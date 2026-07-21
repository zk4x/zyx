// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Tenstorrent backend for zyx.
//!
//! # Architecture Overview
//!
//! The Tenstorrent backend compiles zyx kernel IR into tt-metal kernels that
//! execute on Tensix RISC-V cores. It uses the low-level compute kernel API
//! (`compute_kernel_api.h`), NOT the high-level ttnn op API.
//!
//! ## Three-Process Model
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │              Rust Process (zyx)                         │
//! │  ┌───────────────┐    ┌──────────────────────────────┐  │
//! │  │  TTMemoryPool  │    │         TTDevice             │  │
//! │  │  - alloc/free  │    │  - compile() → hash +        │  │
//! │  │  DMA buffers   │    │    generate_compute_kernel() │  │
//! │  │  - noc_address │    │  - launch() → args + IPC     │  │
//! │  │  - mmap ptr    │    │  - deinitialize() → exit     │  │
//! │  └──────┬─────────┘    └─────────┬────────────────────┘  │
//! │         │                        │                        │
//! │         │ noc_address(u64)       │ JSON stdin/stdout      │
//! │         │ buffer_size(u64)       │ {cmd,hash,n_tiles,     │
//! │         ▼                        │  src_noc,dst_noc}      │
//! │  ┌──────────────────────────────────────────────┐         │
//! │  │         RuntimeProcess                       │         │
//! │  │  - spawns C++ binary as child                │         │
//! │  │  - BufWriter<ChildStdin>  → send JSON lines  │         │
//! │  │  - BufReader<ChildStdout> ← recv JSON lines  │         │
//! │  │  - try_wait() on recv to detect dead child   │         │
//! │  │    (equiv to CUDA channel disconnect detect) │         │
//! │  └──────────────────────┬───────────────────────┘         │
//! └─────────────────────────┼─────────────────────────────────┘
//!                           │ pipe
//! ┌─────────────────────────┼─────────────────────────────────┐
//! │              C++ Process (zyx-tt-runtime)                 │
//! │  ┌──────────────────────────────────────────────────┐     │
//! │  │                  runtime.cpp                         │     │
//! │  │  JSON IPC loop:                                   │     │
//! │  │  "init"  → tt_device.open() → return "ok"         │     │
//! │  │  "run"   → reader.cpp + compute.cpp + writer.cpp  │     │
//! │  │            SetRuntimeArgs(src_noc,dst_noc,n_tiles)│     │
//! │  │            tt_device.run() → return "ok"          │     │
//! │  │  "exit"  → return "bye"                           │     │
//! │  └──────────────────────┬───────────────────────────┘     │
//! │                         │                                  │
//! │                         ▼                                  │
//! │  ┌──────────────────────────────────────────────────┐     │
//! │  │            tt-metal library calls                 │     │
//! │  │  - Device::create(0)                              │     │
//! │  │  - Program::create()                              │     │
//! │  │  - CreateKernel(reader.cpp, BRISC)                │     │
//! │  │  - CreateKernel(compute.cpp, TRISC)               │     │
//! │  │  - CreateKernel(writer.cpp, NCRISC)               │     │
//! │  │  - SetRuntimeArgs(reader, {src_noc, n_tiles})     │     │
//! │  │  - SetRuntimeArgs(compute, {n_tiles})             │     │
//! │  │  - SetRuntimeArgs(writer, {dst_noc, n_tiles})     │     │
//! │  │  - EnqueueProgram(device, program, queue)         │     │
//! │  └──────────────────────┬───────────────────────────┘     │
//! └─────────────────────────┼─────────────────────────────────┘
//!                           │ MMIO + PCIe
//!                           ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │             Blackhole ASIC (Tensix Array)                │
//! │  ┌──────────────────────────────────────────────────┐   │
//! │  │  Tensix Core (1 of 120)                          │   │
//! │  │  ┌─────────┐ ┌──────────┐ ┌──────────┐           │   │
//! │  │  │ BRISC   │ │ TRISC0   │ │ TRISC1   │           │   │
//! │  │  │ reader  │ │ unpack   │ │ math     │           │   │
//! │  │  │ noc_async│ │ copy_tile│ │ sfpu_op  │           │   │
//! │  │  │ _read   │ │          │ │ pack_tile│           │   │
//! │  │  ├─────────┤ ├──────────┤ ├──────────┤           │   │
//! │  │  │ NCRISC  │ │ TRISC2   │ │ DST Regs │           │   │
//! │  │  │ writer  │ │ pack     │ │ (4 tiles)│           │   │
//! │  │  │ noc_async│ └──────────┘ └──────────┘           │   │
//! │  │  │ _write  │                                       │   │
//! │  │  └─────────┘                                       │   │
//! │  └──────────────────────────────────────────────────┘   │
//! │                                                          │
//! │  GDDR6 (via NOC): 28-64 GB, 1 TB/s bandwidth            │
//! │  L1 per core: 1.5 MB                                    │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Design Decisions
//!
//! ### No Data Over IPC
//!
//! Zero tensor data crosses the JSON pipe. Rust allocates DMA buffers via
//! `TENSTORRENT_IOCTL_ALLOCATE_DMA_BUF`, which returns a **NOC address** —
//! a physical address the Tensix NOC can DMA from/to. The NOC address is
//! passed to the C++ runtime as a `u64` and set as a runtime argument on
//! reader/writer kernels via `SetRuntimeArgs`. The flow:
//!
//! ```text
//! 1. Rust allocates DMA buf  → gets noc_address + mmap ptr
//! 2. Rust writes test data   → memcpy into mmap (CPU → GDDR6)
//! 3. Rust sends "run" IPC    → {src_noc, dst_noc, n_tiles}
//! 4. C++ calls SetRuntimeArgs(reader, {src_noc, n_tiles})
//! 5. C++ calls SetRuntimeArgs(writer, {dst_noc, n_tiles})
//! 6. EnqueueProgram → Tensix runs:
//!    reader: noc_async_read(src_noc → CB)
//!    compute: SFPU op on tile in DST regs
//!    writer: noc_async_write(CB → dst_noc)
//! 7. Rust reads result       → memcpy from mmap (GDDR6 → CPU)
//! ```
//!
//! ### Runtime Process Lifecycle
//!
//! - **Spawned lazily**: first `compile()` call spawns the C++ child, not during
//!   `initialize_device()`. This avoids keeping a child process alive when the
//!   TT backend is configured but unused.
//! - **One process per device**: each `TTDevice` gets its own `RuntimeProcess`.
//! - **Tear-down**: `TTDevice::deinitialize()` sends `{"cmd":"exit"}`, waits for
//!   `"bye"`, then `child.wait()`.
//! - **Crash detection**: `recv()` calls `child.try_wait()` before reading stdout.
//!   If the child has exited (e.g., segfault in tt-metal), returns error immediately
//!   instead of blocking forever on a dead pipe.
//!
//! ### Tiling Convention
//!
//! Tiling is handled by IR optimization passes, NOT by the backend. A tile is
//! 32×32 bfloat16 = 1024 elements = 2048 bytes. The backend computes
//! `n_tiles = ceil(buffer_size / 2048)` from the DMA buffer size. IR passes
//! ensure `vlen=1024` before the kernel reaches the TT backend.
//!
//! ## Tensix Processor Roles
//!
//! Each Tensix core runs 5 RISC-V processors in parallel, coordinated by
//! circular buffers (CBs) in L1 memory:
//!
//! | Processor | Role | Kernel | CB Direction |
//! |-----------|------|--------|-------------|
//! | **BRISC** | Data movement master | Reader | DRAM → CB c_0..c_15 |
//! | **NCRISC** | NOC data movement | Writer | CB c_16..c_31 → DRAM |
//! | **TRISC0** | Unpack | unpack tiles from CB → DST regs | CB → DST |
//! | **TRISC1** | Math | execute SFPU ops on DST regs | DST → DST |
//! | **TRISC2** | Pack | pack DST regs → output CB | DST → CB |
//!
//! ## Kernel Pipeline
//!
//! ```text
//!                  ┌──────────┐
//!                  │  DRAM    │
//!                  │  GDDR6   │
//!                  └────┬─────┘
//!                       │ noc_async_read
//!                       ▼
//!                 ┌───────────┐
//!                 │  CB c_0   │  ← reader kernel (BRISC)
//!                 │  (input)  │
//!                 └─────┬─────┘
//!                       │ copy_tile
//!                       ▼
//!                 ┌───────────┐
//!                 │ DST REGS  │  ← compute kernel (TRISCs)
//!                 │ (4 tiles) │     SFPU: exp, recip, neg, etc.
//!                 └─────┬─────┘
//!                       │ pack_tile
//!                       ▼
//!                 ┌───────────┐
//!                 │  CB c_16  │  ← writer kernel (NCRISC)
//!                 │  (output) │
//!                 └─────┬─────┘
//!                       │ noc_async_write
//!                       ▼
//!                  ┌──────────┐
//!                  │  DRAM    │
//!                  │  GDDR6   │
//!                  └──────────┘
//! ```
//!
//! ## Memory Model
//!
//! ### DMA Buffers (GDDR6)
//!
//! Memory is allocated as DMA buffers via ioctl on `/dev/tenstorrent/N`:
//!
//! ```rust,ignore
//! struct TTAllocateDmaBufOut {
//!     physical_address: u64,   // physical addr for PCIe BAR mmap
//!     mapping_offset: u64,     // offset for mmap(fd, offset=mapping_offset)
//!     size: u32,               // actual allocated size (≥ requested, page-aligned)
//!     noc_address: u64,        // NOC addr for Tensix DMA (reader/writer kernels)
//! }
//! ```
//!
//! - **`mmap`**: CPU accesses GDDR6 via `mmap(fd, PROT_READ|PROT_WRITE, MAP_SHARED,
//!   offset=mapping_offset)`. The returned pointer is used for `host_to_pool`/`pool_to_host`
//!   (memcpy).
//! - **`noc_address`**: Reader/writer kernels use this via `noc_async_read(noc_addr, ...)`
//!   and `noc_async_write(noc_addr, ...)`. This is the physical NOC address on the
//!   Blackhole mesh network.
//! - **`flags=1`**: The ioctl `flags` field must be set to 1 to make the buffer
//!   NOC-accessible. With `flags=0`, the buffer is only accessible via CPU mmap.
//! - **Buffer lifecycle**: One fd per buffer. The kernel driver's `FREE_DMA_BUF`
//!   ioctl returns `-EINVAL`, so deallocation happens by closing the fd (the kernel
//!   frees GDDR6 on fd close).
//!
//! ### Circular Buffers (L1)
//!
//! L1-resident circular buffers connect the three kernels on each Tensix core.
//! The naming convention uses `tt::CBIndex`:
//!
//! | CB Index | Content | Reader/Writer |
//! |----------|---------|---------------|
//! | `c_0` | Input tile (from DRAM) | Reader writes, TRISC0 reads |
//! | `c_16` | Output tile (to DRAM) | TRISC2 writes, Writer reads |
//!
//! ### DST Register File
//!
//! The math processor has 4 tile slots in the DST register file. Compute kernels:
//! 1. `tile_regs_acquire()` — lock DST
//! 2. `copy_tile(cb, 0, 0)` — copy tile from CB to DST slot 0
//! 3. `sfpu_op(0)` — apply SFPU unary to DST slot 0
//! 4. `tile_regs_commit()` — unlock DST
//! 5. `tile_regs_wait()` — wait for commit
//! 6. `pack_tile(0, cb_out)` — pack DST slot 0 → output CB
//! 7. `tile_regs_release()` — release
//!
//! ## Compute Kernel Code Generation
//!
//! `generate_compute_kernel()` walks the zyx kernel IR starting from `kernel.head`,
//! looking for the first `Op::Unary`. It maps the `UOp` variant to an SFPU function
//! via `uop_to_sfpu()` and emits a fixed tile-loop template:
//!
//! ```cpp
//! #include "api/compute/eltwise_unary/<op>.h"
//!
//! void kernel_main() {
//!     uint32_t n_tiles = get_arg_val<uint32_t>(0);
//!     unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
//!     <init_fn>();
//!     for (uint32_t i = 0; i < n_tiles; i++) {
//!         tile_regs_acquire();
//!         cb_wait_front(tt::CBIndex::c_0, 1);
//!         copy_tile(tt::CBIndex::c_0, 0, 0);
//!         <tile_fn>(0);
//!         cb_pop_front(tt::CBIndex::c_0, 1);
//!         tile_regs_commit();
//!         tile_regs_wait();
//!         cb_reserve_back(tt::CBIndex::c_16, 1);
//!         pack_tile(0, tt::CBIndex::c_16);
//!         cb_push_back(tt::CBIndex::c_16, 1);
//!         tile_regs_release();
//!     }
//! }
//! ```
//!
//! ### SFPU Op Mapping
//!
//! | `UOp` | Include | Init | Tile function |
//! |-------|---------|------|---------------|
//! | `Exp2` | — | — | **unsupported** (needs IR pass: `Exp2` → `Exp` × ln2) |
//! | `Reciprocal` | `recip.h` | `recip_tile_init` | `recip_tile` |
//! | `Sqrt` | `sqrt.h` | `sqrt_tile_init` | `sqrt_tile` |
//! | `Sin` | `trigonometry.h` | `sin_tile_init` | `sin_tile` |
//! | `Cos` | `trigonometry.h` | `cos_tile_init` | `cos_tile` |
//! | `Neg` | `negative.h` | `negative_tile_init` | `negative_tile` |
//! | `Floor` | `rounding.h` | `floor_tile_init` | `floor_tile` |
//! | `Trunc` | `rounding.h` | `trunc_tile_init` | `trunc_tile` |
//!
//! Unsupported ops return `BackendError` — the user is expected to add IR
//! optimization passes that convert unsupported ops (e.g., `Exp2` → `Exp` +
//! multiply) before the kernel reaches the backend.
//!
//! ### Cache Directory
//!
//! Generated compute kernels are cached to disk so they survive process restarts.
//! The cache directory follows XDG convention:
//!
//! ```text
//! $XDG_CONFIG_HOME/zyx/cache/tt/<hash>.cpp
//! # falls back to:
//! $HOME/.config/zyx/cache/tt/<hash>.cpp
//! # falls back to:
//! /tmp/zyx-tt-cache/<hash>.cpp
//! ```
//!
//! The `<hash>` is `format!("{:016x}", kernel.get_hash())` — 16 hex chars from
//! the zyx kernel IR hash. Both Rust and C++ compute the same cache path.
//!
//! ## IPC Protocol
//!
//! JSON lines over stdin/stdout. No external JSON library — Rust uses `format!()`,
//! C++ uses manual string parsing.
//!
//! ### Commands
//!
//! **`init`**:
//! ```json
//! {"cmd":"init","kernel_dir":"/path/to/kernels"}
//! → {"status":"ok"}
//! ```
//!
//! **`run`**:
//! ```json
//! {"cmd":"run","hash":"<16-hex>","n_tiles":<u32>,"src_noc":<u64>,"dst_noc":<u64>}
//! → {"status":"ok"}
//! ```
//!
//! **`exit`**:
//! ```json
//! {"cmd":"exit"}
//! → {"status":"ok","msg":"bye"}
//! ```
//!
//! ### Error Response
//! ```json
//! {"status":"error","msg":"<description>"}
//! ```
//!
//! ## Hardware Access
//!
//! The Tenstorrent backend communicates with the device through the kernel
//! driver (`/dev/tenstorrent/N`). Memory is allocated as DMA buffers via
//! `TENSTORRENT_IOCTL_ALLOCATE_DMA_BUF` and freed by closing the fd (one
//! fd per buffer — the `FREE_DMA_BUF` ioctl returns `-EINVAL` in this
//! kernel version). Data transfer uses direct mmap of GDDR6 via PCIe BAR.
//!
//! At initialization, the backend reads the PCI subsystem ID via
//! `TENSTORRENT_IOCTL_GET_DEVICE_INFO` and looks up the board's DRAM
//! configuration in a hardcoded table (the kernel driver does not expose
//! GDDR6 capacity). Trial-allocation probing was abandoned because
//! `dma_alloc_coherent` without an IOMMU draws from system memory, not
//! from device GDDR6.

use super::{Device, DeviceId, DeviceInfo, DeviceProgramId, Event, Kernel, MemoryPool, OpCapability, PoolBufferId, PoolId};
use crate::{
    DType,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, MemLayout, Op, UOp},
    shape::Dim,
    slab::Slab,
};
use nanoserde::DeJson;
use std::{
    ffi::CString,
    fmt::Write,
    io::{BufRead, BufReader, BufWriter, Write as IoWrite},
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{Arc, Mutex},
};

// ---------------------------------------------------------------------------
// DRAM size lookup
// ---------------------------------------------------------------------------

/// GDDR6 sizes for known Blackhole PCI subsystem IDs.
///
/// Blackhole has 8 DRAM channels, each connected to a 4 GB GDDR6 chip.
/// Some boards have channels harvested (fused off) for binning.
/// P100/P100A: 7 usable channels → 28 GB.
/// P150:        8 usable channels → 32 GB.
/// P300:        2 chips × 8 channels → 64 GB.
///
/// These are the total per-board values. The kernel driver does not expose
/// GDDR6 capacity — `dma_alloc_coherent` without IOMMU draws from system
/// memory, not device GDDR6 — so we use this table as a fallback.
///
/// Sources:
/// - https://docs.tenstorrent.com/aibs/blackhole/specifications.html
/// - tt-umd `board_upi_map` and `expected_dram_harvested_units_map`
/// - tt-metal `blackhole_140_arch.yaml` (dram_bank_size: 4278190080 ≈ 4 GB)
const MAX_DRAM_BYTES: u64 = 64u64 * 1024 * 1024 * 1024; // p300: 64 GB
// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Default, Debug, DeJson)]
#[nserde(default)]
pub struct TTConfig {
    /// If set to None, then it will automatically use all Tenstorrent devices,
    /// otherwise it uses only selected devices
    device_ids: Option<Vec<i32>>,
}

// ---------------------------------------------------------------------------
// Per-buffer tracking: index into C++ runtime's vector<MeshBuffer>
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct TTBuffer {
    dev_index: u32,
    size: u64,
}

// ---------------------------------------------------------------------------
// Memory pool — device DRAM buffers managed by C++ runtime.
// TTBuffer is a handle (u32 dev_index) into the runtime's buffer list.
// The pool shares the runtime IPC channel with TTDevice via Arc<Mutex>.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TTMemoryPool {
    buffers: Slab<PoolBufferId, TTBuffer>,
    runtime: Arc<Mutex<RuntimeProcess>>,
    free_bytes: Dim,
}

#[derive(Debug, Clone)]
pub struct TTEvent;

pub(super) fn initialize_device(
    config: &TTConfig,
    memory_pools: &mut Slab<PoolId, MemoryPool>,
    devices: &mut Slab<DeviceId, Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if let Some(device_ids) = &config.device_ids
        && device_ids.is_empty()
    {
        if debug_dev {
            println!("[tenstorrent] configured out");
        }
        return Ok(());
    }

    if debug_dev {
        println!("[tenstorrent] device initialized");
    }

    // Compute config dir from XDG convention
    let config_base = std::env::var_os("XDG_CONFIG_HOME")
        .and_then(|p| {
            let p = PathBuf::from(p);
            if p.is_absolute() { Some(p) } else { None }
        })
        .or_else(|| std::env::home_dir().map(|h| h.join(".config")))
        .unwrap_or_else(|| PathBuf::from("/tmp"));

    let cache_dir = config_base.join("zyx/cache/tt");

    // The runtime binary must be installed at the config dir by build.rs
    let runtime_path = config_base.join("zyx/zyx-tt-runtime");
    if !runtime_path.exists() {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: format!("runtime not found at {}. Rebuild with TT_METAL_ROOT set.", runtime_path.display()).into(),
        });
    }

    // Paths provided by build.rs
    let kernel_dir = PathBuf::from(env!("ZYX_TT_KERNEL_DIR"));

    // Spawn the runtime eagerly — both pool and device need it
    let runtime = Arc::new(Mutex::new(RuntimeProcess::new(
        &runtime_path.to_string_lossy(),
        &kernel_dir.to_string_lossy(),
        &cache_dir.to_string_lossy(),
    )?));

    let pool_id = memory_pools.len();
    let pool = MemoryPool::TT(TTMemoryPool { buffers: Slab::new(), runtime: runtime.clone(), free_bytes: Dim::from(MAX_DRAM_BYTES) });
    memory_pools.push(pool);

    let _device_id = devices.len();
    devices.push(Device::TT(TTDevice {
        device_info: DeviceInfo {
            compute: 200_000_000_000_000, // ~200 TFLOPS FP32
            max_global_work_dims: vec![Dim::from(u32::MAX); 3],
            max_local_threads: 1024,
            max_local_work_dims: vec![1, 1024, 1],
            preferred_vector_size: 32,
            local_mem_size: 1_500_000, // 1.5 MB L1 per Tensix core
            max_register_bytes: 128,
            tensor_cores: true,
            warp_size: 1, // Tensix has no SIMT warps
            supported_dtype_ops: [OpCapability::all(); DType::N_DTYPES],
            has_native_exp2: false,
            supported_vec_lens: vec![32],
        },
        memory_pool_id: pool_id,
        runtime,
        programs: Slab::new(),
    }));
    Ok(())
}

fn create_temp_shm(size: u64) -> Result<(CString, *mut u8, u64), BackendError> {
    let pid = std::process::id();
    let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos();
    let name = format!("/zyx-tt-{pid:x}-{ns:x}");
    let cname = CString::new(name.clone())
        .map_err(|_| BackendError { status: ErrorStatus::MemoryAllocation, context: "invalid shm path".into() })?;

    let fd = unsafe { libc::shm_open(cname.as_ptr(), libc::O_CREAT | libc::O_RDWR | libc::O_EXCL, 0o600) };
    if fd < 0 {
        return Err(BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: format!("shm_open errno={}", std::io::Error::last_os_error()).into(),
        });
    }

    if unsafe { libc::ftruncate(fd, size as i64) } < 0 {
        unsafe { libc::close(fd) };
        let _ = unsafe { libc::shm_unlink(cname.as_ptr()) };
        return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "ftruncate shm".into() });
    }

    let ptr =
        unsafe { libc::mmap(std::ptr::null_mut(), size as usize, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, fd, 0) };
    if ptr == libc::MAP_FAILED {
        unsafe { libc::close(fd) };
        let _ = unsafe { libc::shm_unlink(cname.as_ptr()) };
        return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "mmap shm".into() });
    }

    unsafe { libc::close(fd) };
    Ok((cname, ptr as *mut u8, size))
}

impl TTMemoryPool {
    pub fn deinitialize(&mut self) {
        let _ = self.runtime.lock().unwrap().exit();
    }

    pub fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
        let bytes_u64: u64 = u64::try_from(bytes).map_err(|_| BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: "allocation size exceeds 64-bit".into(),
        })?;
        if bytes > self.free_bytes {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "out of device memory".into() });
        }
        let rt = &self.runtime;
        let tile_bytes: u64 = 2048;
        let dev_index = rt.lock().unwrap().alloc_buf(bytes_u64, tile_bytes)?;
        let buf = TTBuffer { dev_index, size: bytes_u64 };
        let id = self.buffers.push(buf);
        Ok((id, Event::TT(TTEvent)))
    }

    pub fn deallocate(&mut self, buffer_id: PoolBufferId, event_wait_list: Vec<Event>) {
        let _ = event_wait_list;
        if self.buffers.contains_key(buffer_id) {
            let buf = unsafe { self.buffers.remove_and_return(buffer_id) };
            let _ = self.runtime.lock().unwrap().free_buf(buf.dev_index);
        }
    }

    pub fn host_to_pool(&mut self, src: &[u8], dst: PoolBufferId, event_wait_list: Vec<Event>) -> Result<Event, BackendError> {
        let _ = event_wait_list;
        let rt = &self.runtime;
        let buf = self
            .buffers
            .get_mut(dst)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyH2P, context: "invalid buffer id".into() })?;
        let len = src.len().min(buf.size as usize);
        let (cname, shm_ptr, _) = create_temp_shm(len as u64)?;
        let shm_path = cname.to_str().unwrap_or("/none");
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), shm_ptr, len) };
        rt.lock().unwrap().write_buf(buf.dev_index, shm_path, len as u64)?;
        unsafe {
            libc::munmap(shm_ptr as *mut libc::c_void, len as usize);
            libc::shm_unlink(cname.as_ptr());
        }
        Ok(Event::TT(TTEvent))
    }

    pub fn pool_to_host(&mut self, src: PoolBufferId, dst: &mut [u8], event_wait_list: Vec<Event>) -> Result<(), BackendError> {
        let _ = event_wait_list;
        let rt = &self.runtime;
        let buf = self
            .buffers
            .get_mut(src)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyP2H, context: "invalid buffer id".into() })?;
        let len = dst.len().min(buf.size as usize);
        let (cname, shm_ptr, _) = create_temp_shm(len as u64)?;
        let shm_path = cname.to_str().unwrap_or("/none");
        rt.lock().unwrap().read_buf(buf.dev_index, shm_path, len as u64)?;
        unsafe {
            std::ptr::copy_nonoverlapping(shm_ptr, dst.as_mut_ptr(), len);
            libc::munmap(shm_ptr as *mut libc::c_void, len as usize);
            libc::shm_unlink(cname.as_ptr());
        }
        Ok(())
    }

    pub fn pool_to_pool(
        &mut self,
        src_pool: &mut MemoryPool,
        src: PoolBufferId,
        dst: PoolBufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        match src_pool {
            MemoryPool::Host(host_pool) => {
                let data = host_pool.get_buffer(src);
                self.host_to_pool(data, dst, event_wait_list)
            }
            _ => todo!(),
        }
    }

    pub fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        let _ = self;
        let _ = events;
        Ok(())
    }

    pub fn release_events(&mut self, events: Vec<Event>) {
        let _ = self;
        let _ = events;
    }

    pub fn buffer_size(&self, buffer_id: PoolBufferId) -> Result<u64, BackendError> {
        if self.buffers.contains_key(buffer_id) {
            Ok(self.buffers[buffer_id].size)
        } else {
            Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "invalid buffer id".into() })
        }
    }

    pub fn dev_index(&self, buffer_id: PoolBufferId) -> Result<u32, BackendError> {
        if self.buffers.contains_key(buffer_id) {
            Ok(self.buffers[buffer_id].dev_index)
        } else {
            Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "invalid buffer id".into() })
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime process management (JSON IPC over stdin/stdout)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RuntimeProcess {
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    child: Child,
    timeout_ms: u64,
}

impl RuntimeProcess {
    fn new(runtime_path: &str, kernel_dir: &str, cache_dir: &str) -> Result<Self, BackendError> {
        let mut child = Command::new(runtime_path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|e| BackendError {
                status: ErrorStatus::Initialization,
                context: format!("spawn tt-runtime {runtime_path}: {e}").into(),
            })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| BackendError { status: ErrorStatus::Initialization, context: "tt-runtime: no stdin".into() })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| BackendError { status: ErrorStatus::Initialization, context: "tt-runtime: no stdout".into() })?;

        let mut rt = RuntimeProcess { stdin: BufWriter::new(stdin), stdout: BufReader::new(stdout), child, timeout_ms: 30000 };

        let init_json = format!(r#"{{"cmd":"init","kernel_dir":"{kernel_dir}","cache_dir":"{cache_dir}"}}"#);
        rt.send(&init_json)?;
        let resp = rt.recv_with_timeout(rt.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError {
                status: ErrorStatus::Initialization,
                context: format!("tt-runtime init error: {msg}").into(),
            });
        }
        Ok(rt)
    }

    fn send(&mut self, json: &str) -> Result<(), BackendError> {
        eprintln!("[RUST_SEND] {}", &json[..json.len().min(200)]);
        self.stdin
            .write_all(json.as_bytes())
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("tt-runtime write: {e}").into() })?;
        self.stdin.write_all(b"\n").map_err(|e| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: format!("tt-runtime write nl: {e}").into(),
        })?;
        self.stdin
            .flush()
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("tt-runtime flush: {e}").into() })?;
        Ok(())
    }

    fn poll_read(&mut self, timeout_ms: u64) -> Result<bool, BackendError> {
        match self.child.try_wait() {
            Ok(Some(status)) => {
                return Err(BackendError {
                    status: ErrorStatus::KernelLaunch,
                    context: format!("tt-runtime exited unexpectedly (status {status})").into(),
                });
            }
            Err(e) => {
                return Err(BackendError {
                    status: ErrorStatus::KernelLaunch,
                    context: format!("tt-runtime wait error: {e}").into(),
                });
            }
            Ok(None) => {}
        }

        let fd = std::os::unix::io::AsRawFd::as_raw_fd(self.stdout.get_mut());
        let mut pollfd = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };

        let timeout_ms = i32::try_from(timeout_ms).unwrap_or(i32::MAX);
        let ret = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };

        match ret {
            -1 => {
                let err = std::io::Error::last_os_error();
                Err(BackendError { status: ErrorStatus::KernelLaunch, context: format!("poll error: {err}").into() })
            }
            0 => Ok(false),
            _ => Ok(pollfd.revents & libc::POLLIN != 0),
        }
    }

    fn recv_with_timeout(&mut self, timeout_ms: u64) -> Result<String, BackendError> {
        let mut attempts = 0;
        let max_attempts = 3;
        let poll_timeout = timeout_ms / max_attempts;

        while attempts < max_attempts {
            if self.poll_read(poll_timeout)? {
                let mut line = String::new();
                match self.stdout.read_line(&mut line) {
                    Ok(0) => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelLaunch,
                            context: "tt-runtime closed stdout".into(),
                        });
                    }
                    Ok(_) => {
                        let trimmed = line.trim().to_string();
                        // Skip non-JSON lines (UMD log messages leaking to stdout)
                        if trimmed.starts_with('{') {
                            eprintln!("[RUST_RECV] {trimmed}");
                            return Ok(trimmed);
                        }
                        // Log line — keep reading
                        continue;
                    }
                    Err(_) => {
                        attempts += 1;
                        continue;
                    }
                }
            }
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelLaunch,
                        context: format!("tt-runtime exited unexpectedly during read (status {status})").into(),
                    });
                }
                Err(e) => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelLaunch,
                        context: format!("tt-runtime wait error during read: {e}").into(),
                    });
                }
                Ok(None) => {
                    attempts += 1;
                }
            }
        }
        Err(BackendError {
            status: ErrorStatus::KernelLaunch,
            context: format!("tt-runtime read timeout after {}ms", timeout_ms).into(),
        })
    }

    fn alloc_buf(&mut self, size: u64, tile_bytes: u64) -> Result<u32, BackendError> {
        let cmd = format!(r#"{{"cmd":"alloc_buf","size":{size},"tile_bytes":{tile_bytes}}}"#);
        self.send(&cmd)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError {
                status: ErrorStatus::MemoryAllocation,
                context: format!("alloc_buf error: {msg}").into(),
            });
        }
        let idx_str = extract_json_str(&resp, "index").ok_or_else(|| BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: "alloc_buf: no index in response".into(),
        })?;
        let idx: u32 = idx_str.parse().map_err(|_| BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: format!("alloc_buf: invalid index '{idx_str}'").into(),
        })?;
        Ok(idx)
    }

    fn free_buf(&mut self, dev_index: u32) -> Result<(), BackendError> {
        let cmd = format!(r#"{{"cmd":"free_buf","index":{dev_index}}}"#);
        self.send(&cmd)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: format!("free_buf error: {msg}").into() });
        }
        Ok(())
    }

    fn write_buf(&mut self, dev_index: u32, shm_path: &str, size: u64) -> Result<(), BackendError> {
        let cmd = format!(r#"{{"cmd":"write_buf","index":{dev_index},"shm_path":"{shm_path}","size":{size}}}"#);
        self.send(&cmd)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError { status: ErrorStatus::MemoryCopyH2P, context: format!("write_buf error: {msg}").into() });
        }
        Ok(())
    }

    fn read_buf(&mut self, dev_index: u32, shm_path: &str, size: u64) -> Result<(), BackendError> {
        let cmd = format!(r#"{{"cmd":"read_buf","index":{dev_index},"shm_path":"{shm_path}","size":{size}}}"#);
        self.send(&cmd)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError { status: ErrorStatus::MemoryCopyP2H, context: format!("read_buf error: {msg}").into() });
        }
        Ok(())
    }

    fn run(
        &mut self,
        hash: &str,
        reader_source: &str,
        compute_source: &str,
        n_tiles: u32,
        src_indices: &[u32],
        dst_index: u32,
        input_formats: &[u32],
        input_tile_bytes: &[u32],
        output_formats: &[u32],
        output_tile_bytes: &[u32],
    ) -> Result<(), BackendError> {
        let n_inputs = src_indices.len();
        let n_outputs = 1;
        let reader_source_len = reader_source.len();
        let compute_source_len = compute_source.len();
        let mut cmd = format!(
            r#"{{"cmd":"run","hash":"{hash}","reader_source_len":{reader_source_len},"compute_source_len":{compute_source_len},"n_inputs":{n_inputs},"n_outputs":{n_outputs},"n_tiles":{n_tiles}"#
        );
        for (i, idx) in src_indices.iter().enumerate() {
            cmd.push_str(&format!(r#","src{i}":{idx}"#));
        }
        cmd.push_str(&format!(r#","dst0":{dst_index}"#));
        for (i, fmt) in input_formats.iter().enumerate() {
            cmd.push_str(&format!(r#","fmt_i{i}":{fmt}"#));
        }
        for (i, tb) in input_tile_bytes.iter().enumerate() {
            cmd.push_str(&format!(r#","tb_i{i}":{tb}"#));
        }
        for (i, fmt) in output_formats.iter().enumerate() {
            cmd.push_str(&format!(r#","fmt_o{i}":{fmt}"#));
        }
        for (i, tb) in output_tile_bytes.iter().enumerate() {
            cmd.push_str(&format!(r#","tb_o{i}":{tb}"#));
        }
        cmd.push('}');
        self.send(&cmd)?;
        // Send raw reader source bytes
        self.stdin.write_all(reader_source.as_bytes()).map_err(|e| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: format!("tt-runtime write reader: {e}").into(),
        })?;
        // Send raw compute source bytes
        self.stdin.write_all(compute_source.as_bytes()).map_err(|e| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: format!("tt-runtime write compute: {e}").into(),
        })?;
        self.stdin
            .flush()
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("tt-runtime flush: {e}").into() })?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("tt-runtime run error: {msg}").into(),
            });
        }
        Ok(())
    }

    fn exit(&mut self) -> Result<(), BackendError> {
        self.send(r#"{"cmd":"exit"}"#)?;
        let resp = self.recv_with_timeout(self.timeout_ms)?;
        if resp.contains("\"error\"") {
            let msg = extract_json_str(&resp, "msg").unwrap_or_else(|| "unknown".into());
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("tt-runtime exit error: {msg}").into(),
            });
        }
        self.child.wait().ok();
        Ok(())
    }
}

fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let k = json.find(&format!("\"{key}\""))?;
    let after_colon = &json[k + key.len() + 3..]; // skip past "key":
    let start = after_colon.find('"')? + 1;
    let end = after_colon[start..].find('"')?;
    Some(after_colon[start..start + end].to_string())
}

// ---------------------------------------------------------------------------
// Compiled program tracking
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct TTProgram {
    hash: String,
    reader_source: String,
    compute_source: String,
    input_dtypes: Vec<DType>,
    output_dtypes: Vec<DType>,
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TTDevice {
    device_info: DeviceInfo,
    memory_pool_id: PoolId,
    runtime: Arc<Mutex<RuntimeProcess>>,
    programs: Slab<DeviceProgramId, TTProgram>,
}

impl TTDevice {
    pub fn deinitialize(&mut self) {}

    pub const fn info(&self) -> &DeviceInfo {
        &self.device_info
    }

    pub const fn memory_pool_id(&self) -> PoolId {
        self.memory_pool_id
    }

    pub const fn free_compute(&self) -> u128 {
        self.device_info.compute
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        let hash = format!("{:016x}", kernel.get_hash());

        let mut n_inputs: usize = 0;
        let mut n_outputs: usize = 0;
        let mut input_dtypes: Vec<DType> = Vec::new();
        let mut output_dtypes: Vec<DType> = Vec::new();
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            match kernel.at(op_id) {
                Op::Load { src, layout: MemLayout::Tile { .. }, .. } => {
                    if let Op::Define { dtype, .. } = kernel.at(*src) {
                        input_dtypes.push(*dtype);
                    }
                    n_inputs += 1;
                }
                Op::Store { dst, layout: MemLayout::Tile { .. }, .. } => {
                    if let Op::Define { dtype, .. } = kernel.at(*dst) {
                        output_dtypes.push(*dtype);
                    }
                    n_outputs += 1;
                }
                _ => {}
            }
            op_id = kernel.next_op(op_id);
        }

        let mut op_id = kernel.head;
        let mut unary_op = None;
        let mut binary_op = None;
        while !op_id.is_null() {
            if let Op::Store { x, layout: MemLayout::Tile { .. }, .. } = kernel.at(op_id) {
                let compute = *x;
                let mut scan = kernel.head;
                while !scan.is_null() {
                    if scan == compute {
                        match kernel.at(scan) {
                            Op::Unary { uop, .. } => {
                                if uop_to_sfpu(*uop).is_ok() {
                                    unary_op = Some(*uop);
                                }
                            }
                            Op::Binary { bop, .. } => {
                                if bop_to_binary_api(*bop).is_ok() {
                                    binary_op = Some(*bop);
                                }
                            }
                            _ => {}
                        }
                        break;
                    }
                    scan = kernel.next_op(scan);
                }
                break;
            }
            op_id = kernel.next_op(op_id);
        }

        let mut compute_source = String::new();
        compute_source.push_str("#include <cstdint>\n");
        compute_source.push_str("#include \"api/compute/common.h\"\n");
        compute_source.push_str("#include \"api/compute/tile_move_copy.h\"\n");
        compute_source.push_str("#include \"api/compute/eltwise_unary/eltwise_unary.h\"\n");

        match (unary_op, binary_op) {
            (Some(uop), None) => {
                let info = uop_to_sfpu(uop)?;
                compute_source.push_str(&format!("#include \"{}\"\n", info.header));
                compute_source.push_str("#include \"api/compute/compute_kernel_api.h\"\n\n");
                write!(compute_source, "void kernel_main() {{\n").unwrap();
                write!(compute_source, "    uint32_t n_tiles = get_arg_val<uint32_t>(0);\n").unwrap();
                for i in 0..n_inputs {
                    write!(compute_source, "    constexpr auto cb_in{i} = tt::CBIndex::c_{i};\n").unwrap();
                }
                write!(compute_source, "    constexpr auto cb_out0 = tt::CBIndex::c_16;\n").unwrap();
                write!(compute_source, "    constexpr uint32_t dst_reg = 0;\n\n").unwrap();
                write!(compute_source, "    unary_op_init_common(cb_in0, cb_out0);\n").unwrap();
                write!(compute_source, "    {}();\n", info.init_fn).unwrap();
                write!(compute_source, "\n").unwrap();
                write!(compute_source, "    for (uint32_t i = 0; i < n_tiles; i++) {{\n").unwrap();
                for i in 0..n_inputs {
                    write!(compute_source, "        cb_wait_front(cb_in{i}, 1);\n").unwrap();
                }
                write!(compute_source, "        tile_regs_acquire();\n").unwrap();
                write!(compute_source, "        copy_tile(cb_in0, 0, dst_reg);\n").unwrap();
                write!(compute_source, "        {}(dst_reg);\n", info.tile_fn).unwrap();
                write!(compute_source, "        tile_regs_commit();\n").unwrap();
                write!(compute_source, "        tile_regs_wait();\n").unwrap();
                write!(compute_source, "        cb_reserve_back(cb_out0, 1);\n").unwrap();
                write!(compute_source, "        pack_tile(dst_reg, cb_out0);\n").unwrap();
                write!(compute_source, "        cb_push_back(cb_out0, 1);\n").unwrap();
                for i in 0..n_inputs {
                    write!(compute_source, "        cb_pop_front(cb_in{i}, 1);\n").unwrap();
                }
                write!(compute_source, "        tile_regs_release();\n").unwrap();
                write!(compute_source, "    }}\n").unwrap();
                write!(compute_source, "}}\n").unwrap();
            }
            (None, Some(bop)) => {
                let info = bop_to_binary_api(bop)?;
                compute_source.push_str(&format!("#include \"{}\"\n", info.header));
                compute_source.push_str("#include \"api/compute/compute_kernel_api.h\"\n\n");
                write!(compute_source, "void kernel_main() {{\n").unwrap();
                write!(compute_source, "    uint32_t n_tiles = get_arg_val<uint32_t>(0);\n").unwrap();
                for i in 0..n_inputs {
                    write!(compute_source, "    constexpr auto cb_in{i} = tt::CBIndex::c_{i};\n").unwrap();
                }
                write!(compute_source, "    constexpr auto cb_out0 = tt::CBIndex::c_16;\n").unwrap();
                write!(compute_source, "    constexpr uint32_t dst_reg = 0;\n\n").unwrap();
                if info.uses_cbs {
                    write!(compute_source, "    binary_op_init_common(cb_in0, cb_in1, cb_out0);\n").unwrap();
                }
                write!(compute_source, "    {}();\n", info.tile_init_fn).unwrap();
                write!(compute_source, "\n").unwrap();
                write!(compute_source, "    for (uint32_t i = 0; i < n_tiles; i++) {{\n").unwrap();
                for i in 0..n_inputs {
                    write!(compute_source, "        cb_wait_front(cb_in{i}, 1);\n").unwrap();
                }
                write!(compute_source, "        tile_regs_acquire();\n").unwrap();
                if info.uses_cbs {
                    write!(compute_source, "        {}(cb_in0, cb_in1, 0, 0, dst_reg);\n", info.tile_fn).unwrap();
                } else {
                    write!(compute_source, "        copy_tile(cb_in0, 0, dst_reg);\n").unwrap();
                    write!(compute_source, "        {}(dst_reg, dst_reg, dst_reg);\n", info.tile_fn).unwrap();
                }
                write!(compute_source, "        tile_regs_commit();\n").unwrap();
                write!(compute_source, "        tile_regs_wait();\n").unwrap();
                write!(compute_source, "        cb_reserve_back(cb_out0, 1);\n").unwrap();
                write!(compute_source, "        pack_tile(dst_reg, cb_out0);\n").unwrap();
                write!(compute_source, "        cb_push_back(cb_out0, 1);\n").unwrap();
                for i in 0..n_inputs {
                    write!(compute_source, "        cb_pop_front(cb_in{i}, 1);\n").unwrap();
                }
                write!(compute_source, "        tile_regs_release();\n").unwrap();
                write!(compute_source, "    }}\n").unwrap();
                write!(compute_source, "}}\n").unwrap();
            }
            _ => {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("no supported unary or binary op found in kernel (inputs={n_inputs}, outputs={n_outputs})")
                        .into(),
                });
            }
        }

        let reader_source = if n_inputs == 0 {
            "#include <cstdint>\nvoid kernel_main() {}".to_string()
        } else {
            let mut code = String::new();
            write!(code, "#include <cstdint>\n#include \"api/dataflow/dataflow_api.h\"\n\nvoid kernel_main() {{\n").unwrap();
            write!(code, "    uint32_t n_tiles = get_arg_val<uint32_t>({});\n", n_inputs * 2).unwrap();
            write!(code, "\n").unwrap();
            write!(code, "    for (uint32_t s = 0; s < {}; s++) {{\n", n_inputs).unwrap();
            write!(code, "        uint32_t src_noc_low = get_arg_val<uint32_t>(s * 2);\n").unwrap();
            write!(code, "        uint32_t src_noc_high = get_arg_val<uint32_t>(s * 2 + 1);\n").unwrap();
            write!(code, "        uint64_t src_noc_addr = (uint64_t)src_noc_high << 32 | src_noc_low;\n").unwrap();
            write!(code, "        uint32_t cb_id = tt::CBIndex::c_0 + s;\n").unwrap();
            write!(code, "        uint32_t tile_bytes = get_tile_size(cb_id);\n").unwrap();
            write!(code, "\n").unwrap();
            write!(code, "        for (uint32_t i = 0; i < n_tiles; i++) {{\n").unwrap();
            write!(code, "            cb_reserve_back(cb_id, 1);\n").unwrap();
            write!(code, "            uint32_t l1_addr = get_write_ptr(cb_id);\n").unwrap();
            write!(code, "            uint64_t noc_addr = src_noc_addr + i * tile_bytes;\n").unwrap();
            write!(code, "            noc_async_read(noc_addr, l1_addr, tile_bytes);\n").unwrap();
            write!(code, "            noc_async_read_barrier();\n").unwrap();
            write!(code, "            cb_push_back(cb_id, 1);\n").unwrap();
            write!(code, "        }}\n").unwrap();
            write!(code, "    }}\n").unwrap();
            write!(code, "}}\n").unwrap();
            code
        };

        if debug_asm {
            eprintln!("[tenstorrent] === reader kernel ===\n{reader_source}\n=== end reader ===");
            eprintln!("[tenstorrent] === compute kernel ===\n{compute_source}\n=== end compute ===");
        }
        let prog_id =
            self.programs.push(TTProgram { hash, reader_source, compute_source, input_dtypes, output_dtypes });
        Ok(prog_id)
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        if self.programs.contains_key(program_id) {
            unsafe { self.programs.remove_and_return(program_id) };
        }
    }

    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        memory_pool: &mut TTMemoryPool,
        args: &[PoolBufferId],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let _ = event_wait_list;
        let prog = if self.programs.contains_key(program_id) {
            &self.programs[program_id]
        } else {
            return Err(BackendError { status: ErrorStatus::KernelLaunch, context: "invalid program id".into() });
        };

        let rt = &self.runtime;

        let n_inputs = prog.input_dtypes.len();
        let n_outputs = prog.output_dtypes.len();

        if args.len() < n_inputs + n_outputs {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("expected {} buffers, got {}", n_inputs + n_outputs, args.len()).into(),
            });
        }

        let mut src_indices: Vec<u32> = Vec::with_capacity(n_inputs);
        for i in 0..n_inputs {
            let idx = memory_pool.dev_index(args[i]).map_err(|e| BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("src{i} dev_index: {e}").into(),
            })?;
            src_indices.push(idx);
        }
        let dst_buf = args[n_inputs];
        let dst_index = memory_pool
            .dev_index(dst_buf)
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("dst dev_index: {e}").into() })?;

        let src_bytes = memory_pool
            .buffer_size(args[0])
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("src buffer size: {e}").into() })?;
        let first_tile_bytes: u32 = {
            let te = 1024u64;
            (match prog.input_dtypes.first().copied().unwrap_or(DType::F32) {
                DType::F32 => 4 * te,
                DType::F16 | DType::BF16 => 2 * te,
                _ => 4 * te,
            }) as u32
        };
        let n_tiles = ((src_bytes + first_tile_bytes as u64 - 1) / first_tile_bytes as u64) as u32;
        if n_tiles == 0 {
            return Err(BackendError { status: ErrorStatus::KernelLaunch, context: "empty buffer".into() });
        }

        fn dtype_to_format(dt: DType) -> u32 {
            match dt {
                DType::F32 => 0,
                DType::F16 => 1,
                DType::BF16 => 2,
                _ => 0,
            }
        }
        fn dtype_to_tile_bytes(dt: DType) -> u32 {
            let te = 1024u64;
            match dt {
                DType::F32 => (4 * te) as u32,
                DType::F16 | DType::BF16 => (2 * te) as u32,
                _ => (4 * te) as u32,
            }
        }
        let input_formats: Vec<u32> = prog.input_dtypes.iter().map(|d| dtype_to_format(*d)).collect();
        let input_tile_bytes: Vec<u32> = prog.input_dtypes.iter().map(|d| dtype_to_tile_bytes(*d)).collect();
        let output_formats: Vec<u32> = prog.output_dtypes.iter().map(|d| dtype_to_format(*d)).collect();
        let output_tile_bytes: Vec<u32> = prog.output_dtypes.iter().map(|d| dtype_to_tile_bytes(*d)).collect();

        let mut rt_guard = rt.lock().unwrap();
        rt_guard.run(
            &prog.hash,
            &prog.reader_source,
            &prog.compute_source,
            n_tiles,
            &src_indices,
            dst_index,
            &input_formats,
            &input_tile_bytes,
            &output_formats,
            &output_tile_bytes,
        )?;

        Ok(Event::TT(TTEvent))
    }
}

// ---------------------------------------------------------------------------
// Compute kernel code generation
// ---------------------------------------------------------------------------

struct SfpuInfo {
    header: &'static str,
    init_fn: &'static str,
    tile_fn: &'static str,
}

struct BinaryApi {
    tile_fn: &'static str,
    tile_init_fn: &'static str,
    header: &'static str,
    /// If true, use the traditional api where tiles read directly from CBs
    /// (add_tiles(cb0, cb1, 0, 0, dst)). If false, use the SFPU api with
    /// copy_tile + bop_tile(dst, dst, dst).
    uses_cbs: bool,
}

fn bop_to_binary_api(bop: BOp) -> Result<BinaryApi, BackendError> {
    match bop {
        // Traditional binary API (eltwise_binary.h) — reads directly from CBs
        BOp::Add => Ok(BinaryApi {
            tile_fn: "add_tiles",
            tile_init_fn: "add_tiles_init",
            header: "api/compute/eltwise_binary.h",
            uses_cbs: true,
        }),
        BOp::Sub => Ok(BinaryApi {
            tile_fn: "sub_tiles",
            tile_init_fn: "sub_tiles_init",
            header: "api/compute/eltwise_binary.h",
            uses_cbs: true,
        }),
        BOp::Mul => Ok(BinaryApi {
            tile_fn: "mul_tiles",
            tile_init_fn: "mul_tiles_init",
            header: "api/compute/eltwise_binary.h",
            uses_cbs: true,
        }),
        // SFPU binary API (eltwise_binary_sfpu.h) — operates on DST regs
        BOp::Div => Ok(BinaryApi {
            tile_fn: "div_binary_tile",
            tile_init_fn: "div_binary_tile_init",
            header: "api/compute/eltwise_binary_sfpu.h",
            uses_cbs: false,
        }),
        BOp::Pow => Ok(BinaryApi {
            tile_fn: "power_binary_tile",
            tile_init_fn: "power_binary_tile_init",
            header: "api/compute/eltwise_binary_sfpu.h",
            uses_cbs: false,
        }),
        BOp::Eq => Ok(BinaryApi {
            tile_fn: "eq_binary_tile",
            tile_init_fn: "eq_binary_tile_init",
            header: "api/compute/eltwise_binary_sfpu.h",
            uses_cbs: false,
        }),
        BOp::NotEq => Ok(BinaryApi {
            tile_fn: "ne_binary_tile",
            tile_init_fn: "ne_binary_tile_init",
            header: "api/compute/eltwise_binary_sfpu.h",
            uses_cbs: false,
        }),
        BOp::Cmplt => Ok(BinaryApi {
            tile_fn: "lt_binary_tile",
            tile_init_fn: "lt_binary_tile_init",
            header: "api/compute/eltwise_binary_sfpu.h",
            uses_cbs: false,
        }),
        BOp::Max => Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: "Max binary op not yet supported for Tenstorrent (add an IR optimization pass)".into(),
        }),
        _ => Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("unsupported binary op {bop:?} for Tenstorrent (add an IR optimization pass)").into(),
        }),
    }
}

fn uop_to_sfpu(uop: UOp) -> Result<SfpuInfo, BackendError> {
    match uop {
        UOp::Exp => Ok(SfpuInfo { header: "api/compute/eltwise_unary/exp.h", init_fn: "exp_tile_init", tile_fn: "exp_tile" }),
        // Exp2 is not available in tt-metal SFPU (no exp2_tile). An IR
        // optimization pass must convert Exp2 → Exp + multiply by ln(2)
        // before the kernel reaches this backend.
        UOp::Reciprocal => {
            Ok(SfpuInfo { header: "api/compute/eltwise_unary/recip.h", init_fn: "recip_tile_init", tile_fn: "recip_tile" })
        }
        UOp::Sqrt => Ok(SfpuInfo { header: "api/compute/eltwise_unary/sqrt.h", init_fn: "sqrt_tile_init", tile_fn: "sqrt_tile" }),
        UOp::Sin => {
            Ok(SfpuInfo { header: "api/compute/eltwise_unary/trigonometry.h", init_fn: "sin_tile_init", tile_fn: "sin_tile" })
        }
        UOp::Cos => {
            Ok(SfpuInfo { header: "api/compute/eltwise_unary/trigonometry.h", init_fn: "cos_tile_init", tile_fn: "cos_tile" })
        }
        UOp::Neg => Ok(SfpuInfo {
            header: "api/compute/eltwise_unary/negative.h",
            init_fn: "negative_tile_init",
            tile_fn: "negative_tile",
        }),
        UOp::Floor => {
            Ok(SfpuInfo { header: "api/compute/eltwise_unary/rounding.h", init_fn: "floor_tile_init", tile_fn: "floor_tile" })
        }
        UOp::Trunc => {
            Ok(SfpuInfo { header: "api/compute/eltwise_unary/rounding.h", init_fn: "trunc_tile_init", tile_fn: "trunc_tile" })
        }
        _ => Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("unsupported unary op {uop:?} for Tenstorrent (add an IR optimization pass)").into(),
        }),
    }
}
