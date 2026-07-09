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
    kernel::{Op, UOp},
    shape::Dim,
    slab::Slab,
};
use nanoserde::DeJson;
use std::{
    fs,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    os::unix::io::AsRawFd,
    path::PathBuf,
    process::{Child, ChildStdin, ChildStdout, Command},
    ptr,
    sync::atomic::{AtomicU8, Ordering},
};

// ---------------------------------------------------------------------------
// Kernel driver ioctl interface (from tenstorrent-2.8.0 DKMS ioctl.h)
// ---------------------------------------------------------------------------

const TENSTORRENT_IOCTL_MAGIC: u8 = 0xFA;

// Helper: ioctl code = (_IOC_TYPE << 8) | nr  (simplified; kernel _IOWR also encodes struct size)
const fn ioctl_code(nr: u32) -> u64 {
    const BASE: u64 = TENSTORRENT_IOCTL_MAGIC as u64;
    (BASE << 8) | (nr as u64)
}

const TENSTORRENT_IOCTL_GET_DEVICE_INFO: u64 = ioctl_code(0);
const TENSTORRENT_IOCTL_ALLOCATE_DMA_BUF: u64 = ioctl_code(3);

/// Linux page size (used for DMA buf alignment)
const PAGE_SIZE: u32 = 4096;

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
struct TTGetDeviceInfo {
    output_size_bytes: u32,     // in.output_size_bytes (offset 0)
    out_output_size_bytes: u32, // out.output_size_bytes (offset 4 — required by kernel ABI)
    vendor_id: u16,
    device_id: u16,
    subsystem_vendor_id: u16,
    subsystem_id: u16,
    bus_dev_fn: u16,
    max_dma_buf_size_log2: u16,
    pci_domain: u16,
    reserved: u16,
}

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
struct TTAllocateDmaBufIn {
    requested_size: u32,
    buf_index: u8,
    flags: u8,
    reserved0: [u8; 2],
    reserved1: [u64; 2],
}

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
struct TTAllocateDmaBufOut {
    physical_address: u64,
    mapping_offset: u64,
    size: u32,
    reserved0: u32,
    noc_address: u64,
    reserved1: u64,
}

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
struct TTAllocateDmaBuf {
    inn: TTAllocateDmaBufIn,
    out: TTAllocateDmaBufOut,
}

// ---------------------------------------------------------------------------
// ioctl syscall wrapper
// ---------------------------------------------------------------------------

unsafe fn ioctl_ptr<T>(fd: i32, request: u64, arg: *mut T) -> i32 {
    // SAFETY: caller must ensure fd is valid and arg points to a properly-sized struct
    unsafe { libc::ioctl(fd, request as libc::c_ulong, arg as *mut libc::c_void) }
}

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
const DRAM_SIZE_TABLE: &[(u16, &str, u64)] = &[
    (0x0036, "p100", 28u64 * 1024 * 1024 * 1024),
    (0x0040, "p150a", 32u64 * 1024 * 1024 * 1024),
    (0x0041, "p150b", 32u64 * 1024 * 1024 * 1024),
    (0x0042, "p150c", 32u64 * 1024 * 1024 * 1024),
    (0x0043, "p100a", 28u64 * 1024 * 1024 * 1024),
    (0x0044, "p300b", 64u64 * 1024 * 1024 * 1024),
    (0x0045, "p300a", 64u64 * 1024 * 1024 * 1024),
    (0x0046, "p300c", 64u64 * 1024 * 1024 * 1024),
];

fn dram_size_for_subsystem_id(subsystem_id: u16) -> Result<Dim, BackendError> {
    for &(id, _name, size) in DRAM_SIZE_TABLE {
        if id == subsystem_id {
            return Ok(size as Dim);
        }
    }
    Err(BackendError {
        status: ErrorStatus::Initialization,
        context: format!("unknown Tenstorrent board (subsystem_id=0x{subsystem_id:04x}, card_type=?), please report this to zyx with `lspci -nn | grep 1e52` output").into(),
    })
}

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
// Per-buffer tracking
// ---------------------------------------------------------------------------

struct TTBuffer {
    file: File,
    mmap_ptr: *mut u8,
    size: u32,
    noc_address: u64,
    buf_index: u8,
}

impl std::fmt::Debug for TTBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TTBuffer")
            .field("file", &self.file)
            .field("mmap_ptr", &self.mmap_ptr)
            .field("size", &self.size)
            .field("noc_address", &self.noc_address)
            .field("buf_index", &self.buf_index)
            .finish()
    }
}

impl Drop for TTBuffer {
    fn drop(&mut self) {
        if !self.mmap_ptr.is_null() {
            unsafe {
                libc::munmap(self.mmap_ptr as *mut libc::c_void, self.size as usize);
            }
        }
    }
}

unsafe impl Send for TTBuffer {}
unsafe impl Sync for TTBuffer {}

// ---------------------------------------------------------------------------
// Memory pool
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TTMemoryPool {
    device_file: Option<File>,
    #[allow(unused)]
    total_bytes: Dim,
    free_bytes: Dim,
    next_buf_index: AtomicU8,
    buffers: Slab<PoolBufferId, TTBuffer>,
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

    let device_file = File::options()
        .read(true)
        .write(true)
        .open("/dev/tenstorrent/0")
        .map_err(|e| BackendError {
            status: ErrorStatus::Initialization,
            context: format!("open /dev/tenstorrent/0: {e}").into(),
        })?;

    let fd = device_file.as_raw_fd();

    // Query device info
    // output_size_bytes must be >= sizeof(out struct). The kernel copies min(in.size, sizeof(out))
    // bytes. The out struct starts at offset 4 (after in.output_size_bytes).
    let out_size = size_of::<TTGetDeviceInfo>() as u32 - 4;
    let mut info = TTGetDeviceInfo { output_size_bytes: out_size, ..Default::default() };
    unsafe {
        let ret = ioctl_ptr(fd, TENSTORRENT_IOCTL_GET_DEVICE_INFO, &mut info);
        if ret != 0 {
            return Err(BackendError {
                status: ErrorStatus::Initialization,
                context: format!("TENSTORRENT_IOCTL_GET_DEVICE_INFO: {ret}").into(),
            });
        }
    }

    let total_bytes = dram_size_for_subsystem_id(info.subsystem_id)?;

    if debug_dev {
        let card_name = DRAM_SIZE_TABLE
            .iter()
            .find(|&&(id, _, _)| id == info.subsystem_id)
            .map(|&(_, name, _)| name)
            .unwrap_or("?");
        println!(
            "[tenstorrent] vendor=0x{:04x} device=0x{:04x} subsys=0x{:04x} card={card_name} (subven=0x{:04x})",
            info.vendor_id, info.device_id, info.subsystem_id, info.subsystem_vendor_id
        );
        println!("[tenstorrent] total_dram={} MB", total_bytes / (1024 * 1024));
        println!("[tenstorrent] max_dma_buf_size_log2={}", info.max_dma_buf_size_log2);
    }

    let pool_id = memory_pools.len();
    let pool = MemoryPool::TT(TTMemoryPool {
        device_file: Some(device_file),
        total_bytes,
        free_bytes: total_bytes,
        next_buf_index: AtomicU8::new(0),
        buffers: Slab::new(),
    });
    memory_pools.push(pool);

    // Compute config dir from XDG convention (same as cache_dir but without /cache/tt)
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
            context: format!(
                "runtime not found at {}. Rebuild with TT_METAL_ROOT set.",
                runtime_path.display()
            )
            .into(),
        });
    }

    // Paths provided by build.rs
    let kernel_dir = PathBuf::from(env!("ZYX_TT_KERNEL_DIR"));

    let _device_id = devices.len();
    devices.push(Device::TT(TTDevice {
        device_info: DeviceInfo {
            compute: 200_000_000_000_000, // ~200 TFLOPS FP32
            max_global_work_dims: vec![Dim::from(u32::MAX); 3],
            max_local_threads: 1024,
            max_local_work_dims: vec![1, 1024, 1],
            preferred_vector_size: 16,
            local_mem_size: 1_500_000, // 1.5 MB L1 per Tensix core
            max_register_bytes: 128,
            tensor_cores: true,
            warp_size: 1, // Tensix has no SIMT warps
            supported_dtype_ops: [OpCapability::all(); DType::N_DTYPES],
            has_native_exp2: false,
            has_vector_ops: true,
        },
        memory_pool_id: pool_id,
        runtime: None,
        kernel_dir,
        cache_dir,
        runtime_path,
        programs: Slab::new(),
    }));
    Ok(())
}

impl TTMemoryPool {
    pub fn deinitialize(&mut self) {
        // buffers are dropped → munmap + fd close for each
        // device_file is dropped → fd close
    }

    pub fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
        let bytes32: u32 = u32::try_from(bytes).map_err(|_| BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: "allocation size exceeds 4 GiB".into(),
        })?;

        if self.device_file.is_none() {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "device not opened".into() });
        }

        // Round up to page boundary (kernel requires page-aligned size)
        let page_aligned = bytes32.next_multiple_of(PAGE_SIZE);

        if self.free_bytes < page_aligned as Dim {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "OOM on tenstorrent device".into() });
        }

        let buf_index = self.next_buf_index.fetch_add(1, Ordering::Relaxed);

        // Open a new fd per buffer (workaround: ioctl_free_dma_buf returns -EINVAL;
        // closing the fd triggers kernel cleanup of the DMA buffer)
        let buf_file = File::options()
            .read(true)
            .write(true)
            .open("/dev/tenstorrent/0")
            .map_err(|e| BackendError {
                status: ErrorStatus::MemoryAllocation,
                context: format!("open /dev/tenstorrent/0 for buffer {buf_index}: {e}").into(),
            })?;

        let buf_fd = buf_file.as_raw_fd();

        let mut alloc = TTAllocateDmaBuf {
            inn: TTAllocateDmaBufIn {
                requested_size: page_aligned,
                buf_index,
                flags: 1, // NOC DMA enabled — device reader/writer kernels access this buffer via NOC
                reserved0: [0; 2],
                reserved1: [0; 2],
            },
            out: TTAllocateDmaBufOut::default(),
        };

        unsafe {
            let ret = ioctl_ptr(buf_fd, TENSTORRENT_IOCTL_ALLOCATE_DMA_BUF, &mut alloc);
            if ret != 0 {
                return Err(BackendError {
                    status: ErrorStatus::MemoryAllocation,
                    context: format!("TENSTORRENT_IOCTL_ALLOCATE_DMA_BUF: {ret}").into(),
                });
            }
        }

        let actual_size = alloc.out.size;

        // mmap the buffer for direct CPU access to GDDR6 via PCIe BAR
        let mmap_ptr = unsafe {
            let ptr = libc::mmap(
                ptr::null_mut(),
                actual_size as usize,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                buf_fd,
                alloc.out.mapping_offset as i64,
            );
            if ptr == libc::MAP_FAILED {
                // buf_file dropped here → fd close → kernel frees DMA buf
                return Err(BackendError {
                    status: ErrorStatus::MemoryAllocation,
                    context: format!(
                        "mmap DMA buffer (size={actual_size}, offset=0x{:x})",
                        alloc.out.mapping_offset
                    )
                    .into(),
                });
            }
            ptr as *mut u8
        };

        self.free_bytes -= actual_size as Dim;

        let buf = TTBuffer { file: buf_file, mmap_ptr, size: actual_size, noc_address: alloc.out.noc_address, buf_index };

        let id = self.buffers.push(buf);
        Ok((id, Event::TT(TTEvent)))
    }

    pub fn deallocate(&mut self, buffer_id: PoolBufferId, event_wait_list: Vec<Event>) {
        let _ = event_wait_list;
        if self.buffers.contains_key(buffer_id) {
            let buf = unsafe { self.buffers.remove_and_return(buffer_id) };
            self.free_bytes += buf.size as Dim;
            // TTBuffer::drop handles munmap, then File::drop closes fd → kernel frees GDDR6
        }
    }

    pub fn host_to_pool(&mut self, src: &[u8], dst: PoolBufferId, event_wait_list: Vec<Event>) -> Result<Event, BackendError> {
        let _ = event_wait_list;
        let buf = self
            .buffers
            .get_mut(dst)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyH2P, context: "invalid buffer id".into() })?;
        let len = src.len().min(buf.size as usize);
        unsafe {
            ptr::copy_nonoverlapping(src.as_ptr(), buf.mmap_ptr, len);
        }
        Ok(Event::TT(TTEvent))
    }

    pub fn pool_to_host(&mut self, src: PoolBufferId, dst: &mut [u8], event_wait_list: Vec<Event>) -> Result<(), BackendError> {
        let _ = event_wait_list;
        let buf = self
            .buffers
            .get_mut(src)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyP2H, context: "invalid buffer id".into() })?;
        let len = dst.len().min(buf.size as usize);
        unsafe {
            ptr::copy_nonoverlapping(buf.mmap_ptr, dst.as_mut_ptr(), len);
        }
        Ok(())
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

    pub fn noc_address(&self, buffer_id: PoolBufferId) -> Result<u64, BackendError> {
        if self.buffers.contains_key(buffer_id) {
            Ok(self.buffers[buffer_id].noc_address)
        } else {
            Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "invalid buffer id".into() })
        }
    }

    pub fn buffer_size(&self, buffer_id: PoolBufferId) -> Result<u64, BackendError> {
        if self.buffers.contains_key(buffer_id) {
            Ok(self.buffers[buffer_id].size as u64)
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

        let mut rt = RuntimeProcess {
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
            child,
            timeout_ms: 30000, // 30 second default timeout
        };

        // Send init
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
        // Check if child is still alive first
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
            Ok(None) => {} // still running
        }

        let fd = std::os::unix::io::AsRawFd::as_raw_fd(self.stdout.get_mut());
        let mut pollfd = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };

        let timeout_ms = i32::try_from(timeout_ms).unwrap_or(i32::MAX);
        let ret = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };

        match ret {
            -1 => {
                let err = std::io::Error::last_os_error();
                return Err(BackendError { status: ErrorStatus::KernelLaunch, context: format!("poll error: {err}").into() });
            }
            0 => Ok(false),                              // timeout
            _ => Ok(pollfd.revents & libc::POLLIN != 0), // data available or error
        }
    }

    fn recv_with_timeout(&mut self, timeout_ms: u64) -> Result<String, BackendError> {
        // Use poll-based timeout to prevent blocking indefinitely
        let mut attempts = 0;
        let max_attempts = 3;
        let poll_timeout = timeout_ms / max_attempts;

        while attempts < max_attempts {
            if self.poll_read(poll_timeout)? {
                // Data available, try to read
                let mut line = String::new();
                match self.stdout.read_line(&mut line) {
                    Ok(0) => {
                        // EOF, process exited
                        return Err(BackendError {
                            status: ErrorStatus::KernelLaunch,
                            context: "tt-runtime closed stdout".into(),
                        });
                    }
                    Ok(_) => {
                        return Ok(line.trim().to_string());
                    }
                    Err(e) => {
                        // Read error
                        if attempts == max_attempts - 1 {
                            return Err(BackendError {
                                status: ErrorStatus::KernelLaunch,
                                context: format!("tt-runtime read error: {e}").into(),
                            });
                        }
                        attempts += 1;
                        continue;
                    }
                }
            }

            // Poll timed out, check if child is still alive
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

    fn run(&mut self, hash: &str, n_tiles: u32, src_noc: u64, dst_noc: u64) -> Result<(), BackendError> {
        let cmd = format!(r#"{{"cmd":"run","hash":"{hash}","n_tiles":{n_tiles},"src_noc":{src_noc},"dst_noc":{dst_noc}}}"#);
        self.send(&cmd)?;
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

    fn set_timeout(&mut self, timeout_ms: u64) {
        self.timeout_ms = timeout_ms;
    }
}

fn extract_json_str(json: &str, key: &str) -> Option<String> {
    let k = json.find(&format!("\"{key}\""))?;
    let after_colon = &json[k + key.len() + 4..]; // skip past "key":
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
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TTDevice {
    device_info: DeviceInfo,
    memory_pool_id: PoolId,
    runtime: Option<RuntimeProcess>,
    kernel_dir: PathBuf,
    cache_dir: PathBuf,
    runtime_path: PathBuf,
    programs: Slab<DeviceProgramId, TTProgram>,
}

impl TTDevice {
    pub fn deinitialize(&mut self) {
        if let Some(mut rt) = self.runtime.take() {
            // Use shorter timeout for exit (10 seconds)
            rt.set_timeout(10000);
            let _ = rt.exit();
        }
    }

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

        // Spawn runtime on first use
        if self.runtime.is_none() {
            let kernel_dir = self.kernel_dir.to_string_lossy().to_string();
            let cache_dir = self.cache_dir.to_string_lossy().to_string();
            let runtime_path = self.runtime_path.to_string_lossy().to_string();
            match RuntimeProcess::new(&runtime_path, &kernel_dir, &cache_dir) {
                Ok(rt) => self.runtime = Some(rt),
                Err(e) => {
                    if debug_asm {
                        eprintln!("[tenstorrent] runtime: {e}");
                    }
                    return Err(e);
                }
            }
        }

        // Check disk cache
        let compute_path = self.cache_dir.join(format!("{hash}.cpp"));
        if !compute_path.exists() {
            if debug_asm {
                eprintln!("[tenstorrent] generating {hash}.cpp");
            }
            let source = generate_compute_kernel(kernel)?;
            fs::create_dir_all(&self.cache_dir).map_err(|e| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("create cache dir: {e}").into(),
            })?;
            fs::write(&compute_path, &source).map_err(|e| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("write {hash}.cpp: {e}").into(),
            })?;
        } else if debug_asm {
            eprintln!("[tenstorrent] using cached {hash}.cpp");
        }

        let prog_id = self.programs.push(TTProgram { hash });
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

        let rt = self
            .runtime
            .as_mut()
            .ok_or_else(|| BackendError { status: ErrorStatus::KernelLaunch, context: "runtime not initialized".into() })?;

        // args: first half are inputs, second half are outputs
        // The scheduler convention is: loads first, then stores
        let n_inputs = args.len() / 2;
        if n_inputs == 0 || args.len() < 2 {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("expected at least 2 buffers, got {}", args.len()).into(),
            });
        }

        // Use first input buffer and first output buffer
        let src_buf = args[0];
        let dst_buf = args[n_inputs];

        let src_noc = memory_pool
            .noc_address(src_buf)
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("src noc address: {e}").into() })?;
        let dst_noc = memory_pool
            .noc_address(dst_buf)
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("dst noc address: {e}").into() })?;

        // Count tiles from buffer size (round up to tile boundary)
        let src_bytes = memory_pool
            .buffer_size(src_buf)
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("src buffer size: {e}").into() })?;
        let tile_bytes: u64 = 2048; // TILE_ELEMS * sizeof(bfloat16) = 1024 * 2
        let n_tiles = ((src_bytes + tile_bytes - 1) / tile_bytes) as u32;
        if n_tiles == 0 {
            return Err(BackendError { status: ErrorStatus::KernelLaunch, context: "empty buffer".into() });
        }

        // Use longer timeout for kernel execution (60 seconds)
        let kernel_timeout_ms = 60000;
        rt.set_timeout(kernel_timeout_ms);

        rt.run(&prog.hash, n_tiles, src_noc, dst_noc)
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("runtime run: {e}").into() })?;

        // Reset timeout to default
        rt.set_timeout(30000);

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

fn generate_compute_kernel(kernel: &Kernel) -> Result<String, BackendError> {
    // Walk the IR to find the first supported unary op
    let mut uop = None;
    let mut op_id = kernel.head;
    while !op_id.is_null() {
        match kernel.at(op_id) {
            Op::Unary { uop: op, .. } => {
                uop = Some(*op);
                break;
            }
            _ => {}
        }
        op_id = kernel.next_op(op_id);
    }

    let sfpu = uop_to_sfpu(uop.ok_or_else(|| BackendError {
        status: ErrorStatus::KernelCompilation,
        context: "no unary op found in kernel".into(),
    })?)?;

    Ok(format!(
        r####"#include <cstdint>
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "{header}"

void kernel_main() {{
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    {init_fn}();
    for (uint32_t i = 0; i < n_tiles; i++) {{
        tile_regs_acquire();
        cb_wait_front(tt::CBIndex::c_0, 1);
        copy_tile(tt::CBIndex::c_0, 0, 0);
        {tile_fn}(0);
        cb_pop_front(tt::CBIndex::c_0, 1);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(tt::CBIndex::c_16, 1);
        pack_tile(0, tt::CBIndex::c_16);
        cb_push_back(tt::CBIndex::c_16, 1);
        tile_regs_release();
    }}
}}
"####,
        header = sfpu.header,
        init_fn = sfpu.init_fn,
        tile_fn = sfpu.tile_fn,
    ))
}
