// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
//
// Tenstorrent backend: a thin caller of `libskamandros.so`.
//
// The compiler and the driver are skamandros's. This file owns only what zyx
// needs from a backend: a device handle, a DRAM pool, one compiled module per
// kernel, and the host-side layout transfer that turns row-major logical bytes
// into the face-order tile pages skamandros reads.
//
// The library is found through the dynamic loader's search path
// (`LD_LIBRARY_PATH`, the default directories), exactly as CUDA's driver is:
// build/copy `libskamandros.so` somewhere on it and zyx finds it.

use super::{ChunkId, Cmd, CmdQueue, Dev, DeviceInfo, DeviceProgramId, LaunchArg, Placement, PlanDim, Pool, Shard};
use crate::{
    DType, Map, Set, ZyxError,
    backend::DTypeCapability,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{Kernel, Op, OpId},
    shape::Dim,
    slab::Slab,
};
use libloading::Library;
use nanoserde::DeJson;
use std::ffi::{CString, c_char, c_void};
use std::sync::{Arc, Mutex, OnceLock};

// ── The C ABI of libskamandros.so ─────────────────────────────────────────────

/// `ttStatus::TT_OK`.
const TT_OK: i32 = 0;
/// The ABI this backend speaks; `ttAbiVersion` must return it.
const TT_ABI_VERSION: u32 = 1;

/// DRAM the pool hands out: one Blackhole channel, which is the range the
/// driver's allocator addresses (`tt_isa::dram::CHANNEL_BYTES`).
const DRAM_BYTES: u64 = 0xFF00_0000;

/// The ABI surface, resolved once. Function pointers are plain addresses and
/// the library stays loaded for the process, so this is `Send + Sync`.
struct Skamandros {
    _lib: Library,
    compile: unsafe extern "C" fn(*const c_char, usize, *mut *mut c_void, *mut *mut c_char) -> i32,
    module_unload: unsafe extern "C" fn(*mut c_void, *mut *mut c_char) -> i32,
    init: unsafe extern "C" fn(*mut *mut c_char) -> i32,
    _release: unsafe extern "C" fn(*mut *mut c_char) -> i32,
    mem_alloc: unsafe extern "C" fn(usize, *mut u64, *mut *mut c_char) -> i32,
    memcpy_h2d: unsafe extern "C" fn(u64, *const u8, usize, *mut *mut c_char) -> i32,
    memcpy_d2h: unsafe extern "C" fn(*mut u8, u64, usize, *mut *mut c_char) -> i32,
    launch: unsafe extern "C" fn(*mut c_void, *const u64, usize, *mut *mut c_char) -> i32,
    free: unsafe extern "C" fn(*mut c_void),
}

// SAFETY: the struct holds a loaded library plus function pointers into it;
// libloading's `Library` is `Send + Sync`, and the pointers are immutable.
unsafe impl Send for Skamandros {}
unsafe impl Sync for Skamandros {}

static SKAMANDROS: OnceLock<Result<Skamandros, String>> = OnceLock::new();

/// Resolve the library and its entry points. `Err` keeps the loader's message
/// so a missing/broken library is a loud backend failure, not a panic.
fn skamandros() -> Result<&'static Skamandros, BackendError> {
    let loaded = SKAMANDROS.get_or_init(|| {
        // SAFETY: `Library::new` is the documented entry point; the symbols
        // have the signatures the skamandros header declares.
        unsafe {
            let lib = Library::new("libskamandros.so").map_err(|e| format!("libskamandros.so: {e}"))?;
            let compile = *lib
                .get::<unsafe extern "C" fn(*const c_char, usize, *mut *mut c_void, *mut *mut c_char) -> i32>(b"ttCompile\0")
                .map_err(|e| format!("ttCompile: {e}"))?;
            let module_unload = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *mut *mut c_char) -> i32>(b"ttModuleUnload\0")
                .map_err(|e| format!("ttModuleUnload: {e}"))?;
            let init =
                *lib.get::<unsafe extern "C" fn(*mut *mut c_char) -> i32>(b"ttInit\0").map_err(|e| format!("ttInit: {e}"))?;
            let release = *lib
                .get::<unsafe extern "C" fn(*mut *mut c_char) -> i32>(b"ttRelease\0")
                .map_err(|e| format!("ttRelease: {e}"))?;
            let mem_alloc = *lib
                .get::<unsafe extern "C" fn(usize, *mut u64, *mut *mut c_char) -> i32>(b"ttMemAlloc\0")
                .map_err(|e| format!("ttMemAlloc: {e}"))?;
            let memcpy_h2d = *lib
                .get::<unsafe extern "C" fn(u64, *const u8, usize, *mut *mut c_char) -> i32>(b"ttMemcpyH2D\0")
                .map_err(|e| format!("ttMemcpyH2D: {e}"))?;
            let memcpy_d2h = *lib
                .get::<unsafe extern "C" fn(*mut u8, u64, usize, *mut *mut c_char) -> i32>(b"ttMemcpyD2H\0")
                .map_err(|e| format!("ttMemcpyD2H: {e}"))?;
            let launch = *lib
                .get::<unsafe extern "C" fn(*mut c_void, *const u64, usize, *mut *mut c_char) -> i32>(b"ttLaunch\0")
                .map_err(|e| format!("ttLaunch: {e}"))?;
            let free = *lib.get::<unsafe extern "C" fn(*mut c_void)>(b"ttFree\0").map_err(|e| format!("ttFree: {e}"))?;
            let version =
                *lib.get::<unsafe extern "C" fn() -> u32>(b"ttAbiVersion\0").map_err(|e| format!("ttAbiVersion: {e}"))?;
            let version = version();
            if version != TT_ABI_VERSION {
                return Err(format!("libskamandros.so ABI version {version}, this backend speaks {TT_ABI_VERSION}"));
            }
            Ok(Skamandros {
                _lib: lib,
                compile,
                module_unload,
                init,
                _release: release,
                mem_alloc,
                memcpy_h2d,
                memcpy_d2h,
                launch,
                free,
            })
        }
    });
    loaded.as_ref().map_err(|e| BackendError { status: ErrorStatus::Initialization, context: e.clone().into_boxed_str() })
}

/// Turn a failed ABI call into a backend error, freeing the message the
/// library handed back.
fn abi_error(ffi: &Skamandros, status: ErrorStatus, code: i32, err: *mut c_char) -> BackendError {
    let message = if err.is_null() {
        format!("libskamandros: status {code}")
    } else {
        // SAFETY: the ABI guarantees `err` is a NUL-terminated string this
        // library allocated and lets the caller free it.
        let text = unsafe { std::ffi::CStr::from_ptr(err) }.to_string_lossy().into_owned();
        // SAFETY: `ttFree` frees exactly what `err` points at.
        unsafe { (ffi.free)(err.cast::<c_void>()) };
        text
    };
    BackendError { status, context: format!("tenstorrent: {message}").into_boxed_str() }
}

// ── Global state ──────────────────────────────────────────────────────────────

static TT: OnceLock<TTGlobal> = OnceLock::new();

struct TTGlobal {
    pools: Vec<Mutex<TTMemoryPool>>,
    devices: Vec<Mutex<TTDevice>>,
}

/// Config section for this backend (`tenstorrent` in the config file).
#[derive(Default, Debug, DeJson)]
#[nserde(default)]
pub struct TTConfig {
    /// `None` uses every Tenstorrent device; `Some([..])` selects only those
    /// ids. The transport opens one device, id 0.
    pub device_ids: Option<Vec<i32>>,
}

fn backend() -> Result<(&'static Vec<Mutex<TTMemoryPool>>, &'static Vec<Mutex<TTDevice>>), BackendError> {
    if let Some(g) = TT.get() {
        return Ok((&g.pools, &g.devices));
    }
    let global = initialize_backend()?;
    let _ = TT.set(global);
    match TT.get() {
        Some(g) => Ok((&g.pools, &g.devices)),
        None => Err(BackendError {
            status: ErrorStatus::Initialization,
            context: "tenstorrent: backend table was not published".into(),
        }),
    }
}

pub(super) fn pool(id: u16) -> Result<&'static Mutex<TTMemoryPool>, BackendError> {
    backend()?.0.get(id as usize).ok_or_else(|| no_pool(id))
}

fn no_pool(id: u16) -> BackendError {
    BackendError { status: ErrorStatus::Initialization, context: format!("Pool::TT({id}) is not available").into() }
}

pub(super) fn device(id: u16) -> Result<&'static Mutex<TTDevice>, BackendError> {
    backend()?.1.get(id as usize).ok_or_else(|| BackendError {
        status: ErrorStatus::Initialization,
        context: format!("Dev::TT({id}) is not available").into(),
    })
}

pub(super) fn device_count() -> u16 {
    backend().map(|(_, devs)| devs.len() as u16).unwrap_or(0)
}

/// The skamandros device is a Blackhole; its Tensix worker grid bounds the
/// launch grid `tt_tile` may use. Larger group axes become loops.
const GRID_ROWS: Dim = 10;
const GRID_COLS: Dim = 13;

fn device_info() -> DeviceInfo {
    // F8E5M2 has no Blackhole DataFormat: not a capable dtype.
    let mut dtype_capability = [DTypeCapability::all(); DType::N_DTYPES];
    dtype_capability[DType::F8E5M2 as usize] = DTypeCapability::ZERO;
    DeviceInfo {
        compute: 200_000_000_000_000, // ~200 TFLOPS BF16
        max_global_work_dims: vec![GRID_ROWS, GRID_COLS],
        max_local_threads: 1024,
        max_local_work_dims: vec![1, 1024, 1],
        preferred_vector_size: 32,
        local_mem_size: 0, // no thread-shared scratchpad; L1 is per-core private
        max_register_bytes: 128,
        tensor_cores: true,
        warp_size: 1, // Tensix has no SIMT warps
        cc: [0, 0],
        dtype_capability,
        has_native_exp2: false,
        supported_vec_lens: vec![32],
        vector_depth: 2,
        tenstorrent: true,
        tile: [32, 32],
        tile_sizes: vec![[32, 32]],
        mma_variants: vec![],
        num_circular_buffers: 32, // architectural CB0-CB31
        has_openmp: false,
    }
}

fn initialize_backend() -> Result<TTGlobal, BackendError> {
    let config = super::config();
    let debug_dev = super::debug_backends();
    if let Some(ids) = &config.tenstorrent.device_ids
        && ids.is_empty()
    {
        if debug_dev {
            println!("[tenstorrent] configured out");
        }
        return Ok(TTGlobal { pools: Vec::new(), devices: Vec::new() });
    }
    let devices = match &config.tenstorrent.device_ids {
        Some(ids) => ids.len(),
        None => 1,
    };
    if devices > 1 {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: format!("tenstorrent: {devices} devices configured, libskamandros opens device 0").into(),
        });
    }
    if let Some(ids) = &config.tenstorrent.device_ids
        && ids.iter().any(|id| *id != 0)
    {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: format!("tenstorrent device ids {ids:?}: libskamandros opens device 0").into(),
        });
    }

    let ffi = skamandros()?;
    let mut err: *mut c_char = std::ptr::null_mut();
    // SAFETY: `err` is a writable out pointer; the library is loaded.
    let code = unsafe { (ffi.init)(&mut err) };
    if code != TT_OK {
        return Err(abi_error(ffi, ErrorStatus::Initialization, code, err));
    }
    if debug_dev {
        println!("[tenstorrent] device 0 initialized through libskamandros");
        println!("[tenstorrent] tensix grid {GRID_ROWS} rows x {GRID_COLS} cols");
    }
    let info = device_info();
    Ok(TTGlobal {
        pools: vec![Mutex::new(TTMemoryPool::new())],
        devices: vec![Mutex::new(TTDevice {
            device_info: Arc::new(info),
            memory_pool: Pool::TT(0),
            programs: Slab::new(),
        })],
    })
}

// ── DRAM pool ─────────────────────────────────────────────────────────────────

/// One allocation in the device's DRAM. `addr` is the opaque word the ABI's
/// `ttMemAlloc` returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DramChunk {
    addr: u64,
    len: u64,
}

fn io_error(status: ErrorStatus, msg: String) -> BackendError {
    BackendError { status, context: msg.into_boxed_str() }
}

#[derive(Debug)]
pub struct TTMemoryPool {
    total: u64,
    free_bytes: Dim,
    chunks: Slab<ChunkId, DramChunk>,
    free_set: Set<ChunkId>,
}

impl TTMemoryPool {
    fn new() -> Self {
        Self { total: DRAM_BYTES, free_bytes: DRAM_BYTES as Dim, chunks: Slab::new(), free_set: Set::default() }
    }

    pub const fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    /// Allocate `bytes` of device DRAM. Best-fit from the free list first, so
    /// repeated graphs keep stable addresses, else a fresh `ttMemAlloc`.
    pub fn allocate(&mut self, bytes: Dim) -> Result<ChunkId, BackendError> {
        let best = self
            .free_set
            .iter()
            .filter_map(|id| {
                let len = self.chunks[*id].len as Dim;
                (len >= bytes).then_some((len, *id))
            })
            .min();
        if let Some((_, id)) = best {
            self.free_set.remove(&id);
            return Ok(id);
        }
        let len: u64 =
            bytes.try_into().map_err(|_| io_error(ErrorStatus::MemoryAllocation, "DRAM allocation size too large".into()))?;
        if len > self.free_bytes as u64 {
            return Err(io_error(
                ErrorStatus::MemoryAllocation,
                format!("OOM: {len} bytes requested, {} bytes of {} free", self.free_bytes, self.total),
            ));
        }
        let ffi = skamandros()?;
        let mut addr: u64 = 0;
        let mut err: *mut c_char = std::ptr::null_mut();
        // SAFETY: `addr` and `err` are writable; the library is loaded.
        let code = unsafe { (ffi.mem_alloc)(len as usize, &mut addr, &mut err) };
        if code != TT_OK {
            return Err(abi_error(ffi, ErrorStatus::MemoryAllocation, code, err));
        }
        self.free_bytes -= len as Dim;
        Ok(self.chunks.push(DramChunk { addr, len }))
    }

    /// Put one chunk on the free list (reclaimable only by [`Self::dispose`]).
    pub fn release(&mut self, buffer_id: ChunkId) {
        if !self.chunks.contains_id(buffer_id) {
            debug_assert!(false, "release of unknown DRAM chunk {buffer_id:?}");
            return;
        }
        debug_assert!(!self.free_set.contains(&buffer_id), "double release of DRAM chunk {buffer_id:?}");
        self.free_set.insert(buffer_id);
    }

    /// Claim existing allocations, all or nothing.
    pub fn try_reuse_allocations(&mut self, buffer_ids: &Set<ChunkId>) -> bool {
        let ok = buffer_ids.iter().all(|id| self.free_set.contains(id));
        if ok {
            for id in buffer_ids {
                self.free_set.remove(id);
            }
        }
        ok
    }

    /// Forget every free chunk. `libskamandros` has no per-allocation free;
    /// the device's DRAM is released with the device (`ttRelease`), so a
    /// chunk removed here stays reserved until then.
    pub fn dispose(&mut self) {
        for id in core::mem::take(&mut self.free_set) {
            if let Some(chunk) = self.chunks.get(id) {
                self.free_bytes += chunk.len as Dim;
            }
            self.chunks.remove(id);
        }
    }

    /// Blocking upload: host bytes into the chunk's first bytes.
    pub fn host_to_pool(&mut self, src: &[u8], dst: ChunkId) -> Result<(), BackendError> {
        self.write_at(dst, 0, src)
    }

    /// Blocking download: the chunk's first bytes into host bytes.
    pub fn pool_to_host(&mut self, src: ChunkId, dst: &mut [u8]) -> Result<(), BackendError> {
        let len = self.chunk_len(src).min(dst.len());
        self.read_at(src, 0, &mut dst[..len])
    }

    /// The chunk's length in bytes.
    pub fn chunk_len(&self, id: ChunkId) -> usize {
        usize::try_from(self.chunks[id].len).expect("DRAM chunk longer than a host slice")
    }

    /// The opaque ABI address of byte `off` in chunk `id`.
    pub fn addr(&self, id: ChunkId, off: usize) -> Result<u64, BackendError> {
        let chunk = &self.chunks[id];
        let off = u64::try_from(off).map_err(|_| io_error(ErrorStatus::MemoryCopyH2P, "DRAM offset does not fit u64".into()))?;
        if off > chunk.len {
            return Err(io_error(ErrorStatus::MemoryCopyH2P, format!("DRAM offset {off} exceeds chunk {}", chunk.len)));
        }
        chunk.addr.checked_add(off).ok_or_else(|| io_error(ErrorStatus::MemoryCopyH2P, "DRAM address overflows".into()))
    }

    /// Write `src` at byte `off` of chunk `id`.
    pub fn write_at(&mut self, id: ChunkId, off: usize, src: &[u8]) -> Result<(), BackendError> {
        const STATUS: ErrorStatus = ErrorStatus::MemoryCopyH2P;
        let addr = self.addr(id, off)?;
        self.check(id, off, src.len()).map_err(|e| io_error(STATUS, e))?;
        if src.is_empty() {
            return Ok(());
        }
        let ffi = skamandros()?;
        let mut err: *mut c_char = std::ptr::null_mut();
        // SAFETY: `src` is readable for `len` bytes and `addr` is this pool's.
        let code = unsafe { (ffi.memcpy_h2d)(addr, src.as_ptr(), src.len(), &mut err) };
        if code != TT_OK {
            return Err(abi_error(ffi, STATUS, code, err));
        }
        Ok(())
    }

    /// Read byte `off` of chunk `id` into `dst`.
    pub fn read_at(&mut self, id: ChunkId, off: usize, dst: &mut [u8]) -> Result<(), BackendError> {
        const STATUS: ErrorStatus = ErrorStatus::MemoryCopyP2H;
        let addr = self.addr(id, off)?;
        self.check(id, off, dst.len()).map_err(|e| io_error(STATUS, e))?;
        if dst.is_empty() {
            return Ok(());
        }
        let ffi = skamandros()?;
        let mut err: *mut c_char = std::ptr::null_mut();
        // SAFETY: `dst` is writable for `len` bytes and `addr` is this pool's.
        let code = unsafe { (ffi.memcpy_d2h)(dst.as_mut_ptr(), addr, dst.len(), &mut err) };
        if code != TT_OK {
            return Err(abi_error(ffi, STATUS, code, err));
        }
        Ok(())
    }

    fn check(&self, id: ChunkId, off: usize, len: usize) -> Result<(), String> {
        let chunk = &self.chunks[id];
        let end = (off as u64).checked_add(len as u64).ok_or_else(|| format!("DRAM access at {off} of {len} bytes overflows"))?;
        if end > chunk.len {
            return Err(format!("DRAM access {off}..{end} of chunk {id:?} exceeds its {} bytes", chunk.len));
        }
        Ok(())
    }
}

// ── Device ───────────────────────────────────────────────────────────────────

/// One compiled program: the module handle `ttCompile` returned. The handle is
/// an address, not a pointer we dereference, so the device stays `Send`.
#[derive(Debug, Clone, Copy)]
struct TTProgram {
    module: usize,
}

#[derive(Debug)]
pub struct TTDevice {
    device_info: Arc<DeviceInfo>,
    memory_pool: Pool,
    programs: Slab<DeviceProgramId, TTProgram>,
}

impl TTDevice {
    pub fn info(&self) -> Arc<DeviceInfo> {
        self.device_info.clone()
    }

    pub fn free_compute(&self) -> u128 {
        // Zero keeps `Dev::Auto` from selecting TT: an auto-selected TT
        // device fails on kernels whose grid shape it cannot dispatch where
        // C/CUDA succeed. Explicit `Dev::TT` is unaffected.
        0
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        let rendered = kernel.render()?;
        // The renderer's payload is the program text, carried in `Op::Source`.
        let mut text: Option<String> = None;
        let mut op_id = rendered.head;
        while !op_id.is_null() {
            if let Op::Source(src) = &rendered.ops[op_id].op {
                text = Some(src.as_str().to_string());
                break;
            }
            op_id = rendered.next_op(op_id);
        }
        let Some(text) = text else {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent: rendered kernel carries no skamandros program text".into(),
            });
        };
        if debug_asm {
            eprintln!("[tenstorrent] program:\n{text}");
        }
        let ffi = skamandros()?;
        let ctext = CString::new(text).map_err(|_| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: "tenstorrent: program text contains a NUL byte".into(),
        })?;
        let mut module: *mut c_void = std::ptr::null_mut();
        let mut err: *mut c_char = std::ptr::null_mut();
        // SAFETY: the text pointer and length are valid; `module` and `err`
        // are writable out pointers.
        let code = unsafe { (ffi.compile)(ctext.as_ptr(), ctext.as_bytes().len(), &mut module, &mut err) };
        if code != TT_OK {
            return Err(abi_error(ffi, ErrorStatus::KernelCompilation, code, err));
        }
        if module.is_null() {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent: ttCompile returned a null module".into(),
            });
        }
        Ok(self.programs.push(TTProgram { module: module as usize }))
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        let Some(program) = self.programs.get(program_id).copied() else {
            return;
        };
        self.programs.remove(program_id);
        if let Ok(ffi) = skamandros() {
            let mut err: *mut c_char = std::ptr::null_mut();
            // SAFETY: the handle came from `ttCompile` and is not unloaded
            // twice (`programs.remove` above is the single owner).
            let code = unsafe { (ffi.module_unload)(program.module as *mut c_void, &mut err) };
            if code != TT_OK {
                let _ = abi_error(ffi, ErrorStatus::KernelCompilation, code, err);
            }
        }
    }

    pub fn launch(&mut self, program_id: DeviceProgramId, pool_handle: Pool, args: &[LaunchArg]) -> Result<(), BackendError> {
        debug_assert_eq!(pool_handle, self.memory_pool);
        let program = if self.programs.contains_id(program_id) {
            self.programs[program_id]
        } else {
            return Err(BackendError { status: ErrorStatus::KernelLaunch, context: "invalid program id".into() });
        };
        // One word per kernel parameter, in head order: a buffer's DRAM
        // address, or the scalar's value.
        let mut params: Vec<u64> = Vec::with_capacity(args.len());
        {
            let Pool::TT(dev_id) = self.memory_pool else {
                unreachable!("TT device carries {:?}, not a TT pool", self.memory_pool)
            };
            let tpool = super::lock(self.memory_pool, pool(dev_id)?);
            for arg in args {
                match arg {
                    LaunchArg::Buffer { chunk, offset, .. } => params.push(tpool.addr(*chunk, *offset)?),
                    LaunchArg::Variable(c) => {
                        let mut word = [0u8; 8];
                        let bytes = c.to_le_bytes();
                        debug_assert!(bytes.len() <= 8, "scalar launch argument wider than a word");
                        word[..bytes.len()].copy_from_slice(&bytes);
                        params.push(u64::from_le_bytes(word));
                    }
                }
            }
        }
        let ffi = skamandros()?;
        let mut err: *mut c_char = std::ptr::null_mut();
        // SAFETY: the module handle is live and `params` is a valid array.
        let code = unsafe { (ffi.launch)(program.module as *mut c_void, params.as_ptr(), params.len(), &mut err) };
        if code != TT_OK {
            return Err(abi_error(ffi, ErrorStatus::KernelLaunch, code, err));
        }
        Ok(())
    }

    /// Timed launch for autotune. skamandros launches synchronously, so a
    /// wall-clock bracket is already an uncontended measurement.
    pub fn launch_timed(&mut self, program_id: DeviceProgramId, args: &[LaunchArg]) -> Result<u64, BackendError> {
        let start = std::time::Instant::now();
        self.launch(program_id, self.memory_pool, args)?;
        Ok(start.elapsed().as_nanos() as u64)
    }

    /// Copy a transfer into this device's pool. The destination is DRAM, so
    /// the move reads the source's pool into a host staging buffer and uploads
    /// it. Single-shard placements only.
    pub fn copy(&self, src: &Placement, dst: &Placement, bytes: Dim) -> Result<(), BackendError> {
        debug_assert!(bytes >= 0, "TT copy of negative bytes");
        let [src_shard] = &src.shards[..] else {
            todo!("TT copy of multi-shard source placement")
        };
        let [dst_shard] = &dst.shards[..] else {
            todo!("TT copy of multi-shard destination placement")
        };
        debug_assert_eq!(dst_shard.pool, self.memory_pool, "TT copy destination is not on this device");
        debug_assert_eq!(src_shard.offset, 0, "TT copy source has a nonzero offset");
        debug_assert_eq!(dst_shard.offset, 0, "TT copy destination has a nonzero offset");
        if src_shard.pool == self.memory_pool {
            // Same device: a placement copy is a no-op (the chunk is the same).
            return Ok(());
        }
        let mut buf = vec![0u8; bytes as usize];
        src_shard.pool.pool_to_host(src_shard.chunk, &mut buf)?;
        let Pool::TT(dev_id) = self.memory_pool else {
            unreachable!("TT device carries {:?}, not a TT pool", self.memory_pool)
        };
        let mut tpool = super::lock(self.memory_pool, pool(dev_id)?);
        tpool.host_to_pool(&buf, dst_shard.chunk)
    }

    /// Schedule a command run for a Tenstorrent device. A slot dies at its
    /// last use unless pinned (a plan output or read after this partition).
    pub(crate) fn schedule(cmds: Vec<Cmd>, outputs: &Set<OpId>, live_out: Set<OpId>, dev: u16) -> TTPartition {
        let mut last_use: Map<OpId, usize> = Map::default();
        for (idx, cmd) in cmds.iter().enumerate() {
            for r in cmd.reads() {
                last_use.insert(r, idx);
            }
        }
        let mut pinned = outputs.clone();
        pinned.extend(live_out);
        let mut deaths: Vec<Vec<OpId>> = vec![Vec::new(); cmds.len()];
        for (slot, idx) in last_use {
            if !pinned.contains(&slot) {
                deaths[idx].push(slot);
            }
        }
        TTPartition { cmds, deaths, dev }
    }
}

/// Preplanned Tenstorrent partition: ordered commands plus per-command death
/// lists. Launches are synchronous, so program order is the execution order.
#[derive(Debug)]
pub(crate) struct TTPartition {
    pub(crate) cmds: Vec<Cmd>,
    deaths: Vec<Vec<OpId>>,
    pub(crate) dev: u16,
}

impl TTPartition {
    pub(crate) fn replay(
        &self,
        dev: &mut TTDevice,
        resolved: &mut Map<OpId, Arc<Placement>>,
        vars: &Map<OpId, Constant>,
    ) -> Result<(), BackendError> {
        let my_pool = dev.memory_pool;
        debug_assert_eq!(my_pool, Pool::TT(self.dev), "TT partition replayed on the wrong device");
        for (idx, cmd) in self.cmds.iter().enumerate() {
            match cmd {
                Cmd::Launch { program, args, outputs } => {
                    debug_assert_eq!(program.dev, Dev::TT(self.dev), "TT partition holds a non-TT program");
                    for (slot, dtype, dims) in outputs {
                        if resolved.contains_key(slot) {
                            continue;
                        }
                        let el = Dim::from(dtype.bit_size() / 8);
                        let bytes = dims.iter().map(|d| d.eval(vars)).fold(el, |a, b| a * b);
                        debug_assert!(bytes >= 0, "TT replay allocated negative bytes");
                        let chunk = my_pool.allocate(bytes)?;
                        resolved.insert(
                            *slot,
                            Arc::new(Placement { shards: vec![Shard { pool: my_pool, chunk, offset: 0, len: bytes as usize }] }),
                        );
                    }
                    let mut launch_args: Vec<LaunchArg> = Vec::with_capacity(args.len());
                    for arg in args {
                        if let Some(placement) = resolved.get(arg) {
                            let [shard] = &placement.shards[..] else {
                                todo!("multi-shard slot in TT launch")
                            };
                            debug_assert_eq!(shard.pool, my_pool, "TT launch arg is not on this device");
                            debug_assert_eq!(shard.offset, 0, "TT launch arg has a nonzero offset");
                            launch_args.push(LaunchArg::Buffer { chunk: shard.chunk, offset: 0, len: shard.len });
                        } else if let Some(constant) = vars.get(arg) {
                            launch_args.push(LaunchArg::Variable(constant.clone()));
                        } else {
                            panic!("TT replay: launch arg {arg:?} is neither placed nor bound");
                        }
                    }
                    if std::env::var("ZYX_DRY_RUN").is_err() {
                        dev.launch(program.program_id, my_pool, &launch_args)?;
                    }
                }
                Cmd::Alias { class, to } => {
                    let placed = resolved.get(to).unwrap_or_else(|| panic!("TT replay: alias target {to:?} is unplaced")).clone();
                    resolved.insert(*class, placed);
                }
                Cmd::Copy { .. } => unreachable!("copies are Copy partitions, never device runs"),
            }
            for dead in &self.deaths[idx] {
                resolved.remove(dead);
            }
        }
        Ok(())
    }
}

// ── Host-side tile layout ─────────────────────────────────────────────────────
//
// A skamandros page is 32x32 datums in face order: four 16x16 faces, face
// `face_row * 2 + face_col`, each row-major. Tiles are row-major outside. The
// host converts on the way in and out; the device only ever sees pages.

impl Dev {
    /// Copy a logical tensor into this device, honoring its layout. Row-major
    /// sources stage through host temps and swizzle to face order (zero-filled
    /// halo); TT sources are already tile layout and copy physical bytes
    /// unchanged. `shape` is the logical shape.
    pub(crate) fn tilize_transfer(&self, src: &Arc<Placement>, shape: &[Dim], dtype: DType) -> Result<Arc<Placement>, ZyxError> {
        let Dev::TT(_) = self else {
            panic!("tilize_transfer: not a TT device: {self:?}");
        };
        let el = dtype.bit_size() as usize / 8;
        debug_assert!(el > 0, "tilize_transfer: zero element width");
        // A rank-1 value is a tile *column*: element `i` is row `i`, lane 0.
        let (rows, cols) = match shape {
            [.., r, c] => (*r, *c),
            [c] => (*c, 1),
            [] => (1, 1),
        };
        debug_assert!(rows > 0 && cols > 0, "tilize_transfer: non-positive dims {rows}x{cols}");
        let (pr, pc) = self.info().expect("tilize_transfer: TT device has no info").tile_physical_dims(rows, cols);
        let rank = shape.len();
        let outer: Dim = if rank >= 2 { shape[..rank - 2].iter().product() } else { 1 };
        let el_d = el as Dim;
        let tile_elems = 1024 * el_d;
        let tiles = (pr * pc) / 1024;
        let mut phys_dims: Vec<Dim> = Vec::with_capacity(rank.max(2));
        if rank >= 2 {
            phys_dims.extend_from_slice(&shape[..rank - 2]);
        }
        phys_dims.push(tiles * tile_elems);
        let [shard] = &src.shards[..] else {
            todo!("tilize_transfer: multi-shard source");
        };
        debug_assert_eq!(shard.offset, 0, "tilize_transfer: nonzero-offset source shard");
        if matches!(shard.pool, Pool::TT(_)) {
            // Already tile layout: physical byte copy, no swizzle.
            let mut queue = CmdQueue::new();
            queue.push(Cmd::Copy {
                src: OpId::from(0),
                dst: OpId::from(1),
                dst_pool: self.pool(),
                dst_dtype: dtype,
                dst_dims: phys_dims.iter().map(|&d| PlanDim::Const(d)).collect(),
            });
            let mut boundary = Map::default();
            boundary.insert(OpId::from(0), Arc::clone(src));
            let mut outputs = Set::default();
            outputs.insert(OpId::from(1));
            let plan = queue.schedule(&outputs);
            let out = plan.replay(boundary, &Map::default())?;
            return Ok(Arc::clone(&out[&OpId::from(1)]));
        }
        // Logical bytes to a host temp through the existing copy path.
        let logical_len = (outer * rows * cols * el as Dim) as usize;
        let mut queue = CmdQueue::new();
        queue.push(Cmd::Copy {
            src: OpId::from(0),
            dst: OpId::from(1),
            dst_pool: Pool::Host,
            dst_dtype: dtype,
            dst_dims: shape.iter().map(|&d| PlanDim::Const(d)).collect(),
        });
        let mut boundary = Map::default();
        boundary.insert(OpId::from(0), Arc::clone(src));
        let mut outputs = Set::default();
        outputs.insert(OpId::from(1));
        let plan = queue.schedule(&outputs);
        let out = plan.replay(boundary, &Map::default())?;
        let staged = Arc::clone(&out[&OpId::from(1)]);
        let [staged_shard] = &staged.shards[..] else {
            unreachable!("tilize_transfer: staged placement changed shape");
        };
        let mut logical = vec![0u8; logical_len];
        Pool::Host.pool_to_host(staged_shard.chunk, &mut logical)?;
        // Swizzle to face order on the host (zero-filled halo).
        let (ntr, ntc) = ((rows + 31) / 32, (cols + 31) / 32);
        let mut physical = vec![0u8; usize::try_from(outer * tiles * tile_elems).expect("TT tile bytes fit usize")];
        for b in 0..outer {
            for r in 0..rows {
                for c in 0..cols {
                    let (tr, tc) = (r / 32, c / 32);
                    let face = (r % 32) / 16 * 2 + (c % 32) / 16;
                    let within = (r % 16) * 16 + (c % 16);
                    let srci = ((b * rows + r) * cols + c) * el_d;
                    let dsti = (((b * ntr + tr) * ntc + tc) * 1024 + face * 256 + within) * el_d;
                    physical[dsti as usize..dsti as usize + el].copy_from_slice(&logical[srci as usize..srci as usize + el]);
                }
            }
        }
        let plen = physical.len();
        let chunk = Pool::Host.insert_host(physical.into_boxed_slice());
        let staged = Arc::new(Placement { shards: vec![Shard { pool: Pool::Host, chunk, offset: 0, len: plen }] });
        let mut queue = CmdQueue::new();
        queue.push(Cmd::Copy {
            src: OpId::from(0),
            dst: OpId::from(1),
            dst_pool: self.pool(),
            dst_dtype: dtype,
            dst_dims: phys_dims.iter().map(|&d| PlanDim::Const(d)).collect(),
        });
        let mut boundary = Map::default();
        boundary.insert(OpId::from(0), staged);
        let mut outputs = Set::default();
        outputs.insert(OpId::from(1));
        let plan = queue.schedule(&outputs);
        let out = plan.replay(boundary, &Map::default())?;
        Ok(Arc::clone(&out[&OpId::from(1)]))
    }

    /// Copy a TT buffer out of this device, honoring its layout. TT
    /// destinations copy physical bytes unchanged; any other destination
    /// downloads physical bytes, unswizzles to logical row-major on the host,
    /// and moves logical bytes. `shape` is the logical shape.
    pub(crate) fn untilize_transfer(
        &self,
        src: &Arc<Placement>,
        shape: &[Dim],
        dtype: DType,
        dst_pool: Pool,
    ) -> Result<Arc<Placement>, ZyxError> {
        let Dev::TT(_) = self else {
            panic!("untilize_transfer: not a TT device: {self:?}");
        };
        let el = dtype.bit_size() as usize / 8;
        debug_assert!(el > 0, "untilize_transfer: zero element width");
        let (rows, cols) = match shape {
            [.., r, c] => (*r, *c),
            [c] => (*c, 1),
            [] => (1, 1),
        };
        debug_assert!(rows > 0 && cols > 0, "untilize_transfer: non-positive dims {rows}x{cols}");
        let (pr, pc) = self.info().expect("untilize_transfer: TT device has no info").tile_physical_dims(rows, cols);
        let rank = shape.len();
        let outer: Dim = if rank >= 2 { shape[..rank - 2].iter().product() } else { 1 };
        let el_d = el as Dim;
        let tile_elems = 1024 * el_d;
        let tiles = (pr * pc) / 1024;
        let mut phys_dims: Vec<Dim> = Vec::with_capacity(rank.max(2));
        if rank >= 2 {
            phys_dims.extend_from_slice(&shape[..rank - 2]);
        }
        phys_dims.push(tiles * tile_elems);
        let [shard] = &src.shards[..] else {
            todo!("untilize_transfer: multi-shard source");
        };
        debug_assert_eq!(shard.offset, 0, "untilize_transfer: nonzero-offset source shard");
        if matches!(dst_pool, Pool::TT(_)) {
            // Physical-to-physical: no layout change.
            let mut queue = CmdQueue::new();
            queue.push(Cmd::Copy {
                src: OpId::from(0),
                dst: OpId::from(1),
                dst_pool,
                dst_dtype: dtype,
                dst_dims: phys_dims.iter().map(|&d| PlanDim::Const(d)).collect(),
            });
            let mut boundary = Map::default();
            boundary.insert(OpId::from(0), Arc::clone(src));
            let mut outputs = Set::default();
            outputs.insert(OpId::from(1));
            let plan = queue.schedule(&outputs);
            let out = plan.replay(boundary, &Map::default())?;
            return Ok(Arc::clone(&out[&OpId::from(1)]));
        }
        // Physical bytes to a host temp (the replay TT-source override counts
        // shard bytes; dims size the staging buffer).
        let mut queue = CmdQueue::new();
        queue.push(Cmd::Copy {
            src: OpId::from(0),
            dst: OpId::from(1),
            dst_pool: Pool::Host,
            dst_dtype: dtype,
            dst_dims: phys_dims.iter().map(|&d| PlanDim::Const(d)).collect(),
        });
        let mut boundary = Map::default();
        boundary.insert(OpId::from(0), Arc::clone(src));
        let mut outputs = Set::default();
        outputs.insert(OpId::from(1));
        let plan = queue.schedule(&outputs);
        let out = plan.replay(boundary, &Map::default())?;
        let staged = Arc::clone(&out[&OpId::from(1)]);
        let [staged_shard] = &staged.shards[..] else {
            unreachable!("untilize_transfer: staged placement changed shape");
        };
        let physical_len = usize::try_from(outer * tiles * tile_elems).expect("TT tile bytes fit usize");
        let mut physical = vec![0u8; physical_len];
        Pool::Host.pool_to_host(staged_shard.chunk, &mut physical)?;
        // Unswizzle on the host, then move logical bytes.
        let (ntr, ntc) = ((rows + 31) / 32, (cols + 31) / 32);
        let mut logical = vec![0u8; (outer * rows * cols * el as Dim) as usize];
        for b in 0..outer {
            for r in 0..rows {
                for c in 0..cols {
                    let (tr, tc) = (r / 32, c / 32);
                    let face = (r % 32) / 16 * 2 + (c % 32) / 16;
                    let within = (r % 16) * 16 + (c % 16);
                    let srci = (((b * ntr + tr) * ntc + tc) * 1024 + face * 256 + within) * el_d;
                    let dsti = ((b * rows + r) * cols + c) * el as Dim;
                    logical[dsti as usize..dsti as usize + el].copy_from_slice(&physical[srci as usize..srci as usize + el]);
                }
            }
        }
        let logical_len = logical.len();
        let chunk = Pool::Host.insert_host(logical.into_boxed_slice());
        let staged = Arc::new(Placement { shards: vec![Shard { pool: Pool::Host, chunk, offset: 0, len: logical_len }] });
        let mut queue = CmdQueue::new();
        queue.push(Cmd::Copy {
            src: OpId::from(0),
            dst: OpId::from(1),
            dst_pool,
            dst_dtype: dtype,
            dst_dims: shape.iter().map(|&d| PlanDim::Const(d)).collect(),
        });
        let mut boundary = Map::default();
        boundary.insert(OpId::from(0), staged);
        let mut outputs = Set::default();
        outputs.insert(OpId::from(1));
        let plan = queue.schedule(&outputs);
        let out = plan.replay(boundary, &Map::default())?;
        Ok(Arc::clone(&out[&OpId::from(1)]))
    }
}
