// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
//
// Tenstorrent backend for zyx.
//
// # Grid indexing (gidx)
//
// Two kernel-index dims map to the core's logical coordinate: gidx0 → row
// (y), gidx1 → col (x). Blackhole P100a: 10 rows × 13 cols (140 physical
// workers minus the dispatch column; harvested boards shrink x). Source:
// tt-metal `core_descriptors/blackhole_140_arch.yaml`.
//
// Device access is direct: every op calls the dlopened `tt_runtime_shim`
// on the calling thread under the pool/device lock. tt-metal serializes
// concurrent callers internally (`MeshDeviceImpl::api_mutex_`), so no
// channel or worker stands between Rust and the engine. Rust holds only
// opaque `*mut c_void` handles; all C++ objects stay in the shim
// (`extern "C"` facade, exceptions never cross).
//
// Only device lifecycle is threaded: one lifecycle thread creates every
// device, then destroys them and tears the context down at process exit.
// UMD's CHIP_IN_USE robust mutex is owned by the creating thread, so
// destroy + teardown must run there, never on the exiting thread.

use super::{ChunkId, Cmd, Dev, DeviceInfo, DeviceProgramId, GwsDim, Kernel, LaunchArg, Placement, Pool, Shard};
use crate::{
    DType, Map, Set,
    backend::DTypeCapability,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{GPUOp, Op, OpId, TTOp},
    shape::Dim,
    slab::Slab,
};
use nanoserde::DeJson;
use std::os::raw::{c_int, c_void};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;

// ── Global state ──────────────────────────────────────────────────────────────

// The single backend global: pools + devices, the engine binding, and the
// lifecycle exit sender. One `OnceLock` runs `init_global` exactly once —
// concurrent threads block until it returns and reuse the tables.
struct TTGlobal {
    backend: TTBackend,
    shim: Shim,
    exit: Option<Sender<LifeMsg>>,
}
static TT: OnceLock<TTGlobal> = OnceLock::new();

/// Pools and their devices, built together by [`initialize_backend`].
struct TTBackend {
    pools: Vec<Mutex<TTMemoryPool>>,
    devices: Vec<Mutex<TTDevice>>,
}

/// The single backend initializer: shim binding first (every later step
/// calls into it), then pools + devices, then the atexit hook (runs before
/// tt-metal's own handlers: it registers after theirs).
fn init_global() -> TTGlobal {
    let shim = load_shim();
    let (backend, exit) = initialize_backend();
    unsafe {
        atexit(Some(tt_atexit_shutdown));
    }
    TTGlobal { backend, shim, exit }
}

/// Engine binding for the unified global. The dlopen itself ran once in
/// `init_global`.
fn shim() -> &'static Shim {
    &TT.get_or_init(init_global).shim
}

/// Builds pools + devices together in one pass and publishes the combined
/// table. Runs once. The exit sender is `Some` exactly when devices were
/// created (nothing to tear down otherwise).
fn initialize_backend() -> (TTBackend, Option<Sender<LifeMsg>>) {
    let config = super::config();
    let debug_dev = super::debug_backends();
    let (pools, exit) = ensure_pool_table(&config.tenstorrent, debug_dev).expect("tenstorrent: pool table init failed");
    let mut devices = Vec::with_capacity(pools.len());
    for (idx, pool) in pools.iter().enumerate() {
        let pool_id = Pool::TT(u16::try_from(idx).expect("So many Tenstorrent devices..."));
        let guard = super::lock(pool_id, pool);
        devices.push(Mutex::new(TTDevice {
            device_info: Arc::new(guard.dev_info.clone()),
            memory_pool: pool_id,
            dev: guard.dev,
            programs: Slab::new(),
        }));
        drop(guard);
    }
    (TTBackend { pools, devices }, exit)
}

// ---------------------------------------------------------------------------
// Device lifecycle: one thread creates every device, parks, then destroys
// every device and tears the context down at process exit. Ops never touch
// this thread — they call the shim directly on the caller thread.
// ---------------------------------------------------------------------------

/// Message to the lifecycle thread. Shutdown carries a reply channel so the
/// exit hook waits for teardown instead of racing tt-metal's exit handlers.
enum LifeMsg {
    Shutdown { done: Sender<()> },
}

/// A created device reported to init: opaque handle + grid snapshot.
/// `Copy` so the lifecycle thread keeps its set for teardown while init
/// takes its own. The handle is only passed back to the shim under the
/// pool/device locks.
#[derive(Clone, Copy)]
struct NewDevice {
    dev: *mut c_void,
    rows: u32,
    cols: u32,
}

// Opaque handles cross threads (init reports, pool/device statics) but are
// only passed back to the shim under the pool/device locks — never
// dereferenced in Rust. Same discipline as `TTBuffer`.
unsafe impl Send for NewDevice {}

/// Spawn the lifecycle thread: loads its own engine binding, makes the
/// process's first tt-metal call here (`num_devices` creates the
/// MetalContext singleton), then creates each requested device, reports
/// over a oneshot, and parks until the exit hook signals. First call, chip
/// start, destroy, and context teardown all share this thread — UMD's
/// CHIP_IN_USE mutex is owned by the creating thread, so splitting init
/// across threads aborts at teardown. With no devices the thread reports
/// empty and exits (exit sender dropped: nothing to tear down).
fn spawn_lifecycle(want: Option<Vec<i32>>) -> Result<(Vec<NewDevice>, Option<Sender<LifeMsg>>), BackendError> {
    // Own binding (same shipped file, refcounted dlopen): lifecycle calls
    // run on this thread against this handle. Op sites use the global one.
    let shim = load_shim();
    let (init_tx, init_rx) = channel();
    let (exit_tx, exit_rx) = channel();
    thread::spawn(move || {
        // First tt-metal call in the process (see doc above); the spawning
        // thread makes none.
        let n_devices = unsafe { (shim.num_devices)() };
        if n_devices < 0 {
            let _ = init_tx.send(Err(cpp_err(&shim, ErrorStatus::Initialization)));
            return;
        }
        // Explicit ids are validated loudly; None means every visible device.
        // Zero visible devices is not an error: the backend contributes
        // nothing, like a missing CUDA/HIP driver.
        let ids: Vec<i32> = match &want {
            Some(want) => {
                for id in want {
                    if *id < 0 || *id >= n_devices {
                        let _ = init_tx.send(Err(BackendError {
                            status: ErrorStatus::Initialization,
                            context: format!("tenstorrent device id {id} out of range (0..{n_devices})").into(),
                        }));
                        return;
                    }
                }
                want.clone()
            }
            None => (0..n_devices).collect(),
        };
        if ids.is_empty() {
            let _ = init_tx.send(Ok(Vec::new()));
            return;
        }
        let mut devs = Vec::with_capacity(ids.len());
        for id in ids {
            let dev = unsafe { (shim.create_device)(id) };
            if dev.is_null() {
                let _ = init_tx.send(Err(cpp_err(&shim, ErrorStatus::Initialization)));
                return;
            }
            let mut rows = 0u32;
            let mut cols = 0u32;
            unsafe { (shim.get_grid_size)(dev, &raw mut rows, &raw mut cols) };
            if unsafe { (shim.has_error)() } {
                unsafe { (shim.destroy_device)(dev) };
                let _ = init_tx.send(Err(cpp_err(&shim, ErrorStatus::Initialization)));
                return;
            }
            devs.push(NewDevice { dev, rows, cols });
        }
        // Init takes a copy; the thread keeps its set for teardown.
        let _ = init_tx.send(Ok(devs.clone()));
        let Ok(LifeMsg::Shutdown { done }) = exit_rx.recv() else { return };
        for d in &devs {
            unsafe { (shim.destroy_device)(d.dev) };
            if unsafe { (shim.has_error)() } {
                eprintln!("tenstorrent lifecycle: {}", cpp_err(&shim, ErrorStatus::Initialization).context);
            }
        }
        unsafe { (shim.teardown_metal)() };
        if unsafe { (shim.has_error)() } {
            eprintln!("tenstorrent lifecycle: {}", cpp_err(&shim, ErrorStatus::Initialization).context);
        }
        let _ = done.send(());
    });

    // Fail loudly here (not lazily on first alloc) when init fails.
    let devs = init_rx
        .recv()
        .map_err(|_| BackendError {
            status: ErrorStatus::Initialization,
            context: "tenstorrent lifecycle thread exited during init".into(),
        })??;
    // An empty report means the thread created nothing and exited (its exit
    // receiver dropped): no teardown to own.
    let exit = if devs.is_empty() { None } else { Some(exit_tx) };
    Ok((devs, exit))
}

/// Translate the shim's thread-local error string into a `BackendError`.
/// Runs on the same thread that made the failing shim call (error state is
/// thread-local); every call site calls and reads back to back.
fn cpp_err(shim: &Shim, status: ErrorStatus) -> BackendError {
    let mut buf = [0i8; 2048];
    let n = unsafe { (shim.get_last_error)(buf.as_mut_ptr(), buf.len() as c_int) };
    let context = if n > 0 {
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf.as_ptr() as *const u8, n as usize)) }.to_string()
    } else {
        "unknown tt-metal error".into()
    };
    BackendError { status, context: context.into_boxed_str() }
}

// The tt-metal shim (`tt_runtime_shim.h`) is a cdylib shipped by zyx's
// build script under `$HOME/.config/zyx/` (versioned filename baked in as
// `ZYX_TT_SHIM`). It is dlopened here at first device use — user binaries
// therefore carry no tt-metal link dependency at all and need no loader
// path of their own. Same C ABI, same process, same calls as a static
// link; only the loading moved. Signatures mirror the header exactly.
// `Library` is `Send + Sync`, so one binding serves every thread; the
// lifecycle thread loads a second binding for its own calls (see
// `spawn_lifecycle`).
struct Shim {
    _lib: libloading::Library,
    get_last_error: unsafe extern "C" fn(*mut i8, c_int) -> c_int,
    has_error: unsafe extern "C" fn() -> bool,
    num_devices: unsafe extern "C" fn() -> c_int,
    create_device: unsafe extern "C" fn(c_int) -> *mut c_void,
    destroy_device: unsafe extern "C" fn(*mut c_void),
    teardown_metal: unsafe extern "C" fn(),
    get_grid_size: unsafe extern "C" fn(*mut c_void, *mut u32, *mut u32),
    alloc_buffer: unsafe extern "C" fn(*mut c_void, u64, u64) -> *mut c_void,
    free_buffer: unsafe extern "C" fn(*mut c_void, *mut c_void),
    write_buffer: unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, u64),
    read_buffer: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, u64),
    compile_program: unsafe extern "C" fn(
        *mut c_void,
        *const u8,
        usize,
        *const u8,
        usize,
        *const u8,
        usize,
        *const u32,
        *const u32,
        *const u32,
        *const u32,
        usize,
        *const u32,
        usize,
        *const u32,
        usize,
        *const u32,
        usize,
        u32,
        bool,
    ) -> u32,
    run_program: unsafe extern "C" fn(
        *mut c_void,
        u32,
        *const *mut c_void,
        usize,
        *const *mut c_void,
        usize,
        u32,
        u32,
        *const u32,
        *const u32,
        usize,
    ) -> bool,
    destroy_program: unsafe extern "C" fn(*mut c_void, u32),
}

/// Load the engine binding. Panics loudly (never a silent fallback) when
/// the shipped cdylib is absent — rebuild zyx with TT_METAL_RUNTIME_ROOT set.
fn load_shim() -> Shim {
    // Ship dir mirror of zyx/build.rs: XDG config dir. Keep in sync.
    let config_base = std::env::var("XDG_CONFIG_HOME").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("tenstorrent: neither XDG_CONFIG_HOME nor HOME is set");
        format!("{home}/.config")
    });
    let path = format!("{config_base}/zyx/{}", env!("ZYX_TT_SHIM"));
    let lib = unsafe { libloading::Library::new(&path) }
        .unwrap_or_else(|e| panic!("tenstorrent: cannot load TT shim {path}: {e}; rebuild zyx with TT_METAL_RUNTIME_ROOT set"));
    macro_rules! bind {
        ($name:literal, $ty:ty) => {
            *unsafe { lib.get::<$ty>(concat!($name, "\0").as_bytes()) }
                .unwrap_or_else(|e| panic!("tenstorrent: shim {path} lacks symbol {}: {e}", $name))
        };
    }
    Shim {
        get_last_error: bind!("get_last_error", unsafe extern "C" fn(*mut i8, c_int) -> c_int),
        has_error: bind!("has_error", unsafe extern "C" fn() -> bool),
        num_devices: bind!("num_devices", unsafe extern "C" fn() -> c_int),
        create_device: bind!("create_device", unsafe extern "C" fn(c_int) -> *mut c_void),
        destroy_device: bind!("destroy_device", unsafe extern "C" fn(*mut c_void)),
        teardown_metal: bind!("teardown_metal", unsafe extern "C" fn()),
        get_grid_size: bind!("get_grid_size", unsafe extern "C" fn(*mut c_void, *mut u32, *mut u32)),
        alloc_buffer: bind!("alloc_buffer", unsafe extern "C" fn(*mut c_void, u64, u64) -> *mut c_void),
        free_buffer: bind!("free_buffer", unsafe extern "C" fn(*mut c_void, *mut c_void)),
        write_buffer: bind!("write_buffer", unsafe extern "C" fn(*mut c_void, *mut c_void, *const c_void, u64)),
        read_buffer: bind!("read_buffer", unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, u64)),
        compile_program: bind!(
            "compile_program",
            unsafe extern "C" fn(
                *mut c_void,
                *const u8,
                usize,
                *const u8,
                usize,
                *const u8,
                usize,
                *const u32,
                *const u32,
                *const u32,
                *const u32,
                usize,
                *const u32,
                usize,
                *const u32,
                usize,
                *const u32,
                usize,
                u32,
                bool,
            ) -> u32
        ),
        run_program: bind!(
            "run_program",
            unsafe extern "C" fn(
                *mut c_void,
                u32,
                *const *mut c_void,
                usize,
                *const *mut c_void,
                usize,
                u32,
                u32,
                *const u32,
                *const u32,
                usize,
            ) -> bool
        ),
        destroy_program: bind!("destroy_program", unsafe extern "C" fn(*mut c_void, u32)),
        _lib: lib,
    }
}

// Process exit hook (libc atexit, declared directly — no libc crate).
unsafe extern "C" {
    fn atexit(cb: Option<unsafe extern "C" fn()>) -> c_int;
}
/// Exit-time shutdown hook: signals the lifecycle thread (which destroys
/// every device and tears the context down before tt-metal's own handlers
/// run — LIFO registration) and bounds the wait. `static`s never drop, so
/// without this open devices would outlive `MetalContext`, whose destructor
/// throws on open devices.
extern "C" fn tt_atexit_shutdown() {
    let Some(g) = TT.get() else { return };
    let Some(exit) = g.exit.as_ref() else { return };
    let (tx, rx) = channel();
    if exit.send(LifeMsg::Shutdown { done: tx }).is_err() {
        return;
    }
    let _ = rx.recv_timeout(std::time::Duration::from_secs(30));
}

// ---------------------------------------------------------------------------
// DRAM size lookup
// ---------------------------------------------------------------------------

/// GDDR6 sizes for known Blackhole PCI subsystem IDs (total per board).
/// P100/P100A: 7 usable channels → 28 GB; P150: 8 → 32 GB; P300: 2×8 → 64 GB.
/// The driver does not expose GDDR6 capacity, so this table is the fallback.
/// Sources: docs.tenstorrent.com/aibs/blackhole, tt-umd `board_upi_map`,
/// tt-metal `blackhole_140_arch.yaml` (dram_bank_size ≈ 4 GB).
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

fn detect_dram_bytes() -> u64 {
    if let Ok(entries) = std::fs::read_dir("/sys/bus/pci/devices") {
        let found = entries.flatten().find_map(|entry| {
            let vendor = std::fs::read_to_string(entry.path().join("vendor")).ok()?;
            if vendor.trim() != "0x1e52" {
                return None;
            }
            let subsys = std::fs::read_to_string(entry.path().join("subsystem_device")).ok()?;
            u16::from_str_radix(subsys.trim().trim_start_matches("0x"), 16).ok()
        });
        if let Some(id) = found {
            for &(sid, _name, size) in DRAM_SIZE_TABLE {
                if sid == id {
                    return size;
                }
            }
        }
    }
    64u64 * 1024 * 1024 * 1024
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
// Per-buffer tracking: opaque handle beside Rust's slab entry.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct TTBuffer {
    handle: *mut c_void,
    pub(crate) size: u64,
}

// Opaque FFI handles are Send/Sync: only the shim dereferences them (under
// the pool/device locks), and they point at C-managed objects.
unsafe impl Send for TTBuffer {}
unsafe impl Sync for TTBuffer {}

// ---------------------------------------------------------------------------
// Memory pool — device DRAM buffers. ChunkId is chosen by Rust's slab;
// the opaque handle beside it is passed back to the shim under this
// pool's lock. Allocation reuses the free_set first (best-fit, stable
// addresses); fresh device memory is requested only on a miss.
// ---------------------------------------------------------------------------

fn backend() -> Result<(&'static Vec<Mutex<TTMemoryPool>>, &'static Vec<Mutex<TTDevice>>), BackendError> {
    let g = TT.get_or_init(init_global);
    Ok((&g.backend.pools, &g.backend.devices))
}

pub(super) fn pool(id: u16) -> Result<&'static Mutex<TTMemoryPool>, BackendError> {
    backend()?.0.get(id as usize).ok_or_else(|| no_pool(id))
}

pub(super) fn pool_count() -> u16 {
    backend().map(|(pools, _)| pools.len() as u16).unwrap_or(0)
}

fn no_pool(id: u16) -> BackendError {
    BackendError { status: ErrorStatus::Initialization, context: format!("Pool::TT({id}) is not available").into() }
}

#[derive(Debug)]
pub struct TTMemoryPool {
    pub(crate) buffers: Slab<ChunkId, TTBuffer>,
    free_set: Set<ChunkId>,
    free_bytes: Dim,
    /// Opaque tt-metal device handle, shared with this pool's `TTDevice`.
    /// Passed back to the shim under this pool's lock; never dereferenced.
    dev: *mut c_void,
    pub(crate) dev_info: DeviceInfo,
}

// See `NewDevice`: the handle only travels back to the shim under this
// pool's lock.
unsafe impl Send for TTMemoryPool {}

fn ensure_pool_table(
    config: &TTConfig,
    debug_dev: bool,
) -> Result<(Vec<Mutex<TTMemoryPool>>, Option<Sender<LifeMsg>>), BackendError> {
    let mut pools: Vec<Mutex<TTMemoryPool>> = Vec::new();
    // Configured out: empty tables, no shim calls, no threads.
    if let Some(device_ids) = &config.device_ids
        && device_ids.is_empty()
    {
        if debug_dev {
            println!("[tenstorrent] configured out");
        }
        return Ok((pools, None));
    }

    // The lifecycle thread makes the process's first tt-metal call and owns
    // create/destroy/teardown on one thread (see `spawn_lifecycle`); this
    // thread makes none.
    let (devs, exit) = spawn_lifecycle(config.device_ids.clone())?;
    let dram_bytes = detect_dram_bytes();

    // F8E5M2 has no Blackhole DataFormat: not a capable dtype, codegen rejects it.
    let mut dtype_capability = [DTypeCapability::all(); DType::N_DTYPES];
    dtype_capability[DType::F8E5M2 as usize] = DTypeCapability::ZERO;
    for (idx, d) in devs.iter().enumerate() {
        if debug_dev {
            println!("[tenstorrent] device {idx} initialized");
            println!("[tenstorrent] device total memory: {} MB", dram_bytes / (1024 * 1024));
            println!("[tenstorrent] tensix grid {} rows x {} cols", d.rows, d.cols);
        }
        pools.push(Mutex::new(TTMemoryPool {
            buffers: Slab::new(),
            free_set: Set::default(),
            free_bytes: Dim::from(dram_bytes as i64),
            dev: d.dev,
            dev_info: DeviceInfo {
                compute: 200_000_000_000_000, // ~200 TFLOPS BF16
                // Grid axes only (gidx0 row, gidx1 col); TT launches at most
                // 2 group axes, and the launch path rejects more.
                max_global_work_dims: vec![Dim::from(d.rows), Dim::from(d.cols)],
                max_local_threads: 1024,
                max_local_work_dims: vec![1, 1024, 1],
                preferred_vector_size: 32,
                local_mem_size: 1_500_000, // 1.5 MB L1 per Tensix core
                max_register_bytes: 128,
                tensor_cores: true,
                warp_size: 1, // Tensix has no SIMT warps
                cc: [0, 0],
                dtype_capability,
                has_native_exp2: false,
                supported_vec_lens: vec![32],
                tenstorrent: true,
                tile: [32, 32],
                tile_sizes: vec![[32, 32]],
                wmma_layouts: vec![],
                num_circular_buffers: 32, // architectural CB0-CB31
                has_openmp: false,
            },
        }));
    }

    Ok((pools, exit))
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

impl TTMemoryPool {
    pub fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<ChunkId, BackendError> {
        // Best-fit from the free_set first (stable addresses — those bytes
        // were already paid for); else fresh from the device.
        let best = self
            .free_set
            .iter()
            .filter_map(|id| {
                let len = self.buffers[*id].size as Dim;
                (len >= bytes).then_some((len, *id))
            })
            .min();
        if let Some((_, id)) = best {
            self.free_set.remove(&id);
            return Ok(id);
        }
        let bytes_u64: u64 = u64::try_from(bytes).map_err(|_| BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: "allocation size exceeds 64-bit".into(),
        })?;
        if bytes > self.free_bytes {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "out of device memory".into() });
        }
        let chunk_id = self.buffers.push(TTBuffer { handle: std::ptr::null_mut(), size: 0 });
        let shim = shim();
        let handle = unsafe { (shim.alloc_buffer)(self.dev, bytes_u64, 2048) };
        if handle.is_null() {
            return Err(cpp_err(shim, ErrorStatus::MemoryAllocation));
        }
        // Device memory is page-granular; the handle covers whole pages.
        let size = (bytes_u64 + 4095) / 4096 * 4096;
        debug_assert!(!handle.is_null(), "shim returned null handle without error");
        self.buffers[chunk_id] = TTBuffer { handle, size };
        self.free_bytes -= size as Dim;
        Ok(chunk_id)
    }

    /// Put a buffer on the free list for stable-address reuse. Frees nothing;
    /// device memory returns only via [`TTMemoryPool::dispose`]. Safe without
    /// a sync: launches are synchronous, so no launch can still be using the
    /// buffer when its placement drops.
    pub fn release(&mut self, buffer_id: ChunkId) {
        if !self.buffers.contains_id(buffer_id) {
            debug_assert!(false, "release of unknown TT buffer {buffer_id:?}");
            return;
        }
        debug_assert!(!self.free_set.contains(&buffer_id), "double release of TT buffer {buffer_id:?}");
        self.free_set.insert(buffer_id);
    }

    /// Claim existing allocations, all or nothing: every id must be on the
    /// free list (stable addresses) or nothing is claimed.
    pub fn try_reuse_allocations(&mut self, buffer_ids: &Set<ChunkId>) -> bool {
        let ok = buffer_ids.iter().all(|id| self.free_set.contains(id));
        if ok {
            for id in buffer_ids {
                self.free_set.remove(id);
            }
        }
        ok
    }

    /// Free every buffer on the free list back to the device at once.
    pub fn dispose(&mut self) {
        let shim = shim();
        for id in core::mem::take(&mut self.free_set) {
            let buf = unsafe { self.buffers.remove_and_return(id) };
            self.free_bytes += buf.size as Dim;
            unsafe { (shim.free_buffer)(self.dev, buf.handle) };
        }
    }

    /// Blocking upload: the shim runs the transfer to completion.
    pub fn host_to_pool(&mut self, src: &[u8], dst: ChunkId) -> Result<(), BackendError> {
        let buf = self
            .buffers
            .get(dst)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyH2P, context: "invalid buffer id".into() })?;
        let len = src.len().min(buf.size as usize);
        let handle = buf.handle;
        let shim = shim();
        unsafe { (shim.write_buffer)(self.dev, handle, src.as_ptr() as *const c_void, len as u64) };
        if unsafe { (shim.has_error)() } {
            return Err(cpp_err(shim, ErrorStatus::MemoryCopyH2P));
        }
        Ok(())
    }

    /// Blocking download: the shim runs the transfer to completion.
    pub fn pool_to_host(&mut self, src: ChunkId, dst: &mut [u8]) -> Result<(), BackendError> {
        let buf = self
            .buffers
            .get(src)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyP2H, context: "invalid buffer id".into() })?;
        let cap = dst.len().min(buf.size as usize);
        let handle = buf.handle;
        let shim = shim();
        unsafe { (shim.read_buffer)(self.dev, handle, dst.as_mut_ptr() as *mut c_void, cap as u64) };
        if unsafe { (shim.has_error)() } {
            return Err(cpp_err(shim, ErrorStatus::MemoryCopyP2H));
        }
        Ok(())
    }

    pub fn buffer_handle(&self, buffer_id: ChunkId) -> Result<*mut c_void, BackendError> {
        if self.buffers.contains_id(buffer_id) {
            Ok(self.buffers[buffer_id].handle)
        } else {
            Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "invalid buffer id".into() })
        }
    }
}

#[derive(Debug)]
struct TTProgram {
    /// Shim id (per-device program cache index).
    shim: u32,
    /// Global param count (inputs) and GlobalMut count (outputs).
    n_inputs: u32,
    n_outputs: u32,
    /// Group-range lengths in axis order: Const resolved at compile,
    /// Param(ordinal) resolved from the launch args.
    gws: Vec<GwsDim>,
    /// Tensix grid rows/cols at compile: dynamic launch sizes check here.
    max_grid: [u32; 2],
}

#[derive(Debug)]
pub struct TTDevice {
    device_info: Arc<DeviceInfo>,
    memory_pool: Pool,
    /// Opaque tt-metal device handle, shared with this device's pool.
    /// Passed back to the shim under this device's lock; never dereferenced.
    dev: *mut c_void,
    programs: Slab<DeviceProgramId, TTProgram>,
}

// See `NewDevice`: the handle only travels back to the shim under this
// device's lock.
unsafe impl Send for TTDevice {}

impl TTDevice {
    pub fn info(&self) -> Arc<DeviceInfo> {
        self.device_info.clone()
    }

    pub fn free_compute(&self) -> u128 {
        // Zero until TT codegen covers the full op/grid space: Dev::Auto
        // ranks by free_compute, and an auto-selected TT device fails
        // compilation (e.g. grid axis restrictions) where C/CUDA succeed.
        // Explicit Dev::TT is unaffected.
        0
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        // Render to descriptor; decode head positions. CB ids, section
        // params, param ordinals, and io counts arrive in the descriptor
        // tables; what stays here is launch-side assembly: the decoded grid,
        // the runtime CB config split, and the program compile call.
        let rendered = kernel.render()?;
        let mut order = rendered.ops_in_order();
        let Some(Op::TT(TTOp::ProgramDesc(desc))) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "head op is not ProgramDesc".into() });
        };
        let Some(GPUOp::Params(_params)) = order.next().and_then(Op::as_gpu) else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "second op is not Params".into() });
        };
        let Some(Op::TT(TTOp::TensixGrid(grid))) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "third op is not TensixGrid".into() });
        };
        let Some(Op::Source(reader)) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "fourth op is not Source".into() });
        };
        let Some(Op::TT(TTOp::EndReader)) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "fifth op is not EndReader".into() });
        };
        let Some(Op::Source(compute)) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "sixth op is not Source".into() });
        };
        let Some(Op::TT(TTOp::EndCompute)) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "seventh op is not EndCompute".into() });
        };
        let Some(Op::Source(writer)) = order.next() else {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "eighth op is not Source".into() });
        };

        // Per-section params (0 = reader, 1 = compute, 2 = writer): param
        // ordinals in ascending head order — Global + Variable interleaved
        // first, GlobalMut at the tail — then the core's grid coordinates
        // gidx0 (row) / gidx1 (col). See `Kernel::generate_tenstorrent` and
        // `tt_runtime_shim.cpp` `section_rt_args` for the consumption side.
        let n_params = desc.n_params;
        // Tensix grid from the descriptor (rank <= 2 enforced at render):
        // Param-backed lengths resolve at launch from the Variable arg.
        let gws: Vec<GwsDim> = grid.to_vec();

        let reader = reader.as_str().as_bytes();
        let reader_params: &[u32] = &desc.reader_params;
        let compute = compute.as_str().as_bytes();
        let compute_params: &[u32] = &desc.compute_params;
        let writer = writer.as_str().as_bytes();
        let writer_params: &[u32] = &desc.writer_params;
        if debug_asm {
            eprintln!("[tenstorrent] reader:\n{}", String::from_utf8_lossy(reader));
            eprintln!("[tenstorrent] compute:\n{}", String::from_utf8_lossy(compute));
            eprintln!("[tenstorrent] writer:\n{}", String::from_utf8_lossy(writer));
        }

        // DST geometry follows the codegen mode: fp32 iff the kernel
        // touches F32 tiles (any F32 tile in DST, per the typecast
        // header).
        let fp32_dest_acc_en = desc.fp32;

        // Snapshot the grid for the launch-time bounds check (dynamic
        // sizes only; const sizes already failed at compile above).
        let mg = &self.device_info.max_global_work_dims;
        let max_grid = [
            u32::try_from(mg[0]).map_err(|_| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent grid rows do not fit u32".into(),
            })?,
            u32::try_from(mg[1]).map_err(|_| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent grid cols do not fit u32".into(),
            })?,
        ];

        let cb_indices: Vec<u32> = (0..desc.cb_config.len() as u32).collect();
        let cb_formats: Vec<u32> = desc.cb_config.iter().map(|c| c.format).collect();
        let cb_tile_bytes: Vec<u32> = desc.cb_config.iter().map(|c| c.tile_bytes).collect();
        let cb_num_tiles: Vec<u32> = desc.cb_config.iter().map(|c| c.num_tiles).collect();
        let shim = shim();
        let prog = unsafe {
            (shim.compile_program)(
                self.dev,
                reader.as_ptr(),
                reader.len(),
                compute.as_ptr(),
                compute.len(),
                writer.as_ptr(),
                writer.len(),
                cb_indices.as_ptr(),
                cb_formats.as_ptr(),
                cb_tile_bytes.as_ptr(),
                cb_num_tiles.as_ptr(),
                cb_indices.len(),
                reader_params.as_ptr(),
                reader_params.len(),
                compute_params.as_ptr(),
                compute_params.len(),
                writer_params.as_ptr(),
                writer_params.len(),
                n_params,
                fp32_dest_acc_en,
            )
        };
        if prog == u32::MAX {
            return Err(cpp_err(shim, ErrorStatus::KernelCompilation));
        }
        Ok(self.programs.push(TTProgram { shim: prog, n_inputs: desc.n_inputs, n_outputs: desc.n_outputs, gws, max_grid }))
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        if self.programs.contains_id(program_id) {
            let prog = unsafe { self.programs.remove_and_return(program_id) };
            let shim = shim();
            unsafe { (shim.destroy_program)(self.dev, prog.shim) };
        }
    }

    pub fn launch(&mut self, program_id: DeviceProgramId, pool_handle: Pool, args: &[LaunchArg]) -> Result<(), BackendError> {
        debug_assert_eq!(pool_handle, self.memory_pool);
        let Pool::TT(id) = pool_handle else {
            unreachable!("TT launch with non-TT pool")
        };
        let pool_arc = pool(id).expect("launch on unavailable TT pool");
        let memory_pool = super::lock(pool_handle, &pool_arc);
        let prog = if self.programs.contains_id(program_id) {
            &self.programs[program_id]
        } else {
            return Err(BackendError { status: ErrorStatus::KernelLaunch, context: "invalid program id".into() });
        };
        let program = prog.shim;

        let n_inputs = prog.n_inputs as usize;
        let n_outputs = prog.n_outputs as usize;

        // One arg per param, head order: Global + Variable interleaved,
        // GlobalMut at the tail. Kinds derive from the args: Variable ->
        // Variable; Buffer above the GlobalMut tail -> Global.
        if args.len() < n_inputs + n_outputs {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("expected {} args ({} inputs + {} outputs), got {}", n_inputs + n_outputs, n_inputs, n_outputs, args.len())
                    .into(),
            });
        }
        let n_params = args.len();
        debug_assert!(n_params >= n_inputs + n_outputs, "tt launch: {n_params} args for {n_inputs} inputs + {n_outputs} outputs");

        let globalmut_start = n_params - n_outputs;
        let mut src_handles: Vec<*mut c_void> = Vec::with_capacity(n_inputs);
        let mut dst_handles: Vec<*mut c_void> = Vec::with_capacity(n_outputs);
        // Variable params: (ordinal, value) pairs.
        let mut vars: Vec<(u32, u32)> = Vec::new();
        for (ordinal, arg) in args.iter().enumerate() {
            let ordinal = ordinal as u32;
            match arg {
                LaunchArg::Buffer { chunk, .. } => {
                    let handle = memory_pool.buffer_handle(*chunk).map_err(|e| BackendError {
                        status: ErrorStatus::KernelLaunch,
                        context: format!("param {ordinal} handle: {e}").into(),
                    })?;
                    if ordinal as usize >= globalmut_start {
                        dst_handles.push(handle);
                    } else {
                        src_handles.push(handle);
                    }
                }
                LaunchArg::Variable(value) => {
                    debug_assert!(
                        (ordinal as usize) < globalmut_start,
                        "tt launch: Variable arg at ordinal {ordinal} inside GlobalMut tail"
                    );
                    let dim = value.as_dim().expect("variable launch arg has a concrete dim");
                    let v = u32::try_from(dim).map_err(|_| BackendError {
                        status: ErrorStatus::KernelLaunch,
                        context: format!("param {ordinal} variable value {dim} does not fit u32").into(),
                    })?;
                    vars.push((ordinal, v));
                }
            }
        }
        debug_assert_eq!(src_handles.len(), n_inputs, "tt launch: {} src args for {} Global params", src_handles.len(), n_inputs);
        debug_assert_eq!(dst_handles.len(), n_outputs, "tt launch: {} dst args for {} outputs", dst_handles.len(), n_outputs);

        // Grid dims from the group-range lengths (gws), in axis order.
        // Variable ordinals index the launch args (head order), which always
        // carry a Variable value at a Variable param's position.
        let mut grid_dims = [1u32, 1u32];
        if prog.gws.len() > 2 {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("tenstorrent supports at most 2 group axes, got {}", prog.gws.len()).into(),
            });
        }
        for (axis, g) in prog.gws.iter().enumerate() {
            let dim = g.eval(&mut |ordinal| {
                vars.iter()
                    .find(|(o, _)| *o == ordinal as u32)
                    .map(|(_, v)| Dim::from(*v))
                    .expect("gws param ordinal has no Variable launch arg")
            });
            grid_dims[axis] = u32::try_from(dim).map_err(|_| BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("gws axis {axis} dim {dim} does not fit u32").into(),
            })?;
            // Dynamic sizes skip the compile check: bound them here.
            if grid_dims[axis] > prog.max_grid[axis] {
                return Err(BackendError {
                    status: ErrorStatus::KernelLaunch,
                    context: format!("tenstorrent grid axis {axis} size {} exceeds device grid {}", grid_dims[axis], prog.max_grid[axis])
                        .into(),
                });
            }
        }
        let shim = shim();
        let ok = unsafe {
            let ordinals = vars.iter().map(|(o, _)| *o).collect::<Vec<u32>>();
            let values = vars.iter().map(|(_, v)| *v).collect::<Vec<u32>>();
            (shim.run_program)(
                self.dev,
                program,
                src_handles.as_ptr(),
                src_handles.len(),
                dst_handles.as_ptr(),
                dst_handles.len(),
                grid_dims[0],
                grid_dims[1],
                ordinals.as_ptr(),
                values.as_ptr(),
                vars.len(),
            )
        };
        if !ok {
            return Err(cpp_err(shim, ErrorStatus::KernelLaunch));
        }
        Ok(())
    }

    /// Timed launch for autotune. Returns the kernel's run time in nanos.
    /// Tenstorrent launches are synchronous (the shim blocks through
    /// Finish), so a wall-clock bracket is already an uncontended
    /// measurement.
    pub fn launch_timed(&mut self, program_id: DeviceProgramId, args: &[LaunchArg]) -> Result<u64, BackendError> {
        let start = std::time::Instant::now();
        self.launch(program_id, self.memory_pool, args)?;
        Ok(start.elapsed().as_nanos() as u64)
    }

    /// Copy a transfer into this device's pool: host uploads go through the
    /// blocking shim upload, CUDA sources stage through a host temp (no peer
    /// DMA between vendors), TT same-device is a no-op and cross-device
    /// routes through host. Single-shard placements only.
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
        let Pool::TT(my_id) = self.memory_pool else {
            unreachable!("TT copy on a non-TT device")
        };
        match src_shard.pool {
            Pool::Host => {
                // The host lock is held across the blocking shim upload: the
                // source pointer stays valid, and the shim never takes it.
                let host = super::host::pool();
                let hpool = super::lock(Pool::Host, host);
                let src_ptr = hpool.get_buffer(src_shard.chunk).as_ptr();
                let src_bytes = unsafe { std::slice::from_raw_parts(src_ptr, bytes as usize) };
                let tt = pool(my_id)?;
                let mut tpool = super::lock(self.memory_pool, tt);
                tpool.host_to_pool(src_bytes, dst_shard.chunk)?;
            }
            Pool::Cuda(id) => {
                // No peer DMA between vendors: stage through a host temp.
                let host = super::host::pool();
                let mut hpool = super::lock(Pool::Host, host);
                let tmp = hpool.allocate(bytes)?;
                let tmp_ptr = hpool.buffer_ptr_mut(tmp);
                {
                    let cuda = super::cuda::pool(id)?;
                    let mut cpool = super::lock(Pool::Cuda(id), cuda);
                    cpool.pool_to_host(src_shard.chunk, unsafe { std::slice::from_raw_parts_mut(tmp_ptr, bytes as usize) })?;
                }
                let tmp_bytes = unsafe { std::slice::from_raw_parts(tmp_ptr, bytes as usize) };
                let tt = pool(my_id)?;
                let mut tpool = super::lock(self.memory_pool, tt);
                let r = tpool.host_to_pool(tmp_bytes, dst_shard.chunk);
                hpool.release(tmp);
                r?;
            }
            Pool::TT(other_id) => {
                if other_id == my_id {
                    return Ok(());
                }
                let mut buf = vec![0u8; bytes as usize];
                let other = pool(other_id)?;
                super::lock(Pool::TT(other_id), other).pool_to_host(src_shard.chunk, &mut buf)?;
                let tt = pool(my_id)?;
                let mut tpool = super::lock(self.memory_pool, tt);
                tpool.host_to_pool(&buf, dst_shard.chunk)?;
            }
            p => {
                // Other device pools route through host memory.
                let mut buf = vec![0u8; bytes as usize];
                p.pool_to_host(src_shard.chunk, &mut buf)?;
                let tt = pool(my_id)?;
                let mut tpool = super::lock(self.memory_pool, tt);
                tpool.host_to_pool(&buf, dst_shard.chunk)?;
            }
        }
        Ok(())
    }
}

/// Preplanned Tenstorrent partition: ordered commands plus per-command death
/// lists (slots whose final read is that command and which nothing later
/// needs). Replay allocates unbound defs on the fly and launches programs
/// back-to-back — launches are synchronous, so program order is the order.
#[derive(Debug)]
pub(crate) struct TTPartition {
    pub(crate) cmds: Vec<Cmd>,
    deaths: Vec<Vec<OpId>>,
    pub(crate) dev: u16,
}

impl TTDevice {
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
                    // Buffers name their chunk (offsets are 0 — no sub-chunk
                    // views exist yet); variables carry their scalar value.
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
                    // Dry run skips execution (buffers stay uninitialized).
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
