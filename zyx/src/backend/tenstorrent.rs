// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
//
// Tenstorrent backend for zyx.
//
// # Grid indexing (gidx)
//
// The tensix cores on a device form a 2D grid. Two kernel-index
// dimensions are available, mapped to the core's logical coordinate:
//
//   - gidx0 → core row (y)
//   - gidx1 → core column (x)
//
// For Blackhole P100a the logical worker grid is 10 rows × 13 columns
// (x ∈ 0..13, y ∈ 0..10), giving 130 cores total (140 physical workers
// minus the 10-core dispatch column; harvested boards shrink x).
// Source: tt-metal `core_descriptors/blackhole_140_arch.yaml` (range
// end [12, 9], sized (end-start)+1 in `core_descriptor.cpp`). A
// single-core launch uses `gidx0 = 0, gidx1 = 0` (also written `{0, 0}`
// in CoreCoord notation).
//
// Device access goes through one worker thread per device, which owns the
// tt-metal `MeshDevice` and serializes every call over an mpsc channel.
// Rust holds only opaque `*mut c_void` handles; all C++ objects stay in
// the `tt_runtime_shim` (`extern "C"` facade, exceptions never cross).

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

// The single backend global: pools + devices. One `OnceLock` runs
// `init_global` exactly once — concurrent threads block until it returns
// and reuse the tables. Without the Once, two threads can both spawn a
// device worker and race the same device's init (deadlock).
struct TTGlobal {
    backend: TTBackend,
}
static TT: OnceLock<TTGlobal> = OnceLock::new();

/// Pools and their devices, built together by [`initialize_backend`].
struct TTBackend {
    pools: Vec<Mutex<TTMemoryPool>>,
    devices: Vec<Mutex<TTDevice>>,
}

/// The single backend initializer: pools + devices, then the atexit hook
/// (registered once, runs before tt-metal's own handlers since it registers
/// after theirs). Device workers load their own shim binding (see `spawn`)
/// — the global never touches the engine.
fn init_global() -> TTGlobal {
    let backend = initialize_backend();
    unsafe {
        libc::atexit(tt_atexit_shutdown);
    }
    TTGlobal { backend }
}

/// The single backend initializer: builds pools + devices together in
/// one pass and publishes the combined table. Runs once.
fn initialize_backend() -> TTBackend {
    let config = super::config();
    let debug_dev = super::debug_backends();
    let pools = ensure_pool_table(&config.tenstorrent, debug_dev).expect("tenstorrent: pool table init failed");
    let mut devices = Vec::with_capacity(pools.len());
    for (idx, pool) in pools.iter().enumerate() {
        let pool_id = Pool::TT(u16::try_from(idx).expect("So many Tenstorrent devices..."));
        let guard = super::lock(pool_id, pool);
        devices.push(Mutex::new(TTDevice {
            device_info: Arc::new(guard.dev_info.clone()),
            memory_pool: pool_id,
            worker: guard.worker.clone(),
            programs: Slab::new(),
        }));
        drop(guard);
    }
    TTBackend { pools, devices }
}

// ---------------------------------------------------------------------------
// Worker thread: owns the device, buffers, and the command channel.
// ---------------------------------------------------------------------------

// The command payload carries opaque FFI handles (*mut c_void) owned by the
// worker. They are Send/Sync because the worker is the only thread that
// dereferences them, and they point to C-managed objects.
unsafe impl Send for Command {}
unsafe impl Sync for Command {}

/// Command sent to the device worker. All tt-metal work happens on the worker
/// thread; replies are sent back on the embedded reply channel.
enum Command {
    /// Query the tensix grid (rows, cols).
    Grid {
        reply: Sender<Result<(u32, u32), BackendError>>,
    },
    /// Allocate device DRAM. The ChunkId is chosen by the caller (Rust's
    /// slab); the worker returns the real size and the opaque handle.
    Allocate {
        bytes: u64,
        chunk_id: ChunkId,
        reply: Sender<Result<(*mut c_void, u64), BackendError>>,
    },
    /// Free a buffer by Rust-side ChunkId (handle freed on the worker).
    Free {
        buffer_id: ChunkId,
        handle: *mut c_void,
        reply: Sender<()>,
    },
    /// Host -> device: enqueue a write and block until it completes. The `src`
    /// vector is owned by the caller; the worker copies it into tt-metal.
    Write {
        buffer_id: ChunkId,
        handle: *mut c_void,
        src: Vec<u8>,
        reply: Sender<Result<(), BackendError>>,
    },
    /// Device -> host: enqueues a blocking read directly into the caller's
    /// buffer and returns the number of bytes copied. The caller blocks on
    /// the reply, so `dst` stays valid for the whole call.
    Read {
        buffer_id: ChunkId,
        handle: *mut c_void,
        dst: *mut u8,
        dst_len: usize,
        reply: Sender<Result<usize, BackendError>>,
    },
    /// Cache the compilation of a kernel (sources + CB config + param ordinals).
    /// Returns the shim's per-device program id.
    Compile {
        reader_src: Vec<u8>,
        compute_src: Vec<u8>,
        writer_src: Vec<u8>,
        cb_idx: Vec<u32>,
        cb_fmt: Vec<u32>,
        cb_tile_bytes: Vec<u32>,
        cb_num_tiles: Vec<u32>,
        reader_params: Vec<u32>,
        compute_params: Vec<u32>,
        writer_params: Vec<u32>,
        n_params: u32,
        fp32_dest_acc_en: bool,
        reply: Sender<Result<u32, BackendError>>,
    },
    /// Launch a cached program (shim id). Handles are passed directly (Rust
    /// owns the slab keys; the worker owns the handles).
    Launch {
        program: u32,
        src_handles: Vec<*mut c_void>,
        dst_handles: Vec<*mut c_void>,
        grid_dims: [u32; 2],
        vars: Vec<(u32, u32)>,
        reply: Sender<Result<(), BackendError>>,
    },
    /// Release a cached program (shim id) back to the device.
    DestroyProgram {
        program: u32,
        reply: Sender<Result<(), BackendError>>,
    },
    /// Drain the channel and shut the worker down.
    Shutdown { reply: Sender<()> },
}

/// Per-buffer tracking on the worker: one opaque handle per Rust-side ChunkId.
struct WorkerBuffer {
    size: u64,
}

/// Device worker: owns the device and the buffer table. Programs live in the
/// shim's per-device cache; the worker only forwards their ids.
struct TTWorker {
    dev: *mut c_void,
    buffers: Map<ChunkId, WorkerBuffer>,
}

/// Worker handle carried by the main thread. Senders are cloned cheaply; each
/// clone routes to the same worker.
#[derive(Debug)]
pub(crate) struct RuntimeWorker {
    sender: Arc<Sender<Command>>,
}

/// Translate the shim's thread-local error string into a `BackendError` of the
/// given status. Must run on the same thread that made the failing shim call
/// (the worker thread); every call site below upholds this.
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
struct Shim {
    _lib: libloading::Library,
    get_last_error: unsafe extern "C" fn(*mut i8, c_int) -> c_int,
    has_error: unsafe extern "C" fn() -> bool,
    create_device: unsafe extern "C" fn() -> *mut c_void,
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

/// Load the engine binding. Each device worker calls this once at thread
/// start and owns the result for the thread's lifetime — every shim call
/// in this file runs on a worker thread, so no sharing primitive is
/// needed. Panics loudly (never a silent fallback) when the shipped cdylib
/// is absent — rebuild zyx with TT_METAL_ROOT set.
fn load_shim() -> Shim {
    // Ship dir mirror of zyx/build.rs: XDG config dir. Keep in sync.
    let config_base = std::env::var("XDG_CONFIG_HOME").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("tenstorrent: neither XDG_CONFIG_HOME nor HOME is set");
        format!("{home}/.config")
    });
    let path = format!("{config_base}/zyx/{}", env!("ZYX_TT_SHIM"));
    let lib = unsafe { libloading::Library::new(&path) }
        .unwrap_or_else(|e| panic!("tenstorrent: cannot load TT shim {path}: {e}; rebuild zyx with TT_METAL_ROOT set"));
    macro_rules! bind {
        ($name:literal, $ty:ty) => {
            *unsafe { lib.get::<$ty>(concat!($name, "\0").as_bytes()) }
                .unwrap_or_else(|e| panic!("tenstorrent: shim {path} lacks symbol {}: {e}", $name))
        };
    }
    Shim {
        get_last_error: bind!("get_last_error", unsafe extern "C" fn(*mut i8, c_int) -> c_int),
        has_error: bind!("has_error", unsafe extern "C" fn() -> bool),
        create_device: bind!("create_device", unsafe extern "C" fn() -> *mut c_void),
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

/// Exit-time shutdown hook.
///
/// `static` items never drop, so the worker's `destroy_device` would never
/// run before tt-metal's own exit handlers tear down `MetalContext` (SIGABRT
/// via `close_device` throwing from a destructor). The single `atexit` entry
/// registered by `init_global` walks the pool table at process end and shuts
/// every worker down first (LIFO: this hook registers after tt-metal's, so
/// it runs before theirs). No registry: every worker is reachable through
/// its pool. The handler never panics and never waits unboundedly — a wedged
/// device must not hang process exit.
extern "C" fn tt_atexit_shutdown() {
    let Some(g) = TT.get() else { return };
    for pool in &g.backend.pools {
        let Ok(guard) = pool.lock() else { continue };
        let sender = guard.worker.sender.as_ref().clone();
        drop(guard);
        let (tx, rx) = channel();
        if sender.send(Command::Shutdown { reply: tx }).is_err() {
            continue;
        }
        let _ = rx.recv_timeout(std::time::Duration::from_secs(30));
    }
}

impl RuntimeWorker {
    /// Spawn the worker thread. The worker creates the tt-metal device
    /// internally, reports init success/failure over a oneshot, then services
    /// commands until the sender is dropped (i.e., process exit).
    pub(crate) fn spawn() -> Result<Self, BackendError> {
        let (tx, rx) = channel();
        let tx = Arc::new(tx);
        let (init_tx, init_rx) = channel();

        // Block SIGABRT during device creation; tt-metal's own signal handling
        // expects SIGABRT to be deliverable once the device is up.
        const SIGABRT: libc::c_int = 6;
        let mut mask = std::mem::MaybeUninit::<libc::sigset_t>::uninit();
        unsafe {
            libc::sigemptyset(mask.as_mut_ptr());
            libc::sigaddset(mask.as_mut_ptr(), SIGABRT);
            libc::pthread_sigmask(libc::SIG_BLOCK, mask.as_ptr(), std::ptr::null_mut());
        }

        let worker_tx = tx.clone();
        thread::spawn(move || {
            let shim = load_shim();
            let dev = unsafe { (shim.create_device)() };
            if dev.is_null() {
                let err = cpp_err(&shim, ErrorStatus::Initialization);
                let _ = init_tx.send(Err(err));
                return;
            }

            // Unblock SIGABRT before servicing commands.
            unsafe {
                libc::pthread_sigmask(libc::SIG_UNBLOCK, mask.as_ptr(), std::ptr::null_mut());
            }
            let _ = init_tx.send(Ok(()));

            let mut worker = TTWorker { dev, buffers: Map::default() };
            'work_thread_loop: loop {
                let cmd = match rx.recv() {
                    Ok(cmd) => cmd,
                    // All senders gone (process teardown): close the device
                    // before tt-metal statics tear down, then exit. Without
                    // this the open MeshDevice outlives MetalContext and
                    // close_device throws from a destructor (SIGABRT).
                    Err(_) => {
                        unsafe { (shim.destroy_device)(worker.dev) };
                        break 'work_thread_loop;
                    }
                };

                match cmd {
                    Command::Grid { reply } => {
                        let mut rows = 0u32;
                        let mut cols = 0u32;
                        unsafe { (shim.get_grid_size)(worker.dev, &raw mut rows, &raw mut cols) };
                        if unsafe { (shim.has_error)() } {
                            reply.send(Err(cpp_err(&shim, ErrorStatus::Initialization))).ok();
                        } else {
                            reply.send(Ok((rows, cols))).ok();
                        }
                        continue 'work_thread_loop;
                    }

                    Command::Allocate { bytes, chunk_id, reply } => {
                        let handle = unsafe { (shim.alloc_buffer)(worker.dev, bytes, 2048) };
                        if handle.is_null() {
                            reply.send(Err(cpp_err(&shim, ErrorStatus::MemoryAllocation))).ok();
                            continue 'work_thread_loop;
                        }
                        let n_pages = (bytes + 4095) / 4096;
                        let size = n_pages * 4096;
                        worker.buffers.insert(chunk_id, WorkerBuffer { size });
                        reply.send(Ok((handle, size))).ok();
                        continue 'work_thread_loop;
                    }

                    Command::Free { buffer_id, handle, reply } => {
                        if worker.buffers.remove(&buffer_id).is_some() {
                            unsafe { (shim.free_buffer)(worker.dev, handle) };
                        }
                        reply.send(()).ok();
                        continue 'work_thread_loop;
                    }

                    Command::Write { buffer_id, handle, src, reply } => {
                        let Some(entry) = worker.buffers.get(&buffer_id) else {
                            reply
                                .send(Err(BackendError {
                                    status: ErrorStatus::MemoryAllocation,
                                    context: "write buffer id not allocated".into(),
                                }))
                                .ok();
                            continue 'work_thread_loop;
                        };
                        if src.len() as u64 > entry.size {
                            reply
                                .send(Err(BackendError {
                                    status: ErrorStatus::MemoryCopyH2P,
                                    context: "write length exceeds buffer size".into(),
                                }))
                                .ok();
                            continue 'work_thread_loop;
                        }
                        unsafe { (shim.write_buffer)(worker.dev, handle, src.as_ptr() as *const c_void, src.len() as u64) };
                        if unsafe { (shim.has_error)() } {
                            reply.send(Err(cpp_err(&shim, ErrorStatus::MemoryCopyH2P))).ok();
                        } else {
                            reply.send(Ok(())).ok();
                        }
                        continue 'work_thread_loop;
                    }

                    Command::Read { buffer_id, handle, dst, dst_len, reply } => {
                        let Some(entry) = worker.buffers.get(&buffer_id) else {
                            reply
                                .send(Err(BackendError {
                                    status: ErrorStatus::MemoryAllocation,
                                    context: "read buffer id not allocated".into(),
                                }))
                                .ok();
                            continue 'work_thread_loop;
                        };
                        let cap = dst_len.min(entry.size as usize);
                        unsafe { (shim.read_buffer)(worker.dev, handle, dst as *mut c_void, cap as u64) };
                        if unsafe { (shim.has_error)() } {
                            reply.send(Err(cpp_err(&shim, ErrorStatus::MemoryCopyP2H))).ok();
                        } else {
                            reply.send(Ok(cap)).ok();
                        }
                        continue 'work_thread_loop;
                    }

                    Command::Compile {
                        reader_src,
                        compute_src,
                        writer_src,
                        cb_idx,
                        cb_fmt,
                        cb_tile_bytes,
                        cb_num_tiles,
                        reader_params,
                        compute_params,
                        writer_params,
                        n_params,
                        fp32_dest_acc_en,
                        reply,
                    } => {
                        let prog = unsafe {
                            (shim.compile_program)(
                                worker.dev,
                                reader_src.as_ptr(),
                                reader_src.len(),
                                compute_src.as_ptr(),
                                compute_src.len(),
                                writer_src.as_ptr(),
                                writer_src.len(),
                                cb_idx.as_ptr(),
                                cb_fmt.as_ptr(),
                                cb_tile_bytes.as_ptr(),
                                cb_num_tiles.as_ptr(),
                                cb_idx.len(),
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
                            reply.send(Err(cpp_err(&shim, ErrorStatus::KernelCompilation))).ok();
                        } else {
                            reply.send(Ok(prog)).ok();
                        }
                        continue 'work_thread_loop;
                    }

                    Command::Launch { program, src_handles, dst_handles, grid_dims, vars, reply } => {
                        let ok = unsafe {
                            let ordinals = vars.iter().map(|(o, _)| *o).collect::<Vec<u32>>();
                            let values = vars.iter().map(|(_, v)| *v).collect::<Vec<u32>>();
                            (shim.run_program)(
                                worker.dev,
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
                            reply.send(Err(cpp_err(&shim, ErrorStatus::KernelLaunch))).ok();
                        } else {
                            reply.send(Ok(())).ok();
                        }
                        continue 'work_thread_loop;
                    }

                    Command::DestroyProgram { program, reply } => {
                        unsafe { (shim.destroy_program)(worker.dev, program) };
                        reply.send(Ok(())).ok();
                        continue 'work_thread_loop;
                    }

                    Command::Shutdown { reply } => {
                        // Full teardown on the worker thread: destroy_device
                        // closes the mesh device, teardown_metal destroys the
                        // MetalContext (releasing UMD's CHIP_IN_USE guard as
                        // its owning thread). Teardown runs even if the
                        // device close reported an error; failures are
                        // reported loudly instead of hanging process exit.
                        unsafe { (shim.destroy_device)(worker.dev) };
                        if unsafe { (shim.has_error)() } {
                            eprintln!("tenstorrent worker: {}", cpp_err(&shim, ErrorStatus::Initialization).context);
                        }
                        unsafe { (shim.teardown_metal)() };
                        if unsafe { (shim.has_error)() } {
                            eprintln!("tenstorrent worker: {}", cpp_err(&shim, ErrorStatus::Initialization).context);
                        }
                        reply.send(()).ok();
                        break 'work_thread_loop;
                    }
                };
            }
        });

        // Wait for device init (fails loudly instead of serving grid/alloc
        // against a dead worker).
        init_rx.recv().map_err(|_| BackendError {
            status: ErrorStatus::Initialization,
            context: "tenstorrent worker exited during init".into(),
        })??;

        Ok(RuntimeWorker { sender: worker_tx })
    }

    pub(crate) fn query_grid(&self) -> Result<(u32, u32), BackendError> {
        let (tx, rx) = channel();
        self.sender
            .send(Command::Grid { reply: tx })
            .map_err(|_| BackendError { status: ErrorStatus::Initialization, context: "tenstorrent worker gone".into() })?;
        rx.recv().map_err(|_| BackendError {
            status: ErrorStatus::Initialization,
            context: "tenstorrent worker dropped grid reply".into(),
        })?
    }

    pub(crate) fn allocate(&self, bytes: u64, chunk_id: ChunkId) -> Result<(*mut c_void, u64), BackendError> {
        let (tx, rx) = channel();
        self.sender
            .send(Command::Allocate { bytes, chunk_id, reply: tx })
            .map_err(|_| BackendError { status: ErrorStatus::MemoryAllocation, context: "tenstorrent worker gone".into() })?;
        rx.recv().map_err(|_| BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: "tenstorrent worker dropped alloc reply".into(),
        })?
    }

    pub(crate) fn free(&self, buffer_id: ChunkId, handle: *mut c_void) {
        let (tx, rx) = channel();
        let _ = self.sender.send(Command::Free { buffer_id, handle, reply: tx });
        let _ = rx.recv();
    }

    pub(crate) fn write(&self, handle: *mut c_void, buffer_id: ChunkId, src: Vec<u8>) -> Result<(), BackendError> {
        let (tx, rx) = channel();
        self.sender
            .send(Command::Write { buffer_id, handle, src, reply: tx })
            .map_err(|_| BackendError { status: ErrorStatus::MemoryCopyH2P, context: "tenstorrent worker gone".into() })?;
        rx.recv().map_err(|_| BackendError {
            status: ErrorStatus::MemoryCopyH2P,
            context: "tenstorrent worker dropped write reply".into(),
        })?
    }

    pub(crate) fn read(&self, handle: *mut c_void, buffer_id: ChunkId, dst: &mut [u8]) -> Result<usize, BackendError> {
        let (tx, rx) = channel();
        self.sender
            .send(Command::Read { buffer_id, handle, dst: dst.as_mut_ptr(), dst_len: dst.len(), reply: tx })
            .map_err(|_| BackendError { status: ErrorStatus::MemoryCopyP2H, context: "tenstorrent worker gone".into() })?;
        rx.recv().map_err(|_| BackendError {
            status: ErrorStatus::MemoryCopyP2H,
            context: "tenstorrent worker dropped read reply".into(),
        })?
    }

    pub(crate) fn compile(
        &self,
        reader_src: &[u8],
        compute_src: &[u8],
        writer_src: &[u8],
        cb_indices: &[u32],
        cb_fmt: &[u32],
        cb_tile_bytes: &[u32],
        cb_num_tiles: &[u32],
        reader_params: &[u32],
        compute_params: &[u32],
        writer_params: &[u32],
        n_params: u32,
        fp32_dest_acc_en: bool,
    ) -> Result<u32, BackendError> {
        let (tx, rx) = channel();
        self.sender
            .send(Command::Compile {
                reader_src: reader_src.to_vec(),
                compute_src: compute_src.to_vec(),
                writer_src: writer_src.to_vec(),
                cb_idx: cb_indices.to_vec(),
                cb_fmt: cb_fmt.to_vec(),
                cb_tile_bytes: cb_tile_bytes.to_vec(),
                cb_num_tiles: cb_num_tiles.to_vec(),
                reader_params: reader_params.to_vec(),
                compute_params: compute_params.to_vec(),
                writer_params: writer_params.to_vec(),
                n_params,
                fp32_dest_acc_en,
                reply: tx,
            })
            .map_err(|_| BackendError { status: ErrorStatus::KernelCompilation, context: "tenstorrent worker gone".into() })?;
        rx.recv().map_err(|_| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: "tenstorrent worker dropped compile reply".into(),
        })?
    }

    pub(crate) fn run(
        &self,
        program: u32,
        src_handles: &[*mut c_void],
        dst_handles: &[*mut c_void],
        grid_dims: [u32; 2],
        vars: &[(u32, u32)],
    ) -> Result<(), BackendError> {
        let (tx, rx) = channel();
        self.sender
            .send(Command::Launch {
                program,
                src_handles: src_handles.to_vec(),
                dst_handles: dst_handles.to_vec(),
                grid_dims,
                vars: vars.to_vec(),
                reply: tx,
            })
            .map_err(|_| BackendError { status: ErrorStatus::KernelLaunch, context: "tenstorrent worker gone".into() })?;
        rx.recv().map_err(|_| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: "tenstorrent worker dropped run reply".into(),
        })?
    }

    pub(crate) fn destroy_program(&self, program: u32) {
        let (tx, rx) = channel();
        let _ = self.sender.send(Command::DestroyProgram { program, reply: tx });
        let _ = rx.recv();
    }
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

fn detect_dram_bytes() -> u64 {
    let pci_devices = std::path::Path::new("/sys/bus/pci/devices");
    if let Ok(entries) = std::fs::read_dir(pci_devices) {
        for entry in entries.flatten() {
            let vendor_path = entry.path().join("vendor");
            let vendor = std::fs::read_to_string(&vendor_path).unwrap();
            if vendor.trim() == "0x1e52" {
                let subsys = std::fs::read_to_string(entry.path().join("subsystem_device")).unwrap();
                if let Ok(id) = u16::from_str_radix(subsys.trim().trim_start_matches("0x"), 16) {
                    for &(sid, _name, size) in DRAM_SIZE_TABLE {
                        if sid == id {
                            return size;
                        }
                    }
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
// Per-buffer tracking: opaque handle owned by the worker thread.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub(crate) struct TTBuffer {
    handle: *mut c_void,
    pub(crate) size: u64,
}

// Opaque FFI handles are Send/Sync: the worker thread is the only thread that
// dereferences them, and they point at C-managed objects.
unsafe impl Send for TTBuffer {}
unsafe impl Sync for TTBuffer {}

// ---------------------------------------------------------------------------
// Memory pool — device DRAM buffers managed by the worker.
// ChunkId is chosen by Rust's slab; the worker owns the opaque handle.
// Allocation reuses the free_set first (best-fit, stable addresses);
// fresh device memory is requested only on a miss.
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
    pub(crate) worker: Arc<RuntimeWorker>,
    pub(crate) dev_info: DeviceInfo,
}

pub(super) fn ensure_pool_table(config: &TTConfig, debug_dev: bool) -> Result<Vec<Mutex<TTMemoryPool>>, BackendError> {
    let mut pools: Vec<Mutex<TTMemoryPool>> = Vec::new();
    if let Some(device_ids) = &config.device_ids
        && device_ids.is_empty()
    {
        if debug_dev {
            println!("[tenstorrent] configured out");
        }
        return Ok(pools);
    }

    let dram_bytes = detect_dram_bytes();

    // Spawn the worker — it creates the tt-metal device internally and
    // fails loudly here (not lazily on first alloc) if init fails.
    let worker = RuntimeWorker::spawn()?;
    let (grid_rows, grid_cols) = worker.query_grid()?;
    if debug_dev {
        println!("[tenstorrent] device initialized");
        println!("[tenstorrent] device total memory: {} MB", dram_bytes / (1024 * 1024));
        println!("[tenstorrent] tensix grid {grid_rows} rows x {grid_cols} cols");
    }

    // F8E5M2 has no Blackhole DataFormat: not a capable dtype, codegen rejects it.
    let mut dtype_capability = [DTypeCapability::all(); DType::N_DTYPES];
    dtype_capability[DType::F8E5M2 as usize] = DTypeCapability::ZERO;
    pools.push(Mutex::new(TTMemoryPool {
        buffers: Slab::new(),
        free_set: Set::default(),
        free_bytes: Dim::from(dram_bytes as i64),
        worker: Arc::new(worker),
        dev_info: DeviceInfo {
            compute: 200_000_000_000_000, // ~200 TFLOPS BF16
            // Grid axes only (gidx0 row, gidx1 col); TT launches at most
            // 2 group axes, and the launch path rejects more.
            max_global_work_dims: vec![Dim::from(grid_rows), Dim::from(grid_cols)],
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

    Ok(pools)
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
        // Best-fit from the free_set first (stable addresses across
        // repeats — those bytes were already paid for); else fresh.
        // A hit costs no device call and no free_bytes change.
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
        let (handle, size) = self.worker.allocate(bytes_u64, chunk_id)?;
        debug_assert!(!handle.is_null(), "worker returned null handle without error");
        self.buffers[chunk_id] = TTBuffer { handle, size };
        self.free_bytes -= size as Dim;
        Ok(chunk_id)
    }

    /// Put a buffer into the free list for stable-address reuse. Frees
    /// nothing; device memory is reclaimed only by [`TTMemoryPool::dispose`].
    /// Safe without a sync: the worker is synchronous, so no launch can
    /// still be using the buffer when its placement drops.
    pub fn release(&mut self, buffer_id: ChunkId) {
        if !self.buffers.contains_id(buffer_id) {
            debug_assert!(false, "release of unknown TT buffer {buffer_id:?}");
            return;
        }
        debug_assert!(!self.free_set.contains(&buffer_id), "double release of TT buffer {buffer_id:?}");
        self.free_set.insert(buffer_id);
    }

    /// Tries to reuse existing allocations, all or nothing: every id must
    /// be on the free list (stable addresses) or nothing is claimed.
    pub fn try_reuse_allocations(&mut self, buffer_ids: &Set<ChunkId>) -> bool {
        let ok = buffer_ids.iter().all(|id| self.free_set.contains(id));
        if ok {
            for id in buffer_ids {
                self.free_set.remove(id);
            }
        }
        ok
    }

    /// Free all buffers in the free list at once: release every free buffer
    /// back to the worker and give the bytes back.
    pub fn dispose(&mut self) {
        for id in core::mem::take(&mut self.free_set) {
            let buf = unsafe { self.buffers.remove_and_return(id) };
            self.free_bytes += buf.size as Dim;
            self.worker.free(id, buf.handle);
        }
    }

    /// Blocking upload (sync — the worker runs the transfer to completion).
    pub fn host_to_pool(&mut self, src: &[u8], dst: ChunkId) -> Result<(), BackendError> {
        let buf = self
            .buffers
            .get(dst)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyH2P, context: "invalid buffer id".into() })?;
        let len = src.len().min(buf.size as usize);
        let data = src[..len].to_vec();
        let handle = buf.handle;
        self.worker.write(handle, dst, data)
    }

    /// Blocking download (sync — the worker runs the transfer to completion).
    pub fn pool_to_host(&mut self, src: ChunkId, dst: &mut [u8]) -> Result<(), BackendError> {
        let buf = self
            .buffers
            .get(src)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyP2H, context: "invalid buffer id".into() })?;
        let _len = dst.len().min(buf.size as usize);
        let handle = buf.handle;
        let _copied = self.worker.read(handle, src, dst)?;
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

// ---------------------------------------------------------------------------
// Compiled program tracking
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct TTProgram {
    /// Opaque shim id (per-device program cache index).
    shim: u32,
    /// Global param count (kernel inputs). Launch reads the count only.
    n_inputs: u32,
    /// GlobalMut param count (kernel outputs). Launch reads the count only.
    n_outputs: u32,
    /// Group-range lengths in axis order (gws): Const resolved at compile,
    /// Param(ordinal) resolved from the launch args.
    gws: Vec<GwsDim>,
    /// Tensix grid rows/cols (from DeviceInfo at compile): launch-resolved
    /// grid dims (dynamic sizes) are bounds-checked against this.
    max_grid: [u32; 2],
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TTDevice {
    device_info: Arc<DeviceInfo>,
    memory_pool: Pool,
    worker: Arc<RuntimeWorker>,
    programs: Slab<DeviceProgramId, TTProgram>,
}

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

    #[allow(unused_must_use)]
    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        // Render to descriptor; decode head positions. CB ids, section
        // params, param ordinals, and io counts arrive in the descriptor
        // tables, computed once at render when the lowered kernel is
        // available. What stays here is launch-side assembly: the
        // decoded grid, the runtime CB config split, and the program
        // compile call.
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

        // Per-section params (0 = reader, 1 = compute, 2 = writer): the
        // ordinals of the params each section's stores depend on, in
        // ascending head order. These lists define the sections' runtime
        // args: each section gets exactly its own params — its Global +
        // Variable params interleaved in head order first, then its
        // GlobalMut params — followed by the core's tensix-grid coordinates
        // gidx0 (row) / gidx1 (col). GlobalMut occupies the tail of the
        // head-order param list, so the ascending sort already yields the
        // Global|Variable-then-GlobalMut layout; see
        // `Kernel::generate_tenstorrent` and `tt_runtime_shim.cpp`
        // `section_rt_args` for the consumption side.
        let n_params = desc.n_params;
        // Tensix grid from the descriptor (rank <= 2 enforced at
        // render): axis-ordered, full dim expressions, const lengths
        // validated against the device max. Param-backed lengths
        // resolve at launch from the Variable arg.
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
        let shim = self.worker.compile(
            reader,
            compute,
            writer,
            &cb_indices,
            &cb_formats,
            &cb_tile_bytes,
            &cb_num_tiles,
            reader_params,
            compute_params,
            writer_params,
            n_params,
            fp32_dest_acc_en,
        )?;
        Ok(self.programs.push(TTProgram { shim, n_inputs: desc.n_inputs, n_outputs: desc.n_outputs, gws, max_grid }))
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        if self.programs.contains_id(program_id) {
            let prog = unsafe { self.programs.remove_and_return(program_id) };
            self.worker.destroy_program(prog.shim);
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
        let shim = prog.shim;

        let n_inputs = prog.n_inputs as usize;
        let n_outputs = prog.n_outputs as usize;

        // One arg per param, head order: Global + Variable interleaved,
        // GlobalMut at the tail. Kinds are derivable from the args themselves:
        // Variable -> Variable; Buffer above the GlobalMut tail -> Global.
        if args.len() < n_inputs + n_outputs {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!(
                    "expected at least {} args ({} inputs + {} outputs), got {}",
                    n_inputs + n_outputs,
                    n_inputs,
                    n_outputs,
                    args.len()
                )
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
        let mut grid_dims = [1u32, 1u32];
        if prog.gws.len() > 2 {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("tenstorrent supports at most 2 group axes, got {}", prog.gws.len()).into(),
            });
        }
        for (axis, g) in prog.gws.iter().enumerate() {
            // Variable ordinals index the launch args (head order), which
            // always carry a Variable value at a Variable param's position.
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
                    context: format!(
                        "tenstorrent grid axis {axis} size {} exceeds device grid {}",
                        grid_dims[axis], prog.max_grid[axis]
                    )
                    .into(),
                });
            }
        }
        self.worker.run(shim, &src_handles, &dst_handles, grid_dims, &vars)
    }

    /// Timed launch for autotune. Returns the kernel's run time in nanos.
    /// Tenstorrent launches are synchronous (the worker blocks through
    /// Finish), so a wall-clock bracket is already an uncontended
    /// measurement.
    pub fn launch_timed(&mut self, program_id: DeviceProgramId, args: &[LaunchArg]) -> Result<u64, BackendError> {
        let start = std::time::Instant::now();
        self.launch(program_id, self.memory_pool, args)?;
        Ok(start.elapsed().as_nanos() as u64)
    }

    /// Copy executing a transfer into this device's pool: host uploads go
    /// through the blocking worker upload, CUDA sources stage through a host
    /// temp (no peer DMA between vendors; CUDA PoolToHost drains first),
    /// TT same-device is a no-op and cross-device routes through host.
    /// Single-shard placements only.
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
                // The host lock is held across the blocking worker upload: the
                // source pointer stays valid. The worker never takes it.
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
                // Other device pools: route through host memory.
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

/// Preplanned Tenstorrent partition: the ordered commands plus
/// per-command death lists (slots whose final read is that command and
/// which nothing later needs). Replay allocates unbound defs on the fly
/// from their specs and launches programs back-to-back — the worker is
/// synchronous, so no queues, no edges, no worker: program order is the
/// ordering guarantee.
#[derive(Debug)]
pub(crate) struct TTPartition {
    pub(crate) cmds: Vec<Cmd>,
    deaths: Vec<Vec<OpId>>,
    pub(crate) dev: u16,
}

impl TTDevice {
    /// Schedule a command run for a Tenstorrent device: orders nothing
    /// (program order is launch order), precomputes per-command death
    /// lists from the final read of every slot. A slot dies at its last
    /// use unless pinned (a plan output or read after this partition).
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
                    // Resolve args to launch values. Buffers name their
                    // chunk (offsets are 0 everywhere — no sub-chunk views
                    // exist yet); variables carry their scalar value.
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
                    // Dry run: skip device execution, keep arg binding validation.
                    // Output buffers hold uninitialized contents; callers must not read them.
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
