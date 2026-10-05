// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `OpenCL` backend

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::question_mark)]
#![allow(clippy::needless_pass_by_ref_mut)]
#![allow(clippy::unused_self)]

use super::{
    ChunkId, Cmd, DTypeCapability, DeviceInfo, DeviceProgramId, GwsDim, LaunchArg, ParamKind, Placement, Pool, Shard,
    gws_from_kernel,
};
use crate::{
    DType,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{Kernel, Op, OpId, RangeKind},
    shape::Dim,
    slab::Slab,
};
use crate::{Map, Set, hashers::FHasher};
use libloading::Library;
use nanoserde::DeJson;
use std::{
    ffi::{CString, c_void},
    hash::BuildHasherDefault,
    ptr,
    sync::Arc,
    sync::atomic::{AtomicU64, Ordering},
    sync::mpsc::{Receiver, Sender, channel},
    sync::{Mutex, OnceLock},
    thread,
    time::Instant,
};

// ── Global state ──────────────────────────────────────────────────────────────

static OPENCL_POOLS: OnceLock<Vec<Mutex<OpenCLMemoryPool>>> = OnceLock::new();
static OPENCL_DEVICES: OnceLock<Vec<Mutex<OpenCLDevice>>> = OnceLock::new();

#[derive(Debug, Default, DeJson)]
#[nserde(default)]
pub struct OpenCLConfig {
    /// Select which platforms will be used by `OpenCL` backend
    /// If set to None, uses all available platforms.
    /// default = None
    pub platform_ids: Option<Vec<usize>>,
    /// Number of in-order command queues per device. Queue assignment is
    /// static: the schedule spreads slot-disjoint launch chains across
    /// queues (slot affinity); the worker submits each command to its
    /// assigned queue with tail-event waits.
    /// default = 8
    pub queues: Option<usize>,
}

// OpenCL does not have the concept of memory pools,
// so we simply say it is all in one memory pool
#[derive(Debug)]
pub struct OpenCLMemoryPool {
    tx: Sender<Command>,
    #[allow(unused)]
    total_bytes: Dim,
    free_bytes: Arc<AtomicU64>,
    dev_info: DeviceInfo,
}

#[derive(Debug)]
pub struct OpenCLBuffer {
    pub ptr: *mut c_void,
    pub bytes: Dim,
}

#[derive(Debug)]
pub struct OpenCLDevice {
    tx: Sender<Command>,
    dev_info: Arc<DeviceInfo>,
    memory_pool: Pool,
}

#[derive(Debug)]
pub(super) struct OpenCLProgram {
    program: *mut c_void,
    kernel: *mut c_void,
    lws: Vec<Dim>,
    gws: Vec<GwsDim>,
    /// Per-`Param` kinds in head order, used at submission to split launch
    /// args into reads (`Global`) and writes (`GlobalMut`).
    params: Vec<ParamKind>,
}

#[derive(Debug)]
pub(super) struct OpenCLQueue {
    queue: *mut c_void, // points to device queue
}

enum Command {
    Allocate {
        bytes: Dim,
        reply: Sender<Result<ChunkId, BackendError>>,
    },
    /// Put a buffer on the free list for stable-address reuse. Frees
    /// nothing; VRAM is reclaimed only by `Dispose`.
    Release {
        buffer_id: ChunkId,
    },
    /// Free every free-list buffer at once (hard sync first). The only
    /// VRAM reclamation; every released id becomes invalid.
    Dispose,
    /// All-or-nothing claim of free-list ids for stable addresses.
    TryReuse {
        buffer_ids: Set<ChunkId>,
        reply: Sender<Result<bool, BackendError>>,
    },
    /// Blocking read-back: the reply is sent after the data arrived in
    /// host memory. Sync point — every queue drains first.
    PoolToHost {
        src: ChunkId,
        dst: *mut u8,
        bytes: Dim,
        reply: Sender<Result<(), BackendError>>,
    },
    /// Blocking copy into this pool's buffer: the worker drains every
    /// queue first, runs the transfer on queue 0, and replies after the
    /// data arrived. Same-pool pairs use CopyBuffer; anything else reads
    /// from `src_ptr` (host staging or a disk file mapping — valid across
    /// the blocking call, released caller-side after the reply).
    Copy {
        src_pool: Pool,
        src_buf: ChunkId,
        src_ptr: *const u8,
        bytes: Dim,
        dst_buf: ChunkId,
        reply: Sender<Result<(), BackendError>>,
    },
    Compile {
        name: Box<str>,
        source: String,
        lws: Vec<Dim>,
        gws: Vec<GwsDim>,
        params: Vec<ParamKind>,
        reply: Sender<Result<DeviceProgramId, BackendError>>,
    },
    /// Run a whole partition at once — one roundtrip per replay. `cmds`
    /// are the schedule-ordered launches/aliases; `queues`/`waits` their
    /// static queue assignment + cross-queue wait sets; `bound` maps
    /// already-placed slots to chunks; `vars` are the scalar slots. The
    /// worker allocates unbound defs up front, submits every launch to
    /// its queue, releases dead chunks, and replies the fresh slot→chunk
    /// bindings. No capture: repeats resubmit (stable addresses come from
    /// free-list best-fit reuse).
    Replay {
        cmds: Vec<Cmd>,
        queues: Vec<usize>,
        waits: Vec<Vec<usize>>,
        bound: Vec<(OpId, ChunkId)>,
        vars: Vec<(OpId, Constant)>,
        deaths: Vec<Vec<OpId>>,
        reply: Sender<Result<Vec<(OpId, ChunkId)>, BackendError>>,
    },
    /// Timed launch for autotune: drain every queue first (uncontended
    /// timing), then run the kernel solo on queue 0 and reply nanos.
    LaunchTimed {
        program_id: DeviceProgramId,
        args: Vec<LaunchArg>,
        reply: Sender<Result<u64, BackendError>>,
    },
    ReleaseProgram {
        program_id: DeviceProgramId,
    },
}

unsafe impl Send for Command {}

/// Single backend initializer: builds pools + devices together in one pass,
/// publishes both tables. Reads config directly; no init locks.
fn backend() -> Result<(&'static Vec<Mutex<OpenCLMemoryPool>>, &'static Vec<Mutex<OpenCLDevice>>), BackendError> {
    if let Some(pools) = OPENCL_POOLS.get()
        && let Some(devs) = OPENCL_DEVICES.get()
    {
        return Ok((pools, devs));
    }
    let (pools, devs) = initialize_backend()?;
    let _ = OPENCL_POOLS.set(pools);
    let _ = OPENCL_DEVICES.set(devs);
    match (OPENCL_POOLS.get(), OPENCL_DEVICES.get()) {
        (Some(pools), Some(devs)) => Ok((pools, devs)),
        _ => Err(BackendError { status: ErrorStatus::Initialization, context: "OpenCL init failed".into() }),
    }
}

fn initialize_backend() -> Result<(Vec<Mutex<OpenCLMemoryPool>>, Vec<Mutex<OpenCLDevice>>), BackendError> {
    let config = super::config();
    let debug_dev = super::debug_backends();
    let pools = ensure_pool_table(&config.opencl, debug_dev)?;
    let mut devs = Vec::with_capacity(pools.len());
    for (idx, pool) in pools.iter().enumerate() {
        let pool_id = Pool::OpenCL(u16::try_from(idx).expect("So many OpenCL devices..."));
        let guard = super::lock(pool_id, pool);
        let tx = guard.tx.clone();
        let dev_info = guard.dev_info.clone();
        drop(guard);
        devs.push(Mutex::new(OpenCLDevice { tx, dev_info: Arc::new(dev_info), memory_pool: pool_id }));
    }
    Ok((pools, devs))
}

pub(super) fn pool(id: u16) -> Result<&'static Mutex<OpenCLMemoryPool>, BackendError> {
    backend()?.0.get(id as usize).ok_or_else(|| no_pool(id))
}

pub(super) fn pool_count() -> u16 {
    backend().map(|(pools, _)| pools.len() as u16).unwrap_or(0)
}

fn no_pool(id: u16) -> BackendError {
    BackendError { status: ErrorStatus::Initialization, context: format!("Pool::OpenCL({id}) is not available").into() }
}

/// Builds the per-device pool table: loads the ICD, enumerates platforms and
/// devices, and spawns one worker (one context) per device. Pure builder,
/// called once from `backend()`.
pub(super) fn ensure_pool_table(config: &OpenCLConfig, debug_dev: bool) -> Result<Vec<Mutex<OpenCLMemoryPool>>, BackendError> {
    let mut pools: Vec<Mutex<OpenCLMemoryPool>> = Vec::new();
    if let Some(device_ids) = &config.platform_ids
        && device_ids.is_empty()
    {
        if debug_dev {
            println!("[opencl] configured out");
        }
        return Ok(pools);
    }

    // Block SIGABRT before any OpenCL library loading or API calls,
    // so that any ICD-created threads inherit SIGABRT as blocked.
    unsafe extern "C" {
        fn sigemptyset(set: *mut c_void) -> i32;
        fn sigaddset(set: *mut c_void, signum: i32) -> i32;
        fn pthread_sigmask(how: i32, set: *const c_void, oldset: *mut c_void) -> i32;
    }
    const SIGABRT: i32 = 6;
    const SIG_BLOCK: i32 = 0;
    let mut sigset = std::mem::MaybeUninit::<[u8; 128]>::uninit();
    unsafe { sigemptyset(sigset.as_mut_ptr().cast()) };
    unsafe { sigaddset(sigset.as_mut_ptr().cast(), SIGABRT) };
    let ret = unsafe { pthread_sigmask(SIG_BLOCK, sigset.as_ptr().cast(), ptr::null_mut()) };
    if ret != 0 {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: format!("pthread_sigmask failed: {ret}").into(),
        });
    }

    // Install a process-wide no-op SIGABRT handler to catch the signal
    // that the ROCm runtime sends via abort() when it detects context loss.
    // Must be on the main thread before any OpenCL calls.
    type SigH = unsafe extern "C" fn(i32);
    unsafe extern "C" {
        fn signal(sig: i32, handler: SigH) -> SigH;
    }
    unsafe extern "C" fn sigabrt_handler(_: i32) {}
    let _prev = unsafe { signal(SIGABRT, sigabrt_handler as SigH) };

    // Search for opencl dynamic library path, kinda primitive, but fast and mostly works
    let mut opencl_paths = Vec::new();
    opencl_paths.push(std::path::PathBuf::from("libOpenCL.so"));
    opencl_paths.push(std::path::PathBuf::from("libOpenCL.so.1"));
    for lib_folder in ["/lib", "/lib64", "/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu"] {
        if let Ok(lib_folder) = std::fs::read_dir(lib_folder) {
            for entry in lib_folder.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let name = path.file_name().map_or("", |x| x.to_str().unwrap());
                    if name.contains("libOpenCL.so") {
                        opencl_paths.push(path);
                    }
                }
            }
        }
    }

    let opencl = opencl_paths.into_iter().find_map(|path| unsafe { Library::new(path) }.ok());
    let Some(opencl) = opencl else {
        return Err(BackendError { status: ErrorStatus::DyLibNotFound, context: "[OPENCL] runtime not found.".into() });
    };
    let clGetPlatformIDs: unsafe extern "C" fn(cl_uint, *mut *mut c_void, *mut cl_uint) -> OpenCLStatus =
        *unsafe { opencl.get(b"clGetPlatformIDs\0") }?;
    let clCreateContext: unsafe extern "C" fn(
        *const isize,
        cl_uint,
        *const *mut c_void,
        Option<unsafe extern "C" fn(*const i8, *const c_void, usize, *mut c_void)>,
        *mut c_void,
        *mut OpenCLStatus,
    ) -> *mut c_void = *unsafe { opencl.get(b"clCreateContext\0") }?;
    let clCreateCommandQueue: unsafe extern "C" fn(*mut c_void, *mut c_void, cl_bitfield, *mut OpenCLStatus) -> *mut c_void =
        *unsafe { opencl.get(b"clCreateCommandQueue\0") }?;
    let clGetDeviceIDs: unsafe extern "C" fn(*mut c_void, cl_bitfield, cl_uint, *mut *mut c_void, *mut cl_uint) -> OpenCLStatus =
        *unsafe { opencl.get(b"clGetDeviceIDs\0") }?;
    let clGetEventInfo: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus =
        *unsafe { opencl.get(b"clGetEventInfo\0") }?;
    let _clReleaseCommandQueue: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus =
        *unsafe { opencl.get(b"clReleaseCommandQueue\0") }?;
    let clEnqueueNDRangeKernel: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        *const usize,
        *const usize,
        *const usize,
        cl_uint,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> OpenCLStatus = *unsafe { opencl.get(b"clEnqueueNDRangeKernel\0") }?;
    let clGetProgramBuildInfo: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        usize,
        *mut c_void,
        *mut usize,
    ) -> OpenCLStatus = *unsafe { opencl.get(b"clGetProgramBuildInfo\0") }?;
    let clBuildProgram: unsafe extern "C" fn(
        *mut c_void,
        cl_uint,
        *const *mut c_void,
        *const i8,
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>,
        *mut c_void,
    ) -> OpenCLStatus = *unsafe { opencl.get(b"clBuildProgram\0") }?;
    let clReleaseProgram: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus = *unsafe { opencl.get(b"clReleaseProgram\0") }?;
    let clReleaseEvent: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus = *unsafe { opencl.get(b"clReleaseEvent\0") }?;
    let _clReleaseContext: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus = *unsafe { opencl.get(b"clReleaseContext\0") }?;
    let clSetKernelArg: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *const c_void) -> OpenCLStatus =
        *unsafe { opencl.get(b"clSetKernelArg\0") }?;
    let clCreateKernel: unsafe extern "C" fn(*mut c_void, *const i8, *mut OpenCLStatus) -> *mut c_void =
        *unsafe { opencl.get(b"clCreateKernel\0") }?;
    let clReleaseMemObject: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus = *unsafe { opencl.get(b"clReleaseMemObject\0") }?;
    let clGetDeviceInfo: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus =
        *unsafe { opencl.get(b"clGetDeviceInfo\0") }?;
    let clCreateProgramWithSource: unsafe extern "C" fn(
        *mut c_void,
        cl_uint,
        *const *const i8,
        *const usize,
        *mut OpenCLStatus,
    ) -> *mut c_void = *unsafe { opencl.get(b"clCreateProgramWithSource\0") }?;
    let clEnqueueReadBuffer: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        usize,
        usize,
        *mut c_void,
        cl_uint,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> OpenCLStatus = *unsafe { opencl.get(b"clEnqueueReadBuffer\0") }?;
    let clEnqueueWriteBuffer: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        usize,
        usize,
        *const c_void,
        cl_uint,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> OpenCLStatus = *unsafe { opencl.get(b"clEnqueueWriteBuffer\0") }?;
    let clEnqueueCopyBuffer: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *mut c_void,
        usize,
        usize,
        usize,
        cl_uint,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> OpenCLStatus = *unsafe { opencl.get(b"clEnqueueCopyBuffer\0") }?;
    let clCreateBuffer: unsafe extern "C" fn(*mut c_void, cl_bitfield, usize, *mut c_void, *mut OpenCLStatus) -> *mut c_void =
        *unsafe { opencl.get(b"clCreateBuffer\0") }?;
    let clFinish: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus = *unsafe { opencl.get(b"clFinish\0") }?;
    let clGetPlatformInfo: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus =
        *unsafe { opencl.get(b"clGetPlatformInfo\0") }?;

    let library = Arc::new(opencl);
    let platform_ids = {
        // Get the number of platforms
        let mut count: cl_uint = 0;
        unsafe { clGetPlatformIDs(0, ptr::null_mut(), &raw mut count) }.check(ErrorStatus::DeviceEnumeration)?;
        if count > 0 {
            // Get the platform ids.
            let len = count as usize;
            let mut ids: Vec<*mut c_void> = Vec::with_capacity(len);
            unsafe { clGetPlatformIDs(count, ids.as_mut_ptr(), ptr::null_mut()) }.check(ErrorStatus::DeviceEnumeration)?;
            unsafe { ids.set_len(len) };
            ids
        } else {
            Vec::new()
        }
    };
    for (_platform_id, platform) in
        platform_ids.iter().enumerate().filter(|(id, _)| config.platform_ids.as_ref().is_none_or(|ids| ids.contains(id)))
    {
        let platform = *platform;
        let Ok(device_ids) = {
            // Get the number of devices of device_type
            let mut count: cl_uint = 0;
            let mut status = unsafe { clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, ptr::null_mut(), &raw mut count) };
            if (OpenCLStatus::CL_SUCCESS != status) && (OpenCLStatus::CL_DEVICE_NOT_FOUND != status) {
                Err(status)
            } else if 0 < count {
                // Get the device ids.
                let len = count as usize;
                let mut ids: Vec<*mut c_void> = Vec::with_capacity(len);
                unsafe {
                    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, count, ids.as_mut_ptr(), ptr::null_mut());
                    ids.set_len(len);
                };
                if OpenCLStatus::CL_SUCCESS == status {
                    Ok(ids)
                } else {
                    Err(status)
                }
            } else {
                Ok(Vec::default())
            }
        }
        .map_err(|err| err.check(ErrorStatus::DeviceEnumeration).err().unwrap()) else {
            continue;
        };
        if debug_dev {
            let platform_name = {
                let mut size: usize = 0;
                let Ok(()) = unsafe { clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, ptr::null_mut(), &raw mut size) }
                    .check(ErrorStatus::Initialization)
                else {
                    continue;
                };
                if size > 0 {
                    let count = size / core::mem::size_of::<u8>();
                    let mut data: Vec<u8> = Vec::with_capacity(count);
                    let Ok(()) = unsafe {
                        data.set_len(count);
                        clGetPlatformInfo(platform, CL_PLATFORM_NAME, size, data.as_mut_ptr().cast(), ptr::null_mut())
                    }
                    .check(ErrorStatus::Initialization) else {
                        continue;
                    };
                    data
                } else {
                    Vec::default()
                }
            };
            println!("[opencl] {} on devices:", String::from_utf8(platform_name).unwrap());
        }
        if device_ids.is_empty() {
            continue;
        }

        // One pool (one context, one worker thread) per device. Device info
        // is queried on the main thread (clGetDeviceInfo doesn't need a
        // context); each pool carries its device's info for later device
        // registration.
        for dev in device_ids.iter().copied() {
            let mut dev_info = DeviceInfo::default();
            let Ok(()) = query_device_info(dev, clGetDeviceInfo, &mut dev_info, debug_dev) else {
                continue;
            };
            let total_bytes = get_device_data(dev, clGetDeviceInfo, CL_DEVICE_GLOBAL_MEM_SIZE)
                .map(|bytes| Dim::from_ne_bytes(bytes.try_into().unwrap()))
                .unwrap_or(0);
            if debug_dev {
                println!("[opencl] device total memory: {} MB", total_bytes / (1024 * 1024));
            }

            let (tx, rx): (Sender<Command>, Receiver<Command>) = channel();
            let free_bytes_atomic = Arc::new(AtomicU64::new(total_bytes as u64));

            // Cast to usize for Send safety through the closure
            let worker_device: usize = dev as usize;
            let worker_library = library.clone();
            // This worker's own pool: submission resolves launch-arg
            // placements through the shard addressed to it.
            let worker_pool = Pool::OpenCL(u16::try_from(pools.len()).expect("So many OpenCL devices..."));
            thread::spawn({
                let free_bytes_atomic = Arc::clone(&free_bytes_atomic);
                move || {
                    let _worker_library = worker_library;
                    let devices: Vec<*mut c_void> = vec![worker_device as *mut c_void];
                    let mut status = OpenCLStatus::CL_SUCCESS;
                    let context = unsafe {
                        clCreateContext(
                            ptr::null(),
                            cl_uint::try_from(devices.len()).expect("So many devices..."),
                            devices.as_ptr(),
                            None,
                            ptr::null_mut(),
                            &raw mut status,
                        )
                    };
                    if status.check(ErrorStatus::Initialization).is_err() {
                        return;
                    }

                    let mut queues: Vec<OpenCLQueue> = Vec::with_capacity(devices.len());
                    for &dev in &devices {
                        // Hardware queues the micro-batch window is distributed
                        // over (`OpenCLConfig::queues`, default 8).
                        let queue_count = super::config().opencl.queues.unwrap_or(8);
                        for _ in 0..queue_count {
                            let mut qstatus = OpenCLStatus::CL_SUCCESS;
                            let new_queue = unsafe { clCreateCommandQueue(context, dev, 0, &raw mut qstatus) };
                            if qstatus.check(ErrorStatus::Initialization).is_ok() {
                                queues.push(OpenCLQueue { queue: new_queue });
                            }
                        }
                    }
                    if queues.is_empty() {
                        if super::debug_backends() {
                            println!("[opencl] device {worker_device}: no command queues created");
                        }
                        return;
                    }

                    let mut buffers: Slab<ChunkId, OpenCLBuffer> = Slab::new();
                    // Free-list ids for stable-address reuse (`Release`
                    // inserts, `TryReuse`/`Allocate` claim, `Dispose`
                    // frees). Slab entries stay so ChunkIds are monotonic and
                    // never alias a different buffer.
                    let mut free_set: Set<ChunkId> = Set::default();
                    let mut programs: Slab<DeviceProgramId, OpenCLProgram> = Slab::new();

                    // Submission state, all worker-local. `tails` holds one
                    // in-flight tail event per queue (schedule-assigned
                    // cross-queue waits resolve against these). `writer`
                    // maps each chunk to its last writer (RAW) and
                    // `last_use` to its last touch of any kind (WAR) —
                    // reused stable addresses stay ordered against prior
                    // in-flight work on other queues.
                    let mut tails: Vec<*mut c_void> = vec![ptr::null_mut(); queues.len()];
                    let mut writer: Map<ChunkId, (usize, *mut c_void)> = Map::with_hasher(BuildHasherDefault::<FHasher>::new());
                    let mut last_use: Map<ChunkId, (usize, *mut c_void)> = Map::with_hasher(BuildHasherDefault::<FHasher>::new());

                    // Unblock SIGABRT so it can be delivered (absorbed by the no-op handler)
                    const SIG_UNBLOCK: i32 = 1;
                    unsafe {
                        let mut mask = std::mem::MaybeUninit::<[u8; 128]>::uninit();
                        sigemptyset(mask.as_mut_ptr().cast());
                        sigaddset(mask.as_mut_ptr().cast(), SIGABRT);
                        pthread_sigmask(SIG_UNBLOCK, mask.as_ptr().cast(), ptr::null_mut());
                    }

                    'work_thread_loop: while let Ok(cmd) = rx.recv() {
                        match cmd {
                            Command::Allocate { bytes, reply } => {
                                // Best-fit from the free list first (stable
                                // addresses across repeats — those bytes were
                                // already paid for); else fresh.
                                let best = free_set
                                    .iter()
                                    .filter_map(|id| {
                                        let b = &buffers[*id];
                                        (b.bytes >= bytes).then_some((b.bytes, *id))
                                    })
                                    .min();
                                if let Some((_, id)) = best {
                                    free_set.remove(&id);
                                    let _ = reply.send(Ok(id));
                                    continue;
                                }
                                if bytes > free_bytes_atomic.load(Ordering::SeqCst) as i64 {
                                    let _ = reply.send(Err(BackendError {
                                        status: ErrorStatus::MemoryAllocation,
                                        context: "Allocation failure".into(),
                                    }));
                                    continue 'work_thread_loop;
                                }
                                let mut status = OpenCLStatus::CL_SUCCESS;
                                let buffer = unsafe {
                                    clCreateBuffer(context, CL_MEM_READ_WRITE, bytes as usize, ptr::null_mut(), &raw mut status)
                                };
                                if let Err(e) = status.check(ErrorStatus::MemoryAllocation) {
                                    let _ = reply.send(Err(e));
                                    continue 'work_thread_loop;
                                }
                                free_bytes_atomic.fetch_sub(bytes as u64, Ordering::SeqCst);
                                let id = buffers.push(OpenCLBuffer { ptr: buffer, bytes });
                                let _ = reply.send(Ok(id));
                            }
                            Command::Release { buffer_id } => {
                                // Put the id on the free list for
                                // stable-address reuse. Frees nothing: VRAM is
                                // reclaimed only by Dispose. Chunk dependency
                                // state (`writer`/`last_use`) stays: a claimed
                                // address keeps waiting on its prior work.
                                if !buffers.contains_id(buffer_id) {
                                    debug_assert!(false, "release of unknown OpenCL buffer {buffer_id:?}");
                                    continue;
                                }
                                debug_assert!(!free_set.contains(&buffer_id), "double release of OpenCL buffer {buffer_id:?}");
                                free_set.insert(buffer_id);
                            }
                            Command::Dispose => {
                                // The only reclamation: drain every in-order
                                // queue, release every tracked event, then
                                // free the whole list at once. Every released
                                // id becomes invalid.
                                for q in &queues {
                                    let _ = unsafe { (clFinish)(q.queue) }.check(ErrorStatus::MemoryDeallocation);
                                }
                                release_all_events(&mut tails, &mut writer, &mut last_use, clReleaseEvent);
                                for buffer_id in core::mem::take(&mut free_set) {
                                    let OpenCLBuffer { ptr, bytes } = buffers[buffer_id];
                                    debug_assert!(!ptr.is_null(), "deallocating null buffer is invalid");
                                    let _ = unsafe { clReleaseMemObject(ptr) }.check(ErrorStatus::MemoryDeallocation);
                                    free_bytes_atomic.fetch_add(bytes as u64, Ordering::SeqCst);
                                    buffers.remove(buffer_id);
                                }
                            }
                            Command::TryReuse { buffer_ids, reply } => {
                                // All-or-nothing: every id must be on the free
                                // list, else some address moved and the graph
                                // must be retraced.
                                let ok = buffer_ids.iter().all(|id| free_set.contains(id));
                                if ok {
                                    for id in &buffer_ids {
                                        free_set.remove(id);
                                    }
                                }
                                let _ = reply.send(Ok(ok));
                            }
                            Command::Copy { src_pool, src_buf, src_ptr, bytes, dst_buf, reply } => {
                                // Blocking: drain every queue (prior launches
                                // on any queue are ordered before this), run
                                // the transfer on queue 0, reply after the
                                // data arrived. Same-pool pairs use
                                // CopyBuffer; hosted sources (host staging or
                                // a disk mapping) upload from `src_ptr`.
                                for q in &queues {
                                    if let Err(err) = unsafe { (clFinish)(q.queue) }.check(ErrorStatus::MemoryCopyP2P) {
                                        let _ = reply.send(Err(err));
                                        continue 'work_thread_loop;
                                    }
                                }
                                release_all_events(&mut tails, &mut writer, &mut last_use, clReleaseEvent);
                                let dst_ptr = buffers[dst_buf].ptr;
                                debug_assert!(!dst_ptr.is_null(), "copy into null memory. Internal bug.");
                                let status = if src_pool == worker_pool {
                                    let src_ptr = buffers[src_buf].ptr;
                                    debug_assert!(!src_ptr.is_null(), "copy from null memory. Internal bug.");
                                    unsafe {
                                        (clEnqueueCopyBuffer)(
                                            queues[0].queue,
                                            src_ptr,
                                            dst_ptr,
                                            0,
                                            0,
                                            bytes as usize,
                                            0,
                                            ptr::null(),
                                            ptr::null_mut(),
                                        )
                                    }
                                    .check(ErrorStatus::MemoryCopyP2P)
                                } else {
                                    debug_assert!(!src_ptr.is_null(), "copy from null host memory. Internal bug.");
                                    unsafe {
                                        (clEnqueueWriteBuffer)(
                                            queues[0].queue,
                                            dst_ptr,
                                            CL_BLOCKING,
                                            0,
                                            bytes as usize,
                                            src_ptr.cast(),
                                            0,
                                            ptr::null(),
                                            ptr::null_mut(),
                                        )
                                    }
                                    .check(ErrorStatus::MemoryCopyH2P)
                                };
                                let _ = reply.send(status);
                            }
                            Command::PoolToHost { src, dst, bytes, reply } => {
                                // Sync point: drain every queue, then read
                                // back on queue 0.
                                for q in &queues {
                                    if let Err(err) = unsafe { (clFinish)(q.queue) }.check(ErrorStatus::MemoryCopyP2H) {
                                        let _ = reply.send(Err(err));
                                        continue 'work_thread_loop;
                                    }
                                }
                                release_all_events(&mut tails, &mut writer, &mut last_use, clReleaseEvent);
                                let OpenCLBuffer { ptr, .. } = buffers[src];
                                debug_assert!(!ptr.is_null(), "Trying to read null memory. Internal bug.");
                                let status = unsafe {
                                    (clEnqueueReadBuffer)(
                                        queues[0].queue,
                                        ptr,
                                        CL_BLOCKING,
                                        0,
                                        bytes as usize,
                                        dst.cast(),
                                        0,
                                        ptr::null(),
                                        ptr::null_mut(),
                                    )
                                }
                                .check(ErrorStatus::MemoryCopyP2H);
                                let _ = reply.send(status);
                            }
                            Command::Compile { name, source, lws, gws, params, reply } => {
                                let sources: &[&str] = &[source.as_str()];
                                let mut status = OpenCLStatus::CL_SUCCESS;
                                let program = unsafe {
                                    clCreateProgramWithSource(
                                        context,
                                        1,
                                        sources.as_ptr().cast(),
                                        [source.len()].as_ptr(),
                                        &raw mut status,
                                    )
                                };
                                if let Err(e) = status.check(ErrorStatus::KernelCompilation) {
                                    let _ = reply.send(Err(e));
                                    continue 'work_thread_loop;
                                }
                                if let Err(e) = unsafe {
                                    clBuildProgram(
                                        program,
                                        cl_uint::try_from(devices.len()).expect("So many devices..."),
                                        devices.as_ptr(),
                                        c"-cl-finite-math-only -cl-no-signed-zeros -cl-mad-enable".as_ptr().cast(),
                                        None,
                                        ptr::null_mut(),
                                    )
                                }
                                .check(ErrorStatus::KernelCompilation)
                                {
                                    // Try to get build log from first device
                                    let build_log =
                                        get_program_build_data(program, devices[0], clGetProgramBuildInfo, CL_PROGRAM_BUILD_LOG);
                                    match build_log {
                                        Ok(build_log) => {
                                            panic!("{e:?} {}", String::from_utf8_lossy(&build_log));
                                        }
                                        Err(status) => {
                                            let _ = reply.send(Err(status.check(ErrorStatus::KernelCompilation).err().unwrap()));
                                            continue 'work_thread_loop;
                                        }
                                    }
                                }
                                let mut status = OpenCLStatus::CL_SUCCESS;
                                let program_name = &CString::new(name.as_ref()).unwrap();
                                let kernel = unsafe { clCreateKernel(program, program_name.as_ptr().cast(), &raw mut status) };
                                if let Err(e) = status.check(ErrorStatus::KernelCompilation) {
                                    let _ = reply.send(Err(e));
                                    continue 'work_thread_loop;
                                }
                                let program_id = programs.push(OpenCLProgram { program, kernel, lws, gws, params });
                                let _ = reply.send(Ok(program_id));
                            }
                            Command::Replay { cmds, queues: assign, waits, bound, vars, deaths, reply } => {
                                // One roundtrip per partition replay.
                                // Ownership split: bound inputs and escaping
                                // defs are caller-side Placements (Drop
                                // releases them). Intermediaries never become
                                // Placements: the worker allocates them here
                                // and returns them to the free list at death.
                                // Allocation is per-command, parallelism-first:
                                // each launch's outputs reuse a free chunk
                                // only when it adds no wait (same queue or
                                // tail already complete); otherwise fresh VRAM
                                // — blocking reuse to save memory would
                                // serialize parallel chains. Deterministic
                                // order in, stable addresses out.
                                let result = (|| -> Result<Vec<(OpId, ChunkId)>, BackendError> {
                                    let vars_map: Map<OpId, Constant> = vars.into_iter().collect();
                                    let mut slot_chunk: Map<OpId, ChunkId> = bound.into_iter().collect();
                                    let mut fresh: Vec<(OpId, ChunkId)> = Vec::new();
                                    debug_assert_eq!(cmds.len(), assign.len(), "replay queue assignment length mismatch");
                                    debug_assert_eq!(cmds.len(), waits.len(), "replay wait set length mismatch");
                                    for (idx, cmd) in cmds.iter().enumerate() {
                                        match cmd {
                                            Cmd::Launch { program, args, outputs } => {
                                                let queue = assign[idx];
                                                for (slot, dtype, dims) in outputs {
                                                    if slot_chunk.contains_key(slot) {
                                                        continue;
                                                    }
                                                    let bytes =
                                                        dims.iter().map(|d| d.eval(&vars_map)).fold(*dtype, |a, b| a * b);
                                                    if bytes < 0 {
                                                        return Err(BackendError {
                                                            status: ErrorStatus::MemoryAllocation,
                                                            context: format!(
                                                                "replay allocated negative bytes for {slot:?}"
                                                            )
                                                            .into(),
                                                        });
                                                    }
                                                    let id = best_parallel_fit(
                                                        &free_set,
                                                        &buffers,
                                                        &writer,
                                                        &last_use,
                                                        queue,
                                                        bytes,
                                                        clGetEventInfo,
                                                    )
                                                    .or_else(|| {
                                                        if bytes > free_bytes_atomic.load(Ordering::SeqCst) as i64 {
                                                            return None;
                                                        }
                                                        let mut status = OpenCLStatus::CL_SUCCESS;
                                                        let buffer = unsafe {
                                                            clCreateBuffer(
                                                                context,
                                                                CL_MEM_READ_WRITE,
                                                                bytes as usize,
                                                                ptr::null_mut(),
                                                                &raw mut status,
                                                            )
                                                        };
                                                        if status.check(ErrorStatus::MemoryAllocation).is_err() {
                                                            return None;
                                                        }
                                                        free_bytes_atomic.fetch_sub(bytes as u64, Ordering::SeqCst);
                                                        Some(buffers.push(OpenCLBuffer { ptr: buffer, bytes }))
                                                    })
                                                    .ok_or(BackendError {
                                                        status: ErrorStatus::MemoryAllocation,
                                                        context: "Allocation failure".into(),
                                                    })?;
                                                    free_set.remove(&id);
                                                    slot_chunk.insert(*slot, id);
                                                    fresh.push((*slot, id));
                                                }
                                                submit_slots(
                                                    &programs,
                                                    &buffers,
                                                    program.program_id,
                                                    args,
                                                    &slot_chunk,
                                                    &vars_map,
                                                    queue,
                                                    &queues,
                                                    &waits[idx],
                                                    &mut tails,
                                                    &mut writer,
                                                    &mut last_use,
                                                    clEnqueueNDRangeKernel,
                                                    clSetKernelArg,
                                                    clReleaseEvent,
                                                )?;
                                            }
                                            Cmd::Alias { class, to } => {
                                                let chunk = *slot_chunk.get(to).ok_or_else(|| BackendError {
                                                    status: ErrorStatus::KernelLaunch,
                                                    context: format!("replay: alias target {to:?} is unplaced").into(),
                                                })?;
                                                slot_chunk.insert(*class, chunk);
                                            }
                                            Cmd::Copy { .. } => {
                                                unreachable!("copies are Copy partitions, never device runs")
                                            }
                                        }
                                        for dead in &deaths[idx] {
                                            if let Some(chunk) = slot_chunk.remove(dead) {
                                                // Worker-allocated intermediary:
                                                // back to the free list, it
                                                // never became a Placement so
                                                // no Drop is involved. Bound
                                                // inputs stay caller-owned —
                                                // the caller's Drop releases
                                                // those. Alias-shared chunks
                                                // die twice; the guard keeps
                                                // one entry.
                                                if fresh.iter().any(|(s, _)| s == dead)
                                                    && !free_set.contains(&chunk)
                                                {
                                                    free_set.insert(chunk);
                                                }
                                            }
                                        }
                                    }
                                    // Survivors escape the partition: caller
                                    // turns them into Placements. The freed
                                    // intermediaries are gone from the map.
                                    Ok(fresh.into_iter().filter(|(s, _)| slot_chunk.contains_key(s)).collect())
                                })();
                                let _ = reply.send(result);
                            }
                            Command::LaunchTimed { program_id, args, reply } => {
                                // Uncontended timing for autotune: drain
                                // every queue, then run the kernel solo on
                                // queue 0.
                                for q in &queues {
                                    if let Err(err) = unsafe { (clFinish)(q.queue) }.check(ErrorStatus::KernelSync) {
                                        let _ = reply.send(Err(err));
                                        continue 'work_thread_loop;
                                    }
                                }
                                release_all_events(&mut tails, &mut writer, &mut last_use, clReleaseEvent);
                                let start = Instant::now();
                                let result = submit_launch(
                                    &programs,
                                    &buffers,
                                    worker_pool,
                                    program_id,
                                    &args,
                                    &queues,
                                    &mut tails,
                                    &mut writer,
                                    &mut last_use,
                                    clEnqueueNDRangeKernel,
                                    clSetKernelArg,
                                    clReleaseEvent,
                                )
                                .and_then(|()| unsafe { (clFinish)(queues[0].queue) }.check(ErrorStatus::KernelSync));
                                release_all_events(&mut tails, &mut writer, &mut last_use, clReleaseEvent);
                                let nanos = start.elapsed().as_nanos() as u64;
                                let _ = reply.send(result.map(|()| nanos));
                            }
                            Command::ReleaseProgram { program_id } => {
                                let _ = unsafe { clReleaseProgram(programs[program_id].program) }
                                    .check(ErrorStatus::Deinitialization);
                                programs.remove(program_id);
                            }
                        }
                    }
                    //println!("DEINIT receiver");
                }
            });

            pools.push(Mutex::new(OpenCLMemoryPool { tx: tx.clone(), total_bytes, free_bytes: free_bytes_atomic, dev_info }));
        }
    }
    #[allow(unused)]
    let _ = library;
    Ok(pools)
}

pub(super) fn device(id: u16) -> Result<&'static Mutex<OpenCLDevice>, BackendError> {
    backend()?.1.get(id as usize).ok_or_else(|| BackendError {
        status: ErrorStatus::Initialization,
        context: format!("Dev::OpenCL({id}) is not available").into(),
    })
}

pub(super) fn device_count() -> u16 {
    backend().map(|(_, devs)| devs.len() as u16).unwrap_or(0)
}

/// Releases every tracked event exactly once and clears the tracking:
///
/// per-queue tails plus the per-chunk writer/last_use maps (one event can
/// sit in several of them). Used after full drains, where completion is
/// guaranteed.
fn release_all_events(
    tails: &mut Vec<*mut c_void>,
    writer: &mut Map<ChunkId, (usize, *mut c_void)>,
    last_use: &mut Map<ChunkId, (usize, *mut c_void)>,
    clReleaseEvent: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
) {
    let mut events: Vec<*mut c_void> = tails.iter().copied().collect();
    events.extend(writer.values().map(|&(_, e)| e));
    events.extend(last_use.values().map(|&(_, e)| e));
    release_distinct(events, clReleaseEvent);
    for tail in tails.iter_mut() {
        *tail = ptr::null_mut();
    }
    writer.clear();
    last_use.clear();
}

/// Enqueues one kernel with pre-resolved args onto the given queue and
/// tracks the completion event: the queue tail plus per-chunk RAW
/// (writer) and WAR (last_use) entries. Waits cover the schedule's
/// cross-queue wait set plus any in-flight chunk tails on other queues —
/// reused stable addresses stay ordered against prior work.
#[allow(clippy::too_many_arguments)]
fn enqueue_tracked(
    programs: &Slab<DeviceProgramId, OpenCLProgram>,
    program_id: DeviceProgramId,
    params: &[(*const c_void, usize)],
    global_size: Vec<Dim>,
    reads: &[ChunkId],
    writes: &[ChunkId],
    queue: usize,
    queues: &[OpenCLQueue],
    wait_queues: &[usize],
    tails: &mut Vec<*mut c_void>,
    writer: &mut Map<ChunkId, (usize, *mut c_void)>,
    last_use: &mut Map<ChunkId, (usize, *mut c_void)>,
    clEnqueueNDRangeKernel: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        *const usize,
        *const usize,
        *const usize,
        cl_uint,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> OpenCLStatus,
    clSetKernelArg: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *const c_void) -> OpenCLStatus,
    clReleaseEvent: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
) -> Result<(), BackendError> {
    debug_assert!(programs.contains_id(program_id), "launch of unknown program {program_id:?}");
    debug_assert!(queue < queues.len(), "launch on missing queue {queue}");
    let program = &programs[program_id];
    // Callers always pass the full 3D grid (missing group factors are
    // single groups); work_dim is 3 and every ranged axis is addressable.
    // A caller-side empty grid would be work_dim=0, which is invalid.
    debug_assert!(!global_size.is_empty(), "empty global work size is work_dim=0");
    let mut i: u32 = 0;
    for (arg_ptr, arg_size) in params {
        unsafe { (clSetKernelArg)(program.kernel, i, *arg_size, *arg_ptr) }.check(ErrorStatus::IncorrectKernelArg)?;
        i += 1;
    }
    // The driver requires exactly work_dim local sizes: the stored triple
    // pairs with the 3D grid by construction.
    let lws_ptr = if program.lws.is_empty() {
        ptr::null()
    } else {
        program.lws[..global_size.len().min(program.lws.len())].as_ptr().cast()
    };
    // Global work size is checked against the device grid limits before
    // enqueueing (same values the compile-time check in gws_from_kernel uses).
    let max_grid = [2_147_483_647i64, 65_535, 65_535];
    for (i, g) in global_size.iter().enumerate().take(3) {
        if *g < 0 || *g > max_grid[i] {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("global work size dim {i} {g} exceeds device max {}", max_grid[i]).into(),
            });
        }
    }
    // One event per needed edge; the enqueued command retains them.
    let mut waits: Vec<*mut c_void> = Vec::new();
    for q in wait_queues {
        if *q != queue
            && let Some(&event) = tails.get(*q)
            && !event.is_null()
            && !waits.contains(&event)
        {
            waits.push(event);
        }
    }
    for b in reads {
        if let Some(&(w, event)) = writer.get(b)
            && w != queue
            && !event.is_null()
            && !waits.contains(&event)
        {
            waits.push(event);
        }
    }
    for b in writes {
        if let Some(&(u, event)) = last_use.get(b)
            && u != queue
            && !event.is_null()
            && !waits.contains(&event)
        {
            waits.push(event);
        }
    }
    let mut event: *mut c_void = ptr::null_mut();
    unsafe {
        (clEnqueueNDRangeKernel)(
            queues[queue].queue,
            program.kernel,
            u32::try_from(global_size.len()).unwrap_or(3),
            ptr::null(),
            global_size.as_ptr().cast(),
            lws_ptr,
            u32::try_from(waits.len()).unwrap_or(0),
            if waits.is_empty() { ptr::null() } else { waits.as_ptr() },
            &raw mut event,
        )
    }
    .check(ErrorStatus::KernelLaunch)?;
    debug_assert!(!event.is_null(), "kernel enqueue returned no event");
    // Overwrite, never eagerly release: a replaced event can still be
    // referenced from another slot (tails of other queues, other chunks),
    // and releasing it there would double-release. Replaced events stay
    // tracked until the next full drain, which releases everything exactly
    // once (drains bound event lifetime: every readback/autotune probe
    // drains, so nothing accumulates beyond one sync interval).
    tails[queue] = event;
    let mut reads = reads.to_vec();
    reads.sort();
    reads.dedup();
    let mut writes = writes.to_vec();
    writes.sort();
    writes.dedup();
    for b in &reads {
        last_use.insert(*b, (queue, event));
    }
    for b in &writes {
        writer.insert(*b, (queue, event));
        last_use.insert(*b, (queue, event));
    }
    Ok(())
}

/// Best-fit free chunk that adds no wait on `queue`: same-queue tails
/// order themselves on the in-order queue, completed tails need nothing,
/// and only an in-flight tail on another queue blocks. `None` means
/// allocate fresh — parallelism over memory savings, always.
fn best_parallel_fit(
    free_set: &Set<ChunkId>,
    buffers: &Slab<ChunkId, OpenCLBuffer>,
    writer: &Map<ChunkId, (usize, *mut c_void)>,
    last_use: &Map<ChunkId, (usize, *mut c_void)>,
    queue: usize,
    bytes: Dim,
    clGetEventInfo: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus,
) -> Option<ChunkId> {
    let mut cands: Vec<(Dim, ChunkId)> = free_set
        .iter()
        .filter_map(|id| {
            let b = &buffers[*id];
            (b.bytes >= bytes).then_some((b.bytes, *id))
        })
        .collect();
    cands.sort_unstable();
    cands.into_iter().find_map(|(_, id)| {
        let blocked = [writer.get(&id), last_use.get(&id)].into_iter().flatten().any(|&(q, event)| {
            q != queue && !event.is_null() && event_in_flight(event, clGetEventInfo)
        });
        (!blocked).then_some(id)
    })
}

/// True while the event's command has not completed. A failed query is
/// in-flight (conservative): never reuse under an unknown tail.
fn event_in_flight(
    event: *mut c_void,
    clGetEventInfo: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus,
) -> bool {
    let mut status: cl_int = 0;
    let queried = unsafe {
        (clGetEventInfo)(
            event,
            CL_EVENT_COMMAND_EXECUTION_STATUS,
            core::mem::size_of::<cl_int>(),
            (&raw mut status).cast(),
            ptr::null_mut(),
        )
    } == OpenCLStatus::CL_SUCCESS;
    !queried || status != CL_COMPLETE
}

/// Resolves queue-local slots against the replay's slot→chunk map (scalars
/// from `vars`), evaluates the grid, and enqueues with tracking. The
/// partition-replay path: args are slots, not placements.
#[allow(clippy::too_many_arguments)]
fn submit_slots(
    programs: &Slab<DeviceProgramId, OpenCLProgram>,
    buffers: &Slab<ChunkId, OpenCLBuffer>,
    program_id: DeviceProgramId,
    args: &[OpId],
    slot_chunk: &Map<OpId, ChunkId>,
    scalars: &Map<OpId, Constant>,
    queue: usize,
    queues: &[OpenCLQueue],
    wait_queues: &[usize],
    tails: &mut Vec<*mut c_void>,
    writer: &mut Map<ChunkId, (usize, *mut c_void)>,
    last_use: &mut Map<ChunkId, (usize, *mut c_void)>,
    clEnqueueNDRangeKernel: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        *const usize,
        *const usize,
        *const usize,
        cl_uint,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> OpenCLStatus,
    clSetKernelArg: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *const c_void) -> OpenCLStatus,
    clReleaseEvent: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
) -> Result<(), BackendError> {
    debug_assert!(programs.contains_id(program_id), "launch of unknown program {program_id:?}");
    let kinds: &[ParamKind] = &programs[program_id].params;
    debug_assert!(args.len() <= kinds.len(), "more launch args than program params");
    // clSetKernelArg copies the value immediately, so stable storage for
    // the cl_mem handles and scalar bytes within this call is enough —
    // reserved up front: `params` below borrows these vecs while they
    // grow, and any reallocation would dangle the stored pointers.
    let mut mem_args: Vec<*mut c_void> = Vec::with_capacity(args.len());
    let mut scalar_values: Vec<Box<[u8]>> = Vec::with_capacity(args.len());
    let mut params: Vec<(*const c_void, usize)> = Vec::with_capacity(args.len());
    let mut reads: Vec<ChunkId> = Vec::new();
    let mut writes: Vec<ChunkId> = Vec::new();
    for (idx, slot) in args.iter().enumerate() {
        if let Some(chunk) = slot_chunk.get(slot) {
            mem_args.push(buffers[*chunk].ptr);
            let value = mem_args.last().unwrap();
            params.push((core::ptr::from_ref(value).cast(), core::mem::size_of::<*mut c_void>()));
            match kinds.get(idx) {
                Some(ParamKind::GlobalMut) => writes.push(*chunk),
                _ => reads.push(*chunk),
            }
        } else if let Some(constant) = scalars.get(slot) {
            scalar_values.push(constant.to_le_bytes().into());
            let value = scalar_values.last().unwrap();
            params.push((value.as_ptr().cast(), value.len()));
        } else {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("replay: launch slot {slot:?} is neither placed nor bound").into(),
            });
        }
    }
    let global_size: Vec<Dim> = (0..3)
        .map(|i| {
            let g = programs[program_id]
                .gws
                .get(i)
                .map(|gdim| {
                    gdim.eval(&mut |ordinal| {
                        scalars
                            .get(&args[ordinal])
                            .and_then(|c| c.as_dim())
                            .expect("gws param must be a Variable slot")
                    })
                })
                .unwrap_or(1);
            // 3D grid always (mirroring CUDA): a missing group factor is a
            // single group, never a dropped axis. Truncating to the zipped
            // length orphans local-only axes, whose indices then read
            // garbage from nonexistent dimensions.
            g * programs[program_id].lws.get(i).copied().unwrap_or(1)
        })
        .collect();
    enqueue_tracked(
        programs,
        program_id,
        &params,
        global_size,
        &reads,
        &writes,
        queue,
        queues,
        wait_queues,
        tails,
        writer,
        last_use,
        clEnqueueNDRangeKernel,
        clSetKernelArg,
        clReleaseEvent,
    )
}

/// Resolves `LaunchArg` placements against the buffer table through the
/// shard addressed to this worker's pool, evaluates the grid, and
/// enqueues with tracking. The autotune (`LaunchTimed`) path: args carry
/// placements, not slots.
#[allow(clippy::too_many_arguments)]
fn submit_launch(
    programs: &Slab<DeviceProgramId, OpenCLProgram>,
    buffers: &Slab<ChunkId, OpenCLBuffer>,
    worker_pool: Pool,
    program_id: DeviceProgramId,
    args: &[LaunchArg],
    queues: &[OpenCLQueue],
    tails: &mut Vec<*mut c_void>,
    writer: &mut Map<ChunkId, (usize, *mut c_void)>,
    last_use: &mut Map<ChunkId, (usize, *mut c_void)>,
    clEnqueueNDRangeKernel: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        *const usize,
        *const usize,
        *const usize,
        cl_uint,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> OpenCLStatus,
    clSetKernelArg: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *const c_void) -> OpenCLStatus,
    clReleaseEvent: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
) -> Result<(), BackendError> {
    debug_assert!(programs.contains_id(program_id), "launch of unknown program {program_id:?}");
    let kinds: &[ParamKind] = &programs[program_id].params;
    debug_assert!(args.len() <= kinds.len(), "more launch args than program params");
    // clSetKernelArg copies the value immediately, so stable storage for
    // the cl_mem handles and scalar bytes within this call is enough —
    // reserved up front: `params` below borrows these vecs while they
    // grow, and any reallocation would dangle the stored pointers.
    let mut mem_args: Vec<*mut c_void> = Vec::with_capacity(args.len());
    let mut scalar_values: Vec<Box<[u8]>> = Vec::with_capacity(args.len());
    let mut params: Vec<(*const c_void, usize)> = Vec::with_capacity(args.len());
    let mut reads: Vec<ChunkId> = Vec::new();
    let mut writes: Vec<ChunkId> = Vec::new();
    for (idx, arg) in args.iter().enumerate() {
        match arg {
            LaunchArg::Buffer(placement) => {
                let chunk = placement
                    .shards
                    .iter()
                    .find(|shard| shard.pool == worker_pool)
                    .ok_or(BackendError {
                        status: ErrorStatus::KernelLaunch,
                        context: "launch arg has no shard on this pool".into(),
                    })?
                    .chunk;
                mem_args.push(buffers[chunk].ptr);
                let value = mem_args.last().unwrap();
                params.push((core::ptr::from_ref(value).cast(), core::mem::size_of::<*mut c_void>()));
                match kinds.get(idx) {
                    Some(ParamKind::GlobalMut) => writes.push(chunk),
                    _ => reads.push(chunk),
                }
            }
            LaunchArg::Variable(constant) => {
                scalar_values.push(constant.to_le_bytes().into());
                let value = scalar_values.last().unwrap();
                params.push((value.as_ptr().cast(), value.len()));
            }
        }
    }
    let global_size: Vec<Dim> = (0..3)
        .map(|i| {
            let g = programs[program_id]
                .gws
                .get(i)
                .map(|gdim| {
                    gdim.eval(&mut |ordinal| match &args[ordinal] {
                        LaunchArg::Variable(c) => c.as_dim().unwrap(),
                        LaunchArg::Buffer(_) => unreachable!("gws param must be a Variable launch arg"),
                    })
                })
                .unwrap_or(1);
            // 3D grid always (mirroring CUDA): a missing group factor is a
            // single group, never a dropped axis. Truncating to the zipped
            // length orphans local-only axes, whose indices then read
            // garbage from nonexistent dimensions.
            g * programs[program_id].lws.get(i).copied().unwrap_or(1)
        })
        .collect();
    enqueue_tracked(
        programs,
        program_id,
        &params,
        global_size,
        &reads,
        &writes,
        0,
        queues,
        &[],
        tails,
        writer,
        last_use,
        clEnqueueNDRangeKernel,
        clSetKernelArg,
        clReleaseEvent,
    )
}

/// Releases each distinct non-null event exactly once (writer and last_use
/// can hold the same event for a buffer).
fn release_distinct(mut events: Vec<*mut c_void>, clReleaseEvent: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus) {
    events.retain(|event| !event.is_null());
    events.sort();
    events.dedup();
    for event in events {
        let _ = unsafe { (clReleaseEvent)(event) }.check(ErrorStatus::Deinitialization);
    }
}

impl OpenCLMemoryPool {
    pub fn free_bytes(&self) -> Dim {
        self.free_bytes.load(Ordering::SeqCst) as i64
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<ChunkId, BackendError> {
        let (reply, reply_rx) = channel();
        self.tx.send(Command::Allocate { bytes, reply }).unwrap();
        reply_rx.recv().unwrap()
    }

    /// Put a buffer on the free list for stable-address reuse. Frees
    /// nothing; VRAM is reclaimed only by [`OpenCLMemoryPool::dispose`].
    pub fn release(&mut self, buffer_id: ChunkId) {
        self.tx.send(Command::Release { buffer_id }).unwrap();
    }

    /// Free all buffers in the free list at once (hard sync first). The
    /// only VRAM reclamation; every released id becomes invalid.
    pub fn dispose(&mut self) {
        self.tx.send(Command::Dispose).unwrap();
    }

    /// Tries to reuse existing allocations, all or nothing: every id must
    /// be on the free list (stable addresses) or nothing is claimed.
    pub fn try_reuse_allocations(&mut self, buffer_ids: &Set<ChunkId>) -> bool {
        let (reply, reply_rx) = channel();
        self.tx.send(Command::TryReuse { buffer_ids: buffer_ids.clone(), reply }).unwrap();
        reply_rx.recv().unwrap().unwrap_or(false)
    }

    /// Blocking read-back (sync point).
    pub fn pool_to_host(&mut self, src: ChunkId, dst: &mut [u8]) -> Result<(), BackendError> {
        let (reply, reply_rx) = channel();
        self.tx.send(Command::PoolToHost { src, dst: dst.as_mut_ptr(), bytes: dst.len() as Dim, reply }).unwrap();
        reply_rx.recv().unwrap()
    }
}

impl OpenCLDevice {
    pub fn info(&self) -> Arc<DeviceInfo> {
        self.dev_info.clone()
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        // --- Codegen ---
        let mut lws = vec![1i64; 3];
        let mut op_id = kernel.head;
        let mut steps_op_id = 0usize;
        while !op_id.is_null() {
            steps_op_id += 1;
            if steps_op_id > 10_000 {
                panic!("compile did not finish in 10000 steps");
            }
            if let Op::Range { axis, kind: scope } = kernel.ops[op_id].op {
                match scope {
                    RangeKind::Group(_) => {}
                    RangeKind::Local(len) => lws[axis as usize] = i64::from(len),
                    // A warp is a view over a local range — adds no threads.
                    RangeKind::Warp(_) => {}
                }
            }
            op_id = kernel.next_op(op_id);
        }

        if lws.iter().product::<i64>() > self.dev_info.max_local_threads as i64 {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "Invalid local work size.".into() });
        }

        let name = format!("k_{}", lws.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"),);

        let source = kernel.generate_opencl(&name)?;
        if debug_asm {
            println!();
            println!("{source}");
        }

        let gws = gws_from_kernel(kernel, &self.dev_info.max_global_work_dims)?;
        // Collect per-Param kinds in head order — used by the submission path
        // to split launch args into reads (Global) and writes (GlobalMut).
        let mut params = Vec::new();
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            if let Op::Param { kind, .. } = kernel.ops[op_id].op {
                params.push(kind);
            }
            op_id = kernel.next_op(op_id);
        }
        let (reply, reply_rx) = channel();
        self.tx.send(Command::Compile { name: name.into(), source, lws, gws, params, reply }).unwrap();
        reply_rx.recv().unwrap()
    }

    /// Timed launch for autotune: the worker drains every queue first,
    /// then the kernel runs solo on queue 0; the wall-clock nanos are
    /// measured around enqueue+finish.
    pub fn launch_timed(&mut self, program_id: DeviceProgramId, args: &[LaunchArg]) -> Result<u64, BackendError> {
        let (reply, reply_rx) = channel();
        self.tx.send(Command::LaunchTimed { program_id, args: args.to_vec(), reply }).unwrap();
        reply_rx.recv().unwrap()
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        //println!("[OPENCL] release program_id={program_id:?}");
        self.tx.send(Command::ReleaseProgram { program_id }).unwrap();
    }

    pub fn free_compute(&self) -> u128 {
        self.dev_info.compute
    }
}

/// Preplanned OpenCL partition: the ordered commands, per-command death
/// lists, and the static queue assignment (per-command queue plus
/// cross-queue wait sets). Replay ships the whole partition to the worker
/// in one roundtrip (bound slots, scalars, assignment); the worker
/// allocates unbound defs per-command (parallelism-first: fresh VRAM over
/// blocking reuse), submits every launch to its queue, and releases dead
/// chunks. No capture: repeats resubmit, stable addresses come from
/// deterministic alloc order over the free list.
#[derive(Debug)]
pub(crate) struct OpenCLPartition {
    cmds: Vec<Cmd>,
    deaths: Vec<Vec<OpId>>,
    queues: Vec<usize>,
    waits: Vec<Vec<usize>>,
    pub(crate) dev: u16,
}

impl OpenCLDevice {
    pub(crate) fn schedule(cmds: Vec<Cmd>, outputs: &Set<OpId>, live_out: Set<OpId>, dev: u16) -> OpenCLPartition {
        // Deaths: a slot dies at its last read unless pinned (a plan
        // output or read after this partition).
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
        // Static queue assignment (slot affinity): a command joins its
        // read-slots' queue when they agree (the RAW wait disappears);
        // slot-disjoint chains spread round-robin so independent launches
        // overlap on different queues. Waits name queues; the worker
        // resolves them against per-queue tails at submit time.
        let nq = super::config().opencl.queues.unwrap_or(8).max(1);
        let mut slot_queue: Map<OpId, usize> = Map::default();
        let mut round_robin = 0usize;
        let mut assign: Vec<usize> = Vec::with_capacity(cmds.len());
        let mut waits: Vec<Vec<usize>> = Vec::with_capacity(cmds.len());
        for cmd in &cmds {
            match cmd {
                Cmd::Launch { args, outputs: defs, .. } => {
                    let mut qs: Vec<usize> = Vec::new();
                    for slot in args {
                        if let Some(&q) = slot_queue.get(slot)
                            && !qs.contains(&q)
                        {
                            qs.push(q);
                        }
                    }
                    let queue = qs.first().copied().unwrap_or_else(|| {
                        let q = round_robin % nq;
                        round_robin += 1;
                        q
                    });
                    waits.push(qs.into_iter().filter(|q| *q != queue).collect());
                    assign.push(queue);
                    for (slot, _, _) in defs {
                        slot_queue.insert(*slot, queue);
                    }
                }
                Cmd::Alias { class, to } => {
                    // Zero-cost rebind: no executable, no queue traffic.
                    let queue = slot_queue.get(to).copied().unwrap_or(0);
                    assign.push(queue);
                    waits.push(Vec::new());
                    slot_queue.insert(*class, queue);
                }
                Cmd::Copy { .. } => unreachable!("copies are Copy partitions, never device runs"),
            }
        }
        OpenCLPartition { cmds, deaths, queues: assign, waits, dev }
    }

    /// Copy executing a transfer into this device's pool: host uploads go
    /// through a blocking WriteBuffer, same-pool pairs through a blocking
    /// CopyBuffer, disk uploads read straight from the file mapping —
    /// everything else is later work. Single-shard placements only.
    pub fn copy(&self, src: &Placement, dst: &Placement, bytes: Dim) -> Result<(), BackendError> {
        debug_assert!(bytes >= 0, "OpenCL copy of negative bytes");
        let [src_shard] = &src.shards[..] else {
            todo!("OpenCL copy of multi-shard source placement")
        };
        let [dst_shard] = &dst.shards[..] else {
            todo!("OpenCL copy of multi-shard destination placement")
        };
        debug_assert_eq!(dst_shard.pool, self.memory_pool, "OpenCL copy destination is not on this device");
        let dead = |_| BackendError { status: ErrorStatus::MemoryCopyP2P, context: "opencl worker thread died".into() };
        let dead_rx = |_: std::sync::mpsc::RecvError| BackendError {
            status: ErrorStatus::MemoryCopyP2P,
            context: "opencl worker hung up".into(),
        };
        let (reply, reply_rx) = channel();
        let send = |cmd| self.tx.send(cmd).map_err(dead);
        match src_shard.pool {
            p if p == self.memory_pool => {
                send(Command::Copy {
                    src_pool: p,
                    src_buf: src_shard.chunk,
                    src_ptr: ptr::null(),
                    bytes,
                    dst_buf: dst_shard.chunk,
                    reply,
                })?;
            }
            Pool::Host => {
                // The host lock is held across the blocking roundtrip: the
                // source pointer stays valid. The worker never takes it.
                let host = super::host::pool();
                let pool = super::lock(Pool::Host, host);
                let src_ptr = pool.get_buffer(src_shard.chunk).as_ptr();
                send(Command::Copy {
                    src_pool: Pool::Host,
                    src_buf: src_shard.chunk,
                    src_ptr,
                    bytes,
                    dst_buf: dst_shard.chunk,
                    reply,
                })?;
            }
            #[cfg(unix)]
            Pool::Disk => {
                // Straight from the file mapping, no staging buffer. The
                // extent holds exact tensor bytes; the destination is
                // over-allocated — never read past the extent. The chunk
                // stays mapped across the blocking call (released
                // caller-side after the reply).
                let disk = super::disk::pool();
                let dpool = super::lock(Pool::Disk, disk);
                let (ptr, extent) = dpool.mapped_ptr(src_shard.chunk);
                let n = bytes.min(extent);
                send(Command::Copy {
                    src_pool: Pool::Disk,
                    src_buf: src_shard.chunk,
                    src_ptr: ptr,
                    bytes: n,
                    dst_buf: dst_shard.chunk,
                    reply,
                })?;
            }
            #[cfg(windows)]
            Pool::Disk => todo!("OpenCL copy from disk on windows"),
            p => todo!("OpenCL copy from {p:?}"),
        }
        reply_rx.recv().map_err(dead_rx)?
    }
}

impl OpenCLPartition {
    pub(crate) fn replay(
        &self,
        dev: &mut OpenCLDevice,
        resolved: &mut Map<OpId, Arc<Placement>>,
        vars: &Map<OpId, Constant>,
    ) -> Result<(), BackendError> {
        let my_pool = dev.memory_pool;
        // Bound slots addressed to this pool; caller-side defs stay
        // unbound for the worker's per-command allocator.
        let mut bound = Vec::with_capacity(resolved.len());
        for (slot, placement) in resolved.iter() {
            if let Some(shard) = placement.shards.iter().find(|s| s.pool == my_pool) {
                bound.push((*slot, shard.chunk));
            }
        }
        let vars_vec: Vec<(OpId, Constant)> = vars.iter().map(|(s, c)| (*s, *c)).collect();
        let dead = |_| BackendError { status: ErrorStatus::KernelLaunch, context: "opencl worker thread died".into() };
        let dead_rx = |_: std::sync::mpsc::RecvError| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: "opencl worker hung up".into(),
        };
        let (reply, reply_rx) = channel();
        dev.tx
            .send(Command::Replay {
                cmds: self.cmds.clone(),
                queues: self.queues.clone(),
                waits: self.waits.clone(),
                bound,
                vars: vars_vec,
                deaths: self.deaths.clone(),
                reply,
            })
            .map_err(dead)?;
        for (slot, chunk) in reply_rx.recv().map_err(dead_rx)?? {
            resolved.insert(slot, Arc::new(Placement { shards: vec![Shard { pool: my_pool, chunk }] }));
        }
        for dead in self.deaths.iter().flatten() {
            resolved.remove(dead);
        }
        Ok(())
    }
}

impl OpenCLStatus {
    fn check(self, status: ErrorStatus) -> Result<(), BackendError> {
        if self == Self::CL_SUCCESS {
            Ok(())
        } else {
            Err(BackendError { status, context: format!("{self:?}").into() })
        }
    }
}

fn query_device_info(
    device: *mut c_void,
    clGetDeviceInfo: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus,
    dev_info: &mut DeviceInfo,
    debug_dev: bool,
) -> Result<(), BackendError> {
    let device_name = get_device_data(device, clGetDeviceInfo, CL_DEVICE_NAME)?;
    let device_name = String::from_utf8(device_name).unwrap();
    let max_work_item_dims = get_device_data(device, clGetDeviceInfo, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)?;
    if debug_dev {
        println!("[opencl] {device_name}");
    }
    let max_work_item_dims = u32::from_ne_bytes(max_work_item_dims.try_into().unwrap()) as usize;
    let mwis = get_device_data(device, clGetDeviceInfo, CL_DEVICE_MAX_WORK_ITEM_SIZES)?;
    let mut max_local_work_dims: Vec<u32> = vec![0; max_work_item_dims];
    for i in 0..max_work_item_dims {
        let max_dim_size: usize = usize::from_ne_bytes([
            mwis[i * 8],
            mwis[i * 8 + 1],
            mwis[i * 8 + 2],
            mwis[i * 8 + 3],
            mwis[i * 8 + 4],
            mwis[i * 8 + 5],
            mwis[i * 8 + 6],
            mwis[i * 8 + 7],
        ]);
        max_local_work_dims[i] = max_dim_size as u32;
    }
    let mlt = 256;
    // No portable global-size query exists in OpenCL; hardcode the limits the
    // major implementations (NVIDIA/AMD) enforce: x up to 2^31-1, y/z 65535.
    let max_global_work_dims: Vec<Dim> = (0..max_work_item_dims)
        .map(|i| {
            if i == 0 {
                Dim::from(2_147_483_647i64)
            } else {
                Dim::from(65_535i64)
            }
        })
        .collect();
    *dev_info = DeviceInfo {
        compute: 1024 * 1024 * 1024,
        max_global_work_dims,
        max_local_threads: mlt,
        max_local_work_dims,
        preferred_vector_size: u8::try_from(u32::from_ne_bytes(
            get_device_data(device, clGetDeviceInfo, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)?.try_into().unwrap(),
        ))
        .expect("What a vector width...")
            * 4,
        local_mem_size: Dim::try_from(usize::from_ne_bytes(
            get_device_data(device, clGetDeviceInfo, CL_DEVICE_LOCAL_MEM_SIZE)?.try_into().unwrap(),
        ))
        .expect("What a memory size..."),
        max_register_bytes: 256,
        has_native_exp2: true,
        supported_vec_lens: vec![2, 3, 4, 8, 16],
        tensor_cores: false,
        tenstorrent: false,
        tile: [1, 1],
        tile_sizes: vec![],
        wmma_layouts: vec![],
        num_circular_buffers: 0,
        has_openmp: false,
        warp_size: {
            if let Ok(device_type_data) = get_device_data(device, clGetDeviceInfo, CL_DEVICE_TYPE) {
                let device_type = u64::from_ne_bytes(device_type_data.try_into().unwrap_or_default());
                if device_type & CL_DEVICE_TYPE_GPU != 0 { 64 } else { 1 }
            } else {
                1
            }
        },
        cc: [0, 0],
        dtype_capability: [DTypeCapability::all(); DType::N_DTYPES],
    };
    dev_info.dtype_capability[DType::BF16 as usize] = DTypeCapability::none();
    if let Ok(extensions) = get_device_data(device, clGetDeviceInfo, CL_DEVICE_EXTENSIONS) {
        let has_fp16 = extensions.split(|&b| b == b' ').any(|token| token == b"cl_khr_fp16");
        if !has_fp16 {
            dev_info.dtype_capability[DType::F16 as usize] = DTypeCapability::none();
        }
        let has_tensor = extensions.split(|&b| b == b' ').any(|token| token == b"cl_intel_subgroup_matrix_multiply_accumulate");
        if has_tensor {
            dev_info.tensor_cores = true;
        }
    }
    Ok(())
}

fn get_device_data(
    device: *mut c_void,
    clGetDeviceInfo: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus,
    param_name: cl_uint,
) -> Result<Vec<u8>, BackendError> {
    let size = {
        let mut size: usize = 0;
        let ocl_status = unsafe { clGetDeviceInfo(device, param_name, 0, ptr::null_mut(), &raw mut size) };
        if OpenCLStatus::CL_SUCCESS != ocl_status {
            return Err(BackendError {
                status: ErrorStatus::DeviceQuery,
                context: format!("Failed to get device info {param_name}, {ocl_status:?}").into(),
            });
        }
        Ok::<usize, BackendError>(size)
    }?;
    if 0 < size {
        let count = size / core::mem::size_of::<u8>();
        let mut data: Vec<u8> = Vec::with_capacity(count);
        unsafe {
            data.set_len(count);
            clGetDeviceInfo(device, param_name, size, data.as_mut_ptr().cast(), ptr::null_mut())
        }
        .check(ErrorStatus::DeviceQuery)?;
        Ok(data)
    } else {
        Ok(Vec::default())
    }
}

fn get_program_build_data(
    program: *mut c_void,
    device: *mut c_void,
    clGetProgramBuildInfo: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        usize,
        *mut c_void,
        *mut usize,
    ) -> OpenCLStatus,
    param_name: cl_uint,
) -> Result<Vec<u8>, OpenCLStatus> {
    let size = {
        let mut size: usize = 0;
        let status = unsafe { clGetProgramBuildInfo(program, device, param_name, 0, ptr::null_mut(), &raw mut size) };
        if OpenCLStatus::CL_SUCCESS == status {
            Ok(size)
        } else {
            Err(status)
        }
    }?;
    if 0 < size {
        let count = size / core::mem::size_of::<u8>();
        let mut data: Vec<u8> = Vec::with_capacity(count);
        let status = unsafe {
            data.set_len(count);
            clGetProgramBuildInfo(program, device, param_name, size, data.as_mut_ptr().cast(), ptr::null_mut())
        };
        if OpenCLStatus::CL_SUCCESS == status {
            Ok(data)
        } else {
            Err(status)
        }
    } else {
        Ok(Vec::default())
    }
}

type cl_int = i32;
type cl_uint = u32;
type cl_bitfield = u64;

const CL_PLATFORM_NAME: cl_uint = 0x0902; // 2306
const CL_DEVICE_NAME: cl_uint = 0x102B; // 4139
const CL_DEVICE_GLOBAL_MEM_SIZE: cl_uint = 0x101F; // 4127
const CL_DEVICE_LOCAL_MEM_SIZE: cl_uint = 0x1023; // 4131
//const CL_DEVICE_MAX_MEM_ALLOC_SIZE: cl_uint = 0x1010; // 4112
//const CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: cl_uint = 0x101A; // 4122
//const CL_DEVICE_MAX_WORK_GROUP_SIZE: cl_uint = 0x1004; // 4100
const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: cl_uint = 0x1003; // 4099
//const CL_DEVICE_MAX_PRIVATE_MEMORY_SIZE: cl_uint = 0x1160; // 4448
const CL_DEVICE_MAX_WORK_ITEM_SIZES: cl_uint = 0x1005; // 4101
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: cl_uint = 0x100A; // 4106
const CL_DEVICE_EXTENSIONS: cl_uint = 0x1030; // 4144

const CL_DEVICE_TYPE: cl_uint = 0x1000;
const CL_DEVICE_TYPE_GPU: cl_bitfield = 1 << 2;
const CL_DEVICE_TYPE_ALL: cl_bitfield = 0xFFFF_FFFF;
const CL_MEM_READ_WRITE: cl_bitfield = 1;
//const CL_MEM_READ_ONLY: cl_bitfield = 4;
const CL_BLOCKING: cl_uint = 1;
const CL_PROGRAM_BUILD_LOG: cl_uint = 0x1183; // 4483
const CL_EVENT_COMMAND_EXECUTION_STATUS: cl_uint = 0x1283; // 4763
const CL_COMPLETE: cl_int = 0x0;

#[allow(clippy::upper_case_acronyms)]
#[derive(Copy, Clone, PartialEq, Debug, Eq)]
#[repr(C)]
enum OpenCLStatus {
    CL_DEVICE_NOT_FOUND = -1, // 0xFFFF_FFFF
    CL_SUCCESS = 0,
    CL_MEM_OBJECT_ALLOCATION_FAILURE = -4,
    CL_OUT_OF_RESOURCES = -5,
    CL_OUT_OF_HOST_MEMORY = -6,
    CL_IMAGE_FORMAT_NOT_SUPPORTED = -10,
    CL_MISALIGNED_SUB_BUFFER_OFFSET = -13,
    CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST = -14,
    CL_INVALID_VALUE = -30,
    CL_INVALID_DEVICE_QUEUE = -33,
    CL_INVALID_CONTEXT = -34,
    CL_INVALID_COMMAND_QUEUE = -36,
    CL_INVALID_MEM_OBJECT = -38,
    CL_INVALID_IMAGE_SIZE = -40,
    CL_INVALID_SAMPLER = -41,
    CL_INVALID_PROGRAM = -44,
    CL_INVALID_PROGRAM_EXECUTABLE = -45,
    CL_INVALID_KERNEL_NAME = -46,
    CL_INVALID_KERNEL_DEFINITION = -47,
    CL_INVALID_KERNEL = -48,
    CL_INVALID_ARG_INDEX = -49,
    CL_INVALID_ARG_VALUE = -50,
    CL_INVALID_ARG_SIZE = -51,
    CL_INVALID_KERNEL_ARGS = -52,
    CL_INVALID_WORK_DIMENSION = -53,
    CL_INVALID_WORK_GROUP_SIZE = -54,
    CL_INVALID_WORK_ITEM_SIZE = -55,
    CL_INVALID_GLOBAL_OFFSET = -56,
    CL_INVALID_EVENT_WAIT_LIST = -57,
    CL_INVALID_EVENT = -58,
    CL_INVALID_OPERATION = -59,
    CL_INVALID_BUFFER_SIZE = -61,
    CL_INVALID_GLOBAL_WORK_SIZE = -63,
    CL_INVALID_PROPERTY = -64,
    CL_MAX_SIZE_RESTRICTION_EXCEEDED = -72,
    UNKNOWN,
}

impl From<cl_int> for OpenCLStatus {
    fn from(status: cl_int) -> Self {
        match status {
            -4 => Self::CL_MEM_OBJECT_ALLOCATION_FAILURE,
            -5 => Self::CL_OUT_OF_RESOURCES,
            -6 => Self::CL_OUT_OF_HOST_MEMORY,
            -10 => Self::CL_IMAGE_FORMAT_NOT_SUPPORTED,
            -13 => Self::CL_MISALIGNED_SUB_BUFFER_OFFSET,
            -14 => Self::CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST,
            -30 => Self::CL_INVALID_VALUE,
            -33 => Self::CL_INVALID_DEVICE_QUEUE,
            -34 => Self::CL_INVALID_CONTEXT,
            -36 => Self::CL_INVALID_COMMAND_QUEUE,
            -38 => Self::CL_INVALID_MEM_OBJECT,
            -40 => Self::CL_INVALID_IMAGE_SIZE,
            -41 => Self::CL_INVALID_SAMPLER,
            -44 => Self::CL_INVALID_PROGRAM,
            -45 => Self::CL_INVALID_PROGRAM_EXECUTABLE,
            -46 => Self::CL_INVALID_KERNEL_NAME,
            -47 => Self::CL_INVALID_KERNEL_DEFINITION,
            -48 => Self::CL_INVALID_KERNEL,
            -49 => Self::CL_INVALID_ARG_INDEX,
            -50 => Self::CL_INVALID_ARG_VALUE,
            -51 => Self::CL_INVALID_ARG_SIZE,
            -52 => Self::CL_INVALID_KERNEL_ARGS,
            -53 => Self::CL_INVALID_WORK_DIMENSION,
            -54 => Self::CL_INVALID_WORK_GROUP_SIZE,
            -55 => Self::CL_INVALID_WORK_ITEM_SIZE,
            -56 => Self::CL_INVALID_GLOBAL_OFFSET,
            -57 => Self::CL_INVALID_EVENT_WAIT_LIST,
            -58 => Self::CL_INVALID_EVENT,
            -59 => Self::CL_INVALID_OPERATION,
            -61 => Self::CL_INVALID_BUFFER_SIZE,
            -63 => Self::CL_INVALID_GLOBAL_WORK_SIZE,
            -64 => Self::CL_INVALID_PROPERTY,
            -72 => Self::CL_MAX_SIZE_RESTRICTION_EXCEEDED,
            _ => Self::UNKNOWN,
        }
    }
}
