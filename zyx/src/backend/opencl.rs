// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! `OpenCL` backend

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::question_mark)]
#![allow(clippy::needless_pass_by_ref_mut)]
#![allow(clippy::unused_self)]

use super::{DTypeCapability, DeviceInfo, DeviceProgramId, GwsDim, LaunchArg, ParamKind, Pool, PoolBufferId, gws_from_kernel};
use crate::{
    DType,
    error::{BackendError, ErrorStatus},
    kernel::{Kernel, Op, RangeKind},
    shape::Dim,
    slab::Slab,
};
use crate::{Map, hashers::FHasher};
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

#[derive(Debug, Default, DeJson)]
#[nserde(default)]
pub struct OpenCLConfig {
    /// Select which platforms will be used by `OpenCL` backend
    /// If set to None, uses all available platforms.
    /// default = None
    pub platform_ids: Option<Vec<usize>>,
    /// Number of in-order command queues per device that the micro-batch
    /// window is distributed over.
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
    rc: u16,
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

/// Pending commands accumulate until the micro-batch window is flushed.
const MICRO_BATCH_WINDOW: usize = 100;

enum Command {
    Allocate {
        bytes: Dim,
        reply: Sender<Result<PoolBufferId, BackendError>>,
    },
    Retain {
        buffer_id: PoolBufferId,
    },
    Release {
        buffer_id: PoolBufferId,
    },
    /// Blocking read-back: the reply is sent after the data arrived in
    /// host memory. This is a sync point — all pending work is submitted
    /// and every queue is drained first.
    PoolToHost {
        src: PoolBufferId,
        dst: *mut u8,
        bytes: Dim,
        reply: Sender<Result<(), BackendError>>,
    },
    /// Async copy into this pool's buffer, fire-and-forget: appended to the
    /// pending micro-batch window, submitted (with computed waits) when the
    /// window flushes. `OpenCLMemoryPool::pool_to_pool` retained the source
    /// pool buffer; once the copy completes, this worker releases it back to
    /// the source pool (foreign sweep).
    Copy {
        src_pool: Pool,
        src_buf: PoolBufferId,
        src_ptr: *const u8,
        bytes: Dim,
        dst_buf: PoolBufferId,
    },
    Compile {
        name: Box<str>,
        source: String,
        lws: Vec<Dim>,
        gws: Vec<GwsDim>,
        params: Vec<ParamKind>,
        reply: Sender<Result<DeviceProgramId, BackendError>>,
    },
    /// Fire-and-forget kernel launch: appended to the pending micro-batch
    /// window; submitted (with computed waits) when the window flushes.
    Launch {
        program_id: DeviceProgramId,
        args: Vec<LaunchArg>,
    },
    /// Timed launch for autotune: flush + drain first (uncontended timing),
    /// then launch solo on one queue, reply nanos.
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

/// Process-wide per-device pools. Owned here — `mod.rs` only holds
/// `Pool::OpenCL(i)` handles. `OPENCL_INIT` serializes first construction
/// only; the alloc/free path never takes it. Each pool owns one device:
/// one context and one worker thread per device.
static OPENCL_POOLS: OnceLock<Vec<Arc<Mutex<OpenCLMemoryPool>>>> = OnceLock::new();
static OPENCL_INIT: Mutex<()> = Mutex::new(());

fn pools_with(config: &OpenCLConfig, debug_dev: bool) -> Result<&'static Vec<Arc<Mutex<OpenCLMemoryPool>>>, BackendError> {
    if let Some(pools) = OPENCL_POOLS.get() {
        return Ok(pools);
    }
    let _init = OPENCL_INIT.lock().unwrap_or_else(|_| panic!("opencl pool init lock poisoned"));
    if let Some(pools) = OPENCL_POOLS.get() {
        return Ok(pools);
    }
    let pools = ensure_pool_table(config, debug_dev)?;
    let _ = OPENCL_POOLS.set(pools);
    OPENCL_POOLS
        .get()
        .ok_or_else(|| BackendError { status: ErrorStatus::Initialization, context: "OpenCL pool init failed".into() })
}

fn pools() -> Result<&'static Vec<Arc<Mutex<OpenCLMemoryPool>>>, BackendError> {
    pools_with(&OpenCLConfig::default(), false)
}

pub(super) fn pool(id: u16) -> Result<Arc<Mutex<OpenCLMemoryPool>>, BackendError> {
    pools()?.get(id as usize).cloned().ok_or_else(|| no_pool(id))
}

pub(super) fn pool_count() -> u16 {
    pools().map(|pools| pools.len() as u16).unwrap_or(0)
}

fn no_pool(id: u16) -> BackendError {
    BackendError { status: ErrorStatus::Initialization, context: format!("Pool::OpenCL({id}) is not available").into() }
}

/// Builds the per-device pool table: loads the ICD, enumerates platforms and
/// devices, and spawns one worker (one context) per device. Runs once under
/// `OPENCL_INIT`.
pub(super) fn ensure_pool_table(
    config: &OpenCLConfig,
    debug_dev: bool,
) -> Result<Vec<Arc<Mutex<OpenCLMemoryPool>>>, BackendError> {
    let mut pools: Vec<Arc<Mutex<OpenCLMemoryPool>>> = Vec::new();
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

                    let mut buffers: Slab<PoolBufferId, OpenCLBuffer> = Slab::new();
                    let mut programs: Slab<DeviceProgramId, OpenCLProgram> = Slab::new();

                    // Pending micro-batch window: launches and copies
                    // accumulate here in program order until
                    // MICRO_BATCH_WINDOW is reached (or a sync point
                    // arrives), then `flush_window` distributes them over
                    // the in-order queues at once.
                    let mut pending: Vec<Pending> = Vec::new();
                    // Last queue that WROTE each buffer (RAW dependencies,
                    // with the tail event of that write) and last queue that
                    // used it at all (WAR dependencies). These persist across
                    // windows: a consumer in window N+1 must still wait for a
                    // producer from window N if it lands on a different queue.
                    let mut writer: Map<PoolBufferId, (usize, *mut c_void)> = Map::with_hasher(BuildHasherDefault::<FHasher>::new());
                    let mut last_use: Map<PoolBufferId, (usize, *mut c_void)> = Map::with_hasher(BuildHasherDefault::<FHasher>::new());
                    // Retained foreign source buffers of in-flight copies:
                    // (source pool, source buffer, completion event). Once the
                    // event completes, the source buffer is released back to
                    // its own pool (sweep_foreign).
                    let mut foreign_dead: Vec<(Pool, PoolBufferId, *mut c_void)> = Vec::new();
                    // First async submission error since the last sync point
                    // (a failed fire-and-forget launch surfaces here, at the
                    // next PoolToHost / LaunchTimed).
                    let mut last_error: Option<BackendError> = None;

                    // Unblock SIGABRT so it can be delivered (absorbed by the no-op handler)
                    const SIG_UNBLOCK: i32 = 1;
                    unsafe {
                        let mut mask = std::mem::MaybeUninit::<[u8; 128]>::uninit();
                        sigemptyset(mask.as_mut_ptr().cast());
                        sigaddset(mask.as_mut_ptr().cast(), SIGABRT);
                        pthread_sigmask(SIG_UNBLOCK, mask.as_ptr().cast(), ptr::null_mut());
                    }

                    'work_thread_loop: while let Ok(cmd) = rx.recv() {
                        // Poll deferred foreign releases: reap any source
                        // buffer whose copy event completed. Cheap and
                        // usually a no-op.
                        sweep_foreign(&mut foreign_dead, clGetEventInfo, clReleaseEvent);
                        match cmd {
                            Command::Allocate { bytes, reply } => {
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
                                let id = buffers.push(OpenCLBuffer { ptr: buffer, bytes, rc: 1 });
                                let _ = reply.send(Ok(id));
                            }
                            Command::Retain { buffer_id } => match buffers.get_mut(buffer_id) {
                                Some(buffer) => buffer.rc = buffer.rc.checked_add(1).expect("OpenCLBuffer rc overflow"),
                                None => debug_assert!(false, "retain of unknown OpenCL buffer {buffer_id:?}"),
                            },
                            Command::Release { buffer_id } => {
                                let Some(buffer) = buffers.get_mut(buffer_id) else {
                                    debug_assert!(false, "release of unknown OpenCL buffer {buffer_id:?}");
                                    continue;
                                };
                                buffer.rc = buffer.rc.checked_sub(1).expect("OpenCLBuffer rc underflow");
                                if buffer.rc > 0 {
                                    continue;
                                }
                                // rc hit zero: the buffer's last use may still
                                // be queued. Flush the pending window, then
                                // drain every in-order queue — completion of
                                // all queues implies completion of every
                                // queued use of this buffer. The worker
                                // drains; callers never block.
                                if let Err(err) = flush_window(
                                    &mut pending,
                                    &queues,
                                    &buffers,
                                    &programs,
                                    &mut writer,
                                    &mut last_use,
                                    &mut foreign_dead,
                                    debug_dev,
                                    clEnqueueNDRangeKernel,
                                    clEnqueueWriteBuffer,
                                    clSetKernelArg,
                                    clReleaseEvent,
                                ) && last_error.is_none()
                                {
                                    last_error = Some(err);
                                }
                                for q in &queues {
                                    let _ = unsafe { (clFinish)(q.queue) }.check(ErrorStatus::MemoryDeallocation);
                                }
                                sweep_foreign(&mut foreign_dead, clGetEventInfo, clReleaseEvent);
                                let mut retired = Vec::new();
                                if let Some((_, old)) = writer.remove(&buffer_id) {
                                    retired.push(old);
                                }
                                if let Some((_, old)) = last_use.remove(&buffer_id) {
                                    retired.push(old);
                                }
                                release_distinct(retired, clReleaseEvent);
                                let OpenCLBuffer { ptr, bytes, .. } = buffers[buffer_id];
                                debug_assert!(!ptr.is_null(), "deallocating null buffer is invalid");
                                let _ = unsafe { clReleaseMemObject(ptr) }.check(ErrorStatus::MemoryDeallocation);
                                free_bytes_atomic.fetch_add(bytes as u64, Ordering::SeqCst);
                                buffers.remove(buffer_id);
                            }
                            Command::Copy { src_pool, src_buf, src_ptr, bytes, dst_buf } => {
                                // Fire-and-forget: append to the micro-batch
                                // window; the batched-submission algorithm
                                // submits it (with computed waits) when the
                                // window flushes.
                                pending.push(Pending::Copy { src_pool, src_buf, src_ptr, bytes, dst: dst_buf });
                                if pending.len() >= MICRO_BATCH_WINDOW
                                    && let Err(err) = flush_window(
                                        &mut pending,
                                        &queues,
                                        &buffers,
                                        &programs,
                                        &mut writer,
                                        &mut last_use,
                                        &mut foreign_dead,
                                        debug_dev,
                                        clEnqueueNDRangeKernel,
                                        clEnqueueWriteBuffer,
                                        clSetKernelArg,
                                        clReleaseEvent,
                                    )
                                    && last_error.is_none()
                                {
                                    last_error = Some(err);
                                }
                            }
                            Command::PoolToHost { src, dst, bytes, reply } => {
                                // Sync point: submit everything pending, drain
                                // all queues, surface async submission errors,
                                // then read back.
                                if let Err(err) = flush_window(
                                    &mut pending,
                                    &queues,
                                    &buffers,
                                    &programs,
                                    &mut writer,
                                    &mut last_use,
                                    &mut foreign_dead,
                                    debug_dev,
                                    clEnqueueNDRangeKernel,
                                    clEnqueueWriteBuffer,
                                    clSetKernelArg,
                                    clReleaseEvent,
                                ) && last_error.is_none()
                                {
                                    last_error = Some(err);
                                }
                                for q in &queues {
                                    if let Err(err) = unsafe { (clFinish)(q.queue) }.check(ErrorStatus::MemoryCopyP2H) {
                                        let _ = reply.send(Err(err));
                                        continue 'work_thread_loop;
                                    }
                                }
                                sweep_foreign(&mut foreign_dead, clGetEventInfo, clReleaseEvent);
                                if let Some(err) = last_error.take() {
                                    let _ = reply.send(Err(err));
                                    continue;
                                }
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
                            Command::Launch { program_id, args } => {
                                // Fire and forget: append to the micro-batch
                                // window; the batched-submission algorithm
                                // assigns it a queue (with computed waits)
                                // when the window flushes.
                                pending.push(Pending::Launch { program_id, args });
                                if pending.len() >= MICRO_BATCH_WINDOW
                                    && let Err(err) = flush_window(
                                        &mut pending,
                                        &queues,
                                        &buffers,
                                        &programs,
                                        &mut writer,
                                        &mut last_use,
                                        &mut foreign_dead,
                                        debug_dev,
                                        clEnqueueNDRangeKernel,
                                        clEnqueueWriteBuffer,
                                        clSetKernelArg,
                                        clReleaseEvent,
                                    )
                                    && last_error.is_none()
                                {
                                    last_error = Some(err);
                                }
                            }
                            Command::LaunchTimed { program_id, args, reply } => {
                                // Uncontended timing for autotune: submit the
                                // pending window, drain every queue, surface
                                // async errors, then run the kernel solo.
                                if let Err(err) = flush_window(
                                    &mut pending,
                                    &queues,
                                    &buffers,
                                    &programs,
                                    &mut writer,
                                    &mut last_use,
                                    &mut foreign_dead,
                                    debug_dev,
                                    clEnqueueNDRangeKernel,
                                    clEnqueueWriteBuffer,
                                    clSetKernelArg,
                                    clReleaseEvent,
                                ) && last_error.is_none()
                                {
                                    last_error = Some(err);
                                }
                                for q in &queues {
                                    if let Err(err) = unsafe { (clFinish)(q.queue) }.check(ErrorStatus::KernelSync) {
                                        let _ = reply.send(Err(err));
                                        continue 'work_thread_loop;
                                    }
                                }
                                sweep_foreign(&mut foreign_dead, clGetEventInfo, clReleaseEvent);
                                if let Some(err) = last_error.take() {
                                    let _ = reply.send(Err(err));
                                    continue;
                                }
                                let start = Instant::now();
                                let result = submit_launch(
                                    &programs,
                                    &buffers,
                                    program_id,
                                    &args,
                                    queues[0].queue,
                                    &[],
                                    clEnqueueNDRangeKernel,
                                    clSetKernelArg,
                                )
                                .and_then(|event| {
                                    let r = unsafe { (clFinish)(queues[0].queue) }.check(ErrorStatus::KernelSync);
                                    if !event.is_null() {
                                        let _ = unsafe { (clReleaseEvent)(event) };
                                    }
                                    r
                                });
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

            pools.push(Arc::new(Mutex::new(OpenCLMemoryPool {
                tx: tx.clone(),
                total_bytes,
                free_bytes: free_bytes_atomic,
                dev_info,
            })));
        }
    }
    #[allow(unused)]
    let _ = library;
    Ok(pools)
}

/// Process-wide per-device OpenCL devices. Owned here — `mod.rs` only holds
/// `Dev::OpenCL(i)` handles. `OPENCL_DEV_INIT` serializes first construction
/// only; compile/launch take the device lock, never the init lock.
static OPENCL_DEVICES: OnceLock<Vec<Arc<Mutex<OpenCLDevice>>>> = OnceLock::new();
static OPENCL_DEV_INIT: Mutex<()> = Mutex::new(());

fn devices_with(config: &OpenCLConfig, debug_dev: bool) -> Result<&'static Vec<Arc<Mutex<OpenCLDevice>>>, BackendError> {
    if let Some(devs) = OPENCL_DEVICES.get() {
        return Ok(devs);
    }
    let _init = OPENCL_DEV_INIT.lock().unwrap_or_else(|_| panic!("opencl device init lock poisoned"));
    if let Some(devs) = OPENCL_DEVICES.get() {
        return Ok(devs);
    }
    let devs = ensure_device_table(config, debug_dev)?;
    let _ = OPENCL_DEVICES.set(devs);
    OPENCL_DEVICES
        .get()
        .ok_or_else(|| BackendError { status: ErrorStatus::Initialization, context: "OpenCL device init failed".into() })
}

fn devices() -> Result<&'static Vec<Arc<Mutex<OpenCLDevice>>>, BackendError> {
    devices_with(&super::config().opencl, super::debug_backends())
}

pub(super) fn device(id: u16) -> Result<Arc<Mutex<OpenCLDevice>>, BackendError> {
    devices()?.get(id as usize).cloned().ok_or_else(|| BackendError {
        status: ErrorStatus::Initialization,
        context: format!("Dev::OpenCL({id}) is not available").into(),
    })
}

pub(super) fn device_count() -> u16 {
    devices().map(|devs| devs.len() as u16).unwrap_or(0)
}

fn ensure_device_table(config: &OpenCLConfig, debug_dev: bool) -> Result<Vec<Arc<Mutex<OpenCLDevice>>>, BackendError> {
    let pools = pools_with(config, debug_dev)?;
    let mut devs = Vec::with_capacity(pools.len());
    for (idx, pool_arc) in pools.iter().enumerate() {
        let pool_id = Pool::OpenCL(u16::try_from(idx).expect("So many OpenCL devices..."));
        let guard = super::lock(pool_id, pool_arc);
        let tx = guard.tx.clone();
        let dev_info = guard.dev_info.clone();
        drop(guard);
        devs.push(Arc::new(Mutex::new(OpenCLDevice { tx, dev_info: Arc::new(dev_info), memory_pool: pool_id })));
    }
    Ok(devs)
}

enum Pending {
    Launch {
        program_id: DeviceProgramId,
        args: Vec<LaunchArg>,
    },
    Copy {
        src_pool: Pool,
        src_buf: PoolBufferId,
        src_ptr: *const u8,
        bytes: Dim,
        dst: PoolBufferId,
    },
}

/// Reads/writes of a pending command. Launch args are in `Param` head order;
/// programs carry their kinds from compile.
fn pending_reads_writes(programs: &Slab<DeviceProgramId, OpenCLProgram>, cmd: &Pending) -> (Vec<PoolBufferId>, Vec<PoolBufferId>) {
    match cmd {
        Pending::Launch { program_id, args } => {
            let kinds: &[ParamKind] = &programs[*program_id].params;
            debug_assert!(args.len() <= kinds.len(), "more launch args than program params");
            let mut reads = Vec::new();
            let mut writes = Vec::new();
            for (idx, arg) in args.iter().enumerate() {
                match arg {
                    LaunchArg::Buffer(id) => match kinds.get(idx).copied().unwrap_or(ParamKind::Global) {
                        ParamKind::GlobalMut => writes.push(*id),
                        ParamKind::Global | ParamKind::Variable => reads.push(*id),
                    },
                    LaunchArg::Variable(_) => {}
                }
            }
            (reads, writes)
        }
        Pending::Copy { dst, .. } => (Vec::new(), vec![*dst]),
    }
}

/// Resolves launch args against the buffer table and enqueues one kernel onto
/// `queue`, waiting on `wait_events`. Returns the completion event. Shared by
/// the batched-submission path and `LaunchTimed`.
#[allow(clippy::too_many_arguments)]
fn submit_launch(
    programs: &Slab<DeviceProgramId, OpenCLProgram>,
    buffers: &Slab<PoolBufferId, OpenCLBuffer>,
    program_id: DeviceProgramId,
    args: &[LaunchArg],
    queue: *mut c_void,
    wait_events: &[*mut c_void],
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
) -> Result<*mut c_void, BackendError> {
    debug_assert!(programs.contains_id(program_id), "launch of unknown program {program_id:?}");
    let program = &programs[program_id];
    // clSetKernelArg copies the value immediately, so stable storage for the
    // scalar bytes within this loop iteration is enough.
    let mut scalar_values: Vec<Box<[u8]>> = Vec::new();
    let mut i: u32 = 0;
    for arg in args {
        let (arg_ptr, arg_size): (*const c_void, usize) = match arg {
            LaunchArg::Buffer(buffer_id) => {
                let ptr = buffers[*buffer_id].ptr;
                let value_ptr: *const _ = &raw const ptr;
                (value_ptr.cast(), core::mem::size_of::<*mut c_void>())
            }
            LaunchArg::Variable(constant) => {
                scalar_values.push(constant.to_le_bytes().into());
                let value = scalar_values.last().unwrap();
                (value.as_ptr().cast(), value.len())
            }
        };
        unsafe { (clSetKernelArg)(program.kernel, i, arg_size, arg_ptr) }.check(ErrorStatus::IncorrectKernelArg)?;
        i += 1;
    }
    let global_size: Vec<Dim> = program
        .gws
        .iter()
        .zip(program.lws.iter())
        .map(|(gdim, l)| {
            let g = gdim.eval(&mut |ordinal| match &args[ordinal] {
                LaunchArg::Variable(c) => c.as_dim().unwrap(),
                LaunchArg::Buffer(_) => unreachable!("gws param must be a Variable launch arg"),
            });
            g * *l
        })
        .collect();
    let lws_ptr = if program.lws.is_empty() { ptr::null() } else { program.lws.as_ptr().cast() };
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
    let mut event: *mut c_void = ptr::null_mut();
    unsafe {
        (clEnqueueNDRangeKernel)(
            queue,
            program.kernel,
            u32::try_from(global_size.len()).unwrap_or(3),
            ptr::null(),
            global_size.as_ptr().cast(),
            lws_ptr,
            u32::try_from(wait_events.len()).unwrap_or(0),
            if wait_events.is_empty() { ptr::null() } else { wait_events.as_ptr() },
            &raw mut event,
        )
    }
    .check(ErrorStatus::KernelLaunch)?;
    debug_assert!(!event.is_null(), "kernel enqueue returned no event");
    Ok(event)
}

/// Submits the pending micro-batch window. Commands are assigned to in-order
/// queues greedily: prefer the queue that last wrote the first input buffer
/// (locality — the RAW wait disappears), else the least-loaded queue.
/// Cross-queue dependencies become event wait lists: RAW via `writer` (last
/// queue that wrote the buffer, with its tail event), WAR via `last_use`.
/// Commands are submitted in program order; the tracking maps persist across
/// windows, so dependencies spanning two windows are handled too. Copy
/// commands additionally record a completion event: the retained foreign
/// source buffer is released when it fires (`sweep_foreign`).
#[allow(clippy::too_many_arguments)]
fn flush_window(
    pending: &mut Vec<Pending>,
    queues: &[OpenCLQueue],
    buffers: &Slab<PoolBufferId, OpenCLBuffer>,
    programs: &Slab<DeviceProgramId, OpenCLProgram>,
    writer: &mut Map<PoolBufferId, (usize, *mut c_void)>,
    last_use: &mut Map<PoolBufferId, (usize, *mut c_void)>,
    foreign_dead: &mut Vec<(Pool, PoolBufferId, *mut c_void)>,
    debug_dev: bool,
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
    clEnqueueWriteBuffer: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        usize,
        usize,
        *const c_void,
        cl_uint,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> OpenCLStatus,
    clSetKernelArg: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *const c_void) -> OpenCLStatus,
    clReleaseEvent: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
) -> Result<(), BackendError> {
    let n_queues = queues.len();
    let mut load = vec![0usize; n_queues];
    let mut first_err: Option<BackendError> = None;
    for cmd in pending.drain(..) {
        let (reads, writes) = pending_reads_writes(programs, &cmd);
        let chosen = reads
            .first()
            .and_then(|b| writer.get(b).map(|&(q, _)| q))
            .filter(|&q| q < n_queues)
            .unwrap_or_else(|| (0..n_queues).min_by_key(|&i| load[i]).unwrap());
        // Cross-queue waits: one event per needed edge, passed as the
        // enqueue's event wait list (the enqueued command retains them).
        let mut waits: Vec<*mut c_void> = Vec::new();
        for b in &reads {
            if let Some(&(w, event)) = writer.get(b)
                && w != chosen
                && !event.is_null()
                && !waits.contains(&event)
            {
                waits.push(event);
            }
        }
        for b in &writes {
            if let Some(&(u, event)) = last_use.get(b)
                && u != chosen
                && !event.is_null()
                && !waits.contains(&event)
            {
                waits.push(event);
            }
        }
        let result: Result<*mut c_void, BackendError> = match cmd {
            Pending::Launch { program_id, args } => {
                submit_launch(programs, buffers, program_id, &args, queues[chosen].queue, &waits, clEnqueueNDRangeKernel, clSetKernelArg)
            }
            Pending::Copy { src_pool, src_buf, src_ptr, bytes, dst } => {
                let mut event: *mut c_void = ptr::null_mut();
                let status = unsafe {
                    (clEnqueueWriteBuffer)(
                        queues[chosen].queue,
                        buffers[dst].ptr,
                        CL_NON_BLOCKING,
                        0,
                        bytes as usize,
                        src_ptr.cast(),
                        u32::try_from(waits.len()).unwrap_or(0),
                        if waits.is_empty() { ptr::null() } else { waits.as_ptr() },
                        &raw mut event,
                    )
                }
                .check(ErrorStatus::MemoryCopyH2P);
                match status {
                    Ok(()) => {
                        // Release the retained source buffer once this
                        // completion event fires (sweep_foreign).
                        foreign_dead.push((src_pool, src_buf, event));
                        Ok(event)
                    }
                    Err(err) => {
                        // The copy was never enqueued: balance the retain now.
                        src_pool.release(src_buf);
                        Err(err)
                    }
                }
            }
        };
        match result {
            Ok(event) => {
                debug_assert!(!event.is_null(), "enqueue returned no event");
                let mut retired: Vec<*mut c_void> = Vec::new();
                for b in &reads {
                    if let Some((_, old)) = last_use.insert(*b, (chosen, event)) {
                        retired.push(old);
                    }
                }
                for b in &writes {
                    if let Some((_, old)) = writer.insert(*b, (chosen, event)) {
                        retired.push(old);
                    }
                    if let Some((_, old)) = last_use.insert(*b, (chosen, event)) {
                        retired.push(old);
                    }
                }
                release_distinct(retired, clReleaseEvent);
            }
            Err(err) => {
                if debug_dev {
                    println!("[opencl] batched submission error: {err:?}");
                }
                first_err.get_or_insert(err);
            }
        }
        load[chosen] += 1;
    }
    match first_err {
        Some(err) => Err(err),
        None => Ok(()),
    }
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

/// Releases retained foreign source buffers whose copy event has completed.
/// Called every loop iteration (a cheap `clGetEventInfo` poll) and after
/// every full drain, where completion is guaranteed.
fn sweep_foreign(
    foreign_dead: &mut Vec<(Pool, PoolBufferId, *mut c_void)>,
    clGetEventInfo: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus,
    clReleaseEvent: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
) {
    foreign_dead.retain(|&(src_pool, src_buf, event)| {
        if !event.is_null() {
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
            if !queried || status != CL_COMPLETE {
                return true;
            }
            let _ = unsafe { (clReleaseEvent)(event) }.check(ErrorStatus::Deinitialization);
        }
        src_pool.release(src_buf);
        false
    });
}

impl OpenCLMemoryPool {
    pub fn free_bytes(&self) -> Dim {
        self.free_bytes.load(Ordering::SeqCst) as i64
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<PoolBufferId, BackendError> {
        let (reply, reply_rx) = channel();
        self.tx.send(Command::Allocate { bytes, reply }).unwrap();
        reply_rx.recv().unwrap()
    }

    /// Increment the buffer's reference count (allocate starts it at 1).
    /// Checked math: overflow panics.
    pub fn retain(&mut self, buffer_id: PoolBufferId) {
        self.tx.send(Command::Retain { buffer_id }).unwrap();
    }

    /// Decrement the buffer's reference count. At zero the worker flushes the
    /// pending window and drains every in-order queue before freeing — the
    /// buffer is never freed behind in-flight work (the worker drains;
    /// callers never block).
    pub fn release(&mut self, buffer_id: PoolBufferId) {
        self.tx.send(Command::Release { buffer_id }).unwrap();
    }

    /// Blocking read-back (sync point).
    pub fn pool_to_host(&mut self, src: PoolBufferId, dst: &mut [u8]) -> Result<(), BackendError> {
        let (reply, reply_rx) = channel();
        self.tx.send(Command::PoolToHost { src, dst: dst.as_mut_ptr(), bytes: dst.len() as Dim, reply }).unwrap();
        reply_rx.recv().unwrap()
    }

    /// Fire-and-forget copy into this pool (dst-owned). The source pool
    /// buffer is retained here and released by this device's worker once the
    /// copy completes — see `flush_window` / `sweep_foreign`.
    pub fn pool_to_pool(&mut self, src: Pool, src_buf: PoolBufferId, dst_buf: PoolBufferId) -> Result<(), BackendError> {
        // Retain the source buffer for the duration of the async copy; this
        // device's worker releases it back once the copy completes.
        src.retain(src_buf);
        match src {
            Pool::Host => {
                let src_pool = super::host::pool();
                let src_pool = super::lock(src, &src_pool);
                let bytes = src_pool.get_buffer(src_buf).len() as Dim;
                let src_ptr = src_pool.get_buffer(src_buf).as_ptr();
                drop(src_pool);
                self.tx.send(Command::Copy { src_pool: Pool::Host, src_buf, src_ptr, bytes, dst_buf }).unwrap();
                Ok(())
            }
            Pool::Disk => {
                // Stage: read the file slice into a temporary host-pool buffer
                // (rc 1) and copy from it like a host source; the worker
                // releases the temp buffer on completion. The disk buffer is
                // only read here, synchronously — its retain is balanced
                // immediately.
                let src_pool = super::disk::pool();
                let mut src_pool = super::lock(src, &src_pool);
                let mut byte_slice = vec![0u8; src_pool.buffer_bytes(src_buf) as usize];
                let staged = src_pool.pool_to_host(src_buf, &mut byte_slice);
                drop(src_pool);
                match staged {
                    Ok(()) => {
                        src.release(src_buf);
                        let tmp = Pool::Host.insert_host(byte_slice.into_boxed_slice());
                        let host_pool = super::host::pool();
                        let host_pool = super::lock(Pool::Host, &host_pool);
                        let bytes = host_pool.get_buffer(tmp).len() as Dim;
                        let src_ptr = host_pool.get_buffer(tmp).as_ptr();
                        drop(host_pool);
                        self.tx.send(Command::Copy { src_pool: Pool::Host, src_buf: tmp, src_ptr, bytes, dst_buf }).unwrap();
                        Ok(())
                    }
                    Err(err) => {
                        src.release(src_buf);
                        Err(err)
                    }
                }
            }
            Pool::Cuda(_) => todo!("cross-pool copy from CUDA to OpenCL"),
            Pool::OpenCL(_) => todo!("cross-pool copy from OpenCL to OpenCL"),
            Pool::Vulkan(_) | Pool::Dummy => todo!("cross-pool copy from {src:?} to OpenCL"),
            #[cfg(feature = "tenstorrent")]
            Pool::TT(_) => todo!("cross-pool copy from TT to OpenCL"),
            #[cfg(feature = "wgpu")]
            Pool::WGPU(_) => todo!("cross-pool copy from WGPU to OpenCL"),
        }
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

    /// Fire-and-forget launch: the command is queued to the device's worker,
    /// which appends it to the micro-batch window and submits it (with
    /// computed waits) when the window flushes.
    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        pool_handle: Pool,
        args: &[LaunchArg],
    ) -> Result<(), BackendError> {
        debug_assert_eq!(pool_handle, self.memory_pool);
        self.tx.send(Command::Launch { program_id, args: args.to_vec() }).unwrap();
        Ok(())
    }

    /// Timed launch for autotune: the worker's pending window is submitted
    /// and every queue drained first, then the kernel runs solo and the
    /// wall-clock nanos are measured around enqueue+finish.
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
const CL_NON_BLOCKING: cl_uint = 0;
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
