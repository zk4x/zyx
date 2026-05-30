// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! `OpenCL` backend

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::question_mark)]
#![allow(clippy::needless_pass_by_ref_mut)]
#![allow(clippy::unused_self)]

const VEC_COMPONENTS: [&str; 16] = [
    "x", "y", "z", "w", "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "sa", "sb",
];

use super::{Device, DeviceId, DeviceInfo, DeviceProgramId, Event, MemoryPool, PoolBufferId, PoolId};
use crate::{
    DType, Map,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::{BOp, Kernel, Op, OpId, Scope, UOp},
    shape::Dim,
    slab::Slab,
};
use libloading::Library;
use nanoserde::DeJson;
use std::{
    ffi::{CString, c_void},
    fmt::Write,
    hash::BuildHasherDefault,
    ptr,
    sync::mpsc::{Receiver, Sender, channel},
    sync::Arc,
    thread,
};

#[derive(Debug, Default, DeJson)]
#[nserde(default)]
pub struct OpenCLConfig {
    /// Select which platforms will be used by `OpenCL` backend
    /// If set to None, uses all available platforms.
    /// default = None
    pub platform_ids: Option<Vec<usize>>,
}

// OpenCL does not have the concept of memory pools,
// so we simply say it is all in one memory pool
#[derive(Debug)]
pub struct OpenCLMemoryPool {
    tx: Sender<Command>,
    #[allow(unused)]
    total_bytes: Dim,
    free_bytes: Dim,
}

#[derive(Debug)]
pub(super) struct OpenCLBuffer {
    buffer: *mut c_void,
    bytes: Dim,
}

#[derive(Debug)]
pub struct OpenCLDevice {
    tx: Sender<Command>,
    dev_info: DeviceInfo,
    memory_pool_id: PoolId,
    device_idx: usize,
}

#[derive(Debug)]
pub(super) struct OpenCLProgram {
    program: *mut c_void,
    kernel: *mut c_void,
    gws: Vec<Dim>,
    lws: Vec<Dim>,
}

#[derive(Debug)]
pub(super) struct OpenCLQueue {
    queue: *mut c_void, // points to device queue
    load: usize,
}

#[derive(Debug, Clone)]
pub struct OpenCLEvent {
    pub event: *mut c_void,
}

enum Command {
    Allocate {
        bytes: Dim,
        reply: Sender<Result<(PoolBufferId, OpenCLEvent), BackendError>>,
    },
    Deallocate {
        buffer_id: PoolBufferId,
        events: Vec<OpenCLEvent>,
    },
    HostToPool {
        src: Vec<u8>,
        dst: PoolBufferId,
        event_wait_list: Vec<OpenCLEvent>,
        reply: Sender<Result<OpenCLEvent, BackendError>>,
    },
    PoolToHost {
        src: PoolBufferId,
        len: usize,
        event_wait_list: Vec<OpenCLEvent>,
        reply: Sender<Result<Vec<u8>, BackendError>>,
    },
    SyncEvents {
        events: Vec<OpenCLEvent>,
        reply: Sender<Result<(), BackendError>>,
    },
    ReleaseEvents {
        events: Vec<OpenCLEvent>,
    },
    Compile {
        name: Box<str>,
        source: String,
        gws: Vec<Dim>,
        lws: Vec<Dim>,
        reply: Sender<Result<DeviceProgramId, BackendError>>,
    },
    Launch {
        device_idx: usize,
        program_id: DeviceProgramId,
        args: Vec<PoolBufferId>,
        event_wait_list: Vec<OpenCLEvent>,
        reply: Sender<Result<OpenCLEvent, BackendError>>,
    },
    ReleaseProgram {
        program_id: DeviceProgramId,
    },
}

unsafe impl Send for Command {}
// This definitely isn't correct, but for now...
unsafe impl Send for OpenCLMemoryPool {}
unsafe impl Send for OpenCLBuffer {}
unsafe impl Send for OpenCLDevice {}
unsafe impl Send for OpenCLProgram {}
unsafe impl Send for OpenCLQueue {}
unsafe impl Send for OpenCLEvent {}

pub(super) fn initialize_device(
    config: &OpenCLConfig,
    memory_pools: &mut Slab<PoolId, MemoryPool>,
    devices: &mut Slab<DeviceId, Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if let Some(device_ids) = &config.platform_ids
        && device_ids.is_empty()
    {
        if debug_dev {
            println!("OpenCL: configured out");
        }
        return Ok(());
    }

    // Search for opencl dynamic library path, kinda primitive, but fast and mostly works
    let mut opencl_paths = Vec::new();
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
        return Err(BackendError { status: ErrorStatus::DyLibNotFound, context: "OpenCL runtime not found.".into() });
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
    let clWaitForEvents: unsafe extern "C" fn(cl_uint, *const *mut c_void) -> OpenCLStatus =
        *unsafe { opencl.get(b"clWaitForEvents\0") }?;
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
    let clReleaseProgram: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus =
        *unsafe { opencl.get(b"clReleaseProgram\0") }?;
    let _clReleaseContext: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus =
        *unsafe { opencl.get(b"clReleaseContext\0") }?;
    //let clReleaseEvent = *unsafe { opencl.get(b"clReleaseContext\0") }?;
    let clSetKernelArg: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *const c_void) -> OpenCLStatus =
        *unsafe { opencl.get(b"clSetKernelArg\0") }?;
    let clCreateKernel: unsafe extern "C" fn(*mut c_void, *const i8, *mut OpenCLStatus) -> *mut c_void =
        *unsafe { opencl.get(b"clCreateKernel\0") }?;
    let clReleaseMemObject: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus =
        *unsafe { opencl.get(b"clReleaseMemObject\0") }?;
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
    let clFinish: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus =
        *unsafe { opencl.get(b"clFinish\0") }?;
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
    let mut memory_pool_id = PoolId::from(usize::from(memory_pools.len()));
    for (_platform_id, platform) in platform_ids
        .iter()
        .enumerate()
        .filter(|(id, _)| config.platform_ids.as_ref().is_none_or(|ids| ids.contains(id)))
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
            println!("OpenCL: {} on devices:", String::from_utf8(platform_name).unwrap());
        }
        if device_ids.is_empty() {
            continue;
        }

        // Query device info on the main thread (clGetDeviceInfo doesn't need a context)
        let mut total_bytes = 0;
        let mut dev_infos: Vec<(usize, DeviceInfo)> = Vec::new();
        for (orig_idx, dev) in device_ids.iter().copied().enumerate() {
            let mut dev_info = DeviceInfo::default();
            let Ok(()) = query_device_info(dev, clGetDeviceInfo, &mut dev_info, debug_dev) else {
                continue;
            };
            if let Ok(bytes) = get_device_data(dev, clGetDeviceInfo, CL_DEVICE_GLOBAL_MEM_SIZE) {
                total_bytes += Dim::from_ne_bytes(bytes.try_into().unwrap());
            }
            dev_infos.push((orig_idx, dev_info));
        }
        if dev_infos.is_empty() {
            continue;
        }

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
        unsafe { pthread_sigmask(SIG_BLOCK, sigset.as_ptr().cast(), ptr::null_mut()) };

        let (tx, rx): (Sender<Command>, Receiver<Command>) = channel();

        // Cast to usize for Send safety through the closure
        let worker_device_ids: Vec<usize> = device_ids.iter().map(|&d| d as usize).collect();
        let worker_library = library.clone();
        thread::spawn(move || {
            let _worker_library = worker_library;
            let devices: Vec<*mut c_void> = worker_device_ids.iter().map(|&d| d as *mut c_void).collect();
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

            let mut queues: Vec<Vec<OpenCLQueue>> = Vec::with_capacity(devices.len());
            for &dev in &devices {
                let mut dev_queues = Vec::new();
                // TODO get max queues per device and limit this to that number
                for _ in 0..2 {
                    let mut qstatus = OpenCLStatus::CL_SUCCESS;
                    let new_queue = unsafe { clCreateCommandQueue(context, dev, 0, &raw mut qstatus) };
                    if qstatus.check(ErrorStatus::Initialization).is_ok() {
                        dev_queues.push(OpenCLQueue { queue: new_queue, load: 0 });
                    }
                }
                queues.push(dev_queues);
            }
            let data_queue = queues.first()
                .and_then(|q| q.first())
                .map(|q| q.queue)
                .expect("no OpenCL command queue created");

            let mut buffers: Slab<PoolBufferId, OpenCLBuffer> = Slab::new();
            let mut programs: Slab<DeviceProgramId, OpenCLProgram> = Slab::new();
            let mut free_bytes: Dim = total_bytes;

            while let Ok(cmd) = rx.recv() {
                match cmd {
                    Command::Allocate { bytes, reply } => {
                        if bytes > free_bytes {
                            let _ = reply.send(Err(BackendError {
                                status: ErrorStatus::MemoryAllocation,
                                context: "Allocation failure".into(),
                            }));
                            continue;
                        }
                        let mut status = OpenCLStatus::CL_SUCCESS;
                        let buffer = unsafe {
                            clCreateBuffer(context, CL_MEM_READ_WRITE, bytes as usize, ptr::null_mut(), &raw mut status)
                        };
                        if let Err(e) = status.check(ErrorStatus::MemoryAllocation) {
                            let _ = reply.send(Err(e));
                            continue;
                        }
                        free_bytes = free_bytes.saturating_sub(bytes);
                        let id = buffers.push(OpenCLBuffer { buffer, bytes });
                        let _ = reply.send(Ok((id, OpenCLEvent { event: ptr::null_mut() })));
                    }
                    Command::Deallocate { buffer_id, events } => {
                        let buffer = &buffers[buffer_id];
                        debug_assert!(!buffer.buffer.is_null(), "Deallocating null buffer is invalid");
                        let event_wait_list: Vec<*mut c_void> = events
                            .into_iter()
                            .map(|e| e.event)
                            .filter(|event| !event.is_null())
                            .collect();
                        if !event_wait_list.is_empty() {
                            let _ = unsafe { clWaitForEvents(event_wait_list.len().try_into().unwrap(), event_wait_list.as_ptr()) }
                                .check(ErrorStatus::Deinitialization);
                        }
                        let _ = unsafe { clReleaseMemObject(buffer.buffer) }.check(ErrorStatus::Deinitialization);
                        free_bytes += buffer.bytes;
                        buffers.remove(buffer_id);
                    }
                    Command::HostToPool { src, dst, event_wait_list, reply } => {
                        let dst = &buffers[dst];
                        debug_assert!(src.len() as u64 <= dst.bytes);
                        let event_wait_list: Vec<*mut c_void> = event_wait_list
                            .into_iter()
                            .map(|e| e.event)
                            .filter(|event| !event.is_null())
                            .collect();
                        let event_wait_list_ptr = if event_wait_list.is_empty() {
                            ptr::null()
                        } else {
                            event_wait_list.as_ptr()
                        };
                        let mut event = ptr::null_mut();
                        let status = unsafe {
                            clEnqueueWriteBuffer(
                                data_queue,
                                dst.buffer,
                                CL_NON_BLOCKING,
                                0,
                                src.len(),
                                src.as_ptr().cast(),
                                event_wait_list.len().try_into().expect("So many events..."),
                                event_wait_list_ptr,
                                &raw mut event,
                            )
                        };
                        if let Err(e) = status.check(ErrorStatus::MemoryCopyH2P) {
                            let _ = reply.send(Err(e));
                            continue;
                        }
                        let _ = reply.send(Ok(OpenCLEvent { event }));
                    }
                    Command::PoolToHost { src, len, event_wait_list, reply } => {
                        let src = &buffers[src];
                        debug_assert!(!src.buffer.is_null(), "Trying to read null memory. Internal bug.");
                        let mut event_wait_list: Vec<*mut c_void> = event_wait_list
                            .into_iter()
                            .map(|e| e.event)
                            .filter(|event| !event.is_null())
                            .collect();
                        if !event_wait_list.is_empty() {
                            let _ = unsafe {
                                clWaitForEvents(
                                    u32::try_from(event_wait_list.len()).expect("So many events..."),
                                    event_wait_list.as_ptr(),
                                )
                            }
                            .check(ErrorStatus::MemoryCopyP2H);
                        }
                        let mut dst = vec![0u8; len];
                        let mut event: *mut c_void = ptr::null_mut();
                        let status = unsafe {
                            clEnqueueReadBuffer(
                                data_queue,
                                src.buffer,
                                CL_NON_BLOCKING,
                                0,
                                dst.len(),
                                dst.as_mut_ptr().cast(),
                                0,
                                ptr::null(),
                                &raw mut event,
                            )
                        };
                        if let Err(e) = status.check(ErrorStatus::MemoryCopyP2H) {
                            let _ = reply.send(Err(e));
                            continue;
                        }
                        let events = [event];
                        let _ = unsafe { clWaitForEvents(1, events.as_ptr()) }.check(ErrorStatus::MemoryCopyP2H);
                        event_wait_list.push(event);
                        let _ = reply.send(Ok(dst));
                    }
                    Command::SyncEvents { events, reply } => {
                        let events: Vec<*mut c_void> = events
                            .into_iter()
                            .map(|e| e.event)
                            .filter(|event| !event.is_null())
                            .collect();
                        let result = if !events.is_empty() {
                            unsafe { clWaitForEvents(events.len().try_into().expect("So many events..."), events.as_ptr()) }
                                .check(ErrorStatus::KernelSync)
                        } else {
                            Ok(())
                        };
                        let _ = reply.send(result);
                    }
                    Command::ReleaseEvents { events: _ } => {}
                    Command::Compile { name, source, gws, lws, reply } => {
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
                            continue;
                        }
                        if let Err(e) = unsafe {
                            clBuildProgram(
                                program,
                                cl_uint::try_from(devices.len()).expect("So many devices..."),
                                devices.as_ptr(),
                                c"-cl-fast-relaxed-math".as_ptr().cast(),
                                None,
                                ptr::null_mut(),
                            )
                        }
                        .check(ErrorStatus::KernelCompilation)
                        {
                            // Try to get build log from first device
                            let build_log = get_program_build_data(program, devices[0], clGetProgramBuildInfo, CL_PROGRAM_BUILD_LOG);
                            match build_log {
                                Ok(build_log) => {
                                    panic!("{e:?} {}", String::from_utf8_lossy(&build_log));
                                }
                                Err(status) => {
                                    let _ = reply.send(Err(status.check(ErrorStatus::KernelCompilation).err().unwrap()));
                                    continue;
                                }
                            }
                        }
                        let mut status = OpenCLStatus::CL_SUCCESS;
                        let program_name = &CString::new(name.as_ref()).unwrap();
                        let kernel = unsafe { clCreateKernel(program, program_name.as_ptr().cast(), &raw mut status) };
                        if let Err(e) = status.check(ErrorStatus::KernelCompilation) {
                            let _ = reply.send(Err(e));
                            continue;
                        }
                        let program_id = programs.push(OpenCLProgram { program, kernel, gws, lws });
                        let _ = reply.send(Ok(program_id));
                    }
                    Command::Launch { device_idx, program_id, args, event_wait_list, reply } => {
                        // Sync events
                        let events: Vec<*mut c_void> = event_wait_list
                            .into_iter()
                            .map(|e| e.event)
                            .filter(|event| !event.is_null())
                            .collect();
                        if !events.is_empty() {
                            let _ = unsafe {
                                clWaitForEvents(events.len().try_into().expect("So many events..."), events.as_ptr())
                            }
                            .check(ErrorStatus::KernelSync);
                        }

                        let queue_id = next_queue(&mut queues[device_idx], clFinish);
                        let program = &programs[program_id];
                        let mut i = 0;
                        for &arg in &args {
                            let arg = &buffers[arg];
                            let ptr: *const _ = &raw const arg.buffer;
                            if let Err(e) = unsafe {
                                clSetKernelArg(program.kernel, i, core::mem::size_of::<*mut c_void>(), ptr.cast())
                            }
                            .check(ErrorStatus::IncorrectKernelArg) {
                                let _ = reply.send(Err(e));
                                continue;
                            }
                            i += 1;
                        }
                        let mut event: *mut c_void = ptr::null_mut();
                        let lws_ptr = if program.lws.is_empty() {
                            ptr::null()
                        } else {
                            program.lws.as_ptr().cast()
                        };
                        if let Err(e) = unsafe {
                            clEnqueueNDRangeKernel(
                                queues[device_idx][queue_id].queue,
                                program.kernel,
                                u32::try_from(program.gws.len()).expect("So many programs..."),
                                ptr::null(),
                                program.gws.as_ptr().cast(),
                                lws_ptr,
                                0,
                                ptr::null(),
                                &raw mut event,
                            )
                        }
                        .check(ErrorStatus::KernelLaunch) {
                            let _ = reply.send(Err(e));
                            continue;
                        }
                        queues[device_idx][queue_id].load += 1;
                        let _ = unsafe { clFinish(queues[device_idx][queue_id].queue) }.check(ErrorStatus::KernelLaunch);
                        let _ = reply.send(Ok(OpenCLEvent { event }));
                    }
                    Command::ReleaseProgram { program_id } => {
                        let _ = unsafe { clReleaseProgram(programs[program_id].program) }.check(ErrorStatus::Deinitialization);
                        programs.remove(program_id);
                    }
                }
            }
        });

        memory_pools.push(MemoryPool::OpenCL(OpenCLMemoryPool {
            tx: tx.clone(),
            total_bytes,
            free_bytes: total_bytes,
        }));
        for (orig_idx, dev_info) in dev_infos.into_iter() {
            devices.push(Device::OpenCL(OpenCLDevice {
                tx: tx.clone(),
                dev_info,
                memory_pool_id,
                device_idx: orig_idx,
            }));
        }
        memory_pool_id += 1;
    }
    #[allow(unused)]
    let _ = library;
    Ok(())
}

impl OpenCLMemoryPool {
    pub fn deinitialize(&mut self) {}

    pub const fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
        let (reply, reply_rx) = channel();
        self.tx.send(Command::Allocate { bytes, reply }).unwrap();
        reply_rx.recv().unwrap().map(|(id, e)| (id, Event::OpenCL(e)))
    }

    pub fn deallocate(&mut self, buffer_id: PoolBufferId, event_wait_list: Vec<Event>) {
        let events = event_wait_list
            .into_iter()
            .map(|e| {
                let Event::OpenCL(e) = e else { unreachable!() };
                e
            })
            .collect();
        self.tx.send(Command::Deallocate { buffer_id, events }).unwrap();
    }

    pub fn host_to_pool(&mut self, src: &[u8], dst: PoolBufferId, event_wait_list: Vec<Event>) -> Result<Event, BackendError> {
        let events = event_wait_list
            .into_iter()
            .map(|e| {
                let Event::OpenCL(e) = e else { unreachable!() };
                e
            })
            .collect();
        let (reply, reply_rx) = channel();
        self.tx
            .send(Command::HostToPool { src: src.to_vec(), dst, event_wait_list: events, reply })
            .unwrap();
        reply_rx.recv().unwrap().map(Event::OpenCL)
    }

    pub fn pool_to_host(&mut self, src: PoolBufferId, dst: &mut [u8], event_wait_list: Vec<Event>) -> Result<(), BackendError> {
        let events = event_wait_list
            .into_iter()
            .map(|e| {
                let Event::OpenCL(e) = e else { unreachable!() };
                e
            })
            .collect();
        let len = dst.len();
        let (reply, reply_rx) = channel();
        self.tx
            .send(Command::PoolToHost { src, len, event_wait_list: events, reply })
            .unwrap();
        let data = reply_rx.recv().unwrap()?;
        dst.copy_from_slice(&data);
        Ok(())
    }

    pub fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        let events = events
            .into_iter()
            .map(|e| {
                let Event::OpenCL(e) = e else { unreachable!() };
                e
            })
            .collect();
        let (reply, reply_rx) = channel();
        self.tx.send(Command::SyncEvents { events, reply }).unwrap();
        reply_rx.recv().unwrap()
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn release_events(&mut self, events: Vec<Event>) {
        let events = events
            .into_iter()
            .map(|e| {
                let Event::OpenCL(e) = e else { unreachable!() };
                e
            })
            .collect();
        self.tx.send(Command::ReleaseEvents { events }).unwrap();
    }
}

impl OpenCLDevice {
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub const fn deinitialize(&mut self) {
        let _ = self;
    }

    pub const fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    pub const fn memory_pool_id(&self) -> PoolId {
        self.memory_pool_id
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        fn new_reg(
            op_id: OpId,
            reg_map: &mut Map<OpId, usize>,
            registers: &mut Vec<((DType, u16), u32, u8)>,
            dtype: (DType, u16),
            rc: u32,
            current_loop_level: u8,
        ) -> usize {
            for (i, (dt, nrc, loop_level)) in registers.iter_mut().enumerate() {
                if *nrc == 0 && *dt == dtype && current_loop_level <= *loop_level {
                    reg_map.insert(op_id, i);
                    *nrc = rc;
                    *loop_level = current_loop_level;
                    return i;
                }
            }
            let i = registers.len();
            registers.push((dtype, rc, current_loop_level));
            reg_map.insert(op_id, i);
            i
        }

        fn get_var(
            op_id: OpId,
            constants: &Map<OpId, Constant>,
            indices: &Map<OpId, u8>,
            reg_map: &Map<OpId, usize>,
            registers: &mut [((DType, u16), u32, u8)],
            loop_level: u8,
        ) -> String {
            if let Some(c) = constants.get(&op_id) {
                c.ocl()
            } else if let Some(id) = indices.get(&op_id) {
                format!("idx{id}")
            } else if let Some(reg) = reg_map.get(&op_id) {
                if registers[*reg].2 == loop_level {
                    registers[*reg].1 -= 1;
                }
                format!("r{reg}")
            } else {
                unreachable!()
            }
        }

        let mut gws = vec![1; 3];
        let mut lws = vec![1; 3];
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            let op = kernel.at(op_id);
            if let &Op::Index { len: dim, scope, axis } = op {
                match scope {
                    Scope::Global => {
                        gws[axis as usize] = dim;
                    }
                    Scope::Local => {
                        lws[axis as usize] = dim;
                    }
                    Scope::Register => {}
                }
            }
            op_id = kernel.next_op(op_id);
        }

        if lws.iter().product::<u64>() > self.dev_info.max_local_threads {
            return Err(BackendError { status: ErrorStatus::KernelCompilation, context: "Invalid local work size.".into() });
        }

        let mut global_args = String::new();
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            let op = kernel.at(op_id);
            if let &Op::Define { dtype, scope, ro, .. } = op {
                if scope == Scope::Global {
                    _ = writeln!(
                        global_args,
                        "  __global {}{}* p{op_id},",
                        if ro { "const " } else { "" },
                        dtype.ocl()
                    );
                }
            } else {
                break;
            }
            op_id = kernel.next_op(op_id);
        }
        global_args.pop();
        global_args.pop();
        global_args.push('\n');

        let mut rcs: Map<OpId, u32> = Map::with_capacity_and_hasher(kernel.ops.len().into(), BuildHasherDefault::new());
        let mut dtypes: Map<OpId, (DType, u16)> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());

        // first we will calculate those reference counts.
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            let op = kernel.at(op_id);
            match op {
                Op::ConstView { .. } | Op::StoreView { .. } | Op::LoadView { .. } | Op::Reduce { .. } | Op::Move { .. } => {
                    unreachable!()
                }
                Op::Const(x) => {
                    dtypes.insert(op_id, (x.dtype(), 1));
                }
                &Op::Define { dtype, .. } => {
                    dtypes.insert(op_id, (dtype, 1));
                }
                &Op::Load { src, index, vlen: len } => {
                    dtypes.insert(op_id, (dtypes[&src].0, len as u16));
                    *rcs.entry(index).or_insert(0) += 1;
                }
                &Op::Store { dst, x: src, index, vlen } => {
                    debug_assert_eq!(dtypes[&src].1, vlen);
                    dtypes.insert(op_id, dtypes[&src]);
                    *rcs.entry(dst).or_insert(0) += 1;
                    *rcs.entry(src).or_insert(0) += 1;
                    *rcs.entry(index).or_insert(0) += 1;
                }
                &Op::Cast { x, dtype } => {
                    dtypes.insert(op_id, (dtype, dtypes[&x].1));
                    *rcs.entry(x).or_insert(0) += 1;
                }
                &Op::Unary { x, .. } => {
                    dtypes.insert(op_id, dtypes[&x]);
                    *rcs.entry(x).or_insert(0) += 1;
                }
                &Op::Binary { x, y, bop } => {
                    let dtype = if bop.returns_bool() {
                        (DType::Bool, dtypes[&x].1)
                    } else {
                        dtypes[&x]
                    };
                    dtypes.insert(op_id, dtype);
                    *rcs.entry(x).or_insert(0) += 1;
                    *rcs.entry(y).or_insert(0) += 1;
                }
                Op::Vectorize { ops } => {
                    let dtype = dtypes[&ops[0]];
                    dtypes.insert(op_id, (dtype.0, ops.len() as u16));
                    for &x in ops {
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                }
                &Op::Devectorize { .. } | Op::Wmma { .. } => todo!(),
                &Op::Mad { x, y, z } => {
                    dtypes.insert(op_id, dtypes[&x]);
                    *rcs.entry(x).or_insert(0) += 1;
                    *rcs.entry(y).or_insert(0) += 1;
                    *rcs.entry(z).or_insert(0) += 1;
                }
                Op::Index { .. } | Op::Loop { .. } => {
                    dtypes.insert(op_id, (DType::U32, 1));
                }
                &Op::If { condition } => {
                    *rcs.entry(condition).or_insert(0) += 1;
                }
                Op::Barrier { .. } | Op::EndIf | Op::EndLoop => {}
            }
            op_id = kernel.next_op(op_id);
        }

        let mut reg_map: Map<OpId, usize> = Map::with_capacity_and_hasher(kernel.ops.len().into(), BuildHasherDefault::new());
        let mut registers: Vec<((DType, u16), u32, u8)> = Vec::new();

        let mut constants: Map<OpId, Constant> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
        let mut indices: Map<OpId, u8> = Map::with_capacity_and_hasher(20, BuildHasherDefault::new());

        let mut loop_id = 0;
        let mut indent = String::from("  ");
        let mut source = String::with_capacity(1000);

        let mut acc_bytes: u64 = 0;
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            let op = kernel.at(op_id);
            match op {
                Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } | Op::Reduce { .. } | Op::Move { .. } => {
                    unreachable!()
                }
                &Op::Const(x) => {
                    constants.insert(op_id, x);
                }
                &Op::Define { dtype, scope, ro, len } => {
                    if scope == Scope::Register {
                        _ = writeln!(
                            source,
                            "{indent}{}{} p{op_id}[{len}] __attribute__ ((aligned));",
                            if ro { "const " } else { "" },
                            dtype.ocl(),
                        );
                        acc_bytes += u64::from(dtype.bit_size() / 8) * len;
                    } else if scope == Scope::Local {
                        _ = writeln!(
                            source,
                            "{indent}__local {}{} p{op_id}[{len}] __attribute__ ((aligned));",
                            if ro { "const " } else { "" },
                            dtype.ocl(),
                        );
                    }
                }
                &Op::Load { src, index, vlen } => {
                    if let Some(&rc) = rcs.get(&op_id) {
                        let dtype = dtypes[&op_id];
                        debug_assert_eq!(dtype.1, vlen);
                        let idx = get_var(index, &constants, &indices, &reg_map, &mut registers, loop_id);
                        let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rc, loop_id);
                        if vlen > 1 {
                            for i in 0..vlen {
                                _ = writeln!(source, "{indent}r{reg}.{} = p{src}[{idx}];", VEC_COMPONENTS[i as usize]);
                            }
                        } else {
                            _ = writeln!(source, "{indent}r{reg} = p{src}[{idx}];");
                        }
                    }
                }
                &Op::Store { dst, x: src, index, vlen } => {
                    let idx = get_var(index, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let x = get_var(src, &constants, &indices, &reg_map, &mut registers, loop_id);
                    if vlen > 1 {
                        for i in 0..vlen {
                            _ = writeln!(source, "{indent}p{dst}[{idx} + {i}] = {x}.{};", VEC_COMPONENTS[i as usize]);
                        }
                    } else {
                        _ = writeln!(source, "{indent}p{dst}[{idx}] = {x};");
                    }
                }
                &Op::Cast { x, dtype } => {
                    let vlen = dtypes[&x].1;
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, (dtype, vlen), rcs[&op_id], loop_id);
                    _ = writeln!(source, "{indent}r{reg} = ({}){x};", dtype.ocl());
                }
                &Op::Unary { x, uop } => {
                    let dtype = dtypes[&x];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    match uop {
                        UOp::BitNot => _ = writeln!(source, "{indent}r{reg} = ~{x};"),
                        UOp::Neg => _ = writeln!(source, "{indent}r{reg} = -{x};"),
                        UOp::Exp => unreachable!(
                            "internal bug: UOp::Exp should be converted to Exp2 + mul by ln2(e) by IR pass before reaching OpenCL backend"
                        ),
                        UOp::Exp2 => {
                            if dtype.0 == DType::F16 {
                                _ = writeln!(source, "{indent}r{reg} = (half)exp2((float){x});");
                            } else {
                                _ = writeln!(source, "{indent}r{reg} = exp2({x});");
                            }
                        }
                        UOp::Log2 => _ = writeln!(source, "{indent}r{reg} = log2({x});"),
                        UOp::Reciprocal => {
                            _ = writeln!(source, "{indent}r{reg} = {}/{x};", dtype.0.one_constant().ocl());
                        }
                        UOp::Sqrt => _ = writeln!(source, "{indent}r{reg} = sqrt({x});"),
                        UOp::Sin => _ = writeln!(source, "{indent}r{reg} = sin({x});"),
                        UOp::Cos => _ = writeln!(source, "{indent}r{reg} = cos({x});"),
                        UOp::Floor => _ = writeln!(source, "{indent}r{reg} = floor({x});"),
                        UOp::Trunc => _ = writeln!(source, "{indent}r{reg} = trunc({x});"),
                        UOp::Ln => _ = writeln!(source, "{indent}r{reg} = log({x});"),
                        UOp::Abs => _ = writeln!(source, "{indent}r{reg} = fabs({x});"),
                    }
                }
                Op::Vectorize { ops } => {
                    let dtype = dtypes[&op_id];
                    let mut vars = String::new();
                    for &x in ops {
                        let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id);
                        _ = write!(vars, "{x}, ");
                    }
                    vars.pop();
                    vars.pop();
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    let dtype = dtypes[&op_id];
                    _ = writeln!(source, "{indent}r{reg} = ({}{})({vars});", dtype.0.ocl(), dtype.1);
                }
                &Op::Devectorize { .. } => todo!(),
                Op::Wmma { .. } => todo!(),
                &Op::Binary { x, y, bop } => {
                    let dtype = dtypes[&op_id];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let y = get_var(y, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    _ = match bop {
                        BOp::Add => writeln!(source, "{indent}r{reg} = {x} + {y};"),
                        BOp::Sub => writeln!(source, "{indent}r{reg} = {x} - {y};"),
                        BOp::Mul => writeln!(source, "{indent}r{reg} = {x} * {y};"),
                        BOp::Div => writeln!(source, "{indent}r{reg} = {x} / {y};"),
                        BOp::Pow => writeln!(source, "{indent}r{reg} = pow((double){x}, (double){y});"),
                        BOp::Mod => writeln!(source, "{indent}r{reg} = {x} % {y};"),
                        BOp::Cmplt => writeln!(source, "{indent}r{reg} = {x} < {y};"),
                        BOp::Cmpgt => writeln!(source, "{indent}r{reg} = {x} > {y};"),
                        BOp::Max => writeln!(source, "{indent}r{reg} = max({x}, {y});"),
                        BOp::Or => writeln!(source, "{indent}r{reg} = {x} || {y};"),
                        BOp::And => writeln!(source, "{indent}r{reg} = {x} && {y};"),
                        BOp::BitXor => writeln!(source, "{indent}r{reg} = {x} ^ {y};"),
                        BOp::BitOr => writeln!(source, "{indent}r{reg} = {x} | {y};"),
                        BOp::BitAnd => writeln!(source, "{indent}r{reg} = {x} & {y};"),
                        BOp::BitShiftLeft => writeln!(source, "{indent}r{reg} = {x} << {y};"),
                        BOp::BitShiftRight => writeln!(source, "{indent}r{reg} = {x} >> {y};"),
                        BOp::NotEq => writeln!(source, "{indent}r{reg} = {x} != {y};"),
                        BOp::Eq => writeln!(source, "{indent}r{reg} = {x} == {y};"),
                    };
                }
                &Op::Mad { x, y, z } => {
                    let dtype = dtypes[&op_id];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let y = get_var(y, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let z = get_var(z, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    _ = writeln!(source, "{indent}r{reg} = {x} * {y} + {z};");
                }
                &Op::Index { len: dim, scope, axis } => {
                    indices.insert(op_id, loop_id);
                    match scope {
                        Scope::Global => {
                            _ = writeln!(
                                source,
                                "{indent}unsigned int idx{loop_id} = get_group_id({axis}); // 0..={}",
                                dim - 1
                            );
                        }
                        Scope::Local => {
                            _ = writeln!(
                                source,
                                "{indent}unsigned int idx{loop_id} = get_local_id({axis}); // 0..={}",
                                dim - 1
                            );
                        }
                        Scope::Register => {}
                    }
                    loop_id += 1;
                }
                &Op::Loop { len, .. } => {
                    indices.insert(op_id, loop_id);
                    _ = writeln!(
                        source,
                        "{indent}for (unsigned int idx{loop_id} = 0; idx{loop_id} < {len}; ++idx{loop_id}) {{"
                    );
                    indent += "  ";
                    loop_id += 1;
                }
                Op::EndLoop => {
                    indent.pop();
                    indent.pop();
                    _ = writeln!(source, "{indent}}}");
                    loop_id -= 1;
                }
                &Op::If { condition } => {
                    let condition = get_var(condition, &constants, &indices, &reg_map, &mut registers, loop_id);
                    _ = writeln!(source, "{indent}if ({condition}) {{");
                    indent += "  ";
                }
                Op::EndIf => {
                    indent.pop();
                    indent.pop();
                    _ = writeln!(source, "{indent}}}");
                }
                Op::Barrier { scope } => match scope {
                    Scope::Global => _ = writeln!(source, "{indent}barrier(CLK_GLOBAL_MEM_FENCE);"),
                    Scope::Local => _ = writeln!(source, "{indent}barrier(CLK_LOCAL_MEM_FENCE);"),
                    Scope::Register => unreachable!(),
                },
            }
            op_id = kernel.next_op(op_id);
        }
        let mut reg_str = String::new();
        if !registers.is_empty() {
            let (dt, _, _) = registers.remove(0);
            let mut prev_dt = dt;
            _ = write!(
                reg_str,
                "{indent}{}{} r0",
                dt.0.ocl(),
                if dt.1 == 1 { String::new() } else { format!("{}", dt.1) }
            );
            let mut i = 1;
            for (dt, _, _) in registers {
                if dt == prev_dt {
                    _ = write!(reg_str, ", r{i}");
                } else {
                    _ = write!(
                        reg_str,
                        ";\n{indent}{}{} r{i}",
                        dt.0.ocl(),
                        if dt.1 == 1 { String::new() } else { format!("{}", dt.1) }
                    );
                }
                prev_dt = dt;
                i += 1;
            }
            _ = writeln!(reg_str, ";");
        }

        let mut pragma = String::new();
        if dtypes.values().any(|&x| x.0 == DType::F16) {
            pragma += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        }
        if dtypes.values().any(|&x| x.0 == DType::F64) {
            pragma += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        }

        let name = format!(
            "k_{}__{}",
            gws.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"),
            lws.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"),
        );

        let source = format!("{pragma}__kernel void {name}(\n{global_args}) {{\n{reg_str}{source}}}\n");
        if debug_asm {
            println!();
            println!("{source}");
        }

        for (i, lwd) in lws.iter().enumerate() {
            gws[i] *= lwd;
        }

        let (reply, reply_rx) = channel();
        self.tx
            .send(Command::Compile {
                name: name.into(),
                source,
                gws,
                lws,
                reply,
            })
            .unwrap();
        reply_rx.recv().unwrap()
    }

    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        _memory_pool: &mut OpenCLMemoryPool,
        args: &[PoolBufferId],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let events = event_wait_list
            .into_iter()
            .map(|e| {
                let Event::OpenCL(e) = e else { unreachable!() };
                e
            })
            .collect();
        let (reply, reply_rx) = channel();
        self.tx
            .send(Command::Launch {
                device_idx: self.device_idx,
                program_id,
                args: args.to_vec(),
                event_wait_list: events,
                reply,
            })
            .unwrap();
        reply_rx.recv().unwrap().map(Event::OpenCL)
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        self.tx.send(Command::ReleaseProgram { program_id }).unwrap();
    }

    pub const fn free_compute(&self) -> u128 {
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
        println!("OpenCL:   {device_name}");
    }
    let max_work_item_dims = u32::from_ne_bytes(max_work_item_dims.try_into().unwrap()) as usize;
    let mwis = get_device_data(device, clGetDeviceInfo, CL_DEVICE_MAX_WORK_ITEM_SIZES)?;
    let mut max_local_work_dims = vec![0; max_work_item_dims];
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
        max_local_work_dims[i] = max_dim_size as Dim;
    }
    let mlt = 256;
    *dev_info = DeviceInfo {
        compute: 1024 * 1024 * 1024,
        max_global_work_dims: vec![100_000; max_work_item_dims],
        max_local_threads: mlt,
        max_local_work_dims,
        preferred_vector_size: u8::try_from(u32::from_ne_bytes(
            get_device_data(device, clGetDeviceInfo, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)?
                .try_into()
                .unwrap(),
        ))
        .expect("What a vector width...")
            * 4,
        local_mem_size: Dim::try_from(usize::from_ne_bytes(
            get_device_data(device, clGetDeviceInfo, CL_DEVICE_LOCAL_MEM_SIZE)?.try_into().unwrap(),
        ))
        .expect("What a memory size..."),
        max_register_bytes: 256,
        has_native_exp2: true,
        tensor_cores: false,
        warp_size: {
            if let Ok(device_type_data) = get_device_data(device, clGetDeviceInfo, CL_DEVICE_TYPE) {
                let device_type = u64::from_ne_bytes(device_type_data.try_into().unwrap_or_default());
                if device_type & CL_DEVICE_TYPE_GPU != 0 {
                    64
                } else {
                    1
                }
            } else {
                1
            }
        },
        supported_dtypes: {
            let mut mask = 0u32;
            let extensions = get_device_data(device, clGetDeviceInfo, CL_DEVICE_EXTENSIONS)
                .ok()
                .map(|d| String::from_utf8_lossy(&d).into_owned())
                .unwrap_or_default();

            if extensions.contains("cl_khr_fp64") {
                mask |= 1u32 << (DType::F64 as u32);
            }
            if extensions.contains("cl_khr_fp16") {
                mask |= 1u32 << (DType::F16 as u32);
            }

            mask
        },
    };
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
    clGetProgramBuildInfo: unsafe extern "C" fn(*mut c_void, *mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus,
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

fn next_queue(
    queues: &mut [OpenCLQueue],
    clFinish: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
) -> usize {
    let mut id = queues.iter().enumerate().min_by_key(|(_, q)| q.load).unwrap().0;
    if queues[id].load > 20 {
        if unsafe { clFinish(queues[id].queue) }.check(ErrorStatus::KernelSync).is_ok() {
            queues[id].load = 0;
        }
        id = queues.iter().enumerate().min_by_key(|(_, q)| q.load).unwrap().0;
    }
    id
}

impl DType {
    fn ocl(self) -> &'static str {
        match self {
            Self::BF16 => todo!("bf16 should be casted to f16 or f32"),
            Self::F16 => "half",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::U8 => "uchar",
            Self::U16 => "ushort",
            Self::I8 => "char",
            Self::I16 => "short",
            Self::I32 => "int",
            Self::I64 => "long",
            Self::Bool => "bool",
            Self::U32 => "uint",
            Self::U64 => "ulong",
        }
    }
}

impl Constant {
    fn ocl(self) -> String {
        match self {
            Self::BF16(x) => format!("{:.16}f", half::bf16::from_le_bytes(x)),
            Self::F16(x) => format!("(half){:.16}", half::f16::from_le_bytes(x)),
            Self::F32(x) => format!("{:.16}f", f32::from_le_bytes(x)),
            Self::F64(x) => format!("(double){:.16}", f64::from_le_bytes(x)),
            Self::U8(x) => format!("{x}"),
            Self::I8(x) => format!("{x}"),
            Self::I16(x) => format!("{x}"),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}"),
            Self::U64(x) => format!("{}", u64::from_le_bytes(x)),
            Self::I32(x) => format!("(int){x}"),
            Self::I64(x) => format!("(long){}", i64::from_le_bytes(x)),
            Self::Bool(x) => format!("{x}"),
        }
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
const CL_DEVICE_EXTENSIONS: cl_uint = 0x1029; // 4137
const CL_DEVICE_TYPE: cl_uint = 0x1000;
const CL_DEVICE_TYPE_GPU: cl_bitfield = 1 << 2;
const CL_DEVICE_TYPE_ALL: cl_bitfield = 0xFFFF_FFFF;
const CL_MEM_READ_WRITE: cl_bitfield = 1;
//const CL_MEM_READ_ONLY: cl_bitfield = 4;
const CL_NON_BLOCKING: cl_uint = 0;
const CL_PROGRAM_BUILD_LOG: cl_uint = 0x1183; // 4483

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


