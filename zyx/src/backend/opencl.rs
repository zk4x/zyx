//! `OpenCL` backend

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::question_mark)]

use super::{BufferId, Device, DeviceInfo, Event, MemoryPool, Pool, ProgramId};
use crate::{
    DType, Map,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    graph::{BOp, UOp},
    kernel::{Kernel, Op, OpId, Scope},
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
    sync::Arc,
};

#[derive(Debug, Default, DeJson)]
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
    // Just to close the connection
    #[allow(unused)]
    library: Arc<Library>,
    #[allow(unused)]
    total_bytes: Dim,
    free_bytes: Dim,
    context: *mut c_void,
    queue: *mut c_void,
    buffers: Slab<BufferId, OpenCLBuffer>,
    // Functions
    clWaitForEvents: unsafe extern "C" fn(cl_uint, *const *mut c_void) -> OpenCLStatus,
    clReleaseCommandQueue: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
    clReleaseContext: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
    clReleaseMemObject: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
    //clReleaseEvent: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
    clEnqueueReadBuffer: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        usize,
        usize,
        *mut c_void,
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
    clCreateBuffer:
        unsafe extern "C" fn(*mut c_void, cl_bitfield, usize, *mut c_void, *mut OpenCLStatus) -> *mut c_void,
}

#[derive(Debug)]
pub(super) struct OpenCLBuffer {
    buffer: *mut c_void,
    bytes: Dim,
}

#[derive(Debug)]
pub struct OpenCLDevice {
    ptr: *mut c_void,
    context: *mut c_void,
    dev_info: DeviceInfo,
    memory_pool_id: u32,
    programs: Slab<ProgramId, OpenCLProgram>,
    queues: Vec<OpenCLQueue>,
    // Functions
    clGetProgramBuildInfo:
        unsafe extern "C" fn(*mut c_void, *mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus,
    clBuildProgram: unsafe extern "C" fn(
        *mut c_void,
        cl_uint,
        *const *mut c_void,
        *const i8,
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>,
        *mut c_void,
    ) -> OpenCLStatus,
    clReleaseProgram: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
    clCreateKernel: unsafe extern "C" fn(*mut c_void, *const i8, *mut OpenCLStatus) -> *mut c_void,
    clGetDeviceInfo: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus,
    clSetKernelArg: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *const c_void) -> OpenCLStatus,
    clCreateProgramWithSource:
        unsafe extern "C" fn(*mut c_void, cl_uint, *const *const i8, *const usize, *mut OpenCLStatus) -> *mut c_void,
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
    //clWaitForEvents: unsafe extern "C" fn(cl_uint, *const *mut c_void) -> OpenCLStatus,
    clFinish: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
    /*clGetCommandQueueInfo: unsafe extern "C" fn(
        command_queue: *mut c_void,
        param_name: cl_uint,
        param_value_size: usize,
        param_value: *mut std::ffi::c_void,
        param_value_size_ret: *mut usize,
    ) -> OpenCLStatus,*/
    //clReleaseCommandQueue: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
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

// This definitely isn't correct, but for now...
unsafe impl Send for OpenCLMemoryPool {}
unsafe impl Send for OpenCLBuffer {}
unsafe impl Send for OpenCLDevice {}
unsafe impl Send for OpenCLProgram {}
unsafe impl Send for OpenCLQueue {}
unsafe impl Send for OpenCLEvent {}

pub(super) fn initialize_device(
    config: &OpenCLConfig,
    memory_pools: &mut Vec<Pool>,
    devices: &mut Vec<Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if let Some(device_ids) = &config.platform_ids
        && device_ids.is_empty()
    {
        if debug_dev {
            println!("OpenCL won't be used, as it was configured out");
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
                    let name = path.file_name().map(|x| x.to_str().unwrap()).unwrap_or("");
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
    let clCreateCommandQueue: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_bitfield,
        *mut OpenCLStatus,
    ) -> *mut c_void = *unsafe { opencl.get(b"clCreateCommandQueue\0") }?;
    let clGetDeviceIDs: unsafe extern "C" fn(
        *mut c_void,
        cl_bitfield,
        cl_uint,
        *mut *mut c_void,
        *mut cl_uint,
    ) -> OpenCLStatus = *unsafe { opencl.get(b"clGetDeviceIDs\0") }?;
    let clWaitForEvents = *unsafe { opencl.get(b"clWaitForEvents\0") }?;
    let clReleaseCommandQueue = *unsafe { opencl.get(b"clReleaseCommandQueue\0") }?;
    let clEnqueueNDRangeKernel = *unsafe { opencl.get(b"clEnqueueNDRangeKernel\0") }?;
    let clGetProgramBuildInfo = *unsafe { opencl.get(b"clGetProgramBuildInfo\0") }?;
    let clBuildProgram = *unsafe { opencl.get(b"clBuildProgram\0") }?;
    let clReleaseProgram = *unsafe { opencl.get(b"clReleaseProgram\0") }?;
    let clReleaseContext = *unsafe { opencl.get(b"clReleaseContext\0") }?;
    //let clReleaseEvent = *unsafe { opencl.get(b"clReleaseContext\0") }?;
    let clSetKernelArg = *unsafe { opencl.get(b"clSetKernelArg\0") }?;
    let clCreateKernel = *unsafe { opencl.get(b"clCreateKernel\0") }?;
    let clReleaseMemObject = *unsafe { opencl.get(b"clReleaseMemObject\0") }?;
    let clGetDeviceInfo = *unsafe { opencl.get(b"clGetDeviceInfo\0") }?;
    let clCreateProgramWithSource = *unsafe { opencl.get(b"clCreateProgramWithSource\0") }?;
    let clEnqueueReadBuffer = *unsafe { opencl.get(b"clEnqueueReadBuffer\0") }?;
    let clEnqueueWriteBuffer = *unsafe { opencl.get(b"clEnqueueWriteBuffer\0") }?;
    let clCreateBuffer = *unsafe { opencl.get(b"clCreateBuffer\0") }?;
    let clFinish = *unsafe { opencl.get(b"clFinish\0") }?;
    let clGetPlatformInfo: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus =
        *unsafe { opencl.get(b"clGetPlatformInfo\0") }?;
    //let clGetCommandQueueInfo = *unsafe { opencl.get(b"clGetCommandQueueInfo\0") }?;

    let library = Arc::new(opencl);
    let platform_ids = {
        // Get the number of platforms
        let mut count: cl_uint = 0;
        unsafe { clGetPlatformIDs(0, ptr::null_mut(), &raw mut count) }.check(ErrorStatus::DeviceEnumeration)?;
        if count > 0 {
            // Get the platform ids.
            let len = count as usize;
            let mut ids: Vec<*mut c_void> = Vec::with_capacity(len);
            unsafe { clGetPlatformIDs(count, ids.as_mut_ptr(), ptr::null_mut()) }
                .check(ErrorStatus::DeviceEnumeration)?;
            unsafe { ids.set_len(len) };
            ids
        } else {
            Vec::new()
        }
    };
    let mut memory_pool_id = u32::try_from(memory_pools.len()).expect("So many memory pools...");
    for (platform_id, platform) in platform_ids
        .iter()
        .enumerate()
        .filter(|(id, _)| config.platform_ids.as_ref().is_none_or(|ids| ids.contains(id)))
    {
        let platform = *platform;
        let Ok(device_ids) = {
            // Get the number of devices of device_type
            let mut count: cl_uint = 0;
            let mut status =
                unsafe { clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, ptr::null_mut(), &raw mut count) };
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
        let mut status = OpenCLStatus::CL_SUCCESS;
        let context = unsafe {
            clCreateContext(
                ptr::null(),
                cl_uint::try_from(device_ids.len()).expect("So many devices..."),
                device_ids.as_ptr(),
                None,
                ptr::null_mut(),
                &raw mut status,
            )
        };
        //println!("init context: {context:?}");
        let Ok(()) = status.check(ErrorStatus::Initialization) else {
            continue;
        };
        let mut total_bytes = 0;
        if debug_dev {
            let platform_name = {
                let mut size: usize = 0;
                let Ok(()) =
                    unsafe { clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, ptr::null_mut(), &raw mut size) }
                        .check(ErrorStatus::Initialization)
                else {
                    continue;
                };
                if size > 0 {
                    let count = size / core::mem::size_of::<u8>();
                    let mut data: Vec<u8> = Vec::with_capacity(count);
                    let Ok(()) = unsafe {
                        data.set_len(count);
                        clGetPlatformInfo(
                            platform,
                            CL_PLATFORM_NAME,
                            size,
                            data.as_mut_ptr().cast(),
                            ptr::null_mut(),
                        )
                    }
                    .check(ErrorStatus::Initialization) else {
                        continue;
                    };
                    data
                } else {
                    Vec::default()
                }
            };
            println!(
                "Using OpenCL platform id {platform_id}: {}, on devices:",
                String::from_utf8(platform_name).unwrap()
            );
        }
        if device_ids.is_empty() {
            continue;
        }
        let mut queue = None;
        for dev in device_ids.iter().copied() {
            // TODO get max queues per device and limit this to that number
            let mut queues = Vec::new();
            for _ in 0..2 {
                let new_queue = unsafe { clCreateCommandQueue(context, dev, 0, &raw mut status) };
                //println!("Initialized queue {new_queue:?}");
                queues.push(OpenCLQueue { queue: new_queue, load: 0 });
                let Ok(()) = status.check(ErrorStatus::Initialization) else {
                    continue;
                };
                if queue.is_none() {
                    queue = Some(new_queue);
                }
            }
            //println!("device: {dev:?}");
            let mut device = OpenCLDevice {
                ptr: dev,
                context,
                dev_info: DeviceInfo::default(),
                memory_pool_id,
                programs: Slab::new(),
                queues,
                clGetProgramBuildInfo,
                clBuildProgram,
                clReleaseProgram,
                clCreateKernel,
                clGetDeviceInfo,
                clSetKernelArg,
                clCreateProgramWithSource,
                clEnqueueNDRangeKernel,
                clFinish,
                //clGetCommandQueueInfo,
            };
            let Ok(()) = device.set_info(debug_dev) else {
                continue;
            };
            if let Ok(bytes) = device.get_device_data(CL_DEVICE_GLOBAL_MEM_SIZE) {
                total_bytes += Dim::from_ne_bytes(bytes.try_into().unwrap());
                devices.push(Device::OpenCL(device));
            }
        }
        let Ok(()) = status.check(ErrorStatus::Initialization) else {
            continue;
        };
        let pool = OpenCLMemoryPool {
            library: library.clone(),
            total_bytes,
            free_bytes: total_bytes,
            context,
            queue: queue.unwrap(),
            buffers: Slab::new(),
            clWaitForEvents,
            clReleaseCommandQueue,
            clReleaseContext,
            clReleaseMemObject,
            //clReleaseEvent,
            clEnqueueReadBuffer,
            clEnqueueWriteBuffer,
            clCreateBuffer,
        };
        memory_pools.push(Pool::new(MemoryPool::OpenCL(pool)));
        memory_pool_id += 1;
    }
    Ok(())
}

impl OpenCLMemoryPool {
    pub fn deinitialize(&mut self) {
        _ = unsafe { (self.clReleaseContext)(self.context) }.check(ErrorStatus::Deinitialization);
        _ = unsafe { (self.clReleaseCommandQueue)(self.queue) }.check(ErrorStatus::Deinitialization);
    }

    pub fn free_bytes(&self) -> Dim {
        //println!("checking free bytes = {}", self.free_bytes);
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(BufferId, Event), BackendError> {
        if bytes > self.free_bytes {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "Allocation failure".into() });
        }
        //println!("OpenCL allocating bytes {bytes}");
        let mut status = OpenCLStatus::CL_SUCCESS;
        let buffer = unsafe {
            (self.clCreateBuffer)(
                self.context,
                CL_MEM_READ_WRITE,
                bytes as usize,
                ptr::null_mut(),
                &raw mut status,
            )
        };
        status.check(ErrorStatus::MemoryAllocation)?;
        //println!("Allocated buffer {buffer:?}, bytes {bytes}");
        self.free_bytes = self.free_bytes.saturating_sub(bytes);
        Ok((
            self.buffers.push(OpenCLBuffer { buffer, bytes }),
            Event::OpenCL(OpenCLEvent { event: ptr::null_mut() }),
        ))
    }

    pub fn deallocate(&mut self, buffer_id: BufferId, event_wait_list: Vec<Event>) {
        /*println!(
            "Deallocate {:?} with events {event_wait_list:?}",
            self.buffers[buffer_id]
        );*/
        let buffer = &mut self.buffers[buffer_id];
        debug_assert!(!buffer.buffer.is_null(), "Deallocating null buffer is invalid");
        let event_wait_list: Vec<*mut c_void> = event_wait_list
            .into_iter()
            .map(|event| {
                let Event::OpenCL(OpenCLEvent { event, .. }) = event else { unreachable!() };
                event
            })
            .filter(|event| !event.is_null())
            .collect();
        if !event_wait_list.is_empty() {
            let event_wait_list_ptr = event_wait_list.as_ptr();
            let _ = unsafe { (self.clWaitForEvents)(event_wait_list.len().try_into().unwrap(), event_wait_list_ptr) }
                .check(ErrorStatus::Deinitialization);
        }
        // This segfaults... AFAIK it shouldn't...
        /*for event in event_wait_list {
            unsafe { (self.clReleaseEvent)(event) }.check(ErrorStatus::Deinitialization)?;
        }*/
        let _ = unsafe { (self.clReleaseMemObject)(buffer.buffer) }.check(ErrorStatus::Deinitialization);
        self.free_bytes += buffer.bytes;
        //println!("free_bytes after deallocation = {}", self.free_bytes);
        self.buffers.remove(buffer_id);
    }

    pub fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: BufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        //println!("Storing {src:?} to {dst:?}");
        let dst = &self.buffers[dst];
        let event_wait_list: Vec<*mut c_void> = event_wait_list
            .into_iter()
            .map(|event| {
                let Event::OpenCL(OpenCLEvent { event, .. }) = event else { unreachable!() };
                event
            })
            .filter(|event| !event.is_null())
            .collect();
        let event_wait_list_ptr = if event_wait_list.is_empty() {
            ptr::null()
        } else {
            event_wait_list.as_ptr()
        };
        let mut event = ptr::null_mut();
        unsafe {
            (self.clEnqueueWriteBuffer)(
                self.queue,
                dst.buffer,
                CL_NON_BLOCKING,
                0,
                src.len(),
                src.as_ptr().cast(),
                event_wait_list.len().try_into().expect("So many events..."),
                event_wait_list_ptr,
                &raw mut event,
            )
        }
        .check(ErrorStatus::MemoryCopyH2P)?;
        let event = Event::OpenCL(OpenCLEvent { event });
        //self.sync_events(vec![event.clone()])?;
        Ok(event)
    }

    pub fn pool_to_host(
        &mut self,
        src: BufferId,
        dst: &mut [u8],
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        let src = &self.buffers[src];
        //println!("OpenCL to host src: {src:?}, bytes {}", dst.len());
        debug_assert!(!src.buffer.is_null(), "Trying to read null memory. Internal bug.");
        let mut event_wait_list: Vec<*mut c_void> = event_wait_list
            .into_iter()
            .map(|event| {
                let Event::OpenCL(OpenCLEvent { event, .. }) = event else { unreachable!() };
                event
            })
            .filter(|event| !event.is_null())
            .collect();
        /*let event_wait_list_ptr = if event_wait_list.is_empty() {
            ptr::null()
        } else {
            event_wait_list.as_ptr()
        };*/
        if !event_wait_list.is_empty() {
            //println!("Syncing events: {event_wait_list:?}");
            unsafe {
                (self.clWaitForEvents)(
                    u32::try_from(event_wait_list.len()).expect("So many events..."),
                    event_wait_list.as_ptr(),
                )
            }
            .check(ErrorStatus::MemoryCopyP2H)?;
        }
        let mut event: *mut c_void = ptr::null_mut();
        unsafe {
            (self.clEnqueueReadBuffer)(
                self.queue,
                src.buffer,
                CL_NON_BLOCKING,
                0,
                dst.len(),
                dst.as_mut_ptr().cast(),
                0,           //event_wait_list.len().try_into().unwrap(),
                ptr::null(), //event_wait_list_ptr,
                &raw mut event,
            )
        }
        .check(ErrorStatus::MemoryCopyP2H)?;
        let events = [event];
        unsafe { (self.clWaitForEvents)(1, events.as_ptr()) }.check(ErrorStatus::MemoryCopyP2H)?;
        event_wait_list.push(event);
        // This segfaults... AFAIK it shouldn't...
        /*for event in event_wait_list {
            unsafe { (self.clReleaseEvent)(event) }.check(ErrorStatus::Deinitialization)?;
        }*/
        //println!("Opencl to host");
        Ok(())
    }

    /*fn pool_to_pool(
        &mut self,
        src: Id,
        dst_pool: &mut dyn MemoryPool,
        dst: Id,
    ) -> Result<(), BackendError> {
        //println!("Moving from {src:?} to {dst:?}");
        // TODO going through host is slow, but likely only way
        //debug_assert_eq!(self.buffers[src].bytes, dst_pool.buffers[dst].bytes);
        let mut data: Vec<u8> = vec![0; self.buffers[src].bytes];
        self.pool_to_host(src, &mut data)?;
        //println!("Copied data: {data:?}");
        dst_pool.host_to_pool(&data, dst)?;
        Ok(())
    }*/

    pub fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        let events: Vec<*mut c_void> = events
            .into_iter()
            .map(|event| {
                let Event::OpenCL(OpenCLEvent { event, .. }) = event else { unreachable!() };
                event
            })
            .filter(|event| !event.is_null())
            .collect();
        let event_wait_list_ptr = if events.is_empty() {
            ptr::null()
        } else {
            events.as_ptr()
        };
        if !events.is_empty() {
            unsafe { (self.clWaitForEvents)(events.len().try_into().expect("So many events..."), event_wait_list_ptr) }
                .check(ErrorStatus::KernelSync)?;
        }
        Ok(())
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn release_events(&mut self, events: Vec<Event>) {
        let _ = self;
        let _ = events;
        // For whatever reason this segfaults... Buggy opencl implementation?
        /*for event in events {
            let Event::OpenCL(OpenCLEvent { event }) = event else { unreachable!() };
            unsafe { (self.clReleaseEvent)(event) }.check(ErrorStatus::Deinitialization)?;
        }*/
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

    pub const fn memory_pool_id(&self) -> u32 {
        self.memory_pool_id
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<ProgramId, BackendError> {
        fn new_reg(
            op_id: OpId,
            reg_map: &mut Map<OpId, usize>,
            registers: &mut Vec<(DType, u32, u8)>,
            dtype: DType,
            rc: u32,
            current_loop_level: u8,
        ) -> usize {
            for (i, (dt, nrc, loop_level)) in registers.iter_mut().enumerate() {
                if *nrc == 0 && *dt == dtype && current_loop_level <= *loop_level {
                    reg_map.insert(op_id, i);
                    registers[i].1 = rc;
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
            registers: &mut [(DType, u32, u8)],
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

        let mut gws = Vec::new();
        let mut lws = Vec::new();
        for &op_id in &kernel.order {
            let op = &kernel[op_id];
            if let &Op::Loop { dim, scope } = op {
                match scope {
                    Scope::Global => {
                        gws.push(dim);
                    }
                    Scope::Local => {
                        lws.push(dim);
                    }
                    Scope::Register => {}
                }
            }
        }

        if lws.iter().product::<usize>() > self.dev_info.max_local_threads {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "Invalid local work size.".into(),
            });
        }

        let mut global_args = String::new();
        for &op_id in &kernel.order {
            let op = &kernel[op_id];
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
        }
        global_args.pop();
        global_args.pop();
        global_args.push('\n');

        let mut rcs: Map<OpId, u32> = Map::with_capacity_and_hasher(kernel.ops.len().into(), BuildHasherDefault::new());
        let mut dtypes: Map<OpId, DType> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());

        #[inline]
        fn dtype_of(dtypes: &Map<OpId, DType>, id: OpId) -> DType {
            *dtypes.get(&id).unwrap_or_else(|| panic!("BUG: dtype missing for op {:?}", id))
        }

        // first we will calculate those reference counts.
        for &op_id in &kernel.order {
            let op = &kernel[op_id];
            match op {
                Op::ConstView { .. } | Op::StoreView { .. } | Op::LoadView { .. } | Op::Reduce { .. } => {
                    unreachable!()
                }
                Op::Const(x) => {
                    dtypes.insert(op_id, x.dtype());
                }
                &Op::Define { dtype, .. } => {
                    dtypes.insert(op_id, dtype);
                }
                &Op::Load { src, index } => {
                    dtypes.insert(op_id, dtype_of(&dtypes, src));
                    *rcs.entry(index).or_insert(0) += 1;
                }
                &Op::Store { dst, x: src, index } => {
                    dtypes.insert(op_id, dtype_of(&dtypes, src));
                    *rcs.entry(dst).or_insert(0) += 1;
                    *rcs.entry(src).or_insert(0) += 1;
                    *rcs.entry(index).or_insert(0) += 1;
                }
                &Op::Cast { x, dtype } => {
                    dtypes.insert(op_id, dtype);
                    *rcs.entry(x).or_insert(0) += 1;
                }
                &Op::Unary { x, .. } => {
                    dtypes.insert(op_id, dtype_of(&dtypes, x));
                    *rcs.entry(x).or_insert(0) += 1;
                }
                &Op::Binary { x, y, bop } => {
                    let dtype = if matches!(bop, BOp::Cmpgt | BOp::Cmplt | BOp::NotEq | BOp::And | BOp::Or) {
                        DType::Bool
                    } else {
                        dtype_of(&dtypes, x)
                    };
                    dtypes.insert(op_id, dtype);
                    *rcs.entry(x).or_insert(0) += 1;
                    *rcs.entry(y).or_insert(0) += 1;
                }
                Op::Loop { .. } => {
                    dtypes.insert(op_id, DType::U32);
                }
                Op::EndLoop => {}
            }
        }

        let mut reg_map: Map<OpId, usize> =
            Map::with_capacity_and_hasher(kernel.ops.len().into(), BuildHasherDefault::new());
        let mut registers: Vec<(DType, u32, u8)> = Vec::new();

        let mut constants: Map<OpId, Constant> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
        let mut indices: Map<OpId, u8> = Map::with_capacity_and_hasher(20, BuildHasherDefault::new());

        let mut n_global_ids = 0;
        let mut loop_id = 0;
        let mut indent = String::from("  ");
        let mut source = String::with_capacity(1000);

        let mut acc_bytes = 0;
        for &op_id in &kernel.order {
            let op = &kernel[op_id];
            //println!("{i} -> {op:?}");
            match op {
                Op::ConstView { .. } | Op::LoadView { .. } | Op::StoreView { .. } | Op::Reduce { .. } => {
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
                        acc_bytes += dtype.byte_size() as usize * len;
                    }
                }
                &Op::Load { src, index } => {
                    if let Some(&rc) = rcs.get(&op_id) {
                        let dtype = dtypes[&src];
                        let idx = get_var(index, &constants, &indices, &reg_map, &mut registers, loop_id);
                        let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rc, loop_id);
                        //if src == OpId(28) { _ = writeln!(source, "{indent}printf(\"Load p%d[%d]\\n\", {src}, {idx});"); }
                        _ = writeln!(source, "{indent}r{reg} = p{src}[{idx}];");
                    }
                }
                &Op::Store { dst, x: src, index } => {
                    _ = writeln!(
                        source,
                        "{indent}p{dst}[{}] = {};",
                        get_var(index, &constants, &indices, &reg_map, &mut registers, loop_id),
                        get_var(src, &constants, &indices, &reg_map, &mut registers, loop_id)
                    );
                }
                &Op::Cast { x, dtype } => {
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    _ = writeln!(source, "{indent}r{reg} = ({}){x};", dtype.ocl());
                }
                &Op::Unary { x, uop } => {
                    let dtype = dtypes[&x];
                    let x = get_var(x, &constants, &indices, &reg_map, &mut registers, loop_id);
                    let reg = new_reg(op_id, &mut reg_map, &mut registers, dtype, rcs[&op_id], loop_id);
                    match uop {
                        UOp::BitNot => _ = writeln!(source, "{indent}r{reg} = ~{x};"),
                        UOp::Neg => _ = writeln!(source, "{indent}r{reg} = -{x};"),
                        UOp::Exp2 => {
                            if dtype == DType::F16 {
                                _ = writeln!(source, "{indent}r{reg} = (half)exp2((float){x});");
                            } else {
                                _ = writeln!(source, "{indent}r{reg} = exp2({x});");
                            }
                            //_ = writeln!(source, "{indent}printf(\"%d\\n\", r{reg});");
                        }
                        UOp::Log2 => _ = writeln!(source, "{indent}r{reg} = log2({x});"),
                        UOp::Reciprocal => {
                            _ = writeln!(source, "{indent}r{reg} = {}/{x};", dtype.one_constant().ocl());
                        }
                        UOp::Sqrt => _ = writeln!(source, "{indent}r{reg} = sqrt({x});"),
                        UOp::Sin => _ = writeln!(source, "{indent}r{reg} = sin({x});"),
                        UOp::Cos => _ = writeln!(source, "{indent}r{reg} = cos({x});"),
                        UOp::Floor => _ = writeln!(source, "{indent}r{reg} = floor({x});"),
                    }
                }
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
                        BOp::Maximum => writeln!(source, "{indent}r{reg} = max({x}, {y});"),
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
                &Op::Loop { dim, scope } => {
                    indices.insert(op_id, loop_id);
                    match scope {
                        Scope::Global => {
                            _ = writeln!(
                                source,
                                "{indent}unsigned int idx{loop_id} = get_group_id({loop_id}); // 0..={}",
                                dim - 1
                            );
                            n_global_ids += 1;
                        }
                        Scope::Local => {
                            _ = writeln!(
                                source,
                                "{indent}unsigned int idx{loop_id} = get_local_id({}); // 0..={}",
                                loop_id - n_global_ids,
                                dim - 1
                            );
                        }
                        Scope::Register => {
                            _ = writeln!(
                                source,
                                "{indent}for (unsigned int idx{loop_id} = 0; idx{loop_id} < {dim}; ++idx{loop_id}) {{"
                            );
                            indent += "  ";
                        }
                    }
                    loop_id += 1;
                }
                Op::EndLoop => {
                    if loop_id as usize > lws.len() + gws.len() {
                        indent.pop();
                        indent.pop();
                        _ = writeln!(source, "{indent}}}");
                        for reg in &mut registers {
                            if reg.2 == loop_id {
                                reg.1 = 0;
                            }
                        }
                        loop_id -= 1;
                    }
                }
            }
        }
        if registers.iter().map(|(dtype, ..)| dtype.byte_size() as usize).sum::<usize>() + acc_bytes > 64 {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "Kernel with too many registers.".into(),
            });
        }

        let mut reg_str = String::new();
        if registers.len() > 0 {
            let (dt, _, _) = registers.remove(0);
            let mut prev_dt = dt;
            _ = write!(reg_str, "{indent}{} r0", dt.ocl());
            let mut i = 1;
            for (dt, _, _) in registers {
                if dt == prev_dt {
                    _ = write!(reg_str, ", r{i}");
                } else {
                    _ = write!(reg_str, ";\n{indent}{} r{i}", dt.ocl());
                }
                prev_dt = dt;
                i += 1;
            }
            _ = writeln!(reg_str, ";");
        }

        let mut pragma = String::new();
        if dtypes.values().any(|&x| x == DType::F16) {
            pragma += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        }
        if dtypes.values().any(|&x| x == DType::F64) {
            pragma += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        }

        let name = format!(
            "k_{}__{}",
            gws.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"),
            lws.iter().map(ToString::to_string).collect::<Vec<_>>().join("_"),
        );

        let source = format!("{pragma}__kernel void {name}(\n{global_args}) {{\n{reg_str}{source}}}\n",);
        if debug_asm {
            println!();
            println!("{source}");
        }

        for (i, lwd) in lws.iter().enumerate() {
            gws[i] *= lwd;
        }

        let context = self.context;
        let device = self.ptr;
        let sources: &[&str] = &[source.as_str()];
        let mut status = OpenCLStatus::CL_SUCCESS;
        let program = unsafe {
            (self.clCreateProgramWithSource)(
                context,
                1,
                sources.as_ptr().cast(),
                [source.len()].as_ptr(),
                &raw mut status,
            )
        };
        status.check(ErrorStatus::KernelCompilation)?;
        if let Err(e) = unsafe {
            (self.clBuildProgram)(
                program,
                1,
                [device].as_ptr(),
                c"-cl-fast-relaxed-math".as_ptr().cast(),
                None,
                ptr::null_mut(),
            )
        }
        .check(ErrorStatus::KernelCompilation)
        {
            let build_log = self.get_program_build_data(program, CL_PROGRAM_BUILD_LOG);
            match build_log {
                Ok(build_log) => {
                    panic!("{e:?} {}", String::from_utf8_lossy(&build_log));
                }
                Err(status) => status.check(ErrorStatus::KernelCompilation)?,
            }
        }
        let mut status = OpenCLStatus::CL_SUCCESS;
        let program_name = &CString::new(name).unwrap();
        let kernel = unsafe { (self.clCreateKernel)(program, program_name.as_ptr().cast(), &raw mut status) };
        status.check(ErrorStatus::KernelCompilation)?;
        let program_id = self.programs.push(OpenCLProgram { program, kernel, gws, lws });
        /*println!(
            "Compiled program {:?} using context: {:?}",
            self.programs[program_id], self.context
        );*/
        Ok(program_id)
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn launch(
        &mut self,
        program_id: ProgramId,
        memory_pool: &mut OpenCLMemoryPool,
        args: &[BufferId],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        /*memory_pool.sync_events(event_wait_list.clone())?;
        for &arg in args {
            let buffer = &memory_pool.buffers[arg];
            let mut dst = vec![0; buffer.bytes];
            println!("arg {:?}:", buffer.buffer);
            memory_pool.pool_to_host(arg, &mut dst, vec![]).unwrap();
            println!("{dst:?}");
        }*/

        let queue_id = self.next_queue();

        /*println!(
            "Launch opencl kernel {:?}, program {:?} on queue {:?}, gws {:?}, lws {:?}",
            self.programs[program_id].kernel,
            self.programs[program_id].program,
            self.queues[queue_id].queue,
            self.programs[program_id].gws,
            self.programs[program_id].lws
        );*/
        let program = &self.programs[program_id];
        let mut i = 0;
        #[allow(clippy::explicit_counter_loop)]
        for &arg in args {
            let arg = &memory_pool.buffers[arg];
            //println!("Kernel arg: {arg:?} at index {i}");
            let ptr: *const _ = &raw const arg.buffer;
            unsafe { (self.clSetKernelArg)(program.kernel, i, core::mem::size_of::<*mut c_void>(), ptr.cast()) }
                .check(ErrorStatus::IncorrectKernelArg)?;
            i += 1;
        }
        let mut event: *mut c_void = ptr::null_mut();
        let event_wait_list: Vec<*mut c_void> = event_wait_list
            .into_iter()
            .map(|event| {
                let Event::OpenCL(OpenCLEvent { event, .. }) = event else { unreachable!() };
                event
            })
            .filter(|event| !event.is_null())
            .collect();
        //println!("Launch kernel with events: {event_wait_list:?}");
        let event_wait_list_ptr = if event_wait_list.is_empty() {
            ptr::null()
        } else {
            event_wait_list.as_ptr()
        };
        let lws_ptr = if program.lws.is_empty() {
            ptr::null()
        } else {
            program.lws.as_ptr().cast()
        };
        unsafe {
            (self.clEnqueueNDRangeKernel)(
                self.queues[queue_id].queue,
                program.kernel,
                u32::try_from(program.gws.len()).expect("So many programs..."),
                ptr::null(),
                program.gws.as_ptr().cast(),
                lws_ptr,
                event_wait_list.len().try_into().expect("So many events..."),
                event_wait_list_ptr,
                &raw mut event,
            )
        }
        .check(ErrorStatus::KernelLaunch)?;
        self.queues[queue_id].load += 1;
        unsafe { (self.clFinish)(self.queues[queue_id].queue) }.check(ErrorStatus::KernelLaunch)?;
        //println!("Launch event: {event:?}");

        /*for &arg in args {
            let buffer = &memory_pool.buffers[arg];
            let mut dst = vec![0; buffer.bytes];
            memory_pool.pool_to_host(arg, &mut dst, vec![]).unwrap();
            println!("{dst:?}");
        }*/

        Ok(Event::OpenCL(OpenCLEvent { event }))
    }

    pub fn release(&mut self, program_id: ProgramId) {
        //println!("Releasing {:?}", program_id);
        let _ =
            unsafe { (self.clReleaseProgram)(self.programs[program_id].program) }.check(ErrorStatus::Deinitialization);
        self.programs.remove(program_id);
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

impl OpenCLDevice {
    fn set_info(&mut self, debug_dev: bool) -> Result<(), BackendError> {
        let device_name = self.get_device_data(CL_DEVICE_NAME)?;
        let device_name = String::from_utf8(device_name).unwrap();
        let max_work_item_dims = self.get_device_data(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)?;
        if debug_dev {
            println!("{device_name}");
        }
        let max_work_item_dims = u32::from_ne_bytes(max_work_item_dims.try_into().unwrap()) as usize;
        let mwis = self.get_device_data(CL_DEVICE_MAX_WORK_ITEM_SIZES)?;
        let mut max_global_work_dims = vec![0; max_work_item_dims];
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
            max_global_work_dims[i] = max_dim_size as Dim;
        }
        let mlt = usize::from_ne_bytes(self.get_device_data(CL_DEVICE_MAX_WORK_GROUP_SIZE)?.try_into().unwrap()) as Dim;
        self.dev_info = DeviceInfo {
            compute: 1024 * 1024 * 1024,
            max_global_work_dims,
            max_local_threads: mlt,
            max_local_work_dims: vec![mlt; max_work_item_dims],
            preferred_vector_size: u8::try_from(u32::from_ne_bytes(
                self.get_device_data(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)?.try_into().unwrap(),
            ))
            .expect("What a vector width...")
                * 4,
            local_mem_size: Dim::try_from(usize::from_ne_bytes(
                self.get_device_data(CL_DEVICE_LOCAL_MEM_SIZE)?.try_into().unwrap(),
            ))
            .expect("What a memory size..."),
            max_register_bytes: 256,
            /*Dim::try_from(usize::from_ne_bytes(
                self.get_device_data(CL_DEVICE_MAX_PRIVATE_MEMORY_SIZE).unwrap().try_into().unwrap(),
            ))
            .expect("What a huge amount of registers"),*/
            tensor_cores: false,
        };
        Ok(())
    }

    fn get_device_data(&mut self, param_name: cl_uint) -> Result<Vec<u8>, BackendError> {
        let size = {
            let object = self.ptr;
            let mut size: usize = 0;
            let ocl_status = unsafe { (self.clGetDeviceInfo)(object, param_name, 0, ptr::null_mut(), &raw mut size) };
            if OpenCLStatus::CL_SUCCESS != ocl_status {
                return Err(BackendError {
                    status: ErrorStatus::DeviceQuery,
                    context: format!("Failed to get device info {param_name}, {ocl_status:?}").into(),
                });
            }
            Ok::<usize, BackendError>(size)
        }?;
        let object = self.ptr;
        if 0 < size {
            let count = size / core::mem::size_of::<u8>();
            let mut data: Vec<u8> = Vec::with_capacity(count);
            unsafe {
                data.set_len(count);
                (self.clGetDeviceInfo)(object, param_name, size, data.as_mut_ptr().cast(), ptr::null_mut())
            }
            .check(ErrorStatus::DeviceQuery)?;
            Ok(data)
        } else {
            Ok(Vec::default())
        }
    }

    fn get_program_build_data(&mut self, program: *mut c_void, param_name: cl_uint) -> Result<Vec<u8>, OpenCLStatus> {
        let size = {
            let idx = self.ptr;
            let mut size: usize = 0;
            let status =
                unsafe { (self.clGetProgramBuildInfo)(program, idx, param_name, 0, ptr::null_mut(), &raw mut size) };
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
                (self.clGetProgramBuildInfo)(
                    program,
                    self.ptr,
                    param_name,
                    size,
                    data.as_mut_ptr().cast(),
                    ptr::null_mut(),
                )
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

    fn next_queue(&mut self) -> usize {
        let mut id = self.queues.iter().enumerate().min_by_key(|(_, q)| q.load).unwrap().0;
        if self.queues[id].load > 20 {
            if unsafe { (self.clFinish)(self.queues[id].queue) }.check(ErrorStatus::KernelSync).is_ok() {
                self.queues[id].load = 0;
            }
            id = self.queues.iter().enumerate().min_by_key(|(_, q)| q.load).unwrap().0;
        }
        id
    }
}

impl DType {
    fn ocl(self) -> &'static str {
        match self {
            Self::BF16 => todo!("bf16 should be casted to f16 or f32"),
            Self::F16 => "half",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::U8 => "unsigned char",
            Self::U16 => "unsigned short",
            Self::I8 => "char",
            Self::I16 => "short",
            Self::I32 => "int",
            Self::I64 => "long",
            Self::Bool => "bool",
            Self::U32 => "unsigned int",
            Self::U64 => "unsigned long",
        }
    }
}

impl Constant {
    fn ocl(self) -> String {
        match self {
            Self::BF16(x) => format!("{:.16}f", half::bf16::from_le_bytes(x)),
            Self::F16(x) => format!("(half){:.16}", half::f16::from_le_bytes(x)),
            Self::F32(x) => format!("(float){:.16}", f32::from_le_bytes(x)),
            Self::F64(x) => format!("(double){:.16}", f64::from_le_bytes(x)),
            Self::U8(x) => format!("{x}"),
            Self::I8(x) => format!("{x}"),
            Self::I16(x) => format!("{x}"),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}"),
            Self::U64(x) => format!("{}", u64::from_le_bytes(x)),
            Self::I32(x) => format!("{x}"),
            Self::I64(x) => format!("{}", i64::from_le_bytes(x)),
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
const CL_DEVICE_MAX_WORK_GROUP_SIZE: cl_uint = 0x1004; // 4100
const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: cl_uint = 0x1003; // 4099
//const CL_DEVICE_MAX_PRIVATE_MEMORY_SIZE: cl_uint = 0x1160; // 4448
const CL_DEVICE_MAX_WORK_ITEM_SIZES: cl_uint = 0x1005; // 4101
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: cl_uint = 0x100A; // 4106
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
