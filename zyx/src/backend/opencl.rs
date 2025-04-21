//! `OpenCL` backend

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::question_mark)]

use super::{BufferId, Device, DeviceInfo, Event, MemoryPool, Pool, ProgramId};
use crate::{
    dtype::Constant, error::{BackendError, ErrorStatus}, graph::{BOp, UOp}, kernel_compiler::{IRKernel, IROp}, shape::Dim, slab::Slab, DType
};
use libloading::Library;
use nanoserde::DeJson;
use std::{
    ffi::{CString, c_void},
    fmt::Write,
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
    clCreateBuffer: unsafe extern "C" fn(
        *mut c_void,
        cl_bitfield,
        usize,
        *mut c_void,
        *mut OpenCLStatus,
    ) -> *mut c_void,
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
    clGetProgramBuildInfo: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        usize,
        *mut c_void,
        *mut usize,
    ) -> OpenCLStatus,
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
    clGetDeviceInfo:
        unsafe extern "C" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus,
    clSetKernelArg:
        unsafe extern "C" fn(*mut c_void, cl_uint, usize, *const c_void) -> OpenCLStatus,
    clCreateProgramWithSource: unsafe extern "C" fn(
        *mut c_void,
        cl_uint,
        *const *const i8,
        *const usize,
        *mut OpenCLStatus,
    ) -> *mut c_void,
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
    //clReleaseCommandQueue: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
}

#[derive(Debug)]
pub(super) struct OpenCLProgram {
    program: *mut c_void,
    kernel: *mut c_void,
    global_work_size: [Dim; 3],
    local_work_size: [Dim; 3],
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
    if let Some(device_ids) = &config.platform_ids {
        if device_ids.is_empty() {
            if debug_dev {
                println!("OpenCL won't be used, as it was configured out");
            }
            return Ok(());
        }
    }
    let opencl_paths = [
        "/lib/libOpenCL.so",
        "/lib64/libOpenCL.so",
        "/lib/x86_64-linux-gnu/libOpenCL.so",
        "/lib64/x86_64-linux-gnu/libOpenCL.so",
        "/usr/lib/libOpenCL.so",
        "/usr/lib64/libOpenCL.so",
        "/usr/lib/x86_64-linux-gnu/libOpenCL.so",
        "/usr/lib64/x86_64-linux-gnu/libOpenCL.so",
    ];
    let opencl = opencl_paths.iter().find_map(|path| unsafe { Library::new(path) }.ok());
    let Some(opencl) = opencl else {
        return Err(BackendError {
            status: ErrorStatus::DyLibNotFound,
            context: "OpenCL runtime not found.".into(),
        });
    };
    let clGetPlatformIDs: unsafe extern "C" fn(
        cl_uint,
        *mut *mut c_void,
        *mut cl_uint,
    ) -> OpenCLStatus = *unsafe { opencl.get(b"clGetPlatformIDs\0") }.unwrap();
    let clCreateContext: unsafe extern "C" fn(
        *const isize,
        cl_uint,
        *const *mut c_void,
        Option<unsafe extern "C" fn(*const i8, *const c_void, usize, *mut c_void)>,
        *mut c_void,
        *mut OpenCLStatus,
    ) -> *mut c_void = *unsafe { opencl.get(b"clCreateContext\0") }.unwrap();
    let clCreateCommandQueue: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_bitfield,
        *mut OpenCLStatus,
    ) -> *mut c_void = *unsafe { opencl.get(b"clCreateCommandQueue\0") }.unwrap();
    let clGetDeviceIDs: unsafe extern "C" fn(
        *mut c_void,
        cl_bitfield,
        cl_uint,
        *mut *mut c_void,
        *mut cl_uint,
    ) -> OpenCLStatus = *unsafe { opencl.get(b"clGetDeviceIDs\0") }.unwrap();
    let clWaitForEvents = *unsafe { opencl.get(b"clWaitForEvents\0") }.unwrap();
    let clReleaseCommandQueue = *unsafe { opencl.get(b"clReleaseCommandQueue\0") }.unwrap();
    let clEnqueueNDRangeKernel = *unsafe { opencl.get(b"clEnqueueNDRangeKernel\0") }.unwrap();
    let clGetProgramBuildInfo = *unsafe { opencl.get(b"clGetProgramBuildInfo\0") }.unwrap();
    let clBuildProgram = *unsafe { opencl.get(b"clBuildProgram\0") }.unwrap();
    let clReleaseProgram = *unsafe { opencl.get(b"clReleaseProgram\0") }.unwrap();
    let clReleaseContext = *unsafe { opencl.get(b"clReleaseContext\0") }.unwrap();
    //let clReleaseEvent = *unsafe { opencl.get(b"clReleaseContext\0") }.unwrap();
    let clSetKernelArg = *unsafe { opencl.get(b"clSetKernelArg\0") }.unwrap();
    let clCreateKernel = *unsafe { opencl.get(b"clCreateKernel\0") }.unwrap();
    let clReleaseMemObject = *unsafe { opencl.get(b"clReleaseMemObject\0") }.unwrap();
    let clGetDeviceInfo = *unsafe { opencl.get(b"clGetDeviceInfo\0") }.unwrap();
    let clCreateProgramWithSource = *unsafe { opencl.get(b"clCreateProgramWithSource\0") }.unwrap();
    let clEnqueueReadBuffer = *unsafe { opencl.get(b"clEnqueueReadBuffer\0") }.unwrap();
    let clEnqueueWriteBuffer = *unsafe { opencl.get(b"clEnqueueWriteBuffer\0") }.unwrap();
    let clCreateBuffer = *unsafe { opencl.get(b"clCreateBuffer\0") }.unwrap();
    let clFinish = *unsafe { opencl.get(b"clFinish\0") }.unwrap();
    let clGetPlatformInfo: unsafe extern "C" fn(
        *mut c_void,
        cl_uint,
        usize,
        *mut c_void,
        *mut usize,
    ) -> OpenCLStatus = *unsafe { opencl.get(b"clGetPlatformInfo\0") }.unwrap();

    let library = Arc::new(opencl);
    let platform_ids = {
        // Get the number of platforms
        let mut count: cl_uint = 0;
        unsafe { clGetPlatformIDs(0, ptr::null_mut(), &mut count) }
            .check(ErrorStatus::DeviceEnumeration)?;
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
    let mut memory_pool_id = u32::try_from(memory_pools.len()).unwrap();
    for (platform_id, platform) in platform_ids
        .iter()
        .enumerate()
        .filter(|(id, _)| config.platform_ids.as_ref().is_none_or(|ids| ids.contains(id)))
    {
        let platform = *platform;
        let Ok(device_ids) = {
            // Get the number of devices of device_type
            let mut count: cl_uint = 0;
            let mut status = unsafe {
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, ptr::null_mut(), &mut count)
            };
            if (OpenCLStatus::CL_SUCCESS != status) && (OpenCLStatus::CL_DEVICE_NOT_FOUND != status)
            {
                Err(status)
            } else if 0 < count {
                // Get the device ids.
                let len = count as usize;
                let mut ids: Vec<*mut c_void> = Vec::with_capacity(len);
                unsafe {
                    status = clGetDeviceIDs(
                        platform,
                        CL_DEVICE_TYPE_ALL,
                        count,
                        ids.as_mut_ptr(),
                        ptr::null_mut(),
                    );
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
                cl_uint::try_from(device_ids.len()).unwrap(),
                device_ids.as_ptr(),
                None,
                ptr::null_mut(),
                &mut status,
            )
        };
        let Ok(()) = status.check(ErrorStatus::Initialization) else {
            continue;
        };
        let mut total_bytes = 0;
        if debug_dev {
            let platform_name = {
                let mut size: usize = 0;
                let Ok(()) = unsafe {
                    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, ptr::null_mut(), &mut size)
                }
                .check(ErrorStatus::Initialization) else {
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
                "Using OpenCL platform, platform id {platform_id}, name {} on devices:",
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
            for _ in 0..8 {
                let new_queue = unsafe { clCreateCommandQueue(context, dev, 0, &mut status) };
                queues.push(OpenCLQueue { queue: new_queue, load: 0 });
                let Ok(()) = status.check(ErrorStatus::Initialization) else {
                    continue;
                };
                if queue.is_none() {
                    queue = Some(new_queue);
                }
            }
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
        unsafe { (self.clReleaseContext)(self.context) }
            .check(ErrorStatus::Deinitialization)
            .unwrap();
        unsafe { (self.clReleaseCommandQueue)(self.queue) }
            .check(ErrorStatus::Deinitialization)
            .unwrap();
    }

    pub const fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(BufferId, Event), BackendError> {
        if bytes > self.free_bytes {
            return Err(BackendError {
                status: ErrorStatus::MemoryAllocation,
                context: "Allocation failure".into(),
            });
        }
        //println!("Allocating bytes {bytes}");
        let mut status = OpenCLStatus::CL_SUCCESS;
        let buffer = unsafe {
            (self.clCreateBuffer)(
                self.context,
                CL_MEM_READ_WRITE,
                bytes,
                ptr::null_mut(),
                &mut status,
            )
        };
        status.check(ErrorStatus::MemoryAllocation)?;
        //println!("Allocated buffer {buffer:?}, bytes {bytes}");
        self.free_bytes = self.free_bytes.checked_sub(bytes).unwrap();
        Ok((
            self.buffers.push(OpenCLBuffer { buffer, bytes }),
            Event::OpenCL(OpenCLEvent { event: ptr::null_mut() }),
        ))
    }

    pub fn deallocate(&mut self, buffer_id: BufferId, event_wait_list: Vec<Event>) {
        //println!("Deallocate {:?}", self.buffers[buffer_id].ptr);
        let buffer = &mut self.buffers[buffer_id];
        debug_assert!(
            !buffer.buffer.is_null(),
            "Deallocating null buffer is invalid"
        );
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
            let _ = unsafe {
                (self.clWaitForEvents)(
                    event_wait_list.len().try_into().unwrap(),
                    event_wait_list_ptr,
                )
            }
            .check(ErrorStatus::Deinitialization);
        }
        // This segfaults... AFAIK it shouldn't...
        /*for event in event_wait_list {
            unsafe { (self.clReleaseEvent)(event) }.check(ErrorStatus::Deinitialization)?;
        }*/
        let _ = unsafe { (self.clReleaseMemObject)(buffer.buffer) }
            .check(ErrorStatus::Deinitialization);
        self.free_bytes += buffer.bytes;
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
                event_wait_list.len().try_into().unwrap(),
                event_wait_list_ptr,
                &mut event,
            )
        }
        .check(ErrorStatus::MemoryCopyH2P)?;
        Ok(Event::OpenCL(OpenCLEvent { event }))
    }

    pub fn pool_to_host(
        &mut self,
        src: BufferId,
        dst: &mut [u8],
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        let src = &self.buffers[src];
        //println!("OpenCL to host src: {src:?}, bytes {}", dst.len());
        debug_assert!(
            !src.buffer.is_null(),
            "Trying to read null memory. Internal bug."
        );
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
                    u32::try_from(event_wait_list.len()).unwrap(),
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
                &mut event,
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
            unsafe {
                (self.clWaitForEvents)(events.len().try_into().unwrap(), event_wait_list_ptr)
            }
            .check(ErrorStatus::KernelSync)?;
        }
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
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
    pub const fn deinitialize(&mut self) {
        let _ = self;
    }

    pub const fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    pub const fn memory_pool_id(&self) -> u32 {
        self.memory_pool_id
    }

    #[allow(clippy::cognitive_complexity)]
    pub fn compile(
        &mut self,
        kernel: &IRKernel,
        debug_asm: bool,
    ) -> Result<ProgramId, BackendError> {
        let mut source = String::from("(\n");
        let mut indent = String::from("  ");

        let mut global_work_size = [0; 3];
        let mut local_work_size = [0; 3];

        for (i, op) in kernel.ops[..6].iter().enumerate() {
            if let IROp::Loop { len } = op {
                if i < 3 {
                    global_work_size[i] = *len;
                } else {
                    local_work_size[i - 3] = *len;
                }
            } else {
                unreachable!()
            }
        }

        // Declare global variables
        for (id, (read_only, dtype)) in kernel.global_variables.iter().enumerate() {
            writeln!(
                source,
                "{indent}__global {}{}* p{id},",
                if *read_only { "const " } else { "" },
                dtype.ocl(),
            ).unwrap();
        }

        source.pop();
        source.pop();
        source += "\n) {\n";

        // Declare local variables
        /*for (id, (scope, dtype, len, _)) in kernel.local_variables.iter().enumerate() {
            if *scope == Scope::Local {
                writeln!(source, "{indent}__local {} p{id}[{len}];", dtype.ocl()).unwrap();
            }
        }*/

        // Declare register variables
        /*for (id, dtype) in kernel.registers.iter().enumerate() {
            writeln!(source, "{indent}{} r{id};", dtype.ocl()).unwrap();
        }*/

        // Add indices for global and local loops
        writeln!(
            source,
            "{indent}r0 = get_group_id(0);  /* 0..{} */\n{indent}r1 = get_group_id(1);  /* 0..{} */\n{indent}r2 = get_group_id(2);  /* 0..{} */\n{indent}r3 = get_local_id(0);  /* 0..{} */\n{indent}r4 = get_local_id(1);  /* 0..{} */\n{indent}r5 = get_local_id(2);  /* 0..{} */",
            global_work_size[0],
            global_work_size[1],
            global_work_size[2],
            local_work_size[0],
            local_work_size[1],
            local_work_size[2]
        ).unwrap();

        //source += &format!("{indent}printf(\"%f, %f, %f, %f\", p0[0], p0[1], p0[2], p0[3]);\n");

        let mut loop_id = 6;
        /*for op in kernel.ops[6..kernel.ops.len() - 6].iter().copied() {
            match op {
                IROp::Load { address, offset } => {
                    //source += &format!("{indent}r{z} = p{address}[{}];\n", offset.ocl());
                    let dtype = kernel.addressables[address as usize].1;
                    /*let dt_bits = dtype.bit_size();
                    if dt_bits <= 64 {
                        source += &format!(
                            "{indent}r{z} = p{address}[{0}/{1}] >> (({0} % {1}) * {dt_bits});\n",
                            offset.ocl(),
                            64 / dt_bits
                        );
                    } else {
                        todo!()
                    }*/
                    match dtype {
                        DType::BF16
                        | DType::F16
                        | DType::F32
                        | DType::F64
                        | DType::U8
                        | DType::U16
                        | DType::U32
                        | DType::U64
                        | DType::I8
                        | DType::I16
                        | DType::I32
                        | DType::I64
                        | DType::Bool => {
                            _ = writeln!(
                                source,
                                "{indent}r{z} = *((__global {}*)p{address} + {});",
                                dtype.ocl(),
                                offset.ocl(),
                            );
                        }
                    }
                }
                IROp::Store { address, offset, x } => {
                    //source += &format!("{indent}p{address}[{}] = {};\n", offset.ocl(), x.ocl());
                    let dtype = kernel.addressables[address as usize].1;
                    match dtype {
                        DType::BF16
                        | DType::F16
                        | DType::F32
                        | DType::F64
                        | DType::U8
                        | DType::U16
                        | DType::U32
                        | DType::U64
                        | DType::I8
                        | DType::I16
                        | DType::I32
                        | DType::I64
                        | DType::Bool => {
                            writeln!(
                                source,
                                "{indent}*((__global {}*)p{address} + {}) = {};",
                                dtype.ocl(),
                                offset.ocl(),
                                x.ocl(),
                            ).unwrap();
                        }
                    }
                }
                IROp::SetLocal { .. } => todo!(),
                IROp::Set { z, value } => {
                    writeln!(source, "{indent}r{z} = {};", value.ocl()).unwrap();
                }
                IROp::Cast { z, x, dtype } => {
                    writeln!(source, "{indent}r{z} = ({})r{x};", dtype.ocl()).unwrap();
                }
                IROp::Unary { z, x, uop } => {
                    let dtype = kernel.registers[z as usize];
                    match uop {
                        UOp::ReLU => writeln!(
                            source,
                            "{indent}r{z} = max(r{x}, {});",
                            dtype.zero_constant().ocl()
                        ).unwrap(),
                        UOp::Neg => writeln!(source, "{indent}r{z} = -r{x};").unwrap(),
                        UOp::Exp2 => writeln!(source, "{indent}r{z} = exp2(r{x});").unwrap(),
                        UOp::Log2 => writeln!(source, "{indent}r{z} = log2(r{x});").unwrap(),
                        UOp::Reciprocal => writeln!(source, "{indent}r{z} = 1/r{x};").unwrap(),
                        UOp::Sqrt => writeln!(
                            source,
                            "{indent}r{z} = sqrt({}r{x});",
                            if matches!(dtype, DType::F16) {
                                "(float)"
                            } else {
                                ""
                            }
                        ).unwrap(),
                        UOp::Sin => writeln!(source, "{indent}r{z} = sin(r{x});").unwrap(),
                        UOp::Cos => writeln!(source, "{indent}r{z} = cos(r{x});").unwrap(),
                        UOp::Not => writeln!(source, "{indent}r{z} = !r{x};").unwrap(),
                    }
                }
                IROp::Binary { z, x, y, bop } => {
                    let dtype = kernel.registers[z as usize];
                    writeln!(
                        source,
                        "{indent}r{z} = {};",
                        match bop {
                            BOp::Add => format!("{} + {}", x.ocl(), y.ocl()),
                            BOp::Sub => format!("{} - {}", x.ocl(), y.ocl()),
                            BOp::Mul => format!("{} * {}", x.ocl(), y.ocl()),
                            BOp::Div => format!("{} / {}", x.ocl(), y.ocl()),
                            BOp::Mod => format!("{} % {}", x.ocl(), y.ocl()),
                            BOp::Pow => format!(
                                "{}({}, {})",
                                if dtype.is_float() { "pow" } else { "pown" },
                                x.ocl(),
                                y.ocl()
                            ),
                            //BOp::Pow => format!("exp2({} * log2(abs({})))", y.ocl(), x.ocl()),
                            BOp::Cmplt => format!("{} < {}", x.ocl(), y.ocl()),
                            BOp::Cmpgt => format!("{} > {}", x.ocl(), y.ocl()),
                            BOp::NotEq => format!("{} != {}", x.ocl(), y.ocl()),
                            BOp::Max => format!("max({}, {})", x.ocl(), y.ocl()),
                            BOp::Or => format!("{} || {}", x.ocl(), y.ocl()),
                            BOp::And => format!("{} && {}", x.ocl(), y.ocl()),
                            BOp::BitOr => format!("{} | {}", x.ocl(), y.ocl()),
                            BOp::BitAnd => format!("{} & {}", x.ocl(), y.ocl()),
                            BOp::BitXor => format!("{} ^ {}", x.ocl(), y.ocl()),
                            BOp::BitShiftLeft => format!("{} << {}", x.ocl(), y.ocl()),
                            BOp::BitShiftRight => format!("{} >> {}", x.ocl(), y.ocl()),
                        }
                    ).unwrap();
                    //if z == 24 && bop == BOp::Sub { source += "  printf(\"r24: %f i2; %u i4: %u\\n\", r24, r2, r4);\n"; }
                }
                /*IROp::MAdd { z, a, b, c } => {
                    let dtype = kernel.registers[z as usize];
                    if dtype.is_float() {
                        writeln!(
                            source,
                            "{indent}r{z} = mad({}, {}, {});\n",
                            a.ocl(),
                            b.ocl(),
                            c.ocl()
                        ).unwrap();
                    } else {
                        writeln!(source, "{indent}r{z} = {} * {} + {};", a.ocl(), b.ocl(), c.ocl()).unwrap();
                    }
                }*/
                IROp::Loop { len } => {
                    writeln!(source, "{indent}for (r{loop_id} = 0; r{loop_id} < {len}; r{loop_id} += 1) {{").unwrap();
                    indent += "  ";
                    loop_id += 1;
                }
                IROp::EndLoop => {
                    indent.pop();
                    indent.pop();
                    writeln!(source, "{indent}}}").unwrap();
                    loop_id -= 1;
                }
                IROp::LocalBarrier => {
                    writeln!(source, "{indent}barrier(CLK_LOCAL_MEM_FENCE);");
                }
            }
        }*/
        source += "}\n";

        let local_work_size = local_work_size;
        let name = format!(
            "k_{}_{}_{}__{}_{}_{}",
            global_work_size[0],
            global_work_size[1],
            global_work_size[2],
            local_work_size[0],
            local_work_size[1],
            local_work_size[2],
        );
        for (i, lwd) in local_work_size.iter().enumerate() {
            global_work_size[i] *= lwd;
        }
        let mut pragma = String::new();
        if source.contains("half") {
            pragma += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        }
        if source.contains("double") {
            pragma += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        }
        let source = format!("{pragma}__kernel void {name}{source}");
        if debug_asm {
            println!("{source}");
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
                &mut status,
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
        let kernel =
            unsafe { (self.clCreateKernel)(program, program_name.as_ptr().cast(), &mut status) };
        status.check(ErrorStatus::KernelCompilation)?;
        Ok(
            self.programs.push(OpenCLProgram {
                program,
                kernel,
                global_work_size,
                local_work_size,
            }),
        )
    }

    pub fn launch(
        &mut self,
        program_id: ProgramId,
        memory_pool: &mut OpenCLMemoryPool,
        args: &[BufferId],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        //memory_pool.sync_events(event_wait_list.clone())?;
        /*for &arg in args {
            let buffer = memory_pool.get_buffer(arg);
            let BufferMut::OpenCL(buffer) = buffer else { unreachable!() };
            let mut dst = vec![0; buffer.bytes];
            memory_pool.pool_to_host(arg, &mut dst, vec![]).unwrap();
            println!("{dst:?}");
        }*/

        let queue_id = self.next_queue()?;
        /*println!(
            "Launch opencl kernel {:?}, program {:?} on queue {:?}, gws {:?}, lws {:?}",
            self.programs[program_id].kernel,
            self.programs[program_id].program,
            self.queues[queue_id].queue,
            self.programs[program_id].global_work_size,
            self.programs[program_id].local_work_size
        );*/
        let program = &self.programs[program_id];
        let mut i = 0;
        #[allow(clippy::explicit_counter_loop)]
        for &arg in args {
            let arg = &memory_pool.buffers[arg];
            //println!("Kernel arg: {arg:?} at index {i}");
            let ptr: *const _ = &arg.buffer;
            unsafe {
                (self.clSetKernelArg)(
                    program.kernel,
                    i,
                    core::mem::size_of::<*mut c_void>(),
                    ptr.cast(),
                )
            }
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
        unsafe {
            (self.clEnqueueNDRangeKernel)(
                self.queues[queue_id].queue,
                program.kernel,
                u32::try_from(program.global_work_size.len()).unwrap(),
                ptr::null(),
                program.global_work_size.as_ptr().cast(),
                program.local_work_size.as_ptr().cast(),
                event_wait_list.len().try_into().unwrap(),
                event_wait_list_ptr,
                &mut event,
            )
        }
        .check(ErrorStatus::KernelLaunch)?;
        self.queues[queue_id].load += 1;
        //unsafe { (self.clFinish)(self.queues[queue_id].queue) }.check(ErrorStatus::KernelLaunch)?;
        //println!("Launch event: {event:?}");
        Ok(Event::OpenCL(OpenCLEvent { event }))
    }

    pub fn release(&mut self, program_id: ProgramId) {
        //println!("Releasing {:?}", program);
        let _ = unsafe { (self.clReleaseProgram)(self.programs[program_id].program) }
            .check(ErrorStatus::Deinitialization);
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
        let mut max_work_item_dims =
            u32::from_ne_bytes(max_work_item_dims.try_into().unwrap()) as usize;
        let mwis = self.get_device_data(CL_DEVICE_MAX_WORK_ITEM_SIZES)?;
        let mut max_global_work_dims: [Dim; 3] = [0; 3];
        if max_work_item_dims > 3 {
            println!(
                "Found device with more than 3 work dimesions, WOW. Please report this. Using only 3 dims for now."
            );
            max_work_item_dims = 3;
        }
        for i in 0..max_work_item_dims {
            let max_dim_size: usize = unsafe {
                core::mem::transmute([
                    mwis[i * 8],
                    mwis[i * 8 + 1],
                    mwis[i * 8 + 2],
                    mwis[i * 8 + 3],
                    mwis[i * 8 + 4],
                    mwis[i * 8 + 5],
                    mwis[i * 8 + 6],
                    mwis[i * 8 + 7],
                ])
            };
            max_global_work_dims[i] = max_dim_size;
        }
        let mlt = usize::from_ne_bytes(
            self.get_device_data(CL_DEVICE_MAX_WORK_GROUP_SIZE)?.try_into().unwrap(),
        );
        self.dev_info = DeviceInfo {
            compute: 1024 * 1024 * 1024 * 1024,
            max_global_work_dims,
            max_local_threads: mlt,
            max_local_work_dims: [mlt, mlt, mlt],
            preferred_vector_size: u8::try_from(u32::from_ne_bytes(
                self.get_device_data(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)?.try_into().unwrap(),
            )).unwrap() * 4,
            local_mem_size: Dim::try_from(u64::from_ne_bytes(
                self.get_device_data(CL_DEVICE_LOCAL_MEM_SIZE)?.try_into().unwrap(),
            ))
            .unwrap(),
            num_registers: 96, // We can only guess or have a map of concrete hardware and respective register counts
            tensor_cores: false,
        };
        Ok(())
    }

    fn get_device_data(&mut self, param_name: cl_uint) -> Result<Vec<u8>, BackendError> {
        let size = {
            let object = self.ptr;
            let mut size: usize = 0;
            let ocl_status = unsafe {
                (self.clGetDeviceInfo)(object, param_name, 0, ptr::null_mut(), &mut size)
            };
            if OpenCLStatus::CL_SUCCESS != ocl_status {
                return Err(BackendError {
                    status: ErrorStatus::DeviceQuery,
                    context: format!("Failed to get device info {param_name}, {ocl_status:?}")
                        .into(),
                });
            }
            Ok(size)
        }?;
        let object = self.ptr;
        if 0 < size {
            let count = size / core::mem::size_of::<u8>();
            let mut data: Vec<u8> = Vec::with_capacity(count);
            unsafe {
                data.set_len(count);
                (self.clGetDeviceInfo)(
                    object,
                    param_name,
                    size,
                    data.as_mut_ptr().cast(),
                    ptr::null_mut(),
                )
            }
            .check(ErrorStatus::DeviceQuery)?;
            Ok(data)
        } else {
            Ok(Vec::default())
        }
    }

    fn get_program_build_data(
        &mut self,
        program: *mut c_void,
        param_name: cl_uint,
    ) -> Result<Vec<u8>, OpenCLStatus> {
        let size = {
            let idx = self.ptr;
            let mut size: usize = 0;
            let status = unsafe {
                (self.clGetProgramBuildInfo)(
                    program,
                    idx,
                    param_name,
                    0,
                    ptr::null_mut(),
                    &mut size,
                )
            };
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

    fn next_queue(&mut self) -> Result<usize, BackendError> {
        let mut id = self.queues.iter().enumerate().min_by_key(|(_, q)| q.load).unwrap().0;
        if self.queues[id].load > 20 {
            unsafe { (self.clFinish)(self.queues[id].queue) }.check(ErrorStatus::KernelSync)?;
            self.queues[id].load = 0;
            id = self.queues.iter().enumerate().min_by_key(|(_, q)| q.load).unwrap().0;
        }
        Ok(id)
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
    fn ocl(&self) -> String {
        match self {
            &Self::BF16(x) => format!("{:.16}f", half::bf16::from_bits(x)),
            &Self::F16(x) => format!("(half){:.16}", half::f16::from_bits(x)),
            &Self::F32(x) => format!("{:.16}f", f32::from_bits(x)),
            &Self::F64(x) => format!("{:.16}", f64::from_bits(x)),
            Self::U8(x) => format!("{x}"),
            Self::I8(x) => format!("{x}"),
            Self::I16(x) => format!("{x}"),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}"),
            Self::U64(x) => format!("{x}"),
            Self::I32(x) => format!("{x}"),
            Self::I64(x) => format!("{x}"),
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
