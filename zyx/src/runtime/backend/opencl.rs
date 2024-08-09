use std::{collections::BTreeSet, ffi::c_void, ptr};

use crate::{
    runtime::{Buffer, BufferId, MemoryKind, MemoryPool, MemoryPoolId},
    DType,
};

pub(crate) struct OpenCLBackend {
    context: *mut c_void,
    devices: BTreeSet<*mut c_void>,
    queues: Box<[*mut c_void]>,
    queue_size: Box<[u8]>,
    queue_id: usize,
    memory_pools: Vec<OpenCLMemoryPool>,
}

struct OpenCLMemoryPool {
    buffers: Vec<OpenCLBuffer>,
    total_bytes: usize,
    free_bytes: usize,
}

struct OpenCLBuffer {
    ptr: *mut c_void,
}

unsafe impl Send for OpenCLBackend {}
unsafe impl Send for OpenCLBuffer {}

impl OpenCLBackend {
    pub(crate) fn new() -> Result<Self, OpenCLError> {
        let platform_id = 0;
        let queues_per_device = 8;
        let platform_ids = {
            // Get the number of platforms
            let mut count: cl_uint = 0;
            let status = unsafe { clGetPlatformIDs(0, ptr::null_mut(), &mut count) };
            check(status, "Unable to get OpenCL platform ids.")?;
            if count > 0 {
                // Get the platform ids.
                let len = count as usize;
                let mut ids: Vec<*mut c_void> = Vec::with_capacity(len);
                let status = unsafe { clGetPlatformIDs(count, ids.as_mut_ptr(), ptr::null_mut()) };
                check(status, "Unable to get OpenCL platform ids.")?;
                unsafe { ids.set_len(len) };
                ids
            } else {
                Vec::new()
            }
        };
        let Some(platform) = platform_ids.get(platform_id) else {
            return Err(OpenCLError {
                status: OpenCLStatus::NO_PLATFORM,
                info: "There are no available OpenCL platforms.".into(),
            }
            .into());
        };
        let platform = *platform;
        #[cfg(feature = "debug_dev")]
        println!(
            "Using OpenCL platform: {}",
            String::from_utf8(get_platform_data(platform, CL_PLATFORM_NAME)?).unwrap()
        );
        let device_ids = get_device_ids(platform, CL_DEVICE_TYPE_ALL)
            .map_err(|err| check(err, "Unable to get OpenCL device ids").err().unwrap())?;
        #[cfg(feature = "debug_dev")]
        println!("Using devices:");
        #[cfg(feature = "debug_dev")]
        for dev in &device_ids {
            println!(
                "{}",
                String::from_utf8(get_device_data(*dev, CL_DEVICE_NAME).map_err(|err| {
                    check(err, "Unable to get OpenCL device name.")
                        .err()
                        .unwrap()
                })?)
                .unwrap()
            );
        }
        let mut status = CL_SUCCESS;
        let context = unsafe {
            clCreateContext(
                ptr::null(),
                device_ids.len() as cl_uint,
                device_ids.as_ptr(),
                None,
                ptr::null_mut(),
                &mut status,
            )
        };
        check(status, "Unable to create OpenCL context")?;
        // This makes our code asynchronous. Creating graph would actually make us 2 times slower (can be optimized),
        // if we couldn't execute kernels asynchronously. We don't need this to be huge. 2 seems to
        // be plenty. And lower values also lower memory usage.
        //device_ids.iter().map(|dev| get_device_info(*dev, CL_DEVICE_MAX_ON_DEVICE_QUEUES)?.into()).min()?;
        #[cfg(feature = "debug_dev")]
        println!("Using {queues_per_device} queues per device.");
        let (queues, errs): (Vec<*mut c_void>, Vec<cl_int>) = (0..queues_per_device)
            .flat_map(|_| {
                device_ids.iter().map(move |dev| {
                    let queue = unsafe { clCreateCommandQueue(context, *dev, 0, &mut status) };
                    (queue, status)
                })
            })
            .unzip();
        for status in errs {
            check(status, "Unable to create command queue")?;
        }
        let mut devices = BTreeSet::new();
        for dev in device_ids {
            devices.insert(dev);
        }
        return Ok(Self {
            context,
            devices,
            queue_size: vec![0; queues.len()].into_boxed_slice(),
            queues: queues.into_boxed_slice(),
            queue_id: 0,
            memory_pools: vec![OpenCLMemoryPool {
                buffers: Vec::new(),
                total_bytes: 4 * 1024 * 1024 * 1024,
                free_bytes: 4 * 1024 * 1024 * 1024,
            }],
        });
    }

    pub(crate) fn memory_pools(&self) -> Vec<MemoryPool> {
        self.memory_pools
            .iter()
            .map(|mp| MemoryPool {
                kind: MemoryKind::OpenCL,
                total_bytes: mp.total_bytes,
                free_bytes: mp.free_bytes,
            })
            .collect()
    }

    pub(crate) fn allocate_memory(
        &mut self,
        byte_size: usize,
        memory_pool: MemoryPoolId,
    ) -> Result<BufferId, OpenCLError> {
        let mut status = CL_SUCCESS;
        let ptr = unsafe {
            clCreateBuffer(
                self.context,
                CL_MEM_READ_ONLY,
                byte_size,
                ptr::null_mut(),
                &mut status,
            )
        };
        check(status, "Unable to allocate memory.")?;
        let id = self.memory_pools[memory_pool].buffers.len();
        self.memory_pools[memory_pool]
            .buffers
            .push(OpenCLBuffer { ptr });
        Ok(id)
    }

    pub(crate) fn deallocate_memory(&mut self, buffer: Buffer) -> Result<(), OpenCLError> {
        let ptr = self.memory_pools[buffer.memory_pool].buffers[buffer.id].ptr;
        let status = unsafe { clReleaseMemObject(ptr) };
        check(status, "Unable to free allocated memory")?;
        Ok(())
    }

    pub(crate) fn host_to_opencl(&mut self, src: &[u8], dst: Buffer) -> Result<(), OpenCLError> {
        let mut event = ptr::null_mut();
        let ptr = self.memory_pools[dst.memory_pool].buffers[dst.id].ptr;
        let status = unsafe {
            clEnqueueWriteBuffer(
                self.queue()?,
                ptr,
                CL_NON_BLOCKING,
                0,
                src.len(),
                src.as_ptr().cast(),
                0,
                ptr::null(),
                &mut event,
            )
        };
        check(status, "Unable to write buffer.")?;
        // Immediattely synchronize because we do not know the lifetime of data
        let status = unsafe { clWaitForEvents(1, (&[event]).as_ptr().cast()) };
        check(status, "Unable to finish buffer write event.")
    }

    // Perhaps this can be done directly, for now we go through host
    //pub(crate) fn cuda_to_opencl(&mut self, src: Buffer, dst: Buffer) -> Result<(), OpenCLError> {}

    pub(crate) fn opencl_to_opencl(&mut self, src: Buffer, dst: Buffer) -> Result<(), OpenCLError> {
        todo!()
    }

    pub(crate) fn opencl_to_host(&mut self, src: Buffer, dst: &mut [u8]) -> Result<(), OpenCLError> {
        let src = self.memory_pools[src.memory_pool].buffers[src.id].ptr;
        assert!(
            !src.is_null(),
            "Trying to read null memory. Internal bug."
        );
        let mut event: *mut c_void = ptr::null_mut();
        let status = unsafe {
            clEnqueueReadBuffer(
                self.queue()?,
                src,
                CL_NON_BLOCKING,
                0,
                dst.len(),
                dst.as_mut_ptr().cast(),
                0,
                ptr::null_mut(),
                &mut event,
            )
        };
        check(status, "Unable to read buffer.")?;
        cl_wait_for_events(&[event])?;
        Ok(())
    }

    fn queue(&mut self) -> Result<*mut c_void, OpenCLError> {
        let res = self.queues[self.queue_id];
        self.queue_size[self.queue_id] += 1;
        // Blocks and waits for queue to finish execution so that
        // we do not overwhelm the device with tasks.
        // Up to two events per queue, before opencl 2.0 we can't do
        // much better than that. After opencl 2.0 we can get status
        // of the queues, but is it really necessary?.
        if self.queue_size[self.queue_id] == 2 {
            let status = unsafe { clFinish(res) };
            check(status, "Unable to finish execution of command queue.")?;
            self.queue_size[self.queue_id] = 0;
        }
        self.queue_id = (self.queue_id + 1) % self.queues.len();
        return Ok(res);
    }
}

impl DType {
    pub(crate) fn ocl(&self) -> &str {
        return match self {
            #[cfg(feature = "half")]
            DType::BF16 => panic!("BF16 is not native to OpenCL, workaround is WIP."),
            #[cfg(feature = "half")]
            DType::F16 => "half",
            DType::F32 => "float",
            DType::F64 => "double",
            #[cfg(feature = "complex")]
            DType::CF32 => panic!("Not native to OpenCL, workaround is WIP"),
            #[cfg(feature = "complex")]
            DType::CF64 => panic!("Not native to OpenCL, workaround is WIP"),
            DType::U8 => "unsigned char",
            DType::I8 => "char",
            DType::I16 => "short",
            DType::I32 => "int",
            DType::I64 => "long",
            DType::Bool => "bool",
        };
    }
}

fn check(status: cl_int, info: &str) -> Result<(), OpenCLError> {
    if status == CL_SUCCESS {
        return Ok(());
    } else {
        return Err(OpenCLError {
            info: info.into(),
            status: OpenCLStatus::new(status),
        });
    }
}

fn cl_wait_for_events(events: &[*mut c_void]) -> Result<(), OpenCLError> {
    let status = unsafe { clWaitForEvents(1, events.as_ptr().cast()) };
    return check(status, "Unable to finish buffer read event.");
}

fn get_program_build_data(
    program: *mut c_void,
    device: *mut c_void,
    param_name: cl_uint,
) -> Result<Vec<u8>, cl_int> {
    fn get_size(
        object: *mut c_void,
        idx: *mut c_void,
        param_name: cl_uint,
    ) -> Result<usize, cl_int> {
        let mut size: usize = 0;
        let status = unsafe {
            clGetProgramBuildInfo(object, idx, param_name, 0, ptr::null_mut(), &mut size)
        };
        if CL_SUCCESS != status {
            return Err(status);
        } else {
            return Ok(size);
        }
    }
    let size = get_size(program, device, param_name)?;
    fn get_vector(
        object: *mut c_void,
        idx: *mut c_void,
        param_name: cl_uint,
        size: usize,
    ) -> Result<Vec<u8>, cl_int> {
        if 0 < size {
            let count = size / core::mem::size_of::<u8>();
            let mut data: Vec<u8> = Vec::with_capacity(count);
            let status = unsafe {
                data.set_len(count);
                clGetProgramBuildInfo(
                    object,
                    idx,
                    param_name,
                    size,
                    data.as_mut_ptr() as *mut c_void,
                    ptr::null_mut(),
                )
            };
            if CL_SUCCESS != status {
                return Err(status);
            } else {
                return Ok(data);
            }
        } else {
            return Ok(Vec::default());
        }
    }
    return get_vector(program, device, param_name, size);
}

fn get_device_data(device: *mut c_void, param_name: cl_uint) -> Result<Vec<u8>, cl_int> {
    fn get_size(object: *mut c_void, param_name: cl_uint) -> Result<usize, cl_int> {
        let mut size: usize = 0;
        let status = unsafe { clGetDeviceInfo(object, param_name, 0, ptr::null_mut(), &mut size) };
        if CL_SUCCESS != status {
            return Err(status);
        } else {
            return Ok(size);
        }
    }
    let size = get_size(device, param_name)?;
    fn get_vector(
        object: *mut c_void,
        param_name: cl_uint,
        size: usize,
    ) -> Result<Vec<u8>, cl_int> {
        if 0 < size {
            let count = size / core::mem::size_of::<u8>();
            let mut data: Vec<u8> = Vec::with_capacity(count);
            let status = unsafe {
                data.set_len(count);
                clGetDeviceInfo(
                    object,
                    param_name,
                    size,
                    data.as_mut_ptr() as *mut c_void,
                    ptr::null_mut(),
                )
            };
            if CL_SUCCESS != status {
                return Err(status);
            } else {
                return Ok(data);
            }
        } else {
            return Ok(Vec::default());
        }
    }
    return get_vector(device, param_name, size);
}

fn get_device_ids(
    platform: *mut c_void,
    device_type: cl_bitfield,
) -> Result<Vec<*mut c_void>, cl_int> {
    // Get the number of devices of device_type
    let mut count: cl_uint = 0;
    let mut status =
        unsafe { clGetDeviceIDs(platform, device_type, 0, ptr::null_mut(), &mut count) };

    if (CL_SUCCESS != status) && (CL_DEVICE_NOT_FOUND != status) {
        return Err(status);
    } else if 0 < count {
        // Get the device ids.
        let len = count as usize;
        let mut ids: Vec<*mut c_void> = Vec::with_capacity(len);
        unsafe {
            status = clGetDeviceIDs(
                platform,
                device_type,
                count,
                ids.as_mut_ptr(),
                ptr::null_mut(),
            );
            ids.set_len(len);
        };

        if CL_SUCCESS != status {
            return Err(status);
        } else {
            return Ok(ids);
        }
    } else {
        return Ok(Vec::default());
    }
}

#[cfg(feature = "debug_dev")]
fn get_platform_data(platform: *mut c_void, param_name: cl_uint) -> Result<Vec<u8>, OpenCLError> {
    let mut size: usize = 0;
    let status = unsafe { clGetPlatformInfo(platform, param_name, 0, ptr::null_mut(), &mut size) };
    check(status, "Unable to get platform info.")?;
    if 0 < size {
        let count = size / core::mem::size_of::<u8>();
        let mut data: Vec<u8> = Vec::with_capacity(count);
        let status = unsafe {
            data.set_len(count);
            clGetPlatformInfo(
                platform,
                param_name,
                size,
                data.as_mut_ptr() as *mut c_void,
                ptr::null_mut(),
            )
        };
        check(status, "Unable to get platform info.")?;
        return Ok(data);
    } else {
        return Ok(Vec::default());
    }
}

type cl_int = i32;
type cl_uint = u32;
type cl_bitfield = u64;

const CL_PLATFORM_NAME: cl_uint = 0x0902; // 2306
const CL_DEVICE_NAME: cl_uint = 0x102B; // 4139
const CL_DEVICE_GLOBAL_MEM_SIZE: cl_uint = 0x101F; // 4127
const CL_DEVICE_LOCAL_MEM_SIZE: cl_uint = 0x1023; // 4131
const CL_DEVICE_MAX_MEM_ALLOC_SIZE: cl_uint = 0x1010; // 4112
const CL_DEVICE_MAX_WORK_GROUP_SIZE: cl_uint = 0x1004; // 4100
const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: cl_uint = 0x1003; // 4099
const CL_DEVICE_MAX_WORK_ITEM_SIZES: cl_uint = 0x1005; // 4101
const CL_DEVICE_MEM_BASE_ADDR_ALIGN: cl_uint = 0x1019; // 4121
const CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: cl_uint = 0x101A; // 4122
const CL_DEVICE_NOT_FOUND: cl_int = -1; // 0xFFFF_FFFF
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: cl_uint = 0x100A; // 4106
const CL_DEVICE_TYPE_ALL: cl_bitfield = 0xFFFF_FFFF;
const CL_MEM_READ_ONLY: cl_bitfield = 4;
const CL_NON_BLOCKING: cl_uint = 0;
const CL_PROGRAM_BUILD_LOG: cl_uint = 0x1183; // 4483
const CL_SUCCESS: cl_int = 0;

#[allow(dead_code)] // Rust for some reason thinks these fields are unused
#[derive(Debug)]
pub struct OpenCLError {
    info: String,
    status: OpenCLStatus,
}

#[derive(Copy, Clone, PartialEq, Debug, Eq)]
#[repr(C)]
enum OpenCLStatus {
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
    NO_PLATFORM,
    UNKNOWN,
}

impl OpenCLStatus {
    fn new(status: cl_int) -> Self {
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

#[cfg_attr(not(target_os = "macos"), link(name = "OpenCL"))]
#[cfg_attr(target_os = "macos", link(name = "OpenCL", kind = "framework"))]
extern "system" {
    // Platform API
    fn clGetPlatformIDs(
        num_entries: cl_uint,
        platforms: *mut *mut c_void,
        num_platforms: *mut cl_uint,
    ) -> cl_int;

    fn clGetPlatformInfo(
        platform: *mut c_void,
        param_name: cl_uint,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> cl_int;

    fn clGetDeviceInfo(
        device: *mut c_void,
        param_name: cl_uint,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> cl_int;

    // Device APIs
    fn clGetDeviceIDs(
        platform: *mut c_void,
        device_type: cl_bitfield,
        num_entries: cl_uint,
        devices: *mut *mut c_void,
        num_devices: *mut cl_uint,
    ) -> cl_int;

    /*#[cfg(feature = "CL_VERSION_1_2")]
    fn clReleaseDevice(device: cl_device_id) -> cl_int;*/

    fn clCreateContext(
        properties: *const isize,
        num_devices: cl_uint,
        devices: *const *mut c_void,
        pfn_notify: Option<
            unsafe extern "C" fn(
                errinfo: *const i8,
                private_info: *const c_void,
                cb: usize,
                user_data: *mut c_void,
            ),
        >,
        user_data: *mut c_void,
        errcode_ret: *mut cl_int,
    ) -> *mut c_void;

    fn clReleaseContext(context: *mut c_void) -> cl_int;

    //#[cfg(not(feature = "CL_VERSION_2_0"))]
    fn clCreateCommandQueue(
        context: *mut c_void,
        device: *mut c_void,
        properties: cl_bitfield,
        errcode_ret: *mut cl_int,
    ) -> *mut c_void;

    /*#[cfg(feature = "CL_VERSION_2_0")]
    fn clCreateCommandQueueWithProperties(
        context: cl_context,
        device: cl_device_id,
        properties: *const cl_queue_properties,
        errcode_ret: *mut cl_int,
    ) -> cl_command_queue;*/

    fn clCreateBuffer(
        context: *mut c_void,
        flags: cl_bitfield,
        size: usize,
        host_ptr: *mut c_void,
        errcode_ret: *mut cl_int,
    ) -> *mut c_void;

    fn clEnqueueWriteBuffer(
        command_queue: *mut c_void,
        buffer: *mut c_void,
        blocking_write: cl_uint,
        offset: usize,
        cb: usize,
        ptr: *const c_void,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const *mut c_void,
        event: *mut *mut c_void,
    ) -> cl_int;

    pub fn clEnqueueReadBuffer(
        command_queue: *mut c_void,
        buffer: *mut c_void,
        blocking_read: cl_uint,
        offset: usize,
        cb: usize,
        ptr: *mut c_void,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const *mut c_void,
        event: *mut *mut c_void,
    ) -> cl_int;

    fn clReleaseMemObject(memobj: *mut c_void) -> cl_int;

    fn clCreateProgramWithSource(
        context: *mut c_void,
        count: cl_uint,
        strings: *const *const i8,
        lengths: *const usize,
        errcode_ret: *mut cl_int,
    ) -> *mut c_void;

    fn clReleaseProgram(program: *mut c_void) -> cl_int;

    fn clCreateKernel(
        program: *mut c_void,
        kernel_name: *const i8,
        errcode_ret: *mut cl_int,
    ) -> *mut c_void;

    fn clSetKernelArg(
        kernel: *mut c_void,
        arg_index: cl_uint,
        arg_size: usize,
        arg_value: *const c_void,
    ) -> cl_int;

    fn clBuildProgram(
        program: *mut c_void,
        num_devices: cl_uint,
        device_list: *const *mut c_void,
        options: *const i8,
        pfn_notify: Option<unsafe extern "C" fn(program: *mut c_void, user_data: *mut c_void)>,
        user_data: *mut c_void,
    ) -> cl_int;

    fn clGetProgramBuildInfo(
        program: *mut c_void,
        device: *mut c_void,
        param_name: cl_uint,
        param_value_size: usize,
        param_value: *mut c_void,
        param_value_size_ret: *mut usize,
    ) -> cl_int;

    fn clEnqueueNDRangeKernel(
        command_queue: *mut c_void,
        kernel: *mut c_void,
        work_dim: cl_uint,
        global_work_offset: *const usize,
        global_work_dims: *const usize,
        local_work_dims: *const usize,
        num_events_in_wait_list: cl_uint,
        event_wait_list: *const *mut c_void,
        event: *mut *mut c_void,
    ) -> cl_int;

    fn clWaitForEvents(num_events: cl_uint, event_list: *const *mut c_void) -> cl_int;

    fn clFinish(command_queue: *mut c_void) -> cl_int;

    //fn clReleaseEvent(event: *mut c_void) -> cl_int;
}
