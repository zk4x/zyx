use alloc::{
    boxed::Box, collections::BTreeSet, ffi::CString, format as f, string::String, vec::Vec,
};
use core::ffi::c_void;
use opencl_sys::{
    clBuildProgram, clCreateBuffer, clCreateCommandQueue, clCreateContext, clCreateKernel,
    clCreateProgramWithSource, clEnqueueNDRangeKernel, clEnqueueReadBuffer, clEnqueueWriteBuffer,
    clGetDeviceIDs, clGetDeviceInfo, clGetPlatformIDs, clGetPlatformInfo, clGetProgramBuildInfo,
    clReleaseEvent, clReleaseMemObject, clReleaseProgram, clSetKernelArg, clWaitForEvents,
    cl_device_id, cl_device_info, cl_device_type, cl_int, cl_platform_id, cl_platform_info,
    cl_program_info, cl_uint, CL_DEVICE_NAME, CL_DEVICE_NOT_FOUND, CL_DEVICE_TYPE_ALL,
    CL_MEM_HOST_READ_ONLY, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_NON_BLOCKING, CL_PLATFORM_NAME,
    CL_PROGRAM_BUILD_LOG, CL_SUCCESS,
};
use zyx_core::{
    compiler::ROp,
    compiler::{Op, AST},
    dtype::DType,
    error::ZyxError,
    scalar::Scalar,
};

fn get_program_build_data(
    program: *mut c_void,
    device: cl_device_id,
    param_name: cl_program_info,
) -> Result<Vec<u8>, cl_int> {
    fn get_size(
        object: *mut c_void,
        idx: cl_device_id,
        param_name: cl_uint,
    ) -> Result<usize, cl_int> {
        let mut size: usize = 0;
        let status = unsafe {
            clGetProgramBuildInfo(object, idx, param_name, 0, core::ptr::null_mut(), &mut size)
        };
        if CL_SUCCESS != status {
            Err(status)
        } else {
            Ok(size)
        }
    }
    let size = get_size(program, device, param_name)?;
    fn get_vector(
        object: *mut c_void,
        idx: cl_device_id,
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
                    core::ptr::null_mut(),
                )
            };
            if CL_SUCCESS != status {
                Err(status)
            } else {
                Ok(data)
            }
        } else {
            Ok(Vec::default())
        }
    }
    get_vector(program, device, param_name, size)
}

pub fn get_device_data(
    device: cl_device_id,
    param_name: cl_device_info,
) -> Result<Vec<u8>, cl_int> {
    fn get_size(object: *mut c_void, param_name: cl_uint) -> Result<usize, cl_int> {
        let mut size: usize = 0;
        let status =
            unsafe { clGetDeviceInfo(object, param_name, 0, core::ptr::null_mut(), &mut size) };
        if CL_SUCCESS != status {
            Err(status)
        } else {
            Ok(size)
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
                    core::ptr::null_mut(),
                )
            };
            if CL_SUCCESS != status {
                Err(status)
            } else {
                Ok(data)
            }
        } else {
            Ok(Vec::default())
        }
    }
    get_vector(device, param_name, size)
}

fn get_device_ids(
    platform: cl_platform_id,
    device_type: cl_device_type,
) -> Result<Vec<cl_device_id>, cl_int> {
    // Get the number of devices of device_type
    let mut count: cl_uint = 0;
    let mut status =
        unsafe { clGetDeviceIDs(platform, device_type, 0, core::ptr::null_mut(), &mut count) };

    if (CL_SUCCESS != status) && (CL_DEVICE_NOT_FOUND != status) {
        Err(status)
    } else if 0 < count {
        // Get the device ids.
        let len = count as usize;
        let mut ids: Vec<cl_device_id> = Vec::with_capacity(len);
        unsafe {
            status = clGetDeviceIDs(
                platform,
                device_type,
                count,
                ids.as_mut_ptr(),
                core::ptr::null_mut(),
            );
            ids.set_len(len);
        };

        if CL_SUCCESS != status {
            Err(status)
        } else {
            Ok(ids)
        }
    } else {
        Ok(Vec::default())
    }
}

fn get_platform_data(
    platform: cl_platform_id,
    param_name: cl_platform_info,
) -> Result<Vec<u8>, cl_int> {
    fn get_size(object: *mut c_void, param_name: cl_uint) -> Result<usize, cl_int> {
        let mut size: usize = 0;
        let status =
            unsafe { clGetPlatformInfo(object, param_name, 0, core::ptr::null_mut(), &mut size) };
        if CL_SUCCESS != status {
            Err(status)
        } else {
            Ok(size)
        }
    }
    let size = get_size(platform, param_name)?;
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
                clGetPlatformInfo(
                    object,
                    param_name,
                    size,
                    data.as_mut_ptr() as *mut c_void,
                    core::ptr::null_mut(),
                )
            };
            if CL_SUCCESS != status {
                Err(status)
            } else {
                Ok(data)
            }
        } else {
            Ok(Vec::default())
        }
    }
    get_vector(platform, param_name, size)
}

fn get_platform_ids() -> Result<Vec<*mut c_void>, cl_int> {
    // Get the number of platforms
    let mut count: cl_uint = 0;
    let mut status = unsafe { clGetPlatformIDs(0, core::ptr::null_mut(), &mut count) };

    if CL_SUCCESS != status {
        Err(status)
    } else if 0 < count {
        // Get the platform ids.
        let len = count as usize;
        let mut ids: Vec<*mut c_void> = Vec::with_capacity(len);
        unsafe {
            status = clGetPlatformIDs(count, ids.as_mut_ptr(), core::ptr::null_mut());
            ids.set_len(len);
        };

        if CL_SUCCESS != status {
            Err(status)
        } else {
            Ok(ids)
        }
    } else {
        Ok(Vec::new())
    }
}

trait OpenCLDType {
    fn ocl_str(self) -> &'static str;
    fn from_ocl_str(str: &str) -> DType;
}

impl OpenCLDType for DType {
    fn ocl_str(self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::I32 => "int",
        }
    }

    fn from_ocl_str(str: &str) -> DType {
        match str {
            "float" => DType::F32,
            "int" => DType::I32,
            _ => panic!(),
        }
    }
}

pub struct Buffer {
    mem: *mut c_void,
    event: *mut c_void,
}

pub struct Program {
    name: String,
    program: *mut c_void,
    global_work_size: Box<[usize]>,
    local_work_size: Box<[usize]>,
    res_byte_size: usize,
}

impl Program {
    pub fn compile(
        source: &str,
        context: *mut c_void,
        devices: &BTreeSet<*mut c_void>,
        global_work_size: &[usize],
        local_work_size: &[usize],
        res_byte_size: usize,
        reduce: bool,
    ) -> Result<Self, ZyxError> {
        let name = f!(
            "{}__{}__{}",
            if reduce { "r" } else { "e" },
            global_work_size
                .iter()
                .map(|x| f!("{x}"))
                .collect::<Vec<_>>()
                .join("_"),
            local_work_size
                .iter()
                .map(|x| f!("{x}"))
                .collect::<Vec<_>>()
                .join("_"),
        );
        let source = f!("__kernel void {name}{source}");
        #[cfg(feature = "debug1")]
        std::println!("{source}");
        let sources: &[&str] = &[&source];
        let mut err = CL_SUCCESS;
        let program = unsafe {
            clCreateProgramWithSource(
                context,
                1,
                sources.as_ptr().cast(),
                &[source.len()] as *const usize,
                &mut err,
            )
        };
        if err != CL_SUCCESS {
            return Err(ZyxError::BackendError(match err {
                -34 => "Unable to compile program. ERR -34: CL_INVALID_CONTEXT",
                -30 => "Unable to compile program. ERR -30: CL_INVALID_VALUE",
                -5 => "Unable to compile program. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to compile program. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to compile program. UNKNOWN ERROR",
            }));
        };
        let devices = devices.iter().copied().collect::<Vec<*mut c_void>>();
        let err = unsafe {
            clBuildProgram(
                program,
                devices.len() as cl_uint,
                devices.as_ptr().cast(),
                core::ffi::CStr::from_bytes_with_nul(b"-cl-fast-relaxed-math\0")
                    .unwrap()
                    .as_ptr()
                    .cast(),
                None,
                core::ptr::null_mut(),
            )
        };
        if err != CL_SUCCESS {
            return Err(ZyxError::CompileError(Box::new(f!(
                "{err}\n{}",
                core::str::from_utf8(
                    &get_program_build_data(program, devices[0], CL_PROGRAM_BUILD_LOG).map_err(
                        |err| {
                            ZyxError::BackendError(match err {
                -33 => "Unable to get info about failed compilation. ERR -33: CL_INVALID_DEVICE",
                -30 => "Unable to get info about failed compilation. ERR -30: CL_INVALID_VALUE",
                -44 => "Unable to get info about failed compilation. ERR -44: CL_INVALID_PROGRAM",
                -5 => "Unable to get info about failed compilation. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to get info about failed compilation. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to get info about failed compilation. UNKNOWN ERROR",
            })
                        }
                    )?
                )
                .unwrap()
            ))));
        }
        Ok(Self {
            name,
            program,
            global_work_size: global_work_size.iter().copied().collect(),
            local_work_size: local_work_size.iter().copied().collect(),
            res_byte_size,
        })
    }
}

pub(crate) struct Compiler {
    context: *mut c_void,
    devices: BTreeSet<*mut c_void>,
    queues: Box<[*mut c_void]>,
    queue_id: usize,
}

impl Compiler {
    pub(crate) fn new() -> Result<Self, ZyxError> {
        let platform_ids = get_platform_ids().map_err(|err| {
            ZyxError::BackendError(match err {
                -30 => "Unable to get OpenCL platform ids. ERR -30: CL_INVALID_VALUE",
                -6 => "Unable to get OpenCL platform ids. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to get OpenCL platform ids. UNKNOWN ERROR",
            })
        })?;
        let Some(platform) = platform_ids.get(0) else {
            return Err(ZyxError::BackendError(
                "There are no available OpenCL platforms.",
            ));
        };
        let platform = *platform;
        #[cfg(feature = "debug1")]
        std::println!(
            "Using OpenCL platform: {}",
            String::from_utf8(
                get_platform_data(platform, CL_PLATFORM_NAME).map_err(|err| {
                    ZyxError::BackendError(match err {
                        -32 => "Unable to get OpenCL platform name. ERR -32: CL_INVALID_PLATFORM",
                        -30 => "Unable to get OpenCL platform name. ERR -30: CL_INVALID_VALUE",
                        -6 => "Unable to get OpenCL platform name. ERR -6: CL_OUT_OF_HOST_MEMORY",
                        _ => "Unable to get OpenCL platform name. UNKNOWN ERROR",
                    })
                })?
            )
            .unwrap()
        );
        let device_ids = get_device_ids(platform, CL_DEVICE_TYPE_ALL).map_err(|err| {
            ZyxError::BackendError(match err {
                -32 => "Unable to get OpenCL device ids. ERR -32: CL_INVALID_PLATFORM",
                -31 => "Unable to get OpenCL device ids. ERR -31: CL_INVALID_DEVICE_TYPE",
                -30 => "Unable to get OpenCL device ids. ERR -30: CL_INVALID_VALUE",
                -1 => "Unable to get OpenCL device ids. ERR -1: CL_DEVICE_NOT_FOUND",
                -5 => "Unable to get OpenCL device ids. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to get OpenCL device ids. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to get OpenCL device ids. UNKNOWN ERROR",
            })
        })?;
        #[cfg(feature = "debug1")]
        std::println!("Using devices:");
        #[cfg(feature = "debug1")]
        for dev in &device_ids {
            std::println!(
                "{}",
                String::from_utf8(get_device_data(*dev, CL_DEVICE_NAME).map_err(|err| {
                    ZyxError::BackendError(match err {
                        -33 => "Unable to get OpenCL device name. ERR -33: CL_INVALID_DEVICE",
                        -30 => "Unable to get OpenCL device name. ERR -30: CL_INVALID_VALUE",
                        -5 => "Unable to get OpenCL device name. ERR -5: CL_OUT_OF_RESOURCES",
                        -6 => "Unable to get OpenCL device name. ERR -6: CL_OUT_OF_HOST_MEMORY",
                        _ => "Unable to get OpenCL device name. UNKNOWN ERROR",
                    })
                })?)
                .unwrap()
            );
        }
        let mut err = CL_SUCCESS;
        let context = unsafe {
            clCreateContext(
                core::ptr::null(),
                device_ids.len() as cl_uint,
                device_ids.as_ptr(),
                None,
                core::ptr::null_mut(),
                &mut err,
            )
        };
        if err != CL_SUCCESS {
            return Err(ZyxError::BackendError(match err {
                -32 => "Unable to create OpenCL context. ERR -32: CL_INVALID_PLATFORM",
                -64 => "Unable to create OpenCL context name. ERR -64: CL_INVALID_PROPERTY",
                -30 => "Unable to crate OpenCL context name. ERR -30: CL_INVALID_VALUE",
                -33 => "Unable to create OpenCL context name. ERR -33: CL_INVALID_DEVICE",
                -59 => "Unable to create OpenCL context name. ERR -59: CL_INVALID_OPERATION",
                -2 => "Unable to create OpenCL context. ERR -32: CL_DEVICE_NOT_AVAILABLE",
                -5 => "Unable to create OpenCL context. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to create OpenCL context. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to create OpenCL context. UNKNOWN ERROR",
            }));
        }
        // This makes our code asynchronous. Creating graph would actually make us 2 times slower (can be optimized),
        // if we couldn't execute kernels asynchronously. We don't need this to be huge. 2 seems to
        // be plenty. And lower values also lower memory usage.
        let queues_per_device: u32 = 8; //device_ids.iter().map(|dev| get_device_info(*dev, CL_DEVICE_MAX_ON_DEVICE_QUEUES)?.into()).min()?;
        #[cfg(feature = "debug1")]
        std::println!("Using {queues_per_device} queues per device.");
        let (queues, errs): (Vec<*mut c_void>, Vec<cl_int>) = (0..queues_per_device)
            .flat_map(|_| {
                device_ids.iter().map(move |dev| {
                    let queue = unsafe { clCreateCommandQueue(context, *dev, 0, &mut err) };
                    (queue, err)
                })
            })
            .unzip();
        for err in errs {
            if err != CL_SUCCESS {
                return Err(ZyxError::BackendError(match err {
                    -34 => "Unable to create command queue. ERR -34: CL_INVALID_CONTEXT",
                    -33 => "Unable to create command queue. ERR -33: CL_INVALID_DEVICE",
                    -30 => "Unable to create command queue. ERR -30: CL_INVALID_VALUE",
                    -35 => "Unable to create command queue. ERR -35: CL_INVALID_QUEUE_PROPERTIES",
                    -5 => "Unable to create command queue. ERR -5: CL_OUT_OF_RESOURCES",
                    -6 => "Unable to create command queue. ERR -6: CL_OUT_OF_HOST_MEMORY",
                    _ => "Unable to create command queue. UNKNOWN ERROR",
                }));
            }
        }
        let mut devices = BTreeSet::new();
        for dev in device_ids {
            devices.insert(dev);
        }
        Ok(Self {
            context,
            devices,
            queues: queues.into_boxed_slice(),
            queue_id: 0,
        })
    }

    fn queue(&mut self) -> *mut c_void {
        let res = self.queues[self.queue_id];
        self.queue_id = (self.queue_id + 1) % self.queues.len();
        res
    }
}

impl zyx_core::compiler::Compiler for Compiler {
    type Buffer = Buffer;

    type Program = Program;

    fn store<T>(&mut self, iter: impl Iterator<Item = T>) -> Result<Self::Buffer, ZyxError> {
        // TODO we can do buffered load, with buffer of say 1 MB size in RAM and offset write buffer
        let data: Vec<T> = iter.collect();
        let size = data.len() * core::mem::size_of::<T>();
        let mut err = CL_SUCCESS;
        let mem = unsafe {
            clCreateBuffer(
                self.context,
                CL_MEM_READ_ONLY,
                size,
                core::ptr::null_mut(),
                &mut err,
            )
        };
        if err != CL_SUCCESS {
            return Err(ZyxError::BackendError(match err {
                -34 => "Unable to create buffer. ERR -34: CL_INVALID_CONTEXT",
                -64 => "Unable to create buffer. ERR -64: CL_INVALID_PROPERTY",
                -30 => "Unable to create buffer. ERR -30: CL_INVALID_VALUE",
                -61 => "Unable to create buffer. ERR -61: CL_INVALID_BUFFER_SIZE",
                -4 => "Unable to create buffer. ERR -4: CL_MEM_OBJECT_ALLOCATION_FAILURE",
                -5 => "Unable to create buffer. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to create buffer. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to create buffer. UNKNOWN ERROR",
            }));
        }
        let mut event: *mut c_void = core::ptr::null_mut();
        let err = unsafe {
            clEnqueueWriteBuffer(
                self.queue(),
                mem,
                CL_NON_BLOCKING,
                0,
                size,
                data.as_ptr().cast(),
                0,
                core::ptr::null(),
                &mut event,
            )
        };
        if err != CL_SUCCESS {
            return Err(ZyxError::BackendError(match err {
                -36 => "Unable to write buffer. ERR -36: CL_INVALID_COMMAND_QUEUE",
                -34 => "Unable to write buffer. ERR -34: CL_INVALID_CONTEXT",
                -38 => "Unable to write buffer. ERR -38: CL_INVALID_MEM_OBJECT",
                -30 => "Unable to write buffer. ERR -30: CL_INVALID_VALUE",
                -57 => "Unable to write buffer. ERR -57: CL_INVALID_EVENT_WAIT_LIST",
                -13 => "Unable to write buffer. ERR -13: CL_MISALIGNED_SUB_BUFFER_OFFSET",
                -14 => {
                    "Unable to write buffer. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"
                }
                -4 => "Unable to write buffer. ERR -4: CL_MEM_OBJECT_ALLOCATION_FAILURE",
                -59 => "Unable to write buffer. ERR -59: CL_INVALID_OPERATION",
                -5 => "Unable to write buffer. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to write buffer. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to write buffer. UNKNOWN ERROR",
            }));
        }
        let err = unsafe { clWaitForEvents(1, (&[event]).as_ptr().cast()) };
        if err != CL_SUCCESS {
            return Err(ZyxError::BackendError(match err {
                -30 => "Unable to finish buffer write event. ERR -30: CL_INVALID_VALUE",
                -34 => "Unable to finish buffer write event. ERR -34: CL_INVALID_CONTEXT",
                -58 => "Unable to finish buffer write event. ERR -58: CL_INVALID_EVENT",
                -14 => "Unable to finish buffer write event. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
                -5 => "Unable to finish buffer write event. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to finish buffer write event. ERR -6: CL_OUT_OF_MEMORY",
                _ => "Unable to finish buffer write event. UNKNOWN ERROR",
            }));
        }
        Ok(Self::Buffer { mem, event })
    }

    fn load<T: Scalar>(&mut self, buffer: &Self::Buffer, numel: usize) -> Result<Vec<T>, ZyxError> {
        let mut data: Vec<T> = Vec::with_capacity(numel);
        let mut event: *mut c_void = core::ptr::null_mut();
        let err = unsafe {
            clEnqueueReadBuffer(
                self.queue(),
                buffer.mem,
                CL_NON_BLOCKING,
                0,
                numel * T::byte_size(),
                data.as_mut_ptr().cast(),
                1,
                (&[buffer.event]).as_ptr().cast(),
                &mut event,
            )
        };
        if err != CL_SUCCESS {
            return Err(ZyxError::BackendError(match err {
                -36 => "Unable to read buffer. ERR -36: CL_INVALID_COMMAND_QUEUE",
                -34 => "Unable to read buffer. ERR -34: CL_INVALID_CONTEXT",
                -38 => "Unable to read buffer. ERR -38: CL_INVALID_MEM_OBJECT",
                -30 => "Unable to read buffer. ERR -30: CL_INVALID_VALUE",
                -57 => "Unable to read buffer. ERR -57: CL_INVALID_EVENT_WAIT_LIST",
                -13 => "Unable to read buffer. ERR -13: CL_MISALIGNED_SUB_BUFFER_OFFSET",
                -14 => {
                    "Unable to read buffer. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"
                }
                -4 => "Unable to read buffer. ERR -4: CL_MEM_OBJECT_ALLOCATION_FAILURE",
                -59 => "Unable to read buffer. ERR -59: CL_INVALID_OPERATION",
                -5 => "Unable to read buffer. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to read buffer. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to read buffer. UNKNOWN ERROR",
            }));
        }
        let err = unsafe { clWaitForEvents(1, (&[event]).as_ptr().cast()) };
        if err != CL_SUCCESS {
            return Err(ZyxError::BackendError(match err {
                -30 => "Unable to finish buffer read event. ERR -30: CL_INVALID_VALUE",
                -34 => "Unable to finish buffer read event. ERR -34: CL_INVALID_CONTEXT",
                -58 => "Unable to finish buffer read event. ERR -58: CL_INVALID_EVENT",
                -14 => "Unable to finish buffer read event. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
                -5 => "Unable to finish buffer read event. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to finish buffer read event. ERR -6: CL_OUT_OF_MEMORY",
                _ => "Unable to finish buffer read event. UNKNOWN ERROR",
            }));
        }
        // We are now done reading, so the vec is initialized
        unsafe { data.set_len(numel) }
        Ok(data)
    }

    fn drop_buffer(&mut self, buffer: &mut Self::Buffer) -> Result<(), ZyxError> {
        let err = unsafe { clReleaseMemObject(buffer.mem) };
        if err != CL_SUCCESS {
            return Err(ZyxError::BackendError(match err {
                -38 => "Unable to release buffer. ERR -38: CL_INVALID_MEM_OBJECT",
                -5 => "Unable to release buffer. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to release buffer. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to release buffer. UNKNOWN ERROR",
            }));
        }
        let err = unsafe { clReleaseEvent(buffer.event) };
        if err != CL_SUCCESS {
            return Err(ZyxError::BackendError(match err {
                -58 => "Unable to release event. ERR -58: CL_INVALID_EVENT",
                -5 => "Unable to release event. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to release event. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to release event. UNKNOWN ERROR",
            }));
        }
        Ok(())
    }

    fn drop_program(&mut self, program: &mut Self::Program) -> Result<(), ZyxError> {
        let err = unsafe { clReleaseProgram(program.program) };
        if err != CL_SUCCESS {
            return Err(ZyxError::BackendError(match err {
                -5 => "Unable to release program. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to release program. ERR -6: CL_OUT_OF_HOST_MEMORY",
                -44 => "Unable to release program. ERR -44: CL_INVALID_PROGRAM",
                _ => "Unable to release program. UNKNOWN ERROR",
            }));
        }
        Ok(())
    }

    fn launch(
        &mut self,
        program: &Self::Program,
        args: &[&Self::Buffer],
    ) -> Result<Self::Buffer, ZyxError> {
        let program_name = &CString::new(program.name.clone()).unwrap();
        let mut err = CL_SUCCESS;
        let kernel =
            unsafe { clCreateKernel(program.program, program_name.as_ptr().cast(), &mut err) };
        if err != CL_SUCCESS {
            return Err(ZyxError::BackendError(match err {
                -44 => "Unable to create kernel. ERR -: CL_INVALID_PROGRAM",
                -45 => "Unable to create kernel. ERR -: CL_INVALID_PROGRAM_EXECUTABLE",
                -46 => "Unable to create kernel. ERR -: CL_INVALID_KERNEL_NAME",
                -47 => "Unable to create kernel. ERR -: CL_INVALID_KERNEL_DEFINITION",
                -30 => "Unable to create kernel. ERR -: CL_INVALID_VALUE",
                -5 => "Unable to create kernel. ERR -: CL_OUT_OF_RESOURCES",
                -6 => "Unable to create kernel. ERR -: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to create kernel. UNKNOWN ERROR",
            }));
        }
        let kernel_arg_err_handler = |err| {
            ZyxError::BackendError(match err {
                -48 => "Unable to set kernel arg. ERR -48: CL_INVALID_KERNEL",
                -49 => "Unable to set kernel arg. ERR -49: CL_INVALID_ARG_INDEX",
                -50 => "Unable to set kernel arg. ERR -50: CL_INVALID_ARG_VALUE",
                -38 => "Unable to set kernel arg. ERR -38: CL_INVALID_MEM_OBJECT",
                -41 => "Unable to set kernel arg. ERR -41: CL_INVALID_SAMPLER",
                -33 => "Unable to set kernel arg. ERR -33: CL_INVALID_DEVICE_QUEUE",
                -51 => "Unable to set kernel arg. ERR -51: CL_INVALID_ARG_SIZE",
                -72 => "Unable to set kernel arg. ERR -: CL_MAX_SIZE_RESTRICTION_EXCEEDED",
                -5 => "Unable to set kernel arg. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to set kernel arg. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to set kernel arg. UNKNOWN ERROR",
            })
        };
        let mut events = Vec::new();
        let mut i = 0;
        for arg in args {
            let (buffer, event) = (arg.mem, arg.event);
            events.push(event);
            //std::println!("Arg: {:?}", self.load::<f32>(arg, 6));
            // This is POINTER MAGIC. Be careful.
            let ptr: *const _ = &buffer;
            err = unsafe {
                clSetKernelArg(kernel, i, core::mem::size_of::<*mut c_void>(), ptr.cast())
            };
            if err != CL_SUCCESS {
                return Err(kernel_arg_err_handler(err));
            }
            i += 1;
        }
        let mem = unsafe {
            clCreateBuffer(
                self.context,
                CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
                program.res_byte_size,
                core::ptr::null_mut(),
                &mut err,
            )
        };
        if err != CL_SUCCESS {
            return Err(ZyxError::BackendError(match err {
                -34 => "Unable to create kernel output buffer. ERR -34: CL_INVALID_CONTEXT",
                -64 => "Unable to create kernel output buffer. ERR -64: CL_INVALID_PROPERTY",
                -30 => "Unable to create kernel output buffer. ERR -30: CL_INVALID_VALUE",
                -61 => "Unable to create kernel output buffer. ERR -61: CL_INVALID_BUFFER_SIZE",
                -4 => "Unable to create kernel output buffer. ERR -4: CL_MEM_OBJECT_ALLOCATION_FAILURE",
                -5 => "Unable to create kernel output buffer. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to create kernel output buffer. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to create kernel output buffer. UNKNOWN ERROR",
            }));
        }
        let ptr: *const _ = &mem;
        let err =
            unsafe { clSetKernelArg(kernel, i, core::mem::size_of::<*mut c_void>(), ptr.cast()) };
        if err != CL_SUCCESS {
            return Err(kernel_arg_err_handler(err));
        }
        let mut event: *mut c_void = core::ptr::null_mut();
        #[cfg(feature = "debug1")]
        let begin = std::time::Instant::now();
        let err = unsafe {
            clEnqueueNDRangeKernel(
                self.queue(),
                kernel,
                u32::try_from(program.global_work_size.len()).unwrap(),
                core::ptr::null(),
                program.global_work_size.as_ptr(),
                program.local_work_size.as_ptr(),
                u32::try_from(events.len()).unwrap(),
                if events.is_empty() {
                    core::ptr::null()
                } else {
                    events.as_ptr()
                },
                &mut event,
            )
        };
        if err != CL_SUCCESS {
            return Err(ZyxError::BackendError(match err {
                -45 => "Unable to enqueue kernel. ERR -45: CL_INVALID_PROGRAM_EXECUTABLE",
                -36 => "Unable to enqueue kernel. ERR -36: CL_INVALID_COMMAND_QUEUE",
                -48 => "Unable to enqueue kernel. ERR -48: CL_INVALID_KERNEL",
                -34 => "Unable to enqueue kernel. ERR -34: CL_INVALID_CONTEXT",
                -52 => "Unable to enqueue kernel. ERR -52: CL_INVALID_KERNEL_ARGS",
                -53 => "Unable to enqueue kernel. ERR -53: CL_INVALID_WORK_DIMENSION",
                -63 => "Unable to enqueue kernel. ERR -63: CL_INVALID_GLOBAL_WORK_SIZE",
                -56 => "Unable to enqueue kernel. ERR -56: CL_INVALID_GLOBAL_OFFSET",
                -54 => "Unable to enqueue kernel. ERR -54: CL_INVALID_WORK_GROUP_SIZE",
                -55 => "Unable to enqueue kernel. ERR -55: CL_INVALID_WORK_ITEM_SIZE",
                -13 => "Unable to enqueue kernel. ERR -13: CL_MISALIGNED_SUB_BUFFER_OFFSET",
                -40 => "Unable to enqueue kernel. ERR -40: CL_INVALID_IMAGE_SIZE",
                -10 => "Unable to enqueue kernel. ERR -10: CL_IMAGE_FORMAT_NOT_SUPPORTED",
                -5 => "Unable to enqueue kernel. ERR -5: CL_OUT_OF_RESOURCES",
                -4 => "Unable to enqueue kernel. ERR -4: CL_MEM_OBJECT_ALLOCATION_FAILURE",
                -57 => "Unable to enqueue kernel. ERR -57: CL_INVALID_EVENT_WAIT_LIST",
                -59 => "Unable to enqueue kernel. ERR -59: CL_INVALID_OPERATION",
                -6 => "Unable to enqueue kernel. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to enqueue kernel. UNKNOWN ERROR",
            }));
        }
        #[cfg(feature = "debug1")]
        {
            let err = unsafe { clWaitForEvents(1, (&[event]).as_ptr().cast()) };
            if err != CL_SUCCESS {
                return Err(ZyxError::BackendError( match err {
                    - 30 => "Unable to finish kernel execution event. ERR -30: CL_INVALID_VALUE",
                    - 34 => "Unable to finish kernel execution event. ERR -34: CL_INVALID_CONTEXT",
                    - 58 => "Unable to finish kernel execution event. ERR -58: CL_INVALID_EVENT",
                    - 14 => "Unable to finish kernel execution event. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
                    - 5 => "Unable to finish kernel execution event. ERR -5: CL_OUT_OF_RESOURCES",
                    - 6 => "Unable to finish kernel execution event. ERR -6: CL_OUT_OF_MEMORY",
                    _ => "Unable to finish kernel execution event. UNKNOWN ERROR",
                }));
            }
            let elapsed = begin.elapsed().as_nanos();
            let elapsed_millis = elapsed as f64 / 1000000.;
            std::println!(
                "Kernel execution took {elapsed_millis:.3}ms, that is {} GFLOPS",
                (1024u128 * 1024 * 1024 * 2) as f64 / elapsed as f64
            );
            //std::println!("Output: {:?}", self.load::<f32>(&Buffer { mem, event }, 6));
        }
        Ok(Buffer { mem, event })
    }

    fn compile(&mut self, ast: &AST) -> Result<Self::Program, ZyxError> {
        if matches!(ast.reduce(), ROp::None) {
            let (source, gws, lws, rbs) = compile_e_kernel(ast);
            Program::compile(&source, self.context, &self.devices, &gws, &lws, rbs, false)
        } else {
            let (source, gws, lws, rbs) = compile_r_kernel(ast);
            Program::compile(&source, self.context, &self.devices, &gws, &lws, rbs, true)
        }
    }
}

/// Elementwise kernel
fn compile_e_kernel(ast: &AST) -> (String, Vec<usize>, Vec<usize>, usize) {
    //std::println!("\nCompiling ast: {ast:#?}");
    // TODO get this to work with different max local work sizes
    let n = ast.args()[0].0.numel();
    let (tile_height, tile_width) = match n / 8 {
        1..=7 => (1, 8),
        8..=15 => (8, 8),
        16..=31 => (8, 16),
        _ => (16, 16),
    };
    let global_work_size = alloc::vec![n / tile_height, tile_height];
    let local_work_size = alloc::vec![tile_height, tile_width];
    let res_byte_size: usize = global_work_size.iter().product();
    let mut source = f!("(\n  ");
    let mut endl = f!(",\n  ");

    let mut res_id = 0;
    for arg in ast.args() {
        source = f!(
            "{source}__global const {}* data{res_id}{endl}",
            arg.1.ocl_str()
        );
        res_id += 1;
    }
    source = f!("{source}__global RES_DTYPE* data{res_id}{endl}");
    source.pop();
    source.pop();
    source.pop();
    source.pop();
    source = f!("{source}\n) {{\n  ");

    endl = f!(";\n  ");
    //source = f!("{source}int idx0 = get_global_id(0){endl}");
    source = f!("{source}int gidx0 = get_group_id(0){endl}");
    source = f!("{source}int gidx1 = get_group_id(1){endl}");
    source = f!("{source}int lidx0 = get_local_id(0){endl}");
    source = f!("{source}int lidx1 = get_local_id(1){endl}");
    source = f!(
        "{source}int idx0 = (gidx0*{tile_height} + lidx0)*{} + gidx1*{tile_width} + lidx1{endl}",
        global_work_size[1]
    );
    //source = f!("{source}int idx0 = gidx0*{tile_width} + lidx0{endl}");
    let mut dtype = DType::F32.ocl_str();
    let mut nid = 0;
    for op in ast.ops().iter() {
        let res = match op {
            // TODO check if this should be tile or data
            // TODO add correct index
            Op::Leaf(x) => {
                let (view, t) = &ast.args()[*x];
                dtype = t.ocl_str();
                f!("{dtype} var{nid} = {}", view.cidx(&f!("data{x}")))
            }
            Op::UniformF32(..) => {
                todo!()
            }
            Op::CastF32(x) => {
                dtype = DType::F32.ocl_str();
                f!("{dtype} var{nid} = ({dtype})var{x}")
            }
            Op::CastI32(x) => {
                dtype = DType::I32.ocl_str();
                f!("{dtype} var{nid} = ({dtype})var{x}")
            }
            Op::Neg(x) => f!("{dtype} var{nid} = -var{x}"),
            Op::ReLU(x) => f!("{dtype} var{nid} = (var{x} > 0)*var{x}"),
            Op::Sin(x) => f!("{dtype} var{nid} = sin(var{x})"),
            Op::Cos(x) => f!("{dtype} var{nid} = cos(var{x})"),
            Op::Ln(x) => f!("{dtype} var{nid} = ln(var{x})"),
            Op::Exp(x) => f!("{dtype} var{nid} = exp(var{x})"),
            Op::Tanh(x) => f!("{dtype} var{nid} = tanh(var{x})"),
            Op::Sqrt(x) => f!("{dtype} var{nid} = sqrt(var{x})"),
            Op::Add(x, y) => f!("{dtype} var{nid} = var{x} + var{y}"),
            Op::Sub(x, y) => f!("{dtype} var{nid} = var{x} - var{y}"),
            Op::Mul(x, y) => f!("{dtype} var{nid} = var{x} * var{y}"),
            Op::Div(x, y) => f!("{dtype} var{nid} = var{x} / var{y}"),
            Op::Pow(x, y) => f!("{dtype} var{nid} = ({dtype})pow((float)var{x}, (float)var{y})"),
            Op::Cmplt(x, y) => f!("{dtype} var{nid} = ({dtype})(var{x} < var{y})"),
        };
        source = f!("{source}{res}{endl}");
        nid += 1;
    }
    source = source.replace("RES_DTYPE", &f!("{dtype}"));
    source = f!("{source}data{res_id}[idx0] = var{}{endl}", nid - 1);
    source.pop();
    source.pop();
    source = f!("{source}}}");
    (
        source,
        global_work_size,
        local_work_size,
        res_byte_size * DType::from_ocl_str(dtype).byte_size(),
    )
}

/// Reduce kernel
fn compile_r_kernel(_ast: &AST) -> (String, Vec<usize>, Vec<usize>, usize) {
    todo!()
}

#[test]
fn exp_test() -> Result<(), ZyxError> {
    let dev = crate::device()?;
    let x = dev.randn([8, 8], DType::I32).cast(DType::F32);
    let y = x.exp() + &x;
    //let x_vec: Vec<f32> = x.to_vec()?;
    let _y_vec: Vec<f32> = y.to_vec()?;
    //panic!("{y_vec:?}");
    Ok(())
}

#[test]
fn sum_test() -> Result<(), ZyxError> {
    let dev = crate::device()?;
    let x = dev.randn([1024, 1024], DType::F32);
    let y = x.sum(-1);
    let y_vec: Vec<f32> = y.to_vec()?;
    panic!("{y_vec:?}");
    //Ok(())
}
