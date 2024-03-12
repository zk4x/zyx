use alloc::{
    boxed::Box, collections::BTreeSet, ffi::CString, format as f, string::String, vec::Vec,
};
use core::{ffi::c_void, ptr};
use opencl_sys::{
    clBuildProgram, clCreateBuffer, clCreateCommandQueue, clCreateContext, clCreateKernel,
    clCreateProgramWithSource, clEnqueueNDRangeKernel, clEnqueueReadBuffer, clEnqueueWriteBuffer,
    clGetDeviceIDs, clGetPlatformIDs, clGetProgramBuildInfo, clReleaseEvent, clReleaseMemObject,
    clReleaseProgram, clSetKernelArg, clWaitForEvents, cl_device_id, cl_device_type, cl_int,
    cl_platform_id, cl_program_info, cl_uint, CL_DEVICE_NOT_FOUND, CL_DEVICE_TYPE_ALL,
    CL_MEM_HOST_READ_ONLY, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_NON_BLOCKING,
    CL_PROGRAM_BUILD_LOG, CL_SUCCESS,
};
use zyx_compiler::Op;
use zyx_core::view::Index;
use zyx_core::{dtype::DType, error::ZyxError, scalar::Scalar};

//const VECTOR_SYMBOLS: [&str; 16] = [".s0", ".s1", ".s2", ".s3", ".s4", ".s5", ".s6", ".s7", ".s8", ".s9", ".sa", ".sb", ".sc", ".sd", ".se", ".sf"];

fn cl_wait_for_events(events: &[*mut c_void]) -> Result<(), ZyxError> {
    let err = unsafe { clWaitForEvents(1, events.as_ptr().cast()) };
    if err != CL_SUCCESS {
        Err(ZyxError::BackendError(match err {
            -30 => "Unable to finish buffer read event. ERR -30: CL_INVALID_VALUE",
            -34 => "Unable to finish buffer read event. ERR -34: CL_INVALID_CONTEXT",
            -58 => "Unable to finish buffer read event. ERR -58: CL_INVALID_EVENT",
            -14 => "Unable to finish buffer read event. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
            -5 => "Unable to finish buffer read event. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to finish buffer read event. ERR -6: CL_OUT_OF_MEMORY",
            _ => "Unable to finish buffer read event. UNKNOWN ERROR",
        }))
    } else {
        Ok(())
    }
}

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
            clGetProgramBuildInfo(object, idx, param_name, 0, ptr::null_mut(), &mut size)
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
                    ptr::null_mut(),
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

#[cfg(feature = "debug1")]
pub fn get_device_data(
    device: cl_device_id,
    param_name: opencl_sys::cl_device_info,
) -> Result<Vec<u8>, cl_int> {
    fn get_size(object: *mut c_void, param_name: cl_uint) -> Result<usize, cl_int> {
        let mut size: usize = 0;
        let status = unsafe {
            opencl_sys::clGetDeviceInfo(object, param_name, 0, ptr::null_mut(), &mut size)
        };
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
                opencl_sys::clGetDeviceInfo(
                    object,
                    param_name,
                    size,
                    data.as_mut_ptr() as *mut c_void,
                    ptr::null_mut(),
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
        unsafe { clGetDeviceIDs(platform, device_type, 0, ptr::null_mut(), &mut count) };

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
                ptr::null_mut(),
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

#[cfg(feature = "debug1")]
fn get_platform_data(
    platform: cl_platform_id,
    param_name: opencl_sys::cl_platform_info,
) -> Result<Vec<u8>, cl_int> {
    fn get_size(object: *mut c_void, param_name: cl_uint) -> Result<usize, cl_int> {
        let mut size: usize = 0;
        let status = unsafe {
            opencl_sys::clGetPlatformInfo(object, param_name, 0, ptr::null_mut(), &mut size)
        };
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
                opencl_sys::clGetPlatformInfo(
                    object,
                    param_name,
                    size,
                    data.as_mut_ptr() as *mut c_void,
                    ptr::null_mut(),
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

trait OpenCLDType {
    fn ocl_str(&self) -> &'static str;
    fn from_ocl_str(str: &str) -> DType;
}

impl OpenCLDType for DType {
    fn ocl_str(&self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::F64 => "double",
            DType::I32 => "int",
        }
    }

    fn from_ocl_str(str: &str) -> DType {
        match str {
            "float" => DType::F32,
            "double" => DType::F64,
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
    ) -> Result<Self, ZyxError> {
        let name = f!(
            "k__{}__{}",
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
        let mut pragma = f!("");
        if source.contains("double") {
            pragma += &"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        }
        let source = f!("{pragma}__kernel void {name}{source}");
        #[cfg(feature = "debug1")]
        std::println!("{source}");
        let sources: &[&str] = &[source.as_str()];
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
                ptr::null_mut(),
            )
        };
        if err != CL_SUCCESS {
            panic!("{}", String::from_utf8_lossy(&get_program_build_data(program, devices[0], CL_PROGRAM_BUILD_LOG).map_err(
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
            )?));
            /*return Err(ZyxError::CompileError(Box::new(f!(
                "{err}\n{}",
                String::from_utf8_lossy(
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
                ).into_owned()
            ))));*/
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
    pub(crate) fn new(platform_id: usize, queues_per_device: usize) -> Result<Self, ZyxError> {
        let platform_ids = {
            // Get the number of platforms
            let mut count: cl_uint = 0;
            let mut err = unsafe { clGetPlatformIDs(0, ptr::null_mut(), &mut count) };
            if err != CL_SUCCESS {
                return Err(ZyxError::BackendError(match err {
                    -30 => "Unable to get OpenCL platform ids. ERR -30: CL_INVALID_VALUE",
                    -6 => "Unable to get OpenCL platform ids. ERR -6: CL_OUT_OF_HOST_MEMORY",
                    _ => "Unable to get OpenCL platform ids. UNKNOWN ERROR",
                }));
            } else if count > 0 {
                // Get the platform ids.
                let len = count as usize;
                let mut ids: Vec<*mut c_void> = Vec::with_capacity(len);
                unsafe {
                    err = clGetPlatformIDs(count, ids.as_mut_ptr(), ptr::null_mut());
                    ids.set_len(len);
                };
                if CL_SUCCESS != err {
                    return Err(ZyxError::BackendError(match err {
                        -30 => "Unable to get OpenCL platform ids. ERR -30: CL_INVALID_VALUE",
                        -6 => "Unable to get OpenCL platform ids. ERR -6: CL_OUT_OF_HOST_MEMORY",
                        _ => "Unable to get OpenCL platform ids. UNKNOWN ERROR",
                    }));
                }
                ids
            } else {
                Vec::new()
            }
        };
        let Some(platform) = platform_ids.get(platform_id) else {
            return Err(ZyxError::BackendError(
                "There are no available OpenCL platforms.",
            ));
        };
        let platform = *platform;
        #[cfg(feature = "debug1")]
        std::println!(
            "Using OpenCL platform: {}",
            String::from_utf8(
                get_platform_data(platform, opencl_sys::CL_PLATFORM_NAME).map_err(|err| {
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
                String::from_utf8(get_device_data(*dev, opencl_sys::CL_DEVICE_NAME).map_err(
                    |err| {
                        ZyxError::BackendError(match err {
                            -33 => "Unable to get OpenCL device name. ERR -33: CL_INVALID_DEVICE",
                            -30 => "Unable to get OpenCL device name. ERR -30: CL_INVALID_VALUE",
                            -5 => "Unable to get OpenCL device name. ERR -5: CL_OUT_OF_RESOURCES",
                            -6 => "Unable to get OpenCL device name. ERR -6: CL_OUT_OF_HOST_MEMORY",
                            _ => "Unable to get OpenCL device name. UNKNOWN ERROR",
                        })
                    }
                )?)
                .unwrap()
            );
        }
        let mut err = CL_SUCCESS;
        let context = unsafe {
            clCreateContext(
                ptr::null(),
                device_ids.len() as cl_uint,
                device_ids.as_ptr(),
                None,
                ptr::null_mut(),
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
        //device_ids.iter().map(|dev| get_device_info(*dev, CL_DEVICE_MAX_ON_DEVICE_QUEUES)?.into()).min()?;
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

impl zyx_compiler::Compiler for Compiler {
    type Buffer = Buffer;

    type Program = Program;

    fn store<T>(&mut self, iter: impl IntoIterator<Item = T>) -> Result<Self::Buffer, ZyxError> {
        //std::println!("Storing");
        // TODO we can do buffered load, with buffer of say 1 MB size in RAM and offset write buffer
        let data: Vec<T> = iter.into_iter().collect();
        let size = data.len() * core::mem::size_of::<T>();
        let mut err = CL_SUCCESS;
        let mem = unsafe {
            clCreateBuffer(
                self.context,
                CL_MEM_READ_ONLY,
                size,
                ptr::null_mut(),
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
        let mut event: *mut c_void = ptr::null_mut();
        let err = unsafe {
            clEnqueueWriteBuffer(
                self.queue(),
                mem,
                CL_NON_BLOCKING,
                0,
                size,
                data.as_ptr().cast(),
                0,
                ptr::null(),
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
        let mut event: *mut c_void = ptr::null_mut();
        cl_wait_for_events(&[buffer.event])?;
        let err = unsafe {
            clEnqueueReadBuffer(
                self.queue(),
                buffer.mem,
                CL_NON_BLOCKING,
                0,
                numel * T::byte_size(),
                data.as_mut_ptr().cast(),
                0,
                // TODO why does this not work?
                ptr::null_mut(), //events.as_ptr().cast(),
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
        cl_wait_for_events(&[event])?;
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
        flop: usize,
        bytes: usize,
    ) -> Result<Self::Buffer, ZyxError> {
        #[cfg(not(feature = "debug1"))]
        let (_, _) = (flop, bytes);
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
                ptr::null_mut(),
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
        let mut event: *mut c_void = ptr::null_mut();
        #[cfg(feature = "debug1")]
        let begin = std::time::Instant::now();
        let err = unsafe {
            clEnqueueNDRangeKernel(
                self.queue(),
                kernel,
                u32::try_from(program.global_work_size.len()).unwrap(),
                ptr::null(),
                program.global_work_size.as_ptr(),
                program.local_work_size.as_ptr(),
                u32::try_from(events.len()).unwrap(),
                if events.is_empty() {
                    ptr::null()
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
                return Err(ZyxError::BackendError(match err {
                    -30 => "Unable to finish kernel execution event. ERR -30: CL_INVALID_VALUE",
                    -34 => "Unable to finish kernel execution event. ERR -34: CL_INVALID_CONTEXT",
                    -58 => "Unable to finish kernel execution event. ERR -58: CL_INVALID_EVENT",
                    -14 => "Unable to finish kernel execution event. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
                    -5 => "Unable to finish kernel execution event. ERR -5: CL_OUT_OF_RESOURCES",
                    -6 => "Unable to finish kernel execution event. ERR -6: CL_OUT_OF_MEMORY",
                    _ => "Unable to finish kernel execution event. UNKNOWN ERROR",
                }));
            }
            let elapsed_nanos = begin.elapsed().as_nanos();
            let elapsed_millis = elapsed_nanos as f64 / 1000000.;
            std::println!(
                "Kernel took {elapsed_millis:.3}ms for {bytes} B, {flop} FLOP ~ {:.2} GFLOPS, {:.2} GB/s",
                flop as f64 / elapsed_nanos as f64,
                bytes as f64 / elapsed_nanos as f64,
            );
            //std::println!("Output: {:?}", self.load::<f32>(&Buffer { mem, event }, 6));
        }
        Ok(Buffer { mem, event })
    }

    fn compile(&mut self, ir: &zyx_compiler::IR) -> Result<Self::Program, ZyxError> {
        let id_t = "unsigned int";
        let mut source = f!("(\n");

        // Kernel arguments
        for (i, (dtype, read_only)) in ir.kernel_args.iter().enumerate() {
            source += &f!(
                "  __global {}{}* gmem{i},\n",
                if *read_only { "const " } else { "" },
                dtype.ocl_str()
            );
        }
        source.pop();
        source.pop();
        source += &f!("\n) {{\n");

        let mut indent = String::from("  ");

        // Global and local indices
        for (i, (gwd, lwd)) in ir
            .global_work_size
            .iter()
            .zip(ir.local_work_size.iter())
            .enumerate()
        {
            source += &f!(
                "{indent}{id_t} gid{i} = get_group_id({i}); /* {} */\n",
                gwd / lwd
            );
            source += &f!("{indent}{id_t} lid{i} = get_local_id({i}); /* {lwd} */\n");
        }

        for op in &ir.ops {
            match op {
                Op::LoadGlobal { res, arg, index } => match index {
                    Index::Normal(idx) => {
                        source += &f!("{indent}{res} = gmem{arg}[{idx}];\n");
                    }
                    Index::Padded(padding, idx) => {
                        source += &f!("{indent}{res} = {padding} ? gmem{arg}[{idx}] : 0;\n");
                    }
                },
                Op::StoreGlobal { res, index, arg } => match index {
                    Index::Normal(idx) => {
                        source += &f!("{indent}gmem{res}[{idx}] = {arg};\n");
                    }
                    Index::Padded(padding, idx) => {
                        source += &f!("if ({padding}) {{ {indent}gmem{res}[{idx}] = {arg}; }}\n");
                    }
                },
                Op::DeclareVar { dtype, id, len } => {
                    if let Some(len) = len {
                        source += &f!("{indent}{} rmem{id}[{len}];\n", dtype.ocl_str());
                    } else {
                        source += &f!("{indent}{} rmem{id};\n", dtype.ocl_str());
                    }
                }
                Op::InitIndex { id, value } => {
                    source += &f!("{indent}{id_t} idx{id} = {value};\n");
                }
                Op::InitAccumulator {
                    id,
                    dtype,
                    is_sum_reduce,
                    len,
                } => {
                    if let Some(len) = len {
                        source += &f!(
                            "{indent}{} rmem{id}[{len}] = {{ {} }};\n",
                            dtype.ocl_str(),
                            if *is_sum_reduce {
                                dtype.zero_value_str()
                            } else {
                                dtype.min_value_str()
                            }
                        );
                    } else {
                        source += &f!(
                            "{indent}{} rmem{id} = {};\n",
                            dtype.ocl_str(),
                            if *is_sum_reduce {
                                dtype.zero_value_str()
                            } else {
                                dtype.min_value_str()
                            }
                        );
                    }
                }
                Op::Cast {
                    res_dtype: _,
                    res,
                    x,
                } => {
                    source += &f!("{indent}{res} = {x};\n");
                }
                Op::Neg { res, x } => {
                    source += &f!("{indent}{res} = -{x};\n");
                }
                Op::Sin { res, x } => {
                    source += &f!("{indent}{res} = sin({x});\n");
                }
                Op::Cos { res, x } => {
                    source += &f!("{indent}{res} = cos({x});\n");
                }
                Op::Exp { res, x } => {
                    source += &f!("{indent}{res} = exp({x});\n");
                }
                Op::Ln { res, x } => {
                    source += &f!("{indent}{res} = log({x});\n");
                }
                Op::Tanh { res, x } => {
                    source += &f!("{indent}{res} = tanh({x});\n");
                }
                Op::Sqrt { res, x } => {
                    source += &f!("{indent}{res} = sqrt({x});\n");
                }
                Op::Add { res, x, y } => {
                    source += &f!("{indent}{res} = {x} + {y};\n");
                }
                Op::Sub { res, x, y } => {
                    source += &f!("{indent}{res} = {x} - {y};\n");
                }
                Op::Mul { res, x, y } => {
                    source += &f!("{indent}{res} = {x} * {y};\n");
                }
                Op::Div { res, x, y } => {
                    source += &f!("{indent}{res} = {x} / {y};\n");
                }
                Op::Pow { res, x, y } => {
                    source += &f!("{indent}{res} = pow({x}, {y});\n");
                }
                Op::Cmplt { res, x, y } => {
                    source += &f!("{indent}{res} = {x} < {y};\n");
                }
                Op::Max { res, x, y } => {
                    source += &f!("{indent}{res} = max({x}, {y});\n");
                }
                Op::Where { res, x, y, z } => {
                    source += &f!("{indent}{res} = {x} ? {y} : {z};\n");
                }
                Op::Loop {
                    id,
                    upper_bound,
                    step,
                } => {
                    source += &f!("{indent}for ({id_t} rid{id} = 0; rid{id} < {upper_bound}; rid{id} += {step}) {{\n");
                    indent += "  ";
                }
                Op::EndLoop => {
                    indent = indent[..indent.len() - 2].into();
                    source += &f!("{indent}}}\n");
                }
            }
        }

        source += &f!("}}");

        Program::compile(
            source.as_str(),
            self.context,
            &self.devices,
            ir.global_work_size.as_slice(),
            ir.local_work_size.as_slice(),
            ir.res_byte_size,
        )
    }
}

/*#[test]
fn exp_test() -> Result<(), ZyxError> {
    let dev = crate::device_builder().platform_id(0).build()?;
    let x = dev.randn([1024], DType::F32);
    //let x = dev.uniform([4, 3], 0f32..1f32);
    //let x = dev.randn([4, 5], DType::F32);
    let y = x.exp();
    //let x_vec: Vec<f32> = x.to_vec()?;
    let _y_vec: Vec<f32> = y.to_vec()?;
    //panic!();
    //panic!("{y_vec:?}");
    Ok(())
}*/

/*#[test]
fn sum_test() -> Result<(), ZyxError> {
    let dev = crate::device()?;
    let x = dev.tensor([[8, 4, 3], [5, 4, 2]]).transpose();
    //let x = dev.randn([1024, 1024], DType::I32);
    //let x = dev.randn([10, 12], DType::F32);
    let y = x.sum(-1);
    //std::println!("y shape: {:?}", y.shape());
    let y = y.exp() + y;
    let y_vec: Vec<i32> = y.to_vec()?;
    assert_eq!(y_vec, [13, 8, 5]);
    //panic!();
    //panic!("{y_vec:?}");
    // res [[9], [7]]
    Ok(())
}*/

#[test]
fn dot_test() -> Result<(), ZyxError> {
    let dev = crate::device_builder().platform_id(0).build()?;
    let x = dev.randn([1024, 1024], DType::F32);
    let y = dev.randn([1024, 1024], DType::F32);
    let z = x.dot(&y).tanh() + x;
    let _: Vec<f32> = z.to_vec()?;
    panic!();
    Ok(())
}

/*#[test]
fn t5() -> Result<(), ZyxError> {
    let dev = crate::device_builder().platform_id(0).build()?;
    let x = dev.uniform([1, 1, 1, 7, 9], 0f32..100f32);
    std::println!("{x:6.2}");
    let z = x.sum(-2);
    std::println!("{z:6.2}");
    panic!();
    Ok(())
}*/
