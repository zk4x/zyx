use crate::runtime::compiler::ir::IRKernel;
use crate::runtime::compiler::{Compiler, CompilerError, HWInfo};
use crate::Scalar;
use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::ptr;
use opencl_sys::{
    clCreateBuffer, clCreateCommandQueue, clCreateContext, clEnqueueReadBuffer,
    clEnqueueWriteBuffer, clFinish, clGetDeviceIDs, clGetPlatformIDs, clGetProgramBuildInfo,
    clReleaseEvent, clReleaseMemObject, clWaitForEvents, cl_device_id, cl_device_type, cl_int,
    cl_platform_id, cl_program_info, cl_uint, CL_DEVICE_NOT_FOUND, CL_DEVICE_TYPE_ALL,
    CL_MEM_READ_ONLY, CL_NON_BLOCKING, CL_SUCCESS,
};

#[cfg(feature = "debug1")]
use alloc::string::String;

//#[cfg(feature = "debug1")]
use libc_print::std_name::println;

pub(crate) struct OpenCLBuffer {
    memory: *mut c_void,
    event: *mut c_void,
}

pub(crate) struct OpenCLCompiler {
    context: *mut c_void,
    devices: BTreeSet<*mut c_void>,
    queues: Box<[*mut c_void]>,
    queue_size: Box<[u8]>,
    queue_id: usize,
}

// TODO we must ensure that this is OK
// Pointers in these structs are OpenCL pointers,
// so they should stay valid no matter the thread.
unsafe impl Send for OpenCLCompiler {}
unsafe impl Send for OpenCLBuffer {}

impl OpenCLCompiler {
    fn queue(&mut self) -> Result<*mut c_void, CompilerError> {
        let res = self.queues[self.queue_id];
        self.queue_size[self.queue_id] += 1;
        // Up to two event per queue, before opencl 2.0 we can't do
        // much better than that.
        if self.queue_size[self.queue_id] == 2 {
            let status = unsafe { clFinish(res) };
            if status != CL_SUCCESS {
                return Err(match status {
                    -36 => CompilerError::GeneralExecutionError(
                        "Unable to finish command queue. ERR -36: CL_INVALID_COMMAND_QUEUE",
                    ),
                    -5 => CompilerError::GeneralExecutionError(
                        "Unable to finish command queue. ERR -5: CL_OUT_OF_RESOURCES",
                    ),
                    -6 => CompilerError::OutOfHostMemory(
                        "Unable to finish command queue. ERR -6: CL_OUT_OF_HOST_MEMORY",
                    ),
                    _ => CompilerError::GeneralExecutionError(
                        "Unable to finish command queue. UNKNOWN ERROR",
                    ),
                });
            }
            self.queue_size[self.queue_id] = 0;
        }
        self.queue_id = (self.queue_id + 1) % self.queues.len();
        Ok(res)
    }
}

impl Compiler for OpenCLCompiler {
    type Buffer = OpenCLBuffer;
    type Program = ();

    fn initialize() -> Result<Self, CompilerError> {
        let platform_id = 0;
        let queues_per_device = 8;
        // TODO
        let platform_ids = {
            // Get the number of platforms
            let mut count: cl_uint = 0;
            let mut err = unsafe { clGetPlatformIDs(0, ptr::null_mut(), &mut count) };
            if err != CL_SUCCESS {
                return Err(CompilerError::InitializationFailure(match err {
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
                    return Err(CompilerError::InitializationFailure(match err {
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
            return Err(CompilerError::InitializationFailure(
                "There are no available OpenCL platforms.",
            ));
        };
        let platform = *platform;
        #[cfg(feature = "debug1")]
        println!(
            "Using OpenCL platform: {}",
            String::from_utf8(
                get_platform_data(platform, opencl_sys::CL_PLATFORM_NAME).map_err(|err| {
                    CompilerError::InitializationFailure(match err {
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
            CompilerError::InitializationFailure(match err {
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
        println!("Using devices:");
        #[cfg(feature = "debug1")]
        for dev in &device_ids {
            println!(
                "{}",
                String::from_utf8(get_device_data(*dev, opencl_sys::CL_DEVICE_NAME).map_err(
                    |err| {
                        CompilerError::InitializationFailure(match err {
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
            return Err(CompilerError::InitializationFailure(match err {
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
        println!("Using {queues_per_device} queues per device.");
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
                return Err(CompilerError::InitializationFailure(match err {
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
            queue_size: alloc::vec![0; queues.len()].into_boxed_slice(),
            queues: queues.into_boxed_slice(),
            queue_id: 0,
        })
    }

    fn hwinfo(&mut self) -> Result<HWInfo, CompilerError> {

    }

    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, CompilerError> {
        let mut err = CL_SUCCESS;
        let memory = unsafe {
            clCreateBuffer(
                self.context,
                CL_MEM_READ_ONLY,
                byte_size,
                ptr::null_mut(),
                &mut err,
            )
        };
        if err != CL_SUCCESS {
            return Err(match err {
                -34 => CompilerError::GeneralExecutionError(
                    "Unable to allocate memory. ERR -34: CL_INVALID_CONTEXT",
                ),
                -64 => CompilerError::GeneralExecutionError(
                    "Unable to allocate memory. ERR -64: CL_INVALID_PROPERTY",
                ),
                -30 => CompilerError::GeneralExecutionError(
                    "Unable to allocate memory. ERR -30: CL_INVALID_VALUE",
                ),
                -61 => CompilerError::GeneralExecutionError(
                    "Unable to allocate memory. ERR -61: CL_INVALID_BUFFER_SIZE",
                ),
                -4 => CompilerError::OutOfDeviceMemory(
                    "Unable to allocate memory. ERR -4: CL_MEM_OBJECT_ALLOCATION_FAILURE",
                ),
                -5 => CompilerError::GeneralExecutionError(
                    "Unable to allocate memory. ERR -5: CL_OUT_OF_RESOURCES",
                ),
                -6 => CompilerError::OutOfHostMemory(
                    "Unable to allocate memory. ERR -6: CL_OUT_OF_HOST_MEMORY",
                ),
                _ => {
                    CompilerError::GeneralExecutionError("Unable to allocate memory. UNKNOWN ERROR")
                }
            });
        }
        Ok(Self::Buffer {
            memory,
            event: ptr::null_mut(),
        })
    }

    fn store_memory<T: Scalar>(
        &mut self,
        buffer: &mut Self::Buffer,
        data: &[T],
    ) -> Result<(), CompilerError> {
        //std::println!("Storing");
        // TODO we can also do async stores with iter being &[T], in then we need a way of making
        // sure that the reference stays valid for the whole duration of the copy.
        // TODO we can do batched load, with buffer of say 1 MB size in RAM and offset write buffer
        let size = data.len() * T::byte_size();
        let status = unsafe {
            clEnqueueWriteBuffer(
                self.queue()?,
                buffer.memory,
                CL_NON_BLOCKING,
                0,
                size,
                data.as_ptr().cast(),
                0,
                ptr::null(),
                &mut buffer.event,
            )
        };
        if status != CL_SUCCESS {
            return Err(match status {
                -36 => CompilerError::GeneralExecutionError(
                    "Unable to write buffer. ERR -36: CL_INVALID_COMMAND_QUEUE",
                ),
                -34 => CompilerError::GeneralExecutionError(
                    "Unable to write buffer. ERR -34: CL_INVALID_CONTEXT",
                ),
                -38 => CompilerError::GeneralExecutionError(
                    "Unable to write buffer. ERR -38: CL_INVALID_MEM_OBJECT",
                ),
                -30 => CompilerError::GeneralExecutionError(
                    "Unable to write buffer. ERR -30: CL_INVALID_VALUE",
                ),
                -57 => CompilerError::GeneralExecutionError(
                    "Unable to write buffer. ERR -57: CL_INVALID_EVENT_WAIT_LIST",
                ),
                -13 => CompilerError::GeneralExecutionError(
                    "Unable to write buffer. ERR -13: CL_MISALIGNED_SUB_BUFFER_OFFSET",
                ),
                -14 => CompilerError::GeneralExecutionError(
                    "Unable to write buffer. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
                ),
                -4 => CompilerError::OutOfDeviceMemory(
                    "Unable to write buffer. ERR -4: CL_MEM_OBJECT_ALLOCATION_FAILURE",
                ),
                -59 => CompilerError::GeneralExecutionError(
                    "Unable to write buffer. ERR -59: CL_INVALID_OPERATION",
                ),
                -5 => CompilerError::GeneralExecutionError(
                    "Unable to write buffer. ERR -5: CL_OUT_OF_RESOURCES",
                ),
                -6 => CompilerError::OutOfHostMemory(
                    "Unable to write buffer. ERR -6: CL_OUT_OF_HOST_MEMORY",
                ),
                _ => CompilerError::GeneralExecutionError("Unable to write buffer. UNKNOWN ERROR"),
            });
        }
        let status = unsafe { clWaitForEvents(1, (&[buffer.event]).as_ptr().cast()) };
        if status != CL_SUCCESS {
            return Err(match status {
                -30 => CompilerError::GeneralExecutionError("Unable to finish buffer write event. ERR -30: CL_INVALID_VALUE"),
                -34 => CompilerError::GeneralExecutionError("Unable to finish buffer write event. ERR -34: CL_INVALID_CONTEXT"),
                -58 => CompilerError::GeneralExecutionError("Unable to finish buffer write event. ERR -58: CL_INVALID_EVENT"),
                -14 => CompilerError::GeneralExecutionError("Unable to finish buffer write event. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"),
                -5 => CompilerError::GeneralExecutionError("Unable to finish buffer write event. ERR -5: CL_OUT_OF_RESOURCES"),
                -6 => CompilerError::OutOfDeviceMemory("Unable to finish buffer write event. ERR -6: CL_OUT_OF_MEMORY"),
                _ => CompilerError::GeneralExecutionError("Unable to finish buffer write event. UNKNOWN ERROR"),
            });
        }
        Ok(())
    }

    fn load_memory<T: Scalar>(
        &mut self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, CompilerError> {
        let mut data: Vec<T> = Vec::with_capacity(length);
        let mut event: *mut c_void = ptr::null_mut();
        cl_wait_for_events(&[buffer.event])?;
        let status = unsafe {
            clEnqueueReadBuffer(
                self.queue()?,
                buffer.memory,
                CL_NON_BLOCKING,
                0,
                length * T::byte_size(),
                data.as_mut_ptr().cast(),
                0,
                // TODO why does this not work?
                ptr::null_mut(), //events.as_ptr().cast(),
                &mut event,
            )
        };
        if status != CL_SUCCESS {
            return Err(match status {
                -36 => CompilerError::GeneralExecutionError(
                    "Unable to read buffer. ERR -36: CL_INVALID_COMMAND_QUEUE",
                ),
                -34 => CompilerError::GeneralExecutionError(
                    "Unable to read buffer. ERR -34: CL_INVALID_CONTEXT",
                ),
                -38 => CompilerError::GeneralExecutionError(
                    "Unable to read buffer. ERR -38: CL_INVALID_MEM_OBJECT",
                ),
                -30 => CompilerError::GeneralExecutionError(
                    "Unable to read buffer. ERR -30: CL_INVALID_VALUE",
                ),
                -57 => CompilerError::GeneralExecutionError(
                    "Unable to read buffer. ERR -57: CL_INVALID_EVENT_WAIT_LIST",
                ),
                -13 => CompilerError::GeneralExecutionError(
                    "Unable to read buffer. ERR -13: CL_MISALIGNED_SUB_BUFFER_OFFSET",
                ),
                -14 => CompilerError::GeneralExecutionError(
                    "Unable to read buffer. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
                ),
                -4 => CompilerError::OutOfDeviceMemory(
                    "Unable to read buffer. ERR -4: CL_MEM_OBJECT_ALLOCATION_FAILURE",
                ),
                -59 => CompilerError::GeneralExecutionError(
                    "Unable to read buffer. ERR -59: CL_INVALID_OPERATION",
                ),
                -5 => CompilerError::GeneralExecutionError(
                    "Unable to read buffer. ERR -5: CL_OUT_OF_RESOURCES",
                ),
                -6 => CompilerError::OutOfHostMemory(
                    "Unable to read buffer. ERR -6: CL_OUT_OF_HOST_MEMORY",
                ),
                _ => CompilerError::GeneralExecutionError("Unable to read buffer. UNKNOWN ERROR"),
            });
        }
        cl_wait_for_events(&[event])?;
        // We are now done reading, so the vec is initialized
        unsafe { data.set_len(length) }
        Ok(data)
    }

    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), CompilerError> {
        let status = unsafe { clReleaseMemObject(buffer.memory) };
        if status != CL_SUCCESS {
            return Err(match status {
                -38 => CompilerError::GeneralExecutionError(
                    "Unable to release buffer. ERR -38: CL_INVALID_MEM_OBJECT",
                ),
                -5 => CompilerError::GeneralExecutionError(
                    "Unable to release buffer. ERR -5: CL_OUT_OF_RESOURCES",
                ),
                -6 => CompilerError::OutOfHostMemory(
                    "Unable to release buffer. ERR -6: CL_OUT_OF_HOST_MEMORY",
                ),
                _ => {
                    CompilerError::GeneralExecutionError("Unable to release buffer. UNKNOWN ERROR")
                }
            });
        }
        let status = unsafe { clReleaseEvent(buffer.event) };
        if status != CL_SUCCESS {
            return Err(match status {
                -58 => CompilerError::GeneralExecutionError(
                    "Unable to release event. ERR -58: CL_INVALID_EVENT",
                ),
                -5 => CompilerError::GeneralExecutionError(
                    "Unable to release event. ERR -5: CL_OUT_OF_RESOURCES",
                ),
                -6 => CompilerError::OutOfHostMemory(
                    "Unable to release event. ERR -6: CL_OUT_OF_HOST_MEMORY",
                ),
                _ => CompilerError::GeneralExecutionError("Unable to release event. UNKNOWN ERROR"),
            });
        }
        Ok(())
    }

    fn compile_program(&mut self, kernel: &IRKernel) -> Result<Self::Program, CompilerError> {
        println!("Compiling IRKernel: {kernel:#?}");
        todo!()
    }

    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &[&mut Self::Buffer],
    ) -> Result<(), CompilerError> {
        todo!()
    }

    fn drop_program(&mut self, program: Self::Program) {
        todo!()
    }
}

fn cl_wait_for_events(events: &[*mut c_void]) -> Result<(), CompilerError> {
    let status = unsafe { clWaitForEvents(1, events.as_ptr().cast()) };
    if status != CL_SUCCESS {
        return Err(match status {
            -30 => CompilerError::GeneralExecutionError("Unable to finish buffer read event. ERR -30: CL_INVALID_VALUE"),
            -34 => CompilerError::GeneralExecutionError("Unable to finish buffer read event. ERR -34: CL_INVALID_CONTEXT"),
            -58 => CompilerError::GeneralExecutionError("Unable to finish buffer read event. ERR -58: CL_INVALID_EVENT"),
            -14 => CompilerError::GeneralExecutionError("Unable to finish buffer read event. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"),
            -5 => CompilerError::OutOfDeviceMemory("Unable to finish buffer read event. ERR -5: CL_OUT_OF_RESOURCES"),
            -6 => CompilerError::OutOfDeviceMemory("Unable to finish buffer read event. ERR -6: CL_OUT_OF_MEMORY"),
            _ => CompilerError::GeneralExecutionError("Unable to finish buffer read event. UNKNOWN ERROR"),
        });
    }
    return Ok(());
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
