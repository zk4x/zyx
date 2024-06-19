use crate::runtime::compiler::ir::{IRKernel, IRKernelArg, IROp};
use crate::runtime::compiler::{BOp, Compiler, CompilerError, HWInfo, Scope, UOp};
use crate::{DType, Scalar};
use alloc::boxed::Box;
use alloc::collections::BTreeSet;
use alloc::ffi::CString;
use alloc::format as f;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::c_void;
use core::ptr;
use opencl_sys::{
    clBuildProgram, clCreateBuffer, clCreateCommandQueue, clCreateContext, clCreateKernel,
    clCreateProgramWithSource, clEnqueueNDRangeKernel, clEnqueueReadBuffer, clEnqueueWriteBuffer,
    clFinish, clGetDeviceIDs, clGetPlatformIDs, clGetProgramBuildInfo, clReleaseEvent,
    clReleaseMemObject, clReleaseProgram, clSetKernelArg, clWaitForEvents, cl_device_id,
    cl_device_type, cl_int, cl_platform_id, cl_program_info, cl_uint, CL_DEVICE_GLOBAL_MEM_SIZE,
    CL_DEVICE_LOCAL_MEM_SIZE, CL_DEVICE_MAX_MEM_ALLOC_SIZE, CL_DEVICE_MAX_WORK_GROUP_SIZE,
    CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, CL_DEVICE_MAX_WORK_ITEM_SIZES,
    CL_DEVICE_MEM_BASE_ADDR_ALIGN, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, CL_DEVICE_NOT_FOUND,
    CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, CL_DEVICE_TYPE_ALL, CL_MEM_READ_ONLY, CL_NON_BLOCKING,
    CL_PROGRAM_BUILD_LOG, CL_SUCCESS,
};

impl DType {
    fn ocl(&self) -> &str {
        return match self {
            DType::BF16 => "TODO",
            DType::F16 => "half",
            DType::F32 => "float",
            DType::F64 => "double",
            DType::CF32 => "TODO",
            DType::CF64 => "TODO",
            DType::U8 => "unsigned char",
            DType::I8 => "char",
            DType::I16 => "short",
            DType::I32 => "int",
            DType::I64 => "long",
        }
    }
}

pub(crate) struct OpenCLBuffer {
    memory: *mut c_void,
    event: *mut c_void,
}

pub(crate) struct OpenCLProgram {
    name: String,
    program: *mut c_void,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
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
unsafe impl Send for OpenCLProgram {}

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
        return Ok(res)
    }
}

impl Compiler for OpenCLCompiler {
    type Buffer = OpenCLBuffer;
    type Program = OpenCLProgram;

    fn initialize() -> Result<Self, CompilerError> {
        let platform_id = 0;
        let queues_per_device = 8;
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
        libc_print::libc_println!(
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
        libc_print::libc_println!("Using devices:");
        #[cfg(feature = "debug1")]
        for dev in &device_ids {
            libc_print::libc_println!(
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
        libc_print::libc_println!("Using {queues_per_device} queues per device.");
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
        return Ok(Self {
            context,
            devices,
            queue_size: alloc::vec![0; queues.len()].into_boxed_slice(),
            queues: queues.into_boxed_slice(),
            queue_id: 0,
        })
    }

    fn hardware_information(&mut self) -> Result<HWInfo, CompilerError> {
        let dev = *self.devices.first().unwrap();
        // TODO get max work item sizes by somehow converting Vec<u8> into Vec<usize>
        let max_work_item_dims = u32::from_ne_bytes(
            get_device_data(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)
                .unwrap()
                .try_into()
                .unwrap(),
        ) as usize;
        let mwis = get_device_data(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES).unwrap();
        let mut max_work_item_sizes = Vec::with_capacity(max_work_item_dims);
        for i in 0..max_work_item_dims {
            let max_dim_size: usize = unsafe {
                core::mem::transmute([
                    mwis[i * 8 + 0],
                    mwis[i * 8 + 1],
                    mwis[i * 8 + 2],
                    mwis[i * 8 + 3],
                    mwis[i * 8 + 4],
                    mwis[i * 8 + 5],
                    mwis[i * 8 + 6],
                    mwis[i * 8 + 7],
                ])
            };
            max_work_item_sizes.push(max_dim_size);
        }
        //libc_print::libc_println!("Max work item sizes: {max_work_item_sizes:?}");
        return Ok(HWInfo {
            max_work_item_sizes,
            max_work_group_size: usize::from_ne_bytes(
                get_device_data(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE)
                    .unwrap()
                    .try_into()
                    .unwrap(),
            ),
            preferred_vector_size: u32::from_ne_bytes(
                get_device_data(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)
                    .unwrap()
                    .try_into()
                    .unwrap(),
            ) as usize
                * 4,
            f16_support: true,
            f64_support: true,
            fmadd: true,
            global_mem_size: u64::from_ne_bytes(
                get_device_data(dev, CL_DEVICE_GLOBAL_MEM_SIZE)
                    .unwrap()
                    .try_into()
                    .unwrap(),
            ) as usize,
            max_mem_alloc: u64::from_ne_bytes(
                get_device_data(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE)
                    .unwrap()
                    .try_into()
                    .unwrap(),
            ) as usize,
            mem_align: u32::from_ne_bytes(
                get_device_data(dev, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE)
                    .unwrap()
                    .try_into()
                    .unwrap(),
            ) as usize
                / 8,
            page_size: u32::from_ne_bytes(
                get_device_data(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN)
                    .unwrap()
                    .try_into()
                    .unwrap(),
            ) as usize
                / 8,
            local_mem_size: u64::from_ne_bytes(
                get_device_data(dev, CL_DEVICE_LOCAL_MEM_SIZE)
                    .unwrap()
                    .try_into()
                    .unwrap(),
            ) as usize,
            num_registers: 128, // We can only guess or have a map of concrete hardware and respective register counts
            native_mm16x16_support: false,
        });
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
        return Ok(data)
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
        return Ok(())
    }

    fn compile_program(&mut self, kernel: &IRKernel) -> Result<Self::Program, CompilerError> {
        //println!("Compiling IRKernel: {kernel:#?}");

        let mut source = String::from("(\n");
        let mut indent = String::from("  ");

        // Transpile kernel args
        for (id, IRKernelArg { dtype, read_only }) in kernel.args.iter().enumerate() {
            source += &f!(
                "{indent}__global {}{}* g{id},\n",
                if *read_only { "const " } else { "" },
                dtype.ocl()
            );
        }

        source.pop();
        source.pop();
        source += "\n) {\n";

        // Add indices for global and local loops
        source += "  unsigned int i0 = get_group_id(0);\n";
        source += "  unsigned int i1 = get_local_id(0);\n";
        source += "  unsigned int i2 = get_group_id(1);\n";
        source += "  unsigned int i3 = get_local_id(1);\n";
        source += "  unsigned int i5 = get_group_id(2);\n";
        source += "  unsigned int i6 = get_local_id(2);\n";

        // Transpile kernel ops, skip ends of global and local loops
        for op in &kernel.ops {
            match op {
                IROp::InitMem {
                    id,
                    scope,
                    read_only,
                    len,
                    dtype,
                } => match scope {
                    Scope::Global => {}
                    Scope::Local => todo!(),
                    Scope::Register => {
                        let read_only = if *read_only { "const " } else { "" };
                        let size = if *len > 1 {
                            f!("[{len}]")
                        } else {
                            String::new()
                        };
                        source += &f!(
                            "{indent}{read_only}{} {}{id}{};\n",
                            dtype.ocl(),
                            match scope {
                                Scope::Global => "g",
                                Scope::Local => "l",
                                Scope::Register => "r",
                            },
                            size,
                        );
                    }
                },
                IROp::AssignMem { z, x } => {
                    source += &f!("{indent}{z} = {x};\n");
                }
                IROp::UnaryMem { z, x, op } => {
                    source += &f!(
                        "{indent}{z} = {}{x});\n",
                        match op {
                            UOp::Cast(dtype) => f!("({})(", dtype.ocl()),
                            UOp::Inv => String::from("1/("),
                            UOp::Neg => String::from("-("),
                            UOp::Sin => String::from("sin("),
                            UOp::Cos => String::from("cos("),
                            UOp::Exp => String::from("exp("),
                            UOp::Ln => String::from("log("),
                            UOp::Sqrt => String::from("sqrt("),
                        }
                    );
                }
                IROp::BinaryMem { z, x, y, op } => {
                    source += &f!(
                        "{indent}{z} = {};\n",
                        match op {
                            BOp::Add => f!("{x}+{y}"),
                            BOp::Sub => f!("{x}-{y}"),
                            BOp::Mul => f!("{x}*{y}"),
                            BOp::Div => f!("{x}/{y}"),
                            BOp::Pow => f!("powf({x}, {y})"),
                            BOp::Max => f!("max({x}, {y})"),
                            BOp::Cmplt => f!("{x}<{y}"),
                        }
                    );
                }
                IROp::Loop { id, max } => {
                    source += &f!("{indent}for (unsigned int i{id}; i{id} < {max}; i{id}++) {{\n");
                    indent += "  ";
                }
                IROp::EndLoop => {
                    indent.pop();
                    indent.pop();
                    source += &f!("{indent}}}\n");
                }
            }
        }

        source += "}";

        return OpenCLProgram::compile_from_source(
            &source,
            self.context,
            &self.devices,
            kernel.global_work_size,
            kernel.local_work_size,
        );
    }

    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &mut [Self::Buffer],
    ) -> Result<(), CompilerError> {
        //#[cfg(not(feature = "debug1"))]
        //let (_, _) = (flop, bytes);
        let program_name = &CString::new(program.name.clone()).unwrap();
        let mut status = CL_SUCCESS;
        let kernel =
            unsafe { clCreateKernel(program.program, program_name.as_ptr().cast(), &mut status) };
        if status != CL_SUCCESS {
            return Err(CompilerError::GeneralExecutionError(match status {
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
            CompilerError::GeneralExecutionError(match err {
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
        for arg in &mut *args {
            let (buffer, event) = (arg.memory, arg.event);
            //libc_print::libc_println!("Buffer {buffer:?}, event {event:?}");
            if !event.is_null() {
                events.push(event);
            }
            //std::println!("Arg: {:?}", self.load::<f32>(arg, 6));
            // This is POINTER MAGIC. Be careful.
            let ptr: *const _ = &buffer;
            status = unsafe {
                clSetKernelArg(kernel, i, core::mem::size_of::<*mut c_void>(), ptr.cast())
            };
            if status != CL_SUCCESS {
                return Err(kernel_arg_err_handler(status));
            }
            i += 1;
        }
        let mut global_work_size = program.global_work_size;
        for (i, lwd) in program.local_work_size.iter().enumerate() {
            global_work_size[i] *= lwd;
        }
        let mut event: *mut c_void = ptr::null_mut();
        //#[cfg(feature = "debug1")]
        //let begin = std::time::Instant::now();
        let status = unsafe {
            clEnqueueNDRangeKernel(
                self.queue()?,
                kernel,
                u32::try_from(program.global_work_size.len()).unwrap(),
                ptr::null(),
                global_work_size.as_ptr(),
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
        if status != CL_SUCCESS {
            return Err(CompilerError::GeneralExecutionError(match status {
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
                return Err(CompilerError::GeneralExecutionError(match err {
                    -30 => "Unable to finish kernel execution event. ERR -30: CL_INVALID_VALUE",
                    -34 => "Unable to finish kernel execution event. ERR -34: CL_INVALID_CONTEXT",
                    -58 => "Unable to finish kernel execution event. ERR -58: CL_INVALID_EVENT",
                    -14 => "Unable to finish kernel execution event. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
                    -5 => "Unable to finish kernel execution event. ERR -5: CL_OUT_OF_RESOURCES",
                    -6 => "Unable to finish kernel execution event. ERR -6: CL_OUT_OF_MEMORY",
                    _ => "Unable to finish kernel execution event. UNKNOWN ERROR",
                }));
            }
            /*let elapsed_nanos = begin.elapsed().as_nanos();
            let elapsed_millis = elapsed_nanos as f64 / 1000000.;
            println!(
                "Kernel took {elapsed_millis:.3}ms for {bytes} B, {flop} FLOP ~ {:.2} GFLOPS, {:.2} GB/s",
                flop as f64 / elapsed_nanos as f64,
                bytes as f64 / elapsed_nanos as f64,
            );*/
        }
        for arg in &mut *args {
            if arg.event.is_null() {
                arg.event = event;
            }
        }
        libc_print::libc_println!("{:?}", self.load_memory::<f32>(&args[1], 4).unwrap());
        return Ok(())
    }

    fn release_program(&mut self, program: Self::Program) -> Result<(), CompilerError> {
        let status = unsafe { clReleaseProgram(program.program) };
        if status != CL_SUCCESS {
            return Err(match status {
                -44 => CompilerError::GeneralExecutionError(
                    "Unable to release program. ERR -44: CL_INVALID_PROGRAM",
                ),
                -5 => CompilerError::GeneralExecutionError(
                    "Unable to release program. ERR -5: CL_OUT_OF_RESOURCES",
                ),
                -6 => CompilerError::OutOfHostMemory(
                    "Unable to release program. ERR -6: CL_OUT_OF_HOST_MEMORY",
                ),
                _ => {
                    CompilerError::GeneralExecutionError("Unable to release program. UNKNOWN ERROR")
                }
            });
        }
        Ok(())
    }
}

impl OpenCLProgram {
    fn compile_from_source(
        source: &str,
        context: *mut c_void,
        devices: &BTreeSet<*mut c_void>,
        global_work_size: [usize; 3],
        local_work_size: [usize; 3],
    ) -> Result<Self, CompilerError> {
        let name = f!(
            "k__{}_{}__{}_{}__{}_{}",
            global_work_size[0],
            local_work_size[0],
            global_work_size[1],
            local_work_size[1],
            global_work_size[2],
            local_work_size[2],
        );
        let mut pragma = f!("");
        if source.contains("double") {
            pragma += &"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        }
        let source = f!("{pragma}__kernel void {name}{source}");
        #[cfg(feature = "debug1")]
        libc_print::libc_println!("{source}");
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
            return Err(match err {
                -34 => CompilerError::GeneralExecutionError(
                    "Unable to compile program. ERR -34: CL_INVALID_CONTEXT",
                ),
                -30 => CompilerError::GeneralExecutionError(
                    "Unable to compile program. ERR -30: CL_INVALID_VALUE",
                ),
                -5 => CompilerError::GeneralExecutionError(
                    "Unable to compile program. ERR -5: CL_OUT_OF_RESOURCES",
                ),
                -6 => CompilerError::OutOfHostMemory(
                    "Unable to compile program. ERR -6: CL_OUT_OF_HOST_MEMORY",
                ),
                _ => {
                    CompilerError::GeneralExecutionError("Unable to compile program. UNKNOWN ERROR")
                }
            });
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
                    match err {
                        -33 => CompilerError::GeneralExecutionError("Unable to get info about failed compilation. ERR -33: CL_INVALID_DEVICE"),
                        -30 => CompilerError::GeneralExecutionError("Unable to get info about failed compilation. ERR -30: CL_INVALID_VALUE"),
                        -44 => CompilerError::GeneralExecutionError("Unable to get info about failed compilation. ERR -44: CL_INVALID_PROGRAM"),
                        -5 => CompilerError::GeneralExecutionError("Unable to get info about failed compilation. ERR -5: CL_OUT_OF_RESOURCES"),
                        -6 => CompilerError::OutOfHostMemory("Unable to get info about failed compilation. ERR -6: CL_OUT_OF_HOST_MEMORY"),
                        _ => CompilerError::GeneralExecutionError("Unable to get info about failed compilation. UNKNOWN ERROR"),
                    }
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
            global_work_size,
            local_work_size,
        })
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
            return Err(status)
        } else {
            return Ok(size)
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
                return Err(status)
            } else {
                return Ok(data)
            }
        } else {
            return Ok(Vec::default())
        }
    }
    return get_vector(program, device, param_name, size)
}

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
            return Err(status)
        } else {
            return Ok(size)
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
                return Err(status)
            } else {
                return Ok(data)
            }
        } else {
            return Ok(Vec::default())
        }
    }
    return get_vector(device, param_name, size)
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
        return Err(status)
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
            return Err(status)
        } else {
            return Ok(ids)
        }
    } else {
        return Ok(Vec::default())
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
            return Err(status)
        } else {
            return Ok(size)
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
                return Err(status)
            } else {
                return Ok(data)
            }
        } else {
            return Ok(Vec::default())
        }
    }
    return get_vector(platform, param_name, size)
}
