use crate::dtype::DType;
use crate::runtime::compiler::ir::{IRArg, IRKernel, IROp};
use crate::runtime::compiler::{BOp, Compiler, CompilerError, HWInfo, Scope, UOp};
use crate::scalar::Scalar;
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
            #[cfg(feature = "half")]
            DType::BF16 => "BF16 is not native to OpenCL, workaround is WIP.",
            #[cfg(feature = "half")]
            DType::F16 => "half",
            DType::F32 => "float",
            DType::F64 => "double",
            #[cfg(feature = "complex")]
            DType::CF32 => "Not native to OpenCL, workaround is WIP",
            #[cfg(feature = "complex")]
            DType::CF64 => "Not native to OpenCL, workaround is WIP",
            DType::U8 => "unsigned char",
            DType::I8 => "char",
            DType::I16 => "short",
            DType::I32 => "int",
            DType::I64 => "long",
        };
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
    args_read_only: Vec<bool>,
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
            handle_status(
                status,
                "Unable to finish execution of command queue.",
                &[-36, -5, -6],
            )?;
            self.queue_size[self.queue_id] = 0;
        }
        self.queue_id = (self.queue_id + 1) % self.queues.len();
        return Ok(res);
    }
}

impl Compiler for OpenCLCompiler {
    type Buffer = OpenCLBuffer;
    type Program = OpenCLProgram;

    fn initialize() -> Result<Self, CompilerError> {
        let platform_id = 1;
        let queues_per_device = 8;
        let platform_ids = {
            // Get the number of platforms
            let mut count: cl_uint = 0;
            let status = unsafe { clGetPlatformIDs(0, ptr::null_mut(), &mut count) };
            handle_status(status, "Unable to get OpenCL platform ids.", &[-30, -6])?;
            if count > 0 {
                // Get the platform ids.
                let len = count as usize;
                let mut ids: Vec<*mut c_void> = Vec::with_capacity(len);
                let status = unsafe { clGetPlatformIDs(count, ids.as_mut_ptr(), ptr::null_mut()) };
                handle_status(status, "Unable to get OpenCL platform ids.", &[-30, -6])?;
                unsafe { ids.set_len(len) };
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
            String::from_utf8(get_platform_data(platform, opencl_sys::CL_PLATFORM_NAME)?).unwrap()
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
        });
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
        let mut status = CL_SUCCESS;
        let memory = unsafe {
            clCreateBuffer(
                self.context,
                CL_MEM_READ_ONLY,
                byte_size,
                ptr::null_mut(),
                &mut status,
            )
        };
        handle_status(
            status,
            "Unable to allocate memory.",
            &[-34, -64, -30, -61, -4, -5, -6],
        )?;
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
        handle_status(
            status,
            "Unable to write buffer.",
            &[-36, -34, -38, -30, -57, -13, -14, -4, -59, -5, -6],
        )?;
        // Immediattely synchronize because we do not know the lifetime of data
        let status = unsafe { clWaitForEvents(1, (&[buffer.event]).as_ptr().cast()) };
        return handle_status(
            status,
            "Unable to finish buffer write event.",
            &[-30, -34, -58, -14, -5, -6],
        );
    }

    fn load_memory<T: Scalar>(
        &mut self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, CompilerError> {
        debug_assert!(
            !buffer.memory.is_null(),
            "Trying to read null memory. Internal bug."
        );
        debug_assert!(
            !buffer.event.is_null(),
            "Trying to read uninitialized memory. Internal bug."
        );
        cl_wait_for_events(&[buffer.event])?;
        let mut data: Vec<T> = Vec::with_capacity(length);
        let mut event: *mut c_void = ptr::null_mut();
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
        handle_status(
            status,
            "Unable to read buffer.",
            &[-36, -34, -38, -30, -57, -13, -14, -4, -59, -5, -6],
        )?;
        cl_wait_for_events(&[event])?;
        // We are now done reading, so the vec is initialized
        unsafe { data.set_len(length) }
        return Ok(data);
    }

    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), CompilerError> {
        let status = unsafe { clReleaseMemObject(buffer.memory) };
        handle_status(status, "Unable to release buffer.", &[-38, -5, -6])?;
        if !buffer.event.is_null() {
            let status = unsafe { clReleaseEvent(buffer.event) };
            handle_status(status, "Unable to release event.", &[-58, -5, -6])?;
        } else {
            #[cfg(feature = "debug1")]
            libc_print::libc_println!("Warning: A buffer was allocated, but never initialized.");
        }
        return Ok(());
    }

    fn compile_program(&mut self, kernel: &IRKernel) -> Result<Self::Program, CompilerError> {
        //println!("Compiling IRKernel: {kernel:#?}");

        let mut source = String::from("(\n");
        let mut indent = String::from("  ");

        // Transpile kernel args
        let mut args_read_only = Vec::new();
        for (id, IRArg { dtype, read_only }) in kernel.args.iter() {
            source += &f!(
                "{indent}__global {}{}* g{id},\n",
                if *read_only { "const " } else { "" },
                dtype.ocl()
            );
            if *read_only {
                args_read_only.push(true);
            } else {
                args_read_only.push(false);
            }
        }

        source.pop();
        source.pop();
        source += "\n) {\n";

        // Add indices for global and local loops
        source += &f!(
            "  unsigned int i0 = get_group_id(0);   /* 0..{} */\n",
            kernel.global_work_size[0]
        );
        source += &f!(
            "  unsigned int i1 = get_local_id(0);   /* 0..{} */\n",
            kernel.local_work_size[0]
        );
        source += &f!(
            "  unsigned int i2 = get_group_id(1);   /* 0..{} */\n",
            kernel.global_work_size[1]
        );
        source += &f!(
            "  unsigned int i3 = get_local_id(1);   /* 0..{} */\n",
            kernel.local_work_size[1]
        );
        source += &f!(
            "  unsigned int i4 = get_group_id(2);   /* 0..{} */\n",
            kernel.global_work_size[2]
        );
        source += &f!(
            "  unsigned int i5 = get_local_id(2);   /* 0..{} */\n",
            kernel.local_work_size[2]
        );
        source += "  unsigned int t0, t1, t2;\n";

        // Transpile kernel ops, skip ends of global and local loops
        for op in &kernel.ops {
            match op {
                IROp::DeclareMem {
                    id,
                    scope,
                    read_only,
                    len,
                    dtype,
                    init: _,
                } => match scope {
                    Scope::Global => {}
                    Scope::Local => {
                        let read_only = if *read_only { "const " } else { "" };
                        let size = if *len > 0 {
                            f!("[{len}]")
                        } else {
                            String::new()
                        };
                        source += &f!(
                            "{indent}__local {read_only}{} l{id}{};\n",
                            dtype.ocl(),
                            size,
                        );
                    }
                    Scope::Register => {
                        let read_only = if *read_only { "const " } else { "" };
                        let size = if *len > 0 {
                            f!("[{len}]")
                        } else {
                            String::new()
                        };
                        source += &f!("{indent}{read_only}{} r{id}{};\n", dtype.ocl(), size,);
                    }
                },
                IROp::AssignMem { z, x } => {
                    let (zt, z) = z.to_str(0);
                    if !zt.is_empty() {
                        for idx in zt.into_iter() {
                            source += &f!("{indent}t0 = {idx};\n");
                        }
                    }
                    let (xt, x) = x.to_str(1);
                    if !xt.is_empty() {
                        for idx in xt.into_iter() {
                            source += &f!("{indent}t1 = {idx};\n");
                        }
                    }
                    source += &f!("{indent}{z} = {x};\n");
                }
                IROp::Unary { z, x, ops } => {
                    let (zt, z) = z.to_str(0);
                    if !zt.is_empty() {
                        for idx in zt.into_iter() {
                            source += &f!("{indent}t0 = {idx};\n");
                        }
                    }
                    let (xt, x) = x.to_str(1);
                    if !xt.is_empty() {
                        for idx in xt.into_iter() {
                            source += &f!("{indent}t1 = {idx};\n");
                        }
                    }
                    let mut inner_op = f!("{x}");
                    for uop in ops {
                        inner_op = match uop {
                            UOp::Noop => inner_op,
                            UOp::Cast(dtype) => f!("({}){inner_op}", dtype.ocl()),
                            UOp::Neg => f!("-({inner_op})"),
                            UOp::Inv => f!("1/{inner_op}"),
                            UOp::Sin => f!("sin({inner_op})"),
                            UOp::Cos => f!("cos({inner_op})"),
                            UOp::Exp => f!("exp({inner_op})"),
                            UOp::Ln => f!("log({inner_op})"),
                            UOp::Sqrt => f!("sqrt({inner_op})"),
                            UOp::ReLU => f!("max({inner_op}, 0)"),
                            UOp::Tanh => f!("tanh({inner_op})"),
                        };
                    }
                    source += &f!("{indent}{z} = {inner_op};\n");
                }
                IROp::Binary { z, x, y, op } => {
                    let (zt, z) = z.to_str(0);
                    if !zt.is_empty() {
                        for idx in zt.into_iter() {
                            source += &f!("{indent}t0 = {idx};\n");
                        }
                    }
                    let (xt, x) = x.to_str(1);
                    if !xt.is_empty() {
                        for idx in xt.into_iter() {
                            source += &f!("{indent}t1 = {idx};\n");
                        }
                    }
                    let (yt, y) = y.to_str(2);
                    if !yt.is_empty() {
                        for idx in yt.into_iter() {
                            source += &f!("{indent}t2 = {idx};\n");
                        }
                    }
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
                IROp::Loop { id, len } => {
                    source +=
                        &f!("{indent}for (unsigned int i{id} = 0; i{id} < {len}; i{id}++) {{   /* 0..{len} */\n");
                    indent += "  ";
                }
                IROp::EndLoop => {
                    indent.pop();
                    indent.pop();
                    source += &f!("{indent}}}\n");
                }
                IROp::Barrier { scope } => {
                    let scope = match scope {
                        Scope::Register => panic!(),
                        Scope::Local => "LOCAL",
                        Scope::Global => "GLOBAL",
                    };
                    source += &f!("{indent}barrier(CLK_{scope}_MEM_FENCE);\n");
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
            args_read_only,
        );
    }

    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &mut [Self::Buffer],
    ) -> Result<(), CompilerError> {
        //#[cfg(feature = "debug1")]
        //libc_print::libc_println!("{:?}", self.load_memory::<f32>(&args[0], 4).unwrap());
        //#[cfg(not(feature = "debug1"))]
        let program_name = &CString::new(program.name.clone()).unwrap();
        let mut status = CL_SUCCESS;
        let kernel =
            unsafe { clCreateKernel(program.program, program_name.as_ptr().cast(), &mut status) };
        handle_status(
            status,
            "Unable to create kernel.",
            &[-44, -45, -46, -47, -30, -5, -6],
        )?;
        let mut events = Vec::new();
        let mut i = 0;
        for arg in &mut *args {
            let (buffer, event) = (arg.memory, arg.event);
            // Memory that is freshly allocated does not need to be awaited using events
            if !event.is_null() {
                events.push(event);
            }
            // This is POINTER MAGIC. Be careful.
            let ptr: *const _ = &buffer;
            status = unsafe {
                clSetKernelArg(kernel, i, core::mem::size_of::<*mut c_void>(), ptr.cast())
            };
            handle_status(
                status,
                "Unable to set kernel arg.",
                &[-48, -49, -50, -38, -41, -33, -51, -72, -5, -6],
            )?;
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
        handle_status(
            status,
            "Unable to enqueue kernel.",
            &[
                -45, -36, -48, -34, -52, -53, -63, -56, -54, -55, -13, -40, -10, -5, -4, -57, -59,
                -6,
            ],
        )?;
        #[cfg(feature = "debug1")]
        {
            let status = unsafe { clWaitForEvents(1, (&[event]).as_ptr().cast()) };
            handle_status(
                status,
                "Unable to finish kernel execution event.",
                &[-30, -34, -58, -14, -5, -6],
            )?;
            /*let elapsed_nanos = begin.elapsed().as_nanos();
            let elapsed_millis = elapsed_nanos as f64 / 1000000.;
            println!(
                "Kernel took {elapsed_millis:.3}ms for {bytes} B, {flop} FLOP ~ {:.2} GFLOPS, {:.2} GB/s",
                flop as f64 / elapsed_nanos as f64,
                bytes as f64 / elapsed_nanos as f64,
            );*/
        }
        // set event for buffers that were written into
        for (arg, read_only) in args.iter_mut().zip(&program.args_read_only) {
            if !read_only {
                arg.event = event;
            }
        }
        //#[cfg(feature = "debug1")]
        //libc_print::libc_println!("{:?}", self.load_memory::<f32>(&args[1], 4).unwrap());
        return Ok(());
    }

    fn release_program(&mut self, program: Self::Program) -> Result<(), CompilerError> {
        let status = unsafe { clReleaseProgram(program.program) };
        handle_status(status, "Unable to release program", &[-44, -5, -6])?;
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
        args_read_only: Vec<bool>,
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
        let mut status = CL_SUCCESS;
        let program = unsafe {
            clCreateProgramWithSource(
                context,
                1,
                sources.as_ptr().cast(),
                &[source.len()] as *const usize,
                &mut status,
            )
        };
        handle_status(status, "Unable to compile program.", &[-34, -30, -5, -6])?;
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
            // TODO perhaps return error instead of panic
            let build_log = get_program_build_data(program, devices[0], CL_PROGRAM_BUILD_LOG);
            match build_log {
                Ok(build_log) => panic!("{}", String::from_utf8_lossy(&build_log)),
                Err(status) => handle_status(
                    status,
                    "Unable to get info about failed compilation.",
                    &[-33, -30, -44, -5, -6],
                )?,
            }
        }
        Ok(Self {
            name,
            program,
            global_work_size,
            local_work_size,
            args_read_only,
        })
    }
}

fn cl_wait_for_events(events: &[*mut c_void]) -> Result<(), CompilerError> {
    let status = unsafe { clWaitForEvents(1, events.as_ptr().cast()) };
    return handle_status(
        status,
        "Unable to finish buffer read event.",
        &[-30, -34, -58, -14, -5, -6],
    );
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
            return Err(status);
        } else {
            return Ok(size);
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
                opencl_sys::clGetDeviceInfo(
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
    platform: cl_platform_id,
    device_type: cl_device_type,
) -> Result<Vec<cl_device_id>, cl_int> {
    // Get the number of devices of device_type
    let mut count: cl_uint = 0;
    let mut status =
        unsafe { clGetDeviceIDs(platform, device_type, 0, ptr::null_mut(), &mut count) };

    if (CL_SUCCESS != status) && (CL_DEVICE_NOT_FOUND != status) {
        return Err(status);
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
            return Err(status);
        } else {
            return Ok(ids);
        }
    } else {
        return Ok(Vec::default());
    }
}

#[cfg(feature = "debug1")]
fn get_platform_data(
    platform: cl_platform_id,
    param_name: opencl_sys::cl_platform_info,
) -> Result<Vec<u8>, CompilerError> {
    let mut size: usize = 0;
    let status = unsafe {
        opencl_sys::clGetPlatformInfo(platform, param_name, 0, ptr::null_mut(), &mut size)
    };
    handle_status(status, "Unable to get platform info.", &[-32, -30, -6])?;
    if 0 < size {
        let count = size / core::mem::size_of::<u8>();
        let mut data: Vec<u8> = Vec::with_capacity(count);
        let status = unsafe {
            data.set_len(count);
            opencl_sys::clGetPlatformInfo(
                platform,
                param_name,
                size,
                data.as_mut_ptr() as *mut c_void,
                ptr::null_mut(),
            )
        };
        handle_status(status, "Unable to get platform info.", &[-32, -30, -6])?;
        return Ok(data);
    } else {
        return Ok(Vec::default());
    }
}

fn handle_status(
    status: cl_int,
    err_msg: &str,
    possible_errors: &[cl_int],
) -> Result<(), CompilerError> {
    if status == CL_SUCCESS {
        return Ok(());
    } else {
        debug_assert!(
            possible_errors.contains(&status),
            "Internal bug. OpenCL error out of range of expected errors."
        );
        #[cfg(feature = "debug1")]
        libc_print::libc_println!("Error: {err_msg}, OpenCL error code {status}");
        return Err(match status {
            -4 => {
                CompilerError::OutOfDeviceMemory("OpenCL Err -4: CL_MEM_OBJECT_ALLOCATION_FAILURE")
            }
            -5 => CompilerError::GeneralExecutionError("OpenCL Err -5: CL_OUT_OF_RESOURCES"),
            -6 => CompilerError::OutOfHostMemory("OpenCL Err -6: CL_OUT_OF_HOST_MEMORY"),
            -10 => CompilerError::GeneralExecutionError(
                "OpenCL Err -10: CL_IMAGE_FORMAT_NOT_SUPPORTED",
            ),
            -13 => CompilerError::GeneralExecutionError(
                "OpenCL Err -13: CL_MISALIGNED_SUB_BUFFER_OFFSET",
            ),
            -14 => CompilerError::GeneralExecutionError(
                "OpenCL Err -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
            ),
            -30 => CompilerError::GeneralExecutionError("OpenCL Err -30: CL_INVALID_VALUE"),
            -33 => CompilerError::GeneralExecutionError("OpenCL Err -33: CL_INVALID_DEVICE_QUEUE"),
            -34 => CompilerError::GeneralExecutionError("OpenCL Err -34: CL_INVALID_CONTEXT"),
            -36 => CompilerError::GeneralExecutionError("OpenCL Err -36: CL_INVALID_COMMAND_QUEUE"),
            -38 => CompilerError::GeneralExecutionError("OpenCL Err -38: CL_INVALID_MEM_OBJECT"),
            -40 => CompilerError::GeneralExecutionError("OpenCL Err -40: CL_INVALID_IMAGE_SIZE"),
            -41 => CompilerError::GeneralExecutionError("OpenCL Err -41: CL_INVALID_SAMPLER"),
            -44 => CompilerError::GeneralExecutionError("OpenCL Err -44: CL_INVALID_PROGRAM"),
            -45 => CompilerError::GeneralExecutionError(
                "OpenCL Err -45: CL_INVALID_PROGRAM_EXECUTABLE",
            ),
            -46 => CompilerError::GeneralExecutionError("OpenCL Err -46: CL_INVALID_KERNEL_NAME"),
            -47 => {
                CompilerError::GeneralExecutionError("OpenCL Err -47: CL_INVALID_KERNEL_DEFINITION")
            }
            -48 => CompilerError::GeneralExecutionError("OpenCL Err -48: CL_INVALID_KERNEL"),
            -49 => CompilerError::GeneralExecutionError("OpenCL Err -49: CL_INVALID_ARG_INDEX"),
            -50 => CompilerError::GeneralExecutionError("OpenCL Err -50: CL_INVALID_ARG_VALUE"),
            -51 => CompilerError::GeneralExecutionError("OpenCL Err -51: CL_INVALID_ARG_SIZE"),
            -52 => CompilerError::GeneralExecutionError("OpenCL Err -52: CL_INVALID_KERNEL_ARGS"),
            -53 => {
                CompilerError::GeneralExecutionError("OpenCL Err -53: CL_INVALID_WORK_DIMENSION")
            }
            -54 => {
                CompilerError::GeneralExecutionError("OpenCL Err -54: CL_INVALID_WORK_GROUP_SIZE")
            }
            -55 => {
                CompilerError::GeneralExecutionError("OpenCL Err -55: CL_INVALID_WORK_ITEM_SIZE")
            }
            -56 => CompilerError::GeneralExecutionError("OpenCL Err -56: CL_INVALID_GLOBAL_OFFSET"),
            -57 => {
                CompilerError::GeneralExecutionError("OpenCL Err -57: CL_INVALID_EVENT_WAIT_LIST")
            }
            -58 => CompilerError::GeneralExecutionError("OpenCL Err -58: CL_INVALID_EVENT"),
            -59 => CompilerError::GeneralExecutionError("OpenCL Err -59: CL_INVALID_OPERATION"),
            -61 => CompilerError::GeneralExecutionError("OpenCL Err -61: CL_INVALID_BUFFER_SIZE"),
            -63 => {
                CompilerError::GeneralExecutionError("OpenCL Err -63: CL_INVALID_GLOBAL_WORK_SIZE")
            }
            -64 => CompilerError::GeneralExecutionError("OpenCL Err -64: CL_INVALID_PROPERTY"),
            -72 => CompilerError::GeneralExecutionError(
                "OpenCL Err -72: CL_MAX_SIZE_RESTRICTION_EXCEEDED",
            ),
            _ => CompilerError::GeneralExecutionError("OpenCL Err: UNKNOWN ERROR"),
        });
    }
}
