#![allow(non_camel_case_types)]

use crate::{index_map::IndexMap, runtime::{
    ir::{IRDType, IRKernel, IROp, Scope}, node::{BOp, UOp}
}};
use std::{
    ffi::{c_void, CString}, ptr
};

use super::DeviceInfo;

#[derive(Debug)]
pub(crate) struct OpenCLBackend {
    context: *mut c_void,
    queues: Box<[*mut c_void]>,
    queue_size: Box<[u8]>,
    queue_id: usize,
}

// OpenCL does not have the concept of memory pools,
// so we simply say it is all in one memory pool
#[derive(Debug)]
pub(crate) struct OpenCLMemoryPool {
    total_bytes: usize,
    free_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct OpenCLBuffer {
    ptr: *mut c_void,
    byte_size: usize,
}

#[derive(Debug)]
pub(crate) struct OpenCLDevice {
    ptr: *mut c_void,
    compute: usize,
    dev_info: DeviceInfo,
}

#[derive(Debug)]
pub(crate) struct OpenCLProgram {
    name: String,
    program: *mut c_void,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    //args_read_only: Vec<bool>,
}

// Event associated with program launch
#[derive(Debug)]
pub(crate) struct OpenCLEvent {
    ptr: *mut c_void,
}

unsafe impl Send for OpenCLBackend {}
unsafe impl Send for OpenCLMemoryPool {}
unsafe impl Send for OpenCLBuffer {}
unsafe impl Send for OpenCLDevice {}
unsafe impl Send for OpenCLProgram {}
unsafe impl Send for OpenCLEvent {}

impl OpenCLDevice {
    pub(crate) fn compute(&self) -> usize {
        self.compute
    }

    pub(crate) fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    // Memory pool id out of OpenCLMemoryPools
    pub(crate) fn memory_pool_id(&self) -> usize {
        0
    }
}

impl OpenCLBackend {
    pub(crate) fn new() -> Result<(Self, Vec<OpenCLMemoryPool>, Vec<OpenCLDevice>), OpenCLError> {
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
                String::from_utf8(get_device_data(*dev, CL_DEVICE_NAME)?).unwrap()
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
        let mut total_bytes = 0;
        let mut devices = Vec::new();
        for dev in device_ids.iter().copied() {
            let max_work_item_dims = u32::from_ne_bytes(
                get_device_data(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)?
                    .try_into()
                    .unwrap(),
            ) as usize;
            let mwis = get_device_data(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES)?;
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
            total_bytes += u64::from_ne_bytes(get_device_data(dev, CL_DEVICE_GLOBAL_MEM_SIZE)?.try_into().unwrap()) as usize;

            let dev_info = DeviceInfo {
                max_work_item_sizes,
                max_work_group_size: usize::from_ne_bytes(
                    get_device_data(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE)?
                        .try_into()
                        .unwrap(),
                ),
                preferred_vector_size: u32::from_ne_bytes(
                    get_device_data(dev, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)?
                        .try_into()
                        .unwrap(),
                ) as usize
                    * 4,
                f16_support: true,
                f64_support: true,
                fmadd: true,
                page_size: u32::from_ne_bytes(
                    get_device_data(dev, CL_DEVICE_MEM_BASE_ADDR_ALIGN)?
                        .try_into()
                        .unwrap(),
                ) as usize
                    / 8,
                local_memory: true,
                local_mem_size: u64::from_ne_bytes(
                    get_device_data(dev, CL_DEVICE_LOCAL_MEM_SIZE)?.try_into()
                        .unwrap(),
                ) as usize,
                num_registers: 128, // We can only guess or have a map of concrete hardware and respective register counts
                wmma: false,
                tensor_cores: false,
            };
            let compute = 1024*1024*1024*1024; // 1 TFLOPs
            devices.push(OpenCLDevice { ptr: dev, compute, dev_info });
        }
        return Ok((Self {
            context,
            queue_size: vec![0; queues.len()].into_boxed_slice(),
            queues: queues.into_boxed_slice(),
            queue_id: 0,
        }, vec![OpenCLMemoryPool {
            total_bytes,
            free_bytes: total_bytes,
        }], devices));
    }

    pub(crate) fn allocate_memory(
        &mut self,
        byte_size: usize,
        memory_pool: &mut OpenCLMemoryPool,
    ) -> Result<OpenCLBuffer, OpenCLError> {
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
        memory_pool.free_bytes -= byte_size;
        Ok(OpenCLBuffer { ptr, byte_size })
    }

    pub(crate) fn deallocate_memory(&mut self, memory_pool: &mut OpenCLMemoryPool, buffer: &mut OpenCLBuffer) -> Result<(), OpenCLError> {
        let status = unsafe { clReleaseMemObject(buffer.ptr) };
        check(status, "Unable to free allocated memory")?;
        memory_pool.free_bytes += buffer.byte_size;
        Ok(())
    }

    pub(crate) fn host_to_opencl(
        &mut self,
        src: &[u8],
        dst: &mut OpenCLBuffer,
    ) -> Result<(), OpenCLError> {
        let mut event = ptr::null_mut();
        let status = unsafe {
            clEnqueueWriteBuffer(
                self.queue()?,
                dst.ptr,
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
        check(status, "Unable to finish buffer write event.")?;
        Ok(())
    }

    // Perhaps this can be done directly, for now we go through host
    //pub(crate) fn cuda_to_opencl(&mut self, src: Buffer, dst: Buffer) -> Result<(), OpenCLError> {}

    pub(crate) fn opencl_to_opencl(
        &mut self,
        src: &OpenCLBuffer,
        dst: &mut OpenCLBuffer,
    ) -> Result<(), OpenCLError> {
        let _ = src;
        let _ = dst;
        Ok(())
    }

    pub(crate) fn opencl_to_host(
        &mut self,
        src: &OpenCLBuffer,
        dst: &mut [u8],
    ) -> Result<(), OpenCLError> {
        assert!(!src.ptr.is_null(), "Trying to read null memory. Internal bug.");
        let mut event: *mut c_void = ptr::null_mut();
        let status = unsafe {
            clEnqueueReadBuffer(
                self.queue()?,
                src.ptr,
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

    pub(crate) fn compile_program(
        &mut self,
        kernel: &IRKernel,
        device: &OpenCLDevice,
    ) -> Result<OpenCLProgram, OpenCLError> {
        let mut source = String::from("(\n");
        let mut indent = String::from("  ");

        let mut global_work_size = [0; 3];
        let mut local_work_size = [0; 3];

        for op in &kernel.ops[..6] {
            if let IROp::Loop { id, len } = op {
                if id % 2 == 0 {
                    global_work_size[*id as usize / 2] = *len;
                } else {
                    local_work_size[*id as usize / 2] = *len;
                }
            }
        }

        // Declare global variables
        for (id, (_, dtype, read_only)) in kernel.addressables.iter().enumerate() {
            source += &format!(
                "{indent}__global {}{}* g{},\n",
                if *read_only { "const " } else { "" },
                dtype.ocl(),
                id
            );
        }

        source.pop();
        source.pop();
        source += "\n) {\n";

        // Declare register variables
        for (id, (dtype, read_only)) in kernel.registers.iter().enumerate() {
            source += &format!("{indent}{}{} r{id};\n", if *read_only { "const " } else { "" }, dtype.ocl());
        }

        // Add indices for global and local loops
        source += &format!(
            "  r0 = get_group_id(0);   /* 0..{} */\n",
            global_work_size[0]
        );
        source += &format!(
            "  r1 = get_local_id(0);   /* 0..{} */\n",
            local_work_size[0]
        );
        source += &format!(
            "  r2 = get_group_id(1);   /* 0..{} */\n",
            global_work_size[1]
        );
        source += &format!(
            "  r3 = get_local_id(1);   /* 0..{} */\n",
            local_work_size[1]
        );
        source += &format!(
            "  r4 = get_group_id(2);   /* 0..{} */\n",
            global_work_size[2]
        );
        source += &format!(
            "  r5 = get_local_id(2);   /* 0..{} */\n",
            local_work_size[2]
        );

        // Declare register variables

        for op in kernel.ops[6..].iter().copied() {
            match op {
                IROp::Set { z, len: _, value } => {
                    source += &format!("{indent}r{z} = {value};\n");
                }
                IROp::Load { z, x, at, dtype: _ } => {
                    source += &format!("{indent}{z} = {x}[{at}];\n");
                }
                IROp::Store { z, x, at, dtype: _ } => {
                    source += &format!("{indent}{z}[{at}] = {x};\n");
                }
                IROp::Unary { z, x, uop, dtype } => {
                    source += &format!("{indent}{z} = {};\n", match uop {
                        UOp::Cast(_) => format!("({}){x}", dtype.ocl()),
                        UOp::ReLU => format!("max({x}, 0)"),
                        UOp::Neg => format!("-{x}"),
                        UOp::Exp => format!("exp({x})"),
                        UOp::Ln => format!("log({x})"),
                        UOp::Tanh => format!("tanh({x})"),
                        UOp::Inv => format!("1/{x}"),
                        UOp::Sqrt => format!("sqrt({x})"),
                        UOp::Sin => format!("sin({x})"),
                        UOp::Cos => format!("cos({x})"),
                        UOp::Not => format!("!{x}"),
                        UOp::Nonzero => format!("{x} != 0"),
                    });
                }
                IROp::Binary {
                    z,
                    x,
                    y,
                    bop,
                    dtype: _,
                } => {
                    source += &format!("{indent}{z} = {};\n", match bop {
                        BOp::Add => format!("{x} + {y}"),
                        BOp::Sub => format!("{x} - {y}"),
                        BOp::Mul => format!("{x} * {y}"),
                        BOp::Div => format!("{x} / {y}"),
                        BOp::Pow => format!("pow({x}, {y})"),
                        BOp::Cmplt => format!("{x} < {y}"),
                        BOp::Max => format!("max({x}, {y})"),
                    });
                }
                IROp::MAdd { z, a, b, c, dtype: _ } => {
                    source += &format!("{indent}{z} = {a} * {b} + {c};\n");
                }
                IROp::Loop { id, len } => {
                    source += &format!(
                        "{indent}for (unsigned int r{id} = 0; r{id} < {len}; r{id} += 1) {{\n"
                    );
                    indent += "  ";
                }
                IROp::EndLoop => {
                    indent.pop();
                    indent.pop();
                    source += &format!("{indent}}}\n");
                }
                IROp::Barrier { scope } => {
                    source += &format!(
                        "{indent}barrier(CLK_{}AL_MEM_FENCE);\n",
                        match scope {
                            Scope::Global => "GLOB",
                            Scope::Local => "LOC",
                            Scope::Register => panic!(),
                        }
                    );
                }
            }
        }
        source += "}\n";

        Ok(OpenCLProgram::compile_from_source(
            &source,
            self.context,
            &[device.ptr],
            global_work_size,
            local_work_size,
        )?)
    }

    pub(crate) fn launch_program<'a>(
        &mut self,
        program: &mut OpenCLProgram,
        buffers: &mut IndexMap<OpenCLBuffer>,
        args: &[usize],
    ) -> Result<OpenCLEvent, OpenCLError> {
        //#[cfg(feature = "debug1")]
        //libc_print::libc_println!("{:?}", self.load_memory::<f32>(&args[0], 4).unwrap());
        //#[cfg(not(feature = "debug1"))]
        let program_name = &CString::new(program.name.clone()).unwrap();
        let mut status = CL_SUCCESS;
        let kernel =
            unsafe { clCreateKernel(program.program, program_name.as_ptr().cast(), &mut status) };
        check(status, "Unable to create kernel.")?;
        let mut i = 0;
        for arg in args {
            let arg = &mut buffers[*arg];
            // This is POINTER MAGIC. Be careful.
            let ptr: *const _ = &arg.ptr;
            status = unsafe {
                clSetKernelArg(kernel, i, core::mem::size_of::<*mut c_void>(), ptr.cast())
            };
            check(status, "Unable to set kernel arg.")?;
            i += 1;
        }
        let mut global_work_size = program.global_work_size;
        for (i, lwd) in program.local_work_size.iter().enumerate() {
            global_work_size[i] *= lwd;
        }
        let local_work_size = program.local_work_size;
        let mut event: *mut c_void = ptr::null_mut();
        let status = unsafe {
            clEnqueueNDRangeKernel(
                self.queue()?,
                kernel,
                u32::try_from(global_work_size.len()).unwrap(),
                ptr::null(),
                global_work_size.as_ptr(),
                local_work_size.as_ptr(),
                0,
                ptr::null(),
                &mut event,
            )
        };
        check(status, "Unable to enqueue kernel.")?;
        return Ok(OpenCLEvent { ptr: event });
    }

    pub(crate) fn finish_event(&mut self, event: OpenCLEvent) -> Result<(), OpenCLError> {
        let status = unsafe { clWaitForEvents(1, (&[event]).as_ptr().cast()) };
        check(status, "Unable to finish program.")?;
        Ok(())
    }

    pub(crate) fn release_program(&mut self, program: OpenCLProgram) -> Result<(), OpenCLError> {
        let status = unsafe { clReleaseProgram(program.program) };
        check(status, "Unable to release program")?;
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

impl IRDType {
    pub(crate) fn ocl(&self) -> &str {
        return match self {
            #[cfg(feature = "half")]
            IRDType::BF16 => panic!("BF16 is not native to OpenCL, workaround is WIP."),
            #[cfg(feature = "half")]
            IRDType::F16 => "half",
            IRDType::F32 => "float",
            IRDType::F64 => "double",
            #[cfg(feature = "complex")]
            IRDType::CF32 => panic!("Not native to OpenCL, workaround is WIP"),
            #[cfg(feature = "complex")]
            IRDType::CF64 => panic!("Not native to OpenCL, workaround is WIP"),
            IRDType::U8 => "unsigned char",
            IRDType::I8 => "char",
            IRDType::I16 => "short",
            IRDType::I32 => "int",
            IRDType::I64 => "long",
            IRDType::Bool => "bool",
            IRDType::Idx => "unsigned int",
        };
    }
}

impl OpenCLMemoryPool {
    pub(crate) fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    pub(crate) fn free_bytes(&self) -> usize {
        self.free_bytes
    }
}

fn check(status: cl_int, info: &str) -> Result<(), OpenCLError> {
    if status == CL_SUCCESS {
        return Ok(());
    } else {
        return Err(OpenCLError {
            info: info.into(),
            status: status.into(),
        });
    }
}

fn cl_wait_for_events(events: &[*mut c_void]) -> Result<(), OpenCLError> {
    let status = unsafe { clWaitForEvents(1, events.as_ptr().cast()) };
    return check(status, "Unable to finish buffer read event.");
}

impl OpenCLProgram {
    fn compile_from_source(
        source: &str,
        context: *mut c_void,
        devices: &[*mut c_void],
        global_work_size: [usize; 3],
        local_work_size: [usize; 3],
        //args_read_only: Vec<bool>,
    ) -> Result<Self, OpenCLError> {
        let name = format!(
            "k__{}_{}__{}_{}__{}_{}",
            global_work_size[0],
            local_work_size[0],
            global_work_size[1],
            local_work_size[1],
            global_work_size[2],
            local_work_size[2],
        );
        let mut pragma = format!("");
        if source.contains("double") {
            pragma += &"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        }
        let source = format!("{pragma}__kernel void {name}{source}");
        #[cfg(feature = "debug_asm")]
        println!("{source}");
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
        check(status, "Unable to compile program.")?;
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
                Err(status) => check(status, "Unable to get info about failed compilation.")?,
            }
        }
        Ok(Self {
            name,
            program,
            global_work_size,
            local_work_size,
            //args_read_only,
        })
    }
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

fn get_device_data(device: *mut c_void, param_name: cl_uint) -> Result<Vec<u8>, OpenCLError> {
    fn get_size(object: *mut c_void, param_name: cl_uint) -> Result<usize, OpenCLError> {
        let mut size: usize = 0;
        let status = unsafe { clGetDeviceInfo(object, param_name, 0, ptr::null_mut(), &mut size) };
        if CL_SUCCESS != status {
            return Err(OpenCLError { status: status.into(), info: format!("Unable to get device info {param_name}") });
        } else {
            return Ok(size);
        }
    }
    let size = get_size(device, param_name)?;
    fn get_vector(
        object: *mut c_void,
        param_name: cl_uint,
        size: usize,
    ) -> Result<Vec<u8>, OpenCLError> {
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
                return Err(OpenCLError { status: status.into(), info: format!("Unable to get {param_name}") });
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
