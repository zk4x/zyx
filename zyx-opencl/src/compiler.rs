use alloc::{boxed::Box, collections::BTreeSet, ffi::CString, format as f, vec::Vec, string::String};
use cl3::{
    ext::{CL_MEM_READ_ONLY, CL_NON_BLOCKING, CL_PROGRAM_BUILD_LOG},
};
use core::ffi::c_void;
use cl3::ext::CL_MEM_HOST_READ_ONLY;
use zyx_core::{
    compiler::{AST, Op},
    dtype::DType,
    scalar::Scalar,
};
use zyx_core::compiler::ROp;
use zyx_core::error::ZyxError;

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
        std::println!("Compiling source:\n{source}");
        let program = cl3::program::create_program_with_source(context, &[&source]).map_err(|err| ZyxError::BackendError(match err {
            -34 => "Unable to compile program. ERR -34: CL_INVALID_CONTEXT",
            -30 => "Unable to compile program. ERR -30: CL_INVALID_VALUE",
            -5 => "Unable to compile program. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to compile program. ERR -6: CL_OUT_OF_HOST_MEMORY",
            _ => "Unable to compile program. UNKNOWN ERROR",
        }))?;
        let devices = devices.iter().copied().collect::<Vec<*mut c_void>>();
        if let Err(err) = cl3::program::build_program(
            program,
            &devices,
            core::ffi::CStr::from_bytes_with_nul(b"-cl-fast-relaxed-math\0").unwrap(),
            None,
            core::ptr::null_mut(),
        ) {
            return Err(ZyxError::CompileError(Box::new(f!("{err}\n{}", cl3::program::get_program_build_info(program, devices[0], CL_PROGRAM_BUILD_LOG).map_err(|err| ZyxError::BackendError(match err {
                -33 => "Unable to get info about failed compilation. ERR -33: CL_INVALID_DEVICE",
                -30 => "Unable to get info about failed compilation. ERR -30: CL_INVALID_VALUE",
                -44 => "Unable to get info about failed compilation. ERR -44: CL_INVALID_PROGRAM",
                -5 => "Unable to get info about failed compilation. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to get info about failed compilation. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to get info about failed compilation. UNKNOWN ERROR",
            }))?))))
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
        use cl3::ext::CL_DEVICE_TYPE_ALL;
        let platform_ids = cl3::platform::get_platform_ids().map_err(|err| ZyxError::BackendError(match err {
            -30 => "Unable to get OpenCL platform ids. ERR -30: CL_INVALID_VALUE",
            -6 => "Unable to get OpenCL platform ids. ERR -6: CL_OUT_OF_HOST_MEMORY",
            _ => "Unable to get OpenCL platform ids. UNKNOWN ERROR",
        }))?;
        let Some(platform) = platform_ids.get(0) else {
            return Err(ZyxError::BackendError("There are no available OpenCL platforms."))
        };
        let platform = *platform;
        #[cfg(feature = "debug1")]
        std::println!(
            "Using OpenCL platform: {}",
            String::from_utf8(cl3::platform::get_platform_data(
                platform,
                cl3::ext::CL_PLATFORM_NAME
            ).map_err(|err| ZyxError::BackendError(match err{
                -32 => "Unable to get OpenCL platform name. ERR -32: CL_INVALID_PLATFORM",
                -30 => "Unable to get OpenCL platform name. ERR -30: CL_INVALID_VALUE",
                -6 => "Unable to get OpenCL platform name. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to get OpenCL platform name. UNKNOWN ERROR",
            }))?).unwrap()
        );
        let device_ids = cl3::device::get_device_ids(platform, CL_DEVICE_TYPE_ALL).map_err(|err| ZyxError::BackendError(match err {
            -32 => "Unable to get OpenCL device ids. ERR -32: CL_INVALID_PLATFORM",
            -31 => "Unable to get OpenCL device ids. ERR -31: CL_INVALID_DEVICE_TYPE",
            -30 => "Unable to get OpenCL device ids. ERR -30: CL_INVALID_VALUE",
            -1 => "Unable to get OpenCL device ids. ERR -1: CL_DEVICE_NOT_FOUND",
            -5 => "Unable to get OpenCL device ids. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to get OpenCL device ids. ERR -6: CL_OUT_OF_HOST_MEMORY",
            _ => "Unable to get OpenCL device ids. UNKNOWN ERROR",
        }))?;
        #[cfg(feature = "debug1")]
        std::println!("Using devices:");
        #[cfg(feature = "debug1")]
        for dev in &device_ids {
            std::println!(
                "{}",
                String::from_utf8(cl3::device::get_device_data(
                    *dev,
                    cl3::ext::CL_DEVICE_NAME
                ).map_err(|err| ZyxError::BackendError(match err {
                    -33 => "Unable to get OpenCL device name. ERR -33: CL_INVALID_DEVICE",
                    -30 => "Unable to get OpenCL device name. ERR -30: CL_INVALID_VALUE",
                    -5 => "Unable to get OpenCL device name. ERR -5: CL_OUT_OF_RESOURCES",
                    -6 => "Unable to get OpenCL device name. ERR -6: CL_OUT_OF_HOST_MEMORY",
                    _ => "Unable to get OpenCL device name. UNKNOWN ERROR",
                }))?).unwrap()
            );
        }
        let context = cl3::context::create_context(
            &device_ids,
            core::ptr::null(),
            None,
            core::ptr::null_mut(),
        ).map_err(|err| ZyxError::BackendError(match err {
            -32 => "Unable to create OpenCL context. ERR -32: CL_INVALID_PLATFORM",
            -64 => "Unable to create OpenCL context name. ERR -64: CL_INVALID_PROPERTY",
            -30 => "Unable to crate OpenCL context name. ERR -30: CL_INVALID_VALUE",
            -33 => "Unable to create OpenCL context name. ERR -33: CL_INVALID_DEVICE",
            -59 => "Unable to create OpenCL context name. ERR -59: CL_INVALID_OPERATION",
            -2 => "Unable to create OpenCL context. ERR -32: CL_DEVICE_NOT_AVAILABLE",
            -5 => "Unable to create OpenCL context. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to create OpenCL context. ERR -6: CL_OUT_OF_HOST_MEMORY",
            _ => "Unable to create OpenCL context. UNKNOWN ERROR",
        }))?;
        // This makes our code asynchronous. Creating graph would actually make us 2 times slower (can be optimized),
        // if we couldn't execute kernels asynchronously. We don't need this to be huge. 2 seems to
        // be plenty. And lower values also lower memory usage.
        let queues_per_device: u32 = 8; //device_ids.iter().map(|dev| cl3::device::get_device_info(*dev, CL_DEVICE_MAX_ON_DEVICE_QUEUES)?.into()).min()?;
        #[cfg(feature = "debug1")]
        std::println!("Using {queues_per_device} queues per device.");
        let queues = (0..queues_per_device)
            .flat_map(|_| {
                device_ids.iter().map(|dev| {
                    unsafe { cl3::command_queue::create_command_queue(context, *dev, 0) }
                })
            })
            .collect::<Result<Box<[*mut c_void]>, i32>>().map_err(|err| ZyxError::BackendError(match err {
            -34 => "Unable to create command queue. ERR -34: CL_INVALID_CONTEXT",
            -33 => "Unable to create command queue. ERR -33: CL_INVALID_DEVICE",
            -30 => "Unable to create command queue. ERR -30: CL_INVALID_VALUE",
            -35 => "Unable to create command queue. ERR -35: CL_INVALID_QUEUE_PROPERTIES",
            -5 => "Unable to create command queue. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to create command queue. ERR -6: CL_OUT_OF_HOST_MEMORY",
            _ => "Unable to create command queue. UNKNOWN ERROR",
        }))?;
        let mut devices = BTreeSet::new();
        for dev in device_ids {
            devices.insert(dev);
        }
        Ok(Self {
            context,
            devices,
            queues,
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
    fn drop_program(&mut self, program: &mut Self::Program) -> Result<(), ZyxError> {
        unsafe { cl3::program::release_program(program.program) }.map_err(|err| ZyxError::BackendError(match err {
            -5 => "Unable to release program. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to release program. ERR -6: CL_OUT_OF_HOST_MEMORY",
            -44 => "Unable to release program. ERR -44: CL_INVALID_PROGRAM",
            _ => "Unable to release program. UNKNOWN ERROR",
        }))
    }
    fn store<T>(&mut self, iter: impl Iterator<Item = T>) -> Result<Self::Buffer, ZyxError> {
        // TODO we can do buffered load, with buffer of say 1 MB size in RAM and offset write buffer
        let data: Vec<T> = iter.collect();
        let size = data.len() * core::mem::size_of::<T>();
        let mem = unsafe {
            cl3::memory::create_buffer(self.context, CL_MEM_READ_ONLY, size, core::ptr::null_mut())
        }.map_err(|err| ZyxError::BackendError(match err {
            -34 => "Unable to create buffer. ERR -34: CL_INVALID_CONTEXT",
            -64 => "Unable to create buffer. ERR -64: CL_INVALID_PROPERTY",
            -30 => "Unable to create buffer. ERR -30: CL_INVALID_VALUE",
            -61 => "Unable to create buffer. ERR -61: CL_INVALID_BUFFER_SIZE",
            -4 => "Unable to create buffer. ERR -4: CL_MEM_OBJECT_ALLOCATION_FAILURE",
            -5 => "Unable to create buffer. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to create buffer. ERR -6: CL_OUT_OF_HOST_MEMORY",
            _ => "Unable to create buffer. UNKNOWN ERROR",
        }))?;
        let event = unsafe {
            cl3::command_queue::enqueue_write_buffer(
                self.queue(),
                mem,
                CL_NON_BLOCKING,
                0,
                size,
                data.as_ptr().cast(),
                0,
                core::ptr::null(),
            )
        }.map_err(|err| ZyxError::BackendError(match err {
            -36 => "Unable to write buffer. ERR -36: CL_INVALID_COMMAND_QUEUE",
            -34 => "Unable to write buffer. ERR -34: CL_INVALID_CONTEXT",
            -38 => "Unable to write buffer. ERR -38: CL_INVALID_MEM_OBJECT",
            -30 => "Unable to write buffer. ERR -30: CL_INVALID_VALUE",
            -57 => "Unable to write buffer. ERR -57: CL_INVALID_EVENT_WAIT_LIST",
            -13 => "Unable to write buffer. ERR -13: CL_MISALIGNED_SUB_BUFFER_OFFSET",
            -14 => "Unable to write buffer. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
            -4 => "Unable to write buffer. ERR -4: CL_MEM_OBJECT_ALLOCATION_FAILURE",
            -59 => "Unable to write buffer. ERR -59: CL_INVALID_OPERATION",
            -5 => "Unable to write buffer. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to write buffer. ERR -6: CL_OUT_OF_HOST_MEMORY",
            _ => "Unable to write buffer. UNKNOWN ERROR",
        }))?;
        cl3::event::wait_for_events(&[event]).map_err(|err| ZyxError::BackendError(match err {
            -30 => "Unable to finish buffer write event. ERR -30: CL_INVALID_VALUE",
            -34 => "Unable to finish buffer write event. ERR -34: CL_INVALID_CONTEXT",
            -58 => "Unable to finish buffer write event. ERR -58: CL_INVALID_EVENT",
            -14 => "Unable to finish buffer write event. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
            -5 => "Unable to finish buffer write event. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to finish buffer write event. ERR -6: CL_OUT_OF_MEMORY",
            _ => "Unable to finish buffer write event. UNKNOWN ERROR",
        }))?;
        Ok(Self::Buffer { mem, event })
    }

    fn load<T: Scalar>(&mut self, buffer: &Self::Buffer, numel: usize) -> Result<Vec<T>, ZyxError> {
        let mut data: Vec<T> = Vec::with_capacity(numel);
        cl3::event::wait_for_events(&[buffer.event]).map_err(|err| ZyxError::BackendError(match err {
            -30 => "Unable to finish buffer event. ERR -30: CL_INVALID_VALUE",
            -34 => "Unable to finish buffer event. ERR -34: CL_INVALID_CONTEXT",
            -58 => "Unable to finish buffer event. ERR -58: CL_INVALID_EVENT",
            -14 => "Unable to finish buffer event. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
            -5 => "Unable to finish buffer event. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to finish buffer event. ERR -6: CL_OUT_OF_MEMORY",
            _ => "Unable to finish buffer event. UNKNOWN ERROR",
        }))?;
        let event = unsafe {
            cl3::command_queue::enqueue_read_buffer(
                self.queue(),
                buffer.mem,
                CL_NON_BLOCKING,
                0,
                numel * T::byte_size(),
                data.as_mut_ptr().cast(),
                0,
                core::ptr::null(),
                // TODO why does this not work?
                //&[mem.event] as *const *mut c_void,
            )
        }.map_err(|err| ZyxError::BackendError(match err {
            -36 => "Unable to read buffer. ERR -36: CL_INVALID_COMMAND_QUEUE",
            -34 => "Unable to read buffer. ERR -34: CL_INVALID_CONTEXT",
            -38 => "Unable to read buffer. ERR -38: CL_INVALID_MEM_OBJECT",
            -30 => "Unable to read buffer. ERR -30: CL_INVALID_VALUE",
            -57 => "Unable to read buffer. ERR -57: CL_INVALID_EVENT_WAIT_LIST",
            -13 => "Unable to read buffer. ERR -13: CL_MISALIGNED_SUB_BUFFER_OFFSET",
            -14 => "Unable to read buffer. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
            -4 => "Unable to read buffer. ERR -4: CL_MEM_OBJECT_ALLOCATION_FAILURE",
            -59 => "Unable to read buffer. ERR -59: CL_INVALID_OPERATION",
            -5 => "Unable to read buffer. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to read buffer. ERR -6: CL_OUT_OF_HOST_MEMORY",
            _ => "Unable to read buffer. UNKNOWN ERROR",
        }))?;
        cl3::event::wait_for_events(&[event]).map_err(|err| ZyxError::BackendError(match err {
            -30 => "Unable to finish buffer read event. ERR -30: CL_INVALID_VALUE",
            -34 => "Unable to finish buffer read event. ERR -34: CL_INVALID_CONTEXT",
            -58 => "Unable to finish buffer read event. ERR -58: CL_INVALID_EVENT",
            -14 => "Unable to finish buffer read event. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
            -5 => "Unable to finish buffer read event. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to finish buffer read event. ERR -6: CL_OUT_OF_MEMORY",
            _ => "Unable to finish buffer read event. UNKNOWN ERROR",
        }))?;
        // We are now done reading, so the vec is initialized
        unsafe { data.set_len(numel) }
        Ok(data)
    }

    fn drop_buffer(&mut self, buffer: &mut Self::Buffer) -> Result<(), ZyxError> {
        unsafe { cl3::memory::release_mem_object(buffer.mem) }.map_err(|err| ZyxError::BackendError(match err {
            -38 => "Unable to release buffer. ERR -38: CL_INVALID_MEM_OBJECT",
            -5 => "Unable to release buffer. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to release buffer. ERR -6: CL_OUT_OF_HOST_MEMORY",
            _ => "Unable to release buffer. UNKNOWN ERROR",
        }))?;
        unsafe { cl3::event::release_event(buffer.event) }.map_err(|err| ZyxError::BackendError(match err {
            -58 => "Unable to release event. ERR -58: CL_INVALID_EVENT",
            -5 => "Unable to release event. ERR -5: CL_OUT_OF_RESOURCES",
            -6 => "Unable to release event. ERR -6: CL_OUT_OF_HOST_MEMORY",
            _ => "Unable to release event. UNKNOWN ERROR",
        }))?;
        Ok(())
    }

    fn launch(&mut self, program: &Self::Program, args: &[&Self::Buffer]) -> Result<Self::Buffer, ZyxError> {
        let kernel =
            cl3::kernel::create_kernel(program.program, &CString::new(program.name.clone()).unwrap()).map_err(|err| ZyxError::BackendError(match err {
                -44 => "Unable to create kernel. ERR -: CL_INVALID_PROGRAM",
                -45 => "Unable to create kernel. ERR -: CL_INVALID_PROGRAM_EXECUTABLE",
                -46 => "Unable to create kernel. ERR -: CL_INVALID_KERNEL_NAME",
                -47 => "Unable to create kernel. ERR -: CL_INVALID_KERNEL_DEFINITION",
                -30 => "Unable to create kernel. ERR -: CL_INVALID_VALUE",
                -5 => "Unable to create kernel. ERR -: CL_OUT_OF_RESOURCES",
                -6 => "Unable to create kernel. ERR -: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to create kernel. UNKNOWN ERROR",
            }))?;
        let mem = unsafe {
            cl3::memory::create_buffer(
                self.context,
                CL_MEM_READ_ONLY | CL_MEM_HOST_READ_ONLY,
                program.res_byte_size,
                core::ptr::null_mut(),
            )}.map_err(|err| ZyxError::BackendError(match err {
                -34 => "Unable to create kernel output buffer. ERR -34: CL_INVALID_CONTEXT",
                -64 => "Unable to create kernel output buffer. ERR -64: CL_INVALID_PROPERTY",
                -30 => "Unable to create kernel output buffer. ERR -30: CL_INVALID_VALUE",
                -61 => "Unable to create kernel output buffer. ERR -61: CL_INVALID_BUFFER_SIZE",
                -4 => "Unable to create kernel output buffer. ERR -4: CL_MEM_OBJECT_ALLOCATION_FAILURE",
                -5 => "Unable to create kernel output buffer. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to create kernel output buffer. ERR -6: CL_OUT_OF_HOST_MEMORY",
                _ => "Unable to create kernel output buffer. UNKNOWN ERROR",
            }))?;
        let ptr: *const _ = &mem;
        let kernel_arg_err_handler = |err| ZyxError::BackendError(match err {
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
        });
        unsafe {
            cl3::kernel::set_kernel_arg(kernel, 0, core::mem::size_of::<*mut c_void>(), ptr.cast())
        }.map_err(kernel_arg_err_handler)?;
        let mut events = Vec::new();
        let mut i = 1;
        for arg in args {
            let (buffer, event) = (arg.mem, arg.event);
            events.push(event);
            // This is POINTER MAGIC. Be careful.
            let ptr: *const _ = &buffer;
            unsafe {
                cl3::kernel::set_kernel_arg(
                    kernel,
                    i,
                    core::mem::size_of::<*mut c_void>(),
                    ptr.cast(),
                )
            }.map_err(kernel_arg_err_handler)?;
            i += 1;
        }
        #[cfg(feature = "debug1")]
        let begin = std::time::Instant::now();
        let event = unsafe {
            cl3::command_queue::enqueue_nd_range_kernel(
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
            )
        }
        .expect("could not execute opencl kernel.");
        #[cfg(feature = "debug1")]
        {
            cl3::event::wait_for_events(&[event]).map_err(|err| ZyxError::BackendError(match err {
                -30 => "Unable to finish kernel execution event. ERR -30: CL_INVALID_VALUE",
                -34 => "Unable to finish kernel execution event. ERR -34: CL_INVALID_CONTEXT",
                -58 => "Unable to finish kernel execution event. ERR -58: CL_INVALID_EVENT",
                -14 => "Unable to finish kernel execution event. ERR -14: CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
                -5 => "Unable to finish kernel execution event. ERR -5: CL_OUT_OF_RESOURCES",
                -6 => "Unable to finish kernel execution event. ERR -6: CL_OUT_OF_MEMORY",
                _ => "Unable to finish kernel execution event. UNKNOWN ERROR",
            }))?;
            let elapsed = begin.elapsed().as_millis();
            std::println!(
                "Kernel execution took {elapsed}ms, that is {} GFLOPS",
                (1024u128 * 1024 * 1024 * 2) as f64 / elapsed as f64 / 1000000 as f64
            );
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
    let tile_width = 16;
    let tile_height = 16;
    let global_work_size = alloc::vec![256, 256];
    let local_work_size = alloc::vec![tile_height, tile_width];
    let res_byte_size: usize = global_work_size.iter().product();
    let mut source = f!("(\n  ");
    let mut endl = f!(",\n  ");

    let mut res_id = 0;
    for arg in ast.args() {
        source = f!("{source}__global const {}* data{res_id}{endl}", arg.1.ocl_str());
        res_id += 1;
    }
    source = f!("{source}__global RES_DTYPE* data{res_id}{endl}");
    source.pop();
    source.pop();
    source.pop();
    source.pop();
    source = f!("{source}\n) {{\n  ");

    endl = f!(";\n  ");
    source = f!("{source}int gidx0 = get_group_id(0){endl}");
    source = f!("{source}int gidx1 = get_group_id(1){endl}");
    source = f!("{source}int lidx0 = get_local_id(0){endl}");
    source = f!("{source}int lidx1 = get_local_id(1){endl}");
    source = f!("{source}int idx0 = (gidx0*{tile_height} + lidx0)*{} + gidx1*{tile_width} + lidx1{endl}", global_work_size[1]);
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
            Op::Exp(x) => f!("{dtype} var{nid}[] = exp(var{x}[])"),
            _ => todo!(),
        };
        source = f!("{source}{res}{endl}");
        nid += 1;
    }
    source = source.replace("RES_DTYPE", &f!("{dtype}"));
    source = f!("{source}data{res_id}[idx0] = var{}{endl}", nid-1);
    source.pop();
    source.pop();
    source = f!("{source}}}");
    (source, global_work_size, local_work_size, res_byte_size * DType::from_ocl_str(dtype).byte_size())
}

/// Reduce kernel
fn compile_r_kernel(_ast: &AST) -> (String, Vec<usize>, Vec<usize>, usize) {
    todo!()
}

#[test]
fn exp_test() -> Result<(), ZyxError> {
    let dev = crate::device()?;
    let x = dev.randn([2, 3], crate::DType::F32);
    let y = x.exp();
    let _y_vec: Vec<f32> = y.to_vec()?;
    //panic!("{y_vec:?}");
    Ok(())
}
