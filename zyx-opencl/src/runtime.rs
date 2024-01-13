use alloc::{boxed::Box, collections::BTreeSet, ffi::CString, format as f, vec::Vec};
use cl3::{
    error_codes::ClError,
    ext::{CL_MEM_READ_ONLY, CL_NON_BLOCKING, CL_PROGRAM_BUILD_LOG},
};
use core::ffi::c_void;
use zyx_core::compiled_backend::AST;
use zyx_core::dtype::DType;
use zyx_core::scalar::Scalar;

trait OpenCLDType {
    fn ocl_str(self) -> &'static str;
}

impl OpenCLDType for DType {
    fn ocl_str(self) -> &'static str {
        match self {
            DType::F32 => "float",
            DType::I32 => "int",
        }
    }
}

pub struct Buffer {
    mem: *mut c_void,
    event: *mut c_void,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { cl3::memory::release_mem_object(self.mem) }.unwrap();
        unsafe { cl3::event::release_event(self.event) }.unwrap();
    }
}

pub struct Program {
    program: *mut c_void,
    global_work_size: Box<[usize]>,
    local_work_size: Box<[usize]>,
    res_byte_size: usize,
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe { cl3::program::release_program(self.program) }.unwrap();
    }
}

impl Program {
    pub fn compile(
        source: &str,
        context: *mut c_void,
        devices: &BTreeSet<*mut c_void>,
        global_work_size: &[usize],
        local_work_size: &[usize],
        res_byte_size: usize,
    ) -> Self {
        let name = f!(
            "e__{}__{}",
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
        let program = cl3::program::create_program_with_source(context, &[&source]).unwrap();
        let devices = devices.iter().copied().collect::<Vec<*mut c_void>>();
        if let Err(er) = cl3::program::build_program(
            program,
            &devices,
            core::ffi::CStr::from_bytes_with_nul(b"-cl-fast-relaxed-math\0").unwrap(),
            None,
            core::ptr::null_mut(),
        ) {
            panic!(
                "Compilation failed with error {er}:\n{}",
                cl3::program::get_program_build_info(program, devices[0], CL_PROGRAM_BUILD_LOG)
                    .unwrap()
            );
        };
        Self {
            program,
            global_work_size: global_work_size.iter().copied().collect(),
            local_work_size: local_work_size.iter().copied().collect(),
            res_byte_size,
        }
    }
}

pub(crate) struct Runtime {
    context: *mut c_void,
    devices: BTreeSet<*mut c_void>,
    queues: Box<[*mut c_void]>,
    queue_id: usize,
}

impl Runtime {
    pub(crate) fn new() -> Result<Self, ClError> {
        use cl3::ext::CL_DEVICE_TYPE_ALL;
        let platform_ids = cl3::platform::get_platform_ids()?;
        let Some(platform) = platform_ids.get(0) else {
            panic!("There are no available OpenCL platforms.");
        };
        let platform = *platform;
        #[cfg(feature = "debug1")]
        std::println!(
            "Using OpenCL platform: {}",
            alloc::string::String::from_utf8(cl3::platform::get_platform_data(
                platform,
                cl3::ext::CL_PLATFORM_NAME
            )?)
            .unwrap()
        );
        let device_ids = cl3::device::get_device_ids(platform, CL_DEVICE_TYPE_ALL)?;
        #[cfg(feature = "debug1")]
        std::println!("Using devices:");
        #[cfg(feature = "debug1")]
        for dev in &device_ids {
            std::println!(
                "{}",
                alloc::string::String::from_utf8(cl3::device::get_device_data(
                    *dev,
                    cl3::ext::CL_DEVICE_NAME
                )?)
                .unwrap()
            );
        }
        let context = cl3::context::create_context(
            &device_ids,
            core::ptr::null(),
            None,
            core::ptr::null_mut(),
        )?;
        // This makes our code asynchronous. Creating graph would actually make us 2 times slower (can be optimized),
        // if we couldn't execute kernels asynchronously. We don't need this to be huge. 2 seems to
        // be plenty. And lower values also lower memory usage.
        let queues_per_device: u32 = 8; //device_ids.iter().map(|dev| cl3::device::get_device_info(*dev, CL_DEVICE_MAX_ON_DEVICE_QUEUES).unwrap().into()).min().unwrap();
        #[cfg(feature = "debug1")]
        std::println!("Working with {queues_per_device} queues on each device.");
        let queues = (0..queues_per_device)
            .flat_map(|_| {
                device_ids.iter().map(|dev| {
                    unsafe { cl3::command_queue::create_command_queue(context, *dev, 0) }.unwrap()
                })
            })
            .collect();
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

impl zyx_core::compiled_backend::Runtime for Runtime {
    type Buffer = Buffer;
    type Program = Program;
    fn store<T>(&mut self, iter: Box<dyn Iterator<Item = T>>) -> Self::Buffer {
        let data: Vec<T> = iter.collect();
        let size = data.len() * core::mem::size_of::<T>();
        let mem = unsafe {
            cl3::memory::create_buffer(self.context, CL_MEM_READ_ONLY, size, core::ptr::null_mut())
        }
        .unwrap();
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
        }
        .unwrap();
        cl3::event::wait_for_events(&[event]).unwrap();
        Self::Buffer { mem, event }
    }

    fn load<T: Scalar>(&mut self, buffer: &Self::Buffer, numel: usize) -> Vec<T> {
        let mut data: Vec<T> = Vec::with_capacity(numel);
        cl3::event::wait_for_events(&[buffer.event]).unwrap();
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
        }
        .unwrap();
        cl3::event::wait_for_events(&[event]).unwrap();
        // We are now done reading, so the vec is initialized
        unsafe { data.set_len(numel) }
        data
    }

    fn compile(&mut self, ast: &AST) -> Self::Program {
        let global_work_size = [];
        let local_work_size = [];
        let res_byte_size = 0;
        let mut source = f!("(\n  ");
        let mut endl = f!(",\n  ");

        for (i, arg) in ast.args().iter().enumerate() {
            source = f!("{source}__global {}* data{i}{endl}", arg.1.ocl_str());
        }

        endl = f!(";\n  ");
        source.pop();
        source.pop();
        source = f!("{source}) {{\n}}");

        Program::compile(&source, self.context, &self.devices, &global_work_size, &local_work_size, res_byte_size)
    }

    fn launch(&mut self, program: &Self::Program, args: &[&Self::Buffer]) -> Self::Buffer {
        let name = "kernel0";
        let kernel =
            cl3::kernel::create_kernel(program.program, &CString::new(name).unwrap()).unwrap();
        let mem = unsafe {
            cl3::memory::create_buffer(
                self.context,
                CL_MEM_READ_ONLY,
                program.res_byte_size,
                core::ptr::null_mut(),
            )
        }
        .unwrap();
        let ptr: *const _ = &mem;
        unsafe {
            cl3::kernel::set_kernel_arg(kernel, 0, core::mem::size_of::<*mut c_void>(), ptr.cast())
        }
        .unwrap();
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
            }
            .unwrap();
            i += 1;
        }

        //#[cfg(feature = "debug1")]
        //let begin = std::time::Instant::now();
        let event = unsafe {
            cl3::command_queue::enqueue_nd_range_kernel(
                self.queue(),
                kernel,
                u32::try_from(program.global_work_size.len()).unwrap(),
                core::ptr::null(),
                program.global_work_size.as_ptr(),
                // Following line is funny. Removing as_ref causes local_work_size to be moved,
                // thus as_ptr would be use after free
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
        //#[cfg(feature = "debug1")]
        cl3::event::wait_for_events(&[event]).unwrap();
        //#[cfg(feature = "debug1")]
        //let elapsed = begin.elapsed().as_millis();
        //std::println!(
        //"Kernel execution took {elapsed}ms, that is {} GFLOPS",
        //(1024u128 * 1024 * 1024 * 2) as f64 / elapsed as f64 / 1000000 as f64
        //);
        Buffer { mem, event }
    }
}

#[test]
fn exp_test() -> Result<(), ClError> {
    let dev = crate::default_device()?;
    let x = dev.randn([2, 3], crate::DType::F32);
    let y = x.exp();
    let y_vec: Vec<f32> = y.to_vec();
    panic!("{y_vec:?}");
    Ok(())
}
