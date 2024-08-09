#![allow(non_camel_case_types)]

use crate::dtype::DType;
use crate::scalar::Scalar;
use core::ffi::c_void;
use core::ptr;
use std::collections::BTreeSet;
use std::ffi::CString;
use std::format as f;

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

impl DType {
    pub(crate) fn ocl(&self) -> &str {
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
            DType::Bool => "bool",
        };
    }
}

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

pub(crate) struct OpenCLDevice {
    context: *mut c_void,
    devices: BTreeSet<*mut c_void>,
    queues: Box<[*mut c_void]>,
    queue_size: Box<[u8]>,
    queue_id: usize,
}

// TODO we must ensure that this is OK
// Pointers in these structs are OpenCL pointers,
// so they should stay valid no matter the thread.
unsafe impl Send for OpenCLDevice {}
unsafe impl Send for OpenCLBuffer {}
unsafe impl Send for OpenCLProgram {}

impl OpenCLDevice {
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

impl Drop for OpenCLDevice {
    fn drop(&mut self) {
        /*#[cfg(feature = "CL_VERSION_1_2")]
        for device in &mut self.devices {
            let status = unsafe { clReleaseDevice(device) };
            handle_status(status, "Unable to release device.", &[-33, -5, -6]).unwrap();
            }*/
        let status = unsafe { clReleaseContext(self.context) };
        check(status, "Unable to release context.").unwrap();
    }
}

impl Device for OpenCLDevice {
    type Buffer = OpenCLBuffer;
    type Program = OpenCLProgram;
    type Error = OpenCLError;

    fn initialize() -> Result<Self, OpenCLError> {
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
            });
        };
        let platform = *platform;
        #[cfg(feature = "debug1")]
        println!(
            "Using OpenCL platform: {}",
            String::from_utf8(get_platform_data(platform, CL_PLATFORM_NAME)?).unwrap()
        );
        let device_ids = get_device_ids(platform, CL_DEVICE_TYPE_ALL)
            .map_err(|err| check(err, "Unable to get OpenCL device ids").err().unwrap())?;
        #[cfg(feature = "debug1")]
        println!("Using devices:");
        #[cfg(feature = "debug1")]
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
        #[cfg(feature = "debug1")]
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
        });
    }

    fn hardware_information(&mut self) -> Result<HWInfo, OpenCLError> {
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
            local_memory: true,
            local_mem_size: u64::from_ne_bytes(
                get_device_data(dev, CL_DEVICE_LOCAL_MEM_SIZE)
                    .unwrap()
                    .try_into()
                    .unwrap(),
            ) as usize,
            num_registers: 128, // We can only guess or have a map of concrete hardware and respective register counts
            wmma: false,
            tensor_cores: false,
        });
    }

    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, OpenCLError> {
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
        check(status, "Unable to allocate memory.")?;
        Ok(Self::Buffer {
            memory,
            event: ptr::null_mut(),
        })
    }

    fn store_memory<T: Scalar>(
        &mut self,
        buffer: &mut Self::Buffer,
        data: Vec<T>,
    ) -> Result<(), OpenCLError> {
        //std::println!("Storing");
        // TODO we can also do async stores with iter being &[T], in then we need a way of making
        // sure that the reference stays valid for the whole duration of the copy.
        // TODO we can do batched load, with buffer of say 1 MB size in RAM and offset write buffer
        let status = unsafe {
            clEnqueueWriteBuffer(
                self.queue()?,
                buffer.memory,
                CL_NON_BLOCKING,
                0,
                data.len() * T::byte_size(),
                data.as_ptr().cast(),
                0,
                ptr::null(),
                &mut buffer.event,
            )
        };
        check(status, "Unable to write buffer.")?;
        // Immediattely synchronize because we do not know the lifetime of data
        let status = unsafe { clWaitForEvents(1, (&[buffer.event]).as_ptr().cast()) };
        return check(status, "Unable to finish buffer write event.");
    }

    fn load_memory<T: Scalar>(
        &mut self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, OpenCLError> {
        assert!(
            !buffer.memory.is_null(),
            "Trying to read null memory. Internal bug."
        );
        assert!(
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
        check(status, "Unable to read buffer.")?;
        cl_wait_for_events(&[event])?;
        // We are now done reading, so the vec is initialized
        unsafe { data.set_len(length) }
        return Ok(data);
    }

    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), OpenCLError> {
        let status = unsafe { clReleaseMemObject(buffer.memory) };
        check(status, "Unable to release buffer.")?;
        if buffer.event.is_null() {
            #[cfg(feature = "debug1")]
            println!("Warning: A buffer was allocated, but never used.");
        }
        /*let status = unsafe { clReleaseEvent(buffer.event) };
        handle_status(status, "Unable to release event.", &[-58, -5, -6])?;*/
        return Ok(());
    }

    fn compile_program(&mut self, kernel: &IRKernel) -> Result<Self::Program, OpenCLError> {
        let mut source = String::from("(\n");
        let mut indent = String::from("  ");

        source.pop();
        source.pop();
        source += "\n) {\n";

        // Add indices for global and local loops
        source += &f!(
            "  unsigned int i0 = get_group_id(0);   /* 0..{} */\n",
            todo!()
        );
        source += &f!(
            "  unsigned int i1 = get_local_id(0);   /* 0..{} */\n",
            todo!()
        );
        source += &f!(
            "  unsigned int i2 = get_group_id(1);   /* 0..{} */\n",
            todo!()
        );
        source += &f!(
            "  unsigned int i3 = get_local_id(1);   /* 0..{} */\n",
            todo!()
        );
        source += &f!(
            "  unsigned int i4 = get_group_id(2);   /* 0..{} */\n",
            todo!()
        );
        source += &f!(
            "  unsigned int i5 = get_local_id(2);   /* 0..{} */\n",
            todo!()
        );
        //source += "  unsigned int t0, t1, t2;\n";

        return OpenCLProgram::compile_from_source(
            &source,
            self.context,
            &self.devices,
            todo!(),
            todo!(),
            todo!(),
        );
    }

    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &mut [Self::Buffer],
    ) -> Result<(), OpenCLError> {
        //#[cfg(feature = "debug1")]
        //libc_print::libc_println!("{:?}", self.load_memory::<f32>(&args[0], 4).unwrap());
        //#[cfg(not(feature = "debug1"))]
        let program_name = &CString::new(program.name.clone()).unwrap();
        let mut status = CL_SUCCESS;
        let kernel =
            unsafe { clCreateKernel(program.program, program_name.as_ptr().cast(), &mut status) };
        check(status, "Unable to create kernel.")?;
        let mut events = Vec::new();
        let mut i = 0;
        for arg in &mut *args {
            // Memory that is freshly allocated does not need to be awaited using events
            if !arg.event.is_null() {
                events.push(arg.event);
            }
            // This is POINTER MAGIC. Be careful.
            let ptr: *const _ = &arg.memory;
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
        check(status, "Unable to enqueue kernel.")?;
        #[cfg(feature = "debug1")]
        {
            let status = unsafe { clWaitForEvents(1, (&[event]).as_ptr().cast()) };
            check(status, "Unable to finish kernel execution event.")?;
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

    fn release_program(&mut self, program: Self::Program) -> Result<(), OpenCLError> {
        let status = unsafe { clReleaseProgram(program.program) };
        check(status, "Unable to release program")?;
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
    ) -> Result<Self, OpenCLError> {
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
            args_read_only,
        })
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

#[cfg(feature = "debug1")]
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
