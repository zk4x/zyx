#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use libloading::Library;

use crate::{
    index_map::IndexMap,
    runtime::{
        ir::{IRDType, IRKernel, IROp, Scope},
        node::{BOp, UOp},
    },
};
use std::{
    ffi::{c_void, CString},
    ptr,
    rc::Rc,
};

use super::DeviceInfo;

// OpenCL does not have the concept of memory pools,
// so we simply say it is all in one memory pool
#[derive(Debug)]
pub(crate) struct OpenCLMemoryPool {
    #[allow(unused)]
    total_bytes: usize,
    free_bytes: usize,
    context: *mut c_void,
    queue: *mut c_void,
    // Just to close the connection
    #[allow(unused)]
    library: Rc<Library>,
    // Functions
    clWaitForEvents: unsafe extern "C" fn(cl_uint, *const *mut c_void) -> cl_int,
    clReleaseCommandQueue: unsafe extern "C" fn(*mut c_void) -> cl_int,
    clReleaseContext: unsafe extern "C" fn(*mut c_void) -> cl_int,
    clReleaseMemObject: unsafe extern "C" fn(*mut c_void) -> cl_int,
    clEnqueueReadBuffer: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        usize,
        usize,
        *mut c_void,
        cl_uint,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> cl_int,
    clEnqueueWriteBuffer: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        usize,
        usize,
        *const c_void,
        cl_uint,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> cl_int,
    clCreateBuffer: unsafe extern "C" fn(
        *mut c_void,
        cl_bitfield,
        usize,
        *mut c_void,
        *mut cl_int,
    ) -> *mut c_void,
}

// Ideally we would want Buffer to have lifetime of MemoryPool
// and Program to have lifetime of Device, but self referential
// lifetimes are not in rust, so we have to do manual memory management
// as they did it in stone age.
#[derive(Debug)]
pub(crate) struct OpenCLBuffer {
    ptr: *mut c_void,
    bytes: usize,
    queue: *mut c_void, // This is the queue held by memory pool
}

#[derive(Debug)]
pub(crate) struct OpenCLDevice {
    ptr: *mut c_void,
    context: *mut c_void,
    dev_info: DeviceInfo,
    memory_pool_id: usize,
    queue: *mut c_void,
    // Functions
    clWaitForEvents: unsafe extern "C" fn(cl_uint, *const *mut c_void) -> cl_int,
    clReleaseCommandQueue: unsafe extern "C" fn(*mut c_void) -> cl_int,
    clEnqueueNDRangeKernel: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        *const usize,
        *const usize,
        *const usize,
        cl_uint,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> cl_int,
    clGetProgramBuildInfo: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        usize,
        *mut c_void,
        *mut usize,
    ) -> cl_int,
    clBuildProgram: unsafe extern "C" fn(
        *mut c_void,
        cl_uint,
        *const *mut c_void,
        *const i8,
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>,
        *mut c_void,
    ) -> cl_int,
    clReleaseProgram: unsafe extern "C" fn(*mut c_void) -> cl_int,
    clSetKernelArg: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *const c_void) -> cl_int,
    clCreateKernel: unsafe extern "C" fn(*mut c_void, *const i8, *mut cl_int) -> *mut c_void,
    clGetDeviceInfo:
        unsafe extern "C" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> cl_int,
    clCreateProgramWithSource: unsafe extern "C" fn(
        *mut c_void,
        cl_uint,
        *const *const i8,
        *const usize,
        *mut cl_int,
    ) -> *mut c_void,
}

#[derive(Debug)]
pub(crate) struct OpenCLProgram {
    program: *mut c_void,
    kernel: *mut c_void,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    queue: *mut c_void, // points to device queue
    // Functions
    clSetKernelArg: unsafe extern "C" fn(*mut c_void, cl_uint, usize, *const c_void) -> cl_int,
    clEnqueueNDRangeKernel: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        *const usize,
        *const usize,
        *const usize,
        cl_uint,
        *const *mut c_void,
        *mut *mut c_void,
    ) -> cl_int,
    clReleaseProgram: unsafe extern "C" fn(*mut c_void) -> cl_int,
    clWaitForEvents: unsafe extern "C" fn(cl_uint, *const *mut c_void) -> cl_int,
}

// Event associated with program launch
#[derive(Debug)]
pub(crate) struct OpenCLEvent {
    ptr: *mut c_void,
    clWaitForEvents: unsafe extern "C" fn(cl_uint, *const *mut c_void) -> cl_int,
}

unsafe impl Send for OpenCLMemoryPool {}
unsafe impl Send for OpenCLBuffer {}
unsafe impl Send for OpenCLDevice {}
unsafe impl Send for OpenCLProgram {}
unsafe impl Send for OpenCLEvent {}

impl OpenCLDevice {
    pub(crate) fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    // Memory pool id out of OpenCLMemoryPools
    pub(crate) fn memory_pool_id(&self) -> usize {
        self.memory_pool_id
    }
}

impl Drop for OpenCLDevice {
    fn drop(&mut self) {
        unsafe { (self.clReleaseCommandQueue)(self.queue) };
    }
}

impl Drop for OpenCLProgram {
    fn drop(&mut self) {
        unsafe { (self.clReleaseProgram)(self.program) };
    }
}

pub struct OpenCLConfig {
    pub platform_ids: Option<Vec<usize>>,
}

impl Drop for OpenCLMemoryPool {
    fn drop(&mut self) {
        unsafe { (self.clReleaseContext)(self.context) };
        unsafe { (self.clReleaseCommandQueue)(self.queue) };
    }
}

pub(crate) fn initialize_opencl_backend(
    config: &OpenCLConfig,
) -> Result<(Vec<OpenCLMemoryPool>, Vec<OpenCLDevice>), OpenCLError> {
    let opencl_paths = ["/lib64/libOpenCL.so", "/lib/x86_64-linux-gnu/libOpenCL.so"];
    let opencl = opencl_paths.iter().find_map(|path| if let Ok(lib) = unsafe { Library::new(path) } { Some(lib) } else { None } );
    let Some(opencl) = opencl else { return Err(OpenCLError { info: "OpenCL runtime not found.".into(), status: OpenCLStatus::UNKNOWN }) };
    let clGetPlatformIDs: unsafe extern "C" fn(cl_uint, *mut *mut c_void, *mut cl_uint) -> cl_int =
        *unsafe { opencl.get(b"clGetPlatformIDs\0") }.unwrap();
    let clCreateContext: unsafe extern "C" fn(
        *const isize,
        cl_uint,
        *const *mut c_void,
        Option<unsafe extern "C" fn(*const i8, *const c_void, usize, *mut c_void)>,
        *mut c_void,
        *mut cl_int,
    ) -> *mut c_void = *unsafe { opencl.get(b"clCreateContext\0") }.unwrap();
    let clCreateCommandQueue: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_bitfield,
        *mut cl_int,
    ) -> *mut c_void = *unsafe { opencl.get(b"clCreateCommandQueue\0") }.unwrap();
    let clGetDeviceIDs: unsafe extern "C" fn(
        *mut c_void,
        cl_bitfield,
        cl_uint,
        *mut *mut c_void,
        *mut cl_uint,
    ) -> cl_int = *unsafe { opencl.get(b"clGetDeviceIDs\0") }.unwrap();
    let clWaitForEvents = *unsafe { opencl.get(b"clWaitForEvents\0") }.unwrap();
    let clReleaseCommandQueue = *unsafe { opencl.get(b"clReleaseCommandQueue\0") }.unwrap();
    let clEnqueueNDRangeKernel = *unsafe { opencl.get(b"clEnqueueNDRangeKernel\0") }.unwrap();
    let clGetProgramBuildInfo = *unsafe { opencl.get(b"clGetProgramBuildInfo\0") }.unwrap();
    let clBuildProgram = *unsafe { opencl.get(b"clBuildProgram\0") }.unwrap();
    let clReleaseProgram = *unsafe { opencl.get(b"clReleaseProgram\0") }.unwrap();
    let clReleaseContext = *unsafe { opencl.get(b"clReleaseContext\0") }.unwrap();
    let clSetKernelArg = *unsafe { opencl.get(b"clSetKernelArg\0") }.unwrap();
    let clCreateKernel = *unsafe { opencl.get(b"clCreateKernel\0") }.unwrap();
    let clReleaseMemObject = *unsafe { opencl.get(b"clReleaseMemObject\0") }.unwrap();
    let clGetDeviceInfo = *unsafe { opencl.get(b"clGetDeviceInfo\0") }.unwrap();
    let clCreateProgramWithSource =
        *unsafe { opencl.get(b"clCreateProgramWithSource\0") }.unwrap();
    let clEnqueueReadBuffer = *unsafe { opencl.get(b"clEnqueueReadBuffer\0") }.unwrap();
    let clEnqueueWriteBuffer = *unsafe { opencl.get(b"clEnqueueWriteBuffer\0") }.unwrap();
    let clCreateBuffer = *unsafe { opencl.get(b"clCreateBuffer\0") }.unwrap();

    #[cfg(feature = "debug_dev")]
    let clGetPlatformInfo: unsafe extern "C" fn(
        *mut c_void,
        cl_uint,
        usize,
        *mut c_void,
        *mut usize,
    ) -> cl_int = *unsafe { opencl.get(b"clGetPlatformInfo\0") }.unwrap();

    let library = Rc::new(opencl);

    let platform_ids = {
        // Get the number of platforms
        let mut count: cl_uint = 0;
        let status = unsafe { clGetPlatformIDs(0, ptr::null_mut(), &mut count) };
        check(status, "Failed to get OpenCL platform ids.")?;
        if count > 0 {
            // Get the platform ids.
            let len = count as usize;
            let mut ids: Vec<*mut c_void> = Vec::with_capacity(len);
            let status = unsafe { clGetPlatformIDs(count, ids.as_mut_ptr(), ptr::null_mut()) };
            check(status, "Failed to get OpenCL platform ids.")?;
            unsafe { ids.set_len(len) };
            ids
        } else {
            Vec::new()
        }
    };
    let mut devices = Vec::new();
    let mut memory_pools = Vec::new();
    let mut memory_pool_id = 0;
    for (_, platform) in platform_ids.iter().enumerate().filter(|(id, _)| {
        if let Some(ids) = config.platform_ids.as_ref() {
            ids.contains(id)
        } else {
            true
        }
    }) {
        let platform = *platform;
        let Ok(device_ids) = {
            // Get the number of devices of device_type
            let mut count: cl_uint = 0;
            let mut status = unsafe {
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, ptr::null_mut(), &mut count)
            };

            if (CL_SUCCESS != status) && (CL_DEVICE_NOT_FOUND != status) {
                Err(status)
            } else if 0 < count {
                // Get the device ids.
                let len = count as usize;
                let mut ids: Vec<*mut c_void> = Vec::with_capacity(len);
                unsafe {
                    status = clGetDeviceIDs(
                        platform,
                        CL_DEVICE_TYPE_ALL,
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
        .map_err(|err| check(err, "Failed to get OpenCL device ids").err().unwrap()) else {
            continue;
        };
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
        let Ok(()) = check(status, "Failed to create OpenCL context") else {
            continue;
        };
        let mut total_bytes = 0;
        #[cfg(feature = "debug_dev")]
        {
            if let Ok(platform_name) = {
                let mut size: usize = 0;
                let status = unsafe {
                    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, ptr::null_mut(), &mut size)
                };
                let Ok(_) = check(status, "Failed to get platform info.") else { continue; };
                if 0 < size {
                    let count = size / core::mem::size_of::<u8>();
                    let mut data: Vec<u8> = Vec::with_capacity(count);
                    let status = unsafe {
                        data.set_len(count);
                        clGetPlatformInfo(
                            platform,
                            CL_PLATFORM_NAME,
                            size,
                            data.as_mut_ptr() as *mut c_void,
                            ptr::null_mut(),
                        )
                    };
                    match check(status, "Failed to get platform info.") {
                        Ok(_) => Ok(data),
                        Err(err) => Err(err),
                    }
                } else {
                    Ok(Vec::default())
                }
            } {
                println!(
                    "Using OpenCL backend, platform {} on devices:",
                    String::from_utf8(platform_name).unwrap()
                );
            }
        }
        for dev in device_ids.iter().copied() {
            let queue = unsafe { clCreateCommandQueue(context, dev, 0, &mut status) };
            let Ok(_) = check(status, "Failed to create device command queue") else {
                continue;
            };
            let mut device = OpenCLDevice {
                ptr: dev,
                context,
                queue,
                dev_info: DeviceInfo::default(),
                memory_pool_id,
                clWaitForEvents,
                clReleaseCommandQueue,
                clEnqueueNDRangeKernel,
                clGetProgramBuildInfo,
                clBuildProgram,
                clReleaseProgram,
                clSetKernelArg,
                clCreateKernel,
                clGetDeviceInfo,
                clCreateProgramWithSource,
            };
            let Ok(_) = device.set_info() else {
                continue;
            };
            if let Ok(bytes) = device.get_device_data(CL_DEVICE_GLOBAL_MEM_SIZE) {
                total_bytes += u64::from_ne_bytes(bytes.try_into().unwrap()) as usize;
            } else {
                continue;
            }
            devices.push(device);
        }
        if device_ids.is_empty() {
            continue;
        }
        let queue = devices.last().unwrap().queue;
        memory_pools.push(OpenCLMemoryPool {
            library: library.clone(),
            total_bytes,
            free_bytes: total_bytes,
            context,
            queue,
            clWaitForEvents,
            clReleaseCommandQueue,
            clReleaseContext,
            clReleaseMemObject,
            clEnqueueReadBuffer,
            clEnqueueWriteBuffer,
            clCreateBuffer,
        });
        memory_pool_id += 1;
    }
    return Ok((memory_pools, devices));
}

impl OpenCLMemoryPool {
    pub(crate) fn allocate(&mut self, bytes: usize) -> Result<OpenCLBuffer, OpenCLError> {
        let mut status = CL_SUCCESS;
        let ptr = unsafe {
            (self.clCreateBuffer)(
                self.context,
                CL_MEM_READ_ONLY,
                bytes,
                ptr::null_mut(),
                &mut status,
            )
        };
        check(status, "Failed to allocate memory.")?;
        //println!("Allocated buffer {ptr:?}");
        self.free_bytes -= bytes;
        Ok(OpenCLBuffer {
            ptr,
            bytes,
            queue: self.queue,
        })
    }

    pub(crate) fn deallocate(&mut self, buffer: OpenCLBuffer) -> Result<(), OpenCLError> {
        let status = unsafe { (self.clReleaseMemObject)(buffer.ptr) };
        check(status, "Failed to free allocated memory")?;
        self.free_bytes += buffer.bytes;
        Ok(())
    }

    pub(crate) fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: &OpenCLBuffer,
    ) -> Result<(), OpenCLError> {
        //println!("Storing {src:?} to {dst:?}");
        let mut event = ptr::null_mut();
        let status = unsafe {
            (self.clEnqueueWriteBuffer)(
                dst.queue,
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
        check(status, "Failed to write buffer.")?;
        // Immediattely synchronize because we do not know the lifetime of data
        let status = unsafe { (self.clWaitForEvents)(1, (&[event]).as_ptr().cast()) };
        check(status, "Failed to finish buffer write event.")?;
        Ok(())
    }

    pub(crate) fn pool_to_host(
        &mut self,
        src: &OpenCLBuffer,
        dst: &mut [u8],
    ) -> Result<(), OpenCLError> {
        assert!(
            !src.ptr.is_null(),
            "Trying to read null memory. Internal bug."
        );
        let mut event: *mut c_void = ptr::null_mut();
        //println!("OpenCL to host src: {src:?}");
        let status = unsafe {
            (self.clEnqueueReadBuffer)(
                src.queue,
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
        check(status, "Failed to read buffer.")?;
        let status = unsafe { (self.clWaitForEvents)(1, (&[event]).as_ptr().cast()) };
        check(status, "Failed to finish buffer write event.")?;
        Ok(())
    }

    pub(crate) fn pool_to_pool(
        &mut self,
        src: &OpenCLBuffer,
        dst: &OpenCLBuffer,
    ) -> Result<(), OpenCLError> {
        //println!("Moving from {src:?} to {dst:?}");
        // TODO going through host is slow
        assert_eq!(src.bytes, dst.bytes);
        let mut data: Vec<u8> = Vec::with_capacity(dst.bytes);
        unsafe { data.set_len(dst.bytes) };
        self.pool_to_host(src, data.as_mut())?;
        //println!("Copied data: {data:?}");
        self.host_to_pool(&data, dst)?;
        Ok(())
    }
}

impl OpenCLDevice {
    fn set_info(&mut self) -> Result<(), OpenCLError> {
        let device_name = self.get_device_data(CL_DEVICE_NAME)?;
        let device_name = String::from_utf8(device_name).unwrap();
        let max_work_item_dims = self.get_device_data(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)?;
        #[cfg(feature = "debug_dev")]
        println!("{device_name}");
        let max_work_item_dims =
            u32::from_ne_bytes(max_work_item_dims.try_into().unwrap()) as usize;
        let mwis = self.get_device_data(CL_DEVICE_MAX_WORK_ITEM_SIZES)?;
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
        self.dev_info = DeviceInfo {
            compute: get_compute(&device_name),
            max_work_item_sizes,
            max_work_group_size: usize::from_ne_bytes(
                self.get_device_data(CL_DEVICE_MAX_WORK_GROUP_SIZE)?
                    .try_into()
                    .unwrap(),
            ),
            preferred_vector_size: u32::from_ne_bytes(
                self.get_device_data(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)?
                    .try_into()
                    .unwrap(),
            ) as usize
                * 4,
            f16_support: true,
            f64_support: true,
            fmadd: true,
            page_size: u32::from_ne_bytes(
                self.get_device_data(CL_DEVICE_MEM_BASE_ADDR_ALIGN)?
                    .try_into()
                    .unwrap(),
            ) as usize
                / 8,
            local_memory: true,
            local_mem_size: u64::from_ne_bytes(
                self.get_device_data(CL_DEVICE_LOCAL_MEM_SIZE)?
                    .try_into()
                    .unwrap(),
            ) as usize,
            num_registers: 128, // We can only guess or have a map of concrete hardware and respective register counts
            wmma: false,
            tensor_cores: false,
        };
        Ok(())
    }

    pub(crate) fn compile(&mut self, kernel: &IRKernel) -> Result<OpenCLProgram, OpenCLError> {
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
            source += &format!(
                "{indent}{}{} r{id};\n",
                if *read_only { "const " } else { "" },
                dtype.ocl()
            );
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

        for op in kernel.ops[6..kernel.ops.len()-6].iter().copied() {
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
                    source += &format!(
                        "{indent}{z} = {};\n",
                        match uop {
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
                        }
                    );
                }
                IROp::Binary {
                    z,
                    x,
                    y,
                    bop,
                    dtype: _,
                } => {
                    source += &format!(
                        "{indent}{z} = {};\n",
                        match bop {
                            BOp::Add => format!("{x} + {y}"),
                            BOp::Sub => format!("{x} - {y}"),
                            BOp::Mul => format!("{x} * {y}"),
                            BOp::Div => format!("{x} / {y}"),
                            BOp::Pow => format!("pow({x}, {y})"),
                            BOp::Cmplt => format!("{x} < {y}"),
                            BOp::Cmpgt => format!("{x} > {y}"),
                            BOp::Max => format!("max({x}, {y})"),
                            BOp::Or => format!("{x} || {y}"),
                        }
                    );
                }
                IROp::MAdd {
                    z,
                    a,
                    b,
                    c,
                    dtype: _,
                } => {
                    source += &format!("{indent}{z} = {a} * {b} + {c};\n");
                }
                IROp::AMAdd {
                    z,
                    a,
                    b,
                    c,
                    d,
                    dtype: _,
                } => {
                    source += &format!("{indent}{z} = ({a} + {b}) * {c} + {d};\n");
                }
                IROp::SMAdd {
                    z,
                    a,
                    b,
                    c,
                    d,
                    dtype: _,
                } => {
                    source += &format!("{indent}{z} = ({a} - {b}) * {c} + {d};\n");
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

        let context = self.context;
        let device = self.ptr;
        let queue = self.queue;
        let mut global_work_size = global_work_size;
        let local_work_size = local_work_size;
        let name = format!(
            "k__{}_{}__{}_{}__{}_{}",
            global_work_size[0],
            local_work_size[0],
            global_work_size[1],
            local_work_size[1],
            global_work_size[2],
            local_work_size[2],
        );
        for (i, lwd) in local_work_size.iter().enumerate() {
            global_work_size[i] *= lwd;
        }
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
            (self.clCreateProgramWithSource)(
                context,
                1,
                sources.as_ptr().cast(),
                &[source.len()] as *const usize,
                &mut status,
            )
        };
        check(status, "Failed to compile program.")?;
        let err = unsafe {
            (self.clBuildProgram)(
                program,
                1,
                [device].as_ptr(),
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
            let build_log = self.get_program_build_data(program, CL_PROGRAM_BUILD_LOG);
            match build_log {
                Ok(build_log) => panic!("{}", String::from_utf8_lossy(&build_log)),
                Err(status) => check(status, "Failed to get info about failed compilation.")?,
            }
        }
        let mut status = CL_SUCCESS;
        let program_name = &CString::new(name.clone()).unwrap();
        let kernel =
            unsafe { (self.clCreateKernel)(program, program_name.as_ptr().cast(), &mut status) };
        check(status, "Failed to create kernel.")?;
        Ok(OpenCLProgram {
            kernel,
            program,
            global_work_size,
            local_work_size,
            queue,
            clWaitForEvents: self.clWaitForEvents,
            clEnqueueNDRangeKernel: self.clEnqueueNDRangeKernel,
            clReleaseProgram: self.clReleaseProgram,
            clSetKernelArg: self.clSetKernelArg,
        })
    }
}

impl OpenCLProgram {
    pub(crate) fn launch(
        &mut self,
        buffers: &mut IndexMap<OpenCLBuffer>,
        args: &[usize],
    ) -> Result<OpenCLEvent, OpenCLError> {
        //#[cfg(feature = "debug1")]
        //libc_print::libc_println!("{:?}", self.load_memory::<f32>(&args[0], 4).unwrap());
        //#[cfg(not(feature = "debug1"))]
        let mut i = 0;
        for arg in args {
            let arg = &mut buffers[*arg];
            //println!("Kernel arg: {arg:?}");
            // This is POINTER MAGIC. Be careful.
            let ptr: *const _ = &arg.ptr;
            let status = unsafe {
                (self.clSetKernelArg)(
                    self.kernel,
                    i,
                    core::mem::size_of::<*mut c_void>(),
                    ptr.cast(),
                )
            };
            check(status, "Failend to set kernel arg.")?;
            i += 1;
        }
        let mut event: *mut c_void = ptr::null_mut();
        let status = unsafe {
            (self.clEnqueueNDRangeKernel)(
                self.queue,
                self.kernel,
                u32::try_from(self.global_work_size.len()).unwrap(),
                ptr::null(),
                self.global_work_size.as_ptr(),
                self.local_work_size.as_ptr(),
                0,
                ptr::null(),
                &mut event,
            )
        };
        check(status, "Failed to enqueue kernel.")?;
        return Ok(OpenCLEvent {
            ptr: event,
            clWaitForEvents: self.clWaitForEvents,
        });
    }
}

impl Drop for OpenCLEvent {
    fn drop(&mut self) {
        let status = unsafe { (self.clWaitForEvents)(1, (&[self.ptr]).as_ptr().cast()) };
        check(status, "Failed to finish program.").unwrap();
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

impl OpenCLDevice {
    fn get_program_build_data(
        &mut self,
        program: *mut c_void,
        param_name: cl_uint,
    ) -> Result<Vec<u8>, cl_int> {
        let size = {
            let idx = self.ptr;
            let mut size: usize = 0;
            let status = unsafe {
                (self.clGetProgramBuildInfo)(
                    program,
                    idx,
                    param_name,
                    0,
                    ptr::null_mut(),
                    &mut size,
                )
            };
            if CL_SUCCESS != status {
                Err(status)
            } else {
                Ok(size)
            }
        }?;
        return if 0 < size {
            let count = size / core::mem::size_of::<u8>();
            let mut data: Vec<u8> = Vec::with_capacity(count);
            let status = unsafe {
                data.set_len(count);
                (self.clGetProgramBuildInfo)(
                    program,
                    self.ptr,
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
        };
    }

    fn get_device_data(&mut self, param_name: cl_uint) -> Result<Vec<u8>, OpenCLError> {
        let size = {
            let object = self.ptr;
            let param_name = param_name;
            let mut size: usize = 0;
            let status = unsafe {
                (self.clGetDeviceInfo)(object, param_name, 0, ptr::null_mut(), &mut size)
            };
            if CL_SUCCESS != status {
                return Err(OpenCLError {
                    status: status.into(),
                    info: format!("Failed to get device info {param_name}"),
                });
            } else {
                Ok(size)
            }
        }?;
        return {
            let object = self.ptr;
            let param_name = param_name;
            if 0 < size {
                let count = size / core::mem::size_of::<u8>();
                let mut data: Vec<u8> = Vec::with_capacity(count);
                let status = unsafe {
                    data.set_len(count);
                    (self.clGetDeviceInfo)(
                        object,
                        param_name,
                        size,
                        data.as_mut_ptr() as *mut c_void,
                        ptr::null_mut(),
                    )
                };
                if CL_SUCCESS != status {
                    Err(OpenCLError {
                        status: status.into(),
                        info: format!("Failed to get {param_name}"),
                    })
                } else {
                    Ok(data)
                }
            } else {
                Ok(Vec::default())
            }
        };
    }
}

type cl_int = i32;
type cl_uint = u32;
type cl_bitfield = u64;

#[cfg(feature = "debug_dev")]
const CL_PLATFORM_NAME: cl_uint = 0x0902; // 2306
const CL_DEVICE_NAME: cl_uint = 0x102B; // 4139
const CL_DEVICE_GLOBAL_MEM_SIZE: cl_uint = 0x101F; // 4127
const CL_DEVICE_LOCAL_MEM_SIZE: cl_uint = 0x1023; // 4131
                                                  //const CL_DEVICE_MAX_MEM_ALLOC_SIZE: cl_uint = 0x1010; // 4112
                                                  //const CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: cl_uint = 0x101A; // 4122
const CL_DEVICE_MAX_WORK_GROUP_SIZE: cl_uint = 0x1004; // 4100
const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: cl_uint = 0x1003; // 4099
const CL_DEVICE_MAX_WORK_ITEM_SIZES: cl_uint = 0x1005; // 4101
const CL_DEVICE_MEM_BASE_ADDR_ALIGN: cl_uint = 0x1019; // 4121
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

fn get_compute(device_name: &str) -> u128 {
    match device_name.to_lowercase() {
        x if x.contains("i5-4460") => 300 * 1024 * 1024 * 1024,
        x if x.contains("i5-2500") => 150 * 1024 * 1024 * 1024,
        x if x.contains("ryzen 5 5500u") => 300 * 1024 * 1024 * 1024,
        x if x.contains("rx 550") => 1200 * 1024 * 1024 * 1024,
        x if x.contains("gtx 745") => 900 * 1024 * 1024 * 1024,
        _ => {
            #[cfg(feature = "debug_dev")]
            println!("Unknown device {device_name}, guessing compute capability");
            1024 * 1024 * 1024 * 1024
        }
    }
}
