#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use libloading::Library;

use crate::{
    dtype::Constant,
    index_map::IndexMap,
    runtime::{
        ir::{IRDType, IRKernel, IROp, Scope, Var},
        node::{BOp, UOp},
    },
};
use std::{
    ffi::{c_void, CString},
    ptr,
    rc::Rc,
};

use super::DeviceInfo;

#[derive(Debug, Default, serde::Deserialize)]
pub struct OpenCLConfig {
    /// Select which platforms will be used by OpenCL backend
    /// If set to None, uses all available platforms.
    /// default = None
    pub platform_ids: Option<Vec<usize>>,
}

// OpenCL does not have the concept of memory pools,
// so we simply say it is all in one memory pool
#[derive(Debug)]
pub(super) struct OpenCLMemoryPool {
    // Just to close the connection
    #[allow(unused)]
    library: Rc<Library>,
    #[allow(unused)]
    total_bytes: usize,
    free_bytes: usize,
    context: *mut c_void,
    queue: *mut c_void,
    // Functions
    clWaitForEvents: unsafe extern "C" fn(cl_uint, *const *mut c_void) -> OpenCLStatus,
    clReleaseCommandQueue: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
    clReleaseContext: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
    clReleaseMemObject: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
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
    ) -> OpenCLStatus,
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
    ) -> OpenCLStatus,
    clCreateBuffer: unsafe extern "C" fn(
        *mut c_void,
        cl_bitfield,
        usize,
        *mut c_void,
        *mut OpenCLStatus,
    ) -> *mut c_void,
}

// Ideally we would want Buffer to have lifetime of MemoryPool
// and Program to have lifetime of Device, but self referential
// lifetimes are not in rust, so we have to do manual memory management
// as they did it in stone age.
#[derive(Debug)]
pub(super) struct OpenCLBuffer {
    ptr: *mut c_void,
    bytes: usize,
    queue: *mut c_void, // This is the queue held by memory pool
}

#[derive(Debug)]
pub(super) struct OpenCLDevice {
    ptr: *mut c_void,
    context: *mut c_void,
    dev_info: DeviceInfo,
    memory_pool_id: usize,
    // Functions
    //clWaitForEvents: unsafe extern "C" fn(cl_uint, *const *mut c_void) -> cl_int,
    clGetProgramBuildInfo: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_uint,
        usize,
        *mut c_void,
        *mut usize,
    ) -> OpenCLStatus,
    clBuildProgram: unsafe extern "C" fn(
        *mut c_void,
        cl_uint,
        *const *mut c_void,
        *const i8,
        Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>,
        *mut c_void,
    ) -> OpenCLStatus,
    clReleaseProgram: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
    clCreateKernel: unsafe extern "C" fn(*mut c_void, *const i8, *mut OpenCLStatus) -> *mut c_void,
    clGetDeviceInfo:
        unsafe extern "C" fn(*mut c_void, cl_uint, usize, *mut c_void, *mut usize) -> OpenCLStatus,
    clCreateProgramWithSource: unsafe extern "C" fn(
        *mut c_void,
        cl_uint,
        *const *const i8,
        *const usize,
        *mut OpenCLStatus,
    ) -> *mut c_void,
}

#[derive(Debug)]
pub(super) struct OpenCLProgram {
    program: *mut c_void,
    kernel: *mut c_void,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    clReleaseProgram: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
}

#[derive(Debug)]
pub(super) struct OpenCLQueue {
    queue: *mut c_void, // points to device queue
    load: usize,
    //events: Vec<*mut c_void>,
    // Functions
    clReleaseCommandQueue: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
    clSetKernelArg:
        unsafe extern "C" fn(*mut c_void, cl_uint, usize, *const c_void) -> OpenCLStatus,
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
    ) -> OpenCLStatus,
    //clWaitForEvents: unsafe extern "C" fn(cl_uint, *const *mut c_void) -> OpenCLStatus,
    clFinish: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
}

unsafe impl Send for OpenCLMemoryPool {}
unsafe impl Send for OpenCLBuffer {}
unsafe impl Send for OpenCLDevice {}
unsafe impl Send for OpenCLProgram {}
unsafe impl Send for OpenCLQueue {}

impl OpenCLDevice {
    pub(super) fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    // Memory pool id out of OpenCLMemoryPools
    pub(super) fn memory_pool_id(&self) -> usize {
        self.memory_pool_id
    }
}

impl Drop for OpenCLMemoryPool {
    fn drop(&mut self) {
        unsafe { (self.clReleaseContext)(self.context) };
        unsafe { (self.clReleaseCommandQueue)(self.queue) };
    }
}

impl Drop for OpenCLProgram {
    fn drop(&mut self) {
        unsafe { (self.clReleaseProgram)(self.program) };
    }
}

impl Drop for OpenCLQueue {
    fn drop(&mut self) {
        let _ = unsafe { (self.clReleaseCommandQueue)(self.queue) };
    }
}

pub(super) fn initialize_devices(
    config: &OpenCLConfig,
    debug_dev: bool,
) -> Result<(Vec<OpenCLMemoryPool>, Vec<(OpenCLDevice, Vec<OpenCLQueue>)>), OpenCLError> {
    let opencl_paths = ["/lib64/libOpenCL.so", "/lib/x86_64-linux-gnu/libOpenCL.so"];
    let opencl = opencl_paths.iter().find_map(|path| {
        if let Ok(lib) = unsafe { Library::new(path) } {
            Some(lib)
        } else {
            None
        }
    });
    let Some(opencl) = opencl else {
        return Err(OpenCLError {
            info: "OpenCL runtime not found.".into(),
            status: OpenCLStatus::UNKNOWN,
        });
    };
    let clGetPlatformIDs: unsafe extern "C" fn(
        cl_uint,
        *mut *mut c_void,
        *mut cl_uint,
    ) -> OpenCLStatus = *unsafe { opencl.get(b"clGetPlatformIDs\0") }.unwrap();
    let clCreateContext: unsafe extern "C" fn(
        *const isize,
        cl_uint,
        *const *mut c_void,
        Option<unsafe extern "C" fn(*const i8, *const c_void, usize, *mut c_void)>,
        *mut c_void,
        *mut OpenCLStatus,
    ) -> *mut c_void = *unsafe { opencl.get(b"clCreateContext\0") }.unwrap();
    let clCreateCommandQueue: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        cl_bitfield,
        *mut OpenCLStatus,
    ) -> *mut c_void = *unsafe { opencl.get(b"clCreateCommandQueue\0") }.unwrap();
    let clGetDeviceIDs: unsafe extern "C" fn(
        *mut c_void,
        cl_bitfield,
        cl_uint,
        *mut *mut c_void,
        *mut cl_uint,
    ) -> OpenCLStatus = *unsafe { opencl.get(b"clGetDeviceIDs\0") }.unwrap();
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
    let clCreateProgramWithSource = *unsafe { opencl.get(b"clCreateProgramWithSource\0") }.unwrap();
    let clEnqueueReadBuffer = *unsafe { opencl.get(b"clEnqueueReadBuffer\0") }.unwrap();
    let clEnqueueWriteBuffer = *unsafe { opencl.get(b"clEnqueueWriteBuffer\0") }.unwrap();
    let clCreateBuffer = *unsafe { opencl.get(b"clCreateBuffer\0") }.unwrap();
    let clFinish = *unsafe { opencl.get(b"clFinish\0") }.unwrap();
    let clGetPlatformInfo: unsafe extern "C" fn(
        *mut c_void,
        cl_uint,
        usize,
        *mut c_void,
        *mut usize,
    ) -> OpenCLStatus = *unsafe { opencl.get(b"clGetPlatformInfo\0") }.unwrap();

    let library = Rc::new(opencl);
    let platform_ids = {
        // Get the number of platforms
        let mut count: cl_uint = 0;
        unsafe { clGetPlatformIDs(0, ptr::null_mut(), &mut count) }
            .check("Failed to get OpenCL platform ids.")?;
        if count > 0 {
            // Get the platform ids.
            let len = count as usize;
            let mut ids: Vec<*mut c_void> = Vec::with_capacity(len);
            unsafe { clGetPlatformIDs(count, ids.as_mut_ptr(), ptr::null_mut()) }
                .check("Failed to get OpenCL platform ids.")?;
            unsafe { ids.set_len(len) };
            ids
        } else {
            Vec::new()
        }
    };
    let mut devices = Vec::new();
    let mut memory_pools = Vec::new();
    let mut memory_pool_id = 0;
    for (platform_id, platform) in platform_ids.iter().enumerate().filter(|(id, _)| {
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
            if (OpenCLStatus::CL_SUCCESS != status) && (OpenCLStatus::CL_DEVICE_NOT_FOUND != status)
            {
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
                if OpenCLStatus::CL_SUCCESS != status {
                    Err(status)
                } else {
                    Ok(ids)
                }
            } else {
                Ok(Vec::default())
            }
        }
        .map_err(|err| err.check("Failed to get OpenCL device ids").err().unwrap()) else {
            continue;
        };
        let mut status = OpenCLStatus::CL_SUCCESS;
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
        let Ok(()) = status.check("Failed to create OpenCL context") else {
            continue;
        };
        let mut total_bytes = 0;
        if debug_dev {
            let platform_name = {
                let mut size: usize = 0;
                let Ok(_) = unsafe {
                    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, ptr::null_mut(), &mut size)
                }
                .check("Failed to get platform info.") else {
                    continue;
                };
                if size > 0 {
                    let count = size / core::mem::size_of::<u8>();
                    let mut data: Vec<u8> = Vec::with_capacity(count);
                    let Ok(_) = unsafe {
                        data.set_len(count);
                        clGetPlatformInfo(
                            platform,
                            CL_PLATFORM_NAME,
                            size,
                            data.as_mut_ptr() as *mut c_void,
                            ptr::null_mut(),
                        )
                    }
                    .check("Failed to get platform info.") else {
                        continue;
                    };
                    data
                } else {
                    Vec::default()
                }
            };
            println!(
                "Using OpenCL platform, platform id {platform_id}, name {} on devices:",
                String::from_utf8(platform_name).unwrap()
            );
        }
        for dev in device_ids.iter().copied() {
            // TODO get max queues per device and limit this to that number
            let mut queues = Vec::new();
            for _ in 0..8 {
                queues.push(OpenCLQueue {
                    queue: unsafe { clCreateCommandQueue(context, dev, 0, &mut status) },
                    load: 0,
                    //events: Vec::new(),
                    clSetKernelArg,
                    //clWaitForEvents,
                    clEnqueueNDRangeKernel,
                    clFinish,
                    clReleaseCommandQueue,
                });
                let Ok(_) = status.check("Failed to create device command queue") else {
                    continue;
                };
            }
            let mut device = OpenCLDevice {
                ptr: dev,
                context,
                dev_info: DeviceInfo::default(),
                memory_pool_id,
                clGetProgramBuildInfo,
                clBuildProgram,
                clReleaseProgram,
                clCreateKernel,
                clGetDeviceInfo,
                clCreateProgramWithSource,
            };
            let Ok(_) = device.set_info(debug_dev) else {
                continue;
            };
            if let Ok(bytes) = device.get_device_data(CL_DEVICE_GLOBAL_MEM_SIZE) {
                total_bytes += u64::from_ne_bytes(bytes.try_into().unwrap()) as usize;
            } else {
                continue;
            }
            devices.push((device, queues));
        }
        if device_ids.is_empty() {
            continue;
        }
        let queue =
            unsafe { clCreateCommandQueue(context, devices.last().unwrap().0.ptr, 0, &mut status) };
        let Ok(_) = status.check("Failed to create device command queue") else {
            continue;
        };
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
    pub(super) fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    pub(super) fn allocate(&mut self, bytes: usize) -> Result<OpenCLBuffer, OpenCLError> {
        if bytes > self.free_bytes {
            return Err(OpenCLError {
                info: "Insufficient free memory.".into(),
                status: OpenCLStatus::CL_MEM_OBJECT_ALLOCATION_FAILURE,
            });
        }
        let mut status = OpenCLStatus::CL_SUCCESS;
        let ptr = unsafe {
            (self.clCreateBuffer)(
                self.context,
                CL_MEM_READ_ONLY,
                bytes,
                ptr::null_mut(),
                &mut status,
            )
        };
        status.check("Failed to allocate memory.")?;
        //println!("Allocated buffer {ptr:?}, bytes {bytes}");
        self.free_bytes -= bytes;
        Ok(OpenCLBuffer {
            ptr,
            bytes,
            queue: self.queue,
        })
    }

    pub(super) fn deallocate(&mut self, buffer: OpenCLBuffer) -> Result<(), OpenCLError> {
        //println!("Deallocate {:?}", buffer.ptr);
        unsafe { (self.clReleaseMemObject)(buffer.ptr) }
            .check("Failed to free allocated memory")?;
        self.free_bytes += buffer.bytes;
        Ok(())
    }

    pub(super) fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: &OpenCLBuffer,
    ) -> Result<(), OpenCLError> {
        //println!("Storing {src:?} to {dst:?}");
        //println!("Storing to {dst:?}");
        let mut event = ptr::null_mut();
        unsafe {
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
        }
        .check("Failed to write buffer.")?;
        // Immediattely synchronize because we do not know the lifetime of data
        unsafe { (self.clWaitForEvents)(1, (&[event]).as_ptr().cast()) }
            .check("Failed to finish buffer write event.")
    }

    pub(super) fn pool_to_host(
        &mut self,
        src: &OpenCLBuffer,
        dst: &mut [u8],
    ) -> Result<(), OpenCLError> {
        //println!("OpenCL to host src: {src:?}");
        assert!(
            !src.ptr.is_null(),
            "Trying to read null memory. Internal bug."
        );
        let mut event: *mut c_void = ptr::null_mut();
        unsafe {
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
        }
        .check("Failed to read buffer.")?;
        unsafe { (self.clWaitForEvents)(1, (&[event]).as_ptr().cast()) }
            .check("Failed to finish buffer write event.")?;
        Ok(())
    }

    pub(super) fn pool_to_pool(
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
    fn set_info(&mut self, debug_dev: bool) -> Result<(), OpenCLError> {
        let device_name = self.get_device_data(CL_DEVICE_NAME)?;
        let device_name = String::from_utf8(device_name).unwrap();
        let max_work_item_dims = self.get_device_data(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)?;
        if debug_dev {
            println!("{device_name}");
        }
        let mut max_work_item_dims =
            u32::from_ne_bytes(max_work_item_dims.try_into().unwrap()) as usize;
        let mwis = self.get_device_data(CL_DEVICE_MAX_WORK_ITEM_SIZES)?;
        let mut max_global_work_dims = [0; 3];
        if max_work_item_dims > 3 {
            println!(
                "Found device with more than 3 work dimesions, WOW. Please report this. Using only 3 dims for now."
            );
            max_work_item_dims = 3;
        }
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
            max_global_work_dims[i] = max_dim_size;
        }
        let mlt = usize::from_ne_bytes(
            self.get_device_data(CL_DEVICE_MAX_WORK_GROUP_SIZE)?
                .try_into()
                .unwrap(),
        );
        self.dev_info = DeviceInfo {
            compute: get_compute(&device_name, debug_dev),
            max_global_work_dims,
            max_local_threads: mlt,
            max_local_work_dims: [mlt, mlt, mlt],
            preferred_vector_size: u32::from_ne_bytes(
                self.get_device_data(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)?
                    .try_into()
                    .unwrap(),
            ) as usize
                * 4,
            local_mem_size: u64::from_ne_bytes(
                self.get_device_data(CL_DEVICE_LOCAL_MEM_SIZE)?
                    .try_into()
                    .unwrap(),
            ) as usize,
            num_registers: 96, // We can only guess or have a map of concrete hardware and respective register counts
            tensor_cores: false,
        };
        Ok(())
    }

    pub(super) fn compile(
        &mut self,
        kernel: &IRKernel,
        debug_asm: bool,
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
            } else {
                panic!()
            }
        }

        // Declare global variables
        for (id, (_, dtype, read_only)) in kernel.addressables.iter().enumerate() {
            source += &format!(
                "{indent}__global {}{}* g{id},\n",
                if *read_only { "const " } else { "" },
                dtype.ocl(),
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

        for op in kernel.ops[6..kernel.ops.len() - 6].iter().copied() {
            match op {
                IROp::Set { z, len: _, value } => {
                    source += &format!("{indent}r{z} = {value};\n");
                }
                IROp::Load { z, x, at, dtype: _ } => {
                    //source += &format!("{indent}if ({0} > 1048575) {{ printf(\"Load %d\\n\", {0}); }}\n", at.ocl());
                    source += &format!("{indent}{} = {}[{}];\n", z.ocl(), x.ocl(), at.ocl());
                }
                IROp::Store { z, x, at, dtype: _ } => {
                    //source += &format!("{indent}if ({0} > 1048570) {{ printf(\"Store %d\\n\", {0}); }}\n", at.ocl());
                    source += &format!("{indent}{}[{}] = {};\n", z.ocl(), at.ocl(), x.ocl());
                }
                IROp::Unary { z, x, uop, dtype } => {
                    source += &match uop {
                        UOp::Cast(_) => {
                            format!("{indent}{} = ({}){};\n", z.ocl(), dtype.ocl(), x.ocl())
                        }
                        UOp::ReLU => format!(
                            "{indent}{} = max({}, {});\n",
                            z.ocl(),
                            x.ocl(),
                            Constant::new(0).unary(UOp::Cast(dtype.dtype())).ocl()
                        ),
                        UOp::Neg => format!("{indent}{} = -{};\n", z.ocl(), x.ocl()),
                        UOp::Exp2 => format!("{indent}{} = exp2({});\n", z.ocl(), x.ocl()),
                        UOp::Log2 => format!("{indent}{} = log2({});\n", z.ocl(), x.ocl()),
                        UOp::Inv => format!("{indent}{} = 1/{};\n", z.ocl(), x.ocl()),
                        UOp::Sqrt => format!("{indent}{} = sqrt({});\n", z.ocl(), x.ocl()),
                        UOp::Sin => format!("{indent}{} = sin({});\n", z.ocl(), x.ocl()),
                        UOp::Cos => format!("{indent}{} = cos({});\n", z.ocl(), x.ocl()),
                        UOp::Not => format!("{indent}{} = !{};\n", z.ocl(), x.ocl()),
                        UOp::Nonzero => format!("{indent}{} = {} != 0;\n", z.ocl(), x.ocl()),
                    };
                }
                IROp::Binary {
                    z,
                    x,
                    y,
                    bop,
                    dtype: _,
                } => {
                    source += &format!(
                        "{indent}{} = {};\n",
                        z.ocl(),
                        match bop {
                            BOp::Add => format!("{} + {}", x.ocl(), y.ocl()),
                            BOp::Sub => format!("{} - {}", x.ocl(), y.ocl()),
                            BOp::Mul => format!("{} * {}", x.ocl(), y.ocl()),
                            BOp::Div => format!("{} / {}", x.ocl(), y.ocl()),
                            BOp::Pow => format!("pow({}, {})", x.ocl(), y.ocl()),
                            BOp::Cmplt => format!("{} < {}", x.ocl(), y.ocl()),
                            BOp::Cmpgt => format!("{} > {}", x.ocl(), y.ocl()),
                            BOp::Max => format!("max({}, {})", x.ocl(), y.ocl()),
                            BOp::Or => format!("{} || {}", x.ocl(), y.ocl()),
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
                    source += &format!(
                        "{indent}{} = {} * {} + {};\n",
                        z.ocl(),
                        a.ocl(),
                        b.ocl(),
                        c.ocl()
                    );
                }
                IROp::Loop { id, len } => {
                    source += &format!(
                        "{indent}for (unsigned int r{id} = 0; r{id} < {len}; r{id} += 1) {{\n"
                    );
                    indent += "  ";
                }
                IROp::EndLoop { .. } => {
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

        let local_work_size = local_work_size;
        let name = format!(
            "k__{}_{}_{}__{}_{}_{}",
            global_work_size[0],
            global_work_size[1],
            global_work_size[2],
            local_work_size[0],
            local_work_size[1],
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
        if debug_asm {
            println!("{source}");
        }
        let context = self.context;
        let device = self.ptr;
        let sources: &[&str] = &[source.as_str()];
        let mut status = OpenCLStatus::CL_SUCCESS;
        let program = unsafe {
            (self.clCreateProgramWithSource)(
                context,
                1,
                sources.as_ptr().cast(),
                &[source.len()] as *const usize,
                &mut status,
            )
        };
        status.check("Failed to compile program.")?;
        if let Err(e) = unsafe {
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
        }
        .check("Failed to build program.")
        {
            let build_log = self.get_program_build_data(program, CL_PROGRAM_BUILD_LOG);
            match build_log {
                Ok(build_log) => panic!("{e:?} {}", String::from_utf8_lossy(&build_log)),
                Err(status) => status.check(&format!(
                    "Failed to get info about failed compilation. {e:?}"
                ))?,
            }
        }
        let mut status = OpenCLStatus::CL_SUCCESS;
        let program_name = &CString::new(name.clone()).unwrap();
        let kernel =
            unsafe { (self.clCreateKernel)(program, program_name.as_ptr().cast(), &mut status) };
        status.check("Failed to create kernel.")?;
        Ok(OpenCLProgram {
            kernel,
            program,
            global_work_size,
            local_work_size,
            clReleaseProgram: self.clReleaseProgram,
        })
    }
}

impl OpenCLQueue {
    pub(super) fn launch(
        &mut self,
        program: &mut OpenCLProgram,
        buffers: &mut IndexMap<OpenCLBuffer>,
        args: &[usize],
    ) -> Result<(), OpenCLError> {
        /*println!(
            "Launch opencl kernel on queue {:?}, gws {:?}, lws {:?}",
            self.queue, program.global_work_size, program.local_work_size
        );*/
        let mut i = 0;
        for arg in args {
            let arg = &mut buffers[*arg];
            //println!("Kernel arg: {arg:?}");
            let ptr: *const _ = &arg.ptr;
            unsafe {
                (self.clSetKernelArg)(
                    program.kernel,
                    i,
                    core::mem::size_of::<*mut c_void>(),
                    ptr.cast(),
                )
            }
            .check("Failend to set kernel arg.")?;
            i += 1;
        }
        let mut event: *mut c_void = ptr::null_mut();
        self.load += 1;
        unsafe {
            (self.clEnqueueNDRangeKernel)(
                self.queue,
                program.kernel,
                u32::try_from(program.global_work_size.len()).unwrap(),
                ptr::null(),
                program.global_work_size.as_ptr(),
                program.local_work_size.as_ptr(),
                0,
                ptr::null(),
                &mut event,
            )
        }
        .check("Failed to enqueue kernel.")?;
        //unsafe { (self.clFinish)(self.queue) }.check("finish fail").unwrap();
        //self.events.push(event);
        Ok(())
    }

    pub(super) fn sync(&mut self) -> Result<(), OpenCLError> {
        //println!("Syncing {:?}", self);
        unsafe { (self.clFinish)(self.queue) }.check("Failed to synchronize device queue.")?;
        self.load = 0;
        //self.events.clear();
        //unsafe { (self.clWaitForEvents)(self.events.len() as u32, self.events.as_ptr()) }.check("Failed to synchronize device queue.")?;
        //panic!();
        Ok(())
    }

    pub(super) fn load(&self) -> usize {
        self.load
    }
}

impl IRDType {
    fn ocl(&self) -> &str {
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
            IRDType::U32 => "unsigned int",
        };
    }
}

impl OpenCLStatus {
    fn check(self, info: &str) -> Result<(), OpenCLError> {
        if self == OpenCLStatus::CL_SUCCESS {
            return Ok(());
        } else {
            return Err(OpenCLError {
                info: info.into(),
                status: self,
            });
        }
    }
}

impl OpenCLDevice {
    pub(super) fn release_program(&self, program: OpenCLProgram) -> Result<(), OpenCLError> {
        unsafe { (self.clReleaseProgram)(program.program) }
            .check("Failed to release OpenCL program")
    }

    fn get_program_build_data(
        &mut self,
        program: *mut c_void,
        param_name: cl_uint,
    ) -> Result<Vec<u8>, OpenCLStatus> {
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
            if OpenCLStatus::CL_SUCCESS != status {
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
            if OpenCLStatus::CL_SUCCESS != status {
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
            if OpenCLStatus::CL_SUCCESS != status {
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
                unsafe {
                    data.set_len(count);
                    (self.clGetDeviceInfo)(
                        object,
                        param_name,
                        size,
                        data.as_mut_ptr() as *mut c_void,
                        ptr::null_mut(),
                    )
                }
                .check(&format!("Failed to get {param_name}"))?;
                Ok(data)
            } else {
                Ok(Vec::default())
            }
        };
    }
}

type cl_int = i32;
type cl_uint = u32;
type cl_bitfield = u64;

const CL_PLATFORM_NAME: cl_uint = 0x0902; // 2306
const CL_DEVICE_NAME: cl_uint = 0x102B; // 4139
const CL_DEVICE_GLOBAL_MEM_SIZE: cl_uint = 0x101F; // 4127
const CL_DEVICE_LOCAL_MEM_SIZE: cl_uint = 0x1023; // 4131
                                                  //const CL_DEVICE_MAX_MEM_ALLOC_SIZE: cl_uint = 0x1010; // 4112
                                                  //const CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE: cl_uint = 0x101A; // 4122
const CL_DEVICE_MAX_WORK_GROUP_SIZE: cl_uint = 0x1004; // 4100
const CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: cl_uint = 0x1003; // 4099
const CL_DEVICE_MAX_WORK_ITEM_SIZES: cl_uint = 0x1005; // 4101
const CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT: cl_uint = 0x100A; // 4106
const CL_DEVICE_TYPE_ALL: cl_bitfield = 0xFFFF_FFFF;
const CL_MEM_READ_ONLY: cl_bitfield = 4;
const CL_NON_BLOCKING: cl_uint = 0;
const CL_PROGRAM_BUILD_LOG: cl_uint = 0x1183; // 4483

#[allow(dead_code)] // Rust for some reason thinks these fields are unused
#[derive(Debug)]
pub struct OpenCLError {
    info: String,
    status: OpenCLStatus,
}

#[derive(Copy, Clone, PartialEq, Debug, Eq)]
#[repr(C)]
enum OpenCLStatus {
    CL_DEVICE_NOT_FOUND = -1, // 0xFFFF_FFFF
    CL_SUCCESS = 0,
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

fn get_compute(device_name: &str, debug_dev: bool) -> u128 {
    match device_name.to_lowercase() {
        x if x.contains("i5-4460") => 300 * 1024 * 1024 * 1024,
        x if x.contains("i5-2500") => 150 * 1024 * 1024 * 1024,
        x if x.contains("ryzen 5 5500u") => 300 * 1024 * 1024 * 1024,
        x if x.contains("rx 550") => 1200 * 1024 * 1024 * 1024,
        x if x.contains("gtx 745") => 900 * 1024 * 1024 * 1024,
        x if x.contains("rtx 2060") => 57 * 1024 * 1024 * 1024 * 1024,
        _ => {
            if debug_dev {
                println!("Unknown device {device_name}, guessing compute capability");
            }
            1024 * 1024 * 1024 * 1024
        }
    }
}

impl Constant {
    fn ocl(&self) -> String {
        use core::mem::transmute as t;
        match self {
            #[cfg(feature = "half")]
            Constant::F16(x) => format!("{:.16}f", unsafe { t::<_, half::f16>(*x) }),
            #[cfg(feature = "half")]
            Constant::BF16(x) => format!("{:.16}f", unsafe { t::<_, half::bf16>(*x) }),
            Constant::F32(x) => format!("{:.16}f", unsafe { t::<_, f32>(*x) }),
            Constant::F64(x) => format!("{:.16}f", unsafe { t::<_, f64>(*x) }),
            #[cfg(feature = "complex")]
            Constant::CF32(..) => todo!("Complex numbers are currently not supported for OpenCL"),
            #[cfg(feature = "complex")]
            Constant::CF64(..) => todo!("Complex numbers are currently not supported for OpenCL"),
            Constant::U8(x) => format!("{x}"),
            Constant::I8(x) => format!("{x}"),
            Constant::I16(x) => format!("{x}"),
            Constant::U32(x) => format!("{x}"),
            Constant::I32(x) => format!("{x}"),
            Constant::I64(x) => format!("{x}"),
            Constant::Bool(x) => format!("{x}"),
        }
    }
}

impl Var {
    fn ocl(&self) -> String {
        match self {
            Var::Id(id, scope) => format!("{scope}{id}"),
            Var::Const(value) => format!("{}", value.ocl()),
        }
    }
}
