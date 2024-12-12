//! `OpenCL` backend

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use super::DeviceInfo;
use crate::{
    dtype::Constant,
    index_map::{Id, IndexMap},
    ir::{IRKernel, IROp, Reg, Scope},
    node::{BOp, UOp},
    DType,
};
use libloading::Library;
use std::{
    ffi::{c_void, CString},
    ptr,
    sync::Arc,
};

#[derive(Debug, Default, serde::Deserialize)]
pub struct OpenCLConfig {
    /// Select which platforms will be used by `OpenCL` backend
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
    library: Arc<Library>,
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
    memory_pool_id: u32,
    // Functions
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
    clReleaseCommandQueue: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
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
}

#[derive(Debug)]
pub(super) struct OpenCLQueue {
    queue: *mut c_void, // points to device queue
    load: usize,
    //events: Vec<*mut c_void>,
    // Functions
    //clReleaseCommandQueue: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
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
    clWaitForEvents: unsafe extern "C" fn(cl_uint, *const *mut c_void) -> OpenCLStatus,
    clFinish: unsafe extern "C" fn(*mut c_void) -> OpenCLStatus,
}

#[derive(Debug)]
pub(super) struct OpenCLEvent {
    event: *mut c_void,
    clWaitForEvents: unsafe extern "C" fn(cl_uint, *const *mut c_void) -> OpenCLStatus,
}

// This definitely isn't correct, but for now...
unsafe impl Send for OpenCLMemoryPool {}
unsafe impl Send for OpenCLBuffer {}
unsafe impl Send for OpenCLDevice {}
unsafe impl Send for OpenCLProgram {}
unsafe impl Send for OpenCLQueue {}
unsafe impl Send for OpenCLEvent {}

impl OpenCLDevice {
    pub(super) const fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    // Memory pool id out of OpenCLMemoryPools
    pub(super) const fn memory_pool_id(&self) -> u32 {
        self.memory_pool_id
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::unnecessary_wraps)]
    pub(super) const fn deinitialize(self) -> Result<(), OpenCLError> {
        // cuReleaseDevice is OpenCL 1.2 only, but we support 1.0, so nothing to do here?
        // TODO better do it conditionally, if the function exists in .so, then load it, do nothing
        // otherwise
        Ok(())
    }
}

impl OpenCLMemoryPool {
    pub(super) fn deinitialize(self) -> Result<(), OpenCLError> {
        unsafe { (self.clReleaseContext)(self.context) }
            .check("Failed to release OpenCL context.")?;
        unsafe { (self.clReleaseCommandQueue)(self.queue) }
            .check("Failed to release OpenCL command queue.")?;
        Ok(())
    }
}

type OpenCLQueuePool = Vec<(OpenCLDevice, Vec<OpenCLQueue>)>;

pub(super) fn initialize_devices(
    config: &OpenCLConfig,
    debug_dev: bool,
) -> Result<(Vec<OpenCLMemoryPool>, OpenCLQueuePool), OpenCLError> {
    let opencl_paths = ["/lib64/libOpenCL.so", "/lib/x86_64-linux-gnu/libOpenCL.so"];
    let opencl = opencl_paths
        .iter()
        .find_map(|path| unsafe { Library::new(path) }.ok());
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

    let library = Arc::new(opencl);
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
        config
            .platform_ids
            .as_ref()
            .map_or(true, |ids| ids.contains(id))
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
                if OpenCLStatus::CL_SUCCESS == status {
                    Ok(ids)
                } else {
                    Err(status)
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
                cl_uint::try_from(device_ids.len()).unwrap(),
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
                let Ok(()) = unsafe {
                    clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, ptr::null_mut(), &mut size)
                }
                .check("Failed to get platform info.") else {
                    continue;
                };
                if size > 0 {
                    let count = size / core::mem::size_of::<u8>();
                    let mut data: Vec<u8> = Vec::with_capacity(count);
                    let Ok(()) = unsafe {
                        data.set_len(count);
                        clGetPlatformInfo(
                            platform,
                            CL_PLATFORM_NAME,
                            size,
                            data.as_mut_ptr().cast(),
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
                    clSetKernelArg,
                    clWaitForEvents,
                    clEnqueueNDRangeKernel,
                    clFinish,
                    //clReleaseCommandQueue,
                });
                let Ok(()) = status.check("Failed to create device command queue") else {
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
                clReleaseCommandQueue,
                clCreateKernel,
                clGetDeviceInfo,
                clCreateProgramWithSource,
            };
            let Ok(()) = device.set_info(debug_dev) else {
                continue;
            };
            if let Ok(bytes) = device.get_device_data(CL_DEVICE_GLOBAL_MEM_SIZE) {
                total_bytes +=
                    usize::try_from(u64::from_ne_bytes(bytes.try_into().unwrap())).unwrap();
                devices.push((device, queues));
            }
        }
        if device_ids.is_empty() {
            continue;
        }
        let queue =
            unsafe { clCreateCommandQueue(context, devices.last().unwrap().0.ptr, 0, &mut status) };
        let Ok(()) = status.check("Failed to create device command queue") else {
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
    Ok((memory_pools, devices))
}

impl OpenCLMemoryPool {
    pub(super) const fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    pub(super) fn allocate(&mut self, bytes: usize) -> Result<OpenCLBuffer, OpenCLError> {
        if bytes > self.free_bytes {
            return Err(OpenCLError {
                info: "Insufficient free memory.".into(),
                status: OpenCLStatus::CL_MEM_OBJECT_ALLOCATION_FAILURE,
            });
        }
        //println!("Allocating bytes {bytes}");
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
        self.free_bytes = self.free_bytes.checked_sub(bytes).unwrap();
        Ok(OpenCLBuffer {
            ptr,
            bytes,
            queue: self.queue,
        })
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn deallocate(&mut self, buffer: OpenCLBuffer) -> Result<(), OpenCLError> {
        //println!("Deallocate {:?}", buffer.ptr);
        assert!(!buffer.ptr.is_null(), "Deallocating null buffer is invalid");
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
        unsafe { (self.clWaitForEvents)(1, [event].as_ptr().cast()) }
            .check("Failed to finish buffer write event.")
    }

    pub(super) fn pool_to_host(
        &mut self,
        src: &OpenCLBuffer,
        dst: &mut [u8],
    ) -> Result<(), OpenCLError> {
        //println!("OpenCL to host src: {src:?}, bytes {}", dst.len());
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
        unsafe { (self.clWaitForEvents)(1, [event].as_ptr().cast()) }
            .check("Failed to finish buffer write event.")?;
        Ok(())
    }

    pub(super) fn pool_to_pool(
        &mut self,
        src: &OpenCLBuffer,
        dst: &OpenCLBuffer,
    ) -> Result<(), OpenCLError> {
        //println!("Moving from {src:?} to {dst:?}");
        // TODO going through host is slow, but likely only way
        // LOL, is maybe unint really better than without it???
        // the only reason to use it is to stop it from running destructor, but u8 does not run it anyway ...
        assert_eq!(src.bytes, dst.bytes);
        let mut data: Vec<u8> = vec![0; dst.bytes];
        self.pool_to_host(src, &mut data)?;
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
                    mwis[i * 8],
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
            compute: 1024 * 1024 * 1024 * 1024,
            max_global_work_dims,
            max_local_threads: mlt,
            max_local_work_dims: [mlt, mlt, mlt],
            preferred_vector_size: u32::from_ne_bytes(
                self.get_device_data(CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)?
                    .try_into()
                    .unwrap(),
            ) as usize
                * 4,
            local_mem_size: usize::try_from(u64::from_ne_bytes(
                self.get_device_data(CL_DEVICE_LOCAL_MEM_SIZE)?
                    .try_into()
                    .unwrap(),
            ))
            .unwrap(),
            num_registers: 96, // We can only guess or have a map of concrete hardware and respective register counts
            tensor_cores: false,
        };
        Ok(())
    }

    #[allow(clippy::cognitive_complexity)]
    pub(super) fn compile(
        &mut self,
        kernel: &IRKernel,
        debug_asm: bool,
    ) -> Result<OpenCLProgram, OpenCLError> {
        let mut source = String::from("(\n");
        let mut indent = String::from("  ");

        let mut global_work_size = [0; 3];
        let mut local_work_size = [0; 3];

        let mut loops = [0; 6];
        for (i, op) in kernel.ops[..6].iter().enumerate() {
            if let IROp::Loop { id, len } = op {
                if i % 2 == 0 {
                    global_work_size[i / 2] = *len;
                } else {
                    local_work_size[i / 2] = *len;
                }
                loops[i] = *id;
            } else {
                unreachable!()
            }
        }

        // Declare global variables
        for (id, (scope, dtype, _, read_only)) in kernel.addressables.iter().enumerate() {
            if *scope == Scope::Global {
                source += &format!(
                    "{indent}__global {}{}* p{id},\n",
                    if *read_only { "const " } else { "" },
                    dtype.ocl(),
                );
            }
        }

        source.pop();
        source.pop();
        source += "\n) {\n";

        // Declare local variables
        for (id, (scope, dtype, len, _)) in kernel.addressables.iter().enumerate() {
            if *scope == Scope::Local {
                source += &format!("{indent}__local {} p{id}[{len}];\n", dtype.ocl(),);
            }
        }

        // Declare register accumulators
        for (id, (scope, dtype, len, read_only)) in kernel.addressables.iter().enumerate() {
            if *scope == Scope::RegTile {
                source += &format!(
                    "{indent}{}{} p{id}[{len}];\n",
                    if *read_only { "const " } else { "" },
                    dtype.ocl(),
                );
            }
        }

        // Declare register variables
        for (id, dtype) in kernel.registers.iter().enumerate() {
            source += &format!("{indent}{} r{id};\n", dtype.ocl(),);
        }

        // Add indices for global and local loops
        source += &format!(
            "{indent}r{} = get_group_id(0);   /* 0..{} */\n",
            loops[0], global_work_size[0]
        );
        source += &format!(
            "{indent}r{} = get_local_id(0);   /* 0..{} */\n",
            loops[1], local_work_size[0]
        );
        source += &format!(
            "{indent}r{} = get_group_id(1);   /* 0..{} */\n",
            loops[2], global_work_size[1]
        );
        source += &format!(
            "{indent}r{} = get_local_id(1);   /* 0..{} */\n",
            loops[3], local_work_size[1]
        );
        source += &format!(
            "{indent}r{} = get_group_id(2);   /* 0..{} */\n",
            loops[4], global_work_size[2]
        );
        source += &format!(
            "{indent}r{} = get_local_id(2);   /* 0..{} */\n",
            loops[5], local_work_size[2]
        );
        //source += &format!("{indent}printf(\"%f, %f, %f, %f\", p0[0], p0[1], p0[2], p0[3]);\n");

        for op in kernel.ops[6..kernel.ops.len() - 6].iter().copied() {
            match op {
                IROp::Load { z, address, offset } => {
                    if let Reg::Var(id) = offset {
                        if id == 11 {
                            //source += &format!("{indent}printf(\"%u, \", r11);\n");
                        }
                    }
                    source += &format!("{indent}r{z} = p{address}[{}];\n", offset.ocl());
                    //source += &format!( "  printf(\"r{z}, p{address} = %f r2 = %u r4 = %u\\n\", r{z}, r2, r4);\n" );
                }
                IROp::Store { address, offset, x } => {
                    source += &format!("{indent}p{address}[{}] = {};\n", offset.ocl(), x.ocl());
                }
                IROp::Unary { z, x, uop } => {
                    let dtype = kernel.registers[z as usize];
                    source += &match uop {
                        UOp::Cast(_) => {
                            format!("{indent}r{z} = ({})r{x};\n", dtype.ocl())
                        }
                        UOp::ReLU => format!(
                            "{indent}r{z} = max(r{x}, {});\n",
                            dtype.zero_constant().ocl()
                        ),
                        UOp::Neg => format!("{indent}r{z} = -r{x};\n"),
                        UOp::Exp2 => format!("{indent}r{z} = exp2(r{x});\n"),
                        UOp::Log2 => format!("{indent}r{z} = log2(r{x});\n"),
                        UOp::Inv => format!("{indent}r{z} = 1/r{x};\n"),
                        UOp::Sqrt => format!(
                            "{indent}r{z} = sqrt({}r{x});\n",
                            if matches!(dtype, DType::F16) {
                                "(float)"
                            } else {
                                ""
                            }
                        ),
                        UOp::Sin => format!("{indent}r{z} = sin(r{x});\n"),
                        UOp::Cos => format!("{indent}r{z} = cos(r{x});\n"),
                        UOp::Not => format!("{indent}r{z} = !r{x};\n"),
                    };
                }
                IROp::Binary { z, x, y, bop } => {
                    source += &format!(
                        "{indent}r{z} = {};\n",
                        match bop {
                            BOp::Add => format!("{} + {}", x.ocl(), y.ocl()),
                            BOp::Sub => format!("{} - {}", x.ocl(), y.ocl()),
                            BOp::Mul => format!("{} * {}", x.ocl(), y.ocl()),
                            BOp::Div => format!("{} / {}", x.ocl(), y.ocl()),
                            BOp::Mod => format!("{} % {}", x.ocl(), y.ocl()),
                            BOp::Pow => format!("pow({}, {})", x.ocl(), y.ocl()),
                            BOp::Cmplt => format!("{} < {}", x.ocl(), y.ocl()),
                            BOp::Cmpgt => format!("{} > {}", x.ocl(), y.ocl()),
                            BOp::NotEq => format!("{} != {}", x.ocl(), y.ocl()),
                            BOp::Max => format!("max({}, {})", x.ocl(), y.ocl()),
                            BOp::Or => format!("{} || {}", x.ocl(), y.ocl()),
                            BOp::And => format!("{} && {}", x.ocl(), y.ocl()),
                            BOp::BitOr => format!("{} | {}", x.ocl(), y.ocl()),
                            BOp::BitAnd => format!("{} & {}", x.ocl(), y.ocl()),
                            BOp::BitXor => format!("{} ^ {}", x.ocl(), y.ocl()),
                        }
                    );
                    //if z == 24 && bop == BOp::Sub { source += "  printf(\"r24: %f i2; %u i4: %u\\n\", r24, r2, r4);\n"; }
                }
                IROp::MAdd { z, a, b, c } => {
                    source += &format!("{indent}r{z} = {} * {} + {};\n", a.ocl(), b.ocl(), c.ocl());
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
                            Scope::Register | Scope::RegTile => unreachable!(),
                        }
                    );
                }
            }
        }
        source += "}\n";

        let local_work_size = local_work_size;
        let name = format!(
            "k_{}_{}_{}__{}_{}_{}",
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
        let mut pragma = String::new();
        if source.contains("half") {
            pragma += "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
        }
        if source.contains("double") {
            pragma += "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
        }
        let source = format!("{pragma}__kernel void {name}{source}");
        //println!("{source}");
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
                [source.len()].as_ptr(),
                &mut status,
            )
        };
        status.check("Failed to compile program.")?;
        if let Err(e) = unsafe {
            (self.clBuildProgram)(
                program,
                1,
                [device].as_ptr(),
                c"-cl-fast-relaxed-math".as_ptr().cast(),
                None,
                ptr::null_mut(),
            )
        }
        .check("Failed to build program.")
        {
            let build_log = self.get_program_build_data(program, CL_PROGRAM_BUILD_LOG);
            match build_log {
                Ok(build_log) => {
                    panic!("{e:?} {}", String::from_utf8_lossy(&build_log));
                }
                Err(status) => status.check(&format!(
                    "Failed to get info about failed compilation. {e:?}"
                ))?,
            }
        }
        let mut status = OpenCLStatus::CL_SUCCESS;
        let program_name = &CString::new(name).unwrap();
        let kernel =
            unsafe { (self.clCreateKernel)(program, program_name.as_ptr().cast(), &mut status) };
        status.check("Failed to create kernel.")?;
        Ok(OpenCLProgram {
            program,
            kernel,
            global_work_size,
            local_work_size,
        })
    }
}

impl OpenCLQueue {
    pub(super) fn launch(
        &mut self,
        program: &mut OpenCLProgram,
        buffers: &mut IndexMap<OpenCLBuffer>,
        args: &[Id],
    ) -> Result<OpenCLEvent, OpenCLError> {
        /*println!(
            "Launch opencl kernel {:?}, program {:?} on queue {:?}, gws {:?}, lws {:?}",
            program.kernel,
            program.program,
            self.queue,
            program.global_work_size,
            program.local_work_size
        );*/
        let mut i = 0;
        #[allow(clippy::explicit_counter_loop)]
        for arg in args {
            let arg = &mut buffers[*arg];
            //println!("Kernel arg: {arg:?} at index {i}");
            let ptr: *const _ = &arg.ptr;
            unsafe {
                (self.clSetKernelArg)(
                    program.kernel,
                    i,
                    core::mem::size_of::<*mut c_void>(),
                    ptr.cast(),
                )
            }
            .check("Failed to set kernel arg.")?;
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
        Ok(OpenCLEvent {
            event,
            clWaitForEvents: self.clWaitForEvents,
        })
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

    pub(super) const fn load(&self) -> usize {
        self.load
    }
}

impl OpenCLEvent {
    pub(super) fn finish(self) -> Result<(), OpenCLError> {
        unsafe { (self.clWaitForEvents)(1, [self.event].as_ptr()) }.check("Failed to finish event")
    }
}

impl OpenCLStatus {
    fn check(self, info: &str) -> Result<(), OpenCLError> {
        if self == Self::CL_SUCCESS {
            Ok(())
        } else {
            Err(OpenCLError {
                info: info.into(),
                status: self,
            })
        }
    }
}

impl OpenCLDevice {
    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn release_program(&self, program: OpenCLProgram) -> Result<(), OpenCLError> {
        //println!("Releasing {:?}", program);
        unsafe { (self.clReleaseProgram)(program.program) }
            .check("Failed to release OpenCL program")
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn release_queue(&self, queue: OpenCLQueue) -> Result<(), OpenCLError> {
        unsafe { (self.clReleaseCommandQueue)(queue.queue) }
            .check("Failed to release OpenCL queue.")
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
            if OpenCLStatus::CL_SUCCESS == status {
                Ok(size)
            } else {
                Err(status)
            }
        }?;
        if 0 < size {
            let count = size / core::mem::size_of::<u8>();
            let mut data: Vec<u8> = Vec::with_capacity(count);
            let status = unsafe {
                data.set_len(count);
                (self.clGetProgramBuildInfo)(
                    program,
                    self.ptr,
                    param_name,
                    size,
                    data.as_mut_ptr().cast(),
                    ptr::null_mut(),
                )
            };
            if OpenCLStatus::CL_SUCCESS == status {
                Ok(data)
            } else {
                Err(status)
            }
        } else {
            Ok(Vec::default())
        }
    }

    fn get_device_data(&mut self, param_name: cl_uint) -> Result<Vec<u8>, OpenCLError> {
        let size = {
            let object = self.ptr;
            let mut size: usize = 0;
            let status = unsafe {
                (self.clGetDeviceInfo)(object, param_name, 0, ptr::null_mut(), &mut size)
            };
            if OpenCLStatus::CL_SUCCESS != status {
                return Err(OpenCLError {
                    status,
                    info: format!("Failed to get device info {param_name}"),
                });
            }
            Ok(size)
        }?;
        let object = self.ptr;
        if 0 < size {
            let count = size / core::mem::size_of::<u8>();
            let mut data: Vec<u8> = Vec::with_capacity(count);
            unsafe {
                data.set_len(count);
                (self.clGetDeviceInfo)(
                    object,
                    param_name,
                    size,
                    data.as_mut_ptr().cast(),
                    ptr::null_mut(),
                )
            }
            .check(&format!("Failed to get {param_name}"))?;
            Ok(data)
        } else {
            Ok(Vec::default())
        }
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

//#[allow(dead_code)] // Rust for some reason thinks these fields are unused
#[derive(Debug)]
pub struct OpenCLError {
    info: String,
    status: OpenCLStatus,
}

impl std::fmt::Display for OpenCLError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "OpenCLError {{ info: {:?}, status: {:?} }}",
            self.info, self.status
        ))
    }
}

#[allow(clippy::upper_case_acronyms)]
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

impl DType {
    fn ocl(self) -> String {
        match self {
            Self::BF16 => todo!("bf16 should be casted to f16 or f32"),
            Self::F8 => format!("f8"),
            Self::F16 => format!("half"),
            Self::F32 => format!("float"),
            Self::F64 => format!("double"),
            Self::U8 => format!("unsigned char"),
            Self::U16 => format!("unsigned short"),
            Self::I8 => format!("char"),
            Self::I16 => format!("short"),
            Self::I32 => format!("int"),
            Self::I64 => format!("long"),
            Self::Bool => "bool".into(),
            Self::U32 => format!("unsigned int"),
            Self::U64 => format!("unsigned long"),
        }
    }
}

impl Constant {
    fn ocl(&self) -> String {
        match self {
            &Self::BF16(x) => format!("{:.16}f", half::bf16::from_bits(x)),
            &Self::F8(x) => format!("{:.8}f", float8::F8E4M3::from_bits(x)),
            &Self::F16(x) => format!("{:.16}f", half::f16::from_bits(x)),
            &Self::F32(x) => format!("{:.16}f", f32::from_bits(x)),
            &Self::F64(x) => format!("{:.16}", f64::from_bits(x)),
            Self::U8(x) => format!("{x}"),
            Self::I8(x) => format!("{x}"),
            Self::I16(x) => format!("{x}"),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}"),
            Self::U64(x) => format!("{x}"),
            Self::I32(x) => format!("{x}"),
            Self::I64(x) => format!("{x}"),
            Self::Bool(x) => format!("{x}"),
        }
    }
}

impl Reg {
    fn ocl(&self) -> String {
        match self {
            Self::Var(id) => format!("r{id}"),
            Self::Const(value) => value.ocl(),
        }
    }
}
