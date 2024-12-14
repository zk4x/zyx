//! Cuda backend

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused)]

use std::ffi::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::Arc;

use float8::F8E4M3;
use libloading::Library;
use nanoserde::DeJson;

use super::DeviceInfo;
use crate::dtype::Constant;
use crate::ir::IRKernel;
use crate::ir::{IROp, Reg, Scope};
use crate::node::{BOp, UOp};
use crate::slab::{Id, Slab};
use crate::DType;

/// CUDA configuration
#[derive(Debug, Default, DeJson)]
pub struct CUDAConfig {
    device_ids: Option<Vec<i32>>,
}

#[derive(Debug)]
pub struct CUDAError {
    info: String,
    status: CUDAStatus,
}

impl std::fmt::Display for CUDAError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "CUDAError {{ info: {:?}, status: {:?} }}",
            self.info, self.status
        ))
    }
}

#[derive(Debug)]
pub(super) struct CUDAMemoryPool {
    // Just to close the connection
    #[allow(unused)]
    cuda: Arc<Library>,
    context: CUcontext,
    device: CUdevice,
    free_bytes: usize,
    cuMemAlloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUDAStatus,
    cuMemcpyHtoD: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUDAStatus,
    cuMemcpyDtoH: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUDAStatus,
    cuMemFree: unsafe extern "C" fn(CUdeviceptr) -> CUDAStatus,
    cuMemcpyPeer:
        unsafe extern "C" fn(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, usize) -> CUDAStatus,
    //cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUDAStatus,
    //cuCtxDestroy: unsafe extern "C" fn(CUcontext) -> CUDAStatus,
}

#[derive(Debug)]
pub(super) struct CUDABuffer {
    ptr: u64,
    context: CUcontext,
    bytes: usize,
}

#[derive(Debug)]
pub(super) struct CUDADevice {
    device: CUdevice,
    memory_pool_id: u32,
    dev_info: DeviceInfo,
    compute_capability: [c_int; 2],
    cuModuleLoadDataEx: unsafe extern "C" fn(
        *mut CUmodule,
        *const c_void,
        c_uint,
        *mut CUjit_option,
        *mut *mut c_void,
    ) -> CUDAStatus,
    cuModuleGetFunction:
        unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUDAStatus,
    cuModuleUnload: unsafe extern "C" fn(CUmodule) -> CUDAStatus,
    cuStreamDestroy: unsafe extern "C" fn(CUstream) -> CUDAStatus,
}

#[derive(Debug)]
pub(super) struct CUDAProgram {
    //name: String,
    module: CUmodule,
    function: CUfunction,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
}

#[derive(Debug)]
pub(super) struct CUDAQueue {
    stream: CUstream,
    load: usize,
    cuLaunchKernel: unsafe extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUDAStatus,
    cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUDAStatus,
}

#[derive(Debug)]
pub(super) struct CUDAEvent {}

impl CUDAEvent {
    pub(super) fn finish(self) -> Result<(), CUDAError> {
        Ok(())
    }
}

// This is currently just wrong, CUDA uses thread locals that can't be send ...
unsafe impl Send for CUDAMemoryPool {}
unsafe impl Send for CUDABuffer {}
unsafe impl Send for CUDAProgram {}
unsafe impl Send for CUDAQueue {}

type CUDAQueuePool = Vec<(CUDADevice, Vec<CUDAQueue>)>;

pub(super) fn initialize_devices(
    config: &CUDAConfig,
    debug_dev: bool,
) -> Result<(Vec<CUDAMemoryPool>, CUDAQueuePool), CUDAError> {
    let _ = config;

    let cuda_paths = ["/lib/x86_64-linux-gnu/libcuda.so", "/lib64/libcuda.so"];
    let cuda = cuda_paths.iter().find_map(|path| unsafe { Library::new(path) }.ok());
    let Some(cuda) = cuda else {
        return Err(CUDAError {
            info: String::from("CUDA runtime not found."),
            status: CUDAStatus::CUDA_ERROR_UNKNOWN,
        });
    };

    let cuInit: unsafe extern "C" fn(c_uint) -> CUDAStatus =
        *unsafe { cuda.get(b"cuInit\0") }.unwrap();
    let cuDriverGetVersion: unsafe extern "C" fn(*mut c_int) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDriverGetVersion\0") }.unwrap();
    let cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDeviceGetCount\0") }.unwrap();
    let cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDeviceGet\0") }.unwrap();
    let cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDeviceGetName\0") }.unwrap();
    let cuDeviceComputeCapability: unsafe extern "C" fn(
        *mut c_int,
        *mut c_int,
        CUdevice,
    ) -> CUDAStatus = *unsafe { cuda.get(b"cuDeviceComputeCapability\0") }.unwrap();
    let cuDeviceTotalMem: unsafe extern "C" fn(*mut usize, CUdevice) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDeviceTotalMem\0") }.unwrap();
    let cuDeviceGetAttribute: unsafe extern "C" fn(
        *mut c_int,
        CUdevice_attribute,
        CUdevice,
    ) -> CUDAStatus = *unsafe { cuda.get(b"cuDeviceGetAttribute\0") }.unwrap();
    let cuCtxCreate: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUDAStatus =
        *unsafe { cuda.get(b"cuCtxCreate\0") }.unwrap();
    let cuMemAlloc = *unsafe { cuda.get(b"cuMemAlloc\0") }.unwrap();
    let cuMemcpyHtoD = *unsafe { cuda.get(b"cuMemcpyHtoD\0") }.unwrap();
    let cuMemFree = *unsafe { cuda.get(b"cuMemFree\0") }.unwrap();
    let cuMemcpyDtoH = *unsafe { cuda.get(b"cuMemcpyDtoH\0") }.unwrap();
    let cuMemcpyPeer = *unsafe { cuda.get(b"cuMemcpyPeer\0") }.unwrap();
    //let cuCtxSetCurrent = *unsafe { cuda.get(b"cuCtxGetCurrent\0") }.unwrap();
    //let cuCtxDestroy = *unsafe { cuda.get(b"cuCtxDestroy\0") }.unwrap();
    let cuModuleLoadDataEx = *unsafe { cuda.get(b"cuModuleLoadDataEx\0") }.unwrap();
    let cuModuleGetFunction = *unsafe { cuda.get(b"cuModuleGetFunction\0") }.unwrap();
    let cuLaunchKernel = *unsafe { cuda.get(b"cuLaunchKernel\0") }.unwrap();
    let cuStreamCreate: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUDAStatus =
        *unsafe { cuda.get(b"cuStreamCreate\0") }.unwrap();
    let cuStreamSynchronize = *unsafe { cuda.get(b"cuStreamSynchronize\0") }.unwrap();
    let cuStreamDestroy = *unsafe { cuda.get(b"cuStreamDestroy\0") }.unwrap();
    let cuModuleUnload = *unsafe { cuda.get(b"cuModuleUnload\0") }.unwrap();
    //let cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUDAStatus = *unsafe { cuda.get(b"cuDevicePrimaryCtxRetain\0") }.unwrap();

    unsafe { cuInit(0) }.check("Failed to init CUDA")?;
    let mut driver_version = 0;
    unsafe { cuDriverGetVersion(&mut driver_version) }
        .check("Failed to get CUDA driver version")?;
    let mut num_devices = 0;
    unsafe { cuDeviceGetCount(&mut num_devices) }.check("Failed to get CUDA device count")?;
    if num_devices == 0 {
        return Err(CUDAError {
            info: "No available cuda device.".into(),
            status: CUDAStatus::CUDA_ERROR_UNKNOWN,
        });
    }
    let device_ids: Vec<i32> = (0..num_devices)
        .filter(|id| config.device_ids.as_ref().map_or(true, |ids| ids.contains(id)))
        .collect();
    if debug_dev && !device_ids.is_empty() {
        println!(
            "Using CUDA driver, driver version: {}.{} on devices:",
            driver_version / 1000,
            (driver_version - (driver_version / 1000 * 1000)) / 10
        );
    }

    let cuda = Arc::new(cuda);
    let mut memory_pools = Vec::new();
    let mut devices = Vec::new();
    for dev_id in device_ids {
        let mut device = 0;
        unsafe { cuDeviceGet(&mut device, dev_id) }.check("Failed to access CUDA device")?;
        let mut device_name = [0; 100];
        let Ok(()) = unsafe { cuDeviceGetName(device_name.as_mut_ptr(), 100, device) }
            .check("Failed to get CUDA device name")
        else {
            continue;
        };
        let mut major = 0;
        let mut minor = 0;
        let Ok(()) = unsafe { cuDeviceComputeCapability(&mut major, &mut minor, device) }
            .check("Failed to get CUDA device compute capability.")
        else {
            continue;
        };
        if debug_dev {
            println!("{:?}, compute capability: {major}.{minor}", unsafe {
                std::ffi::CStr::from_ptr(device_name.as_ptr())
            });
        }
        let mut free_bytes = 0;
        let Ok(()) =
            unsafe { cuDeviceTotalMem(&mut free_bytes, device) }.check("Failed to get dev mem.")
        else {
            continue;
        };
        let mut context: CUcontext = ptr::null_mut();
        if let Err(e) =
            unsafe { cuCtxCreate(&mut context, 0, device) }.check("Failed to create CUDA context.")
        {
            println!("{e:?}");
            continue;
        }
        /*if let Err(e) = unsafe { cuDevicePrimaryCtxRetain(&mut context, device) }.check("Failed to create CUDA context.") {
            println!("{e:?}");
            continue;
        }*/
        //println!("Using context {context:?} and device {device:?}");
        memory_pools.push(CUDAMemoryPool {
            cuda: cuda.clone(),
            context,
            device,
            free_bytes,
            cuMemAlloc,
            cuMemcpyHtoD,
            cuMemFree,
            cuMemcpyDtoH,
            cuMemcpyPeer,
            //cuCtxSetCurrent,
            //cuCtxDestroy,
        });
        let mut queues = Vec::new();
        for _ in 0..8 {
            let mut stream = ptr::null_mut();
            let Ok(()) = unsafe { cuStreamCreate(&mut stream, 0) }.check("") else {
                continue;
            };
            queues.push(CUDAQueue { stream, load: 0, cuLaunchKernel, cuStreamSynchronize });
        }
        devices.push((
            CUDADevice {
                device,
                dev_info: DeviceInfo {
                    compute: 1024 * 1024 * 1024 * 1024,
                    max_global_work_dims: [64, 64, 64],
                    max_local_threads: 1,
                    max_local_work_dims: [1, 1, 1],
                    local_mem_size: 0,
                    num_registers: 96,
                    preferred_vector_size: 16,
                    tensor_cores: major > 7,
                },
                memory_pool_id: u32::try_from(memory_pools.len()).unwrap() - 1,
                cuModuleLoadDataEx,
                cuModuleGetFunction,
                cuModuleUnload,
                cuStreamDestroy,
                compute_capability: [major, minor],
            },
            queues,
        ));
        let dev = &mut devices.last_mut().unwrap().0;
        dev.dev_info = DeviceInfo {
            compute: 1024 * 1024 * 1024 * 1024,
            max_global_work_dims: [
                usize::try_from(dev.get(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                    cuDeviceGetAttribute,
                )?)
                .unwrap(),
                usize::try_from(dev.get(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                    cuDeviceGetAttribute,
                )?)
                .unwrap(),
                usize::try_from(dev.get(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
                    cuDeviceGetAttribute,
                )?)
                .unwrap(),
            ],
            max_local_threads: usize::try_from(dev.get(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                cuDeviceGetAttribute,
            )?)
            .unwrap(),
            max_local_work_dims: [
                usize::try_from(dev.get(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                    cuDeviceGetAttribute,
                )?)
                .unwrap(),
                usize::try_from(dev.get(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                    cuDeviceGetAttribute,
                )?)
                .unwrap(),
                usize::try_from(dev.get(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                    cuDeviceGetAttribute,
                )?)
                .unwrap(),
            ],
            local_mem_size: usize::try_from(dev.get(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                cuDeviceGetAttribute,
            )?)
            .unwrap(),
            num_registers: 96,
            preferred_vector_size: 16,
            tensor_cores: major > 7,
        }
    }

    Ok((memory_pools, devices))
}

impl CUDAMemoryPool {
    #[allow(clippy::unused_self)]
    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn deinitialize(self) -> Result<(), CUDAError> {
        //unsafe { (self.cuCtxDestroy)(self.context) }.check("Failed to destroy CUDA context.")?;
        Ok(())
    }

    pub(super) const fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    pub(super) fn allocate(&mut self, bytes: usize) -> Result<CUDABuffer, CUDAError> {
        if bytes > self.free_bytes {
            return Err(CUDAError {
                info: "Insufficient free memory.".into(),
                status: CUDAStatus::CUDA_ERROR_OUT_OF_MEMORY,
            });
        }
        //println!("Allocating to context {:?}, device {:?}", self.context, self.device);
        self.free_bytes -= bytes;
        let mut ptr = u64::try_from(self.device).unwrap();
        //unsafe { (self.cuCtxSetCurrent)(self.context) }.check("Failed to set current CUDA context.")?;
        unsafe { (self.cuMemAlloc)(&mut ptr, bytes) }.check("Failed to allocate memory.")?;
        Ok(CUDABuffer { ptr, bytes, context: self.context })
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn deallocate(&mut self, buffer: CUDABuffer) -> Result<(), CUDAError> {
        unsafe { (self.cuMemFree)(buffer.ptr) }.check("Failed to free memory.")?;
        self.free_bytes += buffer.bytes;
        Ok(())
    }

    pub(super) fn host_to_pool(&mut self, src: &[u8], dst: &CUDABuffer) -> Result<(), CUDAError> {
        //println!("Copying {src:?} to {dst:?}");
        //unsafe { (self.cuCtxSetCurrent)(self.context) }.check("Failed to set current CUDA context.")?;
        unsafe { (self.cuMemcpyHtoD)(dst.ptr, src.as_ptr().cast(), src.len()) }
            .check("Failed to copy memory from host to pool.")
    }

    pub(super) fn pool_to_host(
        &mut self,
        src: &CUDABuffer,
        dst: &mut [u8],
    ) -> Result<(), CUDAError> {
        unsafe { (self.cuMemcpyDtoH)(dst.as_mut_ptr().cast(), src.ptr, dst.len()) }
            .check("Failed to copy memory from pool to host.")
    }

    pub(super) fn pool_to_pool(
        &mut self,
        src: &CUDABuffer,
        dst: &CUDABuffer,
    ) -> Result<(), CUDAError> {
        unsafe { (self.cuMemcpyPeer)(dst.ptr, dst.context, src.ptr, src.context, dst.bytes) }
            .check("Failed copy memory from pool to pool.")
    }
}

impl CUDADevice {
    #[allow(clippy::unused_self)]
    #[allow(clippy::unnecessary_wraps)]
    pub(super) const fn deinitialize(self) -> Result<(), CUDAError> {
        Ok(())
    }

    fn get(
        &mut self,
        attr: CUdevice_attribute,
        cuDeviceGetAttribute: unsafe extern "C" fn(
            *mut c_int,
            CUdevice_attribute,
            CUdevice,
        ) -> CUDAStatus,
    ) -> Result<c_int, CUDAError> {
        let mut v = 0;
        unsafe { cuDeviceGetAttribute(&mut v, attr, self.device) }
            .check("Failed to get device attribute.")?;
        Ok(v)
    }

    pub(super) const fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    // Memory pool id out of OpenCLMemoryPools
    pub(super) const fn memory_pool_id(&self) -> u32 {
        self.memory_pool_id
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn release_program(&self, program: CUDAProgram) -> Result<(), CUDAError> {
        unsafe { (self.cuModuleUnload)(program.module) }.check("Failed to release CUDA program.")
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn release_queue(&self, queue: CUDAQueue) -> Result<(), CUDAError> {
        unsafe { (self.cuStreamDestroy)(queue.stream) }.check("Failed to release CUDA stream.")
    }

    pub(super) fn compile(
        &mut self,
        kernel: &IRKernel,
        debug_asm: bool,
    ) -> Result<CUDAProgram, CUDAError> {
        let (global_work_size, local_work_size, name, ptx_vec) =
            self.compile_cuda(kernel, debug_asm)?;
        //self.compile_ptx(kernel, debug_asm)?;

        let mut module = ptr::null_mut();
        unsafe {
            (self.cuModuleLoadDataEx)(
                &mut module,
                ptx_vec.as_ptr().cast(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        }
        .check("Module load failed.")?;
        let mut function: CUfunction = ptr::null_mut();
        // Don't forget that the name is null terminated string
        unsafe { (self.cuModuleGetFunction)(&mut function, module, name.as_ptr().cast()) }
            .check("Failed to load function.")?;

        Ok(CUDAProgram {
            //name,
            module,
            function,
            global_work_size,
            local_work_size,
        })
    }

    #[allow(clippy::type_complexity)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    fn compile_cuda(
        &mut self,
        kernel: &IRKernel,
        debug_asm: bool,
    ) -> Result<([usize; 3], [usize; 3], String, Vec<u8>), CUDAError> {
        let mut source = String::from("(\n");
        let mut indent = String::from("  ");

        let mut global_work_size = [0; 3];
        let mut local_work_size = [0; 3];

        let mut loop_ids = [0; 6];
        for (i, op) in kernel.ops[..6].iter().enumerate() {
            if let IROp::Loop { id, len } = op {
                if i % 2 == 0 {
                    global_work_size[i / 2] = *len;
                } else {
                    local_work_size[i / 2] = *len;
                }
                loop_ids[i] = *id;
            } else {
                unreachable!()
            }
        }

        // Declare global variables
        for (id, (scope, dtype, _, read_only)) in kernel.addressables.iter().enumerate() {
            if *scope == Scope::Global {
                source.push_str(&format!(
                    "{indent}{}{}* p{id},\n",
                    if *read_only { "const " } else { "" },
                    dtype.cu(),
                ));
            }
        }

        source.pop();
        source.pop();
        source.push_str("\n) {\n");

        // Declare local variables
        for (id, (scope, dtype, len, _)) in kernel.addressables.iter().enumerate() {
            if *scope == Scope::Local {
                source.push_str(&format!(
                    "{indent}__shared__ {} p{id}[{len}];\n",
                    //if *read_only { "const " } else { "" },
                    dtype.cu(),
                ));
            }
        }

        // Declare accumulators
        for (id, (scope, dtype, len, read_only)) in kernel.addressables.iter().enumerate() {
            if *scope == Scope::RegTile {
                source.push_str(&format!(
                    "{indent}{}{} p{id}[{len}];\n",
                    if *read_only { "const " } else { "" },
                    dtype.cu(),
                ));
            }
        }

        // Declare register variables
        for (id, dtype) in kernel.registers.iter().enumerate() {
            source.push_str(&format!("{indent}{} r{id};\n", dtype.cu()));
        }

        // Add indices for global and local loops
        source.push_str(&format!(
            "  r{} = blockIdx.x;   /* 0..{} */\n",
            loop_ids[0], global_work_size[0]
        ));
        source.push_str(&format!(
            "  r{} = threadIdx.x;   /* 0..{} */\n",
            loop_ids[1], local_work_size[0]
        ));
        source.push_str(&format!(
            "  r{} = blockIdx.y;   /* 0..{} */\n",
            loop_ids[2], global_work_size[1]
        ));
        source.push_str(&format!(
            "  r{} = threadIdx.y;   /* 0..{} */\n",
            loop_ids[3], local_work_size[1]
        ));
        source.push_str(&format!(
            "  r{} = blockIdx.z;   /* 0..{} */\n",
            loop_ids[4], global_work_size[2]
        ));
        source.push_str(&format!(
            "  r{} = threadIdx.z;   /* 0..{} */\n",
            loop_ids[5], local_work_size[2]
        ));

        for op in kernel.ops[6..kernel.ops.len() - 6].iter().copied() {
            match op {
                IROp::Load { z, address, offset } => {
                    source.push_str(&format!("{indent}r{z} = p{address}[{}];\n", offset.cu()));
                }
                IROp::Store { address, offset, x } => {
                    source.push_str(&format!(
                        "{indent}p{address}[{}] = {};\n",
                        offset.cu(),
                        x.cu()
                    ));
                }
                IROp::Unary { z, x, uop } => {
                    let dtype = kernel.registers[z as usize];
                    let zero = Constant::new(0).unary(UOp::Cast(dtype)).cu();
                    source.push_str(&match uop {
                        UOp::Cast(_) => {
                            format!("{indent}r{} = ({})r{};\n", z, dtype.cu(), x)
                        }
                        UOp::ReLU => {
                            if dtype == DType::F16 {
                                format!("{indent}r{z} = r{x} * __float2half(r{x} > {zero});\n")
                            } else {
                                format!("{indent}r{z} = r{x} * (r{x} > {zero});\n")
                            }
                        }
                        UOp::Neg => format!("{indent}r{z} = -r{x};\n"),
                        UOp::Exp2 => format!("{indent}r{z} = exp2(r{x});\n"),
                        UOp::Log2 => format!("{indent}r{z} = log2(r{x});\n"),
                        UOp::Inv => format!("{indent}r{z} = 1/r{x};\n"),
                        UOp::Sqrt => format!("{indent}r{z} = sqrt(r{x});\n"),
                        UOp::Sin => format!("{indent}r{z} = sin(r{x});\n"),
                        UOp::Cos => format!("{indent}r{z} = cos(r{x});\n"),
                        UOp::Not => format!("{indent}r{z} = !r{x};\n"),
                    });
                }
                IROp::Binary { z, x, y, bop } => {
                    source.push_str(&format!(
                        "{indent}r{z} = {};\n",
                        match bop {
                            BOp::Add => format!("{} + {}", x.cu(), y.cu()),
                            BOp::Sub => format!("{} - {}", x.cu(), y.cu()),
                            BOp::Mul => format!("{} * {}", x.cu(), y.cu()),
                            BOp::Div => format!("{} / {}", x.cu(), y.cu()),
                            BOp::Mod => format!("{} % {}", x.cu(), y.cu()),
                            BOp::Pow => format!("pow({}, {})", x.cu(), y.cu()),
                            BOp::Cmplt => format!("{} < {}", x.cu(), y.cu()),
                            BOp::Cmpgt => format!("{} > {}", x.cu(), y.cu()),
                            BOp::Max => format!("max({}, {})", x.cu(), y.cu()),
                            BOp::Or => format!("{} || {}", x.cu(), y.cu()),
                            BOp::And => format!("{} && {}", x.cu(), y.cu()),
                            BOp::BitOr => format!("{} | {}", x.cu(), y.cu()),
                            BOp::BitAnd => format!("{} & {}", x.cu(), y.cu()),
                            BOp::BitXor => format!("{} ^ {}", x.cu(), y.cu()),
                            BOp::NotEq => format!("{} != {}", x.cu(), y.cu()),
                        }
                    ));
                }
                IROp::MAdd { z, a, b, c } => {
                    source.push_str(&format!(
                        "{indent}r{z} = {} * {} + {};\n",
                        a.cu(),
                        b.cu(),
                        c.cu()
                    ));
                }
                IROp::Loop { id, len } => {
                    source.push_str(&format!(
                        "{indent}for (unsigned int r{id} = 0; r{id} < {len}; r{id} += 1) {{\n"
                    ));
                    indent.push_str("  ");
                }
                IROp::EndLoop { .. } => {
                    indent.pop();
                    indent.pop();
                    source.push_str(&format!("{indent}}}\n"));
                }
                IROp::Barrier { scope } => {
                    source.push_str(&format!(
                        "{};\n",
                        match scope {
                            Scope::Global => "__threadfence()",
                            Scope::Local => "__syncthreads()",
                            Scope::Register | Scope::RegTile => unreachable!(),
                        }
                    ));
                }
            }
        }
        source += "}\n";

        let mut name = format!(
            "k_{}_{}_{}__{}_{}_{}",
            global_work_size[0],
            global_work_size[1],
            global_work_size[2],
            local_work_size[0],
            local_work_size[1],
            local_work_size[2],
        );
        let mut pragma = String::new();
        if source.contains("__half") {
            pragma += "#include <cuda_fp16.h>\n";
        }
        let source = format!("{pragma}extern \"C\" __global__ void {name}{source}\0");
        name += "\0";
        if debug_asm {
            println!("{source}");
        }

        let cudartc_paths = [
            "/lib/x86_64-linux-gnu/libnvrtc.so",
            "/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so",
        ];
        let cudartc = cudartc_paths.iter().find_map(|path| unsafe { Library::new(path) }.ok());
        let Some(cudartc) = cudartc else {
            return Err(CUDAError {
                info: "CUDA runtime not found.".into(),
                status: CUDAStatus::CUDA_ERROR_UNKNOWN,
            });
        };
        let nvrtcCreateProgram: unsafe extern "C" fn(
            *mut nvrtcProgram,
            *const c_char,
            *const c_char,
            c_int,
            *const *const c_char,
            *const *const c_char,
        ) -> nvrtcResult = *unsafe { cudartc.get(b"nvrtcCreateProgram\0") }.unwrap();
        let nvrtcCompileProgram: unsafe extern "C" fn(
            nvrtcProgram,
            c_int,
            *const *const c_char,
        ) -> nvrtcResult = *unsafe { cudartc.get(b"nvrtcCompileProgram\0") }.unwrap();
        let nvrtcGetPTXSize: unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult =
            *unsafe { cudartc.get(b"nvrtcGetPTXSize\0") }.unwrap();
        let nvrtcGetPTX: unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> nvrtcResult =
            *unsafe { cudartc.get(b"nvrtcGetPTX\0") }.unwrap();
        let nvrtcGetProgramLogSize: unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult =
            *unsafe { cudartc.get(b"nvrtcGetProgramLogSize\0") }.unwrap();
        let nvrtcGetProgramLog: unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> nvrtcResult =
            *unsafe { cudartc.get(b"nvrtcGetProgramLog\0") }.unwrap();
        let nvrtcDestroyProgram: unsafe extern "C" fn(*mut nvrtcProgram) -> nvrtcResult =
            *unsafe { cudartc.get(b"nvrtcDestroyProgram\0") }.unwrap();

        //let include_folders = ["/usr/local/cuda-12.6/targets/x86_64-linux/include/\0".as_ptr().cast()];
        //let include_files = ["cuda_fp16.h\0".as_ptr().cast()];
        let mut program = ptr::null_mut();
        unsafe {
            nvrtcCreateProgram(
                &mut program,
                source.as_ptr().cast(),
                name.as_ptr().cast(),
                0,
                ptr::null_mut(), //include_folders.as_ptr(),
                ptr::null_mut(), //include_files.as_ptr(),
            )
        }
        .check("nvrtcCreateProgram")?;
        let df = format!(
            "--gpu-architecture=compute_{}{}\0",
            self.compute_capability[0], self.compute_capability[1]
        );
        let opts = [
            df.as_ptr().cast(),
            "-I/usr/local/cuda-12.6/targets/x86_64-linux/include\0".as_ptr().cast(),
        ];
        if let Err(e) =
            unsafe { nvrtcCompileProgram(program, 2, opts.as_ptr()) }.check("nvrtcCompileProgram")
        {
            println!("CUDA compilation error {e:?}");
            let mut program_log_size: usize = 0;
            unsafe { nvrtcGetProgramLogSize(program, &mut program_log_size) }
                .check("nvrtcGetProgramLogSize")?;
            let mut program_log_vec: Vec<u8> = vec![0; program_log_size + 1];
            unsafe { nvrtcGetProgramLog(program, program_log_vec.as_mut_ptr().cast()) }
                .check("nvrtcGetProgramLog")?;
            if let Ok(log) = String::from_utf8(program_log_vec) {
                println!("NVRTC program log:\n{log}",);
            } else {
                println!("NVRTC program log is not valid utf8");
            }
        }
        let mut ptx_size: usize = 0;
        unsafe { nvrtcGetPTXSize(program, &mut ptx_size) }.check("nvrtcGetPTXSize")?;
        let mut ptx_vec: Vec<u8> = vec![0; ptx_size];
        unsafe { nvrtcGetPTX(program, ptx_vec.as_mut_ptr().cast()) }.check("nvrtcGetPTX")?;
        unsafe { nvrtcDestroyProgram(&mut program) }.check("nvrtcDestoyProgram")?;
        Ok((global_work_size, local_work_size, name, ptx_vec))
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    fn compile_ptx(
        &mut self,
        kernel: &IRKernel,
        debug_asm: bool,
    ) -> Result<([usize; 3], [usize; 3], String, Vec<u8>), CUDAError> {
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
        let name = format!(
            "k__{}_{}__{}_{}__{}_{}",
            global_work_size[0],
            local_work_size[0],
            global_work_size[1],
            local_work_size[1],
            global_work_size[2],
            local_work_size[2],
        );

        let indent = "    ";
        let mut source = format!(
            ".version {0}.{1}
.target sm_{0}{1}
.address_size 64
.visible .entry {name}(\n",
            self.compute_capability[0], self.compute_capability[1]
        );
        // Declare global variables
        for (id, (scope, _, _, read_only)) in kernel.addressables.iter().enumerate() {
            if *scope == Scope::Global {
                source += &format!("{indent}.param    .u64 g{id},\n");
            }
        }
        source.pop();
        source.pop();
        source += "\n) {\n";

        // TOOD declare local variables

        // Temporaries
        source += &format!("{indent}.reg  .pred    p;\n");
        source += &format!("{indent}.reg  .s64    a0;\n");
        source += &format!("{indent}.reg  .s64    a1;\n");
        // Declare register variables
        for (id, dtype) in kernel.registers.iter().enumerate() {
            source += &format!("{indent}.reg  .{}    r{id};\n", dtype.ptx());
        }
        // Add indices for global and local loops
        source += &format!("{indent}mov.u32    r0, %ctaid.x;\n");
        source += &format!("{indent}mov.u32    r1, %tid.x;\n");
        source += &format!("{indent}mov.u32    r2, %ctaid.y;\n");
        source += &format!("{indent}mov.u32    r3, %tid.y;\n");
        source += &format!("{indent}mov.u32    r4, %ctaid.z;\n");
        source += &format!("{indent}mov.u32    r5, %tid.z;\n");

        for op in kernel.ops[6..kernel.ops.len() - 6].iter().copied() {
            match op {
                IROp::Load { z, address, offset } => {
                    let dtype = kernel.registers[z as usize];
                    // Get address
                    source += &format!(
                        "{indent}ld.param.u64    a0, [a{address}+{}];\n",
                        offset.ptx()
                    );
                    // Convert address to global
                    source += &format!("{indent}cvta.to.global.u64    a1, a0;\n");
                    // Load from global to register
                    source += &format!("{indent}ld.global.{}    r{}, [a1];\n", dtype.ptx(), z);
                }
                IROp::Store { address, offset, x } => {
                    let dtype = match x {
                        Reg::Var(id) => kernel.registers[id as usize],
                        Reg::Const(constant) => constant.dtype(),
                    };
                    // Get address
                    source += &format!("{indent}ld.param.u64    a0, [a{address}];\n");
                    // Convert address to global
                    source += &format!("{indent}cvta.to.global.u64    a1, a0;\n");
                    // Load from global to register
                    source += &format!("{indent}st.global.{}    [a1], {};\n", dtype.ptx(), x.ptx());
                }
                IROp::Unary { z, x, uop } => {
                    let dtype = kernel.registers[z as usize];
                    source += &match uop {
                        UOp::Cast(cdt) => format!(
                            "{indent}cvt.{}.{}    r{z}, r{x};\n",
                            <DType as Into<DType>>::into(cdt).ptx(),
                            dtype.ptx(),
                        ),
                        UOp::ReLU => todo!(),
                        UOp::Neg => {
                            format!("{indent}neg.{}   r{z}, r{x};\n", dtype.ptx())
                        }
                        UOp::Exp2 => format!("{indent}ex2.approx.{}   r{z}, r{x};\n", dtype.ptx(),),
                        UOp::Log2 => format!("{indent}lg2.approx.{}   r{z}, r{x};\n", dtype.ptx(),),
                        UOp::Inv => todo!(),
                        UOp::Sqrt => {
                            format!("{indent}sqrt.approx.{}   r{z}, r{x};\n", dtype.ptx(),)
                        }
                        UOp::Sin => format!("{indent}sin.approx.{}   r{z}, r{x};\n", dtype.ptx(),),
                        UOp::Cos => format!("{indent}cos.approx.{}   r{z}, r{x};\n", dtype.ptx(),),
                        UOp::Not => {
                            format!("{indent}not.{}   r{z}, r{x};\n", dtype.ptx())
                        }
                    };
                }
                IROp::Binary { z, x, y, bop } => {
                    let dtype = kernel.registers[z as usize];
                    //println!("Adding binary {bop:?}");
                    source += &format!(
                        "{indent}{}.{}   r{z}, {}, {};\n",
                        match bop {
                            BOp::Add => "add",
                            BOp::Sub => "sub",
                            BOp::Mul => "mul",
                            BOp::Div => "div",
                            BOp::Mod => "mod",
                            BOp::Pow => todo!(),
                            BOp::Cmplt => "set.lt",
                            BOp::Cmpgt => "set.gt",
                            BOp::NotEq => "set.ne",
                            BOp::Max => todo!(),
                            BOp::Or => todo!(),
                            BOp::And => todo!(),
                            BOp::BitOr => todo!(),
                            BOp::BitAnd => todo!(),
                            BOp::BitXor => todo!(),
                        },
                        dtype.ptx(),
                        x.ptx(),
                        y.ptx()
                    );
                }
                IROp::MAdd { z, a, b, c } => {
                    let dtype = kernel.registers[z as usize];
                    source += &format!(
                        "{indent}mad.lo.{}    r{z}, {}, {}, {};\n",
                        dtype.ptx(),
                        a.ptx(),
                        b.ptx(),
                        c.ptx()
                    );
                }
                IROp::Loop { id, .. } => {
                    source += &format!("LOOP_{id}:\n");
                }
                IROp::EndLoop { id, len } => {
                    // Increment counter
                    source += &format!("{indent}add.u32    r{id}, r{id}, 1;\n");
                    // Set condition
                    source += &format!("{indent}setp.lt.u32    p, r{id}, {len};\n");
                    // Branch
                    source += &format!("@p  bra    LOOP_{id};\n");
                }
                IROp::Barrier { scope } => {
                    source += &format!(
                        "{};\n",
                        match scope {
                            Scope::Global => "__threadfence()",
                            Scope::Local => "__syncthreads()",
                            Scope::Register | Scope::RegTile => unreachable!(),
                        }
                    );
                }
            }
        }
        // End kernel
        source += &format!("{indent}ret;\n}}\0");
        if debug_asm {
            println!("Compiling kernel {name}, PTX source:\n{source}");
        }
        Ok((
            global_work_size,
            local_work_size,
            name,
            source.bytes().collect(),
        ))
    }
}

impl CUDAQueue {
    pub(super) fn launch(
        &mut self,
        program: &mut CUDAProgram,
        buffers: &mut Slab<CUDABuffer>,
        args: &[Id],
    ) -> Result<CUDAEvent, CUDAError> {
        let mut kernel_params: Vec<*mut core::ffi::c_void> = Vec::new();
        for &arg in args {
            let arg = &mut buffers[arg];
            //let ptr = &mut arg.mem;
            let ptr: *mut _ = &mut arg.ptr;
            kernel_params.push(ptr.cast());
        }
        unsafe {
            (self.cuLaunchKernel)(
                program.function,
                u32::try_from(program.global_work_size[0]).unwrap(),
                u32::try_from(program.global_work_size[1]).unwrap(),
                u32::try_from(program.global_work_size[2]).unwrap(),
                u32::try_from(program.local_work_size[0]).unwrap(),
                u32::try_from(program.local_work_size[1]).unwrap(),
                u32::try_from(program.local_work_size[2]).unwrap(),
                0,
                self.stream,
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            )
        }
        .check("Failed to launch kernel.")?;
        Ok(CUDAEvent {})
    }

    pub(super) fn sync(&mut self) -> Result<(), CUDAError> {
        self.load = 0;
        unsafe { (self.cuStreamSynchronize)(self.stream) }
            .check("Failed to synchronize CUDA stream.")
    }

    pub(super) const fn load(&self) -> usize {
        self.load
    }
}

impl CUDAStatus {
    fn check(self, info: &str) -> Result<(), CUDAError> {
        if self == Self::CUDA_SUCCESS {
            Ok(())
        } else {
            Err(CUDAError { info: info.into(), status: self })
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUctx_st {
    _unused: [u8; 0],
}
type CUcontext = *mut CUctx_st;
type CUdevice = c_int;
type CUdeviceptr = u64;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUmod_st {
    _unused: [u8; 0],
}
type CUmodule = *mut CUmod_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUfunc_st {
    _unused: [u8; 0],
}
type CUfunction = *mut CUfunc_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUstream_st {
    _unused: [u8; 0],
}
type CUstream = *mut CUstream_st;
#[allow(unused)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum CUjit_option {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_NUM_OPTIONS = 20,
}
#[allow(unused)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED = 135,
    CU_DEVICE_ATTRIBUTE_MAX,
}

impl DType {
    pub(super) fn ptx(&self) -> &str {
        match self {
            Self::BF16 => panic!("BF16 is not native to OpenCL, workaround is WIP."),
            Self::F8 => "f8",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::U8 => "u8",
            Self::I8 => "s8",
            Self::I16 => "s16",
            Self::I32 => "s32",
            Self::I64 => "s64",
            Self::Bool => "b8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
        }
    }
}

impl Constant {
    fn ptx(&self) -> String {
        match self {
            &Self::F16(x) => format!("{:.12}", half::f16::from_bits(x)),
            &Self::BF16(x) => format!("{:.12}", half::bf16::from_bits(x)),
            Self::F8(x) => {
                /*let bytes = unsafe { t::<_, f32>(*x).to_ne_bytes() };
                let hex = format!("{:02X}{:02X}{:02X}{:02X}", bytes[0], bytes[1], bytes[2], bytes[3]);
                format!("0f{}", hex)*/
                use float8::F8E4M3 as f8;
                format!("{:.12}", f8::from_bits(*x))
            }
            &Self::F32(x) => {
                /*let bytes = unsafe { t::<_, f32>(*x).to_ne_bytes() };
                let hex = format!("{:02X}{:02X}{:02X}{:02X}", bytes[0], bytes[1], bytes[2], bytes[3]);
                format!("0f{}", hex)*/
                format!("{:.12}", f32::from_bits(x))
            }
            &Self::F64(x) => {
                /*let bytes = unsafe { t::<_, f64>(*x).to_ne_bytes() };
                let hex = format!("{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}", bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
                format!("0d{}", hex)*/
                format!("{:.12}", f64::from_bits(x))
            }
            Self::U8(_) => todo!(),
            Self::I8(_) => todo!(),
            Self::I16(_) => todo!(),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}"),
            Self::U64(x) => format!("{x}"),
            Self::I32(x) => format!("{x}"),
            Self::I64(x) => format!("{x}"),
            Self::Bool(_) => todo!(),
        }
    }
}

impl Reg {
    fn ptx(&self) -> String {
        match self {
            Self::Var(id) => format!("r{id}"),
            Self::Const(value) => value.ptx(),
        }
    }
}

impl DType {
    pub(super) fn cu(&self) -> &str {
        match self {
            Self::BF16 => todo!("BF16 is not native to OpenCL, workaround is WIP."),
            Self::F8 => todo!("F8 is not native to OpenCL, workaround is WIP."),
            Self::F16 => "__half",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::U8 => "unsigned char",
            Self::I8 => "char",
            Self::I16 => "short",
            Self::I32 => "int",
            Self::I64 => "long",
            Self::Bool => "bool",
            Self::U16 => "unsigned short",
            Self::U32 => "unsigned int",
            Self::U64 => "unsigned long",
        }
    }
}

impl Reg {
    fn cu(&self) -> String {
        match self {
            Self::Var(id) => format!("r{id}"),
            Self::Const(value) => value.cu(),
        }
    }
}

impl Constant {
    fn cu(&self) -> String {
        match self {
            &Self::BF16(x) => format!("{}f", half::bf16::from_bits(x)),
            &Self::F8(x) => format!("{:.16}f", F8E4M3::from_bits(x)),
            &Self::F16(x) => format!("__float2half({:.6}f)", half::f16::from_bits(x)),
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

#[repr(C)]
#[derive(Debug)]
struct _nvrtcProgram {
    _unused: [u8; 0],
}
type nvrtcProgram = *mut _nvrtcProgram;

#[allow(unused)]
#[derive(Debug, PartialEq, Eq)]
#[repr(C)]
enum nvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
}

impl nvrtcResult {
    fn check(self, info: &str) -> Result<(), CUDAError> {
        if self == Self::NVRTC_SUCCESS {
            Ok(())
        } else {
            Err(CUDAError { info: info.into(), status: CUDAStatus::CUDA_ERROR_INVALID_SOURCE })
        }
    }
}

#[allow(unused)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum CUDAStatus {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_UNKNOWN = 999,
}
