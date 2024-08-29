#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use super::DeviceInfo;
use crate::dtype::Constant;
use crate::runtime::ir::{IRDType, IROp, Scope, Var};
use crate::runtime::node::{BOp, UOp};
use crate::{index_map::IndexMap, runtime::ir::IRKernel};
use libloading::Library;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::rc::Rc;

#[derive(Debug, serde::Deserialize)]
pub struct HIPConfig {}

#[derive(Debug)]
pub struct HIPError {
    info: String,
    status: HIPStatus,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum HIPStatus {
    HIP_SUCCESS,
    HIP_ERROR_UNKNOWN,
    HIP_ERROR_OUT_OF_MEMORY,
}

#[derive(Debug)]
pub(crate) struct HIPMemoryPool {
    #[allow(unused)]
    cuda: Rc<Library>,
    context: HIPcontext,
    device: HIPdevice,
    free_bytes: usize,
    hipMemAlloc: unsafe extern "C" fn(*mut HIPdeviceptr, usize) -> HIPStatus,
    hipMemcpyHtoD: unsafe extern "C" fn(HIPdeviceptr, *const c_void, usize) -> HIPStatus,
    hipMemcpyDtoH: unsafe extern "C" fn(*mut c_void, HIPdeviceptr, usize) -> HIPStatus,
    hipMemFree: unsafe extern "C" fn(HIPdeviceptr) -> HIPStatus,
    hipMemcpyPeer:
      unsafe extern "C" fn(HIPdeviceptr, HIPcontext, HIPdeviceptr, HIPcontext, usize) -> HIPStatus,
    hipCtxDestroy: unsafe extern "C" fn(HIPcontext) -> HIPStatus,
}

#[derive(Debug)]
pub(crate) struct HIPBuffer {
    ptr: u64,
    context: HIPcontext,
    bytes: usize,
}

#[derive(Debug)]
pub(crate) struct HIPDevice {
    device: HIPdevice,
    memory_pool_id: usize,
    dev_info: DeviceInfo,
    compute_capability: [c_int; 2],
}

#[derive(Debug)]
pub(crate) struct HIPProgram {}

#[derive(Debug)]
pub(crate) struct HIPEvent {}

unsafe impl Send for HIPMemoryPool {}
unsafe impl Send for HIPBuffer {}
unsafe impl Send for HIPProgram {}

pub(crate) fn initialize_hip_backend(
    config: &HIPConfig,
) -> Result<(Vec<HIPMemoryPool>, Vec<HIPDevice>), HIPError> {
    let _ = config;

    let hip_paths = ["/lib64/libamdhip64.so"];
    let hip = hip_paths.iter().find_map(|path| {
        if let Ok(lib) = unsafe { Library::new(path) } {
            Some(lib)
        } else {
            None
        }
    });
    let Some(hip) = hip else {
        return Err(HIPError {
            info: "HIP runtime not found.".into(),
            status: HIPStatus::HIP_ERROR_UNKNOWN,
        });
    };

    let hipInit: unsafe extern "C" fn(c_uint) -> HIPStatus =
        *unsafe { hip.get(b"hipInit\0") }.unwrap();
    let hipDriverGetVersion: unsafe extern "C" fn(*mut c_int) -> HIPStatus =
        *unsafe { hip.get(b"hipDriverGetVersion\0") }.unwrap();
    let hipDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> HIPStatus =
        *unsafe { hip.get(b"hipGetDeviceCount\0") }.unwrap();
    let hipDeviceGet: unsafe extern "C" fn(*mut HIPdevice, c_int) -> HIPStatus =
        *unsafe { hip.get(b"hipDeviceGet\0") }.unwrap();
    let hipDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, HIPdevice) -> HIPStatus =
        *unsafe { hip.get(b"hipDeviceGetName\0") }.unwrap();
    let hipDeviceComputeCapability: unsafe extern "C" fn(
        *mut c_int,
        *mut c_int,
        HIPdevice,
    ) -> HIPStatus = *unsafe { hip.get(b"hipDeviceComputeCapability\0") }.unwrap();
    let hipDeviceTotalMem: unsafe extern "C" fn(*mut usize, HIPdevice) -> HIPStatus =
        *unsafe { hip.get(b"hipDeviceTotalMem\0") }.unwrap();
    let hipDeviceGetAttribute: unsafe extern "C" fn(
        *mut c_int,
        HIPdevice_attribute,
        HIPdevice,
    ) -> HIPStatus = *unsafe { hip.get(b"hipDeviceGetAttribute\0") }.unwrap();
    let hipCtxCreate: unsafe extern "C" fn(*mut HIPcontext, c_uint, HIPdevice) -> HIPStatus =
        *unsafe { hip.get(b"hipCtxCreate\0") }.unwrap();
    let hipMemAlloc = *unsafe { hip.get(b"hipMalloc\0") }.unwrap();
    let hipMemcpyHtoD = *unsafe { hip.get(b"hipMemcpyHtoD\0") }.unwrap();
    let hipMemFree = *unsafe { hip.get(b"hipFree\0") }.unwrap();
    let hipMemcpyDtoH = *unsafe { hip.get(b"hipMemcpyDtoH\0") }.unwrap();
    let hipMemcpyPeer = *unsafe { hip.get(b"hipMemcpyPeer\0") }.unwrap();
    let hipCtxDestroy = *unsafe { hip.get(b"hipCtxDestroy\0") }.unwrap();
    //let hipModuleLoadDataEx = *unsafe { hip.get(b"hipModuleLoadDataEx\0") }.unwrap();
    //let hipModuleGetFunction = *unsafe { hip.get(b"hipModuleGetFunction\0") }.unwrap();
    //let hipLaunchKernel = *unsafe { hip.get(b"hipLaunchKernel\0") }.unwrap();

    unsafe { hipInit(0) }.check("Failed to init HIP")?;
    let mut driver_version = 0;
    unsafe { hipDriverGetVersion(&mut driver_version) }
        .check("Failed to get HIP driver version")?;
    if let Ok(_) = std::env::var("DEBUG_DEV") {
        println!(
            "Using HIP backend, driver version: {}.{} on devices:",
            driver_version / 1000,
            (driver_version - (driver_version / 1000 * 1000)) / 10
        );
    }
    let mut num_devices = 0;
    unsafe { hipDeviceGetCount(&mut num_devices) }.check("Failed to get HIP device count")?;
    if num_devices == 0 {
        return Err(HIPError {
            info: "No available hip device.".into(),
            status: HIPStatus::HIP_ERROR_UNKNOWN,
        });
    }

    let hip = Rc::new(hip);
    let mut memory_pools = Vec::new();
    let mut devices = Vec::new();
    for dev_id in 0..num_devices {
        let mut device = 0;
        unsafe { hipDeviceGet(&mut device, dev_id) }.check("Failed to access HIP device")?;
        let mut device_name = [0; 100];
        let Ok(_) = unsafe { hipDeviceGetName(device_name.as_mut_ptr(), 100, device) }
            .check("Failed to get HIP device name") else { continue; };
        let mut major = 0;
        let mut minor = 0;
        let Ok(_) = unsafe { hipDeviceComputeCapability(&mut major, &mut minor, device) }
            .check("Failed to get HIP device compute capability.") else { continue; };
        if let Ok(_) = std::env::var("DEBUG_DEV") {
            println!("{:?}, compute capability: {major}.{minor}", unsafe {
                std::ffi::CStr::from_ptr(device_name.as_ptr())
            });
        }
        let mut free_bytes = 0;
        let Ok(_) = unsafe { hipDeviceTotalMem(&mut free_bytes, device) }.check("Failed to get dev mem.") else { continue; };
        let mut context: HIPcontext = ptr::null_mut();
        unsafe { hipCtxCreate(&mut context, 0, device) }.check("Unable to create HIP context.")?;
        memory_pools.push(HIPMemoryPool {
            cuda: hip.clone(),
            context,
            device,
            free_bytes,
            hipMemAlloc,
            hipMemcpyHtoD,
            hipMemFree,
            hipMemcpyDtoH,
            hipMemcpyPeer,
            hipCtxDestroy,
        });
        devices.push(HIPDevice {
            device,
            dev_info: DeviceInfo::default(),
            memory_pool_id: 0,
            //hipModuleLoadDataEx,
            //hipModuleGetFunction,
            //hipModuleEnumerateFunctions,
            //hipLaunchKernel,
            compute_capability: [major, minor],
        })
    }

    Ok((memory_pools, devices))
}

impl HIPMemoryPool {
    pub(crate) fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    pub(crate) fn allocate(&mut self, bytes: usize) -> Result<HIPBuffer, HIPError> {
        if bytes > self.free_bytes {
            return Err(HIPError {
                info: "Insufficient free memory.".into(),
                status: HIPStatus::HIP_ERROR_OUT_OF_MEMORY,
            });
        }
        self.free_bytes -= bytes;
        let mut ptr = self.device as u64;
        unsafe { (self.hipMemAlloc)(&mut ptr, bytes) }.check("Failed to allocate memory.")?;
        return Ok(HIPBuffer {
            ptr,
            bytes,
            context: self.context,
        });
    }

    pub(crate) fn deallocate(&mut self, buffer: HIPBuffer) -> Result<(), HIPError> {
        unsafe { (self.hipMemFree)(buffer.ptr) }.check("Failed to free memory.")?;
        self.free_bytes += buffer.bytes;
        Ok(())
    }

    pub(crate) fn host_to_pool(&mut self, src: &[u8], dst: &HIPBuffer) -> Result<(), HIPError> {
        unsafe { (self.hipMemcpyHtoD)(dst.ptr, src.as_ptr().cast(), src.len()) }
            .check("Failed to copy memory from host to pool.")
    }

    pub(crate) fn pool_to_host(&mut self, src: &HIPBuffer, dst: &mut [u8]) -> Result<(), HIPError> {
        unsafe { (self.hipMemcpyDtoH)(dst.as_mut_ptr().cast(), src.ptr, dst.len()) }
            .check("Failed to copy memory from pool to host.")
    }

    pub(crate) fn pool_to_pool(
        &mut self,
        src: &HIPBuffer,
        dst: &HIPBuffer,
    ) -> Result<(), HIPError> {
        unsafe { (self.hipMemcpyPeer)(dst.ptr, dst.context, src.ptr, src.context, dst.bytes) }
            .check("Failed copy memory from pool to pool.")
    }
}

impl Drop for HIPMemoryPool {
    fn drop(&mut self) {
        unsafe { (self.hipCtxDestroy)(self.context) };
    }
}

impl HIPDevice {
    pub(crate) fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    // Memory pool id out of OpenCLMemoryPools
    pub(crate) fn memory_pool_id(&self) -> usize {
        self.memory_pool_id
    }

    pub(crate) fn compile(&mut self, kernel: &IRKernel) -> Result<HIPProgram, HIPError> {
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
                "{indent}{}{}* g{id},\n",
                if *read_only { "const " } else { "" },
                dtype.hip(),
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
                dtype.hip()
            );
        }

        // Add indices for global and local loops
        source += &format!(
            "  r0 = blockIdx.x;   /* 0..{} */\n",
            global_work_size[0]
        );
        source += &format!(
            "  r1 = threadIdx.x;   /* 0..{} */\n",
            local_work_size[0]
        );
        source += &format!(
            "  r2 = blockIdx.y;   /* 0..{} */\n",
            global_work_size[1]
        );
        source += &format!(
            "  r3 = threadIdx.y;   /* 0..{} */\n",
            local_work_size[1]
        );
        source += &format!(
            "  r4 = blockIdx.z;   /* 0..{} */\n",
            global_work_size[2]
        );
        source += &format!(
            "  r5 = threadIdx.z;   /* 0..{} */\n",
            local_work_size[2]
        );

        for op in kernel.ops[6..kernel.ops.len()-6].iter().copied() {
            match op {
                IROp::Set { z, len: _, value } => {
                    source += &format!("{indent}r{z} = {value};\n");
                }
                IROp::Load { z, x, at, dtype: _ } => {
                    source += &format!("{indent}{} = {}[{}];\n", z.hip(), x.hip(), at.hip());
                }
                IROp::Store { z, x, at, dtype: _ } => {
                    source += &format!("{indent}{}[{}] = {};\n", z.hip(), at.hip(), x.hip());
                }
                IROp::Unary { z, x, uop, dtype } => {
                    source += &match uop {
                        UOp::Cast(_) => format!("{indent}{} = ({}){};\n", z.hip(), dtype.hip(), x.hip()),
                        UOp::ReLU => format!("{indent}{} = max({}, 0);\n", z.hip(), x.hip()),
                        UOp::Neg => format!("{indent}{} = -{};\n", z.hip(), x.hip()),
                        UOp::Exp2 => format!("{indent}{} = exp2({});\n", z.hip(), x.hip()),
                        UOp::Log2 => format!("{indent}{} = log2({});\n", z.hip(), x.hip()),
                        UOp::Inv => format!("{indent}{} = 1/{};\n", z.hip(), x.hip()),
                        UOp::Sqrt => format!("{indent}{} = sqrt({});\n", z.hip(), x.hip()),
                        UOp::Sin => format!("{indent}{} = sin({});\n", z.hip(), x.hip()),
                        UOp::Cos => format!("{indent}{} = cos({});\n", z.hip(), x.hip()),
                        UOp::Not => format!("{indent}{} = !{};\n", z.hip(), x.hip()),
                        UOp::Nonzero => format!("{indent}{} = {} != 0;\n", z.hip(), x.hip()),
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
                        z.hip(),
                        match bop {
                            BOp::Add => format!("{} + {}", x.hip(), y.hip()),
                            BOp::Sub => format!("{} - {}", x.hip(), y.hip()),
                            BOp::Mul => format!("{} * {}", x.hip(), y.hip()),
                            BOp::Div => format!("{} / {}", x.hip(), y.hip()),
                            BOp::Pow => format!("pow({}, {})", x.hip(), y.hip()),
                            BOp::Cmplt => format!("{} < {}", x.hip(), y.hip()),
                            BOp::Cmpgt => format!("{} > {}", x.hip(), y.hip()),
                            BOp::Max => format!("max({}, {})", x.hip(), y.hip()),
                            BOp::Or => format!("{} || {}", x.hip(), y.hip()),
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
                    source += &format!("{indent}{} = {} * {} + {};\n", z.hip(), a.hip(), b.hip(), c.hip());
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
        let source = format!("{pragma}extern \"C\" __global__ void {name}{source}");
        if let Ok(_) = std::env::var("DEBUG_ASM") {
            println!("{source}");
        }
        todo!()
    }
}

impl HIPProgram {
    pub(crate) fn launch(
        &mut self,
        buffers: &mut IndexMap<HIPBuffer>,
        args: &[usize],
    ) -> Result<HIPEvent, HIPError> {
        let mut kernel_params: Vec<*mut core::ffi::c_void> = Vec::new();
        for arg in args {
            let arg = &mut buffers[*arg];
            //let ptr = &mut arg.mem;
            let ptr: *mut _ = &mut arg.ptr;
            kernel_params.push(ptr.cast());
        }
        /*unsafe {
            (self.hipLaunchKernel)(
                self.function,
                self.global_work_size[0] as u32,
                self.global_work_size[1] as u32,
                self.global_work_size[2] as u32,
                self.local_work_size[0] as u32,
                self.local_work_size[1] as u32,
                self.local_work_size[2] as u32,
                0,
                ptr::null_mut(),
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            )
        }
        .check("Failed to launch kernel.")?;*/
        // For now just empty event, later we can deal with streams to make it async
        Ok(HIPEvent {})
    }
}

impl HIPStatus {
    fn check(self, info: &str) -> Result<(), HIPError> {
        if self != HIPStatus::HIP_SUCCESS {
            return Err(HIPError {
                info: info.into(),
                status: self,
            });
        } else {
            return Ok(());
        }
    }
}

impl IRDType {
    pub(crate) fn hip(&self) -> &str {
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

impl Var {
    fn hip(&self) -> String {
        match self {
            Var::Id(id, scope) => format!("{scope}{id}"),
            Var::Const(value) => format!("{}", value.hip()),
        }
    }
}

impl Constant {
    fn hip(&self) -> String {
        use core::mem::transmute as t;
        match self {
            #[cfg(feature = "half")]
            Constant::F16(x) => format!("{}f", unsafe { t::<_, half::f16>(*x) }),
            #[cfg(feature = "half")]
            Constant::BF16(x) => format!("{}f", unsafe { t::<_, half::bf16>(*x) }),
            Constant::F32(x) => format!("{}f", unsafe { t::<_, f32>(*x) }),
            Constant::F64(x) => format!("{}f", unsafe { t::<_, f64>(*x) }),
            #[cfg(feature = "complex")]
            Constant::CF32(..) => todo!("Complex numbers are currently not supported for HIP"),
            #[cfg(feature = "complex")]
            Constant::CF64(..) => todo!("Complex numbers are currently not supported for HIP"),
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

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct HIPctx_st {
    _unused: [u8; 0],
}
type HIPcontext = *mut HIPctx_st;
type HIPdevice = c_int;
type HIPdeviceptr = u64;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct HIPmod_st {
    _unused: [u8; 0],
}
type HIPmodule = *mut HIPmod_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct HIPfunc_st {
    _unused: [u8; 0],
}
type HIPfunction = *mut HIPfunc_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum HIPdevice_attribute {
    HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
}
