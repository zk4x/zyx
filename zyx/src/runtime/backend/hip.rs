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

#[derive(Debug, Default, serde::Deserialize)]
pub struct HIPConfig {
    device_ids: Option<Vec<i32>>,
}

#[derive(Debug)]
pub struct HIPError {
    info: String,
    status: HIPStatus,
    hiprtc: hiprtcResult,
}

#[derive(Debug)]
pub(super) struct HIPMemoryPool {
    #[allow(unused)]
    cuda: Rc<Library>,
    context: HIPcontext,
    device: HIPdevice,
    free_bytes: usize,
    hipMemAlloc: unsafe extern "C" fn(*mut HIPdeviceptr, usize) -> HIPStatus,
    hipMemcpyHtoD: unsafe extern "C" fn(HIPdeviceptr, *const c_void, usize) -> HIPStatus,
    hipMemcpyDtoH: unsafe extern "C" fn(*mut c_void, HIPdeviceptr, usize) -> HIPStatus,
    hipMemFree: unsafe extern "C" fn(HIPdeviceptr) -> HIPStatus,
    hipMemcpyPeer: unsafe extern "C" fn(
        HIPdeviceptr,
        HIPcontext,
        HIPdeviceptr,
        HIPcontext,
        usize,
    ) -> HIPStatus,
    hipCtxDestroy: unsafe extern "C" fn(HIPcontext) -> HIPStatus,
}

#[derive(Debug)]
pub(super) struct HIPBuffer {
    ptr: u64,
    context: HIPcontext,
    bytes: usize,
}

#[derive(Debug)]
pub(super) struct HIPDevice {
    device: HIPdevice,
    memory_pool_id: usize,
    dev_info: DeviceInfo,
    compute_capability: [c_int; 2],
}

#[derive(Debug)]
pub(super) struct HIPProgram {
    name: String,
    module: HIPmodule,
    function: HIPfunction,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
}

#[derive(Debug)]
pub(super) struct HIPQueue {
    load: usize,
    hipLaunchKernel: unsafe extern "C" fn(
        HIPfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        HIPstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> HIPStatus,
}

unsafe impl Send for HIPMemoryPool {}
unsafe impl Send for HIPBuffer {}
unsafe impl Send for HIPProgram {}

pub(super) fn initialize_device(
    config: &HIPConfig,
    debug_dev: bool,
) -> Result<(Vec<HIPMemoryPool>, Vec<(HIPDevice, Vec<HIPQueue>)>), HIPError> {
    let _ = config;

    let hip_paths = [
        "/lib64/libamdhip64.so",
        "/lib/x86_64-linux-gnu/libamdhip64.so",
    ];
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
            status: HIPStatus::hipErrorTbd,
            hiprtc: hiprtcResult::HIPRTC_SUCCESS,
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
    let mut num_devices = 0;
    unsafe { hipDeviceGetCount(&mut num_devices) }.check("Failed to get HIP device count")?;
    if num_devices == 0 {
        return Err(HIPError {
            info: "No available hip device.".into(),
            status: HIPStatus::hipErrorTbd,
            hiprtc: hiprtcResult::HIPRTC_SUCCESS,
        });
    }
    let device_ids: Vec<_> = (0..num_devices)
        .filter(|id| {
            if let Some(ids) = config.device_ids.as_ref() {
                ids.contains(id)
            } else {
                true
            }
        })
        .collect();
    if device_ids.is_empty() {
        return Err(HIPError {
            info: format!("No devices available or selected."),
            status: HIPStatus::hipSuccess,
            hiprtc: hiprtcResult::HIPRTC_SUCCESS,
        });
    }
    if debug_dev {
        println!(
            "Using HIP runtime, driver version: {}.{} on devices:",
            driver_version / 1000,
            (driver_version - (driver_version / 1000 * 1000)) / 10
        );
    }

    let hip = Rc::new(hip);
    let mut memory_pools = Vec::new();
    let mut devices = Vec::new();
    for dev_id in device_ids {
        let mut device = 0;
        unsafe { hipDeviceGet(&mut device, dev_id) }.check("Failed to access HIP device")?;
        let mut device_name = [0; 100];
        let Ok(_) = unsafe { hipDeviceGetName(device_name.as_mut_ptr(), 100, device) }
            .check("Failed to get HIP device name")
        else {
            continue;
        };
        let mut major = 0;
        let mut minor = 0;
        let Ok(_) = unsafe { hipDeviceComputeCapability(&mut major, &mut minor, device) }
            .check("Failed to get HIP device compute capability.")
        else {
            continue;
        };
        if debug_dev {
            println!("{:?}, compute capability: {major}.{minor}", unsafe {
                std::ffi::CStr::from_ptr(device_name.as_ptr())
            });
        }
        let mut free_bytes = 0;
        let Ok(_) =
            unsafe { hipDeviceTotalMem(&mut free_bytes, device) }.check("Failed to get dev mem.")
        else {
            continue;
        };
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
        let mut queues = Vec::new();
        devices.push((
            HIPDevice {
                device,
                dev_info: DeviceInfo::default(),
                memory_pool_id: 0,
                compute_capability: [major, minor],
            },
            queues,
        ))
    }

    Ok((memory_pools, devices))
}

impl HIPMemoryPool {
    pub(super) fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    pub(super) fn allocate(&mut self, bytes: usize) -> Result<HIPBuffer, HIPError> {
        if bytes > self.free_bytes {
            return Err(HIPError {
                info: "Insufficient free memory.".into(),
                status: HIPStatus::hipErrorOutOfMemory,
                hiprtc: hiprtcResult::HIPRTC_SUCCESS,
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

    pub(super) fn deallocate(&mut self, buffer: HIPBuffer) -> Result<(), HIPError> {
        unsafe { (self.hipMemFree)(buffer.ptr) }.check("Failed to free memory.")?;
        self.free_bytes += buffer.bytes;
        Ok(())
    }

    pub(super) fn host_to_pool(&mut self, src: &[u8], dst: &HIPBuffer) -> Result<(), HIPError> {
        unsafe { (self.hipMemcpyHtoD)(dst.ptr, src.as_ptr().cast(), src.len()) }
            .check("Failed to copy memory from host to pool.")
    }

    pub(super) fn pool_to_host(&mut self, src: &HIPBuffer, dst: &mut [u8]) -> Result<(), HIPError> {
        unsafe { (self.hipMemcpyDtoH)(dst.as_mut_ptr().cast(), src.ptr, dst.len()) }
            .check("Failed to copy memory from pool to host.")
    }

    pub(super) fn pool_to_pool(
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
    pub(super) fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    // Memory pool id out of OpenCLMemoryPools
    pub(super) fn memory_pool_id(&self) -> usize {
        self.memory_pool_id
    }

    pub(super) fn release_program(&self, program: HIPProgram) -> Result<(), HIPError> {
        todo!()
    }

    pub(super) fn compile(
        &mut self,
        kernel: &IRKernel,
        debug_asm: bool,
    ) -> Result<HIPProgram, HIPError> {
        let mut source = String::from("(\n");
        let mut indent = String::from("  ");
        let mut global_work_size = [0; 3];
        let mut local_work_size = [0; 3];
        for op in &kernel.ops[..6] {
            if let IROp::While { id, len } = op {
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
        for (id, (len, dtype, read_only)) in kernel.registers.iter().enumerate() {
            source += &format!(
                "{indent}{}{} r{id}[{len}];\n",
                if *read_only { "const " } else { "" },
                dtype.hip()
            );
        }
        // Add indices for global and local loops
        source += &format!("  r0 = blockIdx.x;   /* 0..{} */\n", global_work_size[0]);
        source += &format!("  r1 = threadIdx.x;   /* 0..{} */\n", local_work_size[0]);
        source += &format!("  r2 = blockIdx.y;   /* 0..{} */\n", global_work_size[1]);
        source += &format!("  r3 = threadIdx.y;   /* 0..{} */\n", local_work_size[1]);
        source += &format!("  r4 = blockIdx.z;   /* 0..{} */\n", global_work_size[2]);
        source += &format!("  r5 = threadIdx.z;   /* 0..{} */\n", local_work_size[2]);
        for op in kernel.ops[6..kernel.ops.len() - 6].iter().copied() {
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
                        UOp::Cast(_) => {
                            format!("{indent}{} = ({}){};\n", z.hip(), dtype.hip(), x.hip())
                        }
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
                    source += &format!(
                        "{indent}{} = {} * {} + {};\n",
                        z.hip(),
                        a.hip(),
                        b.hip(),
                        c.hip()
                    );
                }
                IROp::While { id, len } => {
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
        let mut name = format!(
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
        // INFO: MUST BE NULL TERMINATED!
        let source = format!("{pragma}extern \"C\" __global__ void {name}{source}\0");
        name += "\0";
        if debug_asm {
            println!("{source}");
        }
        let hiprtc_paths = ["/lib64/libhiprtc.so"];
        let hiprtc = hiprtc_paths.iter().find_map(|path| {
            if let Ok(lib) = unsafe { Library::new(path) } {
                Some(lib)
            } else {
                None
            }
        });
        let Some(hiprtc) = hiprtc else {
            return Err(HIPError {
                info: "HIP runtime compiler (HIPRTC) not found.".into(),
                status: HIPStatus::hipErrorTbd,
                hiprtc: hiprtcResult::HIPRTC_SUCCESS,
            });
        };
        let hiprtcCreateProgram: unsafe extern "C" fn(
            *mut hiprtcProgram,
            *const c_char,
            *const c_char,
            c_int,
            *const *const c_char,
            *const *const c_char,
        ) -> hiprtcResult = *unsafe { hiprtc.get(b"hiprtcCreateProgram\0") }.unwrap();
        let hiprtcCompileProgram: unsafe extern "C" fn(
            hiprtcProgram,
            c_int,
            *const *const c_char,
        ) -> hiprtcResult = *unsafe { hiprtc.get(b"hiprtcCompileProgram\0") }.unwrap();
        let hiprtcGetCodeSize: unsafe extern "C" fn(hiprtcProgram, *mut usize) -> hiprtcResult =
            *unsafe { hiprtc.get(b"hiprtcGetCodeSize\0") }.unwrap();
        let hiprtcGetCode: unsafe extern "C" fn(hiprtcProgram, *mut c_char) -> hiprtcResult =
            *unsafe { hiprtc.get(b"hiprtcGetCode\0") }.unwrap();
        let hiprtcGetProgramLogSize: unsafe extern "C" fn(
            hiprtcProgram,
            *mut usize,
        ) -> hiprtcResult = *unsafe { hiprtc.get(b"hiprtcGetProgramLogSize\0") }.unwrap();
        let hiprtcGetProgramLog: unsafe extern "C" fn(hiprtcProgram, *mut c_char) -> hiprtcResult =
            *unsafe { hiprtc.get(b"hiprtcGetProgramLog\0") }.unwrap();
        let hiprtcDestroyProgram: unsafe extern "C" fn(*mut hiprtcProgram) -> hiprtcResult =
            *unsafe { hiprtc.get(b"hiprtcDestroyProgram\0") }.unwrap();

        #[repr(C)]
        #[derive(Debug)]
        struct _hiprtcProgram {
            _unused: [u8; 0],
        }
        type hiprtcProgram = *mut _hiprtcProgram;
        let mut program = ptr::null_mut();
        unsafe {
            hiprtcCreateProgram(
                &mut program as *mut hiprtcProgram,
                source.as_ptr().cast(),
                name.as_ptr().cast(),
                0,
                ptr::null(),
                ptr::null(),
            )
        }
        .check("hiprtcCreateProgram")?;

        let df = format!(
            "--gpu-architecture=compute_{}{}\0",
            self.compute_capability[0], self.compute_capability[1]
        );
        //let df = format!("");
        let opts = [df.as_str()];
        if let Err(e) = unsafe { hiprtcCompileProgram(program, 0, opts.as_ptr().cast()) }
            .check("hiprtcCompileProgram")
        {
            //println!("Error during compilation {e:?}");
            let mut program_log_size: usize = 0;
            unsafe { hiprtcGetProgramLogSize(program, &mut program_log_size) }
                .check("hiprtcGetProgramLogSize")?;
            //program_log_size = 1000;
            println!("Program log size: {program_log_size}");
            let mut program_log: Vec<u8> = vec![0; program_log_size];
            unsafe { hiprtcGetProgramLog(program, program_log.as_mut_ptr() as *mut i8) }
                .check("hiprtcGetProgramLog")?;
            if let Ok(log) = String::from_utf8(program_log) {
                println!("HIPRTC program log:\n{log}",);
            } else {
                println!("HIPRTC program log is not valid utf8");
            }
            return Err(e);
        }

        let mut code_size: usize = 0;
        unsafe { hiprtcGetCodeSize(program, &mut code_size) }.check("hiprtcGetCodeSize")?;

        let mut code_vec: Vec<u8> = vec![0; code_size];
        unsafe { hiprtcGetCode(program, code_vec.as_mut_ptr() as *mut i8) }
            .check("hiprtcGetCode")?;
        unsafe { hiprtcDestroyProgram(&mut program) }.check("hiprtcDestroyProgram")?;

        /*let mut module = ptr::null_mut();
        unsafe {
            (self.hipModuleLoadDataEx)(
                &mut module,
                code_vec.as_ptr().cast(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        }
        .check("Module load failed.")?;
        let mut function: HIPfunction = ptr::null_mut();
        // Don't forget that the name is null terminated string
        // Name may be mangled, IDK cause hiprtc just does not work
        unsafe { (self.hipModuleGetFunction)(&mut function, module, name.as_ptr().cast()) }
            .check("Failed to load function.")?;*/

        Ok(HIPProgram {
            name,
            module: todo!(),
            function: todo!(),
            global_work_size,
            local_work_size,
        })
    }
}

impl HIPQueue {
    pub(super) fn launch(
        &mut self,
        program: &mut HIPProgram,
        buffers: &mut IndexMap<HIPBuffer>,
        args: &[usize],
    ) -> Result<(), HIPError> {
        let mut kernel_params: Vec<*mut core::ffi::c_void> = Vec::new();
        for arg in args {
            let arg = &mut buffers[*arg];
            //let ptr = &mut arg.mem;
            let ptr: *mut _ = &mut arg.ptr;
            kernel_params.push(ptr.cast());
        }
        unsafe {
            (self.hipLaunchKernel)(
                program.function,
                program.global_work_size[0] as u32,
                program.global_work_size[1] as u32,
                program.global_work_size[2] as u32,
                program.local_work_size[0] as u32,
                program.local_work_size[1] as u32,
                program.local_work_size[2] as u32,
                0,
                ptr::null_mut(),
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            )
        }
        .check("Failed to launch kernel.")
    }

    pub(super) fn sync(&mut self) -> Result<(), HIPError> {
        self.load = 0;
        todo!()
    }

    pub(super) fn load(&self) -> usize {
        self.load
    }
}

impl HIPStatus {
    fn check(self, info: &str) -> Result<(), HIPError> {
        if self != HIPStatus::hipSuccess {
            return Err(HIPError {
                info: format!("Try rerunning with env var AMD_LOG_LEVEL=2 {info}"),
                status: self,
                hiprtc: hiprtcResult::HIPRTC_SUCCESS,
            });
        } else {
            return Ok(());
        }
    }
}

impl IRDType {
    pub(super) fn hip(&self) -> &str {
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
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct HIPstream_st {
    _unused: [u8; 0],
}
type HIPstream = *mut HIPstream_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum hiprtcResult {
    HIPRTC_SUCCESS = 0,                                     // Success
    HIPRTC_ERROR_OUT_OF_MEMORY = 1,                         // Out of memory
    HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,              // Failed to create program
    HIPRTC_ERROR_INVALID_INPUT = 3,                         // Invalid input
    HIPRTC_ERROR_INVALID_PROGRAM = 4,                       // Invalid program
    HIPRTC_ERROR_INVALID_OPTION = 5,                        // Invalid option
    HIPRTC_ERROR_COMPILATION = 6,                           // Compilation error
    HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,             // Failed in builtin operation
    HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8, // No name expression after compilation
    HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,   // No lowered names before compilation
    HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,            // Invalid name expression
    HIPRTC_ERROR_INTERNAL_ERROR = 11,                       // Internal error
    HIPRTC_ERROR_LINKING = 100,                             // Error in linking
}

impl hiprtcResult {
    fn check(self, info: &str) -> Result<(), HIPError> {
        if self != Self::HIPRTC_SUCCESS {
            Err(HIPError {
                info: format!("Try rerunning with env var AMD_LOG_LEVEL=2 {info}"),
                status: HIPStatus::hipSuccess,
                hiprtc: self,
            })
        } else {
            Ok(())
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum HIPStatus {
    hipSuccess = 0,
    ///< Successful completion.
    hipErrorInvalidValue = 1,
    ///< One or more of the parameters passed to the API call is NULL
    ///< or not in an acceptable range.
    hipErrorOutOfMemory = 2,
    ///< out of memory range.
    hipErrorNotInitialized = 3,
    ///< Invalid not initialized
    hipErrorDeinitialized = 4,
    ///< Deinitialized
    hipErrorProfilerDisabled = 5,
    hipErrorProfilerNotInitialized = 6,
    hipErrorProfilerAlreadyStarted = 7,
    hipErrorProfilerAlreadyStopped = 8,
    hipErrorInvalidConfiguration = 9,
    ///< Invalide configuration
    hipErrorInvalidPitchValue = 12,
    ///< Invalid pitch value
    hipErrorInvalidSymbol = 13,
    ///< Invalid symbol
    hipErrorInvalidDevicePointer = 17,
    ///< Invalid Device Pointer
    hipErrorInvalidMemcpyDirection = 21,
    ///< Invalid memory copy direction
    hipErrorInsufficientDriver = 35,
    hipErrorMissingConfiguration = 52,
    hipErrorPriorLaunchFailure = 53,
    hipErrorInvalidDeviceFunction = 98,
    ///< Invalid device function
    hipErrorNoDevice = 100,
    ///< Call to hipGetDeviceCount returned 0 devices
    hipErrorInvalidDevice = 101,
    ///< DeviceID must be in range from 0 to compute-devices.
    hipErrorInvalidImage = 200,
    ///< Invalid image
    hipErrorInvalidContext = 201,
    ///< Produced when input context is invalid.
    hipErrorContextAlreadyCurrent = 202,
    hipErrorMapFailed = 205,
    hipErrorUnmapFailed = 206,
    hipErrorArrayIsMapped = 207,
    hipErrorAlreadyMapped = 208,
    hipErrorNoBinaryForGpu = 209,
    hipErrorAlreadyAcquired = 210,
    hipErrorNotMapped = 211,
    hipErrorNotMappedAsArray = 212,
    hipErrorNotMappedAsPointer = 213,
    hipErrorECCNotCorrectable = 214,
    hipErrorUnsupportedLimit = 215,
    ///< Unsupported limit
    hipErrorContextAlreadyInUse = 216,
    ///< The context is already in use
    hipErrorPeerAccessUnsupported = 217,
    hipErrorInvalidKernelFile = 218,
    ///< In CUDA DRV, it is CUDA_ERROR_INVALID_PTX
    hipErrorInvalidGraphicsContext = 219,
    hipErrorInvalidSource = 300,
    ///< Invalid source.
    hipErrorFileNotFound = 301,
    ///< the file is not found.
    hipErrorSharedObjectSymbolNotFound = 302,
    hipErrorSharedObjectInitFailed = 303,
    ///< Failed to initialize shared object.
    hipErrorOperatingSystem = 304,
    ///< Not the correct operating system
    hipErrorInvalidHandle = 400,
    ///< Invalide handle
    hipErrorIllegalState = 401,
    ///< Resource required is not in a valid state to perform operation.
    hipErrorNotFound = 500,
    ///< Not found
    hipErrorNotReady = 600,
    ///< Indicates that asynchronous operations enqueued earlier are not
    ///< ready.  This is not actually an error, but is used to distinguish
    ///< from hipSuccess (which indicates completion).  APIs that return
    ///< this error include hipEventQuery and hipStreamQuery.
    hipErrorIllegalAddress = 700,
    hipErrorLaunchOutOfResources = 701,
    ///< Out of resources error.
    hipErrorLaunchTimeOut = 702,
    ///< Timeout for the launch.
    hipErrorPeerAccessAlreadyEnabled = 704,
    ///< Peer access was already enabled from the current
    ///< device.
    hipErrorPeerAccessNotEnabled = 705,
    ///< Peer access was never enabled from the current device.
    hipErrorSetOnActiveProcess = 708,
    ///< The process is active.
    hipErrorContextIsDestroyed = 709,
    ///< The context is already destroyed
    hipErrorAssert = 710,
    ///< Produced when the kernel calls assert.
    hipErrorHostMemoryAlreadyRegistered = 712,
    ///< Produced when trying to lock a page-locked
    ///< memory.
    hipErrorHostMemoryNotRegistered = 713,
    ///< Produced when trying to unlock a non-page-locked
    ///< memory.
    hipErrorLaunchFailure = 719,
    ///< An exception occurred on the device while executing a kernel.
    hipErrorCooperativeLaunchTooLarge = 720,
    ///< This error indicates that the number of blocks
    ///< launched per grid for a kernel that was launched
    ///< via cooperative launch APIs exceeds the maximum
    ///< number of allowed blocks for the current device.
    hipErrorNotSupported = 801,
    ///< Produced when the hip API is not supported/implemented
    hipErrorStreamCaptureUnsupported = 900,
    ///< The operation is not permitted when the stream
    ///< is capturing.
    hipErrorStreamCaptureInvalidated = 901,
    ///< The current capture sequence on the stream
    ///< has been invalidated due to a previous error.
    hipErrorStreamCaptureMerge = 902,
    ///< The operation would have resulted in a merge of
    ///< two independent capture sequences.
    hipErrorStreamCaptureUnmatched = 903,
    ///< The capture was not initiated in this stream.
    hipErrorStreamCaptureUnjoined = 904,
    ///< The capture sequence contains a fork that was not
    ///< joined to the primary stream.
    hipErrorStreamCaptureIsolation = 905,
    ///< A dependency would have been created which crosses
    ///< the capture sequence boundary. Only implicit
    ///< in-stream ordering dependencies  are allowed
    ///< to cross the boundary
    hipErrorStreamCaptureImplicit = 906,
    ///< The operation would have resulted in a disallowed
    ///< implicit dependency on a current capture sequence
    ///< from hipStreamLegacy.
    hipErrorCapturedEvent = 907,
    ///< The operation is not permitted on an event which was last
    ///< recorded in a capturing stream.
    hipErrorStreamCaptureWrongThread = 908,
    ///< A stream capture sequence not initiated with
    ///< the hipStreamCaptureModeRelaxed argument to
    ///< hipStreamBeginCapture was passed to
    ///< hipStreamEndCapture in a different thread.
    hipErrorGraphExecUpdateFailure = 910,
    ///< This error indicates that the graph update
    ///< not performed because it included changes which
    ///< violated constraintsspecific to instantiated graph
    ///< update.
    hipErrorInvalidChannelDescriptor = 911,
    ///< Invalid channel descriptor.
    hipErrorInvalidTexture = 912,
    ///< Invalid texture.
    hipErrorUnknown = 999,
    ///< Unknown error.
    // HSA Runtime Error Codes start here.
    hipErrorRuntimeMemory = 1052,
    ///< HSA runtime memory call returned error.  Typically not seen
    ///< in production systems.
    hipErrorRuntimeOther = 1053,
    ///< HSA runtime call other than memory returned error.  Typically
    ///< not seen in production systems.
    hipErrorTbd, // Marker that more error codes are needed.
}
