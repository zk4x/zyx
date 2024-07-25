#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use super::IRKernel;
use super::{Compiler, CompilerError, HWInfo};
use crate::runtime::compiler::{IRArg, IROp, Scope};
use crate::runtime::node::{BOp, UOp};
use crate::{DType, Scalar};
use alloc::format as f;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, CStr};
use core::ptr;
use cuda_driver_sys::{
    cuCtxCreate_v2, cuCtxDetach, cuDeviceComputeCapability, cuDeviceGet, cuDeviceGetCount,
    cuDeviceGetName, cuDriverGetVersion, cuInit, cuMemAlloc_v2, cuMemFree_v2, cuMemcpyDtoH_v2,
    cuMemcpyHtoD_v2, cuModuleLoadDataEx, CUcontext, CUdevice, CUmodule, CUresult,
};
use nvrtc::NvrtcProgram;

type nvrtcProgram = nvrtc::NvrtcProgram;

#[derive(Debug)]
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

use std::println;

fn handle_status(status: CUresult, msg: &str) -> Result<(), CompilerError> {
    // TODO return proper compiler error
    if status != CUresult::CUDA_SUCCESS {
        panic!("CUDA error: {status:?}, {msg}")
    }
    Ok(())
}

impl DType {
    fn cuda(&self) -> &str {
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
        };
    }
}

pub(crate) struct CUDARuntime {
    device: CUdevice,
    context: CUcontext,
}

pub(crate) struct CUDABuffer {
    mem: u64,
}

pub(crate) struct CUDAProgram {}

unsafe impl Send for CUDARuntime {}
unsafe impl Send for CUDABuffer {}
unsafe impl Send for CUDAProgram {}

impl Drop for CUDARuntime {
    fn drop(&mut self) {
        unsafe { cuCtxDetach(self.context) };
    }
}

impl Compiler for CUDARuntime {
    type Buffer = CUDABuffer;
    type Program = CUDAProgram;

    fn initialize() -> Result<Self, CompilerError> {
        handle_status(unsafe { cuInit(0) }, "Failed to init CUDA")?;
        let mut driver_version = 0;
        handle_status(
            unsafe { cuDriverGetVersion(&mut driver_version) },
            "Failed to get CUDA driver version",
        )?;
        println!(
            "CUDA driver version: {}.{}",
            driver_version / 1000,
            (driver_version - (driver_version / 1000 * 1000)) / 10
        );
        let mut num_devices = 0;
        handle_status(
            unsafe { cuDeviceGetCount(&mut num_devices) },
            "Failed to get CUDA device count",
        )?;
        println!("Number of devices: {num_devices}");
        assert!(num_devices > 0, "No available cuda device.");
        let mut device = 0;
        handle_status(
            unsafe { cuDeviceGet(&mut device, 0) },
            "Failed to access CUDA device",
        )?;
        let mut device_name = [0; 100];
        handle_status(
            unsafe { cuDeviceGetName(device_name.as_mut_ptr(), 100, device) },
            "Failed to get CUDA device name",
        )?;
        println!("Using device: {:?}", unsafe {
            CStr::from_ptr(device_name.as_ptr())
        });
        let mut major = 0;
        let mut minor = 0;
        handle_status(
            unsafe { cuDeviceComputeCapability(&mut major, &mut minor, device) },
            "Failed to get CUDA device compute capability.",
        )?;
        println!("Device compute capability: {major}.{minor}");
        let mut context: CUcontext = ptr::null_mut();
        handle_status(
            unsafe { cuCtxCreate_v2(&mut context, 0, device) },
            "Unable to create CUDA context.",
        )?;

        return Ok(CUDARuntime { device, context });
    }

    fn hardware_information(&mut self) -> Result<HWInfo, CompilerError> {
        return Ok(HWInfo {
            max_work_item_sizes: vec![1024, 1024, 1024],
            max_work_group_size: 256,
            preferred_vector_size: 4,
            f16_support: true,
            f64_support: true,
            fmadd: true,
            global_mem_size: 2 * 1024 * 1024 * 1024,
            max_mem_alloc: 512 * 1024 * 1024,
            mem_align: 1024,
            page_size: 1024,
            local_mem_size: 1024 * 1024,
            num_registers: 96,
            native_mm16x16_support: false,
        });
    }

    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, CompilerError> {
        let mut dptr = 0;
        handle_status(
            unsafe { cuMemAlloc_v2(&mut dptr, byte_size) },
            "Failed to allocate memory",
        )?;
        return Ok(CUDABuffer { mem: dptr });
    }

    fn store_memory<T: Scalar>(
        &mut self,
        buffer: &mut Self::Buffer,
        data: &[T],
    ) -> Result<(), CompilerError> {
        handle_status(
            unsafe {
                cuMemcpyHtoD_v2(
                    buffer.mem,
                    data.as_ptr().cast(),
                    data.len() * T::dtype().byte_size(),
                )
            },
            "Failed to store memory",
        )?;
        return Ok(());
    }

    fn load_memory<T: Scalar>(
        &mut self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, CompilerError> {
        let mut res: Vec<T> = Vec::with_capacity(length);
        handle_status(
            unsafe {
                cuMemcpyDtoH_v2(
                    res.as_mut_ptr().cast(),
                    buffer.mem,
                    length * T::dtype().byte_size(),
                )
            },
            "Failed to load memory",
        )?;
        unsafe { res.set_len(length) };
        return Ok(res);
    }

    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), CompilerError> {
        handle_status(
            unsafe { cuMemFree_v2(buffer.mem) },
            "Failed to deallocate memory",
        )?;
        return Ok(());
    }

    fn compile_program(&mut self, kernel: &IRKernel) -> Result<Self::Program, CompilerError> {
        let mut source = String::from("(\n");
        let mut indent = String::from("  ");

        // Transpile kernel args
        let mut args_read_only = Vec::new();
        for (id, IRArg { dtype, read_only }) in kernel.args.iter() {
            source += &f!(
                "{indent}{}{}* g{id},\n",
                if *read_only { "const " } else { "" },
                dtype.cuda()
            );
            if *read_only {
                args_read_only.push(true);
            } else {
                args_read_only.push(false);
            }
        }

        source.pop();
        source.pop();
        source += "\n) {\n";

        // Add indices for global and local loops
        source += &f!(
            "  unsigned int i0 = blockIdx.x;   /* 0..{} */\n",
            kernel.global_work_size[0]
        );
        source += &f!(
            "  unsigned int i1 = threadIdx.x;   /* 0..{} */\n",
            kernel.local_work_size[0]
        );
        source += &f!(
            "  unsigned int i2 = blockIdx.y;   /* 0..{} */\n",
            kernel.global_work_size[1]
        );
        source += &f!(
            "  unsigned int i3 = threadIdx.y;   /* 0..{} */\n",
            kernel.local_work_size[1]
        );
        source += &f!(
            "  unsigned int i4 = blockIdx.z;   /* 0..{} */\n",
            kernel.global_work_size[2]
        );
        source += &f!(
            "  unsigned int i5 = threadIdx.z;   /* 0..{} */\n",
            kernel.local_work_size[2]
        );

        // Transpile kernel ops, skip ends of global and local loops
        for op in &kernel.ops {
            match op {
                IROp::DeclareMem {
                    id,
                    scope,
                    read_only,
                    len,
                    dtype,
                    init,
                } => match scope {
                    Scope::Global => {}
                    Scope::Local => {
                        let read_only = if *read_only { "const " } else { "" };
                        let size = if *len > 0 {
                            f!("[{len}]")
                        } else {
                            String::new()
                        };
                        source += &f!(
                            "{indent}__local {read_only}{} l{id}{};\n",
                            dtype.cuda(),
                            size,
                        );
                    }
                    Scope::Register => {
                        let read_only = if *read_only { "const " } else { "" };
                        let size = if *len > 0 {
                            f!("[{len}]")
                        } else {
                            String::new()
                        };
                        source += &f!("{indent}{read_only}{} r{id}{}", dtype.cuda(), size);
                        if let Some(init) = init {
                            source += &f!(" = {{{}}};\n", init);
                        } else {
                            source += ";\n";
                        }
                    }
                },
                IROp::AssignMem { z, x } => {
                    let (zt, z) = z.to_str(0);
                    if !zt.is_empty() {
                        for idx in zt.into_iter() {
                            source += &f!("{indent}t0 = {idx};\n");
                        }
                    }
                    let (xt, x) = x.to_str(1);
                    if !xt.is_empty() {
                        for idx in xt.into_iter() {
                            source += &f!("{indent}t1 = {idx};\n");
                        }
                    }
                    source += &f!("{indent}{z} = {x};\n");
                }
                IROp::Unary { z, x, ops } => {
                    let (zt, z) = z.to_str(0);
                    if !zt.is_empty() {
                        for idx in zt.into_iter() {
                            source += &f!("{indent}t0 = {idx};\n");
                        }
                    }
                    let (xt, x) = x.to_str(1);
                    if !xt.is_empty() {
                        for idx in xt.into_iter() {
                            source += &f!("{indent}t1 = {idx};\n");
                        }
                    }
                    let mut inner_op = f!("{x}");
                    for uop in ops {
                        inner_op = match uop {
                            UOp::Noop => inner_op,
                            UOp::Cast(dtype) => f!("({}){inner_op}", dtype.cuda()),
                            UOp::Neg => f!("-({inner_op})"),
                            UOp::Inv => f!("1/{inner_op}"),
                            UOp::Sin => f!("sin({inner_op})"),
                            UOp::Cos => f!("cos({inner_op})"),
                            UOp::Exp => f!("exp({inner_op})"),
                            UOp::Ln => f!("log({inner_op})"),
                            UOp::Sqrt => f!("sqrt({inner_op})"),
                            UOp::ReLU => f!("max({inner_op}, 0)"),
                            UOp::Tanh => f!("tanh({inner_op})"),
                        };
                    }
                    source += &f!("{indent}{z} = {inner_op};\n");
                }
                IROp::Binary { z, x, y, op } => {
                    let (zt, z) = z.to_str(0);
                    if !zt.is_empty() {
                        for idx in zt.into_iter() {
                            source += &f!("{indent}t0 = {idx};\n");
                        }
                    }
                    let (xt, x) = x.to_str(1);
                    if !xt.is_empty() {
                        for idx in xt.into_iter() {
                            source += &f!("{indent}t1 = {idx};\n");
                        }
                    }
                    let (yt, y) = y.to_str(2);
                    if !yt.is_empty() {
                        for idx in yt.into_iter() {
                            source += &f!("{indent}t2 = {idx};\n");
                        }
                    }
                    source += &f!(
                        "{indent}{z} = {};\n",
                        match op {
                            BOp::Add => f!("{x}+{y}"),
                            BOp::Sub => f!("{x}-{y}"),
                            BOp::Mul => f!("{x}*{y}"),
                            BOp::Div => f!("{x}/{y}"),
                            BOp::Pow => f!("powf({x}, {y})"),
                            BOp::Max => f!("max({x}, {y})"),
                            BOp::Cmplt => f!("{x}<{y}"),
                        }
                    );
                }
                IROp::Loop { id, len } => {
                    source +=
                        &f!("{indent}for (unsigned int i{id} = 0; i{id} < {len}; i{id}++) {{   /* 0..{len} */\n");
                    indent += "  ";
                }
                IROp::EndLoop => {
                    indent.pop();
                    indent.pop();
                    source += &f!("{indent}}}\n");
                }
                IROp::Barrier { scope } => {
                    let scope = match scope {
                        Scope::Register => panic!(),
                        Scope::Local => "LOCAL",
                        Scope::Global => "GLOBAL",
                    };
                    source += &f!("{indent}barrier(CLK_{scope}_MEM_FENCE);\n");
                }
            }
        }

        source += "}";

        return CUDAProgram::compile_from_source(
            &source,
            //self.context,
            kernel.global_work_size,
            kernel.local_work_size,
            args_read_only,
        );
    }

    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &mut [Self::Buffer],
    ) -> Result<(), CompilerError> {
        let _ = program;
        let _ = args;

        //cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra)
        todo!()
    }

    fn release_program(&mut self, program: Self::Program) -> Result<(), CompilerError> {
        let _ = program;
        todo!()
    }
}

/*#[repr(C)]
#[derive(Debug)]
pub struct _nvrtcProgram {
    _unused: [u8; 0],
}
type nvrtcProgram = *mut _nvrtcProgram;*/

impl CUDAProgram {
    fn compile_from_source(
        source: &str,
        //context: *mut c_void,
        global_work_size: [usize; 3],
        local_work_size: [usize; 3],
        args_read_only: Vec<bool>,
    ) -> Result<Self, CompilerError> {
        let name = f!(
            "k__{}_{}__{}_{}__{}_{}",
            global_work_size[0],
            local_work_size[0],
            global_work_size[1],
            local_work_size[1],
            global_work_size[2],
            local_work_size[2],
        );
        let source = f!("__global__ void {name}{source}");
        #[cfg(feature = "debug1")]
        println!("{source}");

        let program = NvrtcProgram::new(&source, Some(&name), &[], &[]).unwrap();
        program.compile(&[]).unwrap();
        let ptx = program.get_ptx().unwrap();
        #[cfg(feature = "debug1")]
        println!("{ptx}");

        let module = ptr::null_mut();
        handle_status(
            unsafe { cuModuleLoadDataEx(module, image, numOptions, options, optionValues) },
            "Module load failed.",
        )?;

        todo!()
    }
}
