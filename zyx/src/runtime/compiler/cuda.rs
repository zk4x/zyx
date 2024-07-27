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
use core::ffi::{c_char, c_int, c_ulonglong, CStr};
use core::ptr;

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum CUresult {
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
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUctx_st {
    _unused: [u8; 0],
}
type CUcontext = *mut CUctx_st;
type CUdevice = c_int;
type CUdeviceptr = c_ulonglong;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUmod_st {
    _unused: [u8; 0],
}
type CUmodule = *mut CUmod_st;
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
extern "system" {
    fn cuCtxCreate_v2(
        pctx: *mut CUcontext,
        flags: ::std::os::raw::c_uint,
        dev: CUdevice,
    ) -> CUresult;
    fn cuCtxDetach(ctx: CUcontext) -> CUresult;
    fn cuDeviceComputeCapability(
        major: *mut ::std::os::raw::c_int,
        minor: *mut ::std::os::raw::c_int,
        dev: CUdevice,
    ) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: ::std::os::raw::c_int) -> CUresult;
    fn cuDeviceGetCount(count: *mut ::std::os::raw::c_int) -> CUresult;
    fn cuDeviceGetName(
        name: *mut ::std::os::raw::c_char,
        len: ::std::os::raw::c_int,
        dev: CUdevice,
    ) -> CUresult;
    fn cuDriverGetVersion(driverVersion: *mut ::std::os::raw::c_int) -> CUresult;
    fn cuInit(Flags: ::std::os::raw::c_uint) -> CUresult;
    fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: ::std::os::raw::c_uint,
        gridDimY: ::std::os::raw::c_uint,
        gridDimZ: ::std::os::raw::c_uint,
        blockDimX: ::std::os::raw::c_uint,
        blockDimY: ::std::os::raw::c_uint,
        blockDimZ: ::std::os::raw::c_uint,
        sharedMemBytes: ::std::os::raw::c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut ::std::os::raw::c_void,
        extra: *mut *mut ::std::os::raw::c_void,
    ) -> CUresult;
    fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    fn cuMemcpyDtoH_v2(
        dstHost: *mut ::std::os::raw::c_void,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
    ) -> CUresult;
    fn cuMemcpyHtoD_v2(
        dstDevice: CUdeviceptr,
        srcHost: *const ::std::os::raw::c_void,
        ByteCount: usize,
    ) -> CUresult;
    fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const ::std::os::raw::c_char,
    ) -> CUresult;
    fn cuModuleLoadDataEx(
        module: *mut CUmodule,
        image: *const ::std::os::raw::c_void,
        numOptions: ::std::os::raw::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::std::os::raw::c_void,
    ) -> CUresult;
}

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
        #[cfg(feature = "debug1")]
        println!("CUDA error: {status:?}, {msg}");
        return Err(CompilerError::GeneralExecutionError(""));
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
    compute_capability: String,
    context: CUcontext,
}

pub(crate) struct CUDABuffer {
    mem: CUdeviceptr,
}

pub(crate) struct CUDAProgram {
    module: CUmodule,
    name: String,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    args_read_only: Vec<bool>,
}

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

        return Ok(CUDARuntime {
            device,
            context,
            compute_capability: f!("{major}{minor}"),
        });
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
        data: Vec<T>,
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
            &self.compute_capability,
        );
    }

    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &mut [Self::Buffer],
    ) -> Result<(), CompilerError> {
        let mut function = ptr::null_mut();
        handle_status(
            unsafe {
                cuModuleGetFunction(&mut function, program.module, program.name.as_ptr().cast())
            },
            "Failed to load function",
        )?;

        let mut kernel_params: Vec<*mut core::ffi::c_void> = Vec::new();
        for arg in args.iter_mut() {
            //let ptr = &mut arg.mem;
            let ptr: *mut _ = &mut arg.mem;
            kernel_params.push(ptr.cast());
        }

        handle_status(
            unsafe {
                cuLaunchKernel(
                    function,
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
            },
            "Failed to launch kernel.",
        )?;
        Ok(())
    }

    fn release_program(&mut self, program: Self::Program) -> Result<(), CompilerError> {
        let _ = program;
        todo!()
    }
}

impl CUDAProgram {
    fn compile_from_source(
        source: &str,
        //context: *mut c_void,
        global_work_size: [usize; 3],
        local_work_size: [usize; 3],
        args_read_only: Vec<bool>,
        compute_capability: &str,
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
        let source = f!("extern \"C\" __global__ void {name}{source}");
        #[cfg(feature = "debug1")]
        println!("{source}");

        extern "system" {
            fn nvrtcCreateProgram(
                prog: *mut nvrtcProgram,
                src: *const ::std::os::raw::c_char,
                name: *const ::std::os::raw::c_char,
                numHeaders: ::std::os::raw::c_int,
                headers: *const *const ::std::os::raw::c_char,
                includeNames: *const *const ::std::os::raw::c_char,
            ) -> nvrtcResult;
            fn nvrtcCompileProgram(
                prog: nvrtcProgram,
                numOptions: ::std::os::raw::c_int,
                options: *const *const ::std::os::raw::c_char,
            ) -> nvrtcResult;
            fn nvrtcGetPTXSize(prog: nvrtcProgram, ptxSizeRet: *mut usize) -> nvrtcResult;
            fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;
            fn nvrtcGetProgramLogSize(prog: nvrtcProgram, logSizeRet: *mut usize) -> nvrtcResult;
            fn nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult;
            //fn nvrtcGetCUBINSize(prog: nvrtcProgram, cubinSizeRet: *mut usize) -> nvrtcResult;
            //fn nvrtcGetCUBIN(prog: nvrtcProgram, cubin: *mut c_char) -> nvrtcResult;
            fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;
        }

        #[repr(C)]
        #[derive(Debug)]
        struct _nvrtcProgram {
            _unused: [u8; 0],
        }
        type nvrtcProgram = *mut _nvrtcProgram;
        let mut program = ptr::null_mut();
        unsafe {
            nvrtcCreateProgram(
                &mut program as *mut nvrtcProgram,
                (&f!("{source}\0")).as_ptr() as *const c_char,
                (&f!("{name}\0")).as_ptr() as *const c_char,
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        let df = f!("--gpu-architecture=compute_{compute_capability}\0");
        let opts = [df.as_str()];
        unsafe { nvrtcCompileProgram(program, 1, opts.as_ptr().cast()) };
        let mut ptx_size: usize = 0;
        unsafe { nvrtcGetPTXSize(program, &mut ptx_size) };
        let mut ptx_vec: Vec<u8> = Vec::with_capacity(ptx_size);
        unsafe { nvrtcGetPTX(program, ptx_vec.as_mut_ptr() as *mut i8) };
        unsafe { ptx_vec.set_len(ptx_size) };
        //let ptx_c = unsafe { CString::from_vec_unchecked(ptx_vec.clone()) };

        let mut program_log_size: usize = 0;
        unsafe { nvrtcGetProgramLogSize(program, &mut program_log_size) };
        let mut program_log: Vec<u8> = Vec::with_capacity(program_log_size);
        unsafe { nvrtcGetProgramLog(program, program_log.as_mut_ptr() as *mut i8) };
        unsafe { nvrtcDestroyProgram(&mut program) };
        unsafe { program_log.set_len(program_log_size) };
        /*let program_log = unsafe {
            String::from_raw_parts(program_log.as_mut_ptr(), program_log_size, program_log_size)
        };
        println!("NVRTC program log:\n{program_log:?}");*/

        /*let mut cubin_size: usize = 0;
        unsafe { nvrtcGetCUBINSize(program, &mut cubin_size) };
        let mut cubin_vec: Vec<u8> = Vec::with_capacity(cubin_size);
        unsafe { nvrtcGetCUBIN(program, cubin_vec.as_mut_ptr() as *mut i8) };
        unsafe { cubin_vec.set_len(cubin_size) };
        unsafe { nvrtcDestroyProgram(&mut program) };
        let cubin_c = unsafe { CString::from_vec_unchecked(cubin_vec) };
        println!("{cubin_c:?}");*/

        let mut module = ptr::null_mut();
        handle_status(
            unsafe {
                cuModuleLoadDataEx(
                    &mut module,
                    ptx_vec.as_ptr().cast(),
                    0,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            },
            "Module load failed.",
        )?;
        return Ok(CUDAProgram {
            module,
            name,
            global_work_size,
            local_work_size,
            args_read_only,
        });
    }
}
