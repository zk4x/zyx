use crate::Scalar;

use super::{Compiler, CompilerError, HWInfo};
use alloc::vec;
use alloc::vec::Vec;
use core::ptr;
use cuda_driver_sys::{
    cuCtxCreate_v2, cuDeviceComputeCapability, cuDeviceGet, cuDeviceGetCount, cuDeviceGetName,
    cuDriverGetVersion, cuInit, cuMemAlloc_v2, cuMemFree_v2, cuMemcpy, CUcontext, CUctx_st,
    CUdevice, CUresult,
};

use super::IRKernel;

use core::ffi::CStr;
use std::println;

fn handle_status(status: CUresult, msg: &str) -> Result<(), CompilerError> {
    // TODO return proper compiler error
    if status != CUresult::CUDA_SUCCESS {
        panic!("CUDA error: {status:?}, {msg}")
    }
    Ok(())
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
                cuMemcpy(
                    buffer.mem,
                    core::mem::transmute(data.as_ptr()),
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
        let mut res = Vec::with_capacity(length);
        handle_status(
            unsafe {
                cuMemcpy(
                    core::mem::transmute(res.as_mut_ptr()),
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
        let _ = kernel;
        todo!()
    }

    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &mut [Self::Buffer],
    ) -> Result<(), CompilerError> {
        let _ = program;
        let _ = args;
        todo!()
    }

    fn release_program(&mut self, program: Self::Program) -> Result<(), CompilerError> {
        let _ = program;
        todo!()
    }
}
