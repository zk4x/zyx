#![allow(non_camel_case_types)]

use super::{Compiler, CompilerError, HWInfo};
use alloc::vec;
use alloc::{boxed::Box, collections::BTreeSet, string::String, vec::Vec};
use core::ffi::c_void;
use libloading::{Library, Symbol};

#[cfg(feature = "debug1")]
use std::println;

pub(crate) struct HSABuffer {
    memory: *mut c_void,
    event: *mut c_void,
}

pub(crate) struct HSAProgram {
    name: String,
    program: *mut c_void,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    args_read_only: Vec<bool>,
}

pub(crate) struct HSARuntime {
    hsa_runtime_so: Library,
    context: *mut c_void,
    devices: BTreeSet<*mut c_void>,
    queues: Box<[*mut c_void]>,
    queue_size: Box<[u8]>,
    queue_id: usize,
}

// These pointers are on device and do not get invalidated when accessing
// the device from different thread
unsafe impl Send for HSABuffer {}
unsafe impl Send for HSAProgram {}
unsafe impl Send for HSARuntime {}

type size_t = usize;
type hsa_status_t = u32;
type hsa_region_t = *mut c_void;

/*extern "system" {
    fn hsa_init() -> hsa_status_t;

    fn hsa_memory_allocate(
        hsa_region: hsa_region_t,
        size: size_t,
        ptr: *mut *mut c_void,
    ) -> hsa_status_t;
}*/

impl Compiler for HSARuntime {
    type Buffer = HSABuffer;
    type Program = HSAProgram;

    fn initialize() -> Result<Self, CompilerError> {
        /*let hdr = std::fs::read_to_string("/usr/include/linux/kfd_ioctl.h").map_err(|_| {
            CompilerError::InitializationFailure("Unable to read /usr/include/linux/kfd_ioctl.h")
        })?;*/

        //println!("{hdr}");

        let libhsa_path = match std::env::var("ROCM_PATH") {
            Ok(path) => path + "/lib/libhsa-runtime64.so",
            Err(_) => "/usr/lib64/libhsa-runtime64.so.1".into(), // TODO search for hsa-runtime64 path
        };

        //std::fs::File::open(&libhsa_path)
        //.map_err(|_| CompilerError::InitializationFailure("Unable to access hsa-runtime64"))?;

        println!("{libhsa_path}");

        let hsa_runtime_so = unsafe { Library::new(libhsa_path) }.unwrap();
        let hsa_init: Symbol<unsafe extern "system" fn() -> u32> =
            unsafe { hsa_runtime_so.get(b"hsa_init") }.unwrap();
        let status = unsafe { hsa_init() };
        println!("HSA runtime init status: {status}");

        Ok(Self {
            hsa_runtime_so,
            context: todo!(),
            devices: todo!(),
            queues: todo!(),
            queue_size: todo!(),
            queue_id: todo!(),
        })
    }

    fn hardware_information(&mut self) -> Result<HWInfo, CompilerError> {
        Ok(HWInfo {
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
        })
    }

    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, CompilerError> {
        todo!()
    }

    fn store_memory<T: crate::Scalar>(
        &mut self,
        buffer: &mut Self::Buffer,
        data: &[T],
    ) -> Result<(), super::CompilerError> {
        todo!()
    }

    fn load_memory<T: crate::Scalar>(
        &mut self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<alloc::vec::Vec<T>, super::CompilerError> {
        todo!()
    }

    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), CompilerError> {
        // TODO
        return Ok(());
    }

    fn compile_program(
        &mut self,
        kernel: &super::IRKernel,
    ) -> Result<Self::Program, CompilerError> {
        todo!()
    }

    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &mut [Self::Buffer],
    ) -> Result<(), CompilerError> {
        todo!()
    }

    fn release_program(&mut self, program: Self::Program) -> Result<(), CompilerError> {
        // TODO
        return Ok(());
    }
}
