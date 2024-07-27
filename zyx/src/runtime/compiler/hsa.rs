#![allow(non_camel_case_types)]

use super::{Compiler, CompilerError, HWInfo};
use alloc::vec;
use alloc::{boxed::Box, collections::BTreeSet, string::String, vec::Vec};
use core::ffi::c_void;
use hsa::{Agent, Region};

#[cfg(feature = "debug1")]
use std::println;

pub(crate) struct HSABuffer {
    memory: hsa::Memory<u8>,
}

pub(crate) struct HSAProgram {
    name: String,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    args_read_only: Vec<bool>,
    program: *mut c_void,
}

pub(crate) struct HSARuntime {
    agent: Agent,
    mem_region: Region,
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

impl Drop for HSARuntime {
    fn drop(&mut self) {
        hsa::shutdown().unwrap();
    }
}

impl Compiler for HSARuntime {
    type Buffer = HSABuffer;
    type Program = HSAProgram;

    fn initialize() -> Result<Self, CompilerError> {
        hsa::init().unwrap();

        let agents = Agent::list().unwrap();
        println!("Agents:\n{agents:?}");
        let agent = agents[1];
        let device = agent.device().unwrap();
        println!("Device: {device:?}");
        let regions = agent.regions().unwrap();
        let mem_region = regions[0];

        Ok(Self { agent, mem_region })
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
        let memory = hsa::Memory::allocate(self.mem_region, byte_size).unwrap();
        return Ok(HSABuffer { memory });
    }

    fn store_memory<T: crate::Scalar>(
        &mut self,
        buffer: &mut Self::Buffer,
        data: Vec<T>,
    ) -> Result<(), super::CompilerError> {
        //let ptr = buffer.memory.as_mut_ptr();
        /*for i in 0..data.len() {
        // TODO get this working with different dtypes
        let x = data[i].into_f32().to_ne_bytes();
        unsafe {
            *ptr.add(i) = x[i];
            *ptr.add(i + 1) = x[i + 1];
            *ptr.add(i + 2) = x[i + 2];
            *ptr.add(i + 3) = x[i + 3];
        };
        }*/
        return Ok(());
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
