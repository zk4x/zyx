use crate::runtime::compiler::ir::IRKernel;
use crate::runtime::compiler::{Compiler, CompilerError, HWInfo};
use alloc::vec::Vec;
use wgpu::Instance;

pub(crate) struct WGPU {
    instance: Instance,
}

impl Compiler for WGPU {
    type Buffer = ();
    type Program = ();

    fn initialize() -> Result<Self, CompilerError> {
        let instance = Instance::new(todo!());

        return Ok(Self {
            instance,
        })
    }

    fn hardware_information(&mut self) -> Result<HWInfo, CompilerError> {
        todo!()
    }

    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, CompilerError> {
        let _ = byte_size;
        todo!()
    }

    fn store_memory<T>(
        &mut self,
        buffer: &mut Self::Buffer,
        data: &[T],
    ) -> Result<(), CompilerError> {
        let _ = buffer;
        let _ = data;
        todo!()
    }

    fn load_memory<T>(
        &mut self,
        buffer: &Self::Buffer,
        length: usize,
    ) -> Result<Vec<T>, CompilerError> {
        let _ = buffer;
        let _ = length;
        todo!()
    }

    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), CompilerError> {
        let _ = buffer;
        todo!()
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
