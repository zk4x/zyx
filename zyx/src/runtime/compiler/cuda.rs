use crate::runtime::compiler::ir::IRKernel;
use crate::runtime::compiler::{Compiler, CompilerError, HWInfo};
use alloc::vec::Vec;

pub(crate) struct CUDA {}

impl Compiler for CUDA {
    type Buffer = ();
    type Program = ();

    fn initialize() -> Result<Self, CompilerError> {
        // TODO
        return Err(CompilerError::InitializationFailure(
            "CUDA device is not available.",
        ));
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
