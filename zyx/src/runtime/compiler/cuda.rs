use crate::runtime::compiler::{Compiler, CompilerError, HWInfo, IRKernel};
use alloc::vec::Vec;

pub(crate) struct CUDA {}

impl Compiler for CUDA {
    type Buffer = ();
    type Program = ();

    fn initialize() -> Result<Self, CompilerError> {
        todo!()
    }

    fn hwinfo(&mut self) -> Result<HWInfo, CompilerError> {
        todo!()
    }

    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, CompilerError> {
        todo!()
    }

    fn store_memory<T>(&mut self, buffer: &mut Self::Buffer, data: &[T]) -> Result<(), CompilerError> {
        todo!()
    }

    fn load_mem<T>(&mut self, buffer: &Self::Buffer, length: usize) -> Result<Vec<T>, CompilerError> {
        todo!()
    }

    fn deallocate_memory(&mut self, buffer: Self::Buffer) {
        todo!()
    }

    fn compile_program(&mut self, kernel: IRKernel) -> Result<Self::Program, CompilerError> {
        todo!()
    }

    fn launch_program(&mut self, program: &Self::Program, args: &[&mut Self::Buffer]) -> Result<(), CompilerError> {
        todo!()
    }

    fn drop_program(&mut self, program: Self::Program) {
        todo!()
    }
}