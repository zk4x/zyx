use super::{Compiler, HWInfo};

pub(crate) struct HIPBuffer {
    memory: *mut c_void,
    event: *mut c_void,
}

pub(crate) struct HIPProgram {
    name: String,
    program: *mut c_void,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    args_read_only: Vec<bool>,
}

pub(crate) struct HIPCompiler {
    context: *mut c_void,
    devices: BTreeSet<*mut c_void>,
    queues: Box<[*mut c_void]>,
    queue_size: Box<[u8]>,
    queue_id: usize,
}

impl Compiler for OpenCLCompiler {
    type Buffer = OpenCLBuffer;
    type Program = OpenCLProgram;

    fn initialize() -> Result<Self, super::CompilerError> {
        todo!()
    }

    fn hardware_information(&mut self) -> Result<HWInfo, super::CompilerError> {
        Ok(HWInfo {
            max_work_item_sizes: vec![1024, 1024, 1024],
            max_work_group_size: vec![1024, 1024, 1024],
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

    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, super::CompilerError> {
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

    fn deallocate_memory(&mut self, buffer: Self::Buffer) -> Result<(), super::CompilerError> {
        // TODO
        return Ok(());
    }

    fn compile_program(
        &mut self,
        kernel: &super::IRKernel,
    ) -> Result<Self::Program, super::CompilerError> {
        todo!()
    }

    fn launch_program(
        &mut self,
        program: &Self::Program,
        args: &mut [Self::Buffer],
    ) -> Result<(), super::CompilerError> {
        todo!()
    }

    fn release_program(&mut self, program: Self::Program) -> Result<(), super::CompilerError> {
        // TODO
        return Ok(());
    }
}
