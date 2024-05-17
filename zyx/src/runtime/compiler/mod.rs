use alloc::collections::BTreeMap;
use crate::scalar::Scalar;
use alloc::vec::Vec;

pub(super) mod opencl;
pub(super) mod cuda;
pub(super) mod wgpu;

type TensorId = u32;

pub(super) struct CompiledBackend<C: Compiler> {
    compiler: C,
    buffers: BTreeMap<TensorId, C::Buffer>,
}

pub(crate) enum CompilerError {
    InitializationFailure,
    DeviceOutOfMemory,
    HostOutOfMemory,
    // For all unknown errors
    GeneralExecutionError,
}

trait Compiler: Sized {
    type Buffer;
    type Program;
    fn initialize() -> Result<Self, CompilerError>;
    fn hwinfo(&mut self) -> Result<HWInfo, CompilerError>;
    fn allocate_memory(&mut self, byte_size: usize) -> Result<Self::Buffer, CompilerError>;
    fn store_memory<T>(&mut self, buffer: &mut Self::Buffer, data: &[T]) -> Result<(), CompilerError>;
    fn load_mem<T>(&mut self, buffer: &Self::Buffer, length: usize) -> Result<Vec<T>, CompilerError>;
    // Deallocation of resources must not fail.
    fn deallocate_memory(&mut self, buffer: Self::Buffer);
    fn compile_program(&mut self, kernel: IRKernel) -> Result<Self::Program, CompilerError>;
    fn launch_program(&mut self, program: &Self::Program, args: &[&mut Self::Buffer]) -> Result<(), CompilerError>;
    // Deallocation of resources must not fail.
    fn drop_program(&mut self, program: Self::Program);
}

pub(super) struct IRKernel {}

/// Hardware information needed for applying optimizations
#[derive(Debug)]
pub struct HWInfo {
    /// Biggest kernel dimensions
    pub max_work_item_sizes: Vec<usize>,
    /// Maximum local work size threads
    pub max_work_group_size: usize,
    /// Preferred vector size in bytes
    pub preferred_vector_size: usize,
    /// Is half supported?
    pub f16_support: bool,
    /// Is double supported?
    pub f64_support: bool,
    /// Is fused multiply add supported?
    pub fmadd: bool,
    /// Global (VRAM, RAM) memory size in bytes
    pub global_mem_size: usize,
    /// Maximum memory allocation for single buffer in bytes
    pub max_mem_alloc: usize,
    /// Alignment for data types in bytes
    pub mem_align: usize,
    /// Page size (base address alignment) in bytes
    pub page_size: usize,
    /// Local memory size in bytes
    pub local_mem_size: usize,
    /// Number of registers per thread
    pub num_registers: usize,
}

impl<C: Compiler> CompiledBackend<C> {
    pub(super) fn initialize() -> Result<Self, CompilerError> {
        Ok(Self {
            compiler: C::initialize()?,
            buffers: BTreeMap::new(),
        })
    }

    pub(super) fn store<T: Scalar>(&mut self, data: &[T]) -> Result<(), CompilerError> {
        todo!()
    }

    pub(super) fn remove(&mut self, x: TensorId) {
        if let Some(buffer) = self.buffers.remove(&x) {
            self.compiler.deallocate_memory(buffer);
        }
    }
}
