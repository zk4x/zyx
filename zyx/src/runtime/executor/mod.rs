use super::{memory::Buffer, ir::IRKernel};

pub(super) mod cuda;
//pub(super) mod hsa;
//pub(super) mod opencl;
pub(super) mod x86_64;

#[derive(Debug)]
pub enum ExecError {}

#[derive(Debug)]
pub enum Program {}

// Each platform serves both as allocator and executor
pub(super) struct Executor {}

impl Executor {
    pub(super) const fn new() -> Self {
        Self {}
    }

    pub(super) fn hardware_information(&mut self) -> Result<HWInfo, ExecError> {
        todo!()
    }

    pub(super) fn compile_program(&mut self, kernel: &IRKernel) -> Result<Program, ExecError> {
        todo!()
    }

    pub(super) fn launch_program(&mut self, program: &Program, args: &mut [Buffer]) -> Result<(), ExecError> {
        todo!()
    }

    pub(super) fn release_program(&mut self, program: Program) -> Result<(), ExecError> {
        todo!()
    }
}

/// Hardware information needed for applying optimizations
#[allow(unused)]
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
    /// Does this device have local memory?
    pub local_memory: bool,
    /// Does this hardware support native matmul of 16x16 local tiles?
    pub wmma: bool,
    /// Does this hardware have tensor cores?
    pub tensor_cores: bool,
}
