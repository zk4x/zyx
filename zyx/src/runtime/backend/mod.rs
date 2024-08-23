pub(crate) mod cuda;
pub(crate) mod hip;
pub(crate) mod opencl;

/// Hardware information needed for applying optimizations
#[derive(Debug, Default)]
pub(super) struct DeviceInfo {
    /// Device compute in flops
    pub compute: u128,
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
    /// Page size (base address alignment) in bytes
    pub page_size: usize,
    /// Does this device have local memory?
    pub local_memory: bool,
    /// Local memory size in bytes
    pub local_mem_size: usize,
    /// Number of registers per thread
    pub num_registers: usize,
    /// Does this hardware support native matmul of 16x16 local tiles?
    pub wmma: bool,
    /// Does this hardware have tensor cores?
    pub tensor_cores: bool,
}

pub(super) struct MemoryInfo {
    /// Global (VRAM, RAM) memory size in bytes
    pub total_memory: usize,
    /// Maximum memory allocation for single buffer in bytes
    pub max_alloc_size: usize,
    /// Page size, minimum allocatable size
    pub page_size: usize,
    /// Alignment for data types in bytes
    pub alignment: usize,
}
