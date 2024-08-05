use cuda::CUDAMemoryPool;
use host::HostMemoryPool;
use hsa::HSAMemoryPool;
use opencl::OpenCLMemoryPool;

mod host;
mod cuda;
mod hsa;
mod opencl;

#[derive(Debug)]
pub enum MemoryError {
    OutOfMemory,
}

#[derive(Debug)]
pub(super) enum MemoryPool {
    Host(HostMemoryPool),
    CUDA(CUDAMemoryPool),
    HSA(HSAMemoryPool),
    OpenCL(OpenCLMemoryPool),
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