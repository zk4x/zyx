#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Device {
    #[cfg(feature = "cuda")]
    CUDA,
    #[cfg(feature = "hsa")]
    HSA,
    #[cfg(feature = "opencl")]
    OpenCL,
    #[cfg(feature = "wgsl")]
    WGSL,
    CPU,
}
