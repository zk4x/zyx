#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Device {
    #[cfg(feature = "cuda")]
    CUDA,
    #[cfg(feature = "hip")]
    HIP,
    #[cfg(feature = "opencl")]
    OpenCL,
    #[cfg(feature = "wgpu")]
    WGPU,
    CPU,
}
