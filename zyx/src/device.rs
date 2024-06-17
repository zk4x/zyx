#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Device {
    CUDA,
    OpenCL,
    #[cfg(feature = "wgpu")]
    WGPU,
    CPU,
}
