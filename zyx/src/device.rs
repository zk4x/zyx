#[derive(Clone, Copy)]
pub enum Device {
    CUDA,
    OpenCL,
    WGPU,
    CPU,
}