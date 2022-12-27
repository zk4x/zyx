use ocl;

/// OpenCL device
/// 
/// This device provides acces to OpenCL devices on your computer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Device {}

impl super::Device for Device {}
