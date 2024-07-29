#[cfg_attr(feature = "py", pyo3::pyclass(eq, eq_int))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Device {
    CUDA,
    HSA,
    OpenCL,
    WGSL,
    CPU,
}
