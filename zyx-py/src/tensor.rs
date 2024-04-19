use pyo3::pyclass;
use zyx_core::backend::Backend;
use zyx_core::tensor::Id;
use zyx_cpu::CPU;
use zyx_opencl::OpenCL;

#[pyclass]
pub struct Tensor {
    id: Id,
    //backend,
}
