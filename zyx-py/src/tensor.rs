use pyo3::{pyclass, pymethods, PyResult};
use crate::device::PyDevice;

#[derive(Clone)]
#[pyclass(name = "DType")]
pub enum PyDType {
    F16,
    F32,
    F64,
    I32,
}

impl PyDType {
    pub(crate) fn dtype(&self) -> DType {
        match self {
            PyDType::F16 => DType::F16,
            PyDType::F32 => DType::F32,
            PyDType::F64 => DType::F64,
            PyDType::I32 => DType::I32,
        }
    }
}

#[pyclass(name = "Tensor")]
pub struct PyTensor {
    pub(crate) id: Id,
    pub(crate) device: PyDevice,
}

#[pymethods]
impl PyTensor {
    fn ln(&self) -> PyResult<Self> {
        todo!()
    }
}
