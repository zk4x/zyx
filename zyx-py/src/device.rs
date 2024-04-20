use std::sync::{Arc};
use pyo3::{pyclass, pymethods, PyResult};
use pyo3::types::PyList;
use zyx_core::backend::Backend;
use zyx_cpu::CPU;
use crate::tensor::{PyDType, PyTensor};

#[derive(Clone)]
#[pyclass(name = "Device")]
pub struct PyDevice(Arc<Dev>);

pub enum Dev {
    CPU(CPU),
    //OpenCL(OpenCL),
}

#[pymethods]
impl PyDevice {
    pub fn randn(&self, shape: &PyList, dtype: PyDType) -> PyResult<PyTensor> {
        let shape: Vec<usize> = shape.extract()?;
        Ok(PyTensor {
            id: match self.0.as_ref() {
                Dev::CPU(cpu) => cpu.randn(shape, dtype.dtype()).unwrap().id(),
            },
            device: self.clone(),
        })
    }
}
