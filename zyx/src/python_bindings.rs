use std::vec::Vec;

use pyo3::{
    pymethods, pymodule,
    types::{PyAnyMethods, PyModule, PyModuleMethods, PyTuple},
    Bound, PyResult,
};

use crate::{DType, Tensor};

#[pymethods]
impl Tensor {
    #[staticmethod]
    #[pyo3(name = "randn", signature = (*shape, dtype=DType::F32))]
    pub fn randn_py(shape: &Bound<'_, PyTuple>, dtype: DType) -> Tensor {
        let shape: Vec<usize> = shape
            .into_iter()
            .map(|d| {
                d.extract::<usize>()
                    .expect("Shape must be positive integers")
            })
            .collect();
        return Tensor::randn(shape, dtype);
    }
}

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "zyx")]
fn zyx_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Tensor>()?;
    //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
