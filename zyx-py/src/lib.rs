mod tensor;
mod device;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn zyx_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<device::Device>()?;
    m.add_class::<tensor::Tensor>()?;
    //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
