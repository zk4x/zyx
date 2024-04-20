mod tensor;
mod device;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn zyx_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<device::PyDevice>()?;
    m.add_class::<tensor::PyTensor>()?;
    //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
