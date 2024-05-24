use pyo3::prelude::*;

use zyx as _;

/// A Python module implemented in Rust.
#[pymodule]
fn zyx_py(_py: Python, m: &PyModule) -> PyResult<()> {
    //m.add_class::<zyx::Device>()?;
    //m.add_class::<zyx::Tensor>()?;
    //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
