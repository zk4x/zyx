// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Python bindings for zyx ML library

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pymodule]
#[pyo3(name = "zyx")]
fn zyx_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    zyx::py_bindings::register_tensor(m)?;
    zyx::py_bindings::register_dtype(m)?;
    zyx::py_bindings::register_gradient_tape(m)?;
    zyx_optim::register_optimizers(m)?;
    zyx_nn::register_nn(m)?;
    Ok(())
}
