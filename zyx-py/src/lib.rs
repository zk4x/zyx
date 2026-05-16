// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Python bindings for zyx ML library

use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pymodule]
#[pyo3(name = "zyx")]
fn zyx_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Register tensor and core functionality
    zyx::py_bindings::register_tensor(m)?;
    zyx::py_bindings::register_dtype(m)?;
    zyx::py_bindings::register_gradient_tape(m)?;
    
    // Create nn submodule
    let nn_mod = PyModule::new(m.py(), "nn")?;
    zyx_nn::register_nn(&nn_mod)?;
    m.add_submodule(&nn_mod)?;
    
    // Create optim submodule
    let optim_mod = PyModule::new(m.py(), "optim")?;
    zyx_optim::register_optimizers(&optim_mod)?;
    m.add_submodule(&optim_mod)?;
    
    Ok(())
}
