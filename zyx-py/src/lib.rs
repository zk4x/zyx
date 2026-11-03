// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Python bindings for zyx ML library

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Attach `child` to `parent` and also publish it under `dotted` in
/// `sys.modules`.
///
/// `add_submodule` only sets the attribute, so `import zyx.nn` would still
/// fail with `ModuleNotFoundError`; `from zyx import nn` happened to work
/// because that path goes through the attribute. Registering in
/// `sys.modules` is what makes the plain `import` form work, and it also
/// stops a later `import zyx.nn` from re-executing the module.
fn add_submodule(parent: &Bound<'_, PyModule>, child: &Bound<'_, PyModule>, dotted: &str) -> PyResult<()> {
    parent.add_submodule(child)?;
    parent
        .py()
        .import("sys")?
        .getattr("modules")?
        .set_item(dotted, child)?;
    Ok(())
}

#[pymodule]
#[pyo3(name = "zyx")]
fn zyx_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    // Register tensor and core functionality
    zyx::py_bindings::register_tensor(m)?;
    zyx::py_bindings::register_dtype(m)?;
    zyx::py_bindings::register_tape(m)?;

    // Create nn submodule
    let nn_mod = PyModule::new(m.py(), "nn")?;
    zyx_nn::register_nn(&nn_mod)?;
    add_submodule(m, &nn_mod, "zyx.nn")?;

    // Create optim submodule
    let optim_mod = PyModule::new(m.py(), "optim")?;
    zyx_optim::register_optimizers(&optim_mod)?;
    add_submodule(m, &optim_mod, "zyx.optim")?;

    Ok(())
}
