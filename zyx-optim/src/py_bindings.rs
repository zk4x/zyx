// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Python bindings for zyx optimizers

use crate::{Adam, AdamW, RMSprop, SGD};
use pyo3::prelude::*;
use pyo3::types::PyList;
use zyx::Tensor;

fn do_update(
    model: &Bound<'_, PyAny>,
    grads_list: &Bound<'_, PyList>,
    update_fn: impl FnOnce(&mut Vec<Tensor>, Vec<Option<Tensor>>),
    py: Python<'_>,
) -> PyResult<()> {
    let params_obj = model.call_method0("get_params")?;
    let params_list = params_obj.cast::<PyList>().unwrap();
    let params: Vec<Tensor> = params_list
        .iter()
        .map(|t| t.extract::<Tensor>().expect("params must be list of Tensor"))
        .collect();
    let grads: Vec<Option<Tensor>> = grads_list
        .iter()
        .map(|t| {
            if t.is_none() {
                None
            } else {
                Some(t.extract::<Tensor>().expect("gradients must be list of Tensor or None"))
            }
        })
        .collect();
    let mut params_mut = params;
    update_fn(&mut params_mut, grads);
    let new_list = PyList::empty(py);
    for p in params_mut {
        new_list.append(p)?;
    }
    model.call_method1("set_params", (new_list,))?;
    Ok(())
}

/// Python bindings for SGD optimizer.
#[pymethods]
impl SGD {
    /// Create a new SGD optimizer.
    #[new]
    #[pyo3(signature = (learning_rate=0.001, momentum=0.0, weight_decay=0.0, dampening=0.0, nesterov=false, maximize=false))]
    fn py_new(
        learning_rate: f32,
        momentum: f32,
        weight_decay: f32,
        dampening: f32,
        nesterov: bool,
        maximize: bool,
    ) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            dampening,
            nesterov,
            maximize,
            bias: Vec::new(),
        }
    }

    /// Update model parameters using computed gradients.
    #[pyo3(name = "update")]
    fn update_py(&mut self, model: &Bound<'_, PyAny>, gradients: &Bound<'_, PyList>, py: Python<'_>) -> PyResult<()> {
        let s = self;
        do_update(model, gradients, |p, g| SGD::update(s, p.iter_mut(), g.into_iter()), py)
    }
}

/// Python bindings for Adam optimizer.
#[pymethods]
impl Adam {
    /// Create a new Adam optimizer.
    #[new]
    #[pyo3(signature = (learning_rate=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, amsgrad=false))]
    fn py_new(
        learning_rate: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        amsgrad: bool,
    ) -> Self {
        Self {
            learning_rate,
            betas,
            eps,
            weight_decay,
            amsgrad,
            m: Vec::new(),
            v: Vec::new(),
            vm: Vec::new(),
            t: 0,
        }
    }

    /// Update model parameters using computed gradients.
    #[pyo3(name = "update")]
    fn update_py<'py>(&mut self, model: &Bound<'py, PyAny>, gradients: &Bound<'py, PyList>, py: Python<'py>) -> PyResult<()> {
        let s = self;
        do_update(model, gradients, |p, g| Adam::update(s, p.iter_mut(), g.into_iter()), py)
    }
}

/// Python bindings for AdamW optimizer.
#[pymethods]
impl AdamW {
    /// Create a new AdamW optimizer.
    #[new]
    #[pyo3(signature = (learning_rate=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=false))]
    fn py_new(
        learning_rate: f32,
        betas: (f32, f32),
        eps: f32,
        weight_decay: f32,
        amsgrad: bool,
    ) -> Self {
        Self {
            learning_rate,
            betas,
            eps,
            weight_decay,
            amsgrad,
            m: Vec::new(),
            v: Vec::new(),
            vm: Vec::new(),
            t: 0,
        }
    }

    /// Update model parameters using computed gradients.
    #[pyo3(name = "update")]
    fn update_py<'py>(&mut self, model: &Bound<'py, PyAny>, gradients: &Bound<'py, PyList>, py: Python<'py>) -> PyResult<()> {
        let s = self;
        do_update(model, gradients, |p, g| AdamW::update(s, p.iter_mut(), g.into_iter()), py)
    }
}

/// Python bindings for RMSprop optimizer.
#[pymethods]
impl RMSprop {
    /// Create a new RMSprop optimizer.
    #[new]
    #[pyo3(signature = (learning_rate=0.01, alpha=0.99, eps=1e-8, momentum=0.0, centered=false, weight_decay=0.0))]
    fn py_new(
        learning_rate: f32,
        alpha: f32,
        eps: f32,
        momentum: f32,
        centered: bool,
        weight_decay: f32,
    ) -> Self {
        let mut opt = Self::default();
        opt.learning_rate = learning_rate;
        opt.alpha = alpha;
        opt.eps = eps;
        opt.momentum = momentum;
        opt.centered = centered;
        opt.weight_decay = weight_decay;
        opt
    }

    /// Update model parameters using computed gradients.
    #[pyo3(name = "update")]
    fn update_py<'py>(&mut self, model: &Bound<'py, PyAny>, gradients: &Bound<'py, PyList>, py: Python<'py>) -> PyResult<()> {
        let s = self;
        do_update(model, gradients, |p, g| RMSprop::update(s, p.iter_mut(), g.into_iter()), py)
    }
}

/// Helper to register optimizer classes in the zyx-py module.
pub fn register_optimizers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SGD>()?;
    m.add_class::<Adam>()?;
    m.add_class::<AdamW>()?;
    m.add_class::<RMSprop>()?;
    Ok(())
}
