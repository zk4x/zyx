// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Python bindings for zyx optimizers

use crate::{Adam, AdamW, RMSprop, SGD};
use pyo3::prelude::*;
use pyo3::types::PyList;
use zyx::Tensor;

/// Run one optimizer step over `params` from `grads` (parallel lists; `None`
/// means "no gradient for this parameter").
///
/// Every optimizer writes its update with `Tensor::assign`, i.e. in place, so
/// the caller's tensors keep their identity and nothing has to be written back.
fn do_update(
    params_list: &Bound<'_, PyList>,
    grads_list: &Bound<'_, PyList>,
    update_fn: impl FnOnce(&mut Vec<Tensor>, Vec<Tensor>),
) -> PyResult<()> {
    let mut params: Vec<Tensor> = params_list
        .iter()
        .map(|t| t.extract::<Tensor>().expect("params must be a list of Tensor"))
        .collect();
    let grads: Vec<Tensor> = grads_list
        .iter()
        .enumerate()
        .map(|(i, t)| {
            if t.is_none() {
                Tensor::zeros_like(params[i].clone())
            } else {
                t.extract::<Tensor>().expect("gradients must be a list of Tensor or None")
            }
        })
        .collect();
    update_fn(&mut params, grads);
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

    /// Update `params` in place from `gradients`.
    ///
    /// The two lists are parallel; a `None` gradient skips that parameter.
    #[pyo3(name = "update")]
    fn update_py(&mut self, params: &Bound<'_, PyList>, gradients: &Bound<'_, PyList>) -> PyResult<()> {
        let s = self;
        do_update(params, gradients, |p, g| SGD::update(s, p.iter_mut(), g.into_iter()))
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

    /// Update `params` in place from `gradients`.
    ///
    /// The two lists are parallel; a `None` gradient skips that parameter.
    #[pyo3(name = "update")]
    fn update_py(&mut self, params: &Bound<'_, PyList>, gradients: &Bound<'_, PyList>) -> PyResult<()> {
        let s = self;
        do_update(params, gradients, |p, g| Adam::update(s, p.iter_mut(), g.into_iter()))
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

    /// Update `params` in place from `gradients`.
    ///
    /// The two lists are parallel; a `None` gradient skips that parameter.
    #[pyo3(name = "update")]
    fn update_py(&mut self, params: &Bound<'_, PyList>, gradients: &Bound<'_, PyList>) -> PyResult<()> {
        let s = self;
        do_update(params, gradients, |p, g| AdamW::update(s, p.iter_mut(), g.into_iter()))
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

    /// Update `params` in place from `gradients`.
    ///
    /// The two lists are parallel; a `None` gradient skips that parameter.
    #[pyo3(name = "update")]
    fn update_py(&mut self, params: &Bound<'_, PyList>, gradients: &Bound<'_, PyList>) -> PyResult<()> {
        let s = self;
        do_update(params, gradients, |p, g| RMSprop::update(s, p.iter_mut(), g.into_iter()))
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
