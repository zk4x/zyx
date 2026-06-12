// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use zyx::Tensor;
use zyx_derive::Module;

/// # Stochastic gradient descent optimizer
#[derive(Module)]
#[cfg_attr(feature = "py", pyo3::pyclass)]
pub struct SGD {
    /// learning rate (default: 0.001)
    pub learning_rate: f32,
    /// momentum factor (default: 0.0)
    pub momentum: f32,
    /// weight decay (L2 penalty) (default: 0.0)
    pub weight_decay: f32,
    /// dampening for momentum (default: 0.0)
    pub dampening: f32,
    /// enables Nesterov momentum (default: false)
    pub nesterov: bool,
    /// maximize the objective with respect to the params, instead of minimizing (default: false)
    pub maximize: bool,
    /// stores momentum, starts empty and will be initialized on demand
    pub bias: Vec<Tensor>,
}

impl Default for SGD {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.0,
            weight_decay: 0.0,
            dampening: 0.0,
            nesterov: false,
            maximize: false,
            bias: Vec::new(),
        }
    }
}

impl SGD {
    /// Updates parameters with gradients.
    /// Number of parameters must be the same as number of gradients.
    /// Gradients can be None, those are simply skipped.
    pub fn update<'a>(
        &mut self,
        parameters: impl IntoIterator<Item = &'a mut Tensor>,
        gradients: impl IntoIterator<Item = Option<Tensor>>,
    ) {
        let params: Vec<&mut Tensor> = parameters.into_iter().collect();
        let grads: Vec<Option<Tensor>> = gradients.into_iter().collect();

        assert_eq!(
            params.len(),
            grads.len(),
            "Number of parameters != number of gradients."
        );

        let mut bias_idx = 0usize;
        for (param, grad) in params.into_iter().zip(grads) {
            if let Some(mut grad) = grad {
                if self.weight_decay != 0.0 {
                    grad = grad + param.clone() * self.weight_decay;
                }
                if self.momentum != 0.0 {
                    if bias_idx < self.bias.len() {
                        self.bias[bias_idx] =
                            self.bias[bias_idx].clone() * self.momentum + grad.clone() * (1.0 - self.dampening);
                    } else {
                        self.bias.push(grad.clone());
                    }
                    if self.nesterov {
                        grad = grad + self.bias[bias_idx].clone() * self.momentum;
                    } else {
                        grad = self.bias[bias_idx].clone();
                    }
                    bias_idx += 1;
                }
                if self.maximize {
                    *param = (&*param + grad * self.learning_rate).cast(param.dtype());
                } else {
                    *param = (&*param - grad * self.learning_rate).cast(param.dtype());
                }
            }
        }
    }
}
