//! Various optimizers to update [Variables](crate::tensor::Variable).
//! 

use std::ops::{Sub, Mul};

/// # Optimizer trait
/// 
/// All optimizers must implement this trait.
pub trait Optimizer {
    /// Update one of [parameters](crate::module::Parameters)
    fn update_data<S, G>(&self, data: S, grad: G) -> S
    where
        G: Mul<f64>,
        S: Sub<<G as Mul<f64>>::Output, Output = S>;
}

/// # Stochastic gradient descent optimizer
///
/// Updates [parameter's](crate::module::Parameters) data using following function:
/// ```txt
/// x.data = x.data - x.grad * learning_rate;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SGD {
    learning_rate: f64,
}

impl Default for SGD {
    fn default() -> Self {
        Self::new()
    }
}

impl SGD {
    /// Create new [SGD] from given parameters
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
        }
    }

    /// Set learning rate for [SGD]
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }
}

impl Optimizer for SGD {
    fn update_data<S, G>(&self, data: S, grad: G) -> S
    where
        G: Mul<f64>,
        S: Sub<<G as Mul<f64>>::Output, Output = S>,
    {
        data - grad * self.learning_rate
    }
}
