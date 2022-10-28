//! Various optimizers to update [Variables](crate::tensor::Variable).
//! 

use crate::module::Parameters;
use std::ops::{Mul, Sub};

/// # Optimizer trait
/// 
/// All optimizers must implement this trait.
pub trait Optimizer {
    /// Optimizer's [parameters](crate::module::Parameters)
    type P: Parameters;

    /// Get all [parameters](crate::module::Parameters) that optimizer has.
    fn parameters(&self) -> &Self::P;

    /// Update one of [parameters](crate::module::Parameters)
    fn update_data<S>(&self, data: S, grad: S) -> S
    where
        // These are the requirements for SGD. For other optimizers, they may be subject to change.
        S: Sub<Output = S> + Mul<Output = S> + Mul<f64, Output = S>;

    /// Update data in [parameters](crate::module::Parameters) using their gradients.
    fn step(&self)
    where
        Self: Optimizer,
        Self: Sized,
    {
        self.parameters().update_data(self);
    }

    /// Fill [parameter](crate::module::Parameters) gradients with zeros.
    fn zero_grad(&self) {
        self.parameters().zero_grad();
    }
}

/// # Stochastic gradient descent optimizer
///
/// Updates [parameter's](crate::module::Parameters) data using following function:
/// ```txt
/// x.data = x.data - x.grad * learning_rate;
/// ```
pub struct SGD<Params> {
    parameters: Params,
    learning_rate: f64,
}

impl<Params> SGD<Params> {
    /// Create new [SGD] from given parameters
    pub fn new(parameters: Params) -> Self
    where
        Params: Parameters,
    {
        Self {
            parameters,
            learning_rate: 0.01,
        }
    }

    /// Set learning rate for [SGD]
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }
}

impl<Params> Optimizer for SGD<Params>
where
    Params: Parameters,
{
    type P = Params;
    fn parameters(&self) -> &Self::P {
        &self.parameters
    }

    fn update_data<S>(&self, data: S, grad: S) -> S
    where
        S: Sub<Output = S> + Mul<Output = S> + Mul<f64, Output = S>,
    {
        data - grad * self.learning_rate
    }
}
