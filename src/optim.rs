//! Various optimizers to update Buffers with gradients.
//! 

// TODO: add Parameters struct, that will be used instead of Vec<&'a Variable>
// It should be dynamic over all variable storages.

// TODO: currently, all parameter Buffers must have same storage
// we can solve this using dyn, but it would be nice, if we could use
// some kind of dynamic tuples

use crate::{ops::Zeros, tensor::Variable};
use std::ops::Mul;

/// # Optimizer trait
/// 
/// All optimizers must implement this trait.
pub trait Optimizer<'a, S> {
    /// Get all parameters that optimizer has.
    fn parameters(&self) -> &[&'a Variable<S>];
    /// Update data in parameters using their gradients.
    fn step(&self);
    /// Fill parameter gradients with zeros.
    fn zero_grad(&self);
}

/// Stochastic gradient descent optimizer
pub struct SGD<'a, S> {
    parameters: Vec<&'a Variable<S>>,
    learning_rate: f32,
}

impl<'a, S> SGD<'a, S> {
    /// Create new SGD from given parameters
    pub fn new(parameters: &[&'a Variable<S>]) -> Self {
        Self {
            parameters: parameters.to_vec(),
            learning_rate: 0.01,
        }
    }
}

impl<'a, S> Optimizer<'a, S> for SGD<'a, S>
where
    S: Default + Zeros + Mul<f32, Output = S>,
{
    fn parameters(&self) -> &[&'a Variable<S>] {
        &self.parameters
    }

    fn step(&self) {
        self.parameters.iter().for_each(|param| param.update_data(|data| data * (1. - self.learning_rate)));
    }

    fn zero_grad(&self) {
        self.parameters.iter().for_each(|param| param.zero_grad());
    }
}