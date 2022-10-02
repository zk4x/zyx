//! Various optimizers to update Buffers with gradients.
//! 

// TODO: currently, all parameter Buffers must have same storage
// variadic templates would be useful

use crate::tensor::Variable;

pub trait Optimizer<'a, S> {
    fn parameters(&self) -> &[&'a Variable<S>];
    fn step(&self);
    fn zero_grad(&self);
}

pub struct SGD<'a, S> {
    parameters: Vec<&'a Variable<S>>,
    learning_rate: f32,
}

impl<'a, S> SGD<'a, S> {
    pub fn new(parameters: &[&'a Variable<S>]) -> Self {
        Self {
            parameters: parameters.to_vec(),
            learning_rate: 0.01,
        }
    }
}

impl<'a, S> Optimizer<'a, S> for SGD<'a, S>
where
    S: Default + std::ops::Mul<f32, Output = S>,
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