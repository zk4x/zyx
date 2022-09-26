//! Various optimizers to update tensors with gradients.
//! 

// TODO: currently, all parameter tensors must have same storage
// variadic templates would be useful

use std::rc::Rc;
use crate::tensor::TensorGrad;

pub trait Optimizer<'a, S> {
    fn parameters(&self) -> &[&'a TensorGrad<S>];
    fn step(&self);
    fn zero_grad(&self);
}

pub struct SGD<'a, S> {
    parameters: Vec<&'a TensorGrad<S>>,
    learning_rate: f32,
}

impl<'a, S> SGD<'a, S> {
    pub fn new(parameters: &[&'a TensorGrad<S>]) -> Self {
        Self {
            parameters: parameters.to_vec(),
            learning_rate: 0.01,
        }
    }
}

impl<'a, S> Optimizer<'a, S> for SGD<'a, S>
where
    S: Default,
    for<'b> &'b S: std::ops::Mul<f32, Output = S>,
{
    fn parameters(&self) -> &[&'a TensorGrad<S>] {
        &self.parameters
    }

    fn step(&self) {
        self.parameters.iter().for_each(|param| param.update_data(&|data| Rc::new(data.as_ref() * (1. - self.learning_rate))));
    }

    fn zero_grad(&self) {
        self.parameters.iter().for_each(|param| param.zero_grad());
    }
}