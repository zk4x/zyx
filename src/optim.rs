//! Various optimizers to update tensors with gradients.
//! 

// TODO: currently, all parameter tensors must have same storage
// variadic templates would be useful

use std::rc::Rc;

pub(crate) trait OptimizableTensor<S> {
    fn update_data(&self, f: &dyn Fn(&mut Rc<S>) -> Rc<S>);
    fn zero_grad(&self);
}

trait Optimizer<'a, S> {
    fn parameters(&self) -> &Vec<&'a dyn OptimizableTensor<S>>;
    fn step(&self);
    fn zero_grad(&self);
}

pub struct SGD<'a, S> {
    parameters: Vec<&'a dyn OptimizableTensor<S>>,
    learning_rate: f64,
}

impl<'a, S> Optimizer<'a, S> for SGD<'a, S>
where
    for<'b> &'b S: std::ops::Mul<f64, Output = S>,
{
    fn parameters(&self) -> &Vec<&'a dyn OptimizableTensor<S>> {
        &self.parameters
    }

    fn step(&self) {
        self.parameters.iter().for_each(|param| param.update_data(&|data| Rc::new(data.as_ref() * (1. - self.learning_rate))));
    }

    fn zero_grad(&self) {
        self.parameters.iter().for_each(|param| param.zero_grad());
    }
}