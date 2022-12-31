use core::ops::{Mul, Sub};

/// # Stochastic gradient descent optimizer
///
/// Updates [parameter's](crate::nn::parameters::Parameters) data using following function:
/// ```txt
/// x.data = x.data - x.grad * learning_rate;
/// ```
// TODO DOCS
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SGD {
    learning_rate: f32,
}

impl Default for SGD {
    fn default() -> Self {
        Self::new()
    }
}

impl SGD {
    /// # Create new [SGD]
    ///
    /// This creates new [SGD] with learning rate 0.01
    pub fn new() -> Self {
        Self {
            learning_rate: 0.01,
        }
    }

    /// Set learning rate for [SGD]
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }
}

// The optimizer and loader and function that zeros gradients should be directly implemented for Variable and all tuples of Variables.
// If this can't be done in standard rust, we should create macro for that.

/// Step over SGD optimizers parameters
pub trait SGDStep {
    /// Call the step function
    fn step(self, optim: &SGD);
}

use crate::tensor::Variable;
impl<S> SGDStep for &mut Variable<S>
where
    S: Clone + Mul<f32> + Sub<<S as Mul<f32>>::Output, Output = S>,
{
    fn step(self, optim: &SGD) {
        self.data = self.data.clone() - self.grad().clone() * optim.learning_rate;
    }
}

// This should be automatically derived;
impl SGDStep for () {
    fn step(self, _: &SGD) {}
}

// This should be automatically derived;
impl<V1, V2> SGDStep for (V1, V2)
where
    V1: SGDStep,
    V2: SGDStep,
{
    fn step(self, optim: &SGD) {
        self.0.step(optim);
        self.1.step(optim);
    }
}

// This should be automatically derived;
impl<V1, V2, V3> SGDStep for (V1, V2, V3)
where
    V1: SGDStep,
    V2: SGDStep,
    V3: SGDStep,
{
    fn step(self, optim: &SGD) {
        self.0.step(optim);
        self.1.step(optim);
        self.2.step(optim);
    }
}

// This should be automatically derived;
impl<V1, V2, V3, V4> SGDStep for (V1, V2, V3, V4)
where
    V1: SGDStep,
    V2: SGDStep,
    V3: SGDStep,
    V4: SGDStep,
{
    fn step(self, optim: &SGD) {
        self.0.step(optim);
        self.1.step(optim);
        self.2.step(optim);
        self.3.step(optim);
    }
}

// This should be automatically derived;
impl<V1, V2, V3, V4, V5> SGDStep for (V1, V2, V3, V4, V5)
where
    V1: SGDStep,
    V2: SGDStep,
    V3: SGDStep,
    V4: SGDStep,
    V5: SGDStep,
{
    fn step(self, optim: &SGD) {
        self.0.step(optim);
        self.1.step(optim);
        self.2.step(optim);
        self.3.step(optim);
        self.4.step(optim);
    }
}

// This should be automatically derived;
impl<V1, V2, V3, V4, V5, V6> SGDStep for (V1, V2, V3, V4, V5, V6)
where
    V1: SGDStep,
    V2: SGDStep,
    V3: SGDStep,
    V4: SGDStep,
    V5: SGDStep,
    V6: SGDStep,
{
    fn step(self, optim: &SGD) {
        self.0.step(optim);
        self.1.step(optim);
        self.2.step(optim);
        self.3.step(optim);
        self.4.step(optim);
        self.5.step(optim);
    }
}

// This should be automatically derived;
impl<V1, V2, V3, V4, V5, V6, V7> SGDStep for (V1, V2, V3, V4, V5, V6, V7)
where
    V1: SGDStep,
    V2: SGDStep,
    V3: SGDStep,
    V4: SGDStep,
    V5: SGDStep,
    V6: SGDStep,
    V7: SGDStep,
{
    fn step(self, optim: &SGD) {
        self.0.step(optim);
        self.1.step(optim);
        self.2.step(optim);
        self.3.step(optim);
        self.4.step(optim);
        self.5.step(optim);
        self.6.step(optim);
    }
}

// This should be automatically derived;
impl<V1, V2, V3, V4, V5, V6, V7, V8> SGDStep for (V1, V2, V3, V4, V5, V6, V7, V8)
where
    V1: SGDStep,
    V2: SGDStep,
    V3: SGDStep,
    V4: SGDStep,
    V5: SGDStep,
    V6: SGDStep,
    V7: SGDStep,
    V8: SGDStep,
{
    fn step(self, optim: &SGD) {
        self.0.step(optim);
        self.1.step(optim);
        self.2.step(optim);
        self.3.step(optim);
        self.4.step(optim);
        self.5.step(optim);
        self.6.step(optim);
        self.7.step(optim);
    }
}