//! Various optimizers to update [Variables](crate::tensor::Variable).
//!
//! This is a major change from the way PyTorch handles things.
//! [Parameters](crate::nn::parameters::Parameters) are neither stored nor referenced by optimizers.
//!
//! We crate new network. [Parameters](crate::nn::parameters::Parameters) are held in this network.
//! ```
//! # use zyx::prelude::*;
//! # use zyx::nn::Linear;
//! # use zyx::optim::SGD;
//! # use zyx::device::cpu;
//! # let mut device = cpu::Device::default();
//! let net = Linear::<2, 4>::new(&mut device);
//! ```
//! Then we create an optimizer. We do not pass [parameters](crate::nn::parameters::Parameters) into it.
//! ```
//! # use zyx::prelude::*;
//! # use zyx::nn::Linear;
//! # use zyx::optim::SGD;
//! # use zyx::device::cpu;
//! # let mut device = cpu::Device::default();
//! # let mut net = Linear::<2, 4>::new(&mut device);
//! let optim = SGD::new().with_learning_rate(0.03);
//! ```
//! When we want to update our [parameters](crate::nn::parameters::Parameters) using the optimizer we call the step function.
//! Name of the function is similar to PyTorch, but instead of passing [parameters](crate::nn::parameters::Parameters)
//! into optimizer, we pass optimizer into [parameters](crate::nn::parameters::Parameters).
//! ```
//! # use zyx::prelude::*;
//! # use zyx::nn::Linear;
//! # use zyx::device::cpu;
//! # use zyx::optim::SGD;
//! # let mut device = cpu::Device::default();
//! # let mut net = Linear::<2, 4>::new(&mut device);
//! # let optim = SGD::new().with_learning_rate(0.03);
//! net.parameters().step(&optim);
//! ```
//! Calling net.parameters() gives us mutable reference to network's [parameters](crate::nn::parameters::Parameters).
//! If we would like to define our own updatable [parameters](crate::nn::parameters::Parameters), it is easy,
//! since [parameters](crate::nn::parameters::Parameters) is simply a tuple of [Variables](crate::tensor::Variable)
//! ```
//! use zyx::prelude::*;
//! use zyx::device::cpu::{self, Buffer};
//! use zyx::shape::Sh2;
//! use zyx::tensor::Variable;
//! use zyx::optim::SGD;
//!
//! let mut device = cpu::Device::default();
//!
//! let mut x: Variable<Buffer<'_, Sh2<2, 3>, f64>> = device.uniform(0., 1.).with_grad();
//! let mut y: Variable<Buffer<'_, Sh2<2, 3>, i32>> = device.randn().with_grad();
//! let mut z = device.buffer([2, 3, 4]).with_grad();
//!
//! let optim = SGD::new();
//!
//! (&mut x, &mut y, &mut z).step(&optim);
//! ```
//! Note that in these examples we do not populate gradients, therefore calling .step() has no actual effect.
//!

use core::ops::{Mul, Sub};

/// # Stochastic gradient descent optimizer
///
/// Updates [parameter's](crate::nn::parameters::Parameters) data using following function:
/// ```txt
/// x.data = x.data - x.grad * learning_rate;
/// ```
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
