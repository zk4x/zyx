//! Various optimizers to update [Variables](crate::tensor::Variable).
//! 
//! This is a major change from the way PyTorch handles things.
//! [Parameters](crate::module::Parameters) are neither stored nor referenced by [Optimizers](crate::optim::Optimizer).
//!
//! We crate new network. [Parameters](crate::module::Parameters) are held in this network.
//! ```
//! # use zyx::prelude::*;
//! # use zyx::nn::Linear;
//! # use zyx::optim::SGD;
//! # use zyx::accel::cpu::Buffer;
//! let net = Linear::<Buffer<f32>, Buffer<f32>>::new(2, 4);
//! ```
//! Then we create an [optimizer](crate::optim::Optimizer). We do not pass [parameters](crate::module::Parameters) into it.
//! ```
//! # use zyx::prelude::*;
//! # use zyx::nn::Linear;
//! # use zyx::optim::SGD;
//! # use zyx::accel::cpu::Buffer;
//! # let net = Linear::<Buffer<f32>, Buffer<f32>>::new(2, 4);
//! let optim = SGD::new().with_learning_rate(0.03);
//! ```
//! When we want to update our [parameters](crate::module::Parameters) using the optimizer we call the step function.
//! Name of the function is similar to PyTorch, but instead of passing [parameters](crate::module::Parameters)
//! into [optimizer](crate::optim::Optimizer), we pass [optimizer](crate::optim::Optimizer) into [parameters](crate::module::Parameters).
//! ```ignore
//! # use zyx::prelude::*;
//! # use zyx::nn::Linear;
//! # use zyx::optim::SGD;
//! # use zyx::accel::cpu::Buffer;
//! # let net = Linear::<Buffer<f32>, Buffer<f32>>::new(2, 4);
//! # let optim = SGD::new().with_learning_rate(0.03);
//! net.parameters().step(&optim);
//! ```
//! Calling net.parameters() gives us mutable reference to network's [parameters](crate::module::Parameters).
//! If we would like to define our own updatable [parameters](crate::module::Parameters), it is easy,
//! since [parameters](crate::module::Parameters) are simply a tuple of [Variable's](crate::tensor::Variable)
//! ```
//! use zyx::prelude::*;
//! use zyx::accel::cpu::Buffer;
//! use zyx::optim::SGD;
//!
//! let mut x = Buffer::uniform((2, 3), 0., 1.).with_grad();
//! let mut y = Buffer::<f32>::randn((2, 3)).with_grad();
//! let mut z = Buffer::<f32>::eye(3).with_grad();
//!
//! let optim = SGD::new();
//!
//! (&mut x, &mut y, &mut z).step(&optim);
//! ```
//! Note that in these examples we do not populate gradients, therefore calling .step() has no actual effect.
//! 

use core::ops::{Sub, Mul};

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
    /// # Create new [SGD]
    ///
    /// This creates new [SGD] with learning rate 0.01
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
