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

mod sgd;
mod numpy_rw;

pub use numpy_rw::NumpyRW;
pub use sgd::{SGD, SGDStep};
