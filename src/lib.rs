//! # Zyx
//! 
//! [![crates.io](https://img.shields.io/crates/v/zyx.svg)](https://crates.io/crates/zyx)
//! [![Documentation](https://docs.rs/zyx/badge.svg)](https://docs.rs/zyx)
//! [![Repository](https://github.com/zk4x/zyx/badge.svg)](https://github.com/zk4x/zyx)
//! 
//! Zyx is open source tensor library. It defines struct Variable that adds gradient to any datatype.
//! Provided is multidimensional array accel::cpu::Buffer that can optionally use matrixmultiply
//! crate for faster execution.
//! 
//! ## Features
//! 
//! 1. Any datatype is supported, including rust primitives, accel::cpu::Buffer as well as any custom
//!    datatype that you provide. Basically everything is a tensor.
//! 
//! 2. Graph is fully dynamic from user perspective, but is compiled statically. Only last Tensor
//!    in series of operations (tree root) stores references to gradients and data required for backpropagation,
//!    thus everything else is freed. You can clone Tensors to create multiple graphs, or use register_hook to access
//!    gradients as they pass through.
//! 
//! 3. Cpu accelerator code is just 600 lines, so implementing custom accelerators is pretty simple without the need
//!    to rewrite the whole library.
//! 
//! 4. No dyn, no Rc. Performance depends on your choice of accelerator, only overhead from tensors is gradients stored in RefCells.
//! 
//! ## Example of usage
//! 
//! For examples of linear and recurrent neural networks, look at examples directory.
//! 
//! ```rust
//! use zyx::prelude::*;
//! use zyx::accel::cpu::Buffer;
//! 
//! let x = Buffer::uniform(&[2, 3, 2, 3], -1., 1.).with_grad();
//! let y = Buffer::<f32>::randn(&[2, 3, 3, 4]).with_grad();
//! 
//! x.matmul(&y).sum(&[]).backward();
//! 
//! println!("{}", x.grad());
//! println!("{}", y.grad());
//! ```
//! 
//! Want to use scalars? Just give them gradients!
//! 
//! ```rust
//! use zyx::prelude::*;
//! 
//! let x = 3_f32.with_grad();
//! let y = 5_f32;
//! (&x + y).relu().backward();
//! println!("{}", x.grad());
//! ```
//! 
//! ## Installation
//! 
//! The library is available on crates.io: https://crates.io/crates/zyx
//! 
//! ## Important
//! 
//! Not all features are yet implemented and not all tests are written.
//! Therefore this library can not be considered stable yet.
//! Convolution is in the works.
//! With that said, the most important stuff is implemented and working as intended.
//! 
//! ## How to orient yourself in the library
//! 
//! The library contains following modules. The order of these modules is from most to least important.
//! 
//! ### ops
//! 
//! This module contains definitions of operations that can be performed on tensors. The operations should be performable with basic datatypes (those implementations
//! are in this module), all custom accelerators in buffer module and all tensors (implemented in tensor::ops module).
//! 
//! ### tensor
//! 
//! The primary component of the library is tensor module. It contains definition of Variable and Tensor.
//! By calling .with_grad() on any datatype, you get Variable, which stores it's gradient.
//! Variable is leaf tensor.
//! 
//! Tensor is a tensor, that is a result of some computation where at least one of the operands is TensorGrad.
//! Tensor is not a leaf tensor, Tensor does not store it's gradient. This simplifies calculations and saves memory.
//! Tensor stores references to Variable's gradients and some data buffers used during gradient calculation.
//! 
//! tensor::self contains tensor definitions, getters and setters for tensors.
//! tensor::ops contains tensor implementations of operations defined in ops module.
//! tensor::init contains initialization methods for tensors.
//! 
//! ### accel
//! 
//! Buffer module contains implementations of accelerators. The default accelerator is cpu::Buffer. This accelerators is complete, but not optimized.
//! You can define your own accelerators by simply creating a new module in this buffer directory.
//! 
//! These three modules represent the foundation upon which this library stands. There should be minimal to none API changes to ops and tensor modules.
//! As for the buffer module, new accelerators should be added and existing accelerators should become faster, however removal of existing features is not going to be accepted.
//! 
//! ### optim
//! 
//! Optim module has Optimizer trait. Here are all the different optimizers like SGD.
//! 
//! ### module
//! 
//! This module contains traits needed for simple definition of neural network models.
//! Basically all functions and layers and models should have module.forward(input) function and this module also provides input.apply(module) function.
//! It is deemed usefull for the user to have access to both standard module.forward(input) type of API and API with monads.
//! 
//! ### shape
//! 
//! This module defines Shape and Dims traits. These are implemented for &[usize] and &[i32] respectively. Shape stores the size of tensor's dimensions
//! while Dims stores dimension's order, that can also be negative (-1 is last dimension). Dims is used as input into functions as Permute or Sum, when
//! we need to define along which dimensions we want to perform these operations.
//! 
//! ### nn
//! 
//! This is module, which is expected to get most stuff added. This module will contain functors, layers, models, cells, simply averything that can have .forward(input) function.
//! 
//! ## init
//! 
//! This module contains initialization methods for tensors.
//! 

pub mod tensor;
pub mod accel;
pub mod ops;
pub mod shape;
pub mod module;
pub mod nn;
pub mod optim;
pub mod prelude;
pub mod init;

#[cfg(test)]
mod tests;

// TODO: saving of models and buffers (probably via .to_vec() and .shape(), and ::from_vec())
// power and convolution ops for tensor
// opencl buffer
// lazy Buffer (both opencl and cpu (I know, opencl can run on cpu as well))
