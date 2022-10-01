//! # Zyx
//! 
//! [![crates.io](https://img.shields.io/crates/v/zyx.svg)](https://crates.io/crates/zyx)
//! [![Documentation](https://docs.rs/zyx/badge.svg)](https://docs.rs/zyx)
//! 
//! Zyx is open source tensor library.
//! 
//! It defines generic traits for operations that can be performed
//! with tensors and generic tensor struct, that can use any custom accelerator as buffer, provided
//! that this accelerator implements those operations, that are called on the tensor.
//! 
//! That is, if you don't use some operations, there is no need to implement them for your accelerator.
//! 
//! ## Features
//! 
//! 1. Tensor is generic abstraction over underlying type. That is,
//!    it is incredibly simple to provide your own accelerators
//!    and data types. You just need to implement those ops for your datatype, that you
//!    will use and don't need to care about implementing anything else.
//!    The library then calculates gradients for you automatically.
//! 
//! 2. Provided is basic implementation of multidimensional buffer. It is using rayon
//!    for parallel computing, but some functions, notably matmul aren't currently optimized.
//! 
//! 3. Graph of neural network is defined dynamically by user, but is statically compiled
//!    into the type system. Thus there is virtually zero overhead using dynamic graphs.
//!    backward() is just a function call that calls all the operations in reverse without creation
//!    of the graph at runtime. No dyn keyword used, Vecs of operations are not created.
//!    Tensors are basically zero cost abstractions over underlying accelerator. At compile time
//!    they just remeber the operation used and call appropriate functions to calculate the derivatives.
//! 
//! 4. Accelerator code is just 500 lines in one file. Implementing custom accelerators should be simple.
//!    There is no need to make changes across the whole library. Just add a file and you have added an accelerator.
//! 
//! ## Example of usage
//! 
//! For examples of linear and recurrent neural networks, look at examples directory.
//! 
//! ```rust
//! use zyx::prelude::*;
//! use zyx::buffer::cpu;
//! use zyx::tensor::Tensor;
//! 
//! let x = Tensor::uniform(&[20, 30], -1., 1.).with_grad();
//! let y = Tensor::<cpu::Buffer<f32>>::randn(&[30, 15]).with_grad();
//! 
//! x.matmul(&y).sum(&[]).backward();
//! 
//! println!("{}", x.grad().borrow());
//! println!("{}", y.grad().borrow());
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
//! ## Thank you
//! 
//! To all the users and contributors.
//! Without you, this library would have no reason to exist.
//! Any opinions, issue reports, feature requests as well as code contributions are very welcome.
//! 
//! ## How to orient yourself in the library
//! 
//! The library contains following modules. The order of these modules is from most to least important.
//! 
//! ### Ops
//! 
//! This module contains definitions of operations that can be performed on tensors. The operations should be performable with basic datatypes (those implementations
//! are in this module), all custom accelerators in buffer module and all tensors (implemented in tensor::ops module).
//! 
//! ### Tensor
//! 
//! The primary component of the library is tensor module. It contains definition of Tensor, TensorGrad and TensorFunc.
//! Tensor is just an Rc over storage buffer S, so it does not require gradient.
//! By calling .with_grad() on Tensor, you get TensorGrad, which stores it's gradient.
//! These two are called leaf tensors and they are created explicitly by the user.
//! 
//! TensorFunc is a tensor, that is a result of some computation where at least one of the operands is TensorGrad.
//! TensorFunc is not a leaf tensor, TensorFunc does not store it's gradient. This simplifies calculations and saves memory.
//! TensorFunc stores references to TensorGrad's gradients and Rc pointers to some data buffers used during gradient
//! calculation.
//! 
//! tensor::self contains tensor definitions, getters and setters for tensors.
//! tensor::ops contains tensor implementations of operations defined in ops module.
//! tensor::init contains initialization methods for tensors.
//! 
//! ### Buffer
//! 
//! Buffer module contains implementations of accelerators. The default accelerator is cpu::Buffer. This accelerators is complete, but not optimized.
//! You can define your own accelerators by simply creating a new module in this buffer directory.
//! 
//! These three modules represent the foundation upon which this library stands. There should be minimal to none API changes to ops and tensor modules.
//! As for the buffer module, new accelerators should be added and existing accelerators should become faster, however removal of existing features is not going to be accepted.
//! 
//! ### Optim
//! 
//! Optim module has Optimizer trait. Here are all the different optimizers like SGD.
//! 
//! ### Module
//! 
//! This module contains traits needed for simple definition of neural network models.
//! Basically all functions and layers and models should have module.forward(input) function and this module also provides input.apply(module) function.
//! It is deemed usefull for the user to have access to both standard module.forward(input) type of API and API with monads.
//! 
//! ### Shape
//! 
//! This module defines Shape and Dims traits. These are implemented for &[usize] and &[i32] respectively. Shape stores the size of tensor's dimensions
//! while Dims stores dimension's order, that can also be negative (-1 is last dimension). Dims is used as input into functions as Permute or Sum, when
//! we need to define along which dimensions we want to perform these operations.
//! 
//! ### nn
//! 
//! This is module, which is expected to get most stuff added. This module will contain functors, layers, models, cells, simply averything that can have .forward(input) function.
//! 

pub mod tensor;
pub mod buffer;
pub mod ops;
pub mod shape;
pub mod module;
pub mod nn;
pub mod optim;
pub mod prelude;

#[cfg(test)]
mod tests;

// TODO: saving of models and tensors
