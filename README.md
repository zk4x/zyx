# Zyx

[![crates.io](https://img.shields.io/crates/v/zyx.svg)](https://crates.io/crates/zyx)
[![Documentation](https://docs.rs/zyx/badge.svg)](https://docs.rs/zyx)
[![Build Status](https://travis-ci.org/btwiuse/zyx.svg?branch=master)](https://travis-ci.org/btwiuse/zyx)

Zyx is open source tensor library.

It defines generic traits for operations that can be performed
with tensors and generic tensor struct, that can use any datatype as buffer, provided
that this datatype implements those operations, that are called on the tensor.

That is, if you don't use some operations, there is no need to implement them
for your datatype.

## Features:
1. Tensor is generic abstraction over underlying type. That is,
   it is incredibly simple to provide your own accelerators
   and data types.

   You just need to implement those ops for your datatype, that you
   will use and don't need to care about implementing anything else.
   The library then calculates gradients for you automatically.

2. Provided is basic implementation of multidimensional buffer. It is using rayon
   for parallel computing, but some functions, notably matmul aren't currently optimized.

3. Graph of neural network is defined dynamically by user, but is statically compiled
   into the type system. Thus there is virtually zero overhead using dynamic graphs.
   backward() is just a function call that calls all the operations in reverse without creation
   of the graph at runtime.

## Example of usage

```rust
use zyx::tensor::Tensor;
use zyx::buffer::cpu;
use zyx::prelude::*; // includes traits for operations

// Create Tensor using many methods, for more, look at zyx::tensor::init.rs
let x = Tensor::from([1., 2., 3.]);
let x = Tensor::<cpu::Buffer<f32>>::randn(&[2, 3, 1, 4]);
let x = Tensor::<cpu::Buffer<f32>>::uniform(&[2, 3, 1, 4], 0., 1.);

// Add gradient to tensor
let x = x.with_grad();

// You can apply standard operations. For list of operations, look at documentation
// for tensor module.
let y = x.relu();

// Library uses Rc's internally for tracking graphs, therefore you can freely
// clone and pass Tensors by value.

let x = Tensor::uniform(&[2, 3, 1, 4], 0., 1.).with_grad();

for _ in 0..10 {
    let y = x.tanh();
}
```

As for neural networks, there is Module trait:

```rust
pub trait Module<Input> {
    type Output;
    fn forward(&self, x: Input) -> Self::Output;
}
```

Anything that implements this trait (many of those structs can be found in zyx::nn),
can be used in two ways.

```rust
use zyx::nn;

let tanh_fn = nn::Tanh;
let x = Tensor::from([1., 2.]);

// Either like this
let y = tanh_fn.forward(x.clone());
// Or like this
let y = x.clone().apply(&tanh_fn);

// The second way is especially useful if you want to apply multiple operations in sequence
let y = x.clone().apply(&nn::Tanh).apply(&nn::ReLU).apply(&nn::Exp).apply(&nn::Ln);

// Model can be created using tuples
let network = (nn::Tanh, nn::ReLU, nn::Exp);
let y = x.clone().apply(&network);

// These can be chained
let net1 = (nn::Tanh, nn::ReLU);
let net2 = (nn::Exp, nn::ReLU);
let network = (net1, net2);

let y = network.forward(x);
```

## Important

Currently this library requires nightly compiler, for the following feature:

#![feature(type_alias_impl_trait)]

Not all features are yet implemented and not all tests are written.
Therefore this library can not be yet considered stable.
With that said, the most important stuff is implemented and working
as inteded, so you can build for example linear and recurrent models.
Convolution is in the works.

## Installation

The library is available on crates.io: https://crates.io/crates/zyx
