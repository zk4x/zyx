# Zyx

[![crates.io](https://img.shields.io/crates/v/zyx.svg)](https://crates.io/crates/zyx)
[![Documentation](https://docs.rs/zyx/badge.svg)](https://docs.rs/zyx)

Zyx is open source tensor library. It defines struct [Variable](crate::tensor::Variable) that adds gradient to any datatype.
Provided is [multidimensional array](crate::accel::cpu::Buffer) that can optionally use matrixmultiply
crate for faster execution.

From user perspective, it works similar to PyTorch. Also names of functions are mostly the same,
so that you can quickly pick up this library if you are familiar with PyTorch.

## Features

1. Any datatype is supported, including rust primitives, [CPU Buffer](crate::accel::cpu::Buffer), preliminary support for ndarray as well as any custom datatype that you provide. Basically everything is a tensor.
2. No dyn, no Rc, no RefCell, so Variable is true zero cost abstraction. Performance depends on your choice of accelerator. However gradients need to be updated from multiple places, so for that we use UnsafeCell. Since only unsafe mutation of gradients happens while calling backward, it is pretty easy to manually assure safety.
3. [CPU Buffer](crate::accel::cpu::Buffer) code is under 1000 lines, so implementing custom accelerators is pretty simple without the need to rewrite the whole library.
4. Graph is fully dynamic from user perspective, but is compiled statically. Only last [Tensor](crate::tensor::Tensor) in series of operations (tree root) stores references to gradients and data required for backpropagation, thus everything else is freed. You can clone [Tensors](crate::tensor::Tensor) to create multiple graphs, or use [register_hook](crate::tensor::Tensor::register_hook()) to access gradients as they pass through.
5. There are no runtime errors, not even Results that need to be handled. State is stored in the type system. Functions are only implemented for those types that guarantee correct execution. For example [backward](crate::tensor::Tensor::backward()) is not implemented for types that don't have gradients. Accelerators are exception. They may or may not produce runtime errors. [CPU Buffer](crate::accel::cpu::Buffer) panics if you perform operations on [Buffer](crate::accel::cpu::Buffer)s with invalid shapes.
6. Tensors are immutable from user perspective. This greatly simplifies everything, especially correct calculation of gradients.As for the performance, cloning is used when [Variable](crate::tensor::Variable) is passed by reference. How expensive this clone is depends on accelerator. [CPU Buffer](crate::accel::cpu::Buffer) uses [Arc](core::sync::Arc) to avoid copies and make operations inplace if possible.

## Example of usage

For examples of linear and recurrent neural networks, look at examples directory.
If you want to accelerate matrix multiplication using matrixmultiply crate, use `--features=matrimultiply`.

```rust
use zyx::prelude::*;
use zyx::accel::cpu::Buffer;

let x = Buffer::uniform((2, 3, 2, 3), -1., 1.).with_grad();
let y = Buffer::<f32>::randn((2, 3, 3, 4)).with_grad();

let z = x.matmul(&y).sum(());
z.backward();

println!("{}", x.grad());
println!("{}", y.grad());
```

Want to use scalars? Just give them gradients!

```rust
use zyx::prelude::*;

let x = 3_f32.with_grad();
let y = 5.;
let z = (&x + y).relu();
z.backward();
println!("{}", x.grad());
```

Want to use ndarray? Just give it gradients and use `--features=ndarray`!
Note that reduce and movement ops are not yet implemented for ndarray. Support for binary operations is limited.

```rust
# #[cfg(feature = "ndarray")]
# {
use zyx::prelude::*;
use ndarray::array;

let x = array![[2., 3., 4.], [3., 4., 2.]];
let x = x.with_grad();
x.exp().backward();
println!("{}", x.grad());
# }
```

## Installation

The library is available on crates.io: <https://crates.io/crates/zyx>

## Important

Not all features are yet implemented and not all tests are written.
Therefore this library can not be considered stable yet, but we are getting closer to stable release as the main API is not gonna change much anymore.
Preliminary support for convolution is done.
Most stuff is implemented and working as intended.

## How to orient yourself in the library

This is the order of modules from most to least important.
1. ops
2. tensor
3. accel
4. optim
5. module
6. shape
7. nn
8. init

## Thank you

To all the users and contributors. Without you, this library would have no reason to exist.

Any opinions, issue reports, feature requests as well as code contributions are very welcome.
