# Zyx

[![crates.io](https://img.shields.io/crates/v/zyx.svg)](https://crates.io/crates/zyx)
[![Documentation](https://docs.rs/zyx/badge.svg)](https://docs.rs/zyx)

Zyx is open source tensor library. It defines struct [Variable](crate::tensor::Variable) that adds gradient to any datatype.
Provided is [multidimensional array](crate::accel::cpu::Buffer) that can optionally use matrixmultiply crate for faster execution.

From user perspective, it works similar to PyTorch. Also names of functions are mostly the same,
so that you can quickly pick up this library if you are familiar with PyTorch.

## Main Ideas

We want to provide a way to do automatic differentiation and backpropagation for any datatypes, whether are those scalars, arrays, matrices, or tensors.
This library aims to be zero cost abstraction and use simple Rust syntax for this autodiff and backprop.

### Zero cost abstraction

By passing datatype into [.with_grad()](crate::tensor::IntoVariable::with_grad()) function you create Variable. Variable stores your datatype and adds gradient
to this datatype. This gradient is of the same type as your datatype. To manage access to this gradient we use [UnsafeCell](core::cell::UnsafeCell) as gradient must
be accessed from different places.

Tensor is a result of a mathematical or other operation performed on Variable. Tensor creates the graph needed for backpropagation at compile time.

All operations are executed eagerly.

**TL DR:** By zero cost abstraction we mean zero dyn, zero Rc, zero RefCell and minimal number of branches.

### Simple Rust syntax

The syntax you will be using as a user is very close to PyTorch.
Also, although the graph is created at compile time, it behaves completely dynamically (i. e. RNNs are easy). You don't need to do any graph.compile or graph.execute calls.
Tensor and Variable are both immutable.

## Features

1. PyTorch like API.
2. Commitment to support stable Rust.
3. Zero overhead approach with compile time graph.
4. Multithreaded CPU Buffer as the default accelerator.
5. Minimum of runtime errors, primarily thanks to constant shapes that are checked at compile time.

## Missing features

These features are important and in the works but not ready yet:
1. Min opration
2. Max operation
3. Convolution
This means some functions that depend on these, such as Softmax are missing as well.

In particular convolution would be much easier to implement if stable rust supported generic constant expressions.

## Examples

For examples of linear and recurrent neural networks, look at [examples directory](https://github.com/zk4x/zyx/tree/main/examples).
If you want to accelerate matrix multiplication using matrixmultiply crate, use `--features=matrimultiply`.

```rust
# #[cfg(not(feature = "matrixmultiply"))]
# {
use zyx::prelude::*;
use zyx::accel::cpu::Buffer;

let x = Buffer::<f32, Sh4<2, 3, 2, 3>>::uniform(-1., 1.).with_grad();
let y = Buffer::<f32, Sh4<2, 3, 3, 4>>::randn().with_grad();

let z = x.matmul(&y).sum::<Ax4<0, 1, 2, 3>>();

z.backward();

println!("{}", x.grad());
println!("{}", y.grad());
# }
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

## Installation

The library is available on crates.io: <https://crates.io/crates/zyx>

## Important

Not all features are yet implemented and not all tests are written.
Therefore this library can not be considered stable yet.
With that said, we don't have plans to significantly change what has already been written.

## Notes

- Performance depends on your choice of accelerator.
- We support rust primitives and [CPU Buffer](crate::accel::cpu::Buffer). We push EVERYTHING IS A TENSOR approach.
- [CPU Buffer](crate::accel::cpu::Buffer) code is under 1000 lines, so implementing custom accelerators is pretty simple without the need to rewrite the whole library.
- Only last [Tensor](crate::tensor::Tensor) in series of operations (tree root) stores references to gradients and data required for backpropagation, thus everything else is freed. You can clone [Tensors](crate::tensor::Tensor) to create multiple graphs, or use [register_hook](crate::tensor::Tensor::register_hook()) to access gradients as they pass through.
- State is stored in the type system. Functions are only implemented for those types that guarantee correct execution. For example [backward](crate::tensor::Tensor::backward()) is not implemented for types that don't have gradients. Accelerators are exception. They may or may not produce runtime errors. [CPU Buffer](crate::accel::cpu::Buffer) panics if you perform operations on [Buffer](crate::accel::cpu::Buffer)s with invalid shapes.

## How to orient yourself in the library

We would advice you to first look at module ops. There are defined all basic operations you can work with. These are implemented for accelerators. Then any accelerators which implement these operations have implemented operations in tensor::ops, these do automatic gradient calculations.
Tensors can then use optimizers to update their values using these calculated gradients.
To initialize your accelerators, you can use methods in init.rs. These are automatically implemented for all accelerators that implement ops::FromSlice.
Many other functors, such as losses and other higher level functions are in module nn.

## Future options

- **no-std support** - it is not that hard, beacuse only things blocking us are heavy use of rayon in CPU Buffer and some use of random crate.
- **GPU accelerators** - we would like to create opencl and possibly cuda implementations of buffer. GPU cache and VRAM is much harder than CPU, so we will see about the performance.

> Any opinions, issue reports, feature requests as well as code contributions are very welcome.
