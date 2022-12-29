# Zyx

[![crates.io](https://img.shields.io/crates/v/zyx.svg)](https://crates.io/crates/zyx)
[![Documentation](https://docs.rs/zyx/badge.svg)](https://docs.rs/zyx)

Zyx is open source tensor library designed to be zero cost abstraction and to provide lot of compile-time guarantees.

From user perspective it works similar to PyTorch. Also names of functions are mostly the same, so that you can quickly pick up zyx if you are familiar with PyTorch.

## Main Ideas

We want to provide a way to do automatic differentiation and backpropagation for any datatypes, whether are those scalars, arrays, matrices, or tensors.
Zyx aims to be zero cost abstraction and use simple Rust syntax for this autodiff and backprop.

### Zero cost abstraction

By passing datatype into [.with_grad()](crate::ops::IntoVariable::with_grad()) function you create Variable. Variable stores your datatype and adds gradient
to the datatype. Gradient is of the same type as your datatype. To manage access to the gradient we use [UnsafeCell](core::cell::UnsafeCell) as gradients must
be accessed from different places and we saw significant performance improvements over [RefCell](core::cell::RefCell) in certain benchmarks.

[Tensor](crate::tensor::Tensor) is a result of a mathematical or other operation performed on Variable. Tensor creates the graph needed for backpropagation at compile time.

Tensors are immutable and all operations are executed eagerly.

**TL DR:** By zero cost abstraction we mean zero dyn, zero Rc, zero RefCell and minimal number of branches.

### Simple Rust syntax

The syntax you will be using as a user is very close to PyTorch.
Also, although the graph is created at compile time, it behaves completely dynamically (i. e. RNNs are easy). You don't need to do any graph.compile or graph.execute calls.
Buffer, Tensor and Variable are immutable.

## Features

1. PyTorch like API.
2. Zero overhead approach with compile time graph.
3. Typestate API with const [Shapes](crate::shape::Shape) and minimum runtime errors.
4. Works on both [CPU](crate::device::cpu::Device) and [GPU](crate::device::opencl::Device).

Thanks to typestate API there are **ZERO** runtime errors when running on CPU. Well, technically you can run out of RAM...

If you have supported IDE, you can look at your whole graph just by hovering over your loss variable and inspecting it's type.
The second generic parameter of [Tensor](crate::tensor::Tensor) represents the graph.

GPU acceleration uses OpenCL through [ocl](https://github.com/cogciprocate/ocl).

The current architecture makes it easy to add other accelerators should the need arise.
It is because new accelerators can be added gradually and number of required operations is low.

You can also turn custom datatypes into tensors by calling .with_grad(). They will run on CPU.

## Missing features

Convolution is not currently possible on stable rust. We need generic constant expressions to calculate the output shape.

## Examples

For examples of linear neural networks, look at [examples directory](https://github.com/zk4x/zyx/tree/main/examples).
If you want to accelerate matrix multiplication using matrixmultiply crate, use `--features=matrimultiply`.

```rust
# #[cfg(not(feature = "matrixmultiply"))]
# {
use zyx::prelude::*;
use zyx::device::cpu; // If you want this to run on GPU, just use zyx::device::opencl;
use zyx::tensor::Variable;
use zyx::shape::{Sh4, Ax4};

let device = cpu::Device::default();

let x: Variable<cpu::Buffer<'_, Sh4<2, 3, 2, 3>>> = device.uniform(-1., 1.).with_grad();
let y: Variable<cpu::Buffer<'_, Sh4<2, 3, 3, 4>>> = device.randn().with_grad();

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

Zyx is available on crates.io: <https://crates.io/crates/zyx>

## Important

Many features are not implemented and many tests are missing. We would appreciate help writing more thorough tests and feature requests so that we know where to direct our focus.
Therefore zyx can not be considered stable yet.
With that said, we don't have plans to significantly change APIs that have already been written.

## Notes

- Performance depends on your choice of device.
- We support rust primitives, [CPU](crate::device::cpu::Device) and [GPU](crate::device::opencl::Device). We push EVERYTHING IS A TENSOR approach.
- [CPU Buffer](crate::device::cpu::Buffer) code is under 1000 lines, so implementing custom devices is simple.
- Only last [Tensor](crate::tensor::Tensor) in series of operations (tree root) stores references to gradients and data required for backpropagation, thus everything else is freed. You can clone [Tensors](crate::tensor::Tensor) to create multiple graphs, or use [register_hook](crate::tensor::Tensor::register_hook()) to access gradients as they pass through.
- State is stored in the type system. Functions are only implemented for those types that guarantee correct execution. For example [backward](crate::tensor::Tensor::backward()) is not implemented for types that don't have gradients.
- FUN NOTE: If you have supported hardware, you can try using zyx with the new opencl mesa driver rusticl that was written in rust!

## How to orient yourself in zyx

We would advice you to first look at module ops. There are defined all basic operations you can work with. Any device which implements these operations automatically implementes operations in tensor::ops, these do automatic gradient calculations.
Tensors can use optimizers to update their values using calculated gradients.
Many other functors, such as losses and other higher level functions are in module nn.

## Future options

- **no-std support** - it is not that hard, beacuse only things blocking us are heavy use of rayon in CPU Buffer and some use of random crate.
- **CUDA device** - possible cuda implementations.

> Any opinions, issue reports, feature requests as well as code contributions are very welcome.
