# Zyx

Zyx is machine learning library written in Rust. This README is a quick tutorial for people
already familiar with tensors and other machine learning concepts. 
For more comprehensive documentation please see our [book](https://zk4x.github.io/zyx).

## Overview

Most ML models can be created with just tensors and operations on them.
Tensors must be stored somewhere. Zyx can use both RAM and VRAM (on the gpu) to store tensors.

Please do not directly add zyx as your dependency. Instead add one of the backends: zyx-opencl, zyx-cpu.

This is how you add gpu backend.
```shell
cargo add zyx-opencl
```
Initialize the device in your code.
```rust
let dev = zyx_opencl::device()?;
```
Now you can create tensors.
```rust
let x = dev.randn([2, 2], DType::F32);
let y = dev.tensor([[2., -3.], [4., 1.]]);
```
And do some mathematical operations on these tensors.
```rust
let z = (&x + &y).exp() - &x;
```

## Automatic differentiation 👍

Zyx allows for arbitrary differentiation. Using the previously created tensors we calculate derivative of z w.r.t. x and y.
```rust
let grads = z.backward([&x, &y]);
```
You can also calculate higher order derivatives.
```rust
let grad_xx = grads[0].unwrap().backward([&x]).unwrap();
```

## Optimizers ⚙️

Optimizers optimize something (usually minimize loss). For this they need to know what is the derivative of loss w.r.t. model's parameters.
```rust
let optimizer = zyx_optim::SGD { learning_rate: 0.01, momentum: 0.9, ..Default::default() };
let loss = (model.forward(&x) - &label).pow(2);
let grads = loss.backward(&model);
optimizer.update(&mut model, grads);
```
Optimizer updates model's parameters with gradients.

## Performance 🚀

Thanks to its execution model, Zyx should use minimum amount of RAM.
As for the backends, compiled backends (such as zyx-opencl) automatically fuse operations and create custom kernels.
Native rust CPU backend is slow and should not be used. It currently serves only as reference backend.

## Syntax

Initialize devices (you need to add appropriate crate to you project).
```rust
let opencl = zyx_opencl::default();
let cpu = zyx_cpu::default();
```
Initialize tensors.
```rust
let x = dev.tensor([[2, 3, 1], [5, 2, 8]]);
let x = dev.tensor(0..100).reshape([10, 10]);
let x = dev.randn([3, 2, 1], DType::F32);
let x = dev.uniform([3, 2, 1], -1..27);
let x = dev.zeros([3, 2, 1], DType::F32);
let x = dev.ones([3, 2, 1], DType::I32);
let x = dev.full([3, 2, 1], 4.);
let x = dev.eye(3, DType::F16);
```
Index tensors.
```rust
let x = dev.randn([2, 3, 4, 5], DType::F32);
let z = x.get((.., 2, 1..-2, ..-1));
let v: f32 = x.get((1, 2, .., -1)).item().unwrap();
```
IO operations.
```rust
model.save("model.safetensors");
model.load("model.safetensors");
```
Custom models.
```shell
cargo add zyx-nn
```
```rust
use zyx_nn::{IntoTensor, Tensor, Backend};

struct MyModel<B: Backend> {
    l0: zyx_nn::Linear<B>,
    l1: zyx_nn::Linear<B>,
}

impl<B: Backend> MyModel<B> {
    fn forward(&self, x: impl IntoTensor<B>) -> Tensor<B> {
        let x = self.l0.forward(x).relu();
        self.l1.forward(x)
    }
}
```

## Goals

1. Correctness
2. Hardware support
3. Performance

## Notes 🤔

Unlike other libraries, zyx does not require identifying which tensors will be differentiated beforehand,
that means there is no traced() or requires_grad. Everything is automatically handled with performance in mind.
Behind the scenes zyx actually waits with execution of the graph until it knows if it needs to store intermediate
tensors for backpropagation or not, and so it does not unnecessarily allocate memory.

## Contributing

See [CONTRIBUTING.md](https://github.com/zk4x/zyx/blob/main/CONTRIBUTING.md)

## Thanks ❤️

For contributing to Zyx, finding bugs and using it in your ML models.

## Licensing

Dual licensed with MIT and Apache 2.0
