# Zyx

Machine learning library written in rust.

Zyx is semi small and has only few dependencies (i. e. zyx was build from scratch in rust and OpenCL).

## Tensors

[Tensor](crate::tensor::Tensor) is the basic unit of zyx. Tensors are immutable. Context manages all tensors and connects them with [backends](#backends).

Tensors are just reference counted fat pointers. Feel free to clone them.
```rust
# use zyx::context::Context;
let mut ctx = Context::new();
let x = ctx.tensor([[2, 4, 3], [4, 2, 3]]);
let y = x.clone();
```

## Automatic differentation/Backpropagation

Every operation is traced automatically. Thus you can calculate derivative of any tensor with respect to any other tensor. There is no need for gradient tape and you don't need to set requires_grad. See [automatic differentiation](Automatic differentiation.md) for more details.
```rust
# #[cfg(feature = "rand")] {
# use zyx::context::Context;
# use zyx::dtype::DType;
let mut ctx = Context::new();
let x = ctx.randn((2, 4));
let w1 = ctx.randn((4, 4));
let w2 = ctx.randn((4, 3));
let out = x.dot(w1).tanh().dot(w2);
let y = ctx.tensor([2, 1, 4]).cast(DType::F32);
let loss = (out - y).pow(2.);
# }
```
Following function calculates gradients for w1 and w2 (i. e. derivative of loss w.r.t w1 and w.r.t w2).
```rust
# #[cfg(feature = "rand")] {
# use zyx::context::Context;
# use zyx::dtype::DType;
# let mut ctx = Context::new();
# let x = ctx.randn((2, 4));
# let mut w1 = ctx.randn((4, 4));
# let mut w2 = ctx.randn((4, 3));
# let out = x.dot(&w1).tanh().dot(&w2);
# let y = ctx.tensor([2, 1, 4]).cast(DType::F32);
# let loss = (out - y).pow(2.);
loss.backward([&mut w1, &mut w2]);
# }
```

## Graph realization

Neural networks are directed acyclic graphs of many tensors. Thus one of the big tradeoffs of modern machine learning libraries is when and how to do calculations. There is no clear winner here, each method has it's pros and cons.

Zyx uses fully dynamic graph. Realization of tensors happens only when you call [realize](crate::tensor::Tensor::realize). This means tensors are evaluated lazily.
```rust
# #[cfg(all(feature = "rand", feature = "opencl"))] {
# use zyx::context::Context;
let mut ctx = Context::opencl().unwrap();
let x = ctx.randn((256, 1024));
let w1 = ctx.randn((1024, 1024));
let mut z = x.dot(w1);
z.realize().unwrap();
# }
```
This enables certain optimizations, but you need to call realize during training loop.

## Neural networks

Implementing [module](crate::nn::Module) allows for custom high level constructs.
```rust
# #[cfg(feature = "opencl")] {
# use zyx::context::Context;
# use zyx::tensor::Tensor;
# use zyx::nn::{Module, Linear};
# use zyx::optim::SGD;
# use zyx::parameters::Parameters;
struct TinyNet {
    l0: Linear,
    l1: Linear,
}

impl Module for TinyNet {
    fn forward(&self, x: &Tensor) -> Tensor {
        self.l1.forward(&self.l0.forward(x).tanh())
    }

    fn parameters(&mut self) -> Parameters {
        self.l0.parameters().join(self.l1.parameters())
    }
}

let mut ctx = Context::opencl().unwrap();
let mut net = TinyNet {
    l0: ctx.linear(12, 1024),
    l1: ctx.linear(1024, 53),
};
let mut opt = SGD::new().set_lr(0.01);
let x = ctx.randn((32, 12)).set_label("x");
let y = ctx.randn(53);
for _ in 0..10 {
    let out = net.forward(&x);
    let loss = out.mse(&y).sum(());
    out.backward(net.parameters());
    // optimizer.step realizes parameters and zeros gradients
    opt.step(net.parameters()).unwrap();
}
# }
```

## Goals

These are general directions for further development of Zyx.
1. Correctness
2. Performance
3. Hardware support

## Visualization

Networks can be visualized using dot language.
```rust ignore
let graph = ctx.dot_graph();
std::fs::File::create("graph.dot").unwrap().write_all(graph.as_bytes()).unwrap();
```
tiny_net example forward pass:

![Tiny net forward pass image](https://github.com/zk4x/zyx/blob/main/examples/tiny_net_graph.png)

## Backends

Zyx has two backends, CPU and OpenCL (all OpenCL versions).

Backends are easy to add. Only few ops are needed and automatic differentiation works with all backends. However making them fast is very hard.

## Performance

Here is comparison of Zyx, tinygrad, dfdx and PyTorch running TinyNet example (forward + backward). This is **cherry picked** benchmark. Take it with grain of salt.

Table shows running time in seconds. PyTorch uses compiled model. Tinygrad runs OpenCL backend for GPU and numpy for CPU.

We couldn't get dfdx and PyTorch working with given gpu.

| Device         |   Zyx |  tinygrad |  dfdx |  PyTorch |
| -------------- | ----- | --------- | ----- | -------- |
| GPU RX 550     |  4.58 |      5.51 |     - |        - |
| CPU i5 Haswell | 14.39 |     11.03 |  7.79 |     4.74 |

As you can see, Zyx is ok on the GPU, but needs to be further optimized for the CPU. PyTorch looks really impressive here, given that it can only utilize CPU.

## Load/Save

Zyx works with .safetensors format from huggingface. Enable io feature to have this work.
```rust ignore
net.parameters().save("model.safetensors");
// or shorter
net.save("model.safetensors");
```
Loading is not much more complex.
```rust ignore
net.load("model.safetensors");
```

## No-std

Zyx is no-std library, but alloc is required.

## Features

- opencl - enables OpenCL backend
- cpu - enables multithreading, faster cpu operations and std
- io - enables file operations and std
- debug1 - enables printing of debug information during runtime and std

## Multiple GPUs

Zyx should work with multiple GPUs within single OpenCL platform, but this was not tested.

## Syntax

Zyx has syntax similar to other ML libraries (i. e. PyTorch).

|                 | Zyx                                              | PyTorch                                                     |
| --------------- | ------------------------------------------------ | ----------------------------------------------------------- |
| random tensor   | `ctx.randn((5, 3))`                              | `torch.randn((5, 3))`                                       |
| zeros tensor    | `ctx.zeros((5, 3))`                              | `torch.zeros((5, 3))`                                       |
| uniform tensor  | `ctx.uniform((4, 6), 0.0..4.0)`                  | `torch.zero((4, 6)).uniform_(0, 4)`                         |
| matmul          | `let z = x.dot(y);`                              | `z = x @ y`                                                 |
| tanh            | `let y = x.tanh();`                              | `y = x.tanh()`                                              |
| binary ops      | `let z = x + 2;`                                 | `z = x + 2`                                                 |
| dtypes          | `let z = x.cast(DType::I32);`                    | `z = x.type(torch.int32)`                                   |
| saving          | `net.save("net.safetensors");`                   | `torch.save(net.state_dict(), "net.pt")`                    |
| loading         | `net.load("net.safetensors");`                   | `net.load_state_dict(torch.load("net.pt"))`                 |
| backpropagation | `let y = x.exp();`<br>`y.backward(&mut x);`      | `x.requires_grad = True`<br>`y = x.exp()`<br>`y.backward()` |
| optimizers      | `let opt = SGD::new();`<br>`opt.step(&mut net);` | `opt = SGD(net.parameters())`<br>`opt.step()`               |

## Missing features

Zyx is very much *experimental* software. Some notable missing features are convolutions and padding.

## Contributing

Any contributions are welcome. If interested in simple contributions, improving documentation and adding examples is great. For those interested in adding modules, there is folder nn, where you can leverage tensor's existing ops. And if you are interested in low level programming, improving performance of opencl kernels is the most difficult option.

## Bugs

Please report any correctness and performance bugs. Especially please report incorrect/insufficient tests.

## Thanks

Libraries that are used as dependencies for Zyx deserve special thanks, because Zyx would not be possible without them.

We would also like to thank users of Zyx for providing continuous interest and showing that there is a demand for this library.

## License

Zyx is free software licensed under the terms of both the [MIT license](<http://opensource.org/licenses/MIT>) and the [Apache License, Version 2.0](<http://www.apache.org/licenses/LICENSE-2.0>).
For OpenCL licensing see it's [website](<https://www.khronos.org/opencl/>).
