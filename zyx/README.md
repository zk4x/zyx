# Zyx

Zyx is machine learning library written in Rust.
It's main feature is compiled backend. It automatically generates
optimized kernels for CUDA, OpenCL and WGPU.
Zyx is lazy, waits with execution until it is explicitly asked for results
or the graph of tensor operations gets too large.
Zyx automatically finds repetition in your training loops, caches
that part of the graph and uses constant folding on values that are evaluated
only once. Since zyx stores the whole graph and automatically optimizes it,
all tensors are differentiable. In pytorch this is equivalent to all tensors
being set to requires_grad=True.

## Syntax

Zyx uses syntax similar to pytorch.

```rust
use zyx::{Tensor, DType};

let x = Tensor::randn([1024, 1024], DType::BF16);
let y = Tensor::uniform([8, 1024, 1024], -1f32..4f32);
let b = Tensor::zeros([1024], DType::F16);
let z = &x + &y;
let z = (x.dot(&y), + b).gelu();
let b_grad = z.backward([&b]);
```

## Backends

Zyx runs on different devices, current backends are CUDA, OpenCL and CPU.
Zyx automatically tries to select the most performant available device, but you can also manually change it
by creating file backend_config.ron in folder zyx in home config directory (usually ~/.config/zyx/backend_config.ron).
There write [BackendConfig] struct.

## Simple neural network

```shell
cargo add zyx;
cargo add zyx-optim;
cargo add zyx-nn;
```
```rust
use zyx::{Tensor, DType};
use zyx_nn::Linear;

Tensor::set_default_device_best();

let l0 = Linear::new(3, 1024, DType::F16);
let l1 = Linear::new(1024, 2, DType::F16);

let x = Tensor::from([2, 3, 1]).cast(DType::F16);
let target = Tensor::from([2, 4]);

let mut optim = zyx_optim::SGD {
    learning_rate: 0.01,
    momentum: 0.9,
    nesterov: true,
    ..Default::default()
};

let train_steps = 100;
for _ in 0..train_steps {
    let y = l0.forward(&x).relu();
    let y = l1.forward(&y).sigmoid();
    let loss = y.mse_loss(&target):
    let grads = loss.backward(l0.into_iter().chain(l1.into_iter()));
    optim.update(l0.into_iter().chain(l1.into_iter()), grads);
}

l0.into_iter().chain(l1.into_iter()).save("my_net.safetensors");
```

For more details, there is a [book](https://zk4x.github.io/zyx).

# No-std

Zyx does not use rust's std, it only uses alloc.

## Features

debug1 - enables printing of debug information during runtime

## Warning

Zyx breaks many principles of clean code. Clean code was tried in older versions of zyx.
Abstractions, wrappers, dyn (virtual tables), generics and lifetimes made the code hard
to reason about. Zyx now uses enums for everything and almost zero generics (only in functions,
such as impl IntoShape to make API more flexible). If you dislike ugly code, please do not use zyx.

Zyx uses some unsafe code, mostly due to FFI access. If you find unsafe code offensive,
please do not use zyx.

Zyx brings it's own runtime. It is a single global struct behind mutex.
Tensor are indices into graph stored in this runtime. If runtime wasn't
global variable, all tensors would have to keep lifetime to it. It was
tried and it poisoned the whole codebase with lifetimes. If you find global variables
offensive, please do not use zyx.

Zyx uses some code duplication. If you hate code that is not DRY, please do not use zyx.

Zyx has code of conduct that we humbly borrowed from sqlite.

## Contributing

Please read CONTRIBUTING.md

## License

Zyx is free software licensed under the terms of both the MIT license and the Apache License, Version 2.0.
