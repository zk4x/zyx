# Zyx

Zyx is machine learning library written in Rust.
It's main feature is compiled backend. It automatically generates
optimized kernels for CUDA and OpenCL.
Zyx is lazy, waits with execution until it is explicitly asked for results.
All tensors are differentiable.

## Install

```shell
cargo add zyx
```

## Syntax

Zyx uses syntax similar to pytorch.

```rust no_run
use zyx::{Tensor, DType};

let x = Tensor::randn([1024, 1024], DType::F32)?;
let y = Tensor::uniform([8, 1024, 1024], -1f32..4f32)?;
let b = Tensor::zeros([1024], DType::F32);
let z = &x + &y;
let z = (x.dot(&y)? + &b).gelu();
// Zyx allows for arbitrary differentiation
let b_grad = z.backward([&b])[0].clone().unwrap();
// Also higher order derivatives
let bb_grad = b_grad.backward([&b])[0].clone().unwrap();
# Ok::<(), zyx::ZyxError>(())
```

## Backends

Zyx runs on different devices, current backends are CUDA, OpenCL and wgsl through wgpu.
HIP would be supported too, but HIPRTC is currently broken.
Using COMGR directly as a workaround is in the works..
Zyx automatically tries to utilize all available devices, but you can also manually change it by creating file backend_config.json in folder zyx in home config directory (usually ~/.config/zyx/backend_config.json).
There write [DeviceConfig] struct.

## Simple neural network

```shell
cargo add zyx;
cargo add zyx-optim;
cargo add zyx-nn;
```
```rust ignore
use zyx::{Tensor, DType};
use zyx_nn::Linear;

let l0 = Linear::new(3, 1024, DType::F32);
let l1 = Linear::new(1024, 2, DType::F32);

let x = Tensor::from([2, 3, 1]).cast(DType::F32);
let target = Tensor::from([2, 4]);

// Zyx also provides some optimizers like SGD and Adam
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
    let loss = y.mse_loss(&target)?:
    let grads = loss.backward(l0.into_iter().chain(l1.into_iter()));
    optim.update(l0.into_iter().chain(l1.into_iter()), grads);
}

l0.into_iter().chain(l1.into_iter()).save("my_net.safetensors");
# Ok::<(), zyx::ZyxError>(())
```

For more details, there is a [book](https://zk4x.github.io/zyx).

## Lazyness

Tensors do not get realized automatically. Realization happens only when user accesses tensors, or explicitly using Tensor::realize function.
```rust ignore
Tensor::realize([&x, &y]).unwrap();
```
If you do not know when to realize tensors, just do it after updating them with optimizer.
```rust ignore
sgd.update(&mut model, grads);
Tensor::realize(&model).unwrap();
```
This function might get obsolete in the future once detection of repeating graph patterns is implemented.

## Goals

1. Correctness
2. Hardware support
3. Performance

## Rust version

Zyx currently only supports latest rust stable version. Zyx also requires std,
as it accesses files (like cuda, hip and opencl runtimes), env vars (mostly for debugging)
and also some other stuff that requires filesystem and threads.

## Features

- **rand** - enables support for functions that enable random number generation
- **half** - enables support for f16 and bf16 dtypes
- **complex** - enables support for cf32 and cf64 dtypes

## Warning

Zyx breaks many principles of clean code. Clean code was tried in older versions of zyx.
Abstractions, wrappers, dyn (virtual tables), generics and lifetimes made the code hard
to reason about. Zyx now uses enums for everything and almost zero generics (only in functions, such as impl IntoShape to make API more flexible).
If you dislike ugly code, please do not use zyx.

Zyx uses some unsafe code, mostly due to FFI access. If you find unsafe code offensive,
please do not use zyx.

Zyx brings it's own runtime. It is a single global struct behind mutex.
Tensors are indices into graph stored in this runtime. If runtime wasn't
global variable, all tensors would have to keep lifetime to it. It was
tried and it poisoned the whole codebase with lifetimes. If you find global variables
offensive, please do not use zyx.

Zyx uses some code duplication. If you hate code that is not DRY, please do not use zyx.

## Code of conduct

Zyx has [code of conduct](CODE_OF_CONDUCT.md) that we humbly borrowed from sqlite.

## Contributing

Please check out [CONTRIBUTING.md](CONTRIBUTING.md)

## Thanks

For contributing to Zyx, finding bugs and using it in your ML models.

## License

Zyx is free software licensed under the terms of both the MIT license and the Apache License, Version 2.0.
