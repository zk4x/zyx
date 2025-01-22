# Zyx

Zyx is a machine learning library.
Zyx feels dynamic (no static graphs), but is lazy,
waits with execution until it is explicitly asked for results.
Zyx automatically generates and compiles
optimized kernels at runtime for multiple backends.
All tensors are differentiable (that is tensors use `requires_grad=True`),
but thanks to lazyness all unnecessary memory allocations are optimized away.

## Install

Zyx is under lot of development and breaking changes are expected.
Please use latest github version.

```toml
# Only tensors (includes autograd)
zyx = { version = "*", git = "https://github.com/zk4x/zyx", package = "zyx" }
# Neural network modules - Linear, normalization layers, ...
zyx-nn = { version = "*", git = "https://github.com/zk4x/zyx", package = "zyx-nn" }
# Optimizers - SGD, Adam
zyx-optim = { version = "*", git = "https://github.com/zk4x/zyx", package = "zyx-optim" }
```

## Syntax

Zyx uses syntax similar to other ML frameworks.

```rust
use zyx::{Tensor, DType, GradientTape};

let x = Tensor::randn([11024, 1024], DType::F32)?;
let y = Tensor::uniform([8, 1024, 1024], -1f32..4f32)?;
let b = Tensor::zeros([1024], DType::F32);
let tape = GradientTape::new();
let z = &x + &y;
let z = (x.dot(&y)? + &b).gelu();
// Zyx allows for arbitrary differentiation
let b_grad = tape.gradient(&z, [&b])[0].clone().unwrap();
// Also higher order derivatives
let bb_grad = tape.gradient(&b_grad, [&b])[0].clone().unwrap();
# Ok::<(), zyx::ZyxError>(())
```

## Backends

- [x] `CUDA`
- [x] `OpenCL`
- [x] `WGPU`

Please look at file [DEVICE_CONFIG.md](https://github.com/zk4x/zyx/blob/main/zyx/DEVICE_CONFIG.md)
for detailed info how to tell Zyx which hardware it should utilize.

If you'd like to add new backend to zyx, that would be awesome!
Please read [BACKEND.md](https://github.com/zk4x/zyx/blob/main/zyx/BACKEND.md)

With [env var](https://github.com/zk4x/zyx/blob/main/zyx/ENV_VARS.md) `ZYX_DEBUG`=16 zyx prints generated kernel source code.

## Simple neural network

```rust ignore
use zyx::{Tensor, DType};
use zyx_nn::Linear;

    let mut l0 = Linear::init(3, 1024, true, DType::F32)?;
    let mut l1 = Linear::init(1024, 2, true, DType::F32)?;

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
        let tape = GradientTape::new();
        let y = l0.forward(&x)?.relu();
        let y = l1.forward(&y)?.sigmoid();
        let loss = y.mse_loss(&target)?;
        let grads = tape.gradient(&loss, l0.into_iter().chain(l1.into_iter()));
        optim.update((&mut l0).into_iter().chain((&mut l1).into_iter()), grads);
    }

    l0.into_iter().chain(l1.into_iter()).save("my_net.safetensors")?;
# Ok::<(), zyx::ZyxError>(())
```

For more details, there is a [book](https://zk4x.github.io/zyx) in works.

## Lazyness

Tensors do not get realized automatically. Realization happens only when user accesses tensors, or explicitly using `Tensor::realize` function.
```rust ignore
Tensor::realize([&x, &y]).unwrap();
```
If you do not know when to realize tensors, just do it after updating them with optimizer.
```rust ignore
sgd.update(&mut model, grads);
Tensor::realize(&model).unwrap();
```

## Error handling

In case of incorrect user input and incorrect hardware behavior, zyx returns results.
Every panic is a bug.

## Goals

1. Correctness
2. Hardware support
3. Performance

## Rust version

Zyx currently only supports latest stable version of rust. Zyx also requires std,
as it accesses files (like cuda, hip and opencl runtimes), env vars (mostly for debugging)
and also some other stuff that requires filesystem and threads.

## Operating systems

Zyx is currently tested only on linux, but should work with all *nix operating systems.
If it does not work on your system, or if you are interested in Windows support, please
create a github issue.

## Features

- **`disk_cache`** - enables saving of searched kernels to disk
- **wgsl** - enables wgsl backend

## Warning

Zyx breaks many principles of clean code. There are no dyn (virtual tables), no lifetimes on structs, generics
are only used to make user API nicer, mostly in impl Scalar or impl `IntoShape` form.
Since zyx is pretty much a compiler, internally most things are done using enums in loose data-oriented way.
If you dislike ugly code, please do not use zyx.

Zyx uses some unsafe code, mostly due to FFI access. If you find unsafe code offensive, please do not use zyx.

Zyx brings it's own runtime. It is a single global struct behind mutex.
Tensors are indices into graph stored in this runtime. If runtime wasn't
global variable, all tensors would have to keep lifetime to it. It was
tried and it poisoned the whole codebase with lifetimes. If you find global variables
offensive, please do not use zyx.

Zyx uses some code duplication. If you hate code that is not DRY, please do not use zyx.

## Dependencies

Zyx tries to use 0 dependencies, but we are not reinventing the wheel, so we use json for config
parsing, libloading to dynamically load backend dynamic library files (i.e. libcuda.so), float8 and half
for numbers and memmap2 for memory mapped disk reads and writes. All dependencies are carefully considered
and are used only if deemed absolutely necessary, that is only if they do one thing and do it well.

Optional dependencies do not have size limits, so zyx can bring lot of features with those.

## Code of conduct

Zyx has [code of conduct](CODE_OF_CONDUCT.md) that we humbly borrowed from sqlite.

## Contributing

Please check out [CONTRIBUTING.md](CONTRIBUTING.md)

## Thanks

For contributing to Zyx, finding bugs and using it in your ML models.

## License

Zyx is free software licensed under the terms of both the MIT license and the Apache License, Version 2.0.
