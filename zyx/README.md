# Zyx

Zyx is a machine learning library with two execution modes:

- **Eager-ish (default)**: Tensor operations fuse into kernels as you write them.
  When no more fusion is possible, the kernel executes immediately.
  No separate realize step needed. Ideal for one-off ops — data preprocessing,
  initialization — like NumPy with GPU acceleration.
- **Tape**: Create a `Tape` scope to enable lazy graph building, autograd, and complex
  optimizations (egraph-based fusion and device allocation search).
  Use this in training or inference loops where the same structure repeats.
  Think of it as `torch.compile`, but less strict — no miscompilation issues.

Zyx automatically generates and compiles optimized kernels at runtime for multiple backends.
All tensors are differentiable, but thanks to tape-scoped gradient tracing,
unnecessary memory allocations are optimized away.

## Install

Zyx comes with autograd and all backends built in — no feature flags needed (WGPU is optional).

```toml
# Core library (tensors, autograd, all backends)
zyx = "*"
# Neural network modules - Linear, normalization layers, ...
zyx-nn = "*"
# Optimizers - SGD, Adam
zyx-optim = "*"
```

## Syntax

Zyx uses syntax similar to other ML frameworks.
Outside a tape, ops execute eagerly (fused into kernels automatically).
Inside a `Tape`, they build a lazy graph for autograd and complex optimization:

```rust
use zyx::{DType, Tape, Tensor};

let x = Tensor::randn([8, 1024, 1024], DType::F32)?;
let y = Tensor::uniform([8, 1024, 1024], -1f32..4f32)?;
let b = Tensor::zeros([1024], DType::F32);
let tape = Tape::new()?;
let z = &x + &y;
let z = (x.dot(&y)? + &b).gelu();
let b_grad = tape.gradient(&z, [&b])[0].clone();
let bb_grad = tape.gradient(&b_grad, [&b])[0].clone();
# Ok::<(), zyx::ZyxError>(())
```

## Quick Start

No config file needed — all backends try to initialize by default.
Run `ZYX_DEBUG=1` to see which ones found hardware.

See [CONFIG.md](CONFIG.md) for device selection, autotune, and advanced options.
See [ENV_VARS.md](ENV_VARS.md) for debugging with `ZYX_DEBUG`.

## Backends

- [x] `C` — CPU backend via C codegen (clang/gcc)
- [x] `CUDA` — NVIDIA GPU acceleration
- [x] `HIP` — AMD GPU acceleration (ROCm platform)
- [x] `OpenCL` — Cross-platform (CPU via POCL, GPU via native drivers)
- [x] `WGPU` — Modern web and native GPU support via wgpu (WGSL), feature: `wgpu`
- [x] `Vulkan` — Cross-platform GPU acceleration via Vulkan (SPIR-V)

If you'd like to add new backend to zyx, that would be awesome!
Please read [BACKEND.md](https://github.com/zk4x/zyx/blob/main/zyx/BACKEND.md)

## Neural network training loop

Use a `Tape` inside your training loop for autograd and graph caching.
The tape scope detects tensors crossing the boundary (created outside, used inside)
as dynamic inputs and caches everything internal across iterations:

```rust ignore
use zyx::{Tensor, DType, Tape, ZyxError};
use zyx_nn::{Linear, Module};
use zyx_optim::SGD;

#[derive(Module)]
struct TinyNet {
    l0: Linear,
    l1: Linear,
    lr: f32,
}

impl TinyNet {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.l0.forward(x).unwrap().relu();
        self.l1.forward(x).unwrap().sigmoid()
    }
}

let mut net = TinyNet {
    l0: Linear::new(3, 1024, true, DType::F16)?,
    l1: Linear::new(1024, 2, true, DType::F16)?,
    lr: 0.01,
};

let mut optim = SGD {
    learning_rate: net.lr,
    momentum: 0.9,
    nesterov: true,
    ..Default::default()
};

let x = Tensor::from([2, 3, 1]).cast(DType::F16);
let target = Tensor::from([5, 7]);

for _ in 0..100 {
    let tape = Tape::new()?;
    let y = net.forward(&x);
    let loss = y.mse_loss(&target)?;
    let grads = tape.gradient(&loss, &net);
    optim.update(&mut net, grads);
} // Tape drop realizes all alive tensors and caches graph
  // for structurally identical iterations.
# Ok::<(), zyx::ZyxError>(())
```

For more details, there is a [book](https://zk4x.github.io/zyx) in works.

## Execution modes

### Default (eager-ish)

Outside a `Tape`, tensor operations fuse into kernels as you call them.
When fusion is not possible (device mismatch, data dependency chain break, etc.),
the pending kernel executes immediately. No explicit realize step required.
Use this for one-off work: data loading, preprocessing, model initialization — tasks
where spending time on complex optimization would be wasteful.

### Tape (lazy + autograd)

Wrap your training or inference loop in a `Tape` scope to get lazy graph building,
automatic differentiation, and aggressive optimization (egraph-based fusion comparison,
device allocation search, plan caching across structurally identical iterations). Kernel caching (compiled program reuse) is shared across both modes.

The tape detects boundary-crossing tensors (created outside the scope, referenced inside)
as dynamic inputs. Everything internal to the tape is cached by structural hash of the
graph. On cache hit, only the dynamic leaf buffers are resolved — no
graph traversal or recompilation.

Create a tape with [`Tape::new`], get gradients with [`Tape::gradient`],
and explicitly realize specific outputs with [`Tape::realize`] if needed.
On drop, all alive tensors in the scope are realized automatically.

## Error handling

In case of incorrect user input, zyx returns results. Panics are reserved for OOM and hardware issues that are not recoverable.
There are minimal exceptions to this rule, such as binary ops, which will panic if they cannot be broadcasted to a common shape.

## Goals

1. Correctness
2. Hardware support
3. Performance

### Dtype Contract

Backends advertise supported dtypes via `supported_dtypes` mask. zyx will never implicitly downcast (e.g., F32→F16)
when a backend lacks support — the operation fails explicitly. Implicit upcasting (e.g., F16→F32) is permitted
when the backend does not natively support the narrower type — correctness is guaranteed.

## Rust version

Zyx currently only supports latest stable version of rust. Zyx also requires std,
as it accesses files (like cuda, hip and opencl runtimes), env var (for debugging)
and also some other stuff that requires filesystem and threads (loading files,
multithreaded execution, worker threads, etc.).

## Operating systems

Zyx is currently tested only on linux, but should work with all \*nix operating systems.
If it does not work on your system, or if you are interested in Windows support, please
create a github issue. Basically the only difference between operating systems is specifying
proper paths to backend runtimes.

## Features

- **wgpu** - enables wgpu backend

## Warning

Zyx uses some unsafe code, due to FFI/hardware access. Zyx brings it's own runtime.
It is a single global struct behind mutex. Tensors are indices into a graph stored in this runtime.

## Dependencies

Zyx tries to use 0 dependencies, but we are not reinventing the wheel, so we use nanoserde for config
parsing, libloading to dynamically load backend dynamic library files (i.e. libcuda.so) and half
for f16 and bf16 support. All dependencies are carefully considered and are used only if deemed absolutely necessary,
that is only if they do one thing and do it well.

Currently zyx is below 30k LOC. OFC runtimes are needed for respective backends (e.g. libcuda.so).

Optional dependencies do not have size limits. This is currently only WGPU, which has millions
of lines of code with it's dependencies.

## Code of conduct

Zyx has [code of conduct](CODE_OF_CONDUCT.md) that we humbly borrowed from sqlite.

## Contributing

Please check out [CONTRIBUTING.md](CONTRIBUTING.md)

## Thank you

For contributing to Zyx, finding bugs and using it in your ML models.

## License

Zyx is free software licensed under the GNU Lesser General Public License v3.0 (`LGPLv3`)
See the LICENSE file for details.
