# Zyx

Zyx is a machine learning library that runs on your hardware.


## Install

Zyx comes with autograd and all backends built in — no feature flags needed (with the exception of WGPU).

```toml
# Core library (tensors, autograd, all backends)
zyx = "*"
# Neural network modules - Linear, normalization layers, recurrent nets, transformers ...
zyx-nn = "*"
# Optimizers - SGD, Adam, RMSProp ...
zyx-optim = "*"
```


## Syntax

Zyx uses syntax similar to other ML frameworks.
Outside a tape, ops eagerly fuse into kernels based on heuristics.
Inside a `Tape`, autograd is provided and tape also provides more optimizations.
The eager mode is for typical one off kernels where spending time optimizating would be
a waste of time, while Tape is for blocks of repeated computations (training loop, inference loop).

```rust
use zyx::{DType, Tape, Tensor};

let x = Tensor::randn([8, 64, 64], DType::F32)?;
let y = Tensor::uniform([8, 64, 64], -1f32..4f32)?;
let b = Tensor::zeros([64], DType::F32);
let tape = Tape::new([&b])?;
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
- [x] `CUDA` — NVIDIA GPU acceleration, via PTX codegen
- [x] `OpenCL`
- [x] `Vulkan` — via SPIR-V codegen
- [x] `WGPU` - via SPIR-V codegen, feature: `wgpu`
- [ ] `tenstorrent` - Preliminary support, does not pass full test suite yet, feature `tenstorrent`

If you'd like to add new backend to zyx, that would be awesome!
Please read [ADDING_BACKENDS.md](https://github.com/zk4x/zyx/blob/main/ADDING_BACKENDS.md)


## Error handling

In case of incorrect user input, zyx returns results. Panics are reserved for OOM and hardware issues that are
not recoverable. There are minimal exceptions to this rule, such as binary ops, which will panic if they cannot
be broadcasted to a common shape.


### DTypes

Backends advertise supported dtypes via `supported_dtypes` mask. zyx does not implicitly downcast (e.g., F32→F16)
when a backend lacks support — the operation fails explicitly. Implicit upcasting (e.g., F16→F32) is sometimes applied
when the backend does not natively support the narrower type — correctness is guaranteed, not performance.


## Rust version

Minimum supported rust version is 1.88, works with any newer stable version too. Zyx also requires std,
as it accesses files (like cuda, hip and opencl runtimes), env vars (for debugging) and also some other
stuff that requires filesystem and threads (loading files, multithreaded execution, worker threads, etc.).


## Operating systems

Zyx is currently tested only on linux, but should work with all \*nix operating systems.
If it does not work on your system, or if you are interested in Windows support, please
create a github issue. Basically the only difference between operating systems is specifying
proper paths to backend runtimes (e.g. libcuda.so).


## Features

- **wgpu** - enables wgpu backend
- **tenstorrent** - enables tenstorrent backend


## Dependencies

Zyx tries to use 0 dependencies, but we are not reinventing the wheel, so we use nanoserde for config
parsing and libloading to dynamically load backend dynamic library files.
All dependencies are carefully considered and are used only if deemed absolutely necessary.

Currently zyx is below 30k LOC. Runtimes are needed for respective backends (e.g. libcuda.so)
and also hardware drivers.

Optional dependencies do not have size limits. This is currently only WGPU, which has millions
of lines of code with it's dependencies.

For more architecture details, there is a [book](https://zk4x.github.io/zyx).


## Code of conduct

Zyx has [code of conduct](CODE_OF_CONDUCT.md) that we humbly borrowed from sqlite.


## Contributing

Please check out [CONTRIBUTING.md](CONTRIBUTING.md)


## Thank you

For contributing to Zyx, finding bugs and using it in your ML models.


## License

Zyx is free software licensed under the GNU Lesser General Public License v3.0 (`LGPLv3`)
See the LICENSE file for details.
