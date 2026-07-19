# Zyx

Zyx is a machine learning library that runs on your hardware.


# Why is Zyx?

ML was enabled by new kinds of highly parallel, high performance hardware that did not exist before.

Zyx has 3 goals + a bonus goal:
1. Be correct
2. Run everywhere
3. Run fast
Bonus: be pleasant to use while at that

ML won't get better without new hardware and existing libraries are ill-suited to support emerging hardware.
The primary problem is the requirement to write custom kernels to get the required performance.
Manufactures have tough time writing these high performance kernels, therefore they write kernels for only
a few ops and don't support general linear algebra.

Zyx approaches these problems from two angles, while maintaining high flexibility (proven
by several relatively fast rewrites of core components without significant API changes):

1. Linear SSA-ish IR with explicit control flow. This is the hardware unifying interface. Each hardware has
add instructions, can repeat instructions (loop), is highly parallel (work sizes), has multiple types of memory
in a hierarchy and can optionally have vectorization and tiling. This is the core of the instruction set.
Zyx has a series of optimization passes (selected by autotuner) that apply various optimizations for different
levels of these characteristics. The lowering layer from IR to backends is almost 1:1 mapping. If your hardware
can provide this translation, zyx gives you all of linear algebra support.

2. As good as the automatic optimizations can be, writing manual kernels will always be faster, that is why
it is the dominant approach as of now. Zyx acknowledges this and maintains flexibility through it's egraph
system pattern matching of any subgraph structure into a custom kernel written in language of your choosing
(or raw binary blobs, cublas, cblas, etc.), as well as writing custom kernels in zyx IR and taking advantage
of optimization passes zyx provides. Egraph measures their timings, compares with auto-generated zyx kernels
and picks the fastest path through this graph.

The other issue is running on edge and platforms that don't have sufficient resources. A regular CPU only
pytorch install is 100+ MB, while with cuda support, it's 2+ GB. Zyx takes only about 5 MB and uses machine
available drivers to run, such as provided C compiler, provided CUDA runtime, but for example if you don't install
CUDA, any gpu driver with vulkan support is sufficient.


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
- [x] `CUDA` — NVIDIA GPU acceleration
- [x] `OpenCL` — Cross-platform (CPU via POCL, GPU via native drivers)
- [x] `Vulkan` — Cross-platform GPU acceleration via Vulkan (SPIR-V)
- [x] `WGPU` — Modern web and native GPU support via wgpu (WGSL), feature: `wgpu`

If you'd like to add new backend to zyx, that would be awesome!
Please read [BACKEND.md](https://github.com/zk4x/zyx/blob/main/zyx/BACKEND.md)

## Neural network training

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
    let tape = Tape::new(&net)?;
    let y = net.forward(&x);
    let loss = y.mse_loss(&target)?;
    let grads = tape.gradient(&loss, &net);
    optim.update(&mut net, grads);
    tape.realize(net.into_iter().chain(optim.into_iter()))?;
}

# Ok::<(), zyx::ZyxError>(())
```

## Error handling

In case of incorrect user input, zyx returns results. Panics are reserved for OOM and hardware issues that are
not recoverable. There are minimal exceptions to this rule, such as binary ops, which will panic if they cannot
be broadcasted to a common shape.

### DTypes

Backends advertise supported dtypes via `supported_dtypes` mask. zyx will never implicitly downcast (e.g., F32→F16)
when a backend lacks support — the operation fails explicitly. Implicit upcasting (e.g., F16→F32) is permitted
when the backend does not natively support the narrower type — correctness is guaranteed, not performance.

## Rust version

Zyx supports rust 1.88+. Zyx also requires std, as it accesses files (like cuda, hip and opencl runtimes),
env var (for debugging) and also some other stuff that requires filesystem and threads (loading files,
multithreaded execution, worker threads, etc.).

## Operating systems

Zyx is currently tested only on linux, but should work with all \*nix operating systems.
If it does not work on your system, or if you are interested in Windows support, please
create a github issue. Basically the only difference between operating systems is specifying
proper paths to backend runtimes (e.g. libcuda.so).

## Features

- **wgpu** - enables wgpu backend

## Warning

Zyx uses some unsafe code, due to FFI/hardware access. Zyx brings it's own runtime.
It is a single global struct behind mutex. Tensors are indices into a graph stored in this runtime.
It may not be the cleanest approach, but it is the fast and convenient approach.

## Dependencies

Zyx tries to use 0 dependencies, but we are not reinventing the wheel, so we use nanoserde for config
parsing, libloading to dynamically load backend dynamic library files (i.e. libcuda.so).
All dependencies are carefully considered and are used only if deemed absolutely necessary,
that is only if they do one thing and do it well.

Currently zyx is below 30k LOC. OFC runtimes are needed for respective backends (e.g. libcuda.so).

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

