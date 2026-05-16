# zyx

[![crates.io](https://img.shields.io/crates/v/zyx.svg)](https://crates.io/crates/zyx)
[![docs.rs](https://docs.rs/zyx/badge.svg)](https://docs.rs/zyx)
[![build status](https://github.com/zk4x/zyx/actions/workflows/rust.yml/badge.svg)](https://github.com/zk4x/zyx/actions/workflows/rust.yml)
[![license](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://github.com/zk4x/zyx/blob/main/LICENSE)
[![maintenance](https://img.shields.io/badge/maintenance-experimental-yellow.svg)](https://github.com/zk4x/zyx)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Crates](#crates)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Hello World](#hello-world)
- [Basic Neural Network](#basic-neural-network)
- [Advanced Examples](#advanced-examples)
- [Architecture](#architecture)
- [Why zyx is Different](#why-zyx-is-different)
- [Backends](#backends)
- [Documentation](#documentation)
- [Status & License](#status--license)
- [Contributing](#contributing)

## Key Features

- **Unified Graph** — autograd and laziness share the same graph, enabling seamless kernel fusion across all operations.
- **Lazy Evaluation** — operations accumulate until `realize()` triggers execution, reducing temporary allocations.
- **Kernel Fusion** — tensor operations compile into single optimized kernels (CUDA, OpenCL, WebGPU, etc.).
- **Cross‑Platform Backends** — native support for OpenCL (CPU/GPU via POCL), WebGPU, CUDA/ROCm, and more.
- **Full Linear‑Algebra Coverage** — mirrors the PyTorch ops API (matmul, convolutions, pooling, reductions, indexing, etc.) by stacking ops.
- **Immutable Tensors** — tensors cannot be modified in place, preventing back‑prop errors common in PyTorch (`RuntimeError: a tensor was modified in place`).
- **Explicit Gradient Tape** — you control what is recorded via `GradientTape`; no need for `torch.no_grad()` semantics.
- **Arbitrary‑Order Differentiation** — the graph supports 2nd, 3rd, and higher‑order gradients natively.
- **Lazy Disk Loading** — large datasets or models load from disk in parallel with computation.
- **Parallel Pipelining** — work distributes across heterogeneous devices (GPU, CPU, WebGPU) in a pipelined fashion.
- **Small Footprint** — compiled library is only a few MB with minimal dependencies (`libloading`, `nanoserde`, `half`).

## Crates

| Crate | Description |
|-------|-------------|
| `zyx` | Core tensor library with lazy graph and autodiff |
| `zyx-nn` | Neural network layers (Linear, Conv2d, Attention, etc.) |
| `zyx-optim` | Optimizers (SGD, Adam, AdamW, RMSprop) |
| `zyx-derive` | `#[derive(Module)]` procedural macro |
| `zyx-examples` | MNIST, NanoGPT, Phi, RNN examples |

## Quick Start

### Installation

```toml
[dependencies]
zyx = { path = "zyx" }
zyx-nn = { path = "zyx-nn" }
zyx-optim = { path = "zyx-optim" }
zyx-derive = { path = "zyx-derive" }
```

### Hello World

```rust
use zyx::{Tensor, DType};

fn main() -> Result<(), zyx::ZyxError> {
    // Create tensors
    let x = Tensor::randn([2, 3], DType::F32)?;
    let y = Tensor::uniform([2, 3], -1f32..1f32)?;
    
    // Perform operations (lazy evaluation)
    let z = x.relu()? + y.tanh()?;
    
    // Realize computation
    let result = z.realize()?;
    
    println!("Result shape: {:?}", result.shape());
    Ok(())
}
```

## Basic Neural Network

```rust
use zyx::{Tensor, DType, GradientTape};
use zyx_nn::{Linear, Module};
use zyx_optim::SGD;

#[derive(Module)]
struct SimpleNet {
    linear1: Linear,
    linear2: Linear,
}

impl SimpleNet {
    fn new(dtype: DType) -> Result<Self, zyx::ZyxError> {
        Ok(Self {
            linear1: Linear::new(784, 128, true, dtype)?,
            linear2: Linear::new(128, 10, true, dtype)?,
        })
    }
    
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.linear1.forward(x).unwrap().relu();
        self.linear2.forward(&x).unwrap()
    }
}

fn main() -> Result<(), zyx::ZyxError> {
    let mut model = SimpleNet::new(DType::F32)?;
    let mut optim = SGD::default();
    let x = Tensor::randn([64, 784], DType::F32)?;
    let target = Tensor::randn([64, 10], DType::F32)?;
    
    for epoch in 0..10 {
        let tape = GradientTape::new();
        let output = model.forward(&x);
        let loss = output.mse_loss(&target)?;
        
        let grads = tape.gradient(&loss, &model);
        optim.update(&mut model, grads);
        
        // Realize to trigger computation
        Tensor::realize_all()?;
        
        println!("Epoch {}: Loss = {:.4}", epoch, loss.item::<f32>()?);
    }
    
    Ok(())
}
```

## Advanced Examples

```rust
use zyx::{DType, GradientTape, Module, Tensor};
use zyx_nn::{Linear, LayerNorm, MultiheadAttention};
use zyx_optim::AdamW;
use zyx_derive::Module;

#[derive(Module)]
struct TransformerBlock {
    attn: MultiheadAttention,
    mlp: Linear,
    mlp2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerBlock {
    fn new(dim: u64, num_heads: u64, dtype: DType) -> Result<Self, zyx::ZyxError> {
        Ok(Self {
            attn: MultiheadAttention::new(dim, num_heads, 0.0, true, false, false, None, None, true, dtype)?,
            mlp: Linear::new(dim, dim * 4, true, dtype)?,
            mlp2: Linear::new(dim * 4, dim, true, dtype)?,
            norm1: LayerNorm::new([dim], 1e-5, true, true, dtype)?,
            norm2: LayerNorm::new([dim], 1e-5, true, true, dtype)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, zyx::ZyxError> {
        let attn_out = self.attn.forward(x, x, x, None::<Tensor>, false, None::<Tensor>, true, false)?.0;
        let x = self.norm1.forward(&(x + attn_out))?;
        let mlp_out = self.mlp.forward(&x)?.gelu();
        let mlp_out = self.mlp2.forward(&mlp_out)?;
        Ok(self.norm2.forward(&(x + mlp_out))?)
    }
}

fn main() -> Result<(), zyx::ZyxError> {
    let mut model = TransformerBlock::new(64, 4, DType::F32)?;
    let mut optim = AdamW::default();
    let x = Tensor::randn([2, 8, 64], DType::F32)?;

    let tape = GradientTape::new();
    let out = model.forward(&x)?;
    let grads = tape.gradient(&out, &model);

    // Update parameters with gradients
    optim.update(model.iter_mut(), grads);

    // Realize model to trigger computation (zyx uses lazy evaluation)
    model.realize()?;
    Ok(())
}
```

## Architecture

```
Tensor → Lazy Graph → Kernel IR → Backend Code (PTX, OpenCL, WGSL, etc.)
```

The autotune system in `zyx/src/kernel/autotune.rs` searches for optimal kernel configurations (loop tiling, MAD fusion, vectorization) before execution.

### Why zyx is Different

| Feature | zyx | PyTorch |
|---------|-----|---------|
| Gradient recording | Explicit `GradientTape` | Implicit, requires `no_grad()` |
| Tensor mutability | Immutable (no in‑place errors) | Mutable (can cause back‑prop failures) |
| Higher‑order gradients | Arbitrary order natively | Supported but more complex |
| Disk I/O | Lazy loading parallel to compute | Typically blocking |
| Device pipelining | Built‑in heterogeneous pipelining | Manual `to(device)` calls |

## Backends

- [x] **CUDA** - NVIDIA GPU acceleration
- [x] **OpenCL** - Cross-platform CPU/GPU via POCL
- [x] **WebGPU (WGPU)** - Modern web and native GPU support
- [ ] **ROCm** - AMD GPU support (planned)

Please see [DEVICE_CONFIG.md](zyx/DEVICE_CONFIG.md) for detailed information on hardware configuration.

## Documentation

- **📚 Book**: [https://zk4x.github.io/zyx/](https://zk4x.github.io/zyx/) - Comprehensive guide
- **📖 API Reference**: [https://docs.rs/zyx](https://docs.rs/zyx) - Complete API documentation
- **💬 Community**: Join our [Discord](https://discord.gg/zyx) for discussions and support

## Status & License

- **Status**: Experimental — API is stabilizing, performance under active optimization
- **License**: LGPL-3.0-only (all crates)
- **Rust Version**: Requires latest stable Rust
- **Platforms**: Linux (primary), macOS, Windows (experimental)

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Help

- 🐛 **Find bugs**: correctness is our top priority
- 📝 **Write tests**: integration tests are always appreciated
- 📚 **Improve documentation**: typo fixes and better docs
- ⚡ **Add optimizations**: significant performance improvements (>10%)
- 🔌 **Add backends**: CUDA, ROCm, Metal, Vulkan support
- 🎯 **Implement features**: new tensor operations, layers

### Quick Links

- [Examples](zyx-examples/) - MNIST, NanoGPT, RNN implementations
- [Issues](https://github.com/zk4x/zyx/issues) - Bug reports and feature requests
- [Discussions](https://github.com/zk4x/zyx/discussions) - Community discussions
- [Discord](https://discord.gg/zyx) - Real-time chat support

---

<div align="center">
<a href="https://github.com/zk4x/zyx">
    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" height="20">
    Star us on GitHub
</a> | 
<a href="https://discord.gg/zyx">
    <img src="https://simpleicons.org/icons/discord.svg" width="20" height="20">
    Join Discord
</a> | 
<a href="https://docs.rs/zyx">
    <img src="https://simpleicons.org/icons/rust.svg" width="20" height="20">
    API Docs
</a>
</div>
