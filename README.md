# zyx

[![crates.io](https://img.shields.io/crates/v/zyx.svg)](https://crates.io/crates/zyx)
[![PyPI](https://img.shields.io/pypi/v/zyx-py.svg)](https://pypi.org/project/zyx-py/)
[![docs.rs](https://docs.rs/zyx/badge.svg)](https://docs.rs/zyx)
[![build status](https://github.com/zk4x/zyx/workflows/Build%20and%20Publish%20Wheels/badge.svg)](https://github.com/zk4x/zyx/actions/workflows/build-wheels.yml)
[![license](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://github.com/zk4x/zyx/blob/main/LICENSE)
[![maintenance](https://img.shields.io/badge/maintenance-active-green.svg)](https://github.com/zk4x/zyx)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [🐍 Python Bindings](#python-bindings)
- [Crates](#crates)
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
- [Debug Options](#debug-options)
- [Quick Reference](#quick-reference)

## Overview

**zyx** is a complete ML library and compiler that goes from assembly all the way to neural networks. It has a stable API with performance under active optimization.

## Features

- **Unified Graph** — autograd and laziness share the same graph, enabling seamless kernel fusion across all operations.
- **Lazy Evaluation** — operations accumulate until `realize()` triggers execution, reducing temporary allocations.
- **Kernel Fusion** — tensor operations compile into single optimized kernels (CUDA, OpenCL, WebGPU, etc.).
- **Cross‑Platform Backends** — native support for OpenCL (CPU via POCL, GPU via native OpenCL drivers), WebGPU, CUDA, and more.
- **Full Linear‑Algebra Coverage** — mirrors the PyTorch ops API (matmul, convolutions, pooling, reductions, indexing, etc.) by stacking ops.
- **Immutable Tensors** — tensors cannot be modified in place, preventing back‑prop errors common in PyTorch (`RuntimeError: a tensor was modified in place`).
- **Explicit Gradient Tape** — you control what is recorded via `GradientTape`; no need for `torch.no_grad()` semantics.
- **Higher-Order Gradients** — experimental (graph-based, forward-mode autograd planned)
- **No Implicit Downcasting** — if a backend doesn't support a dtype, zyx will never silently downcast (e.g., F32→F16). Upcasting (e.g., F16→F32) is allowed when the backend does not natively support the narrower type — correctness is guaranteed.
- **Lazy Device Loading** — tensors load from their current memory pool (disk, another device) into the compute device only when needed, via the runtime scheduler.
- **Parallel Pipelining** — kernels allocate across heterogeneous devices (GPU, CPU, WebGPU) in a pipelined fashion via the runtime scheduler.
- **Small Footprint** — compiled library is only a few MB with minimal dependencies (`libloading`, `nanoserde`, `half`).

## 🐍 Python Bindings

**zyx** offers Python bindings with full PyTorch API compatibility and multiple backend support:

### Basic Usage
```python
import zyx

x = zyx.Tensor.randn(2, 3)
y = zyx.Tensor.uniform_(2, 3, from_=-1.0, to_=1.0)
z = x.relu() + y.tanh()
print(z.shape())

# Autograd example
tape = zyx.GradientTape()
result = x.relu() * y
grads = tape.gradient(result, [x, y])
```

## Crates

| Crate | Description |
|-------|-------------|
| `zyx` | Core tensor library with lazy graph and autodiff |
| `zyx-nn` | Neural network layers (Linear, Conv2d, Attention, etc.) and `#[derive(Module)]` |
| `zyx-optim` | Optimizers (SGD, Adam, AdamW, RMSprop) |

## Installation

### Python Installation

```bash
# Install from PyPI
pip install zyx-py

# Or install from source for development
pip install git+https://github.com/zk4x/zyx.git#subdirectory=zyx-py
```

### Rust Installation

```bash
# Install from crates.io
cargo add zyx zyx-nn zyx-optim
```

## Hello World

Create tensors, apply operations, and trigger computation with `realize()`:

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

A training loop with a two-layer network, using `GradientTape` for autograd and `SGD` for optimization:

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

A Transformer block with multi-head attention, layer normalization, and AdamW optimization:

```rust
use zyx::{DType, GradientTape, Module, Tensor};
use zyx_nn::{Linear, LayerNorm, MultiheadAttention};
use zyx_optim::AdamW;

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

```mermaid
flowchart LR
    A["Tensor Graph"] --> B["Fusion and Device Schedule Search"]
    B --> C["Device Specific Kernel IR"]
    C --> D["Backend Code / Assembly"]
```

Tensor operations build a lazy computation graph. During realization, the graph is analyzed for fusion opportunities and the optimal execution schedule is searched. The fused operations are lowered to a device-specific intermediate representation, then compiled to native code (PTX, OpenCL C, WGSL, etc.) for the target backend.

## Why zyx is Different

| Feature | zyx | PyTorch | TensorFlow | JAX |
|---------|-----|---------|------------|-----|
| **Execution Model** | Lazy with explicit realization | Eager by default | Eager by default | Functional + XLA |
| **Gradient Recording** | Explicit `GradientTape` | Implicit, requires `no_grad()` | Implicit, tf.function | Explicit + jit |
| **Tensor Mutability** | Immutable (no in-place errors) | Mutable (risk of back-prop failures) | Mutable | Immutable |
| **Kernel Fusion** | Automatic, cross-backend | Manual (torch.jit) | Manual (XLA) | Manual (XLA) |
| **Disk I/O** | Lazy loading parallel to compute | Typically blocking | Blocking | Blocking |
| **Device Pipelining** | Built-in heterogeneous pipelining | Manual `to(device)` calls | Manual device placement | Manual device placement |
| **Compilation** | Runtime kernel compilation | Pre-compiled + jit | Pre-compiled | Just-in-time |
| **Import Time** | ~1ms | ~2s | ~3s | ~0.5s |
| **Wheel Size** | ~4MB (includes CUDA) | hundreds of MB |  |  |

## Backends

- [x] **C** - CPU backend via C codegen (clang/gcc)
- [x] **CUDA** - NVIDIA GPU acceleration  
- [x] **OpenCL** - Cross-platform support (CPU via POCL, GPU via native OpenCL drivers)
- [x] **WebGPU (WGPU)** - Modern web and native GPU support
- [ ] **ROCm** - AMD GPU support (planned)

Please see [DEVICE_CONFIG.md](zyx/DEVICE_CONFIG.md) for detailed information on hardware configuration.

## Documentation

- **📚 Book**: [https://zk4x.github.io/zyx/](https://zk4x.github.io/zyx/) - Comprehensive guide
- **📖 API Reference**: [https://docs.rs/zyx](https://docs.rs/zyx) - Complete API documentation

## Status & License

- **Status**: Stable API with active performance optimization
- **License**: LGPL-3.0-only (all crates)
- **Rust Version**: Requires latest stable Rust
- **Platforms**: Linux (primary), macOS, Windows (experimental)

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md), [STYLE.md](STYLE.md), and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for guidelines.

### How to Help

- 🐛 **Find bugs**: correctness is our top priority
- 📝 **Write tests**: integration tests are always appreciated
- 📚 **Improve documentation**: typo fixes and better docs
- ⚡ **Add optimizations**: significant performance improvements (>10%)
- 🔌 **Add backends**: CUDA, ROCm, Metal, Vulkan support
- 🎯 **Implement features**: new tensor operations, layers

### Quick Links

- [Examples](zyx-examples/) - MNIST, RNN implementations
- [Issues](https://github.com/zk4x/zyx/issues) - Bug reports and feature requests

## Debug Options

| Value | Flag | Description |
|-------|------|-------------|
| 1 | dev | Print hardware devices and configuration |
| 2 | perf | Print graph execution characteristics |
| 4 | sched | Print kernels created by scheduler |
| 8 | ir | Print kernels in intermediate representation |
| 16 | asm | Print native assembly/code (OpenCL, WGSL, etc.) |

Example: `ZYX_DEBUG=16 cargo test --features wgpu relu_1`

## Quick Reference

### Common Tensor Operations

```rust
// Creation
let x = Tensor::randn([2, 3], DType::F32)?;
let y = Tensor::zeros([4, 4], DType::F32)?;
let z = Tensor::from([[1.0, 2.0], [3.0, 4.0]]);

// Operations
let sum = x + y;
let product = x * y;
let relu = x.relu()?;
let matmul = x.dot(&y)?;

// Shape manipulation
let reshaped = x.reshape([6, 1])?;
let sliced = x.slice([0..2, 0..2])?;
let transposed = x.t()?;

// Autograd
let tape = GradientTape::new();
let result = x.relu()? * y;
let grads = tape.gradient(&result, &[x, y]);
```

### Python Quick Reference

```python
import zyx

# Creation
x = zyx.Tensor.randn(2, 3)
y = zyx.Tensor.randn(2, 3)  # Same shape for element-wise operations
z = zyx.Tensor([[1, 2], [3, 4]])

# Operations
sum = x + y  # Same shape required
product = x * y  # Same shape required
relu = x.relu()
matmul = x @ zyx.Tensor.randn(3, 2)  # Matrix multiplication, requires compatible shapes

# Shape manipulation
reshaped = x.reshape(6, 1)
sliced = x[0:2, 0:2]
transposed = x.t()

# Autograd
tape = zyx.GradientTape()
result = x.relu() * y
grads = tape.gradient(result, [x, y])
```

### Error Handling

zyx provides clear error messages for common issues:

#### Shape Mismatch Errors
```python
import zyx

# This will fail - incompatible shapes for matrix multiplication
x = zyx.Tensor.randn(2, 5)
y = zyx.Tensor.randn(17, 8)  # Error: 2x5 @ 17x8 is invalid

try:
    result = x @ y
except Exception as e:
    print(f"Shape error: {e}")

# Correct approach - ensure compatible shapes
x = zyx.Tensor.randn(2, 5)
y = zyx.Tensor.randn(5, 8)  # Valid: 2x5 @ 5x8 = 2x8
result = x @ y
```

```rust
use zyx::Tensor;

let x = Tensor::randn([2, 5], DType::F32)?;
let y = Tensor::randn([17, 8], DType::F32)?;

// This returns an error - incompatible shapes for matrix multiplication
match x.dot(&y) {
    Ok(_) => unreachable!(),
    Err(e) => println!("Shape error: {e}"),
}

// Correct approach
let x = Tensor::randn([2, 5], DType::F32)?;
let y = Tensor::randn([5, 8], DType::F32)?;
let result = x.dot(&y)?;
```

#### Device Errors
```python
import zyx

# Operations may succeed initially but fail during realization
x = zyx.Tensor.randn(1000, 1000)  # Large tensor
y = x @ x  # Operation builds in graph

try:
    result = y.realize()  # May fail if device runs out of memory
except Exception as e:
    print(f"Device error during realization: {e}")
    # Handle device errors (e.g., reduce batch size, use smaller tensors)
```

---

<div align="center">
<a href="https://github.com/zk4x/zyx">
    <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" width="20" height="20">
    Star us on GitHub
</a> | 
<a href="https://docs.rs/zyx">
    <img src="https://simpleicons.org/icons/rust.svg" width="20" height="20">
    API Docs
</a>
</div>
