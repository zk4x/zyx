# zyx

## Overview

**zyx** is a Rust‑based machine‑learning library focused on **kernel fusion**, **lazy evaluation**, and **minimal overhead**. It provides a unified computation graph that powers both automatic differentiation and lazy execution. **zyx is experimental** — performance is still under active optimization and it is not production‑ready.

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

```toml
[dependencies]
zyx = { path = "zyx" }
zyx-nn = { path = "zyx-nn" }
zyx-optim = { path = "zyx-optim" }
zyx-derive = { path = "zyx-derive" }
```

```rust
use zyx::{DType, GradientTape, Tensor};
use zyx_nn::{Linear, LayerNorm, Module, MultiheadAttention};
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

fn main() -> anyhow::Result<()> {
    let mut model = TransformerBlock::new(512, 8, DType::F32)?;
    let mut optim = AdamW::default();
    let x = Tensor::randn([4, 128, 512], DType::F32)?;
    let tape = GradientTape::new();
    let out = model.forward(&x)?;
    let grads = tape.gradient(&out, &model);
    optim.update(&mut model, grads);
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

## Documentation

- **Book**: https://zk4x.github.io/zyx/
- **API Reference**: https://docs.rs/zyx

## Status & License

- **Status**: Experimental — API is stabilizing, performance under active optimization
- **License**: LGPL‑3.0‑only (all crates)

## Contributing

Contributions are welcome! Please read the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
