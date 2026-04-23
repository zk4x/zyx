# zyx

A pure Rust machine learning library focused on **kernel fusion** and **minimal overhead**.

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
    fn new(dim: u64, num_heads: u64, dtype: DType) -> Result<Self, ZyxError> {
        Ok(Self {
            attn: MultiheadAttention::new(dim, num_heads, 0.0, true, false, false, None, None, true, dtype)?,
            mlp: Linear::new(dim, dim * 4, true, dtype)?,
            mlp2: Linear::new(dim * 4, dim, true, dtype)?,
            norm1: LayerNorm::new([dim], 1e-5, true, true, dtype)?,
            norm2: LayerNorm::new([dim], 1e-5, true, true, dtype)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, ZyxError> {
        let attn_out = self.attn.forward(x, x, x, None::<Tensor>, false, None::<Tensor>, true, false)?.0;
        let x = self.norm1.forward(&(x + attn_out))?;
        let mlp_out = self.mlp.forward(&x)?.gelu();
        let mlp_out = self.mlp2.forward(&mlp_out)?;
        Ok(self.norm2.forward(&(x + mlp_out))?)
    }
}

let mut model = TransformerBlock::new(512, 8, DType::F32)?;
let mut optim = AdamW::default();

let x = Tensor::randn([4, 128, 512], DType::F32)?;
let tape = GradientTape::new();
let out = model.forward(&x)?;
let grads = tape.gradient(&out, &model);
optim.update(&mut model, grads);
```

## Key Design Decisions

- **One graph for everything** — autograd and laziness share the same graph
- **Lazy evaluation** — operations accumulate in the graph until `realize()` is called
- **Kernel fusion** — tensor ops compile into single optimized GPU kernels
- **Minimal dependencies** — core has only `libloading`, `nanoserde`, and `half`
- **16 bytes per tensor** — 10k virtual tensors use ~160kB + shape metadata

## Crates

| Crate | Description |
|-------|-------------|
| [`zyx`](zyx/) | Core tensor library with autodiff |
| [`zyx-nn`](zyx-nn/) | Neural network layers (Linear, Conv2d, LSTM, Attention...) |
| [`zyx-optim`](zyx-optim/) | Optimizers (SGD, Adam, AdamW, RMSprop) |
| [`zyx-derive`](zyx-derive/) | `#[derive(Module)]` procedural macro |
| [`zyx-examples`](zyx-examples/) | MNIST, NanoGPT, Phi, RNN examples |

## Features

**Tensor Operations**
- All standard ops: matmul, convolutions, pooling, reductions, indexing
- Lazy graph building with dynamic shape inference
- Supports f32, f64, i32, i64, u32, u64, f16, bf16, bool

**Automatic Differentiation**
- Reverse-mode AD via shared computation graph
- Gradient clipping, zeroing, and accumulation

**Neural Network Layers**
- Dense: Linear, LayerNorm, RMSNorm, GroupNorm, BatchNorm
- Convolutional: Conv2d
- Recurrent: RNNCell, GRUCell, LSTMCell
- Attention: CausalSelfAttention, MultiheadAttention
- Embedding, PositionalEncoding, TransformerEncoder/DecoderLayer

**Optimizers**
- SGD (with momentum, nesterov)
- Adam, AdamW (with weight decay)
- RMSprop

**Backends**
- OpenCL (CPU via pocl, GPU via vendor runtimes)
- WebGPU via wgpu
- CUDA, ROCm/HIP via dynamic `.so` loading

**Utilities**
- Safe serialization (safetensors format)
- GGUF model loading

## Quick Start

```toml
[dependencies]
zyx = { path = "zyx" }
zyx-nn = { path = "zyx-nn" }
zyx-optim = { path = "zyx-optim" }
zyx-derive = { path = "zyx-derive" }
```

See [`zyx-examples`](zyx-examples/) for MNIST and RNN training examples.

## Architecture

```
Tensor → Graph (Lazy) → Kernel IR → Backend Code (PTX, OpenCL, WGSL...)
```

Operations are added to a lazy graph and compiled to low-level kernel IR. The autotune system searches for optimal kernel configurations (loop tiling, MAD fusion, vectorization).

## Documentation

- [Book](https://zk4x.github.io/zyx/) — comprehensive guide
- [docs.rs](https://docs.rs/zyx) — API reference

## Status

Experimental. The API is stabilizing but breaking changes may occur.

## License

LGPL-3.0-only (all crates)