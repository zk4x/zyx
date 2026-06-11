# zyx-nn

Neural network modules for the [zyx](https://github.com/zk4x/zyx) machine learning library.

This crate provides a collection of common neural network building blocks implemented as reusable `Module` traits, designed to work seamlessly with zyx's kernel fusion and autograd system.

## Features

### Linear & Normalization
- `Linear` — Dense fully-connected layer
- `LayerNorm` — Layer normalization
- `BatchNorm` — Batch normalization
- `GroupNorm` — Group normalization
- `RMSNorm` — Root mean square normalization

### Recurrent Layers
- `RNNCell` — Simple recurrent cell
- `GRUCell` — Gated recurrent unit
- `LSTMCell` — Long short-term memory

### Attention Mechanisms
- `CausalSelfAttention` — Causal self-attention for transformers
- `MultiheadAttention` — Multi-head attention with configurable heads

### Embeddings & Convolution
- `Embedding` — Learnable embedding lookup
- `Conv2d` — 2D convolution

### Transformers
- `TransformerEncoderLayer` — Single transformer encoder block
- `TransformerDecoderLayer` — Single transformer decoder block
- `PositionalEncoding` — Sinusoidal positional embeddings

### Python Bindings
- `py` feature enables Python interoperability via `pyo3`

## Usage

```rust
use zyx::Tensor;
use zyx_nn::{Linear, LayerNorm, Embedding, MultiheadAttention};

fn main() -> zyx::Result<()> {
    let x: Tensor = Tensor::ones(32, 128);
    
    // Linear layer
    let linear = Linear::new(128, 64);
    let h = linear.forward(&x)?;
    
    // LayerNorm
    let norm = LayerNorm::new(64);
    let h = norm.forward(&h)?;
    
    // Multi-head attention
    let attn = MultiheadAttention::new(64, 4, None);
    let h = attn.forward(&h, &h, &h)?;
    
    // Embedding
    let emb = Embedding::new(1000, 64);
    let tokens = Tensor::from(0..1000);
    let h = emb.forward(&tokens)?;
    
    Ok(())
}
```

## API

All modules implement the `Module` trait from `zyx-derive`, which provides:
- `forward(&self, x: &Tensor) -> Result<Tensor, ZyxError>` — Forward pass
- `backward(&mut self, grad_output: &Tensor) -> Result<Tensor, ZyxError>` — Backward pass
- `zero_grad(&mut self)` — Reset gradients

## Features

- **`py`** — Enable Python bindings via pyo3

## Documentation

- [Full API docs](https://docs.rs/zyx-nn)
- [zyx book](https://zk4x.github.io/zyx)
- [GitHub](https://github.com/zk4x/zyx)

## License

LGPL-3.0-only

## Badges

[![experimental](https://img.shields.io/badge/maintenance-experimental-yellow)]()
