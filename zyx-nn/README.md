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
use zyx::{Tensor, DType};
use zyx_nn::{Linear, LayerNorm};

// Create a linear layer
let linear = Linear::new(128, 64, true, DType::F32)?;

// Forward pass
let x = Tensor::randn([32, 128], DType::F32)?;
let y = linear.forward(&x)?;

// Layer normalization
let norm = LayerNorm::new([64], 1e-5, true, true, DType::F32)?;
let z = norm.forward(&y)?;

// Compute loss and gradients with autograd
// let loss = z.sum()?;
// let grads = GradientTape::gradient(&loss, [&model.weight, model.bias]);
```

## API

All modules implement the `Module` trait from `zyx-derive`, which provides:
- `forward(x: impl Into<Tensor>) -> Result<Tensor, ZyxError>` — Forward pass

## Autograd

zyx uses `GradientTape` for automatic differentiation. See the [zyx book](https://zk4x.github.io/zyx) for detailed API documentation.

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
