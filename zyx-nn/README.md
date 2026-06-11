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

## API

All modules implement the `Module` trait from `zyx-derive`, which provides:
- `forward(x: &Tensor) -> Result<Tensor, ZyxError>` — Forward pass
- `backward(grad_output: &Tensor) -> Result<Tensor, ZyxError>` — Backward pass
- `zero_grad()` — Reset gradients

Modules are typically created using the `new()` method with appropriate parameters. See the [zyx book](https://zk4x.github.io/zyx) for detailed API documentation.

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
