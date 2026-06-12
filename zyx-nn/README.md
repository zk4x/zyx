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

let linear = Linear::new(128, 64, true, DType::F32).unwrap();
let x = Tensor::randn([32, 128], DType::F32).unwrap();
let y = linear.forward(&x).unwrap();

let norm = LayerNorm::new([64], 1e-5, true, true, DType::F32).unwrap();
let z = norm.forward(&y).unwrap();
```

## API

All modules implement the `Module` trait from `zyx-derive`, which provides:
- `forward(x: impl Into<Tensor>) -> Result<Tensor, ZyxError>` — Forward pass

## Autograd

zyx uses `GradientTape` for automatic differentiation. The tape must be created before the forward pass to capture the computation graph.

```rust
use zyx::{Tensor, DType, GradientTape};
use zyx_nn::Linear;

let mut linear = Linear::new(128, 64, true, DType::F32).unwrap();
let x = Tensor::randn([32, 128], DType::F32).unwrap();
let target = Tensor::from([32, 64]);

// Create gradient tape BEFORE forward pass
let tape = GradientTape::new();

let y = linear.forward(&x).unwrap();
let loss = y.mse_loss(&target)?;

// Compute gradients w.r.t. model parameters
let grads = tape.gradient(&loss, &[&linear.weight, &linear.bias.unwrap()]);
# Ok::<(), zyx::ZyxError>(())
```

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
