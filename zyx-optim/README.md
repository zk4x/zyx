# zyx-optim

Optimizers for the [zyx](https://github.com/zk4x/zyx) machine learning library.

This crate provides a collection of popular gradient-based optimization algorithms designed to work seamlessly with zyx's autograd system. All optimizers support the `Module` trait for easy integration with neural network modules.

## Features

### Optimizers
- `Adam` — Adaptive moment estimation
- `AdamW` — Adam with weight decay decoupling
- `RMSprop` — Root mean square propagation
- `SGD` — Stochastic gradient descent

### Python Bindings
- `py` feature enables Python interoperability via pyo3

## API

All optimizers implement a simple interface via the `Module` trait:

```rust
use zyx_nn::{Linear, LayerNorm};
use zyx_optim::Adam;

// Create model and optimizer
let mut model = Linear::new(/* in_features, out_features, bias, dtype */)?;
let optim = Adam::new(&mut model, /* lr, beta1, beta2, eps */)?;

// Forward pass
let x = Tensor::ones(32, 128);
let y = model.forward(&x)?;

// Backward pass (computes gradients via autograd)
let loss = y.sum()?;
loss.backward()?;

// Optimizer step
optim.step()?;
```

## Hyperparameters

### Adam
- `lr` — Learning rate (default: 0.001)
- `beta1` — First moment decay (default: 0.9)
- `beta2` — Second moment decay (default: 0.999)
- `eps` — Numerical stability (default: 1e-8)

### AdamW
Same as Adam, with additional:
- `weight_decay` — L2 weight decay (default: 0.0)

### RMSprop
- `lr` — Learning rate (default: 0.001)
- `momentum` — Moving average decay (default: 0.9)
- `eps` — Numerical stability (default: 1e-8)

### SGD
- `lr` — Learning rate (default: 0.01)
- `momentum` — Momentum coefficient (default: 0.0)
- `weight_decay` — L2 weight decay (default: 0.0)

## Features

- **`py`** — Enable Python bindings via pyo3
- **`std`** — Enables zyx-core/std (default)

## Documentation

- [Full API docs](https://docs.rs/zyx-optim)
- [zyx book](https://zk4x.github.io/zyx)
- [GitHub](https://github.com/zk4x/zyx)

## License

LGPL-3.0-only
