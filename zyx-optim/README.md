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

## Usage

```rust
use zyx::Tensor;
use zyx_nn::{Linear, LayerNorm};
use zyx_optim::{Adam, AdamW, RMSprop, SGD};

fn main() -> zyx::Result<()> {
    let mut model = Linear::new(128, 64);
    
    // Create optimizer
    let optim = Adam::new(&mut model, 0.001);
    
    // Forward pass
    let x: Tensor = Tensor::ones(32, 128);
    let y = model.forward(&x)?;
    
    // Compute loss (placeholder)
    let loss = y.sum()?;
    
    // Backward pass
    let grad_loss = Tensor::ones_like(&y)?;
    loss.backward(&grad_loss)?;
    
    // Optimizer step
    optim.step()?;
    
    Ok(())
}
```

## API

All optimizers implement a simple interface:

```rust
use zyx_optim::Adam;

let mut model = Linear::new(128, 64);
let optim = Adam::new(&mut model, 0.001);

// Configure hyperparameters
optim.lr = 0.001;      // learning rate
optim.beta1 = 0.9;     // first moment decay
optim.beta2 = 0.999;   // second moment decay
optim.eps = 1e-8;      // numerical stability

// Optimizer step (computes gradients via autograd, then updates parameters)
optim.step()?;

// Or use `step_with` for custom loss gradient
optim.step_with(loss_grad)?;

// Zero optimizer state (for multi-step optimization)
optim.zero_grad()?;
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
