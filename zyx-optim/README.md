# zyx-optim

Optimizers for the [zyx](https://github.com/zk4x/zyx) machine learning library.

This crate provides a collection of popular gradient-based optimization algorithms designed to work seamlessly with zyx's autograd system. All optimizers implement the `Module` trait.

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
use zyx_nn::{Linear, LayerNorm};
use zyx_optim::{Adam, SGD};
use zyx::{Tensor, DType};

// Create model
let mut model = Linear::new(128, 64, true, DType::F32)?;

// Create optimizer with default parameters
let mut optim = Adam {
    learning_rate: 0.001,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weight_decay: 0.0,
    amsgrad: false,
    m: Vec::new(),
    v: Vec::new(),
    vm: Vec::new(),
    t: 0,
};

// Forward pass
let x = Tensor::randn([32, 128], DType::F32)?;
let y = model.forward(&x)?;

// Backward pass (computes gradients via autograd)
let loss = y.sum()?;
loss.backward()?;

// Optimizer step
optim.step()?;
```

## API

All optimizers implement the `Module` trait:

```rust
use zyx_optim::Adam;

let mut optim = Adam {
    learning_rate: 0.001,
    betas: (0.9, 0.999),
    eps: 1e-8,
    weight_decay: 0.0,
    amsgrad: false,
    m: Vec::new(),
    v: Vec::new(),
    vm: Vec::new(),
    t: 0,
};

// Optimizer step (computes gradients via autograd, then updates parameters)
optim.step()?;

// Configure hyperparameters
optim.learning_rate = 0.001;      // learning rate
optim.betas = (0.9, 0.999);       // first and second moment decay
optim.eps = 1e-8;                 // numerical stability
optim.weight_decay = 0.01;        // L2 weight decay

// Zero optimizer state (for multi-step optimization)
optim.zero_grad()?;
```

## Hyperparameters

### Adam
- `learning_rate` — Learning rate (default: 0.001)
- `betas` — First and second moment decay (default: (0.9, 0.999))
- `eps` — Numerical stability (default: 1e-8)
- `weight_decay` — L2 weight decay (default: 0.0)
- `amsgrad` — AMSGrad variant (default: false)

### AdamW
Same as Adam, with additional:
- `weight_decay` — L2 weight decay (default: 0.0)

### RMSprop
- `learning_rate` — Learning rate (default: 0.001)
- `momentum` — Moving average decay (default: 0.9)
- `eps` — Numerical stability (default: 1e-8)

### SGD
- `learning_rate` — Learning rate (default: 0.01)
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
