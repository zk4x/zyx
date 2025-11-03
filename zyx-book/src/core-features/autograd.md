# Automatic Differentiation

## Core Concepts
Zyx implements automatic differentiation through an explicit gradient tape system. Any differentiable mathematical operation can be automatically tracked and differentiated, including functions like ReLU that have points of non-differentiability in traditional calculus.

## Example Workflow

### Forward Pass with Gradient Tape
Create a gradient tape and perform tensor operations:

```rust
use zyx::{Tensor, GradientTape, DType};

let tape = GradientTape::new();
let x = Tensor::randn([1024, 1024], DType::F32);
let y = Tensor::from([2, 3, 1]);
let z = (x + y.pad([(1000, 21)], 8)) * x;
```

### Backward Pass via Gradient Tape
Compute gradients using the tape:

```rust
let grads = tape.gradient(&z, &[&x, &y]);
```

### Gradient Handling
The `gradient` method returns `Vec<Option<Tensor>>` where `None` indicates no computational path:

```rust
let tape = GradientTape::new();
let x = Tensor::randn([2, 3], DType::F32);
let y = Tensor::randn([2, 3], DType::F32);
let z = y.exp();
let grads = tape.gradient(&z, &[&x]);
assert_eq!(grads, vec![None]);  // No gradient for x since z doesn't depend on it
```

## Performance Advantages

### Tape-Based Computation
Zyx's autograd system:
- Uses an explicit `GradientTape` to record operations
- Only computes gradients when explicitly requested
- Optimizes memory usage through lazy evaluation
- Supports higher-order derivatives with persistent gradient tapes

This architecture enables efficient differentiation while maintaining flexibility for complex computational graphs.
