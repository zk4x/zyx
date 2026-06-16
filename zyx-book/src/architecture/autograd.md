# Autograd

Zyx implements automatic differentiation through an explicit `GradientTape`. Unlike other frameworks, there is no separate autograd graph — it uses the **same graph** as computation.

## The Key Insight

In most frameworks, autograd requires a separate graph because the eager execution engine discards intermediate results. Zyx's lazy graph keeps all nodes alive until `realize()` is called, and the gradient tape simply prevents their deletion.

## GradientTape API

```rust
# extern crate zyx;
# use zyx::{DType, GradientTape, Tensor, ZyxError};
# fn main() -> Result<(), ZyxError> {
let tape = GradientTape::new();
let x = Tensor::randn([2, 3], DType::F32)?;
let y = Tensor::randn([2, 3], DType::F32)?;
let z = x.relu() * y.tanh();

let grads = tape.gradient(&z, vec![&x, &y]);
// grads[0] = gradient of z w.r.t. x
// grads[1] = gradient of z w.r.t. y
# Ok(())
# }
```

The `gradient()` method consumes the tape; use `gradient_persistent()` to keep it alive for higher-order derivatives.

## No "requires_grad"

There's no `requires_grad` flag on tensors. The tape records the entire graph; when you call `gradient()`, you specify which tensors you want gradients for:

```rust
# extern crate zyx;
# use zyx::{DType, GradientTape, Tensor, ZyxError};
# fn main() -> Result<(), ZyxError> {
let tape = GradientTape::new();
let x = Tensor::randn([2, 3], DType::F32)?;
let y = Tensor::randn([2, 3], DType::F32)?;
let z = y.exp();

let grads = tape.gradient(&z, vec![&x]);  // None — z doesn't depend on x
# Ok(())
# }
```

This is more flexible — you don't need to decide at tensor creation time which tensors will be differentiated.

## Higher-Order Derivatives

Using `gradient_persistent()`, the tape stays alive and you can compute higher-order derivatives:

```rust,ignore
// Higher-order derivatives example: currently blocked by
// an autograd subtraction overflow bug (autograd.rs:94).
```

## Memory Efficiency

Since the graph is lazy, intermediate tensors needed for backpropagation are **not held in memory** until `realize()` is called for the gradient computation. The gradient tape stores only `TensorId` values — not the actual data. When the tape is dropped, all tape-preserved nodes are released.
