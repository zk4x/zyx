# Autograd

Zyx implements automatic differentiation through an explicit `Tape`. Unlike other frameworks, there is no separate autograd graph — it uses the **same graph** as computation.

## The Key Insight

In most frameworks, autograd requires a separate graph because the eager execution engine discards intermediate results. Zyx's tape keeps all graph nodes alive until `realize()` or drop, and simply prevents their deletion until gradients are computed.

## Tape API

```rust
# extern crate zyx;
# use zyx::{DType, Tape, Tensor, ZyxError};
# fn main() -> Result<(), ZyxError> {
let tape = Tape::new()?;
let x = Tensor::randn([2, 3], DType::F32)?;
let y = Tensor::randn([2, 3], DType::F32)?;
let z = x.relu() * y.tanh();

let grads = tape.gradient(&z, vec![&x, &y]);
// grads[0] = gradient of z w.r.t. x
// grads[1] = gradient of z w.r.t. y
# Ok(())
# }
```

## No "requires_grad"

There's no `requires_grad` flag on tensors. The tape records the entire graph; when you call `gradient()`, you specify which tensors you want gradients for:

```rust
# extern crate zyx;
# use zyx::{DType, Tape, Tensor, ZyxError};
# fn main() -> Result<(), ZyxError> {
let tape = Tape::new()?;
let x = Tensor::randn([2, 3], DType::F32)?;
let y = Tensor::randn([2, 3], DType::F32)?;
let z = y.exp();

let grads = tape.gradient(&z, vec![&x]);  // None — z doesn't depend on x
# Ok(())
# }
```

This is more flexible — you don't need to decide at tensor creation time which tensors will be differentiated.

## Higher-Order Derivatives

The tape stays alive after `gradient()` for higher-order derivatives:

```rust,ignore
let tape = Tape::new()?;
let x = Tensor::randn([2, 3], DType::F32)?;
let z = x.relu();
let g = tape.gradient(&z, vec![&x]);
let h = tape.gradient(&g[0], vec![&x]);  // second derivative
```

## Memory Efficiency

Inside a tape, intermediate tensors needed for backpropagation are not held in memory until realize time. The tape stores only `TensorId` values — not the actual data. When the tape is dropped, all tape-preserved nodes are released.
