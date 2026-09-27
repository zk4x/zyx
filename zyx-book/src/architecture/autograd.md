# Autograd

Zyx implements automatic differentiation through an explicit `Tape`. Unlike other frameworks, there is no separate autograd graph — it uses the **same graph** as computation.

## The Key Insight

In most frameworks, autograd requires a separate graph because the eager execution engine discards intermediate results. Zyx's tape keeps all graph ops alive until `realize()` or drop, and simply prevents their deletion until gradients are computed.

## Tape API

```rust
# extern crate zyx;
# use zyx::{DType, Tape, Tensor, ZyxError};
# fn main() -> Result<(), ZyxError> {
let x = Tensor::randn([2, 3], DType::F32)?;
let y = Tensor::randn([2, 3], DType::F32)?;
let tape = Tape::new([&x, &y])?;
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
let x = Tensor::randn([2, 3], DType::F32)?;
let y = Tensor::randn([2, 3], DType::F32)?;
let tape = Tape::new([&x, &y])?;
let z = y.exp();

let grads = tape.gradient(&z, vec![&x, &y]);  // grads[0] is zero — z doesn't depend on x
# Ok(())
# }
```

This is more flexible — you don't need to decide at tensor creation time which tensors will be differentiated. Gradient sources must be part of the tape's graph: tensors registered via `Tape::new` or produced by ops inside the tape.

## Higher-Order Derivatives

Autograd is an **append-only transform on the tape**: `gradient()` doesn't consume or rewrite anything — it appends backward ops to the same graph, and the gradients it produces are ordinary tensors on the tape. Higher-order derivatives are just another append: differentiate the gradient like any other tensor.

```rust,ignore
let x = Tensor::randn([2, 3], DType::F32)?;
let tape = Tape::new([&x])?;
let z = x.relu();
let g = tape.gradient(&z, vec![&x]);
let h = tape.gradient(&g[0], vec![&x]);  // second derivative
```

Many properties of the autograd system fall out of this fact:

- **No separate backward graph** — backward ops live in the same graph as forward ops, so they get the same treatment: CSE, fusion, and kernel caching apply to backward code for free.
- **Gradients are first-class tensors** — they can feed further computation, be stored, or be differentiated again.
- **Nothing is snapshotted or frozen** — since the transform only appends, the tape needs no copy of the forward graph and no special "backward mode".
- **Memory behavior is uniform** — graph ops accumulate until `realize()` or drop, whether they came from forward ops or backward ops.

## Memory Efficiency

Inside a tape, intermediate tensors needed for backpropagation are not held in memory until realize time. The tape stores only `TensorId` values — not the actual data. When the tape is dropped, all tape-preserved ops are released.
