# Optimizers

```shell
cargo add zyx-optim
```

Optimizers take gradients calculated as your loss w.r.t. your parameters and update those parameters
so that the next time you run your model with the same inputs, the loss will be lower.

```rust
let mut optim = zyx_optim::SGD { ..Default::default() };
let grads = loss.backward();
optim.update(&mut model, grads);
```

Zyx has multiple optimizers. All are accessible from crate [zyx-optim](https://docs.rs/zyx-optim/latest/zyx-optim/index.html).
