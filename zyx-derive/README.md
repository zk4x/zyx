# zyx-derive

This crate contains procedural macros for zyx.

Macro Module automatically implements IntoIterator<Item = &Tensor>
for your module, so that you can use it in backpropagation and save it to disk.
```rust
use zyx_core::backend::Backend;
use zyx_core::tensor::Tensor;

#[derive(Module)]
struct MyNet<B: Backend> {
    b: Tensor<B>,
    w: Tensor<B>,
}

impl<B: Backend> MyNet<B> {
    fn forward(&self, x: &Tensor<B>) -> Tensor<B> {
        x.dot(self.w) + self.b
    }
}
```

For README and source code, please visit [github](https://www.github.com/zk4x/zyx).

For more details, there is a [book](https://www.github.com/zk4x/zyx/tree/main/zyx-book).

# Features

- std - enables zyx-core/std
