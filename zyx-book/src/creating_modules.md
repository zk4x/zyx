# Creating Modules

```shell
cargo add zyx-nn
```

Zyx only has statefull modules. That is all modules must store one or more tensors. One of the simplest modules
is [linear layer](https://docs.rs/zyx-nn/latest/zyx-nn/struct.Linear.html).

In order to initialize modules, you need a device. Modules have traits implemented for all backends to allow for more ergonomic API:
```rust
use zyx_nn::Linear;
let l0 = Linear::new(1024, 128, DType::F32);
```

## Custom Modules

Custom modules are easy to create, you only need to import Backend trait from core.
```rust
struct MyModule {
    l0: Linear,
    l1: Linear,
}
```

For modules to be useful, they need forward function.
```rust
impl MyModule {
    fn forward(&self, x: impl Into<Tensor>) -> Tensor {
        let x = self.l0.forward(x).relu();
        self.l1.forward(x).sigmoid()
    }
}
```
Since relu is stateless, it is not a module, it is just a function on tensor.

Modules can be initialized with any device.
```rust
let dev = zyx_cpu::device()?;

let my_module = MyModule {
    l0: Linear::new(1024, 512, DType::F32),
    l1: Linear::new(512, 128, DType::F32),
};
```

Also you need to implement IntoIterator<Item = &Tensor> to be able to easily save and IntoIterator<Item = &mut Tensor>
to backpropagate over parameters of the module and to load these parameters into the model.
```rust
impl<'a> IntoIterator for &'a MyModule {
    type Item = &'a Tensor;
    type IntoIter = impl IntoIterator<Item = Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.l0.into_iter().chain(self.l1)
    }
}

impl<'a> IntoIterator for &'a mut MyModule {
    type Item = &'a mut Tensor;
    type IntoIter = impl IntoIterator<Item = Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.l0.into_iter().chain(self.l1)
    }
}
```

Both implementations of IntoIterator could be done using procedural macro Module.
So you can choose this simpler method if you prefer.
```shell
cargo add zyx_derive
```
```rust
#[derive(Module)]
struct MyModule {
    l0: Linear,
    l1: Linear,
}
```

Forward function is used for inference.
```rust
let input = Tensor::randn([8, 1024], DType::F32);

let out = my_module.forward(&input);
```

Backpropagation is provided automatically.
```rust
let input = Tensor::randn([8, 1024], DType::F32);
let label = Tensor::randn([8, 128], DType::F32);

let epochs = 100;
for _ in 0..epochs {
    let out = my_module.forward(&input);
    let loss = (out - label).pow(2);
    loss.backward(&my_module);
}
```
