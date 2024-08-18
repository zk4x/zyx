# Zyx

Zyx is machine learning library written in Rust. This README is a quick tutorial for people
already familiar with tensors and other machine learning concepts. 
For more comprehensive documentation please see our [book](https://zk4x.github.io/zyx).

## Getting started

Add zyx to your Cargo.toml
```shell
cargo add zyx
```

Now you can create tensors.
```rust
let x = Tensor::randn([2, 2], DType::F32);
let y = Tensor::from([[2., -3.], [4., 1.]]);
```
And do some mathematical operations on these tensors.
```rust
let z = x.dot(&y);
let z = (&x + y).exp() - &x;
```

## Automatic differentiation 👍

Zyx allows for arbitrary differentiation. Using the previously created tensors we calculate derivative of z w.r.t. x and y.
```rust
let grads = z.backward([&x, &y]);
```
You can also calculate higher order derivatives.
```rust
let grad_xx = grads[0].unwrap().backward([&x]);
```

## Optimizers ⚙️

Optimizers are shipped in separate crate.
```shell
cargo add zyx_optim
```
Optimizers optimize something (usually minimize loss). For this they need to know what is the derivative of loss w.r.t. model's parameters.
```rust
let sgd = zyx_optim::SGD { learning_rate: 0.01, momentum: 0.9, ..Default::default() };
let loss = (model.forward(&x) - &label).pow(2);
let grads = loss.backward(&model);
sgd.update(&mut model, grads);
```
Optimizer updates model's parameters with gradients.

## Performance 🚀

Zyx uses minimal number of operations (no matmul, no conv). At runtime these operations are fused and compiled into native kernels.

## Lazyness

Tensors do not get realized automatically. Realization happens only when user accesses tensors, or explicitly using Tensor::realize function.
```rust
Tensor::realize([&x, &y]);
```
If you do not know when to realize tensors, just do it after updating them with optimizer.
```rust
sgd.update(&mut model, grads);
Tensor::realize(&model);
```

## Syntax

Initialize tensors.
```rust
let x = Tensor::from([[2, 3, 1], [5, 2, 8]]);
let x = Tensor::from(0..100).reshape([10, 10]);
let x = Tensor::randn([3, 2, 1], DType::F32);
let x = Tensor::uniform([3, 2, 1], -1..27);
let x = Tensor::zeros([3, 2, 1], DType::F32);
let x = Tensor::ones([3, 2, 1], DType::I32);
let x = Tensor::full([3, 2, 1], 4.);
let x = Tensor::eye(3, DType::F16);
```
Index tensors.
```rust
let x = Tensor::randn([2, 3, 4, 5], DType::F32);
let z = x.get((.., 2, 1..-2, ..-1));
let v: f32 = x.get((1, 2, .., -1)).item().unwrap();
```
IO operations.
```rust
model.save("model.safetensors");
model.load("model.safetensors");
```

## Custom models

Zyx nn provides some modules, also you can use module macro.
```shell
cargo add zyx-nn
cargo add zyx-derive
```
```rust
use zyx::Tensor;
use zyx_derive::Module;

#[derive(Module)]
struct MyModel {
    l0: zyx_nn::Linear,
    l1: zyx_nn::Linear,
}

impl MyModel {
    fn forward(&self, x: impl Into<Tensor>) -> Tensor {
        let x = self.l0.forward(x).relu();
        self.l1.forward(x)
    }
}
```

## Goals

1. Correctness
2. Hardware support
3. Performance

## Notes 🤔

Unlike other libraries, zyx does not require identifying which tensors will be differentiated beforehand,
that means there is no traced() or requires_grad. Everything is automatically handled with performance in mind.
Behind the scenes zyx actually waits with execution of the graph until it knows if it needs to store intermediate
tensors for backpropagation or not, and so it does not unnecessarily allocate memory.

## Contributing

See [CONTRIBUTING.md](https://github.com/zk4x/zyx/blob/main/CONTRIBUTING.md)

## Thanks ❤️

For contributing to Zyx, finding bugs and using it in your ML models.

## Licensing

Dual licensed with MIT and Apache 2.0
