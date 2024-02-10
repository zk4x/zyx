# Zyx

Zyx is machine learning library written in Rust.
Most ML models can be created with just tensors and operations on them.
Tensors must be stored somewhere. Zyx can use both RAM and VRAM (on the gpu) to store tensors.

This is how you create gpu backend. First add some backend as dependency.
```shell
cargo add zyx-opencl
```
And then initialize the device in your code.
```rust
let dev = zyx_opencl::device()?;
```
Now you can create tensors.
```rust
let x = dev.randn([2, 2], DType::F32);
let y = dev.tensor([[2., -3.], [4., 1.]]);
```
Let's do some mathematical operations on these tensors.
```rust
let z = (&x + &y).exp() - &x;
```

## Automatic differentiation 👍

Zyx allows for arbitrary differentiation. Using the previously created tensors we calculate derivative of z w.r.t. x and y.
```rust
let (grad_x, grad_y) = z.backward([&x, &y]).flatten().collect_tuple().unwrap();
```
You can also calculate higher order derivatives.
```rust
let grad_xx = grad_x.backward([&x]).unwrap();
```

## Optimizers ⚙️

Optimizers optimize something (usually minimize loss). For this they need to know what is the derivative of loss w.r.t. model's parameters.
```rust
let optimizer = zyx_nn::SGD::new();
let loss = (model.forward(&x) - label).pow(2);
let grads = loss.backward(&model);
optimizer.update(&mut model, grads);
```
Optimizer updates model's parameters with gradients.

## Performance 🚀

Thanks to its execution model, Zyx always uses minimum amount of RAM.
As for the backends, OpenCL backend automatically fuses operations and create custom kernels.
In case of hardware where these backends seem slow you can always use libtorch backend.
Native (Rust) backend is slow and should not be used. It currently serves only as reference backend.

## Syntax

Initialize devices (you need to add appropriate crate to you project).
```rust
let opencl = zyx_opencl::default();
let cpu = zyx_native::default();
let torch = zyx_torch::default();
```
Initialize tensors.
```rust
let x = dev.tensor([[2, 3, 1], [5, 2, 8]]);
let x = dev.randn([3, 2, 1], DType::F32);
let x = dev.uniform([3, 2, 1], -1..27);
let x = dev.zeros([3, 2, 1], DType::F32);
let x = dev.ones([3, 2, 1], DType::I32);
let x = dev.full([3, 2, 1], 4.);
let x = dev.eye(3, DType::F16);
```
Index tensors.
```rust
let x = dev.randn([2, 3, 4, 5], DType::F32);
let z = x.get([.., 2.., 1..-2, ..-1]).unwrap();
let v: f32 = x.get([1, 2, 1, -1]).unwrap().item().unwrap();
```
IO operations.
```shell
cargo add zyx-io
```
```rust
model.save("model.safetensors");
model.load("model.safetensors");
```
Custom models.
```shell
cargo add zyx-nn
```
```rust
use zyx_nn::{IntoTensor, Tensor};

struct MyModel<B> {
    l0: zyx_nn::Linear<B>,
    l1: zyx_nn::Linear<B>,
}

impl<B> MyModel<B> {
    fn forward(&self, x: impl IntoTensor<B>) -> Tensor<B> {
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

Unlike other libraries, Zyx does not require identifying which tensors will be differentiated beforehand,
that means there is no traced() or requires_grad. Everything is automatically handled with performance in mind.
Behind the scenes Zyx actually waits with execution of the graph until it knows if it needs to store intermediate
tensors for backpropagation or not, and so it does not unnecessarily allocate memory.

## Contributing

See CONTRIBUTING.md

## Questions

There is matrix chat: ...

## Thanks ❤️

For contributing to Zyx, finding bugs and using it in your ML models.
