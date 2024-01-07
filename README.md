# Zyx

Zyx is machine learning library written in Rust.
Most ML algorithms can be created with tensors and operations on them.
Tensors must be stored somewhere. Zyx can use both RAM and VRAM (on the gpu) to store tensors.
This is how you create gpu backend. First add opencl backend as dependency.
```shell
cargo add zyx-opencl
```
And then create the device in your code.
```rust
let dev = zyx_opencl::default()?;
```
Now you can create Tensors.
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
let grads = z.backward([&x, &y]);
```
You can also calculate higher order derivatives.
```rust
let grad_xx = grads.next().unwrap().backward([&x]);
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
As for the backends, OpenCL and WGPU backend automatically fuse operations and create custom kernels.
In case of hardware where these backends seem slow you can always use libtorch backend.
Cpu backend is slow and should not be used. It currently serves only as reference backend.

Benchmarks always have some bias, because everybody uses libraries differently.
Still, I will throw a few in here, you can find them all in examples folder.
Numbers show time taken in seconds for 100 iterations.

| Library   | Zyx  |        |      |       | PyTorch |      | Burn |      |
|-----------|------|--------|------|-------|---------|------|------|------|
| Backend   | CUDA | OpenCL | WGPU | Torch | CPU     | CUDA | CPU  | WGPU |
| linear    | 0.0  | 1.0    | 4.5  | 6.5   |
| attention | 1.2  | 3.4    |

## Syntax

Initialize devices (you need to add appropriate crate to you project).
```rust
let opencl = zyx_opencl::default();
let wgpu = zyx_wgpu::default();
let cpu = zyx_cpu::default();
let cuda = zyx_cuda::default();
let torch = zyx_torch::default();
```
Initialize tensors.
```rust
let x = dev.tensor([[2, 3, 1], [5, 2, 8]]);
let x = dev.randn([3, 2, 1], DType::F32);
let x = dev.uniform([3, 2, 1], -1., 27.);
let x = dev.zeros([3, 2, 1], DType::F32);
let x = dev.ones([3, 2, 1], DType::I32);
let x = dev.full([3, 2, 1], 4.);
let x = dev.eye(3, DType::F16);
```
Index tensors.
```rust
let x = dev.randn([2, 3, 4, 5], DType::F32);
let z = x[[.., 2.., 1..-2, ..-1]];
let v: f32 = x[[1, 2, 1, -1]].item();
```
IO operations.
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

## Notes 🤔

Unlike other libraries, Zyx does not require identifying which tensors will be differentiated beforehand,
that means there is no traced() or requires_grad. Everything is automatically handled with performance in mind.
Behind the scenes Zyx actually waits with execution of the graph until it knows if it needs to store intermediate
tensors for backpropagation or not, and so it can immediately release unused memory.

## Contributing

See CONTRIBUTING.md

## Questions

There is matrix chat: ...

## Thanks ❤️

For contributing to Zyx, finding bugs and using it in your ML models.
