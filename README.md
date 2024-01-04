# Zyx

Zyx is machine learning library written in Rust.
Most ML algorithms can be created with tensors and operations on them.
Tensors must be stored somewhere. Zyx can use both RAM and VRAM (on the gpu) to store tensors.
This is how you create gpu backend. First add opencl backend as dependency.
```bash
cargo add zyx-opencl
```
And then create the device in your code.
```rust
let dev = zyx_opencl::OpenCL::new()?;
```
Now you can create Tensors.
```rust
let x: Tensor<&OpenCL> = dev.randn([2, 2]);
let y: Tensor<&OpenCL> = dev.tensor([[2., -3.], [4., 1.]]);
```
Let's do some mathematical operations on these tensors.
```rust
let z = (&x + &y).exp() - &x;
```

## Automatic differentiation

Zyx allows for arbitrary differentiation. Using the previously created tensors we calculate derivative of z w.r.t. x and y.
```rust
let grads = z.backward([&x, &y]);
```
You can also calculate higher order derivatives:
```rust
let grad_xx = grads[0].backward([&x]);
```
Automatic differentiation is needed for optimizers.

## Optimizers

Optimizers optimize something (usually minimize loss). For this they need to know what is the derivative of loss w.r.t. model's parameters.
```rust
let optimizer = zyx_nn::SGD::new();
let loss = (model.forward(&x) - label).pow(2);
let grads = loss.backward(model.parameters());
optimizer.update(model.parameters(), grads);
```
Optimizer updates model's parameters with gradients.

## Why Zyx

### Autmatic differentiation

Unlike other libraries, Zyx does not require identifying which tensors will be differentiated beforehand,
that means there is no traced() or requires_grad. Everything is automatically handled with performance in mind.
Behind the scenes Zyx actually waits with execution of the graph until it knows if it needs to store intermediate tensors
for backpropagation or not.

### Performance

Thanks to its execution model, Zyx always uses minimum amount of RAM.
As for the backends, OpenCL and WGPU backend automatically fuse operations and create custom kernels.
In case of hardware where these backends seem slow you can always use libtorch backend.

### Ease of use

It is hard to write graphs in Rust. Zyx exposes API where Tensor has only one generic parameter - backend. This mostly
allows for writing model in direct way, without having to deal with lifetimes or too many generics.
