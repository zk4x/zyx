# Tensor Operations

Zyx supports most important operations used in other ML libraries.

Examples. For full list, please check tensor's [documentation](https://docs.rs/zyx-core/latest/zyx-core/tensor/struct.Tensor.html).
```rust
let x = dev.randn([1024, 1024], DType::F32);
let y = dev.randn([1024, 1024], DType::F32);

let z = x.exp();
let z = x.relu();
let z = x.sin();
let z = x.tanh();

let z = &x + &y;
let z = &x - &y;
let z = &x * &y;
let z = &x / &y;
let z = x.pow(&y);
let z = x.cmplt(&y);
let z = x.dot(&y);
```
