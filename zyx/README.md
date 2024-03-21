# Zyx

Zyx is machine learning library written in Rust.

For README and source code, please visit [github](https://www.github.com/zk4x/zyx).

Just a quick taste:
```shell
cargo add zyx-opencl;
cargo add zyx-optim;
cargo add zyx-nn;
```
```rust
let dev = zyx_opencl::device()?;

let l0 = dev.linear(3, 1024);
let l1 = dev.linear(1024, 2);

let x = dev.tensor([2, 3, 1]).cast(DType::F32);
let target = dev.tensor([2, 4]);

let mut optim = zyx_optim::SGD {
    learning_rate: 0.01,
    momentum: 0.9,
    nesterov: true,
    ..Default::default()
};

let train_steps = 100;
for _ in 0..train_steps {
    let y = l0.forward(&x).relu();
    let y = l1.forward(&y).sigmoid();
    let loss = y.mse_loss(&target):
    loss.backward(l0.into_iter().chain(l1.into_iter()));
    optim.update(l0.into_iter().chain(l1.into_iter()));
}

l0.into_iter().chain(l1.into_iter()).save("my_net.safetensors");
```

For more details, there is a [book](https://zk4x.github.io/zyx).
