# Disk IO

Zyx does not have special trait for modules. Instead all modules implement IntoIterator<&Tensor> and IntoIterator<&mut Tensor>.

Anything that implements the first trait can be saved.
```rust
let model = Linear::new(1024, 128, DType::F32);

model.save("model.safetensors")?;
```

Zyx uses safetensors format for saving tensors.

Loading is similar.
```rust
let mut model = Linear::load("model.safetensors")?;
```

If you don't know the structure of tensors saved on disks, you can load them like this.
```rust
let tensors = Tensor::load("my_tensors.safetensors")?;
```
