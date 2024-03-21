# Disk IO

Zyx does not have special trait for modules. Instead all modules implement IntoIterator<&Tensor> and IntoIterator<&mut Tensor>.

Anything that implements the first trait can be saved.
```rust
let model = dev.linear(1024, 128);

model.save("model.safetensors")?;
```

Zyx uses safetensors format for saving tensors.

Loading is similar.
```rust
let mut model = dev.linear(1024, 128);

model.load("model.safetensors")?;
```

If you don't know the structure of tensors saved on disks, you can load them like this.
```rust
let dev = zyx_opencl::device();
let tensors = dev.load("my_tensors.safetensors")?;
```
