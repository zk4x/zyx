# First tensors

In this chapter we go over tensor initialization and running your first operations.

## Choosing your backend

Before we can create tensors, we need to choose which device they should be stored in.
The recommended backend is zyx-opencl, so that is what we will use in this book,
but if it does not work for you, or you can't figure out how to install OpenCL runtime
on your OS, you can go with zyx-cpu backend which does not have any dependencies outside
of rust.

```shell
cargo add zyx-opencl
```

Let's initialize the backend.
```rust
use zyx_opencl;
use zyx_opencl::ZyxError;

fn main() -> Result<(), ZyxError> {
    let dev = zyx_opencl::device()?;
    Ok(())
}
```
That's it!

## Tensor #1

Now we can create your first tensor with zyx.
```rust
let x = dev.tensor([[[3, 2]], [[3, 4]]]);
```

Tensor is multidimensional array. We can ask how many dimensions it has.
```rust
assert_eq!(x.rank(), 3);
```
And also what those dimensions are.
```rust
assert_eq!(x.shape(), [2, 1, 2]);
```

Tensors can only hold data of a single type. In this case, it is i32.
```rust
assert_eq!(x.dtype(), DType::I32);
```

