# First tensors

In this chapter we go over tensor initialization and running your first operations.

## Choosing your backend

Zyx automatically chooses the best backend for you.

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

