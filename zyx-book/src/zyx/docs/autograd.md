# Automatic differentiation

Everything that is differentiable in math is differentiable in zyx (sometimes even functions that are not differentiable in math, like ReLU at 0).

## Example

You can just do any operations with your tensors.
```rust
# use zyx_opencl;
# let dev = zyx_opencl::device()?;
let x = dev.randn([1024, 1024], DType::F32);
let y = dev.tensor([2, 3, 1]);
let z = (x + y.pad([(1000, 21)], 8)) * x;
```
At any point in time, you can differentiate any tensor w.r.t. any other tensor or set of tensors. This example differentiates
z w.r.t. x and y.
```rust
let grads = z.backward([&x, &y]);
```

Backward function return Vec<Option<Tensor>>. Grads can contain some None values, if there is no direct connection
between dependent and independent variables.
```rust
let x = dev.randn([2, 3], DType::F32);
let y = dev.randn([2, 3], DType::F32);
let z = y.exp();
let grads = z.backward([&x]);
assert_eq!(grads, vec![None]);
```

## Performance

Some other ML libraries require users to provide additional information in order to make backpropagation efficient. This
is needed, because they store intermediate tensors in memory in order to be able to backpropagate. Zyx does not store
anything in memory. Instead, zyx stores a graph of operations, which takes just a few MB even for millions of tensors
and calculation is done only when user accessses the data, for example when saving to the disk. During backpropagation
the graph is traversed and new tensors are added to the graph.
