- [x] deallocation of memory
- [ ] implement sharding
- [x] padded view permute
- [ ] more tests
- [x] scheduler cross device movement
- [ ] disk memory pool implemented as backend without devices
- [x] dynamic loading of backends at runtime
- [x] test reshape split
- [x] repeat
- [ ] hsa/hsail backend
- [ ] clean up ir.rs
- [ ] automatic optimizations with search
- [ ] hip/rocr backend

## Release blockers

- [ ] local memory caching
- [ ] bigger accumulators
- [x] deallocation of intermediate buffers in scheduler
- [ ] check caching of compiled graphs
- [ ] pool
- [ ] cumsum
- [ ] randn
- [ ] conv
- [x] cuda/ptx backend
- [ ] test backpropagation
- [ ] some tests
- [x] remove smadd, amadd
- [ ] test pad after reduce
- [ ] test padded view permute
- [x] fix reshape after reduce
- [ ] test expand after reduce
- [x] todo fix permute on reduced and reshaped kernel

## Advanced graph caching

Currently this is left to the user. User must explicitly call realize on tensors that should be cached or constant folded.
Later we would like to get constant folding and automatic detection of repeating kernels working, so that user never
needs to call realize function and realize can be deprecated.

## Scheduler optimizations

Scheduler splits graph into kernels each kernel is optimized and compiled for specific device.
Scheduler also assigns these kernels to devices and determines memory allocation and copies between devices.

Thus optimizations can happen in three ways:
1. Kernel creation - how many and which operations get fused into one kernel
2. Device side - how kernels are assigned to devices and how memory gets moved around
3. Kernel side - how each kernel gets optimized for each device

### Kernel creation

We need to make better reshape axis splittingn and combine it with expand op in order to make most
models fuse pretty much into single kernel.

### Device side

First get automatic sharding working, then we can see what more can be done.

### Kernel side

Optimizations are done in these ways:
1. Tiling - that is local memory, register, warp tiling, etc. Many different levels of tiling, automatic search
2. Loop unrolling - more or less straight forward, just unroll small loops and convert dtypes into native vector dtypes, automatic search
3. Permuting loops - automatic search, cache best kernels on disk
4. Merging or splitting loops - automatic seach
5. Padding - for kernels with irregular shapes to make them big enough and dimensions to be power of 2, automatic search
6. Multi step reduce - big reduce loops can be split into two steps with use of intermediate sum/max buffers, again tiled in global/local/register
7. Special algorithms - like running max in softmax in fast attention
8. Device specific optimizations - wmma, tensor cores, native ops (like mad) etc.

In scheduler they are written this way:
Kernel functions
```rust
impl Kernel {
    fn pad_loop(&mut self, op_id: usize, lp: isize, rp: isize)
    fn split_loop(&mut self, op_id: usize, dimensions: &[usize])
    fn merge_loops(&mut self, op_id: usize, num_loops: usize)
    fn permute_loops(&mut self, op_id: usize, axes: &[Axis])
    fn vectorize_loop(&mut self, op_id: usize)
    fn multi_step_loop(&mut self, op_id: usize, steps: &[usize])
}
```
All these functions will need to be applied automatically to each loop with automatic search for the fastest version.

As for tiling, each tensor and accumulator (each variable) needs to be represented as tile.
Each tile can exist in each scope at most once. Unary and binary ops always access tiles that exist at register scope.

### IR

When kernels get compiled to IR, currently two things happen:
1. Tensor ids are converted into register ids
2. Views get converted into IR ops

IR should deduplicate all ops if possible are move ops before loops if they can happen outside of loops.

IR should also probably add synchronization barriers automatically just before loads of tiles written by the kernel.
