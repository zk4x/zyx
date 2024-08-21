- [ ] check caching of compiled graphs
- [x] deallocation of memory
- [ ] deallocation of intermediate buffers in scheduler
- [ ] implement sharding
- [x] padded view permute
- [ ] test padded view permute
- [ ] cuda backend
- [ ] ptx compiler
- [ ] more tests
- [ ] local memory caching
- [ ] bigger accumulators
- [x] scheduler cross device movement
- [ ] disk memory pool implemented as backend without devices
- [x] dynamic loading of backends at runtime
- [x] test reshape split
- [ ] randn
- [ ] cumsum
- [ ] pool
- [ ] conv
- [x] repeat
- [ ] remove duplicated code from IR


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

### IR

When kernels get compiled to IR, currently two things happen:
1. Tensor ids are converted into register ids
2. Views get converted into IR ops

IR should deduplicate all ops if possible are move ops before loops if they can happen outside of loops.
