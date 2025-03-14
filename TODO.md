- [x] deallocation of memory
- [x] fix bitcasting
- [ ] implement sharding
- [ ] write optimizer that will reduce number of elementwise ops in IR
      or increase their number, but reduce number of used registers,
      reducing register pressure to avoid spilling,
      this will depend on number of available registers
- [x] padded view permute
- [x] fix mean
- [x] fix var
- [x] deduplication of nodes in graph before graph compilation
- [ ] look at generated kernels, find small kernels, figure why are they small and how to make them bigger
- [x] scheduler cross device movement
- [ ] disk memory pool implemented as backend without devices
- [ ] run kernel launch function on separate thread/threads
- [x] dynamic loading of backends at runtime
- [x] test reshape split
- [x] repeat
- [ ] hsa/hsail backend
- [ ] vulkan backend
- [x] clean up ir.rs
- [ ] automatic optimizations for scheduler
- [x] depends on function for binary ops resolution
- [x] multiple tensors depending on single one (rc > 1)
- [x] multiple tensors depending on single one (rc > 1) for binary ops
- [x] padding
- [x] fix split dimensions for views with large padding
- [x] permute of padded view
- [x] reshape axis split
- [x] local memory work size
- [x] vops renumbering which allows us to set IRMem id to u8 and then directly generate assembly (PTX, HSA)
- [x] multiple memory pools
- [x] device work scheduler that creates graph of kernels, shards them (if needed) and schedules them to devices and memory pools
- [x] Just write the scheduler, temporary variables and stuff does not matter whatsoever
- [x] multiple kernel executors (with different performance)
- [ ] PTX compiler
- [ ] comgr compiler instead of broken hiprtc, although amd's compute firmware is broken,
      so isn't vulkan better anyway? - not on CDNA, but can we even test that?
- [x] ability to use env vars in code blocks
- [x] uniform function
- [x] compiled graph execution performance metrics
- [x] reorder unary and movement ops
- [x] constant folding
- [x] fix search for device with const nodes in graph
- [x] get function
- [x] pad
- [x] cat
- [x] const node
- [x] dot graph of all nodes
- [x] backpropagation
- [x] scalar casting
- [ ] lower/upper triangle mask (for attention)
- [ ] conv
- [x] wgsl init
- [x] wgsl memory copy
- [x] wgsl compilation
- [x] wgsl program launch
- [x] tensor split op
- [x] simplify vops
- [x] tensor id agnostic optimizations for vkernel (rewrite vkernel just before optimizing it)
- [x] local memory tiling
- [ ] register tiling
- [x] tensor arange initialization
- [x] fix weird bug in graph realization when accessing multiple graphs concurrently (delete tensors did not delete shapes and dtypes)
- [x] fix another weird bug in graph realization when accessing multiple graphs concurrently - perhaps this is due to the way
- [ ] runtime should be multithreaded using channels, so that backends are accessed only from one thread
      allocation works and perhaps we are just not using the correct queues and memory pools, if that is the case,
      multi device graph execution is bugged
- [x] deinitialize the runtime once number of tensors drops to zero in the Runtime::release function
- [x] fix double free opencl command queue
- [ ] amd WMMA cores
- [ ] nvidia tensor cores
- [x] better fusion by making expand add new loops to the end of the kernel (expand now requires reshape, so this is no longer needed)
- [ ] better fusion by making reshape add new loops to the end of the kernel
- [x] fix bugs after binary rewrite
- [ ] faster softmax
- [ ] fast attention
- [x] go over all todos in source code and check which are necessary
- [x] fix bug in scheduler when testing cumsum (it was a bug in view)
- [x] rewrite the AI generated docs, they are horrible
- [x] saving and loading for models from and to disk in safetensors format
- [x] automatic optimizations with search on vkernel
- [x] saving of searched kernel to disk
- [x] remove events and instead use queues/cuda streams to launch multiple kernels concurrently with clFinish/cudaStreamSynchronize
- [x] full reduce
- [x] more work per thread
- [x] more work per thread in reduce dim
- [x] bigger accumulators
- [x] deallocation of intermediate buffers in scheduler
- [x] check caching of compiled graphs
- [x] tensor detach (for recurrent nets)
- [x] unsqueeze
- [x] stack
- [x] pool
- [x] cumsum
- [x] randn
- [x] cuda backend
- [x] hip backend
- [x] remove smadd, amadd
- [x] fix reshape after reduce
- [x] todo fix permute on reduced and reshaped kernel
- [x] test pad after reshape after pad
- [x] test pad after reduce
- [x] test padded view permute
- [x] test expand after reduce
- [x] documentation for tensor functions
- [x] documentation for Zyx Error
- [x] documentation for DType
- [x] documentation for Scalar
- [x] documentation for IntoShape
- [x] std::error::Error for ZyxError
- [x] fix softmax
- [x] IR needs to handle single tensor id in multiple loop levels

### Tests

- [x] unary
- [ ] unary backprop
- [x] binary
- [ ] binary backprop
- [ ] movement
- [ ] movement backprop
- [ ] reduce
- [ ] reduce backprop
- [ ] combination of unary and binary
- [ ] combination of movement and unary
- [ ] combination of movent, unary and binary
- [x] combination of all ops
- [x] fuzzy tester with simple cpu tensor, mostly takes care of all combination testing. The longer it runs, the more certain we can be there are no bugs.
- [ ] we need tests for big modules like transformer. If transformer gives correct outputs, it's likely everything else is correct too.

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
