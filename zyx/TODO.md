- [ ] backends
  - [ ] cuda
    - [x] fix async memcopy
    - [ ] tensor cores
    - [ ] fix load calculation, probably using Atomic usize
    - [x] fix event memory leaks, all events must be properly destroyed
  - [ ] hip
  - [x] opencl
    - [ ] fix load calculation, probably using Atomic usize
  - [ ] vulkan
    - [ ] initialization
    - [ ] memory management
    - [ ] spirv compiler (to spirv binary)
    - [ ] kernel launch
  - [x] wgpu
    - [ ] fix load calculation, probably using Atomic usize
    - [ ] spirv compiler
      - [ ] conversion to spirv SSA (dealing with accumulators)
  - [x] dummy
    - [ ] validation for program ids
- [x] runtime
  - [x] fix event handling
  - [ ] node deallocation after realization
  - [ ] static graphs - unfortunately necessary for very high performance networks to achieve millions of tensor ops/second
- [ ] autograd
  - [x] fix t6 test
  - [ ] more backpropagation tests
  - [ ] drop unneded nodes when gradient tape is released
  - [ ] proper realize function with gradient tape
  - [x] proper backprop, since now we don't quite need to calculate requires_grad_nodes, those are now in gradient_tape
- [ ] dtype
  - [ ] quantized dtypes
  - [x] optional implicit dtype casts
- [x] view
  - [x] split on padded view
  - [x] view padding to ir
    - [x] offset
    - [x] padding condition
  - [x] reshaped view to ir
  - [x] axis merging
  - [x] axes reshape
- [x] kernelizer
  - [x] all dim reduce
  - [ ] kernel reshape with shape that contains reduce ops and add new loops after those
  - [x] cache Map<(Kernel, Optimizations), Program> instead of Map<IRKernel, Program>
  - [ ] improve reshape node
    - [x] merges, splits, reshapes of non reduce axes
    - [ ] inserting new loops to the end of the kernel
  - [ ] pad could also work even with kernels that store stuff, just pad the store view
  - [x] expand reduce bug
  - [x] fix is expandable conditions
  - [ ] tests for fusion, test will create it's own graph and check how the fused kernel looks
    - [ ] softmax fusion test (eventually should be single kernel)
    - [ ] just asserts that various graphs fuse into single kernel
  - [x] scheduling to multiple devices
  - [x] fix bug when running phi3, panic on min_kernel function
  - [ ] automatic sharding across devices
- [x] kernel
  - [x] default optimizations
  - [x] indexing for padded views
  - [x] indexing for multi reshape views
  - [ ] vectorization, vector dtypes
  - [ ] common subexpression elimination/deduplication
  - [x] dead store elimination
  - [ ] loop unrolling
  - [ ] loop splitting
  - [ ] loop reordering
  - [ ] loop invariant code motion
  - [ ] merge all mul + add into mad instructions
  - [ ] register tiling of all variables
  - [ ] local tiling of all variables
  - [ ] flash attention
  - [ ] optimizer with tree search
- [ ] testing
  - [ ] lot of testing for scheduler correctness
  - [ ] fuzzy tester
    - [x] unary ops
    - [ ] movemnt ops
    - [ ] binary ops

- [x] docs
  - [x] manual for adding new backends
- [x] dependencies
  - [x] replace serde with nanoserde
  - [x] implement custom progress bar
  - [x] remove indicatiff

- examples
  - [x] get phi working
    - [ ] fix tensor memory leak

- tests
  - [ ] padding on elementwise kernel
  - [ ] expand on elementwise kernel
  - [ ] reshape on elementwise kernel
  - [ ] permute on elementwise kernel
  - [ ] padding on reduce kernel
  - [ ] expand on reduce kernel
  - [ ] reshape on reduce kernel
  - [ ] permute on reduce kernel


## Architecture

We need to support both dynamic and static graph. Once the graph is created by applying ops, it can be stored as static graph, or interpreted dynamically.
Both static and dynamic graphs are send to kernelizer to create kernels. Then these kernels are scheduled. In case of static graph, there will be static scheduler,
that will create static graph with only compiled kernel launch and memory operation instructions. In case of dynamic graph, there will be interpreter
directly interpreting kernelized graph. It assigns kernels to available devices.

Perhaps it's best to just compile all kernels for all devices. Then scheduler will take a list of kernels, which kernel depends on which other kernels and list of tensors.
This can be the same for static and dynamic graph. There is probably little we can get from doing this all ahead of time, probably just best to check the load on the devices
and assign kernels appropriatelly even in static graphs, instead of assigning automatically always to the same devices.


## Final architecture

### Dynamic graphs

So user creates graph. Then orders it's realization.
Graph interpreter:
Kernelizer creates kernels. Kernels are scheduled to devices, compiled for those devices, launched and compilation results cached. This is done repeatedly for each kernel.

### Static graphs

User creates graph. Then orders it's compilation.
Graph compiler:
Kernelizer creates kernels. Kernels are scheduled to devices all at once. Then kernels are compiled all at once. Then compiled kernel ids and buffer ids are stored in graph as list
of kernel launches and memory operations. This static graph can then be executed at any time. Then we just need to increase ref count for included tensors, so that there are no
conflicts with dynamic graphs.

## For both

In both cases kernel compilation includes:
- kernel caching
- kernel optimization

Check if kernel requires further optimization:
  - no - use cached kernel
  - yes - optimize it:
    while we haven't tried enough optimizations:
      1. optimizer picks optimization
      2. kernel with optimization is lowered to ir and compiled for selected device
      3. repeat
