# Runtime

This is explanation of how zyx works under the hood.
Struct runtime holds the global state of zyx.

```rust
struct Runtime {
    graph: Graph,
    compiled_graphs: BTreeMap<Graph, CompiledGraph>,
    opencl: Option<OpenCLBackend>,
    devices: Vec<Device>,
    memory_pools: Vec<MemoryPool>,
    .. // less important fields
}
```

Runtime stores this information:
1. the current graph of tensor ops as called by the user
2. cached compiled graphs
3. all backends
4. all devices, which contain executable programs
5. all memory pools, which contain memory buffers

What runtime does:
1. converts tensor ops into graph nodes
2. tensor backpropagation by adding more nodes to graph
3. stores tensors on devices
4. passes graph into scheduler for compilation and caches compiled graph
5. launches compiled graphs

## Backend model

Each backend needs to provide six structs:

```rust
struct XBackend { .. }
struct XError { .. }
struct XMemoryPool { .. }
struct XBuffer { .. }
struct XDevice { .. }
struct XProgram { .. }
```

XBackend is the global state of each backend.
XError is enum of possible errors.
XMemoryPool is representation of memory pool accessible by that backend.
XBuffer is buffer stored in XMemoryPool.
XDevice is compute device capable of executing XPrograms.

Each backend can allocate memory, deallocate memory, copy memory between devices and host,
compile programs from IRKernels, launch them using devices and release resources that are not in use anymore.
