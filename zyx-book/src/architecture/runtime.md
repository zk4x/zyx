# Runtime and Scheduler

The runtime (`runtime.rs`) is the global state of zyx — the graph, devices, buffers, and kernel cache all live here.

## The Runtime

```rust,ignore
pub(crate) struct Runtime {
    pub graph: Graph,
    pub devices: Slab<DeviceId, Device>,
    pub pools: Slab<PoolId, MemoryPool>,
    pub buffer_map: Map<TensorId, BufferId>,
    pub events: Map<BTreeSet<BufferId>, Event>,
    pub kernel_cache: KernelCache,
    pub rng: Rng,
    pub autotune_config: AutotuneConfig,
    pub debug: DebugMask,
    pub training: bool,
}
```

The runtime is stored in a global `Mutex<Runtime>`:

```rust,ignore
static RT: Mutex<Runtime> = Mutex::new(Runtime::new());
```

Every tensor operation locks this mutex, appends a graph node (microseconds), and releases. The lock is never held during computation — only during graph manipulation.

## Memory Pools

Each backend provides a `MemoryPool` for allocating device buffers:

```rust,ignore
pub enum MemoryPool {
    Host(HostMemoryPool),
    Disk(DiskMemoryPool),
    C(CMemoryPool),
    CUDA(CUDAMemoryPool),
    // ...
}
```

## The Scheduler (Current)

The current scheduler (`schedule.rs`) selects a device for kernel execution by calculating required memory, sorting devices by free compute capacity, and picking the first with enough free memory. It handles cross-device transfers via events.

## New Scheduler (In Development)

A new scheduling approach is under development in `search.rs` and `search2.rs`. It will use an e-graph-like budget-guided exhaustive fusion enumeration, including costs for memory movement operations — replacing the current simple heuristics with exploration of all fusion configurations within a cost budget.

## Async Execution

Events track kernel completion:

```rust,ignore
events: Map<BTreeSet<BufferId>, Event>
```

## Lazy Device I/O

Tensors can reference data on disk without loading it:

```rust,ignore
let t = Tensor::from_safetensors("model.safetensors", "layer.weight")?;
```

The disk pool keeps file offset information. Data is loaded lazily when the tensor needs to be realized on a compute device.
