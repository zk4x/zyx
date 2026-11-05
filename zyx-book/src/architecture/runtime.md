# Runtime and Plans

The runtime (`runtime.rs`) is the process-wide state of zyx: graphs, tensor
states, kernels, compiled programs, measured timings, interned shape
expressions, and the plan cache. Memory pools belong to the backends, not to
the runtime — they outlive it.

## The Runtime

```rust,ignore
pub struct Runtime {
    pub graphs: Slab<GraphId, Graph>,
    pub tensors: Slab<TensorId, TensorData>,
    pub kernels: Slab<KernelId, KernelData>,
    kernel_map: Map<Kernel, KernelId>,          // content-dedup of kernels
    programs: Map<KernelId, DeviceProgramId>,   // compiled programs
    timings: Map<ProgramId, u64>,               // measured nanos
    pub(crate) expr_hash: Map<Expr, ExprId>,    // hashcons of shape expressions
    pub exprs: Slab<ExprId, Expr>,              // append-only interned exprs
    pub rng: Rng,
    pub implicit_casts: bool,
    pub training: bool,
    pub plan_cache: Map<u64, Plan>,
    #[cfg(feature = "viz")]
    pub viz: Viz,
}
```

The runtime is stored in a global mutex:

```rust,ignore
static RT: mutex::Mutex<Runtime> = mutex::Mutex::new(Runtime::new());
```

Every tensor operation locks it. The lock is coarse: it covers graph mutation,
and also the whole realize path — `Tape::realize` holds it while compiling the
plan, replaying launches, and converting outputs to eager tensors. It is not a
fine-grained per-tensor lock, and nothing re-enters it (helpers that need the
runtime take `&mut Runtime` rather than locking again). Backend pools and their
device worker threads are separate, process-wide globals.

Shape expressions are interned: `ExprId`s are stable forever, structural
equality is one `ExprId` compare, and symbolic dims never live in the tensors
slab. Tensor handles are `u32` indices into `tensors`.

## Tensor States

A `TensorData` is one of:

| State | Meaning |
|-------|---------|
| `Symbolic` | An interned `Expr` (a symbolic scalar, dim, or shape) |
| `Leaf` | A realized value with a buffer placement |
| `PendingLeaf` | A realized value still owed by a kernel that has not launched |
| `Eager` | An unrealized value living in a kernel's op chain |
| `Graph` | A value that lives only in a graph class (unrealized) |
| `GraphLeaf` | A graph leaf backed by a buffer |
| `Promoted` | A graph tensor whose eager producer stays alive |

`item()` forces an eager value: the producer kernel is materialized and
launched. `shape()` does not force; symbolic shapes are resolved without
allocating.

## Memory Pools and Placements

Pools are process-wide and never deinitialized:

```rust,ignore
pub enum Pool {
    Host,          // RAM, shared by C, CBLAS and Host
    Disk,          // file-backed chunks (paths, not bytes)
    Cuda(u16),
    OpenCL(u16),
    Vulkan(u16),
    TT(u16),
    WGPU(u16),     // feature = "wgpu"
}
```

Each backend owns its globals — one mutex per pool ordinal — and lazily
initializes them on first use. A value is a `Placement`:

```rust,ignore
pub struct Placement { pub shards: Vec<Shard> }

pub struct Shard {
    pub pool: Pool,
    pub chunk: ChunkId,
    pub offset: usize,
    pub len: usize,
}
```

One shard is the common case; several shards describe a value split across
devices. `Placement`'s `Drop` returns each chunk to its pool's free list;
`Pool::release` frees nothing by itself.

Devices and pools are two views of the same hardware: `Dev::pool()` derives the
pool from the device, never the reverse.

## Device Selection

`Runtime::pick_device(bytes)` is the generic placement rule:

1. start from `Dev::all()` — lazy init of every compiled-in backend, cached in a
   `OnceLock`; backends that are configured out or whose hardware/driver is
   missing contribute nothing;
2. drop AOT-only devices (`Dev::aot_only()`, e.g. CBLAS) for generic kernels;
3. keep devices whose pool has at least `bytes` free;
4. sort by `free_compute()` descending and take the first.

If no device has enough free memory the call is an allocation error. A user can
pin a device with `ZYX_DEV=cpu|cuda|vulkan|...` (parsed by `Dev::from_str`) or by
calling `Tensor::to(Dev)` explicitly; `Dev::Auto` resolves through the rule
above.

## Plans

Every launch is a `Plan` — a scheduled `CmdQueue`:

```rust,ignore
pub enum Cmd {
    Launch { program: ProgramId, args: Vec<OpId>, outputs: Vec<(OpId, DType, Vec<PlanDim>)> },
    Copy   { src: OpId, dst: OpId, dst_pool: Pool, dst_dtype: DType, dst_dims: Vec<PlanDim> },
    Alias  { class: OpId, to: OpId },
}
```

Slots are queue-local `OpId`s. `PlanDim` is a symbolic output size built from
constants and leaf classes; it is evaluated at replay against `class_vars`.

`CmdQueue::schedule(outputs)`:

- slices the queue into maximal same-device launch runs; each run is scheduled
  by its own device;
- executes every `Copy` on the **destination** device (device runs never copy);
- joins `Alias` bindings to the surrounding run as zero-cost map entries;
- pins each run's outputs plus the slots read after it, and frees everything else
  at last use.

`Plan::replay(boundary, class_vars)` binds the boundary table to `Placement`s,
allocates every unbound output from its `PlanDim`s, resolves aliases through the
boundary, and returns the placements of the plan's outputs. No allocation or
liveness decision was made at schedule time beyond this structure.

### The plan cache

`Runtime::plan_cache` is keyed by the graph's content hash (structure + outputs)
folded with the pool of every leaf class at call time. Two realizations of the
same graph shape share a plan only when their leaf pools match; otherwise the
plan recompiles. This is what makes a tape loop cheap after the first iteration:
the kernels are already compiled and the plan is already scheduled, so later
iterations only replay it with new leaf values and (possibly) new variable dims.

### Eager execution

An eager kernel follows the same route with a one-kernel queue. On
materialization the runtime:

1. verifies argument arity;
2. `linearize`s, runs CSE/DCE/instruction scheduling;
3. merges group indices down to the device's `max_global_work_dims` and
   renumbers them;
4. `verify`s;
5. runs `delete_useless_loops` and three rounds of `default_epilogue`;
6. runs `BeamSearch` (or `NoSearch` under `ZYX_DEBUG=256`);
7. stores the winning program and its measured time.

The resulting plan is executed immediately. Graph kernels go through the same
compile path via `autotune_jit_kernels`, but their programs are recorded as
`Op::Program` nodes and extraction decides whether they run.

## Async Execution

Synchronization is per backend and lives below `Plan`. A plan replay issues
launches and copies in program order; each backend decides how far it can run
ahead. CUDA, for example, owns each device with a worker thread behind a command
channel: launches accumulate in a micro-batch window, are distributed over
several hardware queues with computed stream dependencies, and are submitted
together. There are no runtime-level event objects — ordering is expressed by
the plan (`After`, copy direction, argument dependencies) and enforced by the
backend worker.

## Lazy Device I/O

A tensor can describe data that still lives on disk. `Runtime::new_disk_tensor`
creates a `Leaf` whose shard is a `Pool::Disk` chunk holding a path and an offset
instead of bytes. When a plan needs the value on a compute device, the copy
command stages it there; the disk pool keeps the file offset information, so
nothing is read until the value participates in a launch.

The eager loader is `Module::load_safetensors(path)`, which reads a
safetensors file into a `HashMap<String, Tensor>`; disk-backed leaves are the
mechanism underneath for deferring that read.

## Lifecycle

`Runtime::deinitialize` runs at process exit. With the `time` feature it prints
the accumulated timing table first, then drops the tensor and kernel slabs.
Backend pools are static and are not deinitialized — the process owns the
hardware for its lifetime. `Runtime::manual_seed` reseeds the runtime RNG.
