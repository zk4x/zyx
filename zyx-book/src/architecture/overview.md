# Architecture Overview

zyx has two intermediate representations:

- a **tensor graph** — an e-graph of equivalence classes, built lazily inside a
  `Tape`, where fusion variants compete;
- a **kernel IR** — a linear SSA list of ops, the unit that passes optimize,
  autotune searches, and backends compile.

Eager code produces kernel IR directly. Graph code goes through the kernelizer,
which turns structural graph nodes into JIT kernels; those kernels are compiled
and measured, and extraction then chooses the cheapest measured path through the
e-graph.

```text
eager:  Tensor op ──► append to producing kernel ──► compile kernel ──► launch
                                                    (linearize, passes,
                                                     autotune, codegen)

Tape:   Tensor op ──► e-graph ──► kernelizer ──► JIT kernels ──► autotune
                                                                    │
                                              measured times ───────┘
                                                     │
                                            extract cheapest path
                                                     │
                                                Plan (Cmd queue) ──► replay
```

Both paths end in the same place: a `Plan`, which is a scheduled command queue
(launches, copies, aliases) replayed against memory placements.

## Pipeline Stages

### 1. Tensor API

The user creates tensors and applies operations:

```rust
# extern crate zyx;
# use zyx::{DType, Tensor, ZyxError};
# fn main() -> Result<(), ZyxError> {
let x = Tensor::randn([1024, 1024], DType::F32)?;
let y = x.relu();
let z = y * 2.0;
# Ok(())
# }
```

A `Tensor` is a `u32` handle into the runtime slab. Outside a `Tape`, each op is
appended to the kernel that produced its inputs; that kernel is compiled and
launched when its value is first needed, not at every op. Inside a `Tape`, ops
build graph nodes instead, and `Tape::realize` compiles and executes the
subgraph.

### 2. The Graph

The graph is an e-graph: `Graph { hashcons, ops, jit_kernels, leaf_classes,
leaf_map, ... }`. Every node belongs to an equivalence class (`class_of`), and a
class chains its alternative forms through `next_in_class`. Structural
hashconsing makes identical expressions share one class.

The graph and the kernel IR share one `Op` enum (`kernel/ops.rs`). The
graph-level subset is the structural, device-agnostic vocabulary:

| Variant | Meaning |
|---------|---------|
| `Const` | A constant value baked into kernels |
| `Param` | A buffer or scalar argument (a graph leaf) |
| `Expand` | Broadcast a dimension |
| `Permute` | Transpose axes |
| `Reshape` | Change shape without changing data |
| `Pad` | Pad one axis with zeros |
| `Flip` | Reverse one or more axes |
| `Narrow` | Slice one axis |
| `Stack` | Build a vector from operands |
| `Reduce` | Sum, max, etc. |
| `Cast` / `Bitcast` | Change dtype / reinterpret raw bits |
| `Unary` / `Binary` / `Select` | Element-wise arithmetic |
| `After` | Ordering edge between side-effecting values |
| `ToDevice` | Move a value between devices |
| `Contiguous` | Fusion-break hint: materialize an output |
| `Detach` | Autograd-break hint (value identity forward) |
| `Program` | A compiled kernel boundary with a measured time |
| `Kernel` | An opaque user custom kernel |

Kernel-only variants (`Load`, `Store`, `Storage`, `Range`, `Loop`, `Wmma`,
`TT`, ...) are described in [The Kernel IR](./kernel_ir.md). Shape expressions
are interned separately (`Runtime::exprs` / `expr_hash`), so symbolic dims are
cheap to share and compare.

### 3. The Kernelizer

The kernelizer (`graph/kernelizer.rs`) fuses structural graph nodes into
kernels. It walks classes in topological order with reference counts: a class is
fused into the kernel of its consumer when the surrounding ops allow it, and a
value with several consumers is produced once with one output copy per consumer.
Storing a value and reloading it for later consumers creates a load kernel at
that boundary. `fill_gaps` runs the same machinery per connected region so that
AOT kernels (e.g. a cuDNN matmul) and custom kernels can compete with fused
kernels inside the same graph.

Movement ops are not materialized just to be read back. Their index arithmetic
is replayed into the consumer, and the `linearize` pass unfolds it, so a
reshape/expand/permute chain costs index arithmetic rather than extra kernels.

The kernelizer's invariant: after it runs, every output class has a path to the
leaves composed only of `Program` and `ToDevice` nodes.

### 4. Autotune Each Kernel

Every JIT kernel produced by the kernelizer goes through the same compilation
path as an eager kernel:

1. `linearize`
2. readability passes (CSE, DCE, instruction scheduling)
3. global-index merge to the device's limit, renumbering
4. `verify`
5. three rounds of `default_epilogue`
6. `BeamSearch` over `default_optimizations`, ranked by cost and decided by
   measured launches
7. `device_epilogue` (lowering that must not be searched, e.g. `Pow` expansion
   and the Tenstorrent physical IR)
8. codegen

The measured time of the winning kernel is recorded on the graph as an
`Op::Program { info: (ProgramId, nanos) }` node. Its program id stays alive; the
extraction decides whether to use it.

### 5. Extract

`Graph::extract` walks the classes in extraction order and picks, for each
class, the producer with the cheapest **measured** time. There is no estimated
cost model: a kernel competes only after it has been compiled and timed. Leaf
classes are already realized and need no producer.

After extraction, `add_memory_ops` inserts `ToDevice` transfers for the selected
producer/consumer pairs that live on different devices.

### 6. The Plan

Extracted nodes are lowered to a `CmdQueue`:

- `Cmd::Launch` — a compiled program plus its argument and output slots;
- `Cmd::Copy` — a value moved to a destination pool (and device);
- `Cmd::Alias` — an in-place assign output aliasing its base buffer.

`CmdQueue::schedule(outputs)` slices the queue into maximal same-device launch
runs; copies execute on the destination device, and aliases join the surrounding
run as zero-cost bindings. The result is a `Plan`, cached in
`Runtime::plan_cache` and replayed with the leaf values bound at call time.
`PlanDim` keeps output sizes symbolic until replay, so a plan compiled for one
iteration serves later iterations with different variable dims.

### 7. Backend Codegen

Each backend compiles the winning kernel IR: deSSA, then one linear pass over the
ops emitting target code. The IR is deliberately shaped so no backend repeats
optimization work. Hardware whose compute model cannot be expressed as a linear
SSA pass (Tenstorrent) gets an explicit **physical IR** below the kernel IR —
see [Codegen and the Physical IR](./codegen.md).

Backends are dispatched by enum, with no `dyn Backend`:

```rust,ignore
pub enum Dev {
    Auto,
    C,
    Host,
    Cuda(u16),
    TT(u16),     // feature = "tenstorrent"
    Vulkan(u16),
    OpenCL(u16),
    WGPU(u16),   // feature = "wgpu"
}
```

Except for the feature-gated WGPU and Tenstorrent backends, every backend is
compiled in unconditionally and selected at runtime. A backend whose driver,
hardware, or (for Tenstorrent) simulator is missing simply contributes no
devices; initialization failures are errors, not panics.

### 8. Runtime

The runtime (`runtime.rs`) is the process-wide state: graphs, tensors, kernels,
programs, timings, interned shape expressions, RNG, and the plan cache. A global
`Mutex<Runtime>` guards it.

Device selection for generic work is `Runtime::pick_device(bytes)`: start from
`Dev::all()`, drop AOT-only devices, keep devices whose pool has at least
`bytes` free, and pick the one with the most free compute capacity. Cross-device
movement is scheduled as copies in the plan, and asynchronous execution is a
per-backend concern (CUDA's worker micro-batches launches) — see
[Runtime and Plans](./runtime.md).

## Debugging the Pipeline

| `ZYX_DEBUG` | Output |
|-------------|--------|
| 1 | Devices + configuration |
| 2 | Graph/e-graph dump |
| 4 | Kernel IR before linearization (the scheduler's view) |
| 8 | Winning kernel IR after the search |
| 16 | Generated assembly/source |
| 32 | Kernel launches |
| 64 | Alloc/dealloc |
| 128 | Kernel compilation |
| 256 | Disable the search (`NoSearch`) |

`ZYX_DEBUG=4` and `ZYX_DEBUG=8` are complementary: the first shows the
pre-linearize DAG (structural ops, vector values), the second the post-search
linear form the backend receives. `ZYX_TT_IR=1` dumps the Tenstorrent physical
IR phases.

## Key Design Decisions

- **One graph for everything** — autograd and computation share the same graph.
  No need to specify which tensors require gradients.
- **Symbolic dims everywhere** — the eager path and the graph path both work
  with symbolic dimensions, always. Every symbolic dim bottoms out in a
  `Param { Variable }` scalar (`IDX_T`) whose value lives in a backend pool's
  variable slot. Consumers evaluate dims by folding `Const` leaves and reading
  variable slots, never by fabricating placeholders (`0`, `-1`, ...).
- **Measured search, no cost model** — kernel variants are ranked by a cheap
  feature cost, then decided by real launches; extraction also uses measured
  times. This keeps rewrites exact: the winner is the one that actually ran
  fastest.
- **Inline ops** — all kernel ops live in the arena as flat entries: `Op` is 24
  bytes, `OpLinked` 32. No `Box`, no vtables, no per-op indirection.
- **Linear IR** — a linked list of fixed-size ops; passes traverse it
  front-to-back or back-to-back.
- **Backend codegen is trivial** — the hard work lives in the kernel IR search
  and passes.
- **Tape-scoped lazy graph** — ops inside a tape build graph nodes instead of
  executing eagerly. This enables fusion search, device placement, and plan
  caching across iterations.
