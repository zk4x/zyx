# The Kernelizer

The kernelizer (`graph/kernelizer.rs`) fuses tensor operations into kernels. In eager mode, ops are appended directly to kernels. In tape mode, it traverses the accumulated graph and fuses compatible nodes into kernels.

## When the Kernelizer Runs

Outside a tape, there is no graph kernelizer: `Runtime` appends each op to the kernel that produced its inputs and splits (`duplicate_or_store`) only when a fusion boundary is hit. Inside a tape, `Graph::kernelize` runs once during `Tape::realize()` (dropping only cleans up graph state):

```rust
# extern crate zyx;
# use zyx::{DType, Tape, Tensor, ZyxError};
# fn main() -> Result<(), ZyxError> {
// Tape mode — kernelizer runs at realize
let x = Tensor::randn([4], DType::F32)?;
let tape = Tape::new([&x])?;
let y = x.relu();
let z = y * 2.0;
// at this point: no computation done yet
tape.realize([&z])?;
// kernelizer runs → optimizes → executes
# Ok(())
# }
```

## Fusion Logic

In tape mode, the kernelizer processes graph ops **bottom-up** in topological order. Each graph op type adds ops to the kernel its input lives in — fusion is the default, splitting happens only when needed.

### Per-Op Type Behavior

| Graph Op | Kernel Decision |
|------------|-----------------|
| `Unary`, `Cast` | Always fuse — add op to the input's kernel |
| `Expand`, `Permute`, `Reshape`, `Pad`, `Flip`, `Narrow` | Fused into the input's kernel (unfolded to index arithmetic) |
| `Binary` | Merge both input kernels into one |
| `Reduce` | Add reduce op to the input's kernel |
| `Const` | Create a new kernel with a constant |

### When Splitting Happens

The key splitting decision is in `duplicate_or_store_class()`. When a kernel has multiple outputs (a tensor used by >1 downstream), the kernelizer chooses:

1. **If preceded by a reduce** (expensive to recompute) → store the intermediate result to global memory, create a new load kernel for the next consumer
2. **If NOT preceded by a reduce** (cheap to recompute) → duplicate the kernel so each consumer gets its own copy

This is a cost heuristic, not a simple rule: is recomputing cheaper than a global memory store+load?

Requested outputs are consumers too: if a class is needed after its producing kernel is closed, it is stored and a fresh load kernel serves the remaining consumers.

### Practical Outcome

Each kernel tends to center around one reduce loop, with all fused element-wise ops grouped before and after it. A chain of element-wise ops always fuses into one kernel. Reduce-heavy graphs may end up with each reduce in its own kernel separated by store/load boundaries.

## The Kernelizer Struct

The kernelizer is a set of methods on `Graph`; all working state is local to a call:

```rust,ignore
impl Graph {
    pub fn kernelize(&mut self, inputs: &Set<OpId>, outputs: &BTreeSet<OpId>, allowed: Option<&Set<OpId>>) {
        // `jit_kernels: Slab<JitKernelId, JitKernelData>` is the output table.
        let mut rcs: Map<OpId, u32> = Map::default();
        let mut visited: Map<OpId, (JitKernelId, OpId)> = Map::default();
        // ...
    }
}

pub struct JitKernelData {
    pub(crate) kernel: Kernel,
    pub(crate) outputs: Vec<OpId>,
    pub(crate) loads: Vec<OpId>,
    pub(crate) stores: Vec<OpId>,
}
```

The `visited` map tracks which graph ops have been converted to kernel ops. When a graph op is already in `visited`, the kernelizer uses the existing kernel result instead of recomputing — this is how shared subgraphs are handled.

## Building Kernel Ops

Each graph op type maps to kernel IR operations:

| Graph Op | Kernel IR |
|------------|-----------|
| `Const` | `Op::Const` |
| `Param` | `Op::Param` (global memory) |
| `Unary` | `Op::Unary` |
| `Binary` | `Op::Binary` |
| `Reduce` | `Op::Reduce` |
| `Reshape`, `Expand`, `Permute`, `Flip`, `Pad`, `Narrow` | View ops folded into `Load`/`Store` index arithmetic |

View operations (reshape, expand, permute, pad) are unfolded into index arithmetic rather than becoming separate ops. This is how they become "free" — the index computation is inlined into the load/store operations.

## Kernel Caching

Kernels are deduplicated by content, and compiled programs are cached by kernel id:

```rust,ignore
// Runtime
kernel_map: Map<Kernel, KernelId>,          // content-dedup: equal kernels share an id
programs: Map<KernelId, DeviceProgramId>,   // compiled program per kernel id
timings: Map<ProgramId, u64>,               // measured nanos
```

When a kernel finishes optimization its hash is computed; an equal `Kernel` already in
`kernel_map` reuses the `KernelId` (and the compiled program, if one exists). Compiled
programs persist across realizations, so repeated graph patterns — training-loop iterations —
hit the cache after the first one. The plan cache (`Runtime::plan_cache`) does the same for
whole execution plans, keyed by graph content plus leaf pools.
