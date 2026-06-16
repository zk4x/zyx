# The Kernelizer

The kernelizer (`kernelize.rs`) is the bridge between the high-level tensor graph and the low-level kernel IR. It traverses the computation graph and fuses compatible nodes into kernels.

## When the Kernelizer Runs

The kernelizer is invoked during `realize()`:

```rust,ignore
pub fn realize(x: &[&Tensor]) -> Result<(), ZyxError> {
    // 1. Lock the runtime
    // 2. Identify which tensors need evaluation
    // 3. Topological sort based on reference counts
    // 4. Call kernelizer to build kernels
    // 5. Optimize and compile each kernel
    // 6. Execute on device
    // 7. Release unused nodes
}
```

## Fusion Logic

The kernelizer processes graph nodes **bottom-up** in topological order. Each graph node type adds ops to the kernel its input lives in — fusion is the default, splitting happens only when needed.

### Per-Node Type Behavior

| Graph Node | Kernel Decision |
|------------|-----------------|
| `Unary`, `Cast` | Always fuse — add op to the input's kernel |
| `Expand`, `Permute`, `Reshape`, `Pad` | Add `Move` op to the input's kernel (free after unfolding) |
| `Binary` | Merge both input kernels into one |
| `Reduce` | Add reduce op to the input's kernel |
| `Const` | Create a new kernel with a constant |

### When Splitting Happens

The key splitting decision is in `duplicate_or_store()`. When a kernel has multiple outputs (a tensor used by >1 downstream), the kernelizer chooses:

1. **If preceded by a reduce** (expensive to recompute) → store the intermediate result to global memory, create a new load kernel for the next consumer
2. **If NOT preceded by a reduce** (cheap to recompute) → duplicate the kernel so each consumer gets its own copy

This is a cost heuristic, not a simple rule: is recomputing cheaper than a global memory store+load?

Stores also trigger automatically when a graph node is requested as the final output (`to_eval`), creating a natural boundary there.

### Practical Outcome

Each kernel tends to center around one reduce loop, with all fused element-wise ops grouped before and after it. A chain of element-wise ops always fuses into one kernel. Reduce-heavy graphs may end up with each reduce in its own kernel separated by store/load boundaries.

## The Kernelizer Struct

```rust,ignore
struct Kernelizer<'a> {
    must_keep_nodes: Set<TensorId>,
    pending_stores: Set<TensorId>,
    realized_nodes: Set<TensorId>,
    kernels: Slab<KMKernelId, Kernel>,
    visited: Map<TensorId, (KMKernelId, OpId)>,
    rcs: Map<TensorId, u32>,
    graph: &'a Graph,
    // ...
}
```

The `visited` map tracks which graph nodes have been converted to kernel ops. When a graph node is already in `visited`, the kernelizer uses the existing kernel result instead of recomputing — this is how shared subgraphs are handled.

## Building Kernel Ops

Each graph node type maps to kernel IR operations:

| Graph Node | Kernel IR |
|------------|-----------|
| `Const` | `Op::Const` |
| `Leaf` | `Op::Define` (global memory) |
| `Unary` | `Op::Unary` |
| `Binary` | `Op::Binary` or `Op::Mad` |
| `Reduce` | `Op::Reduce` |
| `Reshape` | `Op::Move` (view unfolding) |
| `Expand` | `Op::Move` (view unfolding) |
| `Permute` | `Op::Move` (view unfolding) |
| `Pad` | `Op::Move` (view unfolding) |

View operations (reshape, expand, permute, pad) are unfolded into index arithmetic rather than becoming separate ops. This is how they become "free" — the index computation is inlined into the load/store operations.

## Kernel Caching

```rust,ignore
pub struct KernelCache {
    pub cache: Map<KernelId, Kernel>,
}
```

After a kernel is built and optimized, its hash is computed. If the same kernel was compiled before, the cached compiled program is reused. The cache persists across realize() calls, so repeated graph patterns (like training loop iterations) hit the cache on the second iteration.
