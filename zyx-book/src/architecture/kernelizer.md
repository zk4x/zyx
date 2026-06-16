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

## The Fusion Rule

The kernelizer's core logic is surprisingly simple:

> **A kernel boundary is created when a Reduce node is used by more than one other kernel.**

This means:
- A chain of element-wise ops (unary, binary) always fuses into one kernel
- View ops (reshape, expand, permute, pad) are always free — they become index arithmetic
- A reduce that feeds multiple downstream ops creates a boundary because each consumer needs the reduced result
- A reduce that feeds only one consumer stays fused

Additional heuristics add boundaries when a kernel would be "too unwieldy":
- Kernels with too many parameters get split
- Very large traversal depth triggers a split

## How Fusion Works

The kernelizer performs a bottom-up traversal of the graph:

```text
Input tensors (already realized)
        │
        ▼
    Node A  ──┐
              ├──► Kernel 1 (fused A + B)
    Node B  ──┘
        │
        ▼
    Node C (Reduce) ──► Kernel 2 (reduce only)
        │
        ├──► Node D ──┐
        │             ├──► Kernel 3 (fused D + E)
        └──► Node E ──┘
```

Node C is a reduce used by both Node D and Node E, so it gets its own kernel. Nodes A and B are element-wise ops feeding each other, so they fuse into one kernel.

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
