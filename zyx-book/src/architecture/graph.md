# The Graph

The graph is an e-graph (equivalence graph) used in tape mode for fusion,
autograd, and device placement. Inside a `Tape`, every operation builds an `Op`
in this graph. Outside a tape, there is no graph — ops go directly to kernel
fusion. When a tape is active, the graph is shared between computation and
autograd — there is only one.

## Data Structure

The graph is stored in the `Runtime`:

```rust,ignore
pub struct Graph {
    pub(crate) hashcons: Map<Op, OpId>,
    pub(crate) ops: Slab<OpId, OpNode>,
    pub(crate) jit_kernels: Slab<JitKernelId, JitKernelData>,
    pub(crate) leaf_classes: Vec<OpId>,
    pub(crate) leaf_map: Map<OpId, TensorId>,
    pub(crate) ref_count: u64,
    pub(crate) dead: bool,
    pub(crate) max_cons_id: u32,
}

pub(crate) struct OpNode {
    pub(crate) op: Op,
    pub(crate) class_of: OpId,
    pub(crate) next_in_class: OpId,
}
```

The `hashcons` deduplicates structurally identical ops — if the same operation
on the same inputs already exists, the existing `OpId` is reused. This provides
CSE (common subexpression elimination) for free.

Each `OpId` maps to an `OpNode` entry in the `ops` slab. A node joins exactly
one equivalence class, named by its `class_of` field (the class's root `OpId`);
alternative forms of the same value are chained through `next_in_class`, in
insertion order. `max_cons_id` allocates the identity used by
`Op::Param::id`.

### Op Types

The graph opset is derived from tinygrad. By stacking these variants, zyx can
express ALL linear algebra operations and ALL PyTorch ops:

```rust,ignore
enum Op {
    Const(Constant),
    Param { dtype: DType, kind: ParamKind, shape: OpId, id: u32 },
    Reshape { x: OpId, shape: OpId },
    Expand { x: OpId, shape: OpId },
    Permute { x: OpId, axes: TinyVec<UAxis> },
    Pad { x: OpId, axis: UAxis, lp: OpId, len: OpId },
    Flip { x: OpId, axes: TinyVec<UAxis> },
    Narrow { x: OpId, axis: UAxis, start: OpId, len: OpId },
    Stack { ops: Box<[OpId]> },
    Index { vec: OpId, idx: usize },
    Reduce { x: OpId, rop: BOp, reduce_axis: OpId },
    Cast { x: OpId, dtype: DType },
    Bitcast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    Binary { x: OpId, y: OpId, bop: BOp },
    Select { cond: OpId, if_true: OpId, if_false: OpId },
    Store { dst: OpId, dst_idx: OpId, src: OpId },
    After { x: OpId, dep: OpId },
    ToDevice { x: OpId, device: Dev, time: u64 },
    Contiguous { x: OpId },
    Detach { x: OpId },
    Program { inputs: OpId, outputs: OpId, info: Box<(ProgramId, u64)> },
    Kernel(Box<CustomKernel>),
    // Kernel-only variants (Load, Storage, Range, Loop, Wmma, TT, ...)
    // are described in The Kernel IR.
}
```

All inputs reference `OpId` — ops operate on equivalence classes. View ops
(`Reshape`, `Expand`, `Permute`, `Pad`, `Flip`, `Narrow`) are per-axis and
stackable; `Stack` builds vectors; `Index` selects an element; `Bitcast`
reinterprets bits without value conversion; `Store` is an in-place write
(assigns lower to it); `After` orders side-effecting ops; `Contiguous` forces a
materialization boundary; `Detach` is value identity with no gradient; `Program`
is a compiled-kernel boundary; `Kernel` wraps an opaque user custom kernel.

The reason to use `OpId` instead of `TensorId` is to get canonicalization for
free.

## Lifecycle with Tape

There is no graph outside a tape — ops are fused directly into kernels.

Inside a tape, ops accumulate until `Tape::realize()` or drop. A class can hold
several alternative producers, and extraction chooses between them:

- **CSE** via hashconsing: identical structural ops share one class.
- **Kernel twins**: the kernelizer mints an `Op::Program` node for every fused
  kernel candidate, on the class it produces.
- **AOT pattern matches**: backends add `Op::Program` nodes for precompiled
  kernels (cuDNN, CBLAS, ...), so they compete with fused kernels.
- **Transfers**: extraction inserts `Op::ToDevice` nodes where the chosen
  producer and consumer live on different devices.

There is no cost model: each candidate is compiled and autotuned, and
`Graph::extract` picks the path by measured timing. Realized ops that the tape
references are preserved for autograd; unreferenced ops are released.

## Graph Size

The graph is designed to stay small. Tensor handles are `u32` (4 bytes), so a
vector of 10,000 handles costs ~40 kB. The graph nodes themselves are bigger:
`Op` is 24 bytes and an `OpNode` is 32. When the tape is dropped, the graph is
deleted as a whole, so there is no per-op garbage collection. Per-tensor `rc`
and the graph's `ref_count` decide when that happens; ops are otherwise
append-only (arena style).
