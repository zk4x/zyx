# The Graph

The graph is an e-graph (equivalence graph) used in tape mode for tensor operation rewrites and optimization. Inside a `Tape`, every operation builds an `Op` in this graph. Outside a tape, there is no graph — ops go directly to kernel fusion. When a tape is active, the graph is shared between computation and autograd — there is only one.

## Data Structure

The graph is stored in the `Runtime`:

```rust,ignore
pub struct Graph {
    hashcons: Map<Op, OpId>,
    ops: Slab<OpId, OpNode>,
    jit_kernels: Slab<JitKernelId, JitKernelData>,
    leaf_classes: Vec<OpId>,
    leaf_map: Map<OpId, TensorId>,
    ref_count: u64,
    dead: bool,
    max_cons_id: u32,
}
```

The `hashcons` deduplicates structurally identical ops — if the same operation on the same inputs already exists, the existing `OpId` is reused. This provides CSE (common subexpression elimination) for free.

Each `OpId` maps to an `OpNode` entry in the `ops` slab. Each op belongs to an equivalence class identified by its `class_of` field (the root `OpId` of the class). Equivalent forms of the same computation (e.g. different layouts of a matmul) live in the same class.

### Op Types

The graph opset is derived from tinygrad. By stacking these variants, zyx can express ALL linear algebra operations and ALL PyTorch ops:

```rust,ignore
enum Op {
    Const(Constant),
    Param { dtype: DType, kind: ParamKind, shape: OpId, cons_id: u32 },
    Reshape { x: OpId, shape: OpId },
    Expand { x: OpId, shape: OpId },
    Permute { x: OpId, axes: TinyVec<UAxis> },
    Pad { x: OpId, axis: UAxis, lp: OpId, len: OpId },
    Flip { x: OpId, axes: TinyVec<UAxis> },
    Narrow { x: OpId, axis: UAxis, start: OpId, len: OpId },
    Stack { ops: Box<[OpId]> },
    Reduce { x: OpId, rop: BOp, reduce_axis: OpId },
    Cast { x: OpId, dtype: DType },
    Bitcast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    Binary { x: OpId, y: OpId, bop: BOp },
    After { x: OpId, dep: OpId },
    ToDevice { x: OpId, device: Dev, time: u64 },
    Contiguous { x: OpId },
    Kernel { inputs: OpId, outputs: OpId, info: Box<(ProgramId, u64)> },
    Custom(Box<CustomKernel>),
}
```

All inputs reference `OpId` rather than `TensorId` — ops operate on equivalence classes, not specific tensors. View ops (`Reshape`, `Expand`, `Permute`, `Pad`, `Flip`, `Narrow`) are per-axis and stackable; `Stack` builds vectors; `Bitcast` reinterprets bits without value conversion; `After` orders side-effecting ops; `Contiguous` materializes a layout; `Custom` wraps opaque custom kernels.

## Lifecycle with Tape

There is no graph outside a tape — ops are fused directly into kernels.

Inside a tape, ops accumulate until `Tape::realize()` or drop. The graph supports rewrites that produce equivalent forms of a computation:

- **CSE** via hashconsing
- **Algebraic rewrites** like transpose fusion
- **Layout rewrites**: matmul can be realized as transposed or un-transposed
- **Shape rewrites**: reshape and padding can be fused or split

There is no cost model: each fusion variant is individually autotuned, and `Graph::extract` picks by measured timing. Realized ops that the tape references are preserved for autograd; unreferenced ops are released.

## Graph Size

The graph is designed to stay small. Tensor handles are `u32` (4 bytes), so 10,000 handles cost ~40 kB. When the tape is dropped, the graph shrinks back to baseline.
