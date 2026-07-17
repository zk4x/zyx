# The Graph

The graph is an e-graph (equivalence graph) used in tape mode for tensor operation rewrites and optimization. Inside a `Tape`, every operation builds a node in this graph. Outside a tape, there is no graph — ops go directly to kernel fusion. When a tape is active, the graph is shared between computation and autograd — there is only one.

## Data Structure

The graph is stored in the `Runtime`:

```rust,ignore
pub struct Graph {
    hashcons: Map<Node, NodeId>,
    nodes: Slab<NodeId, NodeData>,
    classes: Slab<ClassId, EClass>,
    ekernels: Slab<EKernelId, EKernelData>,
    kernel_map: Map<NodeId, EKernelId>,
    leaf_map: Map<ClassId, TensorId>,
    rc: u32,
    max_leaf_id: u32,
}
```

The `hashcons` deduplicates structurally identical nodes — if the same operation on the same inputs already exists, the existing `NodeId` is reused. This provides CSE (common subexpression elimination) for free.

Each `NodeId` maps to a `NodeData` entry in the `nodes` slab, and each node belongs to an equivalence `ClassId` in the `classes` slab. Equivalent forms of the same computation (e.g. different layouts of a matmul) live in the same class.

### Node Types

The graph opset is derived from tinygrad. By stacking these types, zyx can express ALL linear algebra operations and ALL PyTorch ops:

```rust,ignore
enum Node {
    Const(Constant),
    Leaf { dtype: DType, leaf_id: u32 },
    Expand { x: ClassId, shape: ShapeId },
    Permute { x: ClassId, axes: Box<[UAxis]> },
    Reshape { x: ClassId, shape: ShapeId },
    PadZeros { x: ClassId, padding: Box<[(i64, i64)]> },
    Reduce { x: ClassId, bop: BOp, axes: Box<[UAxis]> },
    Cast { x: ClassId, dtype: DType },
    Unary { x: ClassId, uop: UOp },
    Binary { x: ClassId, y: ClassId, bop: BOp },
    ToDevice { x: ClassId, device: DeviceId, time: u64 },
    Kernel { inputs: Box<[ClassId]>, outputs: Box<[ClassId]>, program_id: ProgramId, time: u64 },
}
```

All inputs reference `ClassId` rather than `TensorId` — nodes operate on equivalence classes, not specific tensors.

## Lifecycle with Tape

There is no graph outside a tape — ops are fused directly into kernels.

Inside a tape, nodes accumulate until `Tape::realize()` or drop. The graph supports rewrites that produce equivalent forms of a computation:

- **CSE** via hashconsing
- **Algebraic rewrites** like transpose fusion
- **Layout rewrites**: matmul can be realized as transposed or un-transposed
- **Shape rewrites**: reshape and padding can be fused or split

A cost model selects the cheapest extraction from each equivalence class for kernel compilation. Realized nodes that the tape references are preserved for autograd; unreferenced nodes are released.

## Graph Size

The graph is designed to stay small. Each node is ~16 bytes, and a training iteration with 10,000 operations costs ~200 KB. When the tape is dropped, the graph shrinks back to baseline.
