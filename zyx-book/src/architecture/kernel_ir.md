# The Kernel IR

The kernel IR is the intermediate representation used for all computation kernels. It is a doubly-linked list of 32-byte `OpNode`s stored in an arena (the `Slab` allocator).

## Data Structure

```rust,ignore
pub struct Kernel {
    pub ops: Slab<OpId, OpNode>,
    pub head: OpId,
    pub tail: OpId,
    // ...
}

pub struct OpNode {
    pub prev: OpId,  // u32
    pub next: OpId,  // u32
    pub op: Op,      // 24 bytes (enum + payload)
}
```

The `Slab<OpId, OpNode>` is a `Vec<OpNode>` with a free-list. `OpId` is a `u32` index — random access is O(1).

## Unfolding

Before linearization, view ops (`Reshape`, `Expand`, `Permute`, `Flip`, `Pad`, `Narrow`) are represented directly as `Op` variants. They are unfolded into index arithmetic during the linearization pass — after unfolding, all ops are fixed-size inline entries in the arena — no `Box`, no vtables, no per-op indirection.

The IR is in SSA form, except for `Loop`, `If`, and `Define` ops (which can carry mutable state).

## Op Variants

### Parameters
```rust,ignore
Op::Param { dtype, kind: ParamKind, shape: OpId }
// ParamKind::Variable — scalar launch argument (e.g. dynamic dim, IDX_T)
// ParamKind::Global / GlobalMut — read-only / read-write buffer argument
```

### Arithmetic
```rust,ignore
Op::Cast { x: OpId, dtype: DType }
Op::Bitcast { x: OpId, dtype: DType }
Op::Unary { x: OpId, uop: UOp }
Op::Binary { x: OpId, y: OpId, bop: BOp }
Op::Mad { x: OpId, y: OpId, z: OpId }
Op::Stack { ops: Box<[OpId]> }
```

### Memory
```rust,ignore
Op::Storage { dtype, scope: MemScope, len: Dim }  // kernel-internal memory
Op::Load { src, index, layout }
Op::Store { dst, src, index, layout }
Op::Const(Constant)
```

`Op::Param` and `Op::Storage` are the SSA escape hatches: the mutable
stuff values are read from and written to across the linear order.

### Control Flow
```rust,ignore
Op::Loop { len: OpId }
Op::EndLoop
Op::If { condition: OpId }
Op::EndIf
Op::Barrier
```

### Indexing
```rust,ignore
Op::Range { axis, kind: RangeKind }  // Group / Local / Scalar
Op::Index { vec: OpId, idx }         // select a value from a Stack
```

### Hardware Accelerators / Tiles
```rust,ignore
Op::Wmma { dims, layout, dtype, a, b, c }
Op::ReduceTile { x, .. }
Op::MatmulTile { a, b, .. }
Op::TransposeTile { x, .. }
Op::BroadcastTile { x, .. }
```

### Backend-Specific
```rust,ignore
Op::Asm { .. }  // inline assembly for backends with JIT asm (e.g. Tenstorrent)
```

### View Ops
```rust,ignore
Op::Reshape { x: OpId, shape: OpId }
Op::Expand { x: OpId, shape: OpId }
Op::Permute { x: OpId, axes: TinyVec<UAxis> }
Op::Flip { x: OpId, axes: TinyVec<UAxis> }
Op::Pad { x: OpId, axis: UAxis, lp: OpId, len: OpId }
Op::Narrow { x: OpId, axis: UAxis, start: OpId, len: OpId }
Op::Reduce { x: OpId, rop: BOp, reduce_axis: OpId }
```

## Memory Layouts and Scopes

```rust,ignore
pub enum MemLayout {
    Scalar,
    Vector(u8),
    Tile { x, y, stride },
}

pub enum MemScope {
    Global,
    Local,
    Register,
    Variable,
    Circular,
}
```

## Backend Codegen

Because the IR is designed for it, backend codegen is trivial:

1. **deSSA** — resolve SSA references to physical registers/memory
2. **Linear pass** — walk the op linked list once, emitting instructions

No further optimizations, no complex lowering. Backends whose compute
model needs explicit physical state (Tenstorrent) get an additional
physical IR below the kernel IR — see
[Codegen and the Physical IR](./codegen.md).

## Debugging

Set `ZYX_DEBUG=8` to print the kernel IR:

```text
r18: i32 = def global, len=4
r44: u32 = gidx0    // 0..=0
r19: i32 = r18[r1]  // 0..=3 load
```
