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

Before any optimization passes run, the kernel IR is **unfolded** — `LoadView`, `StoreView`, and `Move` ops are converted to direct index arithmetic (`Load`, `Store` with computed indices). After unfolding, the IR is purely linear with no heap-allocating ops. All ops are fixed-size and live in the arena.

The IR is in SSA form, except for `Loop`, `If`, and `Define` ops (which can carry mutable state).

## Op Variants

### Arithmetic
```rust,ignore
Op::Cast { x: OpId, dtype: DType }
Op::Unary { x: OpId, uop: UOp }
Op::Binary { x: OpId, y: OpId, bop: BOp }
Op::Mad { x: OpId, y: OpId, z: OpId }
```

### Memory
```rust,ignore
Op::Define { dtype, scope, ro, len }
Op::Load { src, index, layout }
Op::Store { dst, x, index, layout }
Op::Const(Constant)
```

### Control Flow
```rust,ignore
Op::Loop { len: Dim }
Op::EndLoop
Op::If { condition: OpId }
Op::EndIf
Op::Barrier { scope }
```

### Indexing
```rust,ignore
Op::Index { len, scope, axis }
```

### Hardware Accelerators
```rust,ignore
Op::Wmma { dims, layout, dtype, a, b, c }
```

### Vectorization
```rust,ignore
Op::Vectorize { ops: Vec<OpId> }
Op::Devectorize { vec: OpId, idx }
```

### View (before unfolding)
```rust,ignore
Op::Move { x: OpId, mop: Box<MoveOp> }
Op::Reduce { x: OpId, rop, n_axes }
```

## Memory Layouts and Scopes

```rust,ignore
pub enum MemLayout {
    Scalar,
    Vector(u8),
    Tile { x, y, stride },
}

pub enum Scope {
    Global,
    Local,
    Register,
}
```

## Backend Codegen

Because the IR is designed for it, backend codegen is trivial:

1. **deSSA** — resolve SSA references to physical registers/memory
2. **Linear pass** — walk the op linked list once, emitting instructions

No further optimizations, no complex lowering.

## Debugging

Set `ZYX_DEBUG=8` to print the kernel IR:

```text
r18: i32 = def global, len=4
r44: u32 = gidx0    // 0..=0
r19: i32 = r18[r1]  // 0..=3 load
```
