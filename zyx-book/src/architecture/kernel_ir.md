# The Kernel IR

The kernel IR is the representation that passes optimize and backends compile.
It is a **linear SSA** construct: a doubly-linked list of ops stored in an arena.
This chapter describes its structure and its ops; the passes that transform it
are in [Optimization Passes](./optimizations.md).

## Data Structure

```rust,ignore
pub struct Kernel {
    pub(crate) ops: Slab<OpId, OpLinked>,   // arena
    pub head: OpId,                         // first op
    pub(crate) tail: OpId,                  // last op
    pub(crate) dev: Dev,                    // target device
    pub(crate) dev_info: Option<Arc<DeviceInfo>>, // fixed hardware facts
}

pub struct OpLinked {
    pub prev: OpId,   // u32
    pub next: OpId,   // u32
    pub op: Op,       // 24 bytes (enum + payload)
}
```

`OpId` is a `u32` index: random access is O(1), `OpId::NULL` is `u32::MAX`, and
the list is walked through `head`/`next` (or backwards from `tail`). `Op` is 24
bytes and `OpLinked` 32 — both are compile-time assertions in `kernel/ops.rs`.
There is no `Box` per op and no vtable.

`dev_info` is a snapshot of the device's `DeviceInfo` (warp size, limits,
capabilities) taken when the device is bound. Passes never lock the runtime to
ask hardware questions, and clones share the snapshot through `Arc`.

The graph uses the same `Op` enum, but a different node: a graph class node
carries `class_of` and `next_in_class` instead of `prev`/`next`. The kernel node
type is `OpLinked`; the graph node type is `OpNode`.

## Linearization

Before `linearize`, a kernel is a small DAG with **no loads** and no storage:

- `Param { kind: Global }` — an input buffer;
- `Param { kind: GlobalMut }` — an output buffer / store destination;
- `Param { kind: Variable }` — a scalar launch argument (e.g. a dynamic dim);
- movement ops (`Reshape`, `Expand`, `Permute`, `Flip`, `Pad`, `Narrow`);
- element-wise ops (`Unary`, `Binary`, `Select`, `Cast`, `Bitcast`,
  `Reduce`, ...);
- `Store` at the end.

`linearize` turns that into the linear form:

- movement and `Reduce` ops are **removed** and expanded into index arithmetic;
- `Load` ops are inserted for read-only params, with `Storage` as their base;
- `Index`/arithmetic over group and loop indices computes addresses;
- loops (`Loop`/`EndLoop`) and ranges (`Range`) become explicit;
- every `Store` gets an index expression computed from the shape it writes;
- writable params stay in place as store destinations.

A real example makes the difference visible. Before linearize (`ZYX_DEBUG=4`):

```text
r0: i64 = 3
r1: f32[3] = param global shape=r0
r2: f32 = 0
r3: f32[3] = expand r2 -> ...
r4: bool[3] = r1 > r3
r5: f32[3] = f32(r4)
r6: f32[3] = param global shape=r0
r7: f32[3] = r5 * r6
...
r17: f32[1] = pad r14 axis=0 lp=r15 len=r16
r18: f32[1] = param global mut shape=r16
store vec1 r18 = r17
```

After linearize and the search (`ZYX_DEBUG=8`), the same computation is scalar
ops over explicit indices:

```text
r85: f32 = param global shape=NULL
r18: f32 = param global mut shape=NULL
r52: f32 = 0
r31: i64 = 0
r51: f32 = load r85[r31]
r9: f32 = -r51
r11: bool = r9 > r52
r4: bool = r51 > r52
r13: f32 = r11 ? r9 : r52
r7: f32 = r4 ? r51 : r52
r14: f32 = r7 + r13
store r18[r31] = r14
```

Two details matter:

- `Param` shapes go to `NULL` during linearization. Buffer sizes must be
  resolved **before** it (the allocator and launch-argument code do exactly
  that); after it, a param's shape is no longer how sizes are communicated.
- The load/store index carries the access shape. `load r85[r31]` with a scalar
  index reads one element; an index over a lane vector reads a vector; an index
  over two distinct axes reads a tile. The debug printer spells this out as
  `store vec4` or `store tile 32x32`.

`Kernel::is_linearized` detects the state from the stores: a pre-linearize store
writes a whole view to a bare `Param`; linearization gives every store
destination an index expression.

## Op Variants

### Leaves

```rust,ignore
Op::Const(Constant)
Op::Param { dtype: DType, kind: ParamKind, shape: OpId, id: u32 }
// ParamKind::Variable  — scalar launch argument (e.g. a dynamic dim)
// ParamKind::Global    — read-only buffer argument
// ParamKind::GlobalMut — read-write buffer argument
```

`Param::id` is graph identity for hashconsing: two params name the same buffer
iff their `id`s agree. Kernel passes ignore it — kernel params are bound
positionally at launch — and hashing canonicalizes ids to head-order ordinals so
program caches keep sharing across buffers.

### Arithmetic and selection

```rust,ignore
Op::Cast { x: OpId, dtype: DType }
Op::Bitcast { x: OpId, dtype: DType }
Op::Unary { x: OpId, uop: UOp }
Op::Binary { x: OpId, y: OpId, bop: BOp }
Op::Select { cond: OpId, if_true: OpId, if_false: OpId }
Op::Stack { ops: Box<[OpId]> }       // pack values into a vector
Op::Index { vec: OpId, idx: usize }  // select one element of a vector
```

There is no `Mad` op: `Kernel::mad(x, y, z)` builds a `Binary::Mul` followed by a
`Binary::Add`, and the passes may reassociate/fold it. There is no separate
"fused multiply-add" IR node.

### Memory

```rust,ignore
Op::Storage { dtype: DType, scope: MemScope, len: Dim, id: u32 }
Op::Load  { src: OpId, src_idx: OpId }
Op::Store { dst: OpId, dst_idx: OpId, src: OpId }
Op::Copy  { src: OpId, src_idx: OpId, dst: OpId, dst_idx: OpId }
```

`Storage` is kernel-internal memory — registers, local/shared memory,
Tenstorrent circular buffers. Its `id` is assigned by the register allocator:
before allocation every storage of the same `(dtype, scope, len)` is one request
and compares equal; after allocation `id` separates physical slots so CSE cannot
merge two registers that must stay distinct.

`Load` is memory → SSA. `Store` is SSA → memory and never memory → memory; that
is `Copy`. `Store` and `Copy` are effect-only: they produce no SSA value, are
DCE roots, and are never CSE'd or hoisted.

### Iteration

```rust,ignore
Op::Range { axis: u32, kind: RangeKind }
Op::Loop { len: OpId }
Op::EndLoop
Op::Barrier
Op::Unroll { axis: u32, len: u16 }
```

`RangeKind` has two variants:

- `Group(OpId)` — a grid axis (CUDA blocks, CPU cores, Tenstorrent cores);
- `Local(u32)` — a thread/lane axis. Local axis 0 is the hardware warp; there is
  no separate warp kind, and the lane id is the axis-0 local value.

There is no `Scalar` range kind: a scalar is a rank-0 value, not a range.

`Loop` is a counted loop. A **conditional is a `Loop` with a boolean length**
(0 or 1 trips), closed by `EndLoop` exactly like a counted loop; there is no
`If`/`EndIf`. `Barrier` pins program order as an operand-free effect and DCE
root. `Unroll` is a lane vector produced by splitting: it holds the lane
constants `0..len` tagged with the axis they were split from, and is devectorized
into a `Stack` of constants right before codegen.

### Compute

```rust,ignore
Op::Wmma { dims: MMADims, layout: MMALayout, dtype: MMADType, a: OpId, b: OpId, c: OpId }
Op::Reduce { x: OpId, rop: BOp, reduce_axis: OpId }   // pre-linearize, or over lanes
Op::Asm { asm: TinyString, ops: TinyVec<OpId> }
```

`Wmma` is the fused fragment matmul `d = a @ b + c` (the result is the new
accumulator). `Reduce` is a pre-linearize op: `linearize` rewrites it into a
loop with an accumulator. It reappears after `split` turns a loop into lane
vectors, where the reduction runs over the lanes rather than over a loop; that
is why the variant stays in the enum. `Asm` emits a backend assembly template
with `{i}` operand substitution.

### View and movement ops

```rust,ignore
Op::Reshape { x: OpId, shape: OpId }
Op::Expand  { x: OpId, shape: OpId }
Op::Permute { x: OpId, axes: TinyVec<UAxis> }
Op::Flip    { x: OpId, axes: TinyVec<UAxis> }
Op::Pad     { x: OpId, axis: UAxis, lp: OpId, len: OpId }
Op::Narrow  { x: OpId, axis: UAxis, start: OpId, len: OpId }
```

These are pre-linearize ops. They are replayed into consumers and unfolded into
index arithmetic by `linearize`.

### Graph-only markers

```rust,ignore
Op::After { x: OpId, dep: OpId }          // x may not run before dep
Op::ToDevice { x: OpId, device: Dev, time: u64 }
Op::Contiguous { x: OpId }                // fusion-break hint
Op::Detach { x: OpId }                    // autograd break; erased by kernelizer
```

`After` and `ToDevice` survive extraction; `Contiguous` forces an output
boundary; `Detach` is value identity and the kernelizer remaps it away.

### Opaque kernels and backend payloads

```rust,ignore
Op::Source(SourceBlock)                        // program source block
Op::Program { inputs: OpId, outputs: OpId, info: Box<(ProgramId, u64)> }
Op::Kernel(Box<CustomKernel>)                  // user custom kernel boundary
Op::TT(TTOp)                                   // Tenstorrent physical IR
Op::GPU(Box<GPUOp>)                            // shared GPU launch config
Op::Spirv(Box<SpirvOp>)                        // Vulkan SPIR-V payload
Op::PTX(Box<PTXOp>)                            // CUDA direct-PTX payload
```

`Program`'s `inputs` and `outputs` are `Stack` classes, not per-kernel boxes;
`info` carries the program id and its measured time. Timings are ignored by
`Eq`/`Hash`.

## Memory Scopes

```rust,ignore
pub enum MemScope {
    Global,    // DRAM / device global memory
    Local,     // SRAM / shared memory
    Register,  // registers
    Circular,  // circular buffer, SRAM (Tenstorrent)
}
```

There is no `Variable` scope: scalar variables are `Param { kind: Variable }`,
not storage.

## Shapes

`Kernel::shape(op)` reports the value shape of an op as a rank list (at most
eight dims): `Unroll` contributes one dim of `len` lanes on its axis, and a
`Param` contributes its buffer shape's dims; everything else is scalar. For
loads, stores, and copies, the **index's** shape is the access shape, which is
what the debug printer shows (`store vec1`, `store tile 32x32`).

Loops and ranges are scalar iterators and contribute no element dims: the trip
count is loop structure, not value shape.

## Codegen

Codegen consumes a stabilized kernel:

1. **deSSA** — resolve SSA values to physical registers/memory slots;
2. **linear pass** — walk the op list once, emitting instructions.

No further optimization happens here. Backends whose hardware needs explicit
physical state get one more IR below this one — for Tenstorrent, the
`tt_tile` → `tt_dataflow_select` → SFPU/LREG/DST passes → `tt_verify` sequence in
`device_epilogue`, ending in `Op::TT` ops. See
[Codegen and the Physical IR](./codegen.md).

## Debugging

```bash
ZYX_DEBUG=4 cargo run   # kernel IR before linearization
ZYX_DEBUG=8 cargo run   # winning kernel IR after the search
ZYX_DEBUG=16 cargo run  # generated source/assembly
```

`ZYX_TT_IR=1` additionally dumps the Tenstorrent physical IR before and after
`tt_tile`, `tt_reindex`, and `tt_dataflow_select`. Debug output is colorized only
on an interactive terminal; `AGENT=1` disables color.

`Kernel::verify` checks ordering, dtypes, shapes, and use/def consistency. It is
compiled in but returns immediately outside `debug_assertions`. After the
Tenstorrent phase boundary the IR is native tiles/rows, and `tt_verify` is the
checker instead.
