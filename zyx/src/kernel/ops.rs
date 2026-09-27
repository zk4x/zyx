// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
use std::hash::{Hash, Hasher};

use nanoserde::{DeBin, SerBin};

use crate::backend::{Dev, ProgramId};
use crate::dtype::Constant;
use crate::kernel::{MemLayout, MemScope};
use crate::shape::{Dim, UAxis};
use crate::slab::SlabId;
use crate::types::{TinyString, TinyVec};
use crate::{DType, Map};

/// Kernel parameter kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
pub enum ParamKind {
    /// Single scalar variable
    Variable,
    /// Global read only buffer
    Global,
    /// Global read-write buffer
    GlobalMut,
}

#[derive(Debug, Clone, SerBin)]
pub enum Op {
    // ops that exist in both
    Const(Constant),
    /// A kernel parameter — one of the arguments passed to the compiled GPU
    /// kernel at launch.
    ///
    /// Pre-linearization the kernel is a pure DAG and Params are its leaves.
    /// Post-linearization the kernel is a linear SSA construct; Params remain
    /// as leaves and — together with [`Op::Storage`] — act as the SSA escape
    /// hatches: the mutable stuff values are read from and written to.
    /// Linearize only nulls a Param's `shape` field, so buffer sizes must be
    /// resolved BEFORE linearization (see `Kernel::alloc_buffers`).
    ///
    /// - [`ParamKind::Variable`] — a single scalar argument (e.g. a dynamic
    ///   dim), passed by value.
    /// - [`ParamKind::Global`] / [`ParamKind::GlobalMut`] — a read-only /
    ///   read-write buffer argument, passed by pointer.
    ///
    /// # Null operands
    ///
    /// Data operands (`x`, `y`, ...) of any op can never be null — only shape
    /// operands may be, where null means scalar shape (rank 0). A null data
    /// operand is always a bug.
    Param {
        dtype: DType,
        kind: ParamKind,
        shape: OpId,
        /// Buffer identity for egraph hashconsing (mirrors the former
        /// `Op::Param.cons_id`): two params name the same buffer iff their
        /// `cons_id`s agree. Kernel passes ignore it (params are bound
        /// positionally there); it is compared by `Eq` but skipped by `Hash`
        /// so program caches (`get_hash`) keep sharing across buffers.
        cons_id: u32,
    },
    Cast {
        x: OpId,
        dtype: DType,
    },
    /// Bitcast: reinterprets the raw bits of `x` as `dtype` without a value
    /// conversion. Requires equal bit widths of `x`'s dtype and `dtype`
    /// (`debug_assert` in [`Kernel::bitcast`]).
    Bitcast {
        x: OpId,
        dtype: DType,
    },
    Unary {
        x: OpId,
        uop: UOp,
    },
    // For binary ops, next of x is y, then next of y is the binary op
    Binary {
        x: OpId,
        y: OpId,
        bop: BOp,
    },
    // Vectorization, YAY!
    Stack {
        ops: Box<[OpId]>,
    },

    // ops that only exist after unfolding views and reduces
    /// Memory internal to the kernel — NOT a launch argument. Used for
    /// accumulators, shared/local memory, Tenstorrent circular buffers,
    /// arrays of values in registers, etc.
    ///
    /// Storage ops only exist post-linearization: in the linear SSA construct
    /// they — together with [`Op::Param`] — are the escape hatches, the
    /// mutable stuff that values are written to and read back from across the
    /// linear order. A `MemScope::Variable` storage holds a single scalar.
    Storage {
        dtype: DType,
        scope: MemScope,
        len: Dim,
    },
    Store {
        dst: OpId,
        src: OpId,
        index: OpId,
        layout: MemLayout,
    },
    Load {
        src: OpId,
        index: OpId,
        layout: MemLayout,
    },
    // Like loop, but for dimensions always executed in parallel
    Range {
        axis: u32,
        kind: RangeKind,
    },
    // Control flow
    Loop {
        len: OpId,
    },
    EndLoop,
    If {
        condition: OpId, // must be boolean variable
    },
    EndIf,
    // fused multiply add
    Mad {
        x: OpId,
        y: OpId,
        z: OpId,
    },
    Index {
        vec: OpId,
        idx: usize,
    }, // select a single value from a vector
    Barrier,
    // fused matmul, a, b, c are fragments, each is a vector, c is accumulator, returns new accumulated vector d
    Wmma {
        dims: MMADims,
        layout: MMALayout,
        dtype: MMADType,
        a: OpId,
        b: OpId,
        c: OpId,
    },
    /// Hardware reduce_tile: folds tile `x` into accumulator tile `acc`
    /// with `rop` (TT: `reduce_tile` accumulates into the acc CB directly;
    /// the result tile carries values in its first row). `scaler` is the
    /// LLK-mandated scale tile (ones when unused, e.g. MAX): an explicit
    /// operand so its CB traffic balances like everything else. Explicit
    /// `acc` keeps the accumulation in SSA dataflow instead of a fold
    /// marker.
    ReduceTile {
        x: OpId,
        scaler: OpId,
        acc: OpId,
        rop: BOp,
        kind: TileDim,
    },
    /// Hardware tile matmul: folds `x @ y` into accumulator tile `acc`
    /// (TT: `matmul_tiles` accumulates into DST). Explicit `acc` keeps
    /// the accumulation in SSA dataflow instead of a fold marker.
    MatmulTile {
        x: OpId,
        y: OpId,
        acc: OpId,
    },
    TransposeTile {
        x: OpId,
    },
    /// Marker: tile `x` (a `load_circular` tile) is consumed with
    /// broadcast `kind` by the consuming tiled binary, which emits the
    /// fused broadcast form (TT: `add/sub/mul_tiles_bcast_*`, operands
    /// stay in CBs). Without the marker the binary uses the plain
    /// register form. Carries no traffic itself.
    BroadcastTile {
        x: OpId,
        kind: TileDim,
    },
    // For backend specific assembly
    Asm {
        asm: TinyString,
        ops: TinyVec<OpId>,
    },

    // ops that exist before linearize and linearize converts them into these ops: index, loop and load
    /// Reshape to a new shape.
    Reshape {
        x: OpId,
        shape: OpId,
    },
    /// Expand dimensions.
    Expand {
        x: OpId,
        shape: OpId,
    },
    /// Permute axes.
    Permute {
        x: OpId,
        axes: TinyVec<UAxis>,
    },
    /// Flip axes
    Flip {
        x: OpId,
        axes: TinyVec<UAxis>,
    },
    /// Pad axis
    /// Pad with `lp` zeros on the left, to total axis length `len`. Right padding is `len - lp - orig_len`.
    Pad {
        x: OpId,
        axis: UAxis,
        lp: OpId,
        len: OpId,
    },
    /// Slice axis
    Narrow {
        x: OpId,
        axis: UAxis,
        start: OpId,
        len: OpId,
    },
    /// Reduce `x` with `rop` over the trailing dim. `reduce_axis` always
    /// names the last dim's class (null is tolerated as shorthand for it).
    /// Reductions over any other axis are permuted to trailing before
    /// reaching this op.
    Reduce {
        x: OpId,
        rop: BOp,
        reduce_axis: OpId,
    },
    // Graph-only ops (former `Node` variants). They never appear in ordered
    // kernels: every kernel-side match arms them with `todo!()`.
    // NOTE: there is no `Assign` variant: graph assigns lower to
    // `Op::Store` with a null index (both the kernelizer and eager assign
    // already emit `store(dst, src, OpId::NULL)`).
    /// Ordering edge: `x` may not run before `dep` completes.
    After {
        x: OpId,
        dep: OpId,
    },
    /// Move `x` to `device`. `time` is measured launch timing, ignored by
    /// `Eq`/`Hash` (mirrors the former `Node::ToDevice`).
    ToDevice {
        x: OpId,
        device: Dev,
        time: u64,
    },
    /// Fusion-break hint: forces `x` to materialize as a separate kernel
    /// output. The kernelizer never fuses through it.
    Contiguous {
        x: OpId,
    },
    /// A compiled kernel boundary: `info` is the owning program and measured
    /// timing. Both `inputs` and `outputs` are `OpId`s of `Stack` ops holding
    /// the input/output classes — the lists are shared nodes, not per-kernel
    /// `Box` allocations. `info` is boxed so `Op` keeps its 24-byte budget.
    /// Timing is ignored by `Eq`/`Hash` (mirrors the former `Node::Kernel`).
    Kernel {
        inputs: OpId,
        outputs: OpId,
        info: Box<(ProgramId, u64)>,
    },
    /// A custom (user-built) kernel boundary, boxed to keep `Op` within
    /// its 24-byte budget. `outputs` triples are `(class, shape, dtype)`.
    /// `time` is measured timing, ignored by `Eq`/`Hash` (mirrors the
    /// former `Node::Custom`, which also never compares equal).
    Custom(Box<CustomKernel>),
}

/// Boxed payload of [`Op::Custom`]: a custom kernel boundary's inputs,
/// output `(class, shape, dtype)` triples, owning program, and measured
/// timing. Boxed so `Op` keeps its 24-byte budget.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
pub struct CustomKernel {
    pub inputs: Box<[OpId]>,
    pub outputs: Box<[(OpId, OpId, DType)]>,
    pub program_id: ProgramId,
    pub time: u64,
}

impl Op {
    /// Position of the variant in declaration order. Used to keep the manual
    /// `Ord` identical to the previously derived one (existing variants keep
    /// declaration order; the graph-only variants are appended last).
    fn disc(&self) -> u8 {
        match self {
            Op::Const(_) => 0,
            Op::Param { .. } => 1,
            Op::Cast { .. } => 2,
            Op::Bitcast { .. } => 3,
            Op::Unary { .. } => 4,
            Op::Binary { .. } => 5,
            Op::Stack { .. } => 6,
            Op::Storage { .. } => 7,
            Op::Store { .. } => 8,
            Op::Load { .. } => 9,
            Op::Range { .. } => 10,
            Op::Loop { .. } => 11,
            Op::EndLoop => 12,
            Op::If { .. } => 13,
            Op::EndIf => 14,
            Op::Mad { .. } => 15,
            Op::Index { .. } => 16,
            Op::Barrier => 17,
            Op::Wmma { .. } => 18,
            Op::ReduceTile { .. } => 19,
            Op::MatmulTile { .. } => 20,
            Op::TransposeTile { .. } => 21,
            Op::BroadcastTile { .. } => 22,
            Op::Asm { .. } => 23,
            Op::Reduce { .. } => 25,
            Op::After { .. } => 26,
            Op::ToDevice { .. } => 27,
            Op::Contiguous { .. } => 28,
            Op::Kernel { .. } => 29,
            Op::Custom(_) => 30,
            Op::Reshape { .. } => 31,
            Op::Expand { .. } => 32,
            Op::Permute { .. } => 33,
            Op::Flip { .. } => 34,
            Op::Pad { .. } => 35,
            Op::Narrow { .. } => 36,
        }
    }
}

impl PartialEq for Op {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Op::Const(a), Op::Const(b)) => a == b,
            (
                Op::Param { dtype: ad, kind: ak, shape: as_, cons_id: ac },
                Op::Param { dtype: bd, kind: bk, shape: bs, cons_id: bc },
            ) => ad == bd && ak == bk && as_ == bs && ac == bc,
            (Op::Cast { x: a, dtype: ad }, Op::Cast { x: b, dtype: bd }) => a == b && ad == bd,
            (Op::Bitcast { x: a, dtype: ad }, Op::Bitcast { x: b, dtype: bd }) => a == b && ad == bd,
            (Op::Unary { x: a, uop: au }, Op::Unary { x: b, uop: bu }) => a == b && au == bu,
            (Op::Binary { x: a, y: ay, bop: ab }, Op::Binary { x: b, y: by, bop: bb }) => a == b && ay == by && ab == bb,
            (Op::Stack { ops: a }, Op::Stack { ops: b }) => a == b,
            (Op::Storage { dtype: ad, scope: as_, len: al }, Op::Storage { dtype: bd, scope: bs, len: bl }) => {
                ad == bd && as_ == bs && al == bl
            }
            (Op::Store { dst: ad, src: as_, index: ai, layout: al }, Op::Store { dst: bd, src: bs, index: bi, layout: bl }) => {
                ad == bd && as_ == bs && ai == bi && al == bl
            }
            (Op::Load { src: as_, index: ai, layout: al }, Op::Load { src: bs, index: bi, layout: bl }) => {
                as_ == bs && ai == bi && al == bl
            }
            (Op::Range { axis: aa, kind: ak }, Op::Range { axis: ba, kind: bk }) => aa == ba && ak == bk,
            (Op::Loop { len: a }, Op::Loop { len: b }) => a == b,
            (Op::EndLoop, Op::EndLoop) => true,
            (Op::If { condition: a }, Op::If { condition: b }) => a == b,
            (Op::EndIf, Op::EndIf) => true,
            (Op::Mad { x: a, y: ay, z: az }, Op::Mad { x: b, y: by, z: bz }) => a == b && ay == by && az == bz,
            (Op::Index { vec: a, idx: ai }, Op::Index { vec: b, idx: bi }) => a == b && ai == bi,
            (Op::Barrier, Op::Barrier) => true,
            (
                Op::Wmma { dims: ad, layout: al, dtype: at, a, b: ab, c: ac },
                Op::Wmma { dims: bd, layout: bl, dtype: bt, a: ba, b: bb, c: bc },
            ) => ad == bd && al == bl && at == bt && a == ba && ab == bb && ac == bc,
            (
                Op::ReduceTile { x: a, scaler: as_, acc: aa, rop: ar, kind: ak },
                Op::ReduceTile { x: b, scaler: bs, acc: ba, rop: br, kind: bk },
            ) => a == b && as_ == bs && aa == ba && ar == br && ak == bk,
            (Op::MatmulTile { x: a, y: ay, acc: aa }, Op::MatmulTile { x: b, y: by, acc: ba }) => a == b && ay == by && aa == ba,
            (Op::TransposeTile { x: a }, Op::TransposeTile { x: b }) => a == b,
            (Op::BroadcastTile { x: a, kind: ak }, Op::BroadcastTile { x: b, kind: bk }) => a == b && ak == bk,
            (Op::Asm { asm: aa, ops: ao }, Op::Asm { asm: ba, ops: bo }) => aa == ba && ao == bo,
            (Op::Reduce { x: a, rop: ar, reduce_axis: aa }, Op::Reduce { x: b, rop: br, reduce_axis: ba }) => {
                a == b && ar == br && aa == ba
            }
            // Graph-only ops mirror the former `Node` equality: `After` never
            // merges; `time` is ignored on the boundary ops.
            (Op::After { .. }, Op::After { .. }) => false,
            (Op::ToDevice { x: a, device: ad, .. }, Op::ToDevice { x: b, device: bd, .. }) => a == b && ad == bd,
            (Op::Contiguous { x: a }, Op::Contiguous { x: b }) => a == b,
            (Op::Kernel { inputs: ai, outputs: ao, info: a }, Op::Kernel { inputs: bi, outputs: bo, info: b }) => {
                ai == bi && ao == bo && a.0 == b.0
            }
            (Op::Custom(_), Op::Custom(_)) => false,
            (Op::Reshape { x: a, shape: as_ }, Op::Reshape { x: b, shape: bs }) => a == b && as_ == bs,
            (Op::Expand { x: a, shape: as_ }, Op::Expand { x: b, shape: bs }) => a == b && as_ == bs,
            (Op::Permute { x: a, axes: aa }, Op::Permute { x: b, axes: ba }) => a == b && aa.as_slice() == ba.as_slice(),
            (Op::Flip { x: a, axes: aa }, Op::Flip { x: b, axes: ba }) => a == b && aa.as_slice() == ba.as_slice(),
            (Op::Pad { x: a, axis: aa, lp: al, len: alen }, Op::Pad { x: b, axis: ba, lp: bl, len: blen }) => {
                a == b && aa == ba && al == bl && alen == blen
            }
            (Op::Narrow { x: a, axis: aa, start: as_, len: al }, Op::Narrow { x: b, axis: ba, start: bs, len: bl }) => {
                a == b && aa == ba && as_ == bs && al == bl
            }
            // Different variants are never equal (Hash carries no
            // discriminant, so cross-variant collisions land here).
            _ => false,
        }
    }
}

impl Eq for Op {}

impl Hash for Op {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.disc().hash(state);
        match self {
            Op::Const(c) => c.hash(state),
            // `cons_id` is skipped: program caches (`get_hash`, incl. the C
            // backend disk cache) must keep sharing one program across
            // buffers. `Eq` stays stricter (it compares `cons_id`), which is
            // contract-legal and gives the egraph its leaf identity.
            Op::Param { dtype, kind, shape, .. } => {
                dtype.hash(state);
                kind.hash(state);
                shape.hash(state);
            }
            Op::Cast { x, dtype } => {
                x.hash(state);
                dtype.hash(state);
            }
            Op::Bitcast { x, dtype } => {
                x.hash(state);
                dtype.hash(state);
            }
            Op::Unary { x, uop } => {
                x.hash(state);
                uop.hash(state);
            }
            Op::Binary { x, y, bop } => {
                x.hash(state);
                y.hash(state);
                bop.hash(state);
            }
            Op::Stack { ops } => ops.hash(state),
            Op::Storage { dtype, scope, len } => {
                dtype.hash(state);
                scope.hash(state);
                len.hash(state);
            }
            Op::Store { dst, src, index, layout } => {
                dst.hash(state);
                src.hash(state);
                index.hash(state);
                layout.hash(state);
            }
            Op::Load { src, index, layout } => {
                src.hash(state);
                index.hash(state);
                layout.hash(state);
            }
            Op::Range { axis, kind } => {
                axis.hash(state);
                kind.hash(state);
            }
            Op::Loop { len } => len.hash(state),
            Op::EndLoop | Op::EndIf | Op::Barrier => {}
            Op::If { condition } => condition.hash(state),
            Op::Mad { x, y, z } => {
                x.hash(state);
                y.hash(state);
                z.hash(state);
            }
            Op::Index { vec, idx } => {
                vec.hash(state);
                idx.hash(state);
            }
            Op::Wmma { dims, layout, dtype, a, b, c } => {
                dims.hash(state);
                layout.hash(state);
                dtype.hash(state);
                a.hash(state);
                b.hash(state);
                c.hash(state);
            }
            Op::ReduceTile { x, scaler, acc, rop, kind } => {
                x.hash(state);
                scaler.hash(state);
                acc.hash(state);
                rop.hash(state);
                kind.hash(state);
            }
            Op::MatmulTile { x, y, acc } => {
                x.hash(state);
                y.hash(state);
                acc.hash(state);
            }
            Op::TransposeTile { x } => x.hash(state),
            Op::BroadcastTile { x, kind } => {
                x.hash(state);
                kind.hash(state);
            }
            Op::Asm { asm, ops } => {
                asm.hash(state);
                ops.hash(state);
            }
            Op::Reduce { x, rop, reduce_axis } => {
                x.hash(state);
                rop.hash(state);
                reduce_axis.hash(state);
            }
            Op::After { x, dep } => {
                x.hash(state);
                dep.hash(state);
            }
            Op::ToDevice { x, device, .. } => {
                x.hash(state);
                device.hash(state);
            }
            Op::Contiguous { x } => x.hash(state),
            Op::Kernel { inputs, outputs, info } => {
                inputs.hash(state);
                outputs.hash(state);
                info.0.hash(state);
            }
            Op::Custom(c) => {
                c.inputs.hash(state);
                c.outputs.hash(state);
                c.program_id.hash(state);
            }
            Op::Reshape { x, shape } => {
                x.hash(state);
                shape.hash(state);
            }
            Op::Expand { x, shape } => {
                x.hash(state);
                shape.hash(state);
            }
            Op::Permute { x, axes } => {
                x.hash(state);
                axes.as_slice().hash(state);
            }
            Op::Flip { x, axes } => {
                x.hash(state);
                axes.as_slice().hash(state);
            }
            Op::Pad { x, axis, lp, len } => {
                x.hash(state);
                axis.hash(state);
                lp.hash(state);
                len.hash(state);
            }
            Op::Narrow { x, axis, start, len } => {
                x.hash(state);
                axis.hash(state);
                start.hash(state);
                len.hash(state);
            }
        }
    }
}

impl PartialOrd for Op {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Op {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Op::Const(a), Op::Const(b)) => a.cmp(b),
            (
                Op::Param { dtype: ad, kind: ak, shape: as_, cons_id: ac },
                Op::Param { dtype: bd, kind: bk, shape: bs, cons_id: bc },
            ) => (ad, ak, as_, ac).cmp(&(bd, bk, bs, bc)),
            (Op::Cast { x: a, dtype: ad }, Op::Cast { x: b, dtype: bd }) => (a, ad).cmp(&(b, bd)),
            (Op::Bitcast { x: a, dtype: ad }, Op::Bitcast { x: b, dtype: bd }) => (a, ad).cmp(&(b, bd)),
            (Op::Unary { x: a, uop: au }, Op::Unary { x: b, uop: bu }) => (a, au).cmp(&(b, bu)),
            (Op::Binary { x: a, y: ay, bop: ab }, Op::Binary { x: b, y: by, bop: bb }) => (a, ay, ab).cmp(&(b, by, bb)),
            (Op::Stack { ops: a }, Op::Stack { ops: b }) => a.cmp(b),
            (Op::Storage { dtype: ad, scope: as_, len: al }, Op::Storage { dtype: bd, scope: bs, len: bl }) => {
                (ad, as_, al).cmp(&(bd, bs, bl))
            }
            (Op::Store { dst: ad, src: as_, index: ai, layout: al }, Op::Store { dst: bd, src: bs, index: bi, layout: bl }) => {
                (ad, as_, ai, al).cmp(&(bd, bs, bi, bl))
            }
            (Op::Load { src: as_, index: ai, layout: al }, Op::Load { src: bs, index: bi, layout: bl }) => {
                (as_, ai, al).cmp(&(bs, bi, bl))
            }
            (Op::Range { axis: aa, kind: ak }, Op::Range { axis: ba, kind: bk }) => (aa, ak).cmp(&(ba, bk)),
            (Op::Loop { len: a }, Op::Loop { len: b }) => a.cmp(b),
            (Op::EndLoop, Op::EndLoop) | (Op::EndIf, Op::EndIf) | (Op::Barrier, Op::Barrier) => std::cmp::Ordering::Equal,
            (Op::If { condition: a }, Op::If { condition: b }) => a.cmp(b),
            (Op::Mad { x: a, y: ay, z: az }, Op::Mad { x: b, y: by, z: bz }) => (a, ay, az).cmp(&(b, by, bz)),
            (Op::Index { vec: a, idx: ai }, Op::Index { vec: b, idx: bi }) => (a, ai).cmp(&(b, bi)),
            (
                Op::Wmma { dims: ad, layout: al, dtype: at, a, b: ab, c: ac },
                Op::Wmma { dims: bd, layout: bl, dtype: bt, a: ba, b: bb, c: bc },
            ) => (ad, al, at, a, ab, ac).cmp(&(bd, bl, bt, ba, bb, bc)),
            (
                Op::ReduceTile { x: a, scaler: as_, acc: aa, rop: ar, kind: ak },
                Op::ReduceTile { x: b, scaler: bs, acc: ba, rop: br, kind: bk },
            ) => (a, as_, aa, ar, ak).cmp(&(b, bs, ba, br, bk)),
            (Op::MatmulTile { x: a, y: ay, acc: aa }, Op::MatmulTile { x: b, y: by, acc: ba }) => (a, ay, aa).cmp(&(b, by, ba)),
            (Op::TransposeTile { x: a }, Op::TransposeTile { x: b }) => a.cmp(b),
            (Op::BroadcastTile { x: a, kind: ak }, Op::BroadcastTile { x: b, kind: bk }) => (a, ak).cmp(&(b, bk)),
            (Op::Asm { asm: aa, ops: ao }, Op::Asm { asm: ba, ops: bo }) => (aa, ao).cmp(&(ba, bo)),
            (Op::Reduce { x: a, rop: ar, reduce_axis: aa }, Op::Reduce { x: b, rop: br, reduce_axis: ba }) => {
                (a, ar, aa).cmp(&(b, br, ba))
            }
            // `Ord` ignores exactly what `Eq` ignores, so `Eq`-equal values
            // always compare `Equal`. `After`/`Custom` never compare equal,
            // so any two same-discriminant values order `Equal`.
            (Op::After { .. }, Op::After { .. }) => std::cmp::Ordering::Equal,
            (Op::ToDevice { x: a, device: ad, .. }, Op::ToDevice { x: b, device: bd, .. }) => (a, ad).cmp(&(b, bd)),
            (Op::Contiguous { x: a }, Op::Contiguous { x: b }) => a.cmp(b),
            (Op::Kernel { inputs: ai, outputs: ao, info: a }, Op::Kernel { inputs: bi, outputs: bo, info: b }) => {
                (ai, ao, a.0).cmp(&(bi, bo, b.0))
            }
            (Op::Custom(_), Op::Custom(_)) => std::cmp::Ordering::Equal,
            (Op::Reshape { x: a, shape: as_ }, Op::Reshape { x: b, shape: bs }) => (a, as_).cmp(&(b, bs)),
            (Op::Expand { x: a, shape: as_ }, Op::Expand { x: b, shape: bs }) => (a, as_).cmp(&(b, bs)),
            (Op::Permute { x: a, axes: aa }, Op::Permute { x: b, axes: ba }) => (a, aa.as_slice()).cmp(&(b, ba.as_slice())),
            (Op::Flip { x: a, axes: aa }, Op::Flip { x: b, axes: ba }) => (a, aa.as_slice()).cmp(&(b, ba.as_slice())),
            (Op::Pad { x: a, axis: aa, lp: al, len: alen }, Op::Pad { x: b, axis: ba, lp: bl, len: blen }) => {
                (a, aa, al, alen).cmp(&(b, ba, bl, blen))
            }
            (Op::Narrow { x: a, axis: aa, start: as_, len: al }, Op::Narrow { x: b, axis: ba, start: bs, len: bl }) => {
                (a, aa, as_, al).cmp(&(b, ba, bs, bl))
            }
            // Only different-variant pairs reach here: order by discriminant.
            (a, b) => a.disc().cmp(&b.disc()),
        }
    }
}

/// Which dimension a `Op::ReduceTile` collapses, or a
/// `Op::BroadcastTile` replicates, within each 32x32 tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum TileDim {
    /// One value per row (reduce: 32 values carried in the result
    /// tile's first row; broadcast: one row replicated to all rows).
    Row,
    /// One value per column (reduce: 32 values carried in the result
    /// tile's first row; broadcast: one column replicated to all columns).
    Col,
    /// Whole tile to/from a single scalar.
    Scalar,
}

/// Scope of index. Index is like loop, but purely parallel acess
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum RangeKind {
    /// Group scope. Represents blocks in cuda, cores in CPU and tenstorrent.
    Group(OpId),
    /// Local scope. Represents cuda threads.
    Local(u32),
    /// Warp scope. Represents warps and wavefronts. References a `Local`
    /// range whose threads form hardware warps; the op's value is the lane
    /// id within the warp (`0..warp_size`, warp size from `DeviceInfo`).
    Warp(OpId),
}

impl std::fmt::Display for RangeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RangeKind::Group(x) => f.write_fmt(format_args!("group_r{x}")),
            RangeKind::Local(x) => f.write_fmt(format_args!("local_{x}")),
            RangeKind::Warp(x) => f.write_fmt(format_args!("warp_{x}")),
        }
    }
}

/// Unary operations for element-wise kernel transformations.
///
/// These operations are applied to a single input tensor.
///
/// # Variants
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, SerBin, DeBin)]
pub enum UOp {
    /// Negation: -x
    Neg,
    /// Logical NOT: !x
    Not,
    /// Bitwise NOT: ~x
    BitNot,
    /// Exponential: e^x
    Exp,
    /// Exponential with base 2: 2^x
    Exp2,
    /// Logarithm with base 2: log2(x)
    Log2,
    /// Reciprocal: 1/x
    Reciprocal,
    /// Square root: sqrt(x)
    Sqrt,
    /// Reciprocal square root: 1/sqrt(x)
    Rsqrt,
    /// Sine: sin(x)
    Sin,
    /// Cosine: cos(x)
    Cos,
    /// Floor: floor(x)
    Floor,
    /// Truncate toward zero: trunc(x)
    Trunc,
    /// Absolute value: |x|
    Abs,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, SerBin, DeBin)]
/// Binary operations for element-wise or reduction kernel operations.
///
/// These operations take two input tensors and produce an output.
///
/// # Variants
pub enum BOp {
    /// Addition: x + y
    Add,
    /// Subtraction: x - y
    Sub,
    /// Multiplication: x * y
    Mul,
    /// Division: x / y
    Div,
    /// Power: x^y
    Pow,
    /// Modulo: x % y
    Mod,
    /// Compare less than: x < y
    Cmplt,
    /// Compare greater than: x > y
    Cmpgt,
    /// Compare greater than or equal: x >= y
    Cmpge,
    /// Maximum: max(x, y)
    Max,
    /// Bitwise OR: x | y
    Or,
    /// Bitwise AND: x & y
    And,
    /// Bitwise XOR: x ^ y
    BitXor,
    /// Bitwise OR: x | y
    BitOr,
    /// Bitwise AND: x & y
    BitAnd,
    /// Left shift: x << y
    BitShiftLeft,
    /// Right shift: x >> y
    BitShiftRight,
    /// Not equal: x != y
    NotEq,
    /// Equal: x == y
    Eq,
}

impl BOp {
    /// Returns true if the binary operation is associative:
    /// `(a op b) op c == a op (b op c)`.
    pub const fn is_associative(self) -> bool {
        use BOp::{Add, And, BitAnd, BitOr, BitShiftLeft, BitShiftRight, BitXor, Max, Mul, Or};
        matches!(self, Add | Mul | And | Or | BitXor | BitAnd | BitOr | BitShiftLeft | BitShiftRight | Max)
    }

    /// Returns true if the binary operation is commutative:
    /// `a op b == b op a`.
    pub const fn is_commutative(self) -> bool {
        use BOp::{Add, And, BitAnd, BitOr, BitXor, Max, Mul, Or};
        matches!(self, Add | Mul | And | Or | BitXor | BitAnd | BitOr | Max)
    }

    /// Returns true if the operation produces a boolean result.
    pub const fn returns_bool(self) -> bool {
        use BOp::{And, Cmpge, Cmpgt, Cmplt, Eq, NotEq, Or};
        matches!(self, Cmpgt | Cmpge | Cmplt | NotEq | Eq | And | Or)
    }
}

/// Matrix multiply dimensions for tensor core operations.
///
/// Represents the shape (m, n, k) for matrix multiplication.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMADims {
    /// 8x8 with k=16
    m8n8k16,
    /// 16x8 with k=8
    m16n8k8,
    /// 16x8 with k=16
    m16n8k16,
    /// 32x8 with k=16 (int8)
    m32n8k16,
    /// 8x32 with k=16 (int8)
    m8n32k16,
    /// 8x8 with k=32 (int4)
    m8n8k32,
    /// 8x8 with k=128 (b1)
    m8n8k128,
}

impl MMADims {
    /// Decompose MMAD dimensions into m, n, k components.
    pub const fn decompose_mnk(self) -> (u64, u64, u64) {
        match self {
            MMADims::m8n8k16 => (8, 8, 16),
            MMADims::m16n8k8 => (16, 8, 8),
            MMADims::m16n8k16 => (16, 8, 16),
            MMADims::m32n8k16 => (32, 8, 16),
            MMADims::m8n32k16 => (8, 32, 16),
            MMADims::m8n8k32 => (8, 8, 32),
            MMADims::m8n8k128 => (8, 8, 128),
        }
    }
}

/// Memory layout for tensor core matrix operands.
///
/// Describes how matrix data is stored in memory.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMALayout {
    /// Row-major for A, column-major for B — the only layout `mma.sync` accepts
    row_col,
}

/// Data type for matrix multiply operations.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMADType {
    /// FP16 input with FP32 accumulator
    f16_f16_f16_f32,
    /// FP16 input with FP16 accumulator
    f16_f16_f16_f16,
    /// 8 bit signed integer input with 32 bit signed integer accumulator
    s8_s8_s32_s32,
    /// 4 bit signed integer input with 32 bit signed integer accumulator
    s4_s4_s32_s32,
    /// 1 bit input with 32 bit signed integer accumulator, XOR + popc reduction
    b1_b1_s32_xor_popc,
    /// 1 bit input with 32 bit signed integer accumulator, AND + popc reduction
    b1_b1_s32_and_popc,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
pub struct OpLinked {
    pub prev: OpId,
    pub next: OpId, // Use Vec<OpId> instead for egraph
    pub op: Op,
}

const _: () = assert!(core::mem::size_of::<Op>() == 24);
const _: () = assert!(core::mem::size_of::<OpLinked>() == 32);

/// Operation ID for kernel operations.
///
/// This is a unique identifier for each operation in the kernel IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct OpId(pub(crate) u32);

impl OpId {
    /// NULL
    pub const NULL: Self = Self(u32::MAX);

    /// Check if this OpId is null.
    pub const fn is_null(self) -> bool {
        self.0 == u32::MAX
    }
}

impl std::fmt::Display for OpId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl From<usize> for OpId {
    fn from(value: usize) -> Self {
        OpId(value as u32)
    }
}

impl From<OpId> for usize {
    fn from(value: OpId) -> usize {
        value.0 as usize
    }
}

impl SlabId for OpId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl MemLayout {
    /// Get the number of elements in the memory layout.
    pub(crate) fn n_elements(self) -> Dim {
        match self {
            MemLayout::Scalar => 1,
            MemLayout::Vector(x) => x.into(),
            MemLayout::Tile { x, y, .. } => x as Dim * y as Dim,
        }
    }
}

impl std::fmt::Display for MemLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemLayout::Scalar => f.write_fmt(format_args!("Scalar")),
            MemLayout::Vector(x) => f.write_fmt(format_args!("Vec({x})")),
            MemLayout::Tile { x, y, stride } => f.write_fmt(format_args!("Tile({x}x{y} st={stride})")),
        }
    }
}

impl Op {
    // TODO use custom non allocating iterator instead of allocating a vec
    #[allow(clippy::match_same_arms)]
    pub(crate) fn parameters(&self) -> impl DoubleEndedIterator<Item = OpId> {
        match self {
            Op::Const { .. } | Op::Storage { .. } | Op::EndLoop | Op::Barrier | Op::EndIf => {
                vec![]
            }
            &Op::Param { shape, .. } => {
                // Shape is null after linearize
                if shape.is_null() { vec![] } else { vec![shape] }
            }
            &Op::Range { kind, .. } => match kind {
                RangeKind::Group(len) => vec![len],
                RangeKind::Local(_) => vec![],
                RangeKind::Warp(local_id) => vec![local_id],
            },
            &Op::Loop { len, .. } => vec![len],
            &Op::Reshape { x, shape, .. } | &Op::Expand { x, shape } => vec![x, shape],
            &Op::Permute { x, .. } | &Op::Flip { x, .. } => vec![x],
            &Op::Pad { x, lp, len, .. } => vec![x, lp, len],
            &Op::Narrow { x, start, len, .. } => vec![x, start, len],
            Op::Reduce { x, reduce_axis, .. } => vec![*x, *reduce_axis],
            &Op::Store { dst, src, index, .. } => {
                // Pre-linearize stores carry a NULL index (whole-view write),
                // which names no operand.
                if index.is_null() {
                    vec![dst, src]
                } else {
                    vec![dst, src, index]
                }
            }
            Op::Cast { x, .. } => vec![*x],
            Op::Bitcast { x, .. } => vec![*x],
            Op::Unary { x, .. } => vec![*x],
            &Op::Binary { x, y, .. } => vec![x, y],
            &Op::Load { src, index, .. } => vec![src, index],
            &Op::Mad { x, y, z } => vec![x, y, z],
            Op::Asm { ops, .. } => ops.iter().copied().collect(),
            Op::Stack { ops } => ops.iter().copied().collect(),
            &Op::Index { vec, .. } => vec![vec],
            &Op::Wmma { a, b, c, .. } => vec![a, b, c],
            Op::If { condition } => vec![*condition],
            &Op::MatmulTile { x, y, acc } => vec![x, y, acc],
            &Op::TransposeTile { x } => vec![x],
            &Op::BroadcastTile { x, .. } => vec![x],
            &Op::ReduceTile { x, acc, scaler, .. } => vec![x, acc, scaler],
            Op::After { x, dep } => vec![*x, *dep],
            Op::ToDevice { x, .. } => vec![*x],
            Op::Contiguous { x } => vec![*x],
            Op::Kernel { .. } | Op::Custom(_) => {
                todo!("parameters: graph-only op in ordered kernel")
            }
        }
        .into_iter()
    }

    #[allow(clippy::match_same_arms)]
    pub(crate) fn parameters_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut OpId> {
        match self {
            Op::Const { .. } | Op::Storage { .. } | Op::EndLoop | Op::EndIf | Op::Barrier => vec![],
            Op::Param { shape, .. } => {
                // Shape is null after linearize
                if shape.is_null() { vec![] } else { vec![shape] }
            }
            Op::Range { kind, .. } => match kind {
                RangeKind::Group(len) => vec![len],
                RangeKind::Local(_) => vec![],
                RangeKind::Warp(local_id) => vec![local_id],
            },
            Op::Loop { len, .. } => vec![len],
            Op::Reshape { x, shape, .. } | Op::Expand { x, shape } => vec![x, shape],
            Op::Permute { x, .. } | Op::Flip { x, .. } => vec![x],
            Op::Pad { x, lp, len, .. } => vec![x, lp, len],
            Op::Narrow { x, start, len, .. } => vec![x, start, len],
            Op::Reduce { x, reduce_axis, .. } => vec![x, reduce_axis],
            Op::Store { dst, src: x, index, .. } => {
                // Pre-linearize stores carry a NULL index (whole-view write).
                if index.is_null() { vec![dst, x] } else { vec![dst, x, index] }
            }
            Op::Cast { x, .. } => vec![x],
            Op::Bitcast { x, .. } => vec![x],
            Op::Unary { x, .. } => vec![x],
            Op::Binary { x, y, .. } => vec![x, y],
            Op::Load { src, index, .. } => vec![src, index],
            Op::Mad { x, y, z } => vec![x, y, z],
            Op::Stack { ops } => ops.iter_mut().collect(),
            Op::Index { vec, .. } => vec![vec],
            Op::Wmma { a, b, c, .. } => vec![a, b, c],
            Op::If { condition } => vec![condition],
            Op::MatmulTile { x, y, acc } => vec![x, y, acc],
            Op::ReduceTile { x, acc, scaler, .. } => vec![x, acc, scaler],
            Op::TransposeTile { x } => vec![x],
            Op::BroadcastTile { x, .. } => vec![x],
            Op::Asm { ops, .. } => ops.iter_mut().collect(),
            Op::After { .. } | Op::ToDevice { .. } | Op::Contiguous { .. } | Op::Kernel { .. } | Op::Custom(_) => {
                todo!("parameters_mut: graph-only op in ordered kernel")
            }
        }
        .into_iter()
    }

    /// Check if this operation is a constant.
    pub(crate) const fn is_const(&self) -> bool {
        matches!(self, Op::Cast { .. })
    }

    /// Check if this operation is a load.
    pub(crate) const fn is_load(&self) -> bool {
        matches!(self, Op::Load { .. })
    }

    /// Remap parameter IDs according to a mapping.
    pub(crate) fn remap_params(&mut self, remapping: &Map<OpId, OpId>) {
        for param in self.parameters_mut() {
            if let Some(remapped_id) = remapping.get(param) {
                *param = *remapped_id;
            }
        }
    }
}

impl std::fmt::Display for MemScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            MemScope::Global => "global",
            MemScope::Local => "local",
            MemScope::Register => "reg",
            MemScope::Circular => "cb",
        })
    }
}

impl std::fmt::Display for ParamKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ParamKind::Variable => "var",
            ParamKind::Global => "global",
            ParamKind::GlobalMut => "global mut",
        })
    }
}
