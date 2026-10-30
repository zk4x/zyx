// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent kernel passes: tile the IR, lower it to native ops, and
//! verify the reader/compute/writer stream shape.
//!
//! Pipeline order: `tt_tile` -> `tt_lower_llks` -> `tt_dst_reuse` ->
//! `verify` -> `tt_verify`. The passes are public so external pass authors
//! can reuse or replace stages. Calling them out of order is a loud panic,
//! never silent corruption.
//!
//! Lowering is SSA over DST: every native op writes a fresh
//! `Op::Storage { scope: Register }` and the `TTOp` carries that storage
//! OpId as its `slot`. A separate pass (`tt_dst_reuse`) assigns the 8-16
//! physical tile registers by liveness and rewrites operands, so lowering
//! never needs a refcounted slot map and codegen only maps a physical
//! storage to its DST row. Coalescing an input and output onto one row is
//! also the reuse pass's job, not a lowering special case.

use super::split::SplitMode;
use crate::DType;
use crate::Map;
use crate::Set;
use crate::dtype::Constant;
use crate::kernel::Pat;
use crate::kernel::{BOp, Bindings, Kernel, MemScope, Op, OpId, ParamKind, RangeKind, TTOp, TileDim, UOp};
use crate::shape::Dim;

/// Matmul role hand-off: T2 may pack the accumulator (T1 posts) and T1 may
/// clear it again (T2 posts). The numbers are the run's semaphore indices;
/// both are initialised by the matmul lowering (`TTOp::Semset`).
const TT_SEM_READY: u8 = 0;
const TT_SEM_FREE: u8 = 1;
/// `Before::MATRIX` as codegen's `tt_before` numbering.
const TT_BEFORE_MATRIX: u8 = 2;
/// `Before::PACKER` as codegen's `tt_before` numbering.
const TT_BEFORE_PACKER: u8 = 1;

/// One phase-2 pattern and the native lowering it emits. The list is
/// ordered biggest-first; the recursive walk tries each pattern at a value
/// root and stops at the first emitter that succeeds. The SFPU variant is
/// data, not a per-op arm: adding an elementwise op is a table entry.
#[derive(Clone, Copy)]
enum TtEmit {
    /// Elementwise SFPU: `arity` 1 or 2 DST inputs, one DST result.
    Sfpu { kind: u32, arity: u8 },
    /// Hardware Matrix Unit: CB inputs, DST result.
    Matmul,
    /// Structural tile reduce: DST input, DST result.
    Reduce(TileDim),
}

/// zyx unary op -> metalium SFPU kind. `None` means no native kind yet;
/// the value stays high-level instead of being mis-lowered.
fn uop_kind(uop: UOp) -> Option<u32> {
    use crate::backend::tenstorrent::kern::sfpu::ops::kind_sfpu as sf;
    Some(match uop {
        UOp::Neg => sf::NEG,
        UOp::Not => sf::BOOL_NOT,
        UOp::BitNot => sf::INT_NOT,
        // No direct 2^x / log2 kind in metalium yet; the fused sigmoid
        // pattern covers the exp2 inside sigmoid, and a standalone exp2
        // stays high-level until the table grows a program entry.
        UOp::Exp2 | UOp::Log2 => return None,
        UOp::Reciprocal => sf::RECIP,
        UOp::Sqrt => sf::SQRT,
        UOp::Rsqrt => sf::RSQRT,
        UOp::Sin => sf::SIN,
        UOp::Cos => sf::COS,
        UOp::Floor => sf::FLOOR,
        UOp::Trunc => sf::TRUNC,
        UOp::Abs => sf::ABS,
    })
}

/// zyx binary op -> metalium SFPU kind. The same table covers float,
/// integer, and comparison kinds; `None` is "no native kind yet".
fn bop_kind(bop: BOp) -> Option<u32> {
    use crate::backend::tenstorrent::kern::kind;
    use crate::backend::tenstorrent::kern::sfpu::ops::kind_sfpu as sf;
    Some(match bop {
        BOp::Add => kind::ADD,
        BOp::Sub => kind::SUB,
        BOp::Mul => kind::MUL,
        BOp::Div => sf::DIV,
        BOp::Pow => sf::POW,
        BOp::Mod => sf::REM,
        BOp::Cmplt => sf::LT,
        BOp::Cmpgt => sf::GT,
        BOp::Cmpge => sf::GE,
        // Tensor-tensor max has no direct SFPU kind; reduce Max is a
        // structural `ReduceTile`, not an elementwise op.
        BOp::Max => return None,
        BOp::Or | BOp::BitOr => sf::INT_OR,
        BOp::And | BOp::BitAnd => sf::INT_AND,
        BOp::BitXor => sf::INT_XOR,
        BOp::BitShiftLeft => sf::INT_SHL,
        BOp::BitShiftRight => sf::INT_SHR,
        BOp::NotEq => sf::NE,
        BOp::Eq => sf::EQ,
    })
}

/// Every zyx unary op. `uop_kind` filters out the ones without a native
/// SFPU kind, so the table stays data-only.
const UOPS: &[UOp] = &[
    UOp::Neg,
    UOp::Not,
    UOp::BitNot,
    UOp::Exp2,
    UOp::Log2,
    UOp::Reciprocal,
    UOp::Sqrt,
    UOp::Rsqrt,
    UOp::Sin,
    UOp::Cos,
    UOp::Floor,
    UOp::Trunc,
    UOp::Abs,
];

/// Every zyx binary op. `bop_kind` filters out the ones without a native
/// SFPU kind.
const BOPS: &[BOp] = &[
    BOp::Add,
    BOp::Sub,
    BOp::Mul,
    BOp::Div,
    BOp::Pow,
    BOp::Mod,
    BOp::Cmplt,
    BOp::Cmpgt,
    BOp::Cmpge,
    BOp::Max,
    BOp::Or,
    BOp::And,
    BOp::BitXor,
    BOp::BitOr,
    BOp::BitAnd,
    BOp::BitShiftLeft,
    BOp::BitShiftRight,
    BOp::NotEq,
    BOp::Eq,
];

/// State for the recursive phase-2 lowering. `rcs` is the original
/// high-level use count: a computed value with more than one use is packed
/// to a Circular CB, because DST is single-use scratch and multi-use values
/// live in L1 (the Tensix dataflow model). `materialized` maps that value
/// to its CB; every consumer unpacks from it. The redundant first unpack
/// is deliberate — a later pass can coalesce it with the producer's DST
/// row without complicating lowering.
struct TtLowering<'a> {
    pats: &'a [(Pat, TtEmit)],
    prologue: OpId,
    reader_cb_of: &'a Map<OpId, OpId>,
    rcs: &'a Map<OpId, u32>,
    materialized: Map<OpId, OpId>,
    /// Matmul feeder index patterns (A plain, B K-scaled).
    pat_a: Pat,
    pat_b: Pat,
    /// The matmul accumulator that has a role hand-off: its DST slot, and the
    /// `(ready, free)` semaphores. T1 takes `free` before clearing the
    /// accumulator and posts `ready` when the last `MVMUL` drained; T2 takes
    /// `ready` before packing and posts `free` after. Only the pack of *this*
    /// slot waits — any other pack has nothing to hand off.
    matmul: Option<(OpId, u8, u8)>,
}

impl Kernel {
    /// Section-split constructor: end of the reader section. Section
    /// assignment is a placement decision, so markers are builder
    /// ops — render only splits output at them, never derives them.
    pub fn tt_end_reader(&mut self) -> OpId {
        self.push_back(Op::TT(TTOp::EndReader))
    }

    /// Section-split constructor: end of the compute section. Same
    /// placement rules as [`Kernel::tt_end_reader`].
    pub fn tt_end_compute(&mut self) -> OpId {
        self.push_back(Op::TT(TTOp::EndCompute))
    }

    /// Tenstorrent tiling: pad to multiples of 32 and unroll loops
    /// and group ranges to multiples of 32. No special tiling or
    /// vectorization passes: the 32-wide `Unroll` lanes plus composed
    /// indices are the tile IR the backend matchers read.
    pub fn tt_tile(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        self.default_epilogue();
        self.default_epilogue();

        // Gather loops, pad to 32, split unroll-32 FIRST: split's
        // loop arm allows exactly one acc store per body, so it must
        // run before coarsening replicates the body. pad_to pads
        // lengths only on TT (no guards); already-multiples are
        // untouched. Non-reduce loops are a no-op inside (no acc
        // found); anything else keeps its shape for later cuts.
        let mut pad_loops: Vec<OpId> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if matches!(x, Op::Loop { .. }) {
                pad_loops.push(op);
            }
        }
        for l in pad_loops {
            self.pad_to(l, 32);
            let len = match self.ops[l].op {
                Op::Loop { len, .. } => len,
                _ => unreachable!("pad_to preserves the Loop variant"),
            };
            // resolve_const evaluates const-expr lengths on demand, so
            // freshly padded (unfolded) round-ups resolve directly.
            if self.resolve_const(len).and_then(|c| c.as_dim()).is_some_and(|l| l % 32 == 0) {
                self.split(l, SplitMode::Unroll, 32);
            }
        }

        self.default_epilogue();
        self.default_epilogue();

        // Gather group ranges, pad to 32, coarsen LAST: duplicate
        // the vectorized body, one lane per row. Anything else
        // (symbolic, indivisible) is skipped, never fatal. The device
        // grid is 2D: one group range is pre-split into two nested
        // groups below, two proceed as-is, more than two is a later
        // increment.
        let mut pad_groups: Vec<OpId> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if matches!(x, Op::Range { kind: RangeKind::Group(_), .. }) {
                pad_groups.push(op);
            }
        }
        if pad_groups.len() > 2 {
            todo!("tt_tile: more than two group ranges");
        }
        if pad_groups.len() == 1 {
            // One group range: split into two nested groups of roughly
            // equal size (user spec, unconditional — no loop check, no
            // length threshold), then pad and unroll each normally below.
            let g = pad_groups[0];
            self.pad_to(g, 32);
            let len = match self.ops[g].op {
                Op::Range { kind: RangeKind::Group(len), .. } => self.resolve_const(len).and_then(|c| c.as_dim()),
                _ => unreachable!("pad_to preserves the Group range"),
            };
            if let Some(len) = len {
                // Largest divisor at or below sqrt(len): the two
                // factors are as equal as possible.
                let mut f = (len as f64).sqrt() as Dim;
                while f > 1 && len % f != 0 {
                    f -= 1;
                }
                if f > 1 {
                    let factor = u16::try_from(f).expect("tt_tile: group split factor overflows u16");
                    // The inner group takes a new axis: highest group
                    // axis + 1 (user spec; verify requires unique group
                    // axes, and split reuses the axis). TT-local: split
                    // itself is generic across backends.
                    let mut max_group_axis = 0u32;
                    for (_, x) in self.iter_unordered() {
                        if let Op::Range { axis, kind: RangeKind::Group(_) } = x {
                            max_group_axis = max_group_axis.max(*axis);
                        }
                    }
                    let inner = self.split(g, SplitMode::Group, factor);
                    if !inner.is_null() {
                        let new_axis = max_group_axis.checked_add(1).expect("tt_tile: group axis overflows u32");
                        match &mut self.ops[inner].op {
                            Op::Range { axis, kind: RangeKind::Group(_) } => *axis = new_axis,
                            _ => unreachable!("split Group returns the inner group range"),
                        }
                    }
                }
            }
            pad_groups.clear();
            for (op, x) in self.iter_unordered() {
                if matches!(x, Op::Range { kind: RangeKind::Group(_), .. }) {
                    pad_groups.push(op);
                }
            }
        }
        for g in pad_groups {
            self.pad_to(g, 32);
            let len = match self.ops[g].op {
                Op::Range { kind: RangeKind::Group(len), .. } => len,
                _ => unreachable!("pad_to preserves the Group range"),
            };
            if self.resolve_const(len).and_then(|c| c.as_dim()).is_some_and(|l| l % 32 == 0) {
                self.split(g, SplitMode::Unroll, 32);
            }
        }

        self.default_epilogue();
        self.default_epilogue();

        self.verify();
    }

    /// General TT lowering + register allocation (matmul, reduce,
    /// sigmoid/silu carried). Traffic is part of emission: the pass
    /// matches compute, creates the CBs (DRAM-backed reader/writer CBs
    /// plus interior CBs), emits the DRAM↔CB movement as `TTOp::Copy`
    /// (explicit bytes, scalar base indices — unrolls stripped), places
    /// `EndReader`/`EndCompute`, lowers each match to a structured LLK
    /// op over storages only, and allocates the Register storages in
    /// the same step. Render only substitutes numbers; stripped unrolls
    /// dangle into DCE.
    ///
    /// Lowered values chain through DST: a reduce reads its feeder's
    /// storage and writes a new one; sigmoid/silu run in place on their
    /// feeder's storage; matmul reads CBs and writes a fresh DST row.
    /// Only a value that reaches a DRAM output is packed (phase 3).
    /// Anything unmatched is left in place for later increments.
    /// Canonicalize a single-tile reduce's traffic to the physical tile
    /// pitch.
    ///
    /// `tt_tile` builds the reader index from the *logical* layout: a
    /// reduce over a contiguous axis loads a rank-3 cube (`[32, 32, 32]`
    /// for a whole tile) and a reduce over a tiled rank-2 value loads at
    /// the logical pitch. Neither is what the NOC moves: the reader reads
    /// one 32x32 tile from its base, and the reduce wants the reduced axis
    /// as the innermost tile lane.
    ///
    /// This pass recognizes the one template zyx emits today — exactly one
    /// `Reduce(Add)` fed by exactly one DRAM load and drained by exactly one
    /// DRAM store — and rewrites both indices to `row * 32 + lane`, so the
    /// load publishes one tile and the store drains one row. Anything else
    /// is left untouched (multi-reduce, loops, groups, matmul cubes).
    pub fn tt_reindex(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        // Single-tile kernels only: loops and groups mean extents or
        // multi-tile traffic this rewrite does not model.
        for (_, x) in self.iter_unordered() {
            match x {
                Op::Loop { .. } => return,
                Op::Range { kind: RangeKind::Group(_), .. } => return,
                _ => {}
            }
        }
        // Exactly one reduce, one DRAM load, one DRAM store.
        let mut reduces: Vec<OpId> = Vec::new();
        let mut loads: Vec<OpId> = Vec::new();
        let mut stores: Vec<OpId> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            match self.ops[id].op {
                Op::Reduce { .. } => reduces.push(id),
                Op::Load { src: base, .. } => {
                    if matches!(self.ops[base].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. }) {
                        loads.push(id);
                    }
                }
                Op::Store { dst: base, .. } => {
                    if matches!(self.ops[base].op, Op::Param { kind: ParamKind::GlobalMut, .. }) {
                        stores.push(id);
                    }
                }
                _ => {}
            }
            id = self.ops[id].next;
        }
        if reduces.len() != 1 || loads.len() != 1 || stores.len() != 1 {
            return;
        }
        let root = reduces[0];
        let load = loads[0];
        let store = stores[0];
        // Reduce(Add) over the DRAM load; the store drains the reduce.
        let (x, k) = match &self.ops[root].op {
            Op::Reduce { x, reduce_axis, rop } if matches!(rop, BOp::Add) => (*x, *reduce_axis),
            _ => return,
        };
        if x != load {
            return;
        }
        let ssrc = match &self.ops[store].op {
            Op::Store { src, .. } => *src,
            _ => return,
        };
        if ssrc != root {
            return;
        }
        // Rank-3 cube to a rank-2 result (axis drop), or a
        // logical-pitch rank-2 tile to a row (pitch substitution).
        // Anything else (matmul cubes, scalars) is not this template.
        let drop3 = self.shape(load) == vec![32, 32, 32] && self.shape(root) == vec![32, 32];
        let row2 = self.shape(load) == vec![32, 32] && self.shape(root) == vec![32];
        if !drop3 && !row2 {
            return;
        }
        let lidx = match self.ops[load].op {
            Op::Load { src_idx, .. } => src_idx,
            _ => unreachable!("tt_reindex: load lost its index"),
        };
        // Stride anatomy of the load index: the reduce lane at stride
        // 1, one row axis at stride W, and (cube only) dropped axes at
        // multiples of W. A const offset must be zero (a nonzero
        // logical offset is not a physical one); anything else bails.
        let Some(strides) = self.get_strides(lidx) else {
            return;
        };
        if !matches!(self.ops[k].op, Op::Unroll { len: 32, .. }) {
            return;
        }
        match strides.get(&k) {
            Some((32, 1)) => {}
            _ => return,
        }
        let mut rows: Vec<(OpId, Dim)> = Vec::new();
        for (var, (len, stride)) in strides.iter() {
            if *var == OpId::NULL {
                if *stride != 0 {
                    return;
                }
                continue;
            }
            if *var == k {
                continue;
            }
            if !matches!(self.ops[*var].op, Op::Unroll { len: 32, .. }) {
                return;
            }
            if *len != 32 || *stride <= 0 {
                return;
            }
            rows.push((*var, *stride));
        }
        // Keep the smallest stride above 1 as the row axis; every
        // other axis must stride a multiple of it (pure upward row
        // shift, never a new column structure).
        rows.sort_by_key(|&(_, s)| s);
        let row = match rows.as_slice() {
            [(r, w)] if row2 && *w != 32 && *w > 1 => *r,
            [(r, w), (_d, dw)] if drop3 && *w > 1 && *dw > *w && *dw % *w == 0 => *r,
            _ => return,
        };
        // Canonical traffic indices at the physical tile pitch.
        let phys = self.insert_const_idx_before(load, 32i64);
        let rk = self.insert_before(load, Op::Binary { x: row, y: phys, bop: BOp::Mul });
        let new_idx = self.insert_before(load, Op::Binary { x: rk, y: k, bop: BOp::Add });
        match &mut self.ops[load].op {
            Op::Load { src_idx, .. } => *src_idx = new_idx,
            _ => unreachable!("tt_reindex: load lost its index"),
        }
        match &mut self.ops[store].op {
            Op::Store { dst_idx, .. } => *dst_idx = row,
            _ => unreachable!("tt_reindex: store lost its index"),
        }
        self.verify();
    }

    pub fn tt_lower_llks(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        for (_, x) in self.iter_unordered() {
            if let Op::Loop { .. } = x {
                return;
            }
        }
        // CB Storage definitions live in the prologue (past leading
        // params/storages, before everything else): inserting before
        // `prologue` keeps it stable.
        let mut prologue = self.head;
        for _ in 0..250_000 {
            if prologue.is_null() {
                break;
            }
            match self.ops[prologue].op {
                Op::Param { .. } | Op::Storage { .. } => prologue = self.ops[prologue].next,
                _ => break,
            }
        }
        // Phase-2 pattern table, biggest first. The recursive lowering
        // tries each pattern at a value root and stops at the first
        // emitter that succeeds; a failed emitter falls through.
        let tile = Pat::bind('t').shape([32i64, 32]);
        // Single-tile matmul: Reduce(Add) over a tiled Mul to a
        // tiled root. Both orders (matmul is elementwise-commutative
        // in IR; the A/B roles resolve structurally at install).
        let ma = Pat::bind('a').shape([32i64, 32]);
        let mb = Pat::bind('b').shape([32i64, 32]);
        let mmul = Pat::sum((ma.clone() * mb.clone()).shape([32i64, 32, 32])).shape([32i64, 32]);
        let mmul_swap = Pat::sum((mb.clone() * ma.clone()).shape([32i64, 32, 32])).shape([32i64, 32]);
        // Feeder index shapes: K-plain (A) vs K-scaled (B). Add/Mul
        // commutativity covers operand orders; each match_pat call
        // binds at most 3 (fresh budget per call).
        let pat_a = Pat::bind('k') + (Pat::bind('d') * Pat::bind('e'));
        let pat_b = Pat::bind('f') + (Pat::bind('g') * Pat::bind('k'));
        // Fused sigmoid composite, single tile. Builder spelling
        // (`tensor/elementwise.rs`): `sigmoid = recip(add(exp2(mul(neg(x),
        // LOG2_E)), 1))`; the scalar consts match any `Const` (builders
        // cast them to the tile dtype). Rank-1 handles a reduce result
        // feeding elementwise (`sum` -> `sigmoid`).
        let sx = Pat::bind('x').shape([32i64, 32]);
        let sigmoid_pat = (sx.neg().exp() + Pat::Const).recip().shape([32i64, 32]);
        let sx1 = Pat::bind('x').shape([32i64]);
        let sigmoid_pat1 = (sx1.neg().exp() + Pat::Const).recip().shape([32i64]);
        let mut pats: Vec<(Pat, TtEmit)> = vec![
            (sigmoid_pat, TtEmit::Sfpu { kind: crate::backend::tenstorrent::kern::sfpu::ops::kind_sfpu::SIGMOID, arity: 1 }),
            (sigmoid_pat1, TtEmit::Sfpu { kind: crate::backend::tenstorrent::kern::sfpu::ops::kind_sfpu::SIGMOID, arity: 1 }),
            (mmul, TtEmit::Matmul),
            (mmul_swap, TtEmit::Matmul),
            (Pat::sum(tile.clone()).shape([32i64]), TtEmit::Reduce(TileDim::Row)),
            (Pat::reduce(tile.clone(), BOp::Max).shape([32i64]), TtEmit::Reduce(TileDim::Row)),
            (Pat::sum(tile.clone()).shape([] as [i64; 0]), TtEmit::Reduce(TileDim::Scalar)),
            (Pat::reduce(tile, BOp::Max).shape([] as [i64; 0]), TtEmit::Reduce(TileDim::Scalar)),
        ];
        // Generic elementwise table: every UOp/BOp with a native kind.
        // This is data, not per-op lowering code. Silu needs no entry:
        // it is `Mul(x, sigmoid(x))`, so the Mul entry recurses into the
        // fused sigmoid entry, which allocates its own DST result.
        for &uop in UOPS {
            let Some(kind) = uop_kind(uop) else { continue };
            let p2 = Pat::unary(Pat::bind('x').shape([32i64, 32]), uop).shape([32i64, 32]);
            let p1 = Pat::unary(Pat::bind('x').shape([32i64]), uop).shape([32i64]);
            pats.push((Pat::Any(Box::new([p2, p1])), TtEmit::Sfpu { kind, arity: 1 }));
        }
        for &bop in BOPS {
            let Some(kind) = bop_kind(bop) else { continue };
            let p2 = Pat::binary(bop, Pat::bind('x').shape([32i64, 32]), Pat::bind('y').shape([32i64, 32])).shape([32i64, 32]);
            let p1 = Pat::binary(bop, Pat::bind('x').shape([32i64]), Pat::bind('y').shape([32i64])).shape([32i64]);
            pats.push((Pat::Any(Box::new([p2, p1])), TtEmit::Sfpu { kind, arity: 2 }));
        }
        // Phase 1 — reader publishes: every tile `Load` over a DRAM
        // param gets a `TTOp::Copy` publishing the tile to the param's
        // CB, base index stripped of lanes, explicit byte size. Loads
        // stay in place and dangle once compute is lowered (DCE
        // removes them with their index math). Tiled DRAM traffic
        // only — anything else bails the whole pass, since a
        // half-rewritten kernel is worse than none.
        let mut loads: Vec<(OpId, OpId, OpId)> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            if let Op::Load { src: base, src_idx: off, .. } = self.ops[id].op {
                if matches!(self.ops[base].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. }) {
                    if self.shape(id) != vec![32, 32] {
                        return;
                    }
                    loads.push((id, base, off));
                }
            }
            id = self.ops[id].next;
        }
        // CB per DRAM param, in first-load order; slots counted up front.
        // `reader_cb_of` survives for phase 2 (feeder → CB resolution).
        let mut reader_cb_of: Map<OpId, OpId> = Map::default();
        if !loads.is_empty() {
            for (_, _, off) in loads.iter() {
                let mut stack = vec![*off];
                for _ in 0..250_000 {
                    let Some(n) = stack.pop() else { break };
                    match self.ops[n].op {
                        Op::Const(..) | Op::Range { .. } | Op::Param { .. } | Op::Unroll { .. } => {}
                        Op::Binary { x, y, .. } => {
                            stack.push(x);
                            stack.push(y);
                        }
                        Op::Unary { x, .. } => stack.push(x),
                        Op::Cast { x, .. } => stack.push(x),
                        _ => return,
                    }
                }
            }
            let anchor = loads[0].0;
            // One CB per DRAM param, in first-load order; slots counted up front.
            let mut cbs: Vec<(OpId, DType, i64)> = Vec::new();
            for (_, base, _) in loads.iter() {
                if let Some(entry) = cbs.iter_mut().find(|(b, _, _)| *b == *base) {
                    entry.2 += 1;
                } else {
                    let Op::Param { dtype, .. } = self.ops[*base].op else { return };
                    cbs.push((*base, dtype, 1));
                }
            }
            let mut cb_of: Map<OpId, OpId> = Map::default();
            let mut slot_of: Map<OpId, i64> = Map::default();
            for (base, dtype, ntiles) in cbs.iter() {
                let cb = if prologue.is_null() {
                    self.push_back(Op::Storage { dtype: *dtype, scope: MemScope::Circular, len: ntiles * 1024 })
                } else {
                    self.insert_before(prologue, Op::Storage { dtype: *dtype, scope: MemScope::Circular, len: ntiles * 1024 })
                };
                cb_of.insert(*base, cb);
                reader_cb_of.insert(*base, cb);
                slot_of.insert(*base, 0);
            }
            for (_load, base, off) in loads.iter() {
                let slot = slot_of[base];
                slot_of.insert(*base, slot + 1);
                let cb = cb_of[base];
                let dtype = self.dtype(*base);
                // CB indices are elementwise (uniform with vector traffic):
                // tile k lives at element offset k*1024.
                let slot_c = self.insert_const_idx_before(anchor, slot * 1024);
                // Base address with lanes stripped; the transfer size is
                // explicit, so no lane math rides along.
                let Some(base_idx) = self.strip_unrolls(*off, anchor) else {
                    return;
                };
                let bytes = dtype.bit_size() as u32 / 8 * 1024;
                self.insert_before(anchor, Op::TT(TTOp::Copy { src: *base, src_idx: base_idx, dst: cb, dst_idx: slot_c, bytes }));
            }
            self.insert_before(anchor, Op::TT(TTOp::EndReader));
        }
        // Phase 2 — native lowering. Values are lowered recursively from
        // the DRAM output stores. Every native op writes a fresh logical
        // Register storage. A computed value with more than one use is
        // packed to a Circular CB (DST is single-use scratch; multi-use
        // values live in L1) and every consumer unpacks from that CB. The
        // first unpack is redundant with the producer's DST row on purpose:
        // a later pass can coalesce it, keeping lowering simple.
        let (_, rcs) = self.compute_dtypes_and_rcs();
        let mut lower = TtLowering {
            pats: &pats,
            prologue,
            reader_cb_of: &reader_cb_of,
            rcs: &rcs,
            materialized: Map::default(),
            pat_a,
            pat_b,
            matmul: None,
        };
        let mut roots: Vec<OpId> = Vec::new();
        let mut scan = self.head;
        for _ in 0..250_000 {
            if scan.is_null() {
                break;
            }
            if let Op::Store { dst, src, .. } = self.ops[scan].op {
                if matches!(self.ops[dst].op, Op::Param { kind: ParamKind::GlobalMut, .. }) {
                    roots.push(src);
                }
            }
            scan = self.next_op(scan);
        }
        for root in roots {
            let _ = self.tt_lower_value(root, &mut lower);
        }
        self.common_subexpression_elimination();
        // The high-level cones replaced by native ops are dead now. Drop
        // them before phase 3 so codegen sees only native compute.
        self.dead_code_elimination();

        // The prologue anchor was the first non-param/storage op when the
        // pass started; the DCE above can take exactly that op (a rewritten
        // index cone leaves nothing after the params). Phase 3 inserts CB
        // storages before it, so re-resolve it first.
        prologue = self.head;
        for _ in 0..250_000 {
            if prologue.is_null() {
                break;
            }
            match self.ops[prologue].op {
                Op::Param { .. } | Op::Storage { .. } => prologue = self.ops[prologue].next,
                _ => break,
            }
        }

        // Phase 3 — writer drains: every DRAM store of a lowered value
        // becomes a `TTOp::Copy` in a writer section closed by
        // `EndCompute`. A value already materialized in a CB is drained
        // directly; a DST result is packed to a writer CB first.
        let mut stores: Vec<(OpId, OpId, OpId, OpId)> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            match self.ops[id].op {
                Op::Load { src: base, .. } => {
                    if matches!(self.ops[base].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. })
                        && self.shape(id) != vec![32, 32]
                    {
                        return;
                    }
                }
                Op::Store { dst: base, dst_idx: off, src, .. } => {
                    if matches!(self.ops[base].op, Op::Storage { .. }) {
                        // Already lowered; nothing owned here.
                    } else if matches!(self.ops[base].op, Op::Param { kind: ParamKind::GlobalMut, .. }) {
                        // Tiled stores and row stores (a reduce row is 32
                        // elements); anything else bails the pass.
                        match self.shape(id).as_slice() {
                            [32, 32] | [32] | [1024] => stores.push((id, base, off, src)),
                            _ => return,
                        }
                    } else {
                        return;
                    }
                }
                _ => {}
            }
            id = self.ops[id].next;
        }
        if !stores.is_empty() {
            let first = stores[0].0;
            // Empty reader section for store-only kernels.
            let mut scan = self.head;
            let mut has_reader = false;
            for _ in 0..250_000 {
                if scan.is_null() {
                    break;
                }
                if matches!(self.ops[scan].op, Op::TT(TTOp::EndReader)) {
                    has_reader = true;
                    break;
                }
                scan = self.ops[scan].next;
            }
            if !has_reader {
                // A tiled DRAM load without a reader section means phase 1
                // bailed: don't paper over it.
                let mut scan = self.head;
                for _ in 0..250_000 {
                    if scan.is_null() {
                        break;
                    }
                    if let Op::Load { src: base, .. } = self.ops[scan].op {
                        if matches!(self.ops[base].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. })
                            && self.shape(scan) == vec![32, 32]
                        {
                            return;
                        }
                    }
                    scan = self.ops[scan].next;
                }
                self.insert_before(first, Op::TT(TTOp::EndReader));
            }
            // Compute-side rewrites in head order; drains append at the tail.
            self.push_back(Op::TT(TTOp::EndCompute));
            let slot_zero = self.push_back(Op::Const(Constant::idx(0i64)));
            for (store, base, off, src) in stores.iter() {
                let Some((dtype, storage)) = self.tt_lowered_storage(*src, &lower) else {
                    // Unlowered value: stays for later increments.
                    continue;
                };
                // Single-tile shapes only; anything else bails the pass.
                match self.shape(*store).as_slice() {
                    [32, 32] | [32] | [1024] => {}
                    _ => return,
                };
                if self.dtype(*store) != dtype {
                    return;
                }
                let bytes = match (dtype.bit_size() as u32 / 8).checked_mul(1024) {
                    Some(b) if b > 0 => b,
                    _ => return,
                };
                let tail = self.tail;
                let Some(base_idx) = self.strip_unrolls(*off, tail) else {
                    return;
                };
                if matches!(self.ops[storage].op, Op::Storage { scope: MemScope::Circular, .. }) {
                    // Multi-use value already in L1: drain it directly.
                    let zero = self.insert_const_idx_before(*store, 0i64);
                    self.push_back(Op::TT(TTOp::Copy { src: storage, src_idx: zero, dst: *base, dst_idx: base_idx, bytes }));
                } else {
                    // Output boundary: pack the DST row into a writer CB,
                    // then drain that CB to DRAM. The writer CB is FP32 (the
                    // packer writes FP32 out of `Dst`); when the param is
                    // 16-bit, the host boundary narrows on the way back
                    // (`TTDmMove::narrow`). Packing BF16 straight from `Dst`
                    // is the packer's F32->BF16 conversion, which the pinned
                    // ttsim refuses.
                    let pack_bytes = match (DType::F32.bit_size() as u32 / 8).checked_mul(1024) {
                        Some(b) if b > 0 => b,
                        _ => return,
                    };
                    let writer_cb = self.tt_new_cb(DType::F32, prologue);
                    let zero_pack = self.insert_const_idx_before(*store, 0i64);
                    // The lane tree goes where the pack reads it: right before
                    // the store (its base `zero_pack` is inserted there too),
                    // never before the producer — that would put index math in
                    // the compute section referring to a later const.
                    let pack_lanes = self.lane_tree_before(zero_pack, &[32, 32], *store);
                    // Pack from the resolved DST storage, not the value
                    // root: the reuse pass only sees Storage operands, and
                    // it must know this pack keeps the row live.
                    let pack = Op::Store { dst: writer_cb, dst_idx: pack_lanes, src: storage };
                    // A matmul accumulator hands its work over: T2 takes
                    // `ready` before reading it and posts `free` after, so T1
                    // can clear the rows for the next tile. The wait goes in
                    // first so it precedes the pack.
                    let handoff = lower.matmul.filter(|(slot, _, _)| *slot == storage);
                    if let Some((_, ready, free)) = handoff {
                        self.insert_before(*store, Op::TT(TTOp::Wait { sem: ready, before: TT_BEFORE_PACKER, role: 2 }));
                    }
                    let pack = self.insert_before(*store, pack);
                    if let Some((_, _, free)) = handoff {
                        self.insert_after(pack, Op::TT(TTOp::Post { sem: free, role: 2 }));
                    }
                    self.push_back(Op::TT(TTOp::Copy {
                        src: writer_cb,
                        src_idx: slot_zero,
                        dst: *base,
                        dst_idx: base_idx,
                        bytes: pack_bytes,
                    }));
                }
                self.remove_op(*store);
            }
        }
        self.verify();
    }

    /// Fresh logical DST storage, inserted in the prologue so it dominates
    /// every compute op. The reuse pass later assigns it a physical row.
    fn tt_new_dst(&mut self, dtype: DType, prologue: OpId) -> OpId {
        let op = Op::Storage { dtype, scope: MemScope::Register, len: 1024 };
        if prologue.is_null() {
            self.push_back(op)
        } else {
            self.insert_before(prologue, op)
        }
    }

    /// Fresh Circular CB, inserted in the prologue. Used for multi-use
    /// values and for writer packs.
    fn tt_new_cb(&mut self, dtype: DType, prologue: OpId) -> OpId {
        let op = Op::Storage { dtype, scope: MemScope::Circular, len: 1024 };
        if prologue.is_null() {
            self.push_back(op)
        } else {
            self.insert_before(prologue, op)
        }
    }

    /// Resolve a value to its logical storage. A computed value with one
    /// use is its DST row; a multi-use value is its CB; a DRAM load is its
    /// reader CB. The caller decides whether to unpack.
    fn tt_lower_value(&mut self, id: OpId, lower: &mut TtLowering) -> Option<(DType, OpId)> {
        if let Some(&cb) = lower.materialized.get(&id) {
            return Some((self.dtype(id), cb));
        }
        match self.ops[id].op {
            // A lowered accumulator (`Setrwc`) carries its tile in `dst`:
            // the matmul's value is the DST storage the Matrix Unit filled.
            Op::TT(TTOp::Sfpu { slot, .. } | TTOp::ReduceTile { slot, .. }) => {
                return Some((self.dtype(id), slot));
            }
            Op::TT(TTOp::Setrwc { dst }) => return Some((self.dtype(id), dst)),
            Op::Storage { dtype, .. } => return Some((dtype, id)),
            Op::Load { .. } => return self.tt_lower_load(id, lower),
            _ => {}
        }
        for i in 0..lower.pats.len() {
            let Some(m) = self.match_pat(id, &lower.pats[i].0) else {
                continue;
            };
            let emit = lower.pats[i].1;
            if let Some(v) = self.tt_emit(id, emit, &m, lower) {
                return Some(v);
            }
        }
        None
    }

    /// A DRAM load is already published to its reader CB by phase 1; a
    /// load from another storage is that storage. No DST traffic yet.
    fn tt_lower_load(&self, id: OpId, lower: &TtLowering) -> Option<(DType, OpId)> {
        let Op::Load { src: base, .. } = self.ops[id].op else {
            return None;
        };
        match self.ops[base].op {
            Op::Param { dtype, kind: ParamKind::Global | ParamKind::GlobalMut, .. } => {
                let cb = lower.reader_cb_of.get(&base).copied()?;
                Some((dtype, cb))
            }
            Op::Storage { dtype, .. } => Some((dtype, base)),
            _ => None,
        }
    }

    /// Resolve a value to a DST row, unpacking a CB if needed. This is the
    /// only CB -> DST conversion in compute; the copy is single-use and the
    /// reuse pass can coalesce its row with the producer.
    fn tt_resolve_dst(&mut self, id: OpId, consumer: OpId, lower: &mut TtLowering) -> Option<(DType, OpId)> {
        let (dtype, storage) = self.tt_lower_value(id, lower)?;
        match self.ops[storage].op {
            Op::Storage { scope: MemScope::Register, .. } => Some((dtype, storage)),
            Op::Storage { scope: MemScope::Circular, .. } => {
                let dst = self.tt_new_dst(dtype, lower.prologue);
                let zero = self.insert_const_idx_before(consumer, 0i64);
                let bytes = (dtype.bit_size() as u32 / 8).checked_mul(1024)?;
                self.insert_before(consumer, Op::TT(TTOp::Copy { src: storage, src_idx: zero, dst, dst_idx: zero, bytes }));
                Some((dtype, dst))
            }
            _ => None,
        }
    }

    /// A computed value with more than one use must live in L1: DST is
    /// single-use scratch. Pack it after the producer and return the CB;
    /// every consumer unpacks. The extra first unpack is deliberate.
    fn tt_materialize(&mut self, id: OpId, dtype: DType, slot: OpId, lower: &mut TtLowering) -> Option<(DType, OpId)> {
        let uses = lower.rcs.get(&id).copied().unwrap_or(0);
        if uses <= 1 {
            return Some((dtype, slot));
        }
        let cb = self.tt_new_cb(dtype, lower.prologue);
        let zero = self.insert_const_idx_before(id, 0i64);
        let shape = self.shape(id);
        let block: Vec<Dim> = match shape.as_slice() {
            [32, 32] => vec![32, 32],
            // A scalar reduce result lives in lane 0 of its row; pack the
            // row so a second consumer does not alias the producer's DST.
            [32] | [] => vec![32],
            // Odd shapes stay in DST; a later increment can widen the
            // materialization once the codegen knows the lane layout.
            _ => return Some((dtype, slot)),
        };
        let pack_lanes = self.lane_tree_before(zero, &block, id);
        let pack = Op::Store { dst: cb, dst_idx: pack_lanes, src: slot };
        self.insert_after(id, pack);
        lower.materialized.insert(id, cb);
        Some((dtype, cb))
    }

    /// Emit the first matching pattern at `id`. The pattern table is data;
    /// this only knows the three emitter shapes (SFPU, Matmul, Reduce).
    fn tt_emit(&mut self, id: OpId, emit: TtEmit, m: &Bindings, lower: &mut TtLowering) -> Option<(DType, OpId)> {
        match emit {
            TtEmit::Sfpu { kind, arity } => {
                let (dtype, a) = self.tt_resolve_dst(m.op('x'), id, lower)?;
                let b = if arity == 2 {
                    let (_, b) = self.tt_resolve_dst(m.op('y'), id, lower)?;
                    b
                } else {
                    OpId::NULL
                };
                let slot = self.tt_new_dst(dtype, lower.prologue);
                self.ops[id].op = Op::TT(TTOp::Sfpu { kind, a, b, slot });
                self.tt_materialize(id, dtype, slot, lower)
            }
            TtEmit::Reduce(dim) => {
                let Op::Reduce { rop, .. } = self.ops[id].op else {
                    return None;
                };
                let (dtype, input) = self.tt_resolve_dst(m.op('t'), id, lower)?;
                let slot = self.tt_new_dst(dtype, lower.prologue);
                self.ops[id].op = Op::TT(TTOp::ReduceTile { rop, kind: dim, input, slot });
                self.tt_materialize(id, dtype, slot, lower)
            }
            TtEmit::Matmul => {
                let root = id;
                let (a, b) = (m.op('a'), m.op('b'));
                let (ta, tb) = match (&self.ops[a].op, &self.ops[b].op) {
                    (Op::Load { src: pa, src_idx: la }, Op::Load { src: pb, src_idx: lb }) => {
                        match (&self.ops[*pa].op, &self.ops[*pb].op) {
                            (Op::Param { dtype: da, .. }, Op::Param { dtype: db, .. }) => ((*pa, *la, *da), (*pb, *lb, *db)),
                            _ => return None,
                        }
                    }
                    _ => return None,
                };
                // K-plain feeder is A, K-scaled is B; roles are the order
                // binding the same K lane node. Params ride the roles, so
                // the CB lookup follows A/B, not operand order.
                let mut fab = None;
                for ((pa, la, da), (pb, lb, db)) in [(ta, tb), (tb, ta)] {
                    let kx = self.match_pat(la, &lower.pat_a).map(|m| m.op('k'));
                    let ky = self.match_pat(lb, &lower.pat_b).map(|m| m.op('k'));
                    if let (Some(x), Some(y)) = (kx, ky) {
                        if x == y && matches!(self.ops[x].op, Op::Unroll { .. }) {
                            fab = Some(((pa, da), (pb, db), x));
                            break;
                        }
                    }
                }
                let Some(((pa, dtype_a), (pb, dtype_b), k)) = fab else {
                    return None;
                };
                // The reduced axis must be the shared K lane: contracting
                // any other axis is not a matmul.
                if !matches!(&self.ops[root].op, Op::Reduce { reduce_axis, .. } if *reduce_axis == k) {
                    return None;
                }
                let (Some(&cb_a), Some(&cb_b)) = (lower.reader_cb_of.get(&pa), lower.reader_cb_of.get(&pb)) else {
                    return None;
                };
                // The native matmul is the UNPACR/MVMUL sequence, not a
                // structured op: `SrcA`/`SrcB` are two small banks the unpack
                // and math roles hand back and forth (`flip` on the unpack,
                // `flip_a`/`flip_b` on the `MVMUL` that releases one). One
                // output tile is 64 DST rows: `fi` 0 accumulates rows 0..31,
                // `MATH_AM_FI` moves the base on 32 rows for `fi` 1, so the
                // accumulator is the tile's slot directly.
                use tt_isa::backend::ThreadConfigEntry;
                use tt_isa::cfg::generated::thread;
                use crate::backend::tenstorrent::kern::matmul::{MATH_AM_FI, MATH_AM_GROUP, MATH_AM_HALF, MATH_AM_PHASE};
                // `math_prelude`: one `SETC16` per single-field entry, and a
                // whole-word `Setcfg` for SEC2..SEC4, which carry several
                // fields of one word (`math_addr_mods`).
                // Every `ThreadConfig` word goes out *composed*: a `SETC16`
                // carries the whole word, so a field's value has to be shifted
                // into its place here (`FidelityIncr = 1` is not the word `1`,
                // and the raw `1` lands on `DestIncr`, which then walks the
                // accumulator one row per multiply).
                let one = |f: tt_isa::cfg::ThreadConfigField, v: u16| -> TTOp {
                    let e = ThreadConfigEntry::zeroed(f.addr32()).set(f, v).expect("matmul thread entry is encodable");
                    TTOp::Setc16 {
                        field: u32::from(e.addr32()),
                        value: e.value(),
                    }
                };
                let word = |fields: &[(tt_isa::cfg::ThreadConfigField, u16)]| -> TTOp {
                    let mut e = ThreadConfigEntry::zeroed(fields[0].0.addr32());
                    for &(f, v) in fields {
                        e = e.set(f, v).expect("matmul address modifier is encodable");
                    }
                    // These are `ThreadConfig` words, so they go out as
                    // `SETC16` with the whole word — NOT as a `Config` write:
                    // the same index in the `Config` space is a packer/unpacker
                    // register (writing address modifiers there is how the
                    // packer got its `PCK0_ADDR_CTRL_XY_REG_1` fields poked).
                    TTOp::Setc16 {
                        field: u32::from(e.addr32()),
                        value: e.value(),
                    }
                };
                let mut before = vec![
                    one(thread::ADDR_MOD_AB_SEC0_SrcAIncr, 0),
                    one(thread::ADDR_MOD_DST_SEC0_DestIncr, 0),
                    one(thread::ADDR_MOD_AB_SEC1_SrcAIncr, 0),
                    one(thread::ADDR_MOD_DST_SEC1_FidelityIncr, 1),
                    // 2: end of a first half: `SrcB` on eight rows, phase 0.
                    word(&[
                        (thread::ADDR_MOD_AB_SEC2_SrcBIncr, 8),
                        (thread::ADDR_MOD_AB_SEC2_SrcBClear, 0),
                    ]),
                    word(&[
                        (thread::ADDR_MOD_DST_SEC2_DestIncr, 0),
                        (thread::ADDR_MOD_DST_SEC2_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC2_FidelityClear, 1),
                    ]),
                    // 3: end of a group (its last `MVMUL` is not the `fi`'s):
                    // phase and `SrcB` back to zero.
                    word(&[
                        (thread::ADDR_MOD_AB_SEC3_SrcBIncr, 0),
                        (thread::ADDR_MOD_AB_SEC3_SrcBClear, 1),
                    ]),
                    word(&[
                        (thread::ADDR_MOD_DST_SEC3_DestIncr, 0),
                        (thread::ADDR_MOD_DST_SEC3_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC3_FidelityClear, 1),
                    ]),
                    // 4: end of an `fi`: as 3, and the base on 32 rows.
                    word(&[
                        (thread::ADDR_MOD_AB_SEC4_SrcBIncr, 0),
                        (thread::ADDR_MOD_AB_SEC4_SrcBClear, 1),
                    ]),
                    word(&[
                        (thread::ADDR_MOD_DST_SEC4_DestIncr, 32),
                        (thread::ADDR_MOD_DST_SEC4_DestCR, 0),
                        (thread::ADDR_MOD_DST_SEC4_FidelityClear, 1),
                    ]),
                    one(thread::FIDELITY_BASE_Phase, 0),
                ];
                let mut after = vec![
                    // The role hand-off. `ready` starts empty (T1 posts it) and
                    // `free` starts full (T2 hands it back), so the first take
                    // of each has the value its protocol expects.
                    TTOp::Semset { sem: TT_SEM_READY, value: 0 },
                    TTOp::Semset { sem: TT_SEM_FREE, value: 1 },
                    // T1 does not clear the accumulator until T2 has finished
                    // reading the previous tile out of it.
                    TTOp::Wait { sem: TT_SEM_FREE, before: TT_BEFORE_MATRIX, role: 1 },
                    TTOp::Zeroacc { mode: 3, imm10: 0 },
                ];
                // One `MVMUL` group into rows `base..base + 16` of the `fi`:
                // both `SrcB` halves, every phase, all but the group's last
                // multiply, which the caller issues with its flips and end.
                const PHASES: u32 = 4;
                let group = |after: &mut Vec<TTOp>, base: u32| {
                    for half in 0..2u32 {
                        let phases = if half == 0 { PHASES } else { PHASES - 1 };
                        for phase in 0..phases {
                            let am = if half == 0 && phase + 1 == PHASES {
                                MATH_AM_HALF
                            } else {
                                MATH_AM_PHASE
                            };
                            after.push(TTOp::Mvmul {
                                dst_row: base + 8 * half,
                                addr_mod: am as u8,
                                flip_a: false,
                                flip_b: false,
                                broadcast_b: false,
                            });
                        }
                    }
                };
                // `matmul_items`: `fi`, then `k`, then `fj`. `A` is in0 and
                // goes to `SrcB` (face `(fi, k)`); `B` is in1 and goes to
                // `SrcA` (faces `(k, 0)` and `(k, 1)`). Each face grid step
                // unpacks a whole operand face, so every unpack flips.
                for fi in 0..2u32 {
                    for k in 0..2u32 {
                        after.push(TTOp::Setadczw { unit: 1, face: 2 * fi + k });
                        after.push(TTOp::Unpacr { src: cb_a, unit: 1, count: 1024, flip: true });
                        after.push(TTOp::Setadczw { unit: 0, face: 2 * k });
                        after.push(TTOp::Unpacr { src: cb_b, unit: 0, count: 1024, flip: true });
                        group(&mut after, 0);
                        // The group's last multiply hands `SrcA` back: the
                        // `B` face it was read from is done.
                        after.push(TTOp::Mvmul {
                            dst_row: 8,
                            addr_mod: MATH_AM_GROUP as u8,
                            flip_a: true,
                            flip_b: false,
                            broadcast_b: false,
                        });
                        after.push(TTOp::Setadczw { unit: 0, face: 2 * k + 1 });
                        after.push(TTOp::Unpacr { src: cb_b, unit: 0, count: 1024, flip: true });
                        group(&mut after, 16);
                        let end = if k == 0 { MATH_AM_GROUP } else { MATH_AM_FI };
                        after.push(TTOp::Mvmul {
                            dst_row: 24,
                            addr_mod: end as u8,
                            flip_a: true,
                            flip_b: true,
                            broadcast_b: false,
                        });
                        if fi == 1 && k == 1 {
                            // The tile is complete: T2 may pack it (post_after
                            // drains the Matrix Unit first).
                            after.push(TTOp::Post { sem: TT_SEM_READY, role: 1 });
                        }
                    }
                }
                let slot = self.tt_new_dst(dtype_a, lower.prologue);
                // Remember the accumulator so phase 3 can wrap its pack in the
                // `ready` take and the `free` post.
                lower.matmul = Some((slot, TT_SEM_READY, TT_SEM_FREE));
                self.ops[root].op = Op::TT(TTOp::Setrwc { dst: slot });
                for op in before.drain(..) {
                    self.insert_before(root, Op::TT(op));
                }
                let mut cur = root;
                for op in after {
                    cur = self.insert_after(cur, Op::TT(op));
                }
                self.tt_materialize(root, dtype_a, slot, lower)
            }
        }
    }

    /// Storage of an already-lowered value: its materialized CB, its DST
    /// row, a storage itself, or the reader CB of a load.
    fn tt_lowered_storage(&self, id: OpId, lower: &TtLowering) -> Option<(DType, OpId)> {
        if let Some(&cb) = lower.materialized.get(&id) {
            return Some((self.dtype(id), cb));
        }
        match self.ops[id].op {
            Op::TT(TTOp::Sfpu { slot, .. } | TTOp::ReduceTile { slot, .. }) => {
                Some((self.dtype(id), slot))
            }
            Op::TT(TTOp::Setrwc { dst }) => Some((self.dtype(id), dst)),
            Op::TT(TTOp::Copy { dst, .. }) => match self.ops[dst].op {
                Op::Storage { dtype, scope: MemScope::Register, .. } => Some((dtype, dst)),
                _ => None,
            },
            Op::Storage { dtype, .. } => Some((dtype, id)),
            Op::Load { .. } => self.tt_lower_load(id, lower),
            _ => None,
        }
    }

    /// Physical DST reuse. Every logical Register storage created by
    /// lowering is assigned a physical row by linear-scan liveness over all
    /// operands (compute, packs, unpacks). Dead inputs are released before
    /// the output is assigned, so an in-place native op can coalesce its
    /// input and output onto one row. This is the only place physical rows
    /// are chosen; lowering stays SSA.
    pub fn tt_dst_reuse(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        let mut prologue = self.head;
        for _ in 0..250_000 {
            if prologue.is_null() {
                break;
            }
            match self.ops[prologue].op {
                Op::Param { .. } | Op::Storage { .. } => prologue = self.ops[prologue].next,
                _ => break,
            }
        }
        let mut ops: Vec<OpId> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            ops.push(id);
            id = self.next_op(id);
        }
        // Last use of every logical Register storage over all operands.
        let mut last_use: Map<OpId, usize> = Map::default();
        for (i, &op) in ops.iter().enumerate() {
            for p in self.ops[op].op.parameters() {
                if matches!(self.ops[p].op, Op::Storage { scope: MemScope::Register, .. }) {
                    last_use.insert(p, i);
                }
            }
        }
        let mut physical: Map<OpId, OpId> = Map::default();
        let mut active: Vec<(OpId, OpId, usize)> = Vec::new();
        let mut free: Vec<(DType, OpId)> = Vec::new();
        for (i, &op) in ops.iter().enumerate() {
            // Release rows whose last use was strictly before this op.
            let mut k = 0;
            while k < active.len() {
                if active[k].2 < i {
                    let (_, phys, _) = active.remove(k);
                    if let Op::Storage { dtype, .. } = self.ops[phys].op {
                        free.push((dtype, phys));
                    }
                } else {
                    k += 1;
                }
            }
            let regs: Vec<OpId> = self.ops[op]
                .op
                .parameters()
                .filter(|p| matches!(self.ops[*p].op, Op::Storage { scope: MemScope::Register, .. }))
                .collect();
            if regs.is_empty() {
                continue;
            }
            let output = self.tt_register_output(&self.ops[op].op);
            for &p in &regs {
                if Some(p) == output {
                    continue;
                }
                self.tt_ensure_phys(&mut physical, &mut active, &mut free, &last_use, p, prologue);
            }
            if let Some(out) = output {
                // Release inputs whose last use is this op before assigning
                // the output, so a dead input's row can be reused in place.
                let mut k = 0;
                while k < active.len() {
                    if active[k].2 == i && active[k].0 != out {
                        let (_, phys, _) = active.remove(k);
                        if let Op::Storage { dtype, .. } = self.ops[phys].op {
                            free.push((dtype, phys));
                        }
                    } else {
                        k += 1;
                    }
                }
                self.tt_ensure_phys(&mut physical, &mut active, &mut free, &last_use, out, prologue);
            }
        }
        // Rewrite every logical operand to its physical row.
        let mut rewrites: Vec<(OpId, usize, OpId)> = Vec::new();
        for &op in &ops {
            for (idx, p) in self.ops[op].op.parameters().enumerate() {
                if let Some(&phys) = physical.get(&p) {
                    if phys != p {
                        rewrites.push((op, idx, phys));
                    }
                }
            }
        }
        for (op, idx, phys) in rewrites {
            if let Some(slot) = self.ops[op].op.parameters_mut().nth(idx) {
                *slot = phys;
            }
        }
    }

    /// The Register operand an op writes, if any. Inputs precede the
    /// output in `parameters`, so the reuse pass can release dead inputs
    /// before allocating this row.
    fn tt_register_output(&self, op: &Op) -> Option<OpId> {
        match op {
            Op::TT(TTOp::Sfpu { slot, .. } | TTOp::ReduceTile { slot, .. }) => Some(*slot),
            Op::TT(TTOp::Copy { dst, .. }) => match self.ops[*dst].op {
                Op::Storage { scope: MemScope::Register, .. } => Some(*dst),
                _ => None,
            },
            _ => None,
        }
    }

    /// Ensure `logical` has a physical row, reusing a free row of the same
    /// dtype when possible.
    fn tt_ensure_phys(
        &mut self,
        physical: &mut Map<OpId, OpId>,
        active: &mut Vec<(OpId, OpId, usize)>,
        free: &mut Vec<(DType, OpId)>,
        last_use: &Map<OpId, usize>,
        logical: OpId,
        prologue: OpId,
    ) {
        if physical.contains_key(&logical) {
            return;
        }
        let Op::Storage { dtype, .. } = self.ops[logical].op else {
            return;
        };
        let phys = if let Some(i) = free.iter().position(|(d, _)| *d == dtype) {
            free.remove(i).1
        } else {
            self.tt_new_dst(dtype, prologue)
        };
        physical.insert(logical, phys);
        active.push((logical, phys, last_use.get(&logical).copied().unwrap_or(0)));
    }

    /// Base index with lanes stripped: clone the index expression
    /// with every `Unroll` replaced by 0. No affinity assumption —
    /// substitution works on any expression shape; uncloneable leaves
    /// (traffic, control) bail the caller. Shared leaves are reused
    /// (they dominate `before` exactly like the cloned index math
    /// does); the rest is rebuilt before `before`.
    ///
    /// Group ids rebase to physical tiles: DRAM traffic addresses
    /// tile-layout buffers, so a const-scaled group id takes the
    /// physical tile stride (one full [32, 32] tile per step, times
    /// inner-group extents) and a bare group id inside Add/Sub wraps
    /// to id * stride. Symbolic extents bail the caller.
    fn strip_unrolls(&mut self, idx: OpId, before: OpId) -> Option<OpId> {
        // Clone memo per call (index math is small).
        let mut memo: Map<OpId, OpId> = Map::default();
        // Group axis -> physical tile stride (element units), built
        // lazily on first group use inside the walk below.
        let mut strides: Option<Map<u32, Dim>> = None;
        let zero = self.insert_before(before, Op::Const(Constant::idx(0i64)));
        memo.insert(zero, zero);
        let mut stack = vec![(idx, false)];
        for _ in 0..250_000 {
            let Some((n, expanded)) = stack.pop() else { break };
            if memo.contains_key(&n) {
                continue;
            }
            match self.ops[n].op {
                Op::Const(_) | Op::Range { .. } | Op::Param { .. } => {
                    memo.insert(n, n);
                }
                Op::Unroll { .. } => {
                    memo.insert(n, zero);
                }
                Op::Binary { x, y, bop } => {
                    if !expanded {
                        stack.push((n, true));
                        stack.push((x, false));
                        stack.push((y, false));
                    } else {
                        let nx = memo[&x];
                        let ny = memo[&y];
                        // Group-leaf axes of the memo images (owned
                        // copies; every borrow ends before inserts).
                        let gx: Option<u32> = match &self.ops[nx].op {
                            Op::Range { axis, kind: RangeKind::Group(_) } => Some(*axis),
                            _ => None,
                        };
                        let gy: Option<u32> = match &self.ops[ny].op {
                            Op::Range { axis, kind: RangeKind::Group(_) } => Some(*axis),
                            _ => None,
                        };
                        // Axis -> physical tile stride, built on first
                        // group use. One group step is one full
                        // [32, 32] tile (1024 elems, the unit phase 1
                        // sizes CBs, slots and byte counts in) times
                        // the extents of all inner (higher-axis) groups;
                        // split keeps the outer on the original axis.
                        // Unresolvable (symbolic) extents bail the pass:
                        // never silent wrong code.
                        if (gx.is_some() || gy.is_some()) && strides.is_none() {
                            let mut axes: Vec<(u32, Dim)> = Vec::new();
                            for (_, op) in self.iter_unordered() {
                                if let Op::Range { axis, kind: RangeKind::Group(len) } = op {
                                    let Some(e) = self.resolve_const(*len).and_then(Constant::as_dim) else {
                                        return None;
                                    };
                                    axes.push((*axis, e));
                                }
                            }
                            axes.sort();
                            let mut map: Map<u32, Dim> = Map::default();
                            let mut suffix: Dim = 1024;
                            for (a, e) in axes.iter().rev() {
                                map.insert(*a, suffix);
                                suffix = match suffix.checked_mul(*e) {
                                    Some(s) => s,
                                    None => return None,
                                };
                            }
                            strides = Some(map);
                        }
                        // Physical group bases: a const-scaled group id
                        // takes the tile stride; a bare group id inside
                        // Add/Sub wraps to id * stride. These are the
                        // forms split/pad composition emits; anything
                        // else keeps the faithful clone and trips the
                        // downstream checks loudly.
                        let mut ax = nx;
                        let mut ay = ny;
                        match bop {
                            BOp::Mul => {
                                if let Some(a) = gx {
                                    if gy.is_none() && matches!(self.ops[ny].op, Op::Const(_)) {
                                        let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                        let sc = self.insert_before(before, Op::Const(Constant::idx(s)));
                                        ax = self.insert_before(before, Op::Binary { x: nx, y: sc, bop });
                                    }
                                }
                                if let Some(a) = gy {
                                    if gx.is_none() && matches!(self.ops[nx].op, Op::Const(_)) {
                                        let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                        let sc = self.insert_before(before, Op::Const(Constant::idx(s)));
                                        ay = self.insert_before(before, Op::Binary { x: sc, y: ny, bop });
                                    }
                                }
                            }
                            BOp::Add | BOp::Sub => {
                                if let Some(a) = gx {
                                    let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                    let sc = self.insert_before(before, Op::Const(Constant::idx(s)));
                                    ax = self.insert_before(before, Op::Binary { x: nx, y: sc, bop: BOp::Mul });
                                }
                                if let Some(a) = gy {
                                    let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                    let sc = self.insert_before(before, Op::Const(Constant::idx(s)));
                                    ay = self.insert_before(before, Op::Binary { x: ny, y: sc, bop: BOp::Mul });
                                }
                            }
                            _ => {}
                        }
                        // Fold constant lanes away: the base is fully
                        // static exactly when the index is.
                        let id = match (&self.ops[ax].op, &self.ops[ay].op) {
                            (Op::Const(cx), Op::Const(cy)) => {
                                self.insert_before(before, Op::Const(Constant::binary(*cx, *cy, bop)))
                            }
                            _ => self.insert_before(before, Op::Binary { x: ax, y: ay, bop }),
                        };
                        memo.insert(n, id);
                    }
                }
                Op::Unary { x, uop } => {
                    if !expanded {
                        stack.push((n, true));
                        stack.push((x, false));
                    } else {
                        let nx = memo[&x];
                        let id = match &self.ops[nx].op {
                            Op::Const(cx) => self.insert_before(before, Op::Const(cx.unary(uop))),
                            _ => self.insert_before(before, Op::Unary { x: nx, uop }),
                        };
                        memo.insert(n, id);
                    }
                }
                Op::Cast { x, dtype } => {
                    if !expanded {
                        stack.push((n, true));
                        stack.push((x, false));
                    } else {
                        let nx = memo[&x];
                        let id = self.insert_before(before, Op::Cast { x: nx, dtype });
                        memo.insert(n, id);
                    }
                }
                _ => return None,
            }
        }
        memo.get(&idx).copied()
    }

    /// TT stream verifier: the kernel is one reader section, one compute
    /// section, and one writer section, in that order, each closed by its
    /// marker. Compute programs live only in the compute section; DRAM
    /// copies never do. `Op::Barrier` is a host barrier, never a TT
    /// instruction.
    pub fn tt_verify(&self) {
        let mut seen_reader = false;
        let mut seen_compute = false;
        let mut section = 0u8;
        let mut written: Set<OpId> = Set::default();
        let is_dram = |op: &Op| matches!(op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. });
        let is_written = |op: &Op| matches!(op, Op::Param { kind: ParamKind::GlobalMut, .. });
        let mut id = self.head;
        while !id.is_null() {
            match self.at(id) {
                Op::TT(TTOp::EndReader) => {
                    assert!(section == 0, "tt_verify: duplicate EndReader at {id:?}");
                    assert!(!seen_reader, "tt_verify: duplicate EndReader at {id:?}");
                    seen_reader = true;
                    section = 1;
                }
                Op::TT(TTOp::EndCompute) => {
                    assert!(seen_reader, "tt_verify: EndCompute without EndReader at {id:?}");
                    assert!(!seen_compute, "tt_verify: duplicate EndCompute at {id:?}");
                    assert!(section == 1, "tt_verify: EndCompute outside the compute section at {id:?}");
                    seen_compute = true;
                    section = 2;
                }
                Op::TT(TTOp::Sfpu { .. })
                | Op::TT(TTOp::ReduceTile { .. })
                | Op::TT(TTOp::Unpacr { .. })
                | Op::TT(TTOp::Setrwc { .. })
                | Op::TT(TTOp::Setadczw { .. })
                | Op::TT(TTOp::Setadcx { .. })
                | Op::TT(TTOp::Zeroacc { .. })
                | Op::TT(TTOp::Mvmul { .. })
                | Op::TT(TTOp::Setc16 { .. })
                | Op::TT(TTOp::Setcfg { .. })
                | Op::TT(TTOp::Post { .. })
                | Op::TT(TTOp::Wait { .. })
                | Op::TT(TTOp::Semset { .. }) => {
                    assert!(section == 1, "tt_verify: compute op outside the compute section at {id:?}");
                }
                Op::TT(TTOp::Copy { src, dst, .. }) => {
                    let s = self.tt_storage_of(*src);
                    let d = self.tt_storage_of(*dst);
                    if section == 1 {
                        assert!(!is_dram(self.at(s)), "tt_verify: DRAM read in compute at {id:?}");
                        assert!(!is_dram(self.at(d)), "tt_verify: DRAM write in compute at {id:?}");
                    }
                    if is_written(self.at(d)) {
                        written.insert(d);
                    }
                    assert!(
                        !(is_written(self.at(s)) && written.contains(&s)),
                        "tt_verify: store->load of the same DRAM param at {id:?}"
                    );
                }
                Op::Store { dst, .. } => {
                    let d = self.tt_storage_of(*dst);
                    if section == 1 {
                        assert!(!is_dram(self.at(d)), "tt_verify: DRAM write in compute at {id:?}");
                    }
                    if is_written(self.at(d)) {
                        written.insert(d);
                    }
                }
                Op::Load { src, .. } => {
                    let s = self.tt_storage_of(*src);
                    if section == 1 {
                        assert!(!is_dram(self.at(s)), "tt_verify: DRAM read in compute at {id:?}");
                    }
                    assert!(
                        !(is_written(self.at(s)) && written.contains(&s)),
                        "tt_verify: store->load of the same DRAM param at {id:?}"
                    );
                }
                Op::Barrier => panic!("tt_verify: Op::Barrier in a TT kernel at {id:?}"),
                _ => {}
            }
            id = self.next_op(id);
        }
        assert!(seen_reader, "tt_verify: missing EndReader");
        assert!(seen_compute, "tt_verify: missing EndCompute");
    }

    /// Chase a `Load`/`Store`/`Copy` operand through its index expression
    /// to the underlying `Storage`/`Param`. Tile-compute results name DST
    /// slots, not buffers. The index is not needed (render uses slot 0).
    pub(crate) fn tt_storage_of(&self, op_id: OpId) -> OpId {
        match self.at(op_id) {
            Op::Load { src, .. } | Op::Copy { src, .. } => self.tt_storage_of(*src),
            Op::Store { dst, .. } => self.tt_storage_of(*dst),
            Op::Storage { .. } | Op::Param { .. } => op_id,
            Op::TT(TTOp::ReduceTile { slot, .. }) | Op::TT(TTOp::Sfpu { slot, .. }) => *slot,
            other => panic!("tt_storage_of: {op_id:?} is not a Load/Store/Copy, got {other:?}"),
        }
    }
}
