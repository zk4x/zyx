// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#![allow(unused)]

//! Rangeify movement operations.
//!
//! Reimplements unfold_movement_ops using tinygrad's rangeify approach,
//! without the View abstraction for movement op propagation.
//! Movement ops are applied directly to axis indices, and
//! LoadView/StoreView/ConstView are converted to Load/Store/Const in a single pass.
//!
//! # Kernel structure before and after linearize
//!
//! Before [`linearize`](Kernel::linearize), kernels contain only high-level ops:
//! `Define`, `Move` (reshape/expand/permute/pad/flip), `Reduce`, `Binary`, and
//! `Store`. Notably, they contain **no `Load`s**. All inputs to a kernel are
//! `Define` ops with `MemScope::Global` scope, and every global define is either:
//!
//! - **read-only** (`ro: true`) — an input, later turned into a `Load`
//! - **not read-only** (`ro: false`) — a `Store` destination (an output)
//!
//! Because there are no `Load`s before linearize, the kernel does not yet have a
//! `loads` list.
//!
//! Linearize performs the bulk of the "unfolding":
//!
//! - it **removes `Move` and `Reduce`**, expanding them into index arithmetic,
//! - it **inserts `Load`s**, `Loop`s, and the indexing computation (`Index`,
//!   `Mad`, `Binary` on loop/group indices),
//! - it computes each `Store`'s index from the shape it writes,
//! - read-only global defines become `Load { src, .. }` referencing a freshly
//!   inserted source `Define`,
//! - writable global defines stay in place as `Store` destinations.
//!
//! `Storage` is a post-linearization operation. It must not be expected in
//! pre-linearization movement kernels or used as a movement-chain marker.
//!
//! Only after linearize does the kernel have `Load` ops, so the `loads` list only
//! becomes meaningful then. This matters for any pass that maps kernel args to
//! buffers: pre-linearize, map from the global `Define` ops (in op order), not
//! from a `loads` list.

/// A single symbolic dimension of a value's index view: the loop/group index
/// (`idx`), left/right pad (`lp`/`rp`) and axis length (`len`). All are
/// `OpId`s resolved lazily. Strides are recomputed by each consumer from `len`.
#[derive(Clone, Copy)]
pub(crate) struct SDim {
    pub(crate) idx: OpId,
    pub(crate) lp: OpId,
    pub(crate) rp: OpId,
    pub(crate) len: OpId,
}

impl SDim {
    pub(crate) fn new(idx: OpId, lp: OpId, rp: OpId, len: OpId) -> Self {
        Self { idx, lp, rp, len }
    }
}

use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, IdxKind, Kernel, MemLayout, MemScope, MoveOp, Op, OpId, ParamKind},
    shape::{self, Dim, UAxis},
    slab::SlabId,
};

/// Extract the value of an index constant op.
fn pad_value(k: &Kernel, id: OpId) -> u64 {
    let Op::Const(c) = k.ops[id].op else {
        unreachable!("pad constant expected, got {:?}", k.ops[id].op)
    };
    c.as_dim().expect("pad constant must be a non-negative dim")
}

impl Kernel {
    /// Unfold movement operations into index-based operations using tinygrad's rangeify approach.
    ///
    /// Movement ops (Reshape, Expand, Permute, Pad) are applied directly to axis indices,
    /// and LoadView/StoreView/ConstView are converted to Load/Store/Const in a single pass.
    // TODO Currently it only works if each define has a single move op chain.
    // Make it also work with move op chains when each define is accessed by multiple move ops.
    pub fn linearize(&mut self) {
        if !self.ops.values().any(|n| matches!(n.op, Op::Store { index: OpId::NULL, .. })) {
            return;
        }

        #[cfg(debug_assertions)]
        {
            let has_gidx = self.ops.values().any(|n| matches!(n.op, Op::Index { kind: IdxKind::Group(_), .. }));
            let has_moves = self.ops.values().any(|n| matches!(n.op, Op::Move { .. }));
            if has_gidx && has_moves {
                panic!("unfold_movement_ops: cannot have both explicit gidx and LoadView/StoreView/Move ops");
            }
        }

        // Phase 0: auto-cast mixed-dtype constants in binary ops so the emitted
        // arithmetic is well-typed. For every Binary in list order: if both
        // operands are constants of differing dtype, cast each to their
        // `least_upper_dtype`; if exactly one operand is a constant, cast that
        // constant to the other (non-const) operand's dtype. Runs before Phase 1
        // so the index arithmetic produced by the move handlers is type-correct.
        {
            let binaries: Vec<OpId> = {
                let mut v = Vec::new();
                let mut scan = self.head;
                while !scan.is_null() {
                    v.push(scan);
                    scan = self.next_op(scan);
                }
                v
            };
            for bin in binaries {
                let Op::Binary { x, y, .. } = self.ops[bin].op else {
                    continue;
                };
                let dx = self.dtype(x);
                let dy = self.dtype(y);
                if dx == dy {
                    continue;
                }
                let cx = matches!(self.ops[x].op, Op::Const(_));
                let cy = matches!(self.ops[y].op, Op::Const(_));
                let mut rewrites: Vec<(OpId, OpId)> = Vec::new();
                match (cx, cy) {
                    (true, true) => {
                        let lu = dx.least_upper_dtype(dy);
                        if dx != lu {
                            rewrites.push((x, self.insert_before(bin, Op::Cast { x, dtype: lu })));
                        }
                        if dy != lu {
                            rewrites.push((y, self.insert_before(bin, Op::Cast { x: y, dtype: lu })));
                        }
                    }
                    (true, false) => {
                        rewrites.push((x, self.insert_before(bin, Op::Cast { x, dtype: dy })));
                    }
                    (false, true) => {
                        rewrites.push((y, self.insert_before(bin, Op::Cast { x: y, dtype: dx })));
                    }
                    (false, false) => {}
                }
                if !rewrites.is_empty() {
                    let map: Map<OpId, OpId> = rewrites.into_iter().collect();
                    self.ops[bin].op.remap_params(&map);
                }
            }
        }

        /*debug_assert!({
            let mut live: Set<OpId> = Set::default();
            let mut stack: Vec<OpId> = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if matches!(self.ops[op_id].op, Op::Store { .. }) {
                    stack.push(op_id);
                }
                op_id = self.next_op(op_id);
            }
            while let Some(id) = stack.pop() {
                if !id.is_null() && live.insert(id) {
                    stack.extend(self.ops[id].op.parameters());
                }
            }
            op_id = self.head;
            while !op_id.is_null() {
                if !live.contains(&op_id) {
                    self.debug();
                    panic!("linearize: dead code detected at op {op_id}");
                }
                op_id = self.next_op(op_id);
            }
            true
        });*/

        // Snapshot the order of global defines so linearize can assert it never
        // reorders the buffers' declaration order.
        let global_params: Vec<(DType, ParamKind, OpId)> = {
            let mut params = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Op::Param { dtype, kind, shape } = self.ops[op_id].op {
                    params.push((dtype, kind, shape));
                }
                op_id = self.next_op(op_id);
            }
            params
        };

        // Anchor for everything linearize inserts: the first original op of the
        // kernel. The group-index/stride scaffolding for a writable global is
        // inserted just before it; any index arithmetic inserted by the handlers
        // must land AFTER that whole scaffolding block (it depends on the group
        // indices) but BEFORE the original compute ops.
        let start = self.head;

        // For each op, shape and strides: (index, stride, left pad, right pad, axis length)
        let mut views: Map<OpId, Vec<SDim>> = Map::default();

        // Maps a writable global define to the store that writes into it. The
        // store handler records the entry (walking dst through any moves to the
        // terminal define); the define handler uses it to write back the store's
        // computed index.
        let mut dst_stores: Map<OpId, OpId> = Map::default();

        // Variable defines already materialized as a load by a move handler (e.g.
        // the narrow offset). The Define handler must skip them, otherwise it would
        // emit a duplicate load after the arithmetic that consumes it.
        let mut consumed_vars: Set<OpId> = Set::default();

        // Stack of open reduce loops, as `(loop_start, anchor)`. `loop_start` is
        // the first dependency of the reduce (where its loop opener is inserted
        // and where its scope begins on rescan); `anchor` is the op right after
        // the reduce's loop opener, used with `insert_before` so index arithmetic
        // that depends on reduce loop ids lands inside the loop, after the loop's
        // own infrastructure. The reverse walk pops an entry when it reaches the
        // matching `loop_start`.
        let mut open_loops: Vec<(OpId, OpId)> = Vec::new();
        // Snapshot the original ops in list order. Handlers insert index arithmetic
        // before `start` or the innermost open-loop anchor; walking a snapshot in
        // reverse avoids processing those inserted ops (they are not view ops and
        // have no view entry).
        let mut op_ids: Vec<OpId> = Vec::new();
        let mut scan = self.head;
        while !scan.is_null() {
            op_ids.push(scan);
            scan = self.next_op(scan);
        }
        // Phase 1: unfold movement ops (reshape/narrow/...) into index
        // arithmetic, converting LoadView/StoreView/ConstView into Load/Store/
        // Const. The inserted arithmetic is appended at anchor positions; its
        // relative order is fixed up in Phase 2. Reductions are left intact
        // here (their loops are emitted in Phase 3).
        for &op_id in op_ids.iter().rev() {
            // Leave loop scopes as the reverse walk exits them. Popped after the
            // op is processed: the loop_start op itself sits inside the loop (it
            // is the first dependency of the reduce input), so its own inserted
            // index arithmetic must still land inside the loop.
            let anchor = open_loops.last().map(|&(_, a)| a).unwrap_or(start);
            match self.ops[op_id].op {
                Op::Const(value) => {
                    let Some(view) = views.remove(&op_id) else { continue };
                    // The constant is a scalar whose value must be nullified where the
                    // view's padding condition is false (padded regions read as zero).
                    let mut pc = self.insert_before(anchor, Op::Const(Constant::Bool(true)));
                    let mut has_pad = false;
                    for d in &view {
                        let idx = d.idx;
                        let lp_id = d.lp;
                        let rp_id = d.rp;
                        let len_op = d.len;
                        let lp = pad_value(self, lp_id);
                        let rp = pad_value(self, rp_id);
                        if lp > 0 || rp > 0 {
                            has_pad = true;
                            if lp > 0 {
                                let lp_m1 = self.insert_const_idx_before(anchor, lp - 1);
                                let t = self.insert_before(anchor, Op::Binary { x: idx, y: lp_m1, bop: BOp::Cmpgt });
                                pc = self.insert_before(anchor, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if rp > 0 {
                                let len_mr = self.insert_before(anchor, Op::Binary { x: len_op, y: rp_id, bop: BOp::Sub });
                                let t = self.insert_before(anchor, Op::Binary { x: idx, y: len_mr, bop: BOp::Cmplt });
                                pc = self.insert_before(anchor, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                    }
                    if has_pad {
                        let pcd = self.insert_before(anchor, Op::Cast { x: pc, dtype: value.dtype() });
                        let z = self.insert_before(anchor, Op::Const(value));
                        self.ops[op_id].op = Op::Binary { x: pcd, y: z, bop: BOp::Mul };
                    }
                }
                Op::Param { dtype, kind, shape } => match kind {
                    // Register-scope defines (e.g. reduce accumulators) are managed
                    // by the ops that create them; only global/variable defines are
                    // rangeified here. Writable globals are store destinations,
                    // read-only globals/variables are load sources. Writables with
                    // MemScope::Variable are left alone (stores to variables are
                    // invalid; the verifier rejects them).
                    ParamKind::GlobalMut => {
                        // Write path: this define is the destination of a store. The
                        // store's index is computed from the define's rangeified view
                        // and written back into the matching store op.
                        let store_id = dst_stores.remove(&op_id).unwrap();
                        let view = views.remove(&op_id).unwrap();
                        let zero = self.insert_const_idx_before(anchor, 0u32);
                        let mut write_index = zero;
                        let mut suffix = self.insert_const_idx_before(anchor, 1u32);
                        for d in view.iter().rev() {
                            let src_idx = self.insert_before(anchor, Op::Binary { x: d.idx, y: d.lp, bop: BOp::Sub });
                            write_index = self.insert_before(anchor, Op::Mad { x: src_idx, y: suffix, z: write_index });
                            // Stride by the compact length (see reshape handler).
                            let psum = self.insert_before(anchor, Op::Binary { x: d.lp, y: d.rp, bop: BOp::Add });
                            let compact = self.insert_before(anchor, Op::Binary { x: d.len, y: psum, bop: BOp::Sub });
                            suffix = self.insert_before(anchor, Op::Binary { x: compact, y: suffix, bop: BOp::Mul });
                        }
                        match &mut self.ops[store_id].op {
                            Op::Store { index, .. } => *index = write_index,
                            _ => unreachable!("graph stores are the only stores at linearize time"),
                        }
                    }
                    ParamKind::Variable => {
                        if consumed_vars.contains(&op_id) {
                            continue;
                        }
                        let view = views.remove(&op_id).unwrap();
                        // Variables are single values (no indexing). Like constants,
                        // they only need the padding mask: where the view is out of
                        // bounds, the loaded value is zeroed.
                        let mut pc = self.insert_before(anchor, Op::Const(Constant::Bool(true)));
                        for d in &view {
                            let len_mr = self.insert_before(anchor, Op::Binary { x: d.len, y: d.rp, bop: BOp::Sub });
                            let t_lo = self.insert_before(anchor, Op::Binary { x: d.idx, y: d.lp, bop: BOp::Cmpge });
                            pc = self.insert_before(anchor, Op::Binary { x: t_lo, y: pc, bop: BOp::And });
                            let t_hi = self.insert_before(anchor, Op::Binary { x: d.idx, y: len_mr, bop: BOp::Cmplt });
                            pc = self.insert_before(anchor, Op::Binary { x: t_hi, y: pc, bop: BOp::And });
                        }
                        // Insert the ro source define immediately before this op so the
                        // global/variable define order (which buffer args bind to) is
                        // preserved.
                        let src = self.insert_before(op_id, Op::Param { dtype, kind, shape });
                        let zero = self.insert_before(anchor, Op::Const(Constant::idx(0)));
                        let z = self.insert_before(anchor, Op::Load { src, index: zero, layout: MemLayout::Scalar });
                        let pcd = self.insert_before(anchor, Op::Cast { x: pc, dtype });
                        self.ops[op_id].op = Op::Binary { x: pcd, y: z, bop: BOp::Mul };
                    }
                    ParamKind::Global => {
                        if consumed_vars.contains(&op_id) {
                            continue;
                        }
                        let view = views.remove(&op_id).unwrap();
                        // Padding condition: valid where index is within the source
                        // extent, all symbolic and unconditional. Pads that are
                        // Const(0) simply fold away in later passes.
                        //   index = sum over axes of (idx - lp) * stride
                        //   pc    = and over axes of (idx >= lp) && (idx < len - rp)
                        let zero = self.insert_before(anchor, Op::Const(Constant::idx(0)));
                        let one = self.insert_before(anchor, Op::Const(Constant::idx(1)));
                        let mut index = zero;
                        let mut pc = self.insert_before(anchor, Op::Const(Constant::Bool(true)));
                        let mut suffix = self.insert_before(anchor, Op::Const(Constant::idx(1)));
                        for d in view.iter().rev() {
                            let src_idx = self.insert_before(anchor, Op::Binary { x: d.idx, y: d.lp, bop: BOp::Sub });
                            index = self.insert_before(anchor, Op::Mad { x: src_idx, y: suffix, z: index });
                            // Stride by the compact length (see reshape handler).
                            let psum = self.insert_before(anchor, Op::Binary { x: d.lp, y: d.rp, bop: BOp::Add });
                            let compact = self.insert_before(anchor, Op::Binary { x: d.len, y: psum, bop: BOp::Sub });
                            suffix = self.insert_before(anchor, Op::Binary { x: compact, y: suffix, bop: BOp::Mul });
                            let t_lo = self.insert_before(anchor, Op::Binary { x: d.idx, y: d.lp, bop: BOp::Cmpge });
                            pc = self.insert_before(anchor, Op::Binary { x: t_lo, y: pc, bop: BOp::And });
                            let len_mr = self.insert_before(anchor, Op::Binary { x: d.len, y: d.rp, bop: BOp::Sub });
                            let t_hi = self.insert_before(anchor, Op::Binary { x: d.idx, y: len_mr, bop: BOp::Cmplt });
                            pc = self.insert_before(anchor, Op::Binary { x: t_hi, y: pc, bop: BOp::And });
                        }
                        // Insert the ro source define immediately before this op so the
                        // global define order (which buffer args bind to) is preserved.
                        let src = self.insert_before(op_id, Op::Param { dtype, kind, shape });
                        // Zero the offset where the padding condition fails, so the load
                        // always reads in-bounds, then zero the loaded value itself.
                        let pcu = self.insert_before(anchor, Op::Cast { x: pc, dtype: IDX_T });
                        let offset = self.insert_before(anchor, Op::Binary { x: pcu, y: index, bop: BOp::Mul });
                        let z = self.insert_before(anchor, Op::Load { src, index: offset, layout: MemLayout::Scalar });
                        let pcd = self.insert_before(anchor, Op::Cast { x: pc, dtype });
                        self.ops[op_id].op = Op::Binary { x: pcd, y: z, bop: BOp::Mul };
                    }
                },
                Op::Store { dst, src, index, layout } => {
                    debug_assert_eq!(index, OpId::NULL);
                    debug_assert_eq!(layout, MemLayout::Scalar);
                    // The store writes the kernel's single contiguous output, so its
                    // view is built from the writable global's shape (the dims live in
                    // the dst Param's `shape`), with the group-index/stride scaffolding
                    // inserted before `start`.
                    let mut dst_param = dst;
                    while let Op::Move { x, .. } = self.ops[dst_param].op {
                        dst_param = x;
                    }
                    let dst_param_op = &self.ops[dst_param].op;
                    assert!(
                        matches!(dst_param_op, Op::Param { kind: ParamKind::GlobalMut, .. }),
                        "store dst chain must terminate at a writable global Param, got {dst_param_op:?}"
                    );
                    let Op::Param { shape: dst_shape, .. } = *dst_param_op else {
                        unreachable!()
                    };
                    assert!(
                        dst_stores.insert(dst_param, op_id).is_none(),
                        "store dst chain terminates at Param {dst_param:?}, which is already a store destination"
                    );
                    self.ops[op_id].op = Op::Store { dst, src, index: OpId::NULL, layout: MemLayout::Scalar };
                    let dims = self.shape_ids(dst_shape);
                    let mut view = Vec::new();
                    for (axis, &len) in dims.iter().enumerate().rev() {
                        let len = if self.dtype(len) != IDX_T {
                            self.insert_before(start, Op::Cast { x: len, dtype: IDX_T })
                        } else {
                            len
                        };
                        let idx = self.insert_before(start, Op::Index { axis: axis as u32, kind: IdxKind::Group(len) });
                        let lp = self.insert_before(start, Op::Const(Constant::idx(0)));
                        let rp = self.insert_before(start, Op::Const(Constant::idx(0)));
                        view.push(SDim::new(idx, lp, rp, len));
                    }
                    view.reverse();
                    views.insert(src, view.clone());
                    views.insert(dst, view);
                }
                Op::Reduce { x, rop, reduce_axis } => {
                    // Build the reduce input x's view with all dims: the non-reduced
                    // dims come from the reduce output's view (set by the Store
                    // handler), and the reduced (last) dim is a freshly-opened loop
                    // over `reduce_axis`. The reduce's `reduce_axis` is repointed at
                    // that loop so the loop gets ordered before the reduce's input
                    // computation in Phase 2 (outer loops before inner ones).
                    let out_view = views.remove(&op_id).unwrap();
                    let loop_id = self.insert_before(anchor, Op::Loop { len: reduce_axis });
                    let zero = self.insert_const_idx_before(anchor, 0u32);
                    let mut view = out_view;
                    view.push(SDim::new(loop_id, zero, zero, reduce_axis));
                    views.insert(x, view);
                    self.ops[op_id].op = Op::Reduce { x, rop, reduce_axis: loop_id };
                }
                Op::Move { x, ref mop } => {
                    match mop.as_ref() {
                        MoveOp::Reshape { shape, input_rank } => {
                            // CORRECT (div/mod) VERSION:
                            // Reshape merges/splits contiguous dims, so axis indices don't
                            // align 1:1. Build a single flat index over the output view (all
                            // the arithmetic LoadView would do), then recover each input axis
                            // by successive div/mod against the input's contiguous strides.
                            //
                            //     let out_view = views[&op_id].clone();
                            //     let x_shape = self.shape(x);
                            //     let mut x_strides = vec![1; x_shape.len()];
                            //     let mut st = 1;
                            //     for a in (0..x_shape.len()).rev() {
                            //         x_strides[a] = st;
                            //         st *= x_shape[a];
                            //     }
                            //     let zero = self.insert_const_idx_before(anchor, 0u32);
                            //     let mut base = zero;
                            //     for &(idx, drift, _, _, _) in &out_view {
                            //         base = self.insert_before(anchor, Op::Mad { x: idx, y: drift, z: base });
                            //     }
                            //     let n = x_shape.len();
                            //     let mut view = Vec::with_capacity(n);
                            //     let mut q = base;
                            //     for a in 0..n {
                            //         let s = x_strides[a];
                            //         let s_id = self.insert_const_idx_before(anchor, s);
                            //         let idx_expr = if a == n - 1 {
                            //             q
                            //         } else {
                            //             let div = self.insert_before(anchor, Op::Binary { x: q, y: s_id, bop: BOp::Div });
                            //             let rem = self.insert_before(anchor, Op::Binary { x: q, y: s_id, bop: BOp::Mod });
                            //             q = rem;
                            //             div
                            //         };
                            //         let len_id = self.insert_const_idx_before(anchor, x_shape[a]);
                            //         view.push((idx_expr, s_id, zero, zero, len_id));
                            //     }
                            //     views.insert(x, view);
                            // Reshape merges/splits contiguous dims, so axis indices don't
                            // align 1:1. The input is read as a single flat index over the
                            // whole (contiguous) input, which equals the flat index over the
                            // output. Build `base` from the output view (all the arithmetic
                            // LoadView would do), then recover each input axis by successive
                            // div/mod against the input's contiguous strides. The input view
                            // is built fully contiguous here; any movement ops upstream of `x`
                            // (processed later, in reverse) apply their own transforms on it.
                            let out_view = views[&op_id].clone();
                            // Anchor the index math at the Move op itself (which
                            // follows its shape dims) rather than the global
                            // `anchor`, so the arithmetic is inserted AFTER the
                            // shape dimensions it depends on, not before them.
                            let zero = self.insert_const_idx_before(op_id, 0u32);
                            let one = self.insert_const_idx_before(op_id, 1u32);
                            let mut base = zero;
                            let mut valid = self.insert_before(op_id, Op::Const(Constant::Bool(true)));
                            for d in &out_view {
                                let lo = self.insert_before(op_id, Op::Binary { x: d.idx, y: d.lp, bop: BOp::Cmpge });
                                let len = if self.dtype(d.len) != IDX_T {
                                    self.insert_before(op_id, Op::Cast { x: d.len, dtype: IDX_T })
                                } else {
                                    d.len
                                };
                                let interior_len = self.insert_before(op_id, Op::Binary { x: len, y: d.rp, bop: BOp::Sub });
                                let hi = self.insert_before(op_id, Op::Binary { x: d.idx, y: interior_len, bop: BOp::Cmplt });
                                let in_axis = self.insert_before(op_id, Op::Binary { x: lo, y: hi, bop: BOp::And });
                                valid = self.insert_before(op_id, Op::Binary { x: valid, y: in_axis, bop: BOp::And });
                            }
                            let mut suffix = one;
                            for d in out_view.iter().rev() {
                                // Subtract the left pad so the flat base skips padded
                                // leading regions of the output view.
                                let src_idx = self.insert_before(op_id, Op::Binary { x: d.idx, y: d.lp, bop: BOp::Sub });
                                base = self.insert_before(op_id, Op::Mad { x: src_idx, y: suffix, z: base });
                                // Stride by the *compact* length (padding `lp`/`rp`
                                // inflate `len`; the flat base must not include them, or
                                // it would over-step the source).
                                let psum = self.insert_before(op_id, Op::Binary { x: d.lp, y: d.rp, bop: BOp::Add });
                                let len = if self.dtype(d.len) != IDX_T {
                                    self.insert_before(op_id, Op::Cast { x: d.len, dtype: IDX_T })
                                } else {
                                    d.len
                                };
                                let compact = self.insert_before(op_id, Op::Binary { x: len, y: psum, bop: BOp::Sub });
                                suffix = self.insert_before(op_id, Op::Binary { x: compact, y: suffix, bop: BOp::Mul });
                            }
                            // The input's contiguous strides: the running product of the
                            // trailing dims' lengths, resolved symbolically from `x`.
                            let x_shape = self.store_shape_ids(x);
                            let n = x_shape.len();
                            let mut x_strides = vec![one; n];
                            let mut st = one;
                            for a in (0..n).rev() {
                                x_strides[a] = st;
                                let len = if self.dtype(x_shape[a]) != IDX_T {
                                    self.insert_before(op_id, Op::Cast { x: x_shape[a], dtype: IDX_T })
                                } else {
                                    x_shape[a]
                                };
                                st = self.insert_before(op_id, Op::Binary { x: len, y: st, bop: BOp::Mul });
                            }
                            let mut view = Vec::with_capacity(n);
                            let mut q = base;
                            for a in 0..n {
                                let s = x_strides[a];
                                let idx_expr = if a == n - 1 {
                                    q
                                } else {
                                    let div = self.insert_before(op_id, Op::Binary { x: q, y: s, bop: BOp::Div });
                                    let rem = self.insert_before(op_id, Op::Binary { x: q, y: s, bop: BOp::Mod });
                                    q = rem;
                                    div
                                };
                                let len = if self.dtype(x_shape[a]) != IDX_T {
                                    self.insert_before(op_id, Op::Cast { x: x_shape[a], dtype: IDX_T })
                                } else {
                                    x_shape[a]
                                };
                                let invalid = self.insert_before(op_id, Op::Binary { x: len, y: one, bop: BOp::Add });
                                let idx_expr = self.branchless_where(valid, idx_expr, invalid);
                                view.push(SDim::new(idx_expr, zero, zero, len));
                            }
                            views.insert(x, view);
                        }
                        MoveOp::Expand { shape } => {
                            let x_shape = self.shape(x);
                            let shape = self.shape(*shape);
                            let view = views[&op_id].clone();
                            let zero = self.insert_const_idx_before(anchor, 0u32);
                            let one = self.insert_const_idx_before(anchor, 1u32);
                            // New leading axes are prepended broadcasts; the input axes
                            // align to the tail of the output shape. A broadcast input
                            // axis reads a single constant element.
                            let offset = shape.len() - x_shape.len();
                            let view: Vec<SDim> = (0..x_shape.len())
                                .map(|a| {
                                    if x_shape[a] == shape[offset + a] {
                                        view[offset + a]
                                    } else {
                                        SDim::new(zero, zero, zero, one)
                                    }
                                })
                                .collect();
                            views.insert(x, view);
                        }
                        MoveOp::Permute { axes } => {
                            // output[a] reads input[axes[a]]. Input axis j is
                            // consumed by output axis inv_axes[j]; copy that
                            // output axis's SDim into input axis j's slot.
                            let x_shape = self.shape(x);
                            let mut inv_axes = vec![0; axes.len()];
                            for (i, &a) in axes.iter().enumerate() {
                                inv_axes[a] = i;
                            }
                            let view = &views[&op_id];
                            let view: Vec<SDim> = (0..x_shape.len()).map(|j| view[inv_axes[j]]).collect();
                            views.insert(x, view);
                        }
                        MoveOp::Flip { axes } => {
                            let axes = axes.clone();
                            let view = views[&op_id].clone();
                            let one = self.insert_const_idx_before(anchor, 1u32);
                            let mut new_view = Vec::with_capacity(view.len());
                            for (a, d) in view.into_iter().enumerate() {
                                if axes.contains(&(a as UAxis)) {
                                    // Reverse the axis: input coord = len - 1 - out_idx.
                                    let len_m1 = self.insert_before(anchor, Op::Binary { x: d.len, y: one, bop: BOp::Sub });
                                    let idx = self.insert_before(anchor, Op::Binary { x: len_m1, y: d.idx, bop: BOp::Sub });
                                    // Padding swaps sides under a flip.
                                    new_view.push(SDim::new(idx, d.rp, d.lp, d.len));
                                } else {
                                    new_view.push(d);
                                }
                            }
                            views.insert(x, new_view);
                        }
                        MoveOp::Pad { axis, lp, rp } => {
                            let axis = *axis;
                            let lp = *lp;
                            let rp = *rp;
                            let x_shape = self.shape(x);
                            let view = views[&op_id].clone();
                            let zero = self.insert_const_idx_before(anchor, 0u32);
                            let lp_val = pad_value(self, lp);
                            let rp_val = pad_value(self, rp);
                            let mut new_view = Vec::with_capacity(view.len());
                            for (a, d) in view.into_iter().enumerate() {
                                if a == axis as usize {
                                    // The input-view index ranges over the padded
                                    // coordinates (length x_shape + lp + rp); the pad
                                    // condition zeros everything outside the interior.
                                    let len = self.insert_const_idx_before(anchor, (x_shape[a] + lp_val + rp_val) as u64);
                                    let lp_id = if lp_val > 0 { lp } else { zero };
                                    let rp_id = if rp_val > 0 { rp } else { zero };
                                    new_view.push(SDim::new(d.idx, lp_id, rp_id, len));
                                } else {
                                    new_view.push(d);
                                }
                            }
                            views.insert(x, new_view);
                        }
                        &MoveOp::Narrow { axis, start, .. } => {
                            let x_shape = self.store_shape_ids(x);
                            let view = views[&op_id].clone();
                            // Narrow slices one axis: the input coordinate along the
                            // narrowed axis is `start + out_idx`. Padding is unchanged
                            // (inherited from the parent's view), only the offset shifts.
                            // The axis length must be the *input's* length on that axis
                            // (not the narrow's output length), so the stride and the
                            // padding bound `idx < len - rp` cover the shifted index.
                            let mut new_view = Vec::with_capacity(view.len());
                            for (a, d) in view.into_iter().enumerate() {
                                if a as UAxis == axis {
                                    let start = if self.dtype(start) != IDX_T {
                                        self.insert_before(anchor, Op::Cast { x: start, dtype: IDX_T })
                                    } else {
                                        start
                                    };
                                    let idx = self.insert_before(anchor, Op::Binary { x: d.idx, y: start, bop: BOp::Add });
                                    let len = if self.dtype(x_shape[a]) != IDX_T {
                                        self.insert_before(anchor, Op::Cast { x: x_shape[a], dtype: IDX_T })
                                    } else {
                                        x_shape[a]
                                    };
                                    new_view.push(SDim::new(idx, d.lp, d.rp, len));
                                } else {
                                    new_view.push(d);
                                }
                            }
                            views.insert(x, new_view);
                        }
                    }
                    self.remap(op_id, x);
                    self.remove_op(op_id);
                }
                Op::Cast { x, .. } | Op::Unary { x, .. } => {
                    if let Some(view) = views.get(&op_id).cloned() {
                        views.insert(x, view);
                    }
                }
                Op::Binary { x, y, .. } => {
                    if let Some(view) = views.get(&op_id).cloned() {
                        views.insert(x, view.clone());
                        views.insert(y, view);
                    }
                }
                Op::Index { .. } => {}
                Op::Stack { ref ops } => {
                    // Stack produces `[n] + first_shape`: output element at leading
                    // index `i` and trailing indices `t` reads input `i` at `t`. So
                    // each input reads at the trailing axes of the output view (the
                    // leading axis selects which source). Assign each stacked input
                    // the output view with the leading SDim dropped, then resolve the
                    // op into a chain of branchless selects on the leading group
                    // index: where(lead==n-1, src_{n-1}, ... where(lead==1, src_1,
                    // src_0)). The src_{k} are the input op ids; when the reverse
                    // walk reaches their Param define they are remapped to loads and
                    // this chain follows automatically.
                    let stacked = ops.to_vec();
                    if let Some(view) = views.get(&op_id).cloned() {
                        debug_assert!(!view.is_empty(), "Stack: empty output view");
                        let leading = view[0].idx;
                        let trailing = &view[1..];
                        for &input in stacked.iter() {
                            views.insert(input, trailing.to_vec());
                        }
                        let n = stacked.len();
                        let mut ret = stacked[n - 1];
                        for k in (0..n - 1).rev() {
                            let k_const = self.insert_before(op_id, Op::Const(Constant::idx(k as u32)));
                            let eq = self.insert_before(op_id, Op::Binary { x: leading, y: k_const, bop: BOp::Eq });
                            ret = self.branchless_where(eq, stacked[k], ret);
                        }
                        self.remap(op_id, ret);
                    }
                    self.remove_op(op_id);
                }
                ref op => {
                    self.debug();
                    unreachable!("{op:?}");
                }
            }
            // Leave loop scopes as the reverse walk exits them, after the
            // loop_start op (which lives inside the loop) has been processed.
            if let Some(&(ls, _)) = open_loops.last()
                && ls == op_id
            {
                open_loops.pop();
            }
        }

        // Phase 2: collect reachable ops from the stores and scope markers, then
        // topologically order their dependencies. Phase 1 may leave the linked
        // list temporarily invalid while inserting and replacing ops, so the slab
        // is the source of truth until this phase rebuilds the list.
        {
            let mut roots = Vec::new();
            for (op_id, op) in self.iter_unordered() {
                match op {
                    Op::Store { .. } | Op::Loop { .. } => roots.push(op_id),
                    Op::Param { .. }
                    | Op::Const(_)
                    | Op::Binary { .. }
                    | Op::Unary { .. }
                    | Op::Cast { .. }
                    | Op::Mad { .. }
                    | Op::Load { .. }
                    | Op::Index { .. }
                    | Op::Reduce { .. } => {}
                    Op::Storage { .. } | Op::Wmma { .. } | Op::Barrier | Op::If { .. } | Op::EndIf | Op::EndLoop => {
                        debug_assert!(false, "unexpected root operation after Phase 1: {op:?}");
                    }
                    _ => {}
                }
            }

            let mut reachable = Set::default();
            let mut pending = roots;
            while let Some(op_id) = pending.pop() {
                if reachable.insert(op_id) {
                    if self.ops.contains_id(op_id) {
                        pending.extend(self.at(op_id).parameters());
                    }
                }
            }

            for op_id in self.ops.ids().collect::<Vec<_>>() {
                if !reachable.contains(&op_id) {
                    self.remove_op(op_id);
                }
            }

            let mut order: Vec<OpId> = Vec::new();
            // Params must precede their users. Recover their original argument
            // order from the Phase 1 snapshot, since neither the temporary links
            // nor slab iteration order represent kernel argument order anymore.
            let mut used_params = Set::default();
            let mut expected_params = global_params.clone();
            expected_params.sort_by_key(|(_, kind, _)| *kind == ParamKind::GlobalMut);
            for (dtype, kind, shape) in expected_params {
                let found = self.iter_unordered().find_map(|(op_id, op)| {
                    if used_params.contains(&op_id) {
                        return None;
                    }
                    match op {
                        Op::Param { dtype: d, kind: k, shape: s } if *d == dtype && *k == kind && *s == shape => Some(op_id),
                        _ => None,
                    }
                });
                let op_id = found.expect("linearize lost a parameter during Phase 1");
                order.push(op_id);
                used_params.insert(op_id);
            }
            // Any newly-created parameters not represented by the original
            // snapshot are appended in slab order.
            for (op_id, op) in self.iter_unordered() {
                if matches!(op, Op::Param { .. }) && used_params.insert(op_id) {
                    order.push(op_id);
                }
            }
            let mut placed: Set<OpId> = order.iter().copied().collect();
            // Append only operations whose dependencies have already been
            // emitted. Marking nodes visited during DFS is insufficient here:
            // a discovered dependency can be skipped by another traversal before
            // it has actually been appended to the order.
            loop {
                let mut progress = false;
                for (op_id, op) in self.iter_unordered() {
                    if !placed.contains(&op_id) {
                        let ready = op.parameters().filter(|p| !p.is_null()).all(|p| placed.contains(&p));
                        if ready {
                            order.push(op_id);
                            placed.insert(op_id);
                            progress = true;
                        }
                    }
                }
                if placed.len() == self.ops.values().count() {
                    break;
                }
                assert!(progress, "linearize dependency ordering contains a cycle or missing operation");
            }
            // Rebuild the kernel's linked list in `order`.
            for (i, &op) in order.iter().enumerate() {
                self.ops[op].prev = if i == 0 { OpId::NULL } else { order[i - 1] };
                self.ops[op].next = if i + 1 == order.len() { OpId::NULL } else { order[i + 1] };
            }
            self.head = order.first().copied().unwrap_or(OpId::NULL);
            self.tail = order.last().copied().unwrap_or(OpId::NULL);
        }

        // Verify the relative order of global defines is unchanged by linearize
        // (read-only defines first, then writable ones, both in original order).
        debug_assert!({
            let mut params = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Op::Param { dtype, kind, shape } = self.ops[op_id].op {
                    params.push((dtype, kind, shape));
                }
                op_id = self.next_op(op_id);
            }
            let mut expected = global_params.clone();
            expected.sort_by_key(|(_, kind, _)| *kind == ParamKind::GlobalMut);
            if params != expected {
                self.debug();
                panic!(
                    "linearize: global define order changed:\n  original = {global_params:?}\n  expected = {expected:?}\n  final = {params:?}"
                );
            }
            true
        });

        // After linearization the parameter shapes are no longer meaningful;
        // clear them so the verify below (and later passes) don't require shape
        // consts to be ordered before the params that reference them.
        for node in self.ops.values_mut() {
            if let Op::Param { shape, .. } = &mut node.op {
                *shape = OpId::NULL;
            }
        }

        // Phase 3: move each loop immediately before its first direct user.
        let mut loop_stack = Vec::new();
        let mut first_users = Map::default();
        let mut scan = self.head;
        while !scan.is_null() {
            let next = self.next_op(scan);
            match self.ops[scan].op {
                Op::Loop { .. } => loop_stack.push(scan),
                Op::EndLoop => {}
                _ => {
                    if let Some(pos) =
                        loop_stack.iter().rposition(|&loop_id| self.ops[scan].op.parameters().any(|p| p == loop_id))
                    {
                        let loop_id = loop_stack.remove(pos);
                        first_users.insert(loop_id, scan);
                    }
                }
            }
            scan = next;
        }
        for (loop_id, first_user) in first_users {
            self.move_op_before(loop_id, first_user);
        }

        self.debug();

        // Phase 4: insert accumulators immediately before their exact loops.
        // No loop movement occurs after this point.
        let reduce_ids: Vec<OpId> =
            self.iter_unordered().filter(|(_, op)| matches!(op, Op::Reduce { .. })).map(|(id, _)| id).collect();
        for op_id in reduce_ids {
            let Op::Reduce { x, rop, reduce_axis } = self.ops[op_id].op else {
                unreachable!()
            };
            let loop_id = reduce_axis;

            let acc_dtype = self.dtype(x);
            let zero = self.insert_const_idx_before(loop_id, 0u32);
            let acc_init = self.insert_before(
                loop_id,
                Op::Const(match rop {
                    BOp::Add => acc_dtype.zero_constant(),
                    BOp::Max => acc_dtype.min_constant(),
                    BOp::Mul => acc_dtype.one_constant(),
                    _ => unreachable!(),
                }),
            );
            let acc = self.insert_before(loop_id, Op::Storage { dtype: acc_dtype, scope: MemScope::Register, len: 1 });
            self.insert_before(loop_id, Op::Store { dst: acc, src: acc_init, index: zero, layout: MemLayout::Scalar });

            // Accumulate inside the loop, then close it, then read the result.
            let load_acc = self.insert_before(op_id, Op::Load { src: acc, index: zero, layout: MemLayout::Scalar });
            let bin_acc = self.insert_before(op_id, Op::Binary { x, y: load_acc, bop: rop });
            self.insert_before(op_id, Op::Store { dst: acc, src: bin_acc, index: zero, layout: MemLayout::Scalar });
            self.insert_before(op_id, Op::EndLoop);
            self.ops[op_id].op = Op::Load { src: acc, index: zero, layout: MemLayout::Scalar };
        }
        //
        // The move handlers may leave dead constants (e.g. unused `one`/`total`
        // scaffold) and duplicate arithmetic behind; CSE and DCE clean those up
        // now that the ops are ordered.
        assert!(
            self.ops.values().all(|node| !matches!(node.op, Op::Move { .. } | Op::Stack { .. })),
            "linearize left a movement or stack operation in the kernel"
        );

        self.verify();

        self.common_subexpression_elimination();
        self.dead_code_elimination();
    }

    pub(crate) fn reduce_dims(&self, op_id: OpId) -> Vec<Dim> {
        let mut params = vec![op_id];
        let mut n_reduce_axes = 0;
        let mut visited = Set::default();
        while let Some(param) = params.pop() {
            if visited.insert(param) {
                match self.ops[param].op {
                    Op::Const(_) => return vec![1],
                    Op::Param { .. } => todo!(),
                    Op::Storage { .. } => {
                        todo!()
                    }
                    Op::Reduce { .. } => n_reduce_axes += 1,
                    Op::Move { ref mop, .. } => match mop.as_ref() {
                        MoveOp::Reshape { shape, .. } => {
                            todo!()
                        }
                        MoveOp::Expand { .. } | MoveOp::Permute { .. } | MoveOp::Pad { .. } => {
                            todo!()
                        }
                        MoveOp::Narrow { .. } => {}
                        MoveOp::Flip { .. } => {}
                    },
                    _ => {}
                }
                params.extend(self.at(param).parameters());
            }
        }
        unreachable!();
    }
}
