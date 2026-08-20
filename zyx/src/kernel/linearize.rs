// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Rangeify movement operations.
//!
//! Reimplements unfold_movement_ops
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
/// (`idx`), contiguous row-major stride (`stride`), left/right pad
/// (`lp`/`rp`) and axis length (`len`). All are `OpId`s resolved lazily.
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

use std::collections::BinaryHeap;

use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, IdxKind, Kernel, MemLayout, MemScope, MoveOp, Op, OpId, ParamKind},
    shape::{Dim, UAxis},
    slab::SlabId,
};

impl Kernel {
    /// Unfold movement operations into index-based operations
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
        let global_params: Vec<(DType, ParamKind)> = {
            let mut params = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Op::Param { dtype, kind, .. } = self.ops[op_id].op {
                    params.push((dtype, kind));
                }
                op_id = self.next_op(op_id);
            }
            params
        };

        self.add_indexing();

        // After linearization the parameter shapes are no longer meaningful;
        // clear them so the verify below (and later passes) don't require shape
        // consts to be ordered before the params that reference them.
        for node in self.ops.values_mut() {
            if let Op::Param { shape, .. } = &mut node.op {
                *shape = OpId::NULL;
            }
        }

        // Read-only (Variable + Global) and writable (GlobalMut) params in
        // linked-list order. Since Phase 1 does not reorder params, forward order
        // is the correct kernel argument order.
        let mut ro_params: Vec<OpId> = Vec::new();
        let mut rw_params: Vec<OpId> = Vec::new();
        {
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Op::Param { kind, .. } = self.ops[op_id].op {
                    match kind {
                        ParamKind::Variable | ParamKind::Global => ro_params.push(op_id),
                        ParamKind::GlobalMut => rw_params.push(op_id),
                    }
                }
                op_id = self.next_op(op_id);
            }
        }
        self.toposort(&ro_params, &rw_params);

        // Verify the relative order of global defines is unchanged by linearize
        // (read-only defines first, then writable ones, both in original order).
        debug_assert!({
            let mut params = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Op::Param { dtype, kind, .. } = self.ops[op_id].op {
                    params.push((dtype, kind));
                }
                op_id = self.next_op(op_id);
            }
            let mut expected = global_params.clone();
            expected.sort_by_key(|(_, kind)| *kind == ParamKind::GlobalMut);
            if params != expected {
                self.debug();
                panic!(
                    "linearize: global define order changed:\n  original = {global_params:?}\n  expected = {expected:?}\n  final = {params:?}"
                );
            }
            true
        });

        self.autocast_scalars();

        self.add_control_flow();

        //
        // The move handlers may leave dead constants (e.g. unused `one`/`total`
        // scaffold) and duplicate arithmetic behind; CSE and DCE clean those up
        // now that the ops are ordered.
        assert!(
            self.ops.values().all(|node| !matches!(node.op, Op::Move { .. } | Op::Stack { .. })),
            "linearize left a movement or stack operation in the kernel"
        );

        // Dedup group-index ops: every store handler emits its own set of
        // `Op::Index { kind: Group }` per output axis, so N stores of the same
        // shape produce N duplicate group indices per axis. Keep the first in
        // linked-list order as canonical and remap the rest onto it. All
        // duplicates for an axis must agree on the length, or the kernel is
        // malformed.
        {
            let mut canonical: Map<u32, OpId> = Map::default();
            let mut lengths: Map<u32, Dim> = Map::default();
            let mut op_id = self.head;
            while !op_id.is_null() {
                let next = self.next_op(op_id);
                if let Op::Index { axis, kind: IdxKind::Group(len) } = self.ops[op_id].op {
                    let len_dim = self.resolve_dim(len).unwrap_or(u64::MAX);
                    if let Some(&l) = lengths.get(&axis) {
                        assert!(len_dim == l, "group index axis={axis} has inconsistent lengths ({} vs {})", l, len_dim);
                        self.remap(op_id, canonical[&axis]);
                        self.remove_op(op_id);
                    } else {
                        lengths.insert(axis, len_dim);
                        canonical.insert(axis, op_id);
                    }
                }
                op_id = next;
            }
        }

        self.common_subexpression_elimination();
        self.dead_code_elimination();
        self.debug();
    }

    fn add_indexing(&mut self) {
        // Shared zero/one index constants used throughout the handlers, hoisted
        // once so every branch reuses them instead of inserting fresh constants.
        let zero = self.const_idx(0);
        let one = self.const_idx(1);

        // For each op, shape and strides: (index, stride, left pad, right pad, axis length)
        let mut views: Map<OpId, Vec<SDim>> = Map::default();

        // Maps a writable global define to the store that writes into it. The
        // store handler records the entry (walking dst through any moves to the
        // terminal define); the define handler uses it to write back the store's
        // computed index.
        let mut dst_stores: Map<OpId, OpId> = Map::default();

        // Snapshot the original ops in list order. Handlers insert index arithmetic
        // before `start` or the innermost open-loop anchor; walking a snapshot in
        // reverse avoids processing those inserted ops (they are not view ops and
        // have no view entry).
        // Collect ops reachable from the store roots. Ops not on any store's
        // dependency chain are dead and must be skipped during the reverse walk
        // (their views are never seeded, and touching them would panic).
        let mut roots: Vec<OpId> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if matches!(self.ops[op_id].op, Op::Store { .. }) {
                roots.push(op_id);
            }
            op_id = self.next_op(op_id);
        }
        let mut reachable = Set::default();
        let mut pending = roots;
        while let Some(op_id) = pending.pop() {
            if self.ops.contains_id(op_id) {
                if reachable.insert(op_id) {
                    pending.extend(self.at(op_id).parameters());
                }
            }
        }
        let mut op_ids: Vec<OpId> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if reachable.contains(&op_id) {
                op_ids.push(op_id);
            }
            op_id = self.next_op(op_id);
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
            match self.ops[op_id].op {
                Op::Const(value) => {
                    let Some(view) = views.remove(&op_id) else { continue };
                    // The constant is a scalar whose value must be nullified where the
                    // view's padding condition is false (padded regions read as zero).
                    // The mask is built unconditionally over every axis, all symbolic;
                    // pads that are Const(0) simply fold away in later passes.
                    let mut pc = self.const_val(true);
                    for d in &view {
                        let len_mr = self.sub(d.len, d.rp);
                        let t_lo = self.cmpge(d.idx, d.lp);
                        pc = self.and(t_lo, pc);
                        let t_hi = self.cmplt(d.idx, len_mr);
                        pc = self.and(t_hi, pc);
                    }
                    let z = self.push_back(Op::Const(value));
                    self.ops[op_id].op = Op::Binary { x: pc, y: z, bop: BOp::Mul };
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
                        let mut write_index = zero;
                        for d in view.iter().rev() {
                            let src_idx = self.sub(d.idx, d.lp);
                            write_index = self.mad(src_idx, d.st, write_index);
                        }
                        match &mut self.ops[store_id].op {
                            Op::Store { index, .. } => *index = write_index,
                            _ => unreachable!("graph stores are the only stores at linearize time"),
                        }
                    }
                    ParamKind::Variable => {
                        let view = views.remove(&op_id).unwrap();
                        // Variables are single values (no indexing). Like constants,
                        // they only need the padding mask: where the view is out of
                        // bounds, the loaded value is zeroed.
                        let mut pc = self.const_val(true);
                        for d in &view {
                            let len_mr = self.sub(d.len, d.rp);
                            let t_lo = self.cmpge(d.idx, d.lp);
                            pc = self.and(t_lo, pc);
                            let t_hi = self.cmplt(d.idx, len_mr);
                            pc = self.and(t_hi, pc);
                        }
                        // Insert the ro source define immediately before this op so the
                        // global/variable define order (which buffer args bind to) is
                        // preserved.
                        let src = self.insert_before(op_id, Op::Param { dtype, kind, shape });
                        let z = self.load(src, zero, MemLayout::Scalar);
                        self.ops[op_id].op = Op::Binary { x: pc, y: z, bop: BOp::Mul };
                    }
                    ParamKind::Global => {
                        let view = views.remove(&op_id).unwrap();
                        // Padding condition: valid where index is within the source
                        // extent, all symbolic and unconditional. Pads that are
                        // Const(0) simply fold away in later passes.
                        //   index = sum over axes of (idx - lp) * stride
                        //   pc    = and over axes of (idx >= lp) && (idx < len - rp)
                        let mut index = self.const_idx(0);
                        let mut pc = self.const_val(true);
                        for d in view.iter().rev() {
                            let src_idx = self.sub(d.idx, d.lp);
                            index = self.mad(src_idx, d.st, index);
                            // An axis is only actually padded when its clamped lp/rp are
                            // nonzero. A slice clamps both to 0 (its output length is not
                            // the source extent), so the bounds checks must not apply
                            // there -- otherwise idx < len - rp wrongly rejects the
                            // shifted index. branchless_where keeps this fully symbolic.
                            let lp_gt = self.cmpgt(d.lp, zero);
                            let rp_gt = self.cmpgt(d.rp, zero);
                            let padded = self.or_(lp_gt, rp_gt);
                            let ge = self.cmpge(d.idx, d.lp);
                            let tv = self.const_val(true);
                            let t_lo = self.branchless_where(padded, ge, tv);
                            pc = self.and(t_lo, pc);
                            let len_mr = self.sub(d.len, d.rp);
                            let lt = self.cmplt(d.idx, len_mr);
                            let tv2 = self.const_val(true);
                            let t_hi = self.branchless_where(padded, lt, tv2);
                            pc = self.and(t_hi, pc);
                        }
                        // Insert the ro source define immediately before this op so the
                        // global define order (which buffer args bind to) is preserved.
                        let src = self.insert_before(op_id, Op::Param { dtype, kind, shape });
                        // Zero the offset where the padding condition fails, so the load
                        // always reads in-bounds, then zero the loaded value itself.
                        let offset = self.mul(pc, index);
                        let z = self.load(src, offset, MemLayout::Scalar);
                        self.ops[op_id].op = Op::Binary { x: pc, y: z, bop: BOp::Mul };
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
                    let mut st = one;
                    for (axis, &len) in dims.iter().enumerate().rev() {
                        let idx = self.group_index(axis as u32, len);
                        view.push(SDim::new(idx, st, zero, zero, len));
                        st = self.mul(st, len);
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
                    let loop_id = self.loop_(reduce_axis);
                    // Non-reduced axes must use the reduce input `x`'s row-major
                    // strides (idx/lp/rp/len come from the output view, stride is
                    // recomputed from the input shape, which includes the reduced
                    // axis). The reduced axis is the freshly-opened loop with the
                    // input's contiguous stride and zero padding.
                    let x_shape = self.store_shape_ids(x);
                    let n = x_shape.len();
                    let mut x_strides = vec![one; n];
                    let mut st = one;
                    for a in (0..n).rev() {
                        x_strides[a] = st;
                        st = self.mul(x_shape[a], st);
                    }
                    let non_reduce = out_view.len();
                    let mut view = Vec::with_capacity(n);
                    for (e, d) in out_view.iter().enumerate() {
                        view.push(SDim::new(d.idx, x_strides[e], d.lp, d.rp, d.len));
                    }
                    for a in non_reduce..n {
                        view.push(SDim::new(loop_id, x_strides[a], zero, zero, x_shape[a]));
                    }
                    views.insert(x, view);
                    self.ops[op_id].op = Op::Reduce { x, rop, reduce_axis: loop_id };
                }
                Op::Move { x, ref mop } => {
                    match mop.as_ref() {
                        MoveOp::Reshape { .. } => {
                            // Reshape merges/splits contiguous dims, so axis indices don't
                            // align 1:1. The input is read as a single flat index over the
                            // whole (contiguous) input, which equals the flat index over the
                            // output. Build `base` from the output view using each axis's
                            // stored stride, then recover each input axis by successive
                            // div/mod against the input's contiguous strides. Any pad/crop
                            // offsets on the reshape output's axes have already been baked
                            // into `d.idx` by the pad handlers, so no lp handling is needed
                            // here.
                            let out_view = views[&op_id].clone();
                            let x_shape = self.store_shape_ids(x);
                            let n = x_shape.len();
                            let mut x_strides = vec![one; n];
                            let mut st = one;
                            for a in (0..n).rev() {
                                x_strides[a] = st;
                                st = self.mul(x_shape[a], st);
                            }
                            // Validity mask over the output view: a recovered input
                            // coordinate is only meaningful where the output is within
                            // its own source extent (idx >= lp && idx < len - rp).
                            // Padded output regions must read as zero, so invalid
                            // recovered indices are clamped to len + 1 (out of bounds).
                            let mut valid = self.const_val(true);
                            for d in &out_view {
                                let lo = self.cmpge(d.idx, d.lp);
                                let interior_len = self.sub(d.len, d.rp);
                                let hi = self.cmplt(d.idx, interior_len);
                                let in_axis = self.and(lo, hi);
                                valid = self.and(valid, in_axis);
                            }
                            let mut base = zero;
                            for d in &out_view {
                                base = self.mad(d.idx, d.st, base);
                            }
                            let mut view = Vec::with_capacity(n);
                            let mut q = base;
                            for a in 0..n {
                                let s = x_strides[a];
                                let idx_expr = if a == n - 1 {
                                    q
                                } else {
                                    let div = self.div(q, s);
                                    let rem = self.mod_(q, s);
                                    q = rem;
                                    div
                                };
                                let len = x_shape[a];
                                let invalid = self.add(len, one);
                                let idx_expr = self.branchless_where(valid, idx_expr, invalid);
                                view.push(SDim::new(idx_expr, x_strides[a], zero, zero, len));
                            }
                            views.insert(x, view);
                        }
                        &MoveOp::Expand { shape } => {
                            // Broadcast determination is symbolic: an input axis is
                            // broadcast iff its dim resolves to 1 and the output dim
                            // resolves to something != 1 (mirrors tinygrad's
                            // broadcast_axes/resolve). A dynamic dim resolves to None
                            // and is treated as non-broadcast (identity), the safe
                            // default. No concrete shape() lookup is required.
                            let x_shape = self.store_shape_ids(x);
                            let shape = self.shape_ids(shape);
                            // New leading axes are prepended broadcasts; the input axes
                            // align to the tail of the output shape. A broadcast input
                            // axis reads a single constant element (stride 0); a
                            // non-broadcast axis keeps the input's own contiguous
                            // stride (not the output view's), so the load indexes the
                            // compact input.
                            let offset = shape.len() - x_shape.len();
                            let n = x_shape.len();
                            let mut x_strides = vec![one; n];
                            let mut st = one;
                            for a in (0..n).rev() {
                                x_strides[a] = st;
                                st = self.mul(x_shape[a], st);
                            }
                            let out_view = views[&op_id].clone();
                            let view = if n == 0 {
                                // Scalar input broadcasts to every axis: the input view
                                // is the whole output view, so the pad mask propagates.
                                out_view
                            } else {
                                let mut v = Vec::with_capacity(n);
                                for a in 0..n {
                                    let broadcast = self.resolve_dim(x_shape[a]) == Some(1)
                                        && self.resolve_dim(shape[offset + a]) != Some(1);
                                    let d = out_view[offset + a];
                                    let stride = if broadcast {
                                        zero
                                    } else if matches!(self.ops[d.st].op, Op::Const(c) if c.as_dim() == Some(0)) {
                                        // A downstream op already broadcasts this axis
                                        // (stride 0); keep it zero instead of falling back
                                        // to the input's contiguous stride, which would
                                        // leak a loop index into a broadcast axis.
                                        zero
                                    } else {
                                        x_strides[a]
                                    };
                                    v.push(SDim::new(d.idx, stride, d.lp, d.rp, d.len));
                                }
                                v
                            };
                            views.insert(x, view);
                        }
                        MoveOp::Permute { axes } => {
                            // Pure backwards permutation: input axis j is consumed by
                            // output axis inv_axes[j], so the input view's axis j is
                            // exactly the output view's axis inv_axes[j]. No shape
                            // lookup or stride recomputation is needed -- the SDims are
                            // simply reordered.
                            let axes = axes.clone();
                            let view = views[&op_id].clone();
                            let mut inv_axes = vec![0; axes.len()];
                            for (i, &a) in axes.iter().enumerate() {
                                inv_axes[a] = i;
                            }
                            let view: Vec<SDim> = inv_axes.iter().map(|&j| view[j]).collect();
                            views.insert(x, view);
                        }
                        MoveOp::Flip { axes } => {
                            let axes = axes.clone();
                            let view = views[&op_id].clone();
                            let mut new_view = Vec::with_capacity(view.len());
                            for (a, d) in view.into_iter().enumerate() {
                                if axes.contains(&(a as UAxis)) {
                                    // Reverse the axis: input coord = len - 1 - out_idx.
                                    let len_m1 = self.sub(d.len, one);
                                    let idx = self.sub(len_m1, d.idx);
                                    // Padding swaps sides under a flip.
                                    new_view.push(SDim::new(idx, d.st, d.rp, d.lp, d.len));
                                } else {
                                    new_view.push(d);
                                }
                            }
                            views.insert(x, new_view);
                        }
                        &MoveOp::Pad { axis, lp, rp } => {
                            // Negative left pad is a slice offset: the input index
                            // shifts right by `-lp`. Positive pads are true padding.
                            // All offset/clamp logic is branchless and fully symbolic:
                            //   idx = idx + max(-lp, 0)
                            //   lp  = max(lp, 0)
                            //   rp  = max(rp, 0)
                            //   len = max(x_shape[a] + lp + rp, 0)
                            let mut view = views[&op_id].clone();
                            let d = view[axis].clone();
                            let len = self.sub(d.len, lp);
                            let len = self.sub(len, rp);
                            let lp = self.sub(d.lp, lp);
                            let rp = self.sub(d.rp, rp);
                            view[axis] = SDim { idx: d.idx, st: OpId::NULL, lp, rp, len  };
                            views.insert(x, view);
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
                            let n = x_shape.len();
                            let mut x_strides = vec![one; n];
                            let mut st = one;
                            for a in (0..n).rev() {
                                x_strides[a] = st;
                                st = self.mul(x_shape[a], st);
                            }
                            let mut new_view = Vec::with_capacity(view.len());
                            for (a, d) in view.into_iter().enumerate() {
                                if a as UAxis == axis {
                                    let idx = self.add(d.idx, start);
                                    new_view.push(SDim::new(idx, x_strides[a], d.lp, d.rp, x_shape[a]));
                                } else {
                                    new_view.push(SDim::new(d.idx, x_strides[a], d.lp, d.rp, d.len));
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
                            let k_const = self.push_back(Op::Const(Constant::idx(k as i64)));
                            let eq = self.eq(leading, k_const);
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
        }
    }

    fn toposort(&mut self, ro_params: &[OpId], rw_params: &[OpId]) {
        // Phase 2: collect reachable ops from the store roots, then topologically
        // order their dependencies. Phase 1 may leave the linked list temporarily
        // invalid while inserting and replacing ops, so the slab is the source of
        // truth until this phase rebuilds the list.
        {
            let mut roots = Vec::new();
            for (op_id, op) in self.iter_unordered() {
                match op {
                    Op::Store { .. } => roots.push(op_id),
                    Op::Param { .. }
                    | Op::Const(_)
                    | Op::Binary { .. }
                    | Op::Unary { .. }
                    | Op::Cast { .. }
                    | Op::Mad { .. }
                    | Op::Load { .. }
                    | Op::Index { .. }
                    | Op::Reduce { .. }
                    | Op::Loop { .. } => {}
                    Op::Storage { .. } | Op::Wmma { .. } | Op::Barrier | Op::If { .. } | Op::EndIf | Op::EndLoop => {
                        debug_assert!(false, "unexpected root operation after Phase 1: {op:?}");
                    }
                    _ => {}
                }
            }

            // Reachability from the store roots. Any op not on a store's dependency
            // chain is dead and removed.
            let mut reachable = Set::default();
            let mut pending = roots;
            while let Some(op_id) = pending.pop() {
                if self.ops.contains_id(op_id) {
                    if reachable.insert(op_id) {
                        pending.extend(self.at(op_id).parameters());
                    }
                }
            }

            for op_id in self.ops.ids().collect::<Vec<_>>() {
                if !reachable.contains(&op_id) && !matches!(self.ops[op_id].op, Op::Param { .. }) {
                    self.remove_op(op_id);
                }
            }

            // Get reduce ids in sorted order, from innermost to outermost
            let mut reduce_ids: Vec<OpId> = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if matches!(self.ops[op_id].op, Op::Reduce { .. }) {
                    reduce_ids.push(op_id);
                }
                op_id = self.next_op(op_id);
            }

            let mut region_depth: Map<OpId, u32> = reachable.iter().map(|&x| (x, 0)).collect();
            // Generate region depth for all ops in reduce ids
            let mut n = reduce_ids.len() as u32;
            for reduce_id in reduce_ids {
                let Op::Reduce { x, reduce_axis, .. } = self.ops[reduce_id].op else {
                    unreachable!()
                };
                // Backward
                let mut stack = vec![x];
                let mut descendants: Map<OpId, Set<OpId>> = Map::default();
                while let Some(parent) = stack.pop() {
                    if reachable.contains(&parent) {
                        for child in self.ops[parent].op.parameters() {
                            if let Some(parents) = descendants.get_mut(&child) {
                                parents.insert(parent);
                            } else {
                                stack.push(child);
                                descendants.insert(child, [parent].into_iter().collect());
                            }
                        }
                    }
                }
                // Forward
                let mut stack = vec![reduce_axis];
                let mut visited: Set<OpId> = Set::default();
                while let Some(child) = stack.pop() {
                    if visited.insert(child) {
                        if let Some(parents) = descendants.get(&child) {
                            stack.extend(parents);
                        }
                    }
                }
                for x in visited {
                    *region_depth.get_mut(&x).unwrap() += n;
                }
                n -= 1;
            }

            let mut ideal: Vec<OpId> = reachable.iter().copied().collect();
            ideal.sort_by_key(|&op_id| {
                let op_priority = match self.ops[op_id].op {
                    Op::Param { .. } => -20,
                    Op::Index { .. } => -15,
                    Op::Const(_) => -10,
                    Op::Loop { .. } => 5,
                    Op::Reduce { .. } => -5,
                    _ => 0,
                };
                let pri = op_priority + region_depth.get(&op_id).copied().unwrap_or(0) as i32;
                (pri, op_id)
            });
            let nkey: Map<OpId, u64> = ideal.iter().enumerate().map(|(i, &id)| (id, i as u64)).collect();

            // out_degree[u] = number of consumers of u (ops referencing u).
            let mut out_degree: Map<OpId, u32> = Map::default();
            for (op_id, op) in self.iter_unordered() {
                if !reachable.contains(&op_id) {
                    continue;
                }
                for p in op.parameters() {
                    if !p.is_null() {
                        *out_degree.entry(p).or_default() += 1;
                    }
                }
            }

            // Seed with the sinks (out_degree 0), pop the highest ideal-order key
            // first, and emit each op once all its consumers are emitted.
            let mut heap: BinaryHeap<(u64, OpId)> = BinaryHeap::new();
            for &op_id in &reachable {
                if out_degree.get(&op_id).copied().unwrap_or(0) == 0 {
                    heap.push((nkey[&op_id], op_id));
                }
            }
            let mut order = Vec::new();
            while let Some((_, op_id)) = heap.pop() {
                order.push(op_id);
                for p in self.at(op_id).parameters() {
                    if p.is_null() {
                        continue;
                    }
                    let d = out_degree.get_mut(&p).expect("consumer of a reachable op must have an out_degree entry");
                    *d -= 1;
                    if *d == 0 {
                        heap.push((nkey[&p], p));
                    }
                }
            }
            assert!(order.len() == reachable.len(), "linearize dependency ordering contains a cycle or missing operation");
            order.reverse();

            // Move the params to the front: read-only (Variable + Global) first,
            // then writable (GlobalMut), each in linked-list order.
            order.retain(|op| !matches!(self.ops[*op].op, Op::Param { .. }));
            let mut final_order = Vec::with_capacity(order.len() + ro_params.len() + rw_params.len());
            final_order.extend(ro_params.iter().copied());
            final_order.extend(rw_params.iter().copied());
            final_order.extend(order);

            // Rebuild the kernel's linked list in `final_order`.
            for (i, &op) in final_order.iter().enumerate() {
                self.ops[op].prev = if i == 0 { OpId::NULL } else { final_order[i - 1] };
                self.ops[op].next = if i + 1 == final_order.len() {
                    OpId::NULL
                } else {
                    final_order[i + 1]
                };
            }
            self.head = final_order.first().copied().unwrap_or(OpId::NULL);
            self.tail = final_order.last().copied().unwrap_or(OpId::NULL);
        }
    }

    // Auto-cast scalar operands in arithmetic so mixed-dtype binaries are
    // well-typed. Runs after toposort so it sees every Binary/Mad in the final
    // op list. Symmetric over all operands: if dtypes differ, every operand
    // with shape `[]` (constants, variables, and index/loop scalars) is cast so
    // the operation is well-typed; if no operand is scalar, the kernel is
    // broken and this panics.
    fn autocast_scalars(&mut self) {
        let ops: Vec<OpId> = {
            let mut v = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                v.push(op_id);
                op_id = self.next_op(op_id);
            }
            v
        };
        for op_id in ops {
            let operands: Vec<OpId> = match self.ops[op_id].op {
                Op::Binary { x, y, .. } => vec![x, y],
                Op::Mad { x, y, z } => vec![x, y, z],
                Op::Load { index, .. } | Op::Store { index, .. } => {
                    if index.is_null() || self.dtype(index) == IDX_T {
                        continue;
                    }
                    let cast = self.insert_before(op_id, Op::Cast { x: index, dtype: IDX_T });
                    match &mut self.ops[op_id].op {
                        Op::Load { index, .. } | Op::Store { index, .. } => *index = cast,
                        _ => unreachable!(),
                    }
                    continue;
                }
                _ => continue,
            };
            let dtypes: Vec<DType> = operands.iter().map(|&o| self.dtype(o)).collect();
            if dtypes.iter().all(|&d| d == dtypes[0]) {
                continue;
            }
            // Scalars are operands whose value shape is `[]`.
            let scalars: Vec<(OpId, DType)> =
                operands.iter().copied().zip(dtypes.iter().copied()).filter(|&(o, _)| self.shape(o).is_empty()).collect();
            if scalars.is_empty() {
                self.debug();
                panic!("autocast_scalars: mixed-dtype op {op_id} has no scalar operand to cast");
            }
            let target = if scalars.len() == operands.len() {
                // All operands are scalars: fold least_upper_dtype over the distinct dtypes.
                let mut target = dtypes[0];
                for &d in &dtypes[1..] {
                    target = target.least_upper_dtype(d);
                }
                target
            } else {
                // Mixed: every non-scalar operand must share one dtype; cast the
                // scalars to it.
                let nonscalar: Vec<DType> = operands
                    .iter()
                    .copied()
                    .zip(dtypes.iter().copied())
                    .filter(|&(o, _)| !self.shape(o).is_empty())
                    .map(|(_, d)| d)
                    .collect();
                let target = nonscalar[0];
                if nonscalar.iter().any(|&d| d != target) {
                    self.debug();
                    panic!("autocast_scalars: mixed-dtype op {op_id} has non-scalar operands of differing dtype");
                }
                target
            };
            let mut rewrites: Vec<(OpId, OpId)> = Vec::new();
            for &(o, d) in &scalars {
                if d != target {
                    rewrites.push((o, self.insert_before(op_id, Op::Cast { x: o, dtype: target })));
                }
            }
            if !rewrites.is_empty() {
                let map: Map<OpId, OpId> = rewrites.into_iter().collect();
                self.ops[op_id].op.remap_params(&map);
            }
        }
    }

    fn add_control_flow(&mut self) {
        // Phase 3: insert accumulators immediately before their exact loops.
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
    }
}
