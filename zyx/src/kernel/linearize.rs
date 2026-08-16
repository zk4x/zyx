// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

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
//! Only after linearize does the kernel have `Load` ops, so the `loads` list only
//! becomes meaningful then. This matters for any pass that maps kernel args to
//! buffers: pre-linearize, map from the global `Define` ops (in op order), not
//! from a `loads` list.

#![allow(unused)]

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
    pub fn linearize(&mut self, output_shape: &[OpId]) {
        if !self.ops.values().any(|n| matches!(n.op, Op::Store { index: OpId::NULL, .. })) {
            return;
        }

        #[cfg(debug_assertions)]
        {
            let has_gidx = self.ops.values().any(|n| matches!(n.op, Op::Index { kind: IdxKind::Group, .. }));
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
        let global_defines: Vec<(DType, ParamKind)> = {
            let mut defines = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Op::Param { dtype, kind } = self.ops[op_id].op {
                    defines.push((dtype, kind));
                }
                op_id = self.next_op(op_id);
            }
            defines
        };

        let init_view = {
            let head = self.head;
            let mut init_view = Vec::new();
            // Contiguous row-major stride: axis i has stride = product of the
            // symbolic output dims after it. Walk backwards carrying a running
            // suffix product (built with Mad), then reverse to keep axis order.
            // The innermost axis gets a trailing Const(1) stride.
            let mut suffix = self.insert_before(head, Op::Const(Constant::idx(1)));
            for (axis, &len) in output_shape.iter().enumerate().rev() {
                let idx = self.insert_before(head, Op::Index { len, axis: axis as u32, kind: IdxKind::Group });
                let st = suffix;
                let lp = self.insert_before(head, Op::Const(Constant::idx(0)));
                let rp = self.insert_before(head, Op::Const(Constant::idx(0)));
                init_view.push((idx, st, lp, rp, len));
                suffix = self.insert_before(head, Op::Binary { x: len, y: suffix, bop: BOp::Mul });
            }
            init_view.reverse();
            init_view
        };

        // For each op, shape and strides: (index, stride, left pad, right pad, axis length)
        let mut views: Map<OpId, Vec<(OpId, OpId, OpId, OpId, OpId)>> = Map::default();

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
        let start = self.head;
        while !scan.is_null() {
            op_ids.push(scan);
            scan = self.next_op(scan);
        }
        for &op_id in op_ids.iter().rev() {
            // Leave loop scopes as the reverse walk exits them. Popped after the
            // op is processed: the loop_start op itself sits inside the loop (it
            // is the first dependency of the reduce input), so its own inserted
            // index arithmetic must still land inside the loop.
            let anchor = open_loops.last().map(|&(_, a)| a).unwrap_or(start);
            match self.ops[op_id].op {
                Op::Const(value) => {
                    let view = views.remove(&op_id).unwrap();
                    // The constant is a scalar whose value must be nullified where the
                    // view's padding condition is false (padded regions read as zero).
                    let mut pc = self.insert_before(anchor, Op::Const(Constant::Bool(true)));
                    let mut has_pad = false;
                    for &(idx, _st, lp_id, rp_id, len_op) in &view {
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
                Op::Param { dtype, kind } => {
                    // Register-scope defines (e.g. reduce accumulators) are managed
                    // by the ops that create them; only global/variable defines are
                    // rangeified here. Writable globals are store destinations,
                    // read-only globals/variables are load sources. Writables with
                    // MemScope::Variable are left alone (stores to variables are
                    // invalid; the verifier rejects them).
                    if kind == ParamKind::GlobalMut {
                        // Write path: this define is the destination of a store. The
                        // store's index is computed from the define's rangeified view
                        // and written back into the matching store op.
                        let store_id = dst_stores.remove(&op_id).unwrap();
                        let view = views.remove(&op_id).unwrap();
                        let zero = self.insert_const_idx_before(anchor, 0u32);
                        let mut write_index = zero;
                        let mut has_pad = false;
                        for (index_elem, stride, lp_id, rp_id, _len_op) in &view {
                            let lp = pad_value(self, *lp_id);
                            let rp = pad_value(self, *rp_id);
                            has_pad |= lp > 0 || rp > 0;
                            let src_idx = if lp == 0 {
                                *index_elem
                            } else {
                                self.insert_before(anchor, Op::Binary { x: *index_elem, y: *lp_id, bop: BOp::Sub })
                            };
                            write_index = self.insert_before(anchor, Op::Mad { x: src_idx, y: *stride, z: write_index });
                        }
                        // A store cannot write padding: the store covers exactly the
                        // define's writable extent, so padding here is invalid.
                        debug_assert!(!has_pad, "store destination define has padding: {has_pad}");
                        match &mut self.ops[store_id].op {
                            Op::Store { index, .. } => *index = write_index,
                            _ => unreachable!("graph stores are the only stores at linearize time"),
                        }
                        continue;
                    }
                    if consumed_vars.contains(&op_id) {
                        continue;
                    }
                    let shape = todo!();
                    let view = views.remove(&op_id).unwrap();
                    let zero = self.insert_const_idx_before(anchor, 0u32);
                    if kind == ParamKind::Variable {
                        // Variables are single values (no indexing). Like constants,
                        // they only need the padding mask: where the view is out of
                        // bounds, the loaded value is zeroed.
                        let mut pc = self.insert_before(anchor, Op::Const(Constant::Bool(true)));
                        let mut has_pad = false;
                        for &(idx, _st, lp_id, rp_id, len_op) in &view {
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
                        // Insert the ro source define immediately before this op so the
                        // global/variable define order (which buffer args bind to) is
                        // preserved.
                        let src = self.insert_before(op_id, Op::Param { dtype, kind });
                        if has_pad {
                            let z = self.insert_before(anchor, Op::Load { src, index: zero, layout: MemLayout::Scalar });
                            let pcd = self.insert_before(anchor, Op::Cast { x: pc, dtype });
                            self.ops[op_id].op = Op::Binary { x: pcd, y: z, bop: BOp::Mul };
                        } else {
                            self.ops[op_id].op = Op::Load { src, index: zero, layout: MemLayout::Scalar };
                        }
                        continue;
                    }
                    // Padding condition: valid where index is within the source extent.
                    // index = sum over axes of (idx - lp) * stride
                    // pc = and over padded axes of idx > lp-1 && idx < len-rp
                    let mut index = zero;
                    let mut pc = self.insert_before(anchor, Op::Const(Constant::Bool(true)));
                    let mut has_pad = false;
                    for &(idx, st, lp_id, rp_id, len_op) in &view {
                        let lp = pad_value(self, lp_id);
                        let rp = pad_value(self, rp_id);
                        let src_idx = if lp == 0 {
                            idx
                        } else {
                            self.insert_before(anchor, Op::Binary { x: idx, y: lp_id, bop: BOp::Sub })
                        };
                        index = self.insert_before(anchor, Op::Mad { x: src_idx, y: st, z: index });
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
                    // Insert the ro source define immediately before this op so the
                    // global define order (which buffer args bind to) is preserved.
                    let src = self.insert_before(op_id, Op::Param { dtype, kind });
                    if has_pad {
                        // Zero the offset where the padding condition fails, so the load
                        // always reads in-bounds, then zero the loaded value itself.
                        let pcu = self.insert_before(anchor, Op::Cast { x: pc, dtype: IDX_T });
                        let offset = self.insert_before(anchor, Op::Binary { x: pcu, y: index, bop: BOp::Mul });
                        let z = self.insert_before(anchor, Op::Load { src, index: offset, layout: MemLayout::Scalar });
                        let pcd = self.insert_before(anchor, Op::Cast { x: pc, dtype });
                        self.ops[op_id].op = Op::Binary { x: pcd, y: z, bop: BOp::Mul };
                    } else {
                        self.ops[op_id].op = Op::Load { src, index, layout: MemLayout::Scalar };
                    }
                }
                Op::Store { dst, src, index, layout } => {
                    debug_assert_eq!(index, OpId::NULL);
                    debug_assert_eq!(layout, MemLayout::Scalar);
                    // The store writes the kernel's single contiguous output, so its
                    // view is `init_view` (built from the symbolic `output_shape`).
                    let view = init_view.clone();
                    // The store index is written back by the terminal define (as a
                    // writable global) when its walk reaches it. Walk dst through the
                    // movement ops (these are the only ops allowed between a store and
                    // the define it writes) and record the mapping.
                    let mut dst_define = dst;
                    while let Op::Move { x, .. } = self.ops[dst_define].op {
                        dst_define = x;
                    }
                    let dst_define_op = &self.ops[dst_define].op;
                    assert!(
                        matches!(dst_define_op, Op::Param { kind: ParamKind::GlobalMut, .. }),
                        "store dst chain must terminate at a writable global define, got {dst_define_op:?}"
                    );
                    assert!(
                        dst_stores.insert(dst_define, op_id).is_none(),
                        "store dst chain terminates at define {dst_define:?}, which is already a store destination"
                    );
                    self.ops[op_id].op = Op::Store { dst, src, index: OpId::NULL, layout: MemLayout::Scalar };
                    views.insert(src, view.clone());
                    views.insert(dst, view);
                }
                Op::Reduce { x, rop, n_axes } => {
                    todo!()
                    // Collect all transitive dependencies of the reduce input and the
                    // accumulator dtype. The loop that wraps the reduction is opened at
                    // the soonest dependency that appears in the graph.
                    /*let mut reduce_loop_ops_set = Set::default();
                    let mut params = vec![x];
                    let mut acc_dtype = None;
                    while let Some(param) = params.pop() {
                        if reduce_loop_ops_set.insert(param) {
                            params.extend(self.at(param).parameters());
                            if acc_dtype.is_none() {
                                match self.at(param) {
                                    &Op::Storage { dtype, .. } | &Op::Cast { dtype, .. } => acc_dtype = Some(dtype),
                                    Op::Const(v) => acc_dtype = Some(v.dtype()),
                                    _ => {}
                                }
                            }
                        }
                    }
                    let acc_dtype = acc_dtype.unwrap();

                    let mut loop_start = OpId::NULL;
                    let mut scan = self.head;
                    while !scan.is_null() {
                        if reduce_loop_ops_set.contains(&scan) {
                            loop_start = scan;
                            break;
                        }
                        scan = self.next_op(scan);
                    }

                    // const zero + init accumulator constant.
                    let const_zero = self.insert_const_idx_before(loop_start, 0u32);
                    let acc_init_id = self.insert_before(
                        loop_start,
                        Op::Const(match rop {
                            BOp::Add => acc_dtype.zero_constant(),
                            BOp::Max => acc_dtype.min_constant(),
                            BOp::Mul => acc_dtype.one_constant(),
                            _ => unreachable!(),
                        }),
                    );
                    let acc = self.insert_before(loop_start, Op::Storage { dtype: acc_dtype, scope: MemScope::Register, len: 1 });
                    self.insert_before(
                        loop_start,
                        Op::Store { dst: acc, src: acc_init_id, index: const_zero, layout: MemLayout::Scalar },
                    );

                    // Open the reduce loops over the reduced dims, keeping the loop ids.
                    let dims = self.reduce_dims(op_id);
                    let mut loop_ids = Vec::with_capacity(n_axes);
                    let mut loop_lens = Vec::with_capacity(n_axes);
                    for &dim in &dims[..n_axes] {
                        let len = self.insert_const_idx_before(loop_start, dim);
                        loop_lens.push(len);
                        loop_ids.push(self.insert_before(loop_start, Op::Loop { len }));
                    }

                    // x's view uses the reduce input's row-major strides for the
                    // non-reduced axes, plus the newly opened loops for the reduced
                    // axes with contiguous strides and zero padding.
                    let out_view = views.remove(&op_id).unwrap();
                    let rank = shape.len();
                    let non_reduce = rank - n_axes;
                    let mut strides = vec![1; rank];
                    let mut st = 1;
                    for a in (0..rank).rev() {
                        strides[a] = st;
                        st *= shape[a];
                    }
                    let zero = self.insert_const_idx_before(loop_start, 0u32);
                    let mut view = Vec::with_capacity(rank);
                    for e in 0..non_reduce {
                        let (idx, _st, lp, rp, len) = out_view[e];
                        let stride = self.insert_const_idx_before(loop_start, strides[e]);
                        view.push((idx, stride, lp, rp, len));
                    }
                    for (i, &lid) in loop_ids.iter().enumerate() {
                        let stride = self.insert_const_idx_before(loop_start, strides[non_reduce + i]);
                        view.push((lid, stride, zero, zero, loop_lens[i]));
                    }
                    views.insert(x, view);

                    // Accumulate just before the reduce op (which is inside the loop).
                    let load_acc = self.insert_before(op_id, Op::Load { src: acc, index: const_zero, layout: MemLayout::Scalar });
                    let bin_acc = self.insert_before(op_id, Op::Binary { x, y: load_acc, bop: rop });
                    self.insert_before(op_id, Op::Store { dst: acc, src: bin_acc, index: const_zero, layout: MemLayout::Scalar });

                    // Close the reduce loop.
                    for _ in 0..n_axes {
                        self.insert_before(op_id, Op::EndLoop);
                    }

                    // Replace the reduce with a load of the accumulator result.
                    self.ops[op_id].op = Op::Load { src: acc, index: const_zero, layout: MemLayout::Scalar };

                    // Upstream ops (walked later, in reverse) that depend on the loop
                    // ids/strides just inserted must land inside this loop, right
                    // after its infrastructure. The op immediately after the loop
                    // opener is `loop_start`; insertions before it end up after the
                    // loop's own inserted ops.
                    open_loops.push((loop_start, loop_start));*/
                }
                Op::Move { x, ref mop } => {
                    match mop.as_ref() {
                        MoveOp::Reshape { shape } => {
                            todo!()
                            // Reshape merges/splits contiguous dims, so axis indices don't
                            // align 1:1. Build a single flat index over the output view (all
                            // the arithmetic LoadView would do), then recover each input axis
                            // by successive div/mod against the input's contiguous strides.
                            /*let out_view = views[&op_id].clone();
                            let x_shape = self.shape(x);
                            let mut x_strides = vec![1; x_shape.len()];
                            let mut st = 1;
                            for a in (0..x_shape.len()).rev() {
                                x_strides[a] = st;
                                st *= x_shape[a];
                            }
                            let zero = self.insert_const_idx_before(anchor, 0u32);
                            let mut base = zero;
                            for &(idx, drift, _, _, _) in &out_view {
                                base = self.insert_before(anchor, Op::Mad { x: idx, y: drift, z: base });
                            }
                            let n = x_shape.len();
                            let mut view = Vec::with_capacity(n);
                            let mut q = base;
                            for a in 0..n {
                                let s = x_strides[a];
                                let s_id = self.insert_const_idx_before(anchor, s);
                                let idx_expr = if a == n - 1 {
                                    q
                                } else {
                                    let div = self.insert_before(anchor, Op::Binary { x: q, y: s_id, bop: BOp::Div });
                                    let rem = self.insert_before(anchor, Op::Binary { x: q, y: s_id, bop: BOp::Mod });
                                    q = rem;
                                    div
                                };
                                let len_id = self.insert_const_idx_before(anchor, x_shape[a]);
                                view.push((idx_expr, s_id, zero, zero, len_id));
                            }
                            views.insert(x, view);*/
                        }
                        MoveOp::Expand { shape } => {
                            todo!()
                            /*let x_shape = self.shape(x);
                            let shape = shape.clone();
                            let view = &views[&op_id];
                            let mut x_strides = vec![1; x_shape.len()];
                            let mut st = 1;
                            for a in (0..x_shape.len()).rev() {
                                x_strides[a] = st;
                                st *= x_shape[a];
                            }
                            let zero = self.insert_const_idx_before(anchor, 0);
                            // New leading axes are prepended broadcasts; the input axes
                            // align to the tail of the output shape.
                            let offset = shape.len() - x_shape.len();
                            let view = (0..x_shape.len())
                                .map(|a| {
                                    let idx = view[offset + a].0;
                                    let stride = if x_shape[a] != shape[offset + a] {
                                        zero
                                    } else {
                                        self.insert_const_idx_before(anchor, x_strides[a])
                                    };
                                    (idx, stride, view[offset + a].2, view[offset + a].3, view[offset + a].4)
                                })
                                .collect();
                            views.insert(x, view);*/
                        }
                        MoveOp::Permute { axes } => {
                            todo!()
                            /*let view = &views[&op_id];
                            let mut inv_axes = vec![0; axes.len()];
                            for (i, &a) in axes.iter().enumerate() {
                                inv_axes[a] = i;
                            }
                            let x_shape = self.shape(x);
                            let mut x_strides = vec![1; x_shape.len()];
                            let mut st = 1;
                            for a in (0..x_shape.len()).rev() {
                                x_strides[a] = st;
                                st *= x_shape[a];
                            }
                            let zero = self.insert_const_idx_before(anchor, 0);
                            // Input axis j's coordinate is output axis inv_axes[j]'s. Its
                            // stride is the input's contiguous stride, unless the output
                            // axis is broadcast (stride 0), in which case it stays 0.
                            let view = (0..x_shape.len())
                                .map(|j| {
                                    let (idx, os, lp, rp, len) = view[inv_axes[j]];
                                    let stride = if matches!(self.ops[os].op, Op::Const(c) if c.as_dim() == Some(0)) {
                                        zero
                                    } else {
                                        self.insert_const_idx_before(anchor, x_strides[j])
                                    };
                                    (idx, stride, lp, rp, len)
                                })
                                .collect();
                            views.insert(x, view);*/
                        }
                        MoveOp::Flip { axes } => {
                            todo!()
                            /*let axes = axes.clone();
                            let x_shape = self.shape(x);
                            let view = &views[&op_id];
                            let zero = self.insert_const_idx_before(anchor, 0u32);
                            let one = self.insert_const_idx_before(anchor, 1u32);
                            let view = (0..x_shape.len())
                                .map(|a| {
                                    let (idx, stride, lp_id, rp_id, len_id) = view[a];
                                    if axes.contains(&(a as UAxis)) {
                                        // Reverse the axis: the input coordinate is
                                        // `extent - 1 - out_idx`. The extent is the
                                        // coordinate range of the padded axis.
                                        let len_m1 = self.insert_before(anchor, Op::Binary { x: len_id, y: one, bop: BOp::Sub });
                                        let idx = self.insert_before(anchor, Op::Binary { x: len_m1, y: idx, bop: BOp::Sub });
                                        // Padding swaps sides under a flip.
                                        (idx, stride, rp_id, lp_id, len_id)
                                    } else {
                                        (idx, stride, lp_id, rp_id, len_id)
                                    }
                                })
                                .collect();
                            views.insert(x, view);*/
                        }
                        MoveOp::Pad { padding, .. } => {
                            todo!()
                            /*let x_shape = self.shape(x);
                            let padding = padding.clone();
                            let view = &views[&op_id];
                            let mut x_strides = vec![1; x_shape.len()];
                            let mut st = 1;
                            for a in (0..x_shape.len()).rev() {
                                x_strides[a] = st;
                                st *= x_shape[a];
                            }
                            let zero = self.insert_const_idx_before(anchor, 0u32);
                            let view = (0..x_shape.len())
                                .map(|a| {
                                    let idx = view[a].0;
                                    let lp = padding[a].0;
                                    let rp = padding[a].1;
                                    let stride = self.insert_const_idx_before(anchor, x_strides[a]);
                                    // Negative left padding is a slice offset:
                                    // input index = output index - lp.
                                    let idx = if lp < 0 {
                                        let off = self.insert_const_idx_before(anchor, (-lp) as u64);
                                        self.insert_before(anchor, Op::Binary { x: idx, y: off, bop: BOp::Add })
                                    } else {
                                        idx
                                    };
                                    // The input-view index ranges over the padded
                                    // coordinates (length x_shape + lp + rp), which is
                                    // this axis' extent for pad-condition bounds -- NOT
                                    // the consumer's view extent (that is the slice
                                    // output length when the pad is a slice).
                                    let len_id =
                                        self.insert_const_idx_before(anchor, ((x_shape[a] as i64) + lp + rp).max(0) as u64);
                                    let lp_id = if lp > 0 {
                                        self.insert_const_idx_before(anchor, lp as u64)
                                    } else {
                                        zero
                                    };
                                    let rp_id = if rp > 0 {
                                        self.insert_const_idx_before(anchor, rp as u64)
                                    } else {
                                        zero
                                    };
                                    (idx, stride, lp_id, rp_id, len_id)
                                })
                                .collect();
                            views.insert(x, view);*/
                        }
                        &MoveOp::Narrow { .. } => {
                            todo!()
                        }
                    }
                    self.remap(op_id, x);
                    self.remove_op(op_id);
                }
                Op::Cast { x, .. } | Op::Unary { x, .. } => {
                    views.insert(x, views[&op_id].clone());
                }
                Op::Binary { x, y, .. } => {
                    views.insert(x, views[&op_id].clone());
                    views.insert(y, views[&op_id].clone());
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

        // Put defines in the beginning
        let head = self.head;
        let mut op_id = head;
        let mut first_mut_global = head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Op::Param { kind, .. } = self.ops[op_id].op {
                match kind {
                    ParamKind::Variable | ParamKind::Global => self.move_op_before(op_id, first_mut_global),
                    ParamKind::GlobalMut => {
                        self.move_op_before(op_id, head);
                        if first_mut_global == head {
                            first_mut_global = op_id;
                        }
                    }
                }
            }
            op_id = next;
        }

        // Verify the relative order of global defines is unchanged by linearize
        // (read-only defines first, then writable ones, both in original order).
        debug_assert!({
            let mut defines = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if let Op::Param { dtype, kind } = self.ops[op_id].op {
                    defines.push((dtype, kind));
                }
                op_id = self.next_op(op_id);
            }
            let mut expected = global_defines.clone();
            expected.sort_by_key(|(_, kind)| *kind == ParamKind::GlobalMut);
            if defines != expected {
                self.debug();
                panic!(
                    "linearize: global define order changed:\n  original = {global_defines:?}\n  expected = {expected:?}\n  final = {defines:?}"
                );
            }
            true
        });

        self.verify();
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
                    Op::Reduce { n_axes, .. } => n_reduce_axes += n_axes,
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
