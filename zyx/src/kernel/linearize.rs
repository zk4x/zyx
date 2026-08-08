// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Rangeify movement operations.
//!
//! Reimplements unfold_movement_ops using tinygrad's rangeify approach,
//! without the View abstraction for movement op propagation.
//! Movement ops are applied directly to axis indices, and
//! LoadView/StoreView/ConstView are converted to Load/Store/Const in a single pass.

#![allow(unused)]

use crate::{
    Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, IdxScope, Kernel, MemLayout, MemScope, MoveOp, Op, OpId},
    shape::{self, Dim, UAxis},
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
    pub fn linearize(&mut self) {
        let has_gidx = self.ops.values().any(|n| matches!(n.op, Op::Index { scope: IdxScope::Group, .. }));
        let has_view_moves = self.ops.values().any(|n| matches!(n.op, Op::LoadView(_) | Op::StoreView { .. } | Op::Move { .. }));

        match (has_gidx, has_view_moves) {
            (true, false) => return,
            (true, true) => {
                panic!("unfold_movement_ops: cannot have both explicit gidx and LoadView/StoreView/Move ops");
            }
            (false, true) => {}
            (false, false) => return,
        }

        debug_assert!({
            let mut live: Set<OpId> = Set::default();
            let mut stack: Vec<OpId> = Vec::new();
            let mut op_id = self.head;
            while !op_id.is_null() {
                if matches!(self.ops[op_id].op, Op::Define { .. } | Op::Store { .. } | Op::StoreView { .. }) {
                    stack.push(op_id);
                }
                op_id = self.next_op(op_id);
            }
            while let Some(id) = stack.pop() {
                if live.insert(id) {
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
        });

        // For each op, shape and strides: (index, stride, left pad, right pad, axis length)
        let mut views: Map<OpId, Vec<(OpId, OpId, OpId, OpId, OpId)>> = Map::default();

        // Reused group index per axis, so every store of a result shares one index.
        let mut group_indices: Map<u32, OpId> = Map::default();

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
                Op::Define { .. } => {}
                Op::LoadView(ref x) => {
                    let src = x.0;
                    let dtype = x.1;
                    let view = views.remove(&op_id).unwrap();
                    let zero = self.insert_const_idx_before(anchor, 0u32);
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
                Op::Reduce { x, rop, n_axes } => {
                    // Collect all transitive dependencies of the reduce input and the
                    // accumulator dtype. The loop that wraps the reduction is opened at
                    // the soonest dependency that appears in the graph.
                    let mut reduce_loop_ops_set = Set::default();
                    let shape = self.shape_of(x);
                    let mut params = vec![x];
                    let mut acc_dtype = None;
                    while let Some(param) = params.pop() {
                        if reduce_loop_ops_set.insert(param) {
                            params.extend(self.at(param).parameters());
                            if acc_dtype.is_none() {
                                match self.at(param) {
                                    &Op::Define { dtype, .. } | &Op::Cast { dtype, .. } => acc_dtype = Some(dtype),
                                    Op::Const(v) => acc_dtype = Some(v.dtype()),
                                    Op::LoadView(v) => acc_dtype = Some(v.1),
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
                    let acc = self
                        .insert_before(loop_start, Op::Define { dtype: acc_dtype, scope: MemScope::Register, ro: false, len: 1 });
                    self.insert_before(
                        loop_start,
                        Op::Store { dst: acc, x: acc_init_id, index: const_zero, layout: MemLayout::Scalar },
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
                    self.insert_before(op_id, Op::Store { dst: acc, x: bin_acc, index: const_zero, layout: MemLayout::Scalar });

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
                    open_loops.push((loop_start, loop_start));
                }
                Op::Move { x, ref mop } => {
                    match mop.as_ref() {
                        MoveOp::Reshape { shape } => {
                            // Reshape merges/splits contiguous dims, so axis indices don't
                            // align 1:1. Build a single flat index over the output view (all
                            // the arithmetic LoadView would do), then recover each input axis
                            // by successive div/mod against the input's contiguous strides.
                            let out_view = views[&op_id].clone();
                            let x_shape = self.shape_of(x);
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
                            views.insert(x, view);
                        }
                        MoveOp::Expand { shape } => {
                            let x_shape = self.shape_of(x);
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
                            views.insert(x, view);
                        }
                        MoveOp::Permute { axes, shape } => {
                            let view = &views[&op_id];
                            let mut inv_axes = vec![0; axes.len()];
                            for (i, &a) in axes.iter().enumerate() {
                                inv_axes[a] = i;
                            }
                            let x_shape = self.shape_of(x);
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
                            views.insert(x, view);
                        }
                        MoveOp::Flip { axes } => {
                            let axes = axes.clone();
                            let x_shape = self.shape_of(x);
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
                            views.insert(x, view);
                        }
                        MoveOp::Pad { padding, .. } => {
                            let x_shape = self.shape_of(x);
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
                            views.insert(x, view);
                        }
                    }
                    self.remap(op_id, x);
                    self.remove_op(op_id);
                }
                Op::StoreView { dst, src, dtype } => {
                    let shape = self.shape_of(src);
                    let mut view = Vec::new();
                    let zero = self.insert_const_idx_before(start, 0u32);
                    let mut st = 1;
                    for axis in (0..shape.len() as u32).rev() {
                        let len = shape[axis as usize];
                        let len_id = self.insert_const_idx_before(start, len);
                        let idx = match group_indices.get(&axis) {
                            Some(&id) => id,
                            None => {
                                let id = self.insert_before(start, Op::Index { len: len_id, axis, scope: IdxScope::Group });
                                group_indices.insert(axis, id);
                                id
                            }
                        };
                        let st_id = self.insert_const_idx_before(start, st);
                        view.push((idx, st_id, zero, zero, len_id));
                        st *= len;
                    }
                    view.reverse();
                    let mut index = self.insert_const_idx_before(start, 0);
                    for &(idx, st, _, _, _) in &view {
                        index = self.insert_before(start, Op::Mad { x: idx, y: st, z: index });
                    }
                    self.ops[op_id].op = Op::Store { dst, x: src, index, layout: MemLayout::Scalar };
                    views.insert(src, view);
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
            if let Op::Define { ro, scope: MemScope::Global, .. } = self.ops[op_id].op {
                if ro {
                    self.move_op_before(op_id, first_mut_global);
                } else {
                    self.move_op_before(op_id, head);
                    if first_mut_global == head {
                        first_mut_global = op_id;
                    }
                }
            }
            op_id = next;
        }

        self.verify();
    }

    pub(crate) fn reduce_dims(&self, op_id: OpId) -> Vec<Dim> {
        let mut params = vec![op_id];
        let mut n_reduce_axes = 0;
        let mut visited = Set::default();
        while let Some(param) = params.pop() {
            if visited.insert(param) {
                match self.at(param) {
                    Op::Const(_) => return vec![1],
                    Op::LoadView(x) => {
                        return x.2[x.2.len() - n_reduce_axes..].into();
                    }
                    Op::Reduce { n_axes, .. } => n_reduce_axes += n_axes,
                    Op::Move { mop, .. } => match mop.as_ref() {
                        MoveOp::Reshape { shape, .. }
                        | MoveOp::Expand { shape }
                        | MoveOp::Permute { shape, .. }
                        | MoveOp::Pad { shape, .. } => {
                            return shape[shape.len() - n_reduce_axes..].into();
                        }
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
