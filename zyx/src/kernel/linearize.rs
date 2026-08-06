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
    shape,
};

/// Extract the value of an index constant op.
fn pad_value(k: &Kernel, id: OpId) -> u64 {
    let Op::Const(c) = k.ops[id].op else {
        unreachable!("pad constant expected, got {:?}", k.ops[id].op)
    };
    c.as_dim().expect("pad constant must be a non-negative dim")
}

/// Extract the axis length of an `Op::Index`.
fn index_len(k: &Kernel, idx: OpId) -> u64 {
    match k.ops[idx].op {
        Op::Index { len, .. } => len,
        _ => unreachable!("pad condition needs an Index length, got {:?}", k.ops[idx].op),
    }
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
                if matches!(self.ops[op_id].op, Op::Store { .. } | Op::StoreView { .. }) {
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
                    panic!("unfold_movement_ops: dead code detected at op {op_id}");
                }
                op_id = self.next_op(op_id);
            }
            true
        });

        self.debug();

        // For each op, shape and strides: (index, stride, left pad, right pad)
        let mut views: Map<OpId, Vec<(OpId, OpId, OpId, OpId)>> = Map::default();

        // Reused group index per axis, so every store of a result shares one index.
        let mut group_indices: Map<u32, OpId> = Map::default();

        let start = self.head;
        let mut op_id = self.tail;
        while !op_id.is_null() {
            let prev = self.prev_op(op_id);
            match self.ops[op_id].op {
                Op::LoadView(ref x) => {
                    let dtype = x.0;
                    let len = x.1.original_numel();
                    let view = views.remove(&op_id).unwrap();
                    let zero = self.insert_const_idx_before(start, 0u32);
                    let one = self.insert_const_idx_before(start, 1u32);
                    // Padding condition: valid where index is within the source extent.
                    // index = sum over axes of (idx - lp) * stride
                    // pc = and over padded axes of idx > lp-1 && idx < len-rp
                    let mut index = zero;
                    let mut pc = self.insert_before(start, Op::Const(Constant::Bool(true)));
                    let mut has_pad = false;
                    for &(idx, st, lp_id, rp_id) in &view {
                        let lp = pad_value(self, lp_id);
                        let rp = pad_value(self, rp_id);
                        let src_idx = if lp == 0 {
                            idx
                        } else {
                            self.insert_before(start, Op::Binary { x: idx, y: lp_id, bop: BOp::Sub })
                        };
                        index = self.insert_before(start, Op::Mad { x: src_idx, y: st, z: index });
                        if lp > 0 || rp > 0 {
                            has_pad = true;
                            if lp > 0 {
                                let lp_m1 = self.insert_const_idx_before(start, lp - 1);
                                let t = self.insert_before(start, Op::Binary { x: idx, y: lp_m1, bop: BOp::Cmpgt });
                                pc = self.insert_before(start, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                            if rp > 0 {
                                let axis_len = index_len(self, idx);
                                let len_mr = self.insert_const_idx_before(start, axis_len - rp);
                                let t = self.insert_before(start, Op::Binary { x: idx, y: len_mr, bop: BOp::Cmplt });
                                pc = self.insert_before(start, Op::Binary { x: t, y: pc, bop: BOp::And });
                            }
                        }
                    }
                    let src = self.insert_before(start, Op::Define { dtype, scope: MemScope::Global, ro: true, len });
                    if has_pad {
                        // Zero the offset where the padding condition fails, so the load
                        // always reads in-bounds, then zero the loaded value itself.
                        let pcu = self.insert_before(start, Op::Cast { x: pc, dtype: IDX_T });
                        let offset = self.insert_before(start, Op::Binary { x: pcu, y: index, bop: BOp::Mul });
                        let z = self.insert_before(start, Op::Load { src, index: offset, layout: MemLayout::Scalar });
                        let pcd = self.insert_before(start, Op::Cast { x: pc, dtype });
                        self.ops[op_id].op = Op::Binary { x: pcd, y: z, bop: BOp::Mul };
                    } else {
                        self.ops[op_id].op = Op::Load { src, index, layout: MemLayout::Scalar };
                    }
                }
                Op::ConstView(ref x) => {
                    self.ops[op_id].op = Op::Const(x.0);
                }
                Op::Reduce { x, rop, n_axes } => {
                    // Collect all transitive dependencies of the reduce input and the
                    // accumulator dtype. The loop that wraps the reduction is opened at
                    // the soonest dependency that appears in the graph.
                    let mut reduce_loop_ops_set = Set::default();
                    let mut params = vec![x];
                    let mut acc_dtype = None;
                    while let Some(param) = params.pop() {
                        if reduce_loop_ops_set.insert(param) {
                            params.extend(self.at(param).parameters());
                            if acc_dtype.is_none() {
                                match self.at(param) {
                                    &Op::Define { dtype, .. } | &Op::Cast { dtype, .. } => acc_dtype = Some(dtype),
                                    Op::ConstView(v) => acc_dtype = Some(v.0.dtype()),
                                    Op::LoadView(v) => acc_dtype = Some(v.0),
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
                    for &dim in &dims[..n_axes] {
                        let len = self.insert_const_idx_before(loop_start, dim);
                        loop_ids.push(self.insert_before(loop_start, Op::Loop { len }));
                    }

                    // x's view uses the reduce input's row-major strides for the
                    // non-reduced axes, plus the newly opened loops for the reduced
                    // axes with contiguous strides and zero padding.
                    let out_view = views.remove(&op_id).unwrap();
                    let rank = self.shape_of(x).len();
                    let non_reduce = rank - n_axes;
                    let mut strides = vec![1; rank];
                    let mut st = 1;
                    for a in (0..rank).rev() {
                        strides[a] = st;
                        st *= self.shape_of(x)[a];
                    }
                    let zero = self.insert_const_idx_before(loop_start, 0u32);
                    let mut view = Vec::with_capacity(rank);
                    for e in 0..non_reduce {
                        let (idx, _st, lp, rp) = out_view[e];
                        let stride = self.insert_const_idx_before(loop_start, strides[e]);
                        view.push((idx, stride, lp, rp));
                    }
                    for (i, &lid) in loop_ids.iter().enumerate() {
                        let stride = self.insert_const_idx_before(loop_start, strides[non_reduce + i]);
                        view.push((lid, stride, zero, zero));
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
                            let zero = self.insert_const_idx_before(start, 0u32);
                            let mut base = zero;
                            for &(idx, drift, _, _) in &out_view {
                                base = self.insert_before(start, Op::Mad { x: idx, y: drift, z: base });
                            }
                            let n = x_shape.len();
                            let mut view = Vec::with_capacity(n);
                            let mut q = base;
                            for a in 0..n {
                                let s = x_strides[a];
                                let s_id = self.insert_const_idx_before(start, s);
                                let idx_expr = if a == n - 1 {
                                    q
                                } else {
                                    let div = self.insert_before(start, Op::Binary { x: q, y: s_id, bop: BOp::Div });
                                    let rem = self.insert_before(start, Op::Binary { x: q, y: s_id, bop: BOp::Mod });
                                    q = rem;
                                    div
                                };
                                view.push((idx_expr, s_id, zero, zero));
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
                            let zero = self.insert_const_idx_before(start, 0);
                            let view = (0..x_shape.len())
                                .map(|a| {
                                    let idx = view[a].0;
                                    let stride = if x_shape[a] != shape[a] {
                                        zero
                                    } else {
                                        self.insert_const_idx_before(start, x_strides[a])
                                    };
                                    (idx, stride, view[a].2, view[a].3)
                                })
                                .collect();
                            views.insert(x, view);
                        }
                        MoveOp::Permute { axes, shape } => {
                            let view = &views[&op_id];
                            let mut inv_axes = vec![0; axes.len()];
                            for (i, &a) in axes.iter().enumerate() {
                                inv_axes[a as usize] = i;
                            }
                            let x_shape = self.shape_of(x);
                            let mut x_strides = vec![1; x_shape.len()];
                            let mut st = 1;
                            for a in (0..x_shape.len()).rev() {
                                x_strides[a] = st;
                                st *= x_shape[a];
                            }
                            let view = (0..x_shape.len())
                                .map(|a| {
                                    let i = inv_axes[a];
                                    let stride = self.insert_const_idx_before(start, x_strides[a]);
                                    (view[i].0, stride, view[i].2, view[i].3)
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
                            let zero = self.insert_const_idx_before(start, 0u32);
                            let view = (0..x_shape.len())
                                .map(|a| {
                                    let idx = view[a].0;
                                    let lp = padding[a].0;
                                    let rp = padding[a].1;
                                    let stride = self.insert_const_idx_before(start, x_strides[a]);
                                    let lp_id = if lp > 0 {
                                        self.insert_const_idx_before(start, lp as u64)
                                    } else {
                                        zero
                                    };
                                    let rp_id = if rp > 0 {
                                        self.insert_const_idx_before(start, rp as u64)
                                    } else {
                                        zero
                                    };
                                    (idx, stride, lp_id, rp_id)
                                })
                                .collect();
                            views.insert(x, view);
                        }
                    }
                    self.remap(op_id, x);
                    self.remove_op(op_id);
                }
                Op::StoreView { src, dtype } => {
                    let shape = self.shape_of(src);
                    let len = shape.iter().product();
                    let mut view = Vec::new();
                    let zero = self.insert_const_idx_before(start, 0u32);
                    let mut st = 1;
                    for axis in (0..shape.len() as u32).rev() {
                        let len = shape[axis as usize];
                        let idx = match group_indices.get(&axis) {
                            Some(&id) => id,
                            None => {
                                let id = self.insert_before(start, Op::Index { len, axis, scope: IdxScope::Group });
                                group_indices.insert(axis, id);
                                id
                            }
                        };
                        let st_id = self.insert_const_idx_before(start, st);
                        view.push((idx, st_id, zero, zero));
                        st *= len;
                    }
                    view.reverse();
                    let mut index = self.insert_const_idx_before(start, 0);
                    for &(idx, st, _, _) in &view {
                        index = self.insert_before(start, Op::Mad { x: idx, y: st, z: index });
                    }
                    let dst = self.insert_before(start, Op::Define { dtype, scope: MemScope::Global, ro: false, len });
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
                _ => break,
                /*ref op => {
                    self.debug();
                    unreachable!("{op:?}");
                }*/
            }
            op_id = prev;
        }

        // Reverse the order of globals
        let mut op_id = self.tail;
        let head = self.head;
        while op_id != head {
            let prev = self.prev_op(op_id);
            if let Op::Define { dtype, scope, ro, len } = self.ops[op_id].op {
                self.move_op_before(op_id, head);
            }
            op_id = prev;
        }
        self.debug();

        self.instruction_schedule();
        //panic!();

        self.verify();
    }
}
