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
    kernel::{IdxScope, Kernel, MemLayout, MemScope, MoveOp, Op, OpId},
    shape,
};

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

        // For each op, shape and strides
        let mut views: Map<OpId, Vec<(OpId, OpId)>> = Map::default();

        let start = self.head;
        let mut op_id = self.tail;
        while !op_id.is_null() {
            let prev = self.prev_op(op_id);
            match self.ops[op_id].op {
                Op::LoadView(ref x) => {
                    let dtype = x.0;
                    let len = x.1.original_numel();
                    let view = views.remove(&op_id).unwrap();
                    let mut index = self.insert_const_idx_before(start, 0);
                    for (idx, st) in view {
                        index = self.insert_before(start, Op::Mad { x: idx, y: st, z: index });
                    }
                    let src = self.insert_before(start, Op::Define { dtype, scope: MemScope::Global, ro: true, len });
                    self.ops[op_id].op = Op::Load { src, index, layout: MemLayout::Scalar };
                }
                Op::ConstView(ref x) => {
                    self.ops[op_id].op = Op::Const(x.0);
                }
                Op::Reduce { x, rop, n_axes } => todo!(),
                Op::Move { x, ref mop } => {
                    match mop.as_ref() {
                        MoveOp::Reshape { shape } => todo!(),
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
                                    (idx, stride)
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
                                    (view[i].0, stride)
                                })
                                .collect();
                            views.insert(x, view);
                        }
                        MoveOp::Pad { padding, shape } => todo!(),
                    }
                    self.remap(op_id, x);
                    self.remove_op(op_id);
                }
                Op::StoreView { src, dtype } => {
                    let shape = self.shape_of(src);
                    let len = shape.iter().product();
                    let mut view = Vec::new();
                    let mut st = 1;
                    for axis in (0..shape.len() as u32).rev() {
                        let len = shape[axis as usize];
                        let idx = self.insert_before(start, Op::Index { len, axis, scope: IdxScope::Group });
                        let st_id = self.insert_const_idx_before(start, st);
                        view.push((idx, st_id));
                        st *= len;
                    }
                    view.reverse();
                    let mut index = self.insert_const_idx_before(start, 0);
                    for &(idx, st) in &view {
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
