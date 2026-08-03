// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Rangeify movement operations.
//!
//! Reimplements unfold_movement_ops using tinygrad's rangeify approach,
//! without the View abstraction for movement op propagation.
//! Movement ops are applied directly to axis indices, and
//! LoadView/StoreView/ConstView are converted to Load/Store/Const in a single pass.

use std::collections::BTreeMap;

use crate::{
    Set,
    dtype::Constant,
    kernel::{BOp, IdxScope, Kernel, MemLayout, MemScope, MoveOp, Op, OpId},
    shape::{Dim, UAxis},
};

impl Kernel {
    /// Unfold movement operations into index-based operations using tinygrad's rangeify approach.
    ///
    /// Movement ops (Reshape, Expand, Permute, Pad) are applied directly to axis indices,
    /// and LoadView/StoreView/ConstView are converted to Load/Store/Const in a single pass.
    pub fn unfold_movement_ops(&mut self) {
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

        let shape = self.shape();
        let mut axis = shape.len() as u32;
        for len in shape.into_iter().rev() {
            axis -= 1;
            self.insert_before(self.head, Op::Index { len, axis, scope: IdxScope::Group });
        }

        let mut axes: BTreeMap<u32, OpId> = BTreeMap::default();
        let start = self.head;
        let mut op_id = self.head;

        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match self.ops[op_id].op {
                Op::Index { axis, .. } => {
                    axes.insert(axis, op_id);
                }
                Op::Loop { .. } => {
                    axes.insert(axes.last_key_value().map_or(0, |x| x.0 + 1), op_id);
                }
                Op::EndLoop => {
                    axes.pop_last();
                }
                Op::Move { x, ref mop } => {
                    let mop = mop.clone();
                    self.apply_movement_op(&mop, &mut axes);
                    self.remap(op_id, x);
                    self.remove_op(op_id);
                }
                Op::LoadView(ref x) => {
                    let shape = x.1.shape();
                    self.unfold_load_view(op_id, x.0, &shape, &axes, start);
                }
                Op::StoreView { src, dtype } => {
                    self.unfold_store_view(op_id, src, dtype, &axes, start);
                }
                Op::ConstView(ref x) => {
                    let shape = x.1.shape();
                    self.unfold_const_view(op_id, x.0, &shape, &axes);
                }
                _ => {}
            }
            op_id = next;
        }

        self.verify();
        self.unfold_reduces();
    }

    fn apply_movement_op(&mut self, mop: &MoveOp, axes: &mut BTreeMap<u32, OpId>) {
        match mop {
            MoveOp::Reshape { shape } => self.apply_reshape(shape, axes),
            MoveOp::Expand { shape } => self.apply_expand(shape, axes),
            MoveOp::Permute { axes: perm_axes, .. } => self.apply_permute(perm_axes, axes),
            MoveOp::Pad { padding, shape } => self.apply_pad(padding, shape, axes),
        }
    }

    fn apply_reshape(&mut self, shape: &[Dim], axes: &mut BTreeMap<u32, OpId>) {
        if axes.is_empty() {
            return;
        }
        let old_axes: Vec<(u32, OpId)> = axes.iter().map(|(&k, &v)| (k, v)).collect();
        axes.clear();

        // Compute strides from shape: stride[i] = prod(shape[i+1..])
        let mut strides: Vec<Dim> = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1].saturating_mul(shape[i + 1]);
        }

        // Build flat index: sum(axis_i * stride_i)
        let mut flat_idx = self.insert_const_idx_before(self.head, 0u64);
        for (i, &(_, ax_id)) in old_axes.iter().enumerate() {
            if i >= shape.len() {
                break;
            }
            let stride_c = self.insert_const_idx_before(self.head, strides[i]);
            let scaled = self.insert_after(ax_id, Op::Binary { x: ax_id, y: stride_c, bop: BOp::Mul });
            flat_idx = self.insert_after(scaled, Op::Binary { x: flat_idx, y: scaled, bop: BOp::Add });
        }

        // Decompose flat index into per-axis coordinates using div/mod
        let mut remaining = flat_idx;
        for (i, &(axis_key, _)) in old_axes.iter().enumerate() {
            if i >= shape.len() {
                break;
            }
            let dim = shape[i];
            if dim == 1 {
                axes.insert(axis_key, self.insert_const_idx_before(self.head, 0u64));
                continue;
            }
            let dim_c = self.insert_const_idx_before(self.head, dim);
            let coord = self.insert_after(remaining, Op::Binary { x: remaining, y: dim_c, bop: BOp::Mod });
            remaining = self.insert_after(coord, Op::Binary { x: remaining, y: dim_c, bop: BOp::Div });
            axes.insert(axis_key, coord);
        }
    }

    fn apply_expand(&mut self, shape: &[Dim], axes: &mut BTreeMap<u32, OpId>) {
        let n_add = shape.len().saturating_sub(axes.len());
        for i in 0..n_add {
            let axis_idx = i as u32;
            let one_idx = self.insert_const_idx_before(self.head, 1u64);
            axes.insert(axis_idx, one_idx);
        }
    }

    fn apply_permute(&mut self, perm_axes: &[UAxis], axes: &mut BTreeMap<u32, OpId>) {
        let old_axes: Vec<OpId> = axes.values().copied().collect();
        axes.clear();
        for (new_axis, &old_axis) in perm_axes.iter().enumerate() {
            axes.insert(new_axis as u32, old_axes[old_axis as usize]);
        }
    }

    fn apply_pad(&mut self, padding: &[(i64, i64)], shape: &[Dim], axes: &mut BTreeMap<u32, OpId>) {
        for (i, &(left, right)) in padding.iter().enumerate() {
            if left == 0 && right == 0 {
                continue;
            }
            let ax_id = match axes.get(&(i as u32)) {
                Some(&id) => id,
                None => continue,
            };
            let zero = self.insert_const_idx_before(self.head, 0i64);

            if left > 0 {
                let lp = self.insert_const_idx_before(self.head, (left - 1) as u64);
                let cond = self.insert_after(ax_id, Op::Binary { x: ax_id, y: lp, bop: BOp::Cmpgt });
                let offset = self.insert_after(ax_id, Op::Binary { x: ax_id, y: lp, bop: BOp::Sub });
                axes.insert(i as u32, self.cond_select(ax_id, cond, offset, zero));
            }
            if right > 0 {
                let rp = self.insert_const_idx_before(self.head, (shape[i] as i64 - right) as u64);
                let cond = self.insert_after(ax_id, Op::Binary { x: ax_id, y: rp, bop: BOp::Cmplt });
                let offset = self.insert_after(ax_id, Op::Binary { x: ax_id, y: rp, bop: BOp::Sub });
                axes.insert(i as u32, self.cond_select(ax_id, cond, offset, zero));
            }
        }
    }

    fn cond_select(&mut self, after: OpId, cond: OpId, then_val: OpId, else_val: OpId) -> OpId {
        let sel = self.insert_after(after, Op::Binary { x: cond, y: else_val, bop: BOp::Mul });
        let not_cond_val = self.insert_const_idx_before(sel, true);
        let not_cond = self.insert_after(sel, Op::Binary { x: cond, y: not_cond_val, bop: BOp::BitXor });
        let then_part = self.insert_after(not_cond, Op::Binary { x: not_cond, y: then_val, bop: BOp::Mul });
        self.insert_after(then_part, Op::Binary { x: then_part, y: sel, bop: BOp::Add })
    }

    fn unfold_load_view(&mut self, op_id: OpId, dtype: crate::DType, shape: &[Dim], axes: &BTreeMap<u32, OpId>, start: OpId) {
        let axes_vec: Vec<OpId> = axes.values().copied().collect();
        let mut strides = vec![1u64; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1].saturating_mul(shape[i + 1]);
        }

        let mut offset = self.insert_const_idx_before(op_id, 0u64);
        for (i, &ax_id) in axes_vec.iter().enumerate() {
            if i >= shape.len() || shape[i] == 1 {
                continue;
            }
            let stride_c = self.insert_const_idx_before(op_id, strides[i]);
            let scaled = self.insert_after(ax_id, Op::Binary { x: ax_id, y: stride_c, bop: BOp::Mul });
            offset = self.insert_after(scaled, Op::Binary { x: offset, y: scaled, bop: BOp::Add });
        }

        let src = self.insert_before(start, Op::Define { dtype, scope: MemScope::Global, ro: true, len: shape.iter().product() });
        self.ops[op_id].op = Op::Load { src, index: offset, layout: MemLayout::Scalar };
    }

    fn unfold_store_view(&mut self, op_id: OpId, src: OpId, dtype: crate::DType, axes: &BTreeMap<u32, OpId>, start: OpId) {
        let mut st = 1u64;
        let mut strides = Vec::new();
        for (_, &ax_id) in axes.iter().rev() {
            match self.ops[ax_id].op {
                Op::Index { len, .. } => {
                    strides.push((len, st, ax_id));
                    st *= len;
                }
                Op::Loop { len: len_id, .. } => {
                    let len = self.loop_len_dim(len_id);
                    strides.push((len, st, ax_id));
                    st *= len;
                }
                _ => {}
            }
        }

        let mut index = self.insert_const_idx_before(op_id, 0u64);
        let mut len = 1u64;
        for (dim, st, ax_id) in strides.into_iter().rev() {
            let y = self.insert_const_idx_before(op_id, st as u64);
            index = self.insert_after(ax_id, Op::Mad { x: ax_id, y, z: index });
            len *= dim;
        }

        let dst = self.insert_before(start, Op::Define { dtype, scope: MemScope::Global, ro: false, len });
        self.ops[op_id].op = Op::Store { dst, x: src, index, layout: MemLayout::Scalar };
    }

    fn unfold_const_view(&mut self, op_id: OpId, value: Constant, shape: &[Dim], axes: &BTreeMap<u32, OpId>) {
        let axes_vec: Vec<OpId> = axes.values().copied().collect();
        let mut strides = vec![1u64; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1].saturating_mul(shape[i + 1]);
        }

        let mut offset = self.insert_const_idx_before(op_id, 0u64);
        for (i, &ax_id) in axes_vec.iter().enumerate() {
            if i >= shape.len() || shape[i] == 1 {
                continue;
            }
            let stride_c = self.insert_const_idx_before(op_id, strides[i]);
            let scaled = self.insert_after(ax_id, Op::Binary { x: ax_id, y: stride_c, bop: BOp::Mul });
            offset = self.insert_after(scaled, Op::Binary { x: offset, y: scaled, bop: BOp::Add });
        }

        let z = self.insert_after(op_id, Op::Const(value));
        self.ops[op_id].op = Op::Binary { x: z, y: offset, bop: BOp::Add };
    }
}
