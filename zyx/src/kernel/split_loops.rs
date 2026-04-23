// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use super::autotune::Optimization;
use crate::{
    backend::DeviceInfo,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, Scope},
};

impl Kernel {
    pub fn opt_split_global_to_local(&self, dev_info: &DeviceInfo) -> (Optimization, usize) {
        let max_threads = dev_info.max_local_threads;
        if self.ops.values().any(|node| matches!(node.op, Op::EndIf)) {
            let factors = Vec::new();
            return (Optimization::SplitLoop { factors }, 0);
        }
        let max_threads = dev_info.max_local_threads/self.ops.values().filter_map(|op| if let Op::Index { len, scope: Scope::Local, .. } = op.op { Some(len) } else { None }).product::<u64>();
        let mut op_id = self.head;
        let mut factors = Vec::new();
        let mut seen_axes = crate::Map::default();
        while !op_id.is_null() {
            if let Op::Index { len, scope, axis } = self.ops[op_id].op {
                let mut l_factors: Vec<u64> = vec![64, 32, 16, 8, 4, 2];
                if scope == Scope::Global {
                    let max_per_axis = dev_info.max_local_work_dims[axis as usize] as u64;
                    l_factors.retain(|&f| len.is_multiple_of(f) && f <= max_threads && f <= max_per_axis);
                    for &f in &l_factors {
                        factors.push((op_id, f));
                    }
                    seen_axes.insert(axis, op_id);
                }
                if scope == Scope::Local {
                    if let Some(global_id) = seen_axes.get(&axis) {
                        factors.retain(|(op_id, _)| global_id != op_id);
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        let n_configs = factors.len();
        (Optimization::SplitGlobalToLocal { factors }, n_configs)
    }

    pub fn opt_split_loop(&self) -> (Optimization, usize) {
        let candidates = vec![8, 16, 4, 2];
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Loop { len } = self.ops[op_id].op {
                if len >= 16 {
                    for &factor in &candidates {
                        if len.is_multiple_of(factor as u64) {
                            factors.push((op_id, factor as u64));
                        }
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        let n_configs = factors.len();
        (Optimization::SplitLoop { factors }, n_configs)
    }

    /// Splits dim (index or loop) into multiple indices or loops
    /// Returns the `OpId`s of the created split operations in the order they were provided
    pub fn split_dim(&mut self, dim_id: OpId, mut splits: Vec<Op>) -> Vec<OpId> {
        let is_loop = matches!(self.ops[dim_id].op, Op::Loop { .. });

        #[cfg(debug_assertions)]
        {
            let mut dim = 1;
            for op in splits.iter() {
                match op {
                    Op::Loop { len, .. } | Op::Index { len, .. } => dim *= len,
                    _ => unreachable!("split can be only index or loop"),
                }
            }
            match self.ops[dim_id].op {
                Op::Index { len, .. } | Op::Loop { len, .. } => debug_assert_eq!(len, dim),
                _ => {}
            }
        }

        let last_dim_op = self.get_last_dim_op(dim_id);
        let n_loops = splits.iter().filter(|op| matches!(op, Op::Loop { .. })).count();
        for (i, op) in splits.iter().enumerate() {
            if matches!(op, Op::Loop { .. }) {
                if is_loop && i == n_loops - 1 {
                } else {
                    self.insert_after(last_dim_op, Op::EndLoop);
                }
            }
        }

        let mut strides = Vec::new();
        let mut st = 1;
        for op in splits.iter().rev() {
            strides.push(st);
            match op {
                Op::Loop { len, .. } | Op::Index { len, .. } => st *= len,
                _ => unreachable!(),
            }
        }
        strides.reverse();
        strides.pop();
        let last_op = splits.pop().unwrap();

        let mut split_ids: Vec<OpId> = Vec::new();
        let mut acc = self.insert_before(dim_id, Op::Const(Constant::idx(0)));
        for (&st, op) in strides.iter().zip(splits) {
            let x = self.insert_before(dim_id, Op::Const(Constant::idx(st as u64)));
            let y = self.insert_before(dim_id, op);
            acc = self.insert_before(dim_id, Op::Mad { x, y, z: acc });
            split_ids.push(y);
        }

        let y = self.insert_before(dim_id, last_op);
        split_ids.push(y);
        self.ops[dim_id].op = Op::Binary { x: acc, y, bop: BOp::Add };

        self.verify();
        split_ids
    }
}
