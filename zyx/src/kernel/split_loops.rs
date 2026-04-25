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
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_split_global_to_local");
        if self.ops.values().any(|node| matches!(node.op, Op::EndIf)) {
            let factors = Vec::new();
            return (Optimization::SplitLoop { factors }, 0);
        }
        let mut local_axis_sizes: crate::Map<u32, u64> = crate::Map::default();
        for op in self.ops.values() {
            if let Op::Index { scope: Scope::Local, axis, len } = op.op {
                if let Some(&existing) = local_axis_sizes.get(&axis) {
                    debug_assert_eq!(existing, len);
                } else {
                    local_axis_sizes.insert(axis, len);
                }
            }
        }
        let used_threads: u64 = local_axis_sizes.values().product::<u64>();
        let remaining_threads = if local_axis_sizes.is_empty() {
            dev_info.max_local_threads
        } else {
            dev_info.max_local_threads / used_threads
        };
        //println!("local_axis_sizes={local_axis_sizes:?}, remaining_threads={remaining_threads}, used_threads={used_threads}");
        let mut op_id = self.head;
        let mut factors = Vec::new();
        while !op_id.is_null() {
            if let Op::Index { len, scope, axis } = self.ops[op_id].op {
                let mut l_factors: Vec<u64> = vec![64, 32, 16, 8, 4, 2];
                if scope == Scope::Global && !local_axis_sizes.contains_key(&axis) {
                    let max_per_axis = dev_info.max_local_work_dims[axis as usize] as u64;
                    l_factors.retain(|&f| len.is_multiple_of(f) && f <= remaining_threads && f <= max_per_axis);
                    for &f in &l_factors {
                        factors.push((op_id, f));
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        let n_configs = factors.len();
        (Optimization::SplitGlobalToLocal { factors }, n_configs)
    }

    pub fn opt_split_loop(&self) -> (Optimization, usize) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_split_loop");
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
