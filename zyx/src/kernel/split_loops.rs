// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Split loops optimization.
//!
//! This module provides loop splitting optimizations for kernels,
//! which split large loops into smaller iterations for better
//! instruction scheduling and vectorization.
//!
//! Loop splitting is useful for:
//!
//! - Reducing instruction dependencies
//! - Enabling better instruction-level parallelism
//! - Improving vectorization opportunities
//! - Splitting global indices into local factors

use super::autotune::Optimization;
use crate::{
    backend::DeviceInfo,
    dtype::Constant,
    kernel::{BOp, IdxKind, Kernel, Op, OpId},
};

impl Kernel {
    /// Optimize splitting global indices to local factors.
    ///
    /// This method splits global indices into local factors for
    /// parallelization across threads.
    ///
    /// Returns the optimization variant and number of variants.
    pub(crate) fn opt_split_global_to_local(&self, dev_info: &DeviceInfo) -> (Optimization, usize) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_split_global_to_local");
        if self.ops.values().any(|node| matches!(node.op, Op::EndIf)) {
            let factors = Vec::new();
            return (Optimization::SplitLoop { factors }, 0);
        }
        let mut local_axis_sizes: crate::Map<u32, u32> = crate::Map::default();
        for op in self.ops.values() {
            if let Op::Index { axis, kind: IdxKind::Local(len) } = op.op {
                if let Some(&existing) = local_axis_sizes.get(&axis) {
                    debug_assert_eq!(existing, len);
                } else {
                    local_axis_sizes.insert(axis, len);
                }
            }
        }
        let used_threads: u32 = local_axis_sizes.values().product();
        let remaining_threads = if local_axis_sizes.is_empty() {
            dev_info.max_local_threads
        } else {
            dev_info.max_local_threads / used_threads
        };
        //println!("local_axis_sizes={local_axis_sizes:?}, remaining_threads={remaining_threads}, used_threads={used_threads}");
        let mut op_id = self.head;
        let mut factors = Vec::new();
        while !op_id.is_null() {
            if let Op::Index { axis, kind: IdxKind::Group(len) } = self.ops[op_id].op {
                let mut l_factors: Vec<u32> = vec![64, 32, 16, 8, 4, 2];
                if !local_axis_sizes.contains_key(&axis) {
                    let max_per_axis = dev_info.max_local_work_dims[axis as usize];
                    let Some(len) = self.resolve_dim(len) else {
                        continue;
                    };
                    l_factors.retain(|&f| len.is_multiple_of(f as u64) && f <= remaining_threads && f <= max_per_axis);
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

    /// Optimize splitting large loops.
    ///
    /// This method splits large loops into smaller iterations for
    /// better instruction scheduling and vectorization.
    ///
    /// Returns the optimization variant and number of variants.
    pub(crate) fn opt_split_loop(&self) -> (Optimization, usize) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_split_loop");
        let candidates = vec![8, 16, 4, 2];
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Loop { len: len_id } = self.ops[op_id].op {
                let Some(len) = self.resolve_dim(len_id) else {
                    continue;
                };
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
    pub(crate) fn split_dim(&mut self, dim_id: OpId, mut splits: Vec<Op>) -> Vec<OpId> {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("split_dim");
        let is_loop = matches!(self.ops[dim_id].op, Op::Loop { .. });

        #[cfg(debug_assertions)]
        {
            let mut dim = 1;
            let mut ok = true;
            for op in splits.iter() {
                use crate::shape::Dim;

                match *op {
                    Op::Loop { len, .. } => match self.resolve_dim(len) {
                        Some(l) => dim *= l,
                        None => {
                            ok = false;
                            break;
                        }
                    },
                    Op::Index { kind, .. } => match kind {
                        IdxKind::Group(len) => match self.resolve_dim(len) {
                            Some(l) => dim *= l,
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        IdxKind::Local(len) => dim *= len as Dim,
                        IdxKind::Warp(len) => dim *= len as Dim,
                    },
                    _ => unreachable!("split can be only index or loop"),
                }
            }
            if ok {
                match self.ops[dim_id].op {
                    Op::Index { kind, .. } => {
                        use crate::shape::Dim;

                        match kind {
                            IdxKind::Group(len) => {
                                if let Some(l) = self.resolve_dim(len) {
                                    debug_assert_eq!(l, dim);
                                }
                            }
                            IdxKind::Local(l) => debug_assert_eq!(l as Dim, dim),
                            IdxKind::Warp(l) => debug_assert_eq!(l as Dim, dim),
                        }
                    }
                    Op::Loop { len, .. } => {
                        if let Some(l) = self.resolve_dim(len) {
                            debug_assert_eq!(l, dim);
                        }
                    }
                    _ => {}
                }
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
                Op::Loop { len, .. } => st *= self.resolve_dim(*len).unwrap(),
                Op::Index { kind, .. } => match kind {
                    IdxKind::Group(len) => st *= self.resolve_dim(*len).unwrap(),
                    IdxKind::Local(len) => st *= u64::from(*len),
                    IdxKind::Warp(len) => st *= u64::from(*len),
                },
                _ => unreachable!("split can be only index or loop"),
            }
        }
        strides.reverse();
        strides.pop();
        let last_op = splits.pop().unwrap();

        let mut split_ids: Vec<OpId> = Vec::new();
        let mut acc = self.insert_before(dim_id, Op::Const(Constant::idx(0)));
        for (&st, op) in strides.iter().zip(splits) {
            let x = self.insert_before(dim_id, Op::Const(Constant::idx(st)));
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
