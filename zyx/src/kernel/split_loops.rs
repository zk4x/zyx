// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

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
    kernel::{BOp, Kernel, Op, OpId, RangeKind},
    shape::Dim,
};

/// Split a global index into local factors for parallelization.
#[derive(Debug)]
pub struct SplitGlobalToLocal {
    /// Pairs of (operation_id, split_factor) for each split.
    pub factors: Vec<(OpId, u32)>,
}

impl Optimization for SplitGlobalToLocal {
    fn nconfigs(&self) -> u64 {
        self.factors.len() as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("SplitGlobalToLocal");
        let (op_id, factor) = self.factors[config as usize];
        let Op::Range { axis, kind: RangeKind::Group(len) } = kernel.ops[op_id].op else {
            unreachable!()
        };
        // valid factors are checked by opt init
        let len = kernel.resolve_const(len).and_then(crate::dtype::Constant::as_dim).unwrap();
        let group_len = kernel.insert_const_idx_before(op_id, len / Dim::from(factor));
        kernel.split_dim(
            op_id,
            vec![
                Op::Range { axis, kind: RangeKind::Group(group_len) },
                Op::Range { axis, kind: RangeKind::Local(factor) },
            ],
        );
    }
}

/// Split a loop into smaller iterations.
#[derive(Debug)]
pub struct SplitLoop {
    /// Pairs of (loop_id, split_factor) for each split.
    pub factors: Vec<(OpId, u64)>,
}

impl Optimization for SplitLoop {
    fn nconfigs(&self) -> u64 {
        self.factors.len() as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        let (op_id, factor) = self.factors[config as usize];
        let Op::Loop { len: len_id } = kernel.ops[op_id].op else {
            unreachable!()
        };
        let Some(len) = kernel.resolve_const(len_id).and_then(crate::dtype::Constant::as_dim) else {
            return;
        };
        let len1 = kernel.insert_const_idx_before(op_id, len / factor as Dim);
        let len2 = kernel.insert_const_idx_before(op_id, factor);
        kernel.split_dim(op_id, vec![Op::Loop { len: len1 }, Op::Loop { len: len2 }]);
    }
}

impl Kernel {
    /// Make the `SplitGlobalToLocal` optimization: scan the kernel for
    /// global indices that can be split into local factors.
    ///
    /// Config ids are ordered by factor, hardware-aligned factors first
    /// (e.g. 64/32 for warp-sized groups).
    pub fn opt_split_global_to_local(&self, dev_info: &DeviceInfo) -> Box<dyn Optimization> {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_split_global_to_local");
        if self.ops.values().any(|node| matches!(node.op, Op::EndIf)) {
            return Box::new(SplitGlobalToLocal { factors: Vec::new() });
        }
        let mut local_axis_sizes: crate::Map<u32, u32> = crate::Map::default();
        for op in self.ops.values() {
            if let Op::Range { axis, kind: RangeKind::Local(len) } = op.op {
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
            if let Op::Range { axis, kind: RangeKind::Group(len) } = self.ops[op_id].op {
                let mut l_factors: Vec<u32> = vec![64, 32, 16, 8, 4, 2];
                if !local_axis_sizes.contains_key(&axis) {
                    let max_per_axis = dev_info.max_local_work_dims[axis as usize];
                    let Some(len) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) else {
                        op_id = self.next_op(op_id);
                        continue;
                    };
                    l_factors.retain(|&f| len % f as Dim == 0 && f <= remaining_threads && f <= max_per_axis);
                    for &f in &l_factors {
                        factors.push((op_id, f));
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        Box::new(SplitGlobalToLocal { factors })
    }

    /// Make the `SplitLoop` optimization: scan the kernel for large loops
    /// that can be split into smaller iterations.
    pub fn opt_split_loop(&self, _dev_info: &DeviceInfo) -> Box<dyn Optimization> {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_split_loop");
        let candidates = vec![8, 16, 4, 2];
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Loop { len: len_id } = self.ops[op_id].op {
                let Some(len) = self.resolve_const(len_id).and_then(crate::dtype::Constant::as_dim) else {
                    op_id = self.next_op(op_id);
                    continue;
                };
                if len >= 16 {
                    for &factor in &candidates {
                        if len % factor as Dim == 0 {
                            factors.push((op_id, factor));
                        }
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        Box::new(SplitLoop { factors })
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
                    Op::Loop { len, .. } => match self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) {
                        Some(l) => dim *= l,
                        None => {
                            ok = false;
                            break;
                        }
                    },
                    Op::Range { kind, .. } => match kind {
                        RangeKind::Group(len) => match self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) {
                            Some(l) => dim *= l,
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        RangeKind::Local(len) => dim *= len as Dim,
                        // A warp is a view over a local range — it is never split.
                        RangeKind::Warp(_) => {
                            ok = false;
                            break;
                        }
                    },
                    _ => unreachable!("split can be only index or loop"),
                }
            }
            if ok {
                match self.ops[dim_id].op {
                    Op::Range { kind, .. } => {
                        use crate::shape::Dim;

                        match kind {
                            RangeKind::Group(len) => {
                                if let Some(l) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) {
                                    debug_assert_eq!(l, dim);
                                }
                            }
                            RangeKind::Local(l) => debug_assert_eq!(l as Dim, dim),
                            RangeKind::Warp(_) => unreachable!("warp dims are never split"),
                        }
                    }
                    Op::Loop { len, .. } => {
                        if let Some(l) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) {
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
                Op::Loop { len, .. } => st *= self.resolve_const(*len).and_then(crate::dtype::Constant::as_dim).unwrap(),
                Op::Range { kind, .. } => match kind {
                    RangeKind::Group(len) => st *= self.resolve_const(*len).and_then(crate::dtype::Constant::as_dim).unwrap(),
                    RangeKind::Local(len) => st *= i64::from(*len),
                    RangeKind::Warp(_) => unreachable!("warp dims are never split"),
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
