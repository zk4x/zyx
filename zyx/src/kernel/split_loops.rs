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
use crate::kernel::{Kernel, Op, OpId, RangeKind};
use crate::shape::Dim;

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
    pub fn opt_split_global_to_local(&self) -> Box<dyn Optimization> {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_split_global_to_local");

        let dev_info = self.dev_info();

        // Boolean loops are conditionals, never split (like `EndIf` bailed before).
        if self.ops.values().any(|node| matches!(node.op, Op::Loop { len } if self.dtype(len) == crate::DType::Bool)) {
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
    pub fn opt_split_loop(&self) -> Box<dyn Optimization> {
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

}
