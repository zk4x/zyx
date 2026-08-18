// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Tiled parallel reduction for large single-dimension reduction kernels.
//!
//! This optimization targets kernels that contain a **single large loop reducing
//! into one register accumulator** — e.g., `for i in 0..32000 { sum += data[i] }`.
//! It parallelizes the reduction across threads by splitting the loop iterations,
//! storing partial sums in shared memory, and doing a tree-reduce to produce the
//! final scalar result.
//!
//! **This is NOT a matmul optimization.** Applying it to a matmul kernel
//! would make performance worse than a naive matmul. It is designed for
//! standalone reduction ops (e.g. `Tensor::sum` over a large axis).

use super::autotune::Optimization;
use crate::{
    Map,
    backend::DeviceInfo,
    dtype::Constant,
    kernel::{BOp, IdxKind, Kernel, MemLayout, MemScope, Op, OpId},
    shape::Dim,
    slab::SlabId,
};

impl Kernel {
    pub(crate) fn opt_local_reduce(&self, dev_info: &DeviceInfo) -> (Optimization, usize) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_tiled_reduce");
        // Let's not tile reduce kernel with barriers for now
        // Don't apply tiled reduce if there's already a barrier or local index
        if self.ops.values().any(|node| matches!(node.op, Op::Barrier | Op::Index { kind: IdxKind::Local(_), .. })) {
            return (Optimization::TiledReduce { factors: Vec::new() }, 0);
        }
        // Only apply tiled reduce if there's exactly one loop in the kernel
        let n_loops = self.ops.values().filter(|node| matches!(node.op, Op::Loop { .. })).count();
        if n_loops != 1 {
            return (Optimization::TiledReduce { factors: Vec::new() }, 0);
        }

        let mut local_axis_sizes: Map<u32, u32> = crate::Map::default();
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

        let candidates = vec![1024, 512, 256, 128, 64, 32, 16, 8];
        let tree_branch_candidates = vec![2, 4, 8, 16];
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Op::Loop { len: len_id } = self.ops[op_id].op {
                let Some(len) = self.resolve_dim(len_id) else { continue };
                if len >= 16 {
                    for &factor in &candidates {
                        if len.is_multiple_of(factor) && len / factor >= 4 && remaining_threads as u64 >= factor {
                            for &tree_branch in &tree_branch_candidates {
                                factors.push((op_id, factor, tree_branch));
                            }
                        }
                    }
                }
            }
            op_id = next;
        }
        let n = factors.len();
        (Optimization::TiledReduce { factors }, n)
    }

    /// Apply tiled reduction parallelization.
    ///
    /// This method parallelizes a large single-dimension reduction loop
    /// across threads by splitting the loop iterations and storing
    /// partial sums in shared memory, then performing a tree reduction.
    ///
    /// # Arguments
    ///
    /// * `loop_start` - The loop operation to parallelize
    /// * `factor` - The factor for splitting the loop
    /// * `tree_branch` - The tree reduction branching factor
    pub(crate) fn local_reduce(&mut self, loop_start: OpId, factor: u32, tree_branch: u32) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("tiled_reduce");
        let loop_len_id = if let Op::Loop { len } = self.at(loop_start) {
            *len
        } else {
            return;
        };
        let Some(loop_len) = self.resolve_dim(loop_len_id) else {
            return;
        };

        // Get new free axis for the local dimension
        let laxis = self
            .ops
            .values()
            .filter_map(|node| {
                if let Op::Index { axis, kind: IdxKind::Local(_), .. } = node.op {
                    Some(axis + 1)
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0);
        if laxis > 2 {
            return;
        }

        // Find the acc definition
        let mut op_id = loop_start;
        let reg_acc;
        let acc_dtype;
        loop {
            if let Op::Storage { dtype, scope, len } = self.ops[op_id].op {
                if scope != MemScope::Register || len != 1 {
                    return;
                }
                reg_acc = op_id;
                acc_dtype = dtype;
                break;
            }
            op_id = self.prev_op(op_id);
            if op_id == OpId::NULL {
                // Accumulator was no found
                return;
            }
        }
        debug_assert!(!reg_acc.is_null());

        // Find the reduce loop bop and the op that used to load from the register accumulator
        let mut reduce_bop_id = OpId::NULL;
        let acc_load_id;
        let mut op_id = self.next_op(loop_start);
        let mut depth = 1;
        loop {
            match self.ops[op_id].op {
                // Update store to use the lidx for indexing
                Op::Store { dst, src: x, layout, .. } => {
                    debug_assert_eq!(layout, MemLayout::Scalar);
                    if dst == reg_acc {
                        reduce_bop_id = x;
                    }
                }
                Op::Load { src, layout, .. } if depth == 0 && src == reg_acc => {
                    debug_assert_eq!(layout, MemLayout::Scalar);
                    acc_load_id = op_id;
                    break;
                }
                Op::Loop { .. } => depth += 1,
                Op::EndLoop => depth -= 1,
                _ => {}
            }
            op_id = self.next_op(op_id);
            if op_id.is_null() {
                return;
            }
        }
        debug_assert!(!reduce_bop_id.is_null());
        let Op::Binary { bop, .. } = self.ops[reduce_bop_id].op else {
            return;
        };

        // ***** IMPLEMENTATION ***** //

        // Find the last global define to insert local memory after it
        let mut last_global = None;
        let mut op_id = self.head;
        while !op_id.is_null() {
            if matches!(self.ops[op_id].op, Op::Param { .. }) {
                last_global = Some(op_id);
            }
            op_id = self.next_op(op_id);
        }

        // Insert local memory definitions right after the last global define
        let insert_at = match last_global {
            Some(g) => {
                let n = self.next_op(g);
                if n.is_null() { self.tail } else { n }
            }
            None => self.head,
        };
        let loc_acc = self.insert_before(insert_at, Op::Storage { dtype: acc_dtype, scope: MemScope::Local, len: factor as Dim });
        let lidx = self.insert_before(insert_at, Op::Index { axis: laxis, kind: IdxKind::Local(factor) });

        // Divide reduce loop by factor
        let factor_const = self.insert_before(loop_start, Op::Const(Constant::idx(factor)));
        let new_len = self.insert_const_idx_before(loop_start, loop_len / factor as Dim);
        let ridx = self.insert_before(loop_start, Op::Loop { len: new_len });
        self.ops[loop_start].op = Op::Mad { x: ridx, y: factor_const, z: lidx };

        // Store to local accumulator
        let const_zero = self.insert_before(acc_load_id, Op::Const(Constant::idx(0)));
        let x = self.insert_before(acc_load_id, Op::Load { src: reg_acc, index: const_zero, layout: MemLayout::Scalar });
        self.insert_before(acc_load_id, Op::Store { dst: loc_acc, src: x, index: lidx, layout: MemLayout::Scalar });

        // Sync memory
        self.insert_before(acc_load_id, Op::Barrier);

        // Tree reduce: each step threads with lidx < active_threads load tree_branch elements and sum them
        // For factor=32, tree_branch 4:
        //   level 0: stride=32, active=8, offsets=8,16,24 -> combine for i in 0..8
        //   level 1: stride=8, active=2, offsets=2,4,6 -> combine for i in 0..2
        //   level 2: stride=2 < tree_branch=4, exit first loop
        //   Then binary reduction: stride=2 -> 1
        let mut stride = factor;
        while stride > 1 {
            let use_tree_branch = stride >= tree_branch;
            let active_threads = if use_tree_branch { stride / tree_branch } else { stride / 2 };
            let limit_const = self.insert_before(acc_load_id, Op::Const(Constant::idx(active_threads)));
            let condition = self.insert_before(acc_load_id, Op::Binary { x: lidx, y: limit_const, bop: BOp::Cmplt });
            self.insert_before(acc_load_id, Op::If { condition });

            let branch = if use_tree_branch { tree_branch } else { 2 };
            let mut sum_x = None;
            for i in 1..branch {
                let offset = i * active_threads;
                let offset_const = self.insert_before(acc_load_id, Op::Const(Constant::idx(offset)));
                let offset_idx = self.insert_before(acc_load_id, Op::Binary { x: lidx, y: offset_const, bop: BOp::Add });
                let local_load =
                    self.insert_before(acc_load_id, Op::Load { src: loc_acc, index: offset_idx, layout: MemLayout::Scalar });
                if let Some(prev_sum) = sum_x {
                    sum_x = Some(self.insert_before(acc_load_id, Op::Binary { x: prev_sum, y: local_load, bop }));
                } else {
                    let current_val =
                        self.insert_before(acc_load_id, Op::Load { src: loc_acc, index: lidx, layout: MemLayout::Scalar });
                    sum_x = Some(self.insert_before(acc_load_id, Op::Binary { x: current_val, y: local_load, bop }));
                }
            }
            let bop_id = sum_x.unwrap();
            self.insert_before(acc_load_id, Op::Store { dst: loc_acc, src: bop_id, index: lidx, layout: MemLayout::Scalar });

            self.insert_before(acc_load_id, Op::EndIf);
            self.insert_before(acc_load_id, Op::Barrier);

            stride = active_threads;
        }

        // Load final result from local[0] to register (only thread 0)
        let condition = self.insert_before(acc_load_id, Op::Binary { x: lidx, y: const_zero, bop: BOp::Eq });
        self.insert_before(acc_load_id, Op::If { condition });
        let final_val = self.insert_before(acc_load_id, Op::Load { src: loc_acc, index: const_zero, layout: MemLayout::Scalar });
        self.insert_before(acc_load_id, Op::Store { dst: reg_acc, src: final_val, index: const_zero, layout: MemLayout::Scalar });
        self.insert_after(self.tail, Op::EndIf);

        self.verify();
    }
}
