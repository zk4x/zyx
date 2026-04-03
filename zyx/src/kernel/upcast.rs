// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use super::autotune::Optimization;
use crate::{
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, Scope},
    Map, Set,
};

impl Kernel {
    pub fn opt_upcast(&self) -> (Optimization, usize) {
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Index { len, scope, .. } = self.ops[op_id].op {
                if scope == Scope::Global {
                    for &f in &[4, 8, 2, 16] {
                        if len.is_multiple_of(f) && len / f >= 4 {
                            factors.push((op_id, f));
                        }
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        let n_configs = factors.len();
        (Optimization::Upcast { factors }, n_configs)
    }

    /// Upcast optimization: increase work per thread to reduce global work size.
    ///
    /// When we have a global index `gidx` that iterates over a large range, we can process
    /// multiple elements per thread to reduce the number of thread blocks needed.
    ///
    /// **Example: factor=4**
    ///
    /// Before:
    /// ```c
    /// for (int i = 0; i < N; i++) {           // reduce loop
    ///     acc += data[gidx * N + i];
    /// }
    /// ```
    ///
    /// After (upcast with factor=4):
    /// ```c
    /// // BEFORE: Initialize accumulator array
    /// for (int j = 0; j < 4; j++) {
    ///     acc[j] = 0;
    /// }
    ///
    /// // REDUCE: Inner loop inside reduce, each thread processes 4 elements
    /// for (int i = 0; i < N; i++) {           // reduce loop
    ///     for (int j = 0; j < 4; j++) {       // inner loop (fused into reduce)
    ///         acc[j] += data[gidx * 4 * N + j * N + i];
    ///     }
    /// }
    ///
    /// // AFTER: Read accumulated results
    /// for (int j = 0; j < 4; j++) {
    ///     output[gidx + j * N] = acc[j];
    /// }
    /// ```
    ///
    /// **Effect on global work size:**
    /// - Before: `grid.x = ceil(N / block.x)` thread blocks
    /// - After: `grid.x = ceil(N / (block.x * factor))` thread blocks (reduced by factor)
    ///
    /// The inner loop processes `factor` elements per reduce iteration, so the global
    /// work size is reduced by factor. This also increases register usage (accumulator
    /// is now an array of size `factor`).
    ///
    /// **Why three loops?**
    /// - **Before**: Must initialize `acc[j]` to 0 before reduce starts
    /// - **Reduce + Inner**: Processes `factor` elements per reduce iteration,
    ///   accumulating into `acc[j]` where j is the inner loop index
    /// - **After**: Must READ the accumulated results after reduce completes
    pub fn upcast(&mut self, op_id: OpId, factor: usize) {
        let Op::Index { len, scope, axis } = self.ops[op_id].op else {
            unreachable!("upcast only works on Index ops");
        };
        debug_assert_eq!(scope, Scope::Global);
        debug_assert!(len.is_multiple_of(factor));

        // split_dim returns [Index_id, Loop_id]
        let split_ids = self.split_dim(
            op_id,
            vec![
                Op::Index { len: len / factor, scope: Scope::Global, axis },
                Op::Loop { len: factor },
            ],
        );
        let upcast_loop_id = split_ids[1]; // The Loop created by split_dim

        // Find the first loop nested inside the upcast loop with different length
        let mut reduce_loop_id = OpId::NULL;
        let mut loop_depth = 0;
        let mut op_id_iter = self.next_op(upcast_loop_id);
        while !op_id_iter.is_null() {
            match self.ops[op_id_iter].op {
                Op::Loop { len, .. } => {
                    let current_depth = loop_depth;
                    loop_depth += 1;
                    if current_depth >= 1 && len != factor {
                        reduce_loop_id = op_id_iter;
                        break;
                    }
                }
                Op::EndLoop => {
                    if loop_depth == 0 {
                        break;
                    }
                    loop_depth -= 1;
                }
                _ => {}
            }
            op_id_iter = self.next_op(op_id_iter);
        }

        // Jam the reduce loop into the upcast loop to increase work per thread
        if reduce_loop_id != OpId::NULL {
            self.jam_loop(reduce_loop_id, upcast_loop_id);
        }

        self.verify();
    }

    /// Jam loop moves a loop (jam_loop_id) inside another loop (inner_loop_id).
    ///
    /// This is used in upcast to fuse the upcast loop with the reduce loop:
    /// - The upcast loop iterates over the global index in chunks
    /// - The reduce loop processes those chunks
    /// - After jamming, the reduce loop accumulates all chunks before returning
    ///
    /// The accumulator size is increased by the jam_dim factor to hold all partial results.
    pub fn jam_loop(&mut self, jam_loop_id: OpId, inner_loop_id: OpId) {
        let mut op_id = jam_loop_id;
        while op_id != inner_loop_id {
            op_id = self.next_op(op_id);
            if self.at(op_id).is_load() {
                return;
            }
        }

        let mut op_id = jam_loop_id;
        let mut loop_level = 0;
        let mut middle_loop_id = OpId::NULL;
        let mut end_middle_loop_id = OpId::NULL;
        let mut end_inner_loop_id = OpId::NULL;
        let mut inner_loop_level = None;
        let mut pre_loop_ops = Set::default();
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Loop { .. } => {
                    if loop_level == 1 {
                        middle_loop_id = op_id;
                    }
                    if op_id == inner_loop_id {
                        inner_loop_level = Some(loop_level);
                    }
                    loop_level += 1;
                }
                Op::EndLoop => {
                    loop_level -= 1;
                    if let Some(inner_loop_level) = inner_loop_level {
                        if loop_level == inner_loop_level {
                            end_inner_loop_id = op_id;
                        }
                        if loop_level == 1 {
                            end_middle_loop_id = op_id;
                            break;
                        }
                    }
                }
                _ => {}
            }
            if loop_level == 1 {
                pre_loop_ops.insert(op_id);
            }
            op_id = self.next_op(op_id);
        }

        let mut op_id = middle_loop_id;
        while op_id != inner_loop_id {
            if self.ops[op_id].op.parameters().any(|p| pre_loop_ops.contains(&p)) {
                return;
            }
            op_id = self.next_op(op_id);
        }
        let mut op_id = end_inner_loop_id;
        while op_id != end_middle_loop_id {
            if self.ops[op_id].op.parameters().any(|p| pre_loop_ops.contains(&p)) {
                return;
            }
            op_id = self.next_op(op_id);
        }

        let jam_dim = if let Op::Loop { len, .. } = self.ops[jam_loop_id].op {
            len
        } else {
            return;
        };

        let const_jam_dim = self.insert_before(jam_loop_id, Op::Const(Constant::idx(jam_dim as u64)));

        let mut defines = Set::default();
        let mut op_id = jam_loop_id;
        while op_id != middle_loop_id {
            op_id = self.next_op(op_id);
            if let Op::Define { dtype, scope, ro, len } = self.ops[op_id].op {
                self.ops[op_id].op = Op::Define { dtype, scope, ro, len: len * jam_dim };
                defines.insert(op_id);
                self.move_op_before(op_id, jam_loop_id);
            }
        }

        let mut op_id = jam_loop_id;
        while op_id != middle_loop_id {
            op_id = self.next_op(op_id);
            match *self.at(op_id) {
                Op::Load { .. } | Op::Define { .. } => unreachable!(),
                Op::Store { dst, index, .. } => {
                    if defines.contains(&dst) {
                        let x = self.insert_before(op_id, Op::Binary { x: index, y: const_jam_dim, bop: BOp::Mul });
                        let new_index = self.insert_before(op_id, Op::Binary { x, y: jam_loop_id, bop: BOp::Add });
                        let Op::Store { index, .. } = &mut self.ops[op_id].op else {
                            unreachable!()
                        };
                        *index = new_index;
                    }
                }
                _ => {}
            }
        }
        let end_pre_loop = self.insert_before(middle_loop_id, Op::EndLoop);

        let mut remapping = Map::default();
        let mut op_id = jam_loop_id;
        let mut t_op_id = inner_loop_id;
        while op_id != end_pre_loop {
            let mut op = self.ops[op_id].op.clone();
            match self.at(op_id) {
                Op::Load { .. } | Op::Define { .. } => unreachable!(),
                Op::Store { .. } => {}
                _ => {
                    op.remap_params(&remapping);
                    t_op_id = self.insert_after(t_op_id, op);
                    remapping.insert(op_id, t_op_id);
                }
            }
            op_id = self.next_op(op_id);
        }

        let mut op_id = t_op_id;
        let mut loop_level = 1;
        loop {
            op_id = self.next_op(op_id);
            self.ops[op_id].op.remap_params(&remapping);
            match self.ops[op_id].op {
                Op::Load { src, index, .. } => {
                    if defines.contains(&src) {
                        let x = self.insert_before(op_id, Op::Binary { x: index, y: const_jam_dim, bop: BOp::Mul });
                        let new_index = self.insert_before(op_id, Op::Binary { x, y: remapping[&jam_loop_id], bop: BOp::Add });
                        let Op::Load { index, .. } = &mut self.ops[op_id].op else {
                            unreachable!()
                        };
                        *index = new_index;
                    }
                }
                Op::Store { dst, index, .. } => {
                    if defines.contains(&dst) {
                        let x = self.insert_before(op_id, Op::Binary { x: index, y: const_jam_dim, bop: BOp::Mul });
                        let new_index = self.insert_before(op_id, Op::Binary { x, y: remapping[&jam_loop_id], bop: BOp::Add });
                        let Op::Store { index, .. } = &mut self.ops[op_id].op else {
                            unreachable!()
                        };
                        *index = new_index;
                    }
                }
                Op::Loop { .. } => loop_level += 1,
                Op::EndLoop => {
                    loop_level -= 1;
                    if loop_level == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }
        self.insert_before(op_id, Op::EndLoop);

        remapping.clear();
        let mut t_op_id = end_middle_loop_id;
        let mut op_id = jam_loop_id;
        while op_id != end_pre_loop {
            let mut op = self.ops[op_id].op.clone();
            match self.at(op_id) {
                Op::Load { .. } | Op::Define { .. } => unreachable!(),
                Op::Store { .. } => {}
                _ => {
                    op.remap_params(&remapping);
                    t_op_id = self.insert_after(t_op_id, op);
                    remapping.insert(op_id, t_op_id);
                }
            }
            op_id = self.next_op(op_id);
        }

        let mut op_id = t_op_id;
        let mut loop_level = 1;
        loop {
            op_id = self.next_op(op_id);
            self.ops[op_id].op.remap_params(&remapping);
            match self.ops[op_id].op {
                Op::Load { src, index, .. } => {
                    if defines.contains(&src) {
                        let x = self.insert_before(op_id, Op::Binary { x: index, y: const_jam_dim, bop: BOp::Mul });
                        let new_index = self.insert_before(op_id, Op::Binary { x, y: remapping[&jam_loop_id], bop: BOp::Add });
                        let Op::Load { index, .. } = &mut self.ops[op_id].op else {
                            unreachable!()
                        };
                        *index = new_index;
                    }
                }
                Op::Store { dst, index, .. } => {
                    if defines.contains(&dst) {
                        let x = self.insert_before(op_id, Op::Binary { x: index, y: const_jam_dim, bop: BOp::Mul });
                        let new_index = self.insert_before(op_id, Op::Binary { x, y: remapping[&jam_loop_id], bop: BOp::Add });
                        let Op::Store { index, .. } = &mut self.ops[op_id].op else {
                            unreachable!()
                        };
                        *index = new_index;
                    }
                }
                Op::Loop { .. } => loop_level += 1,
                Op::EndLoop => {
                    loop_level -= 1;
                    if loop_level == 0 {
                        break;
                    }
                }
                _ => {}
            }
        }
    }
}
