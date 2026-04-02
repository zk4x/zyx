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

    /// Helper function for remapping ops in a range and adjusting indices for defines.
    /// Returns the op_id where the remapping loop ended (the EndLoop that closed the range).
    fn remap_range_and_adjust_indices(
        &mut self,
        start_op_id: OpId,
        end_op_id: OpId,
        target_op_id: OpId,
        remapping: &mut Map<OpId, OpId>,
        defines: &Set<OpId>,
        const_jam_dim: OpId,
        jam_loop_id: OpId,
        insert_end_loop: bool,
    ) -> OpId {
        // Clone ops from start_op_id to end_op_id
        let mut op_id = start_op_id;
        let mut t_op_id = target_op_id;
        while op_id != end_op_id {
            let mut op = self.ops[op_id].op.clone();
            match self.at(op_id) {
                Op::Load { .. } | Op::Define { .. } => unreachable!(),
                Op::Store { .. } => {}
                _ => {
                    op.remap_params(remapping);
                    t_op_id = self.insert_after(t_op_id, op);
                    remapping.insert(op_id, t_op_id);
                }
            }
            op_id = self.next_op(op_id);
        }

        // Remap loop
        let mut op_id = t_op_id;
        let mut loop_level = 1;
        loop {
            op_id = self.next_op(op_id);
            self.ops[op_id].op.remap_params(remapping);
            match self.ops[op_id].op {
                Op::Load { src, index, .. } | Op::Store { dst: src, index, .. } => {
                    if defines.contains(&src) {
                        let x = self.insert_before(
                            op_id,
                            Op::Binary {
                                x: index,
                                y: const_jam_dim,
                                bop: BOp::Mul,
                            },
                        );
                        let new_index = self.insert_before(
                            op_id,
                            Op::Binary {
                                x,
                                y: remapping[&jam_loop_id],
                                bop: BOp::Add,
                            },
                        );
                        match &mut self.ops[op_id].op {
                            Op::Load { index: idx, .. } | Op::Store { index: idx, .. } => {
                                *idx = new_index;
                            }
                            _ => unreachable!(),
                        }
                    }
                }
                Op::Loop { .. } => loop_level += 1,
                Op::EndLoop => {
                    loop_level -= 1;
                    if loop_level == 0 {
                        if insert_end_loop {
                            self.insert_before(op_id, Op::EndLoop);
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
        op_id
    }

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
                Op::Index {
                    len: len / factor,
                    scope: Scope::Global,
                    axis,
                },
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
                    loop_depth += 1;
                    // First loop nested inside upcast loop with different length
                    if len != factor && loop_depth == 1 {
                        reduce_loop_id = op_id_iter;
                        break;
                    }
                }
                Op::EndLoop => {
                    if loop_depth == 0 {
                        // Exited the upcast loop
                        break;
                    }
                    loop_depth -= 1;
                }
                _ => {}
            }
            op_id_iter = self.next_op(op_id_iter);
        }

        if reduce_loop_id != OpId::NULL {
            let jam_loop_id = upcast_loop_id;
            let inner_loop_id = reduce_loop_id;

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

            let Op::Loop { len: jam_dim, .. } = self.ops[jam_loop_id].op else {
                unreachable!()
            };

            let const_jam_dim = self.insert_before(jam_loop_id, Op::Const(Constant::idx(jam_dim as u64)));

            // Move all defines before the loop
            let mut defines = Set::default();
            let mut op_id = jam_loop_id;
            while op_id != middle_loop_id {
                op_id = self.next_op(op_id);
                if let Op::Define { dtype, scope, ro, len } = self.ops[op_id].op {
                    self.ops[op_id].op = Op::Define {
                        dtype,
                        scope,
                        ro,
                        len: len * jam_dim,
                    };
                    defines.insert(op_id);
                    self.move_op_before(op_id, jam_loop_id);
                }
            }

            // Reindex stores
            let mut op_id = jam_loop_id;
            while op_id != middle_loop_id {
                op_id = self.next_op(op_id);
                match *self.at(op_id) {
                    Op::Load { .. } | Op::Define { .. } => unreachable!(),
                    Op::Store { dst, index, .. } => {
                        if defines.contains(&dst) {
                            let x = self.insert_before(
                                op_id,
                                Op::Binary {
                                    x: index,
                                    y: const_jam_dim,
                                    bop: BOp::Mul,
                                },
                            );
                            let new_index = self.insert_before(
                                op_id,
                                Op::Binary {
                                    x,
                                    y: jam_loop_id,
                                    bop: BOp::Add,
                                },
                            );
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
            self.remap_range_and_adjust_indices(
                jam_loop_id,
                end_pre_loop,
                inner_loop_id,
                &mut remapping,
                &defines,
                const_jam_dim,
                jam_loop_id,
                true,
            );

            remapping.clear();
            self.remap_range_and_adjust_indices(
                jam_loop_id,
                end_pre_loop,
                end_middle_loop_id,
                &mut remapping,
                &defines,
                const_jam_dim,
                jam_loop_id,
                false,
            );
        }
    }
}
