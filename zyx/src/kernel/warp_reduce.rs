// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use super::autotune::Optimization;
use crate::{
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, Scope},
};

impl Kernel {
    pub fn opt_local_reduce(&self) -> (Optimization, usize) {
        let candidates = vec![32, 16, 8, 64];
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Loop { len } = self.ops[op_id].op {
                if len >= 256 {
                    for &factor in &candidates {
                        if len.is_multiple_of(factor) && len / factor >= factor {
                            factors.push((op_id, factor));
                        }
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        let n = factors.len();
        (Optimization::LocalReduce { factors }, n)
    }

    pub fn tile_reduce_to_local(&mut self, loop_start: OpId, factor: usize) {
        //println!("Local reduce of loop={loop_start} with factor={factor}");
        let loop_len = if let Op::Loop { len } = self.at(loop_start) {
            *len
        } else {
            return;
        };

        // Get new free axis for the local dimension
        let laxis = self
            .ops
            .values()
            .filter_map(|node| {
                if let Op::Index { scope: Scope::Local, axis, .. } = node.op {
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
            if let Op::Define { dtype, scope, ro, len } = self.ops[op_id].op {
                if scope != Scope::Register || ro || len != 1 {
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
                Op::Store { dst, x, vlen, .. } => {
                    debug_assert_eq!(vlen, 1);
                    if dst == reg_acc {
                        reduce_bop_id = x;
                    }
                }
                Op::Load { src, vlen, .. } => {
                    if depth == 0 && src == reg_acc {
                        debug_assert_eq!(vlen, 1);
                        acc_load_id = op_id;
                        break;
                    }
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

        // Add local index
        let lidx = self.insert_before(reg_acc, Op::Index { len: factor, scope: Scope::Local, axis: laxis });

        // Divide reduce loop by factor
        let factor_const = self.insert_before(loop_start, Op::Const(Constant::idx(factor as u64)));
        let ridx = self.insert_before(loop_start, Op::Loop { len: loop_len / factor });
        self.ops[loop_start].op = Op::Mad { x: ridx, y: factor_const, z: lidx };

        // Add local accumulator
        let loc_acc = self.insert_before(
            acc_load_id,
            Op::Define { dtype: acc_dtype, scope: Scope::Local, ro: false, len: factor },
        );

        // Store to local accumulator
        let const_zero = self.insert_before(acc_load_id, Op::Const(Constant::idx(0)));
        let x = self.insert_before(acc_load_id, Op::Load { src: reg_acc, index: const_zero, vlen: 1 });
        self.insert_before(acc_load_id, Op::Store { dst: loc_acc, x, index: lidx, vlen: 1 });

        // Sync memory
        self.insert_before(acc_load_id, Op::Barrier { scope: Scope::Local });

        // Add the branch
        let const_last = self.insert_before(acc_load_id, Op::Const(Constant::idx(factor as u64 - 1)));
        let condition = self.insert_before(acc_load_id, Op::Binary { x: lidx, y: const_last, bop: BOp::Eq });
        self.insert_before(acc_load_id, Op::If { condition });

        // Add the loop that accumulates over local memory
        let ridx = self.insert_before(acc_load_id, Op::Loop { len: factor - 1 });
        let reg_load = self.insert_before(acc_load_id, Op::Load { src: reg_acc, index: const_zero, vlen: 1 });
        let local_load = self.insert_before(acc_load_id, Op::Load { src: loc_acc, index: ridx, vlen: 1 });
        let bop_id = self.insert_before(acc_load_id, Op::Binary { x: reg_load, y: local_load, bop });
        self.insert_before(acc_load_id, Op::Store { dst: reg_acc, x: bop_id, index: const_zero, vlen: 1 });
        self.insert_before(acc_load_id, Op::EndLoop);
        self.insert_after(self.tail, Op::EndIf);
    }
}
