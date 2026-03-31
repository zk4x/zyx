// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use crate::dtype::Constant;
use crate::kernel::{BOp, Kernel, Op, OpId, Scope};
use crate::shape::Dim;
use crate::DType;

impl Kernel {
    pub fn optimize_warp_reduce(&mut self) {
        let Some(loop_start) = find_reduce_loop_start(self) else {
            return;
        };

        let Some(loop_end) = find_reduce_loop_end(self, loop_start) else {
            return;
        };

        let loop_len = if let Op::Loop { len } = self.at(loop_start) {
            *len
        } else {
            return;
        };

        if !should_optimize(loop_len) {
            return;
        };

        let Some((acc_define, acc_store, bop)) = find_reduce_pattern(self, loop_start, loop_end) else {
            return;
        };

        let factor = 32;
        let outer_len = loop_len / factor;
        if outer_len < 2 {
            return;
        }

        // Create local buffer of size factor
        let local_buf = self.insert_before(
            loop_start,
            Op::Define {
                dtype: DType::F32,
                scope: Scope::Local,
                ro: false,
                len: factor,
            },
        );

        // Create local index for the factor
        let lidx = self.insert_before(
            loop_start,
            Op::Index {
                len: factor,
                scope: Scope::Local,
                axis: 0,
            },
        );

        // Create idx_zero constant
        let idx_zero = self.insert_before(loop_start, Op::Const(Constant::idx(0)));

        // Load initial accumulator value
        let load_acc = self.insert_before(
            loop_start,
            Op::Load {
                src: acc_define,
                index: idx_zero,
                vlen: 1,
            },
        );

        // Store initial value to local buffer at lidx
        self.insert_before(
            loop_start,
            Op::Store {
                dst: local_buf,
                x: load_acc,
                index: lidx,
                vlen: 1,
            },
        );

        // Modify the original loop to iterate by factor
        // Change loop len from loop_len to outer_len
        self.ops[loop_start].op = Op::Loop { len: outer_len };

        // Find the index computation inside the loop and modify it to stride by factor
        let mut op_id = self.next_op(loop_start);
        while op_id != loop_end {
            if let Op::Binary { x, y, bop: BOp::Add } = self.at(op_id).clone() {
                // Check if this is an index computation
                if let Op::Binary {
                    x: x2,
                    y: y2,
                    bop: BOp::Mul,
                } = self.at(x).clone()
                {
                    // This looks like: base + offset, where offset might be the loop index
                    // We need to multiply the offset by factor
                    let factor_const = self.insert_before(op_id, Op::Const(Constant::idx(factor as i32)));
                    let new_offset = self.insert_before(
                        op_id,
                        Op::Binary {
                            x: x2,
                            y: factor_const,
                            bop: BOp::Mul,
                        },
                    );
                    self.ops[op_id].op = Op::Binary {
                        x: new_offset,
                        y: y2,
                        bop: BOp::Add,
                    };
                    break;
                }
            }
            op_id = self.next_op(op_id);
        }

        // Change the store inside the loop to store to local buffer at lidx
        if let Op::Store { x, vlen, .. } = self.at(acc_store).clone() {
            self.ops[acc_store].op = Op::Store {
                dst: local_buf,
                x,
                index: lidx,
                vlen,
            };
        }

        // After the original loop, add barrier
        let after_loop_end = self.next_op(loop_end);
        self.insert_before(after_loop_end, Op::Barrier { scope: Scope::Local });

        // Second loop: reduce across local buffer, only thread 0 does it
        let is_thread_0 = self.insert_before(
            after_loop_end,
            Op::Binary {
                x: lidx,
                y: idx_zero,
                bop: BOp::Eq,
            },
        );

        self.insert_before(after_loop_end, Op::If { condition: is_thread_0 });

        self.insert_before(after_loop_end, Op::Loop { len: factor });

        let inner_idx = self.insert_before(
            after_loop_end,
            Op::Index {
                len: factor,
                scope: Scope::Local,
                axis: 0,
            },
        );

        let load_partial = self.insert_before(
            after_loop_end,
            Op::Load {
                src: local_buf,
                index: inner_idx,
                vlen: 1,
            },
        );

        let load_acc2 = self.insert_before(
            after_loop_end,
            Op::Load {
                src: acc_define,
                index: idx_zero,
                vlen: 1,
            },
        );

        let bin_result = self.insert_before(
            after_loop_end,
            Op::Binary {
                x: load_partial,
                y: load_acc2,
                bop,
            },
        );

        self.insert_before(
            after_loop_end,
            Op::Store {
                dst: acc_define,
                x: bin_result,
                index: idx_zero,
                vlen: 1,
            },
        );

        self.insert_before(after_loop_end, Op::EndLoop);
        self.insert_before(after_loop_end, Op::EndIf);
    }
}

fn find_reduce_loop_start(kernel: &Kernel) -> Option<OpId> {
    let mut op_id = kernel.head;
    while !op_id.is_null() {
        if let Op::Loop { len, .. } = kernel.at(op_id) {
            if *len >= 32 {
                return Some(op_id);
            }
        }
        op_id = kernel.next_op(op_id);
    }
    None
}

fn find_reduce_loop_end(kernel: &Kernel, loop_start: OpId) -> Option<OpId> {
    let mut depth = 0;
    let mut op_id = loop_start;
    loop {
        match kernel.at(op_id) {
            Op::Loop { .. } => depth += 1,
            Op::EndLoop => {
                depth -= 1;
                if depth == 0 {
                    return Some(op_id);
                }
            }
            _ => {}
        }
        op_id = kernel.next_op(op_id);
        if op_id.is_null() {
            return None;
        }
    }
}

fn find_reduce_pattern(kernel: &Kernel, loop_start: OpId, loop_end: OpId) -> Option<(OpId, OpId, BOp)> {
    let mut acc_define = None;
    let mut acc_store = None;
    let mut bop = None;

    let mut op_id = kernel.head;
    while !op_id.is_null() {
        if op_id == loop_start {
            break;
        }
        match kernel.at(op_id) {
            Op::Define { scope, len, ro, .. } => {
                if *scope == Scope::Register && *len == 1 && !*ro {
                    acc_define = Some(op_id);
                }
            }
            _ => {}
        }
        op_id = kernel.next_op(op_id);
    }

    op_id = kernel.next_op(loop_start);
    while op_id != loop_end {
        match kernel.at(op_id) {
            Op::Store { dst, .. } => {
                if let Op::Define { scope, len, .. } = kernel.at(*dst) {
                    if *scope == Scope::Register && *len == 1 {
                        acc_store = Some(op_id);
                    }
                }
            }
            Op::Binary { bop: BOp::Max, .. } => {
                bop = Some(BOp::Max);
            }
            _ => {}
        }
        op_id = kernel.next_op(op_id);
    }

    if let (Some(ad), Some(as_), Some(b)) = (acc_define, acc_store, bop) {
        Some((ad, as_, b))
    } else {
        None
    }
}

fn should_optimize(loop_len: Dim) -> bool {
    loop_len >= 32
}
