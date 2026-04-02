// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use super::autotune::Optimization;
use crate::dtype::Constant;
use crate::kernel::{BOp, Kernel, Op, OpId, Scope};

impl Kernel {
    pub fn opt_warp_reduce(&self) -> (Optimization, usize) {
        let candidates = vec![32, 16, 8, 64];
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Loop { len } = self.ops[op_id].op {
                // No point in doing this with small loops
                if len >= 256 {
                    for &factor in &candidates {
                        // no point in this if second loop is too large
                        if len.is_multiple_of(factor) && len / factor >= factor {
                            factors.push((op_id, factor));
                        }
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        let n_configs = factors.len();
        (Optimization::WarpReduce { factors }, n_configs)
    }

    pub fn optimize_warp_reduce_with(&mut self, loop_start: OpId, factor: usize) {
        let Some(loop_end) = find_reduce_loop_end(self, loop_start) else {
            return;
        };

        let loop_len = if let Op::Loop { len } = self.at(loop_start) {
            *len
        } else {
            return;
        };

        let Some((acc_define, acc_store, bop)) = find_reduce_pattern(self, loop_start, loop_end) else {
            return;
        };

        let outer_len = loop_len / factor;

        // Replace register accumulator with local memory accumulator
        // This is the key change: just modify the existing Define op
        self.ops[acc_define].op = Op::Define {
            dtype: if let Op::Define { dtype, .. } = self.at(acc_define) {
                *dtype
            } else {
                return;
            },
            scope: Scope::Local,
            ro: false,
            len: factor,
        };

        // Create local index for accessing the accumulator
        let lidx = self.insert_before(
            loop_start,
            Op::Index {
                len: factor,
                scope: Scope::Local,
                axis: 0,
            },
        );

        // Change the store inside the loop to use local index
        if let Op::Store { x, vlen, .. } = self.at(acc_store).clone() {
            self.ops[acc_store].op = Op::Store {
                dst: acc_define,
                x,
                index: lidx,
                vlen,
            };
        }

        // Find and update the load from acc_define inside the loop to use lidx
        let mut op_id = self.next_op(loop_start);
        while op_id != loop_end {
            if let Op::Load { src, index: _, vlen } = self.at(op_id).clone() {
                if src == acc_define {
                    self.ops[op_id].op = Op::Load {
                        src: acc_define,
                        index: lidx,
                        vlen,
                    };
                }
            }
            op_id = self.next_op(op_id);
        }

        // Modify the original loop to iterate by factor
        self.ops[loop_start].op = Op::Loop { len: outer_len };

        // Find the index computation inside the loop and modify it to stride by factor
        let mut op_id = self.next_op(loop_start);
        while op_id != loop_end {
            if let Op::Binary { x: _, y, bop: BOp::Add } = self.at(op_id).clone() {
                if let Op::Binary {
                    x: x2,
                    y: y2,
                    bop: BOp::Mul,
                } = self.at(y).clone()
                {
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
                        x: y2,
                        y: new_offset,
                        bop: BOp::Add,
                    };
                    break;
                }
            }
            op_id = self.next_op(op_id);
        }

        // After the original loop, add barrier
        let after_loop_end = self.next_op(loop_end);
        self.insert_before(after_loop_end, Op::Barrier { scope: Scope::Local });

        // idx_zero for the second loop
        let idx_zero = self.insert_before(after_loop_end, Op::Const(Constant::idx(0)));

        // Second loop: thread 0 reduces across all slots in local memory
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

        // Inner loop index
        let inner_idx = self.insert_before(
            after_loop_end,
            Op::Index {
                len: factor,
                scope: Scope::Local,
                axis: 0,
            },
        );

        // Load current accumulator value (slot 0)
        let load_acc = self.insert_before(
            after_loop_end,
            Op::Load {
                src: acc_define,
                index: idx_zero,
                vlen: 1,
            },
        );

        // Load next partial result
        let load_partial = self.insert_before(
            after_loop_end,
            Op::Load {
                src: acc_define,
                index: inner_idx,
                vlen: 1,
            },
        );

        // Combine and store back to slot 0
        let bin_result = self.insert_before(
            after_loop_end,
            Op::Binary {
                x: load_acc,
                y: load_partial,
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

    let Some(acc_def) = acc_define else {
        return None;
    };

    op_id = kernel.next_op(loop_start);
    while op_id != loop_end {
        match kernel.at(op_id) {
            Op::Store { dst, .. } => {
                if *dst == acc_def {
                    acc_store = Some(op_id);
                }
            }
            Op::Binary { x, y, bop: op_bop } => {
                if *x == acc_def || *y == acc_def {
                    bop = Some(*op_bop);
                }
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
