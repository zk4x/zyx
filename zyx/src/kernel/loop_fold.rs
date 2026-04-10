// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    dtype::{Constant, DType},
    kernel::{BOp, Kernel, Op, OpId, Scope},
};

fn constant_as_u64(c: &Constant) -> Option<u64> {
    match c {
        Constant::U32(x) => Some(*x as u64),
        Constant::U64(x) => Some(u64::from_le_bytes(*x)),
        _ => None,
    }
}

impl Kernel {
    pub fn simplify_accumulating_loop(&mut self) {
        // Currently only try_fold_conditional is used (for arange pattern)
        // try_fold1 is disabled as it may cause issues
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Define { len, scope, dtype, .. } = self.at(op_id).clone() {
                if len == 1 && scope == Scope::Register {
                    // Only try fold on arange-like patterns
                    self.try_fold_conditional(op_id, dtype);
                }
            }
            op_id = self.next_op(op_id);
        }

        self.verify();
    }

    fn try_fold_conditional(&mut self, def_id: OpId, acc_dtype: DType) {
        // Find first store to def_id (initial value)
        let mut store1_id = self.next_op(def_id);
        while !store1_id.is_null() {
            if let Op::Store { dst, index, .. } = self.at(store1_id) {
                if *dst == def_id {
                    if let Op::Const(cst) = self.at(*index) {
                        if constant_as_u64(cst) == Some(0) {
                            break;
                        }
                    }
                }
            }
            store1_id = self.next_op(store1_id);
        }
        if store1_id.is_null() {
            return;
        }

        // Find loop
        let mut loop_id = self.next_op(store1_id);
        while !loop_id.is_null() {
            if matches!(self.at(loop_id), Op::Loop { .. }) {
                break;
            }
            match self.at(loop_id) {
                Op::Load { src, .. } if *src == def_id => return,
                Op::Store { dst, .. } if *dst == def_id => return,
                _ => {}
            }
            loop_id = self.next_op(loop_id);
        }
        if !matches!(self.at(loop_id), Op::Loop { .. }) {
            return;
        }

        let loop_len = match self.at(loop_id) {
            Op::Loop { len, .. } => *len,
            _ => return,
        };

        // Find Load of accumulator (at index 0) in loop body
        let mut search_id = self.next_op(loop_id);
        let mut load_id = None;
        while !search_id.is_null() {
            if let Op::Load { src, index, vlen: 1 } = self.at(search_id) {
                if *src == def_id {
                    if let Op::Const(cst) = self.at(*index) {
                        if constant_as_u64(cst) == Some(0) {
                            load_id = Some(search_id);
                            break;
                        }
                    }
                }
            }
            search_id = self.next_op(search_id);
        }
        let load_id = match load_id {
            Some(id) => id,
            None => return,
        };

        // Find Add after Load
        let add_id = self.next_op(load_id);
        if !matches!(self.at(add_id), Op::Binary { bop: BOp::Add, .. }) {
            return;
        }
        let Op::Binary { x: add_x, y: add_y, .. } = self.at(add_id) else {
            return;
        };
        if *add_x != load_id && *add_y != load_id {
            return;
        }

        // Get computed value (the other operand of Add)
        let computed_val = if *add_x == load_id { *add_y } else { *add_x };

        // Check: computed_val should be Mul(Cast(Cmpgt(...)), step)
        if !matches!(self.at(computed_val), Op::Binary { bop: BOp::Mul, .. }) {
            return;
        }
        let Op::Binary { x: mul_x, y: mul_y, .. } = self.at(computed_val) else {
            return;
        };

        // Check: cast(compare(...))
        let (cmp_id, step) = if matches!(self.at(*mul_x), Op::Cast { .. }) {
            (*mul_x, *mul_y)
        } else if matches!(self.at(*mul_y), Op::Cast { .. }) {
            (*mul_y, *mul_x)
        } else {
            return;
        };

        // Check: cast
        let Op::Cast { x: cmp_input, .. } = self.at(cmp_id) else {
            return;
        };

        // Check: compare(add(loop_var, gidx), threshold)
        let cmp_compare_id = *cmp_input;
        if !matches!(self.at(cmp_compare_id), Op::Binary { bop: BOp::Cmpgt, .. }) {
            return;
        }
        let Op::Binary { x: cmp_x, y: cmp_y, .. } = self.at(cmp_compare_id) else {
            return;
        };

        // Check: add(loop_var, gidx)
        let add_compare_id = *cmp_x;
        if !matches!(self.at(add_compare_id), Op::Binary { bop: BOp::Add, .. }) {
            return;
        }
        let Op::Binary { x: add_cx, y: add_cy, .. } = self.at(add_compare_id) else {
            return;
        };

        let gidx_id = *add_cy;
        let threshold_id = *cmp_y;

        // Check Store
        let store_id = self.next_op(add_id);
        if !matches!(self.at(store_id), Op::Store { .. }) {
            return;
        }
        let Op::Store { dst: store_dst, index: store_idx, .. } = self.at(store_id) else {
            return;
        };
        if *store_dst != def_id {
            return;
        }
        let Op::Const(cst) = self.at(*store_idx) else {
            return;
        };
        if constant_as_u64(cst) != Some(0) {
            return;
        }

        // Check endloop
        let endloop_id = self.next_op(store_id);
        if !matches!(self.at(endloop_id), Op::EndLoop) {
            return;
        }

        // Check final load
        let load2_id = self.next_op(endloop_id);
        if !matches!(self.at(load2_id), Op::Load { .. }) {
            return;
        }
        let load2_op = self.at(load2_id).clone();
        let load2_idx = if let Op::Load { src: load2_src, index, vlen: 1 } = load2_op {
            if load2_src != def_id {
                return;
            }
            let Op::Const(cst) = self.at(index) else {
                return;
            };
            if constant_as_u64(cst) != Some(0) {
                return;
            }
            index
        } else {
            return;
        };

        // Now replace the loop with conditional computation
        // result = (gidx > threshold ? loop_len : threshold - gidx + 1) * step

        // cond = gidx > threshold (result is bool, but we need it as int for multiply)
        let cond_id = self.insert_before(load2_id, Op::Binary { x: gidx_id, y: threshold_id, bop: BOp::Cmpgt });

        // Cast cond to integer for arithmetic
        let cond_i32_id = self.insert_before(load2_id, Op::Cast { x: cond_id, dtype: DType::I32 });

        // Cast gidx to i32 for arithmetic with i32 constants
        let gidx_i32_id = self.insert_before(load2_id, Op::Cast { x: gidx_id, dtype: DType::I32 });

        // Create constants - using I32 dtype
        let loop_len_const = Constant::new(loop_len as i32);
        let loop_len_id = self.insert_before(load2_id, Op::Const(loop_len_const));

        // b = threshold - gidx + 1 = (threshold + 1) - gidx (will be negative for gidx > threshold)
        let threshold_plus_1 = 9999i32;
        let b_const = Constant::new(threshold_plus_1);
        let b_const_id = self.insert_before(load2_id, Op::Const(b_const));
        let b_id = self.insert_before(load2_id, Op::Binary { x: b_const_id, y: gidx_i32_id, bop: BOp::Sub });

        // result = loop_len * cond + b * (1 - cond)
        // = b + cond * (loop_len - b)
        let diff_id = self.insert_before(load2_id, Op::Binary { x: loop_len_id, y: b_id, bop: BOp::Sub });
        let cond_mul_diff_id = self.insert_before(load2_id, Op::Binary { x: cond_i32_id, y: diff_id, bop: BOp::Mul });
        let result_id = self.insert_before(load2_id, Op::Binary { x: b_id, y: cond_mul_diff_id, bop: BOp::Add });

        // Multiply by step
        let result_with_step_id = self.insert_before(load2_id, Op::Binary { x: result_id, y: step, bop: BOp::Mul });

        // Replace load2 (which loaded from accumulator) with the result
        self.remap(load2_id, result_with_step_id);

        self.verify();
    }

    fn try_fold1(&mut self, def_id: OpId, acc_dtype: DType) {
        // Skip ops until first store to def_id (initial value)
        let mut store1_id = self.next_op(def_id);
        while !store1_id.is_null() {
            if let Op::Store { dst, index, .. } = self.at(store1_id) {
                if *dst == def_id {
                    if let Op::Const(cst) = self.at(*index) {
                        if constant_as_u64(cst) == Some(0) {
                            break;
                        }
                    }
                }
            }
            store1_id = self.next_op(store1_id);
        }
        if store1_id.is_null() {
            return;
        }

        // Skip ops until loop
        let mut loop_id = self.next_op(store1_id);
        while !loop_id.is_null() {
            if matches!(self.at(loop_id), Op::Loop { .. }) {
                break;
            }

            // Check for any other loads/stores from/to def_id before loop - if so, abort
            match self.at(loop_id) {
                Op::Load { src, .. } if *src == def_id => {
                    return;
                }
                Op::Store { dst, .. } if *dst == def_id => {
                    return;
                }
                _ => {}
            }
            loop_id = self.next_op(loop_id);
        }
        if !matches!(self.at(loop_id), Op::Loop { .. }) {
            return;
        }

        let loop_len = match self.at(loop_id) {
            Op::Loop { len, .. } => *len,
            _ => return,
        };

        // Check loop body: load, add, store (in that order, no other ops)
        let load_id = self.next_op(loop_id);
        if !matches!(self.at(load_id), Op::Load { .. }) {
            return;
        }
        let Op::Load { src: load_src, index: load_idx, vlen: 1 } = self.at(load_id) else {
            return;
        };
        if *load_src != def_id {
            return;
        }
        let Op::Const(cst) = self.at(*load_idx) else {
            return;
        };
        if constant_as_u64(cst) != Some(0) {
            return;
        }

        let add_id = self.next_op(load_id);
        if !matches!(self.at(add_id), Op::Binary { bop: BOp::Add, .. }) {
            return;
        }
        let Op::Binary { x: add_x, y: add_y, .. } = self.at(add_id) else {
            return;
        };
        let add_val = if *add_x == load_id {
            *add_y
        } else if *add_y == load_id {
            *add_x
        } else {
            return;
        };

        let store_id = self.next_op(add_id);
        if !matches!(self.at(store_id), Op::Store { .. }) {
            return;
        }
        let Op::Store { dst: store_dst, index: store_idx, .. } = self.at(store_id) else {
            return;
        };
        if *store_dst != def_id {
            return;
        }
        let Op::Const(cst) = self.at(*store_idx) else {
            return;
        };
        if constant_as_u64(cst) != Some(0) {
            return;
        }

        // Check endloop
        let endloop_id = self.next_op(store_id);
        if !matches!(self.at(endloop_id), Op::EndLoop) {
            return;
        }

        // Check final load
        let load2_id = self.next_op(endloop_id);
        if !matches!(self.at(load2_id), Op::Load { .. }) {
            return;
        }
        let Op::Load { src: load2_src, index: load2_idx, vlen: 1 } = self.at(load2_id) else {
            return;
        };
        if *load2_src != def_id {
            return;
        }
        let Op::Const(cst) = self.at(*load2_idx) else {
            return;
        };
        if constant_as_u64(cst) != Some(0) {
            return;
        }

        // Fold: replace final load with loop_len * add_val (using acc dtype)
        let loop_len_const = Constant::idx(loop_len as u64).cast(acc_dtype);
        let loop_len_id = self.insert_before(load2_id, Op::Const(loop_len_const));

        self.ops[load2_id].op = Op::Binary { x: loop_len_id, y: add_val, bop: BOp::Mul };

        self.remove_op(endloop_id);
        self.remove_op(store_id);
        self.remove_op(add_id);
        self.remove_op(load_id);
        self.remove_op(loop_id);
        self.remove_op(store1_id);
    }
}
