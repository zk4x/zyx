// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    dtype::{Constant, DType},
    kernel::{BOp, Kernel, Op, OpId, Scope},
};

impl Kernel {
    pub fn simplify_accumulating_loop(&mut self) {
        // Currently only try_fold_conditional is used (for arange pattern)
        // try_fold1 is disabled as it may cause issues
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Define { len, scope, .. } = self.at(op_id).clone() {
                if len == 1 && scope == Scope::Register {
                    // Only try fold on arange-like patterns
                    self.try_fold_conditional(op_id);
                }
            }
            op_id = self.next_op(op_id);
        }

        self.verify();
    }

    fn try_fold_conditional(&mut self, def_id: OpId) {
        // Find first store to def_id (initial value)
        let mut store1_id = self.next_op(def_id);
        while !store1_id.is_null() {
            if let Op::Store { dst, index, .. } = self.at(store1_id) {
                if *dst == def_id {
                    if let Op::Const(cst) = self.at(*index) {
                        if let Some(0) = cst.as_dim() {
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

        println!("store1_id={store1_id}");

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

        println!("loop_id={loop_id}");

        // Find Load of accumulator (at index 0) in loop body
        let load_id = self.next_op(loop_id);
        let Op::Load { src: load_src, index: load_idx, vlen: 1 } = self.at(load_id) else {
            return;
        };
        if *load_src != def_id {
            return;
        }
        let Op::Const(cst) = self.at(*load_idx) else {
            return;
        };
        if cst.as_dim() != Some(0) {
            return;
        }

        println!("load_id={load_id}");

        // Find Add after Load
        let add_id = self.next_op(load_id);
        let Op::Binary { bop: BOp::Add, x: add_x, y: add_y, .. } = self.at(add_id) else {
            return;
        };
        if *add_x != load_id && *add_y != load_id {
            return;
        }

        println!("add_id={add_id}");

        // Get computed value (the other operand of Add)
        let computed_val = if *add_x == load_id { *add_y } else { *add_x };

        println!("computed_val={computed_val}");

        // Check: computed_val should be Mul(Cast(Cmpgt(...)), step)
        let Op::Binary { bop: BOp::Mul, x: mul_x, y: mul_y, .. } = self.at(computed_val) else {
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

        println!("cast_id={cmp_id}");

        // Check: cast
        let Op::Cast { x: cmp_input, .. } = self.at(cmp_id) else {
            return;
        };

        // Check: compare(add(loop_var, gidx), threshold
        let cmp_compare_id = *cmp_input;
        let Op::Binary { bop: BOp::Cmpgt, x: cmp_x, y: cmp_y, .. } = self.at(cmp_compare_id) else {
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
        if let Some(0) = cst.as_dim() {
            return;
        }

        println!("store_id={store_id}");

        // Check endloop
        let endloop_id = self.next_op(store_id);
        if !matches!(self.at(endloop_id), Op::EndLoop) {
            return;
        }

        println!("endloop_id={endloop_id}");

        // Check final load
        let load2_id = self.next_op(endloop_id);
        if !matches!(self.at(load2_id), Op::Load { .. }) {
            return;
        }
        println!("load2_id={load2_id}");
        let load2_op = self.at(load2_id).clone();
        let load2_idx = if let Op::Load { src: load2_src, index, vlen: 1 } = load2_op {
            if load2_src != def_id {
                return;
            }
            let Op::Const(cst) = self.at(index) else {
                return;
            };
            if let Some(0) = cst.as_dim() {
                return;
            }
            index
        } else {
            return;
        };

        println!("PATTERN FOUND: store_id={store1_id}");

        // Now replace the loop with conditional computation

        self.verify();
    }

    fn try_fold1(&mut self, def_id: OpId, acc_dtype: DType) {
        // Skip ops until first store to def_id (initial value)
        let mut store1_id = self.next_op(def_id);
        while !store1_id.is_null() {
            if let Op::Store { dst, index, .. } = self.at(store1_id) {
                if *dst == def_id {
                    if let Op::Const(cst) = self.at(*index) {
                        if let Some(0) = cst.as_dim() {
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
        if cst.as_dim() != Some(0) {
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
        if cst.as_dim() != Some(0) {
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
        if cst.as_dim() != Some(0) {
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
