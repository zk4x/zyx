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
            if self.fold_arange_loop(op_id) {
                break;
            }
            op_id = self.next_op(op_id);
        }

        self.verify();
    }

    fn fold_arange_loop(&mut self, def_id: OpId) -> bool {
        // Define
        let &Op::Define { dtype: acc_dtype, scope: Scope::Register, ro: false, len: 1 } = self.at(def_id) else { return false };
        //println!("def_id={def_id}");

        // Find first store to def_id (initial value)
        let mut store_id = self.next_op(def_id);
        while !store_id.is_null() {
            if let &Op::Store { dst, index, .. } = self.at(store_id) {
                if dst == def_id {
                    if let Op::Const(cst) = self.at(index) {
                        if let Some(0) = cst.as_dim() {
                            break;
                        }
                    }
                }
            }
            store_id = self.next_op(store_id);
        }
        if store_id.is_null() {
            return false;
        }
        //println!("store_id={store_id}");

        // Find loop
        let mut loop_id = self.next_op(store_id);
        while !loop_id.is_null() {
            if matches!(self.at(loop_id), Op::Loop { .. }) {
                break;
            }
            match self.at(loop_id) {
                Op::Load { src, .. } if *src == def_id => return false,
                Op::Store { dst, .. } if *dst == def_id => return false,
                _ => {}
            }
            loop_id = self.next_op(loop_id);
        }
        let Op::Loop { .. } = self.at(loop_id) else { return false };
        //println!("loop_id={loop_id}");

        // After the loop is found, check add - one operand is load, other is something
        let add_id = self.next_op(loop_id);
        let &Op::Binary { bop: BOp::Add, x, y: gidx_id, .. } = self.at(add_id) else { return false };
        if x != loop_id {
            return false;
        }
        let &Op::Index { len: gidx_len, scope: Scope::Global, axis: gidx_axis } = self.at(gidx_id) else {
            return false;
        };
        //println!("add_id={add_id}");

        // cmpgt
        let cmpgt_id = self.next_op(add_id);
        let &Op::Binary { x, y, bop: BOp::Cmpgt } = self.at(cmpgt_id) else { return false };
        if x != add_id {
            return false;
        }
        let &Op::Const(threshold) = self.at(y) else { return false };
        //println!("cmpgt_id={cmpgt_id}");

        // Cast
        let cast_id = self.next_op(cmpgt_id);
        let &Op::Cast { x, dtype } = self.at(cast_id) else { return false };
        if x != cmpgt_id || dtype != acc_dtype {
            return false;
        }
        //println!("cast_id={cast_id}");

        // Mul
        let mul_id = self.next_op(cast_id);
        let &Op::Binary { x, y, bop: BOp::Mul } = self.at(mul_id) else { return false };
        if x != cast_id {
            return false;
        }
        let &Op::Const(mul_const) = self.at(y) else { return false };
        //println!("mul_id={mul_id}");

        // Load
        let load_id = self.next_op(mul_id);
        let &Op::Load { src, index, vlen: 1 } = self.at(load_id) else { return false };
        let &Op::Const(index) = self.at(index) else { return false };
        if index.as_dim() != Some(0) {
            return false;
        }
        if src != def_id {
            return false;
        }
        //println!("load_id={load_id}");

        // Add
        let add2_id = self.next_op(load_id);
        let &Op::Binary { x, y, bop: BOp::Add } = self.at(add2_id) else { return false };
        if x != mul_id || y != load_id {
            return false;
        }
        //println!("add2_id={add2_id}");

        // Store
        let store2_id = self.next_op(add2_id);
        let &Op::Store { dst, x, index, vlen: 1 } = self.at(store2_id) else { return false };
        let &Op::Const(index) = self.at(index) else { return false };
        if index.as_dim() != Some(0) {
            return false;
        }
        if dst != def_id || x != add2_id {
            return false;
        }
        println!("store2_id={store2_id}");

        // Endloop
        let endloop_id = self.next_op(store2_id);
        let Op::EndLoop = self.at(endloop_id) else { return false };
        println!("endloop_id={endloop_id}");

        // Load
        let load2_id = self.next_op(endloop_id);
        let &Op::Load { src, index, vlen: 1 } = self.at(load2_id) else { return false };
        let &Op::Const(index) = self.at(index) else { return false };
        if index.as_dim() != Some(0) {
            return false;
        }
        if src != def_id {
            return false;
        }
        println!("load2_id={load2_id}");

        self.ops[load2_id].op = Op::Cast { x: gidx_id, dtype: acc_dtype };

        self.remove_op(def_id);
        self.remove_op(store_id);
        self.remove_op(loop_id);
        self.remove_op(add_id);
        self.remove_op(cmpgt_id);
        self.remove_op(cast_id);
        self.remove_op(mul_id);
        self.remove_op(load_id);
        self.remove_op(add2_id);
        self.remove_op(store2_id);
        self.remove_op(endloop_id);

        self.verify();

        true
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
