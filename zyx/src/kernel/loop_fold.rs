// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: GPL-2.0-only

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
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Define { len, scope, dtype, .. } = self.at(op_id).clone() {
                if len == 1 && scope == Scope::Register {
                    self.try_fold(op_id, dtype);
                }
            }
            op_id = self.next_op(op_id);
        }

        self.verify();
    }

    fn try_fold(&mut self, def_id: OpId, acc_dtype: DType) {
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
