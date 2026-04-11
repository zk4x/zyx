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

        println!("loop_id={loop_id}");

        self.verify();
    }
}
