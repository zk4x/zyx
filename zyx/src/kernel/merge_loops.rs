// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: GPL-2.0-only

use std::collections::BTreeMap;

use crate::{
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, Scope},
};

impl Kernel {
    /// Get last op in the given loop scope
    pub fn get_last_dim_op(&self, loop_id: OpId) -> OpId {
        match self.ops[loop_id].op {
            Op::Index { .. } => return self.tail,
            Op::Loop { .. } => {}
            _ => unreachable!(),
        }
        let mut loop_depth = 0;
        let mut op_id = loop_id;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Loop { .. } => {
                    loop_depth += 1;
                }
                Op::EndLoop => {
                    loop_depth -= 1;
                    if loop_depth == 0 {
                        return op_id;
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        op_id
    }

    /// Merges two or more indices together
    pub fn merge_indices(&mut self, loops: &[OpId]) {
        let mut acc = 1;
        let mut axes = BTreeMap::default();
        let mut first_id = None;
        let mut op_id = self.head;
        while axes.len() != loops.len() {
            if loops.contains(&op_id) {
                let Op::Index { len, scope, axis } = self.ops[op_id].op else {
                    unreachable!()
                };
                debug_assert_eq!(scope, Scope::Global);
                acc *= len;
                axes.insert(axis, (op_id, len));
                if first_id.is_none() {
                    first_id = Some(op_id);
                }
            }
            op_id = self.next_op(op_id);
        }

        let Op::Index { scope, axis, .. } = self.ops[first_id.unwrap()].op else {
            unreachable!()
        };
        let mut x = self.insert_before(first_id.unwrap(), Op::Index { len: acc, scope, axis });

        for (.., (loop_id, len)) in axes.into_iter() {
            let y = self.insert_before(loop_id, Op::Const(Constant::idx(len as u64)));
            self.ops[loop_id].op = Op::Binary { x, y, bop: BOp::Mod };
            x = self.insert_after(loop_id, Op::Binary { x, y, bop: BOp::Div });
        }

        self.verify();
    }
}
