// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    Set,
    kernel::{Kernel, Op, OpId},
};

impl Kernel {
    pub fn instruction_schedule(&mut self) {
        let mut index_ops: Set<OpId> = Set::default();
        let mut insert_after: OpId = OpId::NULL;

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match &self.ops[op_id].op {
                Op::Define { .. } => insert_after = op_id,
                Op::Const(_) | Op::Index { .. } | Op::Loop { .. } => {
                    index_ops.insert(op_id);
                    if !insert_after.is_null() && insert_after != op_id
                        && !matches!(self.ops[op_id].op, Op::Loop { .. })
                    {
                        self.move_op_after(op_id, insert_after);
                    }
                    insert_after = op_id;
                }
                Op::Load { index, .. } => {
                    if index_ops.contains(index) && !insert_after.is_null() {
                        self.move_op_after(op_id, insert_after);
                        insert_after = op_id;
                    }
                }
                Op::EndLoop | Op::EndIf => {
                    insert_after = op_id;
                }
                _ => {
                    let params: Vec<OpId> = self.ops[op_id].op.parameters().collect();
                    if !params.is_empty() && params.iter().all(|p| index_ops.contains(p)) {
                        index_ops.insert(op_id);
                        if !insert_after.is_null() {
                            self.move_op_after(op_id, insert_after);
                        }
                        insert_after = op_id;
                    }
                }
            }
            op_id = next;
        }

        #[cfg(debug_assertions)]
        self.verify();
    }
}
