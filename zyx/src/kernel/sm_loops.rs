// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

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

    pub fn get_global_indices(&self) -> BTreeMap<u32, OpId> {
        let mut indices = BTreeMap::new();
        for (op_id, op_node) in self.ops.iter() {
            if let Op::Index { scope, axis, .. } = op_node.op {
                if scope == Scope::Global {
                    indices.insert(axis, op_id);
                }
            }
        }
        indices
    }

    /// Merges two or more indices together
    pub fn merge_indices(&mut self, loops: &[OpId]) {
        //println!("Merging loops {loops:?}");
        let mut acc = 1;
        // BTreeMap is ordered
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
            //println!("len={len}, x={x}, y={y}, loop_id={loop_id}");
            self.ops[loop_id].op = Op::Binary { x, y, bop: BOp::Mod };
            x = self.insert_after(loop_id, Op::Binary { x, y, bop: BOp::Div });
        }

        self.verify();
    }

    /// Splits dim (index or loop) into multiple indices or loops
    /// Returns the OpIds of the created split operations in the order they were provided
    pub fn split_dim(&mut self, dim_id: OpId, mut splits: Vec<Op>) -> Vec<OpId> {
        //println!("splitting dim_id={dim_id}, splits={splits:?}");
        let is_loop = matches!(self.ops[dim_id].op, Op::Loop { .. });

        #[cfg(debug_assertions)]
        {
            let mut dim = 1;
            for op in splits.iter() {
                match op {
                    Op::Loop { len, .. } | Op::Index { len, .. } => dim *= len,
                    _ => unreachable!("split can be only index or loop"),
                }
            }
            match self.ops[dim_id].op {
                Op::Index { len, .. } | Op::Loop { len, .. } => debug_assert_eq!(len, dim),
                _ => {}
            }
        }

        let last_dim_op = self.get_last_dim_op(dim_id);
        let n_loops = splits.iter().filter(|op| matches!(op, Op::Loop { .. })).count();
        for (i, op) in splits.iter().enumerate() {
            if matches!(op, Op::Loop { .. }) {
                // If splitting a loop, add one fewer EndLoop (original loop has its own)
                if is_loop && i == n_loops - 1 {
                    // Don't add EndLoop for the last loop when splitting an existing loop
                } else {
                    self.insert_after(last_dim_op, Op::EndLoop);
                }
            }
        }

        // 12 - > 2, 2, 4
        // 0..12 - > 0..2 * st + 2 * st +  4 * st

        // Get strides
        let mut strides = Vec::new();
        let mut st = 1;
        for op in splits.iter().rev() {
            strides.push(st);
            match op {
                Op::Loop { len, .. } | Op::Index { len, .. } => st *= len,
                _ => unreachable!(),
            }
        }
        strides.reverse();
        strides.pop(); // skip stride 1
        let last_op = splits.pop().unwrap();

        // Insert splits and collect their OpIds
        // Splits are inserted in reverse order (last to first) because insert_before
        // always inserts before dim_id, so later insertions end up earlier in the sequence
        let mut split_ids: Vec<OpId> = Vec::new();
        let mut acc = self.insert_before(dim_id, Op::Const(Constant::idx(0)));
        for (&st, op) in strides.iter().zip(splits) {
            let x = self.insert_before(dim_id, Op::Const(Constant::idx(st as u64)));
            let y = self.insert_before(dim_id, op);
            acc = self.insert_before(dim_id, Op::Mad { x, y, z: acc });
            split_ids.push(y);
        }

        // Replace previous op
        let y = self.insert_before(dim_id, last_op);
        split_ids.push(y);
        self.ops[dim_id].op = Op::Binary {
            x: acc,
            y,
            bop: BOp::Add,
        };

        // Reverse to get the original order (first split first)
        split_ids.reverse();

        self.verify();
        split_ids
    }
}
