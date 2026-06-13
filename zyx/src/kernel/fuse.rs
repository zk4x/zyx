// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Fuse multiply-add operations.
//!
//! This module provides optimization for fusing multiply-add (MAD) operations,
//! which combines `x * y + z` patterns into a single MAD instruction.
//! This reduces instruction count and can improve performance.

use crate::{
    Map,
    kernel::{BOp, Kernel, Op},
};

impl Kernel {
    /// Fuse multiply-add operations into MAD instructions.
    ///
    /// This method identifies patterns of the form `x * y + z` and
    /// fuses them into a single MAD instruction, reducing instruction
    /// count and potentially improving performance.
    ///
    /// The optimization looks for:
    ///
    /// - Binary add where one operand is a multiply
    /// - The multiply has a reference count of 1 (used only once)
    /// - The multiply and add can be fused into a MAD
    pub fn fuse_mad(&mut self) {
        let mut op_id = self.head;
        let mut rcs = Map::default();
        while !op_id.is_null() {
            for param in self.ops[op_id].op.parameters() {
                rcs.entry(param).and_modify(|rc| *rc += 1).or_insert(1);
            }
            if let Op::Binary { x: xo, y: yo, bop } = self.ops[op_id].op {
                if bop == BOp::Add {
                    if let Op::Binary { x, y, bop } = self.ops[xo].op {
                        if bop == BOp::Mul && rcs[&xo] == 1 {
                            self.ops[op_id].op = Op::Mad { x, y, z: yo };
                        }
                    } else if let Op::Binary { x, y, bop } = self.ops[yo].op {
                        if bop == BOp::Mul && rcs[&yo] == 1 {
                            self.ops[op_id].op = Op::Mad { x, y, z: xo };
                        }
                    }
                }
            }
            op_id = self.next_op(op_id);
        }

        self.verify();
    }

    /// Find all multiply add operations and unfuse them
    pub fn unfuse_mad(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Mad { x, y, z } = self.ops[op_id].op {
                let x = self.insert_before(op_id, Op::Binary { x, y, bop: BOp::Mul });
                self.ops[op_id].op = Op::Binary { x, y: z, bop: BOp::Add };
            }
            op_id = self.next_op(op_id);
        }
    }
}
