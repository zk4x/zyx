// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use super::autotune::Optimization;
use crate::{
    Map,
    kernel::{BOp, Kernel, Op},
};

impl Kernel {
    pub const fn opt_fuse_mad(_: &Kernel) -> (Optimization, usize) {
        (Optimization::FuseMad, 1)
    }

    /// Find all multiply add operations and fuse them
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

    #[allow(clippy::unused_self)]
    pub const fn opt_unfuse_mad(&self) -> (Optimization, usize) {
        (Optimization::UnfuseMad, 1)
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
