// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Fuse multiply-add operations.
//!
//! This module provides optimization for fusing multiply-add (MAD) operations,
//! which combines `x * y + z` patterns into a single MAD instruction.
//! This reduces instruction count and can improve performance.

use crate::{
    Map,
    kernel::{Kernel, Op, UOp},
};

impl Kernel {
    /// Fuse reciprocal-of-square-root into a single rsqrt.
    ///
    /// This method identifies `1/sqrt(x)` spelled as
    /// `Reciprocal(Sqrt(x))` and fuses it into one `Rsqrt`, which
    /// lowers to a single tile op (`rsqrt_tile`, `rsqrt.approx`,
    /// `InverseSqrt`, ...) instead of two.
    ///
    /// Like [`Kernel::fuse_mad`], the inner sqrt must be single-use
    /// (reference count 1).
    pub fn fuse_rsqrt(&mut self) {
        let mut op_id = self.head;
        let mut rcs = Map::default();
        while !op_id.is_null() {
            for param in self.ops[op_id].op.parameters() {
                rcs.entry(param).and_modify(|rc| *rc += 1).or_insert(1);
            }
            if let Op::Unary { x: xo, uop } = self.ops[op_id].op
                && uop == UOp::Reciprocal
                && let Op::Unary { x, uop } = self.ops[xo].op
                && uop == UOp::Sqrt
                && rcs[&xo] == 1
            {
                self.ops[op_id].op = Op::Unary { x, uop: UOp::Rsqrt };
            }
            op_id = self.next_op(op_id);
        }

        self.verify();
    }
}
