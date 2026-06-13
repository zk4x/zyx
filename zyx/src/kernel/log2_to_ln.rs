// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Convert log2(x) * ln(2) to ln(x).
//!
//! This module provides optimization for converting `log2(x) * ln(2)`
//! to `ln(x)`, which allows backends that lack native `log2` to use
//! a native `ln` function instead.

use crate::{
    dtype::Constant,
    kernel::{BOp, Kernel, Op, UOp},
};
use half::{bf16, f16};

const LN_2: f64 = std::f64::consts::LN_2;

fn constant_is_ln_2(c: &Constant) -> bool {
    let val = match *c {
        Constant::BF16(x) => bf16::from_le_bytes(x).to_f32() as f64,
        Constant::F16(x) => f16::from_le_bytes(x).to_f32() as f64,
        Constant::F32(x) => f32::from_le_bytes(x) as f64,
        Constant::F64(x) => f64::from_le_bytes(x),
        _ => return false,
    };
    (val - LN_2).abs() < 1e-6
}

impl Kernel {
    /// Finds `log2(x) * ln(2)` and replaces it with `ln(x)`.
    ///
    /// This recognizes the pattern produced by `tensor.ln()` which is
    /// implemented as `log2(x) * (1/log2(e))` = `log2(x) * ln(2)`.
    /// Converting to `ln` allows backends that lack native `log2` to
    /// use a native `ln` function instead.
    pub fn log2_to_ln(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let &Op::Binary { x: left, y: right, bop: BOp::Mul } = self.at(op_id) {
                let ((&Op::Unary { x: log2_op, uop: UOp::Log2 }, const_op)
                | (const_op, &Op::Unary { x: log2_op, uop: UOp::Log2 })) = (self.at(left), self.at(right))
                else {
                    op_id = next;
                    continue;
                };
                if let &Op::Const(c) = const_op {
                    if constant_is_ln_2(&c) {
                        self.ops[op_id].op = Op::Unary { x: log2_op, uop: UOp::Ln };
                    }
                }
            }
            op_id = next;
        }
    }
}
