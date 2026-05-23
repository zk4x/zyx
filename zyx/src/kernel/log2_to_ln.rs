// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    dtype::Constant,
    kernel::{BOp, Kernel, Op, UOp},
};
use half::{bf16, f16};

const LN_2: f64 = 0.693_147_180_559_945_3;

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
                let (log2_op, const_op) = match (self.at(left), self.at(right)) {
                    (&Op::Unary { x: input, uop: UOp::Log2 }, c) => (input, c),
                    (c, &Op::Unary { x: input, uop: UOp::Log2 }) => (input, c),
                    _ => { op_id = next; continue; }
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
