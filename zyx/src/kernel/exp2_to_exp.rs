// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, UOp},
};
use half::{bf16, f16};

const LOG2_E: f64 = std::f64::consts::LOG2_E;

fn constant_is_log2_e(c: &Constant) -> bool {
    let val = match *c {
        Constant::BF16(x) => bf16::from_le_bytes(x).to_f32() as f64,
        Constant::F16(x) => f16::from_le_bytes(x).to_f32() as f64,
        Constant::F32(x) => f32::from_le_bytes(x) as f64,
        Constant::F64(x) => f64::from_le_bytes(x),
        _ => return false,
    };
    (val - LOG2_E).abs() < 1e-6
}

impl Kernel {
    /// Finds `exp2(x * log2(e))` and replaces it with `exp(x)`.
    ///
    /// This recognizes the pattern produced by `tensor.exp()` which is
    /// implemented as `(x * log2(e)).exp2()`. Converting back to `exp`
    /// allows Tenstorrent to use its native `exp_tile` instead of the
    /// unsupported `exp2_tile`.
    pub fn exp2_to_exp(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let &Op::Unary { x, uop: UOp::Exp2 } = self.at(op_id) {
                if let &Op::Binary { x: left, y: right, bop: BOp::Mul } = self.at(x) {
                    let input = match (self.at(left), self.at(right)) {
                        (&Op::Const(c), _) if constant_is_log2_e(&c) => right,
                        (_, &Op::Const(c)) if constant_is_log2_e(&c) => left,
                        _ => OpId::NULL,
                    };
                    if input != OpId::NULL {
                        self.ops[op_id].op = Op::Unary { x: input, uop: UOp::Exp };
                    }
                }
            }
            op_id = next;
        }
    }
}
