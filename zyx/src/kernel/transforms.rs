// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Convert exp2(x * log2(e)) to exp(x).
//!
//! This module provides optimization for converting `exp2(x * log2(e))`
//! to `exp(x)`, which allows backends like Tenstorrent to use their
//! native `exp_tile` instead of the unsupported `exp2_tile`.

use std::f64::consts::{LN_2, LOG2_E};

use crate::scalar::{bf16, f16};
use crate::slab::SlabId;
use crate::{
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, UOp},
};

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
            if let &Op::Unary { x, uop: UOp::Exp2 } = self.at(op_id)
                && let &Op::Binary { x: left, y: right, bop: BOp::Mul } = self.at(x)
            {
                let input = match (self.at(left), self.at(right)) {
                    (&Op::Const(c), _) if constant_is_log2_e(&c) => right,
                    (_, &Op::Const(c)) if constant_is_log2_e(&c) => left,
                    _ => OpId::NULL,
                };
                if input != OpId::NULL {
                    self.ops[op_id].op = Op::Unary { x: input, uop: UOp::Exp };
                }
            }
            op_id = next;
        }
    }

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
                if let &Op::Const(c) = const_op
                    && constant_is_ln_2(&c)
                {
                    self.ops[op_id].op = Op::Unary { x: log2_op, uop: UOp::Ln };
                }
            }
            op_id = next;
        }
    }

    /// Exp to exp2
    /// Converts `exp(x * ln(e))` to `exp2(x)`
    /// This allows backends with native `exp2` but not `exp` to use exp2 instead
    pub fn exp_to_exp2(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let &Op::Unary { x, uop: UOp::Exp } = self.at(op_id) {
                let dtype = self.dtype(x);
                let y = self.insert_before(op_id, Op::Const(Constant::F64(LOG2_E.to_le_bytes()).cast(dtype)));
                let z = self.insert_before(op_id, Op::Binary { x, y, bop: BOp::Mul });
                self.ops[op_id].op = Op::Unary { x: z, uop: UOp::Exp2 };
            }
            op_id = next;
        }
    }

    /// Ln to log2
    /// Converts `ln(x)` to `log2(x) * ln(2)`
    /// This allows backends with native `log2` but not `ln` to use log2 instead
    pub fn ln_to_log2(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let &Op::Unary { x, uop: UOp::Ln } = self.at(op_id) {
                let dtype = self.dtype(x);
                let y = self.insert_before(op_id, Op::Const(Constant::F64(LN_2.to_le_bytes()).cast(dtype)));
                let x = self.insert_before(op_id, Op::Unary { x, uop: UOp::Log2 });
                self.ops[op_id].op = Op::Binary { x, y, bop: BOp::Mul };
            }
            op_id = next;
        }
    }
}
