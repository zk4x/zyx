// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    DType, Map,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId},
    shape::Dim,
};

impl Kernel {
    pub fn div_mod_simplification(&mut self) {
        let bounds = self.compute_bounds();

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);

            if let Op::Binary { x, y, bop } = self.at(op_id).clone() {
                if matches!(bop, BOp::Div | BOp::Mod) {
                    if let Op::Const(divisor) = self.at(y) {
                        let dtype = divisor.dtype();
                        if let Some(divisor) = divisor.as_dim() {
                            match bop {
                                BOp::Mod => self.simplify_mod(op_id, x, divisor, dtype, &bounds),
                                BOp::Div => self.simplify_div(op_id, x, divisor, dtype, &bounds),
                                _ => {}
                            };
                        }
                    }
                }
            }

            op_id = next;
        }

        self.verify();
    }

    fn simplify_div(&mut self, op_id: OpId, x: OpId, divisor: Dim, dtype: DType, bounds: &Map<OpId, (Dim, Dim)>) {
        if let Some((a, c, _)) = mul_add(self, x) {
            if c == divisor {
                self.remap(op_id, a);
                return;
            }
        }

        if let Some((a, c, _)) = mad(self, x) {
            if c == divisor {
                self.remap(op_id, a);
                return;
            }
        }

        let Some(&(_, xu)) = bounds.get(&x) else { return };
        if xu < divisor {
            self.ops[op_id].op = Op::Const(dtype.zero_constant());
        }
    }

    fn simplify_mod(&mut self, op_id: OpId, x: OpId, divisor: Dim, _dtype: DType, bounds: &Map<OpId, (Dim, Dim)>) {
        let Some(&(min_x, max_x)) = bounds.get(&x) else { return };
        if min_x == 0 && max_x < divisor {
            self.remap(op_id, x);
            return;
        }

        if let Some((a, c, b)) = mul_add(self, x) {
            // Pattern 2: (a * c + b) % c -> b
            if c == divisor {
                self.remap(op_id, b);
                return;
            }
            // Pattern 2b: when c % divisor == 1, (a*c + b) % d = (a + b) % d
            // Only apply when both a and b are simple: Const or Index
            if c % divisor == 1 {
                let divisor_const = self.insert_before(op_id, Op::Const(Constant::idx(divisor)));
                let a_plus_b = self.insert_before(op_id, Op::Binary { x: a, y: b, bop: BOp::Add });
                self.ops[op_id].op = Op::Binary { x: a_plus_b, y: divisor_const, bop: BOp::Mod };
                return;
            }
        }

        // Pattern 3: (a + b) % divisor when max(a + b) < divisor
        if let Op::Binary { x: a, y: b, bop: BOp::Add } = self.ops[x].op {
            if let Some(&(min_a, max_a)) = bounds.get(&a) {
                if let Some(&(min_b, max_b)) = bounds.get(&b) {
                    if min_a > 0 && min_b > 0 {
                        let sum = max_a.saturating_add(max_b);
                        if sum < divisor && sum > 0 {
                            self.remap(op_id, x);
                            return;
                        }
                    }
                }
            }
        }

        // Pattern 4: (a * c) % divisor - reduce c modulo divisor
        if let Op::Binary { x: a, y: c, bop: BOp::Mul } = self.ops[x].op {
            if let Op::Const(y) = self.ops[c].op {
                if let Some(c) = y.as_dim() {
                    let c_reduced = c % divisor;
                    if c_reduced != c && c_reduced > 0 {
                        if let Some(&(min_a, max_a)) = bounds.get(&a) {
                            if min_a > 0 {
                                let prod = max_a.saturating_mul(c_reduced);
                                if prod < divisor && prod > 0 {
                                    self.remap(op_id, x);
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Pattern 6:
        if let Op::Binary { x: a, y: b, bop: BOp::Add } = self.ops[x].op {
            if let Op::Const(y) = self.ops[b].op {
                if let Some(y) = y.as_dim() {
                    if let Some(&(_, max_a)) = bounds.get(&a) {
                        if max_a + y < divisor {
                            self.remap(op_id, x);
                            return;
                        }
                    };
                }
            }
        }
    }
}

fn mul_add(k: &Kernel, x: OpId) -> Option<(OpId, u64, OpId)> {
    if let Some(x) = mad(k, x) {
        return Some(x);
    }
    // Case 1: (a * c) + b
    let Op::Binary { x: mul, y: add, bop: BOp::Add } = k.at(x) else {
        return None;
    };
    if let Op::Binary { x: a, y: c, bop: BOp::Mul } = k.at(*mul) {
        if let Op::Const(cst) = k.at(*c) {
            if let Some(cval) = cst.as_dim() {
                return Some((*a, cval, *add));
            }
        }
    }
    // Case 2: b + (a * c)
    let Op::Binary { x: b, y: mul, bop: BOp::Add } = k.at(x) else {
        return None;
    };
    if let Op::Binary { x: a, y: c, bop: BOp::Mul } = k.at(*mul) {
        if let Op::Const(cst) = k.at(*c) {
            if let Some(cval) = cst.as_dim() {
                return Some((*a, cval, *b));
            }
        }
    }
    None
}

fn mad(k: &Kernel, x: OpId) -> Option<(OpId, u64, OpId)> {
    let Op::Mad { x: a, y: c, z: b } = k.at(x) else { return None };
    let Op::Const(cst) = k.at(*c) else { return None };
    let Some(cval) = cst.as_dim() else { return None };
    Some((*a, cval, *b))
}

/*fn mul_c(k: &Kernel, x: OpId) -> Option<(OpId, u64)> {
    let Op::Binary { x: a, y: c, bop: BOp::Mul } = k.at(x) else {
        return None;
    };
    let Op::Const(cst) = k.at(*c) else { return None };
    let Some(cval) = cst.as_dim() else { return None };
    Some((*a, cval))
}*/
