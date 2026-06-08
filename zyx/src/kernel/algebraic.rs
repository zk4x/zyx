// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    DType, Map,
    kernel::{BOp, Kernel, Op, OpId},
    shape::Dim,
};

impl Kernel {
    pub fn algebraic_simplification(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("algebraic_simplification");

        self.simplify_shl_shr_roundtrips();
        self.unfuse_mad();

        let bounds = self.compute_bounds();

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);

            if let &Op::Binary { x, y, bop } = self.at(op_id) {
                if matches!(bop, BOp::Div | BOp::Mod) {
                    if let Op::Const(divisor) = self.at(y) {
                        let dtype = divisor.dtype();
                        if let Some(divisor) = divisor.as_dim() {
                            match bop {
                                BOp::Mod => self.simplify_mod(op_id, x, y, dtype, &bounds),
                                BOp::Div => self.simplify_div(op_id, x, divisor, dtype, &bounds),
                                _ => {}
                            }
                        }
                    }
                }
            }

            op_id = next;
        }

        self.dead_code_elimination();
        self.verify();
    }

    fn simplify_shl_shr_roundtrips(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Some(y) = self.match_shl_shr_roundtrip(op_id) {
                self.remap(op_id, y);
            }
            op_id = next;
        }
        self.dead_code_elimination();
    }

    fn match_shl_shr_roundtrip(&self, op_id: OpId) -> Option<OpId> {
        let Op::Binary { x: add_op, y: shift_amount, bop: BOp::BitShiftRight } = self.at(op_id) else {
            return None;
        };
        let Op::Const(cst) = self.at(*shift_amount) else { return None };
        let n = cst.as_dim()?;
        if n >= 64 {
            return None;
        }
        let Op::Binary { x: add_x, y: add_y, bop: BOp::Add } = self.at(*add_op) else {
            return None;
        };
        for candidate in [add_x, add_y] {
            if let Op::Binary { x: y, y: s, bop: BOp::BitShiftLeft } = self.at(*candidate) {
                if let Op::Const(c) = self.at(*s) {
                    if c.as_dim() == Some(n) {
                        return Some(*y);
                    }
                }
            }
        }
        None
    }

    #[allow(unused)]
    fn const_dim(&self, op_id: OpId) -> Option<Dim> {
        let Op::Const(c) = self.ops[op_id].op else { return None };
        c.as_dim()
    }

    #[allow(unused)]
    fn get_add_sub_chain(&self, op_id: OpId) -> Vec<OpId> {
        todo!()
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

    fn simplify_mod(&mut self, op_id: OpId, x: OpId, divisor_const: OpId, _dtype: DType, bounds: &Map<OpId, (Dim, Dim)>) {
        let Op::Const(divisor) = self.ops[divisor_const].op else { return };
        let Some(divisor) = divisor.as_dim() else { return };

        //self.debug();

        // Pattern 1: x % divisor when 0 <= x < divisor -> x
        if let Some(&(_, max_x)) = bounds.get(&x) {
            if max_x < divisor {
                self.remap(op_id, x);
                return;
            }
        }

        if let Some((a, c, b)) = mul_add(self, x) {
            // Pattern 2: (a*c + b) % c -> b % c (because (a*c) % c = 0)
            // Math: (a*c + b) % c = ((a*c) % c + b % c) % c = (0 + b % c) % c = b % c
            // Since c == divisor: result = b % divisor
            if c == divisor {
                self.ops[op_id].op = Op::Binary { x: b, y: divisor_const, bop: BOp::Mod };
                // Pattern 1 on result: if b < divisor, b % divisor = b
                if let Some(&(_, max_b)) = bounds.get(&b) {
                    if max_b < divisor {
                        self.remap(op_id, b);
                    }
                }
                return;
            }
            // Pattern 2b: (a*c + b) % d when c % d == 1 -> (a + b) % d
            // Math: (a*c + b) % d = ((a*(c%d) + b) % d) = ((a*1 + b) % d) = (a + b) % d
            if c % divisor == 1 {
                let a_plus_b = self.insert_before(op_id, Op::Binary { x: a, y: b, bop: BOp::Add });
                self.ops[op_id].op = Op::Binary { x: a_plus_b, y: divisor_const, bop: BOp::Mod };
                // Pattern 1 on result: if max(a) + max(b) < divisor, (a+b) % divisor = a+b
                if let Some(&(_, max_a)) = bounds.get(&a)
                    && let Some(&(_, max_b)) = bounds.get(&b)
                {
                    if max_a.saturating_add(max_b) < divisor {
                        self.remap(op_id, a_plus_b);
                    }
                }
                return;
            }
            // Pattern 2c: (a*c + b) % d when max(a*c + b) < d -> b % d
            // Need: min_b == 0 AND max(a*c) + max_b < divisor
            if let Some(&(_min_a, max_a)) = bounds.get(&a) {
                let max_a_c = max_a.saturating_mul(c);
                if let Some(&(min_b, max_b)) = bounds.get(&b) {
                    if min_b == 0 && max_a_c.saturating_add(max_b) < divisor {
                        self.ops[op_id].op = Op::Binary { x: b, y: divisor_const, bop: BOp::Mod };
                        // Pattern 1 on result: if b < divisor, b % divisor = b
                        if max_b < divisor {
                            self.remap(op_id, b);
                        }
                        return;
                    }
                }
            }
            // Pattern 2d: (a*c + b) % d when d = c*k and max(a*c+b) < d -> b
            // Need: min_b == 0 AND max(a*c) + max_b < divisor
            // When max(a*c + b) < divisor, (a*c + b) % divisor = a*c + b, so if max < divisor -> result = b
            if divisor > c && divisor.is_multiple_of(c) {
                if let Some(&(_min_a, max_a)) = bounds.get(&a)
                    && let Some(&(min_b, max_b)) = bounds.get(&b)
                {
                    let max_ac = max_a.saturating_mul(c);
                    if min_b == 0 && max_ac.saturating_add(max_b) < divisor {
                        self.remap(op_id, b);
                        return;
                    }
                }
            }
        }

        // Pattern 3: (a + b) % divisor when min_a > 0, min_b > 0, max(a+b) < divisor
        // If both are positive and sum < divisor, no wraparound, so result = a + b
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

        // Pattern 4: (a * c) % divisor -> reduce c modulo divisor
        // Math: (a * c) % d = (a * (c % d)) % d
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

        // Pattern 5: (a + C) % divisor where C is constant and max(a) + C < divisor
        // If max(a) + C < divisor, no wraparound, so result = a + C
        if let Op::Binary { x: a, y: b, bop: BOp::Add } = self.ops[x].op {
            if let Op::Const(y) = self.ops[b].op {
                if let Some(y) = y.as_dim() {
                    if let Some(&(_, max_a)) = bounds.get(&a) {
                        if max_a + y < divisor {
                            self.remap(op_id, x);
                            return;
                        }
                    }
                }
            }
        }
    }
}

fn mul_add(k: &Kernel, x: OpId) -> Option<(OpId, u64, OpId)> {
    if let Some(x) = mad(k, x) {
        return Some(x);
    }
    // Case 1: (a * c) + b  (also (a << c) + b for constant c)
    let Op::Binary { x: mul, y: add, bop: BOp::Add } = k.at(x) else {
        return None;
    };
    if let Some((a, cval)) = match_mul_or_shl(k, *mul) {
        return Some((a, cval, *add));
    }
    // Case 2: b + (a * c)  (also b + (a << c) for constant c)
    let Op::Binary { x: b, y: mul, bop: BOp::Add } = k.at(x) else {
        return None;
    };
    if let Some((a, cval)) = match_mul_or_shl(k, *mul) {
        return Some((a, cval, *b));
    }
    None
}

fn match_mul_or_shl(k: &Kernel, op: OpId) -> Option<(OpId, u64)> {
    if let Op::Binary { x: a, y: c, bop: BOp::Mul } = k.at(op) {
        if let Op::Const(cst) = k.at(*c) {
            if let Some(cval) = cst.as_dim() {
                return Some((*a, cval));
            }
        }
    }
    if let Op::Binary { x: a, y: c, bop: BOp::BitShiftLeft } = k.at(op) {
        if let Op::Const(cst) = k.at(*c) {
            if let Some(cval) = cst.as_dim() {
                if cval < 64 {
                    return Some((*a, 1u64 << cval));
                }
            }
        }
    }
    None
}

fn mad(k: &Kernel, x: OpId) -> Option<(OpId, u64, OpId)> {
    let Op::Mad { x: a, y: c, z: b } = k.at(x) else { return None };
    let Op::Const(cst) = k.at(*c) else { return None };
    let cval = cst.as_dim()?;
    Some((*a, cval, *b))
}
