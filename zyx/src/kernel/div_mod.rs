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
        //self.debug();
        let mut changed = true;
        let mut iterations = 0;
        while changed && iterations < 10 {
            changed = false;
            iterations += 1;
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
                                /*if let Some(result) = result {
                                    changed = true;
                                    match result {
                                        SimplifyResult::ForwardTo(src) => {
                                            self.remap(op_id, src);
                                        }
                                        SimplifyResult::ReplaceWith(new_op) => {
                                            self.ops[op_id].op = new_op;
                                        }
                                        SimplifyResult::ReplaceWithSeq(mut new_ops) => {
                                            let new_op = new_ops.pop().unwrap();
                                            self.ops[op_id].op = new_op;
                                            while let Some(op) = new_ops.pop() {
                                                self.insert_before(op_id, op);
                                            }
                                        }
                                    }
                                }*/
                            }
                        }
                    }
                }

                op_id = next;
            }
        }
        //self.debug();
        //panic!();

        self.verify();
    }

    fn simplify_div(&mut self, op_id: OpId, x: OpId, divisor: Dim, dtype: DType, bounds: &Map<OpId, (Dim, Dim)>) {
        // Pattern: (a * c + b) / c -> a (integer division discards remainder)
        // This always works regardless of b's value!
        if let Some((a, c, _)) = mul_add(self, x) {
            if c == divisor {
                self.remap(op_id, a);
                return;
            }
        }

        // Also handle Mad (multiply-add in one op)
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

    fn simplify_mod(&mut self, op_id: OpId, x: OpId, divisor: Dim, dtype: DType, bounds: &Map<OpId, (Dim, Dim)>) {
        // Pattern 1: x already in range [0, divisor-1]
        let Some(&(min_x, max_x)) = bounds.get(&x) else { return };
        if min_x == 0 && max_x < divisor {
            self.remap(op_id, x);
            return;
        }

        // Pattern 2: (a * c + b) % c -> b (the remainder is just b, c cancels out)
        if let Some((a, c, b)) = mul_add(self, x) {
            /*if c == divisor {
                self.remap(op_id, b);
                return;
            }*/
            // Pattern 2b: congruence - when c % divisor == 1, (a*c + b) % d = (a + b) % d
            if c % divisor == 1 {
                if let Some(&(_, max_a)) = bounds.get(&a) {
                    if let Some(&(_, max_b)) = bounds.get(&b) {
                        if max_a.saturating_add(max_b) < divisor {
                            let a_plus_b = Op::Binary { x: a, y: b, bop: BOp::Add };
                            self.ops[op_id].op = a_plus_b;
                        }
                    }
                }
                let a_plus_b = self.insert_before(op_id, Op::Binary { x: a, y: b, bop: BOp::Add });
                let d = self.insert_before(op_id, Op::Const(Constant::idx(divisor).cast(dtype)));
                self.ops[op_id].op = Op::Binary { x: a_plus_b, y: d, bop: BOp::Mod };
                return;
            }
        }

        // TODO Pattern 2: (a * c + b) % divisor -> b when b < divisor

        // Pattern 3: (a + b) % divisor when max(a+b) < divisor
        /*if let Some((a, b)) = add(self, x) {
            let Some(&(au, _)) = bounds.get(&a) else { return None };
            let Some(&(bu, _)) = bounds.get(&b) else { return None };
            if au.saturating_add(bu) < divisor {
                return Some(SimplifyResult::ForwardTo(x));
            }
        }

        // Pattern 4: (a * c) % divisor - reduce c modulo divisor
        if let Some((a, c)) = mul_c(self, x) {
            let c = c as u32;
            let c_reduced = c % divisor;
            if c_reduced != c && c_reduced > 0 {
                let Some(&(au, _)) = bounds.get(&a) else { return None };
                if au.saturating_mul(c_reduced) < divisor {
                    return Some(SimplifyResult::ForwardTo(x));
                }
            }
        }

        // Pattern 5: (a / d) % d -> a when a < d
        if let Some((inner, div_y)) = div(self, x) {
            if let Some(&(yl, yu)) = bounds.get(&div_y) {
                if yl == yu && yl == divisor {
                    let Some(&(_, inner_u)) = bounds.get(&inner) else {
                        return None;
                    };
                    if inner_u < divisor {
                        return Some(SimplifyResult::ForwardTo(inner));
                    }
                }
            }
        }*/

        // Pattern 6: (a + const) % divisor = a when a < divisor
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
    let Op::Binary { x: mul, y: add, bop: BOp::Add } = k.at(x) else {
        return None;
    };
    let Op::Binary { x: a, y: c, bop: BOp::Mul } = k.at(*mul) else {
        return None;
    };
    let Op::Const(cst) = k.at(*c) else { return None };
    let Some(cval) = cst.as_dim() else { return None };
    Some((*a, cval, *add))
}

fn mad(k: &Kernel, x: OpId) -> Option<(OpId, u64, OpId)> {
    let Op::Mad { x: a, y: c, z: b } = k.at(x) else { return None };
    let Op::Const(cst) = k.at(*c) else { return None };
    let Some(cval) = cst.as_dim() else { return None };
    Some((*a, cval, *b))
}

fn mul_c(k: &Kernel, x: OpId) -> Option<(OpId, u64)> {
    let Op::Binary { x: a, y: c, bop: BOp::Mul } = k.at(x) else {
        return None;
    };
    let Op::Const(cst) = k.at(*c) else { return None };
    let Some(cval) = cst.as_dim() else { return None };
    Some((*a, cval))
}
