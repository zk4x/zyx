// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use crate::{
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId},
    Map,
};

impl Kernel {
    pub fn div_mod_simplification(&mut self) {
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
                        let Some(&(xl, xu)) = bounds.get(&x) else { op_id = next; continue; };
                        let Some(&(yl, yu)) = bounds.get(&y) else { op_id = next; continue; };

                        if yl != yu || yl == 0 { op_id = next; continue; }
                        let divisor = yl;

                        let result = match bop {
                            BOp::Mod => self.simplify_mod(x, divisor, &bounds),
                            BOp::Div => self.simplify_div(x, divisor, &bounds),
                            _ => None,
                        };

                        if let Some(result) = result {
                            changed = true;
                            match result {
                                SimplifyResult::ReplaceWith(new_op) => {
                                    self.ops[op_id].op = new_op;
                                }
                                SimplifyResult::ForwardTo(src) => {
                                    self.remap(op_id, src);
                                }
                            }
                        }
                    }
                }

                op_id = next;
            }
        }

        self.verify();
    }

    fn simplify_div(&self, x: OpId, divisor: u32, bounds: &Map<OpId, (u32, u32)>) -> Option<SimplifyResult> {
        // Pattern: (a * c + b) / c -> a (integer division discards remainder)
        // This always works regardless of b's value!
        if let Some((a, c, _)) = mul_add(self, x) {
            if c == divisor as u64 {
                return Some(SimplifyResult::ForwardTo(a));
            }
        }

        // Also handle Mad (multiply-add in one op)
        if let Some((a, c, _)) = mad(self, x) {
            if c == divisor as u64 {
                return Some(SimplifyResult::ForwardTo(a));
            }
        }

        let Some(&(_, xu)) = bounds.get(&x) else { return None };
        if xu < divisor {
            return Some(SimplifyResult::ReplaceWith(Op::Const(Constant::idx(0))));
        }

        None
    }

    fn simplify_mod(&self, x: OpId, divisor: u32, bounds: &Map<OpId, (u32, u32)>) -> Option<SimplifyResult> {
        // Pattern 1: x already in range [0, divisor-1]
        let Some(&(xl, xu)) = bounds.get(&x) else { return None };
        if xl == 0 && xu < divisor {
            return Some(SimplifyResult::ForwardTo(x));
        }

        // Pattern 2: (a * c + b) % c -> b (the remainder is just b, c cancels out)
        if let Some((a, c, b)) = mul_add(self, x) {
            if c as u32 == divisor {
                return Some(SimplifyResult::ForwardTo(b));
            }
            // Pattern 2b: congruence - when c % divisor == 1, (a*c + b) % d = (a + b) % d
            if c as u32 % divisor == 1 {
                if let Some(&(au, _)) = bounds.get(&a) {
                    if let Some(&(bu, _)) = bounds.get(&b) {
                        if au.saturating_add(bu) < divisor {
                            if let Some((a_plus_b, _)) = add(self, b) {
                                return Some(SimplifyResult::ForwardTo(a_plus_b));
                            }
                        }
                    }
                }
            }
        }

        // Pattern 2b: Handle Mad ops the same way
        if let Some((a, c, b)) = mad(self, x) {
            if c as u32 == divisor {
                return Some(SimplifyResult::ForwardTo(b));
            }
            // Congruence for Mad
            if c as u32 % divisor == 1 {
                if let Some(&(au, _)) = bounds.get(&a) {
                    if let Some(&(bu, _)) = bounds.get(&b) {
                        if au.saturating_add(bu) < divisor {
                            if let Some((a_plus_b, _)) = add(self, b) {
                                return Some(SimplifyResult::ForwardTo(a_plus_b));
                            }
                        }
                    }
                }
            }
        }

        // Pattern 2: (a * c + b) % divisor -> b when b < divisor
        if let Some((a, c, b)) = mul_add(self, x) {
            let b_small = if let Some(_) = constant_le(self, b, divisor) {
                true
            } else if let Some(&(bl, bu)) = bounds.get(&b) {
                bu < divisor
            } else {
                false
            };
            if b_small {
                return Some(SimplifyResult::ForwardTo(b));
            }
            // Pattern 2b: congruence - when c % divisor == 1, (a*c + b) % d = (a + b) % d
            if c as u32 % divisor == 1 {
                if let Some(&(au, _)) = bounds.get(&a) {
                    if let Some(&(bu, _)) = bounds.get(&b) {
                        if au.saturating_add(bu) < divisor {
                            if let Some((a_plus_b, _)) = add(self, b) {
                                return Some(SimplifyResult::ForwardTo(a_plus_b));
                            }
                        }
                    }
                }
            }
        }

        // Pattern 3: (a + b) % divisor when max(a+b) < divisor
        if let Some((a, b)) = add(self, x) {
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
                    let Some(&(_, inner_u)) = bounds.get(&inner) else { return None };
                    if inner_u < divisor {
                        return Some(SimplifyResult::ForwardTo(inner));
                    }
                }
            }
        }

        // Pattern 6: (a + const) % divisor when a < divisor
        if let Some((inner, b)) = add(self, x) {
            if constant_le(self, b, divisor).is_some() {
                let Some(&(_, inner_u)) = bounds.get(&inner) else { return None };
                if inner_u < divisor {
                    return Some(SimplifyResult::ForwardTo(inner));
                }
            }
        }

        // Pattern 7: Handle Mad ops
        if let Some((a, c, b)) = mad(self, x) {
            let b_small = if constant_le(self, b, divisor).is_some() {
                true
            } else if let Some(&(_, bu)) = bounds.get(&b) {
                bu < divisor
            } else {
                false
            };
            if b_small {
                return Some(SimplifyResult::ForwardTo(b));
            }
            // Congruence for Mad
            if c as u32 % divisor == 1 {
                if let Some(&(au, _)) = bounds.get(&a) {
                    if let Some(&(bu, _)) = bounds.get(&b) {
                        if au.saturating_add(bu) < divisor {
                            if let Some((a_plus_b, _)) = add(self, b) {
                                return Some(SimplifyResult::ForwardTo(a_plus_b));
                            }
                        }
                    }
                }
            }
        }

        None
    }
}

fn mul_add(k: &Kernel, x: OpId) -> Option<(OpId, u64, OpId)> {
    let Op::Binary { x: mul, y: add, bop: BOp::Add } = k.at(x) else { return None };
    let Op::Binary { x: a, y: c, bop: BOp::Mul } = k.at(*mul) else { return None };
    let Op::Const(cst) = k.at(*c) else { return None };
    let Some(cval) = constant_as_u64(cst) else { return None };
    Some((*a, cval, *add))
}

fn mul_c(k: &Kernel, x: OpId) -> Option<(OpId, u64)> {
    let Op::Binary { x: a, y: c, bop: BOp::Mul } = k.at(x) else { return None };
    let Op::Const(cst) = k.at(*c) else { return None };
    let Some(cval) = constant_as_u64(cst) else { return None };
    Some((*a, cval))
}

fn add(k: &Kernel, x: OpId) -> Option<(OpId, OpId)> {
    let Op::Binary { x: a, y: b, bop: BOp::Add } = k.at(x) else { return None };
    Some((*a, *b))
}

fn div(k: &Kernel, x: OpId) -> Option<(OpId, OpId)> {
    let Op::Binary { x: a, y: b, bop: BOp::Div } = k.at(x) else { return None };
    Some((*a, *b))
}

fn mad(k: &Kernel, x: OpId) -> Option<(OpId, u64, OpId)> {
    let Op::Mad { x: a, y: c, z: b } = k.at(x) else { return None };
    let Op::Const(cst) = k.at(*c) else { return None };
    let Some(cval) = constant_as_u64(cst) else { return None };
    Some((*a, cval, *b))
}

fn constant_le(k: &Kernel, op: OpId, c: u32) -> Option<u64> {
    let Op::Const(cst) = k.at(op) else { return None };
    let v = constant_as_u64(cst)?;
    if v < c as u64 { Some(v) } else { None }
}

enum SimplifyResult { ReplaceWith(Op), ForwardTo(OpId) }

fn constant_as_u64(c: &Constant) -> Option<u64> {
    match c {
        Constant::U32(x) => Some(*x as u64),
        Constant::U64(x) => Some(u64::from_le_bytes(*x)),
        _ => None,
    }
}