// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use crate::{
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId},
    Map,
};

impl Kernel {
    pub fn div_mod_simplification(&mut self) {
        let bounds = self.compute_bounds();

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);

            if let Op::Binary { x, y, bop } = self.at(op_id).clone() {
                if matches!(bop, BOp::Div | BOp::Mod) {
                    let Some(&(_, _)) = bounds.get(&x) else { op_id = next; continue; };
                    let Some(&(_, _)) = bounds.get(&y) else { op_id = next; continue; };

                    if let Some(result) = self.try_simplify_div_mod(x, y, bop, &bounds) {
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

        self.verify();
    }

    fn try_simplify_div_mod(&self, x: OpId, y: OpId, bop: BOp, bounds: &Map<OpId, (u32, u32)>) -> Option<SimplifyResult> {
        let Some(&(yl, yu)) = bounds.get(&y) else { return None };
        if yl != yu || yl == 0 { return None; }
        let divisor = yl;

        match bop {
            BOp::Mod => self.simplify_mod(x, divisor, bounds),
            BOp::Div => self.simplify_div(x, divisor, bounds),
            _ => None,
        }
    }

    fn simplify_mod(&self, x: OpId, divisor: u32, bounds: &Map<OpId, (u32, u32)>) -> Option<SimplifyResult> {
        let Some(&(xl, xu)) = bounds.get(&x) else { return None };
        if xl == 0 && xu < divisor {
            return Some(SimplifyResult::ForwardTo(x));
        }

        if let Some((a, c, b)) = mul_add(self, x) {
            let Some(&(au, _)) = bounds.get(&a) else { return None };
            let Some(&(bu, _)) = bounds.get(&b) else { return None };
            let max_val: u64 = au as u64 * c + bu as u64;
            if max_val < divisor as u64 {
                return Some(SimplifyResult::ForwardTo(x));
            }
            if c as u32 % divisor == 1 && (au as u64 + bu as u64) < divisor as u64 {
                return Some(SimplifyResult::ForwardTo(b));
            }
        }

        if let Some((a, b)) = add(self, x) {
            let Some(&(au, _)) = bounds.get(&a) else { return None };
            let Some(&(bu, _)) = bounds.get(&b) else { return None };
            if au.saturating_add(bu) < divisor {
                return Some(SimplifyResult::ForwardTo(x));
            }
        }

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

        if let Some((inner, b)) = add(self, x) {
            if let Some(_) = constant_le(self, b, divisor) {
                let Some(&(_, inner_u)) = bounds.get(&inner) else { return None };
                if inner_u < divisor {
                    return Some(SimplifyResult::ForwardTo(inner));
                }
            }
        }

        None
    }

    fn simplify_div(&self, x: OpId, divisor: u32, bounds: &Map<OpId, (u32, u32)>) -> Option<SimplifyResult> {
        if let Some((a, c, b)) = mul_add(self, x) {
            if let Some(_) = constant_le(self, b, divisor) {
                if c == divisor as u64 {
                    return Some(SimplifyResult::ForwardTo(a));
                }
            }
        }

        let Some(&(_, xu)) = bounds.get(&x) else { return None };
        if xu < divisor {
            return Some(SimplifyResult::ReplaceWith(Op::Const(Constant::idx(0))));
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