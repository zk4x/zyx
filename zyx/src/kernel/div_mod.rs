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
                match bop {
                    BOp::Div | BOp::Mod => {
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
                    _ => {}
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
        // Pattern: (a * c + b) % c -> b when b < c
        if let Op::Binary { x: inner, y, bop: BOp::Add } = self.at(x) {
            if let Some(b) = self.try_match_constant_y(*y, divisor) {
                let Some(&(bl, bu)) = bounds.get(&b) else { return None };
                if bl < divisor && bu <= divisor {
                    return Some(SimplifyResult::ForwardTo(b));
                }
            }
        }
        
        // Pattern: (a + b) % c -> a + b when max(a+b) < c
        if let Op::Binary { x: a, y: b, bop: BOp::Add } = self.at(x) {
            let Some(&(au, _)) = bounds.get(&a) else { return None };
            let Some(&(bu, _)) = bounds.get(&b) else { return None };
            if au.saturating_add(bu) < divisor {
                return Some(SimplifyResult::ForwardTo(x));
            }
        }
        
        // Pattern: x % c where x has range [0, c-1] -> x
        let Some(&(xl, xu)) = bounds.get(&x) else { return None };
        if xl == 0 && xu < divisor {
            return Some(SimplifyResult::ForwardTo(x));
        }
        
        None
    }
    
    fn try_match_constant_y(&self, y: OpId, c: u32) -> Option<OpId> {
        // Try to match b where b < c
        if let Op::Const(cst) = self.at(y) {
            if let Some(v) = constant_as_u64(&cst) {
                if v < c as u64 { return Some(y); }
            }
        }
        
        // Also check: b + const where b < c
        if let Op::Binary { x: b, y: k, bop: BOp::Add } = self.at(y) {
            let b = *b;
            if let Op::Const(cst) = self.at(*k) {
                if let Some(_v) = constant_as_u64(&cst) {
                    return Some(b);
                }
            }
        }
        
        None
    }
    
    fn simplify_div(&self, x: OpId, divisor: u32, bounds: &Map<OpId, (u32, u32)>) -> Option<SimplifyResult> {
        // Pattern: (a * c + b) / c -> a when b < c
        if let Op::Binary { x: inner, y, bop: BOp::Add } = self.at(x) {
            if let Some(b) = self.try_match_constant_y(*y, divisor) {
                let Some(&(bl, bu)) = bounds.get(&b) else { return None };
                if bl < divisor && bu <= divisor {
                    // Find a in (a * c + b)
                    let inner = *inner;
                    if let Op::Binary { x: a, y: cst, bop: BOp::Mul } = self.at(inner) {
                        let a = *a;
                        if let Op::Const(cst2) = self.at(*cst) {
                            if let Some(cval) = constant_as_u64(&cst2) {
                                if cval == divisor as u64 {
                                    return Some(SimplifyResult::ForwardTo(a));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        None
    }
}

enum SimplifyResult { ReplaceWith(Op), ForwardTo(OpId) }

fn constant_as_u64(c: &Constant) -> Option<u64> {
    match c {
        Constant::U32(x) => Some(*x as u64),
        Constant::U64(x) => Some(u64::from_le_bytes(*x)),
        _ => None,
    }
}

fn make_constant_u64(val: u64) -> Constant {
    Constant::U64(val.to_le_bytes())
}