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
                    let Some(&(xl, xu)) = bounds.get(&x);
                    let Some(&(yl, yu)) = bounds.get(&y);
                    eprintln!("Op {}: {:?} x={:?} ({}..={}) y={} ({yl}..={yu})", 
                        op_id.0, bop, x, xl, xu, y, yl, yu);
                    
                    if let Some(result) = self.try_simplify_div_mod(x, y, bop, &bounds) {
                        eprintln!("  -> Simplified to {:?}", result);
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
        
        if let Op::Binary { x: inner, y, bop: BOp::Div } = self.at(x) {
            let inner = *inner;
            let Some(&(yl, yu)) = bounds.get(&y) else { return None };
            if yl == yu && yl == divisor {
                let Some(&(inner_l, inner_u)) = bounds.get(&inner) else { return None };
                if inner_u < divisor {
                    return Some(SimplifyResult::ForwardTo(inner));
                }
            }
        }
        
        if let Op::Binary { x: _inner, y, bop: BOp::Add } = self.at(x) {
            if let Some(b) = self.try_match_constant_y(*y, divisor) {
                let Some(&(bl, bu)) = bounds.get(&b) else { return None };
                if bl < divisor && bu <= divisor {
                    return Some(SimplifyResult::ForwardTo(b));
                }
            }
        }
        
        if let Op::Binary { x: a, y: b, bop: BOp::Add } = self.at(x) {
            let Some(&(au, _)) = bounds.get(&a) else { return None };
            let Some(&(bu, _)) = bounds.get(&b) else { return None };
            if au.saturating_add(bu) < divisor {
                return Some(SimplifyResult::ForwardTo(x));
            }
        }
        
        if let Op::Binary { x: a, y: cst, bop: BOp::Mul } = self.at(x) {
            if let Op::Const(cst2) = self.at(*cst) {
                if let Some(c_val) = constant_as_u64(&cst2) {
                    let c = c_val as u32;
                    let c_reduced = c % divisor;
                    if c_reduced != c && c_reduced > 0 {
                        let Some(&(au, _)) = bounds.get(&a) else { return None };
                        if au.saturating_mul(c_reduced) < divisor {
                            return Some(SimplifyResult::ForwardTo(x));
                        }
                    }
                }
            }
        }
        
        if let Op::Binary { x: mul_op, y: add_op, bop: BOp::Add } = self.at(x) {
            if let Op::Binary { x: a, y: cst, bop: BOp::Mul } = self.at(*mul_op) {
                if let Op::Const(cst2) = self.at(*cst) {
                    if let Some(c_val) = constant_as_u64(&cst2) {
                        let c = c_val as u32;
                        let Some(&(au, _)) = bounds.get(&a) else { return None };
                        let Some(&(bu, _)) = bounds.get(&add_op) else { return None };
                        let max_val = au.saturating_mul(c).saturating_add(bu);
                        if max_val < divisor {
                            return Some(SimplifyResult::ForwardTo(x));
                        }
                        if c % divisor == 1 && au.saturating_add(bu) < divisor {
                            return Some(SimplifyResult::ForwardTo(x));
                        }
                    }
                }
            }
        }
        
        None
    }
    
    fn try_match_constant_y(&self, y: OpId, c: u32) -> Option<OpId> {
        if let Op::Const(cst) = self.at(y) {
            if let Some(v) = constant_as_u64(&cst) {
                if v < c as u64 { return Some(y); }
            }
        }
        
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
        if let Op::Binary { x: inner, y, bop: BOp::Add } = self.at(x) {
            if let Some(b) = self.try_match_constant_y(*y, divisor) {
                let Some(&(bl, bu)) = bounds.get(&b) else { return None };
                if bl < divisor && bu <= divisor {
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
        
        let Some(&(xl, xu)) = bounds.get(&x) else { return None };
        if xu < divisor {
            return Some(SimplifyResult::ReplaceWith(Op::Const(make_constant_u64(0))));
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