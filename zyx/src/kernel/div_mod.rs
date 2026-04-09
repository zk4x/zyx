// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use crate::{
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId},
    Map,
};

impl Kernel {
    pub fn div_mod_simplification(&mut self) {
        // Single pass simplification
        let bounds = self.compute_bounds();
        
        // DEBUG: Uncomment to see before/after
        // println!("=== BEFORE div_mod_simplification ===");
        // self.debug_colorless();
        
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
        
            // Only process Div and Mod binary ops
            if let Op::Binary { x, y, bop } = self.at(op_id).clone() {
                match bop {
                    BOp::Div | BOp::Mod => {
                        if let Some(result) = self.try_simplify_div_mod(x, y, bop, &bounds) {
                            match result {
                                SimplifyResult::ReplaceWith(new_op) => {
                                    // println!("Simplifying op {} to constant", op_id.0);
                                    self.ops[op_id].op = new_op;
                                }
                                SimplifyResult::ForwardTo(src) => {
                                    // println!("Forwarding op {} to {}", op_id.0, src.0);
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
        
        // DEBUG: Uncomment to see after
        // println!("=== AFTER div_mod_simplification ===");
        // self.debug_colorless();
        
        self.verify();
    }
    
    fn try_simplify_div_mod(&self, x: OpId, y: OpId, bop: BOp, bounds: &Map<OpId, (u32, u32)>) -> Option<SimplifyResult> {
        // Try to match the pattern: (a * c + b) op c
        // where op is Div or Mod
        
        let Some(&(yl, yu)) = bounds.get(&y) else {
            return None;
        };
        
        // y must be a constant for our simplifications
        if yl != yu {
            return None;
        }
        
        let divisor = yl;
        if divisor == 0 {
            return None;
        }
        
        // Try to find pattern: a * c + b
        if let Some((a, b, c)) = self.extract_mul_add(x, divisor) {
            // Now we have: (a * c + b) op c
            // Try simplification
            
            let Some(&(al, au)) = bounds.get(&a) else {
                return None;
            };
            let Some(&(bl, bu)) = bounds.get(&b) else {
                return None;
            };
            
            match bop {
                BOp::Div => {
                    // (a * c + b) / c = a + b / c
                    // If b < c, then b / c = 0
                    if bl < divisor && bu <= divisor {
                        // b is in range [0, c-1], so b/c = 0
                        // Result is just a
                        return Some(SimplifyResult::ForwardTo(a));
                    }
                    
                    // Try to simplify: if b is constant and < divisor, we can simplify
                    if let Op::Const(bc) = self.at(b) {
                        let b_val = constant_as_u64(&bc);
                        if let Some(val) = b_val {
                            if val < divisor as u64 {
                                // b / c = 0, so result is a
                                return Some(SimplifyResult::ForwardTo(a));
                            }
                            // Otherwise, result is a + (b / c)
                            let quotient = val / divisor as u64;
                            // If a is constant 0, result is constant
                            if al == 0 && au == 0 {
                                let new_const = make_constant_u64(quotient);
                                return Some(SimplifyResult::ReplaceWith(Op::Const(new_const)));
                            }
                        }
                    }
                }
                BOp::Mod => {
                    // (a * c + b) % c = b % c
                    // If b < c, then result is just b
                    if bl < divisor && bu <= divisor {
                        // b is in range [0, c-1], so result is b
                        return Some(SimplifyResult::ForwardTo(b));
                    }
                    
                    // Try constant case: b % c
                    if let Op::Const(bc) = self.at(b) {
                        let b_val = constant_as_u64(&bc);
                        if let Some(val) = b_val {
                            let result = val % divisor as u64;
                            if result == 0 && al == 0 && au == 0 {
                                // Result is 0
                                return Some(SimplifyResult::ReplaceWith(Op::Const(make_constant_u64(0))));
                            }
                            let new_const = make_constant_u64(result);
                            return Some(SimplifyResult::ReplaceWith(Op::Const(new_const)));
                        }
                    }
                }
                _ => {}
            }
        }
        
        // Try range-based modulo elimination: (a + b) % c = a + b if max(a) + max(b) < c
        if bop == BOp::Mod {
            if let Some((a, b)) = self.extract_add(x) {
                let Some(&(al, au)) = bounds.get(&a) else {
                    return None;
                };
                let Some(&(bl, bu)) = bounds.get(&b) else {
                    return None;
                };
                
                // If max(a+b) < divisor, no wrap possible
                if au.saturating_add(bu) < divisor {
                    // No wrap possible, can eliminate modulo
                    return Some(SimplifyResult::ForwardTo(x)); // Keep as a + b (don't change)
                }
            }
            
            // Try congruence simplification: (a * c + b) % d = ((a % d) * c + b) % d
            if let Some((a, b, c)) = self.extract_mul_add(x, divisor) {
                // This is the case we already handled, but we can add another simplification:
                // If divisor is larger than the max possible value of the expression,
                // we can remove the modulo entirely
                let Some(&(al, au)) = bounds.get(&a) else {
                    return None;
                };
                let Some(&(bl, bu)) = bounds.get(&b) else {
                    return None;
                };
                
                // Compute max value of a * c + b: max(a) * c + max(b)
                let max_val = au.saturating_mul(divisor).saturating_add(bu);
                if max_val < divisor {
                    // Result is always a * c + b, no modulo needed
                    return Some(SimplifyResult::ForwardTo(x));
                }
            }
        }
        
        // Try div chain simplification: (x / a) / b = x / (a * b)
        // Disabled for now - requires mutable self to insert constants
        /*
        if bop == BOp::Div {
            if let Op::Binary { x: inner_x, y: inner_y, bop: inner_bop } = self.at(x) {
                if *inner_bop == BOp::Div {
                    let Some(&(iyl, iyu)) = bounds.get(&inner_y) else {
                        return None;
                    };
                    if iyl == iyu && iyl > 0 && divisor > 0 {
                        // (x / a) / b = x / (a * b)
                        let new_divisor = iyl * divisor;
                        if new_divisor > 0 && new_divisor <= u32::MAX as u64 {
                            let new_const = make_constant_u64(new_divisor);
                            return Some(SimplifyResult::ReplaceWith(Op::Binary {
                                x: inner_x,
                                y: self.insert_before(OpId::NULL, Op::Const(new_const)), // Placeholder
                                bop: BOp::Div,
                            }));
                        }
                    }
                }
            }
        }
        */
        
        None
    }
    
    // Try to extract (a * c + b) pattern from x, where c is the given divisor
    fn extract_mul_add(&self, x: OpId, c: u32) -> Option<(OpId, OpId, u32)> {
        // Check if x is of form: a * c + b
        if let Op::Binary { x: mul_op, y: add_op, bop: BOp::Add } = self.at(x) {
            // Check if mul_op is a * c
            if let Op::Binary { x: a, y: c_op, bop: BOp::Mul } = self.at(*mul_op) {
                if let Op::Const(c_const) = self.at(*c_op) {
                    if let Some(c_val) = constant_as_u64(c_const) {
                        if c_val == c as u64 {
                            return Some((*a, *add_op, c));
                        }
                    }
                }
            }
            // Also try: c * a + b (commutative)
            if let Op::Binary { x: c_op, y: a, bop: BOp::Mul } = self.at(*mul_op) {
                if let Op::Const(c_const) = self.at(*c_op) {
                    if let Some(c_val) = constant_as_u64(c_const) {
                        if c_val == c as u64 {
                            return Some((*a, *add_op, c));
                        }
                    }
                }
            }
        }
        
        // Also check for the form: a + b * c (commutative)
        if let Op::Binary { x: add_op, y: mul_op, bop: BOp::Add } = self.at(x) {
            if let Op::Binary { x: b, y: c_op, bop: BOp::Mul } = self.at(*mul_op) {
                if let Op::Const(c_const) = self.at(*c_op) {
                    if let Some(c_val) = constant_as_u64(c_const) {
                        if c_val == c as u64 {
                            return Some((*b, *add_op, c));
                        }
                    }
                }
            }
            // Also: b * c + a
            if let Op::Binary { x: c_op, y: b, bop: BOp::Mul } = self.at(*mul_op) {
                if let Op::Const(c_const) = self.at(*c_op) {
                    if let Some(c_val) = constant_as_u64(c_const) {
                        if c_val == c as u64 {
                            return Some((*b, *add_op, c));
                        }
                    }
                }
            }
        }
        
        None
    }
    
    // Try to extract (a + b) pattern
    fn extract_add(&self, x: OpId) -> Option<(OpId, OpId)> {
        if let Op::Binary { x, y, bop: BOp::Add } = self.at(x) {
            return Some((*x, *y));
        }
        None
    }
}

enum SimplifyResult {
    ReplaceWith(Op),
    ForwardTo(OpId), // Replace current op by forwarding to src
}

// Helper to get u64 value from Constant
fn constant_as_u64(c: &Constant) -> Option<u64> {
    match c {
        Constant::U32(x) => Some(*x as u64),
        Constant::U64(x) => Some(u64::from_le_bytes(*x)),
        _ => None,
    }
}

// Helper to create a U64 constant
fn make_constant_u64(val: u64) -> Constant {
    Constant::U64(val.to_le_bytes())
}