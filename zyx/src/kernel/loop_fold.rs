// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use crate::{
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId},
};

fn add(k: &Kernel, x: OpId) -> Option<(OpId, OpId)> {
    let Op::Binary { x: a, y: b, bop: BOp::Add } = k.at(x) else { return None };
    Some((*a, *b))
}

impl Kernel {
    pub fn simplify_accumulating_loop(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);

            if let Op::Loop { len, .. } = self.at(op_id).clone() {
                if let Some(body) = self.get_loop_body(op_id) {
                    if body.len() == 3 {
                        if let Some(result) = self.try_simplify_accumulating_loop(op_id, len as u32) {
                            eprintln!("Simplified loop op_id={}", op_id.0);
                            match result {
                                LoopSimplifyResult::ReplaceWith(new_op) => {
                                    self.ops[op_id].op = new_op;
                                }
                                LoopSimplifyResult::FoldToConstant(val) => {
                                    self.ops[op_id].op = Op::Const(Constant::idx(val));
                                }
                            }
                        }
                    }
                }
            }

            op_id = next;
        }

        self.verify();
    }

    fn try_simplify_accumulating_loop(&self, loop_id: OpId, len: u32) -> Option<LoopSimplifyResult> {
        let loop_body = self.get_loop_body(loop_id)?;
        
        if loop_body.len() != 3 {
            return None;
        }

        let load_op = loop_body.get(0)?;
        let add_op = loop_body.get(1)?;
        let store_op = loop_body.get(2)?;

        let Op::Load { src, .. } = self.at(*load_op) else {
            return None;
        };

        let Op::Binary { x: add_x, y: add_y, bop: BOp::Add } = self.at(*add_op) else {
            return None;
        };

        let Op::Store { dst, x: store_x, .. } = self.at(*store_op) else {
            return None;
        };

        if *dst != *src {
            return None;
        }

        if *store_x != *add_op {
            return None;
        }

        let add_val = if *add_x == *load_op {
            *add_y
        } else if *add_y == *load_op {
            *add_x
        } else {
            return None;
        };

        let add_val_bounds = self.compute_expression_bounds(add_val)?;
        let (add_val_l, add_val_u) = add_val_bounds;
        
        let loop_iterations = len as u64;
        
        if add_val_l == add_val_u && add_val_l > 0 {
            let total = add_val_l as u64 * loop_iterations;
            return Some(LoopSimplifyResult::FoldToConstant(total as u32));
        }

        if add_val_u > 0 && add_val_l > 0 {
            let max_total = add_val_u as u64 * loop_iterations;
            if max_total < 0xFFFFFFFF {
                return Some(LoopSimplifyResult::ReplaceWith(Op::Binary {
                    x: add_val,
                    y: OpId::from(len as usize),
                    bop: BOp::Mul,
                }));
            }
        }

        None
    }

    fn get_loop_body(&self, loop_id: OpId) -> Option<Vec<OpId>> {
        let mut body = Vec::new();
        let mut op_id = self.next_op(loop_id);
        
        while !op_id.is_null() {
            match self.at(op_id) {
                Op::EndLoop => return Some(body),
                Op::Loop { .. } => return None,
                op => {
                    if !matches!(op, Op::Const(_) | Op::Index { .. }) {
                        body.push(op_id);
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        
        None
    }

    fn get_end_loop(&self, loop_id: OpId) -> Option<OpId> {
        let mut op_id = self.next_op(loop_id);
        
        while !op_id.is_null() {
            match self.at(op_id) {
                Op::EndLoop => return Some(op_id),
                Op::Loop { .. } => return None,
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        
        None
    }

    fn print_op(&self, op_id: OpId) -> String {
        format!("{}: {:?}", op_id.0, self.at(op_id))
    }

    fn compute_expression_bounds(&self, op_id: OpId) -> Option<(u32, u32)> {
        let bounds = self.compute_bounds();
        bounds.get(&op_id).copied()
    }
}

enum LoopSimplifyResult {
    ReplaceWith(Op),
    FoldToConstant(u32),
}