// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    dtype::{Constant, DType},
    kernel::{BOp, Kernel, Op, OpId, Scope},
};

impl Kernel {
    pub fn simplify_accumulating_loop(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            if self.fold_arange_loop(op_id) {
                break;
            }
            op_id = self.next_op(op_id);
        }

        self.verify();
    }

    fn fold_arange_loop(&mut self, acc_id: OpId) -> bool {
        let &Op::Define { dtype: acc_dtype, scope: Scope::Register, ro: false, len: 1 } = self.at(acc_id) else {
            return false;
        };

        let mut store_id = self.next_op(acc_id);
        while !store_id.is_null() {
            if let &Op::Store { dst, index, .. } = self.at(store_id) {
                if dst == acc_id {
                    if let Op::Const(cst) = self.at(index) {
                        if let Some(0) = cst.as_dim() {
                            break;
                        }
                    }
                }
            }
            store_id = self.next_op(store_id);
        }
        if store_id.is_null() {
            return false;
        }

        let mut loop_id = self.next_op(store_id);
        while !loop_id.is_null() {
            if matches!(self.at(loop_id), Op::Loop { .. }) {
                break;
            }
            match self.at(loop_id) {
                Op::Load { src, .. } if *src == acc_id => return false,
                Op::Store { dst, .. } if *dst == acc_id => return false,
                _ => {}
            }
            loop_id = self.next_op(loop_id);
        }
        let Op::Loop { .. } = self.at(loop_id) else { return false };

        let Some((accumulated_value_id, after_loop_load_id)) = self.identify_accumulate_pattern(acc_id, loop_id) else {
            return false;
        };

        let mut search_id = self.next_op(acc_id);
        let mut store_id = OpId::NULL;
        while !search_id.is_null() {
            if let &Op::Store { dst, .. } = self.at(search_id) {
                if dst == acc_id {
                    store_id = search_id;
                    break;
                }
            }
            search_id = self.next_op(search_id);
        }
        if store_id.is_null() {
            return false;
        }
        let &Op::Store { x: init_value, .. } = self.at(store_id) else { return false };

        if !self.replace_loop_with_closed_form(loop_id, init_value, accumulated_value_id, acc_dtype, after_loop_load_id) {
            return false;
        }

        let mut current = self.next_op(loop_id);
        while !current.is_null() {
            let next = self.next_op(current);
            if matches!(self.at(current), Op::EndLoop) {
                self.remove_op(current);
                break;
            }
            self.remove_op(current);
            current = next;
        }
        self.remove_op(loop_id);
        self.remove_op(store_id);
        self.remove_op(acc_id);

        self.verify();
        true
    }

    fn identify_accumulate_pattern(&self, acc_id: OpId, loop_id: OpId) -> Option<(OpId, OpId)> {
        let mut load_id = loop_id;
        loop {
            if let Op::Load { src, .. } = self.ops[load_id].op {
                if src == acc_id {
                    break;
                }
            }
            load_id = self.next_op(load_id);
        }

        let &Op::Load { src, index, vlen: 1 } = self.at(load_id) else { return None };
        let &Op::Const(index) = self.at(index) else { return None };
        if index.as_dim() != Some(0) {
            return None;
        }
        if src != acc_id {
            return None;
        }

        let add_id = self.next_op(load_id);
        let &Op::Binary { x: accumulated_value_id, y, bop: BOp::Add } = self.at(add_id) else { return None };
        if y != load_id {
            return None;
        }

        let store_id = self.next_op(add_id);
        let &Op::Store { dst, x, index, vlen: 1 } = self.at(store_id) else { return None };
        let &Op::Const(index) = self.at(index) else { return None };
        if index.as_dim() != Some(0) {
            return None;
        }
        if dst != acc_id || x != add_id {
            return None;
        }

        let endloop_id = self.next_op(store_id);
        let Op::EndLoop = self.at(endloop_id) else { return None };

        let load2_id = self.next_op(endloop_id);
        let &Op::Load { src, index, vlen: 1 } = self.at(load2_id) else { return None };
        let &Op::Const(index) = self.at(index) else { return None };
        if index.as_dim() != Some(0) {
            return None;
        }
        if src != acc_id {
            return None;
        }

        Some((accumulated_value_id, load2_id))
    }

    fn replace_loop_with_closed_form(
        &mut self,
        loop_id: OpId,
        init_value: OpId,
        accumulated_value_id: OpId,
        acc_dtype: DType,
        after_loop_load_id: OpId,
    ) -> bool {
        let &Op::Loop { len: loop_len } = self.at(loop_id) else { return false };

        let Some((a, b, c, mul_const)) = self.trace_to_linear_comparison(accumulated_value_id) else {
            return false;
        };

        if a != 1 || b != 1 {
            return false;
        }

        let start = if let Op::Const(cst) = self.at(init_value) {
            cst.as_dim().unwrap_or(0)
        } else {
            0
        };
        eprintln!(
            "DETECTED: loop_len={}, c={}, mul_const={}, start={}",
            loop_len, c, mul_const, start
        );

        let mut gidx_id = OpId::NULL;
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let &Op::Index { scope: Scope::Global, .. } = self.at(op_id) {
                gidx_id = op_id;
                break;
            }
            op_id = self.next_op(op_id);
        }
        if gidx_id.is_null() {
            return false;
        }

        let step = mul_const;
        let offset = loop_len.saturating_sub(c).saturating_sub(1);
        eprintln!("FORMULA: loop_len={}, c={}, step={}, offset={}", loop_len, c, step, offset);
        let offset_id = self.insert_before(after_loop_load_id, Op::Const(Constant::idx(offset)));
        let sum_id = self.insert_before(after_loop_load_id, Op::Binary { x: gidx_id, y: offset_id, bop: BOp::Add });
        let step_id = self.insert_before(after_loop_load_id, Op::Const(Constant::idx(step)));
        let result_id = self.insert_before(after_loop_load_id, Op::Binary { x: sum_id, y: step_id, bop: BOp::Mul });

        self.ops[after_loop_load_id].op = Op::Cast { x: result_id, dtype: acc_dtype };

        self.verify();
        true
    }

    fn build_arange_formula_step1(&mut self, loop_len: u64, c: u64, gidx_id: OpId, insert_before: OpId) -> OpId {
        let offset = loop_len.saturating_sub(c).saturating_sub(1);
        eprintln!("FORMULA step1: loop_len={}, c={}, offset={}", loop_len, c, offset);
        let offset_id = self.insert_before(insert_before, Op::Const(Constant::idx(offset)));
        self.insert_before(insert_before, Op::Binary { x: gidx_id, y: offset_id, bop: BOp::Add })
    }

    fn build_arange_formula_step_k(&mut self, loop_len: u64, c: u64, step: u64, gidx_id: OpId, insert_before: OpId) -> OpId {
        let offset = loop_len.saturating_sub(c).saturating_sub(2);
        eprintln!(
            "FORMULA step_k: loop_len={}, c={}, step={}, offset={}",
            loop_len, c, step, offset
        );
        let offset_id = self.insert_before(insert_before, Op::Const(Constant::idx(offset)));
        let sum_id = self.insert_before(insert_before, Op::Binary { x: gidx_id, y: offset_id, bop: BOp::Add });
        let step_id = self.insert_before(insert_before, Op::Const(Constant::idx(step)));
        self.insert_before(insert_before, Op::Binary { x: sum_id, y: step_id, bop: BOp::Mul })
    }

    fn trace_to_linear_comparison(&self, accumulated_value_id: OpId) -> Option<(u64, u64, u64, u64)> {
        if let Op::Index { scope: Scope::Global, .. } = self.at(accumulated_value_id) {
            return None;
        }

        if let Op::Cast { x, .. } = self.at(accumulated_value_id) {
            return self.trace_cmpgt(*x, 1);
        }

        if let Op::Binary { x: mul_x, y: mul_y, bop: BOp::Mul } = self.at(accumulated_value_id) {
            let mul_const = if let Op::Const(c) = self.at(*mul_x) {
                c.as_dim().unwrap_or(1)
            } else if let Op::Const(c) = self.at(*mul_y) {
                c.as_dim().unwrap_or(1)
            } else {
                return None;
            };
            let next_op = if let Op::Const(_) = self.at(*mul_x) { *mul_y } else { *mul_x };
            if let Op::Cast { x, .. } = self.at(next_op) {
                return self.trace_cmpgt(*x, mul_const);
            }
        }

        None
    }

    fn trace_cmpgt(&self, op_id: OpId, mul_const: u64) -> Option<(u64, u64, u64, u64)> {
        if let Op::Binary { x, y, bop: BOp::Cmpgt } = self.at(op_id) {
            let c = if let Op::Const(threshold) = self.at(*y) {
                threshold.as_dim().unwrap_or(0)
            } else {
                return None;
            };

            if let Op::Binary { bop: BOp::Add, .. } = self.at(*x) {
                return Some((1, 1, c, mul_const));
            }
        }
        None
    }
}
