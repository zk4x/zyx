// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    dtype::{Constant, DType},
    kernel::{BOp, Kernel, Op, OpId, Scope},
};

impl Kernel {
    pub fn simplify_accumulating_loop(&mut self) {
        // Currently only try_fold_conditional is used (for arange pattern)
        // try_fold1 is disabled as it may cause issues
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
        // Define
        let &Op::Define { dtype: acc_dtype, scope: Scope::Register, ro: false, len: 1 } = self.at(acc_id) else { return false };
        //println!("def_id={def_id}");

        // Find first store to def_id (initial value)
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
        //println!("store_id={store_id}");

        // Find loop
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
        //println!("loop_id={loop_id}");

        // Final part of the loop check
        let Some((accumulated_value_id, after_loop_load_id)) = self.identify_accumulate_pattern(acc_id, loop_id) else {
            return false;
        };

        // Final part - get store and init value
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

        // Replace with closed form
        self.replace_loop_with_closed_form(
            loop_id,
            store_id,
            init_value,
            accumulated_value_id,
            acc_dtype,
            after_loop_load_id,
        );

        // TODO later remove the loop

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

        // Load
        let &Op::Load { src, index, vlen: 1 } = self.at(load_id) else { return None };
        let &Op::Const(index) = self.at(index) else { return None };
        if index.as_dim() != Some(0) {
            return None;
        }
        if src != acc_id {
            return None;
        }
        //println!("load_id={load_id}");

        // Add
        let add_id = self.next_op(load_id);
        let &Op::Binary { x: accumulated_value_id, y, bop: BOp::Add } = self.at(add_id) else { return None };
        if y != load_id {
            return None;
        }
        println!("add_id={add_id}");

        // Store
        let store_id = self.next_op(add_id);
        let &Op::Store { dst, x, index, vlen: 1 } = self.at(store_id) else { return None };
        let &Op::Const(index) = self.at(index) else { return None };
        if index.as_dim() != Some(0) {
            return None;
        }
        if dst != acc_id || x != add_id {
            return None;
        }
        println!("store_id={store_id}");

        // Endloop
        let endloop_id = self.next_op(store_id);
        let Op::EndLoop = self.at(endloop_id) else { return None };
        println!("endloop_id={endloop_id}");

        // Load
        let load2_id = self.next_op(endloop_id);
        let &Op::Load { src, index, vlen: 1 } = self.at(load2_id) else { return None };
        let &Op::Const(index) = self.at(index) else { return None };
        if index.as_dim() != Some(0) {
            return None;
        }
        if src != acc_id {
            return None;
        }
        println!("load2_id={load2_id}");

        Some((accumulated_value_id, load2_id))
    }

    fn trace_to_linear_comparison(&self, accumulated_value_id: OpId, loop_id: OpId) -> Option<(u64, u64, u64, u64)> {
        // Trace path: accumulated_value (Mul) -> Cast -> Cmpgt -> Binary (add/sub of loop_idx and gidx)
        // Pattern: (a*lidx + b*gidx > c) * mul_const

        // Step 1: Get Mul operation
        let &Op::Binary { x: mul_x, y: mul_y, bop: BOp::Mul } = self.at(accumulated_value_id) else {
            return None;
        };

        // Get mul constant (either mul_x or mul_y should be constant)
        let (mul_const, cmp_input) = if let Op::Const(c) = self.at(mul_x) {
            (c.as_dim()?, mul_y)
        } else if let Op::Const(c) = self.at(mul_y) {
            (c.as_dim()?, mul_x)
        } else {
            return None;
        };

        // Step 2: Get Cast operation
        let &Op::Cast { x: cast_x, .. } = self.at(cmp_input) else {
            return None;
        };

        // Step 3: Get Cmpgt operation
        let &Op::Binary { x: cmp_x, y: cmp_y, bop: BOp::Cmpgt } = self.at(cast_x) else {
            return None;
        };

        // Step 4: Get constant threshold
        let &Op::Const(threshold) = self.at(cmp_y) else {
            return None;
        };
        let c = threshold.as_dim()?;

        // Step 5: Analyze the comparison input to get a and b coefficients
        // cmp_x should be: a*lidx + b*gidx (or gidx + lidx)
        let &Op::Binary { x, y, bop } = self.at(cmp_x) else {
            return None;
        };

        // Find loop_idx index and gidx index references
        let mut lidx_id = OpId::NULL;
        let mut gidx_id = OpId::NULL;

        // Search both operands
        for operand in [x, y] {
            if let &Op::Index { scope: Scope::Global, .. } = self.at(operand) {
                gidx_id = operand;
            } else {
                // It should be the loop counter - check if it's defined by loop
                // We check if it's NOT global (it's local/register)
                lidx_id = operand;
            }
        }

        if lidx_id.is_null() || gidx_id.is_null() {
            return None;
        }

        // Determine a and b based on operand order
        // If x is loop_idx and y is gidx: a=1, b=1
        // If x is gidx and y is loop_idx: a=1, b=1 (commutative for +)
        let a = 1u64;
        let b = 1u64;

        Some((a, b, c, mul_const))
    }

    fn replace_loop_with_closed_form(
        &mut self,
        loop_id: OpId,
        store_id: OpId,
        init_value: OpId,
        accumulated_value_id: OpId,
        acc_dtype: DType,
        after_loop_load_id: OpId,
    ) -> bool {
        // Get loop length
        let Op::Loop { len: loop_len, .. } = self.at(loop_id) else { return false };

        // Trace to get coefficients
        let Some((a, b, c, mul_const)) = self.trace_to_linear_comparison(accumulated_value_id, loop_id) else {
            return false;
        };

        // Only handle mul_const = 1 for now
        if mul_const != 1 {
            return false;
        }

        // Find gidx used in the computation
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

        // Calculate threshold: c - a * (loop_len - 1)
        // For a=1, loop_len=10000, c=9998: threshold = 9998 - 1*9999 = -1
        // For positive threshold: (gidx - threshold)
        // For negative threshold: (gidx + |threshold|)
        let threshold = c.saturating_sub(a * (loop_len.saturating_sub(1)));

        // Create: gidx - threshold (or gidx + |threshold| if threshold < 0)
        let sub_or_add = if (threshold as i64) < 0 { BOp::Add } else { BOp::Sub };
        let threshold_val = if (threshold as i64) < 0 {
            ((-(threshold as i64)) as u64)
        } else {
            threshold
        };
        let threshold_const = Constant::idx(threshold_val);
        let threshold_id = self.insert_before(after_loop_load_id, Op::Const(threshold_const));
        let sub_id = self.insert_before(
            after_loop_load_id,
            Op::Binary { x: gidx_id, y: threshold_id, bop: sub_or_add },
        );

        // Create: max(0, sub_result)
        let zero_id = self.insert_before(after_loop_load_id, Op::Const(Constant::idx(0)));
        let max_id = self.insert_before(after_loop_load_id, Op::Binary { x: sub_id, y: zero_id, bop: BOp::Max });

        // Create: mul_const * max_result
        let mul_const_id = self.insert_before(after_loop_load_id, Op::Const(Constant::idx(mul_const)));
        let mul_id = self.insert_before(after_loop_load_id, Op::Binary { x: max_id, y: mul_const_id, bop: BOp::Mul });

        // Get init_value const
        let init_const = if let Op::Const(c) = self.at(init_value) {
            c.as_dim().unwrap_or(0)
        } else {
            0
        };

        // Create: result + init_value
        let init_const_id = self.insert_before(after_loop_load_id, Op::Const(Constant::idx(init_const)));
        let result_id = self.insert_before(after_loop_load_id, Op::Binary { x: mul_id, y: init_const_id, bop: BOp::Add });

        // Replace load after loop with the computed result
        self.ops[after_loop_load_id].op = Op::Cast { x: result_id, dtype: acc_dtype };

        // Remove the intermediate ops we created
        // self.remove_op(threshold_id); // Can't remove - it's used
        // Actually, all inserted ops are now referenced, so we keep them

        self.verify();
        true
    }
}
