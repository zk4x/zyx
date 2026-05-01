// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! ## Loop Folding (`fold_loops.rs`)
//!
//! This module optimizes loops that iteratively accumulate values into a closed-form
//! computation. The classic pattern this transforms is:
//!
//! ```c
//! acc = 0           // register define, length 1
//! acc[0] = 0       // store init value at index 0
//! for (i = 0; i < n; i++) {
//!     tmp = acc[0]            // load accumulator
//!     tmp = tmp + f(i)        // add new value
//!     acc[0] = tmp            // store back to accumulator
//! }
//! result = acc[0]            // load after loop
//! ```
//!
//! This is essentially computing something like `sum(f(0) + f(1) + ... + f(n-1))` or
//! `arange(0, n, 1).sum()`. The `fold_loops` optimization detects this pattern and replaces
//! it with a direct closed-form computation instead of iterating.
//!
//! The transformation works by:
//! 1. Detecting the accumulate pattern (a register write, loop, load-add-store sequence)
//! 2. Analyzing what value is being accumulated (must be loop-index-based arithmetic)
//! 3. If it's a simple pattern (like sum of 0+1+2+...), replace with arithmetic formula

use crate::{
    dtype::{Constant, DType},
    kernel::{BOp, IDX_T, Kernel, Op, OpId, Scope},
};

impl Kernel {
    /// Main entry point for loop folding optimization.
    /// Scans through operations looking for accumulating loops that can be simplified.
    /// Currently processes only one such loop per call (bails early after first match).
    pub fn simplify_accumulating_loop(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("simplify_accumulating_loop");

        let mut op_id = self.head;
        while !op_id.is_null() {
            if self.fold_loop(op_id) {
                break;
            }
            op_id = self.next_op(op_id);
        }

        self.verify();
    }

    /// Attempts to fold a specific accumulating loop starting at the given define.
    ///
    /// This is the main pattern matcher for `fold_loops`. It looks for:
    ///
    /// 1. A register define with length 1 (the accumulator variable)
    /// 2. An initial store to index 0 (the init value)
    /// 3. A Loop (the accumulating iteration)
    /// 4. The accumulate pattern inside the loop (load, add, store)
    /// 5. A load after the loop (the final value)
    ///
    /// Returns true if the loop was successfully folded, false otherwise.
    /// On success, the loop and accumulator are removed and replaced with closed-form ops.
    fn fold_loop(&mut self, acc_id: OpId) -> bool {
        // Step 1: Check that acc_id is a register define with length 1 (scalar accumulator)
        let &Op::Define { dtype: acc_dtype, scope, ro, len: 1 } = self.at(acc_id) else {
            return false;
        };
        // We only fold register-scoped accumulators; global/local have different semantics
        if scope != Scope::Register || ro {
            return false;
        }

        // Step 2: Find the initial store to the accumulator (acc[0] = init_value)
        let mut store_id = self.next_op(acc_id);
        while !store_id.is_null() {
            if let &Op::Store { dst, index, .. } = self.at(store_id) {
                if dst == acc_id {
                    // Looking for store at index 0 (the init value)
                    if let Op::Const(cst) = self.at(index) {
                        if cst.as_dim() == Some(0) {
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

        // Step 3: Skip forward until we find the Loop, guarding against other uses of accumulator
        // (if accumulator is used elsewhere, we can't fold)
        let mut loop_id = self.next_op(store_id);
        while !loop_id.is_null() {
            if matches!(self.at(loop_id), Op::Loop { .. }) {
                break;
            }
            // If accumulator is touched before the loop by anything other than the init store, abort
            match self.at(loop_id) {
                Op::Load { src, .. } if *src == acc_id => return false,
                Op::Store { dst, .. } if *dst == acc_id => return false,
                _ => {}
            }
            loop_id = self.next_op(loop_id);
        }
        let Op::Loop { .. } = self.at(loop_id) else { return false };

        // Step 4: Identify the accumulate pattern inside the loop
        // Pattern: load(acc[0]) -> add(value) -> store(acc[0])
        let Some((accumulated_value_id, after_loop_load_id)) = self.identify_accumulate_pattern(acc_id, loop_id) else {
            return false;
        };

        // Re-find the initial store to get the init_value (we need it for closed-form)
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

        // : Replace the loop with closed-form arithmetic (arange)
        if self.replace_arange_loop(acc_id, store_id, loop_id, accumulated_value_id, after_loop_load_id) {
            return true;
        }

        // : Replace the loop with closed-form arithmetic (gather)
        if self.replace_gather_loop(acc_dtype, loop_id, accumulated_value_id, after_loop_load_id) {
            return true;
        }

        false
    }

    /// Identifies the accumulate pattern inside a loop.
    ///
    /// Looks for this specific sequence inside the loop:
    /// - Load from accumulator at index 0
    /// - Binary add with some value
    /// - Store back to accumulator at index 0
    ///
    /// If found, returns the accumulated value ID and the load after the loop.
    /// The accumulated value is typically something like `i` or `i*i` (loop-index-based).
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

    /// Detects and replaces the `index_select`/`gather` loop pattern.
    ///
    /// Pattern:
    /// ```c
    /// acc = 0;
    /// for (i = 0; i < dim_size; i++) {
    ///     if (index == i) {
    ///         acc += source;
    ///     }
    /// }
    /// ```
    ///
    /// Replaces with:
    /// ```c
    /// i = index
    /// acc = source;
    /// ```
    fn replace_gather_loop(
        &mut self,
        _acc_dtype: DType,
        loop_id: OpId,
        accumulated_value_id: OpId,
        after_loop_load_id: OpId,
    ) -> bool {
        // accumulated value must be a binary multiply (mask * source)
        let &Op::Binary { x, y, bop: BOp::Mul } = self.at(accumulated_value_id) else {
            return false;
        };

        let (source_id, indices_id) = if let Some(indices_id) = self.get_indices(x, loop_id) {
            (y, indices_id)
        } else if let Some(indices_id) = self.get_indices(y, loop_id) {
            (x, indices_id)
        } else {
            return false;
        };
        //self.debug();

        //println!("Applying loop removal with loop_id={loop_id}, indices_id={indices_id}, source_id={source_id}");

        self.ops[loop_id].op = Op::Const(Constant::idx(0));

        //let Op::Loop { len: loop_len } = self.ops[loop_id].op else { return false };

        // Convert indices to IDX_T
        let loop_replace = self.insert_after(indices_id, Op::Cast { x: indices_id, dtype: IDX_T });
        //let loop_size = self.insert_after(index_casted, Op::Const(Constant::idx(loop_len)));
        //let loop_replace = self.insert_after(loop_size, Op::Binary { x: index_casted, y: loop_size, bop: BOp::Cmplt });

        // Replace loop index
        let endloop_id = self.prev_op(after_loop_load_id);
        let mut op_id = loop_replace;
        while op_id != endloop_id {
            for param in self.ops[op_id].op.parameters_mut() {
                if *param == loop_id {
                    *param = loop_replace;
                }
            }
            op_id = self.next_op(op_id);
        }
        self.remove_op(endloop_id);
        // Replace accumulator load
        self.remap(after_loop_load_id, source_id);
        //self.debug();
        self.verify();
        true
    }

    /// Find the equality op
    fn get_indices(&self, mask_id: OpId, loop_id: OpId) -> Option<OpId> {
        let (x, y) = match self.ops[mask_id].op {
            Op::Binary { x, y, bop: BOp::Eq } => (x, y),
            Op::Cast { x, .. } => match self.ops[x].op {
                Op::Binary { x, y, bop: BOp::Eq } => (x, y),
                _ => return None,
            },
            _ => return None,
        };
        let indices_id = if self.check_loop(x, loop_id) {
            y
        } else if self.check_loop(y, loop_id) {
            x
        } else {
            return None;
        };
        //println!("Found indices");
        Some(indices_id)
    }

    /// Returns all ops to the loop from the mask, the last op is the loop
    fn check_loop(&self, op_id: OpId, loop_id: OpId) -> bool {
        let Op::Cast { x, .. } = self.ops[op_id].op else { return false };
        if x != loop_id {
            return false;
        }
        //println!("Found loop");
        true
    }

    /// Replaces a loop with closed-form arithmetic if possible.
    ///
    /// This analyzes what value is being accumulated and tries to replace the iteration
    /// with a direct formula. For example, if accumulating `i` from 0 to n-1:
    /// - Original: sum = 0; for(i=0;i<n;i++) sum += i;
    /// - Closed form: sum = (n-1) * n / 2
    ///
    /// The arithmetic formula generated is:
    ///   result = (gidx + offset) * step
    /// Where `offset` = `loop_len` - `c` - 1 (for summing `0..n-1`, this is `n-1`)
    /// And step is the multiplication factor if the value is like `i*i` (step=1) or `2*i` (step=2)
    ///
    /// Returns true if closed-form was applied, false if the pattern can't be simplified.
    fn replace_arange_loop(
        &mut self,
        acc_id: OpId,
        store_id: OpId,
        loop_id: OpId,
        accumulated_value_id: OpId,
        after_loop_load_id: OpId,
    ) -> bool {
        let &Op::Loop { len: loop_len } = self.at(loop_id) else { return false };
        let &Op::Define { dtype, scope: Scope::Register, ro: false, len: 1 } = self.at(acc_id) else { return false };

        let Some((a, b, c, mul_const, gidx_id)) = self.trace_to_linear_comparison(accumulated_value_id, loop_id) else {
            return false;
        };

        if a != 1 || b != 1 {
            return false;
        }

        if !self.is_condition_based_accumulation(accumulated_value_id) {
            return false;
        }

        let step = mul_const;
        let offset = loop_len - c - 1;
        let offset_id = self.insert_before(after_loop_load_id, Op::Const(Constant::idx(offset)));
        let sum_id = self.insert_before(after_loop_load_id, Op::Binary { x: gidx_id, y: offset_id, bop: BOp::Add });
        let step_id = self.insert_before(after_loop_load_id, Op::Const(Constant::idx(step)));
        let result_id = self.insert_before(after_loop_load_id, Op::Binary { x: sum_id, y: step_id, bop: BOp::Mul });

        self.ops[after_loop_load_id].op = Op::Cast { x: result_id, dtype };

        // Remove the now-obsolete loop operations (Loop, body, EndLoop, init store, define)
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

    /// Traces through operations to find a linear comparison pattern.
    ///
    /// This walks backwards from the accumulated value to find:
    /// - A multiplication by a constant (like 1*i, 2*i, etc.)
    /// - An addition with the loop index
    /// - A comparison gt with a threshold
    ///
    /// Returns (a, b, c, `mul_const`, gidx) where the pattern being accumulated is:
    ///   `a * (loop_idx + b) * mul_const < c`
    /// Or for simple sum-of-index case: `loop_idx < n`
    ///
    /// For example, if accumulating `i` (the loop index directly):
    ///   a=1, b=1, c=n, `mul_const`=1, gidx is the loop index variable
    fn trace_to_linear_comparison(&self, accumulated_value_id: OpId, loop_id: OpId) -> Option<(u64, u64, u64, u64, OpId)> {
        if let Op::Index { scope: Scope::Global, .. } = self.at(accumulated_value_id) {
            return None;
        }

        if let Op::Cast { x, .. } = self.at(accumulated_value_id) {
            return self.trace_cmpgt(*x, 1, loop_id);
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
                return self.trace_cmpgt(*x, mul_const, loop_id);
            }
        }

        if let Op::Binary { x: add_x, y: add_y, bop: BOp::Add } = self.at(accumulated_value_id) {
            if let Op::Cast { x, .. } = self.at(*add_x) {
                return self.trace_cmpgt(*x, 1, loop_id);
            }
            if let Op::Cast { x, .. } = self.at(*add_y) {
                return self.trace_cmpgt(*x, 1, loop_id);
            }
            let next_op = *add_x;
            if let Op::Cast { x, .. } = self.at(next_op) {
                let mul_const = if let Op::Cast { .. } = self.at(*add_y) { 2 } else { 1 };
                return self.trace_cmpgt(*x, mul_const, loop_id);
            }
            let next_op = *add_y;
            if let Op::Cast { x, .. } = self.at(next_op) {
                let mul_const = if let Op::Cast { .. } = self.at(*add_x) { 2 } else { 1 };
                return self.trace_cmpgt(*x, mul_const, loop_id);
            }
        }

        None
    }

    /// Looks for a comparison pattern: `loop_idx + offset > threshold`
    ///
    /// This is the innermost pattern we expect: a Binary with Cmpgt where one operand
    /// is the loop index plus/minus a constant, and the other is a constant threshold.
    ///
    /// Example: `gidx + 1 > n` returns (1, 1, n, `mul_const`, gidx)
    fn trace_cmpgt(&self, op_id: OpId, mul_const: u64, loop_id: OpId) -> Option<(u64, u64, u64, u64, OpId)> {
        if let Op::Binary { x, y, bop: BOp::Cmpgt } = self.at(op_id) {
            let c = if let Op::Const(threshold) = self.at(*y) {
                threshold.as_dim().unwrap_or(0)
            } else {
                return None;
            };

            if let Op::Binary { x: add_x, y: add_y, bop: BOp::Add } = self.at(*x) {
                let gidx = if *add_x == loop_id {
                    *add_y
                } else if *add_y == loop_id {
                    *add_x
                } else {
                    return None;
                };
                // We need to check gidx is declared before loop
                let mut x = gidx;
                while x != op_id {
                    if x == loop_id {
                        return Some((1, 1, c, mul_const, gidx));
                    }
                    x = self.next_op(x);
                }
            }
        }
        None
    }

    /// Checks if the operation represents accumulation based on the loop condition.
    ///
    /// This detects whether the accumulated value comes from a comparison with the loop index.
    /// The pattern is typically: something * (`loop_idx` < threshold ? 1 : 0)
    /// Which means "add 1 if condition is true, else add 0" - i.e., conditionally accumulate.
    ///
    /// We verify this by walking through Cast and Mul operations until we find a Cmpgt.
    /// If the chain ends in Cmpgt, it's condition-based accumulation.
    fn is_condition_based_accumulation(&self, op_id: OpId) -> bool {
        match self.at(op_id) {
            Op::Cast { x, .. } => self.is_condition_based_accumulation(*x),
            Op::Binary { x: _, y: _, bop: BOp::Mul } => {
                let mut current = op_id;
                loop {
                    match self.at(current) {
                        Op::Cast { x, .. } => current = *x,
                        Op::Binary { x: mul_x, y: mul_y, bop: BOp::Mul } => {
                            if let Op::Const(_) = self.at(*mul_x) {
                                current = *mul_y;
                            } else if let Op::Const(_) = self.at(*mul_y) {
                                current = *mul_x;
                            } else {
                                return false;
                            }
                        }
                        Op::Binary { bop: BOp::Cmpgt, .. } => return true,
                        _ => return false,
                    }
                }
            }
            Op::Binary { bop: BOp::Cmpgt, .. } => true,
            _ => false,
        }
    }
}
