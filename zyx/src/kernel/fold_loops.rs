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
    kernel::{BOp, IDX_T, Kernel, MemLayout, Op, OpId, Scope},
};

impl Kernel {
    /// Main entry point for loop folding optimization.
    /// Scans through operations looking for accumulating loops that can be simplified.
    /// Currently processes only one such loop per call (bails early after first match).
    pub(crate) fn simplify_accumulating_loop(&mut self) {
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

        let &Op::Load { src, index, layout: MemLayout::Scalar } = self.at(load_id) else { return None };
        let &Op::Const(index) = self.at(index) else { return None };
        if index.as_dim() != Some(0) {
            return None;
        }
        if src != acc_id {
            return None;
        }

        let mut add_id = self.next_op(load_id);
        let accumulated_value_id = loop {
            if add_id.is_null() {
                return None;
            }
            match self.at(add_id) {
                Op::EndLoop => return None,
                Op::Store { dst, .. } if *dst == acc_id => return None,
                Op::Binary { x, y, bop: BOp::Add } if *y == load_id => break *x,
                _ => {}
            }
            add_id = self.next_op(add_id);
        };

        let store_id = self.next_op(add_id);
        let &Op::Store { dst, x, index, layout: MemLayout::Scalar } = self.at(store_id) else { return None };
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
        let &Op::Load { src, index, layout: MemLayout::Scalar } = self.at(load2_id) else { return None };
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
        // Peel through Cast ops to find the Mul (e.g., f32(mask * source))
        let mul_id = self.peel_casts(accumulated_value_id);
        let &Op::Binary { x, y, bop: BOp::Mul } = self.at(mul_id) else {
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

        // Convert indices to IDX_T — this is the corrected index value
        let loop_replace = self.insert_after(indices_id, Op::Cast { x: indices_id, dtype: IDX_T });

        // If source is a tensor load, move it after loop_replace and fix its index to use loop_replace.
        // The original source index computes Add(base, loop_id), which loads source[base+0] when
        // loop runs with len=1. We need source[base + indices_adj] instead.
        if let Op::Load { index: old_idx, .. } = self.ops[source_id].op {
            let base_id = match self.ops[old_idx].op {
                Op::Binary { x, y, bop: BOp::Add } if y == loop_id || x == loop_id => {
                    if y == loop_id {
                        x
                    } else {
                        y
                    }
                }
                _ => return false,
            };
            self.move_op_after(source_id, loop_replace);
            let new_idx = self.insert_before(source_id, Op::Binary { x: base_id, y: loop_replace, bop: BOp::Add });
            if let Some(idx_param) = self.ops[source_id].op.parameters_mut().nth(1) {
                *idx_param = new_idx;
            }
        }

        // Replace loop_id with loop_replace in all ops after loop_replace up to endloop
        let endloop_id = self.prev_op(after_loop_load_id);
        let mut op_id = self.next_op(loop_replace);
        while op_id != endloop_id {
            for param in self.ops[op_id].op.parameters_mut() {
                if *param == loop_id {
                    *param = loop_replace;
                }
            }
            op_id = self.next_op(op_id);
        }

        // Make the mask always true: change Eq(_, loop_i32(loop_id)) to Eq(loop_replace, loop_replace).
        // This ensures the accumulator gets the corrected source value regardless of index value.
        let mask_operand = if x == source_id { y } else { x };
        let eq_id = self.peel_casts(mask_operand);
        match &mut self.ops[eq_id].op {
            Op::Binary { x, y, bop: BOp::Eq } => {
                *x = loop_replace;
                *y = loop_replace;
            }
            _ => return false,
        }

        // Set loop to run once — with corrected source index and always-true mask,
        // the body computes source[base + indices_adj] in a single iteration.
        self.ops[loop_id].op = Op::Loop { len: 1 };
        //self.debug();
        self.verify();
        true
    }

    /// Peel through consecutive Cast ops to find the inner op
    fn peel_casts(&self, mut op_id: OpId) -> OpId {
        loop {
            match self.ops[op_id].op {
                Op::Cast { x, .. } => op_id = x,
                _ => return op_id,
            }
        }
    }

    /// Find the equality op
    fn get_indices(&self, mask_id: OpId, loop_id: OpId) -> Option<OpId> {
        let Op::Binary { x, y, bop: BOp::Eq } = self.ops[self.peel_casts(mask_id)].op else { return None };
        let indices_id = if self.check_loop(x, loop_id) {
            y
        } else if self.check_loop(y, loop_id) {
            x
        } else {
            return None;
        };
        Some(indices_id)
    }

    /// Check if `op_id` traces back to `loop_id` through Casts
    fn check_loop(&self, op_id: OpId, loop_id: OpId) -> bool {
        self.peel_casts(op_id) == loop_id
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

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::kernel::{BOp, DeviceId, Kernel, MemLayout, Op, OpId, Scope};

    /// Build a kernel matching the REAL index_select IR pattern
    /// where the accumulated value is computed AFTER load(acc).
    /// This is the pattern that fold_loops FAILS to optimize.
    ///
    /// Kernel IR structure:
    ///   acc = 0
    ///   for i in 0..len:
    ///     src = load(source_tensor, i)         // some computation
    ///     tmp = load(acc, 0)                    // LOAD (found by identify_accumulate_pattern)
    ///     eq = Eq(loop_id, 5)                   // mask computation (interleaved!)
    ///     eq_f32 = Cast(eq, f32)
    ///     mul = Mul(eq_f32, src)               // accumulated value
    ///     add = Add(mul, tmp)                   // ADD (next_op(load) is NOT add!)
    ///     store(acc, add, 0)
    ///   end
    ///   result = load(acc, 0)
    ///
    /// identify_accumulate_pattern fails because next_op(load(tmp)) is eq, not Add.
    fn make_interleaved_gather_kernel(loop_len: u32) -> (Kernel, OpId) {
        let mut k = Kernel::new(DeviceId::AUTO);
        let acc = k.define(DType::F32, Scope::Register, false, 1);

        let zi = k.const_idx(0u32);
        let zf = k.const_val(0.0f32);
        k.store(acc, zf, zi, MemLayout::Scalar);

        let loop_id = k.loop_(loop_len as u64);

        // Some computation before load(acc) — e.g. loading source
        let _source = k.const_val(42.0f32); // simplified: no tensor load

        // LOAD ACC — identify_accumulate_pattern finds this
        let load_acc = k.load(acc, zi, MemLayout::Scalar);

        // Accumulated value computation AFTER load(acc) — interleaved!
        let index_val = k.const_idx(5u32);
        let eq = k.binary(loop_id, index_val, BOp::Eq);
        let eq_f32 = k.cast(eq, DType::F32);
        let _src = k.const_val(42.0f32); // source value (could be from tensor load above)
        let mul = k.binary(eq_f32, _src, BOp::Mul);

        // ADD: references load_acc (tmp), but next_op(load_acc) is NOT add
        let add = k.binary(mul, load_acc, BOp::Add);
        k.store(acc, add, zi, MemLayout::Scalar);
        k.end_loop();
        let _result = k.load(acc, zi, MemLayout::Scalar);

        (k, loop_id)
    }

    /// Sanity test: the simple pattern (accum value BEFORE load) IS optimized.
    fn make_flat_gather_kernel(loop_len: u32) -> (Kernel, OpId, OpId) {
        let mut k = Kernel::new(DeviceId::AUTO);
        let acc = k.define(DType::F32, Scope::Register, false, 1);

        let zi = k.const_idx(0u32);
        let zf = k.const_val(0.0f32);
        k.store(acc, zf, zi, MemLayout::Scalar);

        let loop_id = k.loop_(loop_len as u64);

        let index_val = k.const_idx(5u32);
        let eq = k.binary(loop_id, index_val, BOp::Eq);
        let eq_f32 = k.cast(eq, DType::F32);
        let source = k.const_val(42.0f32);
        let mul = k.binary(eq_f32, source, BOp::Mul);

        let load_acc = k.load(acc, zi, MemLayout::Scalar);
        let add = k.binary(mul, load_acc, BOp::Add);
        k.store(acc, add, zi, MemLayout::Scalar);
        k.end_loop();
        let result = k.load(acc, zi, MemLayout::Scalar);

        (k, loop_id, result)
    }

    #[test]
    fn test_flat_gather_is_optimized() {
        let (mut k, loop_id, _result) = make_flat_gather_kernel(10);
        k.simplify_accumulating_loop();
        assert_eq!(k.at(loop_id), &Op::Loop { len: 1 }, "flat pattern should fold");
    }

    #[test]
    fn test_interleaved_gather_is_optimized() {
        let (mut k, loop_id) = make_interleaved_gather_kernel(10);
        k.simplify_accumulating_loop();
        assert_eq!(k.at(loop_id), &Op::Loop { len: 1 }, "loop should fold");
    }

    /// Build a kernel matching the real gather kernel IR where the source index
    /// computation (which uses loop_id) appears BEFORE indices_id in the op order.
    /// This means replace_gather_loop's parameter replacement (which starts at
    /// loop_replace, inserted after indices_id) misses the source index computation,
    /// leaving it to reference the loop op which later becomes Const(0) — producing
    /// source[row*5+0] instead of source[row*5+indices[row][col]].
    fn make_gather_kernel_with_source_before_indices() -> (Kernel, OpId) {
        let mut k = Kernel::new(DeviceId::AUTO);

        // Constants
        let zi = k.const_idx(0u32);
        let zz = k.const_val(0u16);
        let c0_i32 = k.const_val(0i32);
        let c5_i32 = k.const_val(5i32);
        let base = k.const_idx(0u32); // r125 = row * 5 (simplified to 0 for test)

        // Output tensor globals (simplified — just use ones for layout)
        let _indices_tensor = k.define(DType::U16, Scope::Global, false, 9);
        let _source_tensor = k.define(DType::U16, Scope::Global, false, 15);

        // Accumulator
        let acc = k.define(DType::U16, Scope::Register, false, 1);
        k.store(acc, zz, zi, MemLayout::Scalar);

        let loop_id = k.loop_(5);

        // r20: i32 = i32(r3) — cast loop_id to i32 (for Eq)
        let loop_i32 = k.cast(loop_id, DType::I32);

        // r96: u16 = load(indices_tensor, pos) — load index value
        let indices_load = k.load(_indices_tensor, zi, MemLayout::Scalar);

        // r111: u32 = base + r3 — source index computation (references loop_id!)
        // This comes BEFORE indices_id (r35), so replace_gather_loop misses it!
        let source_idx = k.binary(base, loop_id, BOp::Add);

        // r115: u16 = load(source_tensor, r111) — load source value
        let source_load = k.load(_source_tensor, source_idx, MemLayout::Scalar);

        // r18: u16 = load(acc, 0) — load accumulator (found by identify_accumulate_pattern)
        let load_acc = k.load(acc, zi, MemLayout::Scalar);

        // r24: i32 = i32(r96) — cast indices to i32
        let idx_i32 = k.cast(indices_load, DType::I32);

        // r29: i32 = r24 < 0 (negative index handling)
        let neg_check = k.binary(idx_i32, c0_i32, BOp::Cmplt);

        // r30: i32 = i32(r29)
        let neg_flag = k.cast(neg_check, DType::I32);

        // r34: i32 = r30 * 5
        let neg_adjust = k.binary(neg_flag, c5_i32, BOp::Mul);

        // r35: i32 = r24 + r34 — adjusted indices (this is indices_id!)
        let indices_adj = k.binary(idx_i32, neg_adjust, BOp::Add);

        // r38: i32 = r35 == r20
        let eq = k.binary(indices_adj, loop_i32, BOp::Eq);

        // r39: u16 = u16(r38)
        let mask = k.cast(eq, DType::U16);

        // r45: u16 = r39 * r115
        let mul = k.binary(mask, source_load, BOp::Mul);

        // r42: u16 = r45 + r18
        let add = k.binary(mul, load_acc, BOp::Add);

        // store(acc, r42, 0)
        k.store(acc, add, zi, MemLayout::Scalar);

        k.end_loop();

        // r46: u16 = load(acc, 0) — result after loop
        let _result = k.load(acc, zi, MemLayout::Scalar);

        (k, loop_id)
    }

    /// Test that identifies the bug: source index computation using loop_id
    /// appears BEFORE indices_id, so replace_gather_loop misses it.
    #[test]
    fn test_gather_source_before_indices() {
        let (mut k, loop_id) = make_gather_kernel_with_source_before_indices();
        k.simplify_accumulating_loop();

        // Loop should have been folded
        assert_eq!(k.at(loop_id), &Op::Loop { len: 1 }, "loop should fold");
    }

    /// Reproduce the exact IR from resnet index_select kernel (ZYX_DEBUG=8 output).
    /// The outer loop (6250) + inner loop (8) accumulate pattern has interleaved
    /// ops between load(acc) and Add, so simplify_accumulating_loop should NOT fold it.
    #[test]
    #[should_panic = "outer loop should be zeroed"]
    fn test_resnet_index_select_ir_not_optimized() {
        let mut k = Kernel::new(DeviceId::AUTO);

        let r93 = k.define(DType::I32, Scope::Global, false, 50000);
        let r116 = k.define(DType::F32, Scope::Global, false, 153600000);
        let r128 = k.define(DType::F32, Scope::Global, true, 153600000);
        let r130 = k.const_idx(50000u32);
        let r1 = k.const_idx(0u32);
        let r42 = k.const_val(0.0f32);
        let r25 = k.const_val(0i32);
        let r30 = k.const_val(50000i32);
        let r106 = k.const_idx(3072u32);
        let r84 = k.const_idx(5u32);
        let r97 = k.const_idx(10u32);
        let r10 = k.const_idx(3u32);
        let r16 = k.gidx(0, 75000);
        let r92 = k.lidx(0, 2);
        let r2 = k.lidx(1, 32);
        let r78 = k.gidx(2, 4);
        let r27 = k.lidx(2, 8);
        let r50 = k.binary(r16, r16, BOp::Add);
        let r129 = k.binary(r50, r92, BOp::Add);
        let r104 = k.binary(r78, r10, BOp::BitShiftLeft);
        let r5 = k.binary(r104, r27, BOp::Add);
        let r22 = k.binary(r129, r130, BOp::Mod);
        let r131 = k.binary(r129, r130, BOp::Div);

        let r3 = k.define(DType::F32, Scope::Register, true, 1);
        k.store(r3, r42, r1, MemLayout::Scalar);

        let r135 = k.binary(r2, r84, BOp::BitShiftLeft);
        let r136 = k.binary(r131, r97, BOp::BitShiftLeft);

        let outer_loop = k.loop_(6250);

        let r53 = k.binary(outer_loop, r10, BOp::BitShiftLeft);

        let inner_loop = k.loop_(8);

        let r35 = k.binary(r53, inner_loop, BOp::Add);
        let r20 = k.cast(r35, DType::I32);
        let r94 = k.load(r93, r22, MemLayout::Scalar);
        let r107 = k.binary(r106, r35, BOp::Mul);
        let r109 = k.binary(r5, r107, BOp::Add);
        let r111 = k.binary(r135, r109, BOp::Add);
        let r113 = k.binary(r136, r111, BOp::Add);
        let r117 = k.load(r116, r113, MemLayout::Scalar);
        let r15 = k.load(r3, r1, MemLayout::Scalar);
        let r28 = k.binary(r94, r25, BOp::Cmplt);
        let r29 = k.cast(r28, DType::I32);
        let r71 = k.binary(r29, r30, BOp::Mul);
        let r34 = k.binary(r71, r94, BOp::Add);
        let r37 = k.binary(r34, r20, BOp::Eq);
        let r38 = k.cast(r37, DType::F32);
        let r118 = k.binary(r38, r117, BOp::Mul);
        let r9 = k.binary(r118, r15, BOp::Add);
        k.store(r3, r9, r1, MemLayout::Scalar);

        k.end_loop();
        k.end_loop();

        let r45 = k.load(r3, r1, MemLayout::Scalar);
        let r121 = k.binary(r22, r106, BOp::Mul);
        let r123 = k.binary(r136, r121, BOp::Add);
        let r125 = k.binary(r135, r123, BOp::Add);
        let r127 = k.binary(r5, r125, BOp::Add);
        k.store(r128, r45, r127, MemLayout::Scalar);

        k.simplify_accumulating_loop();

        assert_eq!(k.at(outer_loop), &Op::Loop { len: 1 }, "outer loop should be zeroed");
        assert_eq!(k.at(inner_loop), &Op::Loop { len: 1 }, "inner loop should be zeroed");
    }
}
