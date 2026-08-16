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
    Set,
    dtype::{Constant, DType},
    kernel::{BOp, IDX_T, IdxKind, Kernel, MemLayout, MemScope, Op, OpId},
    slab::SlabId,
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
        // Check that acc_id is a register define with length 1 (scalar accumulator)
        let &Op::Storage { dtype: acc_dtype, scope, len } = self.at(acc_id) else {
            return false;
        };
        if len != 1 {
            return false;
        }
        // We only fold register-scoped accumulators; global/local have different semantics
        if scope != MemScope::Register {
            return false;
        }

        // Find the initial store to the accumulator (acc[0] = init_value)
        let mut store_id = self.next_op(acc_id);
        while !store_id.is_null() {
            if let &Op::Store { dst, index, .. } = self.at(store_id)
                && dst == acc_id
            {
                // Looking for store at index 0 (the init value)
                if let Op::Const(cst) = self.at(index)
                    && cst.as_dim() == Some(0)
                {
                    break;
                }
            }
            store_id = self.next_op(store_id);
        }
        if store_id.is_null() {
            return false;
        }

        // Skip forward until we find the Loop, guarding against other uses of accumulator
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

        // Identify the accumulate pattern inside the loop
        // Pattern: load(acc[0]) -> add(value) -> store(acc[0])
        let Some((accumulated_value_id, after_loop_load_id)) = self.identify_accumulate_pattern(acc_id, loop_id) else {
            return false;
        };

        // Re-find the initial store to get the init_value (we need it for closed-form)
        let mut search_id = self.next_op(acc_id);
        let mut store_id = OpId::NULL;
        while !search_id.is_null() {
            if let &Op::Store { dst, .. } = self.at(search_id)
                && dst == acc_id
            {
                store_id = search_id;
                break;
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
            if let Op::Load { src, .. } = self.ops[load_id].op
                && src == acc_id
            {
                break;
            }
            load_id = self.next_op(load_id);
        }

        let &Op::Load { src, index, layout: MemLayout::Scalar } = self.at(load_id) else {
            return None;
        };
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
        let &Op::Store { dst, src: x, index, layout: MemLayout::Scalar } = self.at(store_id) else {
            return None;
        };
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
        let &Op::Load { src, index, layout: MemLayout::Scalar } = self.at(load2_id) else {
            return None;
        };
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

        // Sort it so that indices_id comes first in the loop body
        let mut parents = Set::default();
        let mut params = vec![indices_id];
        while let Some(parent) = params.pop() {
            if parents.insert(parent) {
                params.extend(self.ops[parent].op.parameters());
            }
        }
        let after_loop = self.next_op(loop_id);
        let after_indices = self.next_op(indices_id);
        let mut op_id = loop_id;
        while op_id != after_indices && op_id != after_loop_load_id {
            let next = self.next_op(op_id);
            if parents.contains(&op_id) {
                self.move_op_before(op_id, after_loop);
            }
            op_id = next;
        }

        //println!("Applying loop removal with loop_id={loop_id}, indices_id={indices_id}, source_id={source_id}");

        self.ops[loop_id].op = Op::Const(Constant::idx(0));

        //let Op::Loop { len: loop_len } = self.ops[loop_id].op else { return false };

        // Convert indices to IDX_T
        let loop_replace = self.insert_after(indices_id, Op::Cast { x: indices_id, dtype: IDX_T });

        // Replace loop index
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
        self.remove_op(endloop_id);
        // Replace accumulator load
        self.remap(after_loop_load_id, source_id);
        self.verify();
        true
    }

    /// Find the equality op
    fn get_indices(&self, mask_id: OpId, loop_id: OpId) -> Option<OpId> {
        let Op::Binary { x, y, bop: BOp::Eq } = self.ops[self.peel_casts(mask_id)].op else {
            return None;
        };
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
        let peeled = self.peel_casts(op_id);
        if peeled == loop_id {
            return true;
        }
        false
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
        let &Op::Loop { len: loop_len_id } = self.at(loop_id) else {
            return false;
        };
        let loop_len = self.loop_len_dim(loop_len_id);
        let &Op::Storage { dtype, scope: MemScope::Register, len } = self.at(acc_id) else {
            return false;
        };
        if len != 1 {
            return false;
        }

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
        if let Op::Index { kind: IdxKind::Group, .. } = self.at(accumulated_value_id) {
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
            // Fallback: the comparison may be a ceil-style mask
            // `((k + gidx + coeff*loop) >> 1) > 0` that folds to `gidx > 0` at
            // `loop == 0` and is always true for `loop >= 1`. See trace_masked_ceil.
            if let Some(gidx) = self.trace_masked_ceil(op_id, loop_id) {
                return Some((1, 1, c, mul_const, gidx));
            }
        }
        None
    }

    /// Matches a ceil-style mask: `((k + gidx + coeff*loop) >> 1) > 0`.
    ///
    /// For unsigned arithmetic, `(x >> 1) > 0` is equivalent to `x >= 2`. With
    /// `k == 1` and `coeff >= 1`, `k + gidx + coeff*loop >= 2` fails only at
    /// `loop == 0` when `gidx == 0`. The number of true iterations over
    /// `loop in 0..len` is therefore `len - 1 + min(gidx, 1)`, which equals
    /// `gidx + len - 1` when the bounds of `gidx` are contained in `{0, 1}`.
    ///
    /// Returns the gidx op if the mask is of this form and the bounds check
    /// holds; the caller can then use the standard `(gidx + len - 1) * step`
    /// closed form.
    fn trace_masked_ceil(&self, mask_id: OpId, loop_id: OpId) -> Option<OpId> {
        let Op::Binary { x, y, bop: BOp::Cmpgt } = self.at(mask_id) else {
            return None;
        };
        let Op::Const(threshold) = self.at(*y) else {
            return None;
        };
        if threshold.as_dim() != Some(0) {
            return None;
        }
        let Op::Binary { x: sh_x, y: sh_y, bop: BOp::BitShiftRight } = self.at(*x) else {
            return None;
        };
        let Op::Const(shift) = self.at(*sh_y) else {
            return None;
        };
        if shift.as_dim() != Some(1) {
            return None;
        }
        let (k, coeff, gidx) = self.peel_mask_add(*sh_x, loop_id, 0, 0, OpId::NULL)?;
        if k != 1 || coeff == 0 || gidx == OpId::NULL {
            return None;
        }
        // Only safe when the outer index is provably in {0, 1}.
        let bounds = self.compute_bounds();
        match bounds.get(&gidx) {
            Some((_, hi)) if *hi <= 1 => Some(gidx),
            _ => None,
        }
    }

    /// Peels an addition tree rooted at `id` into `k + coeff * loop_id + gidx`.
    ///
    /// Returns `(k, coeff, gidx)` where `k` is the accumulated constant, `coeff`
    /// is the coefficient of the loop variable, and `gidx` is the single
    /// non-constant, non-loop term. Returns `None` if the tree contains anything
    /// else (e.g. multiple outer variables or a loop-dependent gidx).
    fn peel_mask_add(&self, id: OpId, loop_id: OpId, k: i64, coeff: u64, gidx: OpId) -> Option<(i64, u64, OpId)> {
        if id == loop_id {
            return Some((k, coeff.saturating_add(1), gidx));
        }
        match self.at(id) {
            Op::Binary { x, y, bop: BOp::Add } => {
                let (kx, cx, gx) = self.peel_mask_add(*x, loop_id, k, coeff, gidx)?;
                self.peel_mask_add(*y, loop_id, kx, cx, gx)
            }
            Op::Binary { x, y, bop: BOp::Mul } => {
                let (c, loop_side) = if *x == loop_id {
                    if let Op::Const(v) = self.at(*y) {
                        (v.as_dim(), true)
                    } else {
                        (None, false)
                    }
                } else if *y == loop_id {
                    if let Op::Const(v) = self.at(*x) {
                        (v.as_dim(), true)
                    } else {
                        (None, false)
                    }
                } else {
                    (None, false)
                };
                match c {
                    Some(d) if loop_side => Some((k, coeff.saturating_add(d), gidx)),
                    _ => None,
                }
            }
            Op::Binary { x, y, bop: BOp::BitShiftLeft } if *x == loop_id => {
                let Op::Const(v) = self.at(*y) else { return None };
                let d = v.as_dim()?;
                Some((k, coeff.saturating_add(1u64.checked_shl(u32::try_from(d).ok()?).unwrap_or(u64::MAX)), gidx))
            }
            Op::Const(v) => {
                let d = v.as_dim()?;
                Some((k.saturating_add(d as i64), coeff, gidx))
            }
            _ => {
                if gidx == OpId::NULL {
                    Some((k, coeff, id))
                } else {
                    None
                }
            }
        }
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
    use crate::dtype::Constant;
    use crate::dtype::DType;
    use crate::kernel::{BOp, DeviceId, Kernel, MemLayout, MemScope, Op, OpId, ParamKind};

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
        let acc = k.storage(DType::F32, MemScope::Register, 1);

        let zi = k.const_idx(0u32);
        let zf = k.const_val(0.0f32);
        k.store(acc, zf, zi, MemLayout::Scalar);

        let lc = k.const_idx(loop_len as u64);
        let loop_id = k.loop_(lc);

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
        let acc = k.storage(DType::F32, MemScope::Register, 1);

        let zi = k.const_idx(0u32);
        let zf = k.const_val(0.0f32);
        k.store(acc, zf, zi, MemLayout::Scalar);

        let lc = k.const_idx(loop_len as u64);
        let loop_id = k.loop_(lc);

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
        assert_eq!(k.at(loop_id), &Op::Const(Constant::idx(0)), "loop should fold");
    }

    #[test]
    fn test_interleaved_gather_is_optimized() {
        let (mut k, loop_id) = make_interleaved_gather_kernel(10);
        k.simplify_accumulating_loop();
        assert_eq!(k.at(loop_id), &Op::Const(Constant::idx(0)), "loop should fold");
    }

    /// Build a kernel matching the real gather kernel IR where the source index
    /// computation (which uses loop_id) appears BEFORE indices_id in the op order.
    /// This means replace_gather_loop's parameter replacement (which starts at
    /// loop_replace, inserted after indices_id) misses the source index computation,
    /// leaving it to reference the loop op which later becomes Const(0) — producing
    /// source[row*5+0] instead of source[row*5+indices[row][col]].
    fn make_gather_kernel_with_source_before_indices() -> (Kernel, OpId) {
        let mut k = Kernel::new(DeviceId::AUTO);

        let r95 = k.param(DType::U16, ParamKind::Global);
        let r114 = k.param(DType::U16, ParamKind::Global);
        let r122 = k.param(DType::U16, ParamKind::Global);
        let r7 = k.const_val(0u32);
        let r22 = k.const_val(0u16);
        let r74 = k.const_val(3u32);
        let r26 = k.const_val(0i32);
        let r31 = k.const_val(5i32);
        let r110 = k.const_val(5u32);
        let r37 = k.group_index(0, 3);
        let r5 = k.group_index(1, 3);
        let r1 = k.storage(DType::U16, MemScope::Register, 1);
        k.store(r1, r22, r7, MemLayout::Scalar);
        let r123 = k.binary(r37, r74, BOp::Mul);
        let r92 = k.binary(r123, r5, BOp::Add);
        let r71 = k.binary(r37, r110, BOp::Mul);

        let c5 = k.const_idx(5u32);
        let loop_id = k.loop_(c5);

        let r20 = k.cast(loop_id, DType::I32);
        let r96 = k.load(r95, r92, MemLayout::Scalar);
        let r111 = k.binary(r71, loop_id, BOp::Add);
        let r115 = k.load(r114, r111, MemLayout::Scalar);
        let r18 = k.load(r1, r7, MemLayout::Scalar);
        let r24 = k.cast(r96, DType::I32);
        let r29 = k.binary(r24, r26, BOp::Cmplt);
        let r30 = k.cast(r29, DType::I32);
        let r118 = k.binary(r30, r31, BOp::Mul);
        let r35 = k.binary(r24, r118, BOp::Add);
        let r38 = k.binary(r35, r20, BOp::Eq);
        let r39 = k.cast(r38, DType::U16);
        let r97 = k.binary(r39, r115, BOp::Mul);
        let r42 = k.binary(r97, r18, BOp::Add);
        k.store(r1, r42, r7, MemLayout::Scalar);

        k.end_loop();

        let r46 = k.load(r1, r7, MemLayout::Scalar);
        let r121 = k.binary(r5, r123, BOp::Add);
        k.store(r122, r46, r121, MemLayout::Scalar);

        (k, loop_id)
    }

    /// Test that identifies the bug: source index computation using loop_id
    /// appears BEFORE indices_id, so replace_gather_loop misses it.
    #[test]
    fn test_gather_source_before_indices() {
        if !crate::Tensor::dtype_capability(crate::DType::U16).any() {
            return;
        }
        let (mut k, loop_id) = make_gather_kernel_with_source_before_indices();
        k.simplify_accumulating_loop();

        assert_eq!(k.at(loop_id), &Op::Const(Constant::idx(0)), "loop should fold");

        let compiled = k.compile().unwrap();
        let source = crate::Tensor::from([[10u16, 20, 30, 40, 50], [11, 21, 31, 41, 51], [12, 22, 32, 42, 52]]);
        let indices = crate::Tensor::from([[0u16, 2, 4], [1, 3, 0], [4, 1, 2]]);
        let result = compiled.forward(&[&indices, &source], vec![[3, 3]]).unwrap().pop().unwrap();
        assert_eq!(result, [[10u16, 30, 50], [21, 41, 11], [52, 22, 32]]);
    }

    /// Reproduce the exact IR from resnet index_select kernel (ZYX_DEBUG=8 output).
    /// The outer loop (6250) + inner loop (8) accumulate pattern has interleaved
    /// ops between load(acc) and Add, so simplify_accumulating_loop should NOT fold it.
    #[test]
    #[should_panic]
    fn test_resnet_index_select_ir_not_optimized() {
        let mut k = Kernel::new(DeviceId::AUTO);

        let r93 = k.param(DType::F32, ParamKind::Global);
        let r116 = k.param(DType::F32, ParamKind::Global);
        let r128 = k.param(DType::F32, ParamKind::Global);
        let r130 = k.const_idx(50000u32);
        let r1 = k.const_idx(0u32);
        let r42 = k.const_val(0.0f32);
        let r25 = k.const_val(0i32);
        let r30 = k.const_val(50000i32);
        let r106 = k.const_idx(3072u32);
        let r84 = k.const_idx(5u32);
        let r97 = k.const_idx(10u32);
        let r10 = k.const_idx(3u32);
        let r16 = k.group_index(0, 75000);
        let r92 = k.local_index(0, 2);
        let r2 = k.local_index(1, 32);
        let r78 = k.group_index(2, 4);
        let r27 = k.local_index(2, 8);
        let r50 = k.binary(r16, r16, BOp::Add);
        let r129 = k.binary(r50, r92, BOp::Add);
        let r104 = k.binary(r78, r10, BOp::BitShiftLeft);
        let r5 = k.binary(r104, r27, BOp::Add);
        let r22 = k.binary(r129, r130, BOp::Mod);
        let r131 = k.binary(r129, r130, BOp::Div);

        let r3 = k.storage(DType::F32, MemScope::Register, 1);
        k.store(r3, r42, r1, MemLayout::Scalar);

        let r135 = k.binary(r2, r84, BOp::BitShiftLeft);
        let r136 = k.binary(r131, r97, BOp::BitShiftLeft);

        let c6250 = k.const_idx(6250u32);
        let one = k.const_idx(1u32);
        let c8 = k.const_idx(8u32);
        let outer_loop = k.loop_(c6250);

        let r53 = k.binary(outer_loop, r10, BOp::BitShiftLeft);

        let inner_loop = k.loop_(c8);

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

        assert_eq!(k.at(outer_loop), &Op::Loop { len: one }, "outer loop should be zeroed");
        assert_eq!(k.at(inner_loop), &Op::Loop { len: one }, "inner loop should be zeroed");
    }

    /// Build the exact IR of the mnist gather (index_select) kernel captured via
    /// ZYX_DUMP_FOLD at simplify_accumulating_loop time (pre-autotune).
    ///
    /// Structure:
    ///   indices = load(indices_tensor, r26)      // loop-invariant
    ///   arange  = load(arange_tensor, loop_id)   // LOOP-DEPENDENT via Load!
    ///   mask    = f32(indices == arange)
    ///   src     = load(source_tensor, r10 + loop_id*784)
    ///   acc     = acc + f32(mask * src)
    ///
    /// The mask's loop operand is a Load indexed by the loop, so check_loop
    /// (which only peels casts) fails to recognize it → not folded.
    fn make_mnist_gather_kernel(dim: u64) -> (Kernel, OpId) {
        let mut k = Kernel::new(DeviceId::AUTO);

        let n: u64 = dim * dim;
        let r29 = k.param(DType::I32, ParamKind::Global);
        let r38 = k.param(DType::I32, ParamKind::Global);
        let r49 = k.param(DType::F32, ParamKind::Global);
        let r57 = k.param(DType::F32, ParamKind::Global);
        let r1 = k.const_idx(0u32);
        let r8 = k.const_val(0.0f32);
        let r15 = k.const_idx(dim);
        let r25 = k.const_idx(dim);
        let r7 = k.group_index(0, dim);
        let r10 = k.group_index(1, dim);

        let r3 = k.storage(DType::F32, MemScope::Register, 1);
        k.store(r3, r8, r1, MemLayout::Scalar);

        let r58 = k.binary(r7, r25, BOp::Mul);
        let r26 = k.binary(r58, r10, BOp::Add);

        let loop_id = k.loop_(r15);

        let r30 = k.load(r29, r26, MemLayout::Scalar);
        let r39 = k.load(r38, loop_id, MemLayout::Scalar);
        let r4 = k.binary(r30, r39, BOp::Eq);
        let r5 = k.cast(r4, DType::F32);
        let r44 = k.binary(loop_id, r25, BOp::Mul);
        let r46 = k.binary(r10, r44, BOp::Add);
        let r50 = k.load(r49, r46, MemLayout::Scalar);
        let r11 = k.binary(r5, r50, BOp::Mul);
        let r12 = k.cast(r11, DType::F32);
        let r17 = k.load(r3, r1, MemLayout::Scalar);
        let r18 = k.binary(r12, r17, BOp::Add);
        k.store(r3, r18, r1, MemLayout::Scalar);

        k.end_loop();

        let r13 = k.load(r3, r1, MemLayout::Scalar);
        let r54 = k.binary(r7, r25, BOp::Mul);
        let r56 = k.binary(r10, r54, BOp::Add);
        k.store(r57, r13, r56, MemLayout::Scalar);

        (k, loop_id)
    }

    /// The mnist gather (index_select) loop must NOT be folded. The mask's
    /// loop-dependent operand is `load(arange_tensor, loop_id)` — a global
    /// arange BUFFER. From the IR alone it is indistinguishable from arbitrary
    /// indices data, so the fold must not fire. Only a kernelizer-fused arange
    /// (mask operand == loop_id directly) would be foldable.
    #[test]
    fn test_mnist_gather_not_folded() {
        let (mut k, loop_id) = make_mnist_gather_kernel(3);
        k.simplify_accumulating_loop();
        assert!(matches!(k.at(loop_id), &Op::Loop { .. }), "loop must NOT be folded");

        let compiled = k.compile().unwrap();
        let source = crate::Tensor::from([[10.0f32, 20.0, 30.0], [11.0, 21.0, 31.0], [12.0, 22.0, 32.0]]);
        let indices = crate::Tensor::from([[2u32, 0, 1], [1, 2, 0], [0, 1, 2]]);
        let arange = crate::Tensor::from([0u32, 1, 2]);
        let result = compiled.forward(&[&indices, &arange, &source], vec![[3, 3]]).unwrap().pop().unwrap();
        assert_eq!(result, [[12.0f32, 20.0, 31.0], [11.0, 22.0, 30.0], [10.0, 21.0, 32.0]]);
    }

    /// Reproduce the exact scatter pre-fold IR (ZYX_DUMP_FOLD output from
    /// `scatter_1d`, /tmp/scatter_dump.txt lines 598-624).
    ///
    /// Structure (note: the loop-dependent mask operand is the INDICES load,
    /// indexed by loop_id; the arange load is loop-invariant at group_index):
    ///   acc = 0
    ///   for i in 0..3:
    ///     idx  = load(indices, i)          // LOOP-DEPENDENT
    ///     cls  = load(arange, group)       // loop-invariant
    ///     mask = i32(idx == cls)
    ///     src  = load(src, i)
    ///     acc += mask * src
    ///   out[group] = acc
    ///
    /// scatter_1d: x=zeros(10), src=[100,200,300], indices=[0,5,9]
    /// expected result = [100, 0, 0, 0, 0, 200, 0, 0, 0, 300]
    fn make_scatter_kernel(dim: u64, num_indices: u64) -> (Kernel, OpId) {
        let mut k = Kernel::new(DeviceId::AUTO);

        let r29 = k.param(DType::I32, ParamKind::Global);
        let r38 = k.param(DType::I32, ParamKind::Global);
        let r47 = k.param(DType::I32, ParamKind::Global);
        let r61 = k.param(DType::I32, ParamKind::Global);
        let r14 = k.const_idx(0u32);
        let r1 = k.const_val(0i32);
        let r10 = k.const_idx(num_indices);
        let r7 = k.group_index(0, dim);

        let r9 = k.storage(DType::I32, MemScope::Register, 1);
        k.store(r9, r1, r14, MemLayout::Scalar);

        let loop_id = k.loop_(r10);

        let r30 = k.load(r29, loop_id, MemLayout::Scalar);
        let r39 = k.load(r38, r7, MemLayout::Scalar);
        let r4 = k.binary(r30, r39, BOp::Eq);
        let r5 = k.cast(r4, DType::I32);
        let r48 = k.load(r47, loop_id, MemLayout::Scalar);
        let r8 = k.binary(r5, r48, BOp::Mul);
        let r11 = k.cast(r8, DType::I32);
        let r19 = k.load(r9, r14, MemLayout::Scalar);
        let r20 = k.binary(r11, r19, BOp::Add);
        k.store(r9, r20, r14, MemLayout::Scalar);

        k.end_loop();

        let r12 = k.load(r9, r14, MemLayout::Scalar);
        k.store(r61, r12, r7, MemLayout::Scalar);

        (k, loop_id)
    }

    /// The scatter (one-hot accumulate) loop must NOT be folded. The loop's
    /// mask operand `load(indices, loop_id)` is arbitrary indices data (not an
    /// arange, and indistinguishable from one at the IR level), so the fold
    /// must not fire — folding would produce `src[group]` for duplicate index
    /// classes. The loop stays intact and computes the correct scatter.
    #[test]
    fn test_scatter_loop_not_folded() {
        let (mut k, loop_id) = make_scatter_kernel(10, 3);
        k.simplify_accumulating_loop();
        assert!(matches!(k.at(loop_id), &Op::Loop { .. }), "loop must NOT be folded");

        let compiled = k.compile().unwrap();
        let indices = crate::Tensor::from([0i32, 5, 9]);
        let arange = crate::Tensor::from([0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let src = crate::Tensor::from([100i32, 200, 300]);
        let result = compiled.forward(&[&indices, &arange, &src], vec![[10]]).unwrap().pop().unwrap();
        assert_eq!(result, [100, 0, 0, 0, 0, 200, 0, 0, 0, 300]);
    }

    /// Reproduce the ceil-style inner accumulation loop from the
    /// `gather_3d_tensor` IR: over `i in 0..2`, accumulate
    /// `i32(((1 + g + 2*i) >> 1) > 0)` for an outer index `g` in `{0, 1}`.
    ///
    /// The closed form for the count is `g + 1`, so after folding the loop the
    /// kernel should write `out[g] = g + 1` (i.e. `[1, 2]`).
    #[test]
    fn test_ceil_mask_loop_folds() {
        let mut k = Kernel::new(DeviceId::AUTO);
        let out = k.param(DType::I32, ParamKind::Global);
        let g = k.group_index(0, 2);
        let acc = k.storage(DType::I32, MemScope::Register, 1);
        let zi = k.const_idx(0u32);
        let ziv = k.const_val(0i32);
        k.store(acc, ziv, zi, MemLayout::Scalar);

        let ilen = k.const_idx(2u32);
        let loop_id = k.loop_(ilen);

        let i2 = k.binary(loop_id, loop_id, BOp::Add);
        let body = k.binary(g, i2, BOp::Add);
        let c1 = k.const_idx(1u32);
        let b2 = k.binary(c1, body, BOp::Add);
        let shr1 = k.const_idx(1u32);
        let sh = k.binary(b2, shr1, BOp::BitShiftRight);
        let z = k.const_idx(0u32);
        let cmp = k.binary(sh, z, BOp::Cmpgt);
        let mask = k.cast(cmp, DType::I32);
        let l = k.load(acc, zi, MemLayout::Scalar);
        let sum = k.binary(mask, l, BOp::Add);
        k.store(acc, sum, zi, MemLayout::Scalar);

        k.end_loop();

        let res = k.load(acc, zi, MemLayout::Scalar);
        k.store(out, res, g, MemLayout::Scalar);

        k.simplify_accumulating_loop();

        // The after-loop load must be rewritten to the closed form (not a Load).
        assert!(matches!(k.at(res), Op::Cast { .. }), "ceil mask loop should fold");

        let compiled = k.compile().unwrap();
        let result = compiled.forward(&[], vec![[2]]).unwrap().pop().unwrap();
        assert_eq!(result, [1, 2]);
    }

    /// Reproduce the llama one-hot/embedding IR from ZYX_DEBUG=8 (llama/src,
    /// the `embedding` reduce). Per (batch, vocab) thread it counts, over a
    /// full vocab loop, `loop + v > V - 2` (via a divmod round-trip packing
    /// through `% 1` and `(a*2V + b) % (2V-1)`), accumulates the count, then
    /// compares `count - 1 == token` after the loop.
    ///
    /// With `V = 8` the closed form of the inner loop is `v + 1`, so after
    /// folding the whole kernel is just a one-hot lookup: `out[b,p,v] =
    /// (v == tokens[b,p])`. The loop must fold.
    #[test]
    fn test_llama_onehot_loop_folds() {
        let mut k = Kernel::new(DeviceId::AUTO);

        let r67 = k.param(DType::U32, ParamKind::Global);
        let r30 = k.param(DType::F16, ParamKind::Global);
        let c0 = k.const_idx(0u32);
        let c1 = k.const_idx(1u32);
        let c2 = k.const_idx(2u32);
        let c8 = k.const_idx(8u32);
        let c16 = k.const_idx(16u32);
        let c15 = k.const_idx(15u32);
        let c6 = k.const_idx(6u32);
        let ctrue = k.const_val(true);
        let c0i = k.const_val(0i64);
        let c1i = k.const_val(1i64);

        let r97 = k.group_index(0, 2);
        let r37 = k.group_index(1, 8);
        let r34 = k.group_index(2, 1);

        let r43 = k.mod_(r97, c1);
        let r3 = k.div(r97, c1);
        let r40 = k.mod_(r3, c2);
        let _r45 = k.div(r3, c2);
        let r46 = k.mad(r43, c16, c0);
        let r47 = k.mad(r40, c8, r46);
        let r48 = k.mad(r37, c1, r47);
        let r49 = k.mad(r34, c1, r48);
        let r54 = k.mad(r43, c2, c0);
        let r55 = k.mad(r40, c1, r54);
        let r56 = k.mad(r37, c0, r55);
        let r57 = k.mad(r34, c1, r56);
        let r59 = k.div(r57, c2);
        let r60 = k.mod_(r57, c2);
        let r65 = k.mad(r59, c2, c0);
        let r66 = k.mad(r60, c1, r65);
        let r72 = k.mad(r43, c8, c0);
        let r73 = k.mad(r40, c0, r72);
        let r74 = k.mad(r37, c1, r73);
        let r75 = k.mad(r34, c1, r74);

        let r81 = k.storage(DType::I64, MemScope::Register, 1);
        k.store(r81, c0i, c0, MemLayout::Scalar);
        let r84 = k.loop_(c8);

        let r88 = k.load(r81, c0, MemLayout::Scalar);
        let r95 = k.mad(r84, c8, c0);
        let r96 = k.mad(r75, c1, r95);
        let r98 = k.div(r96, c8);
        let r99 = k.mod_(r96, c8);
        let _r102 = k.div(r99, c1);
        let r103 = k.mod_(r99, c1);
        let r113 = k.mad(r98, c8, c0);
        let r114 = k.mad(_r102, c1, r113);
        let r115 = k.mad(r103, c1, r114);
        let r117 = k.div(r115, c8);
        let r118 = k.mod_(r115, c8);
        let r126 = k.mad(r117, c16, c0);
        let r127 = k.mad(r118, c1, r126);
        let r132 = k.mad(r127, c1, c0);
        let r134 = k.div(r132, c15);
        let r135 = k.mod_(r132, c15);
        let r140 = k.mad(r134, c0, c0);
        let r141 = k.mad(r135, c1, r140);
        let r148 = k.cmpgt(r141, c6);
        let r149 = k.and_(r148, ctrue);
        let r150 = k.cast(r149, DType::I64);
        let r0 = k.mul(r150, c1i);
        let r13 = k.cast(r0, DType::I64);
        let r89 = k.add(r13, r88);
        k.store(r81, r89, c0, MemLayout::Scalar);
        k.end_loop();

        let r14 = k.load(r81, c0, MemLayout::Scalar);
        let r17 = k.add(r14, c0i);
        let r20 = k.sub(r17, c1i);
        let r22 = k.cast(r20, DType::F32);
        let r24 = k.load(r67, r66, MemLayout::Scalar);
        let r25 = k.cast(r24, DType::F32);
        let r28 = k.binary(r22, r25, BOp::Eq);
        let r29 = k.cast(r28, DType::F16);
        k.store(r30, r29, r49, MemLayout::Scalar);

        k.constant_folding();
        k.algebraic_simplifications();
        k.constant_folding();
        k.algebraic_simplifications();
        k.simplify_accumulating_loop();

        let tokens = crate::Tensor::from([[1u32, 5]]);
        let tokens_host: Vec<u32> = tokens.clone().try_into().unwrap();

        let compiled = k.compile().unwrap();
        let result = compiled.forward(&[&tokens], vec![[1, 2, 8, 1]]).unwrap().pop().unwrap();
        let got: Vec<f32> = result.cast(DType::F32).try_into().unwrap();

        let mut expected = vec![0f32; 1 * 2 * 8 * 1];
        for p in 0..2 {
            expected[p * 8 + tokens_host[p] as usize] = 1.0;
        }
        assert_eq!(got, expected);
    }
}
