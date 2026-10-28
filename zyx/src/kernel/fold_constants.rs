// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Constant folding and dead code elimination for kernels.
//!
//! This module provides optimizations that evaluate constant expressions
//! and eliminate dead code from kernel IR:
//!
//! - Constant folding: evaluate expressions with constant operands
//! - Accumulator folding: simplify accumulator chains
//! - Empty loop removal: remove loops with zero iterations
//! - Dead code elimination: remove unused operations
//! - Constant movement: move constants to the beginning of kernels
//! - Power unfolding: unfold constant power operations
//!
//! These optimizations reduce instruction count and improve performance.

use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, MemScope, Op, OpId, ParamKind, RangeKind, TTOp, UOp},
};
use std::hash::BuildHasherDefault;

impl Kernel {
    /// Fold constant expressions in the kernel IR.
    ///
    /// This method evaluates expressions with constant operands,
    /// reducing instruction count and improving performance.
    ///
    /// It processes all operations in the kernel and replaces
    /// constant expressions with their computed values.
    #[allow(clippy::match_same_arms)]
    pub fn constant_folding(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("constant folding");
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match self.ops[op_id].op {
                Op::Asm { .. }
                | Op::Source { .. }
                | Op::GPU { .. }
                | Op::Spirv { .. }
                | Op::PTX { .. }
                | Op::Expand { .. }
                | Op::Permute { .. }
                | Op::Flip { .. }
                | Op::Narrow { .. }
                | Op::Pad { .. }
                | Op::Reshape { .. }
                | Op::Reduce { .. }
                | Op::TT { .. }
                | Op::Wmma { .. }
                | Op::Barrier
                | Op::Index { .. }
                | Op::Const(_)
                | Op::Param { .. }
                | Op::Storage { .. }
                | Op::Load { .. }
                | Op::Copy { .. }
                | Op::Range { .. }
                | Op::Loop { .. }
                | Op::Unroll { .. }
                | Op::EndLoop => {}
                Op::After { .. } | Op::ToDevice { .. } | Op::Contiguous { .. } | Op::Program { .. } | Op::Kernel(_) => {
                    todo!()
                }
                // Graph-only marker, erased at kernelization: never present.
                Op::Detach { .. } => {
                    todo!()
                }
                Op::Stack { ref ops } => {
                    // vectorize[devec(v,0), devec(v,1), ..., devec(v,n-1)] → v
                    if let Op::Index { vec, idx: 0 } = self.at(ops[0]) {
                        let vec = *vec;
                        if ops
                            .iter()
                            .skip(1)
                            .enumerate()
                            .all(|(i, &sub)| matches!(self.at(sub), Op::Index { vec: v, idx } if *v == vec && *idx == i + 1))
                        {
                            self.remap(op_id, vec);
                        }
                    }
                }
                Op::Store { dst, src: x, .. } => {
                    // If we store something that we just loaded, the store is pointless
                    if let Op::Load { src, .. } = *self.at(x)
                        && src == dst
                    {
                        self.remove_op(op_id);
                    }
                }
                Op::Cast { x, dtype } => {
                    if let Op::Const(cx) = self.at(x) {
                        self.ops[op_id].op = Op::Const(cx.cast(dtype));
                    }
                    // Cast of Cast: remove intermediate cast only when it is
                    // lossless (DType::is_lossless_cast). Collapsing through
                    // a narrowing cast (e.g. F32→F8→F32) would silently
                    // delete quantization. NOTE: collapsing can itself
                    // produce a same-type cast (Cast(Cast(a→B)→A) becomes
                    // Cast(a→A)), so the same-type check must run after
                    // this rewrite, not before.
                    if let Op::Cast { x: inner_x, dtype: inner_dt } = *self.at(x)
                        && self.dtype(inner_x).is_lossless_cast(inner_dt)
                    {
                        self.ops[op_id].op = Op::Cast { x: inner_x, dtype };
                    }
                    // x + c1 - c1 simplifies to x
                    // Handle: Cast(Sub(Cast(Add(x, c1)), c2)) where c1 == c2 → Cast(x)
                    if let Op::Binary { x: sub_x, y: sub_y, bop: BOp::Sub } = *self.at(x) {
                        let add_x = if let Op::Cast { x: inner_cast_x, .. } = *self.at(sub_x) {
                            inner_cast_x
                        } else {
                            sub_x
                        };
                        if let Op::Binary { x: inner_add_x, y: add_y, bop: BOp::Add } = *self.at(add_x)
                            && self.constants_equal(add_y, sub_y)
                        {
                            self.ops[op_id].op = Op::Cast { x: inner_add_x, dtype };
                        }
                    }
                    // Same-type cast is a no-op. Runs after the rewrites
                    // above: collapsing Cast(Cast(a)) can itself produce a
                    // same-type cast, and a single pass never revisits.
                    if let Op::Cast { x: final_x, .. } = self.at(op_id)
                        && self.dtype(*final_x) == dtype
                    {
                        self.remap(op_id, *final_x);
                    }
                }
                Op::Bitcast { x, dtype } => {
                    // Same-type bitcast is a no-op
                    if self.dtype(x) == dtype {
                        self.remap(op_id, x);
                    }
                    if let Op::Const(cx) = self.at(x) {
                        self.ops[op_id].op = Op::Const(cx.bitcast(dtype));
                    }
                }
                Op::Unary { x, uop } => {
                    if let Op::Const(cx) = self.at(x) {
                        self.ops[op_id].op = Op::Const(cx.unary(uop));
                    }
                }
                Op::Select { cond, if_true, if_false } => {
                    // Identical arms: the predicate is irrelevant.
                    // (Select-then-compute fusion is later work.)
                    if if_true == if_false {
                        self.remap(op_id, if_true);
                    } else if let Op::Const(Constant::Bool(p)) = self.at(cond) {
                        // A constant predicate picks its arm outright.
                        self.remap(op_id, if *p { if_true } else { if_false });
                    }
                }
                Op::Binary { x, y, bop } => match (self.at(x).clone(), self.at(y).clone()) {
                    (Op::Const(cx), Op::Const(cy)) => {
                        let folded = Constant::binary(cx, cy, bop);
                        self.ops[op_id].op = Op::Const(folded);
                    }
                    (Op::Const(cx), _) => match bop {
                        BOp::And if cx.dtype() == DType::Bool && cx.is_zero() => self.remap(op_id, x),
                        BOp::And if cx.dtype() == DType::Bool && cx.is_one() => self.remap(op_id, y),
                        BOp::Or if cx.dtype() == DType::Bool && cx.is_zero() => self.remap(op_id, y),
                        BOp::Or if cx.dtype() == DType::Bool && cx.is_one() => self.remap(op_id, x),
                        BOp::Eq if cx.dtype() == DType::Bool && cx.is_zero() => {
                            self.ops[op_id].op = Op::Unary { x: y, uop: UOp::Not }
                        }
                        BOp::Eq if cx.dtype() == DType::Bool && cx.is_one() => self.remap(op_id, y),
                        BOp::NotEq if cx.dtype() == DType::Bool && cx.is_zero() => self.remap(op_id, y),
                        BOp::NotEq if cx.dtype() == DType::Bool && cx.is_one() => {
                            self.ops[op_id].op = Op::Unary { x: y, uop: UOp::Not }
                        }
                        BOp::Add if cx.is_zero() => self.remap(op_id, y),
                        BOp::Sub if cx.is_zero() => self.ops[op_id].op = Op::Unary { x: y, uop: UOp::Neg },
                        BOp::Mul | BOp::Div if cx.is_zero() => self.ops[op_id].op = Op::Const(cx),
                        BOp::Mul if cx.is_one() => self.remap(op_id, y),
                        BOp::Mul if cx.is_two() => self.ops[op_id].op = Op::Binary { x: y, y, bop: BOp::Add },
                        BOp::Mul if cx.is_power_of_two() && cx.dtype() == IDX_T => {
                            let c = self.insert_before(op_id, Op::Const(cx.unary(UOp::Log2)));
                            self.ops[op_id].op = Op::Binary { x: y, y: c, bop: BOp::BitShiftLeft };
                        }
                        BOp::Div if cx.is_zero() => self.remap(op_id, x),
                        BOp::Div if cx.is_one() && cx.dtype().is_float() => {
                            self.ops[op_id].op = Op::Unary { x: y, uop: UOp::Reciprocal }
                        }
                        BOp::Pow if cx.is_one() => self.ops[op_id].op = Op::Const(cx),
                        BOp::Max if cx.is_minimum() => self.remap(op_id, y),
                        BOp::Max if cx.is_max() => self.remap(op_id, x),
                        BOp::BitAnd if cx.is_zero() => self.remap(op_id, x),
                        BOp::BitAnd if cx.is_max() && cx.dtype().is_uint() => self.remap(op_id, y),
                        BOp::BitOr if cx.is_zero() => self.remap(op_id, y),
                        BOp::BitOr if cx.is_max() => self.remap(op_id, x),
                        BOp::BitXor if cx.is_zero() => self.remap(op_id, y),
                        BOp::BitShiftLeft | BOp::BitShiftRight if cx.is_zero() => self.remap(op_id, x),
                        _ => {}
                    },
                    (_, Op::Const(cy)) => match bop {
                        BOp::And if cy.dtype() == DType::Bool && cy.is_zero() => self.remap(op_id, y),
                        BOp::And if cy.dtype() == DType::Bool && cy.is_one() => self.remap(op_id, x),
                        BOp::Or if cy.dtype() == DType::Bool && cy.is_zero() => self.remap(op_id, x),
                        BOp::Or if cy.dtype() == DType::Bool && cy.is_one() => self.remap(op_id, y),
                        BOp::Eq if cy.dtype() == DType::Bool && cy.is_zero() => {
                            self.ops[op_id].op = Op::Unary { x, uop: UOp::Not }
                        }
                        BOp::Eq if cy.dtype() == DType::Bool && cy.is_one() => self.remap(op_id, x),
                        BOp::NotEq if cy.dtype() == DType::Bool && cy.is_zero() => self.remap(op_id, x),
                        BOp::NotEq if cy.dtype() == DType::Bool && cy.is_one() => {
                            self.ops[op_id].op = Op::Unary { x, uop: UOp::Not }
                        }
                        BOp::Add | BOp::Sub if cy.is_zero() => self.remap(op_id, x),
                        BOp::Mul if cy.is_zero() => self.ops[op_id].op = Op::Const(cy),
                        BOp::Mul if cy.is_one() => self.remap(op_id, x),
                        BOp::Mul if cy.is_two() => self.ops[op_id].op = Op::Binary { x, y: x, bop: BOp::Add },
                        BOp::Mul if cy.is_power_of_two() && cy.dtype() == IDX_T => {
                            let c = self.insert_before(op_id, Op::Const(cy.unary(UOp::Log2)));
                            self.ops[op_id].op = Op::Binary { x, y: c, bop: BOp::BitShiftLeft };
                        }
                        BOp::Div if cy.is_zero() && !cy.dtype().is_float() => panic!("Division by constant zero"),
                        BOp::Div if cy.is_one() => self.remap(op_id, x),
                        BOp::Div if cy.is_power_of_two() && cy.dtype() == IDX_T => {
                            let y = self.insert_before(op_id, Op::Const(cy.unary(UOp::Log2)));
                            self.ops[op_id].op = Op::Binary { x, y, bop: BOp::BitShiftRight };
                        }
                        BOp::Mod if cy.is_one() && !cy.dtype().is_float() => {
                            self.ops[op_id].op = Op::Const(cy.dtype().zero_constant())
                        }
                        BOp::Mod if cy.is_zero() && !cy.dtype().is_float() => panic!("Modulo by constant zero"),
                        BOp::Pow if cy.is_zero() => self.ops[op_id].op = Op::Const(cy.dtype().one_constant()),
                        BOp::Pow if cy.is_one() => self.remap(op_id, x),
                        BOp::Pow if cy.is_two() => self.ops[op_id].op = Op::Binary { x, y: x, bop: BOp::Mul },
                        BOp::Max if cy.is_minimum() => self.remap(op_id, x),
                        BOp::Max if cy.is_max() => self.remap(op_id, y),
                        BOp::BitAnd if cy.is_zero() => self.remap(op_id, y),
                        BOp::BitAnd if cy.is_max() && cy.dtype().is_uint() => self.remap(op_id, x),
                        BOp::BitOr if cy.is_zero() => self.remap(op_id, x),
                        BOp::BitOr if cy.is_max() => self.remap(op_id, y),
                        BOp::BitXor if cy.is_zero() => self.remap(op_id, x),
                        BOp::BitShiftLeft if cy.is_zero() => self.remap(op_id, x),
                        BOp::BitShiftRight if cy.is_zero() => self.remap(op_id, x),
                        // Consecutive modulo by constant: (x % a) % b folds to x % b
                        // only when b | a, and to x % a only when a | b. When neither
                        // divides the other the result cannot be reduced to a single mod.
                        BOp::Mod if cy.dtype() == IDX_T => {
                            if let Op::Binary { bop, x: xi, y: yi } = self.ops[x].op
                                && bop == BOp::Mod
                                && let Op::Const(ciy) = self.ops[yi].op
                                && let (Some(a), Some(b)) = (ciy.as_dim(), cy.as_dim())
                                && a != 0
                                && b != 0
                            {
                                if a % b == 0 {
                                    self.ops[op_id].op = Op::Binary { x: xi, y, bop: BOp::Mod };
                                } else if b % a == 0 {
                                    self.ops[op_id].op = Op::Binary { x: xi, y: yi, bop: BOp::Mod };
                                }
                            }
                        }
                        _ => {}
                    },
                    (x_op, y_op) if x_op == y_op => match bop {
                        BOp::Div => {
                            let dtype = self.dtype(x);
                            self.ops[op_id].op = Op::Const(dtype.one_constant());
                        }
                        BOp::Sub => {
                            let dtype = self.dtype(x);
                            self.ops[op_id].op = Op::Const(dtype.zero_constant());
                        }
                        BOp::Mod if !self.dtype(x).is_float() => {
                            let dtype = self.dtype(x);
                            self.ops[op_id].op = Op::Const(dtype.zero_constant());
                        }
                        BOp::BitXor => {
                            let dtype = self.dtype(x);
                            self.ops[op_id].op = Op::Const(dtype.zero_constant());
                        }
                        BOp::BitAnd | BOp::BitOr | BOp::Max => self.remap(op_id, x),
                        // Boolean idempotence: x && x -> x, x || x -> x. Mask
                        // AND-chains (pad/narrow validity terms) can reduce both
                        // operands to the same comparison; without this rule the
                        // leftover And breaks fold_loops' condition-chain patterns.
                        BOp::And | BOp::Or if self.dtype(x) == DType::Bool => self.remap(op_id, x),
                        BOp::Cmpgt | BOp::Cmplt => self.ops[op_id].op = Op::Const(DType::Bool.zero_constant()),
                        BOp::Eq if !self.dtype(x).is_float() => self.ops[op_id].op = Op::Const(DType::Bool.one_constant()),
                        BOp::NotEq if !self.dtype(x).is_float() => self.ops[op_id].op = Op::Const(DType::Bool.zero_constant()),
                        _ => {}
                    },
                    _ => {}
                },
            }
            op_id = next;
        }

        self.verify();
    }

    /// Fold accumulators in the kernel.
    ///
    /// This method eliminates accumulators that are not stored into
    /// in loops, reducing instruction count and improving performance.
    pub(crate) fn fold_accs(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("fold_accs");
        // We have to do constant folding before folding accs to guarantee indices are constants
        self.constant_folding();
        // Check if an accumulator storage exists without a scope that stores into it.
        // Storages used inside `Loop`/`If` must not be
        // folded — they are conditional/loop-carried, not dead accumulators.
        let mut accumulators = Map::default();
        let mut scope_level = 0u32;
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Storage { scope: MemScope::Register, .. } => {
                    accumulators.insert(op_id, scope_level);
                }
                // Stores address the storage directly (post-linearize)
                // or a bare Param (pre-linearize whole-view write):
                // resolve to the buffer before the accumulator check.
                Op::Store { dst, .. } => {
                    if let Some(lvl) = accumulators.get(&dst)
                        && scope_level > *lvl
                    {
                        accumulators.remove(&dst);
                    }
                }
                Op::Loop { .. } => {
                    scope_level += 1;
                }
                Op::EndLoop => {
                    scope_level -= 1;
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        //println!("accumulators: {accumulators:?}");
        for (acc, _) in accumulators {
            self.fold_acc(acc);
        }
        // Cleanup belongs to DCE: the folds above orphan index expressions (and leave
        // other dead ops), which is verify-dangling interim state. Sweep
        // once here; DCE's trailing verify is the guard.
        self.dead_code_elimination();
    }

    /// Fold a single accumulator operation.
    ///
    /// This method simplifies accumulator operations by folding
    /// constant values and eliminating redundant computations.
    pub(crate) fn fold_acc(&mut self, storage_id: OpId) {
        //println!("Folding acc {storage_id}");
        let Op::Storage { ref len, .. } = self.ops[storage_id].op else {
            unreachable!()
        };
        let mut latest_stores = vec![OpId::NULL; *len as usize];

        // All-or-nothing: bail if any access over this storage can't be
        // folded below (non-const, non-unroll index, non-scalar shape,
        // copies, or any other use) — otherwise the survivors dangle.
        // Unroll indices are compile-time-known lane sets: an access under
        // one touches every lane at once, so it folds exactly like a
        // broadcast const index, provided its width covers the storage.
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Store { dst, dst_idx, .. } => {
                    if dst == storage_id {
                        // Const indices fold per lane; an unroll index
                        // covers every lane at once, provided its width
                        // covers the storage. Scalar accesses only: the
                        // index must resolve to a plain scalar (a lane
                        // tree's shape is non-empty).
                        let lane_index = matches!(self.at(dst_idx), Op::Const(_));
                        if !self.shape(dst_idx).is_empty() || !lane_index {
                            return;
                        }
                    }
                }
                Op::Load { src, src_idx } => {
                    if src == storage_id {
                        let lane_index = matches!(self.at(src_idx), Op::Const(_));
                        if !self.shape(src_idx).is_empty() || !lane_index {
                            return;
                        }
                    }
                }
                Op::Copy { src, dst, .. } => {
                    if src == storage_id || dst == storage_id {
                        return;
                    }
                }
                ref op => {
                    if op.parameters().any(|p| p == storage_id) {
                        return;
                    }
                }
            }
            op_id = self.next_op(op_id);
        }

        self.remove_op(storage_id);

        let mut remaps = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match *self.at(op_id) {
                Op::Store { dst, dst_idx, src: x } => {
                    if dst != storage_id || !self.shape(dst_idx).is_empty() {
                        self.ops[op_id].op.remap_params(&remaps);
                        op_id = next;
                        continue;
                    }
                    self.remove_op(op_id);
                    if self.ops.contains_id(x) {
                        let Op::Const(index_c) = self.ops[dst_idx].op else {
                            unreachable!("fold_constants pre-check admits only const store indices");
                        };
                        let index = index_c.as_dim().expect("store index must be a non-negative integer constant") as usize;
                        latest_stores[index] = x;
                    }
                    op_id = next;
                    continue;
                }
                Op::Load { src, src_idx } => {
                    if src == storage_id
                        && self.shape(src_idx).is_empty()
                        && let Op::Const(index_c) = self.ops[src_idx].op
                    {
                        self.remove_op(op_id);
                        let index = index_c.as_dim().expect("load index must be a non-negative integer constant") as usize;
                        remaps.insert(op_id, latest_stores[index]);
                        op_id = next;
                        continue;
                    }
                }
                _ => {}
            }
            self.ops[op_id].op.remap_params(&remaps);
            op_id = next;
        }
    }

    /// Delete empty loops from the kernel.
    ///
    /// This method removes loops that don't contain any stores,
    /// reducing instruction count and improving performance.
    /// Delete useless loops and ranges: length-1 `Loop`s and `Range`s
    /// iterate exactly once, so they become `Const(0)` (a length-1 loop
    /// also drops its paired `EndLoop`); loops whose bodies never escape
    /// (no store to an outside buffer) are removed entirely.
    pub(crate) fn delete_useless_loops(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("delete_empty_loops");

        // Length-1 iteration structure iterates exactly once: replace it
        // with the constant index 0.
        let mut op_id = self.head;
        let mut depth = 0;
        let mut deleted = Vec::new();
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match self.ops[op_id].op {
                Op::Loop { len: len_id } => {
                    depth += 1;
                    // Boolean lengths are conditionals (`if`), not trip
                    // counts: DIM-resolution of a per-thread value
                    // comparison is meaningless (it can fold to 1 even when
                    // the condition is false for most threads), so the
                    // length-1 rule never applies to them.
                    if self.dtype(len_id) == crate::DType::Bool {
                        op_id = next;
                        continue;
                    }
                    if self.resolve_const(len_id).and_then(crate::dtype::Constant::as_dim) == Some(1) {
                        self.ops[op_id].op = Op::Const(Constant::idx(0));
                        deleted.push(depth);
                    }
                }
                Op::EndLoop => {
                    if deleted.pop_if(|x| *x == depth).is_some() {
                        self.remove_op(op_id);
                    }
                    depth -= 1;
                }
                Op::Range { kind, .. } => match kind {
                    RangeKind::Group(len) => {
                        if self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) == Some(1) {
                            self.ops[op_id].op = Op::Const(Constant::idx(0));
                        }
                    }
                    RangeKind::Local(len) => {
                        if len == 1 {
                            self.ops[op_id].op = Op::Const(Constant::idx(0));
                        }
                    }
                    // A warp is a view over a local range — never a removable len-1 index.
                    RangeKind::Warp(_) => {}
                },
                _ => {}
            }
            op_id = next;
        }

        let mut dead = Set::default();
        let mut store_targets_stack: Vec<Set<OpId>> = Vec::new();
        store_targets_stack.push(Set::default());
        let mut ops_stack: Vec<Set<OpId>> = Vec::new();
        ops_stack.push(Set::default());
        let mut delete_stack: Vec<bool> = Vec::new();
        delete_stack.push(false);

        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.at(op_id) {
                Op::Loop { .. } => {
                    ops_stack.push(Set::default());
                    store_targets_stack.push(Set::default());
                    delete_stack.push(true);
                    for slice in &mut ops_stack {
                        slice.insert(op_id);
                    }
                }
                &Op::Param { kind, .. } => {
                    if kind == ParamKind::GlobalMut {
                        store_targets_stack.last_mut().unwrap().insert(op_id);
                    }
                }
                Op::Storage { .. } => {
                    store_targets_stack.last_mut().unwrap().insert(op_id);
                }
                Op::Store { dst, .. } => {
                    // Post-linearize dst names the buffer directly:
                    // the escape check uses it as-is.
                    let buf = *dst;
                    for (i, targets) in store_targets_stack.iter().enumerate().take(store_targets_stack.len() - 1) {
                        if targets.contains(&buf) {
                            for delete_flag in delete_stack.iter_mut().skip(i + 1) {
                                *delete_flag = false;
                            }
                            break;
                        }
                    }
                    for slice in &mut ops_stack {
                        slice.insert(op_id);
                    }
                }
                Op::Copy { dst, .. } => {
                    // A copy to an outside buffer is an escaping effect like
                    // a store: the dst base names the buffer directly.
                    let buf = *dst;
                    for (i, targets) in store_targets_stack.iter().enumerate().take(store_targets_stack.len() - 1) {
                        if targets.contains(&buf) {
                            for delete_flag in delete_stack.iter_mut().skip(i + 1) {
                                *delete_flag = false;
                            }
                            break;
                        }
                    }
                    for slice in &mut ops_stack {
                        slice.insert(op_id);
                    }
                }
                Op::EndLoop => {
                    for slice in &mut ops_stack {
                        slice.insert(op_id);
                    }
                    store_targets_stack.pop();
                    if let Some(delete_slice) = delete_stack.pop() {
                        if delete_slice {
                            dead.extend(ops_stack.pop().unwrap());
                        } else {
                            ops_stack.pop();
                        }
                    } else {
                        ops_stack.pop();
                    }
                }
                _ => {
                    for slice in &mut ops_stack {
                        slice.insert(op_id);
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
        for op_id in dead {
            self.remove_op(op_id);
        }

        self.verify();
    }

    /// Eliminate dead code from the kernel.
    ///
    /// This method removes unused operations from the kernel,
    /// including operations that are not referenced by any output.
    ///
    /// It uses data flow analysis to determine which operations
    /// are dead and can be safely removed.
    pub fn dead_code_elimination(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("dead_code_elimination");

        let mut params = Vec::new();
        let mut visited = Set::default();
        // We go backward from Stores and gather all needed ops, but we can't remove Loop, Param and Storage ops
        for (op_id, op) in self.iter_unordered() {
            if matches!(
                op,
                Op::Store { .. }
                    | Op::Copy { .. }
                    | Op::TT(TTOp::Copy { .. })
                    | Op::TT(TTOp::ReserveBack { .. })
                    | Op::TT(TTOp::PushBack { .. })
                    | Op::TT(TTOp::WaitFront { .. })
                    | Op::TT(TTOp::PopFront { .. })
                    | Op::TT(TTOp::MathLock)
                    | Op::TT(TTOp::MathUnlock)
                    | Op::TT(TTOp::PackLock)
                    | Op::TT(TTOp::PackUnlock)
                    | Op::TT(TTOp::NocReadBarrier)
                    | Op::TT(TTOp::NocWriteBarrier)
                    | Op::TT(TTOp::ReduceUninit)
                    | Op::TT(TTOp::EndReader)
                    | Op::TT(TTOp::EndCompute)
                    | Op::TT(TTOp::ProgramDesc(_))
                    | Op::TT(TTOp::TensixGrid(_))
                    | Op::TT(TTOp::LLK { .. })
                    | Op::TT(TTOp::ReduceTile { .. })
                    | Op::TT(TTOp::FillTile { .. })
                    | Op::TT(TTOp::LLKBcast { .. })
                    | Op::TT(TTOp::MatmulTiles { .. })
                    | Op::TT(TTOp::SigmoidTile { .. })
                    | Op::TT(TTOp::SiluTile { .. })
                    | Op::Param { .. }
                    | Op::Storage { .. }
                    | Op::Wmma { .. }
                    | Op::Asm { .. }
                    | Op::Barrier
                    | Op::Loop { .. }
                    | Op::EndLoop
            ) {
                params.push(op_id);
                continue;
            }
            // A load from Circular storage is a FIFO pop: its value may
            // be dead (e.g. a drain pop that only frees the producer
            // slot) but the pop itself is a side effect, so the load is
            // a root like a store. Removing it would silently unbalance
            // CB push/pop traffic. The scope chases the load's base.
            if let Op::Load { src, .. } = op {
                if matches!(self.at(*src), Op::Storage { scope: MemScope::Circular, .. }) {
                    params.push(op_id);
                }
            }
        }
        while let Some(op_id) = params.pop() {
            if visited.insert(op_id) {
                params.extend(self.at(op_id).parameters());
            }
        }
        //self.ops.retain(|op_id| visited.contains(op_id));
        #[allow(clippy::needless_collect)] // REASON - borrowck
        for op_id in self.ops.ids().collect::<Vec<_>>() {
            if !visited.contains(&op_id) {
                self.remove_op(op_id);
            }
        }

        self.verify();
    }

    /// Eliminate common subexpressions from the kernel.
    ///
    /// This method identifies and eliminates duplicate computations
    /// in the kernel by tracking which operations are needed by
    /// stores and removing operations that are not referenced.
    pub fn common_subexpression_elimination(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("common_subexpression_elimination");
        let mut stack: Vec<Map<Op, OpId>> = Vec::with_capacity(10);
        stack.push(Map::with_capacity_and_hasher(20, BuildHasherDefault::default()));

        let mut stored_stack: Vec<Set<OpId>> = Vec::with_capacity(10);
        stored_stack.push(Set::with_capacity_and_hasher(10, BuildHasherDefault::default()));

        let mut remaps = Map::with_capacity_and_hasher(10, BuildHasherDefault::default());
        let mut op_id = self.head;
        while !op_id.is_null() {
            match &mut self.ops[op_id].op {
                Op::Barrier | Op::Storage { .. } => {} // skip barrier/storage ops, these can not be deduplicated
                Op::Loop { .. } => {
                    stack.push(Map::with_capacity_and_hasher(20, BuildHasherDefault::default()));
                    stored_stack.push(Set::with_capacity_and_hasher(10, BuildHasherDefault::default()));
                }
                Op::EndLoop => {
                    stack.pop();
                    stored_stack.pop();
                }
                &mut Op::Store { .. }
                | &mut Op::Copy { .. }
                | &mut Op::TT(TTOp::Copy { .. })
                | &mut Op::TT(TTOp::ReserveBack { .. })
                | &mut Op::TT(TTOp::PushBack { .. })
                | &mut Op::TT(TTOp::WaitFront { .. })
                | &mut Op::TT(TTOp::PopFront { .. })
                | &mut Op::TT(TTOp::LLK { .. })
                | &mut Op::TT(TTOp::ReduceTile { .. })
                | &mut Op::TT(TTOp::FillTile { .. })
                | &mut Op::TT(TTOp::LLKBcast { .. })
                | &mut Op::TT(TTOp::MatmulTiles { .. })
                | &mut Op::TT(TTOp::SigmoidTile { .. })
                | &mut Op::TT(TTOp::SiluTile { .. })
                | &mut Op::TT(TTOp::MathLock)
                | &mut Op::TT(TTOp::MathUnlock)
                | &mut Op::TT(TTOp::PackLock)
                | &mut Op::TT(TTOp::PackUnlock)
                | &mut Op::TT(TTOp::NocReadBarrier)
                | &mut Op::TT(TTOp::NocWriteBarrier)
                | &mut Op::TT(TTOp::ReduceUninit)
                | &mut Op::TT(TTOp::EndReader)
                | &mut Op::TT(TTOp::EndCompute)
                | &mut Op::TT(TTOp::ProgramDesc(_))
                | &mut Op::TT(TTOp::TensixGrid(_)) => {
                    // Effect ops never participate in CSE: remap operands,
                    // never dedup (two identical effects are two traffic
                    // events). Only buffer references (Storage/Param base
                    // operands) invalidate load merging; values, consts
                    // and indices are not addresses.
                    let op = &mut self.ops[op_id].op;
                    for param in op.parameters_mut() {
                        if let Some(&new_id) = remaps.get(param) {
                            *param = new_id;
                        }
                    }
                    let params: Vec<OpId> = op.parameters().collect();
                    for &x in params.iter() {
                        if matches!(self.at(x), Op::Param { .. } | Op::Storage { .. }) {
                            stored_stack.last_mut().unwrap().insert(x);
                        }
                    }
                }
                op => {
                    let mut remove_op = false;

                    // For Load ops: remap the base before the
                    // stored-location check (bases dedup and a stale base
                    // id would miss its invalidating store and CSE across
                    // it), then check a write to the same address. The
                    // base buffer counts (writers spell it directly now).
                    // Two reads of one buffer uninterrupted dedup;
                    // read-write-read blocks the second read.
                    let can_cse = if let Op::Load { src, .. } = op {
                        if let Some(&new_id) = remaps.get(src) {
                            *src = new_id;
                        }
                        if stored_stack.iter().rev().any(|x| x.contains(src)) {
                            // The buffer was written since earlier loads were
                            // pooled. Dropping markers alone only blocks the
                            // FIRST read-after-write; later reads would pool
                            // with the pre-write entries again and read
                            // stale values across the store. Purge the
                            // buffer's pre-write pool entries instead, so
                            // subsequent reads-after-write pool only with
                            // other post-write reads.
                            for level in stack.iter_mut() {
                                level.retain(|o, _| !matches!(o, Op::Load { src: base, .. } if *base == *src));
                            }
                            for x in stored_stack.iter_mut() {
                                x.remove(src);
                            }
                            false
                        } else {
                            true
                        }
                    } else {
                        true
                    };

                    // Remap operands BEFORE dedup check so the lookup uses
                    // post-remap OpIds, matching what's already in the map.
                    for param in op.parameters_mut() {
                        if let Some(&new_id) = remaps.get(param) {
                            *param = new_id;
                        }
                    }

                    if can_cse {
                        for loop_level in &stack {
                            if let Some(&old_op_id) = loop_level.get(op) {
                                remaps.insert(op_id, old_op_id);
                                remove_op = true;
                                break;
                            }
                        }
                    }

                    if !remove_op {
                        stack.last_mut().unwrap().insert(op.clone(), op_id);
                    }
                }
            }
            op_id = self.next_op(op_id);
        }

        // CSE leaves the deduplicated originals dead in place. A dead op
        // can sit before the params (a deduped shape const, say), which
        // the positional ABI check in `verify` flags even though the op
        // is about to vanish. Drop dead ops first so the check sees the
        // list the caller will actually run.
        self.dead_code_elimination();

        self.verify();
    }

    /// Move constant operations to the beginning of the kernel.
    ///
    /// This method reorders the kernel so that constant operations
    /// appear before variable operations, which can improve
    /// instruction scheduling and performance.
    pub(crate) fn move_constants_to_beginning(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("move_constants_to_beginning");
        // Leading metadata region: params, storages, consts, and the
        // shape-descriptor ops params depend on (`Op::parameters`).
        // Dependency-derived: a shape cone must stay with its params, so
        // ranges are hoisted only past it — never above a param.
        let mut shape_deps: Set<OpId> = Set::default();
        for (_, op) in self.iter_unordered() {
            if let Op::Param { shape, .. } = op
                && !shape.is_null()
            {
                let mut stack = vec![*shape];
                for _ in 0..250_000 {
                    let Some(d) = stack.pop() else { break };
                    if !shape_deps.insert(d) {
                        continue;
                    }
                    for q in self.ops[d].op.parameters() {
                        if !q.is_null() {
                            stack.push(q);
                        }
                    }
                }
            }
        }
        let leading = |k: &Self, id: OpId| {
            matches!(k.at(id), Op::Storage { .. } | Op::Param { .. } | Op::Const(_)) || shape_deps.contains(&id)
        };
        let mut start = self.head;
        while !start.is_null() && leading(self, start) {
            start = self.next_op(start);
        }

        // Move Op::Const ops right after the leading param/storage ops
        let mut op_id = start;
        let mut start = self.prev_op(start);
        // start == NULL means the consts already sit at the very beginning.
        while !op_id.is_null() && !start.is_null() {
            let next = self.next_op(op_id);
            if let Op::Const(_) = self.at(op_id) {
                self.move_op_after(op_id, start);
                start = op_id;
            }
            op_id = next;
        }

        // Find position after last Op::Const (skip past all params, storages and consts)
        let mut start = self.head;
        while !start.is_null() && leading(self, start) {
            start = self.next_op(start);
        }

        // Move Op::Index ops right after Op::Const ops
        let mut op_id = start;
        let mut start = self.prev_op(start);
        while !op_id.is_null() && !start.is_null() {
            let next = self.next_op(op_id);
            if let Op::Range { kind, .. } = self.at(op_id) {
                // Only hoist ranges whose length is already at the front
                // (const or param). A computed length (binary/unary/...) must
                // stay in topological order behind its definition.
                let hoistable = match *kind {
                    RangeKind::Group(len) => self.resolve_const(len).is_some() || matches!(self.at(len), Op::Param { .. }),
                    RangeKind::Warp(_) => true,
                    RangeKind::Local(_) => true,
                };
                if hoistable {
                    self.move_op_after(op_id, start);
                    start = op_id;
                }
            }
            op_id = next;
        }

        #[cfg(debug_assertions)]
        self.verify();
    }

    fn constants_equal(&self, a: OpId, b: OpId) -> bool {
        let a = self.at(a);
        let b = self.at(b);
        match (a, b) {
            (Op::Const(ca), Op::Const(cb)) => {
                let a_val: Option<i64> = match ca {
                    Constant::U32(x) => Some(*x as i64),
                    Constant::I32(x) => Some(*x as i64),
                    Constant::U64(x) | Constant::I64(x) => Some(i64::from_le_bytes(*x)),
                    _ => None,
                };
                let b_val: Option<i64> = match cb {
                    Constant::U32(x) => Some(*x as i64),
                    Constant::I32(x) => Some(*x as i64),
                    Constant::U64(x) | Constant::I64(x) => Some(i64::from_le_bytes(*x)),
                    _ => None,
                };
                match (a_val, b_val) {
                    (Some(av), Some(bv)) => av == bv,
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// Unfold power operations.
    ///
    /// `x^y` with an integer-valued constant exponent unfolds to
    /// multiplies (host-side binary exponentiation; negative exponents
    /// via reciprocal for floats). Integer-dtype powers lower to a
    /// square-and-multiply bit loop over the exponent bits in unsigned
    /// arithmetic (wrapping is defined there; signed overflow is not),
    /// with tinygrad `xpow_int` negative-exponent semantics (0, except
    /// base ±1). Only genuinely dynamic float powers keep
    /// `x^y = exp2(y * log2(x))`: integer `log2`/`exp2` match no NVRTC
    /// overload and PTX has no `pow`, so every int `Pow` must lower here.
    pub(crate) fn unfold_pows(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let &Op::Binary { x, y, bop } = self.at(op_id)
                && bop == BOp::Pow
            {
                let dt = self.dtype(x);
                let edt = self.dtype(y);
                // Float value of a constant exponent (int consts fold
                // through `as_float` too). Range-checked before `as i64`.
                let exp_f: Option<f64> = self.resolve_const(y).and_then(|c| c.as_float());
                let exp_int: Option<i64> = exp_f.and_then(|f| {
                    if f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                        Some(f as i64)
                    } else {
                        None
                    }
                });
                if exp_int == Some(0) && dt.is_int() {
                    // x^0 -> 1 via x*0+1: exact in int arithmetic (no
                    // NaN domain), and the mul/add broadcast the vector
                    // shape. Float x^0 stays on the legacy path below.
                    let zero = self.insert_before(op_id, Op::Const(Constant::new(0i32)));
                    let zero = self.insert_before(op_id, Op::Cast { x: zero, dtype: dt });
                    let one = self.insert_before(op_id, Op::Const(Constant::new(1i32)));
                    let one = self.insert_before(op_id, Op::Cast { x: one, dtype: dt });
                    let z = self.insert_before(op_id, Op::Binary { x, y: zero, bop: BOp::Mul });
                    self.ops[op_id].op = Op::Binary { x: z, y: one, bop: BOp::Add };
                } else if dt.is_float() {
                    match exp_int {
                        Some(n) if n != 0 => {
                            // Binary exponentiation over the host-side
                            // magnitude: exact, no log2/exp2 rounding.
                            let mag = n.unsigned_abs();
                            let mut acc: Option<OpId> = None;
                            let mut sq = x;
                            let mut m = mag;
                            while m > 0 {
                                if m & 1 == 1 {
                                    acc = Some(match acc {
                                        Some(a) => self.insert_before(
                                            op_id,
                                            Op::Binary { x: a, y: sq, bop: BOp::Mul },
                                        ),
                                        None => sq,
                                    });
                                }
                                m >>= 1;
                                if m > 0 {
                                    sq = self.insert_before(
                                        op_id,
                                        Op::Binary { x: sq, y: sq, bop: BOp::Mul },
                                    );
                                }
                            }
                            let p = acc.expect("nonzero pow magnitude leaves a nonempty chain");
                            if n < 0 {
                                let r = self.insert_before(
                                    op_id,
                                    Op::Unary { x: p, uop: UOp::Reciprocal },
                                );
                                self.ops[op_id].op = Op::Unary { x: r, uop: UOp::Reciprocal };
                            } else {
                                self.remap(op_id, p);
                            }
                        }
                        _ => {
                            // Dynamic (or zero) float exponent: keep
                            // `x^y = exp2(y * log2(x))`.
                            let x = self.insert_before(op_id, Op::Unary { x, uop: UOp::Log2 });
                            let x =
                                self.insert_before(op_id, Op::Binary { x, y, bop: BOp::Mul });
                            self.ops[op_id].op = Op::Unary { x, uop: UOp::Exp2 };
                        }
                    }
                } else if dt.is_int() && edt.is_int() {
                    // Unsigned domain for defined wrapping; signed
                    // dtypes bitcast there and back (equal widths).
                    let udt = match dt {
                        DType::I8 => DType::U8,
                        DType::I16 => DType::U16,
                        DType::I32 => DType::U32,
                        DType::I64 => DType::U64,
                        DType::U8 => DType::U8,
                        DType::U16 => DType::U16,
                        DType::U32 => DType::U32,
                        DType::U64 => DType::U64,
                        _ => unreachable!("int pow on non-int dtype {dt:?}"),
                    };
                    if let Some(n) = exp_int {
                        if n > 0 {
                            let mut b = x;
                            if udt != dt {
                                b = self.insert_before(op_id, Op::Bitcast { x, dtype: udt });
                            }
                            let mut acc: Option<OpId> = None;
                            let mut sq = b;
                            let mut m = n as u64;
                            while m > 0 {
                                if m & 1 == 1 {
                                    acc = Some(match acc {
                                        Some(a) => self.insert_before(
                                            op_id,
                                            Op::Binary { x: a, y: sq, bop: BOp::Mul },
                                        ),
                                        None => sq,
                                    });
                                }
                                m >>= 1;
                                if m > 0 {
                                    sq = self.insert_before(
                                        op_id,
                                        Op::Binary { x: sq, y: sq, bop: BOp::Mul },
                                    );
                                }
                            }
                            let mut p = acc.expect("nonzero pow magnitude leaves a nonempty chain");
                            if udt != dt {
                                p = self.insert_before(op_id, Op::Bitcast { x: p, dtype: dt });
                            }
                            self.remap(op_id, p);
                        } else {
                            // Negative int exponent (torch semantics):
                            // 0, except base 1 (1) and base -1 (±1 by
                            // exponent parity, known at compile time).
                            // The outer select carries `x` as its true
                            // arm: where the condition holds the base is
                            // 1, and `Select` takes its shape from there.
                            let c0 = self.insert_before(op_id, Op::Const(Constant::new(0i32)));
                            let c0 = self.insert_before(op_id, Op::Cast { x: c0, dtype: dt });
                            let c1 = self.insert_before(op_id, Op::Const(Constant::new(1i32)));
                            let c1 = self.insert_before(op_id, Op::Cast { x: c1, dtype: dt });
                            let cm1 = self.insert_before(op_id, Op::Const(Constant::new(-1i32)));
                            let cm1 = self.insert_before(op_id, Op::Cast { x: cm1, dtype: dt });
                            let is_one = self.insert_before(
                                op_id,
                                Op::Binary { x, y: c1, bop: BOp::Eq },
                            );
                            let is_negone = self.insert_before(
                                op_id,
                                Op::Binary { x, y: cm1, bop: BOp::Eq },
                            );
                            let picked = if n & 1 == 1 { cm1 } else { c1 };
                            let inner = self.insert_before(
                                op_id,
                                Op::Select { cond: is_negone, if_true: picked, if_false: c0 },
                            );
                            let result = self.insert_before(
                                op_id,
                                Op::Select { cond: is_one, if_true: x, if_false: inner },
                            );
                            self.remap(op_id, result);
                        }
                    } else {
                        // Dynamic int exponent: square-and-multiply over
                        // the exponent bits (tinygrad `xpow_int`). The
                        // first select's true arm is already vector
                        // (`ret` starts scalar but `prod` is vector from
                        // the first set bit); the neg gate below keeps
                        // the vector result as its true arm.
                        let eone = self.insert_before(op_id, Op::Const(Constant::new(1i32)));
                        let eone = if edt == DType::I32 {
                            eone
                        } else {
                            self.insert_before(op_id, Op::Cast { x: eone, dtype: edt })
                        };
                        let ezero = self.insert_before(op_id, Op::Const(Constant::new(0i32)));
                        let ezero = if edt == DType::I32 {
                            ezero
                        } else {
                            self.insert_before(op_id, Op::Cast { x: ezero, dtype: edt })
                        };
                        let uone = self.insert_before(op_id, Op::Const(Constant::new(1i32)));
                        let uone = if udt == DType::I32 {
                            uone
                        } else {
                            self.insert_before(op_id, Op::Cast { x: uone, dtype: udt })
                        };
                        let mut b = x;
                        if udt != dt {
                            b = self.insert_before(op_id, Op::Bitcast { x, dtype: udt });
                        }
                        let mut ret = uone;
                        let mut sq = b;
                        let width = dt.bit_size();
                        for i in 0..width {
                            let ci =
                                self.insert_before(op_id, Op::Const(Constant::idx(i as i64)));
                            let ci = if edt == IDX_T {
                                ci
                            } else {
                                self.insert_before(op_id, Op::Cast { x: ci, dtype: edt })
                            };
                            let shr = self.insert_before(
                                op_id,
                                Op::Binary { x: y, y: ci, bop: BOp::BitShiftRight },
                            );
                            let bit = self.insert_before(
                                op_id,
                                Op::Binary { x: shr, y: eone, bop: BOp::BitAnd },
                            );
                            let cond = self.insert_before(
                                op_id,
                                Op::Binary { x: bit, y: ezero, bop: BOp::NotEq },
                            );
                            let prod = self.insert_before(
                                op_id,
                                Op::Binary { x: ret, y: sq, bop: BOp::Mul },
                            );
                            ret = self.insert_before(
                                op_id,
                                Op::Select { cond, if_true: prod, if_false: ret },
                            );
                            if i + 1 < width {
                                sq = self.insert_before(
                                    op_id,
                                    Op::Binary { x: sq, y: sq, bop: BOp::Mul },
                                );
                            }
                        }
                        let mut out = ret;
                        if udt != dt {
                            out = self.insert_before(op_id, Op::Bitcast { x: out, dtype: dt });
                            // Signed: gate the negative-exponent result
                            // (0, except base ±1; parity from the live
                            // exponent). Unsigned exponents cannot be
                            // negative, no gate needed.
                            let c0 = self.insert_before(op_id, Op::Const(Constant::new(0i32)));
                            let c0 = self.insert_before(op_id, Op::Cast { x: c0, dtype: dt });
                            let c1 = self.insert_before(op_id, Op::Const(Constant::new(1i32)));
                            let c1 = self.insert_before(op_id, Op::Cast { x: c1, dtype: dt });
                            let cm1 = self.insert_before(op_id, Op::Const(Constant::new(-1i32)));
                            let cm1 = self.insert_before(op_id, Op::Cast { x: cm1, dtype: dt });
                            let is_one = self.insert_before(
                                op_id,
                                Op::Binary { x, y: c1, bop: BOp::Eq },
                            );
                            let is_negone = self.insert_before(
                                op_id,
                                Op::Binary { x, y: cm1, bop: BOp::Eq },
                            );
                            let masked = self.insert_before(
                                op_id,
                                Op::Binary { x: y, y: eone, bop: BOp::BitAnd },
                            );
                            let odd = self.insert_before(
                                op_id,
                                Op::Binary { x: masked, y: ezero, bop: BOp::NotEq },
                            );
                            let neg_odd = self.insert_before(
                                op_id,
                                Op::Select { cond: odd, if_true: cm1, if_false: c1 },
                            );
                            let inner = self.insert_before(
                                op_id,
                                Op::Select { cond: is_negone, if_true: neg_odd, if_false: c0 },
                            );
                            let negval = self.insert_before(
                                op_id,
                                Op::Select { cond: is_one, if_true: c1, if_false: inner },
                            );
                            // `Cmpge` (not negated `Cmplt`) so the
                            // vector result stays the true arm: `Select`
                            // takes its shape from there.
                            let nonneg = self.insert_before(
                                op_id,
                                Op::Binary { x: y, y: ezero, bop: BOp::Cmpge },
                            );
                            out = self.insert_before(
                                op_id,
                                Op::Select { cond: nonneg, if_true: out, if_false: negval },
                            );
                        }
                        self.remap(op_id, out);
                    }
                } else {
                    let x = self.insert_before(op_id, Op::Unary { x, uop: UOp::Log2 });
                    let x = self.insert_before(op_id, Op::Binary { x, y, bop: BOp::Mul });
                    self.ops[op_id].op = Op::Unary { x, uop: UOp::Exp2 };
                }
            }
            op_id = self.next_op(op_id);
        }

        self.verify();
    }
}

#[cfg(test)]
mod tests {
    use crate::DType;
    use crate::kernel::{Dev, Kernel, MemScope, Op, TTOp};
    use crate::types::{TinyString, TinyVec};

    /// Two identical LLK calls are two traffic events: CSE must not
    /// merge them (the unpacker streams each CB read separately).
    #[test]
    fn cse_does_not_dedup_llk_calls() {
        let mut k = Kernel::new(Dev::Auto);
        let out = k.param_mut(DType::F32);
        let cb = k.storage(DType::F32, MemScope::Circular, 1);
        let idx = k.const_val(0i64);
        k.tt_end_reader();
        let a = k.push_back(Op::TT(TTOp::LLK { asm: TinyString::new("copy_tile({0}, {1});"), ops: TinyVec::new(&[cb, idx]) }));
        let b = k.push_back(Op::TT(TTOp::LLK { asm: TinyString::new("copy_tile({0}, {1});"), ops: TinyVec::new(&[cb, idx]) }));
        let load_a = k.load(cb, idx);
        let load_b = k.load(cb, idx);
        k.store(out, load_a, idx);
        k.store(out, load_b, idx);
        k.tt_end_compute();
        k.common_subexpression_elimination();
        // Both calls survive; neither is remapped to the other.
        let mut llk_count = 0;
        for (_, op) in k.iter_unordered() {
            if matches!(op, Op::TT(TTOp::LLK { .. })) {
                llk_count += 1;
            }
        }
        assert_eq!(llk_count, 2, "CSE must not merge two identical LLK calls");
        assert!(matches!(k.at(a), Op::TT(TTOp::LLK { .. })));
        assert!(matches!(k.at(b), Op::TT(TTOp::LLK { .. })));
    }
}
