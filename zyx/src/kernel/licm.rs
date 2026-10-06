// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Loop-invariant code motion and reassociation.
//!
//! This module provides optimizations for loop-invariant code motion
//! and reassociation of commutative operations.
//!
//! Optimizations include:
//!
//! - `opt_reassociate_commutative`: Reassociate commutative operations
//! - `swap_commutative`: Swap commutative operands
//! - `loop_invariant_code_motion`: Hoist loop-invariant computations
//!
//! These optimizations reduce redundant computations and improve performance.

use super::autotune::Optimization;
use crate::kernel::{Kernel, Op, OpId, TTOp};
use crate::{DType, Map, Set};

/// Reassociate commutative operations (addition, multiplication)
/// to group them and reduce instruction count.
#[derive(Debug)]
pub struct ReassociateCommutative;

impl Optimization for ReassociateCommutative {
    fn nconfigs(&self) -> u64 {
        1
    }

    fn apply(&self, kernel: &mut Kernel, _config: u64) {
        kernel.reassociate_commutative();
    }
}

impl Kernel {
    /// Make the `ReassociateCommutative` optimization.
    pub fn opt_reassociate_commutative(&self) -> Box<dyn Optimization> {
        Box::new(ReassociateCommutative)
    }

    /// Swap commutative operands for better instruction scheduling.
    ///
    /// This method swaps operands of commutative operations (addition,
    /// multiplication) to improve instruction scheduling and pipeline
    /// utilization.
    pub(crate) fn swap_commutative(&mut self) {
        // Tracks whether a value depends on a loop index
        let mut loop_dep: Map<OpId, usize> = Map::default();
        let mut loop_depth = 0;
        let mut op_id = self.head;
        while !op_id.is_null() {
            let depth = match self.ops[op_id].op {
                Op::Reshape { .. }
                | Op::Pad { .. }
                | Op::Permute { .. }
                | Op::Expand { .. }
                | Op::Flip { .. }
                | Op::Narrow { .. }
                | Op::Reduce { .. }
                | Op::TT(TTOp::ReduceTile { .. }) => {
                    unreachable!()
                }
                Op::After { .. } | Op::ToDevice { .. } | Op::Contiguous { .. } | Op::Program { .. } | Op::Custom(_) => {
                    unreachable!()
                }
                Op::TT(TTOp::MatmulTile { x, y, acc }) => loop_dep[&x].max(loop_dep[&y]).max(loop_dep[&acc]),
                Op::TT(TTOp::TransposeTile { x }) => loop_dep[&x],
                Op::TT(TTOp::BroadcastTile { x, .. }) => loop_dep[&x],
                // Effect-only TT ops: pinned like Store/Copy, never hoisted.
                Op::TT(TTOp::ReserveBack { .. })
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
                | Op::TT(TTOp::LLK { .. })
                | Op::TT(TTOp::LLKReduce { .. })
                | Op::TT(TTOp::LLKBcast { .. }) => loop_depth,
                Op::Asm { .. } | Op::Index { .. } | Op::Wmma { .. } | Op::Stack { .. } => loop_depth,
                Op::Loop { .. } => {
                    loop_depth += 1;
                    loop_depth
                }
                Op::EndLoop => {
                    loop_depth -= 1;
                    loop_depth
                }
                Op::Unary { x, .. } | Op::Cast { x, .. } | Op::Bitcast { x, .. } => loop_dep[&x],
                Op::Binary { x, y, bop } => {
                    if bop.is_commutative()
                        && !self.ops[x].op.is_const()
                        && (loop_dep[&x] > loop_dep[&y] || self.ops[y].op.is_const() || self.ops[x].op.is_load())
                    {
                        //println!("Swapping {x}, {y}, loop dep {} > {}: {:?}, {:?}", loop_dep[&x], loop_dep[&y], self.ops[x].op, self.ops[y].op);
                        if let Op::Binary { x, y, .. } = &mut self.ops[op_id].op {
                            std::mem::swap(x, y);
                        }
                    }
                    loop_dep[&x].max(loop_dep[&y])
                }
                Op::Param { .. }
                | Op::Barrier
                | Op::Range { .. }
                | Op::Load { .. }
                | Op::Store { .. }
                | Op::Copy { .. }
                | Op::Const(_)
                | Op::Storage { .. } => loop_depth,
                Op::GEP { x, index, .. } => loop_dep[&x].max(loop_dep[&index]),
            };
            loop_dep.insert(op_id, depth);
            op_id = self.next_op(op_id);
        }

        self.verify();
    }

    /// Reassociate commutative operations to group them.
    ///
    /// This method reassociates commutative operations (addition,
    /// multiplication) to group them and reduce instruction count.
    ///
    /// For example, `a + b + c` can be transformed to `(a + b) + c`
    /// to enable better instruction scheduling.
    pub(crate) fn reassociate_commutative(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("reassociate_commutative");
        let mut loop_dep: Map<OpId, usize> = Map::default();
        let mut loop_depth = 0;
        let mut op_id = self.head;
        while !op_id.is_null() {
            let depth = match self.at(op_id) {
                Op::Reshape { .. }
                | Op::Pad { .. }
                | Op::Permute { .. }
                | Op::Expand { .. }
                | Op::Flip { .. }
                | Op::Narrow { .. }
                | Op::Reduce { .. } => {
                    unreachable!()
                }
                Op::After { .. } | Op::ToDevice { .. } | Op::Contiguous { .. } | Op::Program { .. } | Op::Custom(_) => {
                    unreachable!()
                }
                Op::TT(TTOp::ReduceTile { x, scaler, acc, .. }) => loop_dep[x].max(loop_dep[scaler]).max(loop_dep[acc]),
                Op::TT(TTOp::MatmulTile { x, y, acc }) => loop_dep[x].max(loop_dep[y]).max(loop_dep[acc]),
                Op::TT(TTOp::TransposeTile { x }) => loop_dep[x],
                Op::TT(TTOp::BroadcastTile { x, .. }) => loop_dep[x],
                // Effect-only TT ops: pinned like Store/Copy, never hoisted.
                Op::TT(TTOp::ReserveBack { .. })
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
                | Op::TT(TTOp::LLK { .. })
                | Op::TT(TTOp::LLKReduce { .. })
                | Op::TT(TTOp::LLKBcast { .. }) => loop_depth,
                Op::Asm { ops, .. } => {
                    let mut max = 0;
                    for op in ops.iter() {
                        max = max.max(loop_dep[op]);
                    }
                    max
                }
                Op::Stack { ops } => {
                    let mut max = 0;
                    for op in ops.iter() {
                        max = max.max(loop_dep[op]);
                    }
                    max
                }
                Op::Index { vec, .. } => loop_dep[vec],
                Op::Loop { .. } => {
                    loop_depth += 1;
                    loop_depth
                }
                Op::EndLoop => {
                    loop_depth -= 1;
                    loop_depth
                }
                Op::Unary { x, .. } | Op::Cast { x, .. } | Op::Bitcast { x, .. } => loop_dep[x],
                Op::Binary { x, y, .. } => loop_dep[x].max(loop_dep[y]),
                Op::Range { .. }
                | Op::Barrier
                | Op::Load { .. }
                | Op::Store { .. }
                | Op::Copy { .. }
                | Op::Const(_)
                | Op::Param { .. }
                | Op::Storage { .. }
                | Op::Wmma { .. } => loop_depth,
                Op::GEP { x, index, .. } => loop_dep[x].max(loop_dep[index]),
            };
            loop_dep.insert(op_id, depth);
            op_id = self.next_op(op_id);
        }

        let mut op_id = self.head;
        'a: while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let &Op::Binary { bop, .. } = self.at(op_id) {
                if !bop.is_commutative() || !bop.is_associative() {
                    op_id = next;
                    continue 'a;
                }

                // Get all the leafs
                let mut params = vec![op_id];
                let mut chain = Vec::new();
                while let Some(param) = params.pop() {
                    if let &Op::Binary { x, y, bop: t_bop } = self.at(param)
                        && t_bop == bop
                    {
                        params.push(x);
                        params.push(y);
                        continue;
                    }
                    chain.push(param);
                    // We have to be somewhat reasonabe about those chains
                    if chain.len() > 20 {
                        op_id = next;
                        continue 'a;
                    }
                }
                if chain.len() < 2 {
                    op_id = next;
                    continue 'a;
                }
                chain.sort_by_key(|id| loop_dep[id]);

                // Rebuild chain
                let mut prev_acc = chain[0];
                let mut j = 1;
                while j < chain.len() - 1 {
                    let op = Op::Binary { x: chain[j], y: prev_acc, bop };
                    let new_acc = self.insert_before(op_id, op);
                    prev_acc = new_acc;
                    j += 1;
                }
                self.ops[op_id].op = Op::Binary { x: chain[j], y: prev_acc, bop };
            }
            op_id = next;
        }

        self.verify();
    }

    /// Hoist loop-invariant computations outside loops.
    ///
    /// This method identifies computations that are invariant within
    /// loops and moves them outside the loop to reduce redundant
    /// computations and improve performance.
    pub fn loop_invariant_code_motion(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("loop_invariant_code_motion");
        let mut endloop_is = Vec::new();
        let mut loop_id = self.tail;
        while !loop_id.is_null() {
            if *self.at(loop_id) == Op::EndLoop {
                endloop_is.push(loop_id);
            }
            if let &Op::Loop { len } = self.at(loop_id) {
                let endloop_id = endloop_is.pop().unwrap();
                // Boolean loops are conditionals: hoisting their body out
                // would execute it unconditionally. Only counted loops hoist
                // (top-level conditionals never hoisted before either).
                // The pop above still runs so enclosing loops pair correctly.
                if self.dtype(len) == DType::Bool {
                    loop_id = self.prev_op(loop_id);
                    continue;
                }
                let mut op_ids_in_loop = Set::default();
                op_ids_in_loop.insert(loop_id); // Loop op is the primary op that breaks LICM

                let mut op_id = loop_id;
                while op_id != endloop_id {
                    let op = self.at(op_id);
                    let next_op_id = self.next_op(op_id);

                    if !matches!(
                        op,
                        Op::Store { .. }
                            | Op::Copy { .. }
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
                            | Op::TT(TTOp::LLK { .. })
                            | Op::TT(TTOp::LLKReduce { .. })
                            | Op::TT(TTOp::LLKBcast { .. })
                            | Op::Load { .. }
                            | Op::Loop { .. }
                            | Op::EndLoop
                            | Op::Storage { .. }
                            | Op::Barrier
                    ) && op.parameters().all(|op_id| !op_ids_in_loop.contains(&op_id))
                    {
                        self.move_op_before(op_id, loop_id);
                    } else {
                        op_ids_in_loop.insert(op_id);
                    }

                    op_id = next_op_id;
                }
            }
            loop_id = self.prev_op(loop_id);
        }

        self.verify();
    }
}
