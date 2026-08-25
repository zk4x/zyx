// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Loop merging optimization.
//!
//! This module provides loop merging optimizations for kernels,
//! which merge nested loops into single loops when possible.
//!
//! Loop merging can improve performance by:
//!
//! - Reducing loop overhead
//! - Enabling better instruction scheduling
//! - Improving vectorization opportunities

use std::collections::BTreeMap;

use super::autotune::Optimization;
use crate::{
    dtype::Constant,
    kernel::{BOp, IdxKind, Kernel, Op, OpId},
    shape::Dim,
};

impl Kernel {
    /// Get last op in the given loop scope
    pub(crate) fn get_last_dim_op(&self, loop_id: OpId) -> OpId {
        match self.ops[loop_id].op {
            Op::Index { .. } => return self.tail,
            Op::Loop { .. } => {}
            _ => unreachable!(),
        }
        let mut loop_depth = 0;
        let mut op_id = loop_id;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Loop { .. } => {
                    loop_depth += 1;
                }
                Op::EndLoop => {
                    loop_depth -= 1;
                    if loop_depth == 0 {
                        return op_id;
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        op_id
    }

    /// Merge nested Op::Loops into a single loop.
    ///
    /// Takes a chain of nested loops (outermost first) and merges them into one
    /// loop whose length is the product of all lengths.  After merging, each
    /// original loop is replaced with arithmetic that decomposes the merged loop
    /// variable back into the original loop variables (via `/` and `%` chain,
    /// like `merge_indices`).  The existing address computation continues to
    /// work correctly because it still references the same OpIds.
    pub(crate) fn merge_nested_loops(&mut self, loop_ids: &[OpId]) {
        if loop_ids.len() < 2 {
            return;
        }

        // Merging nested loops into one flat loop is only valid when the loop
        // body has no mutable register state that depends on the outer/inner
        // boundary. A `def mut reg` inside the outer loop (e.g. a per-row count
        // accumulator that is reset once per outer iteration and accumulated
        // across the inner iterations) runs its reset once per outer iteration
        // originally, but once per *flat* iteration after merging — changing the
        // result. Refuse to merge such groups.
        //
        // The register may be defined outside the outer loop (linearize hoists
        // accumulator storages to the top of the kernel and resets them via
        // Store inside the loop), so also refuse when a Register is stored or
        // loaded anywhere in the region — its value crosses the inner/outer
        // boundary either way.
        let anchor = loop_ids[0];
        let end = self.get_last_dim_op(anchor);
        let mut op_id = self.next_op(anchor);
        while op_id != end {
            if let Op::Store { .. } = self.ops[op_id].op {
                return;
            }
            op_id = self.next_op(op_id);
        }

        let mut total_len: u64 = 1;
        for &id in loop_ids {
            if let Op::Loop { len: len_id } = self.ops[id].op {
                let Some(len) = self.resolve_const(len_id).and_then(crate::dtype::Constant::as_dim) else {
                    return;
                };
                total_len = total_len * len as u64;
            }
        }

        // Replace original loops with merged loop, removing inner EndLoops
        let anchor = loop_ids[0];
        let merged_len = self.insert_const_idx_before(anchor, total_len);
        let mut x = self.insert_before(anchor, Op::Loop { len: merged_len });

        // Single pass: remove inner EndLoops (keep only the last one)
        let mut op_id = self.next_op(anchor);
        let mut depth: u32 = 1;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match self.ops[op_id].op {
                Op::Loop { .. } => depth += 1,
                Op::EndLoop => {
                    depth -= 1;
                    if depth > 0 {
                        self.remove_op(op_id);
                    } else {
                        break;
                    }
                }
                _ => {}
            }
            op_id = next;
        }

        // Decompose the merged loop variable back into original loop variables.
        // Process innermost to outermost (reverse order of loop_ids).
        // Insert all new ops before the anchor so all definitions
        // precede all uses (avoids backward-reference verification errors).
        for i in (0..loop_ids.len()).rev() {
            let Op::Loop { len: len_id } = self.ops[loop_ids[i]].op else {
                unreachable!()
            };
            let len = self.resolve_const(len_id).and_then(crate::dtype::Constant::as_dim).unwrap();
            let y = self.insert_before(anchor, Op::Const(Constant::idx(len)));
            self.ops[loop_ids[i]].op = Op::Binary { x, y, bop: BOp::Mod };
            x = self.insert_before(anchor, Op::Binary { x, y, bop: BOp::Div });
        }
    }

    /// Merges two or more indices together
    pub(crate) fn merge_indices(&mut self, loops: &[OpId]) {
        let mut acc = 1;
        let mut axes = BTreeMap::default();
        let mut first_id = None;
        let mut op_id = self.head;

        // The decomposed Mod/Div chain must land in list order: each index's
        // Mod uses the previous index's Div, so the indices have to appear in
        // the same order they will be processed (ascending axis). Collect the
        // list order while scanning.
        #[cfg(debug_assertions)]
        let mut list_order: Vec<OpId> = Vec::new();
        while axes.len() != loops.len() {
            if loops.contains(&op_id) {
                // TODO check all scopes are the same
                let Op::Index { axis, kind } = self.ops[op_id].op else {
                    unreachable!()
                };
                let len = match kind {
                    IdxKind::Group(op_id) => self.resolve_const(op_id).and_then(crate::dtype::Constant::as_dim),
                    IdxKind::Local(len) => Some(len as Dim),
                    IdxKind::Warp(len) => Some(len as Dim),
                };
                let Some(len) = len else { return };
                acc *= len;
                axes.insert(axis, (op_id, len));
                #[cfg(debug_assertions)]
                list_order.push(op_id);
                if first_id.is_none() {
                    first_id = Some(op_id);
                }
            }
            op_id = self.next_op(op_id);
        }

        #[cfg(debug_assertions)]
        {
            // The index ops must already be ordered by axis in the list, otherwise
            // a Div inserted after one index would land after the next index's Mod
            // that consumes it. This optimization can't handle that input; leave
            // the kernel untouched.
            let process_order: Vec<OpId> = axes.values().map(|&(id, _)| id).collect();
            if process_order != list_order {
                panic!("merge_indices requires order");
            }
        }

        let Op::Index { axis, kind, .. } = self.ops[first_id.unwrap()].op else {
            unreachable!()
        };
        let kind = match kind {
            IdxKind::Group(_) => IdxKind::Group(self.insert_before(first_id.unwrap(), Op::Const(Constant::idx(acc)))),
            IdxKind::Local(_) => IdxKind::Local(acc as u32),
            IdxKind::Warp(_) => IdxKind::Warp(acc as u8),
        };
        let mut x = self.insert_before(first_id.unwrap(), Op::Index { axis, kind });

        for (.., (loop_id, len)) in axes {
            let y = self.insert_before(loop_id, Op::Const(Constant::idx(len)));
            self.ops[loop_id].op = Op::Binary { x, y, bop: BOp::Mod };
            x = self.insert_after(loop_id, Op::Binary { x, y, bop: BOp::Div });
        }

        self.verify();
    }

    /// Returns the Optimization for merging nested loops and the number of nested loop groups.
    /// Each group is a chain of nested loops that can be merged into one loop.
    pub(crate) fn opt_merge_nested_loops(&self) -> (Optimization, usize) {
        let groups = self.find_nested_loop_groups();
        let n = groups.len();
        (Optimization::MergeNestedLoops { groups }, n)
    }

    /// Find all groups of nested loops in the kernel.
    /// Each group is a chain of consecutive nested loops (outermost first).
    fn find_nested_loop_groups(&self) -> Vec<Vec<OpId>> {
        let mut groups: Vec<Vec<OpId>> = Vec::new();
        let mut current_group: Vec<OpId> = Vec::new();
        let mut depth: u32 = 0;
        let mut in_group = false;

        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Loop { .. } => {
                    if depth == 0 {
                        // Start a new group
                        if in_group {
                            groups.push(std::mem::take(&mut current_group));
                        }
                        in_group = true;
                    }
                    current_group.push(op_id);
                    depth += 1;
                }
                Op::EndLoop => {
                    depth -= 1;
                    if depth == 0 {
                        // End of this group
                        if !current_group.is_empty() {
                            groups.push(std::mem::take(&mut current_group));
                        }
                        in_group = false;
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }

        // Flush any remaining group
        if !current_group.is_empty() {
            groups.push(current_group);
        }

        groups.retain(|g| g.len() >= 2);
        groups
    }
}
