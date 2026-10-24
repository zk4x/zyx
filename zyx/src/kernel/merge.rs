// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Pairwise axis merging.
//!
//! `merge(axis_a, axis_b)` fuses exactly two axes into one flat axis and
//! rewrites the two original axes as div/mod decomposition of the flat
//! axis. Two arms, both refuse with `OpId::NULL` (never wrong):
//!
//! - `Loop` + `Loop`: the inner loop must lie inside the outer's region;
//!   a `Store`/`Copy` anywhere in the region refuses (per-iteration
//!   register reset semantics would change).
//! - `Range` + `Range`: both `Group` or both `Local` (`Warp`/`Unroll`
//!   bail); axes strictly ascending and `axis_a` before `axis_b` in list
//!   order; symbolic lengths bail.
//!
//! Chains fold through pairwise calls: nested loop chains outermost-first,
//! index lists in list order. Every step preserves semantics, so a refused
//! later step leaves valid IR behind.

use super::autotune::Optimization;
use crate::{
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, RangeKind},
    shape::Dim,
};

/// Merge axes into flat axes (loops and ranges).
///
/// Each entry is either a nested loop chain (outermost first) or a range
/// pair. Config index selects which entry to merge; chains and pairs fold
/// through pairwise `merge` calls.
#[derive(Debug)]
pub struct Merge {
    /// Loop chains and range pairs to merge.
    pub groups: Vec<Vec<OpId>>,
}

impl Optimization for Merge {
    fn nconfigs(&self) -> u64 {
        self.groups.len() as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        let Some(chain) = self.groups.get(config as usize) else {
            return;
        };
        if chain.len() < 2 {
            return;
        }
        // Fold outermost-first: merge(chain[0], chain[1]), then merge the
        // flat result with chain[2], and so on. The flat loop stays at
        // chain[0]'s site, so each later step is still a valid pair.
        let mut flat = chain[0];
        for &next in &chain[1..] {
            flat = kernel.merge(flat, next);
            if flat.is_null() {
                return;
            }
        }
    }
}

impl Kernel {
    /// Get last op in the given loop scope
    pub(crate) fn get_last_dim_op(&self, loop_id: OpId) -> OpId {
        match self.ops[loop_id].op {
            Op::Range { .. } => return self.tail,
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

    /// Fuse exactly two axes into one flat axis. Returns the flat axis id,
    /// `OpId::NULL` when the pair cannot merge (untouched kernel).
    pub(crate) fn merge(&mut self, axis_a: OpId, axis_b: OpId) -> OpId {
        if axis_a == axis_b {
            return OpId::NULL;
        }
        let a_is_loop = matches!(self.ops[axis_a].op, Op::Loop { .. });
        let b_is_loop = matches!(self.ops[axis_b].op, Op::Loop { .. });
        let a_is_range = matches!(self.ops[axis_a].op, Op::Range { .. });
        let b_is_range = matches!(self.ops[axis_b].op, Op::Range { .. });
        if a_is_loop && b_is_loop {
            self.merge_loops(axis_a, axis_b)
        } else if a_is_range && b_is_range {
            self.merge_ranges(axis_a, axis_b)
        } else {
            OpId::NULL
        }
    }

    /// Fuse two loops: `inner` must lie inside `outer`'s region. The flat
    /// loop lands before `outer`; both originals become div/mod of it.
    fn merge_loops(&mut self, outer: OpId, inner: OpId) -> OpId {
        let end = self.get_last_dim_op(outer);
        let mut op_id = self.next_op(outer);
        let mut inside = false;
        while op_id != end {
            if op_id.is_null() {
                return OpId::NULL;
            }
            if op_id == inner {
                inside = true;
            }
            // A `def mut reg` inside the outer loop (e.g. a per-row count
            // accumulator reset once per outer iteration) would run its
            // reset once per *flat* iteration after merging — changing the
            // result. The register may be created outside the outer loop
            // (linearize hoists accumulator storages to the top and resets
            // them via Store inside the loop), so refuse on any Register
            // store or load in the region.
            if matches!(self.ops[op_id].op, Op::Store { .. } | Op::Copy { .. }) {
                return OpId::NULL;
            }
            op_id = self.next_op(op_id);
        }
        if !inside {
            return OpId::NULL;
        }

        let Op::Loop { len: len_a_id } = self.ops[outer].op else {
            return OpId::NULL;
        };
        let Op::Loop { len: len_b_id } = self.ops[inner].op else {
            return OpId::NULL;
        };
        let (Some(len_a), Some(len_b)) = (
            self.resolve_const(len_a_id).and_then(Constant::as_dim),
            self.resolve_const(len_b_id).and_then(Constant::as_dim),
        ) else {
            return OpId::NULL;
        };
        let total_len: u64 = len_a as u64 * len_b as u64;

        let merged_len = self.insert_const_idx_before(outer, total_len);
        let flat = self.insert_before(outer, Op::Loop { len: merged_len });

        // Remove inner EndLoops, keep only the last one.
        let mut op_id = self.next_op(outer);
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

        // Decompose the flat loop variable back into the originals,
        // innermost first. All new ops land before `outer` so definitions
        // precede uses.
        let mut x = flat;
        for id in [inner, outer] {
            let Op::Loop { len: len_id } = self.ops[id].op else {
                return OpId::NULL;
            };
            let len = self
                .resolve_const(len_id)
                .and_then(Constant::as_dim)
                .expect("merge: loop len resolved above");
            let y = self.insert_before(outer, Op::Const(Constant::idx(len)));
            self.ops[id].op = Op::Binary { x, y, bop: BOp::Mod };
            x = self.insert_before(outer, Op::Binary { x, y, bop: BOp::Div });
        }
        self.verify();
        flat
    }

    /// Fuse two ranges into one flat range at `first`'s site. Both ranges
    /// become div/mod of it, ascending axis order.
    fn merge_ranges(&mut self, first: OpId, second: OpId) -> OpId {
        let Op::Range { axis: axis_a, kind: kind_a } = self.ops[first].op else {
            return OpId::NULL;
        };
        let Op::Range { axis: axis_b, kind: kind_b } = self.ops[second].op else {
            return OpId::NULL;
        };
        let both_group =
            matches!(kind_a, RangeKind::Group(_)) && matches!(kind_b, RangeKind::Group(_));
        let both_local =
            matches!(kind_a, RangeKind::Local(_)) && matches!(kind_b, RangeKind::Local(_));
        if !(both_group || both_local) {
            return OpId::NULL;
        }
        if axis_a >= axis_b {
            return OpId::NULL;
        }
        // `second` must come after `first` in list order.
        let mut found = false;
        let mut op_id = self.next_op(first);
        while !op_id.is_null() {
            if op_id == second {
                found = true;
                break;
            }
            op_id = self.next_op(op_id);
        }
        if !found {
            return OpId::NULL;
        }

        let len_a: Dim = match kind_a {
            RangeKind::Group(len_id) => {
                let Some(len) = self.resolve_const(len_id).and_then(Constant::as_dim) else {
                    return OpId::NULL;
                };
                len
            }
            RangeKind::Local(len) => len as Dim,
            RangeKind::Warp(_) => return OpId::NULL,
        };
        let len_b: Dim = match kind_b {
            RangeKind::Group(len_id) => {
                let Some(len) = self.resolve_const(len_id).and_then(Constant::as_dim) else {
                    return OpId::NULL;
                };
                len
            }
            RangeKind::Local(len) => len as Dim,
            RangeKind::Warp(_) => return OpId::NULL,
        };
        let mut acc = 1;
        acc *= len_a;
        acc *= len_b;
        let kind = match kind_a {
            RangeKind::Group(_) => {
                RangeKind::Group(self.insert_before(first, Op::Const(Constant::idx(acc))))
            }
            RangeKind::Local(_) => RangeKind::Local(acc as u32),
            RangeKind::Warp(_) => return OpId::NULL,
        };
        let merged = self.insert_before(first, Op::Range { axis: axis_a, kind });

        let mut x = merged;
        for (id, len) in [(first, len_a), (second, len_b)] {
            let y = self.insert_before(id, Op::Const(Constant::idx(len)));
            self.ops[id].op = Op::Binary { x, y, bop: BOp::Mod };
            x = self.insert_after(id, Op::Binary { x, y, bop: BOp::Div });
        }
        self.verify();
        merged
    }

    /// Merges two or more indices together, in list order.
    pub(crate) fn merge_indices(&mut self, loops: &[OpId]) {
        if loops.len() < 2 {
            return;
        }
        // The decomposed Mod/Div chain must land in list order: each index's
        // Mod uses the previous index's Div, so collect in list order, not
        // input order. Unknown or stale ids leave the kernel untouched.
        let mut ordered: Vec<OpId> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if loops.contains(&op_id) {
                if !matches!(self.ops[op_id].op, Op::Range { .. }) {
                    return;
                }
                ordered.push(op_id);
            }
            op_id = self.next_op(op_id);
        }
        if ordered.len() != loops.len() {
            return;
        }
        let mut flat = ordered[0];
        for &id in &ordered[1..] {
            flat = self.merge(flat, id);
            if flat.is_null() {
                return;
            }
        }
    }

    /// Make the `Merge` optimization: each config merges one nested loop
    /// chain or one range pair into a single flat axis.
    pub fn opt_merge(&self) -> Box<dyn Optimization> {
        let mut groups = self.find_nested_loop_groups();
        // Group-range pairs fuse grid axes into one flat axis with
        // div/mod index recovery (user spec): propose them only when
        // the kernel holds more than 2 group ranges (excess to reduce
        // toward the grid); 1-2 groups are the legal tiled end-state.
        // Loop chains and Local pairs keep today's behavior.
        let mut n_groups = 0;
        for (_, x) in self.iter_unordered() {
            if matches!(x, Op::Range { kind: RangeKind::Group(_), .. }) {
                n_groups += 1;
            }
        }
        for pair in self.find_range_pairs() {
            let is_group_pair = matches!(self.ops[pair[0]].op, Op::Range { kind: RangeKind::Group(_), .. });
            if !is_group_pair || n_groups > 2 {
                groups.push(pair);
            }
        }
        Box::new(Merge { groups })
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

    /// Find all consecutive same-kind range pairs in list order, ascending
    /// axis. Consecutive means no other range between them (loads, stores
    /// and arithmetic may sit in between). Each pair is one merge config.
    fn find_range_pairs(&self) -> Vec<Vec<OpId>> {
        let mut pairs: Vec<Vec<OpId>> = Vec::new();
        let mut prev: Option<(OpId, u32, bool)> = None;
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Range { axis, kind } = self.ops[op_id].op {
                let is_group = matches!(kind, RangeKind::Group(_));
                let is_local = matches!(kind, RangeKind::Local(_));
                if is_group || is_local {
                    if let Some((prev_id, prev_axis, prev_group)) = prev {
                        if prev_group == is_group && prev_axis < axis {
                            pairs.push(vec![prev_id, op_id]);
                        }
                    }
                    prev = Some((op_id, axis, is_group));
                }
            }
            op_id = self.next_op(op_id);
        }
        pairs
    }
}
