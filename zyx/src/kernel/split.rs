// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Split optimization.
//!
//! Splits dimensions of the index space: a global group range into a smaller
//! group plus local threads ([`SplitMode::Local`]), a global group range or a
//! reduce loop into a smaller part plus unroll lanes ([`SplitMode::Unroll`]),
//! and a loop into two nested loops ([`SplitMode::Loop`]).
//!
//! Every split is expressed through [`Kernel::split_dim`], which rewrites
//! the split index as the affine composition of the parts, so the IR stays
//! linear in op count per variant: lane growth happens only when
//! devectorization desugars the chosen variant.

use super::autotune::Optimization;
use crate::{
    Map, Set,
    dtype::Constant,
    kernel::{BOp, Kernel, MemScope, Op, OpId, RangeKind},
    shape::Dim,
};

/// The kind of the split's secondary part. The site kind (a [`Op::Range`]
/// group or an [`Op::Loop`]) and this variant together decide the split:
/// `Local` splits a group into a smaller group plus local threads;
/// `Unroll` splits a group or a reduce loop into a smaller part plus unroll
/// lanes. A group split grows only the register storages that become
/// lane-dependent (coarsen's slotting); a reduce-loop split keeps its scalar
/// accumulator and folds the per-iteration lane bundle at the store arm.
#[derive(Debug)]
pub enum SplitMode {
    /// Secondary part is a local thread range.
    Local,
    /// Secondary part is unroll lanes; the lane axis is the next free one
    /// at apply time.
    Unroll,
}

/// The split optimization: one config splits one site by one factor.
#[derive(Debug)]
pub struct OptSplit {
    /// (site, mode, factor) per config. Ordering prefers hardware-aligned
    /// factors that are likely to perform better.
    pub configs: Vec<(OpId, SplitMode, u16)>,
}

impl Optimization for OptSplit {
    fn nconfigs(&self) -> u64 {
        self.configs.len() as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("OptSplit::apply");
        let &(site, ref mode, factor) = &self.configs[config as usize];
        match mode {
            SplitMode::Local => {
                let Op::Range { axis, kind: RangeKind::Group(len_id) } = kernel.ops[site].op else {
                    unreachable!("local split applies to a group range site");
                };
                let len = kernel.resolve_const(len_id).and_then(Constant::as_dim).unwrap();
                let factor = Dim::from(factor);
                debug_assert!(len % factor == 0, "group length {len} not divisible by {factor}");
                let group_len = kernel.insert_const_idx_before(site, len / factor);
                kernel.split_dim(
                    site,
                    vec![
                        Op::Range { axis, kind: RangeKind::Group(group_len) },
                        Op::Range { axis, kind: RangeKind::Local(u32::from(factor as u16)) },
                    ],
                );
            }
            SplitMode::Unroll => {
                // The lane axis is the next free one; nothing runs between
                // the make scan and apply, so the state matches make's view.
                let mut next_lane_axis = 0u64;
                for (_, op) in kernel.iter_unordered() {
                    match op {
                        Op::Range { axis, kind: RangeKind::Unroll(_) } => {
                            next_lane_axis = next_lane_axis.max(u64::from(*axis) + 1);
                        }
                        Op::Unroll { axis, .. } => next_lane_axis = next_lane_axis.max(u64::from(*axis) + 1),
                        _ => {}
                    }
                }
                let axis = u32::try_from(next_lane_axis).expect("unroll lane axis overflows u32");
                let lanes = factor;
                match kernel.ops[site].op {
                    Op::Range { axis: group_axis, kind: RangeKind::Group(len_id) } => {
                        // Global -> unroll lanes. Soundness validation first,
                        // no mutation yet: coarsen's gates rooted at the range
                        // whose lanes the split introduces. Whole-view writes
                        // (dst is an SSA value) and shared-memory effects a
                        // lane axis cannot slot refuse the split silently,
                        // mirroring `coarsen`'s refusal contract.
                        let len = kernel.resolve_const(len_id).and_then(Constant::as_dim).unwrap();
                        let factor = Dim::from(factor);
                        debug_assert!(len % factor == 0, "group length {len} not divisible by {factor}");

                        let mut varies: Set<OpId> = Set::default();
                        varies.insert(site);
                        for _ in 0..10_000 {
                            let mut grown = false;
                            for (op, _) in kernel.iter_unordered() {
                                if !varies.contains(&op) && kernel.at(op).parameters().any(|p| varies.contains(&p)) {
                                    varies.insert(op);
                                    grown = true;
                                }
                            }
                            if !grown {
                                break;
                            }
                        }
                        let mut valid = true;
                        for (_, op) in kernel.iter_unordered() {
                            match *op {
                                Op::Store { dst, dst_idx, src, .. } => {
                                    let addr_varies = !dst_idx.is_null() && varies.contains(&dst_idx);
                                    let val_varies = varies.contains(&src);
                                    let base = kernel.at(dst);
                                    let global =
                                        matches!(base, Op::Storage { scope: MemScope::Global, .. } | Op::Param { .. });
                                    if !matches!(base, Op::Storage { .. } | Op::Param { .. }) {
                                        valid = false;
                                    } else if global && !addr_varies {
                                        valid = false;
                                    } else if !global
                                        && !matches!(base, Op::Storage { scope: MemScope::Register, .. })
                                        && !addr_varies
                                        && val_varies
                                    {
                                        valid = false;
                                    }
                                }
                                Op::Copy { dst, dst_idx, .. } => {
                                    let addr_varies = !dst_idx.is_null() && varies.contains(&dst_idx);
                                    let base = kernel.at(dst);
                                    let global =
                                        matches!(base, Op::Storage { scope: MemScope::Global, .. } | Op::Param { .. });
                                    if !matches!(base, Op::Storage { .. } | Op::Param { .. }) {
                                        valid = false;
                                    } else if global && !addr_varies {
                                        valid = false;
                                    }
                                }
                                _ => {}
                            }
                        }
                        if !valid {
                            return;
                        }

                        // Split the group range: [Group(len/factor), Unroll
                        // lanes]; the composed index rides `split_dim` and its
                        // consumers become lane-dependent through the shape
                        // algebra.
                        let group_len = kernel.insert_const_idx_before(site, len / factor);
                        let split_ids = kernel.split_dim(
                            site,
                            vec![
                                Op::Range { axis: group_axis, kind: RangeKind::Group(group_len) },
                                Op::Unroll { axis, len: lanes },
                            ],
                        );
                        let unroll_id = split_ids[1];

                        // Lane-dependent ops: everything transitively consuming
                        // the unroll node.
                        let mut varies: Set<OpId> = Set::default();
                        varies.insert(unroll_id);
                        for _ in 0..10_000 {
                            let mut grown = false;
                            for (op, _) in kernel.iter_unordered() {
                                if !varies.contains(&op) && kernel.at(op).parameters().any(|p| varies.contains(&p)) {
                                    varies.insert(op);
                                    grown = true;
                                }
                            }
                            if !grown {
                                break;
                            }
                        }

                        // Register storages receiving lane-dependent values or
                        // accessed at lane-dependent addresses grow to one slot
                        // block per lane (coarsen's `len *= factor`); nothing
                        // else does. One pass: `grows` is itself consulted, so
                        // a single lane-varying access grows the storage it
                        // belongs to.
                        let op_ids: Vec<OpId> = kernel.ops.ids().collect();
                        let mut grows: Set<OpId> = Set::default();
                        for &walk_id in &op_ids {
                            match kernel.at(walk_id) {
                                Op::Store { dst, dst_idx, src, .. } => {
                                    let is_register =
                                        matches!(kernel.at(*dst), Op::Storage { scope: MemScope::Register, .. });
                                    let lane_dep = varies.contains(src)
                                        || (!dst_idx.is_null() && varies.contains(dst_idx))
                                        || grows.contains(dst);
                                    if is_register && lane_dep {
                                        grows.insert(*dst);
                                    }
                                }
                                Op::Load { src, src_idx, .. } => {
                                    let is_register =
                                        matches!(kernel.at(*src), Op::Storage { scope: MemScope::Register, .. });
                                    let lane_dep =
                                        (!src_idx.is_null() && varies.contains(src_idx)) || grows.contains(src);
                                    if is_register && lane_dep {
                                        grows.insert(*src);
                                    }
                                }
                                Op::Copy { dst, dst_idx, .. } => {
                                    let is_register =
                                        matches!(kernel.at(*dst), Op::Storage { scope: MemScope::Register, .. });
                                    let lane_dep = (!dst_idx.is_null() && varies.contains(dst_idx)) || grows.contains(dst);
                                    if is_register && lane_dep {
                                        grows.insert(*dst);
                                    }
                                }
                                _ => {}
                            }
                        }
                        // Grow the storages themselves: one slot block
                        // per lane.
                        for &growth_id in &grows {
                            match kernel.at(growth_id) {
                                Op::Storage { dtype, scope, len, .. } => {
                                    let (dtype, scope, len) = (*dtype, *scope, *len);
                                    kernel.ops[growth_id].op = Op::Storage { dtype, scope, len: len * factor };
                                }
                                _ => unreachable!("lane split grows a storage"),
                            }
                        }
                        // Re-index accesses to grown storages: every address
                        // becomes `idx*factor + lane` (coarsen's slot blocks),
                        // and lane-invariant stored values broadcast across
                        // the slot block as `value + 0*lane`.
                        for &walk_id in &op_ids {
                            match kernel.ops[walk_id].op {
                                Op::Load { src, src_idx } if grows.contains(&src) => {
                                    debug_assert!(
                                        !src_idx.is_null(),
                                        "lane split over a whole-view register load"
                                    );
                                    let scale = kernel.insert_const_idx_before(walk_id, factor);
                                    let mul =
                                        kernel.insert_before(walk_id, Op::Binary { x: src_idx, y: scale, bop: BOp::Mul });
                                    let new_idx =
                                        kernel.insert_before(walk_id, Op::Binary { x: mul, y: unroll_id, bop: BOp::Add });
                                    kernel.ops[walk_id].op = Op::Load { src, src_idx: new_idx };
                                }
                                Op::Store { dst, dst_idx, src } if grows.contains(&dst) => {
                                    debug_assert!(
                                        !dst_idx.is_null(),
                                        "lane split over a whole-view register store"
                                    );
                                    let scale = kernel.insert_const_idx_before(walk_id, factor);
                                    let mul =
                                        kernel.insert_before(walk_id, Op::Binary { x: dst_idx, y: scale, bop: BOp::Mul });
                                    let new_idx =
                                        kernel.insert_before(walk_id, Op::Binary { x: mul, y: unroll_id, bop: BOp::Add });
                                    let new_src = if varies.contains(&src) {
                                        src
                                    } else {
                                        let dtype = kernel.dtype(src);
                                        let cast = kernel.insert_before(walk_id, Op::Cast { x: unroll_id, dtype });
                                        let zero = kernel.insert_before(walk_id, Op::Const(dtype.zero_constant()));
                                        let delta =
                                            kernel.insert_before(walk_id, Op::Binary { x: cast, y: zero, bop: BOp::Mul });
                                        kernel.insert_before(walk_id, Op::Binary { x: src, y: delta, bop: BOp::Add })
                                    };
                                    kernel.ops[walk_id].op = Op::Store { dst, dst_idx: new_idx, src: new_src };
                                }
                                Op::Copy { src, src_idx, dst, dst_idx } if grows.contains(&src) || grows.contains(&dst) => {
                                    let mut new_src_idx = src_idx;
                                    if !new_src_idx.is_null() {
                                        let scale = kernel.insert_const_idx_before(walk_id, factor);
                                        let mul = kernel.insert_before(
                                            walk_id,
                                            Op::Binary { x: new_src_idx, y: scale, bop: BOp::Mul },
                                        );
                                        new_src_idx =
                                            kernel.insert_before(walk_id, Op::Binary { x: mul, y: unroll_id, bop: BOp::Add });
                                    }
                                    let mut new_dst_idx = dst_idx;
                                    if !new_dst_idx.is_null() {
                                        let scale = kernel.insert_const_idx_before(walk_id, factor);
                                        let mul = kernel.insert_before(
                                            walk_id,
                                            Op::Binary { x: new_dst_idx, y: scale, bop: BOp::Mul },
                                        );
                                        new_dst_idx =
                                            kernel.insert_before(walk_id, Op::Binary { x: mul, y: unroll_id, bop: BOp::Add });
                                    }
                                    kernel.ops[walk_id].op = Op::Copy { src, src_idx: new_src_idx, dst, dst_idx: new_dst_idx };
                                }
                                _ => {}
                            }
                        }
                    }
            Op::Loop { len: len_id } => {
                // Reduce-loop lane split: [Loop(len/factor), Unroll
                // lanes]. The per-iteration value bundle folds over its
                // lane space at the accumulator store: `Op::Reduce`
                // over the lane axis, inserted before the chain binary,
                // which is rewritten in place to
                // `Binary { x: fold, y: carry, bop }` so its consumers
                // keep the identity. The accumulator keeps length 1.
                //
                // One fold per kernel: devectorize expands a single
                // fold's lanes; a second fold (nested or disjoint) has
                // no lowering. A later config in the same sequence that
                // would add one is skipped (the variant dedups away).
                if kernel.iter_unordered().any(|(_, op)| matches!(op, Op::Reduce { .. })) {
                    return;
                }
                let len = kernel.resolve_const(len_id).and_then(Constant::as_dim).unwrap();
                        let factor = Dim::from(factor);
                        debug_assert!(len % factor == 0 && len / factor >= 1, "loop length {len} not divisible by {factor}");

                        let new_len = kernel.insert_const_idx_before(site, len / factor);
                        let split_ids = kernel.split_dim(site, vec![Op::Loop { len: new_len }, Op::Unroll { axis, len: lanes }]);
                        let (loop_new, unroll_body) = (split_ids[0], split_ids[1]);

                        // The accumulator store: exactly one per body (the
                        // make scan validated it); its source is the chain
                        // binary mixing the per-iteration value with the
                        // carried accumulator load.
                        let store_id;
                        let mut op_id = kernel.next_op(loop_new);
                        loop {
                            match kernel.ops[op_id].op {
                                Op::EndLoop => unreachable!("opt_split validated the acc store"),
                                Op::Store { .. } => {
                                    store_id = op_id;
                                    break;
                                }
                                _ => {}
                            }
                            op_id = kernel.next_op(op_id);
                        }
                        let Op::Store { dst, dst_idx, src } = kernel.ops[store_id].op else {
                            unreachable!("opt_split validated the acc store")
                        };
                        debug_assert!(
                            kernel.resolve_const(dst_idx) == Some(Constant::new(0i64)),
                            "opt_split validated the acc store index"
                        );
                        debug_assert!(
                            matches!(kernel.at(dst), Op::Storage { scope: MemScope::Register, .. }),
                            "opt_split validated the acc storage"
                        );
                        let (x, y, bop) = match kernel.at(src) {
                            Op::Binary { x, y, bop } => (*x, *y, *bop),
                            _ => unreachable!("opt_split validated the chain binary"),
                        };

                        // Lane dependence of the two chain operands: the one
                        // transitively consuming the lane node is the
                        // per-iteration bundle; the other is the carry.
                        let mut x_stack: Vec<OpId> = vec![x];
                        let mut x_seen: Set<OpId> = Set::default();
                        let mut x_dep = false;
                        for _ in 0..10_000 {
                            let Some(id) = x_stack.pop() else { break };
                            if !x_seen.insert(id) {
                                continue;
                            }
                            if id == unroll_body {
                                x_dep = true;
                                break;
                            }
                            x_stack.extend(kernel.at(id).parameters());
                        }
                        let mut y_stack: Vec<OpId> = vec![y];
                        let mut y_seen: Set<OpId> = Set::default();
                        let mut y_dep = false;
                        for _ in 0..10_000 {
                            let Some(id) = y_stack.pop() else { break };
                            if !y_seen.insert(id) {
                                continue;
                            }
                            if id == unroll_body {
                                y_dep = true;
                                break;
                            }
                            y_stack.extend(kernel.at(id).parameters());
                        }
                        debug_assert!(x_dep != y_dep, "lane split chain binary operands misclassified");
                        let (bundle, carry) = if x_dep { (x, y) } else { (y, x) };

                        let fold = kernel.insert_before(src, Op::Reduce { x: bundle, rop: bop, reduce_axis: unroll_body });
                        kernel.ops[src].op = Op::Binary { x: fold, y: carry, bop };
                    }
                    _ => unreachable!("lane split applies to a group range or loop site"),
                }
            }
        }
        kernel.verify();
    }
}

impl Kernel {
    /// Make the split optimization: scan the kernel for splittable sites and
    /// emit one config per (site, factor, mode).
    ///
    /// Validity for lane-emitting modes is checked here so `apply` can assume
    /// a validated config; the boolean-loop gate for local splits and the
    /// lane factor caps live here for the same reason.
    pub fn opt_split(&self) -> Box<dyn Optimization> {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_split");
        let dev_info = self.dev_info();

        // Kernel-wide facts, one pass: boolean loops gate local splits;
        // tile-shaped accesses (rank > 1) and barriers refuse to share the
        // kernel with lane splits.
        let mut bool_loop = false;
        let mut tile_access = false;
        let mut barrier = false;
        let mut local_axis_sizes: crate::Map<u32, u32> = crate::Map::default();
        for (_, op) in self.iter_unordered() {
            match *op {
                Op::Loop { len } if self.dtype(len) == crate::DType::Bool => bool_loop = true,
                Op::Range { axis, kind: RangeKind::Local(len) } => {
                    if let Some(&existing) = local_axis_sizes.get(&axis) {
                        debug_assert_eq!(existing, len);
                    } else {
                        local_axis_sizes.insert(axis, len);
                    }
                }
                Op::Load { src_idx, .. } | Op::Store { dst_idx: src_idx, .. } | Op::Copy { src_idx, .. }
                    if !src_idx.is_null() && self.shape(src_idx).len() > 1 =>
                {
                    tile_access = true;
                }
                Op::Barrier => barrier = true,
                _ => {}
            }
        }
        let op_len: u64 = self.ops.len().0 as u64;

        let used_threads: u32 = local_axis_sizes.values().product();
        let remaining_threads = if local_axis_sizes.is_empty() {
            dev_info.max_local_threads
        } else {
            dev_info.max_local_threads / used_threads
        };

        let mut configs = Vec::new();

        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Range { axis, kind: RangeKind::Group(len) } => {
                    let Some(len) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) else {
                        op_id = self.next_op(op_id);
                        continue;
                    };
                    // Global -> local: hardware-aligned factors, bounded by the
                    // free local thread budget and the per-axis local limit.
                    if !bool_loop && !local_axis_sizes.contains_key(&axis) {
                        let max_per_axis = dev_info.max_local_work_dims[axis as usize];
                        let mut l_factors: Vec<u32> = vec![64, 32, 16, 8, 4, 2];
                        l_factors.retain(|&f| {
                            len % Dim::try_from(f).expect("lane factor overflows i64") == 0
                                && f <= remaining_threads
                                && f <= max_per_axis
                                && u64::from(f) * op_len <= 10_000
                        });
                        for f in l_factors {
                            configs.push((op_id, SplitMode::Local, f as u16));
                        }
                    }
                    // Global -> unroll lanes: vectorized per-thread loads and
                    // stores. Barriers and tile accesses refuse to coexist
                    // with a new lane axis; tile accesses would gain a third
                    // lane axis and barriers pin ordering around them.
                    if !tile_access && !barrier {
                        for f in [16u64, 8, 4] {
                            if len % Dim::try_from(f).expect("lane factor overflows i64") == 0
                                && len / Dim::try_from(f).expect("lane factor overflows i64") > 0
                                && f * op_len <= 10_000
                            {
                                configs.push((op_id, SplitMode::Unroll, f as u16));
                            }
                        }
                    }
                }
                Op::Loop { len } => {
                    if self.dtype(len) == crate::DType::Bool {
                        op_id = self.next_op(op_id);
                        continue;
                    }
                    let Some(len) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) else {
                        op_id = self.next_op(op_id);
                        continue;
                    };
                    if len < 2 {
                        op_id = self.next_op(op_id);
                        continue;
                    }
                    for f in [8u64, 16, 4, 2] {
                        if !self.reducible_loop(op_id, len, f) {
                            continue;
                        }
                        if f * op_len <= 10_000 {
                            configs.push((op_id, SplitMode::Unroll, f as u16));
                        }
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        Box::new(OptSplit { configs })
    }

    /// Validity check for a reduce-loop split config: the make scan proves
    /// the loop is a scalar-accumulator reduce the split can restructure.
    fn reducible_loop(&self, loop_id: OpId, len: Dim, factor: u64) -> bool {
        if len % Dim::try_from(factor).expect("split factor overflows i64") != 0 {
            return false;
        }

        // Find the register accumulator, walking backwards over the
        // loop-free prefix between the preceding nested loop and this loop.
        let mut op_id = self.prev_op(loop_id);
        let acc: OpId;
        loop {
            if op_id.is_null() {
                return false; // no register storage before the loop
            }
            match self.ops[op_id].op {
                Op::Loop { .. } => return false, // parent loop: body has nested loops
                Op::Storage { scope: MemScope::Register, .. } => {
                    acc = op_id;
                    break;
                }
                _ => {}
            }
            op_id = self.prev_op(op_id);
        }
        let acc_len = match self.at(acc) {
            Op::Storage { len, .. } => *len,
            _ => unreachable!(),
        };
        // The split restructures exactly the scalar accumulator case.
        if acc_len != 1 {
            return false;
        }

        // An init store for the accumulator (slot zero) must exist between
        // the declaration and the loop.
        let mut init_found = false;
        op_id = self.next_op(acc);
        while op_id != loop_id {
            match self.ops[op_id].op {
                Op::Store { dst, dst_idx, .. } if dst == acc => {
                    init_found = self.resolve_const(dst_idx) == Some(Constant::new(0i64));
                    break;
                }
                Op::Loop { .. } => return false,
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        if !init_found {
            return false;
        }

        // The body must be: no nested loops, no lane-folding ops, no storage
        // declaration, no barrier, exactly one store (to the accumulator at
        // slot zero, via a chain binary) and no other store or copy.
        let mut store_seen = false;
        op_id = self.next_op(loop_id);
        let mut depth = 1usize;
        while depth > 0 {
            if op_id.is_null() {
                return false;
            }
            match self.ops[op_id].op {
                Op::Loop { .. } => return false,
                Op::EndLoop => depth -= 1,
                Op::Storage { .. } | Op::Barrier => return false,
                Op::Stack { .. } | Op::Reduce { .. } | Op::Index { .. } => return false,
                Op::Copy { .. } => return false,
                Op::Store { dst, dst_idx, src } => {
                    match self.at(dst) {
                        Op::Storage { scope: MemScope::Register, .. } => {}
                        _ => return false,
                    }
                    if dst != acc {
                        return false;
                    }
                    if store_seen || self.resolve_const(dst_idx) != Some(Constant::new(0i64)) {
                        return false;
                    }
                    // The chain binary mixing the per-iteration value with
                    // the carried load: the lane fold rewrites it in place.
                    // Exactly one operand must consume the loop index: the
                    // per-iteration bundle. A loop-invariant chain gains
                    // nothing from lane splitting (the fold would combine
                    // identical values), and a doubly-variant chain has no
                    // clean carry — neither is a valid site for this
                    // transform.
                    let (cx, cy) = match self.at(src) {
                        Op::Binary { x, y, .. } => (*x, *y),
                        _ => return false,
                    };
                    let consumes = [cx, cy].map(|root| {
                        let mut seen = Set::default();
                        let mut stack = vec![root];
                        for _ in 0..10_000 {
                            let Some(id) = stack.pop() else { break };
                            if !seen.insert(id) {
                                continue;
                            }
                            if id == loop_id {
                                return true;
                            }
                            stack.extend(self.at(id).parameters());
                        }
                        false
                    });
                    if consumes[0] == consumes[1] {
                        return false;
                    }
                    store_seen = true;
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        store_seen
    }

    /// Splits dim (index or loop) into multiple indices or loops
    /// Returns the `OpId`s of the created split operations in the order they were provided
    pub(crate) fn split_dim(&mut self, dim_id: OpId, mut splits: Vec<Op>) -> Vec<OpId> {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("split_dim");
        let is_loop = matches!(self.ops[dim_id].op, Op::Loop { .. });

        #[cfg(debug_assertions)]
        {
            let mut dim = 1;
            let mut ok = true;
            for op in splits.iter() {
                use crate::shape::Dim;

                match *op {
                    Op::Loop { len, .. } => match self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) {
                        Some(l) => dim *= l,
                        None => {
                            ok = false;
                            break;
                        }
                    },
                    Op::Range { kind, .. } => match kind {
                        RangeKind::Group(len) => match self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) {
                            Some(l) => dim *= l,
                            None => {
                                ok = false;
                                break;
                            }
                        },
                        RangeKind::Local(len) => dim *= len as Dim,
                        // A warp is a view over a local range — it is never split.
                        RangeKind::Warp(_) => {
                            ok = false;
                            break;
                        }
                        RangeKind::Unroll(len) => dim *= len as Dim,
                    },
                    Op::Unroll { len, .. } => dim *= len as Dim,
                    _ => unreachable!("split can be only index or loop"),
                }
            }
            if ok {
                match self.ops[dim_id].op {
                    Op::Range { kind, .. } => {
                        use crate::shape::Dim;

                        match kind {
                            RangeKind::Group(len) => {
                                if let Some(l) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) {
                                    debug_assert_eq!(l, dim);
                                }
                            }
                            RangeKind::Local(l) => debug_assert_eq!(l as Dim, dim),
                            RangeKind::Warp(_) => unreachable!("warp dims are never split"),
                            RangeKind::Unroll(_) => unreachable!("unroll ranges are never split"),
                        }
                    }
                    Op::Loop { len, .. } => {
                        if let Some(l) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) {
                            debug_assert_eq!(l, dim);
                        }
                    }
                    _ => {}
                }
            }
        }

        let last_dim_op = self.get_last_dim_op(dim_id);
        let n_loops = splits.iter().filter(|op| matches!(op, Op::Loop { .. })).count();
        for (i, op) in splits.iter().enumerate() {
            if matches!(op, Op::Loop { .. }) {
                if is_loop && i == n_loops - 1 {
                } else {
                    self.insert_after(last_dim_op, Op::EndLoop);
                }
            }
        }

        let mut strides = Vec::new();
        let mut st = 1;
        for op in splits.iter().rev() {
            strides.push(st);
            match op {
                Op::Loop { len, .. } => st *= self.resolve_const(*len).and_then(crate::dtype::Constant::as_dim).unwrap(),
                Op::Range { kind, .. } => match kind {
                    RangeKind::Group(len) => st *= self.resolve_const(*len).and_then(crate::dtype::Constant::as_dim).unwrap(),
                    RangeKind::Local(len) => st *= i64::from(*len),
                    RangeKind::Warp(_) => unreachable!("warp dims are never split"),
                    RangeKind::Unroll(len) => st *= Dim::from(*len),
                },
                Op::Unroll { len, .. } => st *= Dim::from(*len),
                _ => unreachable!("split can be only index or loop"),
            }
        }
        strides.reverse();
        strides.pop();
        let last_op = splits.pop().unwrap();

        let mut split_ids: Vec<OpId> = Vec::new();
        let mut acc = self.insert_before(dim_id, Op::Const(Constant::idx(0)));
        for (&st, op) in strides.iter().zip(splits) {
            let x = self.insert_before(dim_id, Op::Const(Constant::idx(st)));
            let y = self.insert_before(dim_id, op);
            let xy = self.insert_before(dim_id, Op::Binary { x, y, bop: BOp::Mul });
            acc = self.insert_before(dim_id, Op::Binary { x: xy, y: acc, bop: BOp::Add });
            split_ids.push(y);
        }

        // The reduce-loop caller repairs the restructured accesses only
        // after the composition exists, so `split_dim` itself does not run
        // the shape verifier; each caller verifies.
        let y = self.insert_before(dim_id, last_op);
        split_ids.push(y);
        self.ops[dim_id].op = Op::Binary { x: acc, y, bop: BOp::Add };

        split_ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use crate::backend::{DTypeCapability, DeviceInfo};
    use crate::kernel::{Dev, MemScope, OpId};
    use std::sync::Arc;

    fn cpu() -> DeviceInfo {
        DeviceInfo {
            compute: 1_000_000,
            max_global_work_dims: vec![Dim::from(1_000_000_000); 3],
            max_local_threads: 1024,
            max_local_work_dims: vec![1024, 1024, 1024],
            preferred_vector_size: 8,
            local_mem_size: 8192,
            max_register_bytes: 256_000,
            tensor_cores: false,
            warp_size: 1,
            cc: [0, 0],
            dtype_capability: [DTypeCapability::all(); DType::N_DTYPES],
            has_native_exp2: false,
            supported_vec_lens: vec![2, 4, 8, 16],
            vector_depth: 1,
            tenstorrent: false,
            tile: [1, 1],
            tile_sizes: vec![],
            wmma_layouts: vec![],
            num_circular_buffers: 0,
            has_openmp: false,
        }
    }

    /// A reduce loop over 64 iterations (acc[0] += inp[acc-loop]) splits at
    /// factor 4: the loop becomes a 16-trip loop over 4 lanes and the lane
    /// partials fold at the accumulator store (`Op::Reduce` over the lane
    /// axis). The accumulator keeps length 1.
    #[test]
    fn split_reduce_loop_folds_lanes_at_acc() {
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let inp = k.param(DType::F32);
        let out = k.param_mut(DType::F32);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let init = k.const_val(0.0f32);
        k.store(acc, init, 0i64);
        let loop_id = {
            let mut lv = OpId::NULL;
            let acc_ref = acc;
            k.loop_over(64i64, |k, lv_id| {
                lv = lv_id;
                let x = k.load(inp, lv_id);
                let prev = k.load(acc_ref, 0i64);
                let sum = k.add(prev, x);
                k.store(acc_ref, sum, 0i64);
            });
            lv
        };
        let result = k.load(acc, 0i64);
        k.store(out, result, 0i64);

        // Apply the structural split via the optimization interface: one
        // reduce-loop lane config at factor 4.
        super::super::autotune::Optimization::apply(
            &OptSplit { configs: vec![(loop_id, SplitMode::Unroll, 4)] },
            &mut k,
            0,
        );
        k.verify();

        // The accumulator stays scalar.
        assert!(matches!(k.at(acc), Op::Storage { len, scope: MemScope::Register, .. } if *len == 1));
        // The store arm folds the lane bundle: an `Op::Reduce` over the
        // unroll axis, and the unroll node exists with length 4.
        let mut folds = 0;
        let mut unrolls = 0;
        for (_, op) in k.iter_unordered() {
            match op {
                Op::Reduce { reduce_axis, .. } => {
                    if matches!(k.at(*reduce_axis), Op::Unroll { len: 4, .. }) {
                        folds += 1;
                    }
                }
                Op::Unroll { len: 4, .. } => unrolls += 1,
                _ => {}
            }
        }
        assert_eq!(folds, 1);
        assert!(unrolls >= 1);
    }

    /// A group range whose lane-dependent value flows into a scalar register
    /// accumulator: splitting by unroll grows the accumulator to one slot
    /// per lane (coarsen's slotting), re-indexes its accesses, and verifies.
    #[test]
    fn split_group_unroll_grows_acc() {
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let inp = k.param(DType::F32);
        let out = k.param_mut(DType::F32);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let g = k.group_range(0, 64i64);
        let x = k.load(inp, g);
        let prev = k.load(acc, 0i64);
        let sum = k.add(x, prev);
        k.store(acc, sum, 0i64);
        let result = k.load(acc, 0i64);
        k.store(out, result, g);

        // Apply the structural split via the optimization interface: one
        // group lane config at factor 8.
        super::super::autotune::Optimization::apply(
            &OptSplit { configs: vec![(g, SplitMode::Unroll, 8)] },
            &mut k,
            0,
        );
        k.verify();

        // The accumulator grew to one slot per lane.
        assert!(matches!(k.at(acc), Op::Storage { len, scope: MemScope::Register, .. } if *len == 8));
        let mut unrolls = 0;
        for (_, op) in k.iter_unordered() {
            match op {
                Op::Unroll { len: 8, .. } => unrolls += 1,
                _ => {}
            }
        }
        assert!(unrolls >= 1);
    }

    /// A group range split into local threads composes index = group*8+local.
    #[test]
    fn split_global_to_local_composes_index() {
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let inp = k.param(DType::F32);
        let out = k.param_mut(DType::F32);
        let g = k.group_range(0, 64i64);
        let x = k.load(inp, g);
        k.store(out, x, g);

        // Apply the structural split via the optimization interface: one
        // local config at factor 8.
        super::super::autotune::Optimization::apply(
            &OptSplit { configs: vec![(g, SplitMode::Local, 8)] },
            &mut k,
            0,
        );
        k.verify();

        let mut groups = 0;
        let mut locals = 0;
        for (id, op) in k.iter_unordered() {
            match op {
                Op::Range { kind: RangeKind::Group(len), .. } => {
                    assert_ne!(id, g, "the original range was composed away");
                    assert_eq!(k.resolve_const(*len).and_then(crate::dtype::Constant::as_dim), Some(8));
                    groups += 1;
                }
                Op::Range { kind: RangeKind::Local(8), .. } => locals += 1,
                _ => {}
            }
        }
        assert_eq!(groups, 1);
        assert_eq!(locals, 1);
    }
}
