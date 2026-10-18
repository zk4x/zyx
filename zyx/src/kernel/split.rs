// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Split optimization.
//!
//! Splits dimensions of the index space: a global group range into a smaller
//! group plus local threads ([`SplitMode::Local`]), a global group range into
//! a smaller part plus unroll lanes or a reduce loop unrolled per element
//! ([`SplitMode::Unroll`]), and a loop into two nested loops
//! ([`SplitMode::Loop`]).
//!
//! Every split composes the split index as the affine combination of
//! the parts inline in [`Kernel::split`], so the IR stays linear in op
//! count per variant: lane growth happens only when devectorization
//! desugars the chosen variant.

use super::autotune::Optimization;
use crate::{
    Set,
    dtype::Constant,
    kernel::{BOp, Kernel, MemScope, Op, OpId, RangeKind},
    shape::Dim,
};

/// The kind of the split's secondary part. The site kind (a [`Op::Range`]
/// group or an [`Op::Loop`]) and this variant together decide the split:
/// `Local` splits a group into a smaller group plus local threads;
/// `Unroll` splits a group into a smaller part plus unroll lanes, or
/// unrolls a reduce loop per element with the per-element values chained
/// into the accumulator store. A group split grows only the register
/// storages that become lane-dependent (coarsen's slotting).
#[derive(Debug, Clone, Copy)]
pub enum SplitMode {
    /// Secondary part is a local thread range.
    Local,
    /// Secondary part is unroll lanes; the lane axis is the next free one
    /// at apply time.
    Unroll,
    /// Secondary part is a loop nest; the new loop is inserted right after
    /// the split site and its end lands at the kernel tail.
    Loop,
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
        let &(axis_id, mode, factor) = &self.configs[config as usize];
        kernel.split(axis_id, mode, factor);
        kernel.verify();
    }
}

impl Kernel {
    /// Single split of one axis into `split_mode` by `factor`.
    ///
    /// Returns the new secondary node (`Local`/`Unroll`/`Loop`);
    /// `OpId::NULL` when the config cannot apply. All validity gates
    /// live here so `OptSplit::apply` stays a straight call.
    pub(crate) fn split(&mut self, axis_id: OpId, split_mode: SplitMode, factor: u16) -> OpId {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("split");
        let factor_dim = Dim::from(factor);
        match split_mode {
            SplitMode::Local => {
                let Op::Range { axis, kind: RangeKind::Group(len_id) } = self.ops[axis_id].op else {
                    return OpId::NULL;
                };
                if !self.is_sym_multiple(len_id, factor_dim) {
                    return OpId::NULL;
                }
                let Some(len) = self.resolve_const(len_id).and_then(Constant::as_dim) else {
                    return OpId::NULL;
                };
                let group_len = self.insert_const_idx_before(axis_id, len / factor_dim);
                // Index composition for exactly two parts: group * factor + local.
                let zero = self.insert_before(axis_id, Op::Const(Constant::idx(0)));
                let scale = self.insert_before(axis_id, Op::Const(Constant::idx(factor_dim)));
                let group_id = self.insert_before(axis_id, Op::Range { axis, kind: RangeKind::Group(group_len) });
                let scaled = self.insert_before(axis_id, Op::Binary { x: scale, y: group_id, bop: BOp::Mul });
                let acc = self.insert_before(axis_id, Op::Binary { x: scaled, y: zero, bop: BOp::Add });
                let local_id = self.insert_before(axis_id, Op::Range { axis, kind: RangeKind::Local(u32::from(factor)) });

                // The accesses are repaired by the caller only after the
                // composition exists, so the composition itself does not run
                // the shape verifier; the caller verifies.
                self.ops[axis_id].op = Op::Binary { x: acc, y: local_id, bop: BOp::Add };
                local_id
            }
            SplitMode::Unroll => {
                // The lane axis is the next free one; nothing runs between
                // the make scan and apply, so the state matches make's view.
                let mut next_lane_axis = 0u64;
                for (_, op) in self.iter_unordered() {
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
                match self.ops[axis_id].op {
                    Op::Range { axis: group_axis, kind: RangeKind::Group(len_id) } => {
                        // Global -> unroll lanes. Soundness validation first,
                        // no mutation yet: coarsen's gates rooted at the range
                        // whose lanes the split introduces. Whole-view writes
                        // (dst is an SSA value) and shared-memory effects a
                        // lane axis cannot slot refuse the split silently,
                        // mirroring `coarsen`'s refusal contract.
                        let len = self.resolve_const(len_id).and_then(Constant::as_dim).unwrap();
                        let factor = Dim::from(factor);

                        let mut varies: Set<OpId> = Set::default();
                        varies.insert(axis_id);
                        for _ in 0..10_000 {
                            let mut grown = false;
                            for (op, _) in self.iter_unordered() {
                                if !varies.contains(&op) && self.at(op).parameters().any(|p| varies.contains(&p)) {
                                    varies.insert(op);
                                    grown = true;
                                }
                            }
                            if !grown {
                                break;
                            }
                        }
                        let mut valid = true;
                        for (_, op) in self.iter_unordered() {
                            match *op {
                                Op::Store { dst, dst_idx, src, .. } => {
                                    let addr_varies = !dst_idx.is_null() && varies.contains(&dst_idx);
                                    let val_varies = varies.contains(&src);
                                    let base = self.at(dst);
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
                                    let base = self.at(dst);
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
                            return OpId::NULL;
                        }

                        // Split the group range: [Group(len/factor), Unroll
                        // lanes]; the composed index rides the inline affine
                        // composition and its
                        // consumers become lane-dependent through the shape
                        // algebra.
                        let group_len = self.insert_const_idx_before(axis_id, len / factor_dim);
                        let mut splits = vec![
                            Op::Range { axis: group_axis, kind: RangeKind::Group(group_len) },
                            Op::Unroll { axis, len: lanes },
                        ];
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
                        let mut acc = self.insert_before(axis_id, Op::Const(Constant::idx(0)));
                        for (&st, op) in strides.iter().zip(splits) {
                            let x = self.insert_before(axis_id, Op::Const(Constant::idx(st)));
                            let y = self.insert_before(axis_id, op);
                            let xy = self.insert_before(axis_id, Op::Binary { x, y, bop: BOp::Mul });
                            acc = self.insert_before(axis_id, Op::Binary { x: xy, y: acc, bop: BOp::Add });
                            split_ids.push(y);
                        }

                        // The accesses are repaired by the caller only after the
                        // composition exists, so the composition itself does not run
                        // the shape verifier; the caller verifies.
                        let y = self.insert_before(axis_id, last_op);
                        split_ids.push(y);
                        self.ops[axis_id].op = Op::Binary { x: acc, y, bop: BOp::Add };
                        let unroll_id = split_ids[1];

                        // Lane-dependent ops: everything transitively consuming
                        // the unroll node.
                        let mut varies: Set<OpId> = Set::default();
                        varies.insert(unroll_id);
                        for _ in 0..10_000 {
                            let mut grown = false;
                            for (op, _) in self.iter_unordered() {
                                if !varies.contains(&op) && self.at(op).parameters().any(|p| varies.contains(&p)) {
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
                        let op_ids: Vec<OpId> = self.ops.ids().collect();
                        let mut grows: Set<OpId> = Set::default();
                        for &walk_id in &op_ids {
                            match self.at(walk_id) {
                                Op::Store { dst, dst_idx, src, .. } => {
                                    let is_register =
                                        matches!(self.at(*dst), Op::Storage { scope: MemScope::Register, .. });
                                    let lane_dep = varies.contains(src)
                                        || (!dst_idx.is_null() && varies.contains(dst_idx))
                                        || grows.contains(dst);
                                    if is_register && lane_dep {
                                        grows.insert(*dst);
                                    }
                                }
                                Op::Load { src, src_idx, .. } => {
                                    let is_register =
                                        matches!(self.at(*src), Op::Storage { scope: MemScope::Register, .. });
                                    let lane_dep =
                                        (!src_idx.is_null() && varies.contains(src_idx)) || grows.contains(src);
                                    if is_register && lane_dep {
                                        grows.insert(*src);
                                    }
                                }
                                Op::Copy { dst, dst_idx, .. } => {
                                    let is_register =
                                        matches!(self.at(*dst), Op::Storage { scope: MemScope::Register, .. });
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
                            match self.at(growth_id) {
                                Op::Storage { dtype, scope, len, .. } => {
                                    let (dtype, scope, len) = (*dtype, *scope, *len);
                                    self.ops[growth_id].op = Op::Storage { dtype, scope, len: len * factor };
                                }
                                _ => unreachable!("lane split grows a storage"),
                            }
                        }
                        // Re-index accesses to grown storages: every address
                        // becomes `idx*factor + lane` (coarsen's slot blocks),
                        // and lane-invariant stored values broadcast across
                        // the slot block as `value + 0*lane`.
                        for &walk_id in &op_ids {
                            match self.ops[walk_id].op {
                                Op::Load { src, src_idx } if grows.contains(&src) => {
                                    let scale = self.insert_const_idx_before(walk_id, factor);
                                    let mul =
                                        self.insert_before(walk_id, Op::Binary { x: src_idx, y: scale, bop: BOp::Mul });
                                    let new_idx =
                                        self.insert_before(walk_id, Op::Binary { x: mul, y: unroll_id, bop: BOp::Add });
                                    self.ops[walk_id].op = Op::Load { src, src_idx: new_idx };
                                }
                                Op::Store { dst, dst_idx, src } if grows.contains(&dst) => {
                                    let scale = self.insert_const_idx_before(walk_id, factor);
                                    let mul =
                                        self.insert_before(walk_id, Op::Binary { x: dst_idx, y: scale, bop: BOp::Mul });
                                    let new_idx =
                                        self.insert_before(walk_id, Op::Binary { x: mul, y: unroll_id, bop: BOp::Add });
                                    self.ops[walk_id].op = Op::Store { dst, dst_idx: new_idx, src };
                                }
                                Op::Copy { src, src_idx, dst, dst_idx } if grows.contains(&src) || grows.contains(&dst) => {
                                    let mut new_src_idx = src_idx;
                                    if !new_src_idx.is_null() {
                                        let scale = self.insert_const_idx_before(walk_id, factor);
                                        let mul = self.insert_before(
                                            walk_id,
                                            Op::Binary { x: new_src_idx, y: scale, bop: BOp::Mul },
                                        );
                                        new_src_idx =
                                            self.insert_before(walk_id, Op::Binary { x: mul, y: unroll_id, bop: BOp::Add });
                                    }
                                    let mut new_dst_idx = dst_idx;
                                    if !new_dst_idx.is_null() {
                                        let scale = self.insert_const_idx_before(walk_id, factor);
                                        let mul = self.insert_before(
                                            walk_id,
                                            Op::Binary { x: new_dst_idx, y: scale, bop: BOp::Mul },
                                        );
                                        new_dst_idx =
                                            self.insert_before(walk_id, Op::Binary { x: mul, y: unroll_id, bop: BOp::Add });
                                    }
                                    self.ops[walk_id].op = Op::Copy { src, src_idx: new_src_idx, dst, dst_idx: new_dst_idx };
                                }
                                _ => {}
                            }
                        }
                        return unroll_id;
                    }
                    Op::Loop { len: len_id } => {
                        // Loop unroll: strip-mine into Loop(len/factor) and
                        // duplicate the body per element, chaining the
                        // per-element values into the acc store. No Reduce
                        // op (unroll_tree_reduce mechanics, inline).
                        if factor_dim < 2 || !self.is_sym_multiple(len_id, factor_dim) {
                            return OpId::NULL;
                        }
                        let Some(len) = self.resolve_const(len_id).and_then(Constant::as_dim) else {
                            return OpId::NULL;
                        };

                        // Find the acc.
                        let acc_id;
                        let mut op_id = self.prev_op(axis_id);
                        loop {
                            if op_id.is_null() {
                                return OpId::NULL; // not a reduce, just a loop
                            }
                            match self.ops[op_id].op {
                                Op::Loop { .. } => return OpId::NULL, // nested reduce or no reduce
                                Op::Storage { scope: MemScope::Register, .. } => {
                                    acc_id = op_id;
                                    break;
                                }
                                _ => {}
                            }
                            op_id = self.prev_op(op_id);
                        }

                        // Find the store to acc.
                        let mut op_id = acc_id;
                        let acc_init;
                        loop {
                            if let Op::Store { dst, src: x, .. } = self.ops[op_id].op
                                && dst == acc_id
                            {
                                acc_init = x;
                                break;
                            }
                            op_id = self.next_op(op_id);
                            if op_id == axis_id {
                                unreachable!();
                            }
                        }

                        let mut has_store = false;
                        let mut op_id = self.next_op(axis_id);
                        let endloop_id;
                        loop {
                            match self.ops[op_id].op {
                                Op::Loop { .. } => return OpId::NULL, // no nested loops
                                Op::Store { dst, .. } => {
                                    if has_store {
                                        return OpId::NULL;
                                    }
                                    if !matches!(self.ops[dst].op, Op::Storage { scope: MemScope::Register, .. }) {
                                        return OpId::NULL;
                                    }
                                    has_store = true;
                                }
                                Op::Copy { src, dst, .. } => {
                                    if has_store
                                        || !matches!(self.ops[src].op, Op::Storage { scope: MemScope::Register, .. })
                                        || !matches!(self.ops[dst].op, Op::Storage { scope: MemScope::Register, .. })
                                    {
                                        return OpId::NULL;
                                    }
                                    has_store = true;
                                }
                                Op::EndLoop => {
                                    endloop_id = op_id;
                                    break;
                                }
                                _ => {}
                            }
                            op_id = self.next_op(op_id);
                        }

                        let mut map: crate::Map<OpId, Vec<OpId>> = crate::Map::default();

                        let new_len = self.insert_const_idx_before(axis_id, len / factor_dim);
                        let new_loop = self.insert_before(axis_id, Op::Loop { len: new_len });
                        let mut op_id = self.next_op(axis_id);
                        let stride = self.insert_before(axis_id, Op::Const(Constant::idx(factor_dim)));
                        self.ops[axis_id].op = Op::Binary { x: new_loop, y: stride, bop: BOp::Mul };
                        let mut new_ones = Vec::with_capacity(factor_dim as usize - 1);
                        for i in 1..factor_dim {
                            let offset = self.insert_before(op_id, Op::Const(Constant::idx(i)));
                            let new_id = self.insert_before(op_id, Op::Binary { x: axis_id, y: offset, bop: BOp::Add });
                            new_ones.push(new_id);
                        }
                        map.insert(axis_id, new_ones);

                        while op_id != endloop_id {
                            let this_id = op_id;
                            op_id = self.next_op(op_id);
                            if let Op::Load { src, .. } = self.ops[this_id].op
                                && src == acc_id
                            {
                                map.insert(this_id, vec![acc_init; factor_dim as usize - 1]);
                            } else if let Op::Store { dst, dst_idx, src: x, .. } = self.ops[this_id].op
                                && dst == acc_id
                            {
                                let Op::Binary { bop, .. } = self.ops[x].op else {
                                    unreachable!()
                                };
                                let y = if let Some(mapping) = map.get(&x) { mapping[0] } else { x };
                                let mut carry = this_id;
                                self.ops[this_id].op = Op::Binary { x, y, bop };
                                for i in 1..factor_dim - 1 {
                                    let x = if let Some(mapping) = map.get(&x) {
                                        mapping[i as usize]
                                    } else {
                                        x
                                    };
                                    carry = self.insert_before(op_id, Op::Binary { x, y: carry, bop });
                                }
                                self.insert_before(op_id, Op::Store { dst, dst_idx, src: carry });
                            } else {
                                let mut new_ones = Vec::with_capacity(factor_dim as usize - 1);
                                for i in 1..factor_dim {
                                    let mut new_op = self.ops[this_id].op.clone();
                                    for param in new_op.parameters_mut() {
                                        if let Some(mapping) = map.get(param) {
                                            *param = mapping[i as usize - 1];
                                        }
                                    }
                                    let new_id = self.insert_before(op_id, new_op);
                                    new_ones.push(new_id);
                                }
                                map.insert(this_id, new_ones);
                            }
                        }

                        new_loop
                    }
                    _ => return OpId::NULL,
                }
            }
            SplitMode::Loop => {
                // Strip-mine into a nested loop: the new loop goes right
                // after the split site, its end lands at the kernel tail.
                // No fold: plain nesting needs no lane combine.
                let (outer_len_id, is_loop) = match self.ops[axis_id].op {
                    Op::Range { kind: RangeKind::Group(len_id), .. } => (len_id, false),
                    Op::Loop { len: len_id } => (len_id, true),
                    _ => return OpId::NULL,
                };
                if !self.is_sym_multiple(outer_len_id, factor_dim) {
                    return OpId::NULL;
                }
                let Some(len) = self.resolve_const(outer_len_id).and_then(Constant::as_dim) else {
                    return OpId::NULL;
                };
                let outer_len = self.insert_const_idx_before(axis_id, len / factor_dim);
                let outer = if is_loop {
                    Op::Loop { len: outer_len }
                } else {
                    let Op::Range { axis, .. } = self.ops[axis_id].op else {
                        return OpId::NULL;
                    };
                    Op::Range { axis, kind: RangeKind::Group(outer_len) }
                };
                let inner_len = self.insert_const_idx_before(axis_id, factor_dim);
                // A split loop closes at the original loop's end; a split
                // range extends to the kernel tail.
                if is_loop {
                    self.insert_after(self.get_last_dim_op(axis_id), Op::EndLoop);
                }
                // Index composition for exactly two parts: outer * factor + inner.
                let zero = self.insert_before(axis_id, Op::Const(Constant::idx(0)));
                let scale = self.insert_before(axis_id, Op::Const(Constant::idx(factor_dim)));
                let outer_id = self.insert_before(axis_id, outer);
                let scaled = self.insert_before(axis_id, Op::Binary { x: scale, y: outer_id, bop: BOp::Mul });
                let acc = self.insert_before(axis_id, Op::Binary { x: scaled, y: zero, bop: BOp::Add });
                let inner_id = self.insert_before(axis_id, Op::Loop { len: inner_len });

                // The accesses are repaired by the caller only after the
                // composition exists, so the composition itself does not run
                // the shape verifier; the caller verifies.
                self.ops[axis_id].op = Op::Binary { x: acc, y: inner_id, bop: BOp::Add };
                if !is_loop {
                    self.push_back(Op::EndLoop);
                }
                inner_id
            }
        }
    }
}

impl Kernel {
    /// Make the split optimization: scan the kernel for splittable axis_ids and
    /// emit one config per (axis_id, factor, mode).
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
                    if local_axis_sizes.get(&axis).is_none() {
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
                        if len % f as Dim != 0 {
                            continue;
                        }
                        if f * op_len <= 10_000 {
                            configs.push((op_id, SplitMode::Unroll, f as u16));
                        }
                    }
                    // Plain nested loops (the deleted `SplitLoop`): large
                    // loops strip-mine without lanes or folds.
                    if len >= 16 {
                        for f in [8u64, 16, 4, 2] {
                            if len % f as Dim == 0 {
                                configs.push((op_id, SplitMode::Loop, f as u16));
                            }
                        }
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        Box::new(OptSplit { configs })
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

    /// A reduce loop over 64 iterations splits at factor 4: the loop
    /// becomes a 16-trip loop, the body duplicates per element, and the
    /// per-element values chain into the accumulator store. No `Op::Reduce`,
    /// no `Op::Unroll`; the accumulator keeps length 1.
    #[test]
    fn split_reduce_loop_unrolls_into_tree() {
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
        // reduce-loop unroll config at factor 4.
        super::super::autotune::Optimization::apply(
            &OptSplit { configs: vec![(loop_id, SplitMode::Unroll, 4)] },
            &mut k,
            0,
        );
        k.verify();

        // The accumulator stays scalar.
        assert!(matches!(k.at(acc), Op::Storage { len, scope: MemScope::Register, .. } if *len == 1));
        // No fold, no unroll node: the body is duplicated per element and
        // chained into the acc store.
        for (_, op) in k.iter_unordered() {
            assert!(!matches!(op, Op::Reduce { .. } | Op::Unroll { .. }));
        }
        // The loop strip-mined to 16 trips.
        let mut sixteens = 0;
        for (_, op) in k.iter_unordered() {
            if let Op::Loop { len } = op
                && k.resolve_const(*len) == Some(crate::dtype::Constant::new(16i64))
            {
                sixteens += 1;
            }
        }
        assert_eq!(sixteens, 1);
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
