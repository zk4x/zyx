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
    dtype::Constant,
    kernel::{BOp, Kernel, MemScope, Op, OpId, RangeKind},
    shape::Dim,
};

/// The kind of the split's secondary part. The site kind (a [`Op::Range`]
/// group or an [`Op::Loop`]) and this variant together decide the split:
/// `Local` splits a group into a smaller group plus local threads;
/// `Unroll` splits a group or a reduce loop into a smaller part plus unroll
/// lanes (a reduce loop additionally restructures its scalar accumulator
/// into a lane register array); `Loop` splits a loop into two nested loops.
#[derive(Debug)]
pub enum SplitMode {
    /// Secondary part is a local thread range.
    Local,
    /// Secondary part is unroll lanes; the lane axis is the next free one
    /// at apply time.
    Unroll,
    /// Secondary part is a smaller inner loop.
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
                        let len = kernel.resolve_const(len_id).and_then(Constant::as_dim).unwrap();
                        let factor = Dim::from(factor);
                        debug_assert!(len % factor == 0, "group length {len} not divisible by {factor}");
                        let group_len = kernel.insert_const_idx_before(site, len / factor);
                        kernel.split_dim(
                            site,
                            vec![
                                Op::Range { axis: group_axis, kind: RangeKind::Group(group_len) },
                                Op::Unroll { axis, len: lanes },
                            ],
                        );
                        kernel.verify();
                    }
                    Op::Loop { len: len_id } => {
                        // Reduce-loop lane split: the accumulator restructures
                        // into a lane register array and a post-loop VecReduce
                        // folds the per-lane partials back into the scalar
                        // every downstream consumer expected.
                        let len = kernel.resolve_const(len_id).and_then(Constant::as_dim).unwrap();
                        let factor = Dim::from(factor);
                        debug_assert!(len % factor == 0 && len / factor >= 1, "loop length {len} not divisible by {factor}");

                        // The accumulator: validated by the make scan.
                        let mut op_id = kernel.prev_op(site);
                        let acc_id;
                        loop {
                            if op_id.is_null() {
                                unreachable!("opt_split validated the accumulator");
                            }
                            match kernel.ops[op_id].op {
                                Op::Storage { scope: MemScope::Register, .. } => {
                                    acc_id = op_id;
                                    break;
                                }
                                _ => {}
                            }
                            op_id = kernel.prev_op(op_id);
                        }
                        let (acc_dtype, scope, acc_len) = match kernel.at(acc_id) {
                            Op::Storage { dtype, scope, len } => (*dtype, *scope, *len),
                            _ => unreachable!("lane split applies to a storage site"),
                        };
                        debug_assert_eq!(acc_len, 1, "opt_split validated a scalar accumulator");
                        // The accumulator's length grows by the lane factor;
                        // per-lane slot blocks follow `[len, lanes]` row-major
                        // with stride `len`.
                        kernel.ops[acc_id].op = Op::Storage { dtype: acc_dtype, scope, len: acc_len * factor };

                        // Init store before the loop: broadcast the scalar init
                        // over the grown slot array. A zero-lane-delta lane tree
                        // gives exactly the init value in each slot; the
                        // broadcast rides the shape algebra.
                        op_id = kernel.next_op(acc_id);
                        let init_store;
                        loop {
                            match kernel.ops[op_id].op {
                                Op::Store { dst, dst_idx, src } if dst == acc_id => {
                                    debug_assert_eq!(
                                        kernel.resolve_const(dst_idx),
                                        Some(Constant::new(0i64)),
                                        "opt_split validated the acc init store"
                                    );
                                    init_store = (op_id, src);
                                    break;
                                }
                                Op::Loop { .. } => unreachable!("opt_split validated the acc init store"),
                                _ => {}
                            }
                            op_id = kernel.next_op(op_id);
                        }
                        let lane_idx = kernel.insert_before(init_store.0, Op::Unroll { axis, len: lanes });
                        let lane_cast = kernel.insert_before(init_store.0, Op::Cast { x: lane_idx, dtype: acc_dtype });
                        let zero = kernel.insert_before(init_store.0, Op::Const(acc_dtype.zero_constant()));
                        let lane_delta = kernel.insert_before(init_store.0, Op::Binary { x: lane_cast, y: zero, bop: BOp::Mul });
                        let broadcast_init =
                            kernel.insert_before(init_store.0, Op::Binary { x: init_store.1, y: lane_delta, bop: BOp::Add });
                        kernel.ops[init_store.0].op = Op::Store { dst: acc_id, dst_idx: lane_idx, src: broadcast_init };

                        // Split the loop: [smaller loop, unroll lanes]; the
                        // composed loop variable rides `split_dim` and the body
                        // Unroll node it creates requires in-order scope for the
                        // body rewrites below.
                        let new_len = kernel.insert_const_idx_before(site, len / factor);
                        let split_ids = kernel.split_dim(site, vec![Op::Loop { len: new_len }, Op::Unroll { axis, len: lanes }]);
                        let (loop_new, unroll_body) = (split_ids[0], split_ids[1]);

                        // Body: the accumulator loads read their lane's slot and
                        // the store writes it; everything else flows through the
                        // shape algebra.
                        let endloop_id;
                        op_id = kernel.next_op(loop_new);
                        loop {
                            match kernel.ops[op_id].op {
                                Op::EndLoop => {
                                    endloop_id = op_id;
                                    break;
                                }
                                Op::Load { src, src_idx } if src == acc_id => {
                                    debug_assert_eq!(
                                        kernel.resolve_const(src_idx),
                                        Some(Constant::new(0i64)),
                                        "opt_split validated the acc chain load"
                                    );
                                    kernel.ops[op_id].op = Op::Load { src: acc_id, src_idx: unroll_body };
                                }
                                Op::Store { dst, dst_idx, src } if dst == acc_id => {
                                    debug_assert_eq!(
                                        kernel.resolve_const(dst_idx),
                                        Some(Constant::new(0i64)),
                                        "opt_split validated the acc store"
                                    );
                                    kernel.ops[op_id].op = Op::Store { dst: acc_id, dst_idx: unroll_body, src };
                                }
                                _ => {}
                            }
                            op_id = kernel.next_op(op_id);
                        }

                        // Post-loop: fold the per-lane partials back into one
                        // scalar. The downstream consumers of the old slot-zero
                        // loads now see the VecReduce of the whole slot array.
                        let post_lane = kernel.insert_after(endloop_id, Op::Unroll { axis, len: lanes });
                        let partial_load = kernel.insert_after(post_lane, Op::Load { src: acc_id, src_idx: post_lane });
                        op_id = kernel.next_op(partial_load);
                        while !op_id.is_null() {
                            match kernel.ops[op_id].op {
                                Op::Load { src, src_idx } if src == acc_id => {
                                    debug_assert_eq!(
                                        kernel.resolve_const(src_idx),
                                        Some(Constant::new(0i64)),
                                        "post-loop acc load reads slot zero"
                                    );
                                    kernel.ops[op_id].op = Op::VecReduce { x: partial_load, bop: BOp::Add };
                                }
                                _ => {}
                            }
                            op_id = kernel.next_op(op_id);
                        }
                        kernel.verify();
                    }
                    _ => unreachable!("lane split applies to a group range or loop site"),
                }
            }
            SplitMode::Loop => {
                let Op::Loop { len: len_id } = kernel.ops[site].op else {
                    unreachable!("loop split applies to a loop site");
                };
                let len = kernel.resolve_const(len_id).and_then(Constant::as_dim).unwrap();
                let factor = Dim::from(factor);
                debug_assert!(len % factor == 0 && factor >= 1, "loop length {len} not divisible by {factor}");
                let new_len = kernel.insert_const_idx_before(site, len / factor);
                let inner_len = kernel.insert_const_idx_before(site, factor);
                kernel.split_dim(site, vec![Op::Loop { len: new_len }, Op::Loop { len: inner_len }]);
                kernel.verify();
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
        // slot zero) and no other store or copy.
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
                Op::VecReduce { .. } | Op::Stack { .. } | Op::Reduce { .. } | Op::Index { .. } => return false,
                Op::Copy { .. } => return false,
                Op::Store { dst, dst_idx, .. } => {
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
    /// factor 4: the acc grows to 4 slots, the store and chain load read
    /// their lane's slot and the post-loop consumer becomes a VecReduce.
    #[test]
    fn split_reduce_loop_restructures_acc() {
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

        // The accumulator grew to the lane factor.
        assert!(matches!(k.at(acc), Op::Storage { len, scope: MemScope::Register, .. } if *len == 4));
        // A post-loop VecReduce folds the lane partials.
        let mut vec_reduces = 0;
        let mut unrolls = 0;
        for (_, op) in k.iter_unordered() {
            match op {
                Op::VecReduce { x, bop: BOp::Add } => {
                    if matches!(k.at(*x), Op::Load { src, .. } if *src == acc) {
                        vec_reduces += 1;
                    }
                }
                Op::Unroll { axis: 0, len: 4 } => unrolls += 1,
                _ => {}
            }
        }
        assert_eq!(vec_reduces, 1);
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
