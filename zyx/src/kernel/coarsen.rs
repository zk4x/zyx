// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Thread coarsening and register blocking optimization.
//!
//! This module provides thread coarsening and register blocking
//! optimizations for kernels, which group thread operations to improve
//! performance.
//!
//! Thread coarsening is useful for:
//!
//! - Reducing thread synchronization overhead
//! - Enabling better instruction scheduling
//! - Improving register utilization
//! - Coalescing memory access patterns
//!
//! Register blocking applies to tiled reductions, blocking operations
//! for register tiling to improve performance.

#![allow(clippy::while_let_loop)]

use std::collections::BTreeMap;

use super::autotune::Optimization;
use crate::{
    dtype::Constant,
    kernel::{BOp, Kernel, MemScope, Op, OpId, RangeKind},
    shape::Dim,
};

/// Coarsen thread-level parallelism by grouping operations.
#[derive(Debug)]
pub struct ThreadCoarse {
    /// Pairs of (operation_id, coarsening_factor) for each group.
    pub factors: Vec<(OpId, u64)>,
}

impl Optimization for ThreadCoarse {
    fn nconfigs(&self) -> u64 {
        self.factors.len() as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        if self.factors.is_empty() {
            return;
        }
        let (op_id, factor) = self.factors[config as usize];
        kernel.coarsen(op_id, factor);
    }
}

/// Register blocking optimization for tiled reductions.
#[derive(Debug)]
pub struct RegisterBlocking {
    /// Reduction splits mapped to tile sizes.
    pub reduce_splits: BTreeMap<OpId, Vec<u64>>,
    /// Thread coarsening factors for each global axis.
    pub thread_coarses: BTreeMap<OpId, Vec<u64>>,
}

impl Optimization for RegisterBlocking {
    fn nconfigs(&self) -> u64 {
        if self.reduce_splits.is_empty() || self.thread_coarses.is_empty() {
            return 0;
        }
        let n_global_options: usize = self.thread_coarses.values().map(|v| v.len() + 1).product();
        let n_reduce_options: usize = self.reduce_splits.values().map(Vec::len).product();
        (n_global_options * n_reduce_options) as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        kernel.apply_register_blocking(&self.reduce_splits, &self.thread_coarses, config as usize);
    }
}

// ## Coalesced local+upcast access
//
// After `opt_split_global_to_local` and `upcast` splits axis 0 (M) and `opt_upcast`
// vectorizes, the address expression for the fine M-dimension is:
//
// ```text
// addr = gidx0 * TILE + lidx0 * V + i      (i = 0..V-1)
// ```
//
// where `lidx0 = threadIdx.x` and `V` is the vector width (e.g. 8).
//
// **Problem:** Consecutive threads (lidx0 = 0, 1, 2, ...) access elements
// spaced `V` apart (stride V).  A warp of 32 threads at stride 8 needs
// ~32× more cache line transactions than stride 1.
//
// **Fix:** Swap the multiplier `V` from `lidx0` to the upcast constant `i`:
//
// ```text
// addr = gidx0 * TILE + lidx0 + i * V
// ```
//
// Now consecutive threads access consecutive elements (stride 1) while
// each thread still covers `V` elements (spaced `V` apart within each
// thread's own access stream).  Per-step coalescing goes from 1 util/32 B
// to 8 utils/32 B.
//
// ### IR pattern
//
// The expression tree is nested — the `Mul(lidx, V)` and `Const(i)` are
// typically at different depths:
//
// ```text
// Add(Add(Mul(gidx, TILE), Mul(lidx, V)), Const(i))
// ```
//
// This pass searches for any `Add` where one subtree contains `Mul(lidx, V)`
// and the other is `Const(i)` with `i < V`.  The `Mul(lidx, V)` subtree is
// then rewritten to just `lidx`, and `Const(i)` becomes `Mul(Const(i), V)`.

impl Kernel {
    /// Make the `ThreadCoarse` optimization: scan the kernel for global
    /// indices that can be coarsened.
    pub fn opt_coarsen(&self) -> Box<dyn Optimization> {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_upcast");
        let mut factors = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Op::Range { kind: RangeKind::Group(len), .. } = self.ops[op_id].op {
                let Some(len) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) else {
                    op_id = next;
                    continue;
                };
                for f in [16, 8, 4] {
                    //println!("len={len} f={f}");
                    if len % f as Dim == 0 {
                        factors.push((op_id, f));
                    }
                }
            }
            op_id = next;
        }
        Box::new(ThreadCoarse { factors })
    }

    /// Thread coarsening by range split: divide a group range by `factor`
    /// into a short group plus an unroll range of lanes, with no body
    /// duplication. Every old user of the group index now varies per
    /// lane automatically; Register storages grow by the lane count
    /// with one slot per lane. Unmatched lanes unfold back to scalar
    /// statements in a later devectorize pass.
    pub fn coarsen(&mut self, gidx_id: OpId, factor: u64) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("thread_coarse");
        let Op::Range { axis, kind: RangeKind::Group(len) } = self.ops[gidx_id].op else {
            unreachable!()
        };
        let Some(len) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) else {
            return;
        };
        debug_assert!(len % factor as Dim == 0);

        // Tiled backends match lanes into tiles; other backends would
        // devectorize into scalar IR past the codegen render limit.
        let tiled = self.dev_info.as_ref().is_some_and(|d| d.tenstorrent);
        if !tiled && len as u64 * factor > 5_000 {
            return;
        }

        // One unroll per axis: same-axis ranges denote the same value
        // stream (blockIdx.x used twice is still one value), so splitting
        // an axis that already hosts lanes would alias them (U*16+U covers
        // 16 elements, not 256). Refuse with the kernel untouched.
        let occupied =
            self.iter_unordered().any(|(_, x)| matches!(x, Op::Range { axis: a, kind: RangeKind::Unroll(_) } if *a == axis));
        if occupied {
            return;
        }

        //println!("thread coarse gidx_id={gidx_id} by factor={factor}");

        // Validate before mutating: collect on-chip scratch (Register
        // and Local) for lane-slotting below. A global store whose
        // address does not vary with the split range cannot take
        // per-lane values (extra lanes have no sound output address),
        // so that refuses.
        let mut scratch_storages: Vec<OpId> = Vec::new();
        let mut scratch_geps: Vec<(OpId, OpId)> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if matches!(x, Op::Storage { scope: MemScope::Register | MemScope::Local, .. }) {
                scratch_storages.push(op);
            }
            if let Op::GEP { x: base, index, .. } = x {
                if matches!(self.at(*base), Op::Storage { scope: MemScope::Register | MemScope::Local, .. }) {
                    scratch_geps.push((op, *index));
                }
            }
            if let Op::Store { dst, .. } = x {
                if let Op::GEP { x: base, index, .. } = self.at(*dst) {
                    if matches!(self.at(*base), Op::Storage { scope: MemScope::Global, .. }) {
                        // Inline subtree walk (no helper fn): does the
                        // store address mention the split range?
                        let mut mentions = false;
                        let mut stack = vec![*index];
                        let mut seen: Vec<OpId> = Vec::new();
                        while let Some(id) = stack.pop() {
                            if id == gidx_id {
                                mentions = true;
                                break;
                            }
                            if seen.contains(&id) {
                                continue;
                            }
                            seen.push(id);
                            stack.extend(self.at(id).parameters());
                        }
                        if !mentions {
                            return;
                        }
                    }
                }
            }
        }

        // First skip ops that don't need duplication
        let mut op_id = self.head;
        while !op_id.is_null()
            && matches!(
                self.ops[op_id].op,
                Op::Storage { scope: MemScope::Global | MemScope::Local, .. }
                    | Op::Param { .. }
                    | Op::Range { .. }
                    | Op::Const(_)
            )
        {
            op_id = self.next_op(op_id);
        }

        // Move gidx_id to the beginning, so that it's not getting transformed by it's own function
        self.move_op_before(gidx_id, op_id);

        // Create constant for factor
        let const_factor = self.insert_before(gidx_id, Op::Const(Constant::idx(factor)));

        // Split the group index into a short group plus an unroll range:
        // gidx := short * factor + unroll. The body is NOT duplicated;
        // every old user of gidx now varies per lane automatically.
        // (Unroll widths are small coarsen factors, always fitting u16.)
        debug_assert!(factor <= u16::MAX as u64, "coarsen: factor exceeds unroll range width");
        let new_len = self.insert_const_idx_before(gidx_id, len / factor as Dim);
        let short = self.insert_before(gidx_id, Op::Range { axis, kind: RangeKind::Group(new_len) });
        let unroll = self.insert_before(gidx_id, Op::Range { axis, kind: RangeKind::Unroll(factor as u16) });
        let base = self.insert_before(gidx_id, Op::Binary { x: short, y: const_factor, bop: BOp::Mul });
        self.ops[gidx_id].op = Op::Binary { x: base, y: unroll, bop: BOp::Add };

        // Grow every scratch storage by the lane count (one slot per
        // lane) and remap its indices: const 0 becomes the bare lane,
        // const c becomes c * factor + unroll, a dynamic index already
        // varying with the split range is left alone, anything else
        // becomes idx * factor + unroll. The wrap is value-injective,
        // so uniformly applied readers/writers stay consistent; without
        // it coarsened lanes would share scratch addresses (proven
        // clobber: four lane accs stored to one local slot, three
        // lost). Barriers stay block-wide and correct; tree reductions
        // over these slots are duplicated per lane by devectorize
        // (their indices now mention the unroll range), each lane
        // reducing its own slots.
        for storage in &scratch_storages {
            if let Op::Storage { ref mut len, .. } = self.ops[*storage].op {
                *len *= factor as Dim;
            }
        }
        for (gep, index) in scratch_geps {
            // Inline subtree walk (no helper fn): does the scratch
            // address already vary with the split range?
            let lane = match self.resolve_const(index).and_then(|c| c.as_dim()) {
                Some(0) => unroll,
                Some(_) => {
                    let base = self.insert_before(gep, Op::Binary { x: index, y: const_factor, bop: BOp::Mul });
                    self.insert_before(gep, Op::Binary { x: base, y: unroll, bop: BOp::Add })
                }
                None => {
                    let mut mentions = false;
                    let mut stack = vec![index];
                    let mut seen: Vec<OpId> = Vec::new();
                    while let Some(id) = stack.pop() {
                        if id == gidx_id {
                            mentions = true;
                            break;
                        }
                        if seen.contains(&id) {
                            continue;
                        }
                        seen.push(id);
                        stack.extend(self.at(id).parameters());
                    }
                    if mentions {
                        index
                    } else {
                        let base = self.insert_before(gep, Op::Binary { x: index, y: const_factor, bop: BOp::Mul });
                        self.insert_before(gep, Op::Binary { x: base, y: unroll, bop: BOp::Add })
                    }
                }
            };
            if let Op::GEP { index: slot, .. } = &mut self.ops[gep].op {
                *slot = lane;
            }
        }

        self.verify();
    }
}

impl Kernel {
    /// Make the `RegisterBlocking` optimization: scan the kernel for
    /// reduction loops and global axes that can be register-blocked.
    pub fn opt_register_blocking(&self) -> Box<dyn Optimization> {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("opt_register_tiling");
        let candidates: Vec<u64> = vec![8, 16, 4, 2];
        let mut global_upcasts: BTreeMap<OpId, Vec<u64>> = BTreeMap::new();
        let mut reduce_factor: BTreeMap<OpId, Vec<u64>> = BTreeMap::new();

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Op::Loop { len: len_id } = self.ops[op_id].op {
                // Boolean loops are conditionals, not counted loops.
                if self.dtype(len_id) == crate::DType::Bool {
                    op_id = next;
                    continue;
                }
                let Some(len) = self.resolve_const(len_id).and_then(crate::dtype::Constant::as_dim) else {
                    op_id = next;
                    continue;
                };
                if len >= 16 {
                    let applicable: Vec<u64> =
                        candidates.iter().copied().filter(|&f| len % f as Dim == 0 && len / f as Dim >= 4).collect();
                    if !applicable.is_empty() {
                        reduce_factor.insert(op_id, applicable);
                    }
                }
            }
            if let Op::Range { kind: RangeKind::Group(len), .. } = self.ops[op_id].op {
                let Some(len) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) else {
                    op_id = next;
                    continue;
                };
                let applicable: Vec<u64> =
                    candidates.iter().copied().filter(|&f| len % f as Dim == 0 && len / f as Dim >= 4).collect();
                if !applicable.is_empty() {
                    global_upcasts.insert(op_id, applicable);
                }
            }
            op_id = next;
        }

        if global_upcasts.is_empty() || reduce_factor.is_empty() {
            return Box::new(RegisterBlocking { reduce_splits: reduce_factor, thread_coarses: global_upcasts });
        }

        Box::new(RegisterBlocking { reduce_splits: reduce_factor, thread_coarses: global_upcasts })
    }

    pub(crate) fn apply_register_blocking(
        &mut self,
        reduce_splits: &BTreeMap<OpId, Vec<u64>>,
        global_upcasts: &BTreeMap<OpId, Vec<u64>>,
        config: usize,
    ) {
        let n_global = global_upcasts.len();
        let n_reduce = reduce_splits.len();
        if n_global == 0 || n_reduce == 0 {
            return;
        }

        let n_global_options: usize = global_upcasts.values().map(|v| v.len() + 1).product();

        let mut remaining_global = config % n_global_options;
        let mut remaining_reduce = config / n_global_options;

        let mut reduce_indices: Vec<usize> = Vec::with_capacity(n_reduce);
        for factors in reduce_splits.values() {
            let n_options = factors.len();
            let factor_idx = remaining_reduce % n_options;
            remaining_reduce /= n_options;
            reduce_indices.push(factor_idx);
        }

        let mut global_indices: Vec<usize> = Vec::with_capacity(n_global);
        for factors in global_upcasts.values() {
            let n_options = factors.len() + 1;
            let factor_idx = remaining_global % n_options;
            remaining_global /= n_options;
            global_indices.push(factor_idx);
        }

        // Apply unroll FIRST
        for (i, (&reduce_id, factors)) in reduce_splits.iter().enumerate() {
            let factor_idx = reduce_indices[i];
            let reduce_factor = factors[factor_idx];
            self.unroll_tree_reduce(reduce_id, reduce_factor as Dim);
        }

        // Then apply upcast
        for (idx, (op_id, factors)) in global_upcasts.iter().enumerate() {
            let factor_idx = global_indices[idx];
            let factor = if factor_idx == 0 { 1 } else { factors[factor_idx - 1] };
            if factor > 1 {
                self.coarsen(*op_id, factor);
            }
        }
    }
}
