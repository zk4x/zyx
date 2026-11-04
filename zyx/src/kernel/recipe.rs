// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Recipes: a fixed sequence of known-good steps applied as one search move.
//!
//! The optimization passes are order-agnostic — each works or bails, and the
//! search finds which orders pay. A recipe is the shortcut: a small ordered
//! list of steps that is known to produce a good kernel shape, exposed to the
//! search as one move so the beam can reach it without walking the
//! intermediate states.
//!
//! Every recipe returns a **config space**, never a shape: each config is one
//! combination of step factors (lanes, per-thread tile, strip), and the cost
//! model ranks them. Gates are structural only — divisibility, device limits,
//! the IR growth bound, and the precondition that makes the sequence sound
//! (a lane range must have a load that ignores it). Nothing is probed on a
//! copy: a probe would collapse the space the search exists to explore.
//!
//! - the lane recipe: split a group range into local lanes, then stage;
//! - register tiling (the coarsen era's `opt_register_blocking`): unroll the
//!   reduce loop into accumulators, then each output range into a
//!   per-thread tile;
//! - tile and stage: lanes on one output range, a per-thread tile on both,
//!   then stage the operand the lanes share.
//!
//! Recipes only fire from a plain kernel: lanes, barriers, reduces, local
//! memory or a fused matmul mean another pass already ran and the sequence's
//! assumptions no longer hold.

use super::{Kernel, MemScope, Op, OpId, ParamKind, RangeKind, autotune::Optimization, split::SplitMode};
use crate::shape::Dim;

/// Local factors a lane recipe tries, largest first.
const LANE_FACTORS: &[u16] = &[32, 16, 8, 4];

impl Kernel {
    /// A recipe's sequence is known-good only from a plain kernel: lanes,
    /// barriers, reduces, local memory or a fused matmul mean another pass
    /// already ran and the sequence's assumptions no longer hold. The search
    /// can still reach any other order one step at a time.
    fn plain_state(&self) -> bool {
        self.iter_unordered().all(|(_, op)| {
            !matches!(
                op,
                Op::Unroll { .. }
                    | Op::Barrier
                    | Op::Reduce { .. }
                    | Op::Wmma { .. }
                    | Op::Storage { scope: MemScope::Local, .. }
                    | Op::Range { kind: RangeKind::Local(_), .. }
            )
        })
    }

    /// Threads the kernel already uses: one length per local axis.
    fn threads_in_use(&self) -> u32 {
        let mut axis_lens: Vec<(u32, u32)> = Vec::new();
        for (_, op) in self.iter_unordered() {
            if let Op::Range { axis, kind: RangeKind::Local(len) } = *op
                && !axis_lens.iter().any(|&(a, _)| a == axis)
            {
                axis_lens.push((axis, len));
            }
        }
        axis_lens.iter().fold(1u32, |acc, &(_, len)| acc.saturating_mul(len))
    }

    /// Whether some global load ignores `range` — the precondition for
    /// splitting it into lanes that staging can then deduplicate across.
    fn load_invariant_over(&self, range: OpId) -> bool {
        self.iter_unordered().any(|(_, op)| match *op {
            Op::Load { src, src_idx } if matches!(self.at(src), Op::Param { kind: ParamKind::Global, .. }) => {
                self.get_strides(src_idx).is_some_and(|map| !map.contains_key(&range))
            }
            _ => false,
        })
    }
}

/// One recipe: split a group range into local lanes, then stage the loads the
/// lanes expose. `configs` holds the `(group range, lane factor)` pairs.
#[derive(Debug)]
pub struct Recipe {
    configs: Vec<(OpId, u16)>,
}

impl Optimization for Recipe {
    fn nconfigs(&self) -> u64 {
        self.configs.len() as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("lane recipe apply");
        let Some(&(range, factor)) = self.configs.get(config as usize) else {
            return;
        };
        if kernel.split(range, SplitMode::Local, factor).is_null() {
            return;
        }
        // Planned on the post-split state: the recipe owns both steps, so the
        // split's fresh ids never leave this call. The strip factor is the
        // staging optimization's own config space, not a recipe constant.
        let staging = kernel.opt_local_staging();
        if staging.nconfigs() > 0 {
            staging.apply(kernel, 0);
        }
        kernel.verify();
    }
}

impl Kernel {
    /// Make the lane recipe: one config per (group range, lane factor) —
    /// the search ranks the space, the recipe does not pick a winner.
    ///
    /// Gates are structural only: divisibility, the device's thread budget,
    /// the IR growth bound, and that some load ignores the range (otherwise
    /// there is nothing for staging to deduplicate across the new lanes).
    /// Never mutates.
    pub fn opt_recipe(&self) -> Box<dyn Optimization> {
        let mut configs: Vec<(OpId, u16)> = Vec::new();

        // A recipe is a known-good sequence *from a plain kernel*: lanes,
        // barriers, reduces and local memory mean another pass already ran,
        // and the sequence's assumptions no longer hold.
        if !self.plain_state() {
            return Box::new(Recipe { configs });
        }

        let info = self.device_info();
        if info.local_mem_size <= 0 || info.max_local_threads == 0 {
            return Box::new(Recipe { configs });
        }
        let used = self.threads_in_use();
        let ops_len = self.ops.len().0 as Dim;

        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Range { axis, kind: RangeKind::Group(len) } = self.ops[op_id].op
                && let Some(len) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim)
                && self.load_invariant_over(op_id)
            {
                let max_per_axis = info.max_local_work_dims.get(axis as usize).copied().unwrap_or(0);
                for &factor in LANE_FACTORS {
                    let factor_dim = Dim::from(factor);
                    if len % factor_dim != 0
                        || u32::from(factor) > max_per_axis
                        || used.saturating_mul(u32::from(factor)) > info.max_local_threads
                        || factor_dim * ops_len > 10_000
                    {
                        continue;
                    }
                    configs.push((op_id, factor));
                }
            }
            op_id = self.next_op(op_id);
        }

        Box::new(Recipe { configs })
    }
}

/// Register tiling: unroll the reduce loop into register accumulators, then
/// unroll each output range into a per-thread tile — the coarsen era's
/// `opt_register_blocking`, expressed with [`SplitMode::Unroll`] steps.
///
/// `configs` holds the (site, factor) steps per config, reduce loops first:
/// a loop can only be cut while its body still has one accumulator store.
#[derive(Debug)]
pub struct RegisterTiling {
    configs: Vec<Vec<(OpId, u16)>>,
}

impl Optimization for RegisterTiling {
    fn nconfigs(&self) -> u64 {
        self.configs.len() as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("reg_tiling apply");
        let Some(steps) = self.configs.get(config as usize) else {
            return;
        };
        for &(site, factor) in steps {
            // Each step is the ordinary split with its own gates; a step the
            // grown kernel no longer accepts leaves the rest of the recipe.
            kernel.split(site, SplitMode::Unroll, factor);
            kernel.verify();
        }
    }
}

impl Kernel {
    /// Make the register tiling recipe: one config per combination of site
    /// factors, reduce loops first (no skip — a reduce site that cannot be
    /// cut takes the whole recipe with it), output ranges may be skipped.
    ///
    /// The combination is pruned until the emitted IR stays under 10k ops,
    /// the same bound the coarsen-era pass used. Never mutates.
    pub fn opt_register_tiling(&self) -> Box<dyn Optimization> {
        /// Per-site factors, largest first.
        const FACTORS: &[u32] = &[16, 8, 4, 2];
        /// No emitted config may grow the IR past this.
        const MAX_OPS: Dim = 10_000;
        /// Configs beyond this are not worth enumerating: give up instead of
        /// building a product nobody samples.
        const MAX_CONFIGS: usize = 256;

        let mut configs: Vec<Vec<(OpId, u16)>> = Vec::new();
        let ops_len = self.ops.len().0 as Dim;

        // Known-good only from a plain kernel: see `opt_recipe`.
        if !self.plain_state() {
            return Box::new(RegisterTiling { configs });
        }

        // A site is a factor candidate when it divides, leaves at least 4
        // iterations, and the largest factor alone stays inside the bound.
        let factors = |len: Dim| -> Vec<u16> {
            FACTORS
                .iter()
                .copied()
                .filter(|&f| {
                    let f = Dim::from(f);
                    len % f == 0 && len / f >= 4 && f * ops_len <= MAX_OPS
                })
                .map(|f| u16::try_from(f).expect("register tiling: factor overflows u16"))
                .collect()
        };

        let mut reduce: Vec<(OpId, Vec<u16>)> = Vec::new();
        let mut global: Vec<(OpId, Vec<u16>)> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Loop { len } => {
                    if self.dtype(len) != crate::DType::Bool
                        && let Some(len) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim)
                    {
                        let candidates = factors(len);
                        if !candidates.is_empty() {
                            reduce.push((op_id, candidates));
                        }
                    }
                }
                Op::Range { kind: RangeKind::Group(len), .. } => {
                    if let Some(len) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim) {
                        let candidates = factors(len);
                        if !candidates.is_empty() {
                            global.push((op_id, candidates));
                        }
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        if reduce.is_empty() || global.is_empty() {
            return Box::new(RegisterTiling { configs });
        }

        // Growth bound over the whole combination: drop the largest factor of
        // the site with the largest factor until the worst case fits.
        loop {
            let max_prod: Dim = reduce
                .iter()
                .chain(global.iter())
                .map(|(_, fs)| fs.first().copied().unwrap_or(1))
                .fold(1, |a, f| a.saturating_mul(Dim::from(f)));
            if ops_len.saturating_mul(max_prod) <= MAX_OPS {
                break;
            }
            let mut target: Option<(usize, bool, u16)> = None;
            for (i, (_, fs)) in reduce.iter().enumerate() {
                if let Some(&f) = fs.first()
                    && target.is_none_or(|(_, _, best)| f > best)
                {
                    target = Some((i, false, f));
                }
            }
            for (i, (_, fs)) in global.iter().enumerate() {
                if let Some(&f) = fs.first()
                    && target.is_none_or(|(_, _, best)| f > best)
                {
                    target = Some((i, true, f));
                }
            }
            let Some((i, is_global, f)) = target else {
                break;
            };
            let site = if is_global { &mut global } else { &mut reduce };
            if let Some((_, fs)) = site.get_mut(i) {
                fs.retain(|&x| x != f);
            }
            if reduce.iter().chain(global.iter()).any(|(_, fs)| fs.is_empty()) {
                // A site with no factor left is not a site: the recipe needs
                // at least one step per kind.
                reduce.retain(|(_, fs)| !fs.is_empty());
                global.retain(|(_, fs)| !fs.is_empty());
                if reduce.is_empty() || global.is_empty() {
                    return Box::new(RegisterTiling { configs });
                }
            }
        }

        let n_reduce: usize = reduce.iter().map(|(_, fs)| fs.len()).product();
        let n_global: usize = global.iter().map(|(_, fs)| fs.len() + 1).product();
        if n_reduce.saturating_mul(n_global) > MAX_CONFIGS {
            return Box::new(RegisterTiling { configs });
        }

        for reduce_combo in 0..n_reduce {
            for global_combo in 0..n_global {
                let mut steps: Vec<(OpId, u16)> = Vec::new();
                let mut rem = reduce_combo;
                for (site, fs) in &reduce {
                    steps.push((*site, fs[rem % fs.len()]));
                    rem /= fs.len();
                }
                let mut rem = global_combo;
                for (site, fs) in &global {
                    // Option 0 is "leave this range alone".
                    let option = rem % (fs.len() + 1);
                    rem /= fs.len() + 1;
                    if option > 0 {
                        steps.push((*site, fs[option - 1]));
                    }
                }
                configs.push(steps);
            }
        }

        Box::new(RegisterTiling { configs })
    }
}

/// Register tiling plus shared-memory staging: lanes for the threads, a
/// per-thread register tile on both output ranges, then the operand those
/// lanes share is staged.
///
/// The order is the only one that works: staging needs lanes to exist (its
/// whole job is deduplicating across them), and the barriers it inserts make a
/// later range unroll unsound.
#[derive(Debug)]
pub struct TileAndStage {
    configs: Vec<TileStage>,
}

/// One tile-and-stage config: lanes on one output range (threads) and an
/// unroll on each output range (the per-thread register tile).
#[derive(Debug, Clone, Copy)]
struct TileStage {
    /// Range split into lanes.
    lane: OpId,
    lane_axis: u32,
    lane_len: Dim,
    lanes: u16,
    /// Per-thread outputs along the lane range and along the other one.
    tile_lane: u16,
    tile_other: u16,
    /// The other output range.
    other: OpId,
}

impl Optimization for TileAndStage {
    fn nconfigs(&self) -> u64 {
        self.configs.len() as u64
    }

    fn apply(&self, kernel: &mut Kernel, config: u64) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("tile_stage apply");
        let Some(p) = self.configs.get(config as usize) else {
            return;
        };
        if kernel.split(p.lane, SplitMode::Local, p.lanes).is_null() {
            kernel.verify();
            return;
        }
        // The lane split wrapped the range as `outer * lanes + lid`: the
        // unroll cuts that outer range, which is the only group range of this
        // length left on the axis.
        let outer_len = p.lane_len / Dim::from(p.lanes);
        let outer = kernel.iter_unordered().find_map(|(id, op)| match *op {
            Op::Range { axis, kind: RangeKind::Group(len) }
                if axis == p.lane_axis
                    && kernel.resolve_const(len).and_then(crate::dtype::Constant::as_dim) == Some(outer_len) =>
            {
                Some(id)
            }
            _ => None,
        });
        let Some(outer) = outer else {
            kernel.verify();
            return;
        };
        if kernel.split(outer, SplitMode::Unroll, p.tile_lane).is_null()
            || kernel.split(p.other, SplitMode::Unroll, p.tile_other).is_null()
        {
            kernel.verify();
            return;
        }
        let staging = kernel.opt_local_staging();
        if staging.nconfigs() > 0 {
            staging.apply(kernel, 0);
        }
        kernel.verify();
    }
}

impl Kernel {
    /// Make the tile-and-stage recipe: one config per (lane range, lane factor,
    /// per-thread tile on each range). The space is the search's to rank — the
    /// recipe picks nothing.
    ///
    /// Gates are structural: divisibility, the thread budget, the IR growth
    /// bound, and that some load ignores the lane range (staging's dedup
    /// target). Never mutates.
    pub fn opt_tile_and_stage(&self) -> Box<dyn Optimization> {
        /// Threads along the lane range.
        const LANES: &[u16] = &[32, 16, 8, 4];
        /// Per-thread outputs along each range.
        const TILES: &[u16] = &[8, 4, 2];

        let mut configs: Vec<TileStage> = Vec::new();

        let info = self.device_info();
        if info.local_mem_size <= 0 || !self.plain_state() {
            return Box::new(TileAndStage { configs });
        }
        let used = self.threads_in_use();
        let ops_len = self.ops.len().0 as Dim;

        // The two output ranges; one per axis is the shape this recipe models.
        let mut groups: Vec<(OpId, u32, Dim)> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Range { axis, kind: RangeKind::Group(len) } = self.ops[op_id].op
                && let Some(len) = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim)
                && !groups.iter().any(|g| g.1 == axis)
            {
                groups.push((op_id, axis, len));
            }
            op_id = self.next_op(op_id);
        }
        if groups.len() != 2 {
            return Box::new(TileAndStage { configs });
        }

        for &(lane, lane_axis, lane_len) in &groups {
            let other = groups.iter().find(|g| g.1 != lane_axis).copied();
            let Some((other, _, other_len)) = other else {
                continue;
            };
            if !self.load_invariant_over(lane) {
                continue;
            }
            let max_per_axis = info.max_local_work_dims.get(lane_axis as usize).copied().unwrap_or(0);
            for &lanes in LANES {
                if lane_len % Dim::from(lanes) != 0
                    || u32::from(lanes) > max_per_axis
                    || used.saturating_mul(u32::from(lanes)) > info.max_local_threads
                {
                    continue;
                }
                for &tile_lane in TILES {
                    let block = Dim::from(lanes) * Dim::from(tile_lane);
                    if lane_len % block != 0 {
                        continue;
                    }
                    for &tile_other in TILES {
                        if other_len % Dim::from(tile_other) != 0 {
                            continue;
                        }
                        if block * Dim::from(tile_other) * ops_len > 10_000 {
                            continue;
                        }
                        configs.push(TileStage { lane, lane_axis, lane_len, lanes, tile_lane, tile_other, other });
                    }
                }
            }
        }

        Box::new(TileAndStage { configs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DType,
        backend::{DTypeCapability, DeviceInfo},
        kernel::{Dev, MemScope},
    };
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
            mma_variants: vec![],
            num_circular_buffers: 0,
            has_openmp: false,
        }
    }

    /// A scalar matmul: the M operand ignores the N range, so splitting N into
    /// lanes exposes it for staging. Both group ranges offer the recipe; the
    /// applied one ends with the N-range lanes and a staged tile.
    #[test]
    fn lanes_then_stage() {
        const M: Dim = 32;
        const N: Dim = 32;
        const K: Dim = 32;

        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let a = k.param(DType::F32);
        let b = k.param(DType::F32);
        let c = k.param_mut(DType::F32);
        let m = k.group_range(0, M);
        let n = k.group_range(1, N);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        k.loop_over(K, |k, kv| {
            let a_idx = k.mad(m, K, kv);
            let b_idx = k.mad(kv, N, n);
            let x = k.load(a, a_idx);
            let y = k.load(b, b_idx);
            let prod = k.mul(x, y);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let r = k.load(acc, 0i64);
        let c_idx = k.mad(m, N, n);
        k.store(c, r, c_idx);

        let opt = k.opt_recipe();
        // Both ranges expose the other operand (M split exposes B, N split
        // exposes A), four lane factors each fit a 32-wide range.
        assert_eq!(opt.nconfigs(), 8, "one recipe per (group range, lane factor)");
        // Configs 0..4 are the M range, 4..8 the N range.
        opt.apply(&mut k, 4);
        k.verify();

        let locals: Vec<Dim> = k
            .iter_unordered()
            .filter_map(|(_, op)| match op {
                Op::Storage { scope: MemScope::Local, len, .. } => Some(*len),
                _ => None,
            })
            .collect();
        assert_eq!(locals, vec![32], "the staged tile after the lane split");
        assert!(
            k.iter_unordered().any(|(_, op)| matches!(op, Op::Range { kind: RangeKind::Local(_), .. })),
            "the recipe split a range into lanes"
        );
    }

    /// No shared memory, no lane recipe: there is nothing to stage into.
    #[test]
    fn no_recipe_without_local_memory() {
        let mut k = Kernel::new(Dev::Auto);
        let mut info = cpu();
        info.local_mem_size = 0;
        k.dev_info = Some(Arc::new(info));
        let a = k.param(DType::F32);
        let c = k.param_mut(DType::F32);
        let m = k.group_range(0, 32i64);
        let n = k.group_range(1, 32i64);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        k.loop_over(32i64, |k, kv| {
            let a_idx = k.mad(m, 32i64, kv);
            let b_idx = k.mad(kv, 32i64, n);
            let x = k.load(a, a_idx);
            let y = k.load(a, b_idx);
            let prod = k.mul(x, y);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let r = k.load(acc, 0i64);
        let c_idx = k.mad(m, 32i64, n);
        k.store(c, r, c_idx);

        assert_eq!(k.opt_recipe().nconfigs(), 0);
    }

    /// Register tiling: the reduce loop becomes register accumulators, the
    /// output ranges become per-thread lanes. Every config must end in valid
    /// IR: the recipe only offers combinations its steps can apply.
    #[test]
    fn registers_are_tiled() {
        const M: Dim = 32;
        const N: Dim = 32;
        const K: Dim = 32;

        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let a = k.param(DType::F32);
        let b = k.param(DType::F32);
        let c = k.param_mut(DType::F32);
        let m = k.group_range(0, M);
        let n = k.group_range(1, N);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        k.loop_over(K, |k, kv| {
            let a_idx = k.mad(m, K, kv);
            let b_idx = k.mad(kv, N, n);
            let x = k.load(a, a_idx);
            let y = k.load(b, b_idx);
            let prod = k.mul(x, y);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let r = k.load(acc, 0i64);
        let c_idx = k.mad(m, N, n);
        k.store(c, r, c_idx);

        let opt = k.opt_register_tiling();
        // 3 factors for the 32-trip loop and 4 options (skip + 3) per group
        // range, minus the growth prune: 16 * 16 = 256 is the cap.
        assert!(opt.nconfigs() > 0, "a matmul has a reduce loop and two ranges");

        // Every config must end in valid IR, not just the ones the search
        // happens to sample.
        for config in 0..opt.nconfigs() {
            let mut probe = k.clone();
            opt.apply(&mut probe, config);
            probe.verify();
        }

        // Config 0 leaves both ranges alone: only the loop is cut.
        opt.apply(&mut k, 0);
        k.verify();
        let unrolls = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Unroll { .. })).count();
        assert_eq!(unrolls, 1, "the reduce loop's lanes");
        assert!(
            k.iter_unordered().any(|(_, op)| matches!(op, Op::Reduce { .. })),
            "the loop's per-lane values reduce into the accumulator"
        );

        // The last config of the first reduce combo cuts every output range.
        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let a = k.param(DType::F32);
        let b = k.param(DType::F32);
        let c = k.param_mut(DType::F32);
        let m = k.group_range(0, M);
        let n = k.group_range(1, N);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        k.store(acc, zero, 0i64);
        k.loop_over(K, |k, kv| {
            let a_idx = k.mad(m, K, kv);
            let b_idx = k.mad(kv, N, n);
            let x = k.load(a, a_idx);
            let y = k.load(b, b_idx);
            let prod = k.mul(x, y);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let r = k.load(acc, 0i64);
        let c_idx = k.mad(m, N, n);
        k.store(c, r, c_idx);
        let opt = k.opt_register_tiling();
        // Each range has 3 factors, so options are skip + 3 = 4; the last
        // global combo of the first reduce combo is 4 * 4 - 1 = 15.
        opt.apply(&mut k, 15);
        k.verify();
        let unrolls = k.iter_unordered().filter(|(_, op)| matches!(op, Op::Unroll { .. })).count();
        assert_eq!(unrolls, 3, "the reduce loop and both output ranges");
    }

    /// Lanes, then a register-blocked range, then staging: the combined shape.
    /// The staged operand's range was unrolled, so its value is lane-vector
    /// valued and the tile is read back as one element per lane.
    #[test]
    fn tiles_then_stages() {
        const M: Dim = 32;
        const N: Dim = 32;
        const K: Dim = 32;

        let mut k = Kernel::new(Dev::Auto);
        k.dev_info = Some(Arc::new(cpu()));
        let a = k.param(DType::F32);
        let b = k.param(DType::F32);
        let c = k.param_mut(DType::F32);
        let m = k.group_range(0, M);
        let n = k.group_range(1, N);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0.0f32);
        k.store(acc, zero, 0i64);
        k.loop_over(K, |k, kv| {
            let a_idx = k.mad(m, K, kv);
            let b_idx = k.mad(kv, N, n);
            let x = k.load(a, a_idx);
            let y = k.load(b, b_idx);
            let prod = k.mul(x, y);
            let prev = k.load(acc, 0i64);
            let sum = k.add(prod, prev);
            k.store(acc, sum, 0i64);
        });
        let r = k.load(acc, 0i64);
        let c_idx = k.mad(m, N, n);
        k.store(c, r, c_idx);

        let opt = k.opt_tile_and_stage();
        assert!(opt.nconfigs() > 0, "lanes expose an operand to share");

        // Every config must end in valid IR, not just the sampled ones.
        for config in 0..opt.nconfigs() {
            let mut probe = k.clone();
            opt.apply(&mut probe, config);
            probe.verify();
        }

        opt.apply(&mut k, 0);
        k.verify();
        assert!(
            k.iter_unordered().any(|(_, op)| matches!(op, Op::Storage { scope: MemScope::Local, .. })),
            "the shared operand is staged"
        );
        assert!(k.iter_unordered().any(|(_, op)| matches!(op, Op::Range { kind: RangeKind::Local(_), .. })));
        assert!(k.iter_unordered().any(|(_, op)| matches!(op, Op::Unroll { .. })), "the register tile");
    }
}
