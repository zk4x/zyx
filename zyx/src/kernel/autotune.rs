// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#![allow(unused)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::derived_hash_with_manual_eq)]

use crate::backend::{AutotuneConfig, Device, DeviceInfo, DeviceProgramId, MemoryPool, PoolBufferId};
use crate::error::BackendError;
use crate::kernel::{Kernel, Op, OpId, Scope};
use crate::rng::Rng;
use crate::shape::Dim;
use crate::slab::SlabId;
use crate::{DebugMask, Map, Set};
use nanoserde::{DeBin, SerBin};
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;

type OptConfigFn = fn(&Kernel, &DeviceInfo) -> (Optimization, usize);

const AVAILABLE_OPTIMIZATIONS: [OptConfigFn; 7] = [
    |k, _| Kernel::opt_reassociate_commutative(k),
    Kernel::opt_split_global_to_local,
    |k, _| Kernel::opt_upcast(k),
    |k, _| Kernel::opt_register_tiling(k),
    Kernel::opt_tiled_reduce,
    |k, _| Kernel::opt_split_loop(k),
    |k, _| Kernel::opt_licm(k),
];

#[derive(Debug)]
pub enum Optimization {
    ReassociateCommutative,
    UnrollLoops {
        factors: Vec<u64>,
    },
    SplitGlobalToLocal {
        factors: Vec<(OpId, u64)>,
    },
    Upcast {
        factors: Vec<(OpId, u64)>,
    },
    RegisterTiling {
        reduce_splits: BTreeMap<OpId, Vec<u64>>,
        global_upcasts: BTreeMap<OpId, Vec<u64>>,
    },
    UnrollConstantLoops,
    TiledReduce {
        factors: Vec<(OpId, u64, u64)>,
    },
    SplitLoop {
        factors: Vec<(OpId, u64)>,
    },
    Licm,
}

impl Optimization {
    pub fn debug(&self, config: usize) {
        match self {
            Optimization::ReassociateCommutative => println!("ReassociateCommutative"),
            Optimization::UnrollLoops { factors } => {
                let factor = factors[config];
                println!("unroll loop len={} by {}", factor, factor)
            }
            Optimization::SplitGlobalToLocal { factors } => {
                let (op_id, factor) = factors[config];
                println!("split global index {} to local by {}", op_id, factor)
            }
            Optimization::Upcast { factors } => {
                let (op_id, factor) = factors[config];
                println!("upcast axis {} by {}", op_id, factor)
            }
            Optimization::RegisterTiling { reduce_splits, global_upcasts } => {
                let n_global_options: usize = global_upcasts.values().map(|v| v.len() + 1).product();
                let mut remaining_global = config % n_global_options;
                let mut parts = Vec::new();
                for (op_id, facs) in global_upcasts.iter() {
                    let n_options = facs.len() + 1;
                    let factor_idx = remaining_global % n_options;
                    remaining_global /= n_options;
                    if factor_idx < facs.len() {
                        parts.push(format!("upcast axis {} by {}", op_id, facs[factor_idx]));
                    }
                }
                let mut remaining_reduce = config / n_global_options;
                for (op_id, facs) in reduce_splits.iter() {
                    let n_options = facs.len() + 1;
                    let factor_idx = remaining_reduce % n_options;
                    remaining_reduce /= n_options;
                    if factor_idx < facs.len() {
                        parts.push(format!("unroll {} by {}", op_id, facs[factor_idx]));
                    }
                }
                if parts.is_empty() {
                    println!("register tiling (no-op)");
                } else {
                    println!("register tiling {}", parts.join(", "))
                }
            }
            Optimization::UnrollConstantLoops => println!("UnrollConstantLoops"),
            Optimization::TiledReduce { factors } => {
                let (op_id, local, global) = factors[config];
                println!("tiled reduce index {} local={}, global={}", op_id, local, global)
            }
            Optimization::SplitLoop { factors } => {
                let (op_id, factor) = factors[config];
                println!("split loop {} by {}", op_id, factor)
            }
            Optimization::Licm => println!("Licm"),
        }
    }

    /// Applies the optimization with the given config ID.
    /// Config IDs are ordered such that lower IDs use hardware-aligned factors
    /// (e.g., warp size 32 for CUDA, wavefront size 64 for AMD) which are likely to perform better.
    pub fn apply(&self, kernel: &mut Kernel, config: usize) {
        match self {
            Optimization::ReassociateCommutative => {
                #[cfg(feature = "time")]
                let _timer = crate::Timer::new("ReassociateCommutative");
                kernel.reassociate_commutative();
            }
            Optimization::UnrollLoops { factors } => {
                #[cfg(feature = "time")]
                let _timer = crate::Timer::new("UnrollLoops");
                let factor = factors[config];
                if (kernel.ops.len().0 as usize) < 500 {
                    kernel.unroll_loops(factor);
                }
            }
            Optimization::SplitGlobalToLocal { factors } => {
                #[cfg(feature = "time")]
                let _timer = crate::Timer::new("SplitGlobalToLocal");
                let (op_id, factor) = factors[config];
                let Op::Index { len, scope, axis } = kernel.ops[op_id].op else { unreachable!() };
                debug_assert_eq!(scope, Scope::Global);
                let factor: Dim = factor;
                kernel.split_dim(
                    op_id,
                    vec![
                        Op::Index { len: len / factor, scope: Scope::Global, axis },
                        Op::Index { len: factor, scope: Scope::Local, axis },
                    ],
                );
            }
            Optimization::Upcast { factors } => {
                #[cfg(feature = "time")]
                let _timer = crate::Timer::new("Upcast");
                if factors.is_empty() {
                    return;
                }
                let (op_id, factor) = factors[config];
                kernel.upcast(op_id, factor);
            }
            Optimization::RegisterTiling { reduce_splits, global_upcasts } => {
                #[cfg(feature = "time")]
                let _timer = crate::Timer::new("RegisterTiling");
                kernel.apply_register_tiling(reduce_splits, global_upcasts, config);
            }
            Optimization::UnrollConstantLoops => {
                #[cfg(feature = "time")]
                let _timer = crate::Timer::new("UnrollConstantLoops");
                kernel.unroll_constant_loops();
            }
            Optimization::TiledReduce { factors } => {
                #[cfg(feature = "time")]
                let _timer = crate::Timer::new("TiledReduce");
                let (op_id, factor, tree_branch) = factors[config];
                kernel.tiled_reduce(op_id, factor, tree_branch);
            }
            Optimization::SplitLoop { factors } => {
                #[cfg(feature = "time")]
                let _timer = crate::Timer::new("SplitLoop");
                let (op_id, factor) = factors[config];
                let Op::Loop { len } = kernel.ops[op_id].op else { unreachable!() };
                kernel.split_dim(op_id, vec![Op::Loop { len: len / factor }, Op::Loop { len: factor }]);
            }
            Optimization::Licm => {
                #[cfg(feature = "time")]
                let _timer = crate::Timer::new("Licm");
                kernel.loop_invariant_code_motion();
            }
        }
    }
}

impl Kernel {
    pub fn run_always_on_optimizations(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("always on optimizations");
        self.eliminate_zero_len_index();
        self.unroll_len1_loops();
        self.constant_folding();
        self.move_constants_to_beginning();
        self.loop_invariant_code_motion();
        self.fold_accs();
        self.delete_empty_loops();
        self.unfold_pows();
        self.div_mod_simplification();
        //self.simplify_accumulating_loop();
        self.swap_commutative();
        self.common_subexpression_elimination();
        self.dead_code_elimination();
    }

    /// Autotune for debugging, applying only a selected series of optimizations
    #[allow(unused)]
    pub fn apply_selected_optimizations(
        &self,
        buffers: &[PoolBufferId],
        device: &mut Device,
        memory_pool: &mut MemoryPool,
        config: &AutotuneConfig,
        flop: u64,
        read_bytes: u64,
        write_bytes: u64,
        debug: DebugMask,
    ) -> Result<(DeviceProgramId, OptSeq), BackendError> {
        //eprintln!("=== autotune_debug called ===");
        let mut kernel = self.clone();

        kernel.run_always_on_optimizations();
        {
            kernel.eliminate_zero_len_index();
            kernel.unroll_len1_loops();
            kernel.constant_folding();
            kernel.move_constants_to_beginning();
            kernel.loop_invariant_code_motion();
            kernel.fold_accs();
            kernel.delete_empty_loops();
            kernel.unfold_pows();
            kernel.div_mod_simplification();
            kernel.simplify_accumulating_loop();
            kernel.swap_commutative();
            kernel.common_subexpression_elimination();
            kernel.dead_code_elimination();
        }
        {
            kernel.eliminate_zero_len_index();
            kernel.unroll_len1_loops();
            kernel.constant_folding();
            kernel.move_constants_to_beginning();
            kernel.loop_invariant_code_motion();
            kernel.fold_accs();
            kernel.delete_empty_loops();
            kernel.unfold_pows();
            kernel.div_mod_simplification();
            kernel.simplify_accumulating_loop();
            kernel.swap_commutative();
            kernel.common_subexpression_elimination();
            kernel.dead_code_elimination();
        }
        {
            kernel.eliminate_zero_len_index();
            kernel.unroll_len1_loops();
            kernel.constant_folding();
            kernel.move_constants_to_beginning();
            kernel.loop_invariant_code_motion();
            kernel.fold_accs();
            kernel.delete_empty_loops();
            kernel.unfold_pows();
            kernel.div_mod_simplification();
            kernel.simplify_accumulating_loop();
            kernel.swap_commutative();
            kernel.common_subexpression_elimination();
            kernel.dead_code_elimination();
        }

        /*let (reg_tile_opt, n_reg_tile) = kernel.opt_register_tiling();
        if n_reg_tile > 0 {
            reg_tile_opt.apply(&mut kernel, 10);
        }
        kernel.run_always_on_optimizations();
        kernel.run_always_on_optimizations();*/

        kernel.debug();

        let (program_id, _) = kernel.launch_with_timings(buffers, device, memory_pool, debug, flop, read_bytes, write_bytes)?;

        Ok((program_id, OptSeq { opts: Vec::new(), cost: Cost::default() }))
    }

    /// Release mode autotune with beam like search and multithreading
    pub fn autotune(
        &self,
        buffers: &[PoolBufferId],
        device: &mut Device,
        memory_pool: &mut MemoryPool,
        config: &AutotuneConfig,
        flop: u64,
        read_bytes: u64,
        write_bytes: u64,
        debug: DebugMask,
    ) -> Result<(DeviceProgramId, OptSeq), BackendError> {
        if false {
            return self.apply_selected_optimizations(buffers, device, memory_pool, config, flop, read_bytes, write_bytes, debug);
        }

        let dev_info_ptr: *const DeviceInfo = device.info();
        let dev_info_ref = unsafe { &*dev_info_ptr };

        let n_launches = config.n_launches;
        let n_seeds = config.n_seeds;
        let n_added_per_step = config.n_added_per_step;
        let n_removed_per_step = config.n_removed_per_step;
        let n_total_opts = config.n_total_opts;

        let mut items = Vec::new();
        let mut visited = Set::default();

        // Initial seed
        let mut kernel = self.clone();
        kernel.run_always_on_optimizations();
        kernel.run_always_on_optimizations();
        kernel.run_always_on_optimizations();

        let avail_configs = AVAILABLE_OPTIMIZATIONS.map(|config_fn| config_fn(&kernel, dev_info_ref));
        let total_configs = avail_configs.iter().map(|(_, x)| *x).sum::<usize>();
        let mult = n_seeds.min(total_configs);
        for (opt_id, (_, n_configs)) in avail_configs.iter().enumerate() {
            let n_configs_to_try = ((n_configs * mult) as f32 / total_configs as f32).ceil() as usize;
            let mut config_id = 0;
            while config_id < n_configs_to_try {
                let mut new_kernel = kernel.clone();
                avail_configs[opt_id].0.apply(&mut new_kernel, config_id);
                new_kernel.run_always_on_optimizations();
                let hash = new_kernel.get_hash();
                if visited.contains(&hash) {
                    config_id += 1;
                    continue;
                }
                let new_seq = OptSeq { opts: vec![(opt_id, config_id)], cost: new_kernel.get_cost(dev_info_ref) };
                visited.insert(hash);
                items.push(new_seq);
                config_id += 1;
            }
        }

        let mut rng = Rng::seed_from_u64(3_498_203_498);
        while items.len() < n_total_opts && !items.is_empty() {
            let mut thread_kernel = kernel.clone();
            let opt_seq = sample_best(&items, &mut rng).clone();
            opt_seq.apply(&mut thread_kernel, dev_info_ref);
            thread_kernel.run_always_on_optimizations();

            if thread_kernel.ops.len().0 > 5000 {
                continue;
            }

            let avail_configs = AVAILABLE_OPTIMIZATIONS.map(|config_fn| config_fn(&thread_kernel, dev_info_ref));
            let total_configs = avail_configs.iter().map(|(_, x)| *x).sum::<usize>();
            let mult = n_added_per_step.min(total_configs);

            let mut added = 0;
            for (opt_id, _) in avail_configs.iter().enumerate() {
                let n_configs_to_try = ((avail_configs[opt_id].1 * mult) as f32 / total_configs as f32).ceil() as usize;

                for config_id in 0..n_configs_to_try {
                    let mut opts = opt_seq.opts.clone();
                    opts.push((opt_id, config_id));

                    let mut new_kernel = thread_kernel.clone();
                    avail_configs[opt_id].0.apply(&mut new_kernel, config_id);
                    let hash = new_kernel.get_hash();
                    if visited.contains(&hash) {
                        continue;
                    }
                    let new_seq = OptSeq { opts, cost: new_kernel.get_cost(dev_info_ref) };
                    visited.insert(hash);
                    items.push(new_seq);
                    added += 1;
                }
            }

            if added == 0 {
                break;
            }

            remove_worst(&mut items, n_removed_per_step, &mut rng);
        }

        let mut launched_kernels = Set::default();
        let mut best_time = u64::MAX;
        let mut best_program = DeviceProgramId::NULL;
        let mut best_opt_seq = OptSeq { opts: Vec::new(), cost: Cost::default() };
        let mut i = n_launches;
        let mut any_success = false;
        let mut last_error = None;
        while i > 0 {
            let opt_seq = sample_best(&items, &mut rng);
            let mut kernel = kernel.clone();

            for &(opt_id, opt_cfg) in &opt_seq.opts {
                let (opt, _) = AVAILABLE_OPTIMIZATIONS[opt_id](&kernel, dev_info_ref);
                //opt.debug(opt_cfg);
                opt.apply(&mut kernel, opt_cfg);
            }
            kernel.run_always_on_optimizations();
            kernel.run_always_on_optimizations();
            kernel.run_always_on_optimizations();
            kernel.fuse_mad();
            kernel.run_always_on_optimizations();
            kernel.run_always_on_optimizations();
            kernel.run_always_on_optimizations();

            if launched_kernels.insert(kernel.get_hash()) {
                if debug.ir() {
                    kernel.debug();
                }

                match kernel.launch_with_timings(buffers, device, memory_pool, debug, flop, read_bytes, write_bytes) {
                    Ok((program_id, time)) => {
                        any_success = true;
                        if time < best_time {
                            best_program = program_id;
                            best_time = time;
                            best_opt_seq = opt_seq.clone();
                        }
                    }
                    Err(e) => {
                        last_error = Some(e);
                    }
                }
            }

            i -= 1;
        }

        if !any_success {
            return Err(last_error.unwrap());
        }

        Ok((best_program, best_opt_seq))
    }

    pub fn get_cost(&self, dev_info: &DeviceInfo) -> Cost {
        // TODO add measuring bank conflicts
        let mut n_instructions = 0;
        let mut n_scoped_loads = [0u64, 0, 0];
        let mut n_scoped_stores = [0u64, 0, 0];
        let mut barriers_per_thread = 0u64;

        let mut gws = [1; 3];
        let mut lws = [1; 3];
        let mut loop_mult = 1;
        let mut latest_loop_lengths = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Cast { .. } | Op::Unary { .. } | Op::Binary { .. } | Op::Mad { .. } => {
                    n_instructions += loop_mult;
                }
                #[allow(clippy::match_same_arms)]
                Op::Const(_) | Op::Define { .. } => {}
                Op::Load { src, vlen, .. } => {
                    n_instructions += loop_mult;
                    let Op::Define { scope, .. } = self.ops[src].op else { unreachable!() };
                    match scope {
                        Scope::Global => n_scoped_loads[0] += loop_mult * u64::from(vlen),
                        Scope::Local => n_scoped_loads[1] += loop_mult * u64::from(vlen),
                        Scope::Register => n_scoped_loads[2] += loop_mult * u64::from(vlen),
                    }
                }
                Op::Store { dst, vlen, .. } => {
                    n_instructions += loop_mult * 3;
                    let Op::Define { scope, .. } = self.ops[dst].op else { unreachable!() };
                    match scope {
                        Scope::Global => n_scoped_stores[0] += loop_mult * u64::from(vlen),
                        Scope::Local => n_scoped_stores[1] += loop_mult * u64::from(vlen),
                        Scope::Register => n_scoped_stores[2] += loop_mult * u64::from(vlen),
                    }
                }
                Op::Index { len, scope, axis } => match scope {
                    Scope::Global => gws[axis as usize] = len,
                    Scope::Local => lws[axis as usize] = len,
                    Scope::Register => todo!(),
                },
                Op::Loop { len } => {
                    n_instructions += loop_mult * 3;
                    loop_mult *= len as u64;
                    latest_loop_lengths.push(len as u64);
                }
                Op::EndLoop => {
                    loop_mult /= latest_loop_lengths.pop().unwrap();
                }
                Op::Wmma { dims, .. } => {
                    let (m, n, k) = dims.decompose_mnk();
                    let warp = u64::from(dev_info.warp_size);
                    let cost = (m * n * k) as u64 / warp;
                    n_instructions += loop_mult * cost;
                }
                Op::Barrier { .. } => {
                    barriers_per_thread += loop_mult;
                }
                Op::If { .. } => {
                    n_instructions += loop_mult * 3;
                }
                Op::EndIf => {}
                Op::Devectorize { .. } => {
                    todo!()
                }
                Op::Vectorize { .. } => {
                    // TODO multiply all ops that are vectorized by the vectorization factor
                    todo!()
                }
                Op::ConstView(_) => todo!(),
                Op::LoadView(_) => todo!(),
                Op::StoreView { .. } => todo!(),
                Op::Move { .. } => todo!(),
                Op::Reduce { .. } => todo!(),
            }
            op_id = self.next_op(op_id);
        }

        let register_estimate: u64 = 0;

        let global_ws = gws.iter().product::<Dim>();
        let n_threads = lws.iter().product::<Dim>();
        let instructions_per_thread = n_instructions;
        let global_loads_per_thread = n_scoped_loads[0];
        let local_loads_per_thread = n_scoped_loads[1];
        let global_stores_per_thread = n_scoped_stores[0];
        let local_stores_per_thread = n_scoped_stores[1];

        let total_loads = n_threads * global_ws * global_loads_per_thread;
        let total_stores = n_threads * global_ws * global_stores_per_thread;
        let total_local = n_threads * global_ws * (local_loads_per_thread + local_stores_per_thread);
        let total_instr = n_threads * global_ws * instructions_per_thread;
        let total_barriers = n_threads * global_ws * barriers_per_thread;

        let memory_score =
            ((total_loads * 10 + total_stores * 10 + total_local + total_barriers * 20) as f64 / total_instr as f64).min(1.0);

        let workgroup_score = 1.0 - (n_threads as f64 / dev_info.max_local_threads as f64).min(1.0);

        let register_score = if register_estimate > dev_info.max_register_bytes {
            0.95
        } else {
            0.05
        };

        let cost = ((memory_score + register_score + workgroup_score) * 1_000_000_000.0) as u64;

        Cost { cost }
    }

    pub fn get_hash(&self) -> u64 {
        let mut hasher = crate::chasher::CHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }

    pub fn launch_with_timings(
        &self,
        buffers: &[PoolBufferId],
        device: &mut Device,
        memory_pool: &mut MemoryPool,
        debug: DebugMask,
        flops: u64,
        bytes_read: u64,
        bytes_written: u64,
    ) -> Result<(DeviceProgramId, u64), BackendError> {
        let program_id = device.compile(self, debug.asm())?;
        let begin = std::time::Instant::now();
        let event = device.launch(program_id, memory_pool, buffers, Vec::new())?;
        memory_pool.sync_events(vec![event])?;
        let nanos = begin.elapsed().as_nanos() as u64;
        let perf = crate::kernel_cache::get_perf(flops, bytes_read, bytes_written, nanos);
        if debug.perf() {
            println!("{perf}");
        }
        Ok((program_id, nanos))
    }
}

#[derive(Debug, Default, Clone, Copy, Hash, DeBin, SerBin)]
pub struct Cost {
    cost: u64,
}

impl PartialEq for Cost {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for Cost {}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.cost.cmp(&other.cost)
    }
}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash, DeBin, SerBin)]
pub struct OptSeq {
    opts: Vec<(usize, usize)>,
    cost: Cost,
}

impl OptSeq {
    pub fn apply(&self, kernel: &mut Kernel, dev_info: &DeviceInfo) {
        for &(opt_id, opt_cfg) in &self.opts {
            let (opt, _): (Optimization, usize) = AVAILABLE_OPTIMIZATIONS[opt_id](kernel, dev_info);
            opt.apply(kernel, opt_cfg);
        }
    }
}

fn remove_worst(items: &mut Vec<OptSeq>, mut n: usize, rng: &mut Rng) {
    if items.len() < 10 * n {
        return;
    }
    while n > 0 && !items.is_empty() {
        // Tournament among random samples biased toward high cost
        const K: usize = 2; // number of candidates
        let mut worst_idx = rng.range::<u64>(0..items.len() as u64) as usize;
        let mut worst_cost = items[worst_idx].cost;
        for _ in 1..K {
            let i = rng.range::<u64>(0..items.len() as u64) as usize;
            let cost = items[i].cost;
            if cost > worst_cost {
                worst_idx = i;
                worst_cost = cost;
            }
        }

        items.swap_remove(worst_idx);

        n -= 1;
    }
}

fn sample_best<'a>(items: &'a [OptSeq], rng: &mut Rng) -> &'a OptSeq {
    const K: usize = 2;
    debug_assert!(!items.is_empty(), "sample_best called with empty items");
    let len = items.len();
    let mut best_idx = rng.range::<u64>(0..len as u64) as usize;
    let mut best_cost = items[best_idx].cost;
    for _ in 1..K {
        let i = rng.range::<u64>(0..len as u64) as usize;
        let cost = items[i].cost;
        if cost < best_cost {
            best_idx = i;
            best_cost = cost;
        }
    }

    &items[best_idx]
}
