// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#![allow(unused)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::derived_hash_with_manual_eq)]

use crate::backend::{AutotuneConfig, Device, DeviceInfo, DeviceProgramId, MemoryPool, PoolBufferId};
use crate::error::{BackendError, ErrorStatus};
use crate::kernel::cost::Cost;
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
                println!("unroll loop len={factor} by {factor}");
            }
            Optimization::SplitGlobalToLocal { factors } => {
                let (op_id, factor) = factors[config];
                println!("split global index {op_id} to local by {factor}, cfg_opt={config}");
            }
            Optimization::Upcast { factors } => {
                let (op_id, factor) = factors[config];
                println!("upcast axis {op_id} by {factor}, cfg_opt={config}");
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
                    println!("register tiling {}", parts.join(", "));
                }
            }
            Optimization::UnrollConstantLoops => println!("UnrollConstantLoops"),
            Optimization::TiledReduce { factors } => {
                let (op_id, local, global) = factors[config];
                println!("tiled reduce index {op_id} local={local}, global={global}");
            }
            Optimization::SplitLoop { factors } => {
                let (op_id, factor) = factors[config];
                println!("split loop {op_id} by {factor}");
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
                kernel.reassociate_commutative();
            }
            Optimization::UnrollLoops { factors } => {
                let factor = factors[config];
                if (kernel.ops.len().0 as usize) < 5000 {
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
                if factors.is_empty() {
                    return;
                }
                let (op_id, factor) = factors[config];
                kernel.upcast(op_id, factor);
            }
            Optimization::RegisterTiling { reduce_splits, global_upcasts } => {
                kernel.apply_register_tiling(reduce_splits, global_upcasts, config);
            }
            Optimization::UnrollConstantLoops => {
                kernel.unroll_constant_loops();
            }
            Optimization::TiledReduce { factors } => {
                let (op_id, factor, tree_branch) = factors[config];
                kernel.tiled_reduce(op_id, factor, tree_branch);
            }
            Optimization::SplitLoop { factors } => {
                let (op_id, factor) = factors[config];
                let Op::Loop { len } = kernel.ops[op_id].op else { unreachable!() };
                kernel.split_dim(op_id, vec![Op::Loop { len: len / factor }, Op::Loop { len: factor }]);
            }
            Optimization::Licm => {
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
        self.simplify_accumulating_loop();
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

        let (opt, _) = kernel.opt_split_global_to_local(device.info());
        opt.apply(&mut kernel, 1);
        let (opt, _) = kernel.opt_upcast();
        opt.apply(&mut kernel, 3);
        let (opt, _) = kernel.opt_split_global_to_local(device.info());
        opt.apply(&mut kernel, 1);
        let (opt, _) = kernel.opt_upcast();
        opt.apply(&mut kernel, 5);

        //kernel.run_always_on_optimizations();
        //kernel.run_always_on_optimizations();
        kernel.dead_code_elimination();

        self.verify();

        kernel.debug_colorless();

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
        let mut exhausted = Set::default();
        let mut i = 0;
        while i < n_total_opts && !items.is_empty() {
            i += 1;
            let mut thread_kernel = kernel.clone();
            let Some(opt_seq) = sample_best(&items, &exhausted, &mut rng).cloned() else { break };
            opt_seq.apply(&mut thread_kernel, dev_info_ref);
            thread_kernel.run_always_on_optimizations();

            //println!("Next opt {i}, kernel size: {:?}", thread_kernel.ops.len());

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
                    //avail_configs[opt_id].0.debug(config_id);
                    avail_configs[opt_id].0.apply(&mut new_kernel, config_id);
                    let hash = new_kernel.get_hash();
                    if visited.contains(&hash) {
                        continue;
                    }
                    let new_seq = OptSeq { opts, cost: new_kernel.get_cost(dev_info_ref) };
                    visited.insert(hash);

                    if new_kernel.ops.len().0 > 10000 {
                        exhausted.insert(new_seq.clone());
                    }

                    items.push(new_seq);
                    added += 1;
                }
            }

            if added == 0 {
                // Seed can't be optimized further
                //exhausted.insert(opt_seq);
                break;
            }

            remove_worst(&mut items, n_removed_per_step, &mut rng);
        }

        let mut launched_kernels = Set::default();
        let mut best_time = u64::MAX;
        let mut best_program = DeviceProgramId::NULL;
        let mut best_opt_seq = OptSeq { opts: Vec::new(), cost: Cost::default() };
        let mut any_success = false;
        let mut last_error = None;
        // Sort items by cost and benchmark the cheapest ones
        items.sort_by_key(|s| s.cost);
        for opt_seq in items.iter().take(n_launches) {
            let mut kernel = kernel.clone();

            println!("launch (cost: {}, n_opts: {}):", opt_seq.cost.cost, opt_seq.opts.len());
            for &(opt_id, opt_cfg) in &opt_seq.opts {
                let (opt, _) = AVAILABLE_OPTIMIZATIONS[opt_id](&kernel, dev_info_ref);
                print!("  ");
                opt.debug(opt_cfg);
                opt.apply(&mut kernel, opt_cfg);
            }
            let (gws, lws) = kernel.work_sizes();

            kernel.run_always_on_optimizations();
            kernel.run_always_on_optimizations();
            kernel.run_always_on_optimizations();
            kernel.fuse_mad();
            //kernel.fuse_mma(dev_info_ref); // WMMA fusion is not yet correct
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
                        let perf_line = crate::kernel_cache::get_perf(flop, read_bytes, write_bytes, time);
                        println!("  -> {time} ns  {perf_line}");
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
        }

        if !any_success {
            return Err(last_error.unwrap_or(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "No successful kernel launches.".into(),
            }));
        }

        Ok((best_program, best_opt_seq))
    }

    pub fn get_hash(&self) -> u64 {
        use sha2::Digest;
        struct H(sha2::Sha256);
        impl std::hash::Hasher for H {
            fn finish(&self) -> u64 {
                let hash = self.0.clone().finalize();
                u64::from_le_bytes(hash[..8].try_into().unwrap())
            }
            fn write(&mut self, bytes: &[u8]) {
                self.0.update(bytes);
            }
        }
        let mut hasher = H(sha2::Sha256::new());
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

fn sample_best<'a>(items: &'a [OptSeq], exhausted: &Set<OptSeq>, rng: &mut Rng) -> Option<&'a OptSeq> {
    for _ in 0..5 {
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

        if exhausted.contains(&items[best_idx]) {
            continue;
        }

        return Some(&items[best_idx]);
    }

    None
}
