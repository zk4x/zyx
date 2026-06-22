// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Autotuning system for kernel optimization.
//!
//! This module provides an autotuning framework that explores different kernel
//! configurations to find optimal performance on the target hardware.
//!
//! # How It Works
//!
//! The autotuning system uses a beam-search algorithm to explore optimization
//! sequences:
//!
//! 1. Start with a base kernel and apply always-on optimizations
//! 2. Try each available optimization with its variants
//! 3. Track visited kernel hashes to avoid duplicates
//! 4. Launch kernels and measure actual timing
//! 5. Build a cost model from real measurements
//! 6. Use beam search to find the best configuration
//!
//! # Available Optimizations
//!
//! The system supports 8 optimization types:
//!
//! - `ReassociateCommutative`: Reassociate and group commutative operations
//! - `UnrollLoops`: Unroll loops with constant or large factors
//! - `SplitGlobalToLocal`: Split global indices into local factors
//! - `ThreadCoarse`: Coarsen thread-level parallelism
//! - `RegisterBlocking`: Block operations for register tiling
//! - `TiledReduce`: Apply tiled reduction parallelism
//! - `SplitLoop`: Split large loops into smaller iterations
//! - `PadIndex`: Pad indices to hardware-friendly sizes
//! - `Vectorize`: Combine scalar loads, stores and ops into vectorized ops
//!

#![allow(unused)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::derived_hash_with_manual_eq)]

use crate::backend::{AutotuneConfig, Device, DeviceInfo, DeviceProgramId, MemoryPool, PoolBufferId};
use crate::error::{BackendError, ErrorStatus};
use crate::hashers::AHasher;
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

/// A config function scans the kernel at a stable state and returns an
/// Optimization with OpIds embedded. No other optimizations run between
/// config time and apply time, so OpIds remain valid through apply.
type OptConfigFn = fn(&Kernel, &DeviceInfo) -> (Optimization, usize);

const AVAILABLE_OPTIMIZATIONS: [OptConfigFn; 9] = [
    |k, _| Kernel::opt_reassociate_commutative(k),
    Kernel::opt_split_global_to_local,
    |k, _| Kernel::opt_thread_coarse(k),
    |k, _| Kernel::opt_register_blocking(k),
    Kernel::opt_local_reduce,
    |k, _| Kernel::opt_split_loop(k),
    |k, _| Kernel::opt_pad_index(k),
    Kernel::opt_vectorize,
    |k, _| Kernel::opt_merge_nested_loops(k),
];

#[derive(Debug)]
pub(crate) enum Optimization {
    /// Reassociate commutative operations (addition, multiplication)
    /// to group them and reduce instruction count.
    ReassociateCommutative,
    /// Unroll loops with a specific factor.
    UnrollLoops {
        /// Unroll factors to try for each loop.
        factors: Vec<u64>,
    },
    /// Split a global index into local factors for parallelization.
    SplitGlobalToLocal {
        /// Pairs of (operation_id, split_factor) for each split.
        factors: Vec<(OpId, u64)>,
    },
    /// Coarsen thread-level parallelism by grouping operations.
    ThreadCoarse {
        /// Pairs of (operation_id, coarsening_factor) for each group.
        factors: Vec<(OpId, u64)>,
    },
    /// Register blocking optimization for tiled reductions.
    RegisterBlocking {
        /// Reduction splits mapped to tile sizes.
        reduce_splits: BTreeMap<OpId, Vec<u64>>,
        /// Thread coarsening factors for each global axis.
        thread_coarses: BTreeMap<OpId, Vec<u64>>,
    },
    /// Unroll loops with constant lengths.
    UnrollConstantLoops,
    /// Tiled reduction parallelization.
    TiledReduce {
        /// Pairs of (operation_id, local_factor, global_factor) for each tiled reduce.
        factors: Vec<(OpId, u64, u64)>,
    },
    /// Split a loop into smaller iterations.
    SplitLoop {
        /// Pairs of (loop_id, split_factor) for each split.
        factors: Vec<(OpId, u64)>,
    },
    /// Pad indices to hardware-friendly sizes (e.g., 32 for CUDA warps).
    PadIndex {
        /// Pairs of (index_op_id, target_size) for each padding.
        factors: Vec<(OpId, Dim)>,
    },
    /// Combine scalar loads into vectorized loads for better memory bandwidth.
    Vectorize {
        /// Supported vector lengths for this device.
        supported_lens: Vec<u8>,
        vectorize_ops: bool,
    },
    /// Merge nested loops into a single loop (enables tiled_reduce).
    MergeNestedLoops {
        /// Each entry is a nested loop chain (outermost first).
        /// Config index selects which chain to merge.
        groups: Vec<Vec<OpId>>,
    },
}

impl Optimization {
    /// Debug output for the optimization with the given config ID.
    ///
    /// This is used during autotuning to print what optimizations
    /// are being applied for debugging purposes.
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
            Optimization::ThreadCoarse { factors } => {
                let (op_id, factor) = factors[config];
                println!("thread_coarse axis {op_id} by {factor}, cfg_opt={config}");
            }
            Optimization::RegisterBlocking { reduce_splits, thread_coarses } => {
                use std::fmt::Write;

                let mut info = String::new();

                let n_global = thread_coarses.len();
                let n_reduce = reduce_splits.len();
                if n_global == 0 || n_reduce == 0 {
                    return;
                }

                let n_global_options: usize = thread_coarses.values().map(|v| v.len() + 1).product();

                let mut remaining_global = config % n_global_options;
                let mut remaining_reduce = config / n_global_options;

                let mut reduce_indices: Vec<usize> = Vec::with_capacity(n_reduce);
                for (_, factors) in reduce_splits.iter() {
                    let n_options = factors.len();
                    let factor_idx = remaining_reduce % n_options;
                    remaining_reduce /= n_options;
                    reduce_indices.push(factor_idx);
                }

                let mut global_indices: Vec<usize> = Vec::with_capacity(n_global);
                for (_, factors) in thread_coarses.iter() {
                    let n_options = factors.len() + 1;
                    let factor_idx = remaining_global % n_options;
                    remaining_global /= n_options;
                    global_indices.push(factor_idx);
                }

                // Apply unroll FIRST
                for (i, (&reduce_id, factors)) in reduce_splits.iter().enumerate() {
                    let factor_idx = reduce_indices[i];
                    let reduce_factor = factors[factor_idx];
                    write!(info, "unroll loop_id={reduce_id} by {reduce_factor}");
                }

                // Then apply thread coarsing
                let mut idx = 0;
                for (op_id, factors) in thread_coarses.iter() {
                    let factor_idx = global_indices[idx];
                    let factor = if factor_idx == 0 { 1 } else { factors[factor_idx - 1] };
                    if factor > 1 {
                        write!(info, ", thread coarse gidx op_id={op_id} by {factor}");
                    }
                    idx += 1;
                }
                println!("{info}");
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
            Optimization::PadIndex { factors } => {
                let (op_id, _) = factors[config];
                println!("pad index {op_id} by 32, cfg_opt={config}");
            }
            Optimization::Vectorize { .. } => {
                println!("Vectorize");
            }
            Optimization::MergeNestedLoops { groups } => {
                if let Some(ids) = groups.get(config) {
                    println!("MergeNestedLoops group {} ({} loops)", config, ids.len());
                }
            }
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
            Optimization::ThreadCoarse { factors } => {
                if factors.is_empty() {
                    return;
                }
                let (op_id, factor) = factors[config];
                kernel.thread_coarse(op_id, factor);
            }
            Optimization::RegisterBlocking { reduce_splits, thread_coarses } => {
                kernel.apply_register_blocking(reduce_splits, thread_coarses, config);
            }
            Optimization::UnrollConstantLoops => {
                kernel.unroll_constant_loops();
            }
            Optimization::TiledReduce { factors } => {
                let (op_id, factor, tree_branch) = factors[config];
                kernel.local_reduce(op_id, factor, tree_branch);
            }
            Optimization::SplitLoop { factors } => {
                let (op_id, factor) = factors[config];
                let Op::Loop { len } = kernel.ops[op_id].op else { unreachable!() };
                kernel.split_dim(op_id, vec![Op::Loop { len: len / factor }, Op::Loop { len: factor }]);
            }
            Optimization::PadIndex { factors } => {
                if factors.is_empty() {
                    return;
                }
                let (gidx_id, pad_to) = factors[config];
                let Op::Index { len: current_len, .. } = kernel.ops[gidx_id].op else { unreachable!() };
                let pad_len = (pad_to - current_len % pad_to) % pad_to;
                if pad_len > 0 {
                    kernel.pad_index(gidx_id, current_len, pad_len, crate::dtype::Constant::idx(0));
                }
            }
            Optimization::Vectorize { supported_lens, vectorize_ops } => {
                kernel.vectorize_loads(supported_lens);
                kernel.vectorize_stores(supported_lens);
                if *vectorize_ops {
                    kernel.vectorize_ops_forward(supported_lens);
                    kernel.vectorize_ops_backward(supported_lens);
                }
            }
            Optimization::MergeNestedLoops { groups } => {
                if let Some(loop_ids) = groups.get(config) {
                    kernel.merge_nested_loops(loop_ids);
                }
            }
        }
    }
}

impl Kernel {
    /// Run always-on optimizations on the kernel.
    ///
    /// These optimizations should always be applied before kernel compilation
    /// and are run twice to ensure all opportunities are exploited.
    ///
    /// The optimization pipeline includes:
    ///
    /// 1. `unroll_len1_loops` - Unroll loops with length 1
    /// 2. `constant_folding` - Evaluate constant expressions
    /// 3. `move_constants_to_beginning` - Move constants to the start
    /// 4. `loop_invariant_code_motion` - Hoist loop-invariant code
    /// 5. `fold_accs` - Fold accumulator operations
    /// 6. `delete_empty_loops` - Remove loops with zero iterations
    /// 7. `unfold_pows` - Unfold power operations
    /// 8. `algebraic_simplification` - Simplify algebraic expressions
    /// 9. `simplify_accumulating_loop` - Simplify accumulating loop patterns
    /// 10. `swap_commutative` - Swap commutative operations
    /// 11. `common_subexpression_elimination` - Eliminate duplicate computations
    /// 12. `instruction_schedule` - Schedule instructions for better performance
    /// 13. `dead_code_elimination` - Remove unused code
    ///
    /// Running this twice ensures all optimization opportunities are explored.
    pub fn run_always_on_optimizations(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("always on optimizations");
        self.unroll_len1_loops();
        self.constant_folding();
        self.move_constants_to_beginning();
        self.loop_invariant_code_motion();
        self.fold_accs();
        self.delete_empty_loops();
        self.unfold_pows();
        self.algebraic_simplification();
        self.simplify_accumulating_loop();
        self.swap_commutative();
        self.common_subexpression_elimination();
        self.instruction_schedule();
        self.dead_code_elimination();
    }

    /// Autotune for debugging, applying only a selected series of optimizations
    #[allow(unused)]
    pub(crate) fn apply_selected_optimizations(
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
        let mut kernel = self.clone();

        kernel.run_always_on_optimizations();
        kernel.run_always_on_optimizations();

        let (opt, _) = kernel.opt_thread_coarse();
        opt.apply(&mut kernel, 0);
        kernel.run_always_on_optimizations();

        kernel.vectorize_loads(&[4]);
        kernel.vectorize_stores(&[4]);
        kernel.vectorize_ops_backward(&[4]);
        kernel.vectorize_ops_forward(&[2, 4]);

        kernel.run_always_on_optimizations();
        kernel.run_always_on_optimizations();
        kernel.fuse_mad();
        kernel.run_always_on_optimizations();
        kernel.run_always_on_optimizations();

        kernel.debug();

        let (program_id, _) = kernel.launch_with_timings(
            buffers,
            device,
            memory_pool,
            debug,
            flop,
            read_bytes,
            write_bytes,
            self.get_hash(),
        )?;

        Ok((program_id, OptSeq { opts: Vec::new(), cost: Cost::default() }))
    }

    /// Release mode autotune with beam-like search and multithreading.
    ///
    /// This method explores the optimization space using a beam search algorithm:
    ///
    /// 1. Start with a base kernel and apply always-on optimizations
    /// 2. For each available optimization, generate multiple variants
    /// 3. Track kernel hashes to avoid exploring duplicate states
    /// 4. Launch kernels and measure actual timing
    /// 5. Build a cost model from real measurements
    /// 6. Use beam search to find the best configuration
    ///
    /// The search explores up to `n_total_opts` optimization sequences,
    /// with `n_seeds` initial seeds and `n_added_per_step` new sequences
    /// added at each step. The beam width is `n_launches`.
    ///
    /// # Arguments
    ///
    /// * `buffers` - Memory buffers for the kernel
    /// * `device` - Target device for compilation
    /// * `memory_pool` - Shared memory pool for buffers
    /// * `config` - Autotuning configuration
    /// * `flop`, `read_bytes`, `write_bytes` - Profile information for cost modeling
    /// * `debug` - Debug flags for IR and performance output
    ///
    /// # Returns
    ///
    /// Returns the best program ID and optimization sequence found.
    pub(crate) fn autotune_(
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

        let variant_hash = self.get_hash();
        let n_launches = config.n_launches;
        let n_seeds = config.n_seeds;
        let n_added_per_step = config.n_added_per_step;
        let n_removed_per_step = config.n_removed_per_step;
        let n_total_opts = config.n_total_opts;

        let mut items = Vec::new();
        let mut visited = Set::default();

        // Initial seed
        let mut kernel = self.clone();
        kernel.eliminate_zero_len_index();
        kernel.renumber_indices();
        kernel.run_always_on_optimizations();
        kernel.run_always_on_optimizations();
        kernel.run_always_on_optimizations();

        if !device.info().has_native_exp2 {
            kernel.exp2_to_exp();
            kernel.log2_to_ln();
        }

        let avail_configs = AVAILABLE_OPTIMIZATIONS.map(|config_fn| config_fn(&kernel, device.info()));
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
                let cost = new_kernel.get_cost(device.info());
                let new_seq = OptSeq { opts: vec![(opt_id, config_id)], cost };
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
            opt_seq.apply(&mut thread_kernel, device.info());
            thread_kernel.run_always_on_optimizations();

            //println!("Next opt {i}, kernel size: {:?}", thread_kernel.ops.len());

            let avail_configs = AVAILABLE_OPTIMIZATIONS.map(|config_fn| config_fn(&thread_kernel, device.info()));
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
                    let new_seq = OptSeq { opts, cost: new_kernel.get_cost(device.info()) };
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

        // Sample randomly for variety in cost model data
        /*let n = n_launches.min(items.len());
        let mut rng_launch = Rng::seed_from_u64(0xDEAD_BEEF);
        let mut sampled = Vec::with_capacity(n);
        while sampled.len() < n && !items.is_empty() {
            let idx = rng_launch.range::<u64>(0..items.len() as u64) as usize;
            sampled.push(items.swap_remove(idx));
        }
        items = sampled;*/

        // Sort by cost: try cheaper configs first
        items.sort_by_key(|opt_seq| opt_seq.cost.cost);
        items.truncate(n_launches);

        for opt_seq in items.iter() {
            let mut kernel = kernel.clone();

            //println!("launch (cost: {}, n_opts: {}):", opt_seq.cost.cost, opt_seq.opts.len());
            for &(opt_id, opt_cfg) in &opt_seq.opts {
                let (opt, _) = AVAILABLE_OPTIMIZATIONS[opt_id](&kernel, device.info());
                if debug.autotune() {
                    opt.debug(opt_cfg);
                }
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

                match kernel.launch_with_timings(
                    buffers,
                    device,
                    memory_pool,
                    debug,
                    flop,
                    read_bytes,
                    write_bytes,
                    variant_hash,
                ) {
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
        }

        if !any_success {
            return Err(last_error.unwrap_or_else(|| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "No successful kernel launches.".into(),
            }));
        }

        Ok((best_program, best_opt_seq))
    }

    /// Get a hash of the kernel for deduplication during autotuning.
    ///
    /// This hash is used to track visited kernel states and avoid
    /// exploring duplicate optimization sequences.
    pub(crate) fn get_hash(&self) -> u64 {
        let mut hasher = AHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }

    /// Launch a kernel and measure its timing.
    ///
    /// This method compiles the kernel, launches it on the device,
    /// and returns the execution time in nanoseconds. It is used
    /// during autotuning to build a cost model.
    ///
    /// # Arguments
    ///
    /// * `buffers` - Memory buffers for the kernel
    /// * `device` - Target device for compilation and execution
    /// * `memory_pool` - Shared memory pool for buffers
    /// * `debug` - Debug flags for IR and performance output
    /// * `flops`, `bytes_read`, `bytes_written` - Profile information
    /// * `variant_hash` - Hash of the kernel variant for logging
    ///
    /// # Returns
    ///
    /// Returns a tuple of (program_id, nanoseconds) or an error.
    pub(crate) fn launch_with_timings(
        &self,
        buffers: &[PoolBufferId],
        device: &mut Device,
        memory_pool: &mut MemoryPool,
        debug: DebugMask,
        flops: u64,
        bytes_read: u64,
        bytes_written: u64,
        variant_hash: u64,
    ) -> Result<(DeviceProgramId, u64), BackendError> {
        let program_id = device.compile(self, debug.asm())?;
        let begin = std::time::Instant::now();
        let event = device.launch(program_id, memory_pool, buffers, Vec::new())?;
        memory_pool.sync_events(vec![event])?;
        let nanos = begin.elapsed().as_nanos() as u64;
        let perf = crate::kernel_cache::get_perf(flops, bytes_read, bytes_written, nanos);
        /*self.get_cost(device.info()).debug();
        println!("variant_hash={variant_hash}, {perf}");*/
        if debug.perf() {
            println!("{perf}");
        }
        Ok((program_id, nanos))
    }
}

#[derive(Debug, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub(crate) struct OptSeq {
    /// List of (optimization_id, config_id) pairs to apply.
    opts: Vec<(usize, usize)>,
    /// Cost estimate for this optimization sequence.
    cost: Cost,
}

impl SerBin for OptSeq {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        self.opts.ser_bin(output);
        self.cost.ser_bin(output);
    }
}

impl DeBin for OptSeq {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        let opts = Vec::<(usize, usize)>::de_bin(offset, bytes)?;
        let cost = Cost::de_bin(offset, bytes)?;
        Ok(Self { opts, cost })
    }
}

impl OptSeq {
    /// Apply the optimization sequence to the kernel.
    ///
    /// Applies all optimizations in the sequence to the kernel
    /// using the provided device information.
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
