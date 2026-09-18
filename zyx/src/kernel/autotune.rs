// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Autotuning system for kernel optimization.
//!
//! The autotuner ([`BeamSearch`]) is egraph-agnostic — its world is linear
//! SSA kernels. Its inputs are
//!
//! - a **seed** iterator (kernels to start from),
//! - a list of optimization make functions ([`MakeOpt`], e.g.
//!   [`Kernel::default_optimizations`]),
//! - an **epilogue** closure run on every kernel state after each
//!   optimization step (the default is [`Kernel::default_epilogue`]),
//! - a **cost** closure ranking candidates without launching them (the
//!   default is [`Kernel::base_cost`]).
//!
//! The search expands optimization sequences (`OptSeq`, private to this
//! module) as `state_{i+1} = epilogue(opt_i(state_i))`, ranks candidates by
//! cost and finally launches the best [`n_launches`](BeamSearch::n_launches)
//! candidates, keeping the fastest. Every compiled program is released; the
//! winner is returned as a `Kernel` together with its measured time.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::derived_hash_with_manual_eq)]

use crate::backend::{Dev, DeviceProgramId, LaunchArg};
use crate::dtype::{Constant, DType};
use crate::error::BackendError;
use crate::hashers::AHasher;
use crate::kernel::{IDX_T, Kernel, Op, OpId, ParamKind};
use crate::rng::Rng;
use crate::runtime::Runtime;
use crate::shape::Dim;
use crate::{DebugMask, Set, Tensor, ZyxError};
use nanoserde::{DeBin, SerBin};
use std::hash::{Hash, Hasher};

/// A kernel optimization the autotuner can apply.
///
/// Instances are created by a make function ([`MakeOpt`]) which scans the
/// kernel at a stable state and may embed `OpId`s. No other optimization
/// runs between make and apply, so embedded `OpId`s remain valid through
/// `apply`.
pub trait Optimization: std::fmt::Debug {
    /// Number of configurations this optimization exposes. Config `x` in
    /// `0..nconfigs` selects a variant; lower ids prefer hardware-aligned
    /// factors that are likely to perform better.
    fn nconfigs(&self) -> u64;
    /// Apply configuration `config` to the kernel in place.
    fn apply(&self, kernel: &mut Kernel, config: u64);
}

/// A make function scans the kernel at a stable state and returns an
/// [`Optimization`] instance ready to be applied.
pub type MakeOpt = fn(&Kernel) -> Box<dyn Optimization>;

impl Kernel {
    /// The default optimization set: the search space [`BeamSearch`] explores
    /// unless the caller provides its own list of make functions.
    pub const fn default_optimizations() -> [MakeOpt; 8] {
        [
            Kernel::opt_split_global_to_local,
            Kernel::opt_reassociate_commutative,
            Kernel::opt_coarsen,
            Kernel::opt_register_blocking,
            Kernel::opt_local_reduce,
            Kernel::opt_split_loop,
            // Kernel::opt_vectorize, // TEMP disabled: debugging matmul_1 verify failure
            Kernel::opt_merge_nested_loops,
            Kernel::opt_fuse_mad,
        ]
    }

    /// The default epilogue, run on every kernel state during autotuning and
    /// before compilation: the standard always-on pipeline (unrolling of
    /// length-1 loops, constant folding, LICM, algebraic simplification, CSE,
    /// instruction scheduling, DCE, ...) followed by the hardware-specific
    /// `exp`/`exp2` conversion. Users can provide their own epilogue.
    pub fn default_epilogue(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("default_epilogue");
        self.unroll_len1_loops();
        self.constant_folding();
        self.move_constants_to_beginning();
        self.loop_invariant_code_motion();
        self.fold_accs();
        self.delete_zero_len_indices();
        self.delete_zero_len_loops();
        self.unfold_pows();
        self.algebraic_simplifications();
        self.simplify_accumulating_loop();
        self.swap_commutative();
        self.common_subexpression_elimination();
        self.instruction_schedule();
        self.dead_code_elimination();
        let dev_info = self.device_info();
        if dev_info.has_native_exp2 {
            self.exp_to_exp2();
        } else {
            self.exp2_to_exp();
        }
        if dev_info.tenstorrent {
            self.opt_tenstorrent_tile();
            self.common_subexpression_elimination();
            self.instruction_schedule();
            self.dead_code_elimination();
            self.debug();
            panic!();
        }
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

    /// Compile the kernel, launch it once and measure the execution time.
    ///
    /// Returns the program id and the measured time in nanoseconds.
    fn launch_with_timings(
        &self,
        buffers: &[LaunchArg],
        device: Dev,
        debug: DebugMask,
    ) -> Result<(DeviceProgramId, u64), BackendError> {
        let program_id = device.compile(self, debug.asm())?;
        let nanos = device.launch_timed(program_id, buffers)?;
        Ok((program_id, nanos))
    }
}

/// A candidate optimization sequence explored by [`BeamSearch`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OptSeq {
    /// List of (optimization_id, config_id) pairs to apply.
    opts: Vec<(usize, u64)>,
    /// Cost estimate for this optimization sequence.
    cost: u64,
}

/// Apply an optimization sequence to the kernel, running the epilogue after
/// each step: `state_{i+1} = epilogue(opt_i(state_i))`. This is the
/// deterministic repro of a searched kernel state.
fn apply_seq(kernel: &mut Kernel, seq: &OptSeq, optimizations: &[MakeOpt], epilogue: &impl Fn(&mut Kernel)) {
    for &(opt_id, config) in &seq.opts {
        optimizations[opt_id](kernel).apply(kernel, config);
        epilogue(kernel);
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

/// Beam search autotuner.
///
/// Explores optimization sequences over the seed kernels, ranking candidates
/// with the cost closure and measuring the most promising ones on the device.
#[cfg_attr(feature = "py", pyo3::pyclass)]
#[derive(Debug, Clone, SerBin, DeBin, nanoserde::DeJson)]
#[nserde(default)]
pub struct BeamSearch {
    /// Max number of kernel launches
    pub n_launches: usize,
    /// Number of initial optimization seeds
    pub n_seeds: usize,
    /// How many optimizations to try each iteration
    pub n_added_per_step: usize,
    /// How many iterations to remove each iteration
    pub n_removed_per_step: usize,
    /// Max number of optimizations that can be tried
    pub n_total_opts: usize,
}

impl Default for BeamSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl BeamSearch {
    /// Defaults mirroring the config-file fallbacks.
    pub const fn new() -> Self {
        Self { n_added_per_step: 200, n_launches: 1, n_removed_per_step: 0, n_seeds: 200, n_total_opts: 1000 }
    }
}

impl BeamSearch {
    /// Autotune using beam search, binding buffers from realized tensors.
    ///
    /// Each tensor must be realized: present in the runtime's `buffer_map`
    /// or resolvable to a constant (bound as a scalar variable argument) —
    /// `Runtime::is_realized` decides. Tensors are consumed positionally:
    /// read-only (Global + Variable) params first, then GlobalMut params,
    /// matching the kernel argument order. Returns an error if a tensor is
    /// not realized.
    ///
    /// See [`BeamSearch::run_`] for the search semantics.
    pub fn run(
        &self,
        rt: &mut Runtime,
        seeds: impl IntoIterator<Item = Kernel>,
        tensors: &[&Tensor],
        optimizations: &[fn(&Kernel) -> Box<dyn Optimization>],
        epilogue: impl Fn(&mut Kernel),
        cost: impl Fn(&Kernel) -> u64,
    ) -> Result<(Kernel, u64), ZyxError> {
        let mut args: Vec<LaunchArg> = Vec::with_capacity(tensors.len());
        for tensor in tensors {
            if let Some(buf_id) = rt.leaf_buffer(tensor.id()) {
                args.push(LaunchArg::Buffer(buf_id.buffer_id));
            } else if let Some(value) = rt.resolve_symbolic(tensor.id()) {
                args.push(LaunchArg::Variable(value));
            } else {
                return Err(ZyxError::kernel_error(format!("autotune: tensor {} is not realized", tensor.id()).into()));
            }
        }
        self.run_(rt, seeds, &args, optimizations, epilogue, cost)
    }

    /// Autotune using beam search.
    ///
    /// The seeds must be prepared by the caller: already linearized
    /// ([`Kernel::is_linearized`]), with basic optimizations and the epilogue
    /// applied — the search does no seed preprocessing. All seeds must belong
    /// to the same device and share the same parameter signature; `args` must
    /// match that signature positionally (read-only params first, then
    /// GlobalMut params; interleaved GlobalMut is rejected). The epilogue
    /// runs on every kernel state after each optimization step
    /// (`state_{i+1} = epilogue(opt_i(state_i))`), so the final state of a
    /// sequence is exactly what gets launched. Every program compiled during
    /// measurement is released; the winner is returned as a kernel together
    /// with its measured time in nanoseconds.
    pub fn run_(
        &self,
        _rt: &mut Runtime,
        seeds: impl IntoIterator<Item = Kernel>,
        args: &[LaunchArg],
        optimizations: &[MakeOpt],
        epilogue: impl Fn(&mut Kernel),
        cost: impl Fn(&Kernel) -> u64,
    ) -> Result<(Kernel, u64), ZyxError> {
        let debug = crate::debug_mask();
        let seeds: Vec<Kernel> = seeds.into_iter().collect();
        if seeds.is_empty() {
            return Err(ZyxError::kernel_error("autotune: no seeds".into()));
        }
        let dev = seeds[0].dev;
        if seeds.iter().any(|seed| seed.dev != dev) {
            return Err(ZyxError::kernel_error("autotune: seeds span multiple devices".into()));
        }

        // Every seed must be linearized and share one parameter signature;
        // `args` binds against it positionally (read-only params first, then
        // GlobalMut params). Interleaved GlobalMut would misbind the args.
        let mut signature: Option<Vec<(ParamKind, DType)>> = None;
        for seed in &seeds {
            if !seed.is_linearized() {
                return Err(ZyxError::kernel_error("autotune: seed kernel is not linearized".into()));
            }
            let params: Vec<(ParamKind, DType)> = {
                let mut params = Vec::new();
                let mut op_id = seed.head;
                while !op_id.is_null() {
                    if let Op::Param { dtype, kind, .. } = seed.ops[op_id].op {
                        params.push((kind, dtype));
                    }
                    op_id = seed.next_op(op_id);
                }
                params
            };
            match &signature {
                None => {
                    let mut seen_mut = false;
                    for (kind, _) in &params {
                        if *kind == ParamKind::GlobalMut {
                            seen_mut = true;
                        } else if seen_mut {
                            return Err(ZyxError::kernel_error(
                                "autotune: kernel has interleaved GlobalMut params (read-only params must come first)".into(),
                            ));
                        }
                    }
                    if params.len() != args.len() {
                        return Err(ZyxError::kernel_error(
                            format!("autotune: kernel has {} params but {} args were given", params.len(), args.len()).into(),
                        ));
                    }
                    signature = Some(params);
                }
                Some(sig) if sig != &params => {
                    return Err(ZyxError::kernel_error("autotune: seeds have differing parameter signatures".into()));
                }
                _ => {}
            }
            seed.verify();
        }

        let mut programs: Vec<DeviceProgramId> = Vec::new();
        let mut best_kernel: Option<Kernel> = None;
        let mut best_time = u64::MAX;
        let mut last_error: Option<BackendError> = None;

        for seed in seeds {
            let base = seed;

            let mut visited = Set::default();
            visited.insert(base.get_hash());
            let mut items: Vec<OptSeq> = vec![OptSeq { opts: Vec::new(), cost: cost(&base) }];

            // Initial candidates: one optimization applied to state_0.
            let avail_configs: Vec<Box<dyn Optimization>> = optimizations.iter().map(|make| make(&base)).collect();
            let total_configs: u64 = avail_configs.iter().map(|opt| opt.nconfigs()).sum();
            let mult = self.n_seeds.min(total_configs as usize) as u64;
            for (opt_id, opt) in avail_configs.iter().enumerate() {
                let n_configs = opt.nconfigs();
                let n_configs_to_try = ((n_configs * mult) as f32 / total_configs as f32).ceil() as u64;
                for config_id in 0..n_configs_to_try {
                    let mut new_kernel = base.clone();
                    opt.apply(&mut new_kernel, config_id);
                    epilogue(&mut new_kernel);
                    let hash = new_kernel.get_hash();
                    if visited.contains(&hash) {
                        continue;
                    }
                    visited.insert(hash);
                    items.push(OptSeq { opts: vec![(opt_id, config_id)], cost: cost(&new_kernel) });
                }
            }

            let mut rng = Rng::seed_from_u64(3_498_203_498);
            let mut exhausted = Set::default();
            let mut i = 0;
            while i < self.n_total_opts && !items.is_empty() {
                i += 1;
                let Some(opt_seq) = sample_best(&items, &exhausted, &mut rng).cloned() else {
                    break;
                };
                let mut thread_kernel = base.clone();
                apply_seq(&mut thread_kernel, &opt_seq, optimizations, &epilogue);

                let avail_configs: Vec<Box<dyn Optimization>> = optimizations.iter().map(|make| make(&thread_kernel)).collect();
                let total_configs: u64 = avail_configs.iter().map(|opt| opt.nconfigs()).sum();
                let mult = self.n_added_per_step.min(total_configs as usize) as u64;

                let mut added = 0;
                for (opt_id, opt) in avail_configs.iter().enumerate() {
                    let n_configs = opt.nconfigs();
                    let n_configs_to_try = ((n_configs * mult) as f32 / total_configs as f32).ceil() as u64;
                    for config_id in 0..n_configs_to_try {
                        let mut opts = opt_seq.opts.clone();
                        opts.push((opt_id, config_id));

                        let mut new_kernel = thread_kernel.clone();
                        opt.apply(&mut new_kernel, config_id);
                        epilogue(&mut new_kernel);
                        let hash = new_kernel.get_hash();
                        if visited.contains(&hash) {
                            continue;
                        }
                        visited.insert(hash);

                        let new_seq = OptSeq { opts, cost: cost(&new_kernel) };
                        if new_kernel.ops.len().0 > 10000 {
                            exhausted.insert(new_seq.clone());
                        }

                        items.push(new_seq);
                        added += 1;
                    }
                }

                if added == 0 {
                    // State can't be optimized further
                    break;
                }

                remove_worst(&mut items, self.n_removed_per_step, &mut rng);
            }

            // Measurement: rebuild each candidate deterministically and launch.
            items.sort_by_key(|seq| seq.cost);
            items.truncate(self.n_launches);
            let mut launched = Set::default();
            for opt_seq in &items {
                let mut kernel = base.clone();
                apply_seq(&mut kernel, opt_seq, optimizations, &epilogue);
                if launched.insert(kernel.get_hash()) {
                    if debug.ir() {
                        kernel.debug();
                    }

                    match kernel.launch_with_timings(&args, dev, debug) {
                        Ok((program_id, time)) => {
                            programs.push(program_id);
                            if time < best_time {
                                best_time = time;
                                best_kernel = Some(kernel);
                            }
                        }
                        Err(e) => {
                            last_error = Some(e);
                        }
                    }
                }
            }
        }

        // Drop all compiled programs; the winner is returned as a kernel.
        for program_id in programs {
            dev.release(program_id);
        }

        match best_kernel {
            Some(kernel) => Ok((kernel, best_time)),
            None => Err(last_error
                .map(ZyxError::from)
                .unwrap_or_else(|| ZyxError::kernel_error("autotune: no successful kernel launches".into()))),
        }
    }
}

impl Kernel {
    /// Generate tiling variants of this kernel by replacing the given
    /// `Const` index operations with concrete values.
    ///
    /// Each row of `variants` is one complete configuration — the N values
    /// replace the N `ops` (row-wise, NOT a cross-product). Chain further
    /// variant generators on the returned iterator with `flat_map`.
    ///
    /// Every op in `ops` must be an `Op::Const` of dtype `IDX_T`.
    pub fn generate_tiling_variants<const N: usize>(
        &self,
        ops: [OpId; N],
        variants: Vec<[Dim; N]>,
    ) -> Result<impl Iterator<Item = Kernel>, ZyxError> {
        for &op_id in &ops {
            match self.ops[op_id].op {
                Op::Const(c) if c.dtype() == IDX_T => {}
                _ => return Err(ZyxError::kernel_error("generate_tiling_variants: op must be a Const of dtype IDX_T".into())),
            }
        }
        let base = self.clone();
        Ok(variants.into_iter().map(move |dims| {
            let mut kernel = base.clone();
            for (&op_id, &dim) in ops.iter().zip(dims.iter()) {
                kernel.ops[op_id].op = Op::Const(Constant::idx(dim));
            }
            kernel
        }))
    }
}
