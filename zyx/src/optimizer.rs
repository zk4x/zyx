// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use crate::{
    backend::DeviceInfo,
    kernel::{Kernel, Op, OpId, Scope},
    shape::Dim,
};
use nanoserde::{DeBin, SerBin};
use std::collections::HashSet;

// Indices in 0..max_index for each optimization Opt
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash, DeBin, SerBin)]
pub struct Optimization(u32);

#[derive(Debug, Clone, DeBin, SerBin)]
pub struct Optimizer {
    // optimizations
    work_size_opt: WorkSizeOpt,
    loop_unrolling_opt: LoopUnrollOpt,
    loop_split_opt: LoopSplitOpt,
    //inner_loop_swap_opt: InnerLoopSwapOpt, // a bit harder to know max number of optimizations
    max_indices: [u32; 3],
    default_indices: [Vec<u32>; 3],
    // best optimization found so far
    best_optimization: Optimization,
    // time taken by kernel with the best optimization
    pub best_time_nanos: u64,
    // Which iteration are we on? This is for default and random.
    default_iteration: u32,
    // This is the deterministic order for iterations, up to max_iter
    full_iteration: u32,
    max_iter: u32,
    // Optimizations that were tried during random search and refinement
    tried: HashSet<Optimization>,
    // Last tried optimization
    last: Optimization,
}

impl Optimizer {
    #[must_use]
    pub fn apply_optimization(
        &self,
        kernel: &mut Kernel,
        optimization: Optimization,
        dev_info: &DeviceInfo,
        debug_ir: bool,
    ) -> bool {
        kernel.fuse_reduces();

        let [
            local_work_size_opt_index,
            _loop_unrolling_opt_index,
            _loop_split_opt_index,
        ] = optimization.into_indices(self.max_indices);

        kernel.unfold_movement_ops();

        if !self.work_size_opt.apply_optimization(local_work_size_opt_index, kernel)
        //|| !self.loop_split_opt.apply_optimization(loop_split_opt_index, kernel)
        //|| !self.loop_jam_opt.apply_optimization(loop_jam_opt_index, kernel) // NOTE: cannot run after licm
        //|| !self.loop_unrolling_opt.apply_optimization(loop_unrolling_opt_index, kernel)
        {
            return false;
        }

        // Use tensor cores if possible
        kernel.fuse_mma(dev_info);
        kernel.fuse_mad();

        let mut temp_kernel = kernel.clone();
        for _ in 0..2 {
            kernel.move_constants_to_beginning();
            kernel.swap_commutative();
            kernel.reassociate_commutative(); // TODO This is changes the kernel on every iteration, fix it
            kernel.constant_folding();
            kernel.loop_invariant_code_motion();
            kernel.delete_empty_loops();
            kernel.unroll_constant_loops();
            kernel.common_subexpression_elimination();
            kernel.dead_code_elimination();

            if *kernel == temp_kernel {
                break;
            }
            temp_kernel = kernel.clone();
        }

        // Convert exponentiation (BOp::Pow) to just exp2 and ln2
        kernel.unfold_pows();

        let mut temp_kernel = kernel.clone();
        for _ in 0..10 {
            kernel.move_constants_to_beginning();
            kernel.swap_commutative();
            kernel.constant_folding();
            kernel.loop_invariant_code_motion();
            kernel.delete_empty_loops();
            kernel.unroll_constant_loops();
            kernel.common_subexpression_elimination();
            kernel.dead_code_elimination();

            if *kernel == temp_kernel {
                break;
            }
            temp_kernel = kernel.clone();
        }

        kernel.fold_accs();
        kernel.common_subexpression_elimination();
        kernel.dead_code_elimination();

        kernel.verify();

        if debug_ir {
            kernel.debug();
        }

        true
    }

    pub fn new(kernel: &Kernel, dev_info: &DeviceInfo) -> Self {
        let mut kernel = kernel.clone();
        kernel.unfold_movement_ops();
        kernel.unfold_reduces();
        let (work_size_opt, work_size_opt_max_idx, work_size_opt_defaults) = WorkSizeOpt::new(&kernel, dev_info);
        let (loop_unroll_opt, loop_unroll_opt_max_idx, loop_unroll_opt_defaults) = LoopUnrollOpt::new(&kernel);
        let (loop_split_opt, loop_split_opt_max_idx, loop_split_opt_defaults) = LoopSplitOpt::new(&kernel);
        let max_indices = [work_size_opt_max_idx, loop_unroll_opt_max_idx, loop_split_opt_max_idx];
        let default_indices = [
            work_size_opt_defaults,
            loop_unroll_opt_defaults,
            loop_split_opt_defaults,
        ];

        //println!( "Optimizing work_size_opt_max_idx={work_size_opt_max_idx},\nloop_jam_opt_max_idx={loop_jam_opt_max_idx},\nloop_unrolling_opt_max_idx={loop_unroll_opt_max_idx},\nloop_split_opt_max_idx={loop_split_opt_max_idx}" );
        //println!("Max default opts: {:?}", default_indices);

        Self {
            max_indices,
            default_indices,
            work_size_opt,
            loop_unrolling_opt: loop_unroll_opt,
            loop_split_opt,
            best_optimization: Optimization(0),
            best_time_nanos: u64::MAX,
            tried: HashSet::with_capacity(200),
            default_iteration: 0,
            full_iteration: 0,
            max_iter: max_indices.iter().product(),
            last: Optimization(0),
        }
    }

    pub fn is_new(&self) -> bool {
        self.last == Optimization(0)
    }

    pub fn max_iters(&self) -> u32 {
        self.max_iter
    }

    pub fn best_optimization(&self) -> Optimization {
        self.best_optimization
    }

    pub fn next_optimization(&mut self, last_time_nanos: u64) -> Option<Optimization> {
        if last_time_nanos < self.best_time_nanos {
            self.best_time_nanos = last_time_nanos;
            self.best_optimization = self.last;
        }
        let opt = if let Some(opt) = self.default_search() {
            Some(opt)
        } else if let Some(opt) = self.random_search() {
            Some(opt)
        } else {
            self.deterministic_search()
        };
        if let Some(opt) = opt {
            self.last = opt;
        }
        opt
    }

    pub fn fully_optimized(&self) -> bool {
        self.full_iteration >= self.max_iter
    }

    fn default_search(&mut self) -> Option<Optimization> {
        if self.default_iteration >= 200 {
            return None;
        }
        self.default_iteration += 1;

        for &d0 in &self.default_indices[0] {
            for &d1 in &self.default_indices[1] {
                for &d2 in &self.default_indices[2] {
                    let dims = [d0, d1, d2];
                    let mut index = 0;
                    let mut stride = 1;
                    for i in (0..dims.len()).rev() {
                        index += dims[i] * stride;
                        stride *= self.max_indices[i];
                    }
                    let opt = Optimization(index);
                    if self.tried.insert(opt) {
                        return Some(opt);
                    }
                }
            }
        }
        None
    }

    fn random_search(&mut self) -> Option<Optimization> {
        if self.default_iteration >= 500 {
            return None;
        }
        self.default_iteration += 1;
        let mut rng = crate::rng::Rng::seed_from_u64(642392);
        //let mut rng = crate::rng::Rng::seed_from_systime();
        for _ in 0..1_000_000 {
            let index = rng.range(0..self.max_iter);
            if self.tried.insert(Optimization(index)) {
                return Some(Optimization(index));
            }
        }
        None
    }

    fn deterministic_search(&mut self) -> Option<Optimization> {
        for _ in 0..1_000_000 {
            let temp = self.full_iteration;
            self.full_iteration += 1;
            if !self.tried.contains(&Optimization(temp)) && temp < self.max_iter {
                return Some(Optimization(temp));
            }
            self.tried.remove(&Optimization(temp));
        }
        self.tried.clear();
        None
    }
}

impl Optimization {
    fn into_indices<const N: usize>(self, max_values: [u32; N]) -> [u32; N] {
        let mut value = self.0;
        debug_assert!(
            value < max_values.iter().product(),
            "Index {self:?} out of range for {max_values:?}."
        );
        let mut result: [u32; N] = [0; N];

        for i in (0..N).rev() {
            result[i] = value % max_values[i]; // Get the current component
            value /= max_values[i]; // Divide for the carry-over effect
        }

        result
    }
}

/// loop unrolling
#[derive(Debug, Clone, DeBin, SerBin)]
pub struct LoopUnrollOpt {}

impl LoopUnrollOpt {
    pub fn new(_kernel: &Kernel) -> (Self, u32, Vec<u32>) {
        (Self {}, 1, vec![0])
    }

    #[must_use]
    pub fn apply_optimization(&self, index: u32, kernel: &mut Kernel) -> bool {
        let unroll_dim = [1, 8][index as usize];
        kernel.unroll_loops(unroll_dim);
        true
    }
}

// TODO currently this is not good at all. It's too simplistic and does not try hard enough to find good default work sizes
// TODO also add permutation. It may be potentially beneficial if register and global loops are permuted.

#[derive(Debug, Clone, DeBin, SerBin)]
pub struct WorkSizeOpt {
    gws: Vec<Dim>,
    gws_factors: Vec<Vec<[Dim; 2]>>,
    max_local_threads: Dim,
}

impl WorkSizeOpt {
    pub fn new(kernel: &Kernel, dev_info: &DeviceInfo) -> (Self, u32, Vec<u32>) {
        fn divisors(x: usize, limit: usize) -> Vec<usize> {
            debug_assert_ne!(x, 0);
            let mut res = Vec::new();
            let sqrt_x = (x as f64).sqrt() as usize;
            for i in 1..=sqrt_x.min(limit) {
                if x.is_multiple_of(i) {
                    res.push(i);
                }
            }
            for i in (0..res.len()).rev() {
                res.push(x / res[i]);
            }
            res
        }

        let mut gws = kernel.shape();
        // If gws > dev_info.max_global_work_dims.len(), then we join the starting dimensions
        while gws.len() > dev_info.max_global_work_dims.len() {
            let d = gws.remove(0);
            gws[0] *= d;
        }

        let mut gws_factors = Vec::new();
        //println!("max_local_work_dims = {:?}", dev_info.max_local_work_dims);
        for (d, &max_lwd) in gws.iter().copied().zip(&dev_info.max_local_work_dims) {
            let res = divisors(d, max_lwd);

            let mut factors = Vec::new();
            // here factors needs to contain all possible pairs of values in res
            for i in 0..res.len() {
                for j in 0..res.len() {
                    let a = res[i];
                    let b = res[j];
                    if a * b <= d {
                        if a <= max_lwd && b <= 16 {
                            factors.push([a, b]);
                        }
                    }
                }
            }

            gws_factors.push(factors);
        }

        let max_local_threads = dev_info.max_local_threads;
        let max_idx = gws_factors.iter().map(|gd| gd.len() as u32).product();

        //println!("gws={gws:?}, gws_factors={gws_factors:?}, max_local_threads={max_local_threads}");

        (Self { gws, gws_factors, max_local_threads }, max_idx, vec![0])
    }

    // Returns false if this index is invalid
    #[must_use]
    pub fn apply_optimization(&self, index: u32, kernel: &mut Kernel) -> bool {
        let mut value = index as usize;
        let mut result: Vec<usize> = Vec::new();
        for max in self.gws_factors.iter().map(|f| f.len()).rev() {
            result.push(value % max);
            value /= max;
        }
        result.reverse();

        let mut gws = self.gws.clone();
        let mut lws = Vec::new();
        let mut rws = Vec::new();
        for (i, factors) in result.into_iter().zip(&self.gws_factors) {
            let [l, r] = factors[i];
            lws.push(l);
            rws.push(r);
        }
        if lws.iter().product::<Dim>() > self.max_local_threads {
            return false;
        }
        for ((g, l), r) in gws.iter_mut().zip(&lws).zip(&rws) {
            if !g.is_multiple_of(l * r) {
                return false;
            }
            *g /= l * r;
        }

        debug_assert_eq!(gws.len(), lws.len());
        debug_assert_eq!(lws.len(), rws.len());

        // Fix if kernel has more than 3 dims
        let global_indices = kernel.get_global_indices();
        if global_indices.len() > 3 {
            //kernel.debug();
            let n = global_indices.len() - 2;
            let loops: Vec<OpId> = global_indices.values().copied().take(n).collect();
            kernel.merge_loops(&loops);
        }
        kernel.reindex_indices();

        //kernel.debug();

        //gws = vec![64, 128];
        //lws = vec![8, 4];
        //rws = vec![2, 2];

        // Get existing global dims
        let mut dim_ids = Vec::new();
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            if let Op::Index { scope, axis, .. } = kernel.ops[op_id].op {
                debug_assert_eq!(scope, Scope::Global);
                dim_ids.push((op_id, axis));
            }
            op_id = kernel.next_op(op_id);
        }
        dim_ids.sort_by_key(|x| x.1);
        let dim_ids: Vec<OpId> = dim_ids.into_iter().map(|x| x.0).collect();

        //println!("gws={gws:?} lws={lws:?} rws={rws:?}");
        //println!("dim_ids={dim_ids:?}");

        let mut axis = 0;
        for (((dim_id, g), l), r) in dim_ids.into_iter().zip(gws.into_iter()).zip(lws).zip(rws) {
            let mut splits = Vec::new();
            splits.push(Op::Index { len: g, scope: Scope::Global, axis });
            splits.push(Op::Index { len: l, scope: Scope::Local, axis });
            splits.push(Op::Loop { len: r, axis });
            kernel.split_dim(dim_id, splits);
            axis += 1;
        }

        true
    }
}

#[derive(Debug, Clone, DeBin, SerBin)]
pub struct LoopSplitOpt {
    // For each reduction op, store possible split configurations
    // [reduction_op_index][split_configuration][split_dimensions]
    reduction_splits: Vec<Vec<Vec<Dim>>>,
}

impl LoopSplitOpt {
    pub fn new(kernel: &Kernel) -> (Self, u32, Vec<u32>) {
        //return (LoopSplitOpt { reduction_splits: Vec::new() }, 10);

        let mut reduction_splits = Vec::new();

        // Find all reduction ops
        for (_, op) in kernel.iter_unordered() {
            if let Op::Reduce { n_axes, .. } = *op {
                // Generate all valid splits for these dimensions
                // Calculate the total product of all dimensions
                //let mut reduce_dims = kernel.reduce_dims(op_id);
                let mut reduce_dims = vec![1];
                reduce_dims.truncate(n_axes);
                let total_product: Dim = reduce_dims.iter().product();

                let mut options: Vec<Vec<Dim>> = Vec::new();

                // Add original
                options.push(reduce_dims);

                // TODO, for tensor cores, we need split into K, 4, 2, so three dimensions at least
                let defaults = [8, 16, 2, 4];
                for d in defaults.into_iter().chain((2..=16).filter(|x| !defaults.contains(x))) {
                    if total_product.is_multiple_of(d) {
                        options.push(vec![total_product / d, d]);
                    }
                }

                reduction_splits.push(options);
            }
        }

        let max_index = reduction_splits.iter().map(|splits| splits.len() as u32).product::<u32>();

        //println!("reduction_splits={reduction_splits:?}");
        (LoopSplitOpt { reduction_splits }, max_index, vec![1, 2])
    }

    pub fn apply_optimization(&self, mut index: u32, kernel: &mut Kernel) -> bool {
        // Check if we have any reduction splits
        if self.reduction_splits.is_empty() {
            // No reduction operations found, nothing to optimize
            return true;
        }

        let reduce_ops: Vec<OpId> =
            kernel.iter_unordered().filter(|(_, op)| matches!(op, Op::Reduce { .. })).map(|(op_id, _)| op_id).collect();

        for (i, choices) in self.reduction_splits.iter().enumerate() {
            let n = choices.len() as u32;
            let idx = index % n;
            index /= n;

            let Some(&reduce_id) = reduce_ops.get(i) else { return false };

            let new_dims: &[Dim] = &self.reduction_splits[i][idx as usize];
            let Op::Reduce { x, n_axes, .. } = &mut kernel.ops[reduce_id].op else { return false };
            let x = *x;
            let n = *n_axes;
            *n_axes = new_dims.len();

            //let mut visited = Set::default();
            //kernel.recursively_reshape(x, n, new_dims, &mut visited, 0);
        }
        true
    }
}

#[test]
fn test_get_indices() {
    let opt = Optimization(117);
    let max_values = [20, 10];
    assert_eq!(opt.into_indices(max_values), [11, 7]);

    let opt = Optimization(23902);
    let max_values = [49, 17, 8, 9];
    assert_eq!(opt.into_indices(max_values), [19, 8, 7, 7]);
}
