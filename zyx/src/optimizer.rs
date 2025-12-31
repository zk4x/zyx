use nanoserde::{DeBin, SerBin};

use crate::{
    backend::DeviceInfo,
    kernel::{Kernel, Op, Scope},
    shape::Dim,
};
use std::collections::HashSet;

// Indices in 0..max_index for each optimization Opt
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash, DeBin, SerBin)]
pub struct Optimization(u32);

#[derive(Debug, Clone, DeBin, SerBin)]
pub struct Optimizer {
    // optimizations
    local_work_size_opt: WorkSizeOpt,
    loop_unroll_and_jam_opt: LoopJamOpt,
    loop_unrolling_opt: LoopUnrollingOpt,
    loop_split_opt: LoopSplitOpt,
    //inner_loop_swap_opt: InnerLoopSwapOpt, // a bit harder to know max number of optimizations
    max_indices: [u32; 4],
    // best optimization found so far
    best_optimization: Optimization,
    // time taken by kernel with the best optimization
    pub best_time_nanos: u64,
    // Which iteration are we on? First 30 iterations are random, then 20 iterations of refinement
    // and remaider is just going over all possible optimizations in deterministic order
    rand_iteration: u32,
    full_iteration: u32,
    max_iter: u32,
    // Optimizations that were tried during random search and refinement
    tried: HashSet<Optimization>,
    // Last tried optimization
    last: Optimization,
}

impl Optimizer {
    #[must_use]
    pub fn apply_optimization(&self, kernel: &mut Kernel, optimization: Optimization, debug_ir: bool) -> bool {
        let [
            local_work_size_opt_index,
            loop_unroll_and_jam_opt_index,
            loop_unrolling_opt_index,
            loop_split_opt_index,
        ] = optimization.into_indices(self.max_indices);

        if !self.local_work_size_opt.apply_optimization(local_work_size_opt_index, kernel) {
            return false;
        }
        kernel.close_loops();

        if !self.loop_split_opt.apply_optimization(loop_split_opt_index, kernel) {
            return false;
        }

        kernel.unfold_reduces();
        kernel.unfold_views();

        kernel.move_constants_to_beginning();
        kernel.constant_folding();
        kernel.common_subexpression_elimination();
        kernel.dead_code_elimination();
        kernel.reorder_commutative();
        kernel.loop_invariant_code_motion();
        kernel.delete_empty_loops();

        // Unroll and jam for all loops
        if !self.loop_unroll_and_jam_opt.apply_optimization(loop_unroll_and_jam_opt_index, kernel) {
            return false;
        }

        // Unrolling for all loops
        if !self.loop_unrolling_opt.apply_optimization(loop_unrolling_opt_index, kernel) {
            return false;
        }

        // Do a few more iterations to clean up things
        let mut temp_kernel = kernel.clone();
        for _ in 0..100 {
            kernel.move_constants_to_beginning();
            kernel.constant_folding();
            kernel.common_subexpression_elimination();
            kernel.dead_code_elimination();
            kernel.reorder_commutative();
            kernel.loop_invariant_code_motion();
            kernel.delete_empty_loops();
            kernel.unfold_pows();

            if *kernel == temp_kernel {
                break;
            }
            temp_kernel = kernel.clone();
        }

        kernel.verify();

        if debug_ir {
            kernel.debug();
        }

        true
    }

    pub fn new(kernel: &Kernel, dev_info: &DeviceInfo) -> Self {
        let (local_work_size_opt, local_work_size_opt_max_idx) = WorkSizeOpt::new(kernel, dev_info);
        let (loop_unrolling_opt, loop_unrolling_opt_max_idx) = LoopUnrollingOpt::new(kernel);
        let (loop_unroll_and_jam_opt, loop_unroll_and_jam_opt_max_idx) = LoopJamOpt::new(kernel);
        let (loop_split_opt, loop_split_opt_max_idx) = LoopSplitOpt::new(kernel);
        let max_indices = [
            local_work_size_opt_max_idx,
            loop_unroll_and_jam_opt_max_idx,
            loop_unrolling_opt_max_idx,
            loop_split_opt_max_idx,
        ];
        Self {
            max_indices,
            local_work_size_opt,
            loop_unroll_and_jam_opt,
            loop_unrolling_opt,
            loop_split_opt,
            best_optimization: Optimization(0),
            best_time_nanos: u64::MAX,
            tried: HashSet::with_capacity(200),
            rand_iteration: 0,
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
        let opt = if let Some(opt) = self.random_search() {
            Some(opt)
        //} else if let Some(opt) = self.refinement_search(last_time_nanos) {
        //Some(opt)
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

    fn random_search(&mut self) -> Option<Optimization> {
        if self.rand_iteration >= 200 {
            return None;
        }
        self.rand_iteration += 1;
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

#[derive(Debug, Clone, DeBin, SerBin)]
struct WorkSizeOpt {
    gws: Vec<Dim>,
    gws_factors: Vec<Vec<[Dim; 2]>>,
    max_local_threads: Dim,
}

impl WorkSizeOpt {
    fn new(kernel: &Kernel, dev_info: &DeviceInfo) -> (Self, u32) {
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
        for (d, &max_lwd) in gws.iter().copied().zip(&dev_info.max_local_work_dims) {
            let res = divisors(d, max_lwd);

            let mut factors = Vec::new();
            // here factors needs to contain all possible pairs of values in res
            for i in 0..res.len() {
                for j in 0..res.len() {
                    let a = res[i];
                    let b = res[j];
                    if a * b <= d {
                        factors.push([a, b]);
                    }
                }
            }

            gws_factors.push(factors);
        }

        let max_local_threads = dev_info.max_local_threads;

        let max_idx = gws_factors.iter().map(|gd| gd.len() as u32).product();
        (Self { gws, gws_factors, max_local_threads }, max_idx)
    }

    // Returns false if this index is invalid
    #[must_use]
    fn apply_optimization(&self, index: u32, kernel: &mut Kernel) -> bool {
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

        let shape: Vec<Dim> = gws.iter().chain(&lws).chain(&rws).copied().collect();
        let n = kernel.shape().len();
        kernel.apply_movement(|view| view.reshape(0..n, &shape));

        {
            for &dim in rws.iter().rev() {
                let loop_id = kernel.ops.push(Op::Loop { dim, scope: Scope::Register });
                kernel.order.insert(0, loop_id);
            }
            for &dim in lws.iter().rev() {
                let loop_id = kernel.ops.push(Op::Loop { dim, scope: Scope::Local });
                kernel.order.insert(0, loop_id);
            }
            for &dim in gws.iter().rev() {
                let loop_id = kernel.ops.push(Op::Loop { dim, scope: Scope::Global });
                kernel.order.insert(0, loop_id);
            }
        };
        true
    }
}

/// loop unrolling plus loop invariant code motion
#[derive(Debug, Clone, DeBin, SerBin)]
struct LoopJamOpt {}

impl LoopJamOpt {
    fn new(_kernel: &Kernel) -> (Self, u32) {
        (Self {}, 3) // 8, 16, 32 unfolding
    }

    #[must_use]
    fn apply_optimization(&self, index: u32, kernel: &mut Kernel) -> bool {
        let unroll_dim = 8 << index;
        /*let mut op_id = kernel.ops.len();
        while op_id > 0 {
            op_id -= 1;
            if let Op::Loop { dim, scope } = kernel.ops[op_id] {
                if scope == Scope::Register && dim <= unroll_dim {
                    // Check if there is a loop after this (so that we can jam)
                    if kernel.ops[op_id + 1..].iter().any(|op| matches!(op, Op::Loop { .. })) {
                        // A reasonable limit for max kernel size
                        if kernel.ops.len() * dim > 10000 {
                            continue;
                        }
                        // TODO
                        //kernel.loop_jam(op_id);
                    }
                }
            }
        }*/
        true
    }
}

/// loop unrolling plus loop invariant code motion
#[derive(Debug, Clone, DeBin, SerBin)]
struct LoopUnrollingOpt {}

impl LoopUnrollingOpt {
    fn new(_kernel: &Kernel) -> (Self, u32) {
        (Self {}, 4) // 4, 8, 16, 32 unfolding
    }

    #[must_use]
    fn apply_optimization(&self, index: u32, kernel: &mut Kernel) -> bool {
        let unroll_dim = 4 << index;
        let mut op_id = kernel.ops.len();
        /*while op_id > 0 {
            op_id -= 1;
            if let Op::Loop { dim, scope } = kernel.ops[op_id] {
                if scope == Scope::Register && dim <= unroll_dim {
                    // A reasonable limit for max kernel size
                    if kernel.ops.len() * dim > 10000 {
                        continue;
                    }
                    kernel.loop_unroll(op_id);
                }
            }
        }*/
        true
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

#[test]
fn test_get_indices() {
    let opt = Optimization(117);
    let max_values = [20, 10];
    assert_eq!(opt.into_indices(max_values), [11, 7]);

    let opt = Optimization(23902);
    let max_values = [49, 17, 8, 9];
    assert_eq!(opt.into_indices(max_values), [19, 8, 7, 7]);
}

#[derive(Debug, Clone, DeBin, SerBin)]
struct LoopSplitOpt {
    // For each reduction op, store possible split configurations
    // [reduction_op_index][split_configuration][split_dimensions]
    reduction_splits: Vec<Vec<Vec<Dim>>>,
}

impl LoopSplitOpt {
    fn new(kernel: &Kernel) -> (Self, u32) {
        //return (LoopSplitOpt { reduction_splits: Vec::new() }, 10);

        let mut reduction_splits = Vec::new();

        // Find all reduction ops
        for op in kernel.ops.values() {
            if let Op::Reduce { dims, .. } = op {
                // Generate all valid splits for these dimensions
                // Calculate the total product of all dimensions
                let total_product: Dim = dims.iter().product();

                let mut options: Vec<Vec<Dim>> = Vec::new();

                // Add original
                options.push(dims.clone());

                // Generate all possible factorizations of the total product up to max_depth
                for d in 2..64 {
                    if total_product.is_multiple_of(d) {
                        options.push(vec![total_product / d, d]);
                    }
                }

                reduction_splits.push(options);
            }
        }

        let max_index = reduction_splits.iter().map(|splits| splits.len() as u32).product::<u32>();
        (LoopSplitOpt { reduction_splits }, max_index)
    }

    fn apply_optimization(&self, mut index: u32, kernel: &mut Kernel) -> bool {
        // Check if we have any reduction splits
        if self.reduction_splits.is_empty() {
            // No reduction operations found, nothing to optimize
            return true;
        }

        /*let reduce_ops: Vec<OpId> = kernel
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| matches!(op, Op::Reduce { .. }))
            .map(|(op_id, _)| op_id)
            .collect();

        for (i, choices) in self.reduction_splits.iter().enumerate() {
            let n = choices.len() as u32;
            let idx = index % n;
            index /= n;

            let Some(&reduce_id) = reduce_ops.get(i) else { return false };
            kernel.reshape_reduce(reduce_id, &self.reduction_splits[i][idx as usize]);
        }*/

        true
    }
}

// In the future
/*def wmma_pass(ir):
"""
    WMMA pass for already-tiled IR.
    Assumes:
      - Inner loops are fully tiled (autotuner chooses tile dimensions)
      - Accumulators are sized to hold full tile sums
      - Memory accesses are coalesced
    """
for loop in ir.loops:
    # Find inner-most multiply-accumulate / reduce loops
    if is_inner_tile_reduction(loop):
        mul_reduce_loop = loop.inner_mul_reduce  # e.g., BINARY Mul + REDUCE SUM

        # Map entire loop to WMMA fragment
        mul_reduce_loop.wrap_into_wmma_fragment()

        # Map accumulator to WMMA C fragment
        loop.accumulator.map_to_wmma_fragment()

        # Redirect loads/stores to fragment memory
        mul_reduce_loop.replace_loads_with_wmma_fragments()
        mul_reduce_loop.replace_stores_with_wmma_fragments()

        # Outer loops stay the same — iterate over WMMA tiles
return ir*/
