use crate::{
    backend::DeviceInfo,
    kernel::Kernel,
    optimizer::{
        loop_jam::LoopJamOpt, loop_split::LoopSplitOpt, loop_unrolling::LoopUnrollOpt, work_size::WorkSizeOpt,
    },
};
use nanoserde::{DeBin, SerBin};
use std::collections::HashSet;

mod loop_jam;
mod loop_split;
mod loop_unrolling;
mod vectorize;
mod work_size;

// Indices in 0..max_index for each optimization Opt
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash, DeBin, SerBin)]
pub struct Optimization(u32);

#[derive(Debug, Clone, DeBin, SerBin)]
pub struct Optimizer {
    // optimizations
    work_size_opt: WorkSizeOpt,
    loop_unroll_and_jam_opt: LoopJamOpt,
    loop_unrolling_opt: LoopUnrollOpt,
    loop_split_opt: LoopSplitOpt,
    //inner_loop_swap_opt: InnerLoopSwapOpt, // a bit harder to know max number of optimizations
    max_indices: [u32; 4],
    default_indices: [Vec<u32>; 4],
    // best optimization found so far
    best_optimization: Optimization,
    // time taken by kernel with the best optimization
    pub best_time_nanos: u64,
    // Which iteration are we on? First 30 iterations are random, then 20 iterations of refinement
    // and remaider is just going over all possible optimizations in deterministic order
    default_iteration: u32,
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

        if !self.work_size_opt.apply_optimization(local_work_size_opt_index, kernel) {
            return false;
        }
        kernel.close_loops();

        if !self.loop_split_opt.apply_optimization(loop_split_opt_index, kernel) {
            return false;
        }

        kernel.unfold_reduces();
        kernel.unfold_views();

        kernel.unroll_loops(1);
        kernel.swap_commutative();
        kernel.reassociate_commutative();

        // This is only needed for debugging
        /*let mut temp_kernel = kernel.clone();
        for _i in 0..100 {
            kernel.move_constants_to_beginning();
            kernel.constant_folding();
            kernel.common_subexpression_elimination();
            kernel.swap_commutative();
            kernel.loop_invariant_code_motion();
            kernel.delete_empty_loops();
            kernel.dead_code_elimination();

            if *kernel == temp_kernel {
                break;
            }
            temp_kernel = kernel.clone();
            #[cfg(debug_assertions)]
            if _i == 99 {
                kernel.debug();
                panic!("YO what are you doing bro.");
            }
        }*/

        // Unroll and jam for all loops
        if !self.loop_unroll_and_jam_opt.apply_optimization(loop_unroll_and_jam_opt_index, kernel) {
            return false;
        }

        // Unrolling for all loops
        if !self.loop_unrolling_opt.apply_optimization(loop_unrolling_opt_index, kernel) {
            return false;
        };

        // We have to do constant folding before folding accs to guarantee indices are constants
        kernel.constant_folding();
        kernel.fold_accs();

        // Convert exponentiation (BOp::Pow) to just exp2 and ln2
        kernel.unfold_pows();
        kernel.swap_commutative();
        kernel.reassociate_commutative();

        let mut temp_kernel = kernel.clone();
        for _ in 0..10 {
            kernel.move_constants_to_beginning();
            kernel.swap_commutative();
            kernel.constant_folding();
            kernel.common_subexpression_elimination();
            kernel.loop_invariant_code_motion();
            kernel.delete_empty_loops();
            kernel.dead_code_elimination();

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
        let (work_size_opt, work_size_opt_max_idx, work_size_opt_defaults) = WorkSizeOpt::new(kernel, dev_info);
        let (loop_unroll_opt, loop_unroll_opt_max_idx, loop_unroll_opt_defaults) = LoopUnrollOpt::new(kernel);
        let (loop_jam_opt, loop_jam_opt_max_idx, loop_jam_opt_defaults) = LoopJamOpt::new(kernel, dev_info);
        let (loop_split_opt, loop_split_opt_max_idx, loop_split_opt_defaults) = LoopSplitOpt::new(kernel);
        let max_indices = [
            work_size_opt_max_idx,
            loop_unroll_opt_max_idx,
            loop_jam_opt_max_idx,
            loop_split_opt_max_idx,
        ];
        let default_indices = [
            work_size_opt_defaults,
            loop_unroll_opt_defaults,
            loop_jam_opt_defaults,
            loop_split_opt_defaults,
        ];

        //println!( "Optimizing work_size_opt_max_idx={work_size_opt_max_idx},\nloop_jam_opt_max_idx={loop_jam_opt_max_idx},\nloop_unrolling_opt_max_idx={loop_unroll_opt_max_idx},\nloop_split_opt_max_idx={loop_split_opt_max_idx}" );
        //println!("Max default opts: {:?}", default_indices);

        Self {
            max_indices,
            default_indices,
            work_size_opt,
            loop_unroll_and_jam_opt: loop_jam_opt,
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

    /*pub fn default_search(&mut self) -> Option<Optimization> {
        if self.default_iteration >= 200 {
            return None;
        }
        self.default_iteration += 1;

        // TODO make this fast by checking the last opt and incrementing from there
        // --- Step 1: try defaults on-the-fly ---
        // Loop over all combinations of default_indices
        for &i0 in &self.default_indices[0] {
            for &i1 in &self.default_indices[1] {
                for &i2 in &self.default_indices[2] {
                    for &i3 in &self.default_indices[3] {
                        // Flatten into single Optimization(u32)
                        let mut flat_idx = 0;
                        let mut multiplier = 1;
                        let indices = [i0, i1, i2, i3];
                        for (i, &max) in indices.iter().rev().zip(self.max_indices.iter().rev()) {
                            flat_idx += i * multiplier;
                            multiplier *= max;
                        }

                        let opt = Optimization(flat_idx);
                        if self.tried.insert(opt) {
                            self.last = opt;
                            return Some(opt);
                        }
                    }
                }
            }
        }

        // --- Step 2: fallback to random search over full Cartesian product ---
        let mut rng = crate::rng::Rng::seed_from_u64(642392);
        for _ in 0..1_000_000 {
            let index = rng.range(0..self.max_iter);
            let opt = Optimization(index);
            if self.tried.insert(opt) {
                self.last = opt;
                return Some(opt);
            }
        }

        // --- Step 3: exhausted ---
        None
    }*/

    fn random_search(&mut self) -> Option<Optimization> {
        if self.default_iteration >= 200 {
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

#[test]
fn test_get_indices() {
    let opt = Optimization(117);
    let max_values = [20, 10];
    assert_eq!(opt.into_indices(max_values), [11, 7]);

    let opt = Optimization(23902);
    let max_values = [49, 17, 8, 9];
    assert_eq!(opt.into_indices(max_values), [19, 8, 7, 7]);
}

// We can have optimizers specific for certain GPUs, like this optimizer optimizes for RX 580, etc.

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
