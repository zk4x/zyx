use nanoserde::{DeBin, SerBin};

use crate::{
    backend::DeviceInfo,
    dtype::Constant,
    kernel::{Kernel, Op, OpId, Scope, increment},
    shape::Dim,
};
use std::{collections::HashSet, ops::Range};

// Indices in 0..max_index for each optimization Opt
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash, DeBin, SerBin)]
pub struct Optimization(u64);

#[derive(Debug, Clone, DeBin, SerBin)]
pub struct Optimizer {
    // optimizations
    local_work_size_opt: WorkSizeOpt,
    loop_unrolling_opt: LoopUnrollingOpt,
    loop_split_opt: LoopSplitOpt,
    //inner_loop_swap_opt: InnerLoopSwapOpt, // a bit harder to know max number of optimizations
    max_indices: [u64; 3],
    // best optimization found so far
    best_optimization: Optimization,
    // time taken by kernel with the best optimization
    pub best_time_nanos: u128,
    // Which iteration are we on? First 30 iterations are random, then 20 iterations of refinement
    // and remaider is just going over all possible optimizations in deterministic order
    rand_iteration: u64,
    full_iteration: u64,
    max_iter: u64,
    // Optimizations that were tried during random search and refinement
    tried: HashSet<Optimization>,
    // Last tried optimization
    last: Optimization,
}

impl Optimizer {
    #[must_use]
    pub fn apply_optimization(&self, kernel: &mut Kernel, optimization: Optimization) -> bool {
        let [local_work_size_opt_index, loop_opt_index, loop_split_opt_index] =
            optimization.into_indices(self.max_indices);

        if !self.local_work_size_opt.apply_optimization(local_work_size_opt_index, kernel) {
            return false;
        }

        if !self.loop_split_opt.apply_optimization(loop_split_opt_index, kernel) {
            return false;
        }

        kernel.unfold_pows();
        kernel.unfold_reduces();
        kernel.define_globals();
        kernel.unfold_views();
        kernel.close_loops();

        let mut temp_kernel = kernel.clone();
        for _ in 0..100 {
            // Limit max optimization iterations
            kernel.move_constants_to_beginning();
            kernel.constant_folding();
            kernel.common_subexpression_elimination();
            kernel.dead_code_elimination();

            if !self.loop_unrolling_opt.apply_optimization(loop_opt_index, kernel) {
                return false;
            }

            if *kernel == temp_kernel {
                break;
            }
            temp_kernel = kernel.clone();
        }
        true
    }

    pub fn new(kernel: &Kernel, dev_info: &DeviceInfo) -> Self {
        let local_work_size_opt = WorkSizeOpt::new(kernel, dev_info);
        let loop_unrolling_opt = LoopUnrollingOpt::new(kernel);
        let loop_split_opt = LoopSplitOpt::new(kernel, 3);
        let max_indices = [
            local_work_size_opt.max_index(),
            loop_unrolling_opt.max_index(),
            loop_split_opt.max_index(),
        ];
        Self {
            max_indices,
            local_work_size_opt,
            loop_unrolling_opt,
            loop_split_opt,
            best_optimization: Optimization(0),
            best_time_nanos: u128::MAX,
            tried: HashSet::with_capacity(200),
            rand_iteration: 0,
            full_iteration: 0,
            max_iter: max_indices.iter().product(),
            last: Optimization(0),
        }
    }

    pub fn max_iters(&self) -> u64 {
        self.max_iter
    }

    pub fn best_optimization(&self) -> Optimization {
        self.best_optimization
    }

    pub fn next_optimization(&mut self, last_time_nanos: u128) -> Option<Optimization> {
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
        let mut rng = crate::rng::Rng::seed_from_systime();
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
            } else {
                self.tried.remove(&Optimization(temp));
            }
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
    fn new(kernel: &Kernel, dev_info: &DeviceInfo) -> Self {
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
        Self { gws, gws_factors, max_local_threads }
    }

    fn max_index(&self) -> u64 {
        self.gws_factors.iter().map(|gd| gd.len() as u64).product()
    }

    // Returns false if this index is invalid
    #[must_use]
    fn apply_optimization(&self, index: u64, kernel: &mut Kernel) -> bool {
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
            let k = gws.len() + lws.len() + rws.len();
            let n = kernel.ops.len();
            increment(&mut kernel.ops, k, 0..n);
            for &dim in rws.iter().rev() {
                kernel.ops.insert(0, Op::Loop { dim, scope: Scope::Register });
            }
            for &dim in lws.iter().rev() {
                kernel.ops.insert(0, Op::Loop { dim, scope: Scope::Local });
            }
            for &dim in gws.iter().rev() {
                kernel.ops.insert(0, Op::Loop { dim, scope: Scope::Global });
            }
        };
        true
    }
}

/// loop unrolling plus loop invariant code motion
#[derive(Debug, Clone, DeBin, SerBin)]
struct LoopUnrollingOpt {}

impl LoopUnrollingOpt {
    fn new(_kernel: &Kernel) -> Self {
        Self {}
    }

    fn max_index(&self) -> u64 {
        1
    }

    #[must_use]
    fn apply_optimization(&self, _index: u64, _kernel: &mut Kernel) -> bool {
        // TODO
        true
    }

    /// Unroll all loops with dimension <= `loop_unroll_size`
    #[allow(unused)]
    fn loop_optimization(kernel: &mut Kernel, loop_unroll_size: usize) {
        fn unroll_loop(ir: &mut Vec<Op>, range: Range<usize>) {
            let Op::Loop { dim, .. } = ir[range.start] else {
                unreachable!("Expected Op::Loop at start of matched loop range");
            };

            let mut body = ir.split_off(range.start);
            let mut tail = body.split_off(range.end - range.start);
            body.pop();

            // If body contains accumulator, we replace it with binary ops and DeclareAcc with constant
            /*let mut replace_acc = if body.iter().any(|op| matches!(op, Op::Accumulate { .. })) {
                ir.iter().rposition(|op| matches!(op, Op::DeclareAcc { .. }))
            } else {
                None
            };
            if let Some(decl_acc_id) = replace_acc {
                if let Op::DeclareAcc { dtype, rop } = ir[decl_acc_id] {
                    ir[decl_acc_id] = Op::Const(match rop {
                        ROp::Sum => dtype.zero_constant(),
                        ROp::Max => dtype.min_constant(),
                    });
                }
            }*/

            // Append body dim times
            for i in 0..dim {
                let mut body = body.clone();
                let n = body.len();
                increment(&mut body, i * n, range.clone());
                body[0] = Op::Const(Constant::U32(i as u32));

                /*if let Some(decl_acc_id) = replace_acc {
                    for (op_id, op) in body.iter_mut().enumerate() {
                        if let &mut Op::Accumulate { x, rop } = op {
                            *op = Op::Binary {
                                x,
                                y: decl_acc_id,
                                bop: match rop {
                                    ROp::Sum => BOp::Add,
                                    ROp::Max => BOp::Max,
                                },
                            };
                            replace_acc = Some(op_id + ir.len());
                            break;
                        }
                    }
                }*/

                ir.extend(body);
            }

            increment(&mut tail, (dim - 1) * body.len() - 1, range.end..usize::MAX);
            increment(&mut tail, (dim - 1) * body.len(), range);
            ir.extend(tail);

            /*for (i, op) in ir.iter().enumerate() {
                println!("{i} -> {op:?}");
            }*/
        }

        /*fn loop_invariant_code_motion(ir: &mut Vec<Op>, range: Range<usize>) {
            for op_id in range {
                match &ir[op_id] {
                    Op::ConstView { value, view } => todo!(),
                    Op::LoadView { dtype, view } => todo!(),
                    Op::Reduce { x, rop, dims } => todo!(),
                    Op::Const(constant) => todo!(),
                    Op::Load { dtype, index, arg_id } => todo!(),
                    Op::DeclareAcc { dtype, rop } => todo!(),
                    Op::Loop { dim, vectorize } => todo!(),
                    Op::Accumulate { x, rop } => todo!(),
                    Op::EndLoop => todo!(),
                    Op::Store { x, index } => todo!(),
                    Op::Cast { x, dtype } => todo!(),
                    Op::Unary { x, uop } => todo!(),
                    Op::Binary { x, y, bop } => todo!(),
                }
            }
        }*/

        let mut ranges = Vec::new();
        let mut stack = Vec::new();

        for (i, op) in kernel.ops.iter().enumerate() {
            match op {
                Op::Loop { dim, .. } => {
                    stack.push((i, dim));
                }
                &Op::EndLoop => {
                    if let Some((start, dim)) = stack.pop()
                        && *dim <= loop_unroll_size
                    {
                        ranges.push(start..i + 1);
                    }
                }
                _ => {}
            }
        }
        //println!("{ranges:?}");

        for range in ranges {
            unroll_loop(&mut kernel.ops, range);
        }
    }
}

impl Optimization {
    fn into_indices<const N: usize>(self, max_values: [u64; N]) -> [u64; N] {
        let mut value = self.0;
        debug_assert!(
            value < max_values.iter().product(),
            "Index {self:?} out of range for {max_values:?}."
        );
        let mut result: [u64; N] = [0; N];

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
    fn new(kernel: &Kernel, max_depth: usize) -> Self {
        let mut reduction_splits = Vec::new();

        // Find all reduction ops
        for (_, op) in kernel.ops.iter().enumerate() {
            if let Op::Reduce { dims, .. } = op {
                // Generate all valid splits for these dimensions
                let splits = Self::generate_splits(dims, max_depth);
                reduction_splits.push(splits);
            }
        }

        LoopSplitOpt { reduction_splits }
    }

    fn max_index(&self) -> u64 {
        self.reduction_splits.iter().map(|splits| splits.len() as u64).product::<u64>()
    }

    fn generate_splits(dims: &[Dim], max_depth: usize) -> Vec<Vec<Dim>> {
        // Calculate the total product of all dimensions
        let total_product: Dim = dims.iter().product();

        // Generate all possible factorizations of the total product up to max_depth
        let mut current = Vec::new();
        let mut results = Vec::new();

        // Inline factorization generation
        fn find_factorizations(
            remaining: Dim,
            start: Dim,
            max_depth: usize,
            current: &mut Vec<Dim>,
            results: &mut Vec<Vec<Dim>>,
        ) {
            if current.len() == max_depth || remaining == 1 {
                if !current.is_empty() {
                    results.push(current.clone());
                }
                return;
            }

            for i in start..=remaining {
                if remaining % i == 0 {
                    current.push(i);
                    find_factorizations(remaining / i, i, max_depth, current, results);
                    current.pop();
                }
            }
        }

        find_factorizations(total_product, 1, max_depth, &mut current, &mut results);

        // Filter out splits that contain dimensions with length 1
        results.retain(|split| !split.contains(&1));

        // Only keep splits where the product equals the total product
        results.retain(|split| split.iter().product::<Dim>() == total_product);

        // Add the original dimensions as a valid option (no split)
        results.push(dims.to_vec());

        results
    }

    fn apply_optimization(&self, index: u64, kernel: &mut Kernel) -> bool {
        // Check if we have any reduction splits
        if self.reduction_splits.is_empty() {
            // No reduction operations found, nothing to optimize
            return true;
        }

        // TODO This code has bad vibes. It should by able to apply splits to multiple loops at once.
        let (reduction_idx, split_idx) = self.decode_index(index);

        // 2. Find the target reduction operation using filter
        let reduce_ops: Vec<OpId> = kernel
            .ops
            .iter()
            .enumerate()
            .filter(|(_, op)| matches!(op, Op::Reduce { .. }))
            .map(|(op_id, _)| op_id)
            .collect();
        let Some(&reduce_op_id) = reduce_ops.get(reduction_idx) else { return false };

        // 3. Get the split dimensions and replace the dims in the reduction op
        let split_dims = self.reduction_splits[reduction_idx][split_idx].clone();

        // Replace the dims in the reduction operation
        let n_reduce_dims;
        if let Op::Reduce { ref mut dims, .. } = kernel.ops[reduce_op_id] {
            n_reduce_dims = dims.len();
            *dims = split_dims.clone();
        } else {
            return false; // Should not happen since we checked it's a Reduce op
        }

        let min_param = self.find_min_param(kernel, reduce_op_id);
        let mut n_skipped_axes = 0;
        for (op_id, op) in kernel.ops.iter_mut().enumerate() {
            match op {
                Op::ConstView { view, .. } => {
                    if op_id >= min_param && op_id < reduce_op_id {
                        let rank = view.rank();
                        view.reshape(
                            rank - n_skipped_axes - n_reduce_dims..rank - n_skipped_axes,
                            &split_dims,
                        );
                    }
                }
                Op::LoadView { view, .. } => {
                    if op_id >= min_param && op_id < reduce_op_id {
                        let rank = view.rank();
                        view.reshape(
                            rank - n_skipped_axes - n_reduce_dims..rank - n_skipped_axes,
                            &split_dims,
                        );
                    }
                }
                Op::Reduce { dims, .. } => {
                    n_skipped_axes += dims.len();
                }
                _ => {}
            }
        }

        true
    }

    fn decode_index(&self, index: u64) -> (usize, usize) {
        let mut idx = index as usize;
        for (i, splits) in self.reduction_splits.iter().enumerate() {
            if idx < splits.len() {
                return (i, idx);
            }
            idx -= splits.len();
        }
        panic!("Index {index} out of bounds");
    }

    fn find_min_param(&self, kernel: &Kernel, reduce_op_id: usize) -> usize {
        // Reuse the tracing algorithm from kernel.unfold_reduces (lines 422-502)
        let Op::Reduce { x, .. } = kernel.ops[reduce_op_id] else { unreachable!() };

        let mut min_param = x;
        let mut params = vec![x];

        while let Some(param) = params.pop() {
            match kernel.ops[param] {
                Op::Load { src, .. } => {
                    params.push(src);
                    if src < min_param {
                        min_param = src;
                    }
                }
                Op::Store { x: src, index, .. } => {
                    params.push(index);
                    if index < min_param {
                        min_param = index;
                    }
                    params.push(src);
                    if src < min_param {
                        min_param = src;
                    }
                }
                Op::Cast { x, .. } | Op::Unary { x, .. } | Op::Reduce { x, .. } => {
                    params.push(x);
                    if x < min_param {
                        min_param = x;
                    }
                }
                Op::Binary { x, y, .. } => {
                    params.push(x);
                    if x < min_param {
                        min_param = x;
                    }
                    params.push(y);
                    if y < min_param {
                        min_param = y;
                    }
                }
                _ => {}
            }
        }

        min_param
    }
}

#[test]
fn test_generate_splits_fix() {
    // Test the case mentioned in the issue: dims = [512], max_depth = 3
    let dims = [512];
    let max_depth = 3;
    let splits = LoopSplitOpt::generate_splits(&dims, max_depth);

    println!("Generated splits for dims = {:?}, max_depth = {}:", dims, max_depth);
    for (i, split) in splits.iter().enumerate() {
        let product: usize = split.iter().product();
        println!("  {}: {:?} (product = {})", i + 1, split, product);
    }

    // Verify that all splits have product equal to 512
    for split in &splits {
        let product: usize = split.iter().product();
        assert_eq!(product, 512, "Split {:?} has product {} != 512", split, product);
    }

    // Verify that we don't have splits with more than max_depth elements
    for split in &splits {
        assert!(
            split.len() <= max_depth,
            "Split {:?} has length {} > max_depth {}",
            split,
            split.len(),
            max_depth
        );
    }

    // Verify that we don't have splits containing 1
    for split in &splits {
        assert!(!split.contains(&1), "Split {:?} contains 1", split);
    }

    // Test with multiple dimensions
    let dims = [256, 368, 512];
    let splits = LoopSplitOpt::generate_splits(&dims, 2);

    println!("\nGenerated splits for dims = {:?}, max_depth = {}:", dims, 2);
    for (i, split) in splits.iter().take(10).enumerate() {
        let product: usize = split.iter().product();
        println!("  {}: {:?} (product = {})", i + 1, split, product);
    }
    if splits.len() > 10 {
        println!("  ... and {} more splits", splits.len() - 10);
    }

    // Verify that all splits have product equal to 256*368*512
    let expected_product: usize = dims.iter().product();
    for split in &splits {
        let product: usize = split.iter().product();
        assert_eq!(
            product, expected_product,
            "Split {:?} has product {} != {}",
            split, product, expected_product
        );
    }
}
