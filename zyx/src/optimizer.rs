use nanoserde::{DeBin, SerBin};

use crate::{backend::DeviceInfo, kernel::Kernel, shape::Dim};
use std::collections::HashSet;

// Indices in 0..max_index for each optimization Opt
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord, Hash, DeBin, SerBin)]
pub struct Optimization(u64);

#[derive(Debug, Clone, DeBin, SerBin)]
pub struct Optimizer {
    // optimizations
    local_work_size_opt: LocalWorkSizeOpt,
    loop_opt: LoopOpt,
    max_indices: [u64; 2],
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
    pub fn new(kernel: &Kernel, dev_info: &DeviceInfo) -> Self {
        let local_work_size_opt = LocalWorkSizeOpt::new(kernel, dev_info);
        let loop_opt = LoopOpt::new(kernel);
        let local_work_size_opt_max_index = local_work_size_opt.max_index();
        let loop_opt_max_index = loop_opt.max_index();
        Self {
            local_work_size_opt,
            loop_opt,
            max_indices: [local_work_size_opt_max_index, loop_opt_max_index],
            best_optimization: Optimization(0),
            best_time_nanos: u128::MAX,
            tried: HashSet::with_capacity(50),
            rand_iteration: 0,
            full_iteration: 0,
            max_iter: local_work_size_opt_max_index * loop_opt_max_index,
            last: Optimization(0),
        }
    }

    pub fn max_iters(&self) -> u64 {
        self.max_iter
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

    fn random_search(&mut self) -> Option<Optimization> {
        if self.rand_iteration >= 50 {
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
            }
        }
        None
    }
}

impl Optimizer {
    pub fn apply_optimization(&self, kernel: &mut Kernel, optimization: Optimization) {
        let [local_work_size_opt_index, loop_opt_index] = optimization.into_indices(self.max_indices);
        self.local_work_size_opt.apply_optimization(local_work_size_opt_index, kernel);

        kernel.unfold_reduces();
        kernel.define_globals();
        kernel.unfold_views();

        let mut temp_kernel = kernel.clone();
        loop {
            kernel.move_constants_to_beginning();
            kernel.constant_folding();
            kernel.common_subexpression_elimination();
            kernel.dead_code_elimination();

            self.loop_opt.apply_optimization(loop_opt_index, kernel);

            if *kernel == temp_kernel {
                break;
            }
            temp_kernel = kernel.clone();
        }
    }
}

#[derive(Debug, Clone, DeBin, SerBin)]
struct LocalWorkSizeOpt {
    gws: Vec<Dim>,
    gws_factors: Vec<Vec<Dim>>,
}

impl LocalWorkSizeOpt {
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
        let gws = kernel.shape();
        let mut gws_factors = Vec::new();
        for (d, &max_lwd) in gws.iter().copied().zip(&dev_info.max_local_work_dims) {
            let res = divisors(d, max_lwd);
            gws_factors.push(res);
        }
        Self { gws, gws_factors }
    }

    fn max_index(&self) -> u64 {
        self.gws_factors.iter().map(|gd| gd.len() as u64).product()
    }

    fn apply_optimization(&self, index: u64, kernel: &mut Kernel) {
        // TODO make this work with limitations for amx local work threads

        let mut value = index as usize;
        let mut result: Vec<usize> = Vec::new();
        for max in self.gws_factors.iter().map(|f| f.len()).rev() {
            result.push(value % max);
            value /= max;
        }
        result.reverse();

        let mut lws = Vec::new();
        for (i, factors) in result.into_iter().zip(&self.gws_factors) {
            lws.push(factors[i]);
        }

        let mut gws = self.gws.clone();
        for (g, l) in gws.iter_mut().zip(&lws) {
            *g /= l;
        }

        let shape: Vec<Dim> = gws.iter().chain(&lws).copied().collect();
        let n = kernel.shape().len();
        kernel.apply_movement(|view| view.reshape(0..n, &shape));
        kernel.unfold_shape(&gws, &lws);
    }
}

/// loop unrolling plus loop invariant code motion
#[derive(Debug, Clone, DeBin, SerBin)]
struct LoopOpt {}

impl LoopOpt {
    fn new(kernel: &Kernel) -> Self {
        Self {}
    }

    fn max_index(&self) -> u64 {
        1
    }

    fn apply_optimization(&self, index: u64, kernel: &mut Kernel) {
        // TODO
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
