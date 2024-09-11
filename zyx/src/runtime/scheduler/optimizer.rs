use std::fmt::Display;

use crate::{runtime::{backend::DeviceInfo, scheduler::VOp, view::View}, shape::Dimension};
use super::kernel::Kernel;

#[derive(Debug, bitcode::Encode, bitcode::Decode)]
pub(super) enum KernelOptimizer {
    Optimized(KernelOptimization, u128),
    // All optimizations, best optimization id
    Optimizing(Vec<(KernelOptimization, u128)>, usize),
}

// Optimizations get applied to existing kernels after
// they are assigned to devices.
#[derive(Debug, Clone, bitcode::Encode, bitcode::Decode, PartialEq, Eq)]
pub(crate) struct KernelOptimization {
    // Axis splits to give us global, local and register work sizes
    // as well as work per thread in reduce loops
    pub(crate) splits: Vec<(usize, Vec<Dimension>)>,
    // Permutation so that global and local work sizes are first
    pub(crate) permutation: Vec<usize>,
    // Enable local tiling
    pub(crate) local_tiles: bool,

    // Load tensor first into local tile, then into registers
    // this is used mainly for expanded tensors, so use threads
    // from one local work group to load the tile and then sync loads
    // before loading into registers
    //local_tiles: Vec<(TensorId, View)>,
    // Unrolls loop with given id
    //unroll_loops: Vec<usize>,
    // Converts all variables in loop into native vector dtypes
    // and removes the loop.
    //vectorize_loops: Vec<usize>,
    // Tile tensor in registers with given view
    //register_tiles: Vec<(TensorId, View)>,
    // TensorCores,
    // WMMA
}

// Probably just do all the optimizations including tensor cores here,
// ir will be just a direct translation and can be removed if we replace it with something
// like renderer to c style, assembly and such.
impl Kernel {
    pub(super) fn new_optimizer(&self, dev_info: &DeviceInfo) -> KernelOptimizer {
        let mut opts = Vec::new();

        //let mgwd = dev_info.max_global_work_dims;
        let mlws = dev_info.max_local_threads;
        let mlwd = dev_info.max_local_work_dims;
        let mrws = dev_info.num_registers;
        let mrwd = [16, 16, 16]; // For now 16, can be raised to 32 on some hardware perhaps
        let maxrr = 8; // Max reduce register work dimension

        let mut gws = [0; 3];
        let mut gws_i = 3;
        let mut splits = Vec::new();
        let num_loops = self
            .ops
            .iter()
            .position(|op| {
                if let VOp::Loop { len: dimension, .. } = op {
                    gws_i -= 1;
                    gws[gws_i] = *dimension;
                }
                !matches!(op, VOp::Loop { .. })
            })
            .unwrap();
        assert_ne!(num_loops, 0);
        if num_loops < 3 {
            let dims: Vec<usize> = core::iter::repeat(1)
                .take(3 - num_loops)
                .chain([self.shape[0]])
                .collect();
            gws_i += 1;
            for dim in dims.iter().rev() {
                gws_i -= 1;
                gws[gws_i] = *dim;
            }
            splits.push((0, dims));
        }
        //println!("Using gws {gws:?}");
        // Local work size
        for lx in (1..=mlws.min(mlwd[0])).filter(|x| gws[0] % x == 0) {
            for ly in (1..=(mlws/lx).min(mlwd[1])).filter(|y| gws[1] % y == 0) {
                for lz in (1..=(mlws/(lx*ly)).min(mlwd[2])).filter(|z| gws[2] % z == 0) {
                    // register work size
                    for rx in (1..=mrws.min(mrwd[0])).filter(|x| (gws[0]/lx) % x == 0) {
                        for ry in (1..=(mrws/rx).min(mrwd[1])).filter(|y| (gws[1]/ly) % y == 0) {
                            for rz in (1..=(mrws/(rx*ry)).min(mrwd[2])).filter(|z| (gws[2]/lz) % z == 0) {
                                // Get splits for local and global work dims
                                let mut splits = splits.clone();
                                splits.push((2, vec![gws[2]/(lz*rz), lz, rz]));
                                splits.push((1, vec![gws[1]/(ly*ry), ly, ry]));
                                splits.push((0, vec![gws[0]/(lx*rx), lx, rx]));

                                // For each reduce loop
                                let mut acc_found = false;
                                let mut loop_found = false;
                                let mut reduce_found = false;
                                // Find first non loop op after loop after accumulator
                                // that op_id - 1 is loop and there we split
                                for (id, op) in self.ops[num_loops..].iter().enumerate() {
                                    if loop_found && !matches!(op, VOp::Loop { .. }) {
                                        loop_found = false;
                                        acc_found = false;
                                        reduce_found = true;
                                        let VOp::Loop { len, .. } = self.ops[id-1+num_loops] else { panic!() };
                                        // Register work size in the reduce loop
                                        for rr in (1..=maxrr).filter(|rr| len % rr == 0) {
                                            // Get splits for local and global work dims
                                            let mut splits = splits.clone();
                                            splits.insert(0, (id-1+num_loops, vec![len/rr, rr]));
                                            // Permute, private loops last
                                            opts.push((KernelOptimization {
                                                splits,
                                                permutation: vec![0, 1, 2, 3, 4, 5, 6, 7],
                                                local_tiles: true,
                                            }, 0));
                                        }
                                    }
                                    if acc_found && matches!(op, VOp::Loop { .. }) {
                                        //println!("Loop found at {id}");
                                        loop_found = true;
                                    }
                                    if matches!(op, VOp::Accumulator { .. }) {
                                        //println!("Acc found at {id}");
                                        acc_found = true;
                                    }
                                }
                                if !reduce_found {
                                    // Permute, private loops last
                                    opts.push((KernelOptimization {
                                        splits,
                                        permutation: vec![0, 1, 2, 3, 4, 5, 6, 7],
                                        local_tiles: true,
                                    }, 0));
                                }
                            }
                        }
                    }
                }
            }
        }
        KernelOptimizer::Optimizing(opts, 0)
    }

    // add per device optimizations to each kernel, local memory, accumulators, work per thread, tiling on many levels,
    // split, merge, permute, pad loops and get them to correct dimensionality (3d) for execution on the device.
    // tensor cores, just a ton of stuff. Later add search over different optimizations.
    pub(super) fn optimize(&self, optimization: &KernelOptimization) -> Kernel {
        let mut kernel = self.clone();
        // Apply axis splits
        for (op_id, dimensions) in &optimization.splits {
            kernel.split_axis(*op_id, dimensions);
        }
        let mut rws = [0; 3];
        let VOp::Loop { len, .. } = kernel.ops[2] else { panic!() };
        rws[0] = len;
        let VOp::Loop { len, .. } = kernel.ops[5] else { panic!() };
        rws[1] = len;
        let VOp::Loop { len, .. } = kernel.ops[8] else { panic!() };
        rws[2] = len;
        // Apply permutation
        kernel.permute(&optimization.permutation);

        // Reorder so that register work threads are last
        // Register threads are op_id 1, 4 and 7
        let mut threaded = true;
        let rlz = kernel.ops.remove(8);
        let rly = kernel.ops.remove(5);
        let rlx = kernel.ops.remove(2);
        kernel.ops.insert(6, rlz.clone());
        kernel.ops.insert(6, rly.clone());
        kernel.ops.insert(6, rlx.clone());
        let mut id = 9;
        while id < kernel.ops.len() {
            if threaded && matches!(kernel.ops[id], VOp::Loop { .. }) {
                kernel.ops.insert(id, VOp::EndLoop);
                kernel.ops.insert(id, VOp::EndLoop);
                kernel.ops.insert(id, VOp::EndLoop);
                id += 4;
                threaded = false;
                continue;
            }
            if threaded && matches!(kernel.ops[id], VOp::EndLoop) {
                kernel.ops.insert(id, VOp::EndLoop);
                kernel.ops.insert(id, VOp::EndLoop);
                kernel.ops.insert(id, VOp::EndLoop);
                id += 4;
                threaded = false;
                continue;
            }
            if !threaded && !matches!(kernel.ops[id], VOp::Loop { .. } | VOp::EndLoop) {
                kernel.ops.insert(id, rlz.clone());
                kernel.ops.insert(id, rly.clone());
                kernel.ops.insert(id, rlx.clone());
                id += 4;
                threaded = true;
                continue;
            }
            id += 1;
        }
        // Since we have swaped our threads around, we need bigger accumulator,
        // otherwise the results would be incorrect
        for op in &mut kernel.ops {
            match op {
                VOp::Accumulator { view, .. } => {
                    *view = View::binded(&rws, &[2, 5, 8]);
                }
                _ => {}
            }
        }

        // Local tiling
        if optimization.local_tiles {
            // Local tile all loads that do not use all loop axes
            // Local tiles use local dimensions and register dimensions
            // i.e. [rws[0]*lws[0], rws[1]*lws[1], rws[2]*lws[2]]
            // TODO
        }
        kernel
    }
}

impl Display for KernelOptimization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("splits {:?}, permute: {:?}", self.splits, self.permutation))
    }
}

impl KernelOptimizer {
    // Get next optimization, returns None if fully optimized
    pub(super) fn next(&mut self) -> Option<usize> {
        // Ideally we want to pick random value from normal distribution
        // where mean would be around the best time, we would set standard deviation
        // and the value would be in range 0..opt.len(), but exclude those that have nonzero exec time

        match self {
            KernelOptimizer::Optimized(_, _) => None,
            KernelOptimizer::Optimizing(opts, best) => {
                use rand::seq::SliceRandom;
                use rand::SeedableRng;
                let values: Vec<usize> = (0..opts.len()).filter(|&id| opts[id].1 == 0).collect();
                if values.len() == 0 {
                    *self = KernelOptimizer::Optimized(opts[*best].0.clone(), opts[*best].1);
                }
                let mut rng = rand::rngs::SmallRng::seed_from_u64(190940981234098124);
                values.choose(&mut rng).copied()
            }
        }
    }

    pub(super) fn set_exec_time(&mut self, optimization_id: usize, exec_time: u128) {
        if let KernelOptimizer::Optimizing(opts, best) = self {
            opts[optimization_id].1 = exec_time;
            if exec_time < opts[*best].1 || opts[*best].1 == 0 {
                *best = optimization_id;
            }
        }
    }

    pub(super) fn best(&self) -> &KernelOptimization {
        match self {
            KernelOptimizer::Optimized(optimization, _) => optimization,
            KernelOptimizer::Optimizing(opts, best) => &opts[*best].0,
        }
    }

    pub(super) fn remaining(&self) -> usize {
        match self {
            KernelOptimizer::Optimized(_, _) => 0,
            KernelOptimizer::Optimizing(opts, _) => (0..opts.len()).filter(|&id| opts[id].1 == 0).count(),
        }
    }
}

impl std::ops::Index<usize> for KernelOptimizer {
    type Output = KernelOptimization;
    fn index(&self, index: usize) -> &Self::Output {
        match self {
            KernelOptimizer::Optimized(opt, _) => opt,
            KernelOptimizer::Optimizing(opts, _) => &opts[index].0,
        }
    }
}
