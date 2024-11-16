use std::{collections::BTreeSet, fmt::Display};

use super::kernel::Kernel;
use crate::{
    runtime::{backend::DeviceInfo, ir::Scope, scheduler::VOp, view::View},
    shape::Dimension,
};

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug)]
pub enum KernelOptimizer {
    #[allow(dead_code)]
    Optimized(KernelOptimization, u128),
    // All optimizations, best optimization id
    Optimizing(Vec<(KernelOptimization, u128)>, usize),
}

// Optimizations get applied to existing kernels after
// they are assigned to devices.
#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelOptimization {
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
    #[allow(clippy::similar_names)]
    pub(super) fn new_optimizer(&self, dev_info: &DeviceInfo) -> KernelOptimizer {
        let mut opts = Vec::new();

        //let mgwd = dev_info.max_global_work_dims;
        let mlws = dev_info.max_local_threads;
        let mlwd = dev_info.max_local_work_dims;
        let mrws = dev_info.num_registers;
        let mrwd = [16, 16, 16]; // For now 16, can be raised to 32 on some hardware perhaps
        let maxrr = 8; // Max reduce register work dimension

        let mut splits = Vec::new();
        let num_loops = self
            .ops
            .iter()
            .position(|op| !matches!(op, VOp::Loop { .. }))
            .unwrap();
        assert_ne!(num_loops, 0);
        let mut gws = [1; 3];
        if num_loops < 3 {
            let dims: Vec<usize> = core::iter::repeat(1)
                .take(3 - num_loops)
                .chain([self.shape()[0]])
                .collect();
            splits.push((0, dims));
            let mut gws_i = 3 - num_loops;
            for d in &self.shape() {
                gws[gws_i] = *d;
                gws_i += 1;
            }
        } else {
            for (gws_d, d) in gws.iter_mut().zip(self.shape()[..3].iter()) {
                *gws_d = *d;
            }
        }
        //println!("Using gws {gws:?}");
        // Local work size
        for lx in (1..=mlws.min(mlwd[0])).filter(|x| gws[0] % x == 0) {
            for ly in (1..=(mlws / lx).min(mlwd[1])).filter(|y| gws[1] % y == 0) {
                for lz in (1..=(mlws / (lx * ly)).min(mlwd[2])).filter(|z| gws[2] % z == 0) {
                    // register work size
                    for rx in (1..=mrws.min(mrwd[0])).filter(|x| (gws[0] / lx) % x == 0) {
                        for ry in (1..=(mrws / rx).min(mrwd[1])).filter(|y| (gws[1] / ly) % y == 0)
                        {
                            for rz in (1..=(mrws / (rx * ry)).min(mrwd[2]))
                                .filter(|z| (gws[2] / lz) % z == 0)
                            {
                                // Get splits for local and global work dims
                                let mut splits = splits.clone();
                                splits.push((2, vec![gws[2] / (lz * rz), lz, rz]));
                                splits.push((1, vec![gws[1] / (ly * ry), ly, ry]));
                                splits.push((0, vec![gws[0] / (lx * rx), lx, rx]));

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
                                        let VOp::Loop { len, .. } = self.ops[id - 1 + num_loops]
                                        else {
                                            unreachable!()
                                        };
                                        // Register work size in the reduce loop
                                        for rr in (1..=maxrr).filter(|rr| len % rr == 0) {
                                            // Get splits for local and global work dims
                                            let mut splits = splits.clone();
                                            splits.insert(
                                                0,
                                                (id - 1 + num_loops, vec![len / rr, rr]),
                                            );
                                            // Permute, private loops last
                                            opts.push((
                                                KernelOptimization {
                                                    splits: splits.clone(),
                                                    permutation: vec![0, 1, 2, 3, 4, 5, 6, 7],
                                                    local_tiles: false,
                                                },
                                                0,
                                            ));
                                            if rr == ly && rr == lz {
                                                opts.push((
                                                    KernelOptimization {
                                                        splits,
                                                        permutation: vec![0, 1, 2, 3, 4, 5, 6, 7],
                                                        local_tiles: true,
                                                    },
                                                    0,
                                                ));
                                            }
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
                                    opts.push((
                                        KernelOptimization {
                                            splits,
                                            permutation: vec![0, 1, 2, 3, 4, 5, 6, 7],
                                            local_tiles: false,
                                        },
                                        0,
                                    ));
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
    #[allow(clippy::similar_names)]
    #[allow(clippy::cognitive_complexity)]
    pub(super) fn optimize(&self, optimization: &KernelOptimization) -> Kernel {
        let mut kernel = self.clone();
        // Apply axis splits
        for (op_id, dimensions) in &optimization.splits {
            kernel.split_axis(*op_id, dimensions);
        }

        let mut lws = [0; 3];
        let VOp::Loop { len, .. } = kernel.ops[1] else {
            unreachable!()
        };
        lws[0] = len;
        let VOp::Loop { len, .. } = kernel.ops[4] else {
            unreachable!()
        };
        lws[1] = len;
        let VOp::Loop { len, .. } = kernel.ops[7] else {
            unreachable!()
        };
        lws[2] = len;

        let mut rws = [0; 3];
        let VOp::Loop { len, .. } = kernel.ops[2] else {
            unreachable!()
        };
        rws[0] = len;
        let VOp::Loop { len, .. } = kernel.ops[5] else {
            unreachable!()
        };
        rws[1] = len;
        let VOp::Loop { len, .. } = kernel.ops[8] else {
            unreachable!()
        };
        rws[2] = len;
        // Apply permutation
        kernel.permute(&optimization.permutation);

        // Reorder so that register work threads are last
        // Register threads are op_id 1, 4 and 7
        if true {
            // if register work sizes are enabled
            let mut threaded = true;
            let rlz = kernel.ops.remove(8);
            let rly = kernel.ops.remove(5);
            let rlx = kernel.ops.remove(2);
            kernel.ops.insert(6, rlz.clone());
            kernel.ops.insert(6, rly.clone());
            kernel.ops.insert(6, rlx.clone());
            if kernel
                .ops
                .iter()
                .any(|op| matches!(op, VOp::Accumulator { .. }))
            {
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
                let acc_view = View::binded(&rws, &[2, 5, 8]);
                let mut accs = BTreeSet::new();
                let mut i = 0;
                while i < kernel.ops.len() {
                    match &mut kernel.ops[i] {
                        &mut VOp::Accumulator {
                            ref mut view,
                            z,
                            dtype,
                            ..
                        } => {
                            *view = acc_view.clone();
                            accs.insert((z, dtype));
                        }
                        VOp::Store {
                            z, xscope, xview, ..
                        } => {
                            if *xscope == Scope::Register {
                                if let Some(..) = accs.iter().find(|(x, _)| x == z) {
                                    *xview = acc_view.clone();
                                    *xscope = Scope::RegTile;
                                }
                            }
                        }
                        // This cannot be triggered currently
                        //VOp::Unary { z, .. } => { if accs.contains(z) { todo!(); } }
                        &mut VOp::Binary { z, x, y, .. } => {
                            //let dtype = crate::DType::F32;
                            // We can add new scope called register tile.
                            // That way each tensor will exist in one scope only once.
                            let mut op_i = i;
                            //if accs.contains(&x) {
                            if let Some(&(_, dtype)) = accs.iter().find(|(id, _)| *id == x) {
                                kernel.ops.insert(
                                    op_i + 1,
                                    VOp::Store {
                                        z: x,
                                        zscope: Scope::RegTile,
                                        zview: acc_view.clone(),
                                        zdtype: dtype,
                                        xscope: Scope::Register,
                                        xview: View::none(),
                                    },
                                );
                                kernel.ops.insert(
                                    op_i,
                                    VOp::Load {
                                        z: x,
                                        zscope: Scope::Register,
                                        zview: View::none(),
                                        x,
                                        xscope: Scope::RegTile,
                                        xview: acc_view.clone(),
                                        xdtype: dtype,
                                    },
                                );
                                op_i += 1;
                                i += 2;
                            }
                            if y != x {
                                if let Some(&(_, dtype)) = accs.iter().find(|(id, _)| *id == y) {
                                    kernel.ops.insert(
                                        op_i + 1,
                                        VOp::Store {
                                            z: y,
                                            zscope: Scope::RegTile,
                                            zview: acc_view.clone(),
                                            zdtype: dtype,
                                            xscope: Scope::Register,
                                            xview: View::none(),
                                        },
                                    );
                                    kernel.ops.insert(
                                        op_i,
                                        VOp::Load {
                                            z: y,
                                            zscope: Scope::Register,
                                            zview: View::none(),
                                            x: y,
                                            xscope: Scope::RegTile,
                                            xview: acc_view.clone(),
                                            xdtype: dtype,
                                        },
                                    );
                                    op_i += 1;
                                    i += 2;
                                }
                            }
                            if z != x && z != y {
                                if let Some(&(_, dtype)) = accs.iter().find(|(id, _)| *id == z) {
                                    kernel.ops.insert(
                                        op_i + 1,
                                        VOp::Store {
                                            z,
                                            zscope: Scope::RegTile,
                                            zview: acc_view.clone(),
                                            zdtype: dtype,
                                            xscope: Scope::Register,
                                            xview: View::none(),
                                        },
                                    );
                                    kernel.ops.insert(
                                        op_i,
                                        VOp::Load {
                                            z,
                                            zscope: Scope::Register,
                                            zview: View::none(),
                                            x: z,
                                            xscope: Scope::RegTile,
                                            xview: acc_view.clone(),
                                            xdtype: dtype,
                                        },
                                    );
                                    //op_i += 1;
                                    i += 2;
                                }
                            }
                        }
                        _ => {}
                    }
                    i += 1;
                }
            }
        }

        // TODO local tiling in elementwise kernels

        // Local tiling, for now possible only if both local dims equal reduce work size
        // TODO For now local work sizes must be equal to reduce_ws, later we can add one
        // more loop and then they will just need to be dividable without remainder.
        // TODO also take lws[0] into consideration
        if false {
            // Get reduce work size, TODO should be multiple values for multi reduce kernels
            let mut reduce_ws = 0;
            for op in &kernel.ops {
                if let &VOp::Loop { axis, len } = op {
                    if axis > 9 {
                        reduce_ws = len;
                    }
                }
            }
            //if optimization.local_tiles && lws[1] == reduce_ws && lws[2] == reduce_ws {
            println!("Using local tiling");
            // Local tile all loads that do not use all loop axes
            // Local tiles use local dimensions and register dimensions
            // i.e. [rws[0]*lws[0], rws[1]*lws[1], rws[2]*lws[2]]
            // TODO
            let mut axes = Vec::new();
            let mut lengths = Vec::new();
            let mut rl_id = 0; // id of the global reduce loop
            let mut reduce_axis = 0;
            let mut id = 0;
            while id < kernel.ops.len() {
                match &mut kernel.ops[id] {
                    &mut VOp::Loop { axis, len } => {
                        axes.push(axis);
                        lengths.push(len);
                        if axis > 8 {
                            rl_id = id - 1;
                            reduce_axis = axis;
                        }
                        if axis == 2 && rl_id != 0 {
                            //kernel.ops.insert(id, kernel.ops[rl_id].clone());
                            kernel.ops.insert(
                                id - 1,
                                VOp::Barrier {
                                    scope: Scope::Local,
                                },
                            );
                            //kernel.ops.insert(id, VOp::EndLoop);
                            id += 1;
                        }
                    }
                    VOp::EndLoop => {
                        if let Some(axis) = axes.pop() {
                            if let Some(&VOp::Loop { axis: raxis, .. }) = kernel.ops.get(rl_id) {
                                if axis == raxis {
                                    kernel.ops.insert(
                                        id,
                                        VOp::Barrier {
                                            scope: Scope::Local,
                                        },
                                    );
                                    id += 1;
                                }
                            }
                            if axis == 9 {
                                rl_id = 0;
                            }
                        }
                        lengths.pop().unwrap();
                    }
                    VOp::Load {
                        z,
                        zscope,
                        zview,
                        x,
                        xscope,
                        xview,
                        xdtype,
                    } => {
                        if *zscope == Scope::Register
                            && *xscope == Scope::Global
                            && zview == &View::none()
                        {
                            let mut sorted_axes = axes.clone();
                            sorted_axes.sort_unstable();
                            let used_axes = xview.used_axes();
                            if used_axes != sorted_axes {
                                let global_view = xview.clone();
                                // TODO add rws[0]
                                let axes = if used_axes.contains(&5) {
                                    [4, reduce_axis, 5]
                                } else {
                                    [4, reduce_axis, 8]
                                };

                                let dims = if used_axes.contains(&5) {
                                    [lws[1], reduce_ws, rws[1]]
                                } else {
                                    [lws[1], reduce_ws, rws[2]]
                                };
                                let local_view = View::binded(&dims, &axes);
                                *xview = local_view;
                                *xscope = Scope::Local;
                                let z = *z;
                                let x = *x;
                                let xdtype = *xdtype;

                                let axes = if used_axes.contains(&5) {
                                    [4, 7, 5]
                                } else {
                                    [4, 7, 8]
                                };
                                let dims = if used_axes.contains(&5) {
                                    [lws[1], lws[2], rws[1]]
                                } else {
                                    [lws[1], lws[2], rws[2]]
                                };
                                let local_view = View::binded(&dims, &axes);
                                kernel.ops.insert(rl_id + 1, VOp::EndLoop);
                                kernel.ops.insert(
                                    rl_id + 1,
                                    VOp::Load {
                                        z,
                                        zscope: Scope::Local,
                                        zview: local_view,
                                        x,
                                        xscope: Scope::Global,
                                        xview: global_view,
                                        xdtype,
                                    },
                                );
                                if used_axes.contains(&8) {
                                    kernel.ops.insert(
                                        rl_id + 1,
                                        VOp::Loop {
                                            axis: 8,
                                            len: rws[2],
                                        },
                                    );
                                }
                                if used_axes.contains(&5) {
                                    kernel.ops.insert(
                                        rl_id + 1,
                                        VOp::Loop {
                                            axis: 5,
                                            len: rws[1],
                                        },
                                    );
                                }
                                id += 3;
                            }
                        }
                    }
                    _ => {}
                }
                id += 1;
            }
        }
        kernel
    }
}

impl Display for KernelOptimization {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "splits {:?}, permute: {:?}",
            self.splits, self.permutation
        ))
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
                if values.is_empty() {
                    *self = KernelOptimizer::Optimized(opts[*best].0.clone(), opts[*best].1);
                }
                let mut rng = rand::rngs::SmallRng::seed_from_u64(190_940_981_234_098_124);
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
            KernelOptimizer::Optimizing(opts, _) => {
                (0..opts.len()).filter(|&id| opts[id].1 == 0).count()
            }
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
