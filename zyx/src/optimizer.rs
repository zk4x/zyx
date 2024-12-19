use crate::{
    backend::{Device, DeviceInfo, MemoryPool},
    ir::{IRKernel, Scope},
    kernel::{Kernel, Op},
    shape::Dimension,
    view::View,
    DebugMask, ZyxError,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

#[derive(Debug)]
pub(super) struct Optimizer {
    cache: BTreeMap<(Kernel, DeviceInfo), OptimizerProgress>,
}

#[derive(Debug)]
enum OptimizerProgress {
    Finished {
        optimization: Optimization,
        //time: Duration,
    },
    Optimizing {
        best: Optimization,
        done: BTreeMap<Optimization, Duration>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Optimization {
    splits: Vec<(usize, Vec<Dimension>)>,
    local_tiles: bool,
}

impl Optimizer {
    pub(super) const fn new() -> Optimizer {
        Optimizer { cache: BTreeMap::new() }
    }

    pub(super) fn search_optimization(
        &mut self,
        kernel: &Kernel,
        device: &mut Device,
        memory_pool: &mut MemoryPool,
        search_iters: usize,
        debug: DebugMask,
    ) -> Result<&Optimization, ZyxError> {
        fn optimize_kernel(
            kernel: &Kernel,
            device: &mut Device,
            memory_pool: &mut MemoryPool,
            search_iters: usize,
            done: &mut BTreeMap<Optimization, Duration>,
            debug: DebugMask,
        ) -> (Optimization, bool) {
            //OptimizerProgress::Optimizing { best: Optimization { splits: Vec::new() }, done: BTreeMap::new(), }
            // list untried optimizations
            let mut opts = kernel.available_optimizations(device.info(), done);
            assert!(!opts.is_empty());

            // TODO if we ensure that no buffer that has been stored to is ever loaded afterwards,
            // then we don't need to allocate and we can directly work with the actual buffers.
            // Without worrying about data corruption.
            // Allocate temporary buffers
            let mut allocated_temps = Vec::new();
            for op in &kernel.ops {
                match op {
                    Op::Load { xscope, xview, xdtype, .. } => {
                        if *xscope == Scope::Global {
                            let buffer_id = memory_pool
                                .allocate(xview.original_numel() * xdtype.byte_size())
                                .unwrap();
                            allocated_temps.push(buffer_id);
                        }
                    }
                    Op::Store { zscope, zview, zdtype, .. } => {
                        if *zscope == Scope::Global {
                            let buffer_id = memory_pool
                                .allocate(zview.original_numel() * zdtype.byte_size())
                                .unwrap();
                            allocated_temps.push(buffer_id);
                        }
                    }
                    _ => {}
                }
            }
            let mut best_exec_time = done.values().max().copied().unwrap_or(Duration::MAX);
            /*let flop_mem_rw = if debug_perf {
                Some(kernel.flop_mem_rw())
            } else {
                None
            };*/
            // pick an optimization
            for _ in 0..search_iters.min(opts.len()) {
                if let Some(optimization) = opts.pop() {
                    //println!("{optimization:?}");
                    let optimized_kernel = kernel.optimize(&optimization);
                    //optimized_kernel.debug();
                    //panic!();
                    let ir_kernel = IRKernel::new(&optimized_kernel.ops, debug.ir());
                    if device
                        .compile(optimized_kernel.ops.clone(), &ir_kernel, debug.asm())
                        .is_err()
                    {
                        done.insert(optimization, Duration::MAX);
                        //if debug_sched { println!("Could not compile, {e:?}, skipping"); }
                        continue;
                    }
                    // Launch kernel and measure it's performance
                    let begin = std::time::Instant::now();
                    let Ok(event) =
                        device.launch(&optimized_kernel.ops, memory_pool, &allocated_temps)
                    else {
                        done.insert(optimization, Duration::MAX);
                        //if debug_sched { println!("Could not launch, {e:?}, skipping"); }
                        continue;
                    };
                    if event.finish().is_err() {
                        done.insert(optimization, Duration::MAX);
                        //if debug_sched { println!("Could not sync, {e:?}, skipping"); }
                        continue;
                    };
                    let exec_time = begin.elapsed();
                    let _ = device.release_program(&optimized_kernel.ops);
                    done.insert(optimization, exec_time);
                    if exec_time < best_exec_time {
                        best_exec_time = exec_time;
                    }
                    /*if let Some((f, mr, mw)) = flop_mem_rw {
                        if let Some(bar) = &progress_bar {
                            bar.set_message(format!(
                                "{}/{} {}",
                                i + 1,
                                search_iters.min(opts.len()),
                                perf_string(f, mr, mw, best_exec_time)
                            ));
                        }
                    }
                    #[cfg(feature = "disk_cache")]
                    if timer.elapsed().as_secs() > 60 {
                        store_optimizer_cache(optimizer_cache, config_dir, debug_sched);
                        timer = std::time::Instant::now();
                    }*/
                }
            }

            (
                done.iter().min_by_key(|x| x.1).unwrap().0.clone(),
                opts.is_empty(),
            )
        }

        // TODO do not clone kernel and device info, borrowck really sucks here
        let key = (kernel.clone(), device.info().clone());
        Ok(
            match self
                .cache
                .entry(key)
                .and_modify(|progress| {
                    if search_iters != 0 {
                        if let OptimizerProgress::Optimizing { best, done } = progress {
                            let (optimization, finished) = optimize_kernel(
                                kernel,
                                device,
                                memory_pool,
                                search_iters,
                                done,
                                debug,
                            );
                            if finished {
                                *progress = OptimizerProgress::Finished { optimization };
                            } else {
                                *best = optimization;
                            }
                        }
                    }
                })
                .or_insert_with(|| {
                    if search_iters == 0 {
                        let best = Optimizer::default_optimizations(kernel, device.info());
                        let done = BTreeMap::new();
                        OptimizerProgress::Optimizing { best, done }
                    } else {
                        let mut done = BTreeMap::new();
                        let (optimization, finished) = optimize_kernel(
                            kernel,
                            device,
                            memory_pool,
                            search_iters,
                            &mut done,
                            debug,
                        );
                        if finished {
                            OptimizerProgress::Finished { optimization }
                        } else {
                            OptimizerProgress::Optimizing { best: optimization, done }
                        }
                    }
                }) {
                OptimizerProgress::Finished { optimization } => optimization,
                OptimizerProgress::Optimizing { best, .. } => best,
            },
        )
    }

    fn default_optimizations(kernel: &Kernel, device_info: &DeviceInfo) -> Optimization {
        let _ = kernel;
        let _ = device_info;
        todo!()
    }
}

impl Kernel {
    fn available_optimizations(
        &self,
        dev_info: &DeviceInfo,
        done: &BTreeMap<Optimization, Duration>,
    ) -> Vec<Optimization> {
        let mut opts = Vec::new();

        //let mgwd = dev_info.max_global_work_dims;
        let mlws = dev_info.max_local_threads;
        let mut mlwd = dev_info.max_local_work_dims;
        let mrws = dev_info.num_registers;
        let (maxrr, mrwd) = if true {
            (8, [16, 16, 16]) // For now 16, can be raised to 32 on some hardware perhaps
        } else {
            mlwd = [1, 1, 1];
            (1, [1, 1, 1]) // For debugging
        };

        let mut reshapes = Vec::new();
        let num_loops = self.ops.iter().position(|op| !matches!(op, Op::Loop { .. })).unwrap();
        assert_ne!(num_loops, 0);
        let mut gws = [1; 3];
        if num_loops < 3 {
            let dims: Vec<usize> =
                core::iter::repeat(1).take(3 - num_loops).chain([self.shape()[0]]).collect();
            reshapes.push((0, dims));
            let mut gws_i = 3 - num_loops;
            for d in &self.shape() {
                gws[gws_i] = *d;
                gws_i += 1;
            }
        } else {
            let sh = self.shape();
            for (gws_d, d) in gws.iter_mut().zip(sh[sh.len() - 3..].iter()) {
                *gws_d = *d;
            }
            gws[0] = sh[..sh.len() - 2].iter().product();
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
                                let mut splits = reshapes.clone();
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
                                    if loop_found && !matches!(op, Op::Loop { .. }) {
                                        loop_found = false;
                                        acc_found = false;
                                        reduce_found = true;
                                        let Op::Loop { len, .. } = self.ops[id - 1 + num_loops]
                                        else {
                                            unreachable!()
                                        };
                                        // Register work size in the reduce loop
                                        for rr in (1..=maxrr).filter(|rr| len % rr == 0) {
                                            // Get splits for local and global work dims
                                            let mut splits = splits.clone();
                                            splits.insert(
                                                0,
                                                (
                                                    id - 1
                                                        + if num_loops > 3 { 3 } else { num_loops },
                                                    vec![len / rr, rr],
                                                ),
                                            );
                                            // Permute, private loops last
                                            let mut optimization =
                                                Optimization { splits, local_tiles: false };
                                            if !done.contains_key(&optimization) {
                                                opts.push(optimization.clone());
                                            }
                                            if rr == ly && rr == lz {
                                                optimization.local_tiles = true;
                                                if !done.contains_key(&optimization) {
                                                    opts.push(optimization);
                                                }
                                            }
                                        }
                                    }
                                    if acc_found && matches!(op, Op::Loop { .. }) {
                                        //println!("Loop found at {id}");
                                        loop_found = true;
                                    }
                                    if matches!(op, Op::Accumulator { .. }) {
                                        //println!("Acc found at {id}");
                                        acc_found = true;
                                    }
                                }
                                if !reduce_found {
                                    // Permute, private loops last
                                    opts.push(Optimization { splits, local_tiles: false });
                                }
                            }
                        }
                    }
                }
            }
        }
        opts
    }

    #[allow(clippy::similar_names)]
    #[allow(clippy::cognitive_complexity)]
    pub(super) fn optimize(&self, optimization: &Optimization) -> Kernel {
        let mut kernel = self.clone();
        let sh = kernel.shape();
        if sh.len() > 3 {
            let sh: Vec<usize> = [sh[..sh.len() - 2].iter().product::<usize>()]
                .iter()
                .chain(sh[sh.len() - 2..].iter())
                .copied()
                .collect();
            assert!(kernel.reshape(&sh));
        }

        // Apply axis splits
        for (op_id, dimensions) in &optimization.splits {
            kernel.split_axis(*op_id, dimensions);
            //kernel.debug();
        }

        let mut lws = [0; 3];
        let Op::Loop { len, .. } = kernel.ops[1] else { unreachable!() };
        lws[0] = len;
        let Op::Loop { len, .. } = kernel.ops[4] else { unreachable!() };
        lws[1] = len;
        let Op::Loop { len, .. } = kernel.ops[7] else { unreachable!() };
        lws[2] = len;

        let mut rws = [0; 3];
        let Op::Loop { len, .. } = kernel.ops[2] else { unreachable!() };
        rws[0] = len;
        let Op::Loop { len, .. } = kernel.ops[5] else { unreachable!() };
        rws[1] = len;
        let Op::Loop { len, .. } = kernel.ops[8] else { unreachable!() };
        rws[2] = len;
        // Apply permutation
        //kernel.permute(&optimization.permutation);

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
            if kernel.ops.iter().any(|op| matches!(op, Op::Accumulator { .. })) {
                let mut id = 9;
                while id < kernel.ops.len() {
                    if threaded && matches!(kernel.ops[id], Op::Loop { .. }) {
                        kernel.ops.insert(id, Op::EndLoop);
                        kernel.ops.insert(id, Op::EndLoop);
                        kernel.ops.insert(id, Op::EndLoop);
                        id += 4;
                        threaded = false;
                        continue;
                    }
                    if threaded && matches!(kernel.ops[id], Op::EndLoop) {
                        kernel.ops.insert(id, Op::EndLoop);
                        kernel.ops.insert(id, Op::EndLoop);
                        kernel.ops.insert(id, Op::EndLoop);
                        id += 4;
                        threaded = false;
                        continue;
                    }
                    if !threaded && !matches!(kernel.ops[id], Op::Loop { .. } | Op::EndLoop) {
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
                let acc_view = View::binded(&rws, &[2, 5, 8], 10);
                let mut accs = BTreeSet::new();
                let mut i = 0;
                while i < kernel.ops.len() {
                    match &mut kernel.ops[i] {
                        &mut Op::Accumulator { ref mut view, z, dtype, .. } => {
                            *view = acc_view.clone();
                            accs.insert((z, dtype));
                        }
                        Op::Store { z, xscope, xview, .. } => {
                            if *xscope == Scope::Register && accs.iter().any(|(x, _)| x == z) {
                                *xview = acc_view.clone();
                                *xscope = Scope::RegTile;
                            }
                        }
                        // This cannot be triggered currently
                        //Op::Unary { z, .. } => { if accs.contains(z) { todo!(); } }
                        &mut Op::Binary { z, x, y, .. } => {
                            //let dtype = crate::DType::F32;
                            // We can add new scope called register tile.
                            // That way each tensor will exist in one scope only once.
                            let mut op_i = i;
                            //if accs.contains(&x) {
                            if let Some(&(_, dtype)) = accs.iter().find(|(id, _)| *id == x) {
                                kernel.ops.insert(
                                    op_i + 1,
                                    Op::Store {
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
                                    Op::Load {
                                        z: x,
                                        zscope: Scope::Register,
                                        zview: View::none(),
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
                                        Op::Store {
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
                                        Op::Load {
                                            z: y,
                                            zscope: Scope::Register,
                                            zview: View::none(),
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
                                        Op::Store {
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
                                        Op::Load {
                                            z,
                                            zscope: Scope::Register,
                                            zview: View::none(),
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
                if let &Op::Loop { axis, len } = op {
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
                    &mut Op::Loop { axis, len } => {
                        axes.push(axis);
                        lengths.push(len);
                        if axis > 8 {
                            rl_id = id - 1;
                            reduce_axis = axis;
                        }
                        if axis == 2 && rl_id != 0 {
                            //kernel.ops.insert(id, kernel.ops[rl_id].clone());
                            kernel.ops.insert(id - 1, Op::Barrier { scope: Scope::Local });
                            //kernel.ops.insert(id, Op::EndLoop);
                            id += 1;
                        }
                    }
                    Op::EndLoop => {
                        if let Some(axis) = axes.pop() {
                            if let Some(&Op::Loop { axis: raxis, .. }) = kernel.ops.get(rl_id) {
                                if axis == raxis {
                                    kernel.ops.insert(id, Op::Barrier { scope: Scope::Local });
                                    id += 1;
                                }
                            }
                            if axis == 9 {
                                rl_id = 0;
                            }
                        }
                        lengths.pop().unwrap();
                    }
                    Op::Load { z, zscope, zview, xscope, xview, xdtype } => {
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
                                let local_view = View::binded(&dims, &axes, 10);
                                *xview = local_view;
                                *xscope = Scope::Local;
                                let z = *z;
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
                                let local_view = View::binded(&dims, &axes, 10);
                                kernel.ops.insert(rl_id + 1, Op::EndLoop);
                                kernel.ops.insert(
                                    rl_id + 1,
                                    Op::Load {
                                        z,
                                        zscope: Scope::Local,
                                        zview: local_view,
                                        xscope: Scope::Global,
                                        xview: global_view,
                                        xdtype,
                                    },
                                );
                                if used_axes.contains(&8) {
                                    kernel.ops.insert(rl_id + 1, Op::Loop { axis: 8, len: rws[2] });
                                }
                                if used_axes.contains(&5) {
                                    kernel.ops.insert(rl_id + 1, Op::Loop { axis: 5, len: rws[1] });
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
