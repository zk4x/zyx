use crate::{
    backend::{Device, DeviceInfo, MemoryPool},
    ir::Scope,
    kernel::{Kernel, Op},
    shape::Dimension,
    view::View,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

pub(super) struct Optimizer {
    cache: BTreeMap<(Kernel, DeviceInfo), OptimizerProgress>,
}

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

pub(super) struct Optimization {
    splits: Vec<(usize, Vec<Dimension>)>,
}

impl Optimizer {
    pub(super) const fn new() -> Optimizer {
        Optimizer {
            cache: BTreeMap::new(),
        }
    }

    pub(super) fn get_optimization(
        &mut self,
        kernel: &Kernel,
        device: &mut Device,
        memory_pool: &mut MemoryPool,
        search_iters: usize,
    ) -> &Optimization {
        match self.cache.get(&(kernel.clone(), device.info().clone())) {
            Some(OptimizerProgress::Finished { optimization }) => optimization,
            Some(OptimizerProgress::Optimizing { best, done }) => {
                if search_iters == 0 {
                    best
                } else {
                    //self.optimize_kernel(kernel.clone(), device, memory_pool, search_iters)
                    todo!()
                }
            }
            None => {
                if search_iters == 0 {
                    self.default_optimizations(kernel, device.info())
                } else {
                    //self.optimize_kernel(kernel.clone(), device, memory_pool, search_iters);
                    todo!()
                }
            }
        }
    }

    fn default_optimizations(&self, kernel: &Kernel, device_info: &DeviceInfo) -> &Optimization {
        let _ = kernel;
        let _ = device_info;
        todo!()
    }
    //fn optimize_kernel(&mut self, kernel: Kernel, device: &mut Device, memory_pool: &mut MemoryPool, search_iters: usize) { todo!() }
}

impl Kernel {
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
        }

        let mut lws = [0; 3];
        let Op::Loop { len, .. } = kernel.ops[1] else {
            unreachable!()
        };
        lws[0] = len;
        let Op::Loop { len, .. } = kernel.ops[4] else {
            unreachable!()
        };
        lws[1] = len;
        let Op::Loop { len, .. } = kernel.ops[7] else {
            unreachable!()
        };
        lws[2] = len;

        let mut rws = [0; 3];
        let Op::Loop { len, .. } = kernel.ops[2] else {
            unreachable!()
        };
        rws[0] = len;
        let Op::Loop { len, .. } = kernel.ops[5] else {
            unreachable!()
        };
        rws[1] = len;
        let Op::Loop { len, .. } = kernel.ops[8] else {
            unreachable!()
        };
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
            if kernel
                .ops
                .iter()
                .any(|op| matches!(op, Op::Accumulator { .. }))
            {
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
                        &mut Op::Accumulator {
                            ref mut view,
                            z,
                            dtype,
                            ..
                        } => {
                            *view = acc_view.clone();
                            accs.insert((z, dtype));
                        }
                        Op::Store {
                            z, xscope, xview, ..
                        } => {
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
                            kernel.ops.insert(
                                id - 1,
                                Op::Barrier {
                                    scope: Scope::Local,
                                },
                            );
                            //kernel.ops.insert(id, Op::EndLoop);
                            id += 1;
                        }
                    }
                    Op::EndLoop => {
                        if let Some(axis) = axes.pop() {
                            if let Some(&Op::Loop { axis: raxis, .. }) = kernel.ops.get(rl_id) {
                                if axis == raxis {
                                    kernel.ops.insert(
                                        id,
                                        Op::Barrier {
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
                    Op::Load {
                        z,
                        zscope,
                        zview,
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
                                    kernel.ops.insert(
                                        rl_id + 1,
                                        Op::Loop {
                                            axis: 8,
                                            len: rws[2],
                                        },
                                    );
                                }
                                if used_axes.contains(&5) {
                                    kernel.ops.insert(
                                        rl_id + 1,
                                        Op::Loop {
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
