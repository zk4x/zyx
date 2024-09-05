use std::fmt::Display;

use crate::{runtime::backend::DeviceInfo, shape::Dimension};
use super::kernel::Kernel;

// Optimizations get applied to existing kernels after
// they are assigned to devices.
#[derive(Debug, Clone, bitcode::Encode, bitcode::Decode)]
pub(crate) struct KernelOptimizations {
    // Axis splits to give us global, local and register work sizes
    pub(crate) splits: Vec<(usize, Vec<Dimension>)>,
    // Permutation so that global and local work sizes are first
    pub(crate) permutation: Vec<usize>,
    // Work per thread in reduce loops, one per each reduce
    pub(crate) reduce_loop_wpt: Vec<usize>,

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
    pub(super) fn possible_optimizations(&self, dev_info: &DeviceInfo) -> Vec<KernelOptimizations> {
        todo!()
    }

    pub(super) fn default_optimizations(&self, dev_info: &DeviceInfo) -> KernelOptimizations {
        /*let num_loops = self
            .ops
            .iter()
            .position(|kernel| !matches!(kernel, VOp::Loop { .. }))
            .unwrap();
        assert_ne!(num_loops, 0);
        if num_loops < 3 {
            let dims: Vec<usize> = core::iter::repeat(1)
                .take(3 - num_loops)
                .chain([self.shape[0]])
                .collect();
            kernel.split_axis(0, &dims);
        }
        // Split first three loops into global and local loops.
        for op in &kernel.ops {
            if let VOp::Loop { axis, dimension } = op {
                if *axis > 2 {
                    break;
                }
                gws[*axis] = *dimension;
            }
        }
        let lws = best_work_size(gws, dev_info.max_work_group_size);
        gws[0] /= lws[0];
        gws[1] /= lws[1];
        gws[2] /= lws[2];
        kernel.split_axis(0, &[gws[0], lws[0]]);
        kernel.split_axis(2, &[gws[1], lws[1]]);
        kernel.split_axis(4, &[gws[2], lws[2]]);
        */
        todo!()
    }

    // add per device optimizations to each kernel, local memory, accumulators, work per thread, tiling on many levels,
    // split, merge, permute, pad loops and get them to correct dimensionality (3d) for execution on the device.
    // tensor cores, just a ton of stuff. Later add search over different optimizations.
    pub(crate) fn optimize(&self, optimizations: &KernelOptimizations) -> Kernel {
        let mut kernel = self.clone();
        // Apply axis splits
        for (op_id, dimensions) in &optimizations.splits {
            kernel.split_axis(*op_id, dimensions);
        }
        // Apply permutation
        kernel.permute(&optimizations.permutation);
        kernel
    }

    /*
        // First create a list of all possible global, local and register work sizes
        // given max_work_group_size and number of register.
        // Also needs to contain work per thread size for single reduce.
        // like this: [gws0, gws1, gws2, lws0, lws1, lws2, rws0, rws1, rws2, rwsr]

        // Get the number of loops before any other operation
        let num_loops = self
            .ops
            .iter()
            .position(|kernel| !matches!(kernel, VOp::Loop { .. }))
            .unwrap();
        assert_ne!(num_loops, 0);

        let mut kernel = self.clone();

        // If there is more loops than 3, pick first three loops as global loops,
        // rest is register loops.
        // So nothing needs to be done.
        // If there is less than three loops, add loops with dimension 1
        if num_loops < 3 {
            let dims: Vec<usize> = core::iter::repeat(1)
                .take(3 - num_loops)
                .chain([self.shape[0]])
                .collect();
            kernel.split_axis(0, &dims);
        }

        // Set best local work sizes
        let mut gws = [1; 3];
        let lws = {
            // Split first three loops into global and local loops.
            for op in &kernel.ops {
                if let VOp::Loop { axis, dimension } = op {
                    if *axis > 2 {
                        break;
                    }
                    gws[*axis] = *dimension;
                }
            }
            let lws = best_work_size(gws, dev_info.max_work_group_size);
            gws[0] /= lws[0];
            gws[1] /= lws[1];
            gws[2] /= lws[2];
            kernel.split_axis(0, &[gws[0], lws[0]]);
            kernel.split_axis(2, &[gws[1], lws[1]]);
            kernel.split_axis(4, &[gws[2], lws[2]]);
            lws
        };

        // Set best register work sizes
        let more_wpt = false;
        let rws = if more_wpt {
            let rws = {
                let rws = best_work_size(gws, dev_info.num_registers);
                gws[0] = gws[0]/rws[0];
                gws[1] = gws[1]/rws[1];
                gws[2] = gws[2]/rws[2];
                kernel.split_axis(0, &[gws[0], rws[0]]);
                kernel.split_axis(3, &[gws[1], rws[1]]);
                kernel.split_axis(6, &[gws[2], rws[2]]);
                // Permute so that work per thread loops are after global and local loops
                kernel.permute(&[0, 2, 3, 5, 6, 8, 1, 4, 7]);

                if kernel.ops.iter().any(|op| matches!(op, VOp::Reduce { .. })) {
                    // Handle reduce loops
                    // Split reduce loops for more work per thread
                    let mut splits = Vec::new();
                    for (id, op) in kernel.ops[9..].iter().enumerate() {
                        if let VOp::Accumulator { .. } = op {
                            let VOp::Loop { dimension, .. } = kernel.ops[id+10] else { todo!() };
                            // TODO get this working if there is more than one reduce loop
                            // TODO get this working with different work per thread
                            splits.push((id+10, [dimension/8, 8]));
                        }
                    }
                    for split in splits {
                        println!("Splitting at {split:?}");
                        kernel.split_axis(split.0, &split.1);
                    }

                    // Permute such that register loops come after reduce loops
                    // Just swap register loops after reduce loops
                    let r0loop = kernel.ops.remove(6);
                    let r1loop = kernel.ops.remove(6);
                    let r2loop = kernel.ops.remove(6);
                    let mut start_reg_ids = Vec::new();
                    let mut end_reg_ids = Vec::new();
                    let mut last_loop_id = None;
                    for id in 6..kernel.ops.len() {
                        match kernel.ops[id] {
                            VOp::Loop { .. } => if let Some(last_loop_id) = &mut last_loop_id {
                                *last_loop_id = id;
                            } else {
                                end_reg_ids.push(id);
                                last_loop_id = Some(id);
                            },
                            VOp::Reduce { .. } => {
                                start_reg_ids.push(last_loop_id);
                            }
                            _ => {}
                        }
                    }

                    // Update accumulators such that they use these register loops

                }
                rws
            };
            println!("Work sizes: {gws:?} {lws:?} {rws:?}");
            rws
        } else {
            [1, 1, 1]
        };

        /*let mut local_loads = Vec::new();
        // Add local and register tiles for expanded tensor loads
        for id in 0..self.kernel.len() {
            //if matches!(self.ops[id], VOp::Load { .. }) {
            if let VOp::Load { z, zscope: scope, view, .. } = &mut self.ops[id] {
                local_loads.push((*z, view.clone()));
                *scope = Scope::Local;
                // First load into local
                *view = View::binded(&[lws[0]*rws[0], lws[1]*rws[1], lws[2]*rws[2]], &[6, 7, 8]);
            }
            // Find all uses of this local loads and put them into registers before using them
            // registers can be tiles, correctly wized tiles directly map to tensor cores
        }*/
        //self.debug();

        // Add local caching for loads
        kernel
    }*/
}

// Takes global work size (gws) and maximum work group size (mwgs)
// Also is used to get best register work size
fn best_work_size(mut gws: [usize; 3], mwgs: usize) -> [usize; 3] {
    let mut lws = [1; 3];
    //println!("Max {mwgs:?}");
    let rwgs = (mwgs as f64).sqrt() as usize;
    //println!("Root {rwgs:?}");

    let mut total = 1;
    let mut n = 1;
    while gws[1] % (n * 2) == 0 && n * 2 <= rwgs {
        n *= 2;
    }
    gws[1] /= n;
    lws[1] *= n;
    total *= n;
    // put the rest into third dimension
    let mut n = 1;
    while gws[2] % (n * 2) == 0 && n * 2 * total <= mwgs {
        n *= 2;
    }
    gws[2] /= n;
    lws[2] *= n;
    total *= n;
    // if third dimension was too small, put the rest into second dimension
    let mut n = 1;
    while gws[1] % (n * 2) == 0 && n * 2 * total <= mwgs {
        n *= 2;
    }
    gws[1] /= n;
    lws[1] *= n;

    return lws;
}

impl Display for KernelOptimizations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("splits {:?}, permute: {:?}, rwpt {:?}", self.splits, self.permutation, self.reduce_loop_wpt))
    }
}
