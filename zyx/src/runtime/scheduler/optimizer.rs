use crate::{runtime::{backend::DeviceInfo, scheduler::vop::VOp, view::View}, tensor::TensorId};
use super::kernel::Kernel;

// Optimizations get applied to existing kernels after
// they are assigned to devices.
#[derive(Debug)]
pub(crate) struct KernelOptimizations {
    permutation: Vec<usize>,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    register_work_size: [usize; 3],
    // Load tensor first into local tile, then into registers
    // this is used mainly for expanded tensors, so use threads
    // from one local work group to load the tile and then sync loads
    // before loading into registers
    local_tiles: Vec<(TensorId, View)>,
    // Unrolls loop with given id
    unroll_loops: Vec<usize>,
    // Converts all variables in loop into native vector dtypes
    // and removes the loop.
    vectorize_loops: Vec<usize>,
    // Tile tensor in registers with given view
    register_tiles: Vec<(TensorId, View)>,
    // TensorCores,
    // WMMA
}

// Probably just do all the optimizations including tensor cores here,
// ir will be just a direct translation and can be removed if we replace it with something
// like renderer to c style, assembly and such.
impl Kernel {
    // add per device optimizations to each kernel, local memory, accumulators, work per thread, tiling on many levels,
    // split, merge, permute, pad loops and get them to correct dimensionality (3d) for execution on the device.
    // tensor cores, just a ton of stuff. Later add search over different optimizations.
    pub(super) fn optimize(&mut self, dev_info: &DeviceInfo) -> KernelOptimizations {
        // Get the number of loops before any other operation
        let num_loops = self
            .ops
            .iter()
            .position(|kernel| !matches!(kernel, VOp::Loop { .. }))
            .unwrap();
        assert_ne!(num_loops, 0);

        // If there is more loops than 3, pick first three loops as global loops,
        // rest is register loops.
        // So nothing needs to be done.
        // If there is less than three loops, add loops with dimension 1
        if num_loops < 3 {
            let dims: Vec<usize> = core::iter::repeat(1)
                .take(3 - num_loops)
                .chain([self.shape[0]])
                .collect();
            self.split_axis(0, &dims);
        }

        // Set best local work sizes
        let mut gws = [1; 3];
        let local_work_size = {
            // Split first three loops into global and local loops.
            for op in &self.ops {
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
            self.split_axis(0, &[gws[0], lws[0]]);
            self.split_axis(2, &[gws[1], lws[1]]);
            self.split_axis(4, &[gws[2], lws[2]]);
            lws
        };

        // Set best register work sizes
        let register_work_size = {
            let rws = best_work_size(gws, dev_info.num_registers);
            gws[0] = gws[0]/rws[0];
            gws[1] = gws[1]/rws[1];
            gws[2] = gws[2]/rws[2];
            self.split_axis(0, &[gws[0], rws[0]]);
            self.split_axis(3, &[gws[1], rws[1]]);
            self.split_axis(6, &[gws[2], rws[2]]);
            // Permute so that work per thread loops are after global and local loops
            self.permute(&[0, 2, 3, 5, 6, 8, 1, 4, 7]);


                // Handle reduce loops
                // Split reduce loops for more work per thread

                // Permute such that all register loops are last

                //self.permute_loops(op_id, &[]);

                // Update accumulators such that they use these register loops
            rws
        };
        self.debug();
        println!("Optimizations: {} work sizes: {gws:?} {local_work_size:?} {register_work_size:?}", dev_info.num_registers);

        // Add local caching for loads
        KernelOptimizations {
            permutation: Vec::new(),
            global_work_size: gws,
            local_work_size,
            register_work_size,
            local_tiles: Vec::new(),
            unroll_loops: Vec::new(),
            vectorize_loops: Vec::new(),
            register_tiles: Vec::new(),
        }
    }
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
