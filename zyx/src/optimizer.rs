#![allow(unused)]

use std::collections::{BTreeMap, BTreeSet};

use crate::{backend::{BackendError, Device, DeviceInfo}, ir::IRKernel, kernel::{Kernel, Op}, rng::Rng, runtime::Pool, shape::Dimension, slab::Id, DebugMask, Map, Set};

pub(super) struct Optimizer<'a> {
    rng: Rng,
    kernel: &'a Kernel,
    dev_info: DeviceInfo,
    visited: BTreeMap<Optimization, u128>,
    pub best_node: Optimization,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Optimization {
    pub shape: [Dimension; 9],
    pub opt_ops: BTreeSet<OptOp>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum OptOp {
    //MergeLoop {},
    SplitLoop {
        id: u16,
        order: u16,
        len: Dimension,
    },
    DownCastLoop {
        id: u16,
        order: u16,
    },
    /*UpCastLoop {
        id: u16,
        order: u16,
    },*/
}

impl Optimization {
    pub fn new(kernel: &Kernel, dev_info: &DeviceInfo) -> Self {
        let mlws = dev_info.max_local_threads;
        let mlwd = dev_info.max_local_work_dims;
        //let mut reshapes = Vec::new();
        let num_loops = kernel.ops.iter().position(|op| !matches!(op, Op::Loop { .. })).unwrap();
        debug_assert_ne!(num_loops, 0);
        let mut gws = [1; 3];
        if num_loops < 3 {
            //reshapes.push((0, dims));
            let mut gws_i = 3 - num_loops;
            for d in &kernel.shape() {
                gws[gws_i] = *d;
                gws_i += 1;
            }
        } else {
            let sh = kernel.shape();
            for (gws_d, d) in gws.iter_mut().zip(sh[sh.len() - 3..].iter()) {
                *gws_d = *d;
            }
            gws[0] = sh[..sh.len() - 2].iter().product();
        }

        let [gx, gy, gz] = gws;

        //let mrws = dev_info.num_registers;
        //let max_reg_split = 32;

        //println!("mlwd {mlwd:?}, mlws {mlws:?}");

        let lz = (1..=mlws.min(mlwd[2])).rev().filter(|lz| gz % lz == 0).max().unwrap_or(1);
        let ly =
            (1..=(mlws / lz).min(mlwd[1])).rev().filter(|ly| gy % ly == 0).max().unwrap_or(1);
        let lx = (1..=(mlws / (lz * ly)).min(mlwd[0]))
            .rev()
            .filter(|lx| gx % lx == 0)
            .max()
            .unwrap_or(1);

        //println!("gws = {gws:?}, lws = [{lx}, {ly}, {lz}]");

        // Upcast is only possible if last local dimension (lz) is 1

        Optimization { shape: [gx/lx, gy/ly, gz/lz, lx, ly, lz, 1, 1, 1], opt_ops: BTreeSet::new() }
    }
}

impl<'a> Optimizer<'a> {
    pub fn new(rng: Rng, kernel: &'a Kernel, dev_info: DeviceInfo) -> Self {
        Self {
            kernel,
            rng,
            visited: BTreeMap::new(),
            best_node: Optimization::new(kernel, &dev_info),
            dev_info,
        }
    }

    /// Next tries a new optimization given access to the device and memory, so that it can run it.
    pub fn next(&mut self) -> Option<Optimization> {
        // List all possible optimizations
        let mut possible_optimizations = Vec::new();

        let mut possible_opt_ops: Vec<OptOp> = Vec::new();
        // Get a list of all inner loops. Everyone of those can be split, upcasted or downcasted (permutation)
        let mut loop_map: Map<u16, u16> = Map::default();
        for op in self.kernel.ops.iter().skip_while(|op| matches!(op, Op::Loop { .. })) {
            if let &Op::Loop { axis, len } = op {
                let id = axis as u16;
                let order = *loop_map.entry(id).and_modify(|x| *x += 1).or_insert(0);
                possible_opt_ops.push(OptOp::DownCastLoop { id, order });
                // TODO possibly add other splits too
                if len % 4 == 0 {
                    possible_opt_ops.push(OptOp::SplitLoop { id, order, len: 4 });
                }
            }
        }

        // Now make a list of possible reshapes, multiply or divide each dimension by 2
        // This can be done for local and register dimensions, global dimensions will always have to be changed in opposite direction

        // Decrese global work size, increase local work size
        let shape = self.best_node.shape;
        for i in 0..3 {
            if shape[i] % 2 == 0 && shape[i+3] * 2 <= self.dev_info.max_local_work_dims[i] {
                let mut shape = shape;
                shape[i] /= 2;
                shape[i+3] *= 2;
                for op in &possible_opt_ops {
                    let mut opt_ops = self.best_node.opt_ops.clone();
                    opt_ops.insert(op.clone());
                    possible_optimizations.push(Optimization { shape, opt_ops });
                }
            }
        }
        // Decrese global work size, increase register work size
        let shape = self.best_node.shape;
        for i in 0..3 {
            if shape[i] % 2 == 0 && shape[i+6] * 2 <= 32 {
                let mut shape = shape;
                shape[i] /= 2;
                shape[i+6] *= 2;
                for op in &possible_opt_ops {
                    let mut opt_ops = self.best_node.opt_ops.clone();
                    opt_ops.insert(op.clone());
                    possible_optimizations.push(Optimization { shape, opt_ops });
                }
            }
        }

        // Decrese local work size, increase global work size
        for i in 0..3 {
            if shape[i+3] % 2 == 0 {
                let mut shape = shape;
                shape[i+3] /= 2;
                shape[i] *= 2;
                let opt_ops = self.best_node.opt_ops.clone();
                for op in &possible_opt_ops {
                    let mut opt_ops = opt_ops.clone();
                    opt_ops.insert(op.clone());
                    possible_optimizations.push(Optimization { shape, opt_ops });
                }
                possible_optimizations.push(Optimization { shape, opt_ops });
            }
        }

        // Decrese local work size, increase register work size
        for i in 0..3 {
            if shape[i+3] % 2 == 0 && shape[i+6] * 2 < 32 {
                let mut shape = shape;
                shape[i+3] /= 2;
                shape[i+6] *= 2;
                let opt_ops = self.best_node.opt_ops.clone();
                for op in &possible_opt_ops {
                    let mut opt_ops = opt_ops.clone();
                    opt_ops.insert(op.clone());
                    possible_optimizations.push(Optimization { shape, opt_ops });
                }
                possible_optimizations.push(Optimization { shape, opt_ops });
            }
        }

        // Decrese register work size, increase global work size
        let shape = self.best_node.shape;
        for i in 0..3 {
            if shape[i+6] % 2 == 0 {
                let mut shape = shape;
                shape[i+6] /= 2;
                shape[i] *= 2;
                for op in &possible_opt_ops {
                    let mut opt_ops = self.best_node.opt_ops.clone();
                    opt_ops.insert(op.clone());
                    possible_optimizations.push(Optimization { shape, opt_ops });
                }
            }
        }

        // Decrese register work size, increase local work size
        for i in 0..3 {
            if shape[i+6] % 2 == 0 && shape[i+3] * 2 < self.dev_info.max_local_work_dims[i] {
                let mut shape = shape;
                shape[i+6] /= 2;
                shape[i+3] *= 2;
                let opt_ops = self.best_node.opt_ops.clone();
                for op in &possible_opt_ops {
                    let mut opt_ops = opt_ops.clone();
                    opt_ops.insert(op.clone());
                    possible_optimizations.push(Optimization { shape, opt_ops });
                }
                possible_optimizations.push(Optimization { shape, opt_ops });
            }
        }

        // Then delete those optimizations that were already visited
        possible_optimizations.retain(|optimization| !self.visited.contains_key(optimization));
        //println!("possible opts: {possible_optimizations:?}");

        if possible_optimizations.is_empty() {
            return None;
        }

        // Then randomly pick an optimization
        let i: u32 = self.rng.rand();
        Some(possible_optimizations.swap_remove(i as usize % possible_optimizations.len()))
    }

    pub fn bench_optimization(&mut self, optimization: &Optimization, pool: &mut Pool, device: &mut Device, args: &[Id], debug: DebugMask) -> Result<u128, BackendError> {
        let ir_kernel = IRKernel::new(self.kernel.clone(), optimization, debug);
        let program_id = device.compile(&ir_kernel, debug.asm())?;
        let nanos = std::time::Instant::now();
        let event = device.launch(program_id, &mut pool.pool, args, vec![])?;
        pool.pool.sync_events(vec![event])?;
        let nanos = nanos.elapsed().as_nanos();
        self.visited.insert(optimization.clone(), nanos);
        Ok(nanos)
    }
}
