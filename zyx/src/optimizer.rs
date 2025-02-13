#![allow(unused)]

use crate::{backend::DeviceInfo, kernel::{Kernel, Op}, rng::Rng, shape::Dimension, Set};

pub(super) struct Optimizer<'a> {
    rng: Rng,
    kernel: &'a Kernel,
    visited: Set<Optimization>,
    best_node: Optimization,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Optimization {
    pub shape: [Dimension; 9],
    pub ops: Set<OptOp>,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

        Optimization { shape: [gx/lx, gy/ly, gz/lz, lx, ly, lz, 1, 1, 1], ops: Set::default() }
    }
}

impl<'a> Optimizer<'a> {
    fn new(rng: Rng, kernel: &'a Kernel, dev_info: &DeviceInfo) -> Self {
        Self {
            kernel,
            rng,
            visited: Set::with_hasher(Default::default()),
            best_node: Optimization::new(kernel, dev_info),
        }
    }

    /// Next tries a new optimization given access to the device and memory, so that it can run it.
    fn next(&mut self) {
        // I think tinygrad picks an optimization and then finds the best reshape, then adds another optimization and finds the best reshape and so on
        // First list all possible optimizations

        // Then delete those optimizations that were already visited

        // Then randomly pick n optimizations and test every one of those
    }

    fn bench_optimization(optimization: &[OptOp]) -> u128 {
        // tests given optimization, returning time taken in nanoseconds
        0
    }
}
