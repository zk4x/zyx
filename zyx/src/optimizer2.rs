#![allow(unused)]

use crate::{backend::DeviceInfo, kernel::{Kernel, Op}, rng::Rng, shape::Dimension, Set};

struct Optimizer<'a> {
    rng: Rng,
    kernel: &'a Kernel,
    search_iters: usize,
    visited: Set<Set<OptOp>>,
    best_node: Set<OptOp>,
}

enum OptOp {
    //MergeLoop {},
    SplitLoop {
        id: u16,
        order: u16,
        len: Dimension,
    },
    UpCastLoop {
        id: u16,
        order: u16,
    },
    DownCastLoop {
        id: u16,
        order: u16,
    },
}

impl<'a> Optimizer<'a> {
    fn new(rng: Rng, kernel: &'a Kernel, dev_info: &DeviceInfo, search_iters: usize) -> Self {
        Self {
            kernel,
            rng,
            search_iters,
            visited: Set::with_hasher(Default::default()),
            best_node: Set::with_hasher(Default::default()),
        }
    }

    fn next(&mut self) {
        // go from best, node,
        // try 10 or 100 different nodes, record timings and store to visited
        // pick the fastest node
        // set the fastest node as the new best node
        // repeat

        // First list all possible optimizations

        // Then delete those optimizations that were already visited

        // Then randomly pick n optimizations and test every one of those
    }

    fn bench_optimization(optimization: &[OptOp]) -> u128 {
        // tests given optimization, returning time taken in nanoseconds
        0
    }
}
