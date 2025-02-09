#![allow(unused)]

use crate::{backend::DeviceInfo, kernel::{Kernel, Op}, rng::Rng, shape::Dimension, Set};

struct Optimizer {
    rng: Rng,
    search_iters: usize,
    visited: Set<Vec<OptOp>>,
    tree: Node,
}

struct Node {
    optimization: Vec<OptOp>,
    children: Vec<Node>,
}

enum OptOp {
    MergeLoop {},
    SplitLoop {
        id: u16,
        order: u16,
        len: Dimension,
    },
    UpCastLoop {},
    DownCastLoop {},
}

impl Optimizer {
    fn new(rng: Rng, kernel: &Kernel, dev_info: &DeviceInfo, search_iters: usize) -> Self {
        Self {
            rng,
            search_iters,
            visited: Set::with_hasher(Default::default()),
            tree: Node {
                optimization: Vec::new(),
                children: Vec::new(),
            },
        }
    }

    fn new_node(&self) {
        // Pick a ramdom node, better performing nodes have higher chance to be picked.
        let exploration_prob = 0.3;
        //self.rng.rand();
    }
}
