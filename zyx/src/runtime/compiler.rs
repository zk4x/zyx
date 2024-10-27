//! This is a graph compiler
//!
//! Compiler takes graph, turns it into a series of kernels. THese kernels get scheduled
//! to multiple devices and platforms. Each kernel get optimized multiple times for the single
//! platform.

use crate::runtime::{Node, Graph, BOp};
use crate::tensor::TensorId;
use crate::dtype::Constant;

// Runtime will then contain scheduler instead of CompiledGraph
// After compilation scheduler is immutable.
// Scheduler decides which kernels get assigned to which devices
struct Scheduler {
    //memory_pools,
    //devices,
    kernels: Vec<Kernel>,
}

struct Kernel {}

enum Reg {
    Var(u16),
    Const(Constant),
}

enum Op {
    Binary {
        z: Reg,
        x: Reg,
        y: Reg,
        bop: BOp,
    }
}

impl Kernel {
    fn load() -> Kernel {
        todo!()
    }
}

impl Scheduler {
    // Adds a new unprepared kernel to the scheduler
    fn push(&mut self, kernel: Kernel) {
    }

    fn compile(graph: Graph, order: &[TensorId]) -> Scheduler {
        let mut scheduler = Scheduler {
            kernels: Vec::new(),
        };
        for &nid in order {
            match &graph[nid] {
                Node::Leaf => {
                    scheduler.push(Kernel::load());
                }
                _ => todo!(),
            }
        }
        scheduler
    }

    fn run(&self, memory_pools: &mut (), devices: &mut ()) {
        todo!()
    }
}

