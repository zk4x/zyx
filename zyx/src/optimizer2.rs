#![allow(unused)]

use crate::{backend::DeviceInfo, kernel::{Kernel, Op}, shape::Dimension, Set};

struct Optimization {
    reshape: Vec<Dimension>,
    // id of loop, which loop with this id, into how large dimension should it split
    splits: Vec<(u16, u16, Dimension)>,
}

struct Optimizer {
    search_iters: usize,
    visited: Set<Optimization>,
    tree: Node,
}

struct Node {
    optimization: Optimization,
    children: Vec<Node>,
}

impl Optimizer {
    fn new(kernel: &Kernel, dev_info: &DeviceInfo, search_iters: usize) -> Self {
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
        
        let lx = 1;
        let ly = 1;
        let lz = 1;

        let num_loops = kernel.ops.iter().position(|op| !matches!(op, Op::Loop { .. })).unwrap();
        debug_assert_ne!(num_loops, 0);
        let shape = kernel.shape();
        let reshape = match num_loops {
            0 => unreachable!(),
            1 => [1, 1, 1, 1, shape[0] / lz, lz].into(),
            2 => [1, 1, shape[0] / ly, ly, shape[1] / lz, lz].into(),
            3 => [shape[0] / lx, lx, shape[1] / ly, ly, shape[2] / lz, lz].into(),
            _ => [
                shape[0..num_loops - 2].iter().product::<usize>() / lx,
                lx,
                shape[num_loops - 2] / ly,
                ly,
                shape[num_loops - 1] / lz,
                lz,
            ].into(),
        };

        /*for lx in (1..=mlws.min(mlwd[0])).filter(|x| gws[0] % x == 0) {
            for ly in (1..=(mlws / lx).min(mlwd[1])).filter(|y| gws[1] % y == 0) {
                for lz in (1..=(mlws / (lx * ly)).min(mlwd[2])).filter(|z| gws[2] % z == 0) {
                    // register work size
                    for rx in (1..=mrws.min(mrwd[0])).filter(|x| (gws[0] / lx) % x == 0) {
                        for ry in (1..=(mrws / rx).min(mrwd[1])).filter(|y| (gws[1] / ly) % y == 0)
                        {
                            for rz in (1..=(mrws / (rx * ry)).min(mrwd[2]))
                                .filter(|z| (gws[2] / lz) % z == 0)
                            {
                            }
                        }
                    }
                }
            }
        }*/

        Self {
            search_iters,
            visited: Set::with_hasher(Default::default()),
            tree: Node {
                optimization: Optimization { reshape, splits: Vec::new() },
                children: Vec::new(),
            },
        }
    }
}
