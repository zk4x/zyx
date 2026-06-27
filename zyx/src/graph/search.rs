// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#![allow(unused)]

use std::collections::BTreeSet;

use crate::{
    DType, Map,
    dtype::Constant,
    graph::{Node, compiled::CompiledNode},
    kernel::{BOp, UOp},
    shape::{Dim, UAxis},
    slab::Slab,
    tensor::TensorId,
};

use super::Graph;

trait FusedKernel: std::fmt::Debug {
    fn try_fuse(g: &mut EGraph, nid: TensorId) -> Option<Self>
    where
        Self: Sized;
}

#[derive(Debug, Hash, PartialEq, Eq)]
struct FusedSlot {
    pre_reshape: Option<Vec<Dim>>,
    pre_permute: Option<Vec<UAxis>>,
    fused_nodes: BTreeSet<TensorId>,
    post_permute: Option<Vec<UAxis>>,
    post_reshape: Option<Vec<Dim>>,
}

#[derive(Debug)]
pub struct EGraph<'a> {
    order: &'a [TensorId],
    graph: &'a Graph,
    kernels: Map<FusedSlot, Box<dyn FusedKernel>>,
}

impl<'a> EGraph<'a> {
    pub fn new(order: &'a [TensorId], graph: &'a Graph) -> Self {
        Self { order, graph, kernels: Map::default() }
    }

    /// We have an array of all available fused kernels.
    /// 1. Filter only those for which we have available devices.
    /// 2. Go from largest.
    /// 3. Continue adding kernels to kernels until the end of the graph.
    /// 4. Now start from the beginning of the graph. Skip the already fused ops
    ///    and fuse again if possible, still using the largest fused kernel.
    /// 5. Iterate with the largest kernel as long as at least one largest kernel
    ///    can be added outside of ops that are already fused.
    ///    Since we are skipping already fused ones, this is not perfectly optimal,
    ///    but close enough.
    /// 6. After all largest kernel fusions were exhausted go with the next largest
    ///    kernel and do the same iterative exhaustive process.
    /// 7. Once all fused kernels were tried, fill in the smallest gaps with zyx
    ///    kernels, first just the smallest gaps.
    /// 8. Then continue by filling larger and larger gaps with zyx kernels, depending
    ///    on the budget.
    ///
    /// This whole saturation is driven by budget. The higher budget, the more fusion
    /// variants are tried. With large enough budgets, it's basically fully exhaustive.
    pub fn saturate(&mut self) {
        let mut unfused: BTreeSet<TensorId> = self.order.iter().copied().collect();

        let fusers = [Matmul::try_fuse];

        loop {
            let mut found = false;
            for fuser in fusers {
                // Fixpoint: keep trying until a full pass consumes nothing.
                for &nid in self.order {
                    if !unfused.contains(&nid) {
                        continue;
                    }
                    if let Some(kernel) = fuser(self, nid) {
                        unfused.remove(&nid);
                        let slot = FusedSlot {
                            pre_reshape: None,
                            pre_permute: None,
                            fused_nodes: BTreeSet::from([nid]),
                            post_permute: None,
                            post_reshape: None,
                        };
                        self.kernels.insert(slot, Box::new(kernel));
                        found = true;
                    }
                }
            }
            if !found {
                break;
            }
        }
    }

    pub fn extract(self) -> Vec<CompiledNode> {
        for kernel in self.kernels {
            println!("{kernel:?}");
        }
        todo!()
    }
}

#[derive(Debug)]
struct Matmul {}

impl FusedKernel for Matmul {
    fn try_fuse(g: &mut EGraph, nid: TensorId) -> Option<Self>
    where
        Self: Sized,
    {
        // Pattern: a matmul is compiled as:
        //   reduce_sum over dim 1
        //     ← binary mul
        //         ← expand ← leaf  (A side)
        //         ← expand ← reshape ← permute[1,0] ← leaf  (B^T side)

        // Check nid is a reduction summing over the contracting dimension.
        let mul_id = match g.graph[nid] {
            Node::Reduce { x, rop: BOp::Add } => x,
            _ => return None,
        };
        if g.graph.axes(nid) != &[1] {
            return None;
        }

        // Check the reduction input is an element-wise multiply.
        let (left, right) = match g.graph[mul_id] {
            Node::Binary { x, y, bop: BOp::Mul } => (x, y),
            _ => return None,
        };

        // Try both orderings: the two branches can be on either side of the mul.
        for &(a, b) in &[(left, right), (right, left)] {
            // A side: Expand ← Leaf
            let a_leaf = match g.graph[a] {
                Node::Expand { x } => x,
                _ => continue,
            };
            if !matches!(g.graph[a_leaf], Node::Leaf { .. }) {
                continue;
            }

            // B side: Expand ← Reshape ← Permute[1,0] ← Leaf
            let reshape_id = match g.graph[b] {
                Node::Expand { x } => x,
                _ => continue,
            };
            let permute_id = match g.graph[reshape_id] {
                Node::Reshape { x } => x,
                _ => continue,
            };
            let b_leaf = match g.graph[permute_id] {
                Node::Permute { x } => x,
                _ => continue,
            };
            if g.graph.axes(permute_id) != &[1, 0] {
                continue;
            }
            if !matches!(g.graph[b_leaf], Node::Leaf { .. }) {
                continue;
            }

            return Some(Matmul {});
        }

        None
    }
}
