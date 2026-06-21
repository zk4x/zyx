#![allow(unused)]

use std::collections::BTreeSet;

use crate::{
    DType, Map,
    dtype::Constant,
    graph::{
        Node,
        compiled::{CachedGraph, CompiledGraph, NodeId},
    },
    kernel::{BOp, UOp},
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
};

trait FusedKernel: std::fmt::Debug {
    fn try_fuse(g: &mut EGraph, nid: NodeId) -> Option<Self>
    where
        Self: Sized;
}

#[derive(Hash)]
struct FusedSlot {
    pre_reshape: Option<Vec<Dim>>,
    pre_permute: Option<Vec<UAxis>>,
    fused_nodes: BTreeSet<NodeId>,
    post_permute: Option<Vec<UAxis>>,
    post_reshape: Option<Vec<Dim>>,
}

pub struct EGraph<'a> {
    graph: &'a CachedGraph,
    kernels: Map<FusedSlot, Box<dyn FusedKernel>>,
}

impl<'a> EGraph<'a> {
    pub fn new(graph: &'a CachedGraph) -> Self {
        Self { graph, kernels: Map::default() }
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
    pub fn saturate(&mut self) {}

    pub fn extract(self) -> CompiledGraph {
        todo!()
    }
}

#[derive(Debug)]
struct Matmul {}

impl FusedKernel for Matmul {
    fn try_fuse(g: &mut EGraph, nid: NodeId) -> Option<Self>
    where
        Self: Sized,
    {
        todo!()
    }
}
