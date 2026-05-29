use crate::{
    DType,
    dtype::Constant,
    graph::{
        Node,
        compiled::{CachedGraph, CompiledGraph},
    },
    kernel::{BOp, UOp},
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
};

#[derive(Debug, Copy, Clone, Hash, Ord, PartialEq, PartialOrd, Eq)]
struct NodeId(u32);

impl SlabId for NodeId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl From<usize> for NodeId {
    fn from(value: usize) -> Self {
        Self(value as u32)
    }
}

impl Into<usize> for NodeId {
    fn into(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug)]
enum ENode {
    Const {
        value: Constant,
    },
    Leaf {
        shape: Box<[Dim]>,
        dtype: DType,
    },
    Cast {
        x: NodeId,
        dtype: DType,
    },
    Unary {
        x: NodeId,
        uop: UOp,
    },
    Binary {
        x: NodeId,
        y: NodeId,
        bop: BOp,
    },
    Reshape {
        x: NodeId,
        shape: Box<[Dim]>,
    },
    Expand {
        x: NodeId,
        shape: Box<[Dim]>,
    },
    Permute {
        x: NodeId,
        axes: Box<[UAxis]>,
        shape: Box<[Dim]>,
    },
    Pad {
        x: NodeId,
        padding: Box<[(i64, i64)]>,
        shape: Box<[Dim]>,
    },
    Reduce {
        x: NodeId,
        axes: Box<[UAxis]>,
        rop: BOp,
        shape: Box<[Dim]>,
    },
    Fused(Box<dyn FusedOp>),
}

trait FusedOp: std::fmt::Debug {
    fn try_fuse(g: &mut EGraph, nid: NodeId) -> Option<Self>
    where
        Self: Sized;
}

pub struct EGraph {
    nodes: Slab<NodeId, ENode>,
}

impl EGraph {
    pub fn new(graph: &CachedGraph) -> EGraph {
        let mut nodes: Slab<NodeId, ENode> = Slab::new();

        for (tid, node) in graph.nodes.iter().enumerate() {
            let enode = match *node {
                Node::Const { value } => ENode::Const { value },
                Node::Leaf { dtype } => ENode::Leaf { shape: graph.shape(tid).into(), dtype },
                Node::Expand { x } => ENode::Expand { x: NodeId(x.0), shape: graph.shape(tid).into() },
                Node::Permute { x } => {
                    ENode::Permute { x: NodeId(x.0), axes: graph.axes[&tid].clone(), shape: graph.shape(tid).into() }
                }
                Node::Reshape { x } => ENode::Reshape { x: NodeId(x.0), shape: graph.shape(tid).into() },
                Node::Pad { x } => {
                    ENode::Pad { x: NodeId(x.0), padding: graph.paddings[&tid].clone(), shape: graph.shape(tid).into() }
                }
                Node::Reduce { x, rop } => {
                    ENode::Reduce { x: NodeId(x.0), axes: graph.axes[&tid].clone(), rop, shape: graph.shape(tid).into() }
                }
                Node::Cast { x, dtype } => ENode::Cast { x: NodeId(x.0), dtype },
                Node::Unary { x, uop } => ENode::Unary { x: NodeId(x.0), uop },
                Node::Binary { x, y, bop } => ENode::Binary { x: NodeId(x.0), y: NodeId(x.0), bop },
                Node::Custom(_) => todo!(),
            };

            nodes.push(enode);
        }

        for (nid, node) in nodes.iter() {
            println!("{nid:?} -> {node:?}");
        }

        EGraph { nodes }
    }

    pub fn saturate(&mut self) {
        let ids: Vec<NodeId> = self.nodes.ids().collect();
        for nid in ids {
            if let Some(fused_op) = Matmul::try_fuse(self, nid) {
                self.nodes.push(ENode::Fused(Box::new(fused_op)));
            }
        }
        todo!()
    }

    pub fn extract(self) -> CompiledGraph {
        todo!()
    }
}

#[derive(Debug)]
struct Matmul {}

impl FusedOp for Matmul {
    fn try_fuse(g: &mut EGraph, nid: NodeId) -> Option<Self>
    where
        Self: Sized,
    {
        todo!()
    }
}
