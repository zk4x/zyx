//! E-graph for tensor operation equivalence and optimization.
//!
//! The graph supports rewrites that produce equivalent forms of a computation:
//! - **CSE** (common subexpression elimination) via hashconsing
//! - **Algebraic rewrites** like transpose fusion: `transpose(A) @ transpose(B)` ↔ `(B @ A).transpose()`
//! - **Layout rewrites**: a matmul can be realized as transposed or un-transposed,
//!   with the transpose either fused into the kernel or materialized as a separate
//!   pre-processing step
//! - **Shape rewrites**: reshape and padding can be fused into adjacent ops or
//!   split out as separate nodes
//!
//! Each equivalence class (`EClass`) holds all equivalent node forms. A cost
//! model selects the cheapest extraction for kernel compilation.

use crate::{
    DType, Map,
    backend::ProgramId,
    dtype::Constant,
    kernel::{BOp, DeviceId, UOp},
    runtime::ShapeId,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

impl From<usize> for NodeId {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}
impl From<NodeId> for usize {
    fn from(v: NodeId) -> usize {
        v.0 as usize
    }
}

impl SlabId for NodeId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClassId(pub u32);

impl From<usize> for ClassId {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}
impl From<ClassId> for usize {
    fn from(v: ClassId) -> usize {
        v.0 as usize
    }
}

impl SlabId for ClassId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Node {
    Const(Constant),
    Leaf {
        dtype: DType,
        shape: ShapeId,
    },
    Expand {
        x: ClassId,
        shape: ShapeId,
    },
    Permute {
        x: ClassId,
        axes: Box<[UAxis]>,
    },
    Reshape {
        x: ClassId,
        shape: ShapeId,
    },
    PadZeros {
        x: ClassId,
        padding: Box<[(i64, i64)]>,
    },
    Reduce {
        x: ClassId,
        bop: BOp,
        axes: Box<[UAxis]>,
    },
    Cast {
        x: ClassId,
        dtype: DType,
    },
    Unary {
        x: ClassId,
        uop: UOp,
    },
    Binary {
        x: ClassId,
        y: ClassId,
        bop: BOp,
    },
    ToDevice {
        x: ClassId,
        device: DeviceId,
    },
    Kernel {
        inputs: Box<[ClassId]>,
        outputs: Box<[ClassId]>,
        program_id: ProgramId,
    },
}

#[derive(Debug)]
struct NodeData {
    node: Node,
    class_of: ClassId,
}

#[derive(Debug)]
pub struct EClass {
    pub nodes: Vec<NodeId>,
    pub shape: ShapeId,
    pub dtype: DType,
}

#[derive(Debug)]
pub struct Graph {
    hashcons: Map<Node, NodeId>,
    nodes: Slab<NodeId, NodeData>,
    pub(crate) classes: Slab<ClassId, EClass>,
    // Node -> Kernel, cost
    //kernel_map: Map<NodeId, (KernelId, u64)>,
}

impl Node {
    fn class_params(&self) -> Vec<ClassId> {
        match self {
            Self::Const(_) | Self::Leaf { .. } => vec![],
            Self::Expand { x, .. } => vec![*x],
            Self::Permute { x, .. } => vec![*x],
            Self::Reshape { x, .. } => vec![*x],
            Self::PadZeros { x, .. } => vec![*x],
            Self::Reduce { x, .. } => vec![*x],
            Self::Cast { x, .. } => vec![*x],
            Self::Unary { x, .. } => vec![*x],
            Self::Binary { x, y, .. } => vec![*x, *y],
            Self::ToDevice { x, .. } => vec![*x],
            Self::Kernel { inputs, outputs, .. } => {
                let mut deps = inputs.to_vec();
                deps.extend(outputs.iter().copied());
                deps
            }
        }
    }
}

impl Graph {
    pub fn new() -> Self {
        Self { hashcons: Map::default(), nodes: Slab::new(), classes: Slab::new() }
    }

    pub fn topo_sort_classes(&self, outputs: &[ClassId]) -> Vec<ClassId> {
        let mut rcs: Map<ClassId, u32> = Map::default();
        let mut stack: Vec<ClassId> = outputs.to_vec();
        while let Some(cid) = stack.pop() {
            rcs.entry(cid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                let mut deps = Vec::new();
                for nid in &self.classes[cid].nodes {
                    for p in self.nodes[*nid].node.class_params() {
                        if !deps.contains(&p) {
                            deps.push(p);
                        }
                    }
                }
                stack.extend(deps);
                1
            });
        }

        let mut order = Vec::new();
        let mut internal_rcs: Map<ClassId, u32> = Map::default();
        let mut stack: Vec<ClassId> = outputs.to_vec();
        while let Some(cid) = stack.pop() {
            if let Some(&rc) = rcs.get(&cid) {
                let visited = internal_rcs.entry(cid).and_modify(|c| *c += 1).or_insert(1);
                if rc == *visited {
                    order.push(cid);
                    let mut deps = Vec::new();
                    for nid in &self.classes[cid].nodes {
                        for p in self.nodes[*nid].node.class_params() {
                            if !deps.contains(&p) {
                                deps.push(p);
                            }
                        }
                    }
                    stack.extend(deps);
                }
            }
        }
        order.reverse();
        order
    }

    pub fn fill_remaining(&mut self, outputs: &[ClassId], shapes: &Slab<ShapeId, Vec<Dim>>) {
        let order = self.topo_sort_classes(outputs);
        for cid in &order {
            let class = &self.classes[*cid];
            if class.shape != ShapeId::NULL {
                println!("Class {cid:?}: shape={:?} dtype={:?}", shapes[class.shape], class.dtype);
            } else {
                println!("Class {cid:?}: shape=NULL dtype={:?}", class.dtype);
            }
            for nid in &class.nodes {
                println!("  Node {nid:?}: {:?}", self.nodes[*nid].node);
            }
        }
    }

    pub fn push(&mut self, node: Node, shape: ShapeId, dtype: DType) -> (NodeId, ClassId) {
        if let Some(&nid) = self.hashcons.get(&node) {
            return (nid, self.nodes[nid].class_of);
        }
        let nid = self.nodes.push(NodeData { node: node.clone(), class_of: ClassId::NULL });
        let cid = self.classes.push(EClass { nodes: vec![nid], shape, dtype });
        self.nodes[nid].class_of = cid;
        self.hashcons.insert(node, nid);
        (nid, cid)
    }
}
