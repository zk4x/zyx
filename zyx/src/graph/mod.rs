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
    DType, Map, ZyxError,
    backend::ProgramId,
    dtype::Constant,
    kernel::{BOp, DeviceId, Kernel, UOp},
    runtime::{Runtime, ShapeId},
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
};

mod kernelizer;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EKernelId(pub u32);

impl From<usize> for EKernelId {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}
impl From<EKernelId> for usize {
    fn from(v: EKernelId) -> usize {
        v.0 as usize
    }
}
impl SlabId for EKernelId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug)]
pub struct EKernelData {
    pub(crate) kernel: Kernel,
    pub(crate) outputs: Vec<ClassId>,
    pub(crate) loads: Vec<ClassId>,
    pub(crate) stores: Vec<ClassId>,
}

#[derive(Debug)]
pub struct Graph {
    hashcons: Map<Node, NodeId>,
    nodes: Slab<NodeId, NodeData>,
    pub(crate) classes: Slab<ClassId, EClass>,
    pub(crate) ekernels: Slab<EKernelId, EKernelData>,
    pub(crate) kernel_map: Map<NodeId, EKernelId>,
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
        Self {
            hashcons: Map::default(),
            nodes: Slab::new(),
            classes: Slab::new(),
            ekernels: Slab::new(),
            kernel_map: Map::default(),
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

    pub fn debug_print(&self, shapes: &Slab<ShapeId, Vec<Dim>>) {
        let line = "─".repeat(60);
        println!("\n{}", line);
        println!("  E-Graph");
        println!("{}", line);
        for cid in self.classes.ids() {
            let class = &self.classes[cid];
            let shape_str = format!("{:?}", &shapes[class.shape]);
            let dtype_str = format!("{:?}", &class.dtype);
            println!("Class {:?} shape={} dtype={}", cid, shape_str, dtype_str);
            for &nid in &class.nodes {
                let kind = &self.nodes[nid].node;
                let inputs: Vec<ClassId> = match kind {
                    Node::Kernel { inputs, .. } => inputs.to_vec(),
                    _ => kind.class_params(),
                };
                let name = match kind {
                    Node::Reduce { bop, .. } => format!("Reduce {:?}", bop),
                    Node::Binary { bop, .. } => format!("Binary {:?}", bop),
                    Node::Unary { uop, .. } => format!("Unary {:?}", uop),
                    Node::Cast { dtype, .. } => format!("Cast {:?}", dtype),
                    Node::Kernel { program_id, .. } => format!("Kernel prog={:?}", program_id),
                    Node::Expand { .. } => "Expand".into(),
                    Node::Permute { axes, .. } => format!("Permute {:?}", axes),
                    Node::Reshape { shape, .. } => format!("Reshape {:?}", shapes[*shape]),
                    Node::PadZeros { padding, .. } => format!("Pad {:?}", padding),
                    Node::ToDevice { device, .. } => format!("ToDevice {:?}", device),
                    Node::Const(v) => format!("Const {:?}", v),
                    Node::Leaf { dtype, .. } => format!("Leaf {:?}", dtype),
                };
                println!("  {name} {nid:?}: inputs={inputs:?}");
            }
        }
        println!("{}\n", line);
    }

    pub fn add_memory_ops(&mut self) {
        todo!()
    }
}

impl Runtime {
    pub fn autotune_all_kernels(&mut self) -> Result<(), ZyxError> {
        let kernel_data: Vec<(NodeId, Kernel)> = if let Some(ref graph) = self.graph {
            let mut v = Vec::new();
            for cid in graph.classes.ids() {
                for &nid in &graph.classes[cid].nodes {
                    if matches!(&graph.nodes[nid].node, Node::Kernel { .. }) {
                        if let Some(&kid) = graph.kernel_map.get(&nid) {
                            v.push((nid, graph.ekernels[kid].kernel.clone()));
                        }
                    }
                }
            }
            v
        } else {
            Vec::new()
        };

        for (nid, kernel) in &kernel_data {
            let (flop, read, write) = kernel.flop_mem_rw();
            let device_ids: Vec<DeviceId> = self.devices.ids().collect();
            let (inputs, outputs, class_of) = if let Some(ref graph) = self.graph {
                let node = &graph.nodes[*nid];
                let Node::Kernel { ref inputs, ref outputs, .. } = node.node else {
                    unreachable!()
                };
                (inputs.clone(), outputs.clone(), node.class_of)
            } else {
                continue;
            };
            //println!("device_ids.len={}", device_ids.len());
            for (i, &dev_id) in device_ids.iter().enumerate() {
                let pool_id = self.devices[dev_id].memory_pool_id();
                let mut kernel = kernel.clone();
                kernel.device_id = dev_id;
                let (dev_prog, _timing) = self.get_or_autotune(kernel, pool_id, flop, read, write)?;
                let prog = ProgramId { device: dev_id, program: dev_prog };
                if let Some(ref mut graph) = self.graph {
                    if i == 0 {
                        if let Node::Kernel { program_id, .. } = &mut graph.nodes[*nid].node {
                            *program_id = prog;
                        }
                    } else {
                        let knid = graph.nodes.push(NodeData {
                            node: Node::Kernel { inputs: inputs.clone(), outputs: outputs.clone(), program_id: prog },
                            class_of,
                        });
                        for &ocid in &*outputs {
                            graph.classes[ocid].nodes.push(knid);
                        }
                        if !outputs.contains(&class_of) {
                            graph.classes[class_of].nodes.push(knid);
                        }
                    }
                }
            }
            //println!("len={}", self.graph.as_ref().unwrap().nodes.iter().filter(|x| matches!(x.1.node, Node::Kernel { .. })).count());
        }

        Ok(())
    }
}
