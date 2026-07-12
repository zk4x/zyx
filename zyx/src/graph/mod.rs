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
    kernel::{BOp, DeviceId, Kernel, Op, OpId, UOp},
    runtime::ShapeId,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    view::View,
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
        Self { hashcons: Map::default(), nodes: Slab::new(), classes: Slab::new(), ekernels: Slab::new() }
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
        let mut class_map: Map<ClassId, (EKernelId, OpId)> = Map::default();

        for &cid in &order {
            let class = &self.classes[cid];
            let nid = class.nodes[0];
            let node = &self.nodes[nid].node;

            match *node {
                Node::Leaf { dtype, shape } => {
                    let shape_vec = shapes[shape].clone();
                    let mut kernel = Kernel::new(DeviceId::AUTO);
                    let op_id = kernel.load_contiguous(dtype, &shape_vec);
                    let kid = self.ekernels.push(EKernelData {
                        kernel,
                        outputs: vec![cid],
                        loads: vec![cid],
                        stores: Vec::new(),
                    });
                    class_map.insert(cid, (kid, op_id));
                }
                Node::Const(c) => {
                    let mut kernel = Kernel::new(DeviceId::AUTO);
                    let op_id = kernel.push_back(Op::ConstView(Box::new((c, View::contiguous(&[1])))));
                    let kid = self.ekernels.push(EKernelData {
                        kernel,
                        outputs: vec![cid],
                        loads: Vec::new(),
                        stores: Vec::new(),
                    });
                    class_map.insert(cid, (kid, op_id));
                }
                Node::Unary { x, uop } => {
                    let (kid, op_id) = class_map[&x];
                    let new_op_id = self.ekernels[kid].kernel.unary(op_id, uop);
                    self.ekernels[kid].outputs.retain(|&o| o != x);
                    self.ekernels[kid].outputs.push(cid);
                    class_map.insert(cid, (kid, new_op_id));
                }
                Node::Cast { x, dtype } => {
                    let (kid, op_id) = class_map[&x];
                    let new_op_id = self.ekernels[kid].kernel.cast(op_id, dtype);
                    self.ekernels[kid].outputs.retain(|&o| o != x);
                    self.ekernels[kid].outputs.push(cid);
                    class_map.insert(cid, (kid, new_op_id));
                }
                Node::Binary { x, y, bop } => {
                    let (kid_x, op_id_x) = class_map[&x];
                    let (kid_y, op_id_y) = class_map[&y];

                    let (kid, op_id) = if kid_x == kid_y {
                        let op_id = self.ekernels[kid_x].kernel.binary(op_id_x, op_id_y, bop);
                        (kid_x, op_id)
                    } else {
                        let x_stores = !self.ekernels[kid_x].stores.is_empty();
                        let y_stores = !self.ekernels[kid_y].stores.is_empty();
                        if x_stores || y_stores {
                            todo!("binary with stores not yet handled");
                        }
                        let swap =
                            self.ekernels[kid_y].kernel.is_reduce() && !self.ekernels[kid_x].kernel.is_reduce();
                        let (keep_kid, merge_kid, keep_op, merge_op) = if swap {
                            (kid_y, kid_x, op_id_y, op_id_x)
                        } else {
                            (kid_x, kid_y, op_id_x, op_id_y)
                        };

                        let EKernelData { outputs: merge_outputs, loads: merge_loads, stores: merge_stores, kernel } =
                            unsafe { self.ekernels.remove_and_return(merge_kid) };
                        let Kernel { ops: ref merge_ops, head: merge_head, .. } = kernel;

                        let mut op_map: Map<OpId, OpId> = Map::default();
                        let mut i = merge_head;
                        while !i.is_null() {
                            let mut op = merge_ops[i].op.clone();
                            for param in op.parameters_mut() {
                                if let Some(&new_param) = op_map.get(param) {
                                    *param = new_param;
                                }
                            }
                            let new_op_id = self.ekernels[keep_kid].kernel.push_back(op);
                            op_map.insert(i, new_op_id);
                            i = merge_ops[i].next;
                        }

                        for (_, (ekid, eop_id)) in class_map.iter_mut() {
                            if *ekid == merge_kid {
                                *ekid = keep_kid;
                                if let Some(&new_op_id) = op_map.get(eop_id) {
                                    *eop_id = new_op_id;
                                }
                            }
                        }

                        let keep_data = &mut self.ekernels[keep_kid];
                        keep_data.outputs.extend(merge_outputs);
                        keep_data.loads.extend(merge_loads);
                        keep_data.stores.extend(merge_stores);

                        let op_id = if swap {
                            self.ekernels[keep_kid].kernel.binary(op_map[&merge_op], keep_op, bop)
                        } else {
                            self.ekernels[keep_kid].kernel.binary(keep_op, op_map[&merge_op], bop)
                        };
                        (keep_kid, op_id)
                    };

                    self.ekernels[kid].outputs.push(cid);
                    class_map.insert(cid, (kid, op_id));
                }
                Node::Reduce { x, bop, ref axes } => {
                    let (kid, op_id) = class_map[&x];
                    let input_shape = shapes[self.classes[x].shape].clone();
                    let input_dtype = self.classes[x].dtype;

                    let (mut kid, mut op_id) = Self::duplicate_or_store_class(
                        &mut self.ekernels,
                        kid,
                        op_id,
                        x,
                        &input_shape,
                        input_dtype,
                    );

                    let shape = shapes[self.classes[x].shape].clone();
                    let max_axis = *axes.last().unwrap() as usize;
                    let mut ai = 0;
                    let mut permute_axes = Vec::with_capacity(shape.len());
                    for i in 0..=max_axis {
                        if axes[ai] as usize == i {
                            ai += 1;
                        } else {
                            permute_axes.push(i as UAxis);
                        }
                    }
                    permute_axes.extend((max_axis + 1..shape.len()).map(|i| i as UAxis));
                    permute_axes.extend_from_slice(axes);

                    if !permute_axes.iter().copied().eq(0..permute_axes.len() as UAxis) {
                        op_id = self.ekernels[kid].kernel.permute(op_id, &permute_axes);
                    }

                    op_id = self.ekernels[kid].kernel.push_back(Op::Reduce {
                        x: op_id,
                        rop: bop,
                        n_axes: axes.len() as UAxis,
                    });

                    if shape.len() == axes.len() {
                        op_id = self.ekernels[kid].kernel.reshape(op_id, &[1]);
                    }

                    self.ekernels[kid].outputs.push(cid);
                    class_map.insert(cid, (kid, op_id));
                    self.add_equivalence(
                        Node::Kernel {
                            inputs: Box::new([x]),
                            outputs: Box::new([cid]),
                            program_id: ProgramId::NULL,
                        },
                        cid,
                    );
                }
                Node::Reshape { x, shape } => {
                    let (kid, op_id) = class_map[&x];
                    let input_shape = shapes[self.classes[x].shape].clone();
                    let input_dtype = self.classes[x].dtype;

                    let (mut kid, mut op_id) = Self::duplicate_or_store_class(
                        &mut self.ekernels,
                        kid,
                        op_id,
                        x,
                        &input_shape,
                        input_dtype,
                    );

                    let shape_vec = shapes[shape].clone();
                    op_id = self.ekernels[kid].kernel.reshape(op_id, &shape_vec);

                    self.ekernels[kid].outputs.push(cid);
                    class_map.insert(cid, (kid, op_id));
                    self.add_equivalence(
                        Node::Kernel { inputs: Box::new([x]), outputs: Box::new([cid]), program_id: ProgramId::NULL },
                        cid,
                    );
                }
                Node::Expand { x, shape } => {
                    let (kid, op_id) = class_map[&x];
                    let input_shape = shapes[self.classes[x].shape].clone();
                    let input_dtype = self.classes[x].dtype;

                    let (mut kid, mut op_id) = Self::duplicate_or_store_class(
                        &mut self.ekernels,
                        kid,
                        op_id,
                        x,
                        &input_shape,
                        input_dtype,
                    );

                    let shape_vec = shapes[shape].clone();
                    op_id = self.ekernels[kid].kernel.expand(op_id, &shape_vec);

                    self.ekernels[kid].outputs.push(cid);
                    class_map.insert(cid, (kid, op_id));
                    self.add_equivalence(
                        Node::Kernel { inputs: Box::new([x]), outputs: Box::new([cid]), program_id: ProgramId::NULL },
                        cid,
                    );
                }
                Node::Permute { x, ref axes } => {
                    let (kid, op_id) = class_map[&x];
                    let input_shape = shapes[self.classes[x].shape].clone();
                    let input_dtype = self.classes[x].dtype;

                    let (mut kid, mut op_id) = Self::duplicate_or_store_class(
                        &mut self.ekernels,
                        kid,
                        op_id,
                        x,
                        &input_shape,
                        input_dtype,
                    );

                    op_id = self.ekernels[kid].kernel.permute(op_id, axes);

                    self.ekernels[kid].outputs.push(cid);
                    class_map.insert(cid, (kid, op_id));
                    self.add_equivalence(
                        Node::Kernel { inputs: Box::new([x]), outputs: Box::new([cid]), program_id: ProgramId::NULL },
                        cid,
                    );
                }
                Node::PadZeros { x, ref padding } => {
                    let (kid, op_id) = class_map[&x];
                    let input_shape = shapes[self.classes[x].shape].clone();
                    let input_dtype = self.classes[x].dtype;

                    let (mut kid, mut op_id) = Self::duplicate_or_store_class(
                        &mut self.ekernels,
                        kid,
                        op_id,
                        x,
                        &input_shape,
                        input_dtype,
                    );

                    op_id = self.ekernels[kid].kernel.pad(op_id, padding);

                    self.ekernels[kid].outputs.push(cid);
                    class_map.insert(cid, (kid, op_id));
                    self.add_equivalence(
                        Node::Kernel { inputs: Box::new([x]), outputs: Box::new([cid]), program_id: ProgramId::NULL },
                        cid,
                    );
                }
                _ => {}
            }
        }

        // Add Node::Kernel equivalence to each output class of each fused kernel.
        // The same Node::Kernel is added to every class that this kernel produces.
        let kernel_nodes: Vec<(Node, Vec<ClassId>)> = self
            .ekernels
            .iter()
            .map(|(_, ekdata)| {
                let inputs: Box<[ClassId]> = ekdata.loads.clone().into_boxed_slice();
                let outputs: Box<[ClassId]> = ekdata.outputs.clone().into_boxed_slice();
                (Node::Kernel { inputs, outputs, program_id: ProgramId::NULL }, ekdata.outputs.clone())
            })
            .collect();
        for (kernel_node, output_classes) in kernel_nodes {
            for &output_cid in &output_classes {
                self.add_equivalence(kernel_node.clone(), output_cid);
            }
        }

        self.debug_print(shapes);
    }

    pub fn debug_print(&self, shapes: &Slab<ShapeId, Vec<Dim>>) {
        let line = "─".repeat(60);
        println!("\n{line}");
        println!("  E-Graph");
        println!("{line}");
        for cid in self.classes.ids() {
            let class = &self.classes[cid];
            let shape_str = if class.shape != ShapeId::NULL {
                format!("{:?}", shapes[class.shape])
            } else {
                "NULL".into()
            };
            let dtype_str = format!("{:?}", class.dtype);
            println!("Class {cid:?} shape={shape_str} dtype={dtype_str}");
            for &nid in &class.nodes {
                let node = &self.nodes[nid].node;
                let inputs = node.class_params();
                let name = match node {
                    Node::Reduce { bop, .. } => format!("Reduce {:?}", bop),
                    Node::Binary { bop, .. } => format!("Binary {:?}", bop),
                    Node::Unary { uop, .. } => format!("Unary {:?}", uop),
                    Node::Cast { dtype, .. } => format!("Cast {:?}", dtype),
                    Node::Kernel { program_id, .. } => format!("Kernel prog={:?}", program_id),
                    Node::Expand { .. } => "Expand".into(),
                    Node::Permute { axes, .. } => format!("Permute {:?}", axes),
                    Node::Reshape { .. } => "Reshape".into(),
                    Node::PadZeros { .. } => "Pad".into(),
                    Node::ToDevice { device, .. } => format!("ToDevice {:?}", device),
                    Node::Const(v) => format!("Const {:?}", v),
                    Node::Leaf { .. } => "Leaf".into(),
                };
                println!("  {name} {nid:?}: inputs={inputs:?}");
            }
        }
        println!("{line}\n");
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

    /// Add a node as an equivalence to an existing class, skipping hashcons.
    /// Used for `Node::Kernel` equivalences that are specific to each class.
    pub fn add_equivalence(&mut self, node: Node, class_id: ClassId) -> NodeId {
        let nid = self.nodes.push(NodeData { node, class_of: class_id });
        self.classes[class_id].nodes.push(nid);
        nid
    }

    fn duplicate_or_store_class(
        ekernels: &mut Slab<EKernelId, EKernelData>,
        kid: EKernelId,
        op_id: OpId,
        input_cid: ClassId,
        input_shape: &[Dim],
        input_dtype: DType,
    ) -> (EKernelId, OpId) {
        let contains_stores = ekernels[kid].kernel.contains_stores();
        let preceded_by_reduce = ekernels[kid].kernel.is_preceded_by_reduce(op_id);

        let (mut kid, mut op_id) = if contains_stores || preceded_by_reduce {
            let mut new_kernel = Kernel::new(DeviceId::AUTO);
            let load_op = new_kernel.load_contiguous(input_dtype, input_shape);
            let new_kid = ekernels.push(EKernelData {
                kernel: new_kernel,
                outputs: Vec::new(),
                loads: vec![input_cid],
                stores: Vec::new(),
            });
            (new_kid, load_op)
        } else {
            (kid, op_id)
        };

        let loads = ekernels[kid].loads.clone();
        let kernel = ekernels[kid].kernel.clone();
        kid = ekernels.push(EKernelData {
            kernel,
            outputs: Vec::new(),
            loads,
            stores: Vec::new(),
        });

        (kid, op_id)
    }
}
