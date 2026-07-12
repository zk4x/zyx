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
    DType, Map, Set, ZyxError,
    backend::ProgramId,
    dtype::Constant,
    kernel::{BOp, DeviceId, Kernel, MoveOp, Op, OpId, UOp},
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
        let output_set: Set<ClassId> = outputs.iter().copied().collect();
        let order = self.topo_sort_classes(outputs);

        // Reference counts: how many times each class appears as a child.
        let mut rcs: Map<ClassId, u32> = Map::default();
        for &cid in &order {
            for nid in &self.classes[cid].nodes {
                for child in self.nodes[*nid].node.class_params() {
                    *rcs.entry(child).or_default() += 1;
                }
            }
        }

        // All realized (Leaf) classes start in pending_stores.
        let mut pending_stores: Set<ClassId> = self
            .classes
            .ids()
            .filter(|&cid| {
                self.classes[cid]
                    .nodes
                    .iter()
                    .any(|&nid| matches!(&self.nodes[nid].node, Node::Leaf { .. }))
            })
            .collect();
        let mut visited: Map<ClassId, (EKernelId, OpId)> = Map::default();

        for &cid in &order {
            debug_assert!(!visited.contains_key(&cid), "class {cid:?} already visited");

            // If this class is in pending_stores (realized), create a load kernel.
            if pending_stores.contains(&cid) {
                let kid = self.new_load_kernel(cid, shapes);
                let rc = rcs.get(&cid).copied().unwrap_or(0) as usize;
                for _ in 0..rc {
                    self.ekernels[kid].outputs.push(cid);
                }
                visited.insert(cid, (kid, self.ekernels[kid].kernel.head));
                continue;
            }

            let nid = self.classes[cid].nodes[0];
            let node = &self.nodes[nid].node;

            match node {
                Node::Leaf { .. } => unreachable!(),
                Node::Const(c) => {
                    let mut kernel = Kernel::new(DeviceId::AUTO);
                    let result_op = kernel.push_back(Op::ConstView(Box::new((*c, View::contiguous(&[1])))));
                    let kid = self.ekernels.push(EKernelData {
                        kernel,
                        outputs: Vec::new(),
                        loads: Vec::new(),
                        stores: Vec::new(),
                    });
                    let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
                    for _ in 0..n_consumers {
                        self.ekernels[kid].outputs.push(cid);
                    }
                    visited.insert(cid, (kid, result_op));
                }
                Node::Unary { x, uop } => {
                    self.add_unary(cid, *x, *uop, &mut visited, &rcs, shapes);
                }
                Node::Cast { x, dtype } => {
                    self.add_cast(cid, *x, *dtype, &mut visited, &rcs, shapes);
                }
                Node::Binary { x, y, bop } => {
                    self.add_binary(cid, *x, *y, *bop, &mut visited, &rcs, &mut pending_stores, shapes);
                }
                Node::Reduce { x, bop, axes } => {
                    let axes: Vec<UAxis> = axes.to_vec();
                    self.add_reduce(cid, *x, *bop, axes, &mut visited, &rcs, &mut pending_stores, shapes);
                }
                Node::Expand { x, shape } => {
                    self.add_expand(cid, *x, *shape, &mut visited, &rcs, &mut pending_stores, shapes);
                }
                Node::Permute { x, axes } => {
                    let axes: Vec<UAxis> = axes.to_vec();
                    self.add_permute(cid, *x, axes, &mut visited, &rcs, &mut pending_stores, shapes);
                }
                Node::Reshape { x, shape } => {
                    let shape_vec: Vec<Dim> = shapes[*shape].clone();
                    self.add_reshape(cid, *x, shape_vec, &mut visited, &rcs, &mut pending_stores, shapes);
                }
                Node::PadZeros { x, padding } => {
                    let padding: Vec<(i64, i64)> = padding.to_vec();
                    self.add_pad(cid, *x, padding, &mut visited, &rcs, &mut pending_stores, shapes);
                }
                Node::ToDevice { x, .. } => {
                    let x = *x;
                    let (child_kid, child_op) = self.child_to_kid(x, &mut visited, shapes);
                    self.add_store(x, child_kid, child_op, &mut visited, &mut pending_stores);
                    let (kid, op_id) = self.child_to_kid(x, &mut visited, shapes);
                    visited.insert(cid, (kid, op_id));
                }
                Node::Kernel { .. } => {}
            }

            // Post-processing: store if output or final.
            if !visited.contains_key(&cid) {
                continue;
            }
            let remaining_rc = rcs.get(&cid).copied().unwrap_or(0);
            let is_output = remaining_rc == 0 || output_set.contains(&cid);

            if is_output {
                let (kid, op_id) = visited[&cid];
                self.add_store(cid, kid, op_id, &mut visited, &mut pending_stores);
                if output_set.contains(&cid) && remaining_rc > 0 {
                    let new_kid = self.new_load_kernel(cid, shapes);
                    let new_op = self.ekernels[new_kid].kernel.head;
                    visited.insert(cid, (new_kid, new_op));
                }
            }
        }

        // Force-seal remaining kernels that still have output classes.
        let kid_list: Vec<EKernelId> = self.ekernels.ids().collect();
        for &kid in &kid_list {
            if !self.ekernels.contains_key(kid) {
                continue;
            }
            let remaining: Vec<ClassId> = self.ekernels[kid]
                .outputs
                .iter()
                .copied()
                .filter(|&c| output_set.contains(&c))
                .collect();
            for &cid in &remaining {
                if let Some(&(_, op_id)) = visited.get(&cid) {
                    self.add_store(cid, kid, op_id, &mut visited, &mut pending_stores);
                }
            }
        }

        self.debug_print(shapes);
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

    // ── Kernelizer helpers ─────────────────────────────────

    fn new_load_kernel(&mut self, cid: ClassId, shapes: &Slab<ShapeId, Vec<Dim>>) -> EKernelId {
        let mut kernel = Kernel::new(DeviceId::AUTO);
        let shape: Vec<Dim> = shapes[self.classes[cid].shape].clone();
        let is_const = self.classes[cid]
            .nodes
            .iter()
            .any(|&nid| matches!(&self.nodes[nid].node, Node::Const(_)));
        if is_const {
            let value = self.classes[cid]
                .nodes
                .iter()
                .copied()
                .find_map(|nid| {
                    if let Node::Const(v) = &self.nodes[nid].node {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .unwrap();
            kernel.push_back(Op::ConstView(Box::new((value, View::contiguous(&[1])))));
        } else {
            kernel.load_contiguous(self.classes[cid].dtype, &shape);
        }
        let kid = self.ekernels.push(EKernelData {
            kernel,
            outputs: Vec::new(),
            loads: if is_const { Vec::new() } else { vec![cid] },
            stores: Vec::new(),
        });
        kid
    }

    fn child_to_kid(
        &mut self,
        child: ClassId,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) -> (EKernelId, OpId) {
        match visited.get(&child) {
            Some(&v) => v,
            None => {
                let kid = self.new_load_kernel(child, shapes);
                let op = self.ekernels[kid].kernel.head;
                visited.insert(child, (kid, op));
                (kid, op)
            }
        }
    }

    fn remove_first_output(kernels: &mut Slab<EKernelId, EKernelData>, kid: EKernelId, cid: ClassId) {
        if let Some(pos) = kernels[kid].outputs.iter().position(|&x| x == cid) {
            kernels[kid].outputs.remove(pos);
        }
    }

    fn add_store(
        &mut self,
        cid: ClassId,
        kid: EKernelId,
        op_id: OpId,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        pending_stores: &mut Set<ClassId>,
    ) {
        if pending_stores.contains(&cid) {
            visited.remove(&cid);
            Self::remove_first_output(&mut self.ekernels, kid, cid);
            return;
        }
        pending_stores.insert(cid);
        let dtype = self.classes[cid].dtype;
        self.ekernels[kid].kernel.store_contiguous(op_id, dtype);
        self.ekernels[kid].stores.push(cid);
        visited.remove(&cid);

        let outputs_empty = {
            let outputs = &mut self.ekernels[kid].outputs;
            outputs.retain(|&x| x != cid);
            outputs.is_empty()
        };

        if outputs_empty {
            visited.retain(|_, &mut (k, _)| k != kid);

            let ekdata = &self.ekernels[kid];
            let mut input_cids: Vec<ClassId> = ekdata.loads.clone();
            let output_cids: Vec<ClassId> = ekdata.outputs.clone();
            let output_set: Set<ClassId> = output_cids.iter().copied().collect();
            input_cids.retain(|c| !output_set.contains(c));

            let kind = Node::Kernel {
                inputs: input_cids.into_boxed_slice(),
                outputs: output_cids.clone().into_boxed_slice(),
                program_id: ProgramId::NULL,
            };
            let knid = self.nodes.push(NodeData { node: kind, class_of: cid });
            for &ocid in &output_cids {
                self.classes[ocid].nodes.push(knid);
            }
            if !output_cids.contains(&cid) {
                self.classes[cid].nodes.push(knid);
            }
            self.kernel_map.insert(knid, kid);
        }
    }

    fn merge_kernels(
        &mut self,
        src: EKernelId,
        dst: EKernelId,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
    ) {
        let EKernelData { kernel: src_kernel, outputs, loads, stores } =
            unsafe { self.ekernels.remove_and_return(src) };

        {
            let dst_data = &mut self.ekernels[dst];
            dst_data.outputs.extend(outputs);
            dst_data.loads.extend(loads);
            dst_data.stores.extend(stores);
        }

        let mut op_map: Map<OpId, OpId> = Map::default();
        let mut i = src_kernel.head;
        while !i.is_null() {
            let mut op = src_kernel.ops[i].op.clone();
            for param in op.parameters_mut() {
                if !param.is_null() {
                    if let Some(&new_param) = op_map.get(param) {
                        *param = new_param;
                    }
                }
            }
            let new_id = self.ekernels[dst].kernel.push_back(op);
            op_map.insert(i, new_id);
            i = src_kernel.ops[i].next;
        }

        for (_, (kid, op_id)) in visited.iter_mut() {
            if *kid == src {
                *kid = dst;
                if let Some(&new_op) = op_map.get(op_id) {
                    *op_id = new_op;
                }
            }
        }
    }

    fn duplicate_or_store_class(
        &mut self,
        child: ClassId,
        mut kid: EKernelId,
        mut op_id: OpId,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        pending_stores: &mut Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) -> (EKernelId, OpId) {
        // Phase 1: if kernel has stores, store child and create fresh load kernel
        let has_stores = self.ekernels[kid].kernel.contains_stores();
        if has_stores {
            self.add_store(child, kid, op_id, visited, pending_stores);
            let new_kid = self.new_load_kernel(child, shapes);
            let new_op = self.ekernels[new_kid].kernel.head;
            visited.insert(child, (new_kid, new_op));
            kid = new_kid;
            op_id = new_op;
        }

        // Phase 2: if kernel has multiple outputs, clone the kernel
        let n_outputs = self.ekernels[kid].outputs.len();
        if n_outputs > 1 {
            let preceded_by_reduce = self.ekernels[kid].kernel.is_preceded_by_reduce(op_id);
            if preceded_by_reduce {
                self.add_store(child, kid, op_id, visited, pending_stores);
                let new_kid = self.new_load_kernel(child, shapes);
                let new_op = self.ekernels[new_kid].kernel.head;
                visited.insert(child, (new_kid, new_op));
                kid = new_kid;
                op_id = new_op;
                let n_outputs2 = self.ekernels[kid].outputs.len();
                if n_outputs2 > 1 {
                    let loads = self.ekernels[kid].loads.clone();
                    let kernel = self.ekernels[kid].kernel.clone();
                    let new_kid = self.ekernels.push(EKernelData {
                        kernel,
                        outputs: Vec::new(),
                        loads,
                        stores: Vec::new(),
                    });
                    kid = new_kid;
                }
            } else {
                let loads = self.ekernels[kid].loads.clone();
                let kernel = self.ekernels[kid].kernel.clone();
                let new_kid = self.ekernels.push(EKernelData {
                    kernel,
                    outputs: Vec::new(),
                    loads,
                    stores: Vec::new(),
                });
                kid = new_kid;
            }
        }

        (kid, op_id)
    }

    fn add_unary(
        &mut self,
        cid: ClassId,
        child: ClassId,
        uop: UOp,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &Map<ClassId, u32>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let (kid, op_id) = self.child_to_kid(child, visited, shapes);
        Self::remove_first_output(&mut self.ekernels, kid, child);
        let kernel = &mut self.ekernels[kid].kernel;
        let result_op = kernel.unary(op_id, uop);
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_cast(
        &mut self,
        cid: ClassId,
        child: ClassId,
        dtype: DType,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &Map<ClassId, u32>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let (kid, op_id) = self.child_to_kid(child, visited, shapes);
        Self::remove_first_output(&mut self.ekernels, kid, child);
        let kernel = &mut self.ekernels[kid].kernel;
        let result_op = kernel.cast(op_id, dtype);
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_binary(
        &mut self,
        cid: ClassId,
        lhs: ClassId,
        rhs: ClassId,
        bop: BOp,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let (mut kid, mut op_id) = self.child_to_kid(lhs, visited, shapes);
        let (mut kidy, op_idy) = self.child_to_kid(rhs, visited, shapes);

        let kid_stores = self.ekernels[kid].kernel.contains_stores();
        let kidy_stores = self.ekernels[kidy].kernel.contains_stores();

        if kid == kidy {
            Self::remove_first_output(&mut self.ekernels, kid, lhs);
            Self::remove_first_output(&mut self.ekernels, kid, rhs);
            let result_op = self.ekernels[kid].kernel.binary(op_id, op_idy, bop);
            let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
            for _ in 0..n_consumers {
                self.ekernels[kid].outputs.push(cid);
            }
            visited.insert(cid, (kid, result_op));
        } else {
            match (kid_stores, kidy_stores) {
                (true, true) => {
                    self.add_store(lhs, kid, op_id, visited, pending_stores);
                    let (nk, no) = self.child_to_kid(lhs, visited, shapes);
                    (kid, op_id) = (nk, no);

                    self.add_store(rhs, kidy, op_idy, visited, pending_stores);
                    let (nk, _) = self.child_to_kid(rhs, visited, shapes);
                    (kidy, _) = (nk, nk);
                }
                (true, false) => {
                    self.add_store(lhs, kid, op_id, visited, pending_stores);
                    let (nk, no) = self.child_to_kid(lhs, visited, shapes);
                    (kid, op_id) = (nk, no);
                }
                (false, true) => {
                    self.add_store(rhs, kidy, op_idy, visited, pending_stores);
                    let (nk, _) = self.child_to_kid(rhs, visited, shapes);
                    (kidy, _) = (nk, nk);
                }
                (false, false) => {}
            }

            self.merge_kernels(kidy, kid, visited);
            let (_, op_idy) = visited[&rhs];
            Self::remove_first_output(&mut self.ekernels, kid, lhs);
            Self::remove_first_output(&mut self.ekernels, kid, rhs);
            let result_op = self.ekernels[kid].kernel.binary(op_id, op_idy, bop);
            let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
            for _ in 0..n_consumers {
                self.ekernels[kid].outputs.push(cid);
            }
            visited.insert(cid, (kid, result_op));
        }
    }

    fn add_reduce(
        &mut self,
        cid: ClassId,
        child: ClassId,
        rop: BOp,
        axes: Vec<UAxis>,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let n_axes = axes.len() as UAxis;
        let (mut kid, mut op_id) = self.child_to_kid(child, visited, shapes);
        (kid, op_id) = self.duplicate_or_store_class(child, kid, op_id, visited, pending_stores, shapes);
        Self::remove_first_output(&mut self.ekernels, kid, child);

        let in_shape: Vec<Dim> = shapes[self.classes[child].shape].clone();
        let kernel = &mut self.ekernels[kid].kernel;
        let permuted = {
            let n = in_shape.len();
            let max_axis = *axes.last().unwrap() as usize;
            let mut permute_axes = Vec::with_capacity(n);
            let mut ai = 0;
            for i in 0..=max_axis {
                if axes[ai] as usize == i {
                    ai += 1;
                } else {
                    permute_axes.push(i as UAxis);
                }
            }
            permute_axes.extend((max_axis + 1..n).map(|i| i as UAxis));
            permute_axes.extend_from_slice(&axes);
            if !permute_axes.iter().copied().eq(0..permute_axes.len() as UAxis) {
                let shape = crate::shape::permute(&in_shape, &permute_axes);
                kernel.push_back(Op::Move {
                    x: op_id,
                    mop: Box::new(MoveOp::Permute { axes: permute_axes, shape }),
                })
            } else {
                op_id
            }
        };
        let mut result_op = kernel.push_back(Op::Reduce {
            x: permuted,
            rop,
            n_axes: n_axes as UAxis,
        });
        if in_shape.len() == n_axes as usize {
            result_op = kernel.reshape(result_op, &[1]);
        }

        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_expand(
        &mut self,
        cid: ClassId,
        child: ClassId,
        _shape: ShapeId,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let (mut kid, mut op_id) = self.child_to_kid(child, visited, shapes);
        (kid, op_id) = self.duplicate_or_store_class(child, kid, op_id, visited, pending_stores, shapes);
        Self::remove_first_output(&mut self.ekernels, kid, child);
        let shape: Vec<Dim> = shapes[self.classes[cid].shape].clone();
        let kernel = &mut self.ekernels[kid].kernel;
        let result_op = kernel.push_back(Op::Move {
            x: op_id,
            mop: Box::new(MoveOp::Expand { shape }),
        });
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_permute(
        &mut self,
        cid: ClassId,
        child: ClassId,
        axes: Vec<UAxis>,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let (mut kid, mut op_id) = self.child_to_kid(child, visited, shapes);
        (kid, op_id) = self.duplicate_or_store_class(child, kid, op_id, visited, pending_stores, shapes);
        Self::remove_first_output(&mut self.ekernels, kid, child);
        let shape: Vec<Dim> = shapes[self.classes[cid].shape].clone();
        let kernel = &mut self.ekernels[kid].kernel;
        let result_op = kernel.push_back(Op::Move {
            x: op_id,
            mop: Box::new(MoveOp::Permute { axes, shape }),
        });
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_reshape(
        &mut self,
        cid: ClassId,
        child: ClassId,
        shape: Vec<Dim>,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let (mut kid, mut op_id) = self.child_to_kid(child, visited, shapes);
        (kid, op_id) = self.duplicate_or_store_class(child, kid, op_id, visited, pending_stores, shapes);
        Self::remove_first_output(&mut self.ekernels, kid, child);
        let kernel = &mut self.ekernels[kid].kernel;
        let result_op = kernel.push_back(Op::Move {
            x: op_id,
            mop: Box::new(MoveOp::Reshape { shape }),
        });
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_pad(
        &mut self,
        cid: ClassId,
        child: ClassId,
        padding: Vec<(i64, i64)>,
        visited: &mut Map<ClassId, (EKernelId, OpId)>,
        rcs: &Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
        shapes: &Slab<ShapeId, Vec<Dim>>,
    ) {
        let child_shape: Vec<Dim> = shapes[self.classes[child].shape].clone();
        let child_n: Dim = child_shape.iter().product();
        let cid_shape: Vec<Dim> = shapes[self.classes[cid].shape].clone();
        let pad_n: Dim = cid_shape.iter().product();
        let expands = pad_n > child_n;

        let (mut kid, mut op_id);
        if expands {
            if let Some(&(ckid, cop_id)) = visited.get(&child) {
                self.add_store(child, ckid, cop_id, visited, pending_stores);
            }
            (kid, op_id) = self.child_to_kid(child, visited, shapes);
        } else {
            (kid, op_id) = self.child_to_kid(child, visited, shapes);
            (kid, op_id) = self.duplicate_or_store_class(child, kid, op_id, visited, pending_stores, shapes);
        }
        Self::remove_first_output(&mut self.ekernels, kid, child);
        let shape: Vec<Dim> = cid_shape;
        let kernel = &mut self.ekernels[kid].kernel;
        let result_op = kernel.push_back(Op::Move {
            x: op_id,
            mop: Box::new(MoveOp::Pad { padding, shape }),
        });
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            self.ekernels[kid].outputs.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }
}

impl crate::runtime::Runtime {
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
            for &dev_id in &device_ids {
                let pool_id = self.devices[dev_id].memory_pool_id();
                let (dev_prog, _timing) = self.get_or_autotune(
                    kernel.clone(), dev_id, pool_id, flop, read, write,
                )?;
                let prog = ProgramId { device: dev_id, program: dev_prog };
                if let Some(ref mut graph) = self.graph {
                    if let Node::Kernel { program_id, .. } = &mut graph.nodes[*nid].node {
                        *program_id = prog;
                    }
                }
            }
        }

        Ok(())
    }
}
