// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Graph of tensor operations.

pub mod compiled;
pub mod kernelizer;
mod search;

use crate::kernel::{BOp, DeviceId, UOp};
use crate::slab::SlabId;
use crate::tensor::TensorId;
use crate::{
    DType,
    shape::{Dim, UAxis},
    slab::Slab,
};
use crate::{Map, Set};
use std::hash::BuildHasherDefault;

/// Graph node, each node is one operation. Nodes
/// represent the opset that is available on tensors.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum Node {
    // Constant tensor baked into kernels
    Const {
        value: Constant,
    },
    // Tensor stored on device
    Leaf {
        dtype: DType,
    },
    Expand {
        x: TensorId,
    },
    Permute {
        x: TensorId,
    },
    // Reshape can be sometimes axis split or axis join
    Reshape {
        x: TensorId,
    },
    Pad {
        x: TensorId,
    },
    Reduce {
        x: TensorId,
        rop: BOp,
    },
    Cast {
        x: TensorId,
        dtype: DType,
    },
    Unary {
        x: TensorId,
        uop: UOp,
    },
    Binary {
        x: TensorId,
        y: TensorId,
        bop: BOp,
    },
    #[allow(unused)]
    Custom(Box<crate::kernel::CustomKernel>),
    /// Explicit device placement — copies the tensor to the target device.
    ToDevice {
        x: TensorId,
        device: DeviceId,
    },
}

#[derive(Debug)]
pub struct Graph {
    // First value is reference count, second is node
    pub nodes: Slab<TensorId, (u32, Node)>,
    pub tape_rc: u32,
    pub tape: Option<Set<TensorId>>,
    pub tape_order: Vec<TensorId>,
    pub tape_inputs: Vec<TensorId>,
    pub shapes: Map<TensorId, Box<[Dim]>>,
    paddings: Map<TensorId, Box<[(i64, i64)]>>,
    axes: Map<TensorId, Box<[UAxis]>>,
}

impl Graph {
    pub(super) const fn new() -> Self {
        Self {
            nodes: Slab::new(),
            tape_rc: 0,
            tape: None,
            tape_order: Vec::new(),
            tape_inputs: Vec::new(),
            shapes: Map::with_hasher(BuildHasherDefault::new()),
            paddings: Map::with_hasher(BuildHasherDefault::new()),
            axes: Map::with_hasher(BuildHasherDefault::new()),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.nodes.len() == TensorId::ZERO
    }

    pub(super) fn retain(&mut self, x: TensorId) {
        self.nodes[x].0 += 1;
    }

    /// Returns which tensors should be deallocated
    pub(super) fn release(&mut self, x: &[TensorId]) -> Set<TensorId> {
        let mut params = Vec::with_capacity(10);
        params.extend(x);
        let mut to_remove = Set::with_capacity_and_hasher(10, BuildHasherDefault::default());
        while let Some(x) = params.pop() {
            //println!("Releasing {x}");
            if let Some((rc, node, ..)) = self.nodes.get_mut(x) {
                let a = rc.saturating_sub(1);
                *rc = a;
                if a == 0 {
                    //println!("Dropping {x}");
                    params.extend(node.parameters());
                    to_remove.insert(x);
                    self.nodes.remove(x);
                    _ = self.shapes.remove(&x);
                    _ = self.axes.remove(&x);
                    _ = self.paddings.remove(&x);

                    if let Some(tape) = self.tape.as_mut() {
                        _ = tape.remove(&x);
                    }
                }
            }
        }
        to_remove
    }

    pub(super) fn push(&mut self, node: Node) -> TensorId {
        self.push_inner(node, None)
    }

    pub(super) fn push_wshape(&mut self, node: Node, shape: Vec<Dim>) -> TensorId {
        self.push_inner(node, Some(shape))
    }

    fn push_inner(&mut self, node: Node, shape: Option<Vec<Dim>>) -> TensorId {
        #[cfg(debug_assertions)]
        if !matches!(node, Node::Custom(_)) {
            let mut sh = None;
            for nid in node.parameters() {
                if let Some(sh) = sh {
                    let shape = self.shape(nid);
                    if sh != shape {
                        println!("{:?}", self.shapes);
                        panic!("{sh:?} != {shape:?} Pushing new node {node:?}");
                    }
                } else {
                    sh = Some(self.shape(nid));
                }
            }
        }

        let params = node.parameters();
        for nid in &params {
            self.nodes[*nid].0 += 1;
        }
        let nid = self.nodes.push((1, node));

        let shape = shape.unwrap_or_else(|| match &self.nodes[nid].1 {
            Node::Const { .. } => vec![1],
            Node::Cast { x, .. } | Node::Unary { x, .. } => self.shape(*x).to_vec(),
            Node::Binary { x, .. } => self.shape(*x).to_vec(),
            _ => unreachable!("shape must be provided for shape-changing ops"),
        });
        self.shapes.insert(nid, shape.into_boxed_slice());

        if let Some(tape) = self.tape.as_mut() {
            tape.insert(nid);
            self.tape_order.push(nid);
            for p in params {
                if !tape.contains(&p) {
                    self.tape_inputs.push(p);
                }
            }
        }
        nid
    }

    pub(super) fn push_padding(&mut self, id: TensorId, padding: Vec<(i64, i64)>) {
        self.paddings.insert(id, padding.into_boxed_slice());
    }

    pub(super) fn push_axes(&mut self, id: TensorId, axes: Vec<UAxis>) {
        self.axes.insert(id, axes.into_boxed_slice());
    }

    pub(super) fn add_shape(&mut self, id: TensorId) {
        let shape = self.shape(id).into();
        self.shapes.insert(id, shape);
    }

    /*pub(super) fn delete_tensors_without_deallocation(&mut self, tensors: &Set<TensorId>) {
        /*for &tensor in tensors {
            self.nodes.remove(tensor);
            self.shapes.remove(&tensor);
            self.paddings.remove(&tensor);
            self.axes.remove(&tensor);
        }*/
        let mut params: Vec<TensorId> = tensors.iter().copied().collect();
        while let Some(x) = params.pop() {
            //println!("Releasing {x}");
            if let Some((rc, node, ..)) = self.nodes.get_mut(x) {
                let a = rc.saturating_sub(1);
                *rc = a;
                if a == 0 {
                    //println!("Dropping {x}");
                    params.extend(node.parameters());
                    self.nodes.remove(x);
                    _ = self.shapes.remove(&x);
                    _ = self.axes.remove(&x);
                    _ = self.paddings.remove(&x);

                    if let Some(tape) = self.gradient_tape.as_mut() {
                        _ = tape.remove(&x);
                    }
                }
            }
        }
    }*/

    #[allow(unused)]
    pub(super) fn device(&self, tensor_id: TensorId) -> DeviceId {
        let mut tensor_id = tensor_id;
        for _ in 0..100_000 {
            if let Node::ToDevice { device, .. } = self.nodes[tensor_id].1 {
                return device;
            }
            let params = self.nodes[tensor_id].1.parameters();
            if params.is_empty() {
                return DeviceId::AUTO;
            }
            tensor_id = params.into_iter().next().unwrap();
        }
        DeviceId::AUTO
    }

    pub(super) fn dtype(&self, tensor_id: TensorId) -> DType {
        let mut tensor_id = tensor_id;
        for _ in 0..100_000 {
            match &self.nodes[tensor_id].1 {
                Node::Const { value } => return value.dtype(),
                Node::Leaf { dtype } | Node::Cast { dtype, .. } => return *dtype,
                Node::Custom(ck) => return ck.dtype,
                Node::Binary { bop, .. } if bop.returns_bool() => {
                    return DType::Bool;
                }
                _ => {
                    tensor_id = self.nodes[tensor_id].1.parameters().into_iter().next().unwrap();
                }
            }
        }
        panic!("DType of {tensor_id:?} could not be found. This is internal bug.")
    }

    pub(crate) fn padding(&self, tensor_id: TensorId) -> &[(i64, i64)] {
        &self.paddings[&tensor_id]
    }

    pub(super) fn axes(&self, tensor_id: TensorId) -> &[UAxis] {
        &self.axes[&tensor_id]
    }

    pub(super) fn shape(&self, tensor_id: TensorId) -> &[Dim] {
        let mut tensor_id = tensor_id;
        for _ in 0..1_000_000 {
            if let Some(shape) = self.shapes.get(&tensor_id) {
                //println!("Found shape {shape:?} for tensor {tensor_id}");
                return shape;
            } else if let Node::Const { .. } = self.nodes[tensor_id].1 {
                return &[1];
            }
            //println!("Getting params of id: {tensor_id}, {:?}", self.nodes[tensor_id].1);
            tensor_id = self.nodes[tensor_id].1.param1();
        }
        panic!("Shape of {tensor_id:?} could not be found. This is internal bug.")
    }

    pub(super) fn build_topo(&self, x: TensorId, sources: &Set<TensorId>) -> Vec<TensorId> {
        //self.debug();
        let Some(tape) = self.tape.as_ref() else {
            return Vec::new();
        };
        //for (id, (rc, node)) in self.nodes.iter() { println!("{id} x {rc}  {node:?}"); }
        //println!("Gradient tape: {tape:?}");
        // Make a list of visited nodes and their reference counts.
        let mut params: Vec<TensorId> = vec![x];
        let mut rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                if matches!(
                    self.nodes[nid].1,
                    Node::Binary {
                        bop: BOp::Cmpgt
                            | BOp::Cmplt
                            | BOp::Eq
                            | BOp::NotEq
                            | BOp::Or
                            | BOp::And
                            | BOp::BitAnd
                            | BOp::BitOr
                            | BOp::BitXor
                            | BOp::BitShiftLeft
                            | BOp::BitShiftRight,
                        ..
                    }
                ) {
                    // Non-differentiable ops: don't trace further
                    return 1;
                }
                if tape.contains(&nid) {
                    params.extend(self.nodes[nid].1.parameters());
                }
                1
            });
        }
        //println!("rcs={rcs:?}");
        // Order them using rcs reference counts
        let mut order = Vec::new();
        let mut internal_rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
        let mut params: Vec<TensorId> = vec![x];
        while let Some(nid) = params.pop() {
            if let Some(&rc) = rcs.get(&nid) {
                if rc == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                    order.push(nid);
                    params.extend(self.nodes[nid].1.parameters());
                }
            }
        }
        //println!("order={order:?}");
        // Build topo, this way it ensures that grad is not used in backprop
        // before it was insert_or_add by all parents.
        let mut topo = Vec::new();
        let mut req_grad = sources.clone();
        let mut visited = Set::with_capacity_and_hasher(100, BuildHasherDefault::new());
        for nid in order.into_iter().rev() {
            for p in self.nodes[nid].1.parameters() {
                if req_grad.contains(&p) && visited.insert(nid) {
                    req_grad.insert(nid);
                    topo.push(nid);
                }
            }
        }
        topo.reverse();
        //println!("topo {topo:?}");
        topo
    }

    /// Plot dot graph in dot format between given nodes
    #[must_use]
    pub fn plot_dot_graph(
        &self,
        ids: &Set<TensorId>,
        buffer_map: &crate::Map<crate::tensor::TensorId, crate::backend::BufferId>,
    ) -> String {
        use core::fmt::Write;
        use std::format as f;
        let ids: Set<TensorId> = if ids.is_empty() {
            self.nodes.ids().collect()
        } else {
            ids.clone()
        };
        //println!("{ids:?}");
        // Make a list of visited nodes and their reference counts.
        let mut params: Vec<TensorId> = ids.iter().copied().collect();
        let mut rcs: Map<TensorId, u8> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                //println!("Access {nid:?}");
                params.extend(self.nodes[nid].1.parameters());
                1
            });
        }
        // Order them using rcs reference counts
        let mut order = Vec::new();
        let mut internal_rcs: Map<TensorId, u8> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
        let mut params: Vec<TensorId> = ids.iter().copied().collect();
        while let Some(nid) = params.pop() {
            if rcs[&nid] == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                order.push(nid);
                if rcs.contains_key(&nid) {
                    params.extend(self.nodes[nid].1.parameters());
                }
            }
        }
        let mut topo: Set<TensorId> = ids.iter().copied().collect();
        for nid in order.into_iter().rev() {
            for p in self.nodes[nid].1.parameters() {
                if topo.contains(&p) {
                    topo.insert(nid);
                }
            }
        }
        // Puts graph of nodes into dot language for visualization
        let mut user_rc: Map<TensorId, u32> = self.nodes.iter().map(|(k, (rc, ..))| (k, *rc)).collect();
        for (_, node, ..) in self.nodes.values() {
            for param in node.parameters() {
                *user_rc.get_mut(&param).unwrap() -= 1;
            }
        }
        //std::println!("User {:?}", user_rc);
        let realized_nodes: Set<TensorId> = buffer_map.keys().copied().collect();
        let mut res_dot_graph = String::from("strict digraph {\n  ordering=in\n  rank=source\n  rankdir=LR\n");
        let mut add_node = |i: TensorId, text: &str, shape: &str| {
            let fillcolor = if user_rc[&i] > 0 { "coral" } else { "aqua" };
            /*if let Some(label) = labels.get(&NodeId::new(id)) {
                write!(res, "  {id}[label=\"{}NL{} x {}NL{}NL{}\", shape={}, fillcolor=\"{}\", style=filled]",
                    label, id, rc[id], text, get_shape(NodeId::new(id)), shape, fillcolor).unwrap();
            } else {*/
            let (border_color, border_width) = if realized_nodes.contains(&i) {
                ("darkred", 5)
            } else {
                ("black", 1)
            };
            write!(res_dot_graph, "  {i}[label=\"{} x {}NL{}NL{:?}\", shape={}, fillcolor=\"{}\", style=filled, color=\"{border_color}\", penwidth={border_width}]", i, self.nodes[i].0, text, self.shape(i), shape, fillcolor).unwrap();
            writeln!(res_dot_graph).unwrap();
        };
        let mut edges = String::new();
        for &id in &topo {
            let node = &self.nodes[id].1;
            match node {
                Node::Const { value } => add_node(id, &f!("Const({value:?})"), "box"),
                Node::Leaf { dtype } => {
                    add_node(id, &f!("Leaf({:?}, {})", self.shape(id), dtype), "box");
                }
                Node::Cast { x, dtype } => add_node(id, &f!("C-{dtype}({x})"), "oval"),
                Node::Unary { x, uop } => add_node(id, &f!("{uop:?}({x})"), "oval"),
                Node::Binary { x, y, bop } => add_node(id, &f!("{bop:?}({x}, {y})"), "oval"),
                Node::Reshape { x } => add_node(id, &f!("Reshape({x})"), "oval"),
                Node::Permute { x } => add_node(id, &f!("Permute({x})"), "oval"),
                Node::Expand { x } => add_node(id, &f!("Expand({x})"), "oval"),
                Node::Pad { x } => add_node(id, &f!("Pad({x})"), "oval"),
                Node::Reduce { x, rop } => add_node(id, &f!("{rop:?}({x})"), "oval"),
                Node::Custom(ck) => add_node(id, &f!("Custom(prog={:?}, params={})", ck.program, ck.inputs.len()), "box"),
                Node::ToDevice { x, device } => add_node(id, &f!("ToDevice({x}, {device:?})"), "box"),
            }
            for param in node.parameters() {
                writeln!(edges, "  {param} -> {id}").unwrap();
            }
        }
        res_dot_graph = res_dot_graph.replace("NL", "\n");
        write!(res_dot_graph, "{edges}}}").unwrap();
        res_dot_graph
    }
}

impl std::ops::Index<TensorId> for Graph {
    type Output = Node;
    fn index(&self, index: TensorId) -> &Self::Output {
        &self.nodes[index].1
    }
}

impl std::ops::IndexMut<TensorId> for Graph {
    fn index_mut(&mut self, index: TensorId) -> &mut Self::Output {
        &mut self.nodes[index].1
    }
}

use crate::dtype::Constant;

impl BOp {
    /// Returns true if the binary operation is associative:
    /// `(a op b) op c == a op (b op c)`.
    pub const fn is_associative(self) -> bool {
        use BOp::{Add, And, BitAnd, BitOr, BitShiftLeft, BitShiftRight, BitXor, Max, Mul, Or};
        matches!(
            self,
            Add | Mul | And | Or | BitXor | BitAnd | BitOr | BitShiftLeft | BitShiftRight | Max
        )
    }

    /// Returns true if the binary operation is commutative:
    /// `a op b == b op a`.
    pub const fn is_commutative(self) -> bool {
        use BOp::{Add, And, BitAnd, BitOr, BitXor, Max, Mul, Or};
        matches!(self, Add | Mul | And | Or | BitXor | BitAnd | BitOr | Max)
    }

    /// Returns true if the operation produces a boolean result.
    pub const fn returns_bool(self) -> bool {
        use BOp::{And, Cmpgt, Cmplt, Eq, NotEq, Or};
        matches!(self, Cmpgt | Cmplt | NotEq | Eq | And | Or)
    }
}

impl Node {
    /// Get all parameters of self.
    pub fn parameters(&self) -> Vec<TensorId> {
        match self {
            Node::Const { .. } | Node::Leaf { .. } => Vec::new(),
            Node::Unary { x, .. }
            | Node::Cast { x, .. }
            | Node::Reshape { x, .. }
            | Node::Expand { x, .. }
            | Node::Permute { x, .. }
            | Node::Pad { x, .. }
            | Node::Reduce { x, .. } => vec![*x],
            Node::Binary { x, y, .. } => vec![*x, *y],
            Node::Custom(ck) => ck.inputs.clone(),
            Node::ToDevice { x, .. } => vec![*x],
        }
    }

    pub fn param1(&self) -> TensorId {
        match self {
            Node::Const { .. } | Node::Leaf { .. } => unreachable!(),
            Node::Expand { x }
            | Node::Permute { x }
            | Node::Reshape { x }
            | Node::Pad { x }
            | Node::Reduce { x, .. }
            | Node::Cast { x, .. }
            | Node::Unary { x, .. }
            | Node::Binary { x, .. } => *x,
            Node::Custom(ck) => ck.inputs[0],
            Node::ToDevice { x, .. } => *x,
        }
    }

    pub(super) fn kind_tag(&self) -> u64 {
        match self {
            Node::Const { .. } => 0,
            Node::Leaf { .. } => 1,
            Node::Expand { .. } => 2,
            Node::Permute { .. } => 3,
            Node::Reshape { .. } => 4,
            Node::Pad { .. } => 5,
            Node::Reduce { .. } => 6,
            Node::Cast { .. } => 7,
            Node::Unary { .. } => 8,
            Node::Binary { .. } => 9,
            Node::Custom(_) => 10,
            Node::ToDevice { .. } => 11,
        }
    }

    pub(super) fn extra_hash(&self) -> u64 {
        match self {
            Node::Const { value } => {
                use std::hash::{Hash, Hasher};
                let mut h = crate::hashers::AHasher::default();
                value.hash(&mut h);
                h.finish()
            }
            Node::Reduce { rop, .. } => *rop as u64,
            Node::Unary { uop, .. } => *uop as u64,
            Node::Binary { bop, .. } => *bop as u64,
            Node::ToDevice { device, .. } => device.0 as u64,
            _ => 0,
        }
    }
}
