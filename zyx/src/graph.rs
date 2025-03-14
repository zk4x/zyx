//! Graph of tensor operations.

use crate::node::{BOp, Node};
use crate::tensor::TensorId;
use crate::Set;
use crate::{
    shape::{Axis, Dimension},
    slab::Slab,
    DType,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
pub struct Graph {
    // First value is reference count, second is node
    pub(super) nodes: Slab<(u32, Node)>,
    pub(super) gradient_tape_ref_count: u32,
    pub(super) gradient_tape: Option<Set<TensorId>>,
    // TODO instead of btreemap use data structure that uses single allocation for all shapes, just Vec<u32>
    shapes: BTreeMap<TensorId, Vec<Dimension>>,
    paddings: BTreeMap<TensorId, Vec<(isize, isize)>>,
    axes: BTreeMap<TensorId, Vec<Axis>>,
}

impl Graph {
    pub(super) const fn new() -> Self {
        Self {
            nodes: Slab::new(),
            gradient_tape_ref_count: 0,
            gradient_tape: None,
            shapes: BTreeMap::new(),
            paddings: BTreeMap::new(),
            axes: BTreeMap::new(),
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.nodes.len() == 0
    }

    pub(super) fn retain(&mut self, x: TensorId) {
        self.nodes[x].0 += 1;
    }

    /// Returns which tensors should be deallocated
    pub(super) fn release(&mut self, x: TensorId) -> Set<TensorId> {
        let mut params = Vec::with_capacity(10);
        params.push(x);
        let mut to_remove = Set::with_capacity_and_hasher(10, Default::default());
        while let Some(x) = params.pop() {
            //println!("Releasing {x}");
            let node = &mut self.nodes[x];
            node.0 -= 1;
            if node.0 == 0 {
                //println!("Dropping {x}");
                params.extend(node.1.parameters());
                to_remove.insert(x);
                self.nodes.remove(x);
                self.shapes.remove(&x);
                if let Some(tape) = self.gradient_tape.as_mut() {
                    tape.remove(&x);
                }
                self.axes.remove(&x);
                self.paddings.remove(&x);
            }
        }
        to_remove
    }

    pub(super) fn push(&mut self, node: Node) -> TensorId {
        //println!("Pushing {node:?}");
        #[cfg(debug_assertions)]
        {
            let mut shape = None;
            for nid in node.parameters() {
                if let Some(sh) = shape {
                    let shape = self.shape(nid);
                    if sh != shape {
                        println!("{:?}", self.shapes);
                        /*for (i, (rc, node)) in self.nodes.iter() {
                            println!("ID {i} x {rc} -> {node:?}");
                        }*/
                        panic!("{sh:?} != {shape:?} Pushing new node {node:?}");
                    }
                } else {
                    shape = Some(self.shape(nid));
                }
            }
        }

        for nid in node.parameters() {
            self.nodes[nid].0 += 1;
        }
        let nid = self.nodes.push((1, node));
        if let Some(tape) = self.gradient_tape.as_mut() {
            tape.insert(nid);
        }
        nid
    }

    pub(super) fn push_wshape(&mut self, node: Node, shape: Vec<Dimension>) -> TensorId {
        //println!("Pushing wshape {node:?}");
        let id = self.push(node);
        self.shapes.insert(id, shape);
        id
    }

    pub(super) fn push_padding(&mut self, id: TensorId, padding: Vec<(isize, isize)>) {
        self.paddings.insert(id, padding);
    }

    pub(super) fn push_axes(&mut self, id: TensorId, axes: Vec<Axis>) {
        self.axes.insert(id, axes);
    }

    pub(super) fn add_shape(&mut self, id: TensorId) {
        let shape = self.shape(id).into();
        self.shapes.insert(id, shape);
    }

    pub(super) fn delete_tensors(&mut self, tensors: &Set<TensorId>) {
        for &tensor in tensors {
            self.nodes.remove(tensor);
            self.shapes.remove(&tensor);
            self.paddings.remove(&tensor);
            self.axes.remove(&tensor);
        }
    }

    pub(super) fn dtype(&self, tensor_id: TensorId) -> DType {
        let mut tensor_id = tensor_id;
        for _ in 0..10000 {
            match self.nodes[tensor_id].1 {
                Node::Const { value } => return value.dtype(),
                Node::Leaf { dtype } | Node::Cast { dtype, .. } => return dtype,
                Node::Binary {
                    bop: BOp::Cmpgt | BOp::Cmplt | BOp::NotEq | BOp::And | BOp::Or,
                    ..
                } => return DType::Bool,
                _ => {
                    tensor_id = self.nodes[tensor_id].1.parameters().next().unwrap();
                }
            }
        }
        panic!("DType of {tensor_id} could not be found. This is internal bug.")
    }

    pub(super) fn padding(&self, tensor_id: TensorId) -> &[(isize, isize)] {
        &self.paddings[&tensor_id]
    }

    pub(super) fn axes(&self, tensor_id: TensorId) -> &[Axis] {
        &self.axes[&tensor_id]
    }

    pub(super) fn shape(&self, tensor_id: TensorId) -> &[Dimension] {
        let mut tensor_id = tensor_id;
        for _ in 0..10000 {
            if let Some(shape) = self.shapes.get(&tensor_id) {
                //println!("Found shape {shape:?} for tensor {tensor_id}");
                return shape;
            } else if let Node::Const { .. } = self.nodes[tensor_id].1 {
                return &[1];
            }
            //println!("Getting params of id: {tensor_id}, {:?}", self.nodes[tensor_id].1);
            tensor_id = self.nodes[tensor_id].1.param1();
        }
        panic!("Shape of {tensor_id} could not be found. This is internal bug.")
    }

    pub(super) fn build_topo(&self, x: TensorId, sources: &Set<TensorId>) -> Vec<TensorId> {
        let Some(tape) = self.gradient_tape.as_ref() else { return Vec::new() };
        //for (id, (rc, node)) in self.nodes.iter() { println!("{id} x {rc}  {node:?}"); }
        //println!("Gradient tape: {tape:?}");
        // Make a list of visited nodes and their reference counts.
        let mut params: Vec<TensorId> = vec![x];
        let mut rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                if !sources.contains(&nid)
                    && !matches!(self.nodes[nid].1, Node::Binary { bop: BOp::Cmplt, .. })
                    && tape.contains(&nid)
                {
                    params.extend(self.nodes[nid].1.parameters());
                }
                1
            });
        }
        // Order them using rcs reference counts
        let mut order = Vec::new();
        let mut internal_rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
        let mut params: Vec<TensorId> = vec![x];
        while let Some(nid) = params.pop() {
            if let Some(&rc) = rcs.get(&nid) {
                if rc == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                    order.push(nid);
                    params.extend(self.nodes[nid].1.parameters());
                }
            }
        }
        //println!("order {order:?}");
        // Build topo, this way it ensures that grad is not used in backprop
        // before it was insert_or_add by all parents.
        let mut topo = Vec::new();
        let mut req_grad = sources.clone();
        let mut visited = BTreeSet::new();
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
    pub fn plot_dot_graph(&self, ids: &Set<TensorId>, pools: &[crate::runtime::Pool]) -> String {
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
        let mut rcs: BTreeMap<TensorId, u8> = BTreeMap::new();
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                //println!("Access {nid:?}");
                params.extend(self.nodes[nid].1.parameters());
                1
            });
        }
        // Order them using rcs reference counts
        let mut order = Vec::new();
        let mut internal_rcs: BTreeMap<TensorId, u8> = BTreeMap::new();
        let mut params: Vec<TensorId> = ids.iter().copied().collect();
        while let Some(nid) = params.pop() {
            if rcs[&nid] == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                order.push(nid);
                if rcs.contains_key(&nid) {
                    params.extend(self.nodes[nid].1.parameters());
                }
            }
        }
        let mut topo: BTreeSet<TensorId> = ids.iter().copied().collect();
        for nid in order.into_iter().rev() {
            for p in self.nodes[nid].1.parameters() {
                if topo.contains(&p) {
                    topo.insert(nid);
                }
            }
        }
        // Puts graph of nodes into dot language for visualization
        let mut user_rc: BTreeMap<TensorId, u32> =
            self.nodes.iter().map(|(k, (rc, _))| (k, *rc)).collect();
        for (_, node) in self.nodes.values() {
            for param in node.parameters() {
                *user_rc.get_mut(&param).unwrap() -= 1;
            }
        }
        //std::println!("User {:?}", user_rc);
        let realized_nodes: Set<TensorId> =
            pools.iter().map(|pool| pool.buffer_map.keys()).flatten().copied().collect();
        let mut res_dot_graph =
            String::from("strict digraph {\n  ordering=in\n  rank=source\n  rankdir=LR\n");
        let mut add_node = |i: TensorId, text: &str, shape: &str| {
            let fillcolor = if user_rc[&i] > 0 { "coral" } else { "aqua" };
            /*if let Some(label) = labels.get(&NodeId::new(id)) {
                write!(res, "  {id}[label=\"{}NL{} x {}NL{}NL{}\", shape={}, fillcolor=\"{}\", style=filled]",
                    label, id, rc[id], text, get_shape(NodeId::new(id)), shape, fillcolor).unwrap();
            } else {*/
            let border_color;
            let border_width;
            if realized_nodes.contains(&i) {
                border_color = "darkred";
                border_width = 5;
            } else {
                border_color = "black";
                border_width = 1;
            }
            write!(
                res_dot_graph,
                "  {i}[label=\"{} x {}NL{}NL{:?}\", shape={}, fillcolor=\"{}\", style=filled, color=\"{border_color}\", penwidth={border_width}]",
                i,
                self.nodes[i].0,
                text,
                self.shape(i),
                shape,
                fillcolor
            )
            .unwrap();
            writeln!(res_dot_graph).unwrap();
        };
        let mut edges = String::new();
        for &id in &topo {
            let node = &self.nodes[id].1;
            match node {
                Node::Const { value } => add_node(id, &f!("Const({value:?})"), "box"),
                Node::Leaf { dtype } => {
                    add_node(id, &f!("Leaf({:?}, {})", self.shape(id), dtype), "box")
                }
                Node::Cast { x, dtype } => add_node(id, &f!("C-{dtype}({x})"), "oval"),
                Node::Unary { x, uop } => add_node(id, &f!("{uop:?}({x})"), "oval"),
                Node::Binary { x, y, bop } => add_node(id, &f!("{bop:?}({x}, {y})"), "oval"),
                Node::Reshape { x } => add_node(id, &f!("Reshape({x})"), "oval"),
                Node::Permute { x } => add_node(id, &f!("Permute({x})"), "oval"),
                Node::Expand { x } => add_node(id, &f!("Expand({x})"), "oval"),
                Node::Pad { x } => add_node(id, &f!("Pad({x})"), "oval"),
                Node::Reduce { x, rop } => add_node(id, &f!("{rop:?}({x})"), "oval"),
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
