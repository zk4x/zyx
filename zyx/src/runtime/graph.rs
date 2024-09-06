use super::{node::{BOp, Node}, TensorId};
use crate::{index_map::IndexMap, shape::Dimension, DType};
use std::collections::{BTreeMap, BTreeSet};

// TODO implement PartialOrd such that tensor id does not matter
// This is probably not very high priority. It probably works fine
// even this way.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub(super) struct Graph {
    // Which nodes need to be evaluated
    to_eval: BTreeSet<TensorId>,
    // First value is reference count, second is node
    nodes: IndexMap<(u32, Node)>,
    shapes: BTreeMap<TensorId, Vec<Dimension>>,
    dtypes: BTreeMap<TensorId, DType>,
}

impl Graph {
    pub(super) const fn new() -> Self {
        Self {
            to_eval: BTreeSet::new(),
            nodes: IndexMap::new(),
            shapes: BTreeMap::new(),
            dtypes: BTreeMap::new(),
        }
    }

    pub(super) fn retain(&mut self, x: TensorId) {
        self.nodes[x].0 += 1;
    }

    /// Returns which tensors should be deallocated
    pub(super) fn release(&mut self, x: TensorId) -> BTreeSet<TensorId> {
        let mut params = Vec::with_capacity(10);
        params.push(x);
        let mut to_remove = BTreeSet::new();
        while let Some(x) = params.pop() {
            let node = &mut self.nodes[x];
            node.0 -= 1;
            if node.0 == 0 {
                params.extend(node.1.parameters());
                to_remove.insert(x);
                self.nodes.remove(x);
                self.shapes.remove(&x);
                self.dtypes.remove(&x);
            }
        }
        to_remove
    }

    pub(super) fn push(&mut self, node: Node) -> TensorId {
        //libc_print::libc_println!("Pushing {node:?}");
        for nid in node.parameters() {
            self.nodes[nid].0 += 1;
        }
        self.nodes.push((1, node))
    }

    pub(super) fn push_wshape(&mut self, node: Node, shape: Vec<Dimension>) -> TensorId {
        //libc_print::libc_println!("Pushing {node:?}");
        for nid in node.parameters() {
            self.nodes[nid].0 += 1;
        }
        let id = self.nodes.push((1, node));
        self.shapes.insert(id, shape);
        id
    }

    pub(super) fn push_wdtype(&mut self, node: Node, dtype: DType) -> TensorId {
        //libc_print::libc_println!("Pushing {node:?}");
        for nid in node.parameters() {
            self.nodes[nid].0 += 1;
        }
        let id = self.nodes.push((1, node));
        self.dtypes.insert(id, dtype);
        id
    }

    pub(super) fn push_wshape_and_dtype(
        &mut self,
        node: Node,
        shape: Vec<Dimension>,
        dtype: DType,
    ) -> TensorId {
        //libc_print::libc_println!("Pushing {node:?}");
        for nid in node.parameters() {
            self.nodes[nid].0 += 1;
        }
        let id = self.nodes.push((1, node));
        self.shapes.insert(id, shape);
        self.dtypes.insert(id, dtype);
        id
    }

    pub(super) fn add_shape_dtype(&mut self, id: TensorId) {
        let shape = self.shape(id).into();
        self.shapes.insert(id, shape);
        let dtype = self.dtype(id);
        self.dtypes.insert(id, dtype);
    }

    pub(super) fn dtype(&self, tensor_id: TensorId) -> DType {
        let mut tensor_id = tensor_id;
        let mut i = 0;
        while i < 1000000 {
            if let Some(&dtype) = self.dtypes.get(&tensor_id) {
                return dtype;
            } else if let Node::Const { value } = self.nodes[tensor_id].1 {
                return value.dtype();
            } else {
                tensor_id = self.nodes[tensor_id].1.parameters().next().unwrap();
            }
            i += 1;
        }
        panic!("DType of {tensor_id} could not be found. This is internal bug.")
    }

    pub(super) fn shape(&self, tensor_id: TensorId) -> &[usize] {
        let mut tensor_id = tensor_id;
        let mut i = 0;
        while i < 10000 {
            if let Some(shape) = self.shapes.get(&tensor_id) {
                return shape;
            } else if let Node::Const { .. } = self.nodes[tensor_id].1 {
                return &[1];
            } else {
                //println!("Getting params of id: {tensor_id}, {:?}", self.nodes[tensor_id].1);
                tensor_id = self.nodes[tensor_id].1.parameters().next().unwrap();
            }
            i += 1;
        }
        panic!("Shape of {tensor_id} could not be found. This is internal bug.")
    }

    pub(super) fn rc(&self, x: TensorId) -> u32 {
        self.nodes[x].0
    }

    pub(super) fn delete_tensors(&mut self, tensors: &BTreeSet<TensorId>) {
        for &tensor in tensors {
            self.nodes.remove(tensor);
        }
    }

    pub(super) fn realize_graph(
        &self,
        tensors: BTreeSet<TensorId>,
        is_realized: impl Fn(TensorId) -> bool,
    ) -> (Graph, BTreeSet<TensorId>, Vec<TensorId>) {
        // First topo search for minimum number of required nodes and create graph from it
        // Then replace all realized nodes with Node::Leaf
        // topo search
        let mut params: Vec<TensorId> = tensors.iter().copied().collect();
        let mut visited = BTreeSet::new();
        let mut leafs = BTreeSet::new();
        while let Some(param) = params.pop() {
            if visited.insert(param) {
                if is_realized(param) {
                    leafs.insert(param);
                } else {
                    params.extend(self.nodes[param].1.parameters());
                }
            }
        }
        // Get refcounts of all nodes
        let mut params: Vec<TensorId> = tensors.iter().copied().collect();
        let mut rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                params.extend(self.nodes[nid].1.parameters());
                1
            });
        }
        // Order them using rcs reference counts
        let mut order = Vec::new();
        let mut internal_rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
        let mut params: Vec<TensorId> = tensors.iter().copied().collect();
        while let Some(nid) = params.pop() {
            if let Some(&rc) = rcs.get(&nid) {
                if rc
                    == *internal_rcs
                        .entry(nid)
                        .and_modify(|rc| *rc += 1)
                        .or_insert(1)
                {
                    order.push(nid);
                    params.extend(self.nodes[nid].1.parameters());
                }
            }
        }
        order.reverse();
        // Nodes required outside of realized graph: self.rcs - rcs > 0
        let outside_nodes = self
            .nodes
            .iter()
            .filter_map(|(id, (rc, _))| {
                if let Some(&rc2) = rcs.get(&id) {
                    if *rc > rc2 {
                        Some(id)
                    } else {
                        None
                    }
                } else {
                    Some(id)
                }
            })
            .chain(tensors.iter().copied())
            .collect();

        let mut shapes: BTreeMap<TensorId, Vec<Dimension>> = self
            .shapes
            .iter()
            .filter_map(|(id, sh)| {
                if visited.contains(id) {
                    Some((*id, sh.clone()))
                } else {
                    None
                }
            })
            .collect();
        let mut dtypes: BTreeMap<TensorId, DType> = self
            .dtypes
            .iter()
            .filter_map(|(id, dt)| {
                if visited.contains(id) {
                    Some((*id, *dt))
                } else {
                    None
                }
            })
            .collect();
        for &leaf in &leafs {
            shapes
                .entry(leaf)
                .or_insert_with(|| self.shape(leaf).into());
            dtypes.entry(leaf).or_insert_with(|| self.dtype(leaf));
        }

        // replace realized nodes with leafs
        return (
            Graph {
                shapes,
                to_eval: tensors,
                dtypes,
                nodes: visited
                    .into_iter()
                    .map(|id| {
                        (
                            id,
                            if leafs.contains(&id) {
                                (self.nodes[id].0, Node::Leaf)
                            } else {
                                self.nodes[id].clone()
                            },
                        )
                    })
                    .collect(),
            },
            outside_nodes,
            order,
        );
    }

    // Calculates execution order, recalculates rcs, flop, bytes read and written and optimizes graph:
    // 1. moves all unary ops before movement ops
    // 2. removes unnecessary ops (like exp followed by ln), adding 0, multiply by 0, divide by 1, etc.
    // This function should be pretty fast, because it's also used by the interpreter, which does not do any caching
    pub(super) fn execution_order(
        &mut self,
        to_eval: &BTreeSet<TensorId>,
    ) -> (Vec<TensorId>, u128, u128, u128) {
        let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
        let mut rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                params.extend(self.nodes[nid].1.parameters());
                1
            });
        }
        // Order them using rcs reference counts
        let mut order = Vec::new();
        let mut internal_rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
        let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
        while let Some(nid) = params.pop() {
            if let Some(&rc) = rcs.get(&nid) {
                if rc
                    == *internal_rcs
                        .entry(nid)
                        .and_modify(|rc| *rc += 1)
                        .or_insert(1)
                {
                    order.push(nid);
                    params.extend(self.nodes[nid].1.parameters());
                }
            }
        }
        order.reverse();
        // Recalculate reference counts
        for (id, (rc, _)) in self.nodes.iter_mut().filter(|(rc, ..)| *rc == 0) {
            *rc = rcs[&id];
        }
        for &id in to_eval {
            self.nodes[id].0 += 1;
        }
        //std::println!("Execution order: {order:?}");
        // Reorder nodes in such a way, that movement ops are as late as possible,
        // after all unary ops just before reduce ops. (Do not reorder it after binary ops though.)
        /*#[cfg(feature = "std")]
        for nid in &order {
            std::println!("{nid} -> {:?}", self.nodes[nid]);
            }*/
        let mut node_swap = true;
        while node_swap {
            node_swap = false;
            for (&nid, &nid1) in order.iter().zip(order.iter().skip(1)) {
                if self.nodes[nid].1.is_movement()
                    && self.nodes[nid1].1.is_unary()
                    && !to_eval.contains(&nid)
                    && !to_eval.contains(&nid1)
                    && self.nodes[nid].0 == 1
                    && self.nodes[nid1].0 == 1
                {
                    //println!("Reordering movement and unary ops, swap {nid} and {nid1}");
                    self.swap_nodes(nid, nid1);
                    node_swap = true;
                }
            }
        }
        let mut flop = 0;
        let mut bytes_read = 0;
        for &nid in &order {
            match &self.nodes[nid].1 {
                Node::Const { .. } => {}
                Node::Leaf => {
                    bytes_read +=
                        self.shape(nid).iter().product::<usize>() * self.dtype(nid).byte_size();
                }
                &Node::Unary { x, .. } => {
                    flop += self.shape(x).iter().product::<usize>();
                }
                &Node::Binary { x, .. } => {
                    flop += self.shape(x).iter().product::<usize>();
                }
                &Node::Reduce { x, ref axes, .. } => {
                    flop += self
                        .shape(x)
                        .iter()
                        .enumerate()
                        .map(|(a, d)| {
                            if axes.contains(&a) {
                                if d - 1 > 0 {
                                    d - 1
                                } else {
                                    1
                                }
                            } else {
                                *d
                            }
                        })
                        .product::<usize>();
                }
                Node::Expand { .. }
                | Node::Permute { .. }
                | Node::Reshape { .. }
                | Node::Pad { .. } => {}
            }
        }
        let mut bytes_written = 0;
        for &nid in to_eval {
            bytes_written +=
                self.shape(nid).iter().product::<usize>() * self.dtype(nid).byte_size();
        }
        return (
            order,
            flop as u128,
            bytes_read as u128,
            bytes_written as u128,
        );
    }

    // Swap movement and unary op
    // first and second tensors must have rc == 1!
    fn swap_nodes(&mut self, first: TensorId, second: TensorId) {
        let temp;
        match &mut self.nodes[first].1 {
            Node::Reshape { x, .. }
            | Node::Expand { x, .. }
            | Node::Pad { x, .. }
            | Node::Permute { x, .. } => {
                temp = *x;
                *x = first;
            }
            _ => panic!("First op must be movement"),
        }
        match &mut self.nodes[second].1 {
            Node::Unary { x, .. } => {
                *x = temp;
            }
            _ => panic!("Second op must be unary"),
        }
        // swap the two nodes
        //let first_value = self.nodes.remove(first).unwrap().clone();
        //let second_value = self.nodes.remove(second).unwrap().clone();
        //self.nodes.insert(first, second_value);
        //self.nodes.insert(second, first_value);
        self.nodes.swap(first, second);
        // NOTE: do not forget to swap shapes and dtypes as well...
        if let Some(first_shape) = self.shapes.remove(&first) {
            if let Some(second_shape) = self.shapes.insert(second, first_shape) {
                self.shapes.insert(first, second_shape);
            }
        }
        if let Some(first_dtype) = self.dtypes.remove(&first) {
            if let Some(second_dtype) = self.dtypes.insert(second, first_dtype) {
                self.dtypes.insert(first, second_dtype);
            }
        }
    }

    pub(super) fn build_topo(&self, x: TensorId, sources: &BTreeSet<TensorId>) -> Vec<TensorId> {
        // Make a list of visited nodes and their reference counts.
        let mut params: Vec<TensorId> = vec![x];
        let mut rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                if !sources.contains(&nid)
                    && !matches!(
                        self.nodes[nid].1,
                        Node::Binary {
                            bop: BOp::Cmplt,
                            ..
                        }
                    )
                // or Node::Detach
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
                if rc
                    == *internal_rcs
                        .entry(nid)
                        .and_modify(|rc| *rc += 1)
                        .or_insert(1)
                {
                    order.push(nid);
                    params.extend(self.nodes[nid].1.parameters());
                }
            }
        }
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
        topo
    }

    /// Plot dot graph in dot format between given nodes
    #[must_use]
    pub fn plot_dot_graph(&self, ids: &BTreeSet<TensorId>) -> String {
        let ids: BTreeSet<TensorId> = if ids.is_empty() {
            self.nodes.ids().collect()
        } else {
            ids.clone()
        };
        // Make a list of visited nodes and their reference counts.
        let mut params: Vec<TensorId> = ids.iter().copied().collect();
        let mut rcs: BTreeMap<TensorId, u8> = BTreeMap::new();
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                params.extend(self.nodes[nid].1.parameters());
                1
            });
        }
        // Order them using rcs reference counts
        let mut order = Vec::new();
        let mut internal_rcs: BTreeMap<TensorId, u8> = BTreeMap::new();
        let mut params: Vec<TensorId> = ids.iter().copied().collect();
        while let Some(nid) = params.pop() {
            if rcs[&nid]
                == *internal_rcs
                    .entry(nid)
                    .and_modify(|rc| *rc += 1)
                    .or_insert(1)
            {
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
        /// Puts graph of nodes into dot language for visualization
        use core::fmt::Write;
        use std::format as f;
        let mut user_rc: BTreeMap<TensorId, u32> =
            self.nodes.iter().map(|(k, (rc, _))| (k, *rc)).collect();
        for (_, node) in self.nodes.values() {
            for param in node.parameters() {
                *user_rc.get_mut(&param).unwrap() -= 1;
            }
        }
        //std::println!("User {:?}", user_rc);
        let mut res =
            String::from("strict digraph {\n  ordering=in\n  rank=source\n  rankdir=LR\n");
        let mut add_node = |i: TensorId, text: &str, shape: &str| {
            let fillcolor = if user_rc[&i] > 0 { "lightblue" } else { "grey" };
            /*if let Some(label) = labels.get(&NodeId::new(id)) {
                write!(res, "  {id}[label=\"{}NL{} x {}NL{}NL{}\", shape={}, fillcolor=\"{}\", style=filled]",
                    label, id, rc[id], text, get_shape(NodeId::new(id)), shape, fillcolor).unwrap();
            } else {*/
            write!(
                res,
                "  {i}[label=\"{} x {}NL{}NL{:?}\", shape={}, fillcolor=\"{}\", style=filled]",
                i,
                self.nodes[i].0,
                text,
                self.shape(i),
                shape,
                fillcolor
            )
            .unwrap();
            writeln!(res).unwrap();
        };
        let mut edges = String::new();
        for &id in &topo {
            let node = &self.nodes[id].1;
            match node {
                Node::Const { value } => add_node(id, &f!("Const({value:?})"), "box"),
                Node::Leaf => add_node(
                    id,
                    &f!("Leaf({:?}, {})", self.shape(id), self.dtype(id)),
                    "box",
                ),
                Node::Unary { x, uop } => add_node(id, &f!("{uop:?}({x})"), "oval"),
                Node::Binary { x, y, bop } => add_node(id, &f!("{bop:?}({x}, {y})"), "oval"),
                Node::Reshape { x, .. } => add_node(id, &f!("Reshape({x})"), "oval"),
                Node::Permute { x, axes, .. } => {
                    add_node(id, &f!("Permute({x}, {axes:?})"), "oval")
                }
                Node::Expand { x, .. } => add_node(id, &f!("Expand({x})"), "oval"),
                Node::Pad { x, padding, .. } => add_node(id, &f!("Pad({x}, {padding:?})"), "oval"),
                Node::Reduce { x, axes, rop, .. } => {
                    add_node(id, &f!("{rop:?}({x}, {axes:?})"), "oval")
                }
            }
            for param in node.parameters() {
                writeln!(edges, "  {} -> {id}", param).unwrap();
            }
        }
        res = res.replace("NL", "\n");
        write!(res, "{edges}}}").unwrap();
        res
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
