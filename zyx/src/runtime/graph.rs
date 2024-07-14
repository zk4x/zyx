use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};

use crate::{DType, Device};

use super::{node::Node, TensorId};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone)]
pub(super) struct Graph {
    // First value is reference count, second is node
    nodes: BTreeMap<TensorId, (u32, Node)>,
}

impl Graph {
    pub(crate) const fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
        }
    }

    pub(crate) fn retain(&mut self, x: TensorId) {
        self.nodes.get_mut(&x).unwrap().0 += 1;
    }

    /// Returns which tensors should be deallocated
    pub(crate) fn release(&mut self, x: TensorId) -> BTreeSet<TensorId> {
        let mut params = Vec::with_capacity(10);
        params.push(x);
        let mut to_remove = BTreeSet::new();
        while let Some(x) = params.pop() {
            let node = self.nodes.get_mut(&x).unwrap();
            node.0 -= 1;
            if node.0 == 0 {
                params.extend(node.1.parameters());
                self.nodes.remove(&x);
                to_remove.insert(x);
            }
        }
        to_remove
    }

    pub(crate) fn push(&mut self, node: Node) -> TensorId {
        for nid in node.parameters() {
            self.nodes.get_mut(&nid).unwrap().0 += 1;
        }
        let id = if let Some((id, _)) = self.nodes.last_key_value() {
            *id
        } else {
            1
        };
        self.nodes.insert(id, (0, node));
        return id;
    }

    pub(crate) fn dtype(&self, tensor_id: TensorId) -> DType {
        let mut x = tensor_id;
        let mut i = 0;
        while i < 1000000 {
            let node = &self.nodes[&x].1;
            match node {
                Node::Leaf { dtype, .. } | Node::Cast { dtype, .. } => return *dtype,
                //Node::Const { value, .. } => return value.dtype(),
                _ => x = node.parameters().next().unwrap(),
            }
            i += 1;
        }
        panic!("DType of {x} could not be found. This is internal bug.")
    }

    pub(crate) fn device(&self, tensor_id: TensorId) -> Device {
        let mut x = tensor_id;
        let mut i = 0;
        while i < 1000000 {
            let node = &self.nodes[&x].1;
            match node {
                Node::Leaf { device, .. } => return *device,
                // TODO | Node::ToDevice { device, .. }
                _ => x = node.parameters().next().unwrap(),
            }
            i += 1;
        }
        panic!("DType of {x} could not be found. This is internal bug.")
    }

    pub(crate) fn shape(&self, tensor_id: TensorId) -> &[usize] {
        let mut x = tensor_id;
        let mut i = 0;
        while i < 10000 {
            let node = &self.nodes[&x].1;
            match node {
                //Node::Const { .. } => return &[1],
                Node::Leaf { shape, .. }
                | Node::Reshape { shape, .. }
                | Node::Pad { shape, .. }
                | Node::Permute { shape, .. }
                | Node::Reduce { shape, .. }
                | Node::Expand { shape, .. } => return &shape,
                _ => x = node.parameters().next().unwrap(),
            }
            i += 1;
        }
        panic!("Shape of {x} could not be found. This is internal bug.")
    }

    pub(crate) fn realize_graph(
        &self,
        tensors: &BTreeSet<TensorId>,
        is_realized: impl Fn(TensorId) -> bool,
    ) -> Self {
        // First topo search for minimum number of required nodes and create graph from it
        // Then replace all realized nodes with Node::Leaf
        // topo search
        let mut params: Vec<TensorId> = tensors.iter().map(|tensor_id| *tensor_id).collect();
        let mut visited = BTreeSet::new();
        let mut leafs = BTreeSet::new();
        while let Some(param) = params.pop() {
            if visited.insert(param) {
                if is_realized(param) {
                    leafs.insert(param);
                } else {
                    params.extend(self.nodes[&param].1.parameters());
                }
            }
        }
        return Graph {
            nodes: visited
                .into_iter()
                .map(|id| {
                    (
                        id,
                        if leafs.contains(&id) {
                            (
                                1,
                                Node::Leaf {
                                    shape: self.shape(id).into(),
                                    dtype: self.dtype(id),
                                    device: self.device(id),
                                },
                            )
                        } else {
                            self.nodes[&id].clone()
                        },
                    )
                })
                .collect(),
        };
    }

    // Calculates execution order and optimizes graph by moving all unary ops before movement ops
    pub(crate) fn execution_order(&mut self, to_eval: &BTreeSet<TensorId>) -> Vec<TensorId> {
        let _ = to_eval;
        // TODO actual calculation of execution order once we no longer just use from lowest id to highest id order
        self.nodes.keys().copied().collect()
    }

    pub(crate) fn swap_movement_and_unary_ops(&mut self) {
        todo!()
        // Reorder nodes in such a way, that movement ops are as late as possible,
        // after all unary ops just before reduce ops. (Do not reorder it after binary ops though.)
        /*let mut node_swap = true;
        while node_swap {
            node_swap = false;
            for nid in order.iter().take(order.len() - 1) {
                if graph.nodes[nid].is_movement() && graph.nodes[&(nid + 1)].is_unary() {
                    //libc_print::libc_println!("Reordering movement and unary ops, swap {} and {}", nid, nid+1);
                    graph.swap_nodes(*nid, nid + 1);
                    node_swap = true;
                }
            }
        }*/
    }

    // Swap movement and unary op
    // first and second tensors must have rc == 1!
    fn swap_nodes(&mut self, first: TensorId, second: TensorId) {
        let _ = first;
        let _ = second;
        todo!()
        /*let temp;
        match self.nodes.get_mut(&first).unwrap() {
            Node::Reshape { x, .. }
            | Node::Expand { x, .. }
            | Node::Pad { x, .. }
            | Node::Permute { x, .. } => {
                temp = *x;
                *x = first;
            }
            _ => panic!("First op must be movement"),
        }
        match self.nodes.get_mut(&second).unwrap() {
            Node::Cast { x, .. }
            | Node::Neg { x, .. }
            | Node::Inv { x, .. }
            | Node::ReLU { x, .. }
            | Node::Exp { x, .. }
            | Node::Ln { x, .. }
            | Node::Sin { x, .. }
            | Node::Cos { x, .. }
            | Node::Sqrt { x, .. } => {
                *x = temp;
            }
            _ => panic!("Second op must be unary"),
        }
        // swap the two nodes
        let first_value = self.nodes.remove(&first).unwrap();
        let second_value = self.nodes.remove(&second).unwrap();
        self.nodes.insert(first, second_value);
        self.nodes.insert(second, first_value);*/
    }
}

impl core::ops::Index<TensorId> for Graph {
    type Output = Node;
    fn index(&self, index: TensorId) -> &Self::Output {
        &self.nodes.get(&index).unwrap().1
    }
}

/*
fn calculate_graph_rcs(
    subgraph: &Subgraph,
    to_eval: &BTreeSet<TensorId>,
) -> BTreeMap<TensorId, u32> {
    // Depth first search through graph. Number of visits of each node are reference counts.
    let mut visited_rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
    let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
    params.reserve(100);
    while let Some(nid) = params.pop() {
        //std::println!("{nid} is evaluated: {}", self.runtime_backend.is_evaluated(nid));
        visited_rcs
            .entry(nid)
            .and_modify(|rc| *rc += 1)
            .or_insert_with(|| {
                params.extend(subgraph[nid].parameters());
                1
            });
    }
    //println!("Temp: {visited_rcs:?}");
    return visited_rcs;
}

fn calculate_graph_execution_order(
    graph: &Subgraph,
    to_eval: &BTreeSet<TensorId>,
    temp_rcs: &BTreeMap<TensorId, u32>,
) -> Vec<TensorId> {
    // Calculates dependency graph of nodes and viable execution order, which is not currently
    // optimized. It is depth first search. On each visit of the node, rc is increased. Once
    // rc of the node A reaches A's rc in the whole graph, then A gets added to the order,
    // that is, there are no more nodes that node A depends on, i.e. there are no nodes that
    // need to be evaluated before A.
    let mut order = Vec::new();
    let mut rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
    let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
    params.reserve(100);
    while let Some(nid) = params.pop() {
        if let Some(temp_rc) = temp_rcs.get(&nid) {
            let rc = rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
            if *temp_rc == *rc {
                order.push(nid);
                params.extend(graph[nid].parameters());
            }
        }
    }
    order.reverse();
    return order;
    }*/

//println!("{subgraph:?}");
//println!("{hw_info:?}");
// Find the best order of execution of nodes
//let rcs = calculate_graph_rcs(&graph, &to_eval);
//let mut order = calculate_graph_execution_order(&graph, &to_eval, &rcs);
