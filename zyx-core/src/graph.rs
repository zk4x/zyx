use crate::{axes::Axes, shape::Shape, tensor::Id, Vec, Map, Set, Node};

// We need to remember repeating parts of graph, then find the best way to evaluate them.
// Repeating means single training/inference loop.

pub struct Graph {
    rcs: Vec<u8>,
    order: Vec<Id>,
    nodes: Vec<Node>,
    no_diff: Set<Id>, // Ids that are "leafs" in graph, no backpropagation can be done on them,
                      // because user does not hold references to their ancestors
}

impl Graph {
    fn push_node(&mut self, node: Node, x: Id, y: Id) -> Id {
        self.nodes.push(node);
        todo!()
    }

    fn backward(&self, x: Id, sources: &[Id]) -> Vec<Option<Id>> {
        todo!()
    }
}
