use crate::graph::{Graph, NodeId};

pub struct ExecPlan {}

enum ExecNode {
    Allocate {},
    Copy {},
    Deallocate {},
    Launch {},
}

impl ExecPlan {
    #[must_use]
    pub fn new(graph: &Graph, nodes: &[NodeId]) -> Self {
        todo!()
    }
}
