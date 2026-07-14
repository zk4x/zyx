use crate::graph::{Graph, NodeId};

struct ExecPlan {}

enum ExecNode {
    Allocate {},
    Copy {},
    Deallocate {},
    Launch {},
}

impl ExecPlan {
    fn new(graph: &Graph, nodes: &[NodeId]) -> Self {
        todo!()
    }
}
