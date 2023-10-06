extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use crate::{shape::Shape, node_id::NodeId, graph::Node, OutOfMemoryError};

#[derive(Debug)]
pub(crate) struct TorchStorage<T> {
    _a: T,
}

impl<T> TorchStorage<T> {
    pub(super) fn shape(&self) -> &Shape {
        todo!()
    }
}

#[derive(Debug)]
pub(crate) struct TorchDev {}

impl TorchDev {
    pub(super) fn realize(
        &mut self,
        _graph: &mut BTreeMap<NodeId, (usize, Node)>, // id, refcount and Node
        _order: &[NodeId],                            // recommended realization order
        _nodes: &BTreeSet<NodeId>,                    // which nodes need to be realized
    ) -> Result<(), OutOfMemoryError> {
        todo!()
    }
}
