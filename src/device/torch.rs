extern crate alloc;
use alloc::{boxed::Box, collections::{BTreeMap, BTreeSet}};
use crate::{shape::Shape, node_id::NodeId, graph::Node, OutOfMemoryError};
use tch::Tensor;

#[derive(Debug)]
pub(crate) struct TorchStorage {
    storage: Tensor,
}

impl TorchStorage {
    pub(super) fn shape(&self) -> Shape {
        self.storage.size().into_iter().map(|x| x as usize).collect::<Box<[usize]>>().into()
    }
}

#[derive(Debug)]
pub(crate) struct TorchDev;

impl TorchDev {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(super) fn load(&self, storage: &TorchStorage) -> Box<[f32]> {
        storage.storage.iter::<f32>().unwrap().collect()
    }

    pub(super) fn realize(
        &mut self,
        graph: &mut BTreeMap<NodeId, (usize, Node)>, // id, refcount and Node
        order: &[NodeId],                            // recommended realization order
        nodes: &BTreeSet<NodeId>,                    // which nodes need to be realized
    ) -> Result<(), OutOfMemoryError> {
        todo!()
    }
}
