extern crate alloc;
use crate::{graph::Node, node_id::NodeId, shape::Shape, OutOfMemoryError};
use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
};
use tch::Tensor;

#[derive(Debug)]
pub(crate) struct TorchStorage {
    storage: Tensor,
}

impl TorchStorage {
    pub(super) fn shape(&self) -> Shape {
        self.storage
            .size()
            .into_iter()
            .map(|x| x as usize)
            .collect::<Box<[usize]>>()
            .into()
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
