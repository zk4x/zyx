use alloc::collections::BTreeMap;

use super::TensorId;

pub(super) struct Graph {
    nodes: BTreeMap<TensorId, Node>,
}
