// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

//! Compiled graph caching layer.

use crate::{
    graph::Node,
    runtime::Runtime,
    shape::{Dim, UAxis},
    tensor::TensorId,
    Map, Set, ZyxError,
};
use std::hash::BuildHasherDefault;

/// Cached result of compiling a graph, ready for replay.
#[allow(dead_code)]
pub struct CompiledGraph {}

/// Compact representation of a graph, used as cache key.
pub struct CompactedGraph {
    pub nodes: Vec<Node>,
    pub shapes: Map<TensorId, Box<[Dim]>>,
    pub paddings: Map<TensorId, Box<[(i32, i32)]>>,
    pub axes: Map<TensorId, Box<[UAxis]>>,
}

impl Runtime {
    pub(crate) fn launch_or_store_graph_with_order(
        &mut self,
        _rcs: Map<TensorId, u32>,
        _realized_nodes: Set<TensorId>,
        order: &[TensorId],
        _to_eval: &Set<TensorId>,
    ) -> Result<(), ZyxError> {
        let mut compacted = CompactedGraph {
            nodes: Vec::with_capacity(order.len()),
            shapes: Map::with_capacity_and_hasher(order.len(), BuildHasherDefault::new()),
            paddings: Map::with_capacity_and_hasher(10, BuildHasherDefault::new()),
            axes: Map::with_capacity_and_hasher(10, BuildHasherDefault::new()),
        };
        let mut id_map: Map<TensorId, TensorId> = Map::with_capacity_and_hasher(order.len(), BuildHasherDefault::new());

        for (i, &nid) in order.iter().enumerate() {
            let new_id = TensorId::from(i);
            let node = &self.graph[nid];
            let reindexed = match node {
                Node::Const { value } => Node::Const { value: *value },
                Node::Leaf { dtype } => Node::Leaf { dtype: *dtype },
                Node::Expand { x } => Node::Expand { x: id_map[x] },
                Node::Permute { x } => Node::Permute { x: id_map[x] },
                Node::Reshape { x } => Node::Reshape { x: id_map[x] },
                Node::Pad { x } => Node::Pad { x: id_map[x] },
                Node::Reduce { x, rop } => Node::Reduce { x: id_map[x], rop: *rop },
                Node::Cast { x, dtype } => Node::Cast { x: id_map[x], dtype: *dtype },
                Node::Unary { x, uop } => Node::Unary { x: id_map[x], uop: *uop },
                Node::Binary { x, y, bop } => Node::Binary { x: id_map[x], y: id_map[y], bop: *bop },
            };
            compacted.nodes.push(reindexed);
            id_map.insert(nid, new_id);

            if matches!(
                node,
                Node::Leaf { .. }
                    | Node::Expand { .. }
                    | Node::Permute { .. }
                    | Node::Reshape { .. }
                    | Node::Pad { .. }
                    | Node::Reduce { .. }
            ) {
                compacted.shapes.insert(new_id, self.graph.shape(nid).into());
            }
            if matches!(node, Node::Pad { .. }) {
                compacted.paddings.insert(new_id, self.graph.padding(nid).into());
            }
            if matches!(node, Node::Permute { .. } | Node::Reduce { .. }) {
                compacted.axes.insert(new_id, self.graph.axes(nid).into());
            }
        }

        todo!()
    }
}
