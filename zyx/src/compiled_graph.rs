// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

//! Compiled graph caching layer.

use crate::{
    backend::{BufferId, ProgramId},
    cache::DeviceId,
    graph::Node,
    runtime::Runtime,
    shape::{Dim, UAxis},
    tensor::TensorId,
    Map, Set, ZyxError,
};
use std::collections::BTreeMap;
use std::hash::BuildHasherDefault;

/// Cached result of compiling a graph, ready for replay.
#[allow(dead_code)]
pub struct CompiledGraph {
    pub nodes: Vec<CompiledNode>,
    pub buffer_slots: Vec<BufferId>,
}

/// Index into CompiledGraph::buffer_slots, used instead of raw BufferId
/// so the compiled graph is stable across runs with different buffer IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferSlot(pub u32);

/// A single step in a compiled graph execution.
#[derive(Debug, Clone)]
pub enum CompiledNode {
    Allocate {
        pool: usize,
        size: usize,
        slot: BufferSlot,
    },
    Deallocate {
        pool: usize,
        slot: BufferSlot,
    },
    CopyMemory {
        src_pool: usize,
        src_buffer: BufferSlot,
        dst_pool: usize,
        dst_buffer: BufferSlot,
    },
    LaunchProgram {
        program: ProgramId,
        device: DeviceId,
    },
}

/// Compact representation of a graph, used as cache key.
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct CompactedGraph {
    pub nodes: Vec<Node>,
    pub shapes: BTreeMap<TensorId, Box<[Dim]>>,
    pub paddings: BTreeMap<TensorId, Box<[(i32, i32)]>>,
    pub axes: BTreeMap<TensorId, Box<[UAxis]>>,
}

impl Runtime {
    pub(crate) fn launch_or_store_graph_with_order(
        &mut self,
        _rcs: Map<TensorId, u32>,
        realized_nodes: Set<TensorId>,
        order: &[TensorId],
        to_eval: &Set<TensorId>,
    ) -> Result<(), ZyxError> {
        let mut compacted = CompactedGraph {
            nodes: Vec::with_capacity(order.len()),
            shapes: BTreeMap::new(),
            paddings: BTreeMap::new(),
            axes: BTreeMap::new(),
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

        if let Some(_cached_graph) = self.graph_cache.get(&compacted) {
            // TODO: replay cached graph
        } else {
            let compiled = self.kernelize(&compacted, realized_nodes, to_eval)?;
            self.graph_cache.insert(compacted, compiled);
        }

        Ok(())
    }
}
