// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Compiled graph caching layer.

use crate::{
    DType, Map, Set, ZyxError,
    backend::{BufferId, DeviceId, ProgramId},
    graph::Node,
    kernel::{Kernel, MoveOp, Op, OpId, OpNode},
    kernelize::KMKernelId,
    runtime::Runtime,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
    view::View,
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
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct CachedGraph {
    pub nodes: Vec<Node>,
    pub shapes: BTreeMap<TensorId, Box<[Dim]>>,
    pub paddings: BTreeMap<TensorId, Box<[(i64, i64)]>>,
    pub axes: BTreeMap<TensorId, Box<[UAxis]>>,
}

impl Runtime {
    pub(crate) fn launch_or_store_graph_with_order(
        &mut self,
        rcs: Map<TensorId, u32>,
        realized_nodes: Set<TensorId>,
        order: &[TensorId],
        to_eval: &Set<TensorId>,
    ) -> Result<(), ZyxError> {
        let mut compacted = CachedGraph {
            nodes: Vec::with_capacity(order.len()),
            shapes: BTreeMap::new(),
            paddings: BTreeMap::new(),
            axes: BTreeMap::new(),
        };
        let mut id_map: Map<TensorId, TensorId> = Map::with_capacity_and_hasher(order.len(), BuildHasherDefault::new());

        for (i, &nid) in order.iter().enumerate() {
            println!("{nid} -> {:?}", self.graph[nid]);
            let new_id = TensorId::from(i);
            let node = &self.graph[nid];
            let reindexed = if realized_nodes.contains(&nid) {
                Node::Leaf { dtype: self.graph.dtype(nid) }
            } else {
                match node {
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
                    Node::Custom { .. } => todo!(),
                }
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
            return Ok(());
        }

        let compacted_realized: Set<TensorId> = realized_nodes.iter().filter_map(|&tid| id_map.get(&tid).copied()).collect();


        Ok(())
    }
}
