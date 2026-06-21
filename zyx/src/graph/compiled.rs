// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#![allow(unused)]

//! Compiled graph caching layer.
use crate::{
    DType, Map, Set, ZyxError,
    backend::{BufferId, DeviceId, PoolId, ProgramId},
    graph::{Node, search::EGraph},
    runtime::Runtime,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
};
use std::collections::BTreeMap;
use std::hash::BuildHasherDefault;

/// Cached result of compiling a graph, ready for replay.
#[allow(dead_code)]
pub struct CompiledGraph {
    pub nodes: Vec<CompiledNode>,
    pub buffer_slots: Vec<BufferId>,
}

/// Index into [`CompiledGraph::buffer_slots`], used instead of raw [`BufferId`]
/// so the compiled graph is stable across runs with different buffer IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferSlot(pub u32);

/// A single step in a compiled graph execution.
#[derive(Debug, Clone)]
pub enum CompiledNode {
    Allocate {
        pool: PoolId,
        size: Dim,
        slot: BufferSlot,
    },
    Deallocate {
        pool: PoolId,
        slot: BufferSlot,
    },
    CopyMemory {
        src_pool: PoolId,
        src_buffer: BufferSlot,
        dst_pool: PoolId,
        dst_buffer: BufferSlot,
    },
    LaunchProgram {
        program: ProgramId,
        device: DeviceId,
    },
}

#[derive(Debug, Copy, Clone, Hash, Ord, PartialEq, PartialOrd, Eq)]
pub struct NodeId(pub u32);

impl SlabId for NodeId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl From<usize> for NodeId {
    fn from(value: usize) -> Self {
        Self(value as u32)
    }
}

impl From<NodeId> for usize {
    fn from(val: NodeId) -> Self {
        val.0 as usize
    }
}

/// Compact representation of a graph, used as cache key.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct CachedGraph {
    pub nodes: Slab<NodeId, Node>,
    pub shapes: BTreeMap<NodeId, Box<[Dim]>>,
    pub paddings: BTreeMap<NodeId, Box<[(i64, i64)]>>,
    pub axes: BTreeMap<NodeId, Box<[UAxis]>>,
}

impl CachedGraph {
    pub(super) fn shape(&self, mut node_id: NodeId) -> &[Dim] {
        for _ in 0..1_000_000 {
            if let Some(shape) = self.shapes.get(&node_id) {
                //println!("Found shape {shape:?} for tensor {tensor_id}");
                return shape;
            } else if let Node::Const { .. } = self.nodes[node_id] {
                return &[1];
            }
            //println!("Getting params of id: {tensor_id}, {:?}", self.nodes[tensor_id].1);
            node_id = NodeId(self.nodes[node_id].param1().0);
        }
        panic!("Shape of {node_id:?} could not be found. This is internal bug.")
    }

    pub(super) fn dtype(&self, mut node_id: NodeId) -> DType {
        for _ in 0..100_000 {
            match self.nodes[node_id] {
                Node::Const { value } => return value.dtype(),
                Node::Leaf { dtype } | Node::Cast { dtype, .. } => return dtype,
                Node::Binary { bop, .. } if bop.returns_bool() => {
                    return DType::Bool;
                }
                _ => {
                    node_id = NodeId(self.nodes[node_id].parameters().into_iter().next().unwrap().0);
                }
            }
        }
        panic!("DType of {node_id:?} could not be found. This is internal bug.")
    }
}

impl Runtime {
    /// Launch graph or store in cache for faster replay.
    ///
    /// This function implements the following flow:
    /// 1. **Build compacted [`CachedGraph`]**: Create a minimal representation of the graph by
    ///    re-indexing all tensor IDs to 0..N and extracting shapes/paddings/axes. This compacted
    ///    form is used as the hash key for cache lookup.
    /// 2. **Cache hit**: If [`CachedGraph`] exists in `graph_cache`, execute its `CompiledGraph.nodes`
    ///    which contains an optimized sequence of Allocate/CopyMemory/LaunchProgram/Deallocate ops.
    /// 3. **Cache miss**: If not found, call `search::search(&compacted)` to:
    ///    - Build an [`EGraph`] from the [`CachedGraph`] nodes
    ///    - Enumerate all fusion strategies by add ing fused variants (matmul, elementwise, reduce chains)
    ///    - Compute and compare costs for each variant combination
    ///    - Select the fastest execution path and convert to `CompiledGraph`
    ///    - Store `(egraph, compiled_graph)` in cache for future reuse
    /// 4. **Execute**: After either cache hit or miss, perform necessary memory operations (allocate,
    ///    copy, deallocate buffers) before launching programs.
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn launch_or_store_graph_with_order(
        &mut self,
        _rcs: &Map<TensorId, u32>,
        realized_nodes: &Set<TensorId>,
        order: &[TensorId],
        _to_eval: &Set<TensorId>,
    ) -> Result<(), ZyxError> {
        let mut compacted = CachedGraph {
            nodes: Slab::with_capacity(order.len()),
            shapes: BTreeMap::new(),
            paddings: BTreeMap::new(),
            axes: BTreeMap::new(),
        };
        let mut id_map: Map<TensorId, NodeId> = Map::with_capacity_and_hasher(order.len(), BuildHasherDefault::new());

        for (id, &nid) in order.iter().enumerate() {
            let new_id = NodeId::from(id);
            let node = &self.graph[nid];
            let reindexed = if realized_nodes.contains(&nid) {
                Node::Leaf { dtype: self.graph.dtype(nid) }
            } else {
                match node {
                    Node::Const { value } => Node::Const { value: *value },
                    Node::Leaf { dtype } => Node::Leaf { dtype: *dtype },
                    Node::Expand { x } => Node::Expand { x: TensorId(id_map[x].0) },
                    Node::Permute { x } => Node::Permute { x: TensorId(id_map[x].0) },
                    Node::Reshape { x } => Node::Reshape { x: TensorId(id_map[x].0) },
                    Node::Pad { x } => Node::Pad { x: TensorId(id_map[x].0) },
                    Node::Reduce { x, rop } => Node::Reduce { x: TensorId(id_map[x].0), rop: *rop },
                    Node::Cast { x, dtype } => Node::Cast { x: TensorId(id_map[x].0), dtype: *dtype },
                    Node::Unary { x, uop } => Node::Unary { x: TensorId(id_map[x].0), uop: *uop },
                    Node::Binary { x, y, bop } => Node::Binary { x: TensorId(id_map[x].0), y: TensorId(id_map[y].0), bop: *bop },
                    Node::Custom { .. } => todo!(),
                    Node::ToDevice { x, device } => Node::ToDevice { x: TensorId(id_map[x].0), device: *device },
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

        if let Some(compiled_graph) = self.graph_cache.get(&compacted) {
            // TODO: Execute cached_graph
            return Ok(());
        }

        //let compacted_realized: Set<TensorId> = realized_nodes.iter().filter_map(|&tid| id_map.get(&tid).copied()).collect();

        let mut egraph = EGraph::new(&compacted);
        egraph.saturate();
        let compiled_graph = egraph.extract();

        self.graph_cache.insert(compacted.clone(), compiled_graph);

        if let Some(compiled_graph) = self.graph_cache.get(&compacted) {
            // TODO: Execute cached_graph
            return Ok(());
        }

        Ok(())
    }
}
