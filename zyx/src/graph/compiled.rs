// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#![allow(unused)]

//! Compiled graph caching layer.
use crate::{
    DType, Map, ZyxError,
    backend::{BufferId, Device, DeviceId, MemoryPool, PoolId, ProgramId},
    graph::{Graph, search::EGraph},
    hashers,
    runtime::Runtime,
    shape::Dim,
    slab::Slab,
    tensor::TensorId,
};
use std::hash::BuildHasherDefault;

/// Logical index into a replay-time slot table, used instead of raw [`BufferId`]
/// so the compiled graph is stable across runs with different buffer IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferSlot(pub u32);

/// A single step in a compiled graph execution.
#[derive(Debug, Clone)]
pub enum CompiledNode {
    Leaf {
        pool: PoolId,
        slot: BufferSlot,
    },
    Allocate {
        size: Dim,
        pool: PoolId,
        slot: BufferSlot,
    },
    Deallocate {
        pool: PoolId,
        slot: BufferSlot,
    },
    CopyMemory {
        src_pool: PoolId,
        src: BufferSlot,
        dst_pool: PoolId,
        dst: BufferSlot,
    },
    LaunchProgram {
        program: ProgramId,
        args: Vec<BufferSlot>,
    },
}

/// Compute a structural hash for the subgraph in `order`.
///
/// The hash is position-based: for each node in topological order, its input
/// hashes are the hashes of positions `order[i]` for its input TensorIds.
/// This gives the same key for structurally identical subgraphs regardless
/// of TensorId or buffer contents.
fn hash_order(order: &[TensorId], graph: &Graph) -> u128 {
    use std::hash::{Hash, Hasher};
    let mut hashes: Vec<u64> = Vec::with_capacity(order.len());
    let mut pos_of: Map<TensorId, usize> = Map::with_capacity_and_hasher(order.len(), BuildHasherDefault::new());
    for (i, &tid) in order.iter().enumerate() {
        pos_of.insert(tid, i);
    }

    for &tid in order {
        let node = &graph[tid];
        let dtype = graph.dtype(tid);
        let shape = graph.shape(tid);
        let params = node.parameters();

        let h1 = params.first().map(|&p| hashes[pos_of[&p]]).unwrap_or(0);
        let h2 = params.get(1).map(|&p| hashes[pos_of[&p]]).unwrap_or(0);

        let mut hasher = hashers::AHasher::default();
        node.kind_tag().hash(&mut hasher);
        node.extra_hash().hash(&mut hasher);
        dtype.hash(&mut hasher);
        shape.hash(&mut hasher);
        h1.hash(&mut hasher);
        h2.hash(&mut hasher);
        hashes.push(hasher.finish());
    }

    let mut h1 = 0u128;
    let mut h2 = 0u128;
    for &h in &hashes {
        h1 = h1.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(h as u128);
        h2 = h2.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add((h >> 1) as u128);
    }
    h1 | (h2 << 64)
}

impl Runtime {
    /// Launch graph or store in cache for faster replay.
    ///
    /// Computes a structural hash from `order`, checks the cache, and either
    /// replays the cached [`CompiledNode`] sequence or compiles via [`EGraph`],
    /// stores it, and replays.
    pub(crate) fn launch_or_store_graph_with_order(&mut self, inputs: &[TensorId], order: &[TensorId]) -> Result<(), ZyxError> {
        let key = hash_order(order, &self.graph);

        let input_buffers: Vec<BufferId> = inputs
            .iter()
            .map(|tid| {
                self.buffer_map
                    .get(tid)
                    .copied()
                    .unwrap_or_else(|| panic!("input tensor {tid:?} not realized"))
            })
            .collect();

        if let Some(compiled_nodes) = self.graph_cache.get(&key) {
            return replay_compiled(&mut self.pools, &mut self.devices, compiled_nodes, &input_buffers);
        }

        let mut egraph = EGraph::new(order, &self.graph);
        egraph.saturate();
        let compiled_nodes = egraph.extract();

        replay_compiled(&mut self.pools, &mut self.devices, &compiled_nodes, &input_buffers)?;
        self.graph_cache.insert(key, compiled_nodes);
        Ok(())
    }
}

/// Replay a cached compiled graph.
///
/// Maintains an ephemeral slot table (`Vec<Option<BufferId>>`) that is indexed
/// by [`BufferSlot`].
///
/// [`CompiledNode::Leaf`] pre-populates a slot from the `inputs` slice
/// (one per Leaf, in order of appearance).
/// [`CompiledNode::Allocate`] allocates and fills a slot.
/// [`CompiledNode::LaunchProgram`] reads argument slots.
/// [`CompiledNode::Deallocate`] drains them.
fn replay_compiled(
    pools: &mut Slab<PoolId, MemoryPool>,
    devices: &mut Slab<DeviceId, Device>,
    nodes: &[CompiledNode],
    inputs: &[BufferId],
) -> Result<(), ZyxError> {
    let mut slots: Vec<Option<BufferId>> = Vec::new();
    let mut input_idx = 0;

    for node in nodes {
        match node {
            CompiledNode::Leaf { slot, .. } => {
                let buf = inputs[input_idx];
                input_idx += 1;
                let idx = slot.0 as usize;
                if idx >= slots.len() {
                    slots.resize(idx + 1, None);
                }
                slots[idx] = Some(buf);
            }
            CompiledNode::Allocate { pool, size, slot } => {
                let (buf, _event) = pools[*pool].allocate(*size)?;
                let idx = slot.0 as usize;
                if idx >= slots.len() {
                    slots.resize(idx + 1, None);
                }
                slots[idx] = Some(BufferId { pool: *pool, buffer: buf });
            }
            CompiledNode::Deallocate { pool, slot } => {
                if let Some(buf) = slots[slot.0 as usize].take() {
                    pools[*pool].deallocate(buf.buffer, vec![]);
                }
            }
            CompiledNode::CopyMemory { src_pool, src, dst_pool, dst } => {
                let _src_buf = slots[src.0 as usize].unwrap();
                let _dst_buf = slots[dst.0 as usize].unwrap();
                // TODO: device-to-device copy between pools
            }
            CompiledNode::LaunchProgram { program, args } => {
                let pool_id = devices[program.device].memory_pool_id();
                let pool = &mut pools[pool_id];
                let kernel_args: Vec<_> = args.iter().map(|s| slots[s.0 as usize].unwrap().buffer).collect();
                let _event = devices[program.device].launch(program.program, pool, &kernel_args, vec![])?;
            }
        }
    }
    Ok(())
}
