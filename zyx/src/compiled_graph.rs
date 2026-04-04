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

/// Cached result of compiling a graph, ready for replay.
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
        rcs: Map<TensorId, u32>,
        realized_nodes: Set<TensorId>,
        order: &[TensorId],
        to_eval: &Set<TensorId>,
    ) -> Result<(), ZyxError> {
        todo!()
    }
}
