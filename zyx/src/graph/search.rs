// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! E-graph based search for fusion strategies.
//!
//! The e-graph stores multiple equivalent implementations for each buffer.
//! Each buffer can be produced by the original operation, a fused kernel,
//! or a copy from another pool. Saturation adds fused variants without
//! replacing existing ones.

use crate::Map;
use crate::Set;
use crate::DType;
use crate::backend::PoolId;
use crate::dtype::Constant;
use crate::graph::compiled::{BufferSlot, CachedGraph, CompiledGraph};
use crate::graph::Node;
use crate::kernel::{BOp, UOp};
use crate::shape::{Dim, UAxis};
use crate::slab::{Slab, SlabId};
use crate::tensor::TensorId;

/// Index into the e-nodes slab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ENodeId(u32);

impl From<usize> for ENodeId {
    fn from(value: usize) -> Self {
        ENodeId(value as u32)
    }
}

impl From<ENodeId> for usize {
    fn from(value: ENodeId) -> Self {
        value.0 as usize
    }
}

impl SlabId for ENodeId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

// SlabId impl for BufferSlot (defined in compiled.rs)
impl From<usize> for BufferSlot {
    fn from(value: usize) -> Self {
        BufferSlot(value as u32)
    }
}

impl From<BufferSlot> for usize {
    fn from(value: BufferSlot) -> Self {
        value.0 as usize
    }
}

impl SlabId for BufferSlot {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

/// A single operation node in the e-graph. Can be a kernel or a memory operation.
pub enum ENode {
    Leaf { output: BufferSlot },
    Const { output: BufferSlot, value: Constant },
    Expand { input: BufferSlot, output: BufferSlot },
    Permute { input: BufferSlot, output: BufferSlot, axes: Box<[UAxis]> },
    Reshape { input: BufferSlot, output: BufferSlot },
    Pad { input: BufferSlot, output: BufferSlot, padding: Box<[(i64, i64)]> },
    Reduce { input: BufferSlot, output: BufferSlot, rop: BOp, axes: Box<[UAxis]> },
    Cast { input: BufferSlot, output: BufferSlot },
    Unary { input: BufferSlot, output: BufferSlot, uop: UOp },
    Binary { x: BufferSlot, y: BufferSlot, output: BufferSlot, bop: BOp },
    Copy { src: BufferSlot, dst: BufferSlot, cost: u64 },
    Fused { inputs: Vec<BufferSlot>, outputs: Vec<BufferSlot>, cost: u64, op: Box<dyn FusedOp> },
}

/// Trait for fused kernel operations.
pub trait FusedOp {
    /// Returns the estimated execution cost.
    fn cost(&self) -> u64;
}

/// Fused matmul kernel (Reduce(Add) over Binary(Mul) with expands).
pub struct MatMulOp {}

impl FusedOp for MatMulOp {
    fn cost(&self) -> u64 {
        0
    }
}

/// Metadata for a buffer slot.
pub struct BufferSlotInfo {
    /// Memory pool this buffer lives in.
    pub pool: PoolId,
    /// Tensor shape.
    pub shape: Box<[Dim]>,
    /// Data type.
    pub dtype: DType,
}

/// The e-graph: a hypergraph of buffer producers and consumers.
/// Each buffer can have multiple producers (original op, fused kernel, copy).
/// Extraction selects the cheapest combination.
pub struct EGraph {
    /// All operation nodes (kernels and memory ops).
    enodes: Slab<ENodeId, ENode>,
    /// Buffer metadata indexed by BufferSlot.
    buffers: Slab<BufferSlot, BufferSlotInfo>,
    /// Which nodes produce each buffer?
    producers: Map<BufferSlot, Vec<ENodeId>>,
    /// Which nodes consume each buffer?
    consumers: Map<BufferSlot, Vec<ENodeId>>,
}

impl ENode {
    /// Returns a list of input buffers.
    pub fn inputs(&self) -> Vec<&BufferSlot> {
        match self {
            ENode::Leaf { .. } | ENode::Const { .. } => vec![],
            ENode::Expand { input, .. }
            | ENode::Permute { input, .. }
            | ENode::Reshape { input, .. }
            | ENode::Pad { input, .. }
            | ENode::Reduce { input, .. }
            | ENode::Cast { input, .. }
            | ENode::Unary { input, .. }
            | ENode::Copy { src: input, .. } => vec![input],
            ENode::Binary { x, y, .. } => vec![x, y],
            ENode::Fused { inputs, .. } => inputs.iter().collect(),
        }
    }

    /// Returns a list of output buffers.
    pub fn outputs(&self) -> Vec<&BufferSlot> {
        match self {
            ENode::Leaf { output }
            | ENode::Const { output, .. }
            | ENode::Expand { output, .. }
            | ENode::Permute { output, .. }
            | ENode::Reshape { output, .. }
            | ENode::Pad { output, .. }
            | ENode::Reduce { output, .. }
            | ENode::Cast { output, .. }
            | ENode::Unary { output, .. }
            | ENode::Binary { output, .. }
            | ENode::Copy { dst: output, .. } => vec![output],
            ENode::Fused { outputs, .. } => outputs.iter().collect(),
        }
    }
}

impl EGraph {
    /// Build e-graph from a cached graph. Each CachedGraph node becomes a BufferSlot
    /// on the default pool (pool 1). Memory ops (Leaf, Copy) are added as needed.
    pub fn new(graph: &CachedGraph) -> EGraph {
        let mut enodes = Slab::new();
        let mut buffers = Slab::new();
        let mut producers: Map<BufferSlot, Vec<ENodeId>> = Map::default();
        let mut consumers: Map<BufferSlot, Vec<ENodeId>> = Map::default();

        let default_pool = PoolId::from(1);

        let buf_id_from_tensor_id = |tid: usize| BufferSlot::from(tid);

        for (tid, node) in graph.nodes.iter().enumerate() {
            let buf_slot = buf_id_from_tensor_id(tid);
            let tensor_id = TensorId::from(tid);
            let shape = graph.shapes.get(&tensor_id).cloned().unwrap_or_else(|| Box::new([1]));
            let dtype = Self::infer_dtype(node, &graph.nodes, tid);

            buffers.push(BufferSlotInfo { pool: default_pool, shape, dtype });

            let enode = match node {
                Node::Leaf { .. } => ENode::Leaf { output: buf_slot },
                Node::Const { value } => ENode::Const { output: buf_slot, value: *value },
                Node::Expand { x } => ENode::Expand {
                    input: buf_id_from_tensor_id((*x).into()),
                    output: buf_slot,
                },
                Node::Permute { x } => ENode::Permute {
                    input: buf_id_from_tensor_id((*x).into()),
                    output: buf_slot,
                    axes: graph.axes.get(&tensor_id).cloned().unwrap_or_else(|| Box::new([])),
                },
                Node::Reshape { x } => ENode::Reshape {
                    input: buf_id_from_tensor_id((*x).into()),
                    output: buf_slot,
                },
                Node::Pad { x } => ENode::Pad {
                    input: buf_id_from_tensor_id((*x).into()),
                    output: buf_slot,
                    padding: graph.paddings.get(&tensor_id).cloned().unwrap_or_else(|| Box::new([])),
                },
                Node::Reduce { x, rop } => ENode::Reduce {
                    input: buf_id_from_tensor_id((*x).into()),
                    output: buf_slot,
                    rop: *rop,
                    axes: graph.axes.get(&tensor_id).cloned().unwrap_or_else(|| Box::new([])),
                },
                Node::Cast { x, .. } => ENode::Cast {
                    input: buf_id_from_tensor_id((*x).into()),
                    output: buf_slot,
                },
                Node::Unary { x, uop } => ENode::Unary {
                    input: buf_id_from_tensor_id((*x).into()),
                    output: buf_slot,
                    uop: *uop,
                },
                Node::Binary { x, y, bop } => ENode::Binary {
                    x: buf_id_from_tensor_id((*x).into()),
                    y: buf_id_from_tensor_id((*y).into()),
                    output: buf_slot,
                    bop: *bop,
                },
                Node::Custom(_) => todo!(),
            };

            let inputs_clone: Vec<BufferSlot> = enode.inputs().into_iter().copied().collect();
            let outputs_clone: Vec<BufferSlot> = enode.outputs().into_iter().copied().collect();

            let enode_id = enodes.push(enode);

            for &input in &inputs_clone {
                consumers.entry(input).or_default().push(enode_id);
            }
            for &output in &outputs_clone {
                producers.entry(output).or_default().push(enode_id);
            }
        }

        EGraph { enodes, buffers, producers, consumers }
    }

    fn infer_dtype(node: &crate::graph::Node, nodes: &[crate::graph::Node], _idx: usize) -> DType {
        match node {
            crate::graph::Node::Leaf { dtype } => *dtype,
            crate::graph::Node::Const { value } => value.dtype(),
            crate::graph::Node::Cast { dtype, .. } => *dtype,
            crate::graph::Node::Binary { bop, .. } if bop.returns_bool() => DType::Bool,
            crate::graph::Node::Binary { .. }
            | crate::graph::Node::Unary { .. }
            | crate::graph::Node::Reduce { .. }
            | crate::graph::Node::Expand { .. }
            | crate::graph::Node::Permute { .. }
            | crate::graph::Node::Reshape { .. }
            | crate::graph::Node::Pad { .. } => {
                let input_idx: usize = node.param1().into();
                Self::infer_dtype(&nodes[input_idx], nodes, input_idx)
            }
            crate::graph::Node::Custom(_) => todo!(),
        }
    }

    /// Apply fusion rules to saturate the e-graph with fused kernel alternatives.
    pub fn saturate(&mut self) {
        self.fuse_matmul();
    }

    /// Fuse Reduce(Add) over Binary(Mul, Expand, Expand) into a matmul kernel.
    pub fn fuse_matmul(&mut self) {
        let mut new_enodes: Vec<ENode> = Vec::new();

        for (_enode_id, enode) in self.enodes.iter() {
            let ENode::Reduce { input: red_input, rop: BOp::Add, output, axes: _ } = enode else {
                continue;
            };

            let red_input = *red_input;
            let Some(producers) = self.producers.get(&red_input) else {
                continue;
            };

            for &bin_id in producers {
                let bin = &self.enodes[bin_id];
                let ENode::Binary { x, y, output: _, bop: BOp::Mul } = bin else {
                    continue;
                };

                let x_input = *x;
                let y_input = *y;

                let x_is_expand = self.producers.get(&x_input).is_some_and(|prods| {
                    prods.iter().any(|&id| matches!(self.enodes[id], ENode::Expand { .. }))
                });
                let y_is_expand = self.producers.get(&y_input).is_some_and(|prods| {
                    prods.iter().any(|&id| matches!(self.enodes[id], ENode::Expand { .. }))
                });

                if !x_is_expand || !y_is_expand {
                    continue;
                }

                let op = Box::new(MatMulOp {});
                let cost = op.cost();
                let fused = ENode::Fused {
                    inputs: vec![x_input, y_input],
                    outputs: vec![*output],
                    cost,
                    op,
                };
                new_enodes.push(fused);
            }
        }

        for enode in new_enodes {
            let outputs_clone: Vec<BufferSlot> = enode.outputs().into_iter().copied().collect();
            let enode_id = self.enodes.push(enode);
            let new_enode = &self.enodes[enode_id];
            for input in new_enode.inputs() {
                self.consumers.entry(*input).or_default().push(enode_id);
            }
            for &output in &outputs_clone {
                self.producers.entry(output).or_default().push(enode_id);
            }
        }
    }

    /// Extract the cheapest combination of e-nodes and produce a CompiledGraph.
    pub fn extract(self) -> CompiledGraph {
        let mut cumulative_costs: Map<BufferSlot, u64> = Map::default();
        let mut claimed: crate::Set<BufferSlot> = crate::Set::default();
        let mut selected: Map<BufferSlot, ENodeId> = Map::default();

        let slots: Vec<BufferSlot> = self.buffers.ids().collect();

        for slot in slots {
            if claimed.contains(&slot) {
                continue;
            }

            let Some(producer_ids) = self.producers.get(&slot) else {
                continue;
            };

            let mut best_id = None;
            let mut best_cumulative = u64::MAX;

            for &enode_id in producer_ids {
                let enode = &self.enodes[enode_id];
                let own_cost = match enode {
                    ENode::Copy { cost, .. } | ENode::Fused { cost, .. } => *cost,
                    _ => 0,
                };

                let inputs_cost: u64 = enode.inputs().into_iter().map(|s| *cumulative_costs.get(s).unwrap_or(&0)).sum();
                let cumulative = own_cost + inputs_cost;

                if cumulative < best_cumulative {
                    best_cumulative = cumulative;
                    best_id = Some(enode_id);
                }
            }

            let Some(enode_id) = best_id else {
                continue;
            };

            let enode = &self.enodes[enode_id];
            for output in enode.outputs() {
                claimed.insert(*output);
                selected.insert(*output, enode_id);
                cumulative_costs.insert(*output, best_cumulative);
            }
        }

        // TODO: Validate schedule (check deps, memory limits)
        // TODO: Convert selected plan to CompiledGraph

        todo!()
    }
}
