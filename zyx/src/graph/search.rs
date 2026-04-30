// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

#![allow(unused)]

//! E-graph based search for fusion strategies.
//!
//! The e-graph stores multiple equivalent implementations for each buffer.
//! Each buffer can be produced by the original operation, a fused kernel,
//! or a copy from another pool. Saturation adds fused variants without
//! replacing existing ones.

use crate::DType;
use crate::Map;
use crate::ZyxError;
use crate::backend::PoolId;
use crate::dtype::Constant;
use crate::graph::Node;
use crate::graph::compiled::{BufferSlot, CachedGraph, CompiledGraph};
use crate::kernel::{BOp, Kernel, MoveOp, Op, OpId, OpNode, UOp};
use crate::shape::{Dim, UAxis};
use crate::slab::{Slab, SlabId};
use crate::tensor::TensorId;
use crate::view::View;

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
    Leaf {
        output: BufferSlot,
    },
    Const {
        output: BufferSlot,
        value: Constant,
    },
    Expand {
        input: BufferSlot,
        output: BufferSlot,
    },
    Permute {
        input: BufferSlot,
        output: BufferSlot,
        axes: Box<[UAxis]>,
    },
    Reshape {
        input: BufferSlot,
        output: BufferSlot,
    },
    Pad {
        input: BufferSlot,
        output: BufferSlot,
        padding: Box<[(i64, i64)]>,
    },
    Reduce {
        input: BufferSlot,
        output: BufferSlot,
        rop: BOp,
        axes: Box<[UAxis]>,
    },
    Cast {
        input: BufferSlot,
        output: BufferSlot,
    },
    Unary {
        input: BufferSlot,
        output: BufferSlot,
        uop: UOp,
    },
    Binary {
        x: BufferSlot,
        y: BufferSlot,
        output: BufferSlot,
        bop: BOp,
    },
    Copy {
        src: BufferSlot,
        dst: BufferSlot,
        cost: u64,
    },
    Fused {
        inputs: Vec<BufferSlot>,
        outputs: Vec<BufferSlot>,
        cost: u64,
        covered: Vec<ENodeId>,
        op: Box<dyn FusedOp>,
    },
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

/// Zyx fused kernel — wraps a full `Kernel` IR.
/// (Note: Kernel uses TensorId internally; it will later be switched to BufferSlot.)
pub struct ZyxOp {
    kernel: Kernel,
    covered: Vec<ENodeId>,
}

impl FusedOp for ZyxOp {
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
    /// Reference counts: how many consumers does each buffer have?
    pub rcs: Map<BufferSlot, u32>,
}

/// Active zyx kernel during zyx_fuse.
#[derive(Clone)]
struct ActiveZyx {
    kernel: Kernel,
    visited: Map<BufferSlot, OpId>,
    covered: Vec<ENodeId>,
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
                Node::Expand { x } => ENode::Expand { input: buf_id_from_tensor_id((*x).into()), output: buf_slot },
                Node::Permute { x } => ENode::Permute {
                    input: buf_id_from_tensor_id((*x).into()),
                    output: buf_slot,
                    axes: graph.axes.get(&tensor_id).cloned().unwrap_or_else(|| Box::new([])),
                },
                Node::Reshape { x } => ENode::Reshape { input: buf_id_from_tensor_id((*x).into()), output: buf_slot },
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
                Node::Cast { x, .. } => ENode::Cast { input: buf_id_from_tensor_id((*x).into()), output: buf_slot },
                Node::Unary { x, uop } => ENode::Unary { input: buf_id_from_tensor_id((*x).into()), output: buf_slot, uop: *uop },
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

        // Compute reference counts from consumers map
        let mut rcs: Map<BufferSlot, u32> = Map::default();
        for (buf_slot, consumer_ids) in &consumers {
            rcs.insert(*buf_slot, consumer_ids.len() as u32);
        }
        // Ensure all buffers have an entry (even if 0 consumers — e.g., graph output)
        for buf_id in buffers.ids() {
            rcs.entry(buf_id).or_insert(1);
        }

        EGraph { enodes, buffers, producers, consumers, rcs }
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
        self.zyx_fuse();
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

                let x_is_expand = self
                    .producers
                    .get(&x_input)
                    .is_some_and(|prods| prods.iter().any(|&id| matches!(self.enodes[id], ENode::Expand { .. })));
                let y_is_expand = self
                    .producers
                    .get(&y_input)
                    .is_some_and(|prods| prods.iter().any(|&id| matches!(self.enodes[id], ENode::Expand { .. })));

                if !x_is_expand || !y_is_expand {
                    continue;
                }

                let op = Box::new(MatMulOp {});
                let cost = op.cost();
                // Collect covered ENodeIds: the reduce, binary, and both expands
                let mut covered = Vec::new();
                // Find the reduce enode id
                if let Some(prod_ids) = self.producers.get(&red_input) {
                    for &pid in prod_ids {
                        if matches!(self.enodes[pid], ENode::Reduce { .. }) {
                            covered.push(pid);
                        }
                    }
                }
                // Find the binary enode id
                covered.push(bin_id);
                // Find expand enode ids
                if let Some(x_prods) = self.producers.get(&x_input) {
                    for &pid in x_prods {
                        if matches!(self.enodes[pid], ENode::Expand { .. }) {
                            covered.push(pid);
                        }
                    }
                }
                if let Some(y_prods) = self.producers.get(&y_input) {
                    for &pid in y_prods {
                        if matches!(self.enodes[pid], ENode::Expand { .. }) {
                            covered.push(pid);
                        }
                    }
                }

                let fused = ENode::Fused { inputs: vec![x_input, y_input], outputs: vec![*output], cost, covered, op };
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

    /// Fill gaps left by specialized fusion kernels (oneDNN, cuDNN, ROCm, etc.).
    ///
    /// Walks naive ENodes in topological order, accumulating ops into zyx kernels
    /// following the same fusion logic as kernelize.rs. All kernelizer logic is
    /// copied here and adapted to work with BufferSlot/ENodeId instead of TensorId.
    /// (Note: Kernel will later be switched to use BufferSlot instead of TensorId.)
    ///
    /// Algorithm:
    /// 1. Walk naive ENodes in order, fusing into active zyx kernels.
    /// 2. When a buffer that is an INPUT to an existing Fused node is encountered:
    ///    - Clone each active zyx kernel
    ///    - One copy registers as ENode::Fused (gap kernel for that fusion's inputs)
    ///    - The other copy continues accumulating
    /// 3. When a buffer that is an OUTPUT of an existing Fused node is encountered:
    ///    - Start a new zyx kernel with just LoadView
    /// 4. When zyx would need to store (contains_stores, is_preceded_by_reduce,
    ///    multiple outputs), clone the kernel and call store on one so that
    ///    its outputs can be used as inputs to the fused kernel — exactly as
    ///    kernelize.rs does.
    ///
    /// This keeps the number of generated kernels low while ensuring all nodes
    /// get at least one producer. No deletion is needed — zyx-only graph is
    /// part of the options that extract can pick.
    pub fn zyx_fuse(&mut self) {
        // Build maps: BufferSlot → Vec<ENodeId> for fused inputs and outputs
        let mut fused_inputs: Map<BufferSlot, Vec<ENodeId>> = Map::default();
        let mut fused_outputs: Map<BufferSlot, Vec<ENodeId>> = Map::default();
        let mut fused_enode_inputs: Map<ENodeId, Vec<BufferSlot>> = Map::default();
        let mut fused_enode_outputs: Map<ENodeId, Vec<BufferSlot>> = Map::default();
        for (enode_id, enode) in self.enodes.iter() {
            let ENode::Fused { .. } = enode else {
                continue;
            };
            let inputs: Vec<BufferSlot> = enode.inputs().into_iter().copied().collect();
            let outputs: Vec<BufferSlot> = enode.outputs().into_iter().copied().collect();
            for input in &inputs {
                fused_inputs.entry(*input).or_default().push(enode_id);
            }
            for output in &outputs {
                fused_outputs.entry(*output).or_default().push(enode_id);
            }
            fused_enode_inputs.insert(enode_id, inputs);
            fused_enode_outputs.insert(enode_id, outputs);
        }

        // Active zyx kernels
        let mut active_kernels: Vec<ActiveZyx> = Vec::new();

        // Collected fused zyx nodes to register at the end
        let mut pending_zyx_nodes: Vec<ENode> = Vec::new();

        // Walk naive ENodes in creation order (CachedGraph order is topo-sorted)
        // Fused nodes are at the end — skip them
        let naive_count = self
            .enodes
            .iter()
            .take_while(|(_, e)| !matches!(e, ENode::Fused { .. }))
            .count();

        for (_enode_id, enode) in self.enodes.iter().take(naive_count) {
            let enode_id = _enode_id;
            let inputs: Vec<BufferSlot> = enode.inputs().into_iter().copied().collect();
            let outputs: Vec<BufferSlot> = enode.outputs().into_iter().copied().collect();

            // For each input buffer: check if it's an input to an existing Fused node
            for input_buf in &inputs {
                if let Some(_fused_ids) = fused_inputs.get(input_buf) {
                    let to_clone: Vec<ActiveZyx> = active_kernels.iter().cloned().collect();
                    for clone in to_clone {
                        let mut c = clone;
                        c.register_as_fused(&mut pending_zyx_nodes);
                    }
                }
            }

            // For each output buffer: check if it's an output of an existing Fused node
            for output_buf in &outputs {
                if let Some(_fused_ids) = fused_outputs.get(output_buf) {
                    // Start a new zyx kernel with just LoadView for this buffer
                    let shape = &self.buffers[*output_buf].shape;
                    let dtype = self.buffers[*output_buf].dtype;
                    let mut new_zk = ActiveZyx::new_load(*output_buf, shape, dtype);
                    new_zk.covered.push(enode_id);
                    active_kernels.push(new_zk);
                }
            }

            // Apply op to all active kernels
            if active_kernels.is_empty() {
                // Create first kernel(s) for leaf/const nodes
                match enode {
                    ENode::Leaf { output } => {
                        let shape = &self.buffers[*output].shape;
                        let dtype = self.buffers[*output].dtype;
                        let zk = ActiveZyx::new_load(*output, shape, dtype);
                        active_kernels.push(zk);
                    }
                    ENode::Const { output, value } => {
                        let shape = &self.buffers[*output].shape;
                        let zk = ActiveZyx::new_const(*output, shape, *value);
                        active_kernels.push(zk);
                    }
                    _ => {
                        // First non-leaf node — create a load kernel for its input(s)
                        for input_buf in &inputs {
                            let has_load = active_kernels.iter().any(|zk| zk.visited.contains_key(input_buf));
                            if !has_load {
                                let shape = &self.buffers[*input_buf].shape;
                                let dtype = self.buffers[*input_buf].dtype;
                                let zk = ActiveZyx::new_load(*input_buf, shape, dtype);
                                active_kernels.push(zk);
                            }
                        }
                    }
                }
            }

            // Apply the actual op
            match enode {
                ENode::Leaf { output } => {
                    for zk in &mut active_kernels {
                        if !zk.visited.contains_key(output) {
                            // Already has a load for this buffer
                        }
                    }
                }
                ENode::Const { output, value } => {
                    for zk in &mut active_kernels {
                        if !zk.visited.contains_key(output) {
                            // Create const kernel for this buffer
                            let shape = &self.buffers[*output].shape;
                            let mut ops = Slab::with_capacity(100);
                            let op = Op::ConstView(Box::new((*value, View::contiguous(shape))));
                            let op_id = ops.push(OpNode { prev: OpId::NULL, next: OpId::NULL, op });
                            zk.kernel = Kernel {
                                outputs: vec![ActiveZyx::tid(*output)],
                                loads: Vec::new(),
                                stores: Vec::new(),
                                ops,
                                head: op_id,
                                tail: op_id,
                            };
                            zk.visited.clear();
                            zk.visited.insert(*output, op_id);
                            zk.covered.push(enode_id);
                        }
                    }
                }
                ENode::Expand { input, output } => {
                    let shape = &self.buffers[*output].shape;
                    let dtype = self.buffers[*input].dtype;
                    for zk in &mut active_kernels {
                        zk.add_expand_op(enode_id, *input, *output, shape, dtype);
                    }
                }
                ENode::Permute { input, output, axes } => {
                    let shape = &self.buffers[*output].shape;
                    for zk in &mut active_kernels {
                        zk.add_permute_op(enode_id, *input, *output, axes, shape);
                    }
                }
                ENode::Reshape { input, output } => {
                    let shape = &self.buffers[*output].shape;
                    for zk in &mut active_kernels {
                        zk.add_reshape_op(enode_id, *input, *output, shape);
                    }
                }
                ENode::Pad { input, output, padding } => {
                    let shape = &self.buffers[*output].shape;
                    for zk in &mut active_kernels {
                        zk.add_pad_op(enode_id, *input, *output, padding, shape);
                    }
                }
                ENode::Reduce { input, output, rop, axes } => {
                    let shape = &self.buffers[*input].shape;
                    let dtype = self.buffers[*input].dtype;
                    for zk in &mut active_kernels {
                        zk.add_reduce_op(enode_id, *input, *output, *rop, axes, shape, dtype);
                    }
                }
                ENode::Cast { input, output } => {
                    let dtype = self.buffers[*output].dtype;
                    for zk in &mut active_kernels {
                        zk.add_cast_op(enode_id, *input, *output, dtype);
                    }
                }
                ENode::Unary { input, output, uop } => {
                    for zk in &mut active_kernels {
                        zk.add_unary_op(enode_id, *input, *output, *uop);
                    }
                }
                ENode::Binary { x, y, output, bop } => {
                    for zk in &mut active_kernels {
                        let _ = zk.add_binary_op(enode_id, *x, *y, *output, *bop);
                    }
                }
                _ => {}
            }
        }

        // Register remaining active kernels as ENode::Fused
        for zk in &mut active_kernels {
            zk.register_as_fused(&mut pending_zyx_nodes);
        }

        // Add all pending zyx nodes to the e-graph
        for enode in pending_zyx_nodes {
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

                let inputs_cost: u64 = enode
                    .inputs()
                    .into_iter()
                    .map(|s| *cumulative_costs.get(s).unwrap_or(&0))
                    .sum();
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

// ---- ActiveZyx implementation ----

fn empty_kernel() -> Kernel {
    let mut ops = Slab::with_capacity(1);
    let op_id = ops.push(OpNode { prev: OpId::NULL, next: OpId::NULL, op: Op::Const(Constant::I32(0)) });
    Kernel { outputs: Vec::new(), loads: Vec::new(), stores: Vec::new(), ops, head: op_id, tail: op_id }
}

impl ActiveZyx {
    fn tid(buf: BufferSlot) -> TensorId {
        TensorId::from(buf.0 as usize)
    }

    fn new_load(buf: BufferSlot, shape: &[Dim], dtype: DType) -> Self {
        let mut ops = Slab::with_capacity(100);
        let op = Op::LoadView(Box::new((dtype, View::contiguous(shape))));
        let op_id = ops.push(OpNode { prev: OpId::NULL, next: OpId::NULL, op });
        let kernel = Kernel {
            outputs: vec![Self::tid(buf)],
            loads: vec![Self::tid(buf)],
            stores: Vec::new(),
            ops,
            head: op_id,
            tail: op_id,
        };
        let mut visited: Map<BufferSlot, OpId> = Map::default();
        visited.insert(buf, op_id);
        Self { kernel, visited, covered: Vec::new() }
    }

    fn new_const(buf: BufferSlot, shape: &[Dim], value: Constant) -> Self {
        let mut ops = Slab::with_capacity(100);
        let op = Op::ConstView(Box::new((value, View::contiguous(shape))));
        let op_id = ops.push(OpNode { prev: OpId::NULL, next: OpId::NULL, op });
        let kernel =
            Kernel { outputs: vec![Self::tid(buf)], loads: Vec::new(), stores: Vec::new(), ops, head: op_id, tail: op_id };
        let mut visited: Map<BufferSlot, OpId> = Map::default();
        visited.insert(buf, op_id);
        Self { kernel, visited, covered: Vec::new() }
    }

    fn add_store(&mut self, buf: BufferSlot, dtype: DType) {
        let op_id = self.visited[&buf];
        self.kernel.push_back(Op::StoreView { src: op_id, dtype });
        self.kernel.stores.push(Self::tid(buf));
        self.visited.remove(&buf);
    }

    fn create_load_kernel(&mut self, buf: BufferSlot, shape: &[Dim], dtype: DType) -> OpId {
        let mut ops = Slab::with_capacity(100);
        let op = Op::LoadView(Box::new((dtype, View::contiguous(shape))));
        let op_id = ops.push(OpNode { prev: OpId::NULL, next: OpId::NULL, op });
        self.kernel.outputs.push(Self::tid(buf));
        self.kernel.loads.push(Self::tid(buf));
        self.kernel.ops = ops;
        self.kernel.head = op_id;
        self.kernel.tail = op_id;
        self.kernel.stores.clear();
        self.visited.insert(buf, op_id);
        op_id
    }

    fn add_unary_op(&mut self, enode_id: ENodeId, input: BufferSlot, output: BufferSlot, uop: UOp) {
        let Some(&op_id) = self.visited.get(&input) else { return };
        let new_op_id = self.kernel.push_back(Op::Unary { x: op_id, uop });
        self.kernel.remove_first_output(Self::tid(input));
        self.kernel.outputs.push(Self::tid(output));
        self.visited.insert(output, new_op_id);
        self.covered.push(enode_id);
    }

    fn add_binary_op(
        &mut self,
        enode_id: ENodeId,
        x: BufferSlot,
        y: BufferSlot,
        output: BufferSlot,
        bop: BOp,
    ) -> Result<(), ZyxError> {
        let Some(&op_id) = self.visited.get(&x) else { return Ok(()) };
        let Some(&op_idy) = self.visited.get(&y) else { return Ok(()) };
        let new_op_id = self.kernel.push_back(Op::Binary { x: op_id, y: op_idy, bop });
        self.kernel.remove_first_output(Self::tid(x));
        self.kernel.remove_first_output(Self::tid(y));
        self.kernel.outputs.push(Self::tid(output));
        self.visited.insert(output, new_op_id);
        self.covered.push(enode_id);
        Ok(())
    }

    fn add_expand_op(&mut self, enode_id: ENodeId, input: BufferSlot, output: BufferSlot, shape: &[Dim], dtype: DType) {
        let Some(op_id) = self.visited.get(&input).copied() else { return };
        let mut op_id = op_id;
        if self.kernel.contains_stores() || self.kernel.is_preceded_by_reduce(op_id) {
            self.add_store(input, dtype);
            op_id = self.create_load_kernel(input, shape, dtype);
        }
        if self.kernel.outputs.len() > 1 {
            let reduce_dims_big = self.kernel.is_preceded_by_reduce(op_id);
            if reduce_dims_big {
                self.add_store(input, dtype);
                op_id = self.create_load_kernel(input, shape, dtype);
            }
        }
        let new_op_id = self
            .kernel
            .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Expand { shape: shape.into() }) });
        self.kernel.remove_first_output(Self::tid(input));
        self.kernel.outputs.push(Self::tid(output));
        self.visited.insert(output, new_op_id);
        self.covered.push(enode_id);
    }

    fn add_permute_op(&mut self, enode_id: ENodeId, input: BufferSlot, output: BufferSlot, axes: &[UAxis], shape: &[Dim]) {
        let Some(&op_id) = self.visited.get(&input) else { return };
        let new_op_id = self
            .kernel
            .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes: axes.into(), shape: shape.into() }) });
        self.kernel.remove_first_output(Self::tid(input));
        self.kernel.outputs.push(Self::tid(output));
        self.visited.insert(output, new_op_id);
        self.covered.push(enode_id);
    }

    fn add_reshape_op(&mut self, enode_id: ENodeId, input: BufferSlot, output: BufferSlot, shape: &[Dim]) {
        let Some(&op_id) = self.visited.get(&input) else { return };
        let new_op_id = self
            .kernel
            .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Reshape { shape: shape.into() }) });
        self.kernel.remove_first_output(Self::tid(input));
        self.kernel.outputs.push(Self::tid(output));
        self.visited.insert(output, new_op_id);
        self.covered.push(enode_id);
    }

    fn add_pad_op(&mut self, enode_id: ENodeId, input: BufferSlot, output: BufferSlot, padding: &[(i64, i64)], shape: &[Dim]) {
        let Some(&op_id) = self.visited.get(&input) else { return };
        let new_op_id = self
            .kernel
            .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Pad { padding: padding.into(), shape: shape.into() }) });
        self.kernel.remove_first_output(Self::tid(input));
        self.kernel.outputs.push(Self::tid(output));
        self.visited.insert(output, new_op_id);
        self.covered.push(enode_id);
    }

    fn add_reduce_op(
        &mut self,
        enode_id: ENodeId,
        input: BufferSlot,
        output: BufferSlot,
        rop: BOp,
        axes: &[UAxis],
        shape: &[Dim],
        dtype: DType,
    ) {
        let Some(op_id) = self.visited.get(&input).copied() else { return };
        let mut op_id = op_id;
        if self.kernel.contains_stores() {
            self.add_store(input, dtype);
            op_id = self.create_load_kernel(input, shape, dtype);
        }
        if self.kernel.outputs.len() > 1 {
            let reduce_dims_big = self.kernel.is_preceded_by_reduce(op_id);
            if reduce_dims_big {
                self.add_store(input, dtype);
                op_id = self.create_load_kernel(input, shape, dtype);
            }
        }
        // Permute before reduce so that reduce axes are last
        let n = shape.len();
        let mut permute_axes = Vec::with_capacity(n);
        let max_axis = *axes.last().unwrap_or(&0);
        let mut ai = 0;
        for i in 0..=max_axis {
            if axes.get(ai) == Some(&i) {
                ai += 1;
            } else {
                permute_axes.push(i);
            }
        }
        permute_axes.extend(max_axis + 1..n);
        permute_axes.extend_from_slice(axes);
        if !permute_axes.iter().copied().eq(0..permute_axes.len()) {
            let permuted_shape = crate::shape::permute(shape, &permute_axes);
            op_id = self
                .kernel
                .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes: permute_axes, shape: permuted_shape }) });
            self.visited.insert(input, op_id);
        }
        let new_op_id = self.kernel.push_back(Op::Reduce { x: op_id, rop, n_axes: axes.len() });
        self.kernel.remove_first_output(Self::tid(input));
        self.kernel.outputs.push(Self::tid(output));
        // If all dims are reduced
        if shape.len() == axes.len() {
            let _ = self
                .kernel
                .push_back(Op::Move { x: new_op_id, mop: Box::new(MoveOp::Reshape { shape: vec![1] }) });
        }
        self.visited.insert(output, new_op_id);
        self.covered.push(enode_id);
    }

    fn add_cast_op(&mut self, enode_id: ENodeId, input: BufferSlot, output: BufferSlot, dtype: DType) {
        let Some(&op_id) = self.visited.get(&input) else { return };
        let new_op_id = self.kernel.push_back(Op::Cast { x: op_id, dtype });
        self.kernel.remove_first_output(Self::tid(input));
        self.kernel.outputs.push(Self::tid(output));
        self.visited.insert(output, new_op_id);
        self.covered.push(enode_id);
    }

    fn register_as_fused(&mut self, pending: &mut Vec<ENode>) {
        if self.kernel.ops.is_empty() {
            return;
        }
        let inputs: Vec<BufferSlot> = self.kernel.loads.iter().map(|tid| BufferSlot::from(tid.0 as usize)).collect();
        let outputs: Vec<BufferSlot> = self
            .kernel
            .outputs
            .iter()
            .map(|tid| BufferSlot::from(tid.0 as usize))
            .collect();
        if inputs.is_empty() || outputs.is_empty() {
            return;
        }
        let cost = 0; // TODO: compute from kernel
        let covered = std::mem::take(&mut self.covered);
        pending.push(ENode::Fused {
            inputs,
            outputs,
            cost,
            covered,
            op: Box::new(ZyxOp { kernel: std::mem::replace(&mut self.kernel, empty_kernel()), covered: Vec::new() }),
        });
    }
}
