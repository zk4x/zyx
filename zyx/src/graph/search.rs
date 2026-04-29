// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! E-graph based search for fusion strategies.
//!
//! The e-graph stores multiple equivalent implementations for each node.
//! Each node can have both the original operations and fused kernels.
//! Saturation adds fused variants without replacing existing ones.

use std::collections::BTreeMap;

use crate::DType;
use crate::dtype::Constant;
use crate::graph::Node;
use crate::graph::compiled::{CachedGraph, CompiledGraph};
use crate::kernel::{BOp, UOp};
use crate::shape::{Dim, UAxis};
use crate::slab::{Slab, SlabId};
use crate::tensor::TensorId;

#[derive(Debug)]
pub enum ENode {
    Const {
        value: Constant,
    },
    Leaf {
        dtype: DType,
    },
    Expand {
        x: TensorId,
    },
    Permute {
        x: TensorId,
    },
    Reshape {
        x: TensorId,
    },
    Pad {
        x: TensorId,
    },
    Reduce {
        x: TensorId,
        rop: BOp,
    },
    Cast {
        x: TensorId,
        dtype: DType,
    },
    Unary {
        x: TensorId,
        uop: UOp,
    },
    Binary {
        x: TensorId,
        y: TensorId,
        bop: BOp,
    },
    /// Fused operation with associated cost (e.g., fused kernel execution time)
    Fused {
        cost: u64,
        kernel: FusedKernel,
    },
}

#[derive(Debug, Clone)]
pub enum FusedKernel {
    MatMul { a: TensorId, b: TensorId },
}

#[derive(Debug)]
pub struct EGraph {
    pub nodes: Slab<TensorId, Vec<ENode>>,
    pub shapes: BTreeMap<TensorId, Box<[Dim]>>,
    pub paddings: BTreeMap<TensorId, Box<[(i64, i64)]>>,
    pub axes: BTreeMap<TensorId, Box<[UAxis]>>,
}

/// Builds  egraph by enumerating all fusion and memory allocation possibilities.
/// Evaluates each fused node time
///
/// Full implementation would include:
/// - Multiple fusion rules (matmul, elementwise chains, reduce chains, etc.)
/// - Cost computation for each variant combination
/// - Path selection to minimize total execution time
impl EGraph {
    pub fn new(graph: &CachedGraph) -> EGraph {
        let mut nodes = Slab::new();
        for (_, node) in graph.nodes.iter().enumerate() {
            let enode = match node {
                Node::Const { value } => ENode::Const { value: *value },
                Node::Leaf { dtype } => ENode::Leaf { dtype: *dtype },
                Node::Expand { x } => ENode::Expand { x: *x },
                Node::Permute { x } => ENode::Permute { x: *x },
                Node::Reshape { x } => ENode::Reshape { x: *x },
                Node::Pad { x } => ENode::Pad { x: *x },
                Node::Reduce { x, rop } => ENode::Reduce { x: *x, rop: *rop },
                Node::Cast { x, dtype } => ENode::Cast { x: *x, dtype: *dtype },
                Node::Unary { x, uop } => ENode::Unary { x: *x, uop: *uop },
                Node::Binary { x, y, bop } => ENode::Binary { x: *x, y: *y, bop: *bop },
                Node::Custom(_) => todo!(),
            };
            nodes.push(vec![enode]);
        }

        for (id, enodes) in nodes.iter() {
            println!("{id}: {enodes:?}");
        }

        EGraph { nodes, shapes: graph.shapes.clone(), paddings: graph.paddings.clone(), axes: graph.axes.clone() }
    }

    pub fn saturate(&mut self) {
        self.one_dnn_fuse();
    }

    pub fn extract(self) -> CompiledGraph {
        todo!()
    }

    pub fn one_dnn_fuse(&mut self) {
        let mut new_nodes: Vec<(TensorId, ENode)> = Vec::new();
        for (node_id, enodes) in self.nodes.iter() {
            for enode in enodes {
                let ENode::Reduce { x: red_input, rop: BOp::Add } = &enode else {
                    continue;
                };
                let bin_enodes = &self.nodes[*red_input];
                for bin_enode in bin_enodes {
                    let ENode::Binary { x: x_input, y: y_input, bop: BOp::Mul } = &bin_enode else {
                        continue;
                    };
                    let mut a = TensorId::NULL;
                    let mut b = TensorId::NULL;
                    for x_node in &self.nodes[*x_input] {
                        if let ENode::Expand { x } = &x_node {
                            a = *x;
                        }
                    }
                    for y_node in &self.nodes[*y_input] {
                        if let ENode::Expand { x } = &y_node {
                            b = *x;
                        }
                    }
                    if !a.is_null() && !b.is_null() {
                        let new_enode = ENode::Fused { cost: 0, kernel: FusedKernel::MatMul { a, b } };
                        new_nodes.push((node_id, new_enode));
                    }
                }
            }
        }
        for (node_id, new_enode) in new_nodes {
            self.nodes[node_id].push(new_enode);
        }
    }
}
