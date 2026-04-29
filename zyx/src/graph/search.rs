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
use crate::kernel::{BOp, UOp};
use crate::shape::{Dim, UAxis};
use crate::slab::Slab;
use crate::tensor::TensorId;

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone)]
pub struct EGraph {
    pub nodes: Slab<TensorId, Vec<ENode>>,
    pub shapes: BTreeMap<TensorId, Box<[Dim]>>,
    pub paddings: BTreeMap<TensorId, Box<[(i64, i64)]>>,
    pub axes: BTreeMap<TensorId, Box<[UAxis]>>,
}

pub fn one_dnn_fuse(
    egraph: &mut Slab<TensorId, Vec<ENode>>,
    shapes: &BTreeMap<TensorId, Box<[Dim]>>,
    axes: &BTreeMap<TensorId, Box<[UAxis]>>,
) {
    let mut new_nodes: Vec<(TensorId, ENode)> = Vec::new();
    for (node_id, enodes) in egraph.iter() {
        for enode in enodes {
            let ENode::Reduce { x: red_input, rop: BOp::Add } = &enode else {
                continue;
            };
            let bin_enodes = &egraph[*red_input];
            for bin_enode in bin_enodes {
                let ENode::Binary { x: x_input, y: y_input, bop: BOp::Mul } = &bin_enode else {
                    continue;
                };
                let x_enodes = &egraph[*x_input];
                let y_enodes = &egraph[*y_input];
                let mut x_has_expand = false;
                let mut y_has_expand = false;
                for x_node in x_enodes {
                    let ENode::Expand { .. } = &x_node else { continue };
                    x_has_expand = true;
                }
                for y_node in y_enodes {
                    let ENode::Expand { .. } = &y_node else { continue };
                    y_has_expand = true;
                }
                if x_has_expand && y_has_expand {
                    let mut a_src = TensorId::from(0);
                    let mut b_src = TensorId::from(0);
                    for x_node in x_enodes {
                        if let ENode::Expand { x } = &x_node {
                            a_src = *x;
                        }
                    }
                    for y_node in y_enodes {
                        if let ENode::Expand { x } = &y_node {
                            b_src = *x;
                        }
                    }
                    let new_enode = ENode::Fused { cost: 0, kernel: FusedKernel::MatMul { a: a_src, b: b_src } };
                    new_nodes.push((node_id, new_enode));
                }
            }
        }
    }
    for (node_id, new_enode) in new_nodes {
        egraph[node_id].push(new_enode);
    }
}

/// Builds  egraph by enumerating all fusion and memory allocation possibilities.
/// Evaluates each fused node time
///
/// Full implementation would include:
/// - Multiple fusion rules (matmul, elementwise chains, reduce chains, etc.)
/// - Cost computation for each variant combination
/// - Path selection to minimize total execution time
pub fn search(cached_graph: &crate::graph::compiled::CachedGraph) {
    let mut egraph: Slab<TensorId, Vec<ENode>> = Slab::new();
    for (nid, node) in cached_graph.nodes.iter().enumerate() {
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
        egraph.push(vec![enode]);
    }

    one_dnn_fuse(&mut egraph, &cached_graph.shapes, &cached_graph.axes);

    for (id, enodes) in egraph.iter() {
        println!("{id}: {enodes:?}");
    }
}
