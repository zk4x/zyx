// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! E-graph based search for fusion strategies.
//!
//! The e-graph stores multiple equivalent implementations for each node.
//! Each node can have both the original operations and fused kernels.
//! Saturation adds fused variants without replacing existing ones.

use std::collections::BTreeMap;

use crate::dtype::Constant;
use crate::graph::Node;
use crate::kernel::{BOp, UOp};
use crate::shape::{Dim, UAxis};
use crate::slab::Slab;
use crate::tensor::TensorId;
use crate::DType;

#[derive(Debug, Clone)]
pub enum FusedKernel {
    MatMul {
        a: TensorId,
        b: TensorId,
    },
}

#[derive(Debug, Clone)]
pub struct ENode {
    pub cost: u64,
    pub kind: ENodeKind,
}

#[derive(Debug, Clone)]
pub enum ENodeKind {
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
    Fused(FusedKernel),
}

#[derive(Debug, Clone)]
pub struct EGraph {
    pub nodes: Slab<TensorId, Vec<ENode>>,
    pub shapes: BTreeMap<TensorId, Box<[Dim]>>,
    pub paddings: BTreeMap<TensorId, Box<[(i64, i64)]>>,
    pub axes: BTreeMap<TensorId, Box<[UAxis]>>,
}

pub fn one_dnn_fuse(egraph: &mut Slab<TensorId, Vec<ENode>>, shapes: &BTreeMap<TensorId, Box<[Dim]>>, axes: &BTreeMap<TensorId, Box<[UAxis]>>) {
    let mut new_nodes: Vec<(TensorId, ENode)> = Vec::new();
    for (node_id, enodes) in egraph.iter() {
        for enode in enodes {
            let ENodeKind::Reduce { x: red_input, rop: BOp::Add } = &enode.kind else { continue; };
            let bin_enodes = &egraph[*red_input];
            for bin_enode in bin_enodes {
                let ENodeKind::Binary { x: x_input, y: y_input, bop: BOp::Mul } = &bin_enode.kind else { continue; };
                let x_enodes = &egraph[*x_input];
                let y_enodes = &egraph[*y_input];
                let mut x_has_expand = false;
                let mut y_has_expand = false;
                for x_node in x_enodes {
                    let ENodeKind::Expand { .. } = &x_node.kind else { continue };
                    x_has_expand = true;
                }
                for y_node in y_enodes {
                    let ENodeKind::Expand { .. } = &y_node.kind else { continue };
                    y_has_expand = true;
                }
                if x_has_expand && y_has_expand {
                    let mut a_src = TensorId::from(0);
                    let mut b_src = TensorId::from(0);
                    for x_node in x_enodes {
                        if let ENodeKind::Expand { x } = &x_node.kind {
                            a_src = *x;
                        }
                    }
                    for y_node in y_enodes {
                        if let ENodeKind::Expand { x } = &y_node.kind {
                            b_src = *x;
                        }
                    }
                    let new_enode = ENode { cost: u64::MAX, kind: ENodeKind::Fused(FusedKernel::MatMul { a: a_src, b: b_src }) };
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
    for (id, node) in cached_graph.nodes.iter().enumerate() {
        let enode = ENode {
            cost: u64::MAX,
            kind: match node {
                Node::Const { value } => ENodeKind::Const { value: *value },
                Node::Leaf { dtype } => ENodeKind::Leaf { dtype: *dtype },
                Node::Expand { x } => ENodeKind::Expand { x: *x },
                Node::Permute { x } => ENodeKind::Permute { x: *x },
                Node::Reshape { x } => ENodeKind::Reshape { x: *x },
                Node::Pad { x } => ENodeKind::Pad { x: *x },
                Node::Reduce { x, rop } => ENodeKind::Reduce { x: *x, rop: *rop },
                Node::Cast { x, dtype } => ENodeKind::Cast { x: *x, dtype: *dtype },
                Node::Unary { x, uop } => ENodeKind::Unary { x: *x, uop: *uop },
                Node::Binary { x, y, bop } => ENodeKind::Binary { x: *x, y: *y, bop: *bop },
                Node::Custom(_) => todo!(),
            },
        };
        egraph.push(vec![enode]);
    }

    one_dnn_fuse(&mut egraph, &cached_graph.shapes, &cached_graph.axes);

    for (id, enodes) in egraph.iter() {
        println!("{id}: {enodes:?}");
    }
}
