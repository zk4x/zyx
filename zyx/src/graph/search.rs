// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::dtype::Constant;
use crate::graph::Node;
use crate::kernel::{BOp, UOp};
use crate::slab::Slab;
use crate::tensor::TensorId;
use crate::DType;

#[derive(Debug, Clone)]
pub enum FusedKernel {
    OneDNN,
}

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
    Fused {
        x: TensorId,
        fused_kernel: FusedKernel,
    },
}

pub fn search(cached_graph: &crate::graph::compiled::CachedGraph) {
    let mut egraph: Slab<TensorId, Vec<ENode>> = Slab::new();
    for (id, node) in cached_graph.nodes.iter().enumerate() {
        let enode: ENode = match node {
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

    for (id, enodes) in egraph.iter() {
        println!("{id}: {enodes:?}");
    }
}