// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Pattern matching of subgraph structures onto specialized kernels.
//!
//! The graph op set is intentionally tiny (Const, Leaf, Expand, Permute,
//! Reshape, Reduce, Cast, Unary, Binary, ToDevice, Kernel). Every tensor
//! expression collapses into this fixed set, which means any computation —
//! however it was written by the user — decomposes into a handful of canonical
//! subgraphs. This module matches those subgraphs so they can be replaced by
//! specialized AOT kernels (cblas, cublas, ...) instead of being fused into
//! generic zyx kernels. The graph measures both alternatives and picks the
//! fastest path through extraction.

use crate::{
    graph::{ClassId, Graph, Node},
    kernel::BOp,
    runtime::ShapeId,
    shape::Dim,
    slab::Slab,
};

mod matmul;

impl Graph {
    /// Finds a `Reduce(Add)` over the single trailing axis of a 3D product.
    /// Returns the product class and the contraction dim `k`.
    fn reduce_add_last(&self, cid: ClassId, shapes: &Slab<ShapeId, Vec<Dim>>) -> Option<(ClassId, Dim)> {
        self.classes[cid].nodes.iter().find_map(|&nid| match &self.nodes[nid].node {
            Node::Reduce { x, bop: BOp::Add, axes } => {
                let prod_shape = &shapes[self.classes[*x].shape];
                if prod_shape.len() == 3 && axes.len() == 1 && axes[0] == prod_shape.len() - 1 {
                    Some((*x, prod_shape[2]))
                } else {
                    None
                }
            }
            _ => None,
        })
    }

    /// Finds the elementwise `Mul` beneath the product class, through an optional
    /// accumulator `Cast` (`dot` casts the product before reducing).
    fn mul_of(&self, cid: ClassId) -> Option<(ClassId, ClassId)> {
        if let Some((x, y)) = self.classes[cid].nodes.iter().find_map(|&nid| match &self.nodes[nid].node {
            Node::Binary { x, y, bop: BOp::Mul } => Some((*x, *y)),
            _ => None,
        }) {
            return Some((x, y));
        }
        let x = self.classes[cid].nodes.iter().find_map(|&nid| match &self.nodes[nid].node {
            Node::Cast { x, .. } => Some(*x),
            _ => None,
        })?;
        self.classes[x].nodes.iter().find_map(|&nid| match &self.nodes[nid].node {
            Node::Binary { x, y, bop: BOp::Mul } => Some((*x, *y)),
            _ => None,
        })
    }

    /// Unwraps `Expand -> Reshape`, returning the reshaped 2D operand class and
    /// the 3D reshape shape.
    ///
    /// The reshape may have been folded into the expand source itself (e.g. an
    /// eager tensor promoted to a graph leaf already at the reshaped shape), in
    /// which case there is no `Reshape` node and the source class is returned
    /// as the operand. The 3D shape is always taken from the expand source's
    /// class shape, so matching does not depend on how the shape was produced.
    fn reshape_operand(&self, cid: ClassId, shapes: &Slab<ShapeId, Vec<Dim>>) -> Option<(ClassId, Vec<Dim>)> {
        let r = self.classes[cid].nodes.iter().find_map(|&nid| match &self.nodes[nid].node {
            Node::Expand { x, .. } => Some(*x),
            _ => None,
        })?;
        let r_shape = shapes[self.classes[r].shape].clone();
        let operand = self.classes[r].nodes.iter().find_map(|&nid| match &self.nodes[nid].node {
            Node::Reshape { x, .. } => Some(*x),
            _ => None,
        });
        Some((operand.unwrap_or(r), r_shape))
    }

    /// Finds the source of a 2D `Permute [1, 0]` (a `[n, k]` transposed from `[k, n]`).
    fn transpose_src(&self, cid: ClassId) -> Option<ClassId> {
        self.classes[cid].nodes.iter().find_map(|&nid| match &self.nodes[nid].node {
            Node::Permute { x, axes } if axes.len() == 2 && axes[0] == 1 && axes[1] == 0 => Some(*x),
            _ => None,
        })
    }
}
