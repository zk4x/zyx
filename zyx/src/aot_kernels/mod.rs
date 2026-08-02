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

    /// Finds the source of an `Expand` and its class shape.
    ///
    /// The shape is read from the source class itself, so matching does not
    /// depend on how the shape was produced (e.g. a `Reshape` in the canonical
    /// matmul form, or an eager tensor already at the broadcast shape).
    fn expand_src(&self, cid: ClassId, shapes: &Slab<ShapeId, Vec<Dim>>) -> Option<(ClassId, Vec<Dim>)> {
        let x = self.classes[cid].nodes.iter().find_map(|&nid| match &self.nodes[nid].node {
            Node::Expand { x, .. } => Some(*x),
            _ => None,
        })?;
        Some((x, shapes[self.classes[x].shape].clone()))
    }

    /// Finds the source of a 2D `Permute [1, 0]` (a `[n, k]` transposed from
    /// `[k, n]`), looking through shape-only wrappers such as the `Reshape` to
    /// `[1, n, k]` in the broadcast matmul form.
    fn transpose_src(&self, cid: ClassId) -> Option<ClassId> {
        self.classes[cid].nodes.iter().find_map(|&nid| match &self.nodes[nid].node {
            Node::Reshape { x, .. } => self.transpose_src(*x),
            Node::Permute { x, axes } if axes.len() == 2 && axes[0] == 1 && axes[1] == 0 => Some(*x),
            _ => None,
        })
    }
}
