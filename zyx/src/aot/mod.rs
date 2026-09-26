// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

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

#![allow(unused)]

use crate::{
    graph::Graph,
    kernel::{BOp, MoveOp, Op, OpId},
    shape::Dim,
};

mod matmul;

impl Graph {
    /// Resolves a class's symbolic shape to numeric dims, or `None` if any
    /// dim is not a constant.
    fn const_shape(&self, cid: OpId) -> Option<Vec<Dim>> {
        self.shape(cid)
            .into_iter()
            .map(|dim| match &self.ops[dim].op {
                Op::Const(c) => c.as_dim(),
                _ => None,
            })
            .collect()
    }

    /// Finds a `ReduceLast(Add)` over the trailing axis of a 3D product.
    /// Returns the product class and the contraction dim `k`.
    fn reduce_add_last(&self, cid: OpId) -> Option<(OpId, Dim)> {
        self.class_nodes(cid).find_map(|nid| match &self.ops[nid].op {
            Op::Reduce { x, rop: BOp::Add, .. } => {
                let prod_shape = self.const_shape(*x)?;
                if prod_shape.len() == 3 {
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
    fn mul_of(&self, cid: OpId) -> Option<(OpId, OpId)> {
        if let Some((x, y)) = self.class_nodes(cid).find_map(|nid| match &self.ops[nid].op {
            Op::Binary { x, y, bop: BOp::Mul } => Some((*x, *y)),
            _ => None,
        }) {
            return Some((x, y));
        }
        let x = self.class_nodes(cid).find_map(|nid| match &self.ops[nid].op {
            Op::Cast { x, .. } => Some(*x),
            _ => None,
        })?;
        self.class_nodes(x).find_map(|nid| match &self.ops[nid].op {
            Op::Binary { x, y, bop: BOp::Mul } => Some((*x, *y)),
            _ => None,
        })
    }

    /// Finds the source of an `Expand` and its class shape.
    ///
    /// The shape is read from the source class itself, so matching does not
    /// depend on how the shape was produced (e.g. a `Reshape` in the canonical
    /// matmul form, or an eager tensor already at the broadcast shape).
    fn expand_src(&self, cid: OpId) -> Option<(OpId, Vec<Dim>)> {
        let x = self.class_nodes(cid).find_map(|nid| match &self.ops[nid].op {
            Op::Move { x, mop } => match mop.as_ref() {
                MoveOp::Expand { .. } => Some(*x),
                _ => None,
            },
            _ => None,
        })?;
        Some((x, self.const_shape(x)?))
    }

    /// Finds the source of a 2D `Permute [1, 0]` (a `[n, k]` transposed from
    /// `[k, n]`), looking through shape-only wrappers such as the `Reshape` to
    /// `[1, n, k]` in the broadcast matmul form.
    fn transpose_src(&self, cid: OpId) -> Option<OpId> {
        self.class_nodes(cid).find_map(|nid| match &self.ops[nid].op {
            Op::Move { x, mop } => match mop.as_ref() {
                MoveOp::Reshape { .. } => self.transpose_src(*x),
                MoveOp::Permute { axes } if axes.len() == 2 && axes[0] == 1 && axes[1] == 0 => Some(*x),
                _ => None,
            },
            _ => None,
        })
    }
}
