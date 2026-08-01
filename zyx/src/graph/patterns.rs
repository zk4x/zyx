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

/// A matmul subgraph matched in the graph: `out = a @ b`, where `a` is `[m, k]`,
/// `b` is `[k, n]` and `out` is `[m, n]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct MatMul {
    pub(crate) a: ClassId,
    pub(crate) b: ClassId,
    pub(crate) out: ClassId,
    pub(crate) m: Dim,
    pub(crate) n: Dim,
    pub(crate) k: Dim,
}

impl Graph {
    /// Attempts to match the canonical matmul subgraph ending at class `cid`.
    ///
    /// `dot` lowers a matmul to the canonical broadcast-reduce form:
    ///
    /// ```text
    /// a [m, k] ──────Reshape [m, 1, k]──────▶ Expand ─┐
    /// b [k, n] ─Permute [n, k]─Reshape [1, n, k]─▶ Expand ──▶ Mul [m, n, k] ─Cast─▶ Reduce(Add, last) ─▶ [m, n]
    /// ```
    ///
    /// The matched operands are the original `a` and `b` classes (the reshape
    /// input and the transpose source respectively), so a backend kernel can
    /// consume them directly without any transposition.
    pub(crate) fn match_matmul_class(&self, cid: ClassId, shapes: &Slab<ShapeId, Vec<Dim>>) -> Option<MatMul> {
        let out_shape = &shapes[self.classes[cid].shape];
        if out_shape.len() != 2 {
            return None;
        }
        let [m, n] = [out_shape[0], out_shape[1]];

        let (prod_cid, k) = self.reduce_add_last(cid, shapes)?;
        let (ea, eb) = self.mul_of(prod_cid)?;

        let (a, a3) = self.reshape_operand(ea, shapes)?;
        let (bt, b3) = self.reshape_operand(eb, shapes)?;

        if a3 != [m, 1, k] || b3 != [1, n, k] {
            return None;
        }

        let b = self.transpose_src(bt)?;
        if shapes[self.classes[a].shape] != [m, k] || shapes[self.classes[b].shape] != [k, n] {
            return None;
        }

        Some(MatMul { a, b, out: cid, m, n, k })
    }

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
    fn reshape_operand(&self, cid: ClassId, shapes: &Slab<ShapeId, Vec<Dim>>) -> Option<(ClassId, Vec<Dim>)> {
        let r = self.classes[cid].nodes.iter().find_map(|&nid| match &self.nodes[nid].node {
            Node::Expand { x, .. } => Some(*x),
            _ => None,
        })?;
        self.classes[r].nodes.iter().find_map(|&nid| match &self.nodes[nid].node {
            Node::Reshape { x, shape } => Some((*x, shapes[*shape].clone())),
            _ => None,
        })
    }

    /// Finds the source of a 2D `Permute [1, 0]` (a `[n, k]` transposed from `[k, n]`).
    fn transpose_src(&self, cid: ClassId) -> Option<ClassId> {
        self.classes[cid].nodes.iter().find_map(|&nid| match &self.nodes[nid].node {
            Node::Permute { x, axes } if axes.len() == 2 && axes[0] == 1 && axes[1] == 0 => Some(*x),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DType,
        graph::{EClass, Node, NodeData},
        slab::SlabId,
    };
    use std::collections::BTreeSet;

    fn class(graph: &mut Graph, shapes: &mut Slab<ShapeId, Vec<Dim>>, node: Node, shape: Vec<Dim>) -> ClassId {
        let sid = shapes.push(shape);
        let nid = graph.nodes.push(NodeData { node, class_of: ClassId::NULL });
        let cid = graph.classes.push(EClass { nodes: vec![nid], shape: sid, dtype: DType::F32 });
        graph.nodes[nid].class_of = cid;
        cid
    }

    fn leaf(graph: &mut Graph, shapes: &mut Slab<ShapeId, Vec<Dim>>, shape: Vec<Dim>) -> ClassId {
        let cid = class(graph, shapes, Node::Leaf { dtype: DType::F32, leaf_id: graph.max_leaf_id }, shape);
        graph.max_leaf_id += 1;
        cid
    }

    /// Builds the exact graph `dot` produces for `a [m, k] @ b [k, n]`.
    fn matmul_graph(m: Dim, n: Dim, k: Dim) -> (Graph, Slab<ShapeId, Vec<Dim>>, BTreeSet<ClassId>) {
        let mut graph = Graph::new();
        let mut shapes = Slab::new();
        let a = leaf(&mut graph, &mut shapes, vec![m, k]);
        let b = leaf(&mut graph, &mut shapes, vec![k, n]);
        let s_m1k = shapes.push(vec![m, 1, k]);
        let s_1nk = shapes.push(vec![1, n, k]);
        let s_mnk = shapes.push(vec![m, n, k]);
        let ra = class(&mut graph, &mut shapes, Node::Reshape { x: a, shape: s_m1k }, vec![m, 1, k]);
        let bt = class(&mut graph, &mut shapes, Node::Permute { x: b, axes: vec![1, 0].into_boxed_slice() }, vec![n, k]);
        let rb = class(&mut graph, &mut shapes, Node::Reshape { x: bt, shape: s_1nk }, vec![1, n, k]);
        let ea = class(&mut graph, &mut shapes, Node::Expand { x: ra, shape: s_mnk }, vec![m, n, k]);
        let eb = class(&mut graph, &mut shapes, Node::Expand { x: rb, shape: s_mnk }, vec![m, n, k]);
        let mul = class(&mut graph, &mut shapes, Node::Binary { x: ea, y: eb, bop: BOp::Mul }, vec![m, n, k]);
        let cast = class(&mut graph, &mut shapes, Node::Cast { x: mul, dtype: DType::F32 }, vec![m, n, k]);
        let out = class(&mut graph, &mut shapes, Node::Reduce { x: cast, bop: BOp::Add, axes: vec![2].into_boxed_slice() }, vec![m, n]);
        let outputs: BTreeSet<ClassId> = [out].into();
        (graph, shapes, outputs)
    }

    #[test]
    fn matches_matmul() {
        let (graph, shapes, outputs) = matmul_graph(2, 3, 4);
        let out = outputs.iter().next().copied().unwrap();
        let mm = graph.match_matmul_class(out, &shapes).unwrap();
        assert_eq!(mm.m, 2);
        assert_eq!(mm.n, 3);
        assert_eq!(mm.k, 4);
    }

    #[test]
    fn does_not_match_reduce_over_first_axis() {
        let (mut graph, mut shapes, _) = matmul_graph(2, 3, 4);
        let out = class(&mut graph, &mut shapes, Node::Reduce { x: ClassId::from(6), bop: BOp::Add, axes: vec![0].into_boxed_slice() }, vec![3, 4]);
        assert!(graph.match_matmul_class(out, &shapes).is_none());
    }
}
