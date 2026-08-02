use crate::{
    DType,
    graph::{ClassId, Graph},
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
    /// Accumulate (output) dtype.
    pub(crate) acc_dtype: DType,
    /// Operand buffer dtype, shared by `a` and `b`.
    pub(crate) in_dtype: DType,
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
    /// The matched operands are the original `a` and `b` classes, consumed
    /// directly by the backend kernel without any transposition:
    ///
    /// - `a` is the expand source itself — `[m, 1, k]` is the same row-major
    ///   layout as `[m, k]`, whether the reshape is explicit or was folded into
    ///   a leaf at the broadcast shape.
    /// - `b` is found by looking through the expand source's `Reshape` for the
    ///   `Permute [1, 0]` and returning its `[k, n]` source.
    pub(crate) fn match_matmul(&self, cid: ClassId, shapes: &Slab<ShapeId, Vec<Dim>>) -> Option<MatMul> {
        let out_shape = &shapes[self.classes[cid].shape];
        if out_shape.len() != 2 {
            return None;
        }
        let [m, n] = [out_shape[0], out_shape[1]];

        let (prod, k) = self.reduce_add_last(cid, shapes)?;
        let (ea, eb) = self.mul_of(prod)?;

        let (a, a3) = self.expand_src(ea, shapes)?;
        let (bt, b3) = self.expand_src(eb, shapes)?;

        if a3 != [m, 1, k] || b3 != [1, n, k] {
            return None;
        }

        let b = self.transpose_src(bt)?;
        if shapes[self.classes[b].shape] != [k, n] {
            return None;
        }

        let in_dtype = self.classes[a].dtype;
        if self.classes[b].dtype != in_dtype {
            return None;
        }

        Some(MatMul { a, b, out: cid, m, n, k, acc_dtype: self.classes[cid].dtype, in_dtype })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        DType,
        graph::{EClass, Node, NodeData},
        kernel::BOp,
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
        let out =
            class(&mut graph, &mut shapes, Node::Reduce { x: cast, bop: BOp::Add, axes: vec![2].into_boxed_slice() }, vec![m, n]);
        let outputs: BTreeSet<ClassId> = [out].into();
        (graph, shapes, outputs)
    }

    #[test]
    fn matches_matmul() {
        let (graph, shapes, outputs) = matmul_graph(2, 3, 4);
        let out = outputs.iter().next().copied().unwrap();
        let mm = graph.match_matmul(out, &shapes).unwrap();
        assert_eq!(mm.m, 2);
        assert_eq!(mm.n, 3);
        assert_eq!(mm.k, 4);
    }

    #[test]
    fn does_not_match_reduce_over_first_axis() {
        let (mut graph, mut shapes, _) = matmul_graph(2, 3, 4);
        let out = class(
            &mut graph,
            &mut shapes,
            Node::Reduce { x: ClassId::from(6), bop: BOp::Add, axes: vec![0].into_boxed_slice() },
            vec![3, 4],
        );
        assert!(graph.match_matmul(out, &shapes).is_none());
    }

    /// Matches even when the `a` operand's reshape is folded into the leaf
    /// (e.g. an eager tensor promoted at the reshaped shape) — no `Reshape`
    /// node on the expand source.
    #[test]
    fn matches_matmul_with_folded_a_reshape() {
        let (mut graph, mut shapes, _) = matmul_graph(2, 3, 4);
        // Replace the a-side Reshape class (id 3) with a leaf of the same shape.
        let folded_a = leaf(&mut graph, &mut shapes, vec![2, 1, 4]);
        let m = 2;
        let n = 3;
        let k = 4;
        let s_mnk = shapes.push(vec![m, n, k]);
        let ea = class(&mut graph, &mut shapes, Node::Expand { x: folded_a, shape: s_mnk }, vec![m, n, k]);
        // Rewrite the Mul (id 7) to consume the new Expand instead of the old one.
        let mul = class(&mut graph, &mut shapes, Node::Binary { x: ea, y: ClassId::from(6), bop: BOp::Mul }, vec![m, n, k]);
        let cast = class(&mut graph, &mut shapes, Node::Cast { x: mul, dtype: DType::F32 }, vec![m, n, k]);
        let out =
            class(&mut graph, &mut shapes, Node::Reduce { x: cast, bop: BOp::Add, axes: vec![2].into_boxed_slice() }, vec![m, n]);
        let mm = graph.match_matmul(out, &shapes).unwrap();
        assert_eq!(mm.a, folded_a);
        assert_eq!(mm.m, m);
        assert_eq!(mm.n, n);
        assert_eq!(mm.k, k);
    }
}
