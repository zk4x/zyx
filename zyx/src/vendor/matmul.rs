#![allow(unused)]

use crate::{
    DType,
    graph::{ClassId, Graph},
    shape::Dim,
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
    pub(crate) fn match_matmul(&self, cid: ClassId) -> Option<MatMul> {
        let out_shape = self.const_shape(cid)?;
        if out_shape.len() != 2 {
            return None;
        }
        let [m, n] = [out_shape[0], out_shape[1]];

        let (prod, k) = self.reduce_add_last(cid)?;
        let (ea, eb) = self.mul_of(prod)?;

        let (a, a3) = self.expand_src(ea)?;
        let (bt, b3) = self.expand_src(eb)?;

        if a3 != [m, 1, k] || b3 != [1, n, k] {
            return None;
        }

        let b = self.transpose_src(bt)?;
        if self.const_shape(b).as_deref() != Some(&[k, n][..]) {
            return None;
        }

        let in_dtype = self.dtype(b);
        if self.dtype(a) != in_dtype {
            return None;
        }

        Some(MatMul { a, b, out: cid, m, n, k, acc_dtype: self.dtype(cid), in_dtype })
    }
}
