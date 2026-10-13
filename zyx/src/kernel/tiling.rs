// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent tiling pass: canonicalize scalar IR for tile matching.
//! (Under construction, one step at a time.)

use super::{Kernel, Op, OpId, Pat, RangeKind, TTOp, TileDim};
use crate::Set;

impl Kernel {
    /// Tenstorrent tiling pass, first cut: reshape only, no tile ops yet.
    /// Unrolls every counted loop with constant length divisible by 32
    /// into explicit 32-wide bundles, then coarsens every group range by
    /// 32, so per-thread work becomes explicit 32-wide bundles (32x32
    /// contraction units for the tile op pattern matcher). Matching
    /// bundles into tile ops lands in later cuts.
    pub fn tt_tile(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        // Fold first: unroll/coarsen replicate the body, so wrappers
        // (x*1, casts of consts) must die before replication, not after.
        self.constant_folding();
        // Tree-unroll every counted loop with constant length divisible
        // by 4 FIRST: unroll_tree_reduce allows exactly one acc store
        // per body, so it must run before coarsening replicates the
        // body. Non-reduce loops are a no-op inside (no acc found);
        // anything else keeps its shape for later cuts.
        let mut loops: Vec<OpId> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if let Op::Loop { len, .. } = x {
                if self.resolve_const(*len).and_then(|c| c.as_dim()).is_some_and(|l| l % 32 == 0) {
                    loops.push(op);
                }
            }
        }
        for l in loops {
            self.unroll_tree_reduce(l, 32);
        }

        self.default_epilogue();
        self.default_epilogue();

        // Bundle scalar work into vectors (loads, stores, then ops)
        // BEFORE coarsening: coarsen duplicates vectorized bodies per
        // lane, materializing one vector load per lane for the tile
        // former to stack. TT declares supported_vec_lens=[32], so the
        // bundles match exactly.
        let vectorize = self.opt_vectorize();
        vectorize.apply(self, 0);

        self.default_epilogue();
        self.default_epilogue();

        // Coarsen group ranges LAST: duplicate the vectorized body, one
        // lane per row. Anything else (symbolic, indivisible) is
        // skipped, never fatal.
        let mut groups: Vec<OpId> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if let Op::Range { kind: RangeKind::Group(len), .. } = x {
                if self.resolve_const(*len).and_then(|c| c.as_dim()).is_some_and(|l| l % 32 == 0) {
                    groups.push(op);
                }
            }
        }
        for g in groups {
            self.coarsen(g, 32);
        }

        self.default_epilogue();
        self.default_epilogue();

        // Vectorize AGAIN: coarsening duplicated the vectorized body,
        // one vector load per row; the tile former stacks them now.
        let vectorize = self.opt_vectorize();
        vectorize.apply(self, 0);

        self.tt_form_reduce_tiles();

        let vectorize = self.opt_vectorize();
        vectorize.apply(self, 0);

        self.default_epilogue();
        self.default_epilogue();

        self.verify();
    }

    /// Former: tiled source + 32 row reduces -> ReduceTile(Row).
    /// One cover match finds every tile group; only the final mutation
    /// lives here. Scalar casts around the reduce are vectorize_ops'
    /// leftovers and must be absorbed there, never matched here. The
    /// result tile carries the sums in its first row, so every reduce
    /// becomes an Index lane of row 0. Dead chains die in DCE.
    pub fn tt_form_reduce_tiles(&mut self) {
        let pat = Pat::vecreduce(Pat::index(Pat::bind('t'), 'r'), 'o');
        for g in self.match_cover(&pat, &[32]) {
            let t = g.bindings.op('t');
            // Tile-shaped source of any kind; lane coverage below ties
            // the lanes to it.
            if self.shape(t) != vec![32, 32] {
                continue;
            }
            let rop = g.bindings.bop('o');
            let mut rows: [Option<OpId>; 32] = [None; 32];
            for (root, lane) in g.roots.iter().zip(g.lanes.iter()) {
                rows[*lane as usize] = Some(*root);
            }
            debug_assert!(rows.iter().all(|r| r.is_some()), "cover guarantees exact lanes");
            let set: Set<OpId> = g.roots.iter().copied().collect();
            let mut first = OpId::NULL;
            let mut scan = self.head;
            while !scan.is_null() {
                if set.contains(&scan) {
                    first = scan;
                    break;
                }
                scan = self.next_op(scan);
            }
            let rt = self.insert_before(first, Op::TT(TTOp::ReduceTile { x: t, rop, kind: TileDim::Row }));
            let row0 = self.insert_before(first, Op::Index { vec: rt, idx: 0 });
            for (r, outer) in rows.iter().enumerate() {
                self.ops[outer.unwrap()].op = Op::Index { vec: row0, idx: r };
            }
        }
    }
}

#[cfg(feature = "tenstorrent")]
#[cfg(test)]
mod tests {
    use super::Kernel;
    use crate::dtype::DType;
    use crate::kernel::{Dev, Op, RangeKind};

    #[test]
    fn tt_tile_2d() {
        let mut k = Kernel::new(Dev::TT(0));
        let a = k.param(DType::BF16);
        let o = k.param_mut(DType::BF16);
        let rows = k.const_idx(32);
        let cols = k.const_idx(256);
        let gr = k.group_range(0, rows);
        let gc = k.group_range(1, cols);
        let stride = k.mul(gr, cols);
        let idx = k.add(gc, stride);
        let v = k.load(a, idx);
        let n = k.neg(v);
        k.store(o, n, idx);
        k.tt_tile();
        // First cut only canonicalizes: groups /4, traffic still scalar.
        // (coarsen rewrites the group op itself into `gidx * 4`, so find
        // the shortened ranges by axis.)
        let mut lens: [Option<i64>; 2] = [None, None];
        let mut id = k.head;
        while !id.is_null() {
            if let Op::Range { axis, kind: RangeKind::Group(len) } = k.at(id).clone() {
                lens[axis as usize] = k.resolve_const(len).and_then(|c| c.as_dim());
            }
            id = k.next_op(id);
        }
        assert_eq!(lens, [Some(8), Some(64)]);
        assert!(k.shape(v).is_empty());
        assert!(k.shape(n).is_empty());
    }
}
