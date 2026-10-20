// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent tiling pass: canonicalize scalar IR for tile matching.
//! (Under construction, one step at a time.)

use super::{BOp, Kernel, Op, OpId, Pat, RangeKind, TTOp, split::SplitMode};
use crate::Map;
use crate::Set;
use crate::dtype::Constant;
use crate::shape::Dim;

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
        self.default_epilogue();
        self.default_epilogue();

        // Fold first: split replicates the body, so wrappers
        // (x*1, casts of consts) must die before replication, not after.
        self.constant_folding();
        // Tree-unroll every counted loop with constant length divisible
        // by 32 FIRST: split's loop arm allows exactly one acc store
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
            self.split(l, SplitMode::Unroll, 32);
        }

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
            self.split(g, SplitMode::Unroll, 32);
        }

        self.default_epilogue();
        self.default_epilogue();

        // Retile the reduce row (see below) on the final IR.
        self.tilize_stores();
        self.debug();

        // Fold dot-cone families into matmul tiles (see below).
        self.vectorize_matmul_tile(32, 32, 32);

        self.default_epilogue();
        self.default_epilogue();

        self.debug();
        todo!();

        self.verify();
    }

    /// Former: tiled source + 32 row reduces -> ReduceTile(Row).
    /// One cover match finds every tile group; only the final mutation
    /// lives here. Scalar casts around the reduce are vectorize_ops'
    /// leftovers and must be absorbed there, never matched here. The
    /// result tile carries the sums in its first row, so every reduce
    /// becomes an Index lane of row 0. Dead chains die in DCE.
    pub fn vectorize_reduce_tile(&mut self) {
        // The matched lane folds rode the deleted vectorize IR; with
        // the tile matcher absent this lowering never fires.
        todo!("tiling: vectorize-based bundling is deleted");
    }

    /// Backward tile-push over vector stores (tile equivalent of
    /// `vectorize_stores`/`vectorize_ops_backward`, which are fully
    /// general over stored values and op heads — this is too, no
    /// enumerated op list). Anchored at each vector-length `Store` via
    /// `Pat`, then structural descent through whatever heads appear:
    /// single-input elementwise and tile-native heads retile their
    /// input, binaries retile both sides, and the walk terminates at
    /// leaves: a tile-typed value is kept, an `Index` of a tile becomes
    /// the tile itself (the extract dies in DCE), a scalar const becomes
    /// an `Expand` to the tile dims (lowered later; render rejects it
    /// loudly until then). Anything else stops the tree — skipped,
    /// never fatal. The store bytes become the tile dims (the packer
    /// masks the non-result lanes, so the whole tile drains).
    pub fn tilize_stores(&mut self) {
        #[derive(Clone, Copy)]
        enum Target {
            Direct(OpId),
            Expanded(OpId),
        }
        struct Rewire {
            user: OpId,
            input: OpId,
            target: Target,
        }
        struct Job {
            store: OpId,
            rewires: Vec<Rewire>,
            scalars: Vec<OpId>,
            dims: [u16; 2],
        }
        fn tile_dims(k: &Kernel, v: OpId) -> Option<[u16; 2]> {
            let dims = k.shape(v);
            if dims.len() != 2 || dims.iter().any(|&d| d <= 0 || d > 65535) {
                return None;
            }
            Some([dims[0] as u16, dims[1] as u16])
        }
        fn descend(
            k: &Kernel,
            v: OpId,
            memo: &mut Map<OpId, Target>,
            active: &mut Set<OpId>,
            rewires: &mut Vec<Rewire>,
            scalars: &mut Vec<OpId>,
            dims: &mut Option<[u16; 2]>,
        ) -> Option<Target> {
            if let Some(t) = memo.get(&v) {
                return Some(*t);
            }
            if !active.insert(v) {
                return None;
            }
            // Tile-typed values (and the reduce tile itself) need nothing.
            if k.shape(v).len() == 2 {
                *dims = tile_dims(k, v);
                if dims.is_none() {
                    active.remove(&v);
                    return None;
                }
                memo.insert(v, Target::Direct(v));
                active.remove(&v);
                return Some(Target::Direct(v));
            }
            // An extract of a tile is the tile (narrowing dies in DCE).
            if let Op::Index { vec, .. } = k.at(v) {
                if k.shape(*vec).len() == 2 {
                    *dims = tile_dims(k, *vec);
                    if dims.is_none() {
                        active.remove(&v);
                        return None;
                    }
                    memo.insert(v, Target::Direct(*vec));
                    active.remove(&v);
                    return Some(Target::Direct(*vec));
                }
                active.remove(&v);
                return None;
            }
            // A reshape is metadata: the chain uses the tile, the view
            // stays for other users (or dies in DCE). The store below
            // always drains the whole tile.
            if let Op::Reshape { x, .. } = k.at(v) {
                let Some(t) = descend(k, *x, memo, active, rewires, scalars, dims) else {
                    active.remove(&v);
                    return None;
                };
                memo.insert(v, t);
                active.remove(&v);
                return Some(t);
            }
            // Scalar leaves broadcast later.
            if k.shape(v).is_empty() && matches!(k.at(v), Op::Const(_)) {
                if !scalars.contains(&v) {
                    scalars.push(v);
                }
                memo.insert(v, Target::Expanded(v));
                active.remove(&v);
                return Some(Target::Expanded(v));
            }
            // Single-input heads retile their input (elementwise and
            // tile-native alike — all valid on tiles).
            let ins: Vec<OpId> = match k.at(v) {
                Op::Unary { x, .. } | Op::Cast { x, .. } | Op::Bitcast { x, .. } => vec![*x],
                Op::Binary { x, y, .. } => vec![*x, *y],
                _ => {
                    active.remove(&v);
                    return None;
                }
            };
            for i in ins {
                let Some(t) = descend(k, i, memo, active, rewires, scalars, dims) else {
                    active.remove(&v);
                    return None;
                };
                rewires.push(Rewire { user: v, input: i, target: t });
            }
            memo.insert(v, Target::Direct(v));
            active.remove(&v);
            Some(Target::Direct(v))
        }
        let anchor = Pat::store(Pat::bind('d'), Pat::bind('i'), Pat::bind('x'));
        let mut jobs: Vec<Job> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if !matches!(x, Op::Store { .. }) {
                continue;
            }
            let Some(m) = self.match_pat(op, &anchor) else { continue };
            let src = m.op('x');
            // Scalar stored values stay scalar; anything lane-vector
            // goes through the tile push.
            if self.shape(src).is_empty() {
                continue;
            }
            let mut memo = Map::default();
            let mut active = Set::default();
            let mut rewires = Vec::new();
            let mut scalars = Vec::new();
            let mut dims = None;
            let Some(root) = descend(self, src, &mut memo, &mut active, &mut rewires, &mut scalars, &mut dims) else {
                continue;
            };
            // Tile dims come from the terminal tile, not the root (the
            // root is still vector-shaped at match time). Pure tile
            // chains retile to themselves: nothing to do. An Index or
            // Reshape root (`root != src`) needs the store rewired to
            // the terminal tile — the extract/view dies in DCE.
            let Some(dims) = dims else { continue };
            let pure_tile = matches!(root, Target::Direct(id) if id == src);
            if rewires.is_empty() && scalars.is_empty() && pure_tile {
                continue;
            }
            rewires.push(Rewire { user: op, input: src, target: root });
            jobs.push(Job { store: op, rewires, scalars, dims });
        }
        for job in jobs {
            // One tile stack per store; scalar leaves share it as their
            // `Expand` shape.
            let a = self.insert_before(job.store, Op::Const(Constant::new(job.dims[0])));
            let b = self.insert_before(job.store, Op::Const(Constant::new(job.dims[1])));
            let stack = self.insert_before(job.store, Op::Stack { ops: [a, b].into() });
            let mut expanded: Map<OpId, OpId> = Map::default();
            for s in job.scalars {
                let e = self.insert_before(job.store, Op::Expand { x: s, shape: stack });
                expanded.insert(s, e);
            }
            for r in job.rewires {
                let t = match r.target {
                    Target::Direct(id) => id,
                    Target::Expanded(s) => expanded[&s],
                };
                match &mut self.ops[r.user].op {
                    Op::Unary { x, .. } | Op::Cast { x, .. } | Op::Bitcast { x, .. } | Op::Index { vec: x, .. } => {
                        if *x == r.input {
                            *x = t;
                        }
                    }
                    Op::Binary { x, y, .. } => {
                        if *x == r.input {
                            *x = t;
                        }
                        if *y == r.input {
                            *y = t;
                        }
                    }
                    Op::Store { src: x, .. } => {
                        if *x == r.input {
                            *x = t;
                        }
                    }
                    _ => {}
                }
            }
            todo!("tt_tile store descriptor rewrite is deferred to the TT rework (bytes no longer exists)")
        }
        self.dead_code_elimination();
    }

    /// Split-IR lane fold -> MatmulTile, values only.
    /// Matches the lane-vector matmul family split produces: a single
    /// `Reduce` over one `Mul` of two 2D-shaped operands, folding to a
    /// 2D result. Shapes only — `[m, k]` and `[k, n]` operands, `[m, n]`
    /// result. No memory op is ever matched: operands may be any op
    /// producing tile values.
    /// Emit is stubbed pending the redesign (reader/writer first, then
    /// match directly to LLK): reaching it is a loud `todo!()`.
    /// Anything else keeps its shape (never fatal).
    pub fn vectorize_matmul_tile(&mut self, m: Dim, n: Dim, k: Dim) {
        // Collect the Reduce sites first; mutation happens after the scan.
        let mut sites: Vec<OpId> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if matches!(x, Op::Reduce { .. }) {
                sites.push(op);
            }
        }
        for root in sites {
            // A single Reduce over one Mul of two 2D-shaped operands:
            // split's loop arm folds the accumulation.
            let prod = match self.ops[root].op {
                Op::Reduce { x, rop, .. } if rop == BOp::Add => x,
                _ => continue,
            };
            let (a, b) = match self.ops[prod].op {
                Op::Binary { x, y, bop } if bop == BOp::Mul => (x, y),
                _ => continue,
            };
            // Shapes only: [m, k] and [k, n] operands folding to [m, n].
            if self.shape(a) != vec![m, k] || self.shape(b) != vec![k, n] {
                continue;
            }
            if self.shape(root) != vec![m, n] {
                continue;
            }
            // Emit: the matmul tile op is deleted pending the redesign
            // (reader/writer first, then match directly to LLK).
            // Reaching it is a loud `todo!()`.
            todo!("vectorize_matmul_tile emit pending redesign");
            // (obsolete Stack-matcher body removed by the merge
            // resolution; HEAD's Reduce matcher above plus the stubbed
            // emit stands.)
        }
        self.dead_code_elimination();
    }
}
