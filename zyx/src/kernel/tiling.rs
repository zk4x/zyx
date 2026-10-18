// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent tiling pass: canonicalize scalar IR for tile matching.
//! (Under construction, one step at a time.)

use super::{BOp, Kernel, Op, OpId, Pat, TTOp};
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

        self.debug();
        // Fold first: split replicates the body, so wrappers
        // (x*1, casts of consts) must die before replication, not after.
        self.constant_folding();

        self.default_epilogue();
        self.default_epilogue();

        // TODO: retile via shaped index approaches (split-based lane
        // vectors); the vectorize pass is deleted.
        todo!("tiling: vectorize-based bundling is deleted");

        self.default_epilogue();
        // Deleted vectorize pass (see above todo).

        //self.vectorize_reduce_tile();

        self.default_epilogue();
        self.default_epilogue();

        // Retile the reduce row (see below) on the final IR.
        self.tilize_stores();

        // Fold dot-cone families into matmul tiles (see below).
        self.vectorize_matmul_tile(32, 32, 32);

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
                Op::Unary { x, .. }
                | Op::Cast { x, .. }
                | Op::Bitcast { x, .. }
                | Op::TT(TTOp::TransposeTile { x, .. })
                | Op::TT(TTOp::ReduceTile { x, .. }) => vec![*x],
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
            let (src, bytes) = (m.op('x'), m.op('b'));
            // Any store with a descriptor (scalar lengths and sub-tile
            // descriptors alike); unit stores (null bytes) stay scalar.
            if bytes.is_null() {
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
                    Op::Unary { x, .. }
                    | Op::Cast { x, .. }
                    | Op::Bitcast { x, .. }
                    | Op::TT(TTOp::TransposeTile { x, .. })
                    | Op::TT(TTOp::ReduceTile { x, .. })
                    | Op::Index { vec: x, .. } => {
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

    /// Former: m x n dots over a shared [k, n] tile -> MatmulTile.
    /// Matches the vectorized matmul family with no reference to loads
    /// or stores: per dot a `Mul` of a broadcast `Stack[k]` of one
    /// A-column against a tile-shaped `[k, n]` op (dot lanes fed by the
    /// vectorize pass, since deleted via the tile matcher), the m*n dots
    /// assembled [row][col] into an outer stack.
    /// The `a` root is whatever tile-shaped op the A-lane Index chains
    /// bottom out at (`shape == [m, k]`); the `b` root is whatever
    /// tile-shaped op the prods share (`shape == [k, n]`). Emits by
    /// replacing the outer stack op in place with
    /// `MatmulTile { x: a_tile, y: b_tile }`, so consumers keep pointing
    /// at the same `OpId`; the replaced family dies in DCE. Anything
    /// else keeps its shape (never fatal).
    pub fn vectorize_matmul_tile(&mut self, m: Dim, n: Dim, k: Dim) {
        let (m_len, n_len, k_len) = (m as usize, n as usize, k as usize);
        let mut outers: Vec<OpId> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if matches!(x, Op::Stack { .. }) {
                outers.push(op);
            }
        }
        for outer in outers {
            let rows: Vec<OpId> = match &self.ops[outer].op {
                Op::Stack { ops } if ops.len() == m_len => ops.to_vec(),
                _ => continue,
            };
            // Dots per row; dot -> (prod, lane); prod -> lanes.
            let mut row_dots: Vec<Vec<OpId>> = Vec::with_capacity(m_len);
            let mut dot_cone: Map<OpId, (OpId, usize)> = Map::default();
            let mut prod_lanes: Map<OpId, Vec<usize>> = Map::default();
            let mut shaped = true;
            for &row in &rows {
                let dots: Vec<OpId> = match &self.ops[row].op {
                    Op::Stack { ops } if ops.len() == n_len => ops.to_vec(),
                    _ => {
                        shaped = false;
                        break;
                    }
                };
                for &dot in &dots {
                    // Stacked dot-lane lowers came from the deleted
                    // vectorize pass; with no lane folds in IR, the
                    // stack members never shape.
                    let _ = dot;
                    shaped = false;
                    break;
                }
                if !shaped {
                    break;
                }
                row_dots.push(dots);
            }
            if !shaped {
                continue;
            }
            // n prods, lanes 0..m each, no gaps, no dups.
            if prod_lanes.len() != n_len {
                continue;
            }
            let mut prods: Vec<OpId> = Vec::with_capacity(n_len);
            let mut covered = true;
            for (prod, lanes) in prod_lanes.iter() {
                if lanes.len() != m_len {
                    covered = false;
                    break;
                }
                let mut seen = vec![false; m_len];
                for lane in lanes.iter() {
                    if *lane >= m_len || seen[*lane] {
                        covered = false;
                        break;
                    }
                    seen[*lane] = true;
                }
                if !covered {
                    break;
                }
                prods.push(*prod);
            }
            if !covered {
                continue;
            }
            // Rows: lane == row, prods distinct, same prod set.
            let mut cone_set: Option<Set<OpId>> = None;
            let mut assembled = true;
            for (r, dots) in row_dots.iter().enumerate() {
                let mut set = Set::default();
                for &dot in dots.iter() {
                    let (prod, lane) = match dot_cone.get(&dot) {
                        Some(&(p, l)) => (p, l),
                        None => {
                            assembled = false;
                            break;
                        }
                    };
                    if lane != r {
                        assembled = false;
                        break;
                    }
                    set.insert(prod);
                }
                if !assembled {
                    break;
                }
                if set.len() != n_len {
                    assembled = false;
                    break;
                }
                if let Some(first) = &cone_set {
                    if *first != set {
                        assembled = false;
                        break;
                    }
                } else {
                    cone_set = Some(set);
                }
            }
            if !assembled {
                continue;
            }
            // Prods: Mul(broadcast-stack, shared tile-shaped op).
            let mut b_tile = OpId::NULL;
            let mut acols: Vec<OpId> = Vec::with_capacity(n_len);
            let mut cones = true;
            for &prod in &prods {
                let (bl, br) = match self.ops[prod].op {
                    Op::Binary { x, y, bop } if bop == BOp::Mul => (x, y),
                    _ => {
                        cones = false;
                        break;
                    }
                };
                if !cones {
                    break;
                }
                let (bcast, b) = match (&self.ops[bl].op, &self.ops[br].op) {
                    (Op::Stack { .. }, _) => (bl, br),
                    (_, Op::Stack { .. }) => (br, bl),
                    _ => {
                        cones = false;
                        break;
                    }
                };
                if !cones {
                    break;
                }
                if self.shape(b) != vec![k, n] {
                    cones = false;
                    break;
                }
                if b_tile.is_null() {
                    b_tile = b;
                } else if b_tile != b {
                    cones = false;
                    break;
                }
                let acol = match &self.ops[bcast].op {
                    Op::Stack { ops } if ops.len() == k_len && ops.iter().all(|o| *o == ops[0]) => ops[0],
                    _ => {
                        cones = false;
                        break;
                    }
                };
                if !cones {
                    break;
                }
                acols.push(acol);
            }
            if !cones {
                continue;
            }
            if b_tile.is_null() {
                continue;
            }
            // All A-lane Index chains bottom out at one shared tile-shaped
            // root; every lane op distinct (no shared lanes across columns).
            let mut lane_ops = Set::default();
            let mut seen_bcast = Set::default();
            let mut a_tile = OpId::NULL;
            let mut tiled = true;
            for &acol in &acols {
                if !seen_bcast.insert(acol) {
                    tiled = false;
                    break;
                }
                let members: Vec<OpId> = match &self.ops[acol].op {
                    Op::Stack { ops } if ops.len() == k_len => ops.to_vec(),
                    _ => {
                        tiled = false;
                        break;
                    }
                };
                if !tiled {
                    break;
                }
                for &mem in &members {
                    if !lane_ops.insert(mem) {
                        tiled = false;
                        break;
                    }
                    let mut t = mem;
                    while let Op::Index { vec, .. } = &self.ops[t].op {
                        t = *vec;
                    }
                    if a_tile.is_null() {
                        if self.shape(t) != vec![m, k] {
                            tiled = false;
                            break;
                        }
                        a_tile = t;
                    } else if t != a_tile {
                        tiled = false;
                        break;
                    }
                }
                if !tiled {
                    break;
                }
            }
            if !tiled || a_tile.is_null() {
                continue;
            }
            // Emit: replace the outer stack in place with the matmul tile.
            // Consumers keep pointing at the same OpId; the replaced
            // family dies in DCE.
            self.ops[outer].op = Op::TT(TTOp::MatmulTile { x: a_tile, y: b_tile });
        }
        self.dead_code_elimination();
    }
}
