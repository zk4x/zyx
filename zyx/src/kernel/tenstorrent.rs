// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent kernel passes: TT ops and their insertion.
//!
//! Batch A covers DST locks, NOC barriers, `ReduceUninit`, and the
//! engine-config init constructors. Inits are [`Op::Asm`] statement
//! templates (LLK calls are software wrappers, not hardware ops), so
//! they need no `TTOp` variants; locks/barriers stay structured
//! effects. Pass-generated LLK compute calls live one step sideways
//! in [`TTOp::LLK`]: same template shape as [`Op::Asm`] but excluded
//! from CSE (two identical calls are two traffic events). Insertion
//! passes (`tt_lock_dst`, `tt_init_math`, ...) land in batch D; the
//! CB-sync family (`ReserveBack`/`PushBack`/`WaitFront`/`PopFront` +
//! `tt_sync_cbs`) is already here from slice 1.
//!
//! Pass pipeline order: `tt_tile` → reader/writer → `tt_lower_llks` → `tt_lock_dst`
//! → `tt_dedup_pushes` → `tt_init_math` → `tt_sync_cbs` → `tt_place_pops`, then `verify`. The passes are
//! public so external pass authors can reuse or replace stages.
//! Calling them out of order is a loud panic, never silent corruption.

use super::split::SplitMode;
use crate::DType;
use crate::Map;
use crate::Set;
use crate::dtype::Constant;
use crate::error::{BackendError, ErrorStatus};
use crate::kernel::Pat;
use crate::kernel::{BOp, IDX_T, Kernel, MemScope, Op, OpId, ParamKind, RangeKind, TTOp, TileDim, UOp};
use crate::shape::Dim;
use crate::types::{TinyString, TinyVec};

/// Register allocator: lowered values share Register tiles by
/// liveness. `alloc` mints (one tile) or reuses a dead value's
/// storage; `release` returns it once its last use is emitted.
/// Matches lower in program order with all uses emitted inline,
/// so a value dies when its match is done — the free list is
/// exact. Program order is execution order here (no pass
/// reorders lowered code), so sequential reuse is sound.
struct TtRegAlloc {
    free: Vec<(DType, OpId)>,
}

impl TtRegAlloc {
    fn alloc(&mut self, k: &mut Kernel, dtype: DType, prologue: OpId) -> OpId {
        if let Some(i) = self.free.iter().position(|(d, _)| *d == dtype) {
            return self.free.remove(i).1;
        }
        if prologue.is_null() {
            k.push_back(Op::Storage { dtype, scope: MemScope::Register, len: 1024 })
        } else {
            k.insert_before(prologue, Op::Storage { dtype, scope: MemScope::Register, len: 1024 })
        }
    }

    fn release(&mut self, dtype: DType, storage: OpId) {
        if !self.free.iter().any(|(_, s)| *s == storage) {
            self.free.push((dtype, storage));
        }
    }
}

impl Kernel {
    /// CB sync constructor: `cb.reserve_back(n)` — open `n` back slots
    /// of circular buffer `cb` for writing.
    pub fn tt_reserve_back(&mut self, cb: OpId, n: u8) -> OpId {
        debug_assert!(
            matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_reserve_back: cb {cb} is not a Circular storage"
        );
        debug_assert!(n != 0, "tt_reserve_back: n must be positive");
        self.push_back(Op::TT(TTOp::ReserveBack { cb, n }))
    }

    /// CB sync constructor: `cb.push_back(n)` — publish `n` written slots.
    pub fn tt_push_back(&mut self, cb: OpId, n: u8) -> OpId {
        debug_assert!(
            matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_push_back: cb {cb} is not a Circular storage"
        );
        debug_assert!(n != 0, "tt_push_back: n must be positive");
        self.push_back(Op::TT(TTOp::PushBack { cb, n }))
    }

    /// CB sync constructor: `cb.wait_front(n)` — block until `n` front
    /// slots hold data.
    pub fn tt_wait_front(&mut self, cb: OpId, n: u8) -> OpId {
        debug_assert!(
            matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_wait_front: cb {cb} is not a Circular storage"
        );
        debug_assert!(n != 0, "tt_wait_front: n must be positive");
        self.push_back(Op::TT(TTOp::WaitFront { cb, n }))
    }

    /// CB sync constructor: `cb.pop_front(n)` — release `n` consumed
    /// slots. No producer in slice 1 (waits only); lands with pop
    /// placement.
    pub fn tt_pop_front(&mut self, cb: OpId, n: u8) -> OpId {
        debug_assert!(
            matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_pop_front: cb {cb} is not a Circular storage"
        );
        self.push_back(Op::TT(TTOp::PopFront { cb, n }))
    }

    /// Section-split constructor: end of the reader section. Section
    /// assignment is a placement decision, so markers are builder
    /// ops — render only splits output at them, never derives them.
    pub fn tt_end_reader(&mut self) -> OpId {
        self.push_back(Op::TT(TTOp::EndReader))
    }

    /// Section-split constructor: end of the compute section. Same
    /// placement rules as [`Kernel::tt_end_reader`].
    pub fn tt_end_compute(&mut self) -> OpId {
        self.push_back(Op::TT(TTOp::EndCompute))
    }

    /// Tenstorrent tiling: pad to multiples of 32 and unroll loops
    /// and group ranges to multiples of 32. No special tiling or
    /// vectorization passes: the 32-wide `Unroll` lanes plus composed
    /// indices are the tile IR the backend matchers read.
    pub fn tt_tile(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        self.default_epilogue();
        self.default_epilogue();

        // Gather loops, pad to 32, split unroll-32 FIRST: split's
        // loop arm allows exactly one acc store per body, so it must
        // run before coarsening replicates the body. pad_to pads
        // lengths only on TT (no guards); already-multiples are
        // untouched. Non-reduce loops are a no-op inside (no acc
        // found); anything else keeps its shape for later cuts.
        let mut pad_loops: Vec<OpId> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if matches!(x, Op::Loop { .. }) {
                pad_loops.push(op);
            }
        }
        for l in pad_loops {
            self.pad_to(l, 32);
            let len = match self.ops[l].op {
                Op::Loop { len, .. } => len,
                _ => unreachable!("pad_to preserves the Loop variant"),
            };
            // resolve_const evaluates const-expr lengths on demand, so
            // freshly padded (unfolded) round-ups resolve directly.
            if self.resolve_const(len).and_then(|c| c.as_dim()).is_some_and(|l| l % 32 == 0) {
                self.split(l, SplitMode::Unroll, 32);
            }
        }

        self.default_epilogue();
        self.default_epilogue();

        // Gather group ranges, pad to 32, coarsen LAST: duplicate
        // the vectorized body, one lane per row. Anything else
        // (symbolic, indivisible) is skipped, never fatal. The device
        // grid is 2D: one group range is pre-split into two nested
        // groups below, two proceed as-is, more than two is a later
        // increment.
        let mut pad_groups: Vec<OpId> = Vec::new();
        for (op, x) in self.iter_unordered() {
            if matches!(x, Op::Range { kind: RangeKind::Group(_), .. }) {
                pad_groups.push(op);
            }
        }
        if pad_groups.len() > 2 {
            todo!("tt_tile: more than two group ranges");
        }
        if pad_groups.len() == 1 {
            // One group range: split into two nested groups of roughly
            // equal size (user spec, unconditional — no loop check, no
            // length threshold), then pad and unroll each normally below.
            let g = pad_groups[0];
            self.pad_to(g, 32);
            let len = match self.ops[g].op {
                Op::Range { kind: RangeKind::Group(len), .. } => self.resolve_const(len).and_then(|c| c.as_dim()),
                _ => unreachable!("pad_to preserves the Group range"),
            };
            if let Some(len) = len {
                // Largest divisor at or below sqrt(len): the two
                // factors are as equal as possible.
                let mut f = (len as f64).sqrt() as Dim;
                while f > 1 && len % f != 0 {
                    f -= 1;
                }
                if f > 1 {
                    let factor = u16::try_from(f).expect("tt_tile: group split factor overflows u16");
                    // The inner group takes a new axis: highest group
                    // axis + 1 (user spec; verify requires unique group
                    // axes, and split reuses the axis). TT-local: split
                    // itself is generic across backends.
                    let mut max_group_axis = 0u32;
                    for (_, x) in self.iter_unordered() {
                        if let Op::Range { axis, kind: RangeKind::Group(_) } = x {
                            max_group_axis = max_group_axis.max(*axis);
                        }
                    }
                    let inner = self.split(g, SplitMode::Group, factor);
                    if !inner.is_null() {
                        let new_axis = max_group_axis.checked_add(1).expect("tt_tile: group axis overflows u32");
                        match &mut self.ops[inner].op {
                            Op::Range { axis, kind: RangeKind::Group(_) } => *axis = new_axis,
                            _ => unreachable!("split Group returns the inner group range"),
                        }
                    }
                }
            }
            pad_groups.clear();
            for (op, x) in self.iter_unordered() {
                if matches!(x, Op::Range { kind: RangeKind::Group(_), .. }) {
                    pad_groups.push(op);
                }
            }
        }
        for g in pad_groups {
            self.pad_to(g, 32);
            let len = match self.ops[g].op {
                Op::Range { kind: RangeKind::Group(len), .. } => len,
                _ => unreachable!("pad_to preserves the Group range"),
            };
            if self.resolve_const(len).and_then(|c| c.as_dim()).is_some_and(|l| l % 32 == 0) {
                self.split(g, SplitMode::Unroll, 32);
            }
        }

        self.default_epilogue();
        self.default_epilogue();

        self.verify();
    }

    /// General TT lowering + register allocation (matmul-first,
    /// reduce carried). Traffic is part of emission: the pass matches
    /// compute, creates the CBs (DRAM-backed reader/writer CBs plus
    /// interior CBs), emits the DRAM↔CB movement as `TTOp::Copy`
    /// (explicit bytes, scalar base indices — unrolls stripped),
    /// places `EndReader`/`EndCompute`, lowers each match to a
    /// structured LLK op over storages only, and allocates the
    /// Register storages in the same step. Render only substitutes
    /// numbers; stripped unrolls dangle into DCE.
    ///
    /// This increment lowers `Reduce` (rop `Add`/`Max` over a
    /// `[32, 32]` tile) and single-tile matmul (`Reduce(Add)` over a
    /// `[32, 32, 32]` tiled Mul to `[32, 32]`). Anything else is left
    /// in place for later increments.
    pub fn tt_lower_llks(&mut self) {
        if !self.device_info().tenstorrent {
            return;
        }
        for (_, x) in self.iter_unordered() {
            if let Op::Loop { .. } = x {
                return;
            }
        }
        // Register allocator: lowered values share Register tiles
        // by liveness (free list + explicit release at last use).
        let mut regs = TtRegAlloc { free: Vec::new() };
        // CB Storage definitions live in the prologue (past leading
        // params/storages, before everything else): inserting before
        // `prologue` keeps it stable.
        let mut prologue = self.head;
        for _ in 0..250_000 {
            if prologue.is_null() {
                break;
            }
            match self.ops[prologue].op {
                Op::Param { .. } | Op::Storage { .. } => prologue = self.ops[prologue].next,
                _ => break,
            }
        }
        // Patterns only — matching happens per-install in the
        // phase-2 fixpoint below (match one, rewrite it, rescan).
        // Priority mirrors the old install order: reduces first,
        // then matmuls. Each install consumes its root, so the
        // loop terminates.
        let tile = Pat::bind('t').shape([32i64, 32]);
        let pats = [
            (Pat::sum(tile.clone()).shape([32i64]), TileDim::Row),
            (Pat::reduce(tile.clone(), BOp::Max).shape([32i64]), TileDim::Row),
            (Pat::sum(tile.clone()).shape([] as [i64; 0]), TileDim::Scalar),
            (Pat::reduce(tile, BOp::Max).shape([] as [i64; 0]), TileDim::Scalar),
        ];
        // Single-tile matmul: Reduce(Add) over a tiled Mul to a
        // tiled root. Both orders (matmul is elementwise-commutative
        // in IR; the A/B roles resolve structurally at install).
        let ma = Pat::bind('a').shape([32i64, 32]);
        let mb = Pat::bind('b').shape([32i64, 32]);
        let mmul = Pat::sum((ma.clone() * mb.clone()).shape([32i64, 32, 32])).shape([32i64, 32]);
        let mmul_swap = Pat::sum((mb.clone() * ma.clone()).shape([32i64, 32, 32])).shape([32i64, 32]);
        // Feeder index shapes: K-plain (A) vs K-scaled (B). Add/Mul
        // commutativity covers operand orders; each match_pat call
        // binds at most 3 (fresh budget per call).
        let pat_a = Pat::bind('k') + (Pat::bind('d') * Pat::bind('e'));
        let pat_b = Pat::bind('f') + (Pat::bind('g') * Pat::bind('k'));
        // Phase 1 — reader publishes: every tile `Load` over a DRAM
        // param gets a `TTOp::Copy` publishing the tile to the param's
        // CB, base index stripped of lanes, explicit byte size. Loads
        // stay in place and dangle once compute is lowered (DCE
        // removes them with their index math). Tiled DRAM traffic
        // only — anything else bails the whole pass, since a
        // half-rewritten kernel is worse than none.
        let mut loads: Vec<(OpId, OpId, OpId)> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            if let Op::Load { src: base, src_idx: off, .. } = self.ops[id].op {
                if matches!(self.ops[base].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. }) {
                    if self.shape(id) != vec![32, 32] {
                        return;
                    }
                    loads.push((id, base, off));
                }
            }
            id = self.ops[id].next;
        }
        // CB per DRAM param, in first-load order; slots counted up front.
        // `reader_cb_of` survives for phase 2 (feeder → CB resolution).
        let mut reader_cb_of: Map<OpId, OpId> = Map::default();
        if !loads.is_empty() {
            for (_, _, off) in loads.iter() {
                let mut stack = vec![*off];
                for _ in 0..250_000 {
                    let Some(n) = stack.pop() else { break };
                    match self.ops[n].op {
                        Op::Const(..) | Op::Range { .. } | Op::Param { .. } | Op::Unroll { .. } => {}
                        Op::Binary { x, y, .. } => {
                            stack.push(x);
                            stack.push(y);
                        }
                        Op::Unary { x, .. } => stack.push(x),
                        Op::Cast { x, .. } => stack.push(x),
                        _ => return,
                    }
                }
            }
            let anchor = loads[0].0;
            // One CB per DRAM param, in first-load order; slots counted up front.
            let mut cbs: Vec<(OpId, DType, i64)> = Vec::new();
            for (_, base, _) in loads.iter() {
                if let Some(entry) = cbs.iter_mut().find(|(b, _, _)| *b == *base) {
                    entry.2 += 1;
                } else {
                    let Op::Param { dtype, .. } = self.ops[*base].op else { return };
                    cbs.push((*base, dtype, 1));
                }
            }
            let mut cb_of: Map<OpId, OpId> = Map::default();
            let mut slot_of: Map<OpId, i64> = Map::default();
            for (base, dtype, ntiles) in cbs.iter() {
                let cb = if prologue.is_null() {
                    self.push_back(Op::Storage { dtype: *dtype, scope: MemScope::Circular, len: ntiles * 1024 })
                } else {
                    self.insert_before(prologue, Op::Storage { dtype: *dtype, scope: MemScope::Circular, len: ntiles * 1024 })
                };
                cb_of.insert(*base, cb);
                reader_cb_of.insert(*base, cb);
                slot_of.insert(*base, 0);
            }
            for (_load, base, off) in loads.iter() {
                let slot = slot_of[base];
                slot_of.insert(*base, slot + 1);
                let cb = cb_of[base];
                let dtype = self.dtype(*base);
                // CB indices are elementwise (uniform with vector traffic):
                // tile k lives at element offset k*1024.
                let slot_c = self.insert_const_idx_before(anchor, slot * 1024);
                // Base address with lanes stripped; the transfer size is
                // explicit, so no lane math rides along.
                let Some(base_idx) = self.strip_unrolls(*off, anchor) else {
                    return;
                };
                let bytes = dtype.bit_size() as u32 / 8 * 1024;
                self.insert_before(anchor, Op::TT(TTOp::Copy { src: *base, src_idx: base_idx, dst: cb, dst_idx: slot_c, bytes }));
            }
            self.insert_before(anchor, Op::TT(TTOp::EndReader));
        }
        // Phase 2 fixpoint — match one, rewrite it, rescan.
        // Feeders resolve to reader CBs through the DRAM param
        // (phase 1 owns all DRAM loads); the unpacker streams CBs
        // only. Unmatched compute stays in place for later
        // increments.
        // Lowered root -> writer CB, for phase-3 drains.
        let mut lowered: Map<OpId, OpId> = Map::default();
        // Fresh writer CB + pack store for a lowered value. The pack
        // carries a tile-shaped index so the store check sees a tile
        // value into a tile access; render packs the whole DST segment
        // regardless. Pack must follow the LLK in program order.
        // (Single pack per CB for now; multi-pack sharing comes with
        // multi-consumer values.)
        // Fused sigmoid/silu composites, single tile. Builder spelling
        // (`tensor/elementwise.rs`): `sigmoid = recip(add(exp2(mul(neg(x),
        // LOG2_E)), 1))`, `silu = x * sigmoid(x)`; the scalar consts
        // match any `Const` (builders cast them to the tile dtype).
        // Silu contains a sigmoid cone, so silu gets its own walk
        // first: claiming an inner sigmoid first would rewrite it and
        // the silu root could never match. A CSE+DCE between the walks
        // prunes orphaned inners (same as the old fuse pass).
        let sx = Pat::bind('x').shape([32i64, 32]);
        let sigmoid_pat = (sx.neg().exp() + Pat::Const).recip().shape([32i64, 32]);
        let silu_pat = (&sx * &sigmoid_pat).shape([32i64, 32]);
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Some(m) = self.match_pat(op_id, &silu_pat) {
                let feeder = m.op('x');
                // Feeder resolves through its DRAM param (phase 1 owns
                // all DRAM loads); the unpacker streams CBs only.
                // Non-DRAM feeders stay for later increments.
                let resolved = match self.ops[feeder].op {
                    Op::Load { src: base, .. } => match self.ops[base].op {
                        Op::Param { dtype, .. } => Some((base, dtype)),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some((base, dtype)) = resolved {
                    if let Some(&cb_in) = reader_cb_of.get(&base) {
                        let root = op_id;
                        // Result slot: allocator-minted Register tile.
                        let slot = regs.alloc(self, dtype, prologue);
                        self.ops[root].op = Op::TT(TTOp::SiluTile { cb_in, slot });
                        // Writer CB + pack after the LLK (program order is exact).
                        let writer_cb = if prologue.is_null() {
                            self.push_back(Op::Storage { dtype, scope: MemScope::Circular, len: 1024 })
                        } else {
                            self.insert_before(prologue, Op::Storage { dtype, scope: MemScope::Circular, len: 1024 })
                        };
                        let zero_pack = self.insert_const_idx_before(root, 0i64);
                        let pack_lanes = self.lane_tree_before(zero_pack, &[32, 32], root);
                        let pack_at = self.next_op(root);
                        let pack = Op::Store { dst: writer_cb, dst_idx: pack_lanes, src: root };
                        if pack_at.is_null() {
                            self.push_back(pack);
                        } else {
                            self.insert_before(pack_at, pack);
                        }
                        regs.release(dtype, slot);
                        lowered.insert(root, writer_cb);
                    }
                }
            }
            op_id = next;
        }
        self.common_subexpression_elimination();
        self.dead_code_elimination();
        // Single pass, biggest first: at each Reduce, try matmul,
        // then reduces. Installs consume their root and create no
        // matchable roots, so one head-order walk suffices.
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Op::Reduce { .. } = self.ops[op_id].op {
                // — matmul attempt —
                let mut mhit = None;
                for pat in [&mmul, &mmul_swap] {
                    if let Some(m) = self.match_pat(op_id, pat) {
                        mhit = Some((m.op('a'), m.op('b')));
                        break;
                    }
                }
                if let Some((a, b)) = mhit {
                    let root = op_id;
                    let (ta, tb) = match (&self.ops[a].op, &self.ops[b].op) {
                        (Op::Load { src: pa, src_idx: la }, Op::Load { src: pb, src_idx: lb }) => {
                            match (&self.ops[*pa].op, &self.ops[*pb].op) {
                                (Op::Param { dtype: da, .. }, Op::Param { dtype: db, .. }) => ((*pa, *la, *da), (*pb, *lb, *db)),
                                _ => continue,
                            }
                        }
                        _ => continue,
                    };
                    // K-plain feeder is A, K-scaled is B; roles are the order
                    // binding the same K lane node. Params ride the roles, so
                    // the CB lookup follows A/B, not operand order.
                    // Anything else is left for later increments.
                    let mut fab = None;
                    for ((pa, la, da), (pb, lb, db)) in [(ta, tb), (tb, ta)] {
                        let kx = self.match_pat(la, &pat_a).map(|m| m.op('k'));
                        let ky = self.match_pat(lb, &pat_b).map(|m| m.op('k'));
                        if let (Some(x), Some(y)) = (kx, ky) {
                            if x == y && matches!(self.ops[x].op, Op::Unroll { .. }) {
                                fab = Some(((pa, da), (pb, db), x));
                                break;
                            }
                        }
                    }
                    let Some(((pa, dtype_a), (pb, dtype_b), k)) = fab else {
                        continue;
                    };
                    // The reduced axis must be the shared K lane:
                    // contracting any other axis is not a matmul.
                    // Anything else is left for later increments.
                    if !matches!(&self.ops[root].op, Op::Reduce { reduce_axis, .. } if *reduce_axis == k) {
                        continue;
                    }
                    let (Some(&cb_a), Some(&cb_b)) = (reader_cb_of.get(&pa), reader_cb_of.get(&pb)) else {
                        continue;
                    };
                    debug_assert_eq!(dtype_a, dtype_b, "tt_lower_llks: matmul dtype mismatch");
                    let slot = regs.alloc(self, dtype_a, prologue);
                    self.ops[root].op = Op::TT(TTOp::MatmulTiles { cb_a, cb_b, slot });
                    let writer_cb = if prologue.is_null() {
                        self.push_back(Op::Storage { dtype: dtype_a, scope: MemScope::Circular, len: 1024 })
                    } else {
                        self.insert_before(prologue, Op::Storage { dtype: dtype_a, scope: MemScope::Circular, len: 1024 })
                    };
                    let zero_pack = self.insert_const_idx_before(root, 0i64);
                    let pack_lanes = self.lane_tree_before(zero_pack, &[32, 32], root);
                    let pack_at = self.next_op(root);
                    let pack = Op::Store { dst: writer_cb, dst_idx: pack_lanes, src: root };
                    if pack_at.is_null() {
                        self.push_back(pack);
                    } else {
                        self.insert_before(pack_at, pack);
                    }
                    regs.release(dtype_a, slot);
                    lowered.insert(root, writer_cb);
                } else {
                    // — reduce attempt —
                    let mut hit = None;
                    for (pat, kind) in pats.iter() {
                        if let Some(m) = self.match_pat(op_id, pat) {
                            if let Op::Reduce { x, rop, .. } = self.ops[op_id].op {
                                debug_assert_eq!(x, m.op('t'), "tt_lower_llks: feeder bind drifted");
                                hit = Some((x, rop, *kind));
                                break;
                            }
                        }
                    }
                    if let Some((x, rop, kind)) = hit {
                        let root = op_id;
                        // Feeder resolves through its DRAM param (phase 1 owns
                        // all DRAM loads); the unpacker streams CBs only.
                        // Non-DRAM feeders stay for later increments.
                        let (base, dtype) = match self.ops[x].op {
                            Op::Load { src: base, .. } => match self.ops[base].op {
                                Op::Param { dtype, .. } => (base, dtype),
                                _ => continue,
                            },
                            _ => continue,
                        };
                        let Some(&cb_in) = reader_cb_of.get(&base) else {
                            continue;
                        };
                        // Scaler: fresh const-1 tile CB per match, filled on-device
                        // by a Fill plus a pack publish (no scaler load — nothing
                        // references the CB value in SSA).
                        let cb_sc = if prologue.is_null() {
                            self.push_back(Op::Storage { dtype, scope: MemScope::Circular, len: 1024 })
                        } else {
                            self.insert_before(prologue, Op::Storage { dtype, scope: MemScope::Circular, len: 1024 })
                        };
                        let one = self.insert_before(root, Op::Const(dtype.one_constant()));
                        let fill_slot = regs.alloc(self, dtype, prologue);
                        let fill = self.insert_before(root, Op::TT(TTOp::FillTile { slot: fill_slot, val: one }));
                        let zero_sc = self.insert_const_idx_before(root, 0i64);
                        let one_lanes = self.lane_tree_before(zero_sc, &[32, 32], root);
                        self.insert_before(root, Op::Store { dst: cb_sc, dst_idx: one_lanes, src: fill });
                        regs.release(dtype, fill_slot);
                        // Result slot: allocator-minted Register tile.
                        let slot = regs.alloc(self, dtype, prologue);
                        self.ops[root].op = Op::TT(TTOp::ReduceTile { rop, kind, cb_in, cb_sc, slot });
                        // Writer CB + pack after the LLK (program order is exact).
                        let writer_cb = if prologue.is_null() {
                            self.push_back(Op::Storage { dtype, scope: MemScope::Circular, len: 1024 })
                        } else {
                            self.insert_before(prologue, Op::Storage { dtype, scope: MemScope::Circular, len: 1024 })
                        };
                        let zero_pack = self.insert_const_idx_before(root, 0i64);
                        let pack_lanes = self.lane_tree_before(zero_pack, &[32, 32], root);
                        let pack_at = self.next_op(root);
                        let pack = Op::Store { dst: writer_cb, dst_idx: pack_lanes, src: root };
                        if pack_at.is_null() {
                            self.push_back(pack);
                        } else {
                            self.insert_before(pack_at, pack);
                        }
                        regs.release(dtype, slot);
                        lowered.insert(root, writer_cb);
                    }
                }
            }
            // — fused sigmoid attempt (single tile). Silu cones were
            // claimed by the walk above, so any sigmoid left is a plain
            // one. Runs for every op: the root is a Recip, not a Reduce.
            if let Some(m) = self.match_pat(op_id, &sigmoid_pat) {
                let feeder = m.op('x');
                let resolved = match self.ops[feeder].op {
                    Op::Load { src: base, .. } => match self.ops[base].op {
                        Op::Param { dtype, .. } => Some((base, dtype)),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some((base, dtype)) = resolved {
                    if let Some(&cb_in) = reader_cb_of.get(&base) {
                        let root = op_id;
                        let slot = regs.alloc(self, dtype, prologue);
                        self.ops[root].op = Op::TT(TTOp::SigmoidTile { cb_in, slot });
                        let writer_cb = if prologue.is_null() {
                            self.push_back(Op::Storage { dtype, scope: MemScope::Circular, len: 1024 })
                        } else {
                            self.insert_before(prologue, Op::Storage { dtype, scope: MemScope::Circular, len: 1024 })
                        };
                        let zero_pack = self.insert_const_idx_before(root, 0i64);
                        let pack_lanes = self.lane_tree_before(zero_pack, &[32, 32], root);
                        let pack_at = self.next_op(root);
                        let pack = Op::Store { dst: writer_cb, dst_idx: pack_lanes, src: root };
                        if pack_at.is_null() {
                            self.push_back(pack);
                        } else {
                            self.insert_before(pack_at, pack);
                        }
                        regs.release(dtype, slot);
                        lowered.insert(root, writer_cb);
                    }
                }
            }
            op_id = next;
        }
        // Phase 3 — writer drains: every DRAM store of a lowered
        // value becomes a `TTOp::Copy` from its writer CB in a writer
        // section closed by `EndCompute`. Base index stripped of
        // lanes, explicit byte size; the store itself is unlinked.
        // Stores of unlowered values stay for later increments (loud
        // downstream: a DRAM store in compute renders nothing).
        let mut stores: Vec<(OpId, OpId, OpId, OpId)> = Vec::new();
        let mut id = self.head;
        for _ in 0..250_000 {
            if id.is_null() {
                break;
            }
            match self.ops[id].op {
                Op::Load { src: base, .. } => {
                    if matches!(self.ops[base].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. })
                        && self.shape(id) != vec![32, 32]
                    {
                        return;
                    }
                }
                Op::Store { dst: base, dst_idx: off, src, .. } => {
                    if matches!(self.ops[base].op, Op::Storage { .. }) {
                        // Already lowered; nothing owned here.
                    } else if matches!(self.ops[base].op, Op::Param { kind: ParamKind::GlobalMut, .. }) {
                        // Tiled stores and row stores (a reduce row is 32
                        // elements); anything else bails the pass.
                        match self.shape(id).as_slice() {
                            [32, 32] | [32] => stores.push((id, base, off, src)),
                            _ => {
                                return;
                            }
                        }
                    } else {
                        return;
                    }
                }
                _ => {}
            }
            id = self.ops[id].next;
        }
        if !stores.is_empty() {
            let first = stores[0].0;
            // Empty reader section for store-only kernels.
            let mut scan = self.head;
            let mut has_reader = false;
            for _ in 0..250_000 {
                if scan.is_null() {
                    break;
                }
                if matches!(self.ops[scan].op, Op::TT(TTOp::EndReader)) {
                    has_reader = true;
                    break;
                }
                scan = self.ops[scan].next;
            }
            if !has_reader {
                // A tiled DRAM load without a reader section means
                // phase 1 bailed: don't paper over it.
                let mut scan = self.head;
                for _ in 0..250_000 {
                    if scan.is_null() {
                        break;
                    }
                    if let Op::Load { src: base, .. } = self.ops[scan].op {
                        if matches!(self.ops[base].op, Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. })
                            && self.shape(scan) == vec![32, 32]
                        {
                            return;
                        }
                    }
                    scan = self.ops[scan].next;
                }
                self.insert_before(first, Op::TT(TTOp::EndReader));
            }
            // Compute-side rewrites in head order; drains append at the tail.
            self.push_back(Op::TT(TTOp::EndCompute));
            // Single pack per writer CB: slot 0 throughout.
            let slot_zero = self.push_back(Op::Const(Constant::idx(0i64)));
            for (store, base, off, src) in stores.iter() {
                let Some(&cb) = lowered.get(src) else {
                    // Unlowered value: stays for later increments.
                    continue;
                };
                // Drain the whole packed producer page, sized from the
                // writer CB allocation — not the seed store width. The
                // packer writes full tiles (`pack_tile`), and the
                // face-ordered column layout means a seed-width drain
                // starves `untilize` gathers (a `[32]` reduce row lives
                // at column 0 across the tile, not in a contiguous row).
                // Single-tile shapes only; anything else bails the pass.
                match self.shape(*store).as_slice() {
                    [32, 32] | [32] => {}
                    _ => return,
                };
                let (cb_dtype, cb_len) = match self.ops[cb].op {
                    Op::Storage { dtype, scope: MemScope::Circular, len } => (dtype, len),
                    _ => return,
                };
                if self.dtype(*store) != cb_dtype {
                    return;
                }
                let Ok(page_elems) = u32::try_from(cb_len) else {
                    return;
                };
                let bytes = match (cb_dtype.bit_size() as u32 / 8).checked_mul(page_elems) {
                    Some(b) if b > 0 => b,
                    _ => return,
                };
                let tail = self.tail;
                let Some(base_idx) = self.strip_unrolls(*off, tail) else {
                    return;
                };
                self.push_back(Op::TT(TTOp::Copy { src: cb, src_idx: slot_zero, dst: *base, dst_idx: base_idx, bytes }));
                self.remove_op(*store);
            }
        }
        self.verify();
    }

    /// Base index with lanes stripped: clone the index expression
    /// with every `Unroll` replaced by 0. No affinity assumption —
    /// substitution works on any expression shape; uncloneable leaves
    /// (traffic, control) bail the caller. Shared leaves are reused
    /// (they dominate `before` exactly like the cloned index math
    /// does); the rest is rebuilt before `before`.
    ///
    /// Group ids rebase to physical tiles: DRAM traffic addresses
    /// tile-layout buffers, so a const-scaled group id takes the
    /// physical tile stride (one full [32, 32] tile per step, times
    /// inner-group extents) and a bare group id inside Add/Sub wraps
    /// to id * stride. Symbolic extents bail the caller.
    fn strip_unrolls(&mut self, idx: OpId, before: OpId) -> Option<OpId> {
        // Clone memo per call (index math is small).
        let mut memo: Map<OpId, OpId> = Map::default();
        // Group axis -> physical tile stride (element units), built
        // lazily on first group use inside the walk below.
        let mut strides: Option<Map<u32, Dim>> = None;
        let zero = self.insert_before(before, Op::Const(Constant::idx(0i64)));
        memo.insert(zero, zero);
        let mut stack = vec![(idx, false)];
        for _ in 0..250_000 {
            let Some((n, expanded)) = stack.pop() else { break };
            if memo.contains_key(&n) {
                continue;
            }
            match self.ops[n].op {
                Op::Const(_) | Op::Range { .. } | Op::Param { .. } => {
                    memo.insert(n, n);
                }
                Op::Unroll { .. } => {
                    memo.insert(n, zero);
                }
                Op::Binary { x, y, bop } => {
                    if !expanded {
                        stack.push((n, true));
                        stack.push((x, false));
                        stack.push((y, false));
                    } else {
                        let nx = memo[&x];
                        let ny = memo[&y];
                        // Group-leaf axes of the memo images (owned
                        // copies; every borrow ends before inserts).
                        let gx: Option<u32> = match &self.ops[nx].op {
                            Op::Range { axis, kind: RangeKind::Group(_) } => Some(*axis),
                            _ => None,
                        };
                        let gy: Option<u32> = match &self.ops[ny].op {
                            Op::Range { axis, kind: RangeKind::Group(_) } => Some(*axis),
                            _ => None,
                        };
                        // Axis -> physical tile stride, built on first
                        // group use. One group step is one full
                        // [32, 32] tile (1024 elems, the unit phase 1
                        // sizes CBs, slots and byte counts in) times
                        // the extents of all inner (higher-axis) groups;
                        // split keeps the outer on the original axis.
                        // Unresolvable (symbolic) extents bail the pass:
                        // never silent wrong code.
                        if (gx.is_some() || gy.is_some()) && strides.is_none() {
                            let mut axes: Vec<(u32, Dim)> = Vec::new();
                            for (_, op) in self.iter_unordered() {
                                if let Op::Range { axis, kind: RangeKind::Group(len) } = op {
                                    let Some(e) = self.resolve_const(*len).and_then(Constant::as_dim) else {
                                        return None;
                                    };
                                    axes.push((*axis, e));
                                }
                            }
                            axes.sort();
                            let mut map: Map<u32, Dim> = Map::default();
                            let mut suffix: Dim = 1024;
                            for (a, e) in axes.iter().rev() {
                                map.insert(*a, suffix);
                                suffix = match suffix.checked_mul(*e) {
                                    Some(s) => s,
                                    None => return None,
                                };
                            }
                            strides = Some(map);
                        }
                        // Physical group bases: a const-scaled group id
                        // takes the tile stride; a bare group id inside
                        // Add/Sub wraps to id * stride. These are the
                        // forms split/pad composition emits; anything
                        // else keeps the faithful clone and trips the
                        // downstream checks loudly.
                        let mut ax = nx;
                        let mut ay = ny;
                        match bop {
                            BOp::Mul => {
                                if let Some(a) = gx {
                                    if gy.is_none() && matches!(self.ops[ny].op, Op::Const(_)) {
                                        let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                        let sc = self.insert_before(before, Op::Const(Constant::idx(s)));
                                        ax = self.insert_before(before, Op::Binary { x: nx, y: sc, bop });
                                    }
                                }
                                if let Some(a) = gy {
                                    if gx.is_none() && matches!(self.ops[nx].op, Op::Const(_)) {
                                        let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                        let sc = self.insert_before(before, Op::Const(Constant::idx(s)));
                                        ay = self.insert_before(before, Op::Binary { x: sc, y: ny, bop });
                                    }
                                }
                            }
                            BOp::Add | BOp::Sub => {
                                if let Some(a) = gx {
                                    let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                    let sc = self.insert_before(before, Op::Const(Constant::idx(s)));
                                    ax = self.insert_before(before, Op::Binary { x: nx, y: sc, bop: BOp::Mul });
                                }
                                if let Some(a) = gy {
                                    let s = strides.as_ref().expect("strip_unrolls: stride map missing")[&a];
                                    let sc = self.insert_before(before, Op::Const(Constant::idx(s)));
                                    ay = self.insert_before(before, Op::Binary { x: ny, y: sc, bop: BOp::Mul });
                                }
                            }
                            _ => {}
                        }
                        // Fold constant lanes away: the base is fully
                        // static exactly when the index is.
                        let id = match (&self.ops[ax].op, &self.ops[ay].op) {
                            (Op::Const(cx), Op::Const(cy)) => {
                                self.insert_before(before, Op::Const(Constant::binary(*cx, *cy, bop)))
                            }
                            _ => self.insert_before(before, Op::Binary { x: ax, y: ay, bop }),
                        };
                        memo.insert(n, id);
                    }
                }
                Op::Unary { x, uop } => {
                    if !expanded {
                        stack.push((n, true));
                        stack.push((x, false));
                    } else {
                        let nx = memo[&x];
                        let id = match &self.ops[nx].op {
                            Op::Const(cx) => self.insert_before(before, Op::Const(cx.unary(uop))),
                            _ => self.insert_before(before, Op::Unary { x: nx, uop }),
                        };
                        memo.insert(n, id);
                    }
                }
                Op::Cast { x, dtype } => {
                    if !expanded {
                        stack.push((n, true));
                        stack.push((x, false));
                    } else {
                        let nx = memo[&x];
                        let id = self.insert_before(before, Op::Cast { x: nx, dtype });
                        memo.insert(n, id);
                    }
                }
                _ => return None,
            }
        }
        memo.get(&idx).copied()
    }

    /// DST lock constructor: `tile_regs_acquire()`.
    pub fn tt_math_lock(&mut self) -> OpId {
        self.push_back(Op::TT(TTOp::MathLock))
    }

    /// DST unlock constructor: `tile_regs_commit()`.
    pub fn tt_math_unlock(&mut self) -> OpId {
        self.push_back(Op::TT(TTOp::MathUnlock))
    }

    /// Pack lock constructor: `tile_regs_wait()`.
    pub fn tt_pack_lock(&mut self) -> OpId {
        self.push_back(Op::TT(TTOp::PackLock))
    }

    /// Pack unlock constructor: `tile_regs_release()`.
    pub fn tt_pack_unlock(&mut self) -> OpId {
        self.push_back(Op::TT(TTOp::PackUnlock))
    }

    /// NOC read barrier constructor: `noc_async_read_barrier()`.
    pub fn tt_noc_read_barrier(&mut self) -> OpId {
        self.push_back(Op::TT(TTOp::NocReadBarrier))
    }

    /// NOC write barrier constructor: `noc_async_write_barrier()`.
    pub fn tt_noc_write_barrier(&mut self) -> OpId {
        self.push_back(Op::TT(TTOp::NocWriteBarrier))
    }

    /// DST lock insertion. MATH tile-compute ops run under
    /// `tile_regs_acquire()..commit()`; PACK drains (a tile value
    /// stored into a Circular buffer) run under `tile_regs_wait()..release()`.
    /// Pure insertion, no allocation: the physical DST slots are the
    /// `MemScope::Register` storages already in the IR, and the CBs are
    /// the `MemScope::Circular` storages — this pass never assigns a
    /// number. Per section (delimited by `TTOp::EndReader` /
    /// `TTOp::EndCompute`): one MATH block and one PACK block. Acquire precedes the first MATH op, commit
    /// follows the last; wait precedes the first pack store, release
    /// follows the last. A section with no MATH ops emits no MATH
    /// locks; a section with no pack stores emits no pack locks.
    ///
    /// Back-edge rule (old DST discipline): re-executing
    /// `tile_regs_acquire()` on a loop back edge wedges the DST, so the
    /// acquire hoists to the preheader of the outermost enclosing loop
    /// whose body holds no pack store, and the commit mirrors to that
    /// loop's end (PACK transitions stop the bubbling, like the old
    /// MATH-held check at each `EndLoop`). PACK locks stay per-trip:
    /// the old pass closed PACK at every back edge instead.
    ///
    /// MATH covers the four SSA tile ops and `LLK`, the tiled
    /// elementwise SSA ops (`Unary`/`Binary`/`Cast`/`Bitcast` over
    /// tiles — scalar index math over consts/loop vars is not tile
    /// math), tiled user `Asm`, and circular loads consumed straight
    /// by a pack store (their unpack copy renders at the load, under
    /// MATH). A pack drain is any `Store` of a tile value into a
    /// Circular buffer: from a Register slot, from a circular load
    /// (CB→CB spelled load+store), or from SSA/`LLK`/`Asm` tile
    /// results.
    pub fn tt_lock_dst(&mut self) {
        // Users: direct-pack loads (below) read it.
        let mut users: Map<OpId, Vec<OpId>> = Map::default();
        let mut scan = self.head;
        while !scan.is_null() {
            for p in self.at(scan).parameters() {
                if !p.is_null() {
                    users.entry(p).or_default().push(scan);
                }
            }
            scan = self.next_op(scan);
        }
        let is_math = |kernel: &Kernel, users: &Map<OpId, Vec<OpId>>, id: OpId| {
            let op = kernel.at(id);
            if matches!(
                op,
                Op::TT(TTOp::ReduceTile { .. })
                    | Op::TT(TTOp::LLKBcast { .. })
                    | Op::TT(TTOp::MatmulTiles { .. })
                    | Op::TT(TTOp::SigmoidTile { .. })
                    | Op::TT(TTOp::SiluTile { .. })
                    | Op::TT(TTOp::FillTile { .. })
                    | Op::TT(TTOp::LLK { .. })
            ) {
                return true;
            }
            // A circular load consumed straight by a pack store unpacks
            // at its own position (render copies there) — it is MATH
            // work like any other unpack.
            if let Op::Load { src, .. } = op {
                if matches!(kernel.at(*src), Op::Storage { scope: MemScope::Circular, .. }) {
                    if let Some(us) = users.get(&id) {
                        return us.iter().all(|u| matches!(kernel.at(*u), Op::Store { .. }))
                            && us.iter().any(|u| matches!(kernel.at(*u), Op::Store { .. }));
                    }
                }
                return false;
            }
            match op {
                Op::Asm { .. } => true,
                Op::Load { .. } | Op::Unary { .. } | Op::Binary { .. } | Op::Cast { .. } | Op::Bitcast { .. } => {
                    // Index math is scalar RISC work, never DST math.
                    kernel.dtype(id) != IDX_T && kernel.shape(id).len() == 2
                }
                _ => false,
            }
        };
        // Pack drain: a Store into a Circular buffer of a tile or row
        // value. Single source of truth is `tt_pack_store_cb` (shared
        // with sync accounting and init_math); this delegates to it.
        let is_pack = |kernel: &Kernel, id: OpId| kernel.tt_pack_store_cb(id).is_some();
        let mut op_id = self.head;
        while !op_id.is_null() {
            // Section scan: ordered math/pack positions, per-math
            // loop-stack snapshots, Loop→EndLoop matching,
            // section-relative positions, and pack-store positions
            // (a pack boundary stops the hoist bubbling).
            let mut maths: Vec<OpId> = Vec::new();
            let mut packs: Vec<OpId> = Vec::new();
            let mut math_loops: Map<OpId, Vec<OpId>> = Map::default();
            let mut loop_stack: Vec<OpId> = Vec::new();
            let mut loop_end: Map<OpId, OpId> = Map::default();
            let mut pos_of: Map<OpId, usize> = Map::default();
            let mut pack_pos: Set<OpId> = Set::default();
            let mut scan = op_id;
            let mut section_end: Option<OpId> = None;
            let mut pos = 0usize;
            while !scan.is_null() {
                if matches!(self.at(scan), Op::TT(TTOp::EndReader) | Op::TT(TTOp::EndCompute)) {
                    section_end = Some(scan);
                    break;
                }
                pos_of.insert(scan, pos);
                pos += 1;
                match self.at(scan) {
                    Op::Loop { .. } => loop_stack.push(scan),
                    Op::EndLoop => {
                        if let Some(start) = loop_stack.pop() {
                            loop_end.insert(start, scan);
                        }
                    }
                    _ => {}
                }
                if is_math(self, &users, scan) {
                    maths.push(scan);
                    math_loops.insert(scan, loop_stack.clone());
                }
                if is_pack(self, scan) {
                    packs.push(scan);
                    pack_pos.insert(scan);
                }
                scan = self.next_op(scan);
            }
            let pos_of_op = |what: &str, id: OpId| -> usize {
                pos_of.get(&id).copied().unwrap_or_else(|| panic!("tt_lock_dst: {what} without position"))
            };
            // Regions: math runs split at pack boundaries (a pack
            // drains DST, so it closes its region — the old one
            // acquire/commit/wait/pack/release region per math→pack
            // group). Packs attach to the preceding run (leading
            // packs, before any math, to the first run).
            let mut runs: Vec<(Vec<OpId>, Vec<OpId>)> = Vec::new();
            let mut cur_run: Vec<OpId> = Vec::new();
            let mut prev_math: Option<OpId> = None;
            for m in maths.iter().copied() {
                let mp = pos_of_op("math", m);
                if let Some(pm) = prev_math {
                    let pp = pos_of_op("math", pm);
                    if packs.iter().any(|p| {
                        let qp = pos_of_op("pack", *p);
                        qp > pp && qp < mp
                    }) {
                        runs.push((core::mem::take(&mut cur_run), Vec::new()));
                    }
                }
                cur_run.push(m);
                prev_math = Some(m);
            }
            if !cur_run.is_empty() {
                runs.push((cur_run, Vec::new()));
            }
            if !runs.is_empty() {
                for p in packs.iter().copied() {
                    let qp = pos_of_op("pack", p);
                    let mut owner = 0usize;
                    for (i, (ms, _)) in runs.iter().enumerate() {
                        let fp = pos_of_op("math", ms[0]);
                        if fp <= qp {
                            owner = i;
                        }
                    }
                    runs[owner].1.push(p);
                }
            }
            // Back-edge hoist: bubble the acquire outward across
            // enclosing loops whose bodies hold no pack store (a pack
            // boundary stops the bubbling — the old MATH-held check).
            // The commit mirrors up to the acquire's level. PACK locks
            // stay per-trip.
            let body_has_pack = |pos_of: &Map<OpId, usize>, pack_pos: &Set<OpId>, l: OpId, end: OpId| {
                let (Some(&s), Some(&e)) = (pos_of.get(&l), pos_of.get(&end)) else {
                    return true;
                };
                pack_pos.iter().any(|p| pos_of.get(p).is_some_and(|q| *q > s && *q < e))
            };
            // Pack-only section (pure movement, no math): one pack
            // region, no MATH locks.
            if runs.is_empty() && !packs.is_empty() {
                self.insert_before(packs[0], Op::TT(TTOp::PackLock));
                self.insert_after(packs[packs.len() - 1], Op::TT(TTOp::PackUnlock));
            }
            for (ms, ps) in runs.iter() {
                let first = ms[0];
                let last = ms[ms.len() - 1];
                let first_loops = math_loops.get(&first).cloned().expect("tt_lock_dst: math without loop snapshot");
                let last_loops = math_loops.get(&last).cloned().expect("tt_lock_dst: math without loop snapshot");
                let mut crossed = 0usize;
                let mut target = first;
                for l in first_loops.iter().rev() {
                    let end = match loop_end.get(l) {
                        Some(e) => *e,
                        None => break,
                    };
                    if body_has_pack(&pos_of, &pack_pos, *l, end) {
                        break;
                    }
                    target = *l;
                    crossed += 1;
                }
                let acquire_at = Some(target);
                let mut target = last;
                let mut n = 0usize;
                for l in last_loops.iter().rev() {
                    if n >= crossed {
                        break;
                    }
                    let end = match loop_end.get(l) {
                        Some(e) => *e,
                        None => break,
                    };
                    if body_has_pack(&pos_of, &pack_pos, *l, end) {
                        break;
                    }
                    target = end;
                    n += 1;
                }
                let commit_at = Some(target);
                // Commit before wait: MATH drains to registers, then
                // packs read them. Inserts are by OpId, so order among
                // the four is independent of position.
                if let Some(at) = acquire_at {
                    self.insert_before(at, Op::TT(TTOp::MathLock));
                }
                if let Some(at) = commit_at {
                    self.insert_after(at, Op::TT(TTOp::MathUnlock));
                }
                if let Some(pf) = ps.first() {
                    self.insert_before(*pf, Op::TT(TTOp::PackLock));
                }
                if let Some(pl) = ps.last() {
                    self.insert_after(*pl, Op::TT(TTOp::PackUnlock));
                }
            }
            op_id = match section_end {
                Some(b) => self.next_op(b),
                None => break,
            }
        }

        self.verify();
    }

    /// Duplicate-publish elimination: drop repeated identical publishes
    /// (`Copy` of one global tile into one circular slot) beyond what
    /// the consuming waits need, so pushed pages match waits/pops.
    ///
    /// Per circular buffer + slot index, with all copies carrying
    /// identical data: `C` publish copies, `W` predicted waits (one per
    /// live load with a non-LLK, non-marker user, plus one per
    /// LLK-structured call consuming a bucket load — sync's exact wait
    /// rules; dead loads draw none). When `C > W`, keep the earliest
    /// `W` copies and drop the rest with the dead loads. Push/wait/pop
    /// balance is preserved by construction; anything else (loops,
    /// distinct pages or data, fills sufficing for the waits) is left
    /// in place — dead loads included, since a dead load is some kept
    /// fill's drain pop.
    ///
    /// All matching is exact-`OpId` identity plus resolved constant
    /// slot indices; unresolvable indices never match, and any
    /// CB-touching op between duplicate copies (or an opaque `Asm`)
    /// drops the group. No new helpers: the predicate is inline below.
    pub fn tt_dedup_pushes(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("tt_dedup_pushes");
        // Positional push/load pairing breaks under loops: skip the
        // whole pass rather than reason trip counts here.
        let mut scan = self.head;
        while !scan.is_null() {
            if matches!(self.at(scan), Op::Loop { .. } | Op::EndLoop) {
                return;
            }
            scan = self.next_op(scan);
        }
        // Order index for gap scans, plus the linear user map.
        let mut order: Vec<OpId> = Vec::new();
        let mut pos_of: Map<OpId, usize> = Map::default();
        scan = self.head;
        while !scan.is_null() {
            pos_of.insert(scan, order.len());
            order.push(scan);
            scan = self.next_op(scan);
        }
        let mut users: Map<OpId, Vec<OpId>> = Map::default();
        for &id in &order {
            for p in self.at(id).parameters() {
                if !p.is_null() {
                    users.entry(p).or_default().push(id);
                }
            }
        }
        // Circular base of a Copy side (the base names its storage
        // directly), if it is one.
        let circ_base = |kernel: &Kernel, side: OpId| -> Option<OpId> {
            if matches!(kernel.at(side), Op::Storage { scope: MemScope::Circular, .. }) {
                return Some(side);
            }
            None
        };
        // Resolved slot index of an access; unresolvable never matches.
        let slot_of_index =
            |kernel: &Kernel, index: OpId| -> Option<Dim> { kernel.resolve_const(index).and_then(|c| c.as_dim()) };
        // Whether an op moves pages of `cb` (gap dirtiness). SSA and
        // marker ops are inert; user `Asm` and anything unlisted are
        // opaque and invalidate.
        let touches = |kernel: &Kernel, id: OpId, cb: OpId| -> bool {
            let load_cb = |load: OpId| -> bool {
                if let Op::Load { src, .. } = kernel.at(load) {
                    return *src == cb;
                }
                false
            };
            match kernel.at(id) {
                Op::Copy { src, dst, .. } => *src == cb || *dst == cb,
                Op::Load { src, .. } => *src == cb,
                Op::Store { dst, .. } => *dst == cb,
                Op::TT(TTOp::LLK { ops, .. }) => ops.iter().copied().any(|o| {
                    !o.is_null()
                        && (matches!(kernel.at(o), Op::Storage { scope: MemScope::Circular, .. } if o == cb) || load_cb(o))
                }),
                Op::TT(TTOp::ReduceTile { cb_in, cb_sc, .. }) => *cb_in == cb || *cb_sc == cb,
                Op::TT(TTOp::MatmulTiles { cb_a, cb_b, .. }) => *cb_a == cb || *cb_b == cb,
                Op::TT(TTOp::SigmoidTile { cb_in, .. }) => *cb_in == cb,
                Op::TT(TTOp::SiluTile { cb_in, .. }) => *cb_in == cb,
                Op::TT(TTOp::LLKBcast { cb_a, cb_b, mx, plain, .. }) => {
                    *cb_a == cb || *cb_b == cb || load_cb(*mx) || load_cb(*plain)
                }
                Op::TT(TTOp::ReserveBack { .. } | TTOp::PushBack { .. } | TTOp::WaitFront { .. } | TTOp::PopFront { .. }) => true,
                Op::Const(_)
                | Op::Param { .. }
                | Op::Storage { .. }
                | Op::Range { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Cast { .. }
                | Op::Bitcast { .. }
                | Op::TT(TTOp::EndReader)
                | Op::TT(TTOp::EndCompute)
                | Op::TT(TTOp::MathLock)
                | Op::TT(TTOp::MathUnlock)
                | Op::TT(TTOp::PackLock)
                | Op::TT(TTOp::PackUnlock)
                | Op::TT(TTOp::NocReadBarrier)
                | Op::TT(TTOp::NocWriteBarrier)
                | Op::TT(TTOp::ReduceUninit) => false,
                _ => true,
            }
        };
        // Publish copies: (cb, slot, src) with positions, slot resolved.
        // Loads: (cb, slot, src) with positions. Unresolvable slots and
        // non-publish copies never enter the maps.
        let mut pubs: Vec<(OpId, Dim, OpId, OpId, usize, OpId)> = Vec::new();
        let mut loads: Vec<(OpId, Dim, OpId, OpId, usize, OpId)> = Vec::new();
        for &id in &order {
            match self.at(id) {
                Op::Copy { src, src_idx, dst, dst_idx, .. } => {
                    if circ_base(self, *src).is_some() {
                        continue;
                    }
                    let (Some(cb), Some(slot)) = (circ_base(self, *dst), slot_of_index(self, *dst_idx)) else {
                        continue;
                    };
                    // Data identity: one DRAM base (Param) as the source;
                    // anything else (a value op) has no page identity.
                    if !matches!(self.at(*src), Op::Param { .. } | Op::Storage { .. }) {
                        continue;
                    }
                    pubs.push((cb, slot, *src, *src_idx, pos_of[&id], id));
                }
                Op::Load { src, src_idx, .. } => {
                    let (Some(cb), Some(slot)) = (circ_base(self, *src), slot_of_index(self, *src_idx)) else {
                        continue;
                    };
                    loads.push((cb, slot, *src, *src_idx, pos_of[&id], id));
                }
                _ => {}
            }
        }
        // Bucket keys present in the publishes.
        let mut keys: Vec<(OpId, Dim)> = Vec::new();
        for (cb, slot, _, _, _, _) in &pubs {
            if !keys.iter().any(|(c, s)| c == cb && s == slot) {
                keys.push((*cb, slot.clone()));
            }
        }
        let mut removals: Vec<OpId> = Vec::new();
        for (cb, slot) in keys {
            // One (base, index) pair per slot: ambiguous pages never
            // match.
            let mut data: Vec<(OpId, OpId)> = Vec::new();
            for (_, _, src, src_idx, _, _) in pubs.iter().filter(|(c, s, _, _, _, _)| c == &cb && s == &slot) {
                if !data.contains(&(*src, *src_idx)) {
                    data.push((*src, *src_idx));
                }
            }
            if data.len() != 1 {
                continue;
            }
            // One address pair per slot on the consume side.
            let mut addrs: Vec<(OpId, OpId)> = Vec::new();
            for (_, _, src, src_idx, _, _) in loads.iter().filter(|(c, s, _, _, _, _)| c == &cb && s == &slot) {
                if !addrs.contains(&(*src, *src_idx)) {
                    addrs.push((*src, *src_idx));
                }
            }
            if addrs.len() != 1 {
                continue;
            }
            let mut copies: Vec<(usize, OpId)> =
                pubs.iter().filter(|(c, s, _, _, _, _)| c == &cb && s == &slot).map(|(_, _, _, _, p, id)| (*p, *id)).collect();
            copies.sort();
            // Predicted waits for this slot: each live load with a
            // non-LLK, non-marker user draws one load-wait, and each
            // LLK-structured call draws one wait per consumed bucket
            // load (sync waits every provenance CB per call; markers
            // are transparent, dead loads draw none). Fills must match
            // waits, so keep one copy per predicted wait.
            let mut load_waits = 0usize;
            let mut call_waits = 0usize;
            for (_, _, _, _, _, id) in loads.iter().filter(|(c, s, _, _, _, _)| c == &cb && s == &slot) {
                let Some(us) = users.get(id) else { continue };
                let mut draws_load_wait = false;
                for u in us {
                    match self.at(*u) {
                        Op::TT(TTOp::LLK { .. } | TTOp::ReduceTile { .. } | TTOp::LLKBcast { .. } | TTOp::MatmulTiles { .. }) => {
                            call_waits += 1;
                        }
                        _ => {
                            draws_load_wait = true;
                        }
                    }
                }
                if draws_load_wait {
                    load_waits += 1;
                }
            }
            let dead: Vec<OpId> = loads
                .iter()
                .filter(|(c, s, _, _, _, _)| c == &cb && s == &slot)
                .filter(|(_, _, _, _, _, id)| !users.contains_key(id))
                .map(|(_, _, _, _, _, id)| *id)
                .collect();
            let (c, l) = (copies.len(), load_waits + call_waits);
            // Fills suffice: leave copies AND dead loads in place (a
            // dead load is some kept fill's drain pop).
            if c < 2 || l >= c {
                continue;
            }
            // Gap dirtiness: any CB-touching op (or opaque Asm) strictly
            // between consecutive duplicate copies drops the bucket.
            let mut clean = true;
            for w in copies.windows(2) {
                for &mid in &order[w[0].0 + 1..w[1].0] {
                    if touches(self, mid, cb) {
                        clean = false;
                        break;
                    }
                }
                if !clean {
                    break;
                }
            }
            if !clean {
                continue;
            }
            // Keep the earliest `l` copies; drop the rest with the dead loads.
            for (_, id) in copies.iter().skip(l) {
                removals.push(*id);
            }
            removals.extend(dead);
        }
        for id in removals {
            self.remove_op(id);
        }
        // DCE runs after `tt_storage`: `Op::parameters` skips NULL LLK
        // operands, so the reachability walk cannot hit them. No CSE:
        // removals only delete exact-duplicate copies and dead loads
        // whose nodes stay alive via the kept twins.
        self.dead_code_elimination();
        self.verify();
    }
    /// CB sync insertion: wrap every CB traffic op with straight-line
    /// single-tile syncs (`n = 1`; hoisted batches land with batching).
    /// Direction reads off which copy side is circular:
    /// - publish (`dst` index over a Circular storage): `ReserveBack`
    ///   immediately before, `PushBack` immediately after;
    /// - drain (`src` index over a Circular storage): `WaitFront`
    ///   immediately before, no pop yet;
    /// - CB-to-CB copy: `WaitFront` on the read side immediately
    ///   before plus `ReserveBack`/`PushBack` on the write side
    ///   (the copy packs into `dst`, like a pack store).
    /// A pack `Store` (tile value into a Circular buffer) reserves
    /// before and pushes after, mirroring the old `TilePack` rule.
    /// Scalar `Store`s into a Circular buffer fill pages
    /// element-wise (1024 per page): a section run totaling an
    /// exact page multiple publishes whole pages (reserve ahead,
    /// push past); partial or uncountable fills stay unpublished.
    /// A circular `Load` is the compute-consume read (the kernel Load IS
    /// the CB read; codegen bundles read+consume, so its wait sits at the
    /// consumer — here it sits before the Load): `WaitFront` immediately
    /// before iff some user is a non-fusion tile consumer — the four SSA
    /// tile ops pre-storage, tiled elementwise SSA (`Unary`/`Binary`/
    /// `Cast`/`Bitcast`), a pack `Store` (CB→CB spelled load+store), or
    /// tiled user `Asm`. A load drained only through fusion (every user
    /// an `LLK` provenance position) carries
    /// no wait of its own: the fused call's provenance waits cover those
    /// pages (old "fused-draining loads carry no syncs"). Any other consumer
    /// is malformed CB traffic — DRAM↔CB moves are `Copy` (see
    /// `copy_global_to_circular`), never load+store — loud, never
    /// silently skipped. A circular buffer reached outside an index
    /// expression is likewise malformed — loud, never silently skipped.
    /// An `LLK` consumes its input CBs straight from the buffers, so
    /// each trailing provenance load gets a `WaitFront` immediately
    /// before the call (waits are level checks — sharing one page
    /// across calls stays correct; pops count it, see `tt_place_pops`).
    pub fn tt_sync_cbs(&mut self) {
        // Linear user map: users[v] = ops taking v as a data operand.
        let mut users: Map<OpId, Vec<OpId>> = Map::default();
        let mut scan = self.head;
        while !scan.is_null() {
            for p in self.at(scan).parameters() {
                users.entry(p).or_default().push(scan);
            }
            scan = self.next_op(scan);
        }

        // Indexed-window batching (old place_waits/place_pops parity):
        // a circular read with a loop-varying slot index observes CB
        // slots window-relative, so a per-trip pop slides the window
        // under later trips and the index reads the wrong tile. When
        // every data access to `cb` inside the innermost const-trip
        // loop is such an indexed read, `batched[(loop, cb)]` holds
        // trips × reads and one WaitFront/PopFront of that count
        // brackets the loop instead of per-trip n:1 syncs. Head
        // reads, stores, LLK traffic, symbolic trips, and over-depth
        // windows keep the per-trip syncs.
        let mut batched: Map<(OpId, OpId), u32> = Map::default();
        let mut batched_read: Set<OpId> = Set::default();
        scan = self.head;
        while !scan.is_null() {
            let (read_op, gep_index, cb) = match self.ops[scan].op {
                Op::Load { src: cb, src_idx: index, .. }
                    if matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. }) =>
                {
                    (scan, index, cb)
                }
                Op::Copy { src: cb, src_idx: index, .. }
                    if matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. }) =>
                {
                    (scan, index, cb)
                }
                _ => {
                    scan = self.next_op(scan);
                    continue;
                }
            };
            // Head slot (const 0) reads the CB front: per-trip syncs
            // stay correct. Anything else is window-relative.
            if self.resolve_const(gep_index).and_then(|c| c.as_dim()) == Some(0) {
                scan = self.next_op(scan);
                continue;
            }
            // Innermost enclosing loop of the read.
            let mut inner: Option<OpId> = None;
            let mut cur = self.prev_op(scan);
            let mut depth = 0u32;
            while !cur.is_null() {
                match self.ops[cur].op {
                    Op::EndLoop => depth += 1,
                    Op::Loop { .. } => {
                        if depth == 0 {
                            inner = Some(cur);
                            break;
                        }
                        depth -= 1;
                    }
                    _ => {}
                }
                cur = self.prev_op(cur);
            }
            let Some(loop_op) = inner else {
                scan = self.next_op(scan);
                continue;
            };
            let Op::Loop { len } = self.ops[loop_op].op else {
                unreachable!("tt_sync_cbs: batch loop is a Loop");
            };
            let Some(trips) =
                self.resolve_const(len).and_then(|c| c.as_dim()).and_then(|t| u32::try_from(t).ok()).filter(|t| *t > 0)
            else {
                scan = self.next_op(scan);
                continue;
            };
            // Matching EndLoop of `loop_op`.
            let mut end = self.next_op(loop_op);
            let mut depth = 0u32;
            while !end.is_null() {
                match self.ops[end].op {
                    Op::Loop { .. } => depth += 1,
                    Op::EndLoop => {
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                    }
                    _ => {}
                }
                end = self.next_op(end);
            }
            if end.is_null() {
                scan = self.next_op(scan);
                continue;
            }
            // Span scan: every data access to `cb` must be a direct
            // qualifying indexed read, else the window cannot be held.
            let mut direct = 0u32;
            let mut held = true;
            let mut cur = self.next_op(loop_op);
            let mut depth = 0u32;
            while cur != end {
                match self.ops[cur].op {
                    Op::Loop { .. } => depth += 1,
                    Op::EndLoop => depth -= 1,
                    Op::Load { src: c, src_idx: index, .. } => {
                        if c == cb {
                            let indexed = self.resolve_const(index).and_then(|c| c.as_dim()) != Some(0);
                            let ok_users = users.get(&cur).is_some_and(|us| {
                                us.iter().all(|u| match &self.ops[*u].op {
                                    Op::Store { .. } | Op::Asm { .. } => true,
                                    op => {
                                        self.shape(*u).len() == 2
                                            && matches!(
                                                op,
                                                Op::Unary { .. } | Op::Binary { .. } | Op::Cast { .. } | Op::Bitcast { .. }
                                            )
                                    }
                                })
                            });
                            if depth == 0 && indexed && ok_users {
                                direct += 1;
                            } else {
                                held = false;
                                break;
                            }
                        }
                    }
                    Op::Store { dst, .. } => {
                        if dst == cb {
                            held = false;
                            break;
                        }
                    }
                    Op::Copy { src, src_idx: index, dst, .. } => {
                        let src_hit = if src == cb { Some(index) } else { None };
                        let dst_hit = if dst == cb { Some(dst) } else { None };
                        match (src_hit, dst_hit) {
                            (None, None) => {}
                            (Some(index), None) => {
                                let indexed = self.resolve_const(index).and_then(|c| c.as_dim()) != Some(0);
                                if depth == 0 && indexed {
                                    direct += 1;
                                } else {
                                    held = false;
                                    break;
                                }
                            }
                            _ => {
                                held = false;
                                break;
                            }
                        }
                    }
                    Op::TT(TTOp::LLK { .. })
                    | Op::TT(TTOp::ReduceTile { .. })
                    | Op::TT(TTOp::LLKBcast { .. })
                    | Op::TT(TTOp::MatmulTiles { .. })
                    | Op::TT(TTOp::SigmoidTile { .. })
                    | Op::TT(TTOp::SiluTile { .. }) => {
                        let mut feeds = false;
                        for o in self.at(cur).parameters() {
                            if o.is_null() {
                                continue;
                            }
                            if let Op::Load { src: c, .. } = self.ops[o].op
                                && c == cb
                            {
                                feeds = true;
                                break;
                            }
                            // Storage-only LLKs name their CBs directly.
                            if matches!(self.ops[o].op, Op::Storage { scope: MemScope::Circular, .. }) && o == cb {
                                feeds = true;
                                break;
                            }
                        }
                        if feeds {
                            held = false;
                            break;
                        }
                    }
                    _ => {}
                }
                cur = self.next_op(cur);
            }
            if !held || direct == 0 {
                scan = self.next_op(scan);
                continue;
            }
            let Some(total) = trips.checked_mul(direct) else {
                scan = self.next_op(scan);
                continue;
            };
            let Op::Storage { len, .. } = self.ops[cb].op else {
                unreachable!("tt_sync_cbs: batch cb is a Storage");
            };
            if len <= 0 {
                scan = self.next_op(scan);
                continue;
            }
            let depth_tiles = (len / 1024) as u32;
            if total == 0 || total > depth_tiles || total > 255 {
                scan = self.next_op(scan);
                continue;
            }
            batched.entry((loop_op, cb)).or_insert(total);
            batched_read.insert(read_op);
            scan = self.next_op(scan);
        }

        // Unified publish tally: every sub-page publish counts
        // ELEMENTS per (section, cb, run anchor) — scalar `Store`s
        // (1 element) and vector `Copy`s (`size` elements) through
        // one normalized arm below. Whole-page traffic (tile copies,
        // tile pack stores) never tallies: one page per execution
        // streams per-op inline below, trip by trip through small
        // CBs, where a hoisted multi-page reserve would deadlock. A
        // section run totaling an exact multiple of 1024 publishes
        // whole pages like tile publishes: one `ReserveBack` ahead
        // of the run, one `PushBack` past it. Runs anchor outside
        // the outermost const-trip loop (per-trip syncs would
        // multiply the count); straight-line stores merge per
        // section. Partial pages and anything under
        // symbolic/conditional loops stay unpublished — a consumer
        // wait then fails loudly downstream, which is correct: an
        // unknowable or incomplete fill cannot satisfy a tile wait.
        {
            enum CountFrame {
                Trip { op: OpId, trips: u64 },
                Other,
            }
            let mut stack: Vec<CountFrame> = Vec::new();
            let mut mult: u64 = 1;
            let mut uncountable: u32 = 0;
            let mut section = 0u8;
            // (section, cb, anchor-before) -> (anchor-after, elements).
            // Straight-line runs share the NULL anchor per section.
            let mut tallies: Map<(u8, OpId, OpId), (OpId, u64)> = Map::default();
            let mut scan = self.head;
            while !scan.is_null() {
                match self.ops[scan].op {
                    Op::TT(TTOp::EndReader) => section = 1,
                    Op::TT(TTOp::EndCompute) => section = 2,
                    Op::Loop { len } => match self.at(len) {
                        Op::Const(c) => match c.as_dim().and_then(|t| u64::try_from(t).ok()).filter(|t| *t > 0) {
                            Some(t) => {
                                mult *= t;
                                stack.push(CountFrame::Trip { op: scan, trips: t });
                            }
                            None => {
                                uncountable += 1;
                                stack.push(CountFrame::Other);
                            }
                        },
                        _ => {
                            uncountable += 1;
                            stack.push(CountFrame::Other);
                        }
                    },
                    Op::EndLoop => match stack.pop() {
                        Some(CountFrame::Trip { trips, .. }) => mult /= trips,
                        Some(CountFrame::Other) => uncountable -= 1,
                        None => panic!("tt_sync_cbs: EndLoop without Loop at {scan:?}"),
                    },
                    Op::Store { .. } | Op::Copy { .. } => {
                        // Elements published per execution, if this op
                        // is a sub-page publish into a Circular buffer:
                        // scalar stores and scalar copies write 1
                        // element, vector copies `size`. Whole-page
                        // traffic matches neither (tile layouts) and
                        // streams inline below.
                        let published: Option<(OpId, u64)> = match self.ops[scan].op {
                            Op::Store { src: x, dst, .. } if uncountable == 0 && self.is_circular_base(dst) => {
                                match self.shape(x).as_slice() {
                                    [] => Some((dst, 1)),
                                    _ => None,
                                }
                            }
                            Op::Copy { src, dst, .. } if uncountable == 0 => {
                                let src_is_cb = matches!(self.ops[src].op, Op::Storage { scope: MemScope::Circular, .. });
                                if matches!(self.ops[dst].op, Op::Storage { scope: MemScope::Circular, .. }) && !src_is_cb {
                                    match self.shape(scan).as_slice() {
                                        [] => Some((dst, 1)),
                                        [size] => Some((dst, *size as u64)),
                                        _ => None,
                                    }
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        };
                        if let Some((cb, units)) = published {
                            let elems = mult * units;
                            match stack.iter().find_map(|f| match f {
                                CountFrame::Trip { op, .. } => Some(*op),
                                CountFrame::Other => None,
                            }) {
                                Some(loop_op) => {
                                    // Matching EndLoop of the anchor.
                                    let mut end = self.next_op(loop_op);
                                    let mut depth = 0u32;
                                    while !end.is_null() {
                                        match self.ops[end].op {
                                            Op::Loop { .. } => depth += 1,
                                            Op::EndLoop => {
                                                if depth == 0 {
                                                    break;
                                                }
                                                depth -= 1;
                                            }
                                            _ => {}
                                        }
                                        end = self.next_op(end);
                                    }
                                    if !end.is_null() {
                                        tallies
                                            .entry((section, cb, loop_op))
                                            .and_modify(|e| e.1 += elems)
                                            .or_insert((end, elems));
                                    }
                                }
                                None => {
                                    tallies
                                        .entry((section, cb, OpId::NULL))
                                        .and_modify(|e| {
                                            e.0 = scan;
                                            e.1 += elems;
                                        })
                                        .or_insert((scan, elems));
                                }
                            }
                        }
                    }
                    _ => {}
                }
                scan = self.next_op(scan);
            }
            for ((_, cb, before), (after, elems)) in tallies {
                if elems % 1024 != 0 {
                    continue;
                }
                let pages = elems / 1024;
                if pages == 0 || pages > 255 {
                    continue;
                }
                let n = pages as u8;
                self.insert_before(before, Op::TT(TTOp::ReserveBack { cb, n }));
                self.insert_after(after, Op::TT(TTOp::PushBack { cb, n }));
            }
        }

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match self.ops[op_id].op {
                Op::Copy { src, dst, .. } => {
                    let src_cb = if self.is_circular_base(src) { Some(src) } else { None };
                    let dst_cb = if self.is_circular_base(dst) { Some(dst) } else { None };
                    match (src_cb, dst_cb) {
                        (None, None) => {}
                        (None, Some(cb)) => {
                            // Vector and scalar publishes tally
                            // element-wise above (exact page multiples
                            // publish); the per-op publish here would
                            // over-count. Partial pages stay
                            // unpublished (loud downstream). Tile traffic
                            // keys off the Copy's access shape, not the
                            // CB buffer's ([len] is always length-1).
                            let tallied = self.shape(op_id).len() <= 1;
                            if !tallied {
                                self.insert_before(op_id, Op::TT(TTOp::ReserveBack { cb, n: 1 }));
                                self.insert_after(op_id, Op::TT(TTOp::PushBack { cb, n: 1 }));
                            }
                        }
                        (Some(cb), None) => {
                            // Drains wait page-granular like tile drains;
                            // partial-page use needs no special tally.
                            // Indexed-window reads carry no per-trip
                            // wait: the bracket wait covers the window.
                            if !batched_read.contains(&op_id) {
                                self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb, n: 1 }));
                            }
                        }
                        (Some(scb), Some(dcb)) => {
                            self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb: scb, n: 1 }));
                            self.insert_before(op_id, Op::TT(TTOp::ReserveBack { cb: dcb, n: 1 }));
                            self.insert_after(op_id, Op::TT(TTOp::PushBack { cb: dcb, n: 1 }));
                        }
                    }
                }
                Op::Store { dst, .. } => {
                    // Pack store: a tile or row value into a Circular
                    // buffer reserves a back slot and pushes it, like a
                    // publish. A `Const` src is a fill, not a pack — left
                    // alone (render rejects it loudly). Single source of
                    // truth is `tt_pack_store_cb`.
                    if self.is_circular_base(dst) && self.tt_pack_store_cb(op_id).is_some() {
                        self.insert_before(op_id, Op::TT(TTOp::ReserveBack { cb: self.tt_storage_of(dst), n: 1 }));
                        let cb = self.tt_storage_of(dst);
                        self.insert_after(op_id, Op::TT(TTOp::PushBack { cb, n: 1 }));
                    }
                }
                Op::Load { src, .. } => {
                    // Non-CB reads are not sync business. Circularity gates
                    // the whole arm as a nested `if` — a `continue` here
                    // would skip the cursor advance at the loop bottom and
                    // spin forever.
                    if matches!(self.ops[src].op, Op::Storage { scope: MemScope::Circular, .. }) {
                        match users.get(&op_id) {
                            // Dead load: no SSA consumer reads. The drain pop
                            // consumes an already-waited page (fifo count,
                            // not push-wait-pop pairs); a dead load draws
                            // no wait of its own. Merge-orphaned dead loads
                            // (provenance twin fused, never-deduped copy
                            // kept by the c<=l guard) otherwise break the
                            // push-wait balance with waits no push feeds.
                            None => {}
                            Some(use_list) => {
                                for &u in use_list {
                                    let ok = match &self.ops[u].op {
                                        Op::TT(
                                            TTOp::ReduceTile { .. }
                                            | TTOp::LLKBcast { .. }
                                            | TTOp::MatmulTiles { .. }
                                            | TTOp::SigmoidTile { .. }
                                            | TTOp::SiluTile { .. }
                                            | TTOp::LLK { .. },
                                        ) => true,
                                        Op::Store { .. } | Op::Asm { .. } => true,
                                        op => {
                                            self.shape(u).len() == 2
                                                && matches!(
                                                    op,
                                                    Op::Unary { .. } | Op::Binary { .. } | Op::Cast { .. } | Op::Bitcast { .. }
                                                )
                                        }
                                    };
                                    if !ok {
                                        panic!("tt_sync_cbs: circular load {op_id:?} feeds non-compute {u:?}");
                                    }
                                }
                                // Fusion-drained loads (every user a
                                // marker or an LLK provenance position)
                                // carry no wait of their own: the fused
                                // call's provenance waits cover those
                                // pages (old "fused-draining loads carry
                                // no syncs"). Any SSA/Store/Asm consumer
                                // copies at its own position and needs
                                // the wait here.
                                let fused_only = use_list.iter().all(|u| {
                                    matches!(
                                        self.ops[*u].op,
                                        Op::TT(TTOp::LLK { .. })
                                            | Op::TT(TTOp::ReduceTile { .. })
                                            | Op::TT(TTOp::LLKBcast { .. })
                                            | Op::TT(TTOp::MatmulTiles { .. })
                                            | Op::TT(TTOp::SigmoidTile { .. })
                                            | Op::TT(TTOp::SiluTile { .. })
                                    )
                                });
                                if !fused_only && !batched_read.contains(&op_id) {
                                    self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb: src, n: 1 }));
                                }
                            }
                        }
                    }
                }
                Op::TT(TTOp::Copy { src, dst, .. }) => {
                    // Staged traffic: explicit size, no per-op tally.
                    // Publish reserves and pushes one page; drain waits
                    // one page (batched reads keep their bracket wait).
                    let src_cb = if self.is_circular_base(src) { Some(src) } else { None };
                    let dst_cb = if self.is_circular_base(dst) { Some(dst) } else { None };
                    match (src_cb, dst_cb) {
                        (None, Some(cb)) => {
                            self.insert_before(op_id, Op::TT(TTOp::ReserveBack { cb, n: 1 }));
                            self.insert_after(op_id, Op::TT(TTOp::PushBack { cb, n: 1 }));
                        }
                        (Some(cb), None) => {
                            if !batched_read.contains(&op_id) {
                                self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb, n: 1 }));
                            }
                        }
                        (Some(scb), Some(dcb)) => {
                            self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb: scb, n: 1 }));
                            self.insert_before(op_id, Op::TT(TTOp::ReserveBack { cb: dcb, n: 1 }));
                            self.insert_after(op_id, Op::TT(TTOp::PushBack { cb: dcb, n: 1 }));
                        }
                        (None, None) => {
                            panic!("tt_sync_cbs: staged copy {op_id:?} touches no CB");
                        }
                    }
                }
                Op::TT(TTOp::LLK { .. })
                | Op::TT(TTOp::ReduceTile { .. })
                | Op::TT(TTOp::LLKBcast { .. })
                | Op::TT(TTOp::MatmulTiles { .. })
                | Op::TT(TTOp::SigmoidTile { .. })
                | Op::TT(TTOp::SiluTile { .. }) => {
                    // Provenance waits: the call consumes its input CBs
                    // straight from the buffers, so each Circular
                    // storage operand gets a wait immediately before
                    // the call. Storage-only operands: no feeder loads
                    // to chase.
                    let mut wait_cbs: Vec<OpId> = Vec::new();
                    for o in self.at(op_id).parameters() {
                        if o.is_null() {
                            continue;
                        }
                        if matches!(self.ops[o].op, Op::Storage { scope: MemScope::Circular, .. }) && !wait_cbs.contains(&o) {
                            wait_cbs.push(o);
                        }
                    }
                    for cb in wait_cbs {
                        self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb, n: 1 }));
                    }
                }
                _ => {}
            }
            op_id = next;
        }

        // Bracket waits for indexed windows: one WaitFront of the
        // whole window ahead of the loop (the per-trip waits above
        // were skipped for these reads).
        let done: Vec<((OpId, OpId), u32)> = batched.iter().map(|(k, v)| (*k, *v)).collect();
        for ((loop_op, cb), total) in done {
            self.insert_before(loop_op, Op::TT(TTOp::WaitFront { cb, n: total as u8 }));
        }

        self.verify();
    }

    /// Engine-config init insertion: ports the old `init_math` +
    /// `reconfig_pack` + `fill_out_cbs` + `close_reduce` passes
    /// plus the compute-startup insertion. Inits are [`Op::Asm`]
    /// statement templates (see the `tt_*_init` constructors), placed
    /// inline immediately before their op — always correct (LLK config
    /// writes are idempotent); hoisting/dedup out of loops is a later
    /// optimization pass, not a correctness item.
    ///
    /// Per compute section (delimited by `EndReader`/`EndCompute`):
    /// - pre-scan: the first pack store's CB (`first_pack`, the old
    ///   `fill_out_cbs` out — no new buffer, no acc), whether a
    ///   matmul call exists, and the IR-order distinct CBs of tile
    ///   loads plus the `EndReader` position (startup triple);
    /// - `compute_kernel_hw_startup(in0, in1, out)` right after
    ///   `EndReader` iff no matmul call exists and loads+store exist
    ///   (single input repeats `in0`; pure movement emits none);
    /// - before each circular load consumed through a copy (tiled
    ///   elementwise SSA, a direct pack store): `copy_tile_init(cb)`
    ///   first, then `copy_tile_to_dst_init_short_with_dt(prev, cb)`
    ///   on unpack source change (ahead of every copy site in walk
    ///   order, so the unpack config always precedes its copies);
    /// - before each tiled `Unary`/`Binary`/`Cast`: the matching init;
    ///   tiled `Mad` is a hard error; scalar binary lanes follow the
    ///   old scalar rule (float const-exprs for Add/Mul/Div-right,
    ///   int 0..=31 on the right for shifts; scalar `Sub` and anything
    ///   else is a loud error — the old render rejected scalar `Sub`
    ///   too);
    /// - before each matmul/transpose/reduce/fused-broadcast `LLK`:
    ///   `mm_init(a, b, first_pack)` / `transpose_wh_init(cb,
    ///   first_pack)` / `reduce_init<op, dim>(ci, cs, acc)` /
    ///   `<bop>_bcast_*_init_short(a, b)` (missing pack target errors
    ///   like the old unfilled-out error);
    /// - before every pack store: `pack_reconfig_data_format(cb)`;
    /// - closing a pending reduce acc: packing it inserts
    ///   `ReduceUninit` immediately before the nearest preceding
    ///   `MathUnlock` (none → loud error).
    ///
    /// Tile compute outside the compute section is malformed (locks
    /// and inits only exist there) — loud panic, never silently
    /// skipped.
    pub fn tt_init_math(&mut self) -> Result<(), BackendError> {
        // Users (consumer-position decisions below read it).
        let mut users: Map<OpId, Vec<OpId>> = Map::default();
        let mut scan = self.head;
        while !scan.is_null() {
            for p in self.at(scan).parameters() {
                if !p.is_null() {
                    users.entry(p).or_default().push(scan);
                }
            }
            scan = self.next_op(scan);
        }
        // Pre-scan: first pack CB, matmul presence, startup triple.
        let mut first_pack: Option<OpId> = None;
        let mut has_matmul = false;
        let mut has_matmul_block = false;
        let mut startup_cbs: Vec<OpId> = Vec::new();
        let mut end_reader: Option<OpId> = None;
        let mut in_compute = false;
        let mut scan = self.head;
        while !scan.is_null() {
            match &self.ops[scan].op {
                Op::TT(TTOp::EndReader) => {
                    in_compute = true;
                    end_reader = Some(scan);
                }
                Op::TT(TTOp::EndCompute) => in_compute = false,
                Op::Load { src, .. } if in_compute => {
                    if matches!(self.ops[*src].op, Op::Storage { scope: MemScope::Circular, .. }) && !startup_cbs.contains(src) {
                        startup_cbs.push(*src);
                    }
                }
                // Structured lowered ops reference their CBs directly
                // (no compute loads left); seed the startup triple in
                // operand order. Legacy string templates keep the load
                // path above; their defs precede uses, so walk order
                // stays load-first wherever loads exist.
                Op::TT(TTOp::MatmulTiles { cb_a, cb_b, .. }) if in_compute => {
                    for cb in [*cb_a, *cb_b] {
                        if !startup_cbs.contains(&cb) {
                            startup_cbs.push(cb);
                        }
                    }
                }
                Op::TT(TTOp::ReduceTile { cb_in, cb_sc, .. }) if in_compute => {
                    for cb in [*cb_in, *cb_sc] {
                        if !startup_cbs.contains(&cb) {
                            startup_cbs.push(cb);
                        }
                    }
                }
                Op::TT(TTOp::LLKBcast { cb_a, cb_b, .. }) if in_compute => {
                    for cb in [*cb_a, *cb_b] {
                        if !startup_cbs.contains(&cb) {
                            startup_cbs.push(cb);
                        }
                    }
                }
                Op::TT(TTOp::SigmoidTile { cb_in, .. }) if in_compute => {
                    if !startup_cbs.contains(&cb_in) {
                        startup_cbs.push(*cb_in);
                    }
                }
                Op::TT(TTOp::SiluTile { cb_in, .. }) if in_compute => {
                    if !startup_cbs.contains(&cb_in) {
                        startup_cbs.push(*cb_in);
                    }
                }
                op if in_compute && first_pack.is_none() => {
                    if let Some(cb) = self.tt_pack_store_cb(scan) {
                        let _ = op;
                        first_pack = Some(cb);
                    }
                }
                _ => {}
            }
            if let Op::TT(TTOp::LLK { asm, .. }) = &self.ops[scan].op
                && asm.as_str().starts_with("matmul_tiles(")
            {
                has_matmul = true;
            }
            if matches!(&self.ops[scan].op, Op::TT(TTOp::MatmulTiles { .. })) {
                has_matmul_block = true;
            }
            scan = self.next_op(scan);
        }
        if let (Some(reader), Some(&in0), Some(out)) = (end_reader, startup_cbs.first(), first_pack) {
            let in1 = startup_cbs.get(1).copied().unwrap_or(in0);
            if has_matmul_block {
                // matmul needs in0->SrcB, in1->SrcA. The installed
                // headers have no SrcOrder-templated startup overload
                // (docs describe a newer tree), so swap the CB args:
                // srca=in1, srcb=in0. The legacy matmul_tiles string
                // path below keeps its old skip.
                self.insert_after(
                    reader,
                    Op::Asm {
                        asm: TinyString::new("compute_kernel_hw_startup({1}, {0}, {2});"),
                        ops: TinyVec::new(&[in0, in1, out]),
                    },
                );
            } else if !has_matmul {
                // Same template as `tt_compute_startup`, placed right after
                // `EndReader` (the ctor pushes back — wrong end).
                self.insert_after(
                    reader,
                    Op::Asm {
                        asm: TinyString::new("compute_kernel_hw_startup({0}, {1}, {2});"),
                        ops: TinyVec::new(&[in0, in1, out]),
                    },
                );
            }
        }

        // Main walk: per-op inits, pending-acc closing, section discipline.
        let mut in_compute = false;
        let mut unpack_src: Option<OpId> = None;
        let mut pending_acc: Option<OpId> = None;
        // Values threading the pending acc (forward marks, O(1) per
        // op): packs consult this set instead of rechasing def chains
        // backward over already-visited ops.
        let mut acc_values: Set<OpId> = Set::default();
        let mut last_math_unlock: Option<OpId> = None;
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match &self.ops[op_id].op {
                Op::TT(TTOp::EndReader) => {
                    in_compute = true;
                    unpack_src = None;
                    pending_acc = None;
                    acc_values.clear();
                }
                Op::TT(TTOp::EndCompute) => {
                    in_compute = false;
                    pending_acc = None;
                    acc_values.clear();
                }
                Op::TT(TTOp::MathUnlock) if in_compute => last_math_unlock = Some(op_id),
                _ => {}
            }
            if !in_compute {
                // Reader/writer move through `Copy` only; any tile
                // compute, fused call, pack store, or CB load outside
                // the compute section is malformed.
                let bad = match &self.ops[op_id].op {
                    Op::TT(TTOp::LLK { .. })
                    | Op::TT(TTOp::ReduceTile { .. })
                    | Op::TT(TTOp::LLKBcast { .. })
                    | Op::TT(TTOp::MatmulTiles { .. })
                    | Op::TT(TTOp::SigmoidTile { .. })
                    | Op::TT(TTOp::SiluTile { .. }) => true,
                    Op::Store { .. } => self.tt_pack_store_cb(op_id).is_some(),
                    Op::Load { src, .. } => {
                        matches!(self.ops[*src].op, Op::Storage { scope: MemScope::Circular, .. })
                    }
                    op => {
                        // Tile compute is float-typed and 2D-shaped;
                        // integer-typed 2D index math (drain index
                        // expressions, lane trees) is address arithmetic,
                        // allowed in any section.
                        // in any section.
                        matches!(op, Op::Unary { .. } | Op::Binary { .. } | Op::Cast { .. } | Op::Bitcast { .. } | Op::Asm { .. })
                            && self.shape(op_id).len() == 2
                            && self.dtype(op_id).is_float()
                    }
                };
                if bad {
                    panic!("tt_init_math: tile compute {op_id:?} outside the compute section");
                }
                op_id = next;
                continue;
            }
            // In-compute walk. Borrow-safe: collect the action first.
            #[derive(Clone, Copy)]
            enum InitAction {
                None,
                UnaryInit(UOp),
                BinaryInit(BOp),
                BinScalarInit(BOp),
                CastInit(DType, DType),
                PackReconfig(OpId),
                MatmulInit(OpId, OpId, OpId),
                MatmulShortInit(OpId, OpId, OpId),
                TransposeInit(OpId, OpId),
                FillInit,
                ReduceInit(OpId, OpId, OpId, BOp, TileDim),
                BcastInit(BOp, TileDim, OpId, OpId),
                SigmoidInit(OpId),
                SiluInit(OpId),
            }
            let action = match &self.ops[op_id].op {
                Op::Unary { uop, .. } => InitAction::UnaryInit(*uop),
                Op::Binary { .. } => {
                    let Op::Binary { x, y, bop } = self.ops[op_id].op else {
                        unreachable!("tt_init_math: not a Binary");
                    };
                    // Integer-typed Binary is address math (index expressions,
                    // lane trees), not tile compute: no init.
                    if !self.dtype(op_id).is_float() {
                        InitAction::None
                    } else {
                        match (self.shape(x).len() == 2, self.shape(y).len() == 2) {
                            (true, true) => InitAction::BinaryInit(bop),
                            (true, false) => {
                                self.tt_check_scalar_lane(op_id, bop, false, y)?;
                                InitAction::BinScalarInit(bop)
                            }
                            (false, true) => {
                                self.tt_check_scalar_lane(op_id, bop, true, x)?;
                                InitAction::BinScalarInit(bop)
                            }
                            (false, false) => panic!("tt_init_math: tiled binary {op_id:?} with no tile lane"),
                        }
                    }
                }
                Op::Cast { .. } => {
                    let Op::Cast { x, dtype } = self.ops[op_id].op else {
                        unreachable!("tt_init_math: not a Cast");
                    };
                    InitAction::CastInit(self.dtype(x), dtype)
                }
                Op::Bitcast { .. } => InitAction::None,
                Op::Copy { src, dst, .. } => {
                    if self.is_circular_base(*src) && self.is_circular_base(*dst) {
                        InitAction::PackReconfig(*dst)
                    } else {
                        InitAction::None
                    }
                }
                // Staged traffic needs no engine init: the NOC calls
                // carry their addresses inline.
                Op::TT(TTOp::Copy { .. }) => InitAction::None,
                Op::Store { .. } => match self.tt_pack_store_cb(op_id) {
                    Some(cb) => InitAction::PackReconfig(cb),
                    None => InitAction::None,
                },
                Op::TT(TTOp::ReduceTile { rop, kind, cb_in, cb_sc, slot, .. }) => {
                    pending_acc = Some(*slot);
                    acc_values.clear();
                    InitAction::ReduceInit(*cb_in, *cb_sc, *slot, *rop, *kind)
                }
                Op::TT(TTOp::FillTile { .. }) => InitAction::FillInit,
                Op::TT(TTOp::LLKBcast { bop, kind, cb_a, cb_b, .. }) => InitAction::BcastInit(*bop, *kind, *cb_a, *cb_b),
                Op::TT(TTOp::SigmoidTile { cb_in, .. }) => InitAction::SigmoidInit(*cb_in),
                Op::TT(TTOp::SiluTile { cb_in, .. }) => InitAction::SiluInit(*cb_in),
                Op::TT(TTOp::MatmulTiles { cb_a, cb_b, .. }) => {
                    let out = first_pack.unwrap_or_else(|| panic!("tt_init_math: matmul {op_id:?} with no packed CB"));
                    InitAction::MatmulShortInit(*cb_a, *cb_b, out)
                }
                Op::TT(TTOp::LLK { .. }) => {
                    let Op::TT(TTOp::LLK { asm, ops }) = &self.ops[op_id].op else {
                        unreachable!("tt_init_math: not an LLK");
                    };
                    let text = asm.as_str();
                    if text.starts_with("matmul_tiles(") {
                        let out = first_pack.unwrap_or_else(|| panic!("tt_init_math: matmul {op_id:?} with no packed CB"));
                        InitAction::MatmulInit(ops[0], ops[1], out)
                    } else if text.starts_with("transpose_wh_tile(") {
                        let out = first_pack.unwrap_or_else(|| panic!("tt_init_math: transpose {op_id:?} with no packed CB"));
                        InitAction::TransposeInit(ops[0], out)
                    } else {
                        // User asm / startup template: no init.
                        InitAction::None
                    }
                }
                _ => InitAction::None,
            };
            // Forward acc-threading mark: a value computed from the
            // pending acc joins the set, so the pack check below is a
            // single lookup. Defs precede uses in walk order, so every
            // contributor is already marked when its consumer arrives.
            if let Some(acc) = pending_acc {
                let threads = match &self.ops[op_id].op {
                    Op::Load { src, .. } => *src == acc,
                    Op::Unary { x, .. } | Op::Cast { x, .. } | Op::Bitcast { x, .. } => acc_values.contains(x),
                    Op::Binary { x, y, .. } => acc_values.contains(x) || acc_values.contains(y),
                    Op::Asm { ops, .. } => ops.iter().copied().any(|o| !o.is_null() && acc_values.contains(&o)),
                    _ => false,
                };
                if threads {
                    acc_values.insert(op_id);
                }
            }
            // Unpack init at the load: a circular load consumed through
            // a copy (tiled elementwise SSA, a direct pack store, or a
            // fused-unary LLK whose render copies its feeder into a
            // fresh DST slot) is configured here, ahead of every copy
            // site in walk (hence runtime) order. Other fused consumers
            // (LLK provenance positions read straight from the CB,
            // user Asm, markers) and dead loads need none.
            if let Op::Load { src, .. } = self.ops[op_id].op {
                if matches!(self.ops[src].op, Op::Storage { scope: MemScope::Circular, .. })
                    && users.get(&op_id).is_some_and(|us| {
                        us.iter().any(|u| {
                            matches!(self.ops[*u].op, Op::Unary { .. } | Op::Binary { .. } | Op::Cast { .. } | Op::Bitcast { .. })
                                || matches!(self.ops[*u].op, Op::Store { src: x, .. } if x == op_id)
                                || match &self.ops[*u].op {
                                    // Fused unary calls copy their feeder
                                    // load into a fresh DST slot at the
                                    // call, so the feeder needs its unpack
                                    // init like any copied load. Any other
                                    // LLK reads CBs/storage slots directly.
                                    Op::TT(TTOp::LLK { asm, .. }) => {
                                        let text = asm.as_str();
                                        text.starts_with("sigmoid_tile(")
                                            || text.starts_with("silu_tile(")
                                            || text.starts_with("exp_tile(")
                                    }
                                    _ => false,
                                }
                        })
                    })
                {
                    // Copy unpack init: same shapes as the `tt_copy_init`
                    // ctors (which push back — wrong end for a pass).
                    let cb = src;
                    match unpack_src {
                        Some(prev) if prev == cb => {
                            self.insert_before(
                                op_id,
                                Op::Asm { asm: TinyString::new("copy_tile_init({0});"), ops: TinyVec::new(&[cb]) },
                            );
                        }
                        Some(prev) => {
                            self.insert_before(
                                op_id,
                                Op::Asm {
                                    asm: TinyString::new("copy_tile_to_dst_init_short_with_dt({0}, {1});"),
                                    ops: TinyVec::new(&[prev, cb]),
                                },
                            );
                            unpack_src = Some(cb);
                        }
                        None => {
                            self.insert_before(
                                op_id,
                                Op::Asm { asm: TinyString::new("copy_tile_init({0});"), ops: TinyVec::new(&[cb]) },
                            );
                            unpack_src = Some(cb);
                        }
                    }
                }
            }
            // The op init itself. Same templates as the `tt_*_init`
            // constructors (which push back — wrong end for a pass).
            let asm_before = |kernel: &mut Kernel, at: OpId, template: &str, ops: &[OpId]| {
                kernel.insert_before(at, Op::Asm { asm: TinyString::new(template), ops: TinyVec::new(ops) });
            };
            match action {
                InitAction::None => {}
                InitAction::UnaryInit(uop) => {
                    let init = match uop {
                        UOp::Neg => "negative_tile_init();",
                        UOp::BitNot => "bitwise_not_tile_init();",
                        UOp::Exp2 => "exp2_tile_init();",
                        UOp::Log2 => "log_with_base_tile_init();",
                        UOp::Reciprocal => "recip_tile_init();",
                        UOp::Sqrt => "sqrt_tile_init();",
                        UOp::Rsqrt => "rsqrt_tile_init();",
                        UOp::Sin => "sin_tile_init();",
                        UOp::Cos => "cos_tile_init();",
                        UOp::Floor | UOp::Trunc => "rounding_op_tile_init();",
                        UOp::Abs => "abs_tile_init();",
                        UOp::Not => "logical_not_tile_init();",
                    };
                    asm_before(self, op_id, init, &[]);
                }
                InitAction::BinaryInit(bop) => {
                    let init = match bop {
                        BOp::Add => "add_binary_tile_init();",
                        BOp::Sub => "sub_binary_tile_init();",
                        BOp::Mul => "mul_binary_tile_init();",
                        BOp::Div => "div_binary_tile_init();",
                        BOp::Max => "binary_max_tile_init();",
                        BOp::BitShiftLeft | BOp::BitShiftRight => "binary_shift_tile_init();",
                        _ => panic!("tt_init_math: {bop:?} has no init call"),
                    };
                    asm_before(self, op_id, init, &[]);
                }
                InitAction::BinScalarInit(bop) => {
                    let init = match bop {
                        BOp::Add | BOp::Sub | BOp::Mul | BOp::Div => "binop_with_scalar_tile_init();",
                        BOp::BitShiftLeft => "left_shift_tile_init();",
                        BOp::BitShiftRight => "right_shift_tile_init();",
                        _ => panic!("tt_init_math: scalar init for {bop:?} has no init call"),
                    };
                    asm_before(self, op_id, init, &[]);
                }
                InitAction::CastInit(in_dtype, out_dtype) => {
                    /// TT `DataFormat` code for a dtype on the tile path (the
                    /// `typecast_tile_init<in, out>` template args). This is NOT the CB
                    /// descriptor code. An unmappable dtype panics — mirrors the old
                    /// render error at the constructor, the exact spot.
                    fn tt_tile_fmt(dtype: DType) -> u32 {
                        match dtype {
                            DType::F32 => 0,
                            DType::F16 | DType::BF16 => 5,
                            DType::I32 => 8,
                            DType::U16 => 9,
                            DType::I8 => 14,
                            DType::U32 => 24,
                            DType::F8E4M3 => 26,
                            DType::U8 => 30,
                            dt => panic!("tt_tile_fmt: dtype {dt:?} has no tt tile format"),
                        }
                    }
                    let template = format!("typecast_tile_init<{}, {}>();", tt_tile_fmt(in_dtype), tt_tile_fmt(out_dtype));
                    asm_before(self, op_id, &template, &[]);
                }
                InitAction::PackReconfig(cb) => {
                    asm_before(self, op_id, "pack_reconfig_data_format({0});", &[cb]);
                    // Packing the pending acc closes it.
                    if pending_acc.is_some() {
                        let Op::Store { src: x, .. } = self.ops[op_id].op else {
                            unreachable!("tt_init_math: pack is a Store");
                        };
                        if acc_values.contains(&x) {
                            let unlock = last_math_unlock
                                .unwrap_or_else(|| panic!("tt_init_math: pack {op_id:?} without an open MathUnlock"));
                            self.insert_before(unlock, Op::TT(TTOp::ReduceUninit));
                            pending_acc = None;
                            acc_values.clear();
                        }
                    }
                }
                InitAction::MatmulInit(a, b, out) => {
                    unpack_src = Some(a);
                    asm_before(self, op_id, "mm_init({0}, {1}, {2});", &[a, b, out]);
                }
                InitAction::MatmulShortInit(a, b, out) => {
                    unpack_src = Some(a);
                    // mm_init_short = state_configure(in1, in0) [the
                    // Reverse mapping] + math/unpack matmul-mode init,
                    // all internally thread-dispatched, so it compiles
                    // in the replicated unpack/pack sources too.
                    asm_before(self, op_id, "mm_init_short({0}, {1}, 0);", &[a, b, out]);
                }
                InitAction::TransposeInit(cb, out) => {
                    asm_before(self, op_id, "transpose_wh_init({0}, {1});", &[cb, out]);
                }
                InitAction::ReduceInit(ci, cs, acc, rop, kind) => {
                    let op_name = match rop {
                        BOp::Max => "PoolType::MAX",
                        BOp::Add => "PoolType::SUM",
                        _ => panic!("tt_init_math: reduce op {rop:?} has no init call"),
                    };
                    let dim_name = match kind {
                        TileDim::Row => "ReduceDim::REDUCE_ROW",
                        TileDim::Col => "ReduceDim::REDUCE_COL",
                        TileDim::Scalar => "ReduceDim::REDUCE_SCALAR",
                    };
                    let template = format!("reduce_init<{op_name}, {dim_name}>({{0}}, {{1}}, {{2}});");
                    asm_before(self, op_id, &template, &[ci, cs, acc]);
                }
                InitAction::BcastInit(bop, kind, cb_a, cb_b) => {
                    let init = match (bop, kind) {
                        (BOp::Add, TileDim::Row) => "add_bcast_rows_init_short",
                        (BOp::Add, TileDim::Col) => "add_bcast_cols_init_short",
                        (BOp::Add, TileDim::Scalar) => "add_bcast_scalar_init_short",
                        (BOp::Sub, TileDim::Row) => "sub_bcast_rows_init_short",
                        (BOp::Sub, TileDim::Col) => "sub_bcast_cols_init_short",
                        (BOp::Sub, TileDim::Scalar) => "sub_tiles_bcast_scalar_init_short",
                        (BOp::Mul, TileDim::Row) => "mul_bcast_rows_init_short",
                        (BOp::Mul, TileDim::Col) => "mul_bcast_cols_init_short",
                        (BOp::Mul, TileDim::Scalar) => "mul_tiles_bcast_scalar_init_short",
                        _ => panic!("tt_init_math: broadcast ({bop:?}, {kind:?}) has no init call"),
                    };
                    let template = format!("{init}({{0}}, {{1}});");
                    asm_before(self, op_id, &template, &[cb_a, cb_b]);
                }
                InitAction::FillInit => {
                    asm_before(self, op_id, "fill_tile_init();", &[]);
                }
                InitAction::SigmoidInit(cb) => {
                    // The render copies the CB tile into DST before the
                    // call: configure the unpacker for that copy, then
                    // the MATH init. (The fused LLK's DRAM feeder load is
                    // dead by now, so the load-scan unpack init that
                    // served the old template form does not fire.)
                    asm_before(self, op_id, "copy_tile_init({0});", &[cb]);
                    asm_before(self, op_id, "sigmoid_tile_init();", &[]);
                }
                InitAction::SiluInit(cb) => {
                    asm_before(self, op_id, "copy_tile_init({0});", &[cb]);
                    asm_before(self, op_id, "silu_tile_init();", &[]);
                }
            }
            op_id = next;
        }

        // Sticky-init hoist (old `hoist_dedup_inits`): an `mm_init`
        // programs the MATH unit once per config (sticky) — re-issuing
        // it per loop trip is at best redundant. Bubble each one
        // outward across enclosing loops whose bodies hold no lock ops
        // (a lock in the body means per-trip inits; crossing it would
        // move config out of its region). Per-call unpacker/packer inits
        // (copy, transpose, reduce, bcast, reconfig) stay per-trip, like
        // the old pass. Never crosses a section marker. Lands after a
        // hoisted acquire (old acquire-then-init order).
        let mut inits = Vec::new();
        let mut scan = self.head;
        while !scan.is_null() {
            if let Op::Asm { asm, .. } = self.at(scan) {
                if asm.as_str().starts_with("mm_init(") {
                    inits.push(scan);
                }
            }
            scan = self.next_op(scan);
        }
        for mut init in inits {
            loop {
                // Innermost enclosing Loop (backward nesting scan).
                let mut l = self.ops[init].prev;
                let mut depth = 0u32;
                let mut enclosing = None;
                while !l.is_null() {
                    match self.at(l) {
                        Op::EndLoop => depth += 1,
                        Op::Loop { .. } => {
                            if depth == 0 {
                                enclosing = Some(l);
                                break;
                            }
                            depth -= 1;
                        }
                        Op::TT(TTOp::EndReader) | Op::TT(TTOp::EndCompute) => break,
                        _ => {}
                    }
                    l = self.ops[l].prev;
                }
                let l = match enclosing {
                    Some(l) => l,
                    None => break,
                };
                // L's matching EndLoop (forward nesting scan).
                let mut e = self.next_op(l);
                let mut d = 1u32;
                while !e.is_null() && d > 0 {
                    match self.at(e) {
                        Op::Loop { .. } => d += 1,
                        Op::EndLoop => d -= 1,
                        _ => {}
                    }
                    if d > 0 {
                        e = self.next_op(e);
                    }
                }
                if e.is_null() {
                    break;
                }
                // A lock in the body pins the init per-trip.
                let mut s = self.next_op(l);
                let mut has_lock = false;
                while s != e {
                    if matches!(
                        self.at(s),
                        Op::TT(TTOp::MathLock) | Op::TT(TTOp::MathUnlock) | Op::TT(TTOp::PackLock) | Op::TT(TTOp::PackUnlock)
                    ) {
                        has_lock = true;
                        break;
                    }
                    s = self.next_op(s);
                }
                if has_lock {
                    break;
                }
                let op = self.at(init).clone();
                self.remove_op(init);
                let before = self.ops[l].prev;
                init = if !before.is_null() && matches!(self.at(before), Op::TT(TTOp::MathLock)) {
                    self.insert_after(before, op)
                } else {
                    self.insert_before(l, op)
                };
            }
        }

        self.verify();
        Ok(())
    }

    /// Pop placement + FIFO accounting: ports the old `place_pops`
    /// (compute `Strict` pops, writer drain pops) with the old
    /// `Counted` matmul-side sharing, keyed on the LLK provenance
    /// loads (`tt_storage` records them past the template range).
    /// Ends with the old FIFO simulation (reserve/wait balance per
    /// section, pushed==popped program-wide per CB, producer-past-
    /// depth, symbolic-loop traffic) — a hang or stall miscompiles to
    /// a loud panic here, never a wedged launch.
    ///
    /// Rules, in IR order:
    /// - every `Copy` with a circular src (writer drain, CB→CB move)
    ///   pops one src page immediately after — each copy is an
    ///   independent consume. A CB→CB copy outside the compute
    ///   section is malformed (nothing there can unpack);
    /// - every `LLK` pops each distinct input CB once per call, and
    ///   shared feeder pages pop when their last call runs (provenance
    ///   `(cb, load)` remaining counts — the old `Counted` case);
    /// - a circular load consumed through SSA (`Unary`/`Binary`/
    ///   `Cast`/`Bitcast`, a pack store, user `Asm`) pops once after
    ///   its last same-section consumer (the fused call consumes
    ///   through provenance);
    /// - a circular load with no users at all is a drain pop: pop
    ///   after the load itself.
    pub fn tt_place_pops(&mut self) {
        // Users + positions + sections.
        let mut users: Map<OpId, Vec<OpId>> = Map::default();
        let mut pos_index: Map<OpId, usize> = Map::default();
        let mut op_section: Map<OpId, u8> = Map::default();
        let mut section = 0u8;
        let mut scan = self.head;
        let mut idx = 0usize;
        while !scan.is_null() {
            pos_index.insert(scan, idx);
            idx += 1;
            if matches!(self.at(scan), Op::TT(TTOp::EndReader)) {
                section = 1;
            } else if matches!(self.at(scan), Op::TT(TTOp::EndCompute)) {
                section = 2;
            }
            op_section.insert(scan, section);
            for p in self.at(scan).parameters() {
                if !p.is_null() {
                    users.entry(p).or_default().push(scan);
                }
            }
            scan = self.next_op(scan);
        }

        // Indexed-window batching (old place_waits/place_pops parity):
        // same decision as `tt_sync_cbs` (which skipped the per-trip
        // waits for these reads): one PopFront of trips × reads after
        // the loop instead of per-trip n:1 pops. Re-derived here on
        // the wait-carrying IR — sync ops are not data traffic, so the
        // decision is identical. See `tt_sync_cbs` for the full rule.
        let mut batched: Map<(OpId, OpId), u32> = Map::default();
        let mut batched_read: Set<OpId> = Set::default();
        scan = self.head;
        while !scan.is_null() {
            let (read_op, gep_index, cb) = match self.ops[scan].op {
                Op::Load { src: cb, src_idx: index, .. }
                    if matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. }) =>
                {
                    (scan, index, cb)
                }
                Op::Copy { src: cb, src_idx: index, .. }
                    if matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. }) =>
                {
                    (scan, index, cb)
                }
                _ => {
                    scan = self.next_op(scan);
                    continue;
                }
            };
            // Head slot (const 0) reads the CB front: per-trip syncs
            // stay correct. Anything else is window-relative.
            if self.resolve_const(gep_index).and_then(|c| c.as_dim()) == Some(0) {
                scan = self.next_op(scan);
                continue;
            }
            // Innermost enclosing loop of the read.
            let mut inner: Option<OpId> = None;
            let mut cur = self.prev_op(scan);
            let mut depth = 0u32;
            while !cur.is_null() {
                match self.ops[cur].op {
                    Op::EndLoop => depth += 1,
                    Op::Loop { .. } => {
                        if depth == 0 {
                            inner = Some(cur);
                            break;
                        }
                        depth -= 1;
                    }
                    _ => {}
                }
                cur = self.prev_op(cur);
            }
            let Some(loop_op) = inner else {
                scan = self.next_op(scan);
                continue;
            };
            let Op::Loop { len } = self.ops[loop_op].op else {
                unreachable!("tt_place_pops: batch loop is a Loop");
            };
            let Some(trips) =
                self.resolve_const(len).and_then(|c| c.as_dim()).and_then(|t| u32::try_from(t).ok()).filter(|t| *t > 0)
            else {
                scan = self.next_op(scan);
                continue;
            };
            // Matching EndLoop of `loop_op`.
            let mut end = self.next_op(loop_op);
            let mut depth = 0u32;
            while !end.is_null() {
                match self.ops[end].op {
                    Op::Loop { .. } => depth += 1,
                    Op::EndLoop => {
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                    }
                    _ => {}
                }
                end = self.next_op(end);
            }
            if end.is_null() {
                scan = self.next_op(scan);
                continue;
            }
            // Span scan: every data access to `cb` must be a direct
            // qualifying indexed read, else the window cannot be held.
            let mut direct = 0u32;
            let mut held = true;
            let mut cur = self.next_op(loop_op);
            let mut depth = 0u32;
            while cur != end {
                match self.ops[cur].op {
                    Op::Loop { .. } => depth += 1,
                    Op::EndLoop => depth -= 1,
                    Op::Load { src: c, src_idx: index, .. } => {
                        if c == cb {
                            let indexed = self.resolve_const(index).and_then(|c| c.as_dim()) != Some(0);
                            let ok_users = users.get(&cur).is_some_and(|us| {
                                us.iter().all(|u| match &self.ops[*u].op {
                                    Op::Store { .. } | Op::Asm { .. } => true,
                                    op => {
                                        self.shape(*u).len() == 2
                                            && matches!(
                                                op,
                                                Op::Unary { .. } | Op::Binary { .. } | Op::Cast { .. } | Op::Bitcast { .. }
                                            )
                                    }
                                })
                            });
                            if depth == 0 && indexed && ok_users {
                                direct += 1;
                            } else {
                                held = false;
                                break;
                            }
                        }
                    }
                    Op::Store { dst, .. } => {
                        if dst == cb {
                            held = false;
                            break;
                        }
                    }
                    Op::Copy { src, src_idx: index, dst, .. } => {
                        let src_hit = if src == cb { Some(index) } else { None };
                        let dst_hit = if dst == cb { Some(dst) } else { None };
                        match (src_hit, dst_hit) {
                            (None, None) => {}
                            (Some(index), None) => {
                                let indexed = self.resolve_const(index).and_then(|c| c.as_dim()) != Some(0);
                                if depth == 0 && indexed {
                                    direct += 1;
                                } else {
                                    held = false;
                                    break;
                                }
                            }
                            _ => {
                                held = false;
                                break;
                            }
                        }
                    }
                    Op::TT(TTOp::LLK { .. })
                    | Op::TT(TTOp::ReduceTile { .. })
                    | Op::TT(TTOp::LLKBcast { .. })
                    | Op::TT(TTOp::MatmulTiles { .. })
                    | Op::TT(TTOp::SigmoidTile { .. })
                    | Op::TT(TTOp::SiluTile { .. }) => {
                        let mut feeds = false;
                        for o in self.at(cur).parameters() {
                            if o.is_null() {
                                continue;
                            }
                            if let Op::Load { src: c, .. } = self.ops[o].op
                                && c == cb
                            {
                                feeds = true;
                                break;
                            }
                            // Storage-only LLKs name their CBs directly.
                            if matches!(self.ops[o].op, Op::Storage { scope: MemScope::Circular, .. }) && o == cb {
                                feeds = true;
                                break;
                            }
                        }
                        if feeds {
                            held = false;
                            break;
                        }
                    }
                    _ => {}
                }
                cur = self.next_op(cur);
            }
            if !held || direct == 0 {
                scan = self.next_op(scan);
                continue;
            }
            let Some(total) = trips.checked_mul(direct) else {
                scan = self.next_op(scan);
                continue;
            };
            let Op::Storage { len, .. } = self.ops[cb].op else {
                unreachable!("tt_place_pops: batch cb is a Storage");
            };
            if len <= 0 {
                scan = self.next_op(scan);
                continue;
            }
            let depth_tiles = (len / 1024) as u32;
            if total == 0 || total > depth_tiles || total > 255 {
                scan = self.next_op(scan);
                continue;
            }
            batched.entry((loop_op, cb)).or_insert(total);
            batched_read.insert(read_op);
            scan = self.next_op(scan);
        }

        // Remaining pages per CB: each lowered LLK consumes one page
        // of each Circular CB operand. Storage-only operands (no
        // feeder loads to count through).
        let mut remaining: Map<OpId, u32> = Map::default();
        let mut walk = self.head;
        while !walk.is_null() {
            if matches!(
                self.ops[walk].op,
                Op::TT(TTOp::LLK { .. })
                    | Op::TT(TTOp::ReduceTile { .. })
                    | Op::TT(TTOp::LLKBcast { .. })
                    | Op::TT(TTOp::MatmulTiles { .. })
                    | Op::TT(TTOp::SigmoidTile { .. })
                    | Op::TT(TTOp::SiluTile { .. })
            ) {
                for o in self.at(walk).parameters() {
                    if o.is_null() {
                        continue;
                    }
                    if matches!(self.ops[o].op, Op::Storage { scope: MemScope::Circular, .. }) {
                        *remaining.entry(o).or_insert(0) += 1;
                    }
                }
            }
            walk = self.next_op(walk);
        }

        // First WaitFront per CB: dead-load drain anchor (see below).
        let mut first_wait: Map<OpId, OpId> = Map::default();
        let mut deferred_drains: Vec<(OpId, OpId)> = Vec::new();
        walk = self.head;
        while !walk.is_null() {
            if let Op::TT(TTOp::WaitFront { cb, .. }) = self.ops[walk].op {
                first_wait.entry(cb).or_insert(walk);
            }
            walk = self.next_op(walk);
        }

        // Last same-section non-LLK, non-marker consumer per load.
        let mut last_use: Map<OpId, OpId> = Map::default();
        for (load, us) in users.iter() {
            if !matches!(self.ops[*load].op, Op::Load { .. }) {
                continue;
            }
            let lsec = op_section[load];
            let mut best: Option<OpId> = None;
            for u in us {
                if op_section[u] != lsec {
                    continue;
                }
                if matches!(
                    self.ops[*u].op,
                    Op::TT(TTOp::LLK { .. })
                        | Op::TT(TTOp::ReduceTile { .. })
                        | Op::TT(TTOp::LLKBcast { .. })
                        | Op::TT(TTOp::MatmulTiles { .. })
                        | Op::TT(TTOp::SigmoidTile { .. })
                        | Op::TT(TTOp::SiluTile { .. })
                ) {
                    continue;
                }
                if best.is_none_or(|b| pos_index[u] > pos_index[&b]) {
                    best = Some(*u);
                }
            }
            if let Some(b) = best {
                last_use.insert(*load, b);
            }
        }

        // Insertion walk.
        let mut op_id = self.head;
        let mut in_compute = false;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if matches!(self.ops[op_id].op, Op::TT(TTOp::EndReader)) {
                in_compute = true;
            } else if matches!(self.ops[op_id].op, Op::TT(TTOp::EndCompute)) {
                in_compute = false;
            }
            // Pops after this op.
            let mut pops: Vec<OpId> = Vec::new();
            match self.ops[op_id].op {
                Op::Copy { src, .. } => {
                    if matches!(self.ops[src].op, Op::Storage { scope: MemScope::Circular, .. }) {
                        if !in_compute {
                            // Writer drain (reader never holds a
                            // circular src — its copies publish).
                            // A CB→CB move outside compute cannot
                            // unpack — malformed.
                            let is_drain = matches!(self.ops[op_id].op, Op::Copy { dst, .. } if !self.is_circular_base(dst));
                            if !is_drain {
                                panic!("tt_place_pops: CB→CB copy {op_id:?} outside the compute section");
                            }
                        }
                        // Indexed-window reads carry no per-trip pop:
                        // the bracket pop covers the window.
                        if !batched_read.contains(&op_id) {
                            pops.push(src);
                        }
                    }
                }
                Op::TT(TTOp::Copy { src, dst, .. }) => {
                    // Staged drain (CB→DRAM): pop the CB side. Publishes
                    // hold no pop here — the LLK consumption above pops
                    // input CBs after their last consuming call.
                    let src_cb = matches!(self.ops[src].op, Op::Storage { scope: MemScope::Circular, .. });
                    let dst_cb = matches!(self.ops[dst].op, Op::Storage { scope: MemScope::Circular, .. });
                    match (src_cb, dst_cb) {
                        (true, false) => {
                            if !batched_read.contains(&op_id) {
                                pops.push(src);
                            }
                        }
                        (false, true) => {}
                        _ => panic!("tt_place_pops: staged copy {op_id:?} is not DRAM↔CB traffic"),
                    }
                }
                Op::TT(TTOp::LLK { .. })
                | Op::TT(TTOp::ReduceTile { .. })
                | Op::TT(TTOp::LLKBcast { .. })
                | Op::TT(TTOp::MatmulTiles { .. })
                | Op::TT(TTOp::SigmoidTile { .. })
                | Op::TT(TTOp::SiluTile { .. }) => {
                    let mut seen: Vec<OpId> = Vec::new();
                    for o in self.at(op_id).parameters() {
                        if o.is_null() {
                            continue;
                        }
                        if matches!(self.ops[o].op, Op::Storage { scope: MemScope::Circular, .. }) && !seen.contains(&o) {
                            seen.push(o);
                        }
                    }
                    for cb in seen {
                        let left = remaining
                            .get_mut(&cb)
                            .unwrap_or_else(|| panic!("tt_place_pops: LLK {op_id:?} consumes uncounted page {cb:?}"));
                        if *left == 0 {
                            panic!("tt_place_pops: use beyond counted total on CB{cb:?}");
                        }
                        *left -= 1;
                        if *left == 0 {
                            pops.push(cb);
                        }
                    }
                }
                _ => {}
            }
            // SSA last-use pops landing here.
            for (load, user) in last_use.iter() {
                if *user == op_id && !batched_read.contains(load) {
                    let Op::Load { src, .. } = self.ops[*load].op else {
                        unreachable!("tt_place_pops: last-use of a Load");
                    };
                    if matches!(self.ops[src].op, Op::Storage { scope: MemScope::Circular, .. }) {
                        pops.push(src);
                    }
                }
            }
            // Dead-load drain pops anchor after the CB's first wait, not
            // at the dead load: the orphan sits in the load preamble,
            // before any wait has fired, so a pop there starves. The
            // first wait has just made a page available that no pop can
            // yet have consumed. A CB with no wait keeps the old
            // at-load pop (fifo stays the loud guard for that shape).
            if matches!(self.ops[op_id].op, Op::Load { .. }) && !users.contains_key(&op_id) && !batched_read.contains(&op_id) {
                let Op::Load { src, .. } = self.ops[op_id].op else {
                    unreachable!("tt_place_pops: dead op is a Load");
                };
                if matches!(self.ops[src].op, Op::Storage { scope: MemScope::Circular, .. }) {
                    match first_wait.get(&src) {
                        Some(wait) => deferred_drains.push((*wait, src)),
                        None => pops.push(src),
                    }
                }
            }
            for cb in pops {
                self.insert_after(op_id, Op::TT(TTOp::PopFront { cb, n: 1 }));
            }
            op_id = next;
        }
        if remaining.values().any(|r| *r != 0) {
            panic!("tt_place_pops: partially consumed pages at end of stream");
        }
        for (wait, cb) in deferred_drains {
            self.insert_after(wait, Op::TT(TTOp::PopFront { cb, n: 1 }));
        }

        // Bracket pops for indexed windows: one PopFront of the whole
        // window after the loop (the per-trip pops above were skipped
        // for these reads; the bracket wait came from `tt_sync_cbs`).
        let done: Vec<((OpId, OpId), u32)> = batched.iter().map(|(k, v)| (*k, *v)).collect();
        for ((loop_op, cb), total) in done {
            // Matching EndLoop of `loop_op`.
            let mut end = self.next_op(loop_op);
            let mut depth = 0u32;
            while !end.is_null() {
                match self.ops[end].op {
                    Op::Loop { .. } => depth += 1,
                    Op::EndLoop => {
                        if depth == 0 {
                            break;
                        }
                        depth -= 1;
                    }
                    _ => {}
                }
                end = self.next_op(end);
            }
            if end.is_null() {
                panic!("tt_place_pops: batch loop {loop_op:?} has no EndLoop");
            }
            self.insert_after(end, Op::TT(TTOp::PopFront { cb, n: total as u8 }));
        }

        // Pop-relative rebase (consume-side addressing): Metal
        // advances a CB's read pointer on every PopFront, so
        // Load-src and Copy-src index expressions past pops address
        // pop-relative pages. Rebase const indices by popped
        // pages (x1024 elems). Copy-dst (reserve-relative) and
        // Store-dst (push-relative) index expressions are other traffic and
        // stay, as does the whole reader section (no pops there).
        // Loop-varying indices stay: bracket windows hold the
        // pointer still mid-loop, so only const indices rebase.
        // A shared index expression is cloned, never mutated in place.
        let mut popped: Map<OpId, u64> = Map::default();
        section = 0;
        scan = self.head;
        while !scan.is_null() {
            let next = self.next_op(scan);
            match self.ops[scan].op {
                Op::TT(TTOp::EndReader) => section = 1,
                Op::TT(TTOp::EndCompute) => section = 2,
                Op::TT(TTOp::PopFront { cb, n }) => {
                    if section != 0 {
                        *popped.entry(cb).or_insert(0) += u64::from(n);
                    }
                }
                _ => {}
            }
            let op = self.ops[scan].clone().op;
            if section != 0 {
                if let Op::Load { src: base, src_idx: index, .. } | Op::Copy { src: base, src_idx: index, .. } = op
                    && matches!(self.ops[base].op, Op::Storage { scope: MemScope::Circular, .. })
                    && popped.get(&base).copied().unwrap_or(0) > 0
                {
                    let pages = popped[&base];
                    let dim = self
                        .resolve_const(index)
                        .and_then(|c| c.as_dim())
                        .unwrap_or_else(|| panic!("tt_place_pops: rebase of non-const CB index at {scan:?}"));
                    if dim < 0 {
                        panic!("tt_place_pops: negative CB index at {scan:?}");
                    }
                    let rebased = (dim as u64)
                        .checked_sub(pages * 1024)
                        .unwrap_or_else(|| panic!("tt_place_pops: rebase underflows CB{base:?} at {scan:?}"));
                    if rebased != dim as u64 {
                        let nc = self.insert_const_idx_before(scan, rebased as i64);
                        match &mut self.ops[scan].op {
                            Op::Load { src_idx, .. } => *src_idx = nc,
                            Op::Copy { src_idx, .. } => *src_idx = nc,
                            _ => unreachable!("tt_place_pops: rebase src vanished"),
                        }
                    }
                }
            }
            scan = next;
        }

        self.tt_fifo_check();
        self.verify();
    }

    /// DST lock firewall: port of the tenstorrent2 stream verifier's
    /// lock half (the FIFO half already lives in `tt_fifo_check`;
    /// both died with tenstorrent2.rs in 67bb272a "cleanup" and only
    /// FIFO was ported back). Runs at the end of the TT
    /// device-epilogue branch: whatever lowering did, lock pairing
    /// must hold, or the sim wedges at launch with no message.
    /// Loud panic at the exact op.
    ///
    /// Checks (compute section only): MathLock acquires from Unlocked
    /// (or Pack), MathUnlock commits a held MATH lock, PackLock waits
    /// from Unlocked, PackUnlock releases a held PACK lock; no lock
    /// op outside the compute section; no section ends (or stream
    /// ends) with the DST held. Lock ORDER (commit-before-wait) is
    /// NOT checked here — known gap, needs its own rule.
    pub fn tt_verify(&self) {
        #[derive(PartialEq)]
        enum Lock {
            Unlocked,
            Math,
            Pack,
        }
        let mut lock = Lock::Unlocked;
        let mut section = 0usize;
        let mut seen_reader = false;
        let mut id = self.head;
        while !id.is_null() {
            match self.at(id) {
                Op::TT(TTOp::EndReader) => {
                    assert!(section == 0, "tt_verify: duplicate EndReader at {id:?}");
                    assert!(lock == Lock::Unlocked, "tt_verify: reader section ends with DST locked at {id:?}");
                    section = 1;
                    seen_reader = true;
                }
                Op::TT(TTOp::EndCompute) => {
                    assert!(seen_reader, "tt_verify: EndCompute without EndReader at {id:?}");
                    assert!(section == 1, "tt_verify: duplicate EndCompute at {id:?}");
                    assert!(lock == Lock::Unlocked, "tt_verify: compute section ends with DST locked at {id:?}");
                    section = 2;
                }
                Op::TT(TTOp::MathLock) => {
                    assert!(section == 1, "tt_verify: DST lock outside compute at {id:?}");
                    assert!(lock == Lock::Unlocked || lock == Lock::Pack, "tt_verify: acquire with DST already held at {id:?}");
                    lock = Lock::Math;
                }
                Op::TT(TTOp::MathUnlock) => {
                    assert!(lock == Lock::Math, "tt_verify: commit without MATH lock at {id:?}");
                    lock = Lock::Unlocked;
                }
                Op::TT(TTOp::PackLock) => {
                    assert!(section == 1, "tt_verify: DST lock outside compute at {id:?}");
                    assert!(lock == Lock::Unlocked, "tt_verify: pack wait without release at {id:?}");
                    lock = Lock::Pack;
                }
                Op::TT(TTOp::PackUnlock) => {
                    assert!(lock == Lock::Pack, "tt_verify: release without PACK lock at {id:?}");
                    lock = Lock::Unlocked;
                }
                _ => {}
            }
            id = self.next_op(id);
        }
        assert!(lock == Lock::Unlocked, "tt_verify: stream ends with DST locked");
    }

    /// 64-byte congruence for one DRAM-touching copy: the board
    /// crashes on incongruent NOC addresses, so both ends must agree
    /// mod 64. Offsets are index x unit x elsize: tile transfers
    /// count slots (1024 els), vector/scalar count elements — the
    /// same units render_copy emits. Affine indices prove by stride
    /// (get_strides, bare var = stride 1); anything else is
    /// unprovable and rejected, never silently skipped.
    fn fifo_check_copy(&self, op_id: OpId, src: OpId, src_idx: OpId, dst: OpId, dst_idx: OpId, unit: i64) {
        let is_dram = |base: OpId| matches!(self.at(base), Op::Param { kind: ParamKind::Global | ParamKind::GlobalMut, .. });
        if !(is_dram(src) || is_dram(dst)) {
            return;
        }
        let elsize = |base: OpId| match self.at(base) {
            Op::Param { dtype, .. } | Op::Storage { dtype, .. } => {
                let bits = dtype.bit_size();
                if bits % 8 != 0 {
                    panic!("tt_fifo_check: copy {op_id:?} side {base:?} has sub-byte dtype {dtype:?}");
                }
                (bits / 8) as i64
            }
            other => panic!("tt_fifo_check: copy {op_id:?} side {base:?} is not a buffer: {other:?}"),
        };
        // Affine decomposition: per-var element strides plus the
        // const base (the NULL entry); a null index is const 0.
        let decomp = |idx: OpId| -> Map<OpId, i64> {
            if idx.is_null() {
                return Map::default();
            }
            self.get_strides(idx)
                .unwrap_or_else(|| panic!("tt_fifo_check: copy {op_id:?} has unprovable index {idx:?}"))
                .into_iter()
                .map(|(v, (_, st))| (v, st))
                .collect()
        };
        let sm = decomp(src_idx);
        let dm = decomp(dst_idx);
        let es = elsize(src);
        let ed = elsize(dst);
        let check = |var: &str, ss: i64, ds: i64| {
            if (ss * unit * es - ds * unit * ed).rem_euclid(64) != 0 {
                panic!(
                    "tt_fifo_check: copy {op_id:?} is 64-byte incongruent at {var}: src {ss} vs dst {ds} (unit {unit}, elsize {es}/{ed})"
                );
            }
        };
        for (v, ss) in sm.iter() {
            if v.is_null() {
                continue;
            }
            check(&std::format!("{v:?}"), *ss, dm.get(v).copied().unwrap_or(0));
        }
        for (v, ds) in dm.iter() {
            if v.is_null() || sm.contains_key(v) {
                continue;
            }
            check(&std::format!("{v:?}"), 0, *ds);
        }
        let bs = sm.get(&OpId::NULL).copied().unwrap_or(0);
        let bd = dm.get(&OpId::NULL).copied().unwrap_or(0);
        check("base", bs, bd);
    }

    /// FIFO simulation over the placed sync ops (ports the old
    /// `verify` FIFO block): per-CB reserve/wait balance at section
    /// ends, pushed==popped program-wide per CB, producer-past-depth,
    /// and the symbolic-loop traffic error. Loop trip counts multiply
    /// straight-line counts; boolean loops (conditionals) don't;
    /// non-const non-bool loop lengths are symbolic and reject any
    /// sync traffic under them. Read-only; panics loudly, never
    /// launches a stalled kernel.
    pub(crate) fn tt_fifo_check(&self) {
        #[derive(Default)]
        struct Fifo {
            reserved: u64,
            avail: u64,
            waited: u64,
            pushed: u64,
            popped: u64,
        }
        let mut fifos: Map<OpId, Fifo> = Map::default();
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Frame {
            Trip(u64),
            Symbolic,
            Conditional,
        }
        let mut mult: u64 = 1;
        let mut stack: Vec<Frame> = Vec::new();
        let mut sym_loops: u32 = 0;
        let mut run_cb: Option<OpId> = None;
        let mut run_tiles: u64 = 0;
        let depth_of = |kernel: &Kernel, cb: OpId| -> u64 {
            match kernel.at(cb) {
                Op::Storage { len, .. } => (*len as u64) / 1024,
                other => panic!("tt_fifo_check: CB {cb:?} is not a storage op, got {other:?}"),
            }
        };
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::TT(TTOp::EndReader) | Op::TT(TTOp::EndCompute) => {
                    for (cb, f) in fifos.iter() {
                        if f.reserved != 0 {
                            panic!("tt_fifo_check: CB{cb:?} reserve open at section end");
                        }
                        if f.waited != 0 {
                            panic!("tt_fifo_check: CB{cb:?} wait open at section end");
                        }
                    }
                    if let Some(prev) = run_cb {
                        if run_tiles > depth_of(self, prev) {
                            panic!(
                                "tt_fifo_check: producer pushes {run_tiles} tiles to CB{prev:?} (depth {}) before any other CB",
                                depth_of(self, prev)
                            );
                        }
                        run_cb = None;
                        run_tiles = 0;
                    }
                }
                Op::Loop { len } => {
                    if let Op::Const(c) = self.at(len) {
                        match c.as_dim() {
                            Some(t) => {
                                mult *= t as u64;
                                stack.push(Frame::Trip(t as u64));
                            }
                            None => panic!("tt_fifo_check: negative loop length under {op_id:?}"),
                        }
                    } else if self.dtype(len) == DType::Bool {
                        stack.push(Frame::Conditional);
                    } else {
                        sym_loops += 1;
                        stack.push(Frame::Symbolic);
                    }
                }
                Op::EndLoop => match stack.pop() {
                    Some(Frame::Trip(t)) => mult /= t,
                    Some(Frame::Symbolic) => sym_loops -= 1,
                    Some(Frame::Conditional) => {}
                    None => panic!("tt_fifo_check: EndLoop without Loop at {op_id:?}"),
                },
                Op::TT(TTOp::ReserveBack { cb, n }) => {
                    if sym_loops != 0 {
                        panic!("tt_fifo_check: FIFO traffic under a symbolic loop at {op_id:?}");
                    }
                    let f = fifos.entry(cb).or_default();
                    if f.reserved != 0 || f.waited != 0 {
                        panic!("tt_fifo_check: reserve CB{cb:?} with open transaction at {op_id:?}");
                    }
                    f.reserved = u64::from(n) * mult;
                }
                Op::TT(TTOp::PushBack { cb, n }) => {
                    if sym_loops != 0 {
                        panic!("tt_fifo_check: FIFO traffic under a symbolic loop at {op_id:?}");
                    }
                    let need = u64::from(n) * mult;
                    let f = fifos.entry(cb).or_default();
                    if f.reserved < need {
                        panic!("tt_fifo_check: push CB{cb:?} with {} reserved at {op_id:?}", f.reserved);
                    }
                    f.reserved -= need;
                    f.avail += need;
                    f.pushed += need;
                    if run_cb != Some(cb) {
                        if let Some(prev) = run_cb {
                            if run_tiles > depth_of(self, prev) {
                                panic!(
                                    "tt_fifo_check: producer pushes {run_tiles} tiles to CB{prev:?} (depth {}) before any other CB",
                                    depth_of(self, prev)
                                );
                            }
                        }
                        run_cb = Some(cb);
                        run_tiles = 0;
                    }
                    run_tiles += u64::from(n);
                }
                Op::TT(TTOp::WaitFront { cb, n }) => {
                    if sym_loops != 0 {
                        panic!("tt_fifo_check: FIFO traffic under a symbolic loop at {op_id:?}");
                    }
                    let need = u64::from(n) * mult;
                    let f = fifos.entry(cb).or_default();
                    if f.avail < need {
                        panic!("tt_fifo_check: wait CB{cb:?} with {} available at {op_id:?}", f.avail);
                    }
                    f.waited += need;
                    f.avail -= need;
                }
                Op::TT(TTOp::PopFront { cb, n }) => {
                    if sym_loops != 0 {
                        panic!("tt_fifo_check: FIFO traffic under a symbolic loop at {op_id:?}");
                    }
                    let need = u64::from(n) * mult;
                    let f = fifos.entry(cb).or_default();
                    if f.waited < need {
                        panic!("tt_fifo_check: pop CB{cb:?} with {} waited at {op_id:?}", f.waited);
                    }
                    f.waited -= need;
                    f.popped += need;
                }
                Op::Copy { src, src_idx, dst, dst_idx, .. } => {
                    let unit: i64 = match self.shape(op_id).as_slice() {
                        [_, _] => 1024,
                        [] | [_] => 1,
                        _ => panic!("tt_fifo_check: copy {op_id:?} has non-tile/vector access shape"),
                    };
                    self.fifo_check_copy(op_id, src, src_idx, dst, dst_idx, unit);
                }
                Op::TT(TTOp::Copy { src, src_idx, dst, dst_idx, .. }) => {
                    // Staged traffic: lane-free bases, unit elementwise.
                    self.fifo_check_copy(op_id, src, src_idx, dst, dst_idx, 1);
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        if !stack.is_empty() {
            panic!("tt_fifo_check: kernel ends inside a loop");
        }
        if let Some(prev) = run_cb {
            if run_tiles > depth_of(self, prev) {
                panic!(
                    "tt_fifo_check: producer pushes {run_tiles} tiles to CB{prev:?} (depth {}) before any other CB",
                    depth_of(self, prev)
                );
            }
        }
        for (cb, f) in fifos.iter() {
            if f.pushed != f.popped {
                panic!("tt_fifo_check: CB{cb:?} pushed {} but popped {}", f.pushed, f.popped);
            }
        }
    }

    /// Chase a `Load`/`Store`/`Copy` operand through its index expression
    /// to the underlying `Storage`/`Param`. Tile-compute results name DST
    /// slots, not buffers. A post-lowering `LLK` resolves to
    /// the Register storage in its operands (its DST slot); an `LLK` with
    /// no Register operand names no slot — loud panic. The index is
    /// not needed (render uses slot 0).
    pub(crate) fn tt_storage_of(&self, op_id: OpId) -> OpId {
        match self.at(op_id) {
            Op::Load { src, .. } | Op::Copy { src, .. } => self.tt_storage_of(*src),
            Op::Store { dst, .. } => self.tt_storage_of(*dst),
            Op::Storage { .. } | Op::Param { .. } => op_id,
            Op::TT(TTOp::LLK { ops, .. }) => {
                let mut slot = None;
                for o in ops.iter().copied() {
                    if o.is_null() {
                        continue;
                    }
                    if matches!(self.at(o), Op::Storage { scope: MemScope::Register, .. }) {
                        slot = Some(o);
                        break;
                    }
                }
                slot.unwrap_or_else(|| panic!("tt_storage_of: LLK {op_id:?} names no Register slot"))
            }
            Op::TT(TTOp::ReduceTile { slot, .. }) => *slot,
            Op::TT(TTOp::MatmulTiles { slot, .. }) => *slot,
            Op::TT(TTOp::LLKBcast { .. }) => panic!("tt_storage_of: LLKBcast {op_id:?} names no Register slot"),
            other => panic!("tt_storage_of: {op_id:?} is not a Load/Store/Copy, got {other:?}"),
        }
    }

    /// Pack-store CB: `Some(cb)` iff `id` is a `Store` of a tile value
    /// into a Circular buffer (shared by `tt_lock_dst`, `tt_sync_cbs`
    /// accounting, and `tt_init_math` — single source of truth).
    fn tt_pack_store_cb(&self, id: OpId) -> Option<OpId> {
        if let Op::Store { src: x, dst, .. } = self.at(id) {
            if matches!(self.at(*dst), Op::Storage { scope: MemScope::Circular, .. }) {
                // Tile and row packs (rank 1 or 2 values); scalar stores
                // stay rejected loudly downstream.
                let n = self.shape(*x).len();
                if n == 1 || n == 2 {
                    return Some(*dst);
                }
            }
        }
        None
    }

    /// True iff `id` is a Circular storage base of a traffic access
    /// (a CB side of a Copy or the dst of a pack store).
    fn is_circular_base(&self, id: OpId) -> bool {
        matches!(self.at(id), Op::Storage { scope: MemScope::Circular, .. })
    }

    /// Validate a scalar binary lane at init time (old scalar rule): int
    /// consts only fold for shifts (right side, 0..=31); float
    /// const-exprs fold for Add/Mul either side and Div on the right;
    /// scalar `Sub` is unrenderable (the old render rejected it too).
    /// Shift rejections return the old `tenstorrent2` errors (tests assert
    /// them via `expect_err`); anything else is a loud error.
    fn tt_check_scalar_lane(&self, id: OpId, bop: BOp, left: bool, s: OpId) -> Result<(), BackendError> {
        match bop {
            BOp::BitShiftLeft | BOp::BitShiftRight => {
                // Shifts lower to the shift LLKs, which are int-only
                // (Int32/UInt32/UInt16): anything else fails here, never
                // on the device.
                let dt = self.dtype(id);
                match dt {
                    DType::I32 | DType::U32 | DType::U16 => {}
                    _ => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!(
                                "tenstorrent2: tiled shift on {dt:?}, LLK supports Int32/UInt32/UInt16 only, op {id}"
                            )
                            .into(),
                        });
                    }
                }
                let Op::Binary { x, y, .. } = self.at(id) else {
                    unreachable!("tt_init_math: shift check on non-Binary {id:?}");
                };
                let (x, y) = (*x, *y);
                // A const side resolving to a compile-time constant has no
                // scalar call. A const amount folds into the
                // unary-immediate LLK (`tile << amount`); amounts outside
                // 0..=31 are UB in every other backend — fail loudly,
                // never emit.
                if self.resolve_const(x).is_some() {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: const-first {bop:?} has no scalar call, op {id}").into(),
                    });
                }
                let v: i128 = match self.resolve_const(y) {
                    Some(Constant::U8(v)) => v as i128,
                    Some(Constant::U16(v)) => v as i128,
                    Some(Constant::U32(v)) => v as i128,
                    Some(Constant::U64(v)) => u64::from_le_bytes(v) as i128,
                    Some(Constant::I8(v)) => v as i128,
                    Some(Constant::I16(v)) => v as i128,
                    Some(Constant::I32(v)) => v as i128,
                    Some(Constant::I64(v)) => i64::from_le_bytes(v) as i128,
                    _ => {
                        return Err(BackendError {
                            status: ErrorStatus::KernelCompilation,
                            context: format!("tenstorrent2: shift amount is no integer const, op {id}").into(),
                        });
                    }
                };
                let Some(amount) = u32::try_from(v).ok() else {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: shift amount is no integer const, op {id}").into(),
                    });
                };
                if amount > 31 {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: shift amount {amount} outside 0..=31, op {id}").into(),
                    });
                }
                if bop == BOp::BitShiftRight && dt == DType::U32 {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent2: U32 right-shift by immediate is arithmetic-only, op {id}").into(),
                    });
                }
                Ok(())
            }
            BOp::Add | BOp::Mul => {
                if !matches!(self.resolve_const(s), Some(Constant::F32(_) | Constant::F16(_) | Constant::BF16(_))) {
                    panic!("tt_init_math: scalar lane {s:?} is not a foldable float const");
                }
                Ok(())
            }
            BOp::Div => {
                if left {
                    panic!("tt_init_math: const-first div {id:?} has no scalar call");
                }
                if !matches!(self.resolve_const(s), Some(Constant::F32(_) | Constant::F16(_) | Constant::BF16(_))) {
                    panic!("tt_init_math: scalar lane {s:?} is not a foldable float const");
                }
                Ok(())
            }
            BOp::Sub => {
                panic!("tt_init_math: scalar sub {id:?} has no LLK call (the old render rejected it too)")
            }
            _ => panic!("tt_init_math: scalar {bop:?} {id:?} has no scalar call"),
        }
    }
}
