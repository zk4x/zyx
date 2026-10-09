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
//! Pass pipeline order: `tt_fuse_llks` → `tt_storage` → `tt_lock_dst`
//! → `tt_dedup_pushes` → `tt_init_math` → `tt_sync_cbs` → `tt_place_pops`, then `verify`. The passes are
//! public so external pass authors can reuse or replace stages.
//! Calling them out of order is a loud panic, never silent corruption.

use crate::DType;
use crate::Map;
use crate::Set;
use crate::dtype::Constant;
use crate::error::{BackendError, ErrorStatus};
use crate::kernel::Pat;
use crate::kernel::{BOp, Kernel, MemLayout, MemScope, Op, OpId, TTOp, TileDim, UOp};
use crate::shape::Dim;
use crate::types::{TinyString, TinyVec};

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
                Op::TT(TTOp::MatmulTile { .. })
                    | Op::TT(TTOp::ReduceTile { .. })
                    | Op::TT(TTOp::TransposeTile { .. })
                    | Op::TT(TTOp::BroadcastTile { .. })
                    | Op::TT(TTOp::LLKReduce { .. })
                    | Op::TT(TTOp::LLKBcast { .. })
                    | Op::TT(TTOp::LLK { .. })
            ) {
                return true;
            }
            // A circular load consumed straight by a pack store unpacks
            // at its own position (render copies there) — it is MATH
            // work like any other unpack. Markers are invisible (the
            // fused call consumes through provenance).
            if let Op::Load { src } = op {
                if let Op::GEP { x: base, .. } = kernel.at(*src) {
                    if matches!(kernel.at(*base), Op::Storage { scope: MemScope::Circular, .. }) {
                        if let Some(us) = users.get(&id) {
                            return us.iter().all(|u| {
                                matches!(kernel.at(*u), Op::TT(TTOp::BroadcastTile { .. }))
                                    || matches!(kernel.at(*u), Op::Store { src: x, .. } if *x == id)
                            }) && us.iter().any(|u| matches!(kernel.at(*u), Op::Store { .. }));
                        }
                    }
                }
                return false;
            }
            match op {
                Op::Asm { .. } => true,
                Op::Load { .. } | Op::Unary { .. } | Op::Binary { .. } | Op::Cast { .. } | Op::Bitcast { .. } => {
                    matches!(kernel.layout(id), MemLayout::Tile { .. })
                }
                _ => false,
            }
        };
        // Pack drain: a Store into a Circular buffer of a tile value
        // (Register slot, circular load, or SSA/LLK/Asm tile result).
        // The drain empties DST into a CB.
        let is_pack = |kernel: &Kernel, id: OpId| {
            if let Op::Store { src: x, dst } = kernel.at(id) {
                if let Op::GEP { x: g, .. } = kernel.at(*dst) {
                    if matches!(kernel.at(*g), Op::Storage { scope: MemScope::Circular, .. }) {
                        return matches!(kernel.layout(*x), MemLayout::Tile { .. });
                    }
                }
            }
            false
        };
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

    /// Fused unary LLK claiming: `sigmoid`/`silu` composites built by
    /// the plain builders (`neg` → `exp` → `add` → `recip`, plus `mul`
    /// for silu, with `exp` spelled as `exp2(x * log2(e))` per the
    /// builders) become one `sigmoid_tile` / one `silu_tile` call.
    /// Same approach as [`Kernel::fuse_rsqrt`](super::fuse): use counts
    /// gate the rewrite, orphaned inners stay dead in place, `verify`.
    ///
    /// Biggest first: a silu cone contains a sigmoid cone, so the silu
    /// pass walks the whole kernel before the sigmoid pass sees it.
    /// Claiming the inner sigmoid first would rewrite it and the silu
    /// pattern could never match. Each pass recomputes use counts from
    /// the current IR (a rewrite changes them).
    ///
    /// Containment (the test's pattern-exclusive-input rule): every
    /// non-const op of the pattern must have all its users inside the
    /// pattern. A shared input (or a shared inner) correctly falls
    /// back to the plain composite — fusing it would re-time CB page
    /// traffic the second consumer still counts on. The pattern root
    /// itself may have outside users; they read the fused tile value
    /// exactly as they read the unfused one. Dead roots (no users —
    /// orphaned inners of an already-fused bigger pattern) are skipped.
    ///
    /// The rewrite is an opaque [`TTOp::LLK`] over the feeder value
    /// (the sigmoid input `x`): `sync_cbs` waits its CB page through
    /// the trailing provenance operand, render copies it into a fresh
    /// DST slot at the call (`operand_slot`, same as the unfused
    /// `exp_tile(s)` path). The engine-config init
    /// (`sigmoid_tile_init();` / `silu_tile_init();`) and the feeder
    /// copy init go right before the call, mirroring `init_math`'s
    /// `asm_before` shapes. Orphaned inner ops are pruned by
    /// `dead_code_elimination` at the end of the pass, so downstream
    /// sync accounting never sees them (a dead inner left in place
    /// would draw a second CB wait for the feeder's page).
    pub fn tt_fuse_llks(&mut self) {
        let x = Pat::bind('x');
        let sigmoid_pat = (x.neg().exp() + 1.).recip();
        let silu_pat = &x * &sigmoid_pat;
        for (pat, init, call) in [
            (&silu_pat, "silu_tile_init();", "silu_tile({0});"),
            (&sigmoid_pat, "sigmoid_tile_init();", "sigmoid_tile({0});"),
            (&Pat::exp(x), "exp_tile_init();", "exp_tile({0});"),
        ] {
            // Total user counts: users[v] = ops taking v as a data operand.
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
            let mut op_id = self.head;
            while !op_id.is_null() {
                let next = self.next_op(op_id);
                // Dead roots stay dead: without a user the fused call
                // would wait a CB page nobody pushed.
                if users.contains_key(&op_id)
                    && let Some(m) = self.match_pat(op_id, pat)
                {
                    let feeder = m.op('x');
                    // Cone: DFS from the root, stopping at consts (shared
                    // consts are free) and the feeder (everything below it
                    // is outside the pattern).
                    let mut cone = Vec::new();
                    let mut stack = vec![op_id];
                    while let Some(id) = stack.pop() {
                        if id.is_null() || id == feeder || matches!(self.at(id), Op::Const(_)) {
                            continue;
                        }
                        if !cone.contains(&id) {
                            cone.push(id);
                            stack.extend(self.at(id).parameters());
                        }
                    }
                    // Exclusive: the feeder and every cone op except the
                    // root have all their users inside the cone (or at the
                    // root). The root itself may feed outside readers.
                    let feeder_ok =
                        users.get(&feeder).map(|us| us.iter().all(|u| *u == op_id || cone.contains(u))).unwrap_or(true);
                    let inners_ok = cone
                        .iter()
                        .filter(|c| **c != op_id)
                        .all(|c| users.get(c).map(|us| us.iter().all(|u| cone.contains(u))).unwrap_or(true));
                    if feeder_ok && inners_ok {
                        // Quick gate: the fused unary renders its feeder
                        // via `operand_slot` (Load or slotted value
                        // only). A call-op feeder (e.g. exp over a fused
                        // broadcast) is unrenderable — leave it unfused
                        // for the general lowering. Sigmoid/silu keep
                        // their existing shapes.
                        if call == "exp_tile({0});" && !matches!(self.at(feeder), Op::Load { .. }) {
                            op_id = next;
                            continue;
                        }
                        self.insert_before(op_id, Op::Asm { asm: TinyString::new(init), ops: TinyVec::new(&[]) });
                        self.ops[op_id].op = Op::TT(TTOp::LLK { asm: TinyString::new(call), ops: TinyVec::new(&[feeder]) });
                    }
                }
                op_id = next;
            }
            // Prune orphaned inners before the next pattern: a fused
            // bigger cone leaves its inner chain structurally linked,
            // and the smaller pattern would otherwise match it and
            // fuse a second traffic event for the same page.
            self.common_subexpression_elimination();
            self.dead_code_elimination();
        }
        // Prune the orphaned inner ops: downstream sync accounting
        // counts structural users, and a dead inner left in place
        // would draw CB waits for pages the fused call already covers.
        // LLK calls and Asm inits are DCE roots, so the fused calls
        // (and their inits) survive.
        self.common_subexpression_elimination();
        self.dead_code_elimination();
        self.verify();
    }

    /// Storage lowering: the SSA tile-compute ops (`MatmulTile`/
    /// `ReduceTile`/`TransposeTile`, plus a tiled `Binary` fused with
    /// a `BroadcastTile` marker) are the stream's compute representation;
    /// the physical CB and DST slots are the `Op::Storage`s already in
    /// the IR. This pass rewrites each compute op to a [`TTOp::LLK`]
    /// call template over those bare storages. After this pass the
    /// compute ops are opaque effect calls; `tt_lock_dst`/
    /// `tt_sync_cbs`/render see only LLK.
    ///
    /// Operands are CB storages plus the register slot; the op/kind/bop
    /// go into the template text (the render substitutes `{i}` for the
    /// leading operands and emits the rest verbatim, mirroring the init
    /// constructors). Trailing operands past the template's `{i}` range
    /// are provenance only: the feeder `Load` ids the call consumes
    /// from, for sync accounting (`tt_sync_cbs` waits,
    /// `tt_place_pops` pops). Render ignores them. A `Load` that feeds
    /// a compute op must chase through its `GEP` to a `Storage`; a
    /// `Load` over a `Param` (DRAM) feeding a compute op is a lowering
    /// bug — loud `panic!`. `BroadcastTile` markers never lower alone:
    /// unfused (dead) markers stay in place and are ignored downstream.
    pub fn tt_storage(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            // Canonicalize CB→CB moves to load+store spelling first: a
            // fused copy would straddle the MATH/PACK cone boundary
            // (unpack needs MATH-open, pack needs PACK-open), while the
            // split form lowers each half under its own cone like
            // every other pack. Same CBs, same indices, same traffic.
            if let Op::Copy { src, dst } = self.ops[op_id].op {
                if self.is_circular_gep(src) && self.is_circular_gep(dst) {
                    let load = self.insert_before(op_id, Op::Load { src });
                    self.ops[op_id].op = Op::Store { dst, src: load };
                    op_id = next;
                    continue;
                }
            }
            let (asm, ops) = match self.at(op_id) {
                Op::TT(TTOp::MatmulTile { x, y, acc }) => {
                    let cb_a = self.tt_storage_of(*x);
                    let cb_b = self.tt_storage_of(*y);
                    let slot = self.tt_storage_of(*acc);
                    debug_assert!(
                        matches!(self.at(cb_a), Op::Storage { scope: MemScope::Circular, .. }),
                        "tt_storage: matmul left {cb_a:?} is not a Circular storage"
                    );
                    debug_assert!(
                        matches!(self.at(cb_b), Op::Storage { scope: MemScope::Circular, .. }),
                        "tt_storage: matmul right {cb_b:?} is not a Circular storage"
                    );
                    debug_assert!(
                        matches!(self.at(slot), Op::Storage { scope: MemScope::Register, .. }),
                        "tt_storage: matmul acc {slot:?} is not a Register slot"
                    );
                    (TinyString::new("matmul_tiles({0}, {1}, 0, 0, {2});"), TinyVec::new(&[cb_a, cb_b, slot, *x, *y]))
                }
                Op::TT(TTOp::ReduceTile { x, scaler, acc, rop, kind }) => {
                    let cb_in = self.tt_storage_of(*x);
                    let cb_sc = self.tt_storage_of(*scaler);
                    let slot = self.tt_storage_of(*acc);
                    debug_assert!(
                        matches!(self.at(cb_in), Op::Storage { scope: MemScope::Circular, .. }),
                        "tt_storage: reduce input {cb_in:?} is not a Circular storage"
                    );
                    debug_assert!(
                        matches!(self.at(cb_sc), Op::Storage { scope: MemScope::Circular, .. }),
                        "tt_storage: reduce scaler {cb_sc:?} is not a Circular storage"
                    );
                    debug_assert!(
                        matches!(self.at(slot), Op::Storage { scope: MemScope::Register, .. }),
                        "tt_storage: reduce acc {slot:?} is not a Register slot"
                    );
                    self.ops[op_id].op =
                        Op::TT(TTOp::LLKReduce { rop: *rop, kind: *kind, cb_in, cb_sc, slot, x: *x, scaler: *scaler });
                    op_id = next;
                    continue;
                }
                Op::TT(TTOp::TransposeTile { x }) => {
                    let cb = self.tt_storage_of(*x);
                    debug_assert!(
                        matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
                        "tt_storage: transpose input {cb:?} is not a Circular storage"
                    );
                    // The transpose result lands in a DST slot filled by
                    // render (`{1}`); the trailing entry is the feeder
                    // load, for sync accounting only.
                    (TinyString::new("transpose_wh_tile({0}, 0, {1});"), TinyVec::new(&[cb, OpId::NULL, *x]))
                }
                Op::TT(TTOp::BroadcastTile { .. }) => {
                    // Marker only: the consuming tiled binary fuses it
                    // (see the `Op::Binary` arm below). Left in place;
                    // `tt_place_pops` ignores markers as load users and
                    // pairs the marked load with the fused call.
                    op_id = next;
                    continue;
                }
                Op::Binary { x, y, bop } => {
                    let x_marked = matches!(self.at(*x), Op::TT(TTOp::BroadcastTile { .. }));
                    let y_marked = matches!(self.at(*y), Op::TT(TTOp::BroadcastTile { .. }));
                    if !x_marked && !y_marked {
                        op_id = next;
                        continue;
                    }
                    if x_marked && y_marked {
                        panic!("tt_storage: binary {op_id:?} marks both sides for broadcast");
                    }
                    // The marked side consumes its CB straight from the
                    // buffer (`cb_b`); the plain side must be a plain
                    // circular load (`cb_a`). The fused result is a
                    // normal DST tile (operand {2}, filled by render).
                    let (marked, plain) = if x_marked { (*x, *y) } else { (*y, *x) };
                    let Op::TT(TTOp::BroadcastTile { x: mx, kind }) = self.at(marked) else {
                        unreachable!("tt_storage: marked side is not a BroadcastTile");
                    };
                    let cb_b = self.tt_storage_of(*mx);
                    if !matches!(self.at(cb_b), Op::Storage { scope: MemScope::Circular, .. }) {
                        panic!("tt_storage: broadcast {op_id:?} marked side is no CB tile load");
                    }
                    if !matches!(self.at(plain), Op::Load { .. } | Op::TT(TTOp::BroadcastTile { .. })) {
                        panic!("tt_storage: broadcast {op_id:?} plain side is no CB tile load");
                    }
                    let cb_a = self.tt_storage_of(plain);
                    if !matches!(self.at(cb_a), Op::Storage { scope: MemScope::Circular, .. }) {
                        panic!("tt_storage: broadcast {op_id:?} plain side is no CB tile load");
                    }
                    self.ops[op_id].op = Op::TT(TTOp::LLKBcast { bop: *bop, kind: *kind, cb_a, cb_b, mx: *mx, plain });
                    op_id = next;
                    continue;
                }
                _ => {
                    op_id = next;
                    continue;
                }
            };
            self.ops[op_id].op = Op::TT(TTOp::LLK { asm, ops });
            op_id = next;
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
        // Circular base of a GEP, if it names one.
        let circ_base = |kernel: &Kernel, gep: OpId| -> Option<OpId> {
            if let Op::GEP { x, .. } = kernel.at(gep) {
                if matches!(kernel.at(*x), Op::Storage { scope: MemScope::Circular, .. }) {
                    return Some(*x);
                }
            }
            None
        };
        // Resolved slot index of a GEP; unresolvable never matches.
        let slot_of = |kernel: &Kernel, gep: OpId| -> Option<Dim> {
            if let Op::GEP { index, .. } = kernel.at(gep) {
                return kernel.resolve_const(*index).and_then(|c| c.as_dim());
            }
            None
        };
        // Whether an op moves pages of `cb` (gap dirtiness). SSA and
        // marker ops are inert; user `Asm` and anything unlisted are
        // opaque and invalidate.
        let touches = |kernel: &Kernel, id: OpId, cb: OpId| -> bool {
            let load_cb = |load: OpId| -> bool {
                if let Op::Load { src } = kernel.at(load) {
                    return circ_base(kernel, *src) == Some(cb);
                }
                false
            };
            match kernel.at(id) {
                Op::Copy { src, dst } => circ_base(kernel, *src) == Some(cb) || circ_base(kernel, *dst) == Some(cb),
                Op::Load { src } => circ_base(kernel, *src) == Some(cb),
                Op::Store { dst, .. } => circ_base(kernel, *dst) == Some(cb),
                Op::TT(TTOp::LLK { ops, .. }) => ops.iter().copied().any(|o| {
                    !o.is_null()
                        && (matches!(kernel.at(o), Op::Storage { scope: MemScope::Circular, .. } if o == cb) || load_cb(o))
                }),
                Op::TT(TTOp::LLKReduce { cb_in, cb_sc, x, scaler, .. }) => {
                    *cb_in == cb || *cb_sc == cb || load_cb(*x) || load_cb(*scaler)
                }
                Op::TT(TTOp::LLKBcast { cb_a, cb_b, mx, plain, .. }) => {
                    *cb_a == cb || *cb_b == cb || load_cb(*mx) || load_cb(*plain)
                }
                Op::TT(
                    TTOp::MatmulTile { .. }
                    | TTOp::TransposeTile { .. }
                    | TTOp::ReduceTile { .. }
                    | TTOp::BroadcastTile { .. }
                    | TTOp::ReserveBack { .. }
                    | TTOp::PushBack { .. }
                    | TTOp::WaitFront { .. }
                    | TTOp::PopFront { .. },
                ) => true,
                Op::Const(_)
                | Op::Param { .. }
                | Op::Storage { .. }
                | Op::GEP { .. }
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
        let mut pubs: Vec<(OpId, Dim, OpId, usize, OpId)> = Vec::new();
        let mut loads: Vec<(OpId, Dim, OpId, usize, OpId)> = Vec::new();
        for &id in &order {
            match self.at(id) {
                Op::Copy { src, dst } => {
                    let src_circ = circ_base(self, *src).is_some();
                    if src_circ {
                        continue;
                    }
                    let (Some(cb), Some(slot)) = (circ_base(self, *dst), slot_of(self, *dst)) else {
                        continue;
                    };
                    // Source must be an address (a GEP), not a bare op:
                    // bare sources have no slot identity to match on.
                    if !matches!(self.at(*src), Op::GEP { .. }) {
                        continue;
                    }
                    pubs.push((cb, slot, *src, pos_of[&id], id));
                }
                Op::Load { src } => {
                    let (Some(cb), Some(slot)) = (circ_base(self, *src), slot_of(self, *src)) else {
                        continue;
                    };
                    loads.push((cb, slot, *src, pos_of[&id], id));
                }
                _ => {}
            }
        }
        // Bucket keys present in the publishes.
        let mut keys: Vec<(OpId, Dim)> = Vec::new();
        for (cb, slot, _, _, _) in &pubs {
            if !keys.iter().any(|(c, s)| c == cb && s == slot) {
                keys.push((*cb, slot.clone()));
            }
        }
        let mut removals: Vec<OpId> = Vec::new();
        for (cb, slot) in keys {
            // One data source per slot: ambiguous pages never match.
            let mut data: Vec<OpId> = Vec::new();
            for (_, _, src, _, _) in pubs.iter().filter(|(c, s, _, _, _)| c == &cb && s == &slot) {
                if !data.contains(src) {
                    data.push(*src);
                }
            }
            if data.len() != 1 {
                continue;
            }
            // One address node per slot on the consume side.
            let mut addrs: Vec<OpId> = Vec::new();
            for (_, _, src, _, _) in loads.iter().filter(|(c, s, _, _, _)| c == &cb && s == &slot) {
                if !addrs.contains(src) {
                    addrs.push(*src);
                }
            }
            if addrs.len() != 1 {
                continue;
            }
            let mut copies: Vec<(usize, OpId)> =
                pubs.iter().filter(|(c, s, _, _, _)| c == &cb && s == &slot).map(|(_, _, _, p, id)| (*p, *id)).collect();
            copies.sort();
            // Predicted waits for this slot: each live load with a
            // non-LLK, non-marker user draws one load-wait, and each
            // LLK-structured call draws one wait per consumed bucket
            // load (sync waits every provenance CB per call; markers
            // are transparent, dead loads draw none). Fills must match
            // waits, so keep one copy per predicted wait.
            let mut load_waits = 0usize;
            let mut call_waits = 0usize;
            for (_, _, _, _, id) in loads.iter().filter(|(c, s, _, _, _)| c == &cb && s == &slot) {
                let Some(us) = users.get(id) else { continue };
                let mut draws_load_wait = false;
                for u in us {
                    match self.at(*u) {
                        Op::TT(TTOp::LLK { .. } | TTOp::LLKReduce { .. } | TTOp::LLKBcast { .. }) => {
                            call_waits += 1;
                        }
                        Op::TT(TTOp::BroadcastTile { .. }) => {}
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
                .filter(|(c, s, _, _, _)| c == &cb && s == &slot)
                .filter(|(_, _, _, _, id)| !users.contains_key(id))
                .map(|(_, _, _, _, id)| *id)
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
    /// - publish (`dst` GEP over a Circular storage): `ReserveBack`
    ///   immediately before, `PushBack` immediately after;
    /// - drain (`src` GEP over a Circular storage): `WaitFront`
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
    /// a `BroadcastTile` marker or an `LLK` provenance position) carries
    /// no wait of its own: the fused call's provenance waits cover those
    /// pages (old "fused-draining loads carry no syncs"). Any other consumer
    /// is malformed CB traffic — DRAM↔CB moves are `Copy` (see
    /// `copy_global_to_circular`), never load+store — loud, never
    /// silently skipped. A circular buffer reached outside a GEP is
    /// likewise malformed — loud, never silently skipped.
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
                Op::Load { src } => match self.ops[src].op {
                    Op::GEP { x: cb, index, .. } if matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. }) => {
                        (scan, index, cb)
                    }
                    _ => {
                        scan = self.next_op(scan);
                        continue;
                    }
                },
                Op::Copy { src, .. } => match self.ops[src].op {
                    Op::GEP { x: cb, index, .. } if matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. }) => {
                        (scan, index, cb)
                    }
                    _ => {
                        scan = self.next_op(scan);
                        continue;
                    }
                },
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
                    Op::Load { src } => {
                        if let Op::GEP { x: c, index, .. } = self.ops[src].op
                            && matches!(self.ops[c].op, Op::Storage { scope: MemScope::Circular, .. })
                            && c == cb
                        {
                            let indexed = self.resolve_const(index).and_then(|c| c.as_dim()) != Some(0);
                            let ok_users = users.get(&cur).is_some_and(|us| {
                                us.iter().all(|u| match &self.ops[*u].op {
                                    Op::Store { .. } | Op::Asm { .. } => true,
                                    op => {
                                        matches!(self.layout(*u), MemLayout::Tile { .. })
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
                        if let Op::GEP { x: c, .. } = self.ops[dst].op
                            && c == cb
                            && matches!(self.ops[c].op, Op::Storage { scope: MemScope::Circular, .. })
                        {
                            held = false;
                            break;
                        }
                    }
                    Op::Copy { src, dst } => {
                        let src_hit = match self.ops[src].op {
                            Op::GEP { x: c, index, .. }
                                if c == cb && matches!(self.ops[c].op, Op::Storage { scope: MemScope::Circular, .. }) =>
                            {
                                Some(index)
                            }
                            _ => None,
                        };
                        let dst_hit = match self.ops[dst].op {
                            Op::GEP { x: c, .. }
                                if c == cb && matches!(self.ops[c].op, Op::Storage { scope: MemScope::Circular, .. }) =>
                            {
                                Some(c)
                            }
                            _ => None,
                        };
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
                    Op::TT(TTOp::LLK { .. }) | Op::TT(TTOp::LLKReduce { .. }) | Op::TT(TTOp::LLKBcast { .. }) => {
                        let mut feeds = false;
                        for o in self.at(cur).parameters() {
                            if o.is_null() {
                                continue;
                            }
                            if let Op::Load { src } = self.ops[o].op
                                && let Op::GEP { x: c, .. } = self.ops[src].op
                                && c == cb
                            {
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
                        Op::Const(c) => {
                            match c.as_dim().and_then(|t| u64::try_from(t).ok()).filter(|t| *t > 0) {
                                Some(t) => {
                                    mult *= t;
                                    stack.push(CountFrame::Trip { op: scan, trips: t });
                                }
                                None => {
                                    uncountable += 1;
                                    stack.push(CountFrame::Other);
                                }
                            }
                        }
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
                            Op::Store { src: x, dst } if uncountable == 0 && self.is_circular_gep(dst) => {
                                match self.layout(x) {
                                    MemLayout::Scalar => Some((self.tt_storage_of(dst), 1)),
                                    _ => None,
                                }
                            }
                            Op::Copy { src, dst } if uncountable == 0 => {
                                let src_is_cb = match self.ops[src].op {
                                    Op::GEP { x, .. } => {
                                        matches!(self.ops[x].op, Op::Storage { scope: MemScope::Circular, .. })
                                    }
                                    _ => false,
                                };
                                match self.ops[dst].op {
                                    Op::GEP { x: d, .. }
                                        if matches!(self.ops[d].op, Op::Storage { scope: MemScope::Circular, .. })
                                            && !src_is_cb =>
                                    {
                                        match self.layout(dst) {
                                            MemLayout::Vector(size) => Some((d, size as u64)),
                                            MemLayout::Scalar => Some((d, 1)),
                                            MemLayout::Tile { .. } => None,
                                        }
                                    }
                                    _ => None,
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
                Op::Copy { src, dst } => {
                    let src_cb = match self.ops[src].op {
                        Op::GEP { x, .. } if matches!(self.ops[x].op, Op::Storage { scope: MemScope::Circular, .. }) => Some(x),
                        Op::GEP { .. } => None,
                        Op::Storage { scope: MemScope::Circular, .. } => {
                            panic!("tt_sync_cbs: copy src {src:?} names a Circular storage directly, must go through a GEP")
                        }
                        _ => None,
                    };
                    let dst_cb = match self.ops[dst].op {
                        Op::GEP { x, .. } if matches!(self.ops[x].op, Op::Storage { scope: MemScope::Circular, .. }) => Some(x),
                        Op::GEP { .. } => None,
                        Op::Storage { scope: MemScope::Circular, .. } => {
                            panic!("tt_sync_cbs: copy dst {dst:?} names a Circular storage directly, must go through a GEP")
                        }
                        _ => None,
                    };
                    match (src_cb, dst_cb) {
                        (None, None) => {}
                        (None, Some(cb)) => {
                            // Vector and scalar publishes tally
                            // element-wise above (exact page multiples
                            // publish); the per-op publish here would
                            // over-count. Partial pages stay
                            // unpublished (loud downstream).
                            let tallied = matches!(self.layout(dst), MemLayout::Vector(_))
                                || matches!(self.layout(dst), MemLayout::Scalar);
                            if !tallied {
                                self.insert_before(op_id, Op::TT(TTOp::ReserveBack { cb, n: 1 }));
                                self.insert_after(op_id, Op::TT(TTOp::PushBack { cb, n: 1 }));
                            }
                        }
                        (Some(cb), None) => {
                            if matches!(self.layout(src), MemLayout::Vector(_)) {
                                panic!("tt_sync_cbs: vector drain {op_id:?} has no sync support");
                            }
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
                Op::Store { src: x, dst } => {
                    // Pack store: a tile value into a Circular buffer
                    // reserves a back slot and pushes it, like a publish.
                    // A `Const` src is a fill, not a pack — left alone
                    // (render rejects it loudly).
                    if self.is_circular_gep(dst) && matches!(self.layout(x), MemLayout::Tile { .. }) {
                        self.insert_before(op_id, Op::TT(TTOp::ReserveBack { cb: self.tt_storage_of(dst), n: 1 }));
                        let cb = self.tt_storage_of(dst);
                        self.insert_after(op_id, Op::TT(TTOp::PushBack { cb, n: 1 }));
                    }
                }
                Op::Load { src } => {
                    // Non-CB reads are not sync business. Circularity gates
                    // the whole arm as a nested `if` — a `continue` here
                    // would skip the cursor advance at the loop bottom and
                    // spin forever.
                    if let Op::GEP { x: cb, .. } = self.ops[src].op
                        && matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. })
                    {
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
                                            TTOp::MatmulTile { .. }
                                            | TTOp::TransposeTile { .. }
                                            | TTOp::ReduceTile { .. }
                                            | TTOp::BroadcastTile { .. }
                                            | TTOp::LLKReduce { .. }
                                            | TTOp::LLKBcast { .. }
                                            | TTOp::LLK { .. },
                                        ) => true,
                                        Op::Store { .. } | Op::Asm { .. } => true,
                                        op => {
                                            matches!(self.layout(u), MemLayout::Tile { .. })
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
                                        Op::TT(TTOp::BroadcastTile { .. })
                                            | Op::TT(TTOp::LLK { .. })
                                            | Op::TT(TTOp::LLKReduce { .. })
                                            | Op::TT(TTOp::LLKBcast { .. })
                                    )
                                });
                                if !fused_only && !batched_read.contains(&op_id) {
                                    self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb, n: 1 }));
                                }
                            }
                        }
                    }
                }
                Op::TT(TTOp::LLK { .. }) | Op::TT(TTOp::LLKReduce { .. }) | Op::TT(TTOp::LLKBcast { .. }) => {
                    // Provenance waits: the call consumes its input CBs
                    // straight from the buffers, so each trailing feeder
                    // load gets a wait immediately before the call.
                    let mut wait_cbs: Vec<OpId> = Vec::new();
                    for o in self.at(op_id).parameters() {
                        if o.is_null() {
                            continue;
                        }
                        if !matches!(self.ops[o].op, Op::Load { .. }) {
                            continue;
                        }
                        let Op::Load { src } = self.ops[o].op else {
                            unreachable!("tt_sync_cbs: provenance is a Load");
                        };
                        if let Op::GEP { x: cb, .. } = self.ops[src].op
                            && matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. })
                        {
                            wait_cbs.push(cb);
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
    /// `reconfig_pack` + `fill_out_cbs` + `close_reduce_cones` passes
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
    /// - reduce cones: packing a pending reduce acc inserts
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
        let mut startup_loads: Vec<OpId> = Vec::new();
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
                Op::Load { src } if in_compute => {
                    if let Op::GEP { x: cb, .. } = self.ops[*src].op
                        && matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. })
                        && !startup_loads.contains(&cb)
                    {
                        startup_loads.push(cb);
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
            scan = self.next_op(scan);
        }
        if let (Some(reader), false) = (end_reader, has_matmul)
            && let (Some(&in0), Some(out)) = (startup_loads.first(), first_pack)
        {
            let in1 = startup_loads.get(1).copied().unwrap_or(in0);
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

        // Main walk: per-op inits, reduce cones, section discipline.
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
                    Op::TT(TTOp::LLK { .. }) | Op::TT(TTOp::LLKReduce { .. }) | Op::TT(TTOp::LLKBcast { .. }) => true,
                    Op::Store { .. } => self.tt_pack_store_cb(op_id).is_some(),
                    Op::Load { src } => {
                        matches!(self.ops[*src].op, Op::GEP { x, .. } if matches!(self.ops[x].op, Op::Storage { scope: MemScope::Circular, .. }))
                    }
                    op => {
                        matches!(op, Op::Unary { .. } | Op::Binary { .. } | Op::Cast { .. } | Op::Bitcast { .. } | Op::Asm { .. })
                            && matches!(self.layout(op_id), MemLayout::Tile { .. })
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
                TransposeInit(OpId, OpId),
                ReduceInit(OpId, OpId, OpId, BOp, TileDim),
                BcastInit(BOp, TileDim, OpId, OpId),
            }
            let action = match &self.ops[op_id].op {
                Op::Unary { uop, .. } => InitAction::UnaryInit(*uop),
                Op::Binary { .. } => {
                    let Op::Binary { x, y, bop } = self.ops[op_id].op else {
                        unreachable!("tt_init_math: not a Binary");
                    };
                    match (matches!(self.layout(x), MemLayout::Tile { .. }), matches!(self.layout(y), MemLayout::Tile { .. })) {
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
                Op::Cast { .. } => {
                    let Op::Cast { x, dtype } = self.ops[op_id].op else {
                        unreachable!("tt_init_math: not a Cast");
                    };
                    InitAction::CastInit(self.dtype(x), dtype)
                }
                Op::Bitcast { .. } => InitAction::None,
                Op::Copy { src, dst } => {
                    if self.is_circular_gep(*src) && self.is_circular_gep(*dst) {
                        let Op::GEP { x: dcb, .. } = self.ops[*dst].op else {
                            unreachable!("tt_init_math: CB copy dst is a GEP");
                        };
                        InitAction::PackReconfig(dcb)
                    } else {
                        InitAction::None
                    }
                }
                Op::Store { .. } => match self.tt_pack_store_cb(op_id) {
                    Some(cb) => InitAction::PackReconfig(cb),
                    None => InitAction::None,
                },
                Op::TT(TTOp::LLKReduce { rop, kind, cb_in, cb_sc, slot, .. }) => {
                    pending_acc = Some(*slot);
                    acc_values.clear();
                    InitAction::ReduceInit(*cb_in, *cb_sc, *slot, *rop, *kind)
                }
                Op::TT(TTOp::LLKBcast { bop, kind, cb_a, cb_b, .. }) => InitAction::BcastInit(*bop, *kind, *cb_a, *cb_b),
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
            // Forward acc-threading mark: a value defined from the
            // pending acc joins the set, so the pack check below is a
            // single lookup. Defs precede uses in walk order, so every
            // contributor is already marked when its consumer arrives.
            if let Some(acc) = pending_acc {
                let threads = match &self.ops[op_id].op {
                    Op::Load { src } => matches!(self.ops[*src].op, Op::GEP { x, .. } if x == acc),
                    Op::Unary { x, .. } | Op::Cast { x, .. } | Op::Bitcast { x, .. } => acc_values.contains(x),
                    Op::Binary { x, y, .. } => acc_values.contains(x) || acc_values.contains(y),
                    Op::TT(TTOp::BroadcastTile { x, .. }) => acc_values.contains(x),
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
            if let Op::Load { src } = self.ops[op_id].op {
                if let Op::GEP { x: cb, .. } = self.ops[src].op
                    && matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. })
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
                    // Reduce cone: packing the pending acc closes it.
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
            }
            op_id = next;
        }

        // Sticky-init hoist (old `hoist_dedup_inits`): an `mm_init`
        // programs the MATH unit once per config (sticky) — re-issuing
        // it per loop trip is at best redundant. Bubble each one
        // outward across enclosing loops whose bodies hold no lock ops
        // (a lock in the body means per-trip cones; crossing it would
        // move config out of its cone). Per-call unpacker/packer inits
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
    ///   its last same-section consumer (`BroadcastTile` markers are
    ///   not consumers — the fused call consumes through provenance);
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
                Op::Load { src } => match self.ops[src].op {
                    Op::GEP { x: cb, index, .. } if matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. }) => {
                        (scan, index, cb)
                    }
                    _ => {
                        scan = self.next_op(scan);
                        continue;
                    }
                },
                Op::Copy { src, .. } => match self.ops[src].op {
                    Op::GEP { x: cb, index, .. } if matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. }) => {
                        (scan, index, cb)
                    }
                    _ => {
                        scan = self.next_op(scan);
                        continue;
                    }
                },
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
                    Op::Load { src } => {
                        if let Op::GEP { x: c, index, .. } = self.ops[src].op
                            && matches!(self.ops[c].op, Op::Storage { scope: MemScope::Circular, .. })
                            && c == cb
                        {
                            let indexed = self.resolve_const(index).and_then(|c| c.as_dim()) != Some(0);
                            let ok_users = users.get(&cur).is_some_and(|us| {
                                us.iter().all(|u| match &self.ops[*u].op {
                                    Op::Store { .. } | Op::Asm { .. } => true,
                                    op => {
                                        matches!(self.layout(*u), MemLayout::Tile { .. })
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
                        if let Op::GEP { x: c, .. } = self.ops[dst].op
                            && c == cb
                            && matches!(self.ops[c].op, Op::Storage { scope: MemScope::Circular, .. })
                        {
                            held = false;
                            break;
                        }
                    }
                    Op::Copy { src, dst } => {
                        let src_hit = match self.ops[src].op {
                            Op::GEP { x: c, index, .. }
                                if c == cb && matches!(self.ops[c].op, Op::Storage { scope: MemScope::Circular, .. }) =>
                            {
                                Some(index)
                            }
                            _ => None,
                        };
                        let dst_hit = match self.ops[dst].op {
                            Op::GEP { x: c, .. }
                                if c == cb && matches!(self.ops[c].op, Op::Storage { scope: MemScope::Circular, .. }) =>
                            {
                                Some(c)
                            }
                            _ => None,
                        };
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
                    Op::TT(TTOp::LLK { .. }) | Op::TT(TTOp::LLKReduce { .. }) | Op::TT(TTOp::LLKBcast { .. }) => {
                        let mut feeds = false;
                        for o in self.at(cur).parameters() {
                            if o.is_null() {
                                continue;
                            }
                            if let Op::Load { src } = self.ops[o].op
                                && let Op::GEP { x: c, .. } = self.ops[src].op
                                && c == cb
                            {
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

        // Provenance remaining counts: (cb storage, load) -> calls.
        let mut remaining: Map<(OpId, OpId), u32> = Map::default();
        let mut walk = self.head;
        while !walk.is_null() {
            if matches!(
                self.ops[walk].op,
                Op::TT(TTOp::LLK { .. }) | Op::TT(TTOp::LLKReduce { .. }) | Op::TT(TTOp::LLKBcast { .. })
            ) {
                for o in self.at(walk).parameters() {
                    if o.is_null() || !matches!(self.ops[o].op, Op::Load { .. }) {
                        continue;
                    }
                    let Op::Load { src } = self.ops[o].op else {
                        unreachable!("tt_place_pops: provenance is a Load");
                    };
                    if let Op::GEP { x: cb, .. } = self.ops[src].op
                        && matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. })
                    {
                        *remaining.entry((cb, o)).or_insert(0) += 1;
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
                        | Op::TT(TTOp::LLKReduce { .. })
                        | Op::TT(TTOp::LLKBcast { .. })
                        | Op::TT(TTOp::BroadcastTile { .. })
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
                    if let Op::GEP { x: cb, .. } = self.ops[src].op
                        && matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. })
                    {
                        if !in_compute {
                            // Writer drain (reader never holds a
                            // circular src — its copies publish).
                            // A CB→CB move outside compute cannot
                            // unpack — malformed.
                            let is_drain = matches!(self.ops[op_id].op, Op::Copy { dst, .. } if !self.is_circular_gep(dst));
                            if !is_drain {
                                panic!("tt_place_pops: CB→CB copy {op_id:?} outside the compute section");
                            }
                        }
                        // Indexed-window reads carry no per-trip pop:
                        // the bracket pop covers the window.
                        if !batched_read.contains(&op_id) {
                            pops.push(cb);
                        }
                    }
                }
                Op::TT(TTOp::LLK { .. }) | Op::TT(TTOp::LLKReduce { .. }) | Op::TT(TTOp::LLKBcast { .. }) => {
                    let mut seen: Vec<OpId> = Vec::new();
                    let mut cbs: Vec<(OpId, OpId)> = Vec::new();
                    for o in self.at(op_id).parameters() {
                        if o.is_null() || !matches!(self.ops[o].op, Op::Load { .. }) {
                            continue;
                        }
                        let Op::Load { src } = self.ops[o].op else {
                            unreachable!("tt_place_pops: provenance is a Load");
                        };
                        if let Op::GEP { x: cb, .. } = self.ops[src].op
                            && matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. })
                            && !seen.contains(&cb)
                        {
                            seen.push(cb);
                            cbs.push((cb, o));
                        }
                    }
                    for (cb, load) in cbs {
                        let left = remaining
                            .get_mut(&(cb, load))
                            .unwrap_or_else(|| panic!("tt_place_pops: LLK {op_id:?} consumes uncounted page ({cb:?}, {load:?})"));
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
                    let Op::Load { src } = self.ops[*load].op else {
                        unreachable!("tt_place_pops: last-use of a Load");
                    };
                    if let Op::GEP { x: cb, .. } = self.ops[src].op
                        && matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. })
                    {
                        pops.push(cb);
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
                let Op::Load { src } = self.ops[op_id].op else {
                    unreachable!("tt_place_pops: dead op is a Load");
                };
                if let Op::GEP { x: cb, .. } = self.ops[src].op
                    && matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. })
                {
                    match first_wait.get(&cb) {
                        Some(wait) => deferred_drains.push((*wait, cb)),
                        None => pops.push(cb),
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
        // Load-src and Copy-src GEP indices past pops address
        // pop-relative pages. Rebase const indices by popped
        // pages (x1024 elems). Copy-dst (reserve-relative) and
        // Store-dst (push-relative) GEPs are other traffic and
        // stay, as does the whole reader section (no pops there).
        // Loop-varying indices stay: bracket windows hold the
        // pointer still mid-loop, so only const indices rebase.
        // A shared GEP is cloned, never mutated in place.
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
            if section != 0 {
                let read_src = match self.ops[scan].op {
                    Op::Load { src } => Some(src),
                    Op::Copy { src, .. } => Some(src),
                    _ => None,
                };
                if let Some(gep) = read_src
                    && let Op::GEP { x: base, index, layout } = self.ops[gep].op
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
                    let rebased = (dim as u64).checked_sub(pages * 1024).unwrap_or_else(|| {
                        panic!("tt_place_pops: rebase underflows CB{base:?} at {scan:?}")
                    });
                    if rebased != dim as u64 {
                        let nc = self.insert_const_idx_before(gep, rebased as i64);
                        let shared = users.get(&gep).is_some_and(|us| us.len() > 1);
                        if shared {
                            let ng = self.insert_before(scan, Op::GEP { x: base, index: nc, layout });
                            if matches!(self.ops[scan].op, Op::Copy { .. }) {
                                let Op::Copy { src, .. } = &mut self.ops[scan].op else {
                                    unreachable!("tt_place_pops: rebase src vanished")
                                };
                                *src = ng;
                            } else {
                                let Op::Load { src } = &mut self.ops[scan].op else {
                                    unreachable!("tt_place_pops: rebase src vanished")
                                };
                                *src = ng;
                            }
                        } else {
                            self.ops[gep].op = Op::GEP { x: base, index: nc, layout };
                        }
                    }
                }
            }
            scan = next;
        }

        self.tt_fifo_check();
        self.verify();
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

    /// Chase a `Load`/`Store`/`Copy` operand through its `GEP` to the
    /// underlying `Storage`/`Param`. Tile-compute results name DST slots,
    /// not buffers: `MatmulTile`/`ReduceTile` see through to their `acc`
    /// (a `load_register_tile`), so matmul→reduce acc chains resolve to
    /// the Register slot instead of panicking. `BroadcastTile` is a pure
    /// marker and forwards to its input. A post-lowering `LLK` resolves to
    /// the Register storage in its operands (its DST slot); an `LLK` with
    /// no Register operand names no slot — loud panic. The GEP index is
    /// not needed (render uses slot 0).
    pub(crate) fn tt_storage_of(&self, op_id: OpId) -> OpId {
        match self.at(op_id) {
            Op::Load { src, .. } | Op::Copy { src, .. } => self.tt_storage_of(*src),
            Op::Store { dst, .. } => self.tt_storage_of(*dst),
            Op::GEP { x, .. } => self.tt_storage_of(*x),
            Op::Storage { .. } | Op::Param { .. } => op_id,
            Op::TT(TTOp::MatmulTile { acc, .. }) | Op::TT(TTOp::ReduceTile { acc, .. }) => self.tt_storage_of(*acc),
            Op::TT(TTOp::BroadcastTile { x, .. }) => self.tt_storage_of(*x),
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
            Op::TT(TTOp::LLKReduce { slot, .. }) => *slot,
            Op::TT(TTOp::LLKBcast { .. }) => panic!("tt_storage_of: LLKBcast {op_id:?} names no Register slot"),
            other => panic!("tt_storage_of: {op_id:?} is not a Load/Store/Copy/GEP, got {other:?}"),
        }
    }

    /// Pack-store CB: `Some(cb)` iff `id` is a `Store` of a tile value
    /// into a Circular buffer (shared by `tt_lock_dst`, `tt_sync_cbs`
    /// accounting, and `tt_init_math` — single source of truth).
    fn tt_pack_store_cb(&self, id: OpId) -> Option<OpId> {
        if let Op::Store { src: x, dst } = self.at(id) {
            if let Op::GEP { x: g, .. } = self.at(*dst) {
                if matches!(self.at(*g), Op::Storage { scope: MemScope::Circular, .. })
                    && matches!(self.layout(*x), MemLayout::Tile { .. })
                {
                    return Some(*g);
                }
            }
        }
        None
    }

    /// True iff `id` is a `GEP` over a Circular storage (a CB slot
    /// address, either side of a traffic `Copy`).
    fn is_circular_gep(&self, id: OpId) -> bool {
        if let Op::GEP { x, .. } = self.at(id) {
            matches!(self.at(*x), Op::Storage { scope: MemScope::Circular, .. })
        } else {
            false
        }
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
