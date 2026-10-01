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

use crate::DType;
use crate::Map;
use crate::Set;
use crate::kernel::{BOp, FusedKind, Kernel, MemLayout, MemScope, Op, OpId, TTOp, TileDim, UOp};
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

    /// Unpack init constructor: `copy_tile_init(cb);`.
    ///
    /// Engine-config inits are LLK software wrappers, not hardware ops,
    /// so they live as [`Op::Asm`] statement templates over bare buffer
    /// ids — never as structured variants. The TT renderer substitutes
    /// the `{i}` placeholders and emits the template as a statement.
    /// Scope safety (bare Circular storages) is a constructor
    /// `debug_assert!`; TT render re-checks at compile time.
    pub fn tt_copy_init(&mut self, cb: OpId) -> OpId {
        debug_assert!(
            matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_copy_init: cb {cb} is not a Circular storage"
        );
        self.asm("copy_tile_init({0});", &[cb])
    }

    /// Unpack re-init constructor:
    /// `copy_tile_to_dst_init_short_with_dt(prev, cb);`.
    pub fn tt_copy_init_with_dt(&mut self, prev: OpId, cb: OpId) -> OpId {
        debug_assert!(
            matches!(self.at(prev), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_copy_init_with_dt: prev {prev} is not a Circular storage"
        );
        debug_assert!(
            matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_copy_init_with_dt: cb {cb} is not a Circular storage"
        );
        self.asm("copy_tile_to_dst_init_short_with_dt({0}, {1});", &[prev, cb])
    }

    /// Pack reconfig constructor: `pack_reconfig_data_format(cb);`.
    pub fn tt_pack_reconfig(&mut self, cb: OpId) -> OpId {
        debug_assert!(
            matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_pack_reconfig: cb {cb} is not a Circular storage"
        );
        self.asm("pack_reconfig_data_format({0});", &[cb])
    }

    /// Unary init constructor (`exp_tile_init();`, ...). The op selects
    /// the call — the table mirrors the LLK init names exactly.
    pub fn tt_unary_init(&mut self, uop: UOp) -> OpId {
        let init = match uop {
            UOp::Neg => "negative_tile_init();",
            UOp::BitNot => "bitwise_not_tile_init();",
            UOp::Exp => "exp_tile_init();",
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
        self.asm(init, &[])
    }

    /// Binary init constructor (`add_binary_tile_init();`, ...).
    pub fn tt_binary_init(&mut self, bop: BOp) -> OpId {
        let init = match bop {
            BOp::Add => "add_binary_tile_init();",
            BOp::Sub => "sub_binary_tile_init();",
            BOp::Mul => "mul_binary_tile_init();",
            BOp::Div => "div_binary_tile_init();",
            BOp::Max => "binary_max_tile_init();",
            BOp::BitShiftLeft | BOp::BitShiftRight => "binary_shift_tile_init();",
            _ => panic!("tt_binary_init: {bop:?} has no init call"),
        };
        self.asm(init, &[])
    }

    /// Scalar-binary init constructor.
    pub fn tt_bin_scalar_init(&mut self, bop: BOp) -> OpId {
        let init = match bop {
            BOp::Add | BOp::Sub | BOp::Mul | BOp::Div => "binop_with_scalar_tile_init();",
            BOp::BitShiftLeft => "left_shift_tile_init();",
            BOp::BitShiftRight => "right_shift_tile_init();",
            _ => panic!("tt_bin_scalar_init: scalar init for {bop:?} has no init call"),
        };
        self.asm(init, &[])
    }

    /// Cast init constructor: `typecast_tile_init<in, out>();` with TT
    /// `DataFormat` codes (not CB descriptor codes).
    pub fn tt_cast_init(&mut self, in_dtype: DType, out_dtype: DType) -> OpId {
        let template = format!("typecast_tile_init<{}, {}>();", tt_tile_fmt(in_dtype), tt_tile_fmt(out_dtype));
        self.asm(&template, &[])
    }

    /// Transpose init constructor: `transpose_wh_init(cb, out);`.
    pub fn tt_transpose_init(&mut self, cb: OpId, out: OpId) -> OpId {
        debug_assert!(
            matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_transpose_init: cb {cb} is not a Circular storage"
        );
        debug_assert!(
            matches!(self.at(out), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_transpose_init: out {out} is not a Circular storage"
        );
        self.asm("transpose_wh_init({0}, {1});", &[cb, out])
    }

    /// Matmul init constructor: `mm_init(a, b, out);`.
    pub fn tt_matmul_init(&mut self, a: OpId, b: OpId, out: OpId) -> OpId {
        for (name, buf) in [("a", a), ("b", b), ("out", out)] {
            debug_assert!(
                matches!(self.at(buf), Op::Storage { scope: MemScope::Circular, .. }),
                "tt_matmul_init: {name} {buf} is not a Circular storage"
            );
        }
        self.asm("mm_init({0}, {1}, {2});", &[a, b, out])
    }

    /// Compute startup constructor:
    /// `compute_kernel_hw_startup(in0, in1, out);`.
    pub fn tt_compute_startup(&mut self, in0: OpId, in1: OpId, out: OpId) -> OpId {
        for (name, buf) in [("in0", in0), ("in1", in1), ("out", out)] {
            debug_assert!(
                matches!(self.at(buf), Op::Storage { scope: MemScope::Circular, .. }),
                "tt_compute_startup: {name} {buf} is not a Circular storage"
            );
        }
        self.asm("compute_kernel_hw_startup({0}, {1}, {2});", &[in0, in1, out])
    }

    /// Reduce init constructor: `reduce_init<op, dim>(ci, cs, acc);`.
    pub fn tt_reduce_init(&mut self, ci: OpId, cs: OpId, acc: OpId, rop: BOp, kind: TileDim) -> OpId {
        for (name, buf) in [("ci", ci), ("cs", cs)] {
            debug_assert!(
                matches!(self.at(buf), Op::Storage { scope: MemScope::Circular, .. }),
                "tt_reduce_init: {name} {buf} is not a Circular storage"
            );
        }
        debug_assert!(
            matches!(self.at(acc), Op::Storage { scope: MemScope::Register, .. }),
            "tt_reduce_init: acc {acc} is not a Register slot"
        );
        let op_name = match rop {
            BOp::Max => "PoolType::MAX",
            BOp::Add => "PoolType::SUM",
            _ => panic!("tt_reduce_init: reduce op {rop:?} has no init call"),
        };
        let dim_name = match kind {
            TileDim::Row => "ReduceDim::REDUCE_ROW",
            TileDim::Col => "ReduceDim::REDUCE_COL",
            TileDim::Scalar => "ReduceDim::REDUCE_SCALAR",
        };
        let template = format!("reduce_init<{op_name}, {dim_name}>({{0}}, {{1}}, {{2}});");
        self.asm(&template, &[ci, cs, acc])
    }

    /// Reduce uninit constructor: `reduce_uninit();`. A structural
    /// effect (closes the reduce cone), so it stays a `TTOp`.
    pub fn tt_reduce_uninit(&mut self) -> OpId {
        self.push_back(Op::TT(TTOp::ReduceUninit))
    }

    /// Fused broadcast-binary init constructor.
    pub fn tt_bcast_init(&mut self, bop: BOp, kind: TileDim, cb_a: OpId, cb_b: OpId) -> OpId {
        debug_assert!(
            matches!(self.at(cb_a), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_bcast_init: cb_a {cb_a} is not a Circular storage"
        );
        debug_assert!(
            matches!(self.at(cb_b), Op::Storage { scope: MemScope::Circular, .. }),
            "tt_bcast_init: cb_b {cb_b} is not a Circular storage"
        );
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
            _ => panic!("tt_bcast_init: broadcast ({bop:?}, {kind:?}) has no init call"),
        };
        let template = format!("{init}({{0}}, {{1}});");
        self.asm(&template, &[cb_a, cb_b])
    }

    /// Fused-unary init constructor.
    pub fn tt_fused_init(&mut self, kind: FusedKind) -> OpId {
        let init = match kind {
            FusedKind::Sigmoid => "sigmoid_tile_init();",
            FusedKind::Silu => "silu_tile_init();",
        };
        self.asm(init, &[])
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
    pub(crate) fn tt_lock_dst(&mut self) {
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
            matches!(op, Op::Asm { .. }) || tt_is_tile_value(kernel, id)
        };
        // Pack drain: a Store into a Circular buffer of a tile value
        // (Register slot, circular load, or SSA/LLK/Asm tile result).
        // The drain empties DST into a CB.
        let is_pack = |kernel: &Kernel, id: OpId| {
            if let Op::Store { src: x, dst } = kernel.at(id) {
                if let Op::GEP { x: g, .. } = kernel.at(*dst) {
                    if matches!(kernel.at(*g), Op::Storage { scope: MemScope::Circular, .. }) {
                        return tt_is_tile_value(kernel, *x);
                    }
                }
            }
            false
        };
        let mut op_id = self.head;
        while !op_id.is_null() {
            let mut math_first: Option<OpId> = None;
            let mut math_last: Option<OpId> = None;
            let mut pack_first: Option<OpId> = None;
            let mut pack_last: Option<OpId> = None;
            // Loop nesting for the back-edge hoist below: per-op loop
            // stack snapshots at the MATH endpoints, Loop→EndLoop
            // matching, section-relative positions, and pack-store
            // positions (a pack boundary stops the hoist bubbling).
            let mut loop_stack: Vec<OpId> = Vec::new();
            let mut loop_end: Map<OpId, OpId> = Map::default();
            let mut first_loops: Vec<OpId> = Vec::new();
            let mut last_loops: Vec<OpId> = Vec::new();
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
                    if math_first.is_none() {
                        math_first = Some(scan);
                        first_loops = loop_stack.clone();
                    }
                    math_last = Some(scan);
                    last_loops = loop_stack.clone();
                }
                if is_pack(self, scan) {
                    if pack_first.is_none() {
                        pack_first = Some(scan);
                    }
                    pack_last = Some(scan);
                    pack_pos.insert(scan);
                }
                scan = self.next_op(scan);
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
            let mut acquire_at = math_first;
            let mut crossed = 0usize;
            if let Some(first) = math_first {
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
                acquire_at = Some(target);
            }
            let mut commit_at = math_last;
            if let Some(last) = math_last {
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
                commit_at = Some(target);
            }
            // Commit before wait: MATH drains to registers, then packs
            // read them. Inserts are by OpId, so order among the four
            // is independent of position.
            if let Some(at) = acquire_at {
                self.insert_before(at, Op::TT(TTOp::MathLock));
            }
            if let Some(at) = commit_at {
                self.insert_after(at, Op::TT(TTOp::MathUnlock));
            }
            if let Some(first) = pack_first {
                self.insert_before(first, Op::TT(TTOp::PackLock));
            }
            if let Some(last) = pack_last {
                self.insert_after(last, Op::TT(TTOp::PackUnlock));
            }
            op_id = match section_end {
                Some(b) => self.next_op(b),
                None => break,
            }
        }

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
    pub(crate) fn tt_storage(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            // Canonicalize CB→CB moves to load+store spelling first: a
            // fused copy would straddle the MATH/PACK cone boundary
            // (unpack needs MATH-open, pack needs PACK-open), while the
            // split form lowers each half under its own cone like
            // every other pack. Same CBs, same indices, same traffic.
            if let Op::Copy { src, dst } = self.ops[op_id].op {
                if is_circular_gep(self, src) && is_circular_gep(self, dst) {
                    let load = self.insert_before(op_id, Op::Load { src });
                    self.ops[op_id].op = Op::Store { dst, src: load };
                    op_id = next;
                    continue;
                }
            }
            let (asm, ops) = match self.at(op_id) {
                Op::TT(TTOp::MatmulTile { x, y, acc }) => {
                    let cb_a = tt_storage_of(self, *x);
                    let cb_b = tt_storage_of(self, *y);
                    let slot = tt_storage_of(self, *acc);
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
                    (
                        TinyString::new("matmul_tiles({0}, {1}, 0, 0, {2});"),
                        TinyVec::new(&[cb_a, cb_b, slot, *x, *y]),
                    )
                }
                Op::TT(TTOp::ReduceTile { x, scaler, acc, rop, kind }) => {
                    let cb_in = tt_storage_of(self, *x);
                    let cb_sc = tt_storage_of(self, *scaler);
                    let slot = tt_storage_of(self, *acc);
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
                    let op_name = match rop {
                        BOp::Max => "PoolType::MAX",
                        BOp::Add => "PoolType::SUM",
                        _ => panic!("tt_storage: reduce op {rop:?} has no LLK call"),
                    };
                    let dim_name = match kind {
                        TileDim::Row => "ReduceDim::REDUCE_ROW",
                        TileDim::Col => "ReduceDim::REDUCE_COL",
                        TileDim::Scalar => "ReduceDim::REDUCE_SCALAR",
                    };
                    let template = format!("reduce_tile<{op_name}, {dim_name}>({{0}}, {{1}}, 0, 0, {{2}});");
                    (TinyString::new(&template), TinyVec::new(&[cb_in, cb_sc, slot, *x, *scaler]))
                }
                Op::TT(TTOp::TransposeTile { x }) => {
                    let cb = tt_storage_of(self, *x);
                    debug_assert!(
                        matches!(self.at(cb), Op::Storage { scope: MemScope::Circular, .. }),
                        "tt_storage: transpose input {cb:?} is not a Circular storage"
                    );
                    // The transpose result lands in a DST slot filled by
                    // render (`{1}`); the trailing entry is the feeder
                    // load, for sync accounting only.
                    (
                        TinyString::new("transpose_wh_tile({0}, 0, {1});"),
                        TinyVec::new(&[cb, OpId::NULL, *x]),
                    )
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
                    let cb_b = tt_storage_of(self, *mx);
                    if !matches!(self.at(cb_b), Op::Storage { scope: MemScope::Circular, .. }) {
                        panic!("tt_storage: broadcast {op_id:?} marked side is no CB tile load");
                    }
                    if !matches!(
                        self.at(plain),
                        Op::Load { .. } | Op::TT(TTOp::BroadcastTile { .. })
                    ) {
                        panic!("tt_storage: broadcast {op_id:?} plain side is no CB tile load");
                    }
                    let cb_a = tt_storage_of(self, plain);
                    if !matches!(self.at(cb_a), Op::Storage { scope: MemScope::Circular, .. }) {
                        panic!("tt_storage: broadcast {op_id:?} plain side is no CB tile load");
                    }
                    let name = match (*bop, *kind) {
                        (BOp::Add, TileDim::Row) => "add_tiles_bcast_rows",
                        (BOp::Add, TileDim::Col) => "add_tiles_bcast_cols",
                        (BOp::Add, TileDim::Scalar) => "add_tiles_bcast_scalar",
                        (BOp::Sub, TileDim::Row) => "sub_tiles_bcast_rows",
                        (BOp::Sub, TileDim::Col) => "sub_tiles_bcast_cols",
                        (BOp::Sub, TileDim::Scalar) => "sub_tiles_bcast_scalar",
                        (BOp::Mul, TileDim::Row) => "mul_tiles_bcast_rows",
                        (BOp::Mul, TileDim::Col) => "mul_tiles_bcast_cols",
                        (BOp::Mul, TileDim::Scalar) => "mul_tiles_bcast_scalar",
                        _ => panic!("tt_storage: broadcast ({bop:?}, {kind:?}) has no fused call"),
                    };
                    let template = if matches!(kind, TileDim::Row) {
                        format!("{name}({{0}}, {{1}}, 0, 0, {{2}}, 0);")
                    } else {
                        format!("{name}({{0}}, {{1}}, 0, 0, {{2}});")
                    };
                    (
                        TinyString::new(&template),
                        TinyVec::new(&[cb_a, cb_b, OpId::NULL, *mx, plain]),
                    )
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
    pub(crate) fn tt_sync_cbs(&mut self) {
        // Linear user map: users[v] = ops taking v as a data operand.
        let mut users: Map<OpId, Vec<OpId>> = Map::default();
        let mut scan = self.head;
        while !scan.is_null() {
            for p in self.at(scan).parameters() {
                users.entry(p).or_default().push(scan);
            }
            scan = self.next_op(scan);
        }

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match self.ops[op_id].op {
                Op::Copy { src, dst } => {
                    let src_cb = match self.ops[src].op {
                        Op::GEP { x, .. }
                            if matches!(self.ops[x].op, Op::Storage { scope: MemScope::Circular, .. }) =>
                        {
                            Some(x)
                        }
                        Op::GEP { .. } => None,
                        Op::Storage { scope: MemScope::Circular, .. } => {
                            panic!("tt_sync_cbs: copy src {src:?} names a Circular storage directly, must go through a GEP")
                        }
                        _ => None,
                    };
                    let dst_cb = match self.ops[dst].op {
                        Op::GEP { x, .. }
                            if matches!(self.ops[x].op, Op::Storage { scope: MemScope::Circular, .. }) =>
                        {
                            Some(x)
                        }
                        Op::GEP { .. } => None,
                        Op::Storage { scope: MemScope::Circular, .. } => {
                            panic!("tt_sync_cbs: copy dst {dst:?} names a Circular storage directly, must go through a GEP")
                        }
                        _ => None,
                    };
                    match (src_cb, dst_cb) {
                        (None, None) => {}
                        (None, Some(cb)) => {
                            self.insert_before(op_id, Op::TT(TTOp::ReserveBack { cb, n: 1 }));
                            self.insert_after(op_id, Op::TT(TTOp::PushBack { cb, n: 1 }));
                        }
                        (Some(cb), None) => {
                            self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb, n: 1 }));
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
                    if is_circular_gep(self, dst) && tt_is_tile_value(self, x) {
                        self.insert_before(op_id, Op::TT(TTOp::ReserveBack { cb: tt_storage_of(self, dst), n: 1 }));
                        let cb = tt_storage_of(self, dst);
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
                            // Dead load: no SSA consumer reads. Either an
                            // LLK consumes it through provenance (waited
                            // at the call, see below) or it is a drain
                            // pop (popped by `tt_place_pops`).
                            None => {}
                            Some(use_list) => {
                                for &u in use_list {
                                    let ok = match &self.ops[u].op {
                                        Op::TT(
                                            TTOp::MatmulTile { .. }
                                            | TTOp::TransposeTile { .. }
                                            | TTOp::ReduceTile { .. }
                                            | TTOp::BroadcastTile { .. }
                                            | TTOp::LLK { .. },
                                        ) => true,
                                        Op::Store { .. } | Op::Asm { .. } => true,
                                        op => {
                                            tt_is_tile_value(self, u)
                                                && matches!(
                                                    op,
                                                    Op::Unary { .. }
                                                        | Op::Binary { .. }
                                                        | Op::Cast { .. }
                                                        | Op::Bitcast { .. }
                                                        | Op::Mad { .. }
                                                )
                                        }
                                    };
                                    if !ok {
                                        panic!(
                                            "tt_sync_cbs: circular load {op_id:?} feeds non-compute {u:?}"
                                        );
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
                                        Op::TT(TTOp::BroadcastTile { .. }) | Op::TT(TTOp::LLK { .. })
                                    )
                                });
                                if !fused_only {
                                    self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb, n: 1 }));
                                }
                            }
                        }
                    }
                }
                Op::TT(TTOp::LLK { .. }) => {
                    // Provenance waits: the call consumes its input CBs
                    // straight from the buffers, so each trailing feeder
                    // load gets a wait immediately before the call.
                    let Op::TT(TTOp::LLK { ops, .. }) = &self.ops[op_id].op else {
                        unreachable!("tt_sync_cbs: not an LLK");
                    };
                    let mut wait_cbs: Vec<OpId> = Vec::new();
                    for o in ops.iter().copied() {
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
    pub(crate) fn tt_init_math(&mut self) {
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
                    if let Some(cb) = tt_pack_store_cb(self, scan) {
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
        let mut last_math_unlock: Option<OpId> = None;
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            match &self.ops[op_id].op {
                Op::TT(TTOp::EndReader) => {
                    in_compute = true;
                    unpack_src = None;
                    pending_acc = None;
                }
                Op::TT(TTOp::EndCompute) => {
                    in_compute = false;
                    pending_acc = None;
                }
                Op::TT(TTOp::MathUnlock) if in_compute => last_math_unlock = Some(op_id),
                _ => {}
            }
            if !in_compute {
                // Reader/writer move through `Copy` only; any tile
                // compute, fused call, pack store, or CB load outside
                // the compute section is malformed.
                let bad = match &self.ops[op_id].op {
                    Op::TT(TTOp::LLK { .. }) => true,
                    Op::Store { .. } => tt_pack_store_cb(self, op_id).is_some(),
                    Op::Load { src } => {
                        matches!(self.ops[*src].op, Op::GEP { x, .. } if matches!(self.ops[x].op, Op::Storage { scope: MemScope::Circular, .. }))
                    }
                    op => {
                        let id = op_id;
                        tt_is_tile_value(self, id)
                            && matches!(
                                op,
                                Op::Unary { .. }
                                    | Op::Binary { .. }
                                    | Op::Cast { .. }
                                    | Op::Bitcast { .. }
                                    | Op::Mad { .. }
                                    | Op::Asm { .. }
                            )
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
                    match tt_scalar_lane(self, x, y) {
                        None => InitAction::BinaryInit(bop),
                        Some((left, s)) => {
                            tt_check_scalar_lane(self, op_id, bop, left, s);
                            InitAction::BinScalarInit(bop)
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
                Op::Mad { .. } => panic!("tt_init_math: tiled mad {op_id:?} has no LLK call"),
                Op::Copy { src, dst } => {
                    if is_circular_gep(self, *src) && is_circular_gep(self, *dst) {
                        let Op::GEP { x: dcb, .. } = self.ops[*dst].op else {
                            unreachable!("tt_init_math: CB copy dst is a GEP");
                        };
                        InitAction::PackReconfig(dcb)
                    } else {
                        InitAction::None
                    }
                }
                Op::Store { .. } => match tt_pack_store_cb(self, op_id) {
                    Some(cb) => InitAction::PackReconfig(cb),
                    None => InitAction::None,
                },
                Op::TT(TTOp::LLK { .. }) => {
                    let Op::TT(TTOp::LLK { asm, ops }) = &self.ops[op_id].op else {
                        unreachable!("tt_init_math: not an LLK");
                    };
                    let text = asm.as_str();
                    if text.starts_with("matmul_tiles(") {
                        let out = first_pack.unwrap_or_else(|| {
                            panic!("tt_init_math: matmul {op_id:?} with no packed CB")
                        });
                        InitAction::MatmulInit(ops[0], ops[1], out)
                    } else if text.starts_with("transpose_wh_tile(") {
                        let out = first_pack.unwrap_or_else(|| {
                            panic!("tt_init_math: transpose {op_id:?} with no packed CB")
                        });
                        InitAction::TransposeInit(ops[0], out)
                    } else if text.starts_with("reduce_tile<") {
                        let (rop, kind) = tt_parse_reduce(text, op_id);
                        let mut acc = None;
                        for o in ops.iter().copied() {
                            if o.is_null() {
                                continue;
                            }
                            if matches!(self.ops[o].op, Op::Storage { scope: MemScope::Register, .. }) {
                                acc = Some(o);
                                break;
                            }
                        }
                        let acc = acc.unwrap_or_else(|| {
                            panic!("tt_init_math: reduce {op_id:?} names no acc slot")
                        });
                        pending_acc = Some(acc);
                        InitAction::ReduceInit(ops[0], ops[1], acc, rop, kind)
                    } else if let Some((bop, kind)) = tt_parse_bcast(text) {
                        InitAction::BcastInit(bop, kind, ops[0], ops[1])
                    } else {
                        // User asm / startup template: no init.
                        InitAction::None
                    }
                }
                _ => InitAction::None,
            };
            // Unpack init at the load: a circular load consumed through
            // a copy (tiled elementwise SSA, a direct pack store) is
            // configured here, ahead of every copy site in walk (hence
            // runtime) order. Fused consumers (LLK provenance, user Asm,
            // markers) and dead loads need none.
            if let Op::Load { src } = self.ops[op_id].op {
                if let Op::GEP { x: cb, .. } = self.ops[src].op
                    && matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. })
                    && users.get(&op_id).is_some_and(|us| {
                        us.iter().any(|u| {
                            matches!(
                                self.ops[*u].op,
                                Op::Unary { .. } | Op::Binary { .. } | Op::Cast { .. } | Op::Bitcast { .. } | Op::Mad { .. }
                            ) || matches!(self.ops[*u].op, Op::Store { src: x, .. } if x == op_id)
                        })
                    })
                {
                    tt_emit_copy_init(self, op_id, cb, &mut unpack_src);
                }
            }
            // The op init itself. Same templates as the `tt_*_init`
            // constructors (which push back — wrong end for a pass).
            let asm_before = |kernel: &mut Kernel, at: OpId, template: &str, ops: &[OpId]| {
                kernel.insert_before(at, Op::Asm {
                    asm: TinyString::new(template),
                    ops: TinyVec::new(ops),
                });
            };
            match action {
                InitAction::None => {}
                InitAction::UnaryInit(uop) => {
                    let init = match uop {
                        UOp::Neg => "negative_tile_init();",
                        UOp::BitNot => "bitwise_not_tile_init();",
                        UOp::Exp => "exp_tile_init();",
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
                    let template =
                        format!("typecast_tile_init<{}, {}>();", tt_tile_fmt(in_dtype), tt_tile_fmt(out_dtype));
                    asm_before(self, op_id, &template, &[]);
                }
                InitAction::PackReconfig(cb) => {
                    asm_before(self, op_id, "pack_reconfig_data_format({0});", &[cb]);
                    // Reduce cone: packing the pending acc closes it.
                    if let Some(acc) = pending_acc {
                        let Op::Store { src: x, .. } = self.ops[op_id].op else {
                            unreachable!("tt_init_math: pack is a Store");
                        };
                        if tt_reaches_acc(self, x, acc) {
                            let unlock = last_math_unlock.unwrap_or_else(|| {
                                panic!("tt_init_math: pack {op_id:?} without an open MathUnlock")
                            });
                            self.insert_before(unlock, Op::TT(TTOp::ReduceUninit));
                            pending_acc = None;
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
                        Op::TT(TTOp::MathLock)
                            | Op::TT(TTOp::MathUnlock)
                            | Op::TT(TTOp::PackLock)
                            | Op::TT(TTOp::PackUnlock)
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
    pub(crate) fn tt_place_pops(&mut self) {
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

        // Provenance remaining counts: (cb storage, load) -> calls.
        let mut remaining: Map<(OpId, OpId), u32> = Map::default();
        let mut walk = self.head;
        while !walk.is_null() {
            if let Op::TT(TTOp::LLK { ops, .. }) = &self.ops[walk].op {
                for o in ops.iter().copied() {
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
                    Op::TT(TTOp::LLK { .. }) | Op::TT(TTOp::BroadcastTile { .. })
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
                            let is_drain = matches!(self.ops[op_id].op, Op::Copy { dst, .. } if !is_circular_gep(self, dst));
                            if !is_drain {
                                panic!("tt_place_pops: CB→CB copy {op_id:?} outside the compute section");
                            }
                        }
                        pops.push(cb);
                    }
                }
                Op::TT(TTOp::LLK { .. }) => {
                    let Op::TT(TTOp::LLK { ops, .. }) = &self.ops[op_id].op else {
                        unreachable!("tt_place_pops: not an LLK");
                    };
                    let mut seen: Vec<OpId> = Vec::new();
                    let mut cbs: Vec<(OpId, OpId)> = Vec::new();
                    for o in ops.iter().copied() {
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
                        let left = remaining.get_mut(&(cb, load)).unwrap_or_else(|| {
                            panic!("tt_place_pops: LLK {op_id:?} consumes uncounted page ({cb:?}, {load:?})")
                        });
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
                if *user == op_id {
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
            // Dead-load drain pops land at the load itself.
            if matches!(self.ops[op_id].op, Op::Load { .. }) && !users.contains_key(&op_id) {
                let Op::Load { src } = self.ops[op_id].op else {
                    unreachable!("tt_place_pops: dead op is a Load");
                };
                if let Op::GEP { x: cb, .. } = self.ops[src].op
                    && matches!(self.ops[cb].op, Op::Storage { scope: MemScope::Circular, .. })
                {
                    pops.push(cb);
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
pub(crate) fn tt_storage_of(kernel: &Kernel, op_id: OpId) -> OpId {
    match kernel.at(op_id) {
        Op::Load { src, .. } | Op::Copy { src, .. } => tt_storage_of(kernel, *src),
        Op::Store { dst, .. } => tt_storage_of(kernel, *dst),
        Op::GEP { x, .. } => tt_storage_of(kernel, *x),
        Op::Storage { .. } | Op::Param { .. } => op_id,
        Op::TT(TTOp::MatmulTile { acc, .. }) | Op::TT(TTOp::ReduceTile { acc, .. }) => {
            tt_storage_of(kernel, *acc)
        }
        Op::TT(TTOp::BroadcastTile { x, .. }) => tt_storage_of(kernel, *x),
        Op::TT(TTOp::LLK { ops, .. }) => {
            let mut slot = None;
            for o in ops.iter().copied() {
                if o.is_null() {
                    continue;
                }
                if matches!(kernel.at(o), Op::Storage { scope: MemScope::Register, .. }) {
                    slot = Some(o);
                    break;
                }
            }
            slot.unwrap_or_else(|| panic!("tt_storage_of: LLK {op_id:?} names no Register slot"))
        }
        other => panic!("tt_storage_of: {op_id:?} is not a Load/Store/Copy/GEP, got {other:?}"),
    }
}

/// Pack-store CB: `Some(cb)` iff `id` is a `Store` of a tile value
/// into a Circular buffer (shared by `tt_lock_dst`, `tt_sync_cbs`
/// accounting, and `tt_init_math` — single source of truth).
fn tt_pack_store_cb(kernel: &Kernel, id: OpId) -> Option<OpId> {
    if let Op::Store { src: x, dst } = kernel.at(id) {
        if let Op::GEP { x: g, .. } = kernel.at(*dst) {
            if matches!(kernel.at(*g), Op::Storage { scope: MemScope::Circular, .. })
                && tt_is_tile_value(kernel, *x)
            {
                return Some(*g);
            }
        }
    }
    None
}

/// Validate a scalar binary lane at init time (old scalar rule): int
/// consts only fold for shifts (right side, 0..=31); float
/// const-exprs fold for Add/Mul either side and Div on the right;
/// scalar `Sub` is unrenderable (the old render rejected it too);
/// anything else is a loud error.
fn tt_check_scalar_lane(kernel: &Kernel, id: OpId, bop: BOp, left: bool, s: OpId) {
    match bop {
        BOp::BitShiftLeft | BOp::BitShiftRight => {
            if left {
                panic!("tt_init_math: const-first shift {id:?} has no scalar call");
            }
            match kernel.at(s) {
                Op::Const(c) => match c.as_dim() {
                    Some(v) if v <= 31 => {}
                    _ => panic!("tt_init_math: shift amount {id:?} is not a u32 0..=31"),
                },
                _ => panic!("tt_init_math: shift amount {id:?} is not an int const"),
            }
        }
        BOp::Add | BOp::Mul => {
            tt_scalar_f32(kernel, s);
        }
        BOp::Div => {
            if left {
                panic!("tt_init_math: const-first div {id:?} has no scalar call");
            }
            tt_scalar_f32(kernel, s);
        }
        BOp::Sub => {
            panic!("tt_init_math: scalar sub {id:?} has no LLK call (the old render rejected it too)")
        }
        _ => panic!("tt_init_math: scalar {bop:?} {id:?} has no scalar call"),
    }
}

/// Parse a baked reduce template (`reduce_tile<PoolType::MAX,
/// ReduceDim::REDUCE_COL>(...)`, as produced by `tt_storage`) back to
/// `(rop, kind)`. Six combos — anything else is not a storage
/// template, loud panic.
fn tt_parse_reduce(text: &str, id: OpId) -> (BOp, TileDim) {
    if let Some(rest) = text.strip_prefix("reduce_tile<PoolType::") {
        let (rop, rest) = if let Some(r) = rest.strip_prefix("MAX, ") {
            (BOp::Max, r)
        } else if let Some(r) = rest.strip_prefix("SUM, ") {
            (BOp::Add, r)
        } else {
            panic!("tt_init_math: reduce {id:?} has unknown pool type");
        };
        let kind = if rest.starts_with("ReduceDim::REDUCE_ROW>(") {
            TileDim::Row
        } else if rest.starts_with("ReduceDim::REDUCE_COL>(") {
            TileDim::Col
        } else if rest.starts_with("ReduceDim::REDUCE_SCALAR>(") {
            TileDim::Scalar
        } else {
            panic!("tt_init_math: reduce {id:?} has unknown dim");
        };
        return (rop, kind);
    }
    panic!("tt_init_math: {id:?} is not a storage reduce template");
}

/// Parse a fused broadcast template (`add_tiles_bcast_rows(...)`, as
/// produced by `tt_storage` fusion) back to `(bop, kind)`. Nine
/// combos; anything else is not a fused template (`None` = leave the
/// `LLK` alone — user asm / startup templates).
fn tt_parse_bcast(text: &str) -> Option<(BOp, TileDim)> {
    let (bop, rest) = if let Some(r) = text.strip_prefix("add_tiles_bcast_") {
        (BOp::Add, r)
    } else if let Some(r) = text.strip_prefix("sub_tiles_bcast_") {
        (BOp::Sub, r)
    } else if let Some(r) = text.strip_prefix("mul_tiles_bcast_") {
        (BOp::Mul, r)
    } else {
        return None;
    };
    if rest.starts_with("rows(") {
        Some((bop, TileDim::Row))
    } else if rest.starts_with("cols(") {
        Some((bop, TileDim::Col))
    } else if rest.starts_with("scalar(") {
        Some((bop, TileDim::Scalar))
    } else {
        None
    }
}

/// True iff the value of `id` is (or threads) the Register acc
/// `acc`: a load of it, elementwise SSA over it, an `LLK` writing
/// it, or user `Asm` over it. Drives `close_reduce_cones`: packing
/// such a value closes the pending reduce. Explicit arms throughout
/// (no catch-all); graph-only ops panic.
fn tt_reaches_acc(kernel: &Kernel, id: OpId, acc: OpId) -> bool {
    match kernel.at(id) {
        Op::Load { src } => matches!(kernel.at(*src), Op::GEP { x, .. } if *x == acc),
        Op::Unary { x, .. } | Op::Cast { x, .. } | Op::Bitcast { x, .. } => tt_reaches_acc(kernel, *x, acc),
        Op::Binary { x, y, .. } => tt_reaches_acc(kernel, *x, acc) || tt_reaches_acc(kernel, *y, acc),
        Op::Mad { x, y, z, .. } => {
            tt_reaches_acc(kernel, *x, acc) || tt_reaches_acc(kernel, *y, acc) || tt_reaches_acc(kernel, *z, acc)
        }
        Op::TT(TTOp::LLK { ops, .. }) => ops.iter().copied().any(|o| !o.is_null() && o == acc),
        Op::TT(TTOp::MatmulTile { acc: a, .. }) | Op::TT(TTOp::ReduceTile { acc: a, .. }) => {
            *a == acc || tt_reaches_acc(kernel, *a, acc)
        }
        Op::TT(TTOp::BroadcastTile { x, .. }) => tt_reaches_acc(kernel, *x, acc),
        Op::TT(TTOp::TransposeTile { .. }) => false,
        Op::Asm { ops, .. } => {
            ops.iter().copied().any(|o| !o.is_null() && tt_reaches_acc(kernel, o, acc))
        }
        Op::Const { .. }
        | Op::Param { .. }
        | Op::Storage { .. }
        | Op::GEP { .. }
        | Op::Store { .. }
        | Op::Copy { .. }
        | Op::Range { .. }
        | Op::Loop { .. }
        | Op::EndLoop
        | Op::Barrier => false,
        Op::TT(_)
        | Op::Stack { .. }
        | Op::Index { .. }
        | Op::Wmma { .. }
        | Op::Reshape { .. }
        | Op::Expand { .. }
        | Op::Permute { .. }
        | Op::Flip { .. }
        | Op::Pad { .. }
        | Op::Narrow { .. }
        | Op::Reduce { .. }
        | Op::After { .. }
        | Op::ToDevice { .. }
        | Op::Contiguous { .. }
        | Op::Kernel { .. }
        | Op::Custom(_) => panic!("tt_reaches_acc: graph-only {id:?} in ordered TT kernel"),
    }
}

/// Emit `copy_tile_init(cb)` / `copy_tile_to_dst_init_short_with_dt
/// (prev, cb)` before `at`, tracking the unpack source. Same shapes
/// as `tt_copy_init` / `tt_copy_init_with_dt` (the ctors push back —
/// wrong end for a pass; templates are stable LLK names).
fn tt_emit_copy_init(kernel: &mut Kernel, at: OpId, cb: OpId, unpack_src: &mut Option<OpId>) {
    match *unpack_src {
        Some(prev) if prev == cb => {
            kernel.insert_before(at, Op::Asm {
                asm: TinyString::new("copy_tile_init({0});"),
                ops: TinyVec::new(&[cb]),
            });
        }
        Some(prev) => {
            kernel.insert_before(at, Op::Asm {
                asm: TinyString::new("copy_tile_to_dst_init_short_with_dt({0}, {1});"),
                ops: TinyVec::new(&[prev, cb]),
            });
            *unpack_src = Some(cb);
        }
        None => {
            kernel.insert_before(at, Op::Asm {
                asm: TinyString::new("copy_tile_init({0});"),
                ops: TinyVec::new(&[cb]),
            });
            *unpack_src = Some(cb);
        }
    }
}

/// True iff `id` is a `GEP` over a Circular storage (a CB slot
/// address, either side of a traffic `Copy`).
pub(crate) fn is_circular_gep(kernel: &Kernel, id: OpId) -> bool {
    if let Op::GEP { x, .. } = kernel.at(id) {
        matches!(kernel.at(*x), Op::Storage { scope: MemScope::Circular, .. })
    } else {
        false
    }
}

/// True iff the value of `id` is a Tenstorrent tile (unpacks from /
/// packs into a CB, lives in DST). `Load` of a tile layout, the four
/// SSA tile ops, `LLK`, tiled user `Asm`, and elementwise SSA over
/// tile operands are tiles; `Const`/`Param`/`Storage`/`Range`/`Loop`
/// values and scalar index math (`Const`/`Mad`/`Binary` over
/// non-tiles) are not. Graph-only ops can never appear in an
/// ordered TT kernel — loud `panic`, never a default arm.
pub(crate) fn tt_is_tile_value(kernel: &Kernel, id: OpId) -> bool {
    match kernel.at(id) {
        Op::Load { src } => matches!(
            kernel.at(*src),
            Op::GEP { layout: MemLayout::Tile { .. }, .. }
        ),
        Op::Unary { x, .. } | Op::Cast { x, .. } | Op::Bitcast { x, .. } => tt_is_tile_value(kernel, *x),
        Op::Binary { x, y, .. } => tt_is_tile_value(kernel, *x) || tt_is_tile_value(kernel, *y),
        Op::Mad { x, y, z, .. } => {
            tt_is_tile_value(kernel, *x) || tt_is_tile_value(kernel, *y) || tt_is_tile_value(kernel, *z)
        }
        Op::Asm { .. } => true,
        Op::TT(_) => true,
        Op::Const { .. }
        | Op::Param { .. }
        | Op::Storage { .. }
        | Op::GEP { .. }
        | Op::Store { .. }
        | Op::Copy { .. }
        | Op::Range { .. }
        | Op::Loop { .. }
        | Op::EndLoop
        | Op::Barrier => false,
        Op::Stack { .. }
        | Op::Index { .. }
        | Op::Wmma { .. }
        | Op::Reshape { .. }
        | Op::Expand { .. }
        | Op::Permute { .. }
        | Op::Flip { .. }
        | Op::Pad { .. }
        | Op::Narrow { .. }
        | Op::Reduce { .. }
        | Op::After { .. }
        | Op::ToDevice { .. }
        | Op::Contiguous { .. }
        | Op::Kernel { .. }
        | Op::Custom(_) => panic!("tt_is_tile_value: graph-only {id:?} in ordered TT kernel"),
    }
}

/// Scalar lane of a tiled `Binary`: which side (if any) is the
/// scalar, and the scalar op. A tiled binary has at least one tile
/// lane; the other lane is scalar iff it is not tile-valued. Both
/// lanes tile → `None` (plain tile-tile call).
pub(crate) fn tt_scalar_lane(kernel: &Kernel, x: OpId, y: OpId) -> Option<(bool, OpId)> {
    let tx = tt_is_tile_value(kernel, x);
    let ty = tt_is_tile_value(kernel, y);
    match (tx, ty) {
        (true, true) => None,
        (true, false) => Some((false, y)),
        (false, true) => Some((true, x)),
        (false, false) => panic!("tt_scalar_lane: tiled binary with no tile lane"),
    }
}

/// Resolve a float scalar const-expr to `f32`: a float `Const`, or a
/// `Cast` of one (rounded through the cast target, mirroring the
/// device-side value). Anything else is not a foldable scalar — loud
/// panic. Int consts never fold (except shift amounts, resolved with
/// `as_dim` at the use site).
pub(crate) fn tt_scalar_f32(kernel: &Kernel, id: OpId) -> f32 {
    use crate::dtype::Constant;
    use crate::scalar::{bf16, f16};
    match kernel.at(id) {
        Op::Const(Constant::F32(b)) => f32::from_le_bytes(*b),
        Op::Const(Constant::F16(b)) => f16(u16::from_le_bytes(*b)).to_f32(),
        Op::Const(Constant::BF16(b)) => bf16(u16::from_le_bytes(*b)).to_f32(),
        Op::Const(_) => panic!("tt_scalar_f32: {id:?} is not a float const"),
        Op::Cast { x, dtype } => {
            let v = tt_scalar_f32(kernel, *x);
            match dtype {
                crate::DType::F32 => v,
                crate::DType::BF16 => bf16::from_f32(v).to_f32(),
                crate::DType::F16 => f16::from_f32(v).to_f32(),
                dt => panic!("tt_scalar_f32: cannot fold scalar through cast to {dt:?}"),
            }
        }
        _ => panic!("tt_scalar_f32: {id:?} is not a float const-expr"),
    }
}

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

#[cfg(test)]
mod tests {
    use crate::DType;
    use crate::kernel::{BOp, Dev, Kernel, MemScope, Op, OpId, TTOp, TileDim};

    /// MATH tile-compute runs under `tile_regs_acquire..commit`;
    /// the pack drain (Register slot stored into a Circular buffer)
    /// runs under `tile_regs_wait..release`. Pure insertion: the
    /// physical DST slots are the `MemScope::Register` storages
    /// already in the IR, this pass assigns no numbers.
    #[test]
    fn lock_dst_wraps_math_and_pack() {
        let mut k = Kernel::from_device_id(Dev::Auto, None);
        let cb_a = k.storage(DType::F32, MemScope::Circular, 1024);
        let cb_b = k.storage(DType::F32, MemScope::Circular, 1024);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let cout = k.storage(DType::F32, MemScope::Circular, 1024);
        let zero = k.const_val(0i64);
        k.tt_end_reader();
        let va = k.load_circular(cb_a, zero);
        let vb = k.load_circular(cb_b, zero);
        let av = k.load_register_tile(acc, zero);
        let f = k.matmul_tile(va, vb, av);
        k.store_register_tile(acc, f, zero);
        let out = k.load_register_tile(acc, zero);
        k.store_circular(cout, out, zero);
        k.tt_end_compute();
        k.tt_lock_dst();

        let mut ops: Vec<Op> = Vec::new();
        let mut op_id = k.head;
        while !op_id.is_null() {
            ops.push(k.at(op_id).clone());
            op_id = k.next_op(op_id);
        }
        let math_locks = ops.iter().filter(|op| matches!(op, Op::TT(TTOp::MathLock))).count();
        let math_unlocks = ops.iter().filter(|op| matches!(op, Op::TT(TTOp::MathUnlock))).count();
        let pack_locks = ops.iter().filter(|op| matches!(op, Op::TT(TTOp::PackLock))).count();
        let pack_unlocks = ops.iter().filter(|op| matches!(op, Op::TT(TTOp::PackUnlock))).count();
        assert_eq!(math_locks, 1, "one MATH acquire per section with MATH ops");
        assert_eq!(math_unlocks, 1, "one MATH commit per section with MATH ops");
        assert_eq!(pack_locks, 1, "one pack wait per section with a pack drain");
        assert_eq!(pack_unlocks, 1, "one pack release per section with a pack drain");
        // Acquire precedes the matmul; release follows the store.
        let pos = |op: &Op| ops.iter().position(|o| o == op).unwrap();
        assert!(pos(&Op::TT(TTOp::MathLock)) < pos(&k.at(f).clone()));
        assert!(pos(&Op::TT(TTOp::PackUnlock)) > pos(&k.at(f).clone()));
    }

    /// `tt_sync_cbs` wraps CB traffic with straight-line single-tile
    /// syncs: a publish copy gets reserve/push, a drain copy and a
    /// compute-feeding circular load get wait-front. DRAM↔CB moves are
    /// `Copy` ops; a circular load feeding anything but tile compute
    /// panics.
    #[test]
    fn sync_cbs_wraps_cb_traffic() {
        let mut k = Kernel::from_device_id(Dev::Auto, None);
        let a = k.param(DType::F32);
        let out = k.param_mut(DType::F32);
        let ca = k.storage(DType::F32, MemScope::Circular, 1024);
        let cb = k.storage(DType::F32, MemScope::Circular, 1024);
        let cout = k.storage(DType::F32, MemScope::Circular, 1024);
        let acc = k.storage(DType::F32, MemScope::Register, 1);
        let zero = k.const_val(0i64);
        let pub_a = k.copy_global_to_circular(a, zero, ca, zero);
        k.tt_end_reader();
        let va = k.load_circular(ca, zero);
        let vb = k.load_circular(cb, zero);
        let av = k.load_register_tile(acc, zero);
        let f = k.matmul_tile(va, vb, av);
        k.store_register_tile(acc, f, zero);
        k.tt_end_compute();
        let drain = k.copy_circular_to_global(cout, zero, out, zero);
        k.tt_sync_cbs();

        let mut order: Vec<OpId> = Vec::new();
        let mut op_id = k.head;
        while !op_id.is_null() {
            order.push(op_id);
            op_id = k.next_op(op_id);
        }
        let pos = |id: OpId| order.iter().position(|&o| o == id).unwrap();
        let find = |want: &Op| order.iter().find(|&&id| k.at(id) == want).copied().unwrap();
        let res = find(&Op::TT(TTOp::ReserveBack { cb: ca, n: 1 }));
        let push = find(&Op::TT(TTOp::PushBack { cb: ca, n: 1 }));
        assert_eq!(pos(res) + 1, pos(pub_a), "reserve sits immediately before the publish copy");
        assert_eq!(pos(push), pos(pub_a) + 1, "push sits immediately after the publish copy");
        let w = find(&Op::TT(TTOp::WaitFront { cb: ca, n: 1 }));
        assert_eq!(pos(w) + 1, pos(va), "wait sits immediately before the compute-feeding load");
        let wd = find(&Op::TT(TTOp::WaitFront { cb: cout, n: 1 }));
        assert_eq!(pos(wd) + 1, pos(drain), "wait sits immediately before the drain copy");
    }
    
    /// `tt_storage` rewrites the SSA tile-compute ops to opaque
    /// `LLK` call templates over the bare `Storage`s already in
    /// the IR. The matmul op becomes `matmul_tiles({0}, {1}, 0, 0, {2})`
    /// over its two CB storages and the register slot; the reduce op
    /// becomes `reduce_tile<{op},{dim}>({0}, {1}, 0, 0, {2})` over
    /// input CB, scaler CB, register slot.
    #[test]
    fn storage_lowering_rewrites_matmul_and_reduce() {
        let mut k = Kernel::from_device_id(Dev::Auto, None);
        let cb_a = k.storage(DType::F32, MemScope::Circular, 1024);
        let cb_b = k.storage(DType::F32, MemScope::Circular, 1024);
let acc = k.storage(DType::F32, MemScope::Register, 1);
            let csc = k.storage(DType::F32, MemScope::Circular, 1024);
            let cout = k.storage(DType::F32, MemScope::Circular, 1024);
        let zero = k.const_val(0i64);
        k.tt_end_reader();
        let va = k.load_circular(cb_a, zero);
        let vb = k.load_circular(cb_b, zero);
        let av = k.load_register_tile(acc, zero);
let f = k.matmul_tile(va, vb, av);
            let vs = k.load_circular(csc, zero);
            let r = k.reduce_tile(va, vs, av, BOp::Max, TileDim::Col);
            // Pack drain: the Register slot is read out and stored to a
            // CB. The matmul/reduce results are effect ops after
            // lowering (no SSA value to thread).
            let out = k.load_register_tile(acc, zero);
            k.store_circular(cout, out, zero);
            k.tt_end_compute();
            k.tt_storage();

        let matmul = k.at(f);
        let reduce = k.at(r);
        match matmul {
            Op::TT(TTOp::LLK { asm, ops }) => {
                assert_eq!(asm.as_str(), "matmul_tiles({0}, {1}, 0, 0, {2});");
                assert_eq!(ops.as_slice(), &[cb_a, cb_b, acc, va, vb]);
            }
            other => panic!("matmul not lowered to LLK, got {other:?}"),
        }
        match reduce {
            Op::TT(TTOp::LLK { asm, ops }) => {
                assert_eq!(
                    asm.as_str(),
                    "reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>({0}, {1}, 0, 0, {2});"
                );
                assert_eq!(ops.as_slice(), &[cb_a, csc, acc, va, vs]);
            }
            other => panic!("reduce not lowered to LLK, got {other:?}"),
        }
        // No SSA tile-compute ops survive.
        let mut leftover = 0;
        let mut op_id = k.head;
        while !op_id.is_null() {
            if matches!(
                k.at(op_id),
                Op::TT(TTOp::MatmulTile { .. } | TTOp::ReduceTile { .. })
            ) {
                leftover += 1;
            }
            op_id = k.next_op(op_id);
        }
        assert_eq!(leftover, 0, "no SSA tile-compute ops survive storage lowering");
    }
}
