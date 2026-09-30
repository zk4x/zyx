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
use crate::kernel::{BOp, FusedKind, Kernel, MemScope, Op, OpId, TTOp, TileDim, UOp};

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
        debug_assert!(n != 0, "tt_pop_front: n must be positive");
        self.push_back(Op::TT(TTOp::PopFront { cb, n }))
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

    /// CB sync insertion: wrap every CB traffic op with straight-line
    /// single-tile syncs (`n = 1`; hoisted batches land with batching).
    /// Direction reads off which copy side is circular:
    /// - publish (`dst` GEP over a Circular storage): `ReserveBack`
    ///   immediately before, `PushBack` immediately after;
    /// - drain (`src` GEP over a Circular storage): `WaitFront`
    ///   immediately before, no pop yet;
    /// - CB-to-CB copy: `TileCopy` rule — `WaitFront` on the read side
    ///   immediately before, no pop yet.
    /// A circular `Load` is the compute-consume read (the kernel Load IS
    /// the CB read; codegen bundles read+consume, so its wait sits at the
    /// consumer — here it sits before the Load): `WaitFront` immediately
    /// before iff every user is a tile compute op, loud `panic!`
    /// otherwise. A circular buffer reached outside a GEP is malformed
    /// CB traffic — loud, never silently skipped.
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
                        (Some(scb), Some(_dcb)) => {
                            self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb: scb, n: 1 }));
                        }
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
                            // Dead load: no consumer reads, no wait needed.
                            None => {}
                            Some(use_list) => {
                                for &u in use_list {
                                    if !matches!(
                                        self.ops[u].op,
                                        Op::TT(
                                            TTOp::MatmulTile { .. }
                                                | TTOp::TransposeTile { .. }
                                                | TTOp::ReduceTile { .. }
                                                | TTOp::BroadcastTile { .. }
                                        )
                                    ) {
                                        panic!(
                                            "tt_sync_cbs: circular load {op_id:?} feeds non-compute {u:?}"
                                        );
                                    }
                                }
                                self.insert_before(op_id, Op::TT(TTOp::WaitFront { cb, n: 1 }));
                            }
                        }
                    }
                }
                _ => {}
            }
            op_id = next;
        }

        self.verify();
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
