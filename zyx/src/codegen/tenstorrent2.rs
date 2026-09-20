// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! TTIR — Tenstorrent physical IR (see `TTIR_DESIGN.md`).
//!
//! One program = one ordered `Vec<TTOp>` covering all three RISC-V kernel
//! sources, split at render time at the `EndReader`/`EndCompute`/`EndWriter`
//! markers.

use super::tenstorrent::CBId;
use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel::{BOp, Kernel, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind, TileDim, UOp},
    shape::Dim,
    types::{TinyString, TinyVec},
};
use std::fmt::{Display, Formatter};

/// SFPU LREG slot. Budget: 64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LRegId(pub u8);

impl LRegId {
    /// Hardware budget of SFPU LREG slots.
    pub const BUDGET: usize = 64;
}

/// DST tile slot. Budget: 16 in BF16 mode, 8 in FP32 mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileId(pub u8);

impl TileId {
    /// DST budget in BF16 mode (each FP32 tile occupies 2 BF16 slots).
    pub const BUDGET_BF16: usize = 16;
    /// DST budget in FP32 mode.
    pub const BUDGET_FP32: usize = 8;
}

/// Scalar C register slot (`r{reg}`). Unbounded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VarId(pub u32);

/// One physical instruction. Each variant is one emitted line (or, after
/// decomposition, one instruction) and carries a static signature (CB
/// traffic, DST lock effects, executing thread) used by verify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TTOp {
    // -- Structure: section boundaries and control flow --
    /// End of the reader (NCRISC/BRISC movement) section.
    EndReader,
    /// End of the compute (TRISC) section.
    EndCompute,
    /// End of the writer section. Explicit, never implicit end-of-Vec.
    EndWriter,
    /// Loop with trip count in a scalar register (defined by a prior
    /// `SConst`/`SBinary` statement).
    Loop {
        /// Number of iterations.
        len: VarId,
    },
    /// End of loop body.
    EndLoop,
    /// Branch on a scalar register.
    If {
        /// Condition value.
        cond: VarId,
    },
    /// End of branch body.
    EndIf,
    // -- Scalar C: rendering statements, no passes over these --
    /// Kernel launch argument read: `z = get_arg_val(ordinal)`.
    Arg {
        /// Destination scalar register.
        z: VarId,
        /// Global launch-arg ordinal (survives replicate_ops_per_section).
        ordinal: u32,
    },
    /// Scalar constant definition.
    Const {
        /// Destination scalar register.
        z: VarId,
        /// Literal value.
        value: i64,
    },
    /// Scalar binary: `z = x <bop> y`.
    Binary {
        /// Destination scalar register.
        z: VarId,
        /// Left operand.
        x: VarId,
        /// Right operand.
        y: VarId,
        /// Operation.
        bop: BOp,
    },
    /// Scalar unary: `z = <uop> x`.
    Unary {
        /// Destination scalar register.
        z: VarId,
        /// Source scalar register.
        x: VarId,
        /// Operation.
        uop: UOp,
    },
    /// NOC address computation: `z = get_noc_addr(addr)` (one `uint64_t` line).
    NocAddr {
        /// Destination scalar register.
        z: VarId,
        /// Address value (a scalar register holding prior page/offset math).
        addr: VarId,
    },
    // -- CB traffic: FIFO sync, one op per event --
    /// Declare one hardware circular buffer (single object, never duplicated
    /// across sections).
    CbDeclare {
        /// Circular buffer.
        cb: CBId,
        /// Depth in tiles.
        n_tiles: u32,
    },
    /// `cb.reserve_back(n)`: open n back slots for writing.
    ReserveBack {
        /// Circular buffer.
        cb: CBId,
        /// Slots to reserve.
        n: u32,
    },
    /// `cb.push_back(n)`: publish n written slots.
    PushBack {
        /// Circular buffer.
        cb: CBId,
        /// Slots to publish.
        n: u32,
    },
    /// `cb.wait_front(m)`: block until m front slots hold data.
    WaitFront {
        /// Circular buffer.
        cb: CBId,
        /// Slots to wait for.
        m: u32,
    },
    /// `cb.pop_front(n)`: release n consumed slots.
    PopFront {
        /// Circular buffer.
        cb: CBId,
        /// Slots to release.
        n: u32,
    },
    // -- NOC: real firmware calls, stay C forever --
    /// `noc_async_read(addr, cb.get_write_ptr(), bytes)` + one tile write.
    AsyncRead {
        /// Source NOC address (scalar register holding `get_noc_addr` result).
        addr: VarId,
        /// Destination circular buffer.
        dst_cb: CBId,
        /// Transfer size in bytes.
        bytes: u32,
    },
    /// `noc_async_read_barrier()`.
    NocReadBarrier,
    /// `noc_async_write(cb.get_read_ptr(), addr, bytes)`.
    AsyncWrite {
        /// Source circular buffer.
        src_cb: CBId,
        /// Destination NOC address.
        addr: VarId,
        /// Transfer size in bytes.
        bytes: u32,
    },
    /// `noc_async_write_barrier()`.
    NocWriteBarrier,
    // -- DST state: operand-free lock effects --
    /// `tile_regs_acquire()`.
    MathLock,
    /// `tile_regs_commit()`.
    MathUnlock,
    /// `tile_regs_wait()`.
    PackLock,
    /// `tile_regs_release()`.
    PackUnlock,
    // -- Engine config: init/reconfig calls, one variant per family --
    /// `copy_tile_init(cb);`.
    CopyInit {
        /// Input circular buffer.
        cb: CBId,
    },
    /// `copy_tile_to_dst_init_short_with_dt(prev, cb);` (reprograms
    /// UNPACK+MATH SRCA to `cb`'s format).
    CopyInitWithDt {
        /// Previously programmed CB (reconfig guard source).
        prev: CBId,
        /// Input circular buffer.
        cb: CBId,
    },
    /// `pack_reconfig_data_format(cb);`.
    PackReconfig {
        /// Output circular buffer.
        cb: CBId,
    },
    /// Tile unary init (`exp_tile_init();`, ...).
    UnaryInit {
        /// Operation (selects the init call).
        uop: UOp,
    },
    /// Tile binary init (`add_binary_tile_init();`, ...).
    BinaryInit {
        /// Operation (selects the init call).
        bop: BOp,
    },
    /// `binop_with_scalar_tile_init();` (tile-scalar binary).
    BinScalarInit,
    /// `typecast_tile_init<in, out>();`.
    CastInit {
        /// Source dtype.
        in_dtype: DType,
        /// Target dtype.
        out_dtype: DType,
    },
    /// `transpose_wh_init(cb, out);`.
    TransposeInit {
        /// Input circular buffer.
        cb: CBId,
        /// Output circular buffer (startup triple).
        out: CBId,
    },
    /// `mm_init(a, b, out);` (full init before every matmul; the
    /// hoist/dedup pass later folds these).
    MatmulInit {
        /// Left input circular buffer.
        a: CBId,
        /// Right input circular buffer.
        b: CBId,
        /// Output circular buffer (startup triple).
        out: CBId,
    },
    /// `reduce_init<op, dim>(ci, cs, acc);` (always inline: the acc
    /// slot lives at the op).
    ReduceInit {
        /// Input circular buffer.
        ci: CBId,
        /// Scaler circular buffer.
        cs: CBId,
        /// Accumulator value (render maps it to the DST slot).
        acc: OpId,
        /// Reduction op.
        rop: BOp,
        /// In-tile dimension.
        kind: TileDim,
    },
    /// Fused broadcast binary init (`add_bcast_cols_init_short`, ...).
    BcastInit {
        /// Operation.
        bop: BOp,
        /// Broadcast dimension.
        kind: TileDim,
        /// Full-tile circular buffer.
        cb_a: CBId,
        /// Broadcast-lane circular buffer.
        cb_b: CBId,
    },
    // -- SSA ports: 1:1 kernel-IR lowering, `OpId` operands verbatim.
    //    Only ops Tenstorrent lowers (`Wmma` excluded; `Move`/`Reduce`
    //    never survive linearization so they never reach conversion). --
    /// Constant value.
    SSAConst {
        /// Constant value.
        value: Constant,
        /// Result value (this op's own id).
        z: OpId,
    },
    /// Kernel parameter (launch argument or buffer).
    SSAParam {
        /// Element type.
        dtype: DType,
        /// Scalar-by-value or buffer-by-pointer.
        kind: ParamKind,
        /// Shape operand.
        shape: OpId,
        /// Result value (this op's own id).
        z: OpId,
    },
    /// Value cast.
    SSACast {
        /// Result value (this op's own id).
        z: OpId,
        /// Source value.
        x: OpId,
        /// Target type.
        dtype: DType,
    },
    /// Bitcast (no value conversion, equal bit widths).
    SSABitcast {
        /// Result value (this op's own id).
        z: OpId,
        /// Source value.
        x: OpId,
        /// Target type.
        dtype: DType,
    },
    /// High-level unary (decomposes to SFPU sequences per the opcode table).
    SSAUnary {
        /// Destination value.
        z: OpId,
        /// Source value.
        x: OpId,
        /// Operation.
        uop: UOp,
    },
    /// High-level binary.
    SSABinary {
        /// Destination value.
        z: OpId,
        /// Left operand.
        x: OpId,
        /// Right operand.
        y: OpId,
        /// Operation.
        bop: BOp,
    },
    /// Vector pack.
    SSAStack {
        /// Result value (this op's own id).
        z: OpId,
        /// Element values.
        ops: Box<[OpId]>,
    },
    /// Kernel-internal memory (accumulators, circular buffers, ...).
    SSAStorage {
        /// Result handle (this op's own id).
        z: OpId,
        /// Element type.
        dtype: DType,
        /// Memory scope.
        scope: MemScope,
        /// Length.
        len: Dim,
    },
    /// Indexed store into storage.
    SSAStore {
        /// Result value (this op's own id).
        z: OpId,
        /// Destination storage.
        dst: OpId,
        /// Value to write.
        src: OpId,
        /// Index value.
        index: OpId,
        /// Access layout.
        layout: MemLayout,
    },
    /// Indexed load from storage.
    SSALoad {
        /// Result value (this op's own id).
        z: OpId,
        /// Source storage.
        src: OpId,
        /// Index value.
        index: OpId,
        /// Access layout.
        layout: MemLayout,
    },
    /// Parallel range (group/local/warp).
    SSARange {
        /// Result value (this op's own id).
        z: OpId,
        /// Axis.
        axis: u32,
        /// Range kind.
        kind: RangeKind,
    },
    /// Loop with trip count value.
    SSALoop {
        /// Result value (this op's own id).
        z: OpId,
        /// Number of iterations.
        len: OpId,
    },
    /// End of loop body.
    SSAEndLoop,
    /// Branch on a boolean value.
    SSAIf {
        /// Condition value.
        condition: OpId,
    },
    /// End of branch body.
    SSAEndIf,
    /// Fused multiply-add.
    SSAMad {
        /// Multiplicand.
        x: OpId,
        /// Multiplier.
        y: OpId,
        /// Addend.
        z: OpId,
        /// Result value (this op's own id).
        w: OpId,
    },
    /// Single value out of a vector.
    SSAIndex {
        /// Result value (this op's own id).
        z: OpId,
        /// Source vector.
        vec: OpId,
        /// Element position.
        idx: usize,
    },
    /// Execution barrier.
    SSABarrier,
    /// Hardware tile reduce into accumulator tile.
    SSAReduceTile {
        /// Result value (this op's own id).
        z: OpId,
        /// Input tile.
        x: OpId,
        /// LLK scale tile.
        scaler: OpId,
        /// Accumulator tile.
        acc: OpId,
        /// Reduction op.
        rop: BOp,
        /// In-tile dimension.
        kind: TileDim,
    },
    /// Hardware tile matmul into accumulator tile.
    SSAMatmulTile {
        /// Result value (this op's own id).
        z: OpId,
        /// Left tile.
        x: OpId,
        /// Right tile.
        y: OpId,
        /// Accumulator tile.
        acc: OpId,
    },
    /// Hardware tile transpose.
    SSATransposeTile {
        /// Result value (this op's own id).
        z: OpId,
        /// Input tile.
        x: OpId,
    },
    /// Broadcast marker: tile `x` consumed with broadcast `kind`.
    SSABroadcastTile {
        /// Input tile.
        x: OpId,
        /// In-tile dimension.
        kind: TileDim,
    },
    /// Backend-specific assembly escape hatch.
    SSAAsm {
        /// Result value (this op's own id).
        z: OpId,
        /// Assembly template.
        asm: TinyString,
        /// Operand values.
        ops: TinyVec<OpId>,
    },
}

/// Hoisted init call for a tile unary op. Duplicated from the legacy
/// codegen (`tenstorrent.rs`): one table per IR, never shared.
fn unary_init_name(uop: UOp) -> &'static str {
    match uop {
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
    }
}

/// Hoisted init call for a tile binary op, if it needs one.
fn binary_init_name(bop: BOp) -> Option<&'static str> {
    match bop {
        BOp::Add => Some("add_binary_tile_init();"),
        BOp::Sub => Some("sub_binary_tile_init();"),
        BOp::Mul => Some("mul_binary_tile_init();"),
        BOp::Div => Some("div_binary_tile_init();"),
        BOp::Max => Some("binary_max_tile_init();"),
        BOp::BitShiftLeft | BOp::BitShiftRight => Some("binary_shift_tile_init();"),
        _ => None,
    }
}

/// Hoisted init call for a fused broadcast binary. `None` means the
/// (op, kind) pair has no LLK (notably every `Div`).
fn bcast_init_name(bop: BOp, kind: TileDim) -> Option<&'static str> {
    match (bop, kind) {
        (BOp::Add, TileDim::Row) => Some("add_bcast_rows_init_short"),
        (BOp::Add, TileDim::Col) => Some("add_bcast_cols_init_short"),
        (BOp::Add, TileDim::Scalar) => Some("add_bcast_scalar_init_short"),
        (BOp::Sub, TileDim::Row) => Some("sub_bcast_rows_init_short"),
        (BOp::Sub, TileDim::Col) => Some("sub_bcast_cols_init_short"),
        (BOp::Sub, TileDim::Scalar) => Some("sub_tiles_bcast_scalar_init_short"),
        (BOp::Mul, TileDim::Row) => Some("mul_bcast_rows_init_short"),
        (BOp::Mul, TileDim::Col) => Some("mul_bcast_cols_init_short"),
        (BOp::Mul, TileDim::Scalar) => Some("mul_tiles_bcast_scalar_init_short"),
        _ => None,
    }
}

/// LLK reduce dimension for a tile reduce kind.
fn reduce_dim_name(kind: TileDim) -> &'static str {
    match kind {
        TileDim::Row => "ReduceDim::REDUCE_ROW",
        TileDim::Col => "ReduceDim::REDUCE_COL",
        TileDim::Scalar => "ReduceDim::REDUCE_SCALAR",
    }
}

/// TT `DataFormat` code for a dtype on the tile path (the
/// `typecast_tile_init<in, out>` template args). This is NOT the CB
/// descriptor code (`cb_fmt` below): the two numberings differ.
fn tt_fmt(dtype: DType) -> u32 {
    match dtype {
        DType::F32 => 0,
        DType::F16 | DType::BF16 => 5,
        DType::I32 => 8,
        DType::U16 => 9,
        DType::I8 => 14,
        DType::U32 => 24,
        DType::F8E4M3 => 26,
        DType::U8 => 30,
        dt => panic!("tenstorrent2: dtype {dt:?} has no tt tile format"),
    }
}

/// Runtime CB descriptor format code for a CB storage dtype
/// (F32=0, F16=1, BF16=2, ...). Follows the CB storage dtype;
/// an unmappable dtype is a compilation error, never a default.
fn cb_fmt(dtype: DType) -> u32 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::U16 => 3,
        DType::F8E4M3 => 4,
        DType::U8 => 5,
        DType::I8 => 6,
        DType::U32 => 7,
        DType::I32 => 8,
        dt => panic!("tenstorrent2: CB dtype {dt:?} has no tt format"),
    }
}

struct Compiler {    ops: Vec<TTOp>,
    /// Circular storage (`OpId` of `SSAStorage`) to hardware CB assignment,
    /// in stream order. One hardware object, one `CBId`, never duplicated.
    cbs: Map<OpId, CBId>,
    /// CB depth in tiles, by `CBId` (storage `len / 1024`).
    cb_depth: Map<CBId, u32>,
    /// Data operands per kernel op, recorded at lowering (the sync pass
    /// reasons about dataflow through this table, never over TTIR).
    params: Map<OpId, Vec<OpId>>,
    /// Load `z` to (source storage, tile layout, slot index), recorded at lowering.
    load_info: Map<OpId, (OpId, bool, OpId)>,
    /// Fused-draining load `z` to consuming tile-op `z`s, recorded at lowering.
    drain_consumers: Map<OpId, Vec<OpId>>,
    /// Loop `z` to constant trip count (`None` = symbolic), recorded at lowering.
    trips: Map<OpId, Option<u32>>,
    /// `Const` ops whose value is dim 0, recorded at lowering.
    zeros: Set<OpId>,
}

/// Loop body tracking for sync insertion: direct traffic per CB, branch
/// presence, and the constant trip count. A loop hoists a CB's
/// transaction iff the body holds exactly one traffic op for that CB,
/// no branch, and a constant trip > 1 that fits the CB depth.
struct LoopFrame {
    /// The `SSALoop` op (open anchor; its counter is the block position).
    op: OpId,
    /// Position in `ops` (open anchor for hoisted blocks).
    pos: usize,
    /// Compile-time trip count; `None` = symbolic (never hoisted, and
    /// any CB traffic under it is a compilation error).
    trip: Option<u32>,
    /// A branch sits in the body: hoisting would emit sync ops for a
    /// body that may not run.
    if_seen: bool,
    /// One traffic op per CB directly in the body (deeper loops own
    /// their own traffic): `z` of the traffic op.
    traffic: Map<CBId, OpId>,
    /// CBs with more than one traffic op in the body: no hoist.
    dirty: Set<CBId>,
}

impl Compiler {
    fn new(kernel: &Kernel) -> Self {
        // Pre-pass over the kernel: dataflow facts the sync pass consumes.
        let mut params: Map<OpId, Vec<OpId>> = Map::default();
        let mut load_info: Map<OpId, (OpId, bool, OpId)> = Map::default();
        let mut load_section: Map<OpId, u8> = Map::default();
        let mut trips: Map<OpId, Option<u32>> = Map::default();
        let mut zeros: Set<OpId> = Set::default();
        let mut section = 0u8;
        let mut pre = kernel.head;
        while !pre.is_null() {
            match &kernel.ops[pre].op {
                Op::Barrier => section += 1,
                Op::Const(c) => {
                    if c.as_dim() == Some(0) {
                        zeros.insert(pre);
                    }
                }
                Op::Load { src, index, layout, .. } => {
                    load_info.insert(pre, (*src, matches!(layout, MemLayout::Tile { .. }), *index));
                    load_section.insert(pre, section);
                }
                Op::Loop { len } => {
                    trips.insert(
                        pre,
                        match kernel.resolve_const(*len).and_then(|c| c.as_dim()) {
                            Some(d) if d >= 0 => Some(d as u32),
                            Some(d) => panic!("tenstorrent2: negative loop trip count {d}, op {pre}"),
                            None => None,
                        },
                    );
                }
                _ => {}
            }
            params.insert(pre, kernel.ops[pre].op.parameters().collect());
            pre = kernel.next_op(pre);
        }
        // Fused drains: a compute-section tile load fed only by fused
        // tile ops drains at the consuming ops (legacy 1-tile path),
        // never at the load itself. The load still counts as consumption.
        let mut consumers: Map<OpId, Vec<OpId>> = Map::default();
        for (&op, ps) in params.iter() {
            for &p in ps {
                consumers.entry(p).or_default().push(op);
            }
        }
        let is_bcast_side = |side: OpId| -> bool { matches!(kernel.ops[side].op, Op::BroadcastTile { .. }) };
        let mut drain_consumers: Map<OpId, Vec<OpId>> = Map::default();
        for (&load, &(_, tile, index)) in load_info.iter() {
            if !tile || load_section[&load] != 1 {
                continue;
            }
            let empty = Vec::new();
            let cs = consumers.get(&load).unwrap_or(&empty);
            if cs.is_empty() {
                continue;
            }
            let fused = cs.iter().all(|&c| match &kernel.ops[c].op {
                Op::ReduceTile { .. } | Op::MatmulTile { .. } | Op::TransposeTile { .. } | Op::BroadcastTile { .. } => true,
                Op::Binary { x, y, .. } => is_bcast_side(*x) || is_bcast_side(*y),
                _ => false,
            });
            if !fused {
                continue;
            }
            if !zeros.contains(&index) {
                panic!("tenstorrent2: fused consumer drains at slot 0; load index op {index} must be 0");
            }
            drain_consumers.insert(load, cs.clone());
        }
        let mut ops = Vec::new();
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            let op = match kernel.ops[op_id].op {
                Op::Const(constant) => TTOp::SSAConst { value: constant, z: op_id },
                Op::Param { dtype, kind, shape } => TTOp::SSAParam { dtype, kind, shape, z: op_id },
                Op::Cast { x, dtype } => TTOp::SSACast { z: op_id, x, dtype },
                Op::Bitcast { x, dtype } => TTOp::SSABitcast { z: op_id, x, dtype },
                Op::Unary { x, uop } => TTOp::SSAUnary { z: op_id, x, uop },
                Op::Binary { x, y, bop } => TTOp::SSABinary { z: op_id, x, y, bop },
                Op::Stack { ref ops } => TTOp::SSAStack { z: op_id, ops: ops.clone() },
                Op::Storage { dtype, scope, len } => TTOp::SSAStorage { z: op_id, dtype, scope, len },
                Op::Store { dst, src, index, layout } => TTOp::SSAStore { z: op_id, dst, src, index, layout },
                Op::Load { src, index, layout } => TTOp::SSALoad { z: op_id, src, index, layout },
                Op::Range { axis, kind } => TTOp::SSARange { z: op_id, axis, kind },
                Op::Loop { len } => TTOp::SSALoop { z: op_id, len },
                Op::EndLoop => TTOp::SSAEndLoop,
                Op::If { condition } => TTOp::SSAIf { condition },
                Op::EndIf => TTOp::SSAEndIf,
                Op::Mad { x, y, z } => TTOp::SSAMad { x, y, z, w: op_id },
                Op::Index { vec, idx } => TTOp::SSAIndex { z: op_id, vec, idx },
                Op::Barrier => TTOp::SSABarrier,
                Op::Wmma { .. } => unreachable!("tenstorrent2: Wmma has no Tenstorrent lowering"),
                Op::ReduceTile { x, scaler, acc, rop, kind } => TTOp::SSAReduceTile { z: op_id, x, scaler, acc, rop, kind },
                Op::MatmulTile { x, y, acc } => TTOp::SSAMatmulTile { z: op_id, x, y, acc },
                Op::TransposeTile { x } => TTOp::SSATransposeTile { z: op_id, x },
                Op::BroadcastTile { x, kind } => TTOp::SSABroadcastTile { x, kind },
                Op::Asm { ref asm, ref ops } => TTOp::SSAAsm { z: op_id, asm: asm.clone(), ops: ops.clone() },
                Op::Move { .. } => unreachable!("tenstorrent2: Move never survives linearization"),
                Op::Reduce { .. } => unreachable!("tenstorrent2: Reduce never survives linearization"),
            };
            ops.push(op);
            op_id = kernel.next_op(op_id);
        }
        Self { ops, cbs: Map::default(), cb_depth: Map::default(), params, load_info, drain_consumers, trips, zeros }
    }

    /// Replicate_ops_per_section: values consumed in multiple sections are
    /// duplicated per section (each section is a separate kernel with its own
    /// registers and runtime args; arg ordinals stay global so the launch
    /// contract survives). Circular-buffer storages are the exception: one
    /// hardware object, one `CBId`, never duplicated — they are hoisted once
    /// to the stream head. Mirrors the legacy `get_needed_ops` closure per
    /// section (stores + structural starters, transitive data deps), then
    /// re-emits each section's needed ops in stream order.
    fn replicate_ops_per_section(&mut self) {
        use MemScope::Circular;
        // Section index per position (0 reader, 1 compute, 2 writer).
        let mut section_of: Vec<u8> = vec![0; self.ops.len()];
        let mut section = 0u8;
        for (pos, op) in self.ops.iter().enumerate() {
            section_of[pos] = section;
            match op {
                TTOp::EndReader | TTOp::EndCompute => section += 1,
                _ => {}
            }
        }
        debug_assert!(section == 2, "tenstorrent2: expected two section markers, found {section}");
        // Data operands per position (closure edges), and whether the op is
        // a circular storage (global, never replicated).
        fn operands(op: &TTOp) -> Vec<OpId> {
            match op {
                TTOp::SSAConst { .. }
                | TTOp::SSAStorage { .. }
                | TTOp::SSALoop { .. }
                | TTOp::SSARange { .. }
                | TTOp::SSAEndLoop
                | TTOp::SSAEndIf
                | TTOp::SSABarrier
                | TTOp::EndReader
                | TTOp::EndCompute
                | TTOp::EndWriter => Vec::new(),
                TTOp::SSAParam { shape, .. } => vec![*shape],
                TTOp::SSACast { x, .. }
                | TTOp::SSABitcast { x, .. }
                | TTOp::SSAUnary { x, .. }
                | TTOp::SSATransposeTile { x, .. }
                | TTOp::SSABroadcastTile { x, .. } => vec![*x],
                TTOp::SSABinary { x, y, .. } => vec![*x, *y],
                TTOp::SSAStack { ops, .. } => ops.iter().copied().collect(),
                TTOp::SSAStore { dst, src, index, .. } => vec![*dst, *src, *index],
                TTOp::SSALoad { src, index, .. } => vec![*src, *index],
                TTOp::SSAIf { condition } => vec![*condition],
                TTOp::SSAMad { x, y, z, .. } => vec![*x, *y, *z],
                TTOp::SSAIndex { vec, .. } => vec![*vec],
                TTOp::SSAReduceTile { x, scaler, acc, .. } => vec![*x, *scaler, *acc],
                TTOp::SSAMatmulTile { x, y, acc, .. } => vec![*x, *y, *acc],
                TTOp::SSAAsm { ops, .. } => ops.iter().copied().collect(),
                _ => Vec::new(),
            }
        }
        // Range/loop length starters need their own edge (walk references them).
        fn starters(op: &TTOp) -> Vec<OpId> {
            match op {
                TTOp::SSARange { kind, .. } => match kind {
                    RangeKind::Group(len) | RangeKind::Warp(len) => vec![*len],
                    RangeKind::Local(_) => Vec::new(),
                },
                TTOp::SSALoop { len, .. } => vec![*len],
                TTOp::SSAIf { condition } => vec![*condition],
                TTOp::SSAAsm { ops, .. } => ops.iter().copied().collect(),
                _ => Vec::new(),
            }
        }
        fn defines(op: &TTOp) -> Option<OpId> {
            match op {
                TTOp::SSAMad { w, .. } => Some(*w),
                TTOp::SSAConst { z, .. }
                | TTOp::SSAParam { z, .. }
                | TTOp::SSACast { z, .. }
                | TTOp::SSABitcast { z, .. }
                | TTOp::SSAUnary { z, .. }
                | TTOp::SSABinary { z, .. }
                | TTOp::SSAStack { z, .. }
                | TTOp::SSAStorage { z, .. }
                | TTOp::SSAStore { z, .. }
                | TTOp::SSALoad { z, .. }
                | TTOp::SSARange { z, .. }
                | TTOp::SSALoop { z, .. }
                | TTOp::SSAIndex { z, .. }
                | TTOp::SSAReduceTile { z, .. }
                | TTOp::SSAMatmulTile { z, .. }
                | TTOp::SSATransposeTile { z, .. }
                | TTOp::SSAAsm { z, .. } => Some(*z),
                TTOp::SSABroadcastTile { .. } => None,
                _ => None,
            }
        }
        let is_circular_storage = |op: &TTOp| -> bool {
            matches!(op, TTOp::SSAStorage { scope: Circular, .. })
        };
        // Loops/branches must not span sections: collect loop open sections.
        {
            let mut loop_sections: Map<OpId, u8> = Map::default();
            for (pos, op) in self.ops.iter().enumerate() {
                match op {
                    TTOp::SSALoop { z, .. } => {
                        loop_sections.insert(*z, section_of[pos]);
                    }
                    TTOp::SSAEndLoop => {}
                    _ => {}
                }
            }
            let _ = loop_sections;
        }
        // Per-section needed sets.
        let mut needed: Vec<Set<OpId>> = vec![Set::default(), Set::default(), Set::default()];
        let mut structural: Vec<Set<OpId>> = vec![Set::default(), Set::default(), Set::default()];
        for target in 0..3u8 {
            let mut stack: Vec<OpId> = Vec::new();
            for (pos, op) in self.ops.iter().enumerate() {
                if section_of[pos] != target {
                    continue;
                }
                match op {
                    TTOp::SSAStore { .. } => {
                        if let Some(z) = defines(op) {
                            needed[target as usize].insert(z);
                        }
                        stack.extend(operands(op));
                    }
                    TTOp::SSALoop { .. }
                    | TTOp::SSARange { .. }
                    | TTOp::SSAIf { .. }
                    | TTOp::SSAEndLoop
                    | TTOp::SSAEndIf
                    | TTOp::SSAAsm { .. } => {
                        if let Some(z) = defines(op) {
                            structural[target as usize].insert(z);
                        } else {
                            structural[target as usize].insert(OpId::NULL);
                        }
                        stack.extend(starters(op));
                    }
                    _ => {}
                }
            }
            while let Some(id) = stack.pop() {
                if id.is_null() || needed[target as usize].contains(&id) {
                    continue;
                }
                // Circular storages stay global; consumers reference the
                // single head copy without pulling them into the section.
                if let Some(pos) = self.ops.iter().position(|o| defines(o) == Some(id)) {
                    if is_circular_storage(&self.ops[pos]) {
                        continue;
                    }
                }
                needed[target as usize].insert(id);
                let def_pos = self.ops.iter().position(|o| defines(o) == Some(id));
                match def_pos {
                    Some(pos) => stack.extend(operands(&self.ops[pos])),
                    None => panic!("tenstorrent2: replicate_ops_per_section: no def for op {id}"),
                }
            }
        }
        // Rebuild: circular storages once at head, then each section's
        // needed + structural ops in stream order with section markers.
        let mut out: Vec<TTOp> = Vec::new();
        for op in &self.ops {
            if is_circular_storage(op) {
                out.push(op.clone());
            }
        }
        for target in 0..3u8 {
            for (pos, op) in self.ops.iter().enumerate() {
                if is_circular_storage(op) {
                    continue;
                }
                match op {
                    TTOp::EndReader | TTOp::EndCompute | TTOp::EndWriter => {}
                    _ => {
                        let keep = match defines(op) {
                            // Value defs replicate: emit wherever consumed,
                            // regardless of lexical section.
                            Some(z) => {
                                needed[target as usize].contains(&z)
                                    || structural[target as usize].contains(&z)
                            }
                            // Id-less ops stay lexical (structure, broadcast
                            // markers, closers).
                            None => {
                                section_of[pos] == target
                                    && (structural[target as usize].contains(&OpId::NULL)
                                        || matches!(
                                            op,
                                            TTOp::SSAEndLoop
                                                | TTOp::SSAEndIf
                                                | TTOp::SSABroadcastTile { .. }
                                        ))
                            }
                        };
                        if keep {
                            out.push(op.clone());
                        }
                    }
                }
            }
            out.push(match target {
                0 => TTOp::EndReader,
                1 => TTOp::EndCompute,
                _ => TTOp::EndWriter,
            });
        }
        self.ops = out;
    }

    /// Sectioning: the first barrier becomes `EndReader`, the second
    /// `EndCompute`, `EndWriter` closes the stream. A third barrier is a
    /// malformed kernel — there are only three sections.
    fn sectioning(&mut self) {
        let mut barriers = 0;
        for op in &mut self.ops {
            if matches!(op, TTOp::SSABarrier) {
                barriers += 1;
                *op = match barriers {
                    1 => TTOp::EndReader,
                    2 => TTOp::EndCompute,
                    _ => panic!("tenstorrent2: kernel has more than two barriers"),
                };
            }
        }
        self.ops.push(TTOp::EndWriter);
    }

    /// Assign CBs: number each circular storage in stream order, recording
    /// storage `OpId` to hardware `CBId`. Non-circular storages are untouched.
    fn assign_cbs(&mut self) {
        for op in &self.ops {
            if let TTOp::SSAStorage { z, scope, len, .. } = op
                && matches!(scope, MemScope::Circular)
                && !self.cbs.contains_key(z)
            {
                let cb = CBId(self.cbs.len() as u32);
                self.cbs.insert(*z, cb);
                self.cb_depth.insert(cb, (*len / 1024) as u32);
            }
        }
    }

    /// Counter check: the operand closure of `root` contains the loop
    /// counter `target` (block slots must tie to the counter or be 0).
    fn counter_tied(&self, root: OpId, target: OpId) -> bool {
        if root.is_null() {
            return false;
        }
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if id == target {
                return true;
            }
            if let Some(ps) = self.params.get(&id) {
                stack.extend(ps.iter().copied());
            }
        }
        false
    }

    /// Sync insertion: `ReserveBack`/`WaitFront`/`PushBack`/`PopFront`
    /// around every CB traffic op. Single-tile transactions wrap the op
    /// inline; a loop body holding exactly one traffic op for a CB, no
    /// branch, and a constant trip > 1 fitting the CB depth upgrades to
    /// a hoisted block (open before the loop, close after its end).
    /// Fused-draining loads carry no syncs; their consumers drain.
    /// `CbDeclare` ops head the stream in `CBId` order.
    fn sync_cbs(&mut self) {
        // Fused drains resolve to consumer wraps (CB known only now).
        let mut drain_at: Map<OpId, Vec<CBId>> = Map::default();
        for (&load, cs) in self.drain_consumers.iter() {
            let (storage, _, _) = self.load_info[&load];
            let cb = self.cbs[&storage];
            for &c in cs {
                drain_at.entry(c).or_default().push(cb);
            }
        }
        // Transaction per traffic op: (cb, tiles per pass, hoist op,
        // produce?). Provisional single-tile; EndLoop may upgrade.
        struct Batch {
            cb: CBId,
            per_op: u32,
            hoist: OpId,
            produce: bool,
        }
        let mut batches: Map<OpId, Batch> = Map::default();
        let mut produced: Map<CBId, u32> = Map::default();
        let mut consumed: Map<CBId, u32> = Map::default();
        for &cb in self.cbs.values() {
            produced.insert(cb, 0);
            consumed.insert(cb, 0);
        }
        let mut section = 0u8;
        let mut stack: Vec<LoopFrame> = Vec::new();
        let mut loop_end: Map<OpId, usize> = Map::default();
        // Stream position of every `z`-carrying op.
        let mut pos_of: Map<OpId, usize> = Map::default();
        for (pos, op) in self.ops.iter().enumerate() {
            match op {
                TTOp::SSAStore { z, .. }
                | TTOp::SSALoad { z, .. }
                | TTOp::SSALoop { z, .. }
                | TTOp::SSAUnary { z, .. }
                | TTOp::SSABinary { z, .. }
                | TTOp::SSAMad { w: z, .. }
                | TTOp::SSAIndex { z, .. }
                | TTOp::SSAReduceTile { z, .. }
                | TTOp::SSAMatmulTile { z, .. }
                | TTOp::SSATransposeTile { z, .. } => {
                    pos_of.insert(*z, pos);
                }
                _ => {}
            }
        }
        for (pos, op) in self.ops.iter().enumerate() {
            match op {
                TTOp::EndReader | TTOp::EndCompute => section += 1,
                TTOp::SSALoop { z, .. } => {
                    let trip = self.trips[z];
                    stack.push(LoopFrame { op: *z, pos, trip, if_seen: false, traffic: Map::default(), dirty: Set::default() });
                }
                TTOp::SSAIf { .. } => {
                    if let Some(f) = stack.last_mut() {
                        f.if_seen = true;
                    }
                }
                TTOp::SSAEndLoop => {
                    let f = stack.pop().expect("tenstorrent2: EndLoop without Loop");
                    loop_end.insert(f.op, pos);
                    for (&cb, &op) in f.traffic.iter() {
                        if f.dirty.contains(&cb) {
                            continue;
                        }
                        let Some(t) = f.trip else { continue };
                        if t <= 1 {
                            continue;
                        }
                        if t > self.cb_depth[&cb] {
                            // The CB cannot hold the whole block:
                            // streaming. Keep the per-iteration
                            // single-tile transaction, always correct.
                            continue;
                        }
                        if !batches.contains_key(&op) {
                            // Fused-only loads count for balance but
                            // carry no batch to upgrade.
                            continue;
                        }
                        let index = match &self.ops[pos_of[&op]] {
                            TTOp::SSALoad { index, .. } | TTOp::SSAStore { index, .. } => *index,
                            _ => panic!("tenstorrent2: hoisted op {op} is not traffic"),
                        };
                        if !self.counter_tied(index, f.op) && !self.zeros.contains(&index) {
                            panic!("tenstorrent2: CB{cb} index op {index} is neither the hoist loop counter nor 0");
                        }
                        if let Some(b) = batches.get_mut(&op) {
                            b.per_op = t;
                            b.hoist = f.op;
                        }
                    }
                }
                TTOp::SSAStore { z, dst, src, index, .. } => {
                    let traffic = match section {
                        0 | 1 => self.cbs.get(dst).map(|&cb| (true, cb, *index)),
                        _ => match self.load_info.get(src) {
                            Some(&(storage, tile, load_index)) if tile => {
                                self.cbs.get(&storage).map(|&cb| (false, cb, load_index))
                            }
                            _ => None,
                        },
                    };
                    if let Some((produce, cb, index)) = traffic {
                        // Per-pass tile count: product of enclosing trips.
                        // Symbolic trips cannot bound CB traffic.
                        let mut count = 1u32;
                        for fr in stack.iter() {
                            match fr.trip {
                                Some(t) => count *= t,
                                None => panic!("tenstorrent2: CB{cb} traffic inside a non-constant loop"),
                            }
                        }
                        if produce {
                            *produced.get_mut(&cb).expect("tenstorrent2: CB totals pre-recorded") += count;
                        } else {
                            *consumed.get_mut(&cb).expect("tenstorrent2: CB totals pre-recorded") += count;
                        }
                        if let Some(f) = stack.last_mut() {
                            if f.traffic.insert(cb, *z).is_some() {
                                f.dirty.insert(cb);
                            }
                        }
                        let counter_tied = stack.iter().any(|fr| self.counter_tied(index, fr.op));
                        if !self.zeros.contains(&index) && !counter_tied {
                            panic!("tenstorrent2: single-tile transaction on CB{cb} needs slot index 0 or the loop counter");
                        }
                        batches.insert(*z, Batch { cb, per_op: 1, hoist: *z, produce });
                    }
                }
                TTOp::SSALoad { z, src, index, .. } => {
                    if section != 1 {
                        continue;
                    }
                    let Some(&cb) = self.cbs.get(src) else { continue };
                    if !self.load_info.get(z).is_some_and(|&(_, tile, _)| tile) {
                        continue;
                    }
                    let mut count = 1u32;
                    for fr in stack.iter() {
                        match fr.trip {
                            Some(t) => count *= t,
                            None => panic!("tenstorrent2: CB{cb} traffic inside a non-constant loop"),
                        }
                    }
                    *consumed.get_mut(&cb).expect("tenstorrent2: CB totals pre-recorded") += count;
                    if let Some(f) = stack.last_mut() {
                        if f.traffic.insert(cb, *z).is_some() {
                            f.dirty.insert(cb);
                        }
                    }
                    if self.drain_consumers.contains_key(z) {
                        continue;
                    }
                    let counter_tied = stack.iter().any(|fr| self.counter_tied(*index, fr.op));
                    if !self.zeros.contains(index) && !counter_tied {
                        panic!("tenstorrent2: single-tile transaction on CB{cb} needs slot index 0 or the loop counter");
                    }
                    batches.insert(*z, Batch { cb, per_op: 1, hoist: *z, produce: false });
                }
                _ => {}
            }
        }
        // Anchors: inline transactions wrap the traffic op; hoisted
        // blocks open before the loop and close after its end.
        // (`pos_of` was built before the walk.)
        let mut before: Map<usize, Vec<TTOp>> = Map::default();
        let mut after: Map<usize, Vec<TTOp>> = Map::default();
        let mut open_close: Vec<(usize, usize, CBId, u32, bool)> = Vec::new();
        for (&op, b) in batches.iter() {
            if b.hoist == op {
                let pos = pos_of[&op];
                open_close.push((pos, pos, b.cb, b.per_op, b.produce));
            } else {
                let open = pos_of[&b.hoist];
                let close = loop_end[&b.hoist] + 1;
                open_close.push((open, close, b.cb, b.per_op, b.produce));
            }
        }
        open_close.sort();
        for (open, close, cb, n, produce) in open_close {
            let (first, second) = if produce {
                (TTOp::ReserveBack { cb, n }, TTOp::PushBack { cb, n })
            } else {
                (TTOp::WaitFront { cb, m: n }, TTOp::PopFront { cb, n })
            };
            before.entry(open).or_default().push(first);
            after.entry(close).or_default().push(second);
        }
        // Fused consumers drain inline (legacy 1-tile path).
        for (&c, cbs) in drain_at.iter() {
            let pos = pos_of[&c];
            for &cb in cbs {
                before.entry(pos).or_default().push(TTOp::WaitFront { cb, m: 1 });
                after.entry(pos).or_default().push(TTOp::PopFront { cb, n: 1 });
            }
        }
        // Cross-section balance: every produced tile per pass is
        // consumed per pass, per CB.
        for (&cb, &p) in produced.iter() {
            let c = consumed[&cb];
            if p != c {
                panic!("tenstorrent2: CB{cb} produces {p} but consumes {c} tiles per pass");
            }
        }
        // Rebuild with sync ops spliced in; CB declarations head the stream.
        let mut cbs: Vec<CBId> = self.cbs.values().copied().collect();
        cbs.sort();
        let mut next = Vec::with_capacity(self.ops.len());
        for cb in cbs {
            next.push(TTOp::CbDeclare { cb, n_tiles: self.cb_depth[&cb] });
        }
        let old = std::mem::take(&mut self.ops);
        for (pos, op) in old.into_iter().enumerate() {
            if let Some(v) = before.remove(&pos) {
                next.extend(v);
            }
            next.push(op);
            if let Some(v) = after.remove(&pos) {
                next.extend(v);
            }
        }
        self.ops = next;
    }

    /// DST lock insertion: `MathLock`/`MathUnlock`/`PackLock`/`PackUnlock`
    /// around compute-section tile traffic. Faithful port of the legacy
    /// `TileEmitter` state machine (`tenstorrent.rs`): lazy MATH acquire
    /// (first take acquires, later takes in the same cone keep), deferred
    /// PACK release (consecutive packs share one cone; the release flushes
    /// at the next MATH take or at section/loop end). MATH ops are tile
    /// compute ops; PACK ops are tile stores draining to a CB. Scalar and
    /// movement sections carry no locks.
    fn lock_dst(&mut self) {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum DstState {
            Unlocked,
            MathLock,
            PackLock,
        }
        // Register-acc storages: loads/stores threading them are alias
        // moves, not MATH/PACK traffic.
        let mut register: Set<OpId> = Set::default();
        for op in &self.ops {
            if let TTOp::SSAStorage { z, scope, .. } = op
                && matches!(scope, MemScope::Register)
            {
                register.insert(*z);
            }
        }
        let is_math = |op: &TTOp, drain_consumers: &Map<OpId, Vec<OpId>>| -> bool {
            match op {
                TTOp::SSAStorage { scope, .. } => matches!(scope, MemScope::Register),
                TTOp::SSALoad { z, src, layout, .. } => {
                    if !matches!(layout, MemLayout::Tile { .. }) {
                        return false;
                    }
                    if register.contains(src) {
                        return false;
                    }
                    if drain_consumers.contains_key(z) {
                        return false;
                    }
                    true
                }
                TTOp::SSAUnary { .. }
                | TTOp::SSABinary { .. }
                | TTOp::SSACast { .. }
                | TTOp::SSAMad { .. }
                | TTOp::SSAReduceTile { .. }
                | TTOp::SSAMatmulTile { .. }
                | TTOp::SSATransposeTile { .. } => true,
                _ => false,
            }
        };
        let old = std::mem::take(&mut self.ops);
        let mut next = Vec::with_capacity(old.len());
        let mut state = DstState::Unlocked;
        let mut section = 0u8;
        let math_lock = |next: &mut Vec<TTOp>, state: &mut DstState| match *state {
            DstState::MathLock => {}
            DstState::PackLock => {
                next.push(TTOp::PackUnlock);
                next.push(TTOp::MathLock);
                *state = DstState::MathLock;
            }
            DstState::Unlocked => {
                next.push(TTOp::MathLock);
                *state = DstState::MathLock;
            }
        };
        for op in old {
            match &op {
                TTOp::EndReader | TTOp::EndCompute => {
                    if section == 1 && state == DstState::PackLock {
                        next.push(TTOp::PackUnlock);
                        state = DstState::Unlocked;
                    }
                    section += 1;
                    next.push(op);
                    continue;
                }
                TTOp::EndWriter => {
                    if section == 1 && state == DstState::PackLock {
                        next.push(TTOp::PackUnlock);
                        state = DstState::Unlocked;
                    }
                    next.push(op);
                    continue;
                }
                TTOp::SSAEndLoop => {
                    // No open cone across the back-edge: the body runs N
                    // times, so a PACK-held file here would deadlock the
                    // next iteration on an acquire past the loop.
                    if section == 1 && state == DstState::PackLock {
                        next.push(TTOp::PackUnlock);
                        state = DstState::Unlocked;
                    }
                    next.push(op);
                    continue;
                }
                _ => {}
            }
            if section != 1 {
                next.push(op);
                continue;
            }
            // Pack path: tile store draining to a CB. Acc-threading
            // stores (dst is a Register acc) alias tiles, no lock.
            if let TTOp::SSAStore { dst, .. } = &op {
                if self.cbs.contains_key(dst) {
                    match state {
                        DstState::MathLock => {
                            next.push(TTOp::MathUnlock);
                            next.push(TTOp::PackLock);
                            state = DstState::PackLock;
                        }
                        DstState::PackLock => {}
                        DstState::Unlocked => {
                            panic!("tenstorrent2: pack with DST Unlocked, no live cone (pack of a dead slot)");
                        }
                    }
                    next.push(op);
                    continue;
                }
                next.push(op);
                continue;
            }
            if is_math(&op, &self.drain_consumers) {
                math_lock(&mut next, &mut state);
            }
            next.push(op);
        }
        self.ops = next;
    }

    /// MATH/config init insertion (pass 1 of 2): a full init before
    /// every compute-section tile compute op. Naive: no hoisting, no
    /// dedup (the hoist+dedup pass folds these later); redundancy is
    /// safe. Copy inits track the unpack-A source like the legacy
    /// emitter (`with_dt` form on format change, plain short
    /// otherwise); matmul notes it the same way. Anything the naive
    /// pass cannot resolve fails loudly at the exact op.
    fn init_math(&mut self) {
        // Storage dtypes, register accs, consts, and every defined id.
        let mut storage_dtype: Map<OpId, DType> = Map::default();
        let mut register: Set<OpId> = Set::default();
        let mut const_ids: Set<OpId> = Set::default();
        let mut defined: Set<OpId> = Set::default();
        for op in &self.ops {
            match op {
                TTOp::SSAStorage { z, dtype, scope, .. } => {
                    storage_dtype.insert(*z, *dtype);
                    if matches!(scope, MemScope::Register) {
                        register.insert(*z);
                    }
                    defined.insert(*z);
                }
                TTOp::SSAConst { z, .. }
                | TTOp::SSAParam { z, .. }
                | TTOp::SSACast { z, .. }
                | TTOp::SSABitcast { z, .. }
                | TTOp::SSAUnary { z, .. }
                | TTOp::SSABinary { z, .. }
                | TTOp::SSAStack { z, .. }
                | TTOp::SSALoad { z, .. }
                | TTOp::SSALoop { z, .. }
                | TTOp::SSAMad { w: z, .. }
                | TTOp::SSAIndex { z, .. }
                | TTOp::SSAReduceTile { z, .. }
                | TTOp::SSAMatmulTile { z, .. }
                | TTOp::SSATransposeTile { z, .. }
                | TTOp::SSAAsm { z, .. } => {
                    defined.insert(*z);
                }
                _ => {}
            }
            if let TTOp::SSAConst { z, .. } = op {
                const_ids.insert(*z);
            }
        }
        // CB descriptor format per CBId, from the storage dtype.
        let mut cb_format: Map<CBId, u32> = Map::default();
        for (&storage, &cb) in self.cbs.iter() {
            let dtype = storage_dtype.get(&storage).expect("tenstorrent2: CB storage without dtype");
            cb_format.insert(cb, cb_fmt(*dtype));
        }
        // Broadcast marker kinds by marked value: SSABroadcastTile
        // carries no own id, so consumers find the kind through the
        // marker's operand (recorded in params at lowering).
        let mut bcast_kinds: Map<OpId, Vec<TileDim>> = Map::default();
        for op in &self.ops {
            if let TTOp::SSABroadcastTile { x, kind } = op {
                bcast_kinds.entry(*x).or_default().push(*kind);
            }
        }
        // First pack destination in compute: the startup-triple output
        // for `mm_init`/`transpose_wh_init` (single-output kernels;
        // pass 2's unconditional pack reconfigs make the choice
        // non-load-bearing for the packer side).
        let mut first_pack_cb: Option<CBId> = None;
        {
            let mut section = 0u8;
            for op in &self.ops {
                match op {
                    TTOp::EndReader | TTOp::EndCompute => section += 1,
                    TTOp::SSAStore { dst, .. } if section == 1 => {
                        if let Some(&cb) = self.cbs.get(dst) {
                            first_pack_cb = Some(cb);
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
        // Load value to its CB (tile loads from Circular storage only;
        // acc loads and fused-draining loads resolve to none).
        let load_cb = |v: OpId,
                       load_info: &Map<OpId, (OpId, bool, OpId)>,
                       cbs: &Map<OpId, CBId>,
                       drain_consumers: &Map<OpId, Vec<OpId>>|
         -> Option<CBId> {
            let (storage, tile, _) = load_info.get(&v)?;
            if !tile || drain_consumers.contains_key(&v) {
                return None;
            }
            cbs.get(storage).copied()
        };
        // Broadcast marker kind for a binary operand: the operand is a
        // marker op id (in params, defined nowhere) whose recorded
        // operand names a broadcast-marked value.
        let bcast_of = |o: OpId,
                        params: &Map<OpId, Vec<OpId>>,
                        defined: &Set<OpId>,
                        bcast_kinds: &Map<OpId, Vec<TileDim>>|
         -> Option<TileDim> {
            if defined.contains(&o) {
                return None;
            }
            let mx = params.get(&o).and_then(|ps| match ps.as_slice() {
                [single] => Some(*single),
                _ => None,
            })?;
            match bcast_kinds.get(&mx).map(Vec::as_slice) {
                Some([kind]) => Some(*kind),
                Some(_) => panic!("tenstorrent2: broadcast marker value r{mx} has ambiguous kinds"),
                None => None,
            }
        };
        let old = std::mem::take(&mut self.ops);
        let mut next = Vec::with_capacity(old.len());
        let mut section = 0u8;
        let mut unpack_src: Option<(CBId, u32)> = None;
        for op in old {
            match &op {
                TTOp::EndReader | TTOp::EndCompute => {
                    section += 1;
                    next.push(op);
                    continue;
                }
                TTOp::EndWriter => {
                    next.push(op);
                    continue;
                }
                _ => {}
            }
            if section != 1 {
                next.push(op);
                continue;
            }
            match op {
                TTOp::SSALoad { z, src, index, layout } => {
                    let tile = matches!(layout, MemLayout::Tile { .. });
                    if tile && !register.contains(&src) && !self.drain_consumers.contains_key(&z) {
                        let cb = self.cbs.get(&src).expect("tenstorrent2: tile load from unmapped CB storage");
                        let fmt = cb_format[cb];
                        if unpack_src.map(|(_, f)| f) != Some(fmt) {
                            let prev = unpack_src.map(|(c, _)| c).unwrap_or(*cb);
                            next.push(TTOp::CopyInitWithDt { prev, cb: *cb });
                            unpack_src = Some((*cb, fmt));
                        } else {
                            next.push(TTOp::CopyInit { cb: *cb });
                        }
                    }
                    next.push(TTOp::SSALoad { z, src, index, layout });
                }
                TTOp::SSAUnary { z, x, uop } => {
                    next.push(TTOp::UnaryInit { uop });
                    next.push(TTOp::SSAUnary { z, x, uop });
                }
                TTOp::SSABinary { z, x, y, bop } => {
                    let bx = bcast_of(x, &self.params, &defined, &bcast_kinds);
                    let by = bcast_of(y, &self.params, &defined, &bcast_kinds);
                    match (bx, by) {
                        (Some(kind), None) => {
                            let cb_b = load_cb(x, &self.load_info, &self.cbs, &self.drain_consumers)
                                .expect("tenstorrent2: broadcast lane is no CB tile load");
                            let cb_a = load_cb(y, &self.load_info, &self.cbs, &self.drain_consumers)
                                .expect("tenstorrent2: broadcast op reads a non-CB tile");
                            if bcast_init_name(bop, kind).is_none() {
                                panic!("tenstorrent2: broadcast ({bop:?}, {kind:?}) has no init call");
                            }
                            next.push(TTOp::BcastInit { bop, kind, cb_a, cb_b });
                            unpack_src = Some((cb_a, cb_format[&cb_a]));
                        }
                        (None, Some(kind)) => {
                            let cb_b = load_cb(y, &self.load_info, &self.cbs, &self.drain_consumers)
                                .expect("tenstorrent2: broadcast lane is no CB tile load");
                            let cb_a = load_cb(x, &self.load_info, &self.cbs, &self.drain_consumers)
                                .expect("tenstorrent2: broadcast op reads a non-CB tile");
                            if bcast_init_name(bop, kind).is_none() {
                                panic!("tenstorrent2: broadcast ({bop:?}, {kind:?}) has no init call");
                            }
                            next.push(TTOp::BcastInit { bop, kind, cb_a, cb_b });
                            unpack_src = Some((cb_a, cb_format[&cb_a]));
                        }
                        (Some(_), Some(_)) => {
                            panic!("tenstorrent2: broadcast op r{z} marks both sides");
                        }
                        (None, None) => {
                            if const_ids.contains(&x) || const_ids.contains(&y) {
                                next.push(TTOp::BinScalarInit);
                            } else if binary_init_name(bop).is_some() {
                                next.push(TTOp::BinaryInit { bop });
                            }
                        }
                    }
                    next.push(TTOp::SSABinary { z, x, y, bop });
                }
                TTOp::SSACast { z, x, dtype } => {
                    let in_dtype = match self.load_info.get(&x) {
                        Some((storage, _, _)) => storage_dtype
                            .get(storage)
                            .copied()
                            .expect("tenstorrent2: cast source storage without dtype"),
                        None => panic!("tenstorrent2: cast r{z} reads a non-load value r{x}"),
                    };
                    next.push(TTOp::CastInit { in_dtype, out_dtype: dtype });
                    next.push(TTOp::SSACast { z, x, dtype });
                }
                TTOp::SSAMatmulTile { z, x, y, acc } => {
                    let a = load_cb(x, &self.load_info, &self.cbs, &self.drain_consumers)
                        .expect("tenstorrent2: matmul reads a non-CB tile");
                    let b = load_cb(y, &self.load_info, &self.cbs, &self.drain_consumers)
                        .expect("tenstorrent2: matmul reads a non-CB tile");
                    let out = first_pack_cb
                        .expect("tenstorrent2: matmul needs a startup output CB but compute packs nothing");
                    next.push(TTOp::MatmulInit { a, b, out });
                    unpack_src = Some((a, cb_format[&a]));
                    next.push(TTOp::SSAMatmulTile { z, x, y, acc });
                }
                TTOp::SSAReduceTile { z, x, scaler, acc, rop, kind } => {
                    if !matches!(rop, BOp::Max | BOp::Add) {
                        panic!("tenstorrent2: reduce op {rop:?} unsupported, op r{z}");
                    }
                    let ci = load_cb(x, &self.load_info, &self.cbs, &self.drain_consumers)
                        .expect("tenstorrent2: reduce reads a non-CB tile");
                    let cs = load_cb(scaler, &self.load_info, &self.cbs, &self.drain_consumers)
                        .expect("tenstorrent2: reduce scaler is no CB tile load");
                    next.push(TTOp::ReduceInit { ci, cs, acc, rop, kind });
                    next.push(TTOp::SSAReduceTile { z, x, scaler, acc, rop, kind });
                }
                TTOp::SSATransposeTile { z, x } => {
                    let cb = load_cb(x, &self.load_info, &self.cbs, &self.drain_consumers)
                        .expect("tenstorrent2: transpose reads a non-CB tile");
                    let out = first_pack_cb
                        .expect("tenstorrent2: transpose needs a startup output CB but compute packs nothing");
                    next.push(TTOp::TransposeInit { cb, out });
                    next.push(TTOp::SSATransposeTile { z, x });
                }
                other => next.push(other),
            }
        }
        self.ops = next;
    }

    /// Pack reconfig insertion (pass 2 of 2): a `PackReconfig` before
    /// every compute-section pack. Naive and unconditional — the first
    /// pack must always reconfigure (silicon starts the packer on
    /// BF16), and redundant reconfigs are safe; the hoist+dedup pass
    /// folds them later.
    fn reconfig_pack(&mut self) {
        let old = std::mem::take(&mut self.ops);
        let mut next = Vec::with_capacity(old.len());
        let mut section = 0u8;
        for op in old {
            match &op {
                TTOp::EndReader | TTOp::EndCompute => {
                    section += 1;
                    next.push(op);
                    continue;
                }
                TTOp::EndWriter => {
                    next.push(op);
                    continue;
                }
                _ => {}
            }
            if section == 1 {
                if let TTOp::SSAStore { dst, .. } = &op {
                    if let Some(&cb) = self.cbs.get(dst) {
                        next.push(TTOp::PackReconfig { cb });
                    }
                }
            }
            next.push(op);
        }
        self.ops = next;
    }

    /// Hoist + dedup of init/reconfig ops (compute section only).
    ///
    /// Hoist: a constant-trip loop body holding exactly one distinct init
    /// config per unit moves that init to the loop preheader. Single-config
    /// is the only safe shape (sticky hardware state: hoisting two configs
    /// would leave the last programmed for the first use). Trip must be a
    /// known `>= 1` (a zero-trip loop would program state for uses that
    /// never run while dedup believes it did); symbolic or zero-trip loops
    /// keep their inits inside, which is always correct. Bodies holding
    /// lock ops are never hoisted across.
    ///
    /// Dedup: per-unit last-programmed state drops redundant inits within
    /// one lock epoch (`MathLock` clears, `PackUnlock` clears). Compute and
    /// sync ops never touch the state. Rebuild, locks/sync/SSA untouched.
    fn hoist_dedup_inits(&mut self) {
        fn is_init(op: &TTOp) -> bool {
            matches!(
                op,
                TTOp::CopyInit { .. }
                    | TTOp::CopyInitWithDt { .. }
                    | TTOp::UnaryInit { .. }
                    | TTOp::BinaryInit { .. }
                    | TTOp::BinScalarInit
                    | TTOp::CastInit { .. }
                    | TTOp::TransposeInit { .. }
                    | TTOp::MatmulInit { .. }
                    | TTOp::ReduceInit { .. }
                    | TTOp::BcastInit { .. }
                    | TTOp::PackReconfig { .. }
            )
        }
        fn is_lock(op: &TTOp) -> bool {
            matches!(op, TTOp::MathLock | TTOp::MathUnlock | TTOp::PackLock | TTOp::PackUnlock)
        }
        // Unit class: unpack programs the unpacker source, math programs the
        // compute unit, pack programs the packer. Dedup tracks one state
        // per unit; hoisting requires a single distinct config per unit.
        #[derive(Clone, Copy, PartialEq, Eq)]
        enum Unit {
            Unpack,
            Math,
            Pack,
        }
        fn unit(op: &TTOp) -> Unit {
            match op {
                TTOp::CopyInit { .. } | TTOp::CopyInitWithDt { .. } => Unit::Unpack,
                TTOp::PackReconfig { .. } => Unit::Pack,
                _ => Unit::Math,
            }
        }
        // Recursive hoist over one level. `section` threads the caller
        // position; only section 1 (compute) holds inits.
        fn hoist_level(ops: Vec<TTOp>, compiler: &Compiler, section: &mut u8) -> Vec<TTOp> {
            let mut out: Vec<TTOp> = Vec::with_capacity(ops.len());
            let mut idx = 0;
            while idx < ops.len() {
                match &ops[idx] {
                    TTOp::EndReader | TTOp::EndCompute => {
                        *section += 1;
                        out.push(ops[idx].clone());
                        idx += 1;
                    }
                    TTOp::SSALoop { z, .. } if *section == 1 => {
                        // Find the matching end (nesting depth counted).
                        let mut depth = 1;
                        let mut end = idx + 1;
                        while end < ops.len() && depth > 0 {
                            match &ops[end] {
                                TTOp::SSALoop { .. } => depth += 1,
                                TTOp::SSAEndLoop => depth -= 1,
                                _ => {}
                            }
                            end += 1;
                        }
                        if depth != 0 {
                            panic!("tenstorrent2: hoist_dedup_inits: unbalanced loop");
                        }
                        let open = ops[idx].clone();
                        let body = hoist_level(ops[idx + 1..end - 1].to_vec(), compiler, &mut 1u8);
                        let close = ops[end - 1].clone();
                        let trip = compiler.trips.get(z).copied().flatten();
                        let single = trip.is_some_and(|t| t >= 1)
                            && !body.iter().any(is_lock);
                        if single {
                            // Distinct init configs per unit, first-occurrence order.
                            let mut seen: Vec<TTOp> = Vec::new();
                            for op in &body {
                                if is_init(op) && !seen.contains(op) {
                                    seen.push(op.clone());
                                }
                            }
                            let mut units: Vec<Unit> = Vec::new();
                            let mut multi = false;
                            for op in &seen {
                                let u = unit(op);
                                if units.contains(&u) {
                                    multi = true;
                                    break;
                                }
                                units.push(u);
                            }
                            if !multi && !seen.is_empty() {
                                out.extend(seen.iter().cloned());
                                out.push(open);
                                out.extend(body.into_iter().filter(|op| !seen.contains(op)));
                                out.push(close);
                            } else {
                                out.push(open);
                                out.extend(body);
                                out.push(close);
                            }
                        } else {
                            out.push(open);
                            out.extend(body);
                            out.push(close);
                        }
                        idx = end;
                    }
                    _ => {
                        out.push(ops[idx].clone());
                        idx += 1;
                    }
                }
            }
            out
        }
        let old = std::mem::take(&mut self.ops);
        let mut section = 0u8;
        let hoisted = hoist_level(old, self, &mut section);
        // CB descriptor formats for unpack-state tracking.
        let mut storage_dtype: Map<OpId, DType> = Map::default();
        for op in &hoisted {
            if let TTOp::SSAStorage { z, dtype, .. } = op {
                storage_dtype.insert(*z, *dtype);
            }
        }
        let mut cb_format: Map<CBId, u32> = Map::default();
        for (&storage, &cb) in self.cbs.iter() {
            let dtype = storage_dtype.get(&storage).expect("tenstorrent2: CB storage without dtype");
            cb_format.insert(cb, cb_fmt(*dtype));
        }
        // Linear dedup over the hoisted stream.
        let mut next: Vec<TTOp> = Vec::with_capacity(hoisted.len());
        let mut section = 0u8;
        let mut unpack: Option<(CBId, u32)> = None;
        let mut math: Option<TTOp> = None;
        let mut pack: Option<CBId> = None;
        for op in hoisted {
            match &op {
                TTOp::EndReader | TTOp::EndCompute => {
                    section += 1;
                    next.push(op);
                    continue;
                }
                TTOp::EndWriter => {
                    next.push(op);
                    continue;
                }
                _ => {}
            }
            if section != 1 {
                next.push(op);
                continue;
            }
            match op {
                TTOp::MathLock => {
                    unpack = None;
                    math = None;
                    pack = None;
                    next.push(TTOp::MathLock);
                }
                TTOp::PackUnlock => {
                    unpack = None;
                    math = None;
                    pack = None;
                    next.push(TTOp::PackUnlock);
                }
                TTOp::MathUnlock | TTOp::PackLock => next.push(op),
                TTOp::CopyInitWithDt { prev, cb } => {
                    let fmt = cb_format[&cb];
                    if unpack != Some((cb, fmt)) {
                        next.push(TTOp::CopyInitWithDt { prev, cb });
                        unpack = Some((cb, fmt));
                    }
                }
                TTOp::CopyInit { cb } => {
                    let fmt = cb_format[&cb];
                    if unpack != Some((cb, fmt)) {
                        next.push(TTOp::CopyInit { cb });
                        unpack = Some((cb, fmt));
                    }
                }
                TTOp::PackReconfig { cb } => {
                    if pack != Some(cb) {
                        next.push(TTOp::PackReconfig { cb });
                        pack = Some(cb);
                    }
                }
                TTOp::UnaryInit { .. }
                | TTOp::BinaryInit { .. }
                | TTOp::BinScalarInit
                | TTOp::CastInit { .. }
                | TTOp::TransposeInit { .. }
                | TTOp::MatmulInit { .. }
                | TTOp::ReduceInit { .. }
                | TTOp::BcastInit { .. } => {
                    if math.as_ref() != Some(&op) {
                        // Broadcast/matmul/transpose/reduce inits also
                        // program the unpacker side; keep it in sync.
                        match &op {
                            TTOp::BcastInit { cb_a, .. } => {
                                unpack = Some((*cb_a, cb_format[cb_a]));
                            }
                            TTOp::MatmulInit { a, .. } => {
                                unpack = Some((*a, cb_format[a]));
                            }
                            TTOp::TransposeInit { cb, .. } => {
                                unpack = Some((*cb, cb_format[cb]));
                            }
                            TTOp::ReduceInit { ci, .. } => {
                                unpack = Some((*ci, cb_format[ci]));
                            }
                            _ => {}
                        }
                        next.push(op.clone());
                        math = Some(op);
                    }
                }
                other => next.push(other),
            }
        }
        self.ops = next;
    }

    /// Print the TTIR stream, one op per line (`r{id}` = SSA values,
    /// `v{id}` = scalar registers, `cb{id}` = circular buffers).
    pub fn debug(&self) {
        println!("{self}");
    }
}

impl Kernel {
    pub fn generate_tenstorrent2(kernel: &Kernel) {
        let mut compiler = Compiler::new(kernel);
        compiler.sectioning();
        compiler.replicate_ops_per_section();
        compiler.assign_cbs();
        compiler.sync_cbs();
        compiler.lock_dst();
        compiler.init_math();
        compiler.reconfig_pack();
        compiler.hoist_dedup_inits();
        compiler.debug();

        todo!()
    }
}

impl Display for Compiler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut indent = String::from(" ");
        let dedent = |indent: &mut String| {
            if indent.len() > 1 {
                indent.pop();
                indent.pop();
            }
        };
        for op in &self.ops {
            match op {
                TTOp::EndReader => writeln!(f, "{indent}end_reader")?,
                TTOp::EndCompute => writeln!(f, "{indent}end_compute")?,
                TTOp::EndWriter => writeln!(f, "{indent}end_writer")?,
                TTOp::Loop { len } => {
                    writeln!(f, "{indent}for _ in 0..v{} {{", len.0)?;
                    indent += "  ";
                }
                TTOp::EndLoop => {
                    dedent(&mut indent);
                    writeln!(f, "{indent}}}")?;
                }
                TTOp::If { cond } => {
                    writeln!(f, "{indent}if v{} {{", cond.0)?;
                    indent += "  ";
                }
                TTOp::EndIf => {
                    dedent(&mut indent);
                    writeln!(f, "{indent}}}")?;
                }
                TTOp::Arg { z, ordinal } => writeln!(f, "{indent}v{} = arg({ordinal})", z.0)?,
                TTOp::Const { z, value } => writeln!(f, "{indent}v{} = {value}", z.0)?,
                TTOp::Binary { z, x, y, bop } => {
                    writeln!(f, "{indent}v{} = {}(v{}, v{})", z.0, format!("{bop:?}").to_lowercase(), x.0, y.0)?;
                }
                TTOp::Unary { z, x, uop } => {
                    writeln!(f, "{indent}v{} = {}(v{})", z.0, format!("{uop:?}").to_lowercase(), x.0)?;
                }
                TTOp::NocAddr { z, addr } => writeln!(f, "{indent}v{} = noc_addr(v{})", z.0, addr.0)?,
                TTOp::CbDeclare { cb, n_tiles } => writeln!(f, "{indent}cb{cb}[{n_tiles}]")?,
                TTOp::ReserveBack { cb, n } => writeln!(f, "{indent}reserve_back(cb{cb}, {n})")?,
                TTOp::PushBack { cb, n } => writeln!(f, "{indent}push_back(cb{cb}, {n})")?,
                TTOp::WaitFront { cb, m } => writeln!(f, "{indent}wait_front(cb{cb}, {m})")?,
                TTOp::PopFront { cb, n } => writeln!(f, "{indent}pop_front(cb{cb}, {n})")?,
                TTOp::AsyncRead { addr, dst_cb, bytes } => {
                    writeln!(f, "{indent}noc_async_read(v{}, cb{dst_cb}, {bytes})", addr.0)?;
                }
                TTOp::NocReadBarrier => writeln!(f, "{indent}noc_async_read_barrier()")?,
                TTOp::AsyncWrite { src_cb, addr, bytes } => {
                    writeln!(f, "{indent}noc_async_write(cb{src_cb}, v{}, {bytes})", addr.0)?;
                }
                TTOp::NocWriteBarrier => writeln!(f, "{indent}noc_async_write_barrier()")?,
                TTOp::MathLock => writeln!(f, "{indent}tile_regs_acquire()")?,
                TTOp::MathUnlock => writeln!(f, "{indent}tile_regs_commit()")?,
                TTOp::PackLock => writeln!(f, "{indent}tile_regs_wait()")?,
                TTOp::PackUnlock => writeln!(f, "{indent}tile_regs_release()")?,
                TTOp::CopyInit { cb } => writeln!(f, "{indent}copy_tile_init({cb});")?,
                TTOp::CopyInitWithDt { prev, cb } => {
                    writeln!(f, "{indent}copy_tile_to_dst_init_short_with_dt({prev}, {cb});")?;
                }
                TTOp::PackReconfig { cb } => writeln!(f, "{indent}pack_reconfig_data_format({cb});")?,
                TTOp::UnaryInit { uop } => writeln!(f, "{indent}{}", unary_init_name(*uop))?,
                TTOp::BinaryInit { bop } => writeln!(
                    f,
                    "{indent}{}",
                    binary_init_name(*bop).expect("tenstorrent2: placed binary init without an init call")
                )?,
                TTOp::BinScalarInit => writeln!(f, "{indent}binop_with_scalar_tile_init();")?,
                TTOp::CastInit { in_dtype, out_dtype } => {
                    writeln!(f, "{indent}typecast_tile_init<{}, {}>();", tt_fmt(*in_dtype), tt_fmt(*out_dtype))?;
                }
                TTOp::TransposeInit { cb, out } => writeln!(f, "{indent}transpose_wh_init({cb}, {out});")?,
                TTOp::MatmulInit { a, b, out } => writeln!(f, "{indent}mm_init({a}, {b}, {out});")?,
                TTOp::ReduceInit { ci, cs, acc, rop, kind } => {
                    let (op_name, dim_name) = match rop {
                        BOp::Max => ("PoolType::MAX", reduce_dim_name(*kind)),
                        BOp::Add => ("PoolType::SUM", reduce_dim_name(*kind)),
                        _ => panic!("tenstorrent2: reduce op {rop:?} has no init call"),
                    };
                    writeln!(f, "{indent}reduce_init<{op_name}, {dim_name}>({ci}, {cs}, r{acc});")?;
                }
                TTOp::BcastInit { bop, kind, cb_a, cb_b } => {
                    let Some(init) = bcast_init_name(*bop, *kind) else {
                        panic!("tenstorrent2: broadcast ({bop:?}, {kind:?}) has no init call")
                    };
                    writeln!(f, "{indent}{init}({cb_a}, {cb_b});")?;
                }
                TTOp::SSAConst { value, z } => writeln!(f, "{indent}r{z} = {value}")?,
                TTOp::SSAParam { dtype, kind, shape, z } => {
                    writeln!(f, "{indent}r{z} = param {kind:?} {dtype} shape=r{shape}")?;
                }
                TTOp::SSACast { z, x, dtype } => writeln!(f, "{indent}r{z} = {dtype}(r{x})")?,
                TTOp::SSABitcast { z, x, dtype } => writeln!(f, "{indent}r{z} = bits({dtype})r{x}")?,
                TTOp::SSAUnary { z, x, uop } => {
                    writeln!(f, "{indent}r{z} = {}(r{x})", format!("{uop:?}").to_lowercase())?;
                }
                TTOp::SSABinary { z, x, y, bop } => {
                    writeln!(f, "{indent}r{z} = {}(r{x}, r{y})", format!("{bop:?}").to_lowercase())?;
                }
                TTOp::SSAStack { z, ops } => writeln!(f, "{indent}r{z} = stack{ops:?}")?,
                TTOp::SSAStorage { z, dtype, scope, len } => {
                    writeln!(f, "{indent}r{z} = storage {scope:?} {dtype}, len={len}")?;
                }
                TTOp::SSAStore { z: _, dst, src, index, layout } => {
                    writeln!(f, "{indent}r{dst}[r{index} @ {layout:?}] = r{src}")?;
                }
                TTOp::SSALoad { z, src, index, layout } => {
                    writeln!(f, "{indent}r{z} = r{src}[r{index} @ {layout:?}]")?;
                }
                TTOp::SSARange { z, axis, kind } => writeln!(f, "{indent}r{z} = range({axis}) {kind:?}")?,
                TTOp::SSALoop { z, len } => {
                    writeln!(f, "{indent}for r{z} in 0..r{len} {{")?;
                    indent += "  ";
                }
                TTOp::SSAEndLoop => {
                    dedent(&mut indent);
                    writeln!(f, "{indent}}}")?;
                }
                TTOp::SSAIf { condition } => {
                    writeln!(f, "{indent}if r{condition} {{")?;
                    indent += "  ";
                }
                TTOp::SSAEndIf => {
                    dedent(&mut indent);
                    writeln!(f, "{indent}}}")?;
                }
                TTOp::SSAMad { x, y, z, w } => writeln!(f, "{indent}r{w} = mad(r{x}, r{y}, r{z})")?,
                TTOp::SSAIndex { z, vec, idx } => writeln!(f, "{indent}r{z} = r{vec}.s{idx}")?,
                TTOp::SSABarrier => writeln!(f, "{indent}barrier")?,
                TTOp::SSAReduceTile { z, x, scaler, acc, rop, kind } => {
                    writeln!(
                        f,
                        "{indent}r{z} = reduce_tile_{kind:?}({})(r{x}, r{scaler}, r{acc})",
                        format!("{rop:?}").to_lowercase()
                    )?;
                }
                TTOp::SSAMatmulTile { z, x, y, acc } => {
                    writeln!(f, "{indent}r{z} = matmul_tile(r{x}, r{y}, r{acc})")?;
                }
                TTOp::SSATransposeTile { z, x } => writeln!(f, "{indent}r{z} = transpose_tile(r{x})")?,
                TTOp::SSABroadcastTile { x, kind } => writeln!(f, "{indent}broadcast_tile_{kind:?}(r{x})")?,
                TTOp::SSAAsm { z, asm, ops } => writeln!(f, "{indent}r{z} = asm {asm:?} {ops:?}")?,
            }
        }
        writeln!(f)
    }
}
