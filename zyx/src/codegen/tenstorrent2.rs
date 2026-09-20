// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! TTIR — Tenstorrent physical IR (see `TTIR_DESIGN.md`).
//!
//! One program = one ordered `Vec<TTOp>` covering all three RISC-V kernel
//! sources, split at render time at the `EndReader`/`EndCompute`/`EndWriter`
//! markers.

use super::tenstorrent::{CBId, NocEmitter, TT_DRAM_PAGE_BYTES, TtSection};
use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind, TileDim, UOp},
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

/// One `Asm` template operand: a CB renders as its index, a tile as
/// its DST slot, a scalar as its register.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsmOperand {
    /// Circular buffer operand.
    Cb(CBId),
    /// DST slot operand.
    Tile(TileId),
    /// Scalar register operand.
    Var(VarId),
}

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
    /// `Const`/`Binary` statement).
    Loop {
        /// Number of iterations (input).
        len: VarId,
        /// Loop counter slot (output, used by body index math).
        counter: VarId,
        /// Counter dtype (always `IDX_T`; carried for render declarations).
        dtype: DType,
        /// Compile-time trip count (`None` = symbolic; traffic under a
        /// symbolic loop is a compilation error in `sync_cbs`).
        trip: Option<u32>,
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
        /// Element dtype (render declaration).
        dtype: DType,
        /// Global launch-arg ordinal (backend runtime-arg tables).
        ordinal: u32,
    },
    /// Scalar constant definition (rendered via `c_type`/`c_code`).
    Const {
        /// Destination scalar register.
        z: VarId,
        /// Literal value.
        value: Constant,
    },
    /// Scalar binary: `z = x <bop> y`.
    Binary {
        /// Destination scalar register.
        z: VarId,
        /// Result dtype (render declaration).
        dtype: DType,
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
        /// Result dtype (render declaration).
        dtype: DType,
        /// Source scalar register.
        x: VarId,
        /// Operation.
        uop: UOp,
    },
    /// Scalar cast: `z = (dtype)x`.
    Cast {
        /// Destination scalar register.
        z: VarId,
        /// Result dtype (render declaration).
        dtype: DType,
        /// Source scalar register.
        x: VarId,
    },
    /// Scalar fused multiply-add: `z = x * y + w`.
    Mad {
        /// Destination scalar register.
        z: VarId,
        /// Result dtype (render declaration).
        dtype: DType,
        /// Multiplicand.
        x: VarId,
        /// Multiplier.
        y: VarId,
        /// Addend.
        w: VarId,
    },
    /// Backend assembly escape hatch: `{i}` substitutes the i-th
    /// operand (CBs as indices, tiles as slots, scalars as registers).
    Asm {
        /// Assembly template.
        asm: TinyString,
        /// Operand values.
        ops: Vec<AsmOperand>,
    },
    /// NOC address computation:
    /// `z = accessor.get_noc_addr(page, offset)` (one `uint64_t` line;
    /// page/offset render inline from the index register).
    NocAddr {
        /// Destination scalar register (the address, always `u64`).
        z: VarId,
        /// Source param ordinal (names the accessor `p{ordinal}`).
        ordinal: u32,
        /// Element index (a scalar register holding the tile index).
        index: VarId,
        /// Element size in bytes (scales the index).
        elem_size: u32,
    },
    /// DRAM accessor declaration for one Global/GlobalMut param
    /// (`TensorAccessor` triple; section-implied `p`/`p_out` naming,
    /// chained compile-time offsets). Rendered once per param per
    /// section, ahead of its first traffic op.
    NocAccessor {
        /// Source param ordinal (names the accessor `p{ordinal}`).
        ordinal: u32,
        /// Element dtype (backend input/output dtype tables).
        dtype: DType,
        /// Global (reader) or GlobalMut (writer).
        kind: ParamKind,
    },
    /// Tensix core grid X coordinate (group range axis 0). Render lowers
    /// to the section's group-arg read (`get_arg_val`, one per axis after
    /// the section args, mirroring the legacy `group_arg` layout).
    TensixGridX {
        /// Destination scalar register.
        z: VarId,
        /// Coordinate dtype (render declaration).
        dtype: DType,
        /// Section-local runtime arg index (lowering precomputes the
        /// section param layout, so render stays a single emitter).
        arg: u32,
    },
    /// Tensix core grid Y coordinate (group range axis 1).
    TensixGridY {
        /// Destination scalar register.
        z: VarId,
        /// Coordinate dtype (render declaration).
        dtype: DType,
        /// Section-local runtime arg index.
        arg: u32,
    },
    // -- CB traffic: FIFO sync, one op per event --
    /// Declare one hardware circular buffer (single object, never duplicated
    /// across sections).
    CbDeclare {
        /// Circular buffer.
        cb: CBId,
        /// Depth in tiles.
        n_tiles: u32,
        /// Runtime descriptor format code (CB storage dtype).
        format: u32,
    },
    /// DST register-file mode (kernel-wide header op, first in the
    /// stream): 32-bit iff the kernel touches F32 tiles. Lowering
    /// asserts the alloc budget; `verify` re-checks the max slot;
    /// render forwards the flag to the backend.
    DstMode {
        /// True = 16-tile BF16 file, false = 8-tile FP32 file.
        bf16: bool,
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
    // -- Movement: pre-sync tile transfers, one op per transfer --
    /// Reader transfer: DRAM tile into a CB (`Global` param source).
    /// `sync_cbs` wraps it with reserve/push; `expand_movement` lowers
    /// it to accessor + address + async read + barrier.
    ReadTile {
        /// Source param ordinal (names the accessor `p{ordinal}`).
        ordinal: u32,
        /// Element dtype (accessor + backend input table).
        dtype: DType,
        /// Element index (scalar register holding the tile index).
        index: VarId,
        /// Destination circular buffer.
        cb: CBId,
        /// Transfer size in bytes.
        bytes: u32,
        /// Element size in bytes (scales the index).
        elem_size: u32,
    },
    /// Writer transfer: CB tile into DRAM (`GlobalMut` param sink).
    /// `sync_cbs` wraps it with wait/pop; `expand_movement` lowers it
    /// to accessor + address + async write + barrier.
    WriteTile {
        /// Source circular buffer.
        cb: CBId,
        /// Destination param ordinal (names `p_out{ordinal}`).
        ordinal: u32,
        /// Element dtype (accessor + backend output table).
        dtype: DType,
        /// Element index (scalar register holding the tile index).
        index: VarId,
        /// Transfer size in bytes.
        bytes: u32,
        /// Element size in bytes (scales the index).
        elem_size: u32,
    },
    // -- NOC: real firmware calls, stay C forever --
    /// `noc_async_read(addr, cb.get_write_ptr()[+ off*bytes], bytes)` —
    /// no barrier (one barrier follows each movement sequence, plus a
    /// trailing reader barrier).
    AsyncRead {
        /// Source NOC address (scalar register holding `get_noc_addr` result).
        addr: VarId,
        /// Destination circular buffer.
        dst_cb: CBId,
        /// Transfer size in bytes.
        bytes: u32,
        /// Tile-slot offset within a hoisted block (`None` = slot 0,
        /// plain write pointer).
        off: Option<VarId>,
    },
    /// `noc_async_read_barrier()`.
    NocReadBarrier,
    /// `noc_async_write(cb.get_read_ptr()[+ off*bytes], addr, bytes)` —
    /// no barrier (one barrier follows each movement sequence).
    AsyncWrite {
        /// Source circular buffer.
        src_cb: CBId,
        /// Destination NOC address.
        addr: VarId,
        /// Transfer size in bytes.
        bytes: u32,
        /// Tile-slot offset within a hoisted block (`None` = slot 0,
        /// plain read pointer).
        off: Option<VarId>,
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
    /// `compute_kernel_hw_startup(in0, in1, out);` (compute front,
    /// ahead of all loops; single-input kernels repeat in0). Emitted
    /// by `verify` when compute both loads and packs (pure movement
    /// needs no startup); matmul kernels carry none (`mm_init` owns
    /// the long init). Render emits it verbatim.
    ComputeStartup {
        /// First-loaded circular buffer (second load, or in0).
        in0: CBId,
        /// Second-loaded circular buffer.
        in1: CBId,
        /// First-packed circular buffer.
        out: CBId,
    },
    /// `reduce_init<op, dim>(ci, cs, acc);` (always inline at its op:
    /// the acc slot only exists after tile allocation, so `tile_regs`
    /// emits this together with the `TileReduce`, never `init_math`).
    ReduceInit {
        /// Input circular buffer.
        ci: CBId,
        /// Scaler circular buffer.
        cs: CBId,
        /// Accumulator DST slot (also the result tile).
        acc: TileId,
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
    // -- Tile DST: physical tile registers, no passes over these --
    /// Streaming copy in: `copy_tile(cb, index, slot)` (runs under MATH).
    TileCopy {
        /// Destination DST slot.
        slot: TileId,
        /// Source circular buffer.
        cb: CBId,
        /// Tile slot within the waited block (scalar register).
        index: VarId,
    },
    /// Pack out: `pack_tile(slot, cb)` (runs under PACK).
    TilePack {
        /// Source DST slot.
        slot: TileId,
        /// Destination circular buffer.
        cb: CBId,
    },
    /// Tiled binary ALU: `op(x, y, dst)` (inputs stay live).
    TileBinary {
        /// Destination DST slot.
        dst: TileId,
        /// Left operand slot.
        x: TileId,
        /// Right operand slot.
        y: TileId,
        /// Operation.
        bop: BOp,
    },
    /// Fused broadcast binary (`add_tiles_bcast_rows`, ...): the full
    /// tile stays in `cb_a`, the broadcast lane in `cb_b`, the result
    /// lands in a fresh DST slot.
    TileBcastBinary {
        /// Destination DST slot.
        dst: TileId,
        /// Full-tile circular buffer.
        cb_a: CBId,
        /// Broadcast-lane circular buffer.
        cb_b: CBId,
        /// Operation.
        bop: BOp,
        /// Broadcast dimension.
        kind: TileDim,
        /// First-packed compute CB (init pass placeholder: `None` from
        /// lowering, filled when the init pass places `BcastInit`).
        out: Option<CBId>,
    },
    /// Tile-scalar binary (`add_unary_tile`, ... with an fp32-bits
    /// immediate): DST-inplace like unary, no CB traffic for the
    /// scalar side.
    TileBinScalar {
        /// Operand and result slot.
        slot: TileId,
        /// Operation (selects the `*_unary_tile` call).
        bop: BOp,
        /// Scalar side as fp32 bits.
        bits: u32,
    },
    /// Tiled unary ALU: in-place `op(slot)` (SFPU mutates the slot).
    TileUnary {
        /// Operand and result slot.
        slot: TileId,
        /// Operation.
        uop: UOp,
    },
    /// Tiled cast: `typecast_tile<in, out>(slot)` (in-place like unary;
    /// Tenstorrent has no DST->DST copy).
    TileCast {
        /// Operand and result slot.
        slot: TileId,
        /// Source dtype.
        in_dtype: DType,
        /// Target dtype.
        out_dtype: DType,
    },
    /// Streaming transpose: `transpose_wh_tile(cb, 0, dst)` (the input
    /// streams from its CB, fused drain, never copied to DST first).
    TileTranspose {
        /// Destination DST slot.
        dst: TileId,
        /// Source circular buffer.
        cb: CBId,
        /// First-packed compute CB (init pass placeholder: `None` from
        /// lowering, filled when the init pass places `TransposeInit`).
        out: Option<CBId>,
    },
    /// Fused matmul: `matmul_tiles(cb_a, cb_b, acc, acc, acc)`
    /// (inputs stay in CBs, accumulate into the acc slot).
    TileMatmul {
        /// Accumulator DST slot (also the result).
        acc: TileId,
        /// Left input circular buffer.
        cb_a: CBId,
        /// Right input circular buffer.
        cb_b: CBId,
        /// First-packed compute CB (init pass placeholder: `None` from
        /// lowering, filled when the init pass places `MatmulInit`).
        out: Option<CBId>,
    },
    /// Fused reduce: `reduce_tile<op, dim>(cb_in, cb_sc, 0, 0, acc)`.
    TileReduce {
        /// Accumulator DST slot (also the result).
        acc: TileId,
        /// Input circular buffer.
        cb_in: CBId,
        /// Scaler circular buffer.
        cb_sc: CBId,
        /// Reduction op.
        rop: BOp,
        /// In-tile dimension.
        kind: TileDim,
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

/// Lowered TTIR: one physical op stream with section boundaries.
/// Built once by [`Compiler::new`]; every later pass rewrites only this vector.
struct Compiler {
    ops: Vec<TTOp>,
}

impl Compiler {
    /// Lowering: one walk per section op list from `get_needed_ops`
    /// (already replicated per section: each section is a separate
    /// kernel with its own registers, arg ordinals stay global).
    /// Scalar values bind `VarId`s, tiled values bind `TileId`s,
    /// circular storages bind one global `CBId`; a use count reaching
    /// zero frees its slot for reuse (shared slots stay live while
    /// any bound id is). Single pass, nothing else: fusion, sync,
    /// locks, and inits are later passes over the ops.
    fn new(kernel: &Kernel) -> Self {
        // Section gate, same rule as the legacy `check_sections`:
        // exactly 2 barriers delimiting reader/compute/writer, and no
        // GPU-only Wmma (tenstorrent has `MatmulTile`, no WMMA units).
        let mut barriers = 0u32;
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            match &kernel.ops[scan].op {
                Op::Barrier => barriers += 1,
                Op::Wmma { .. } => panic!("tenstorrent2: Wmma is GPU-only, tenstorrent uses Op::MatmulTile"),
                _ => {}
            }
            scan = kernel.next_op(scan);
        }
        if barriers != 2 {
            panic!("tenstorrent2: need exactly 2 barriers (3 sections), found {barriers}");
        }
        // DST mode, same scan as the legacy `generate_tenstorrent`:
        // 32-bit iff the kernel touches F32 tiles (F32 storage, or an
        // F8 circular sharing the core per the Blackhole mandate).
        let mut dst_bf16 = true;
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            if let Op::Storage { dtype, scope, .. } = kernel.ops[scan].op {
                match (dtype, scope) {
                    (DType::F32, _) | (DType::F8E4M3, MemScope::Circular) => {
                        dst_bf16 = false;
                        break;
                    }
                    _ => {}
                }
            }
            scan = kernel.next_op(scan);
        }
        let budget = if dst_bf16 { TileId::BUDGET_BF16 } else { TileId::BUDGET_FP32 };
        let param_ordinal_of = NocEmitter::new(kernel).param_ordinal_of;
        // CB allocation: first-touch order over loads/stores of
        // Circular storages, with the legacy validity checks (whole
        // 1024-element tiles, whole pages, single-core L1 budget,
        // hardware CB count fit).
        let mut cbs: Map<OpId, CBId> = Map::default();
        let mut cb_order: Vec<OpId> = Vec::new();
        let mut scan = kernel.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            let storage = match &kernel.ops[scan].op {
                Op::Load { src, .. } => Some(*src),
                Op::Store { dst, .. } => Some(*dst),
                _ => None,
            };
            if let Some(st) = storage
                && matches!(kernel.ops[st].op, Op::Storage { scope: MemScope::Circular, .. })
                && !cbs.contains_key(&st)
            {
                cbs.insert(st, CBId(cb_order.len() as u32));
                cb_order.push(st);
            }
            scan = kernel.next_op(scan);
        }
        let num_circular_buffers = kernel.device_info().num_circular_buffers;
        if cb_order.len() > num_circular_buffers as usize {
            panic!("tenstorrent2: kernel needs {} circular buffers, device holds {num_circular_buffers}", cb_order.len());
        }
        let mut ops = vec![TTOp::DstMode { bf16: dst_bf16 }];
        for (cb, &st) in cb_order.iter().enumerate() {
            let Op::Storage { dtype, len, .. } = kernel.ops[st].op else {
                unreachable!("tenstorrent2: CB map entry {st} is not a storage op")
            };
            let elem = dtype.bit_size() as i64 / 8;
            let bytes = len * elem;
            if len % 1024 != 0 {
                panic!("tenstorrent2: CB{cb} holds {len} elements, not whole 1024-element tiles");
            }
            let page = 1024 * elem;
            if bytes % page != 0 {
                panic!("tenstorrent2: CB{cb} holds {bytes} bytes, not whole {page}B pages");
            }
            if bytes > 32768 {
                panic!("tenstorrent2: CB{cb} needs {bytes} bytes, single-core L1 budget is 32768");
            }
            ops.push(TTOp::CbDeclare { cb: CBId(cb as u32), n_tiles: (len / 1024) as u32, format: cb_fmt(dtype) });
        }
        // A side resolving to a compile-time float constant (follows
        // const expressions): folds into a `*_unary_tile` immediate.
        // Integer constants are NOT converted (a tile op's scalar lane
        // is float; silent int→float would hide dtype bugs).
        let const_f32_bits = |op: OpId| -> Option<u32> {
            use crate::scalar::{bf16, f16};
            match kernel.resolve_const(op)? {
                Constant::F32(b) => Some(f32::from_le_bytes(b).to_bits()),
                Constant::F16(b) => Some(f16::from_le_bytes(b).to_f32().to_bits()),
                Constant::BF16(b) => Some(bf16::from_le_bytes(b).to_f32().to_bits()),
                _ => None,
            }
        };
        let sections = [TtSection::Reader, TtSection::Compute, TtSection::Writer];
        for (s, tt_section) in sections.into_iter().enumerate() {
            let data = kernel.get_needed_ops(tt_section);
            let total = data.rcs.clone();
            let mut remaining = data.rcs.clone();
            let mut vars: Map<OpId, VarId> = Map::default();
            let mut tiles: Map<OpId, TileId> = Map::default();
            let mut free_vars: Vec<VarId> = Vec::new();
            let mut free_tiles: Vec<TileId> = Vec::new();
            let mut next_var = 0u32;
            let mut next_tile = 0u8;
            // Section params in IR order: this section kernel's runtime args.
            let section_params: Vec<OpId> =
                data.ops.iter().copied().filter(|op| matches!(kernel.ops[*op].op, Op::Param { .. })).collect();
            // Bind a fresh (or freed) scalar register to a value.
            let mut def_var = |vars: &mut Map<OpId, VarId>, free_vars: &mut Vec<VarId>, next_var: &mut u32, id: OpId| -> VarId {
                let v = free_vars.pop().unwrap_or_else(|| {
                    let v = VarId(*next_var);
                    *next_var += 1;
                    v
                });
                vars.insert(id, v);
                v
            };
            // Consume one use of a scalar value, freeing its register at zero.
            let mut use_var =
                |vars: &Map<OpId, VarId>, remaining: &mut Map<OpId, u32>, free_vars: &mut Vec<VarId>, id: OpId| -> VarId {
                    let &v = vars.get(&id).unwrap_or_else(|| panic!("tenstorrent2: scalar op {id} has no register"));
                    let left = remaining.get_mut(&id).unwrap_or_else(|| panic!("tenstorrent2: scalar op {id} has no use count"));
                    assert!(*left > 0, "tenstorrent2: scalar op {id} used past its uses");
                    *left -= 1;
                    if *left == 0 {
                        free_vars.push(v);
                    }
                    v
                };
            // Bind a fresh (or freed) DST slot to a tiled value.
            let mut def_tile =
                |tiles: &mut Map<OpId, TileId>, free_tiles: &mut Vec<TileId>, next_tile: &mut u8, id: OpId| -> TileId {
                    let t = free_tiles.pop().unwrap_or_else(|| {
                        let t = TileId(*next_tile);
                        *next_tile += 1;
                        t
                    });
                    assert!((t.0 as usize) < budget, "tenstorrent2: DST budget exceeded");
                    tiles.insert(id, t);
                    t
                };
            for &id in &data.ops {
                match &kernel.ops[id].op {
                    Op::Const(c) => {
                        let z = def_var(&mut vars, &mut free_vars, &mut next_var, id);
                        ops.push(TTOp::Const { z, value: c.clone() });
                    }
                    Op::Param { dtype, kind, .. } => match kind {
                        ParamKind::Variable => {
                            let z = def_var(&mut vars, &mut free_vars, &mut next_var, id);
                            let ordinal =
                                param_ordinal_of.get(&id).copied().expect("tenstorrent2: variable param missing ordinal");
                            ops.push(TTOp::Arg { z, dtype: *dtype, ordinal });
                        }
                        ParamKind::Global | ParamKind::GlobalMut => {
                            if s == 1 {
                                panic!("tenstorrent2: compute touches DRAM through param {id}");
                            }
                            let ordinal = param_ordinal_of.get(&id).copied().expect("tenstorrent2: DRAM param missing ordinal");
                            ops.push(TTOp::NocAccessor { ordinal, dtype: *dtype, kind: *kind });
                        }
                    },
                    Op::Storage { scope, .. } => match scope {
                        MemScope::Circular => {}
                        MemScope::Register => {
                            def_tile(&mut tiles, &mut free_tiles, &mut next_tile, id);
                        }
                        MemScope::Local => unreachable!(
                            "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                        ),
                        MemScope::Global => todo!("tenstorrent2 storage scope, op {id}"),
                    },
                    Op::Load { src, index, layout } => {
                        if !matches!(layout, MemLayout::Tile { .. }) {
                            if s == 1 {
                                todo!("tenstorrent2 compute only supports tile loads");
                            }
                            continue;
                        }
                        if s != 1 {
                            continue;
                        }
                        if matches!(kernel.ops[*src].op, Op::Storage { scope: MemScope::Register, .. }) {
                            let tile = tiles.get(src).copied().expect("tenstorrent2: compute acc load reads an undeclared acc");
                            tiles.insert(id, tile);
                            continue;
                        }
                        let Some(&cb) = cbs.get(src) else {
                            panic!("tenstorrent2: compute load targets unmapped CB, op {id}");
                        };
                        let slot = def_tile(&mut tiles, &mut free_tiles, &mut next_tile, id);
                        let index = use_var(&vars, &mut remaining, &mut free_vars, *index);
                        ops.push(TTOp::TileCopy { slot, cb, index });
                    }
                    Op::Store { dst, src, index, layout } => {
                        if !matches!(layout, MemLayout::Tile { .. }) {
                            todo!("tenstorrent2 only supports tile stores, op {id}");
                        }
                        if s == 0 {
                            let Op::Load { src: ld_src, index: ld_idx, layout: ld_layout } = kernel.ops[*src].op else {
                                panic!("tenstorrent2: reader supports only global to local stores, op {id} has ops in between");
                            };
                            let Op::Param { kind: ParamKind::Global, .. } = kernel.ops[ld_src].op else {
                                panic!("tenstorrent2: reader load op {id} is not from a Global param");
                            };
                            let Op::Storage { dtype, scope: MemScope::Circular, .. } = kernel.ops[*dst].op else {
                                panic!("tenstorrent2: reader store op {id} does not target a Circular CB");
                            };
                            let Some(&cb) = cbs.get(dst) else {
                                panic!("tenstorrent2: reader store op {id} targets unmapped CB");
                            };
                            let MemLayout::Tile { x, y, .. } = ld_layout else {
                                todo!("tenstorrent2 reader only supports tile stores");
                            };
                            let elem_size = dtype.bit_size() as u32 / 8;
                            let index = use_var(&vars, &mut remaining, &mut free_vars, ld_idx);
                            ops.push(TTOp::ReadTile {
                                ordinal: param_ordinal_of[&ld_src],
                                dtype,
                                index,
                                cb,
                                bytes: x as u32 * y as u32 * elem_size,
                                elem_size,
                            });
                            continue;
                        }
                        if s == 2 {
                            let Op::Load { src: cb_src, index: _, layout: ld_layout } = kernel.ops[*src].op else {
                                panic!("tenstorrent2: writer supports only CB to DRAM stores, op {id} has ops in between");
                            };
                            let Some(&cb) = cbs.get(&cb_src) else {
                                panic!("tenstorrent2: writer load op {id} targets unmapped CB");
                            };
                            let Op::Param { dtype, kind: ParamKind::GlobalMut, .. } = kernel.ops[*dst].op else {
                                panic!("tenstorrent2: writer store dst must be a GlobalMut param, op {id}");
                            };
                            let MemLayout::Tile { x, y, .. } = ld_layout else {
                                todo!("tenstorrent2 writer only supports tile stores");
                            };
                            let elem_size = dtype.bit_size() as u32 / 8;
                            let index = use_var(&vars, &mut remaining, &mut free_vars, *index);
                            ops.push(TTOp::WriteTile {
                                cb,
                                ordinal: param_ordinal_of[dst],
                                dtype,
                                index,
                                bytes: x as u32 * y as u32 * elem_size,
                                elem_size,
                            });
                            continue;
                        }
                        if let Op::Storage { scope: MemScope::Register, .. } = kernel.ops[*dst].op {
                            if let Some(&tile) = tiles.get(src) {
                                tiles.insert(*dst, tile);
                            } else if let Op::Load { src: lsrc, .. } = kernel.ops[*src].op
                                && let Some(&tile) = tiles.get(&lsrc)
                            {
                                tiles.insert(*dst, tile);
                            }
                            continue;
                        }
                        let Some(&cb) = cbs.get(dst) else {
                            panic!("tenstorrent2: compute store op {id} targets unmapped CB");
                        };
                        let slot =
                            tiles.get(src).copied().expect("tenstorrent2: compute acc store reads a tile with no DST slot");
                        ops.push(TTOp::TilePack { slot, cb });
                    }
                    Op::Cast { x, dtype } => {
                        if matches!(data.dtypes[&id].1, MemLayout::Tile { .. }) {
                            let slot = tiles.get(x).copied().expect("tenstorrent2: tiled cast reads a value with no DST slot");
                            if total[&x] != 1 {
                                todo!("tenstorrent2 multi-use tiled cast operand, op {id}");
                            }
                            let in_dtype = data.dtypes[&x].0;
                            tiles.insert(id, slot);
                            ops.push(TTOp::TileCast { slot, in_dtype, out_dtype: *dtype });
                        } else {
                            let z = def_var(&mut vars, &mut free_vars, &mut next_var, id);
                            let x = use_var(&vars, &mut remaining, &mut free_vars, *x);
                            ops.push(TTOp::Cast { z, dtype: *dtype, x });
                        }
                    }
                    Op::Bitcast { x, .. } => {
                        if matches!(data.dtypes[&id].1, MemLayout::Tile { .. }) {
                            let tile = tiles.get(x).copied().expect("tenstorrent2: tiled bitcast reads a value with no DST slot");
                            tiles.insert(id, tile);
                        } else {
                            todo!("tenstorrent2 scalar bitcast, op {id}");
                        }
                    }
                    Op::Unary { x, uop } => {
                        if matches!(data.dtypes[&id].1, MemLayout::Tile { .. }) {
                            let slot = tiles.get(x).copied().expect("tenstorrent2: tiled unary reads a value with no DST slot");
                            tiles.insert(id, slot);
                            ops.push(TTOp::TileUnary { slot, uop: *uop });
                        } else {
                            let z = def_var(&mut vars, &mut free_vars, &mut next_var, id);
                            let x = use_var(&vars, &mut remaining, &mut free_vars, *x);
                            ops.push(TTOp::Unary { z, dtype: data.dtypes[&id].0, x, uop: *uop });
                        }
                    }
                    Op::Binary { x, y, bop } => {
                        if !matches!(data.dtypes[&id].1, MemLayout::Tile { .. }) {
                            let z = def_var(&mut vars, &mut free_vars, &mut next_var, id);
                            let x = use_var(&vars, &mut remaining, &mut free_vars, *x);
                            let y = use_var(&vars, &mut remaining, &mut free_vars, *y);
                            ops.push(TTOp::Binary { z, dtype: data.dtypes[&id].0, x, y, bop: *bop });
                            continue;
                        }
                        let marker = |side: OpId| match kernel.ops[side].op {
                            Op::BroadcastTile { x: mx, kind } => Some((kind, mx)),
                            _ => None,
                        };
                        let plain_cb = |side: OpId| match kernel.ops[side].op {
                            Op::Load { src: lsrc, .. } => cbs.get(&lsrc).copied(),
                            _ => None,
                        };
                        match (marker(*x), marker(*y)) {
                            (Some(_), Some(_)) => {
                                panic!("tenstorrent2: broadcast op {id} marks both sides");
                            }
                            (Some((kind, mx)), None) => {
                                let (Some(cb_b), Some(cb_a)) = (plain_cb(mx), plain_cb(*y)) else {
                                    panic!("tenstorrent2: broadcast op {id} side is no CB tile load");
                                };
                                let dst = def_tile(&mut tiles, &mut free_tiles, &mut next_tile, id);
                                ops.push(TTOp::TileBcastBinary { dst, cb_a, cb_b, bop: *bop, kind, out: None });
                            }
                            (None, Some((kind, my))) => {
                                let (Some(cb_b), Some(cb_a)) = (plain_cb(my), plain_cb(*x)) else {
                                    panic!("tenstorrent2: broadcast op {id} side is no CB tile load");
                                };
                                let dst = def_tile(&mut tiles, &mut free_tiles, &mut next_tile, id);
                                ops.push(TTOp::TileBcastBinary { dst, cb_a, cb_b, bop: *bop, kind, out: None });
                            }
                            (None, None) => {
                                let xc = const_f32_bits(*x);
                                let yc = const_f32_bits(*y);
                                let scalar = match (xc, yc) {
                                    (None, Some(bits)) => Some(("tile", bits, *x)),
                                    (Some(bits), None) => Some(("const", bits, *y)),
                                    _ => None,
                                };
                                if let Some((side, bits, tile_op)) = scalar {
                                    match (*bop, side) {
                                        (BOp::Add, _) | (BOp::Mul, _) | (BOp::Sub, _) | (BOp::Div, "tile") => {}
                                        _ => {
                                            panic!("tenstorrent2: const-first {bop:?} has no scalar call, op {id}");
                                        }
                                    };
                                    let t = tiles
                                        .get(&tile_op)
                                        .copied()
                                        .expect("tenstorrent2: scalar binary reads a value with no DST slot");
                                    if total[&tile_op] != 1 {
                                        panic!("tenstorrent2: scalar binary {id} reads a multi-use operand");
                                    }
                                    tiles.insert(id, t);
                                    ops.push(TTOp::TileBinScalar { slot: t, bop: *bop, bits });
                                } else {
                                    let ta =
                                        tiles.get(x).copied().expect("tenstorrent2: tiled binary reads a value with no DST slot");
                                    let tb =
                                        tiles.get(y).copied().expect("tenstorrent2: tiled binary reads a value with no DST slot");
                                    let dst = def_tile(&mut tiles, &mut free_tiles, &mut next_tile, id);
                                    ops.push(TTOp::TileBinary { dst, x: ta, y: tb, bop: *bop });
                                }
                            }
                        }
                    }
                    Op::Mad { x, y, z } => {
                        if matches!(data.dtypes[&id].1, MemLayout::Tile { .. }) {
                            todo!("tenstorrent2 tiled mad, op {id}");
                        }
                        let v = def_var(&mut vars, &mut free_vars, &mut next_var, id);
                        let x = use_var(&vars, &mut remaining, &mut free_vars, *x);
                        let y = use_var(&vars, &mut remaining, &mut free_vars, *y);
                        let z = use_var(&vars, &mut remaining, &mut free_vars, *z);
                        ops.push(TTOp::Mad { z: v, dtype: data.dtypes[&id].0, x, y, w: z });
                    }
                    Op::Stack { .. } => todo!("tenstorrent2 scalar stack, op {id}"),
                    Op::Index { .. } => todo!("tenstorrent2 scalar index, op {id}"),
                    Op::Range { axis, kind } => match kind {
                        RangeKind::Group(_) => {
                            let z = def_var(&mut vars, &mut free_vars, &mut next_var, id);
                            let arg = section_params.len() as u32 + axis;
                            match axis {
                                0 => ops.push(TTOp::TensixGridX { z, dtype: data.dtypes[&id].0, arg }),
                                1 => ops.push(TTOp::TensixGridY { z, dtype: data.dtypes[&id].0, arg }),
                                _ => todo!("tenstorrent2 group range axis {axis}, op {id}"),
                            }
                        }
                        RangeKind::Local(_) => {
                            unreachable!(
                                "tenstorrent does not have local threads; local indices should have been converted to loops by the opt_tenstorrent_tile optimization pass"
                            )
                        }
                        RangeKind::Warp(_) => {
                            unreachable!("tenstorrent has no warps; warp ranges are gpu-only")
                        }
                    },
                    Op::Loop { len } => {
                        let bound = use_var(&vars, &mut remaining, &mut free_vars, *len);
                        let trip = match kernel.resolve_const(*len).and_then(|c| c.as_dim()) {
                            Some(d) if d >= 0 => Some(d as u32),
                            Some(d) => panic!("tenstorrent2: negative loop trip count {d}, op {id}"),
                            None => None,
                        };
                        let counter = def_var(&mut vars, &mut free_vars, &mut next_var, id);
                        ops.push(TTOp::Loop { len: bound, counter, dtype: IDX_T, trip });
                    }
                    Op::EndLoop => ops.push(TTOp::EndLoop),
                    Op::If { condition } => {
                        let cond = use_var(&vars, &mut remaining, &mut free_vars, *condition);
                        ops.push(TTOp::If { cond });
                    }
                    Op::EndIf => ops.push(TTOp::EndIf),
                    Op::Barrier => unreachable!("should've been filtered by kernel sections decomposition"),
                    Op::Wmma { .. } => unreachable!("tenstorrent2: Wmma has no Tenstorrent lowering"),
                    Op::Move { .. } => unreachable!("tenstorrent2: Move never survives linearization"),
                    Op::Reduce { .. } => unreachable!("tenstorrent2: Reduce never survives linearization"),
                    Op::ReduceTile { x, scaler, acc, rop, kind } => {
                        let Op::Load { src: lx, layout: MemLayout::Tile { x: wx, y: hx, .. }, .. } = kernel.ops[*x].op else {
                            panic!("tenstorrent2: reduce side op {x} is no CB tile load");
                        };
                        if wx as u32 != 32 || hx as u32 != 32 {
                            panic!("tenstorrent2: reduce is fixed 32x32, op {x} is {wx}x{hx}");
                        }
                        let Some(&cb_in) = cbs.get(&lx) else {
                            panic!("tenstorrent2: reduce side op {x} targets unmapped CB");
                        };
                        let Op::Load { src: la, .. } = kernel.ops[*acc].op else {
                            panic!("tenstorrent2: reduce acc op {acc} is no acc tile load");
                        };
                        if !matches!(kernel.ops[la].op, Op::Storage { scope: MemScope::Register, .. }) {
                            panic!("tenstorrent2: reduce acc op {acc} does not thread a Register acc");
                        }
                        let Op::Load { src: ls, layout: MemLayout::Tile { .. }, .. } = kernel.ops[*scaler].op else {
                            panic!("tenstorrent2: reduce scaler op {scaler} is no scaler tile load");
                        };
                        let Some(&cb_sc) = cbs.get(&ls) else {
                            panic!("tenstorrent2: reduce scaler op {scaler} targets unmapped CB");
                        };
                        let Some(&acc_slot) = tiles.get(&la) else {
                            panic!("tenstorrent2: reduce acc op {acc} reads an undeclared acc");
                        };
                        tiles.insert(id, acc_slot);
                        ops.push(TTOp::TileReduce { acc: acc_slot, cb_in, cb_sc, rop: *rop, kind: *kind });
                    }
                    Op::MatmulTile { x, y, acc } => {
                        let Op::Load { src: la, layout: MemLayout::Tile { .. }, .. } = kernel.ops[*x].op else {
                            panic!("tenstorrent2: matmul side op {x} is no CB tile load");
                        };
                        let Some(&cb_a) = cbs.get(&la) else {
                            panic!("tenstorrent2: matmul side op {x} targets unmapped CB");
                        };
                        let Op::Load { src: lb, layout: MemLayout::Tile { .. }, .. } = kernel.ops[*y].op else {
                            panic!("tenstorrent2: matmul side op {y} is no CB tile load");
                        };
                        let Some(&cb_b) = cbs.get(&lb) else {
                            panic!("tenstorrent2: matmul side op {y} targets unmapped CB");
                        };
                        let Op::Load { src: lacc, .. } = kernel.ops[*acc].op else {
                            panic!("tenstorrent2: matmul acc op {acc} is no acc tile load");
                        };
                        let Some(&tile) = tiles.get(&lacc) else {
                            panic!("tenstorrent2: matmul acc op {acc} reads an undeclared acc");
                        };
                        tiles.insert(id, tile);
                        ops.push(TTOp::TileMatmul { acc: tile, cb_a, cb_b, out: None });
                    }
                    Op::TransposeTile { x } => {
                        let Op::Load { src: lx, layout: MemLayout::Tile { x: wx, y: hx, .. }, .. } = kernel.ops[*x].op else {
                            panic!("tenstorrent2: transpose side op {x} is no CB tile load");
                        };
                        if wx as u32 != 32 || hx as u32 != 32 {
                            panic!("tenstorrent2: transpose is fixed 32x32, op {x} is {wx}x{hx}");
                        }
                        let Some(&cb) = cbs.get(&lx) else {
                            panic!("tenstorrent2: transpose side op {x} targets unmapped CB");
                        };
                        let dst = def_tile(&mut tiles, &mut free_tiles, &mut next_tile, id);
                        ops.push(TTOp::TileTranspose { dst, cb, out: None });
                    }
                    Op::BroadcastTile { .. } => {}
                    Op::Asm { asm, ops: operands } => {
                        let mut resolved = Vec::with_capacity(operands.len());
                        for &operand in operands.iter() {
                            if let Op::Storage { scope: MemScope::Circular, .. } = kernel.ops[operand].op {
                                let Some(&cb) = cbs.get(&operand) else {
                                    panic!("tenstorrent2: asm operand {operand} targets unmapped CB");
                                };
                                resolved.push(AsmOperand::Cb(cb));
                            } else if let Some(&slot) = tiles.get(&operand) {
                                resolved.push(AsmOperand::Tile(slot));
                            } else if let Some(&reg) = vars.get(&operand) {
                                resolved.push(AsmOperand::Var(reg));
                            } else {
                                panic!("tenstorrent2: asm operand {operand} is not a CB or live tile");
                            }
                        }
                        ops.push(TTOp::Asm { asm: asm.clone(), ops: resolved });
                    }
                }
            }
            ops.push(match s {
                0 => TTOp::EndReader,
                1 => TTOp::EndCompute,
                _ => TTOp::EndWriter,
            });
        }
        Self { ops }
    }

    /// Replicate_ops_per_section: values consumed in multiple sections are
    /// duplicated per section (each section is a separate kernel with its own
    /// registers and runtime args; arg ordinals stay global so the launch
    /// contract survives). Circular-buffer storages are the exception: one
    /// hardware object, one `CBId`, never duplicated — they are hoisted once
    /// to the stream head. Mirrors the legacy `get_needed_ops` closure per
    /// section (stores + structural starters, transitive data deps), then
    /// re-emits each section's needed ops in stream order.
    /// Sync insertion: `ReserveBack`/`WaitFront`/`PushBack`/`PopFront`
    /// around every CB traffic op. Single-tile transactions wrap the op
    /// inline; a loop body holding exactly one traffic op for a CB, no
    /// branch, and a constant trip > 1 fitting the CB depth upgrades to
    /// a hoisted block (open before the loop, close after its end).
    /// Fused-draining loads carry no syncs; their consumers drain.
    /// `CbDeclare` ops head the stream in `CBId` order.
    fn sync_cbs(&mut self) {
        // BIGBANG: physical port lands after compile; lowering emits no syncs yet.
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
        // MATH ops are tile compute ops (copies stream under MATH
        // like every unpack-side op); PACK ops are `TilePack`s draining
        // to a CB. Scalar and movement sections carry no locks. Acc
        // threading leaves no ops (alias bindings in lowering), so no
        // register scan: only emitted ops lock.
        let is_math = |op: &TTOp| -> bool {
            match op {
                TTOp::TileCopy { .. }
                | TTOp::TileUnary { .. }
                | TTOp::TileBinary { .. }
                | TTOp::TileBcastBinary { .. }
                | TTOp::TileBinScalar { .. }
                | TTOp::TileCast { .. }
                | TTOp::TileMatmul { .. }
                | TTOp::TileReduce { .. }
                | TTOp::TileTranspose { .. } => true,
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
                TTOp::EndLoop => {
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
            // Pack path: `TilePack` draining to a CB.
            if let TTOp::TilePack { .. } = &op {
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
            if is_math(&op) {
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
        // BIGBANG: physical port lands after compile; lowering emits no inits yet.
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
                if let TTOp::TilePack { cb, .. } = &op {
                    next.push(TTOp::PackReconfig { cb: *cb });
                }
            }
            next.push(op);
        }
        self.ops = next;
    }

    /// NOC movement lowering (reader/writer sections only; runs after
    /// `scalar_regs`, before the compute-only passes).
    ///
    /// Reader `Store{dst: Circular, src: Load{src: Global param}}` becomes
    /// `NocAccessor` (once per param) + `NocAddr` + `AsyncRead` + barrier;
    /// writer `Store{dst: GlobalMut param, src: Load{src: CB storage}}`
    /// becomes the `AsyncWrite` form. Loads drop (consumed at the store);
    /// a load with no consuming store is a compilation error, like the
    /// legacy "supports only global to local stores" rule.
    ///
    /// The slot offset derives from the op stream alone (no side tables):
    /// the open sync transaction for the CB (`ReserveBack`/`WaitFront`
    /// with `n == 1` means slot 0) plus the loop-counter provenance of
    /// the index (`off` = the index when counter-tied, else the hoist
    /// loop's counter — the legacy `slot_offset` rule).
    ///
    /// v1: tile layouts only, reader `Global→Circular`, writer
    /// `CB→GlobalMut`. Everything else stays loud.
    fn noc_movement(&mut self) {
        // BIGBANG: expansion of `ReadTile`/`WriteTile` lands after
        // compile; this pass currently forwards the stream.
        let old = std::mem::take(&mut self.ops);
        let mut next = Vec::with_capacity(old.len());
        let mut section = 0usize;
        // Pending movement loads per section: `z` to (storage, index, layout).
        let mut loads: Map<OpId, (OpId, OpId, MemLayout)> = Map::default();
        let mut consumed: Set<OpId> = Set::default();
        // Accessors already declared per section.
        let mut accessors: Set<OpId> = Set::default();
        // Open sync transaction per CB: (tiles, loop depth at open).
        let mut open: Map<CBId, (u32, usize)> = Map::default();
        // Loop-counter stack (scalar registers).
        let mut loops: Vec<VarId> = Vec::new();
        // Counter provenance per scalar register (for slot offsets).
        let mut dep: Map<VarId, Set<VarId>> = Map::default();
        // Scalar registers holding constant 0.
        let mut zeros: Set<VarId> = Set::default();
        for op in old {
            match op {
                TTOp::EndReader | TTOp::EndCompute => {
                    if section < 2 {
                        for &z in loads.keys() {
                            debug_assert!(
                                consumed.contains(&z),
                                "tenstorrent2: noc_movement: movement load r{z} has no consuming store"
                            );
                        }
                    }
                    loads.clear();
                    consumed.clear();
                    accessors.clear();
                    open.clear();
                    loops.clear();
                    section += 1;
                    if section == 1 {
                        // Trailing reader barrier: every async read lands
                        // before exit (legacy `final_read_barrier`).
                        next.push(TTOp::NocReadBarrier);
                        next.push(TTOp::EndReader);
                    } else {
                        next.push(TTOp::EndCompute);
                    }
                    continue;
                }
                TTOp::EndWriter => {
                    for &z in loads.keys() {
                        debug_assert!(
                            consumed.contains(&z),
                            "tenstorrent2: noc_movement: movement load r{z} has no consuming store"
                        );
                    }
                    next.push(TTOp::EndWriter);
                    continue;
                }
                _ => {}
            }
            debug_assert!(section < 3, "tenstorrent2: noc_movement: op past EndWriter");
            // Provenance + loop/sync tracking runs over all sections
            // (compute ops flow through untouched below).
            match &op {
                TTOp::Const { z, value } => {
                    if value.as_dim() == Some(0) {
                        zeros.insert(*z);
                    }
                }
                TTOp::Binary { z, x, y, .. } => {
                    let mut d = dep.get(x).cloned().unwrap_or_default();
                    d.extend(dep.get(y).cloned().unwrap_or_default());
                    dep.insert(*z, d);
                }
                TTOp::Unary { z, x, .. } => {
                    dep.insert(*z, dep.get(x).cloned().unwrap_or_default());
                }
                TTOp::Loop { counter, .. } => {
                    let mut d = Set::default();
                    d.insert(*counter);
                    dep.insert(*counter, d);
                }
                TTOp::ReserveBack { cb, n } => {
                    open.insert(*cb, (*n, loops.len()));
                }
                TTOp::PushBack { cb, .. } => {
                    open.remove(cb);
                }
                TTOp::WaitFront { cb, m } => {
                    open.insert(*cb, (*m, loops.len()));
                }
                TTOp::PopFront { cb, .. } => {
                    open.remove(cb);
                }
                _ => {}
            }
            if section == 1 {
                next.push(op);
                continue;
            }
            // Loop stack pushes need the counter out of the op.
            match op {
                TTOp::Loop { len, counter, .. } => {
                    loops.push(counter);
                    next.push(op);
                    continue;
                }
                other => next.push(other),
            }
        }
        self.ops = next;
    }

    /// Hoist writer DRAM accessors to the writer-section front.
    ///
    /// Legacy v1 declares all writer accessors up front (chained
    /// `TensorAccessor` triples) while the reader declares inline at
    /// first traffic. Render is a single dumb emitter, so the ordering
    /// lives here as a stable partition: writer-section `NocAccessor`
    /// ops move to immediately after `EndCompute`, keeping their
    /// relative (chain) order; every other op keeps its position.
    fn hoist_writer_accessors(&mut self) {
        let old = std::mem::take(&mut self.ops);
        let mut hoisted = Vec::new();
        let mut next = Vec::with_capacity(old.len());
        let mut section = 0u8;
        let mut writer_front = 0usize;
        for op in old {
            match &op {
                TTOp::EndReader | TTOp::EndCompute => {
                    section += 1;
                    next.push(op);
                    if section == 2 {
                        writer_front = next.len();
                    }
                }
                TTOp::NocAccessor { .. } if section == 2 => hoisted.push(op),
                _ => next.push(op),
            }
        }
        assert!(section == 2, "tenstorrent2: hoist_writer_accessors: stream has no writer section");
        next.splice(writer_front..writer_front, hoisted);
        self.ops = next;
    }
    fn tile_regs(&mut self) {
        // BIGBANG: subsumed by lowering.
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
                    TTOp::Loop { trip, .. } if *section == 1 => {
                        // Find the matching end (nesting depth counted).
                        let mut depth = 1;
                        let mut end = idx + 1;
                        while end < ops.len() && depth > 0 {
                            match &ops[end] {
                                TTOp::Loop { .. } => depth += 1,
                                TTOp::EndLoop => depth -= 1,
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
                        let single = trip.is_some_and(|t| t >= 1) && !body.iter().any(is_lock);
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
        // CB descriptor formats for unpack-state tracking, off the
        // `CbDeclare` head.
        let mut cb_format: Map<CBId, u32> = Map::default();
        for op in &hoisted {
            if let TTOp::CbDeclare { cb, format, .. } = op {
                cb_format.insert(*cb, *format);
            }
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

    /// Verify the fully-physical stream (runs after `tile_regs`, before
    /// render). Structural firewall: whatever the passes did, the final
    /// stream must be launchable. Loud panic at the exact op.
    ///
    /// Checks: section termination (any marker prefix, so reader-only
    /// kernels pass; in order, nothing past the last marker), zero `SSA*`
    /// ops (phase invariant), DST lock pairing per section (compute-only),
    /// CB open/close balance per section + pushed==popped program-wide,
    /// declare-before-use (CBs, accessors) and def-before-use (`VarId`,
    /// `TileId`) per section, tile-slot budgets, Loop/If balance.
    fn verify(&mut self) {
        // Section markers seen (bitmask: 1 reader, 2 compute, 4 writer).
        let mut seen = 0u8;
        let mut section = 0usize;
        // DST lock state (compute only).
        #[derive(PartialEq)]
        enum Lock {
            Unlocked,
            Math,
            Pack,
        }
        let mut lock = Lock::Unlocked;
        // CB FIFO: declared set, per-CB (reserved, avail, waited),
        // program-wide (pushed, popped).
        let mut declared: Set<CBId> = Set::default();
        let mut fifo: Map<CBId, (u32, u32, u32)> = Map::default();
        let mut totals: Map<CBId, (u32, u32)> = Map::default();
        // Per-section defined values (cleared at each End*).
        let mut scalars: Set<VarId> = Set::default();
        let mut accessors: Set<u32> = Set::default();
        // Tile liveness, replayed from the op stream: a pre-walk records
        // the def/use event order per (section, slot); the main walk
        // replays it, so reuse-after-death passes but clobbering a live
        // slot or using an undefined one panics. (`true` = use.)
        let mut events: Map<(usize, TileId), std::collections::VecDeque<bool>> = Map::default();
        {
            let mut sec = 0usize;
            for op in self.ops.iter() {
                match op {
                    TTOp::EndReader | TTOp::EndCompute | TTOp::EndWriter => sec += 1,
                    TTOp::ReduceInit { acc, .. } => {
                        events.entry((sec, *acc)).or_default().push_back(false);
                    }
                    TTOp::TileCopy { slot, .. } => {
                        events.entry((sec, *slot)).or_default().push_back(false);
                    }
                    TTOp::TilePack { slot, .. } => {
                        events.entry((sec, *slot)).or_default().push_back(true);
                    }
                    TTOp::TileBinary { dst, x, y, .. } => {
                        events.entry((sec, *x)).or_default().push_back(true);
                        events.entry((sec, *y)).or_default().push_back(true);
                        events.entry((sec, *dst)).or_default().push_back(false);
                    }
                    TTOp::TileUnary { slot, .. } => {
                        events.entry((sec, *slot)).or_default().push_back(true);
                        events.entry((sec, *slot)).or_default().push_back(false);
                    }
                    TTOp::TileCast { slot, .. } => {
                        events.entry((sec, *slot)).or_default().push_back(true);
                        events.entry((sec, *slot)).or_default().push_back(false);
                    }
                    TTOp::TileTranspose { dst, .. } => {
                        events.entry((sec, *dst)).or_default().push_back(false);
                    }
                    TTOp::TileMatmul { acc, .. } => {
                        // Acc storage reuse: first mention allocates
                        // (def), later mentions reuse (use) — exactly the
                        // `acc_tile` rule.
                        let reuse = events.contains_key(&(sec, *acc));
                        events.entry((sec, *acc)).or_default().push_back(reuse);
                    }
                    TTOp::TileReduce { acc, .. } => {
                        events.entry((sec, *acc)).or_default().push_back(true);
                    }
                    TTOp::TileBcastBinary { dst, .. } => {
                        events.entry((sec, *dst)).or_default().push_back(false);
                    }
                    TTOp::TileBinScalar { slot, .. } => {
                        events.entry((sec, *slot)).or_default().push_back(true);
                        events.entry((sec, *slot)).or_default().push_back(false);
                    }
                    _ => {}
                }
            }
        }
        let mut defined: Set<(usize, TileId)> = Set::default();
        // Startup triple inputs, same scan as the legacy
        // `generate_compute` (`loaded_order` over compute loads,
        // `stored_first` over compute stores, single-input kernels
        // repeat in0). Fused drains count as loads — they are `Load`
        // ops in the legacy list with no `TileCopy` here.
        let mut loaded_order: Vec<CBId> = Vec::new();
        let mut stored_first: Option<CBId> = None;
        let mut note_load = |loaded_order: &mut Vec<CBId>, cb: CBId| {
            if !loaded_order.contains(&cb) {
                loaded_order.push(cb);
            }
        };
        // Replay one event against the pre-walk order: a def at the
        // queue front proves all prior uses consumed (reuse-after-death
        // passes, clobbering a live slot is order-impossible); a use
        // requires a prior def.
        let tile_def = |events: &mut Map<(usize, TileId), std::collections::VecDeque<bool>>,
                        defined: &mut Set<(usize, TileId)>,
                        sec: usize,
                        slot: TileId| {
            match events.get_mut(&(sec, slot)).map(|q| q.pop_front()) {
                Some(Some(false)) => {}
                Some(_) => panic!("tenstorrent2: verify: tile t{} redefined while live", slot.0),
                None => panic!("tenstorrent2: verify: tile t{} defined with no recorded event", slot.0),
            }
            defined.insert((sec, slot));
        };
        let tile_use = |events: &mut Map<(usize, TileId), std::collections::VecDeque<bool>>,
                        defined: &mut Set<(usize, TileId)>,
                        sec: usize,
                        slot: TileId| {
            assert!(defined.contains(&(sec, slot)), "tenstorrent2: verify: tile t{} undefined", slot.0);
            match events.get_mut(&(sec, slot)).map(|q| q.pop_front()) {
                Some(Some(true)) => {}
                _ => panic!("tenstorrent2: verify: tile t{} use out of event order", slot.0),
            }
        };
        let mut max_slot = 0u8;
        let mut depth = 0u32;
        let end_section = |seen: &mut u8,
                           section: &mut usize,
                           lock: &mut Lock,
                           fifo: &mut Map<CBId, (u32, u32, u32)>,
                           scalars: &mut Set<VarId>,
                           accessors: &mut Set<u32>,
                           depth: &mut u32,
                           marker: u8| {
            assert!(*seen & marker == 0, "tenstorrent2: verify: duplicate section marker");
            *seen |= marker;
            *section += 1;
            assert!(*lock == Lock::Unlocked, "tenstorrent2: verify: section ends with DST locked");
            for (cb, (reserved, _, waited)) in fifo.iter() {
                assert!(*reserved == 0, "tenstorrent2: verify: section ends with CB{cb} reserve open");
                assert!(*waited == 0, "tenstorrent2: verify: section ends with CB{cb} wait open");
            }
            assert!(*depth == 0, "tenstorrent2: verify: section ends inside a walk");
            scalars.clear();
            accessors.clear();
        };
        for op in self.ops.iter() {
            match op {
                TTOp::EndReader => {
                    end_section(&mut seen, &mut section, &mut lock, &mut fifo, &mut scalars, &mut accessors, &mut depth, 1)
                }
                TTOp::EndCompute => {
                    assert!(seen & 1 != 0, "tenstorrent2: verify: EndCompute without EndReader");
                    end_section(&mut seen, &mut section, &mut lock, &mut fifo, &mut scalars, &mut accessors, &mut depth, 2);
                }
                TTOp::EndWriter => {
                    assert!(seen & 3 != 0, "tenstorrent2: verify: EndWriter without a prior section");
                    end_section(&mut seen, &mut section, &mut lock, &mut fifo, &mut scalars, &mut accessors, &mut depth, 4);
                }
                _ => {}
            }
            if matches!(op, TTOp::EndReader | TTOp::EndCompute | TTOp::EndWriter) {
                continue;
            }
            assert!(seen != 7, "tenstorrent2: verify: op past EndWriter");
            match op {
                TTOp::Loop { len, counter, .. } => {
                    assert!(scalars.contains(len), "tenstorrent2: verify: loop bound v{} undefined", len.0);
                    assert!(scalars.insert(*counter), "tenstorrent2: verify: loop counter v{} redefined", counter.0);
                    depth += 1;
                }
                TTOp::EndLoop => {
                    assert!(depth > 0, "tenstorrent2: verify: EndLoop without Loop");
                    depth -= 1;
                }
                TTOp::If { cond } => {
                    assert!(scalars.contains(cond), "tenstorrent2: verify: branch cond v{} undefined", cond.0);
                    depth += 1;
                }
                TTOp::EndIf => {
                    assert!(depth > 0, "tenstorrent2: verify: EndIf without If");
                    depth -= 1;
                }
                TTOp::Arg { z, .. } | TTOp::Const { z, .. } | TTOp::TensixGridX { z, .. } | TTOp::TensixGridY { z, .. } => {
                    assert!(scalars.insert(*z), "tenstorrent2: verify: scalar v{} redefined", z.0);
                }
                TTOp::Binary { z, x, y, .. } => {
                    assert!(scalars.contains(x), "tenstorrent2: verify: scalar v{} undefined", x.0);
                    assert!(scalars.contains(y), "tenstorrent2: verify: scalar v{} undefined", y.0);
                    assert!(scalars.insert(*z), "tenstorrent2: verify: scalar v{} redefined", z.0);
                }
                TTOp::Cast { z, x, .. } => {
                    assert!(scalars.contains(x), "tenstorrent2: verify: scalar v{} undefined", x.0);
                    assert!(scalars.insert(*z), "tenstorrent2: verify: scalar v{} redefined", z.0);
                }
                TTOp::Mad { z, x, y, w, .. } => {
                    assert!(scalars.contains(x), "tenstorrent2: verify: scalar v{} undefined", x.0);
                    assert!(scalars.contains(y), "tenstorrent2: verify: scalar v{} undefined", y.0);
                    assert!(scalars.contains(w), "tenstorrent2: verify: scalar v{} undefined", w.0);
                    assert!(scalars.insert(*z), "tenstorrent2: verify: scalar v{} redefined", z.0);
                }
                TTOp::Asm { ops: operands, .. } => {
                    for operand in operands {
                        match operand {
                            AsmOperand::Cb(cb) => {
                                assert!(declared.contains(cb), "tenstorrent2: verify: asm on undeclared CB{cb}");
                            }
                            AsmOperand::Tile(slot) => {
                                tile_use(&mut events, &mut defined, section, *slot);
                            }
                            AsmOperand::Var(v) => {
                                assert!(scalars.contains(v), "tenstorrent2: verify: scalar v{} undefined", v.0);
                            }
                        }
                    }
                }
                TTOp::ReadTile { cb, index, .. } => {
                    assert!(declared.contains(cb), "tenstorrent2: verify: read on undeclared CB{cb}");
                    assert!(scalars.contains(index), "tenstorrent2: verify: read index v{} undefined", index.0);
                }
                TTOp::WriteTile { cb, index, .. } => {
                    assert!(declared.contains(cb), "tenstorrent2: verify: write on undeclared CB{cb}");
                    assert!(scalars.contains(index), "tenstorrent2: verify: write index v{} undefined", index.0);
                }
                TTOp::DstMode { .. } => {}
                TTOp::ComputeStartup { in0, in1, out } => {
                    assert!(declared.contains(in0), "tenstorrent2: verify: startup on undeclared CB{in0}");
                    assert!(declared.contains(in1), "tenstorrent2: verify: startup on undeclared CB{in1}");
                    assert!(declared.contains(out), "tenstorrent2: verify: startup on undeclared CB{out}");
                }
                TTOp::Unary { z, x, .. } => {
                    assert!(scalars.contains(x), "tenstorrent2: verify: scalar v{} undefined", x.0);
                    assert!(scalars.insert(*z), "tenstorrent2: verify: scalar v{} redefined", z.0);
                }
                TTOp::NocAccessor { ordinal, .. } => {
                    assert!(accessors.insert(*ordinal), "tenstorrent2: verify: duplicate accessor p{ordinal}");
                }
                TTOp::NocAddr { z, ordinal, index, .. } => {
                    assert!(accessors.contains(ordinal), "tenstorrent2: verify: address uses undeclared accessor p{ordinal}");
                    assert!(scalars.contains(index), "tenstorrent2: verify: address index v{} undefined", index.0);
                    assert!(scalars.insert(*z), "tenstorrent2: verify: scalar v{} redefined", z.0);
                }
                TTOp::AsyncRead { addr, dst_cb, off, .. } => {
                    assert!(scalars.contains(addr), "tenstorrent2: verify: read addr v{} undefined", addr.0);
                    assert!(declared.contains(dst_cb), "tenstorrent2: verify: read on undeclared CB{dst_cb}");
                    if let Some(o) = off {
                        assert!(scalars.contains(o), "tenstorrent2: verify: read offset v{} undefined", o.0);
                    }
                }
                TTOp::AsyncWrite { src_cb, addr, off, .. } => {
                    assert!(scalars.contains(addr), "tenstorrent2: verify: write addr v{} undefined", addr.0);
                    assert!(declared.contains(src_cb), "tenstorrent2: verify: write on undeclared CB{src_cb}");
                    if let Some(o) = off {
                        assert!(scalars.contains(o), "tenstorrent2: verify: write offset v{} undefined", o.0);
                    }
                }
                TTOp::NocReadBarrier | TTOp::NocWriteBarrier => {}
                TTOp::CbDeclare { cb, .. } => {
                    assert!(declared.insert(*cb), "tenstorrent2: verify: duplicate CB{cb} declaration");
                    fifo.insert(*cb, (0, 0, 0));
                    totals.insert(*cb, (0, 0));
                }
                TTOp::ReserveBack { cb, n } => {
                    assert!(declared.contains(cb), "tenstorrent2: verify: reserve on undeclared CB{cb}");
                    let e = fifo.get_mut(cb).expect("tenstorrent2: verify: reserve on undeclared CB");
                    assert!(e.0 == 0 && e.2 == 0, "tenstorrent2: verify: reserve on CB{cb} with open transaction");
                    e.0 = *n;
                }
                TTOp::PushBack { cb, n } => {
                    let e = fifo.get_mut(cb).expect("tenstorrent2: verify: push on undeclared CB");
                    assert!(e.0 >= *n, "tenstorrent2: verify: push of {n} on CB{cb} with {e:?} reserved");
                    e.0 -= *n;
                    e.1 += *n;
                    totals.get_mut(cb).expect("tenstorrent2: verify: push on undeclared CB").0 += *n;
                }
                TTOp::WaitFront { cb, m } => {
                    let e = fifo.get_mut(cb).expect("tenstorrent2: verify: wait on undeclared CB");
                    assert!(e.2 == 0, "tenstorrent2: verify: wait on CB{cb} with open wait");
                    assert!(e.1 >= *m, "tenstorrent2: verify: wait of {m} on CB{cb} with {e:?} available");
                    e.2 = *m;
                    e.1 -= *m;
                }
                TTOp::PopFront { cb, n } => {
                    let e = fifo.get_mut(cb).expect("tenstorrent2: verify: pop on undeclared CB");
                    assert!(e.2 >= *n, "tenstorrent2: verify: pop of {n} on CB{cb} with {e:?} waited");
                    e.2 -= *n;
                    totals.get_mut(cb).expect("tenstorrent2: verify: pop on undeclared CB").1 += *n;
                }
                TTOp::MathLock => {
                    assert!(section == 1, "tenstorrent2: verify: DST lock outside compute");
                    assert!(lock == Lock::Unlocked || lock == Lock::Pack, "tenstorrent2: verify: acquire with DST already held");
                    lock = Lock::Math;
                }
                TTOp::MathUnlock => {
                    assert!(lock == Lock::Math, "tenstorrent2: verify: commit without MATH lock");
                    lock = Lock::Unlocked;
                }
                TTOp::PackLock => {
                    assert!(section == 1, "tenstorrent2: verify: DST lock outside compute");
                    assert!(lock == Lock::Unlocked, "tenstorrent2: verify: pack wait without release");
                    lock = Lock::Pack;
                }
                TTOp::PackUnlock => {
                    assert!(lock == Lock::Pack, "tenstorrent2: verify: release without PACK lock");
                    lock = Lock::Unlocked;
                }
                TTOp::CopyInit { .. }
                | TTOp::CopyInitWithDt { .. }
                | TTOp::PackReconfig { .. }
                | TTOp::UnaryInit { .. }
                | TTOp::BinaryInit { .. }
                | TTOp::BinScalarInit
                | TTOp::CastInit { .. }
                | TTOp::TransposeInit { .. }
                | TTOp::MatmulInit { .. }
                | TTOp::BcastInit { .. } => {}
                TTOp::ReduceInit { acc, .. } => {
                    // Defines the acc slot (init precedes its op in the stream).
                    tile_def(&mut events, &mut defined, section, *acc);
                    max_slot = max_slot.max(acc.0);
                }
                TTOp::TileCopy { slot, cb, index } => {
                    assert!(declared.contains(cb), "tenstorrent2: verify: copy on undeclared CB{cb}");
                    assert!(scalars.contains(index), "tenstorrent2: verify: copy index v{} undefined", index.0);
                    tile_def(&mut events, &mut defined, section, *slot);
                    max_slot = max_slot.max(slot.0);
                    if section == 1 {
                        note_load(&mut loaded_order, *cb);
                    }
                }
                TTOp::TilePack { slot, cb } => {
                    tile_use(&mut events, &mut defined, section, *slot);
                    assert!(declared.contains(cb), "tenstorrent2: verify: pack on undeclared CB{cb}");
                    if section == 1 && stored_first.is_none() {
                        stored_first = Some(*cb);
                    }
                }
                TTOp::TileBinary { dst, x, y, .. } => {
                    tile_use(&mut events, &mut defined, section, *x);
                    tile_use(&mut events, &mut defined, section, *y);
                    tile_def(&mut events, &mut defined, section, *dst);
                    max_slot = max_slot.max(dst.0);
                }
                TTOp::TileUnary { slot, .. } => {
                    // In-place: use then redefine (matches the pre-walk order).
                    tile_use(&mut events, &mut defined, section, *slot);
                    tile_def(&mut events, &mut defined, section, *slot);
                }
                TTOp::TileCast { slot, .. } => {
                    // In-place: use then redefine (matches the pre-walk order).
                    tile_use(&mut events, &mut defined, section, *slot);
                    tile_def(&mut events, &mut defined, section, *slot);
                }
                TTOp::TileTranspose { dst, cb } => {
                    assert!(declared.contains(cb), "tenstorrent2: verify: transpose on undeclared CB{cb}");
                    tile_def(&mut events, &mut defined, section, *dst);
                    max_slot = max_slot.max(dst.0);
                    if section == 1 {
                        note_load(&mut loaded_order, *cb);
                    }
                }
                TTOp::TileMatmul { acc, cb_a, cb_b } => {
                    assert!(declared.contains(cb_a), "tenstorrent2: verify: matmul on undeclared CB{cb_a}");
                    assert!(declared.contains(cb_b), "tenstorrent2: verify: matmul on undeclared CB{cb_b}");
                    // Acc storage reuse across ops: first mention defines.
                    if defined.contains(&(section, *acc)) {
                        tile_use(&mut events, &mut defined, section, *acc);
                    } else {
                        tile_def(&mut events, &mut defined, section, *acc);
                    }
                    max_slot = max_slot.max(acc.0);
                    if section == 1 {
                        note_load(&mut loaded_order, *cb_a);
                        note_load(&mut loaded_order, *cb_b);
                    }
                }
                TTOp::TileBcastBinary { dst, cb_a, cb_b, .. } => {
                    assert!(declared.contains(cb_a), "tenstorrent2: verify: bcast on undeclared CB{cb_a}");
                    assert!(declared.contains(cb_b), "tenstorrent2: verify: bcast on undeclared CB{cb_b}");
                    tile_def(&mut events, &mut defined, section, *dst);
                    max_slot = max_slot.max(dst.0);
                    if section == 1 {
                        note_load(&mut loaded_order, *cb_a);
                        note_load(&mut loaded_order, *cb_b);
                    }
                }
                TTOp::TileBinScalar { slot, .. } => {
                    // In-place: use then redefine (matches the pre-walk order).
                    tile_use(&mut events, &mut defined, section, *slot);
                    tile_def(&mut events, &mut defined, section, *slot);
                }
                TTOp::TileReduce { acc, cb_in, cb_sc, .. } => {
                    assert!(declared.contains(cb_in), "tenstorrent2: verify: reduce on undeclared CB{cb_in}");
                    assert!(declared.contains(cb_sc), "tenstorrent2: verify: reduce on undeclared CB{cb_sc}");
                    tile_use(&mut events, &mut defined, section, *acc);
                    if section == 1 {
                        note_load(&mut loaded_order, *cb_in);
                        note_load(&mut loaded_order, *cb_sc);
                    }
                }
                _ => panic!("tenstorrent2: verify: op {op:?} is not fully lowered (SSA remains)"),
            }
        }
        assert!(seen != 0, "tenstorrent2: verify: stream holds no section");
        assert!(lock == Lock::Unlocked, "tenstorrent2: verify: stream ends with DST locked");
        assert!(depth == 0, "tenstorrent2: verify: stream ends inside a walk");
        for ((sec, slot), q) in events.iter() {
            assert!(q.is_empty(), "tenstorrent2: verify: section {sec} tile t{} has unreplayed events", slot.0);
        }
        for (cb, (pushed, popped)) in totals.iter() {
            assert!(pushed == popped, "tenstorrent2: verify: CB{cb} pushed {pushed} but popped {popped} program-wide");
        }
        let bf16 = self
            .ops
            .iter()
            .find_map(|op| match op {
                TTOp::DstMode { bf16 } => Some(*bf16),
                _ => None,
            })
            .expect("tenstorrent2: verify: stream has no DstMode head");
        let budget = if bf16 {
            TileId::BUDGET_BF16 as u8
        } else {
            TileId::BUDGET_FP32 as u8
        };
        assert!(max_slot < budget, "tenstorrent2: verify: tile t{max_slot} exceeds the DST budget {budget}");
        // Startup triple, legacy rule: needs a load and a store
        // (pure movement needs no startup); single-input kernels
        // repeat in0. Goes in the stream as a compute-front op so
        // render emits it with no scan and no state.
        if let (Some(&in0), Some(out)) = (loaded_order.first(), stored_first) {
            let in1 = loaded_order.get(1).copied().unwrap_or(in0);
            let front = self
                .ops
                .iter()
                .position(|op| matches!(op, TTOp::EndReader))
                .expect("tenstorrent2: verify: stream has no reader section")
                + 1;
            self.ops.insert(front, TTOp::ComputeStartup { in0, in1, out });
        }
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
        compiler.sync_cbs();
        compiler.lock_dst();
        compiler.init_math();
        compiler.reconfig_pack();
        compiler.hoist_dedup_inits();
        compiler.noc_movement();
        compiler.hoist_writer_accessors();
        compiler.tile_regs();
        compiler.verify();
        compiler.debug();

        todo!()
    }
}

impl Display for Compiler {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut indent = String::from(" ");
        let mut section = 0u8;
        let dedent = |indent: &mut String| {
            if indent.len() > 1 {
                indent.pop();
                indent.pop();
            }
        };
        for op in &self.ops {
            match op {
                TTOp::EndReader => {
                    writeln!(f, "{indent}end_reader")?;
                    section += 1;
                }
                TTOp::EndCompute => {
                    writeln!(f, "{indent}end_compute")?;
                    section += 1;
                }
                TTOp::EndWriter => writeln!(f, "{indent}end_writer")?,
                TTOp::Loop { len, counter } => {
                    writeln!(f, "{indent}for v{} in 0..v{} {{", counter.0, len.0)?;
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
                TTOp::NocAddr { z, param, index, elem_size } => {
                    let page = TT_DRAM_PAGE_BYTES;
                    let prefix = if section == 2 { "p_out" } else { "p" };
                    writeln!(
                        f,
                        "{indent}v{} = {prefix}{param}.get_noc_addr((v{}*{elem_size})/{page}, (v{}*{elem_size})%{page})",
                        z.0, index.0, index.0
                    )?;
                }
                TTOp::NocAccessor { param, ordinal, .. } => {
                    let prefix = if section == 2 { "p_out" } else { "p" };
                    writeln!(f, "{indent}{prefix}{param} = dram_accessor(arg({ordinal}))")?;
                }
                TTOp::TensixGridX { z } => writeln!(f, "{indent}v{} = tensix_grid_x()", z.0)?,
                TTOp::TensixGridY { z } => writeln!(f, "{indent}v{} = tensix_grid_y()", z.0)?,
                TTOp::CbDeclare { cb, n_tiles } => writeln!(f, "{indent}cb{cb}[{n_tiles}]")?,
                TTOp::ReserveBack { cb, n } => writeln!(f, "{indent}reserve_back(cb{cb}, {n})")?,
                TTOp::PushBack { cb, n } => writeln!(f, "{indent}push_back(cb{cb}, {n})")?,
                TTOp::WaitFront { cb, m } => writeln!(f, "{indent}wait_front(cb{cb}, {m})")?,
                TTOp::PopFront { cb, n } => writeln!(f, "{indent}pop_front(cb{cb}, {n})")?,
                TTOp::AsyncRead { addr, dst_cb, bytes, off } => match off {
                    None => writeln!(f, "{indent}noc_async_read(v{}, cb{dst_cb}, {bytes})", addr.0)?,
                    Some(o) => writeln!(f, "{indent}noc_async_read(v{}, cb{dst_cb} + v{}*{bytes}, {bytes})", addr.0, o.0)?,
                },
                TTOp::NocReadBarrier => writeln!(f, "{indent}noc_async_read_barrier()")?,
                TTOp::AsyncWrite { src_cb, addr, bytes, off } => match off {
                    None => writeln!(f, "{indent}noc_async_write(cb{src_cb}, v{}, {bytes})", addr.0)?,
                    Some(o) => writeln!(f, "{indent}noc_async_write(cb{src_cb} + v{}*{bytes}, v{}, {bytes})", o.0, addr.0)?,
                },
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
                TTOp::ComputeStartup { in0, in1, out } => writeln!(f, "{indent}compute_kernel_hw_startup({in0}, {in1}, {out});")?,
                TTOp::ReduceInit { ci, cs, acc, rop, kind } => {
                    let (op_name, dim_name) = match rop {
                        BOp::Max => ("PoolType::MAX", reduce_dim_name(*kind)),
                        BOp::Add => ("PoolType::SUM", reduce_dim_name(*kind)),
                        _ => panic!("tenstorrent2: reduce op {rop:?} has no init call"),
                    };
                    writeln!(f, "{indent}reduce_init<{op_name}, {dim_name}>({ci}, {cs}, t{});", acc.0)?;
                }
                TTOp::BcastInit { bop, kind, cb_a, cb_b } => {
                    let Some(init) = bcast_init_name(*bop, *kind) else {
                        panic!("tenstorrent2: broadcast ({bop:?}, {kind:?}) has no init call")
                    };
                    writeln!(f, "{indent}{init}({cb_a}, {cb_b});")?;
                }
                TTOp::TileCopy { slot, cb, index } => {
                    writeln!(f, "{indent}copy_tile({cb}, v{}, t{});", index.0, slot.0)?;
                }
                TTOp::TilePack { slot, cb } => {
                    writeln!(f, "{indent}pack_tile(t{}, {cb});", slot.0)?;
                }
                TTOp::TileBinary { dst, x, y, bop } => {
                    let name = match bop {
                        BOp::Add => "add_binary_tile",
                        BOp::Sub => "sub_binary_tile",
                        BOp::Mul => "mul_binary_tile",
                        BOp::Div => "div_binary_tile",
                        BOp::Max => "binary_max_tile",
                        BOp::BitShiftLeft => "binary_left_shift_tile",
                        BOp::BitShiftRight => "binary_right_shift_tile",
                        _ => panic!("tenstorrent2: tiled binary {bop:?} has no LLK call"),
                    };
                    writeln!(f, "{indent}{name}(t{}, t{}, t{});", x.0, y.0, dst.0)?;
                }
                TTOp::TileUnary { slot, uop } => {
                    // Log2 passes its base scale explicitly (legacy form).
                    if *uop == UOp::Log2 {
                        writeln!(f, "{indent}log_with_base_tile(t{}, 0x3fb8aa3b);", slot.0)?;
                    } else {
                        let name = match uop {
                            UOp::Neg => "negative_tile",
                            UOp::BitNot => "bitwise_not_tile",
                            UOp::Exp => "exp_tile",
                            UOp::Exp2 => "exp2_tile",
                            UOp::Log2 => unreachable!("tenstorrent2: log2 is emitted above"),
                            UOp::Reciprocal => "recip_tile",
                            UOp::Sqrt => "sqrt_tile",
                            UOp::Rsqrt => "rsqrt_tile",
                            UOp::Sin => "sin_tile",
                            UOp::Cos => "cos_tile",
                            UOp::Floor => "floor_tile",
                            UOp::Trunc => "trunc_tile",
                            UOp::Abs => "abs_tile",
                            UOp::Not => "logical_not_tile",
                        };
                        writeln!(f, "{indent}{name}(t{});", slot.0)?;
                    }
                }
                TTOp::TileCast { slot, in_dtype, out_dtype } => {
                    writeln!(f, "{indent}typecast_tile<{}, {}>(t{});", tt_fmt(*in_dtype), tt_fmt(*out_dtype), slot.0)?;
                }
                TTOp::TileTranspose { dst, cb } => {
                    writeln!(f, "{indent}transpose_wh_tile({cb}, 0, t{});", dst.0)?;
                }
                TTOp::TileMatmul { acc, cb_a, cb_b } => {
                    writeln!(f, "{indent}matmul_tiles({cb_a}, {cb_b}, t{}, t{}, t{});", acc.0, acc.0, acc.0)?;
                }
                TTOp::TileReduce { acc, cb_in, cb_sc, rop, kind } => {
                    let (op_name, dim_name) = match rop {
                        BOp::Max => ("PoolType::MAX", reduce_dim_name(*kind)),
                        BOp::Add => ("PoolType::SUM", reduce_dim_name(*kind)),
                        _ => panic!("tenstorrent2: reduce op {rop:?} has no LLK call"),
                    };
                    writeln!(f, "{indent}reduce_tile<{op_name}, {dim_name}>({cb_in}, {cb_sc}, 0, 0, t{});", acc.0)?;
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
