// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! TTIR — Tenstorrent physical IR (see `TTIR_DESIGN.md`).
//!
//! One program = one ordered `Vec<TTOp>` covering all three RISC-V kernel
//! sources, split at render time at the `EndReader`/`EndCompute`/`EndWriter`
//! markers.

use super::tenstorrent::CBId;
use crate::{
    DType,
    dtype::Constant,
    kernel::{BOp, Kernel, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind, TileDim, UOp},
    shape::Dim,
    types::{TinyString, TinyVec},
};

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
    /// Loop with trip count in a scalar register (materialized by a prior
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
        /// Global launch-arg ordinal (survives section materialization).
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

struct Compiler {
    ops: Vec<TTOp>,
}

impl Compiler {
    fn new(kernel: &Kernel) -> Self {
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
                Op::Store { dst, src, index, layout } => TTOp::SSAStore { dst, src, index, layout },
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
        Self { ops }
    }
}

impl Kernel {
    fn generate_tenstorrent2(kernel: &Kernel) {
        let mut compiler = Compiler::new(kernel);

        todo!()
    }
}
