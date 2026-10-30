// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent codegen — emission only.
//!
//! Lowering lives in [`crate::kernel::tenstorrent`]: the kernel TT
//! passes (`tt_storage`, `tt_lock_dst`, ...) rewrite the kernel IR in
//! place, and this file is a pure consumer. It walks `Op` + the kernel
//! [`TTOp`](crate::kernel::TTOp) and emits three RISC-V section sources
//! plus the launch tables the backend needs.
//!
//! Physical ids are derived at emission, never stored:
//! - CBs come from `Op::Storage { scope: Circular }` (first touch
//!   order, `CbDeclare` + runtime config per CB);
//! - DST slots come from `Op::Storage { scope: Register }` (slot
//!   reuse by liveness, `DstMode` from the dtype scan);
//! - param ordinals are the flat head order of `Op::Param`;
//! - grid coordinates are the group `Range` axes 0/1;
//! - NOC traffic comes from `Op::Copy` between a `Param` and a
//!   Circular storage (async read one way, async write the other).

use crate::DType;
use crate::Map;
use crate::Set;
use crate::backend::tenstorrent::{TTDmMove, TTDstRow, TTNativeProgram, TTSemaphoreInit, TT_NARROW_BF16, TT_NARROW_F16, TT_NARROW_NONE};
use crate::backend::{GwsDim, gws_from_kernel};
use crate::dtype::Constant;
use crate::error::{BackendError, ErrorStatus};
use crate::kernel::{
    BOp, GPUOp, Kernel, MemScope, Op, OpId, ParamKind, RangeKind, SourceBlock, TTCbConfig, TTOp, TTProgramDesc, TileDim, UOp,
};
use crate::scalar::{bf16, f16};
use crate::slab::Slab;
use tt_isa::isa::Instruction;
use crate::backend::tenstorrent::kern::sfpu::kernel::{A_ROW, B_ROW, OUT_ROW};
use crate::backend::tenstorrent::kern::sfpu::reduce::{Axis, ReduceOp, accumulate, finish};

/// DRAM page size in bytes.
pub(crate) const TT_DRAM_PAGE_BYTES: u32 = 4096;

/// TT `DataFormat` code for a dtype on the tile path (the
/// `typecast_tile_init<in, out>` template args). Distinct from the
/// CB descriptor codes in `generate_tenstorrent` below: the two
/// numberings differ.
/// zyx dtype -> the hardware's L1/`Src` data format code.
///
/// The codes are metalium's (`L1Format::code`), the ones its gates measured:
/// FP32 0, FP16 1, TF32 4, BF16 5, INT32 8. Not a zyx-wide table — the
/// hardware's.
fn tt_fmt(dtype: DType) -> Result<u32, BackendError> {
    use tt_isa::tile::L1Format;
    let fmt = match dtype {
        DType::F32 => L1Format::Fp32,
        DType::F16 => L1Format::Fp16,
        DType::BF16 => L1Format::Bf16,
        DType::I32 => L1Format::Int32,
        dt => {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("tenstorrent: dtype {dt:?} has no tile format").into(),
            });
        }
    };
    fmt.code().ok_or_else(|| BackendError {
        status: ErrorStatus::KernelCompilation,
        context: format!("tenstorrent: {fmt:?} has no measured format code").into(),
    })
}

/// Launch tables for the Tenstorrent backend, built from the kernel IR
/// by [`Kernel::generate_tenstorrent`].
pub struct TTProgram {
    /// Reader section source.
    pub reader_src: String,
    /// Compute section source (empty when the kernel is pure copy).
    pub compute_src: String,
    /// Writer section source.
    pub writer_src: String,
    /// Raw native compute instruction words (`Instruction::word()`), in
    /// emission order. The native backend stages these; the C++ source is
    /// only the not-yet-ported fallback.
    pub compute_native: Vec<u32>,
    /// The staged native program: DM boundary moves, role streams,
    /// semaphores, and the L1/DST placement baked into the instructions.
    pub native: TTNativeProgram,
    /// Global head-order ordinals of the reader section params.
    pub reader_params: Vec<u32>,
    /// Global head-order ordinals of the compute section params.
    pub compute_params: Vec<u32>,
    /// Global head-order ordinals of the writer section params.
    pub writer_params: Vec<u32>,
    /// Total param count (all kinds, global head order).
    pub n_params: u32,
    /// Global params in head order (kernel inputs).
    pub input_dtypes: Vec<DType>,
    /// GlobalMut params in head order (kernel outputs).
    pub output_dtypes: Vec<DType>,
    /// Runtime CB config: (tt format, tile bytes, tile count) per CB,
    /// indexed by CB number.
    pub cb_config: Vec<(u32, u32, u32)>,
    /// True iff the kernel touches F32 tiles (32-bit DST mode).
    pub fp32: bool,
}

impl Kernel {
    /// Tenstorrent render: lower to a descriptor-only kernel — head
    /// order `ProgramDesc`, `Params`, `TensixGrid`, then the three
    /// section sources separated by the existing `EndReader` /
    /// `EndCompute` markers. Lowering lives in `device_epilogue`
    /// (which verifies the lowered IR); `generate_tenstorrent`
    /// renders 1:1. Backend `compile` only decodes positions and
    /// drives the shim.
    /// The unlowered ops are consumed, never carried. Every kernel
    /// goes through here, including hand-built custom kernels (raw IR
    /// lowers exactly once; kernels already holding a `Source` pass
    /// through untouched in `render` and never reach this).
    pub(crate) fn render_tt(&self) -> Result<Kernel, BackendError> {
        let program = self.generate_tenstorrent()?;
        let gws_vec = gws_from_kernel(self, &self.dev_info().max_global_work_dims)?;
        if gws_vec.len() > 2 {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("tenstorrent render: grid rank {} exceeds 2D", gws_vec.len()).into(),
            });
        }
        let mut grid = [GwsDim::Const(1), GwsDim::Const(1)];
        for (i, g) in gws_vec.into_iter().enumerate() {
            grid[i] = g;
        }
        // Param head order is lowering-invariant: the TT passes only
        // append effect ops at the tail and never create, remove, or
        // replace Param ops — so the pre-lowering order matches the
        // ordinals `generate_tenstorrent` numbered post-lowering.
        let mut params = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Param { kind, .. } = self.ops[op_id].op {
                params.push(kind);
            }
            op_id = self.next_op(op_id);
        }
        let desc = TTProgramDesc {
            reader_params: program.reader_params.into_boxed_slice(),
            compute_params: program.compute_params.into_boxed_slice(),
            writer_params: program.writer_params.into_boxed_slice(),
            n_params: program.n_params,
            n_inputs: program.input_dtypes.len() as u32,
            n_outputs: program.output_dtypes.len() as u32,
            cb_config: program
                .cb_config
                .into_iter()
                .map(|(format, tile_bytes, num_tiles)| TTCbConfig { format, tile_bytes, num_tiles })
                .collect(),
            native_compute: program.compute_native.into_boxed_slice(),
            native: program.native,
            fp32: program.fp32,
        };
        let mut rendered =
            Kernel { ops: Slab::new(), head: OpId::NULL, tail: OpId::NULL, dev: self.dev, dev_info: self.dev_info.clone() };
        rendered.push_back(Op::TT(TTOp::ProgramDesc(Box::new(desc))));
        rendered.push_back(Op::GPU(Box::new(GPUOp::Params(params.into_boxed_slice()))));
        rendered.push_back(Op::TT(TTOp::TensixGrid(Box::new(grid))));
        rendered.push_back(Op::Source(SourceBlock(program.reader_src.into_boxed_str())));
        rendered.push_back(Op::TT(TTOp::EndReader));
        rendered.push_back(Op::Source(SourceBlock(program.compute_src.into_boxed_str())));
        rendered.push_back(Op::TT(TTOp::EndCompute));
        rendered.push_back(Op::Source(SourceBlock(program.writer_src.into_boxed_str())));
        Ok(rendered)
    }

    /// Full TTIR codegen: run the kernel TT passes, then emit three
    /// RISC-V section sources plus the launch tables. The passes run on
    /// a clone of this kernel (the caller's IR is untouched); emission
    /// derives CBs, DST slots, param ordinals, and grid axes from the
    /// resulting ops.
    pub(crate) fn generate_tenstorrent(&self) -> Result<TTProgram, BackendError> {
        // Render-only: lowering lives in `device_epilogue` (which
        // verifies the lowered IR); this consumes the result 1:1.
        // Never dumps IR here: every candidate compile would print.
        // The single ZYX_DEBUG=8 print lives at the runtime layer
        // (winner / seed, once before its compile).
        let k = self;

        // Param ordinals + input/output dtypes, flat head order.
        let mut param_ordinal_of: Map<OpId, u32> = Map::default();
        let mut next_param = 0u32;
        let mut input_dtypes: Vec<DType> = Vec::new();
        let mut output_dtypes: Vec<DType> = Vec::new();
        let mut scan = k.head;
        for _ in 0..50_000 {
            if scan.is_null() {
                break;
            }
            if let Op::Param { dtype, kind, .. } = k.at(scan) {
                param_ordinal_of.insert(scan, next_param);
                next_param += 1;
                match kind {
                    ParamKind::Global => input_dtypes.push(*dtype),
                    ParamKind::GlobalMut => output_dtypes.push(*dtype),
                    ParamKind::Variable => {}
                }
            }
            scan = k.next_op(scan);
        }
        if !scan.is_null() {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "tenstorrent: param scan did not finish in 50000 steps".into(),
            });
        }

        // CB allocation: first touch over loads/stores/copies of
        // Circular storages (chased through their index expressions), with the
        // legacy validity checks.
        let mut cbs: Map<OpId, u32> = Map::default();
        let mut cb_order: Vec<OpId> = Vec::new();
        let mut scan = k.head;
        for _ in 0..50_000 {
            if scan.is_null() {
                break;
            }
            let mut touched: [Option<OpId>; 2] = [None, None];
            match k.at(scan) {
                Op::Load { src, .. } | Op::Copy { src, .. } => touched[0] = Some(k.tt_storage_of(*src)),
                Op::Store { dst, .. } => touched[0] = Some(k.tt_storage_of(*dst)),
                // Lowered traffic: publish/drain endpoints are bare
                // storages (DRAM param or Circular CB), never wrapped
                // in Load/Store, so `tt_storage_of` cannot see them.
                Op::TT(TTOp::Copy { src, dst, .. }) => {
                    touched[0] = Some(*src);
                    touched[1] = Some(*dst);
                }
                _ => {}
            };
            for st in touched.into_iter().flatten() {
                if matches!(k.at(st), Op::Storage { scope: MemScope::Circular, .. }) && !cbs.contains_key(&st) {
                    cbs.insert(st, cb_order.len() as u32);
                    cb_order.push(st);
                }
            }
            scan = k.next_op(scan);
        }
        let num_circular_buffers = k.device_info().num_circular_buffers;
        if cb_order.len() > num_circular_buffers as usize {
            return Err(BackendError {
                status: ErrorStatus::TooManyCircularBuffers,
                context: format!(
                    "tenstorrent: kernel needs {} circular buffers, device holds {num_circular_buffers}",
                    cb_order.len()
                )
                .into(),
            });
        }

        // Render.
        let mut native = TTNativeProgram {
            reader: Vec::new(),
            writer: Vec::new(),
            compute: [Vec::new(), Vec::new(), Vec::new()],
            compute_loops: [Vec::new(), Vec::new(), Vec::new()],
            sems: Vec::new(),
            l1: tt_place_cbs(k, &cb_order)?,
            dst: Vec::new(),
        };
        let (reader_src, compute_src, writer_src, reader_params, compute_params, writer_params, fp32, compute_native) =
            render(k, &param_ordinal_of, &cbs, &mut native)?;

        // The DRAM boundary of the native program. Role streams, their
        // loops, and the semaphores are filled by `render` for the compute
        // section (roles run in order, so no semaphores yet).
        let (reader_moves, writer_moves) = tt_boundary_moves(k, &cbs, &native.l1, &param_ordinal_of)?;
        native.reader = reader_moves;
        native.writer = writer_moves;

        // CB config per CB, in index order.
        let mut cb_config: Vec<(u32, u32, u32)> = Vec::new();
        for st in &cb_order {
            let Op::Storage { dtype, len, .. } = k.at(*st) else {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent: cb entry {st} is not a storage op").into(),
                });
            };
            let (fmt, tb) = match *dtype {
                // Index and predicate storages are moved as raw bytes: they
                // are not tile data, so they declare the FP32 code and no
                // instruction ever reads that field for them.
                DType::I64 | DType::Bool => (0, (*len as u32) * 8 / 1024),
                dt => {
                    let bits = dt.bit_size() as u32;
                    (tt_fmt(dt)?, bits / 8 * 1024)
                }
            };
            cb_config.push((fmt, tb, (len / 1024) as u32));
            debug_assert_eq!(cb_config.len() - 1, cbs[st] as usize, "tenstorrent: CB config out of sync with allocation");
        }

        Ok(TTProgram {
            reader_src,
            compute_src,
            compute_native,
            writer_src,
            native,
            reader_params,
            compute_params,
            writer_params,
            n_params: next_param,
            input_dtypes,
            output_dtypes,
            cb_config,
            fp32,
        })
    }
}
fn tt_err(context: String) -> BackendError {
    BackendError { status: ErrorStatus::KernelCompilation, context: context.into() }
}

/// Tile bytes for a CB dtype (matches the `cb_config` table in
/// `generate_tenstorrent`).
fn tt_tile_bytes(dtype: DType) -> Result<u32, BackendError> {
    match dtype {
        DType::F32 | DType::U32 | DType::I32 => Ok(4096),
        DType::F16 | DType::BF16 | DType::U16 => Ok(2048),
        DType::F8E4M3 | DType::U8 | DType::I8 => Ok(1024),
        dt => Err(tt_err(format!("tenstorrent: tile dtype {dt:?} has no tile size"))),
    }
}

/// C++ literal for an index/const value.
fn tt_const_lit(c: &Constant) -> Result<String, BackendError> {
    match c {
        Constant::U8(v) => Ok(v.to_string()),
        Constant::U16(v) => Ok(v.to_string()),
        Constant::U32(v) => Ok(v.to_string()),
        Constant::U64(b) => Ok(u64::from_le_bytes(*b).to_string()),
        Constant::I8(v) => Ok(v.to_string()),
        Constant::I16(v) => Ok(v.to_string()),
        Constant::I32(v) => Ok(v.to_string()),
        Constant::I64(b) => Ok(i64::from_le_bytes(*b).to_string()),
        Constant::Bool(v) => Ok(v.to_string()),
        Constant::F32(b) => Ok(format!("{:?}f", f32::from_le_bytes(*b))),
        Constant::F64(b) => Ok(format!("{:?}", f64::from_le_bytes(*b))),
        Constant::F16(b) => Ok(format!("{:?}f", f16(u16::from_le_bytes(*b)).to_f32())),
        Constant::BF16(b) => Ok(format!("{:?}f", bf16(u16::from_le_bytes(*b)).to_f32())),
        Constant::F8E4M3(v) => Ok(format!("{:?}f", crate::scalar::f8e4m3(*v).to_f32())),
        Constant::F8E5M2(v) => Ok(format!("{:?}f", crate::scalar::f8e5m2(*v).to_f32())),
    }
}

/// C++ type name for an index cast target.
fn tt_ctype(dtype: DType) -> Result<&'static str, BackendError> {
    match dtype {
        DType::Bool => Ok("bool"),
        DType::U8 => Ok("uint8_t"),
        DType::U16 => Ok("uint16_t"),
        DType::U32 => Ok("uint32_t"),
        DType::U64 => Ok("uint64_t"),
        DType::I8 => Ok("int8_t"),
        DType::I16 => Ok("int16_t"),
        DType::I32 => Ok("int32_t"),
        DType::I64 => Ok("int64_t"),
        DType::F32 => Ok("float"),
        DType::F64 => Ok("double"),
        dt => Err(tt_err(format!("tenstorrent: index cast to {dt:?} has no C++ type"))),
    }
}

/// Rewrite the absolute DST rows of a metalium SFPU program. The math
/// programs are written against `A_ROW`/`B_ROW`/`OUT_ROW`; zyx gives them the
/// actual DST rows. Only `SFPLOAD`/`SFPSTORE` carry a row, in `Imm10`;
/// `Program::new` zeroes the DST RWC, so the row is absolute.
fn tt_row_remap(mut code: Vec<Instruction>, a_row: u32, b_row: Option<u32>, out_row: u32) -> Vec<Instruction> {
    for insn in &mut code {
        let def = insn.def();
        if def.mnemonic() != "SFPLOAD" && def.mnemonic() != "SFPSTORE" {
            continue;
        }
        let Some(field) = def.field("Imm10") else { continue };
        let addr = field.extract(insn.word());
        let mapped = if (A_ROW..A_ROW + 64).contains(&addr) {
            a_row + (addr - A_ROW)
        } else if (B_ROW..B_ROW + 64).contains(&addr) {
            b_row.unwrap_or(B_ROW) + (addr - B_ROW)
        } else if (OUT_ROW..OUT_ROW + 64).contains(&addr) {
            out_row + (addr - OUT_ROW)
        } else {
            addr
        };
        let word = (insn.word() & !field.mask()) | field.place(mapped);
        *insn = Instruction::new(word, def);
    }
    code
}

/// The native math program for one generic SFPU op: `a_row` (and `b_row`) in,
/// `slot_row` out, every other DST row untouched. The metalium opcode is the
/// `kind` the lowering stored; the arity must match the operands the zyx op
/// carries, otherwise the program would read a row that was never written.
fn tt_sfpu_program(kind: u32, a_row: u32, b_row: Option<u32>, slot_row: u32) -> Result<Vec<Instruction>, BackendError> {
    use crate::backend::tenstorrent::kern::sfpu::kernel::Operands;
    use crate::backend::tenstorrent::kern::sfpu::ops::{Broadcast, program_for};
    let Some((operands, code)) = program_for(kind, [0.0, 0.0], Broadcast::None) else {
        return Err(tt_err(format!("tenstorrent: SFPU kind {kind:#x} has no metalium program")));
    };
    match (operands, b_row) {
        (Operands::Unary, None) | (Operands::Binary, Some(_)) => {}
        (operands, b_row) => {
            return Err(tt_err(format!(
                "tenstorrent: SFPU kind {kind:#x} takes {operands:?} operands, the op carries {}",
                if b_row.is_some() { "two" } else { "one" }
            )));
        }
    }
    Ok(tt_row_remap(code, a_row, b_row, slot_row))
}

/// A bit-preserving DST -> DST copy of one tile (64 rows).
fn tt_dst_copy(src: u32, dst: u32) -> Vec<Instruction> {
    use crate::backend::tenstorrent::kern::sfpu::{Format, LReg, Program};
    let mut p = Program::new();
    for o in 0..64 {
        p.load(LReg::L0, Format::Int32, src + o);
        p.store(LReg::L0, Format::Int32, dst + o);
    }
    p.finish()
}

/// The native reduce program for one tile, reading `input_row` and writing
/// `slot_row`. `scratch` is needed for a whole-tile scalar reduce (two
/// passes) and for a sum over columns, whose in-order accumulator zeroes
/// its output before reading the input.
fn tt_reduce_program(
    rop: BOp,
    kind: TileDim,
    input_row: u32,
    slot_row: u32,
    scratch: Option<u32>,
) -> Result<Vec<Instruction>, BackendError> {
    let op = match rop {
        BOp::Add => ReduceOp::Sum,
        BOp::Max => ReduceOp::Max,
        _ => return Err(tt_err(format!("tenstorrent: reduce op {rop:?} has no native reduce program"))),
    };
    let mut code = Vec::new();
    match kind {
        TileDim::Row => {
            let mut c = accumulate(op, Axis::Cols, true, None);
            c.extend(finish(op, Axis::Cols));
            code.extend(tt_row_remap(c, input_row, None, slot_row));
        }
        TileDim::Col => {
            // `accumulate_in_order` (sum over rows) zeroes its output
            // before reading the input, so an in-place reduce would erase
            // the data. Always stage that one through the scratch row.
            if op == ReduceOp::Sum {
                let scratch =
                    scratch.ok_or_else(|| tt_err("tenstorrent: sum-over-columns reduce needs a DST scratch row".to_string()))?;
                let mut c = accumulate(op, Axis::Rows, true, None);
                c.extend(finish(op, Axis::Rows));
                code.extend(tt_row_remap(c, input_row, None, scratch));
                code.extend(tt_dst_copy(scratch, slot_row));
            } else {
                let mut c = accumulate(op, Axis::Rows, true, None);
                c.extend(finish(op, Axis::Rows));
                code.extend(tt_row_remap(c, input_row, None, slot_row));
            }
        }
        TileDim::Scalar => {
            let scratch = scratch.ok_or_else(|| tt_err("tenstorrent: scalar reduce needs a DST scratch row".to_string()))?;
            // First reduce columns into `slot`, then rows from `slot` into
            // the scratch, then move the scalar back into `slot`.
            let mut c = accumulate(op, Axis::Cols, true, None);
            c.extend(finish(op, Axis::Cols));
            code.extend(tt_row_remap(c, input_row, None, slot_row));
            let mut c = accumulate(op, Axis::Rows, true, None);
            c.extend(finish(op, Axis::Rows));
            code.extend(tt_row_remap(c, slot_row, None, scratch));
            code.extend(tt_dst_copy(scratch, slot_row));
        }
    }
    Ok(code)
}

/// Section-local emission state. Sections are separate programs: SSA
/// values, loop counters, grid reads, and DST slots never cross a
/// section boundary.
struct TtSection<'a> {
    k: &'a Kernel,
    cbs: &'a Map<OpId, u32>,
    ordinals: &'a Map<OpId, u32>,
    users: &'a Map<OpId, Vec<OpId>>,
    section: u8,
    params: Vec<OpId>,
    loop_names: Map<OpId, String>,
    range_names: Map<OpId, String>,
    var_names: Map<OpId, String>,
    slots: Map<OpId, u32>,
    next_slot: u32,
    budget: u32,
    transfer: u32,
    indent: String,
    math_open: bool,
    pack_open: bool,
    out: String,
    /// Raw native compute instruction words, in emission order. The C++
    /// source is still emitted for the not-yet-ported ops; native ops
    /// append here instead of calling an LLK.
    native: Vec<u32>,
    /// L1 byte address per CB index: what the unpack/pack programs bake in.
    l1: &'a [u32],
    /// Native T0 (unpack) words: one CB -> DST unpack per lowered `Copy`.
    native_t0: Vec<u32>,
    /// Native T2 (pack) words: one DST -> CB pack per lowered `Store`.
    native_t2: Vec<u32>,
    /// Which role stream the last emitted op went to (0 = T0, 1 = T1,
    /// 2 = T2): a semaphore op joins the stream of the instruction before it.
    last_role: u8,
    /// Semaphore init list from `Semset` ops (the run's concurrent schedule).
    sems: Vec<TTSemaphoreInit>,
    /// Compute value ops emitted so far: `ZEROACC` mode 3 (`CLR_ALL`) is only
    /// safe before any of them, or it would wipe live DST rows.
    compute_ops: u32,
    /// Per unpacker (0 = SrcA, 1 = SrcB): its one-time Src configuration has
    /// been emitted, so later unpacks only retarget the base address.
    src_unpacker_ready: [bool; 2],
    /// The T0 body unpacks into `SrcA`/`SrcB` (matmul path), so its thread
    /// config must be `src_thread_config` (`SRCA_SET_SetOvrdWithAddr = 1`),
    /// not the `Dst` path's `thread_config`.
    has_src_unpack: bool,
}

impl<'a> TtSection<'a> {
    fn err(&self, context: String) -> BackendError {
        tt_err(format!("tenstorrent section {}: {context}", self.section))
    }

    /// Inline C++ for a scalar/index expression: consts inline, loop
    /// counters / grid coords / Variable args by their section names,
    /// arithmetic parenthesized. Tile values are not scalar
    /// expressions — loud error.
    fn expr(&self, id: OpId) -> Result<String, BackendError> {
        match self.k.at(id) {
            Op::Const(c) => tt_const_lit(c),
            Op::Loop { .. } => self.loop_names.get(&id).cloned().ok_or_else(|| self.err(format!("{id:?} is no loop counter"))),
            Op::Range { axis, kind, .. } => match kind {
                RangeKind::Group(_) => self
                    .range_names
                    .get(&id)
                    .cloned()
                    .ok_or_else(|| self.err(format!("range {id:?} (axis {axis}) has no grid read"))),
                _ => Err(self.err(format!("range {id:?} is not a group range"))),
            },
            Op::Param { kind, .. } => match kind {
                ParamKind::Variable => {
                    self.var_names.get(&id).cloned().ok_or_else(|| self.err(format!("variable {id:?} has no arg read")))
                }
                _ => Err(self.err(format!("DRAM param {id:?} in an index expression"))),
            },
            Op::Binary { x, y, bop, .. } => {
                let op = match bop {
                    BOp::Add => "+",
                    BOp::Sub => "-",
                    BOp::Mul => "*",
                    BOp::Div => "/",
                    BOp::Mod => "%",
                    BOp::Cmplt => "<",
                    BOp::Cmpgt => ">",
                    BOp::Cmpge => ">=",
                    BOp::Eq => "==",
                    BOp::NotEq => "!=",
                    BOp::And => "&&",
                    BOp::Or => "||",
                    BOp::BitAnd => "&",
                    BOp::BitOr => "|",
                    BOp::BitXor => "^",
                    BOp::BitShiftLeft => "<<",
                    BOp::BitShiftRight => ">>",
                    _ => return Err(self.err(format!("binary {bop:?} has no C++ operator"))),
                };
                Ok(format!("({}{op}{})", self.expr(*x)?, self.expr(*y)?))
            }
            Op::Cast { x, dtype, .. } => Ok(format!("(({}){})", tt_ctype(*dtype)?, self.expr(*x)?)),
            _ => Err(self.err(format!("{id:?} is not a scalar expression"))),
        }
    }

    /// Fresh DST slot, budget-checked (old slot-budget error).
    fn fresh_slot(&mut self) -> Result<u32, BackendError> {
        if self.next_slot >= self.budget {
            return Err(self.err(format!("tile t{} exceeds the DST budget {}", self.next_slot, self.budget)));
        }
        let s = self.next_slot;
        self.next_slot += 1;
        Ok(s)
    }

    /// Slot of a Register storage, assigned on first need.
    fn reg_slot(&mut self, storage: OpId) -> Result<u32, BackendError> {
        if let Some(s) = self.slots.get(&storage) {
            return Ok(*s);
        }
        let s = self.fresh_slot()?;
        self.slots.insert(storage, s);
        Ok(s)
    }

    /// Slot holding a tile operand: an SSA tile already lowered, a
    /// circular load (copied in fresh), or a Register slot. Anything
    /// else is malformed tile traffic — loud error (old "not a CB or
    /// live tile"), never a silent zero.
    fn operand_slot(&mut self, v: OpId) -> Result<u32, BackendError> {
        if let Some(s) = self.slots.get(&v) {
            return Ok(*s);
        }
        if let Op::Load { src, .. } = self.k.at(v) {
            match self.k.at(*src) {
                Op::Storage { scope: MemScope::Circular, .. } => {
                    let cb = self.cbs.get(src).copied().ok_or_else(|| self.err(format!("CB {src:?} has no number")))?;
                    let s = self.fresh_slot()?;
                    self.out.push_str(&format!("{}copy_tile({cb}, 0, {s});\n", self.indent));
                    self.slots.insert(v, s);
                    return Ok(s);
                }
                Op::Storage { scope: MemScope::Register, .. } => return self.reg_slot(*src),
                _ => {}
            }
        }
        Err(self.err(format!("{v:?} is not a CB or live tile")))
    }

    /// Slot already holding a tile value: an SSA tile lowered
    /// earlier, a Register slot, or a circular load unpacked at its
    /// own position (all-pack consumers, see `render_load`). No
    /// silent copies here — a miss is malformed pack traffic, loud,
    /// never a zero. SSA compute uses (`render_tile_op`) take
    /// `operand_slot`, which copies circular loads in at the use.
    fn slot_of(&mut self, v: OpId) -> Result<u32, BackendError> {
        if let Some(s) = self.slots.get(&v) {
            return Ok(*s);
        }
        if let Op::Load { src, .. } = self.k.at(v) {
            if matches!(self.k.at(*src), Op::Storage { scope: MemScope::Register, .. }) {
                return self.reg_slot(*src);
            }
        }
        Err(self.err(format!("{v:?} has no tile slot")))
    }

    /// CB number of a Circular storage (sync ops carry storages).
    fn cb_num(&self, cb: OpId) -> Result<u32, BackendError> {
        self.cbs.get(&cb).copied().ok_or_else(|| self.err(format!("CB {cb:?} has no number")))
    }

    /// L1 byte address of a Circular storage (its place in the data arena).
    fn cb_l1(&self, cb: OpId) -> Result<u64, BackendError> {
        let num = self.cb_num(cb)?;
        let addr = self.l1.get(num as usize).copied().ok_or_else(|| self.err(format!("CB {cb:?} has no L1 address")))?;
        Ok(addr as u64)
    }

    /// Append one word to the role stream of the last emitted instruction.
    /// Semaphore ops carry no role of their own: a `Post` after an unpack is
    /// the unpack role's, one after a math program the math role's.
    fn push_role_word(&mut self, word: u32) {
        match self.last_role {
            0 => self.native_t0.push(word),
            1 => self.native.push(word),
            2 => self.native_t2.push(word),
            other => unreachable!("role {other} is not 0..2"),
        }
    }

    /// Push a word into a named role stream (a semaphore op names the role it
    /// belongs to instead of inheriting the previous op's).
    fn push_role_word_for(&mut self, role: u8, word: u32) {
        match role {
            0 => self.native_t0.push(word),
            1 => self.native.push(word),
            2 => self.native_t2.push(word),
            other => unreachable!("role {other} is not 0..2"),
        }
    }

    /// Runtime arg index of a section param (position in the section
    /// param list — the backend passes exactly these in order).
    fn arg_of(&self, param: OpId) -> Result<usize, BackendError> {
        self.params.iter().position(|p| *p == param).ok_or_else(|| self.err(format!("param {param:?} is not a section param")))
    }

    /// Global head-order ordinal of a param (accessor naming).
    fn ord_of(&self, param: OpId) -> Result<u32, BackendError> {
        self.ordinals.get(&param).copied().ok_or_else(|| self.err(format!("param {param:?} has no ordinal")))
    }

    /// Substitute `{i}` placeholders: Circular storages become CB
    /// numbers, Register storages become DST slots (assigned on
    /// demand), circular loads chase to their CB, live SSA tiles to
    /// their slots. Scalars and NULLs are unrenderable here (the old
    /// render errored on scalar asm the same way).
    fn substitute(&mut self, template: &str, ops: &[OpId]) -> Result<String, BackendError> {
        let mut text = template.to_string();
        for (i, o) in ops.iter().copied().enumerate() {
            if o.is_null() {
                continue;
            }
            let num = match self.k.at(o) {
                Op::Storage { scope: MemScope::Circular, .. } => {
                    self.cbs.get(&o).copied().ok_or_else(|| self.err(format!("CB {o:?} has no number")))?
                }
                Op::Storage { scope: MemScope::Register, .. } => self.reg_slot(o)?,
                Op::Load { .. } => {
                    let chained = self.k.at(o);
                    if let Op::Load { src, .. } = chained {
                        if matches!(self.k.at(*src), Op::Storage { scope: MemScope::Circular, .. }) {
                            self.cbs.get(src).copied().ok_or_else(|| self.err(format!("CB {src:?} has no number")))?
                        } else if matches!(self.k.at(*src), Op::Storage { scope: MemScope::Register, .. }) {
                            self.reg_slot(*src)?
                        } else {
                            return Err(self.err(format!("asm operand {o:?} is not a CB or live tile")));
                        }
                    } else {
                        return Err(self.err(format!("asm operand {o:?} is not a CB or live tile")));
                    }
                }
                _ => {
                    self.slots.get(&o).copied().ok_or_else(|| self.err(format!("asm operand {o:?} is not a CB or live tile")))?
                }
            };
            text = text.replace(&format!("{{{i}}}"), &num.to_string());
        }
        // A surviving placeholder names an operand render never
        // resolved (NULL dst or scalar) — loud, never silently kept.
        if text.contains('{') {
            return Err(self.err(format!("template left placeholders unsubstituted: {text}")));
        }
        Ok(text)
    }
}

/// Render one section body: loops, traffic, sync, locks, compute.
/// Locks are discipline-checked (compute-only, paired, closed at
/// section end — the old verify lock block).
fn render_section(sec: &mut TtSection, ops: &[OpId]) -> Result<(), BackendError> {
    let mut loop_stack: Vec<OpId> = Vec::new();
    for id in ops.iter().copied() {
        match sec.k.at(id) {
            Op::Const { .. } | Op::Param { .. } | Op::Storage { .. } | Op::Stack { .. } => {}
            // Unrolled lanes emit no code: traffic addresses are
            // lane-stripped bases at lowering, and tile-shaped pack
            // indices are shape-only metadata. The defs stay alive
            // only as long as something references them.
            Op::Unroll { .. } => {}
            Op::Range { kind, .. } => match kind {
                RangeKind::Group(_) => {}
                _ => return Err(sec.err(format!("range {id:?} is not a group range"))),
            },
            Op::Loop { len } => {
                if sec.k.dtype(*len) == DType::Bool {
                    sec.out.push_str(&format!("{}if ({}) {{\n", sec.indent, sec.expr(*len)?));
                } else {
                    let bound = sec.expr(*len)?;
                    let name = sec.loop_names.get(&id).cloned().ok_or_else(|| sec.err(format!("{id:?} has no counter")))?;
                    sec.out.push_str(&format!("{}for (uint32_t {name} = 0; {name} < {bound}; {name}++) {{\n", sec.indent));
                }
                sec.indent += "  ";
                loop_stack.push(id);
            }
            Op::EndLoop => {
                if loop_stack.pop().is_none() {
                    return Err(sec.err("EndLoop without Loop".to_string()));
                }
                sec.out.push_str(&format!("{}}}\n", sec.indent));
                sec.indent.pop();
                sec.indent.pop();
            }
            Op::Copy { src, src_idx, dst, dst_idx, .. } => render_copy(sec, id, *src, *src_idx, *dst, *dst_idx)?,
            Op::TT(TTOp::Copy { src, src_idx, dst, dst_idx, bytes }) => {
                // Staged traffic: both indices are lane-free bases
                // (unrolls stripped at lowering), size is explicit.
                // Publish (DRAM→CB) lives in the reader, drain
                // (CB→DRAM) in the writer.
                let src_cb = if matches!(sec.k.at(*src), Op::Storage { scope: MemScope::Circular, .. }) {
                    Some(*src)
                } else {
                    None
                };
                let dst_cb = if matches!(sec.k.at(*dst), Op::Storage { scope: MemScope::Circular, .. }) {
                    Some(*dst)
                } else {
                    None
                };
                let dst_reg = matches!(sec.k.at(*dst), Op::Storage { scope: MemScope::Register, .. });
                match (src_cb, dst_cb, dst_reg) {
                    (None, Some(dcb), false) => {
                        if sec.section != 0 {
                            return Err(sec.err(format!("publish {id:?} outside the reader")));
                        }
                        let dcb = sec.cb_num(dcb)?;
                        let (param, base) = match sec.k.at(*src) {
                            Op::Param { .. } => (*src, *src_idx),
                            _ => return Err(sec.err(format!("publish {id:?} src is not a DRAM param"))),
                        };
                        let ord = sec.ord_of(param)?;
                        let idx = sec.expr(base)?;
                        let eb = elem_bytes(sec.k, param)?;
                        let slot = cb_slot_expr(sec, *dst_idx, eb)?;
                        let noc = sec.transfer;
                        sec.transfer += 1;
                        sec.out.push_str(&format!(
                            "{}uint64_t rnoc{noc} = p{ord}.get_noc_addr((uint32_t)((( {idx} )*{eb})/{page}), (uint32_t)((( {idx} )*{eb})%{page}));\n",
                            sec.indent,
                            page = TT_DRAM_PAGE_BYTES
                        ));
                        sec.out.push_str(&format!(
                            "{}noc_async_read(rnoc{noc}, cb{dcb}.get_write_ptr(){slot}, {bytes});\n",
                            sec.indent
                        ));
                        sec.out.push_str(&format!("{}noc_async_read_barrier();\n", sec.indent));
                        Ok(())
                    }
                    (Some(scb), None, true) => {
                        // CB -> DST unpack. The compute role consumes the
                        // slot, so this lives in the compute section.
                        if sec.section != 1 {
                            return Err(sec.err(format!("unpack {id:?} outside the compute section")));
                        }
                        let cb = sec.cb_num(scb)?;
                        let dst = match sec.k.at(*dst) {
                            Op::Storage { scope: MemScope::Register, dtype, .. } => {
                                if *dtype != DType::F32 {
                                    return Err(sec.err(format!(
                                        "native unpack {id:?} needs an F32 DST tile, got {dtype:?} (BF16/F16 staging is not implemented)"
                                    )));
                                }
                                sec.reg_slot(*dst)?
                            }
                            _ => return Err(sec.err(format!("unpack {id:?} dst is not a Register storage"))),
                        };
                        let l1 = sec.cb_l1(scb)?;
                        let words = crate::backend::tenstorrent::kern::datapath::unpack_tile_to_dst(l1, dst);
                        sec.native_t0.extend(words.iter().map(|i| i.word()));
                        sec.last_role = 0;
                        sec.compute_ops += 1;
                        sec.out.push_str(&format!("{}copy_tile({cb}, 0, {dst});\n", sec.indent));
                        Ok(())
                    }
                    (Some(scb), None, false) => {
                        if sec.section != 2 {
                            return Err(sec.err(format!("drain {id:?} outside the writer")));
                        }
                        let scb = sec.cb_num(scb)?;
                        let (param, base) = match sec.k.at(*dst) {
                            Op::Param { .. } => (*dst, *dst_idx),
                            _ => return Err(sec.err(format!("drain {id:?} dst is not a DRAM param"))),
                        };
                        let ord = sec.ord_of(param)?;
                        let idx = sec.expr(base)?;
                        let eb = elem_bytes(sec.k, param)?;
                        let slot = cb_slot_expr(sec, *src_idx, eb)?;
                        let noc = sec.transfer;
                        sec.transfer += 1;
                        sec.out.push_str(&format!(
                            "{}uint64_t wnoc{noc} = p_out{ord}.get_noc_addr((uint32_t)((( {idx} )*{eb})/{page}), (uint32_t)((( {idx} )*{eb})%{page}));\n",
                            sec.indent,
                            page = TT_DRAM_PAGE_BYTES
                        ));
                        sec.out.push_str(&format!(
                            "{}noc_async_write(cb{scb}.get_read_ptr(){slot}, wnoc{noc}, {bytes});\n",
                            sec.indent
                        ));
                        sec.out.push_str(&format!("{}noc_async_write_barrier();\n", sec.indent));
                        Ok(())
                    }
                    _ => Err(sec.err(format!("staged copy {id:?} is not DRAM↔CB traffic"))),
                }?;
            }
            Op::Load { src, .. } => render_load(sec, id, *src)?,
            Op::Store { src: x, dst, .. } => render_store(sec, id, *x, *dst)?,
            Op::Unary { .. } | Op::Binary { .. } | Op::Cast { .. } | Op::Bitcast { .. } => {
                // Index math is IDX_T-typed: address arithmetic, never
                // tile compute. Only float 2D values render as LLKs.
                if sec.k.shape(id).len() == 2 && sec.k.dtype(id).is_float() {
                    render_tile_op(sec, id)?;
                }
            }
            Op::Asm { asm, ops } => {
                let ops_vec: Vec<OpId> = ops.iter().copied().collect();
                let text = sec.substitute(asm.as_str(), &ops_vec)?;
                sec.out.push_str(&format!("{}{text}\n", sec.indent));
            }
            Op::TT(TTOp::ReduceTile { rop, kind, input, slot, .. }) => {
                // Native reduce: the metalium reduce math program reads the
                // input DST row and writes the accumulator DST row. There is
                // no LLK `reduce_tile` and no CB operand here.
                let input_row = sec.reg_slot(*input)?;
                let slot_row = sec.reg_slot(*slot)?;
                sec.slots.insert(id, slot_row);
                let needs_scratch = *kind == TileDim::Scalar || (*kind == TileDim::Col && *rop == BOp::Add);
                let scratch = if needs_scratch { Some(sec.fresh_slot()?) } else { None };
                let code = tt_reduce_program(*rop, *kind, input_row, slot_row, scratch)?;
                for insn in code {
                    sec.native.push(insn.word());
                }
                sec.compute_ops += 1;
            }
            Op::TT(TTOp::Unpacr { src, unit, count, flip }) => {
                use tt_isa::backend::{Before, cond, stallwait, wait_for_unpacker0, wait_for_unpacker1};
                use tt_isa::cfg::generated::{alu, thcon};
                use crate::backend::tenstorrent::kern::datapath::{Unpacker, config_program, set_adc_x, tile_base_units};
                if *count != 1024 {
                    return Err(sec.err(format!("unpacr {id:?} of {count} datums: only whole tiles are wired up")));
                }
                let unpacker = tt_unpacker(*unit).map_err(|e| sec.err(format!("unpacr {id:?}: {e}")))?;
                let l1 = tt_src_base(sec.cb_l1(*src)?);
                let mut words = tt_isa::backend::ConfigWords::new();
                let fresh = !sec.src_unpacker_ready[*unit as usize];
                if fresh {
                    // One-time Src configuration: metalium's public
                    // `datapath::unpack_src_config` (base, format, descriptor,
                    // DST_BASE, zero strides), with the accumulator set to
                    // 32-bit (`ALU_ACC_CTRL`). The route follows the tile
                    // format: FP32 in L1 is truncated to TF32 by the unpacker,
                    // BF16 in L1 moves as is (`SrcRoute`).
                    let src_dtype = sec.k.dtype(*src);
                    let route = match src_dtype {
                        DType::F32 => crate::backend::tenstorrent::kern::matmul::SrcRoute::Tf32FromFp32,
                        DType::BF16 => crate::backend::tenstorrent::kern::matmul::SrcRoute::Bf16FromBf16,
                        dt => {
                            return Err(sec.err(format!(
                                "unpacr {id:?}: no Src route from {dt:?} (F32 and BF16 are wired up)"
                            )));
                        }
                    };
                    let (in_fmt, out_fmt) = route.formats();
                    crate::backend::tenstorrent::kern::datapath::unpack_src_config(
                        &mut words,
                        unpacker,
                        crate::backend::tenstorrent::kern::matmul::tile_descriptor(in_fmt),
                        l1,
                        out_fmt,
                    );
                    words
                        .set(alu::ALU_ACC_CTRL_Fp32_enabled, 1)
                        .map_err(|e| sec.err(format!("unpacr {id:?}: fp32 acc: {e:?}")))?;
                    words
                        .set(alu::ALU_ACC_CTRL_Zero_Flag_disabled_src, 0)
                        .map_err(|e| sec.err(format!("unpacr {id:?}: zero flag: {e:?}")))?;
                } else {
                    // Steady state: only the tile's base address moves.
                    let field = match unpacker {
                        Unpacker::SrcA => thcon::THCON_SEC0_REG3_Base_address,
                        Unpacker::SrcB => thcon::THCON_SEC1_REG3_Base_address,
                    };
                    words
                        .set(field, tile_base_units(l1))
                        .map_err(|e| sec.err(format!("unpacr {id:?}: base address: {e:?}")))?;
                }
                // `matmul_items` issues no wait around a Src unpack: the
                // prelude's `WRCFG` and the `UNPACR` come from the same
                // thread, and the unpacker's config block is not in flight.
                // (The Dst path's `wait_for_unpackerX(CONFIG)` +
                // `stallwait(CONFIG_BUSY)` pair is for `UnpackToDest`, where
                // the config is rewritten between whole-tile copies.)
                let mut code = config_program(&words);
                if fresh {
                    // `matmul_items`: the X range is the whole face, set once.
                    code.push(set_adc_x(unpacker, 0, crate::backend::tenstorrent::kern::matmul::FACE_DATUMS - 1));
                    sec.src_unpacker_ready[*unit as usize] = true;
                }
                code.push(crate::backend::tenstorrent::kern::datapath::unpack_src_instruction(unpacker, *flip));
                sec.native_t0.extend(code.iter().map(|i| i.word()));
                sec.last_role = 0;
                sec.has_src_unpack = true;
            }
            Op::TT(TTOp::Setadczw { unit, face }) => {
                use tt_isa::isa::generated::encode;
                let z = encode::Setadczw::ZERO.z0(1).z0_val(*face);
                let insn = match tt_unpacker(*unit).map_err(|e| sec.err(format!("setadczw {id:?}: {e}")))? {
                    crate::backend::tenstorrent::kern::datapath::Unpacker::SrcA => z.u0(1),
                    crate::backend::tenstorrent::kern::datapath::Unpacker::SrcB => z.u1(1),
                }
                .encode()
                .map_err(|e| sec.err(format!("setadczw {id:?}: {e:?}")))?;
                sec.native_t0.push(insn.word());
                sec.last_role = 0;
            }
            Op::TT(TTOp::Setadcx { unit, first, count }) => {
                if *count == 0 {
                    return Err(sec.err(format!("setadcx {id:?}: empty datum range")));
                }
                let insn = crate::backend::tenstorrent::kern::datapath::set_adc_x(
                    tt_unpacker(*unit).map_err(|e| sec.err(format!("setadcx {id:?}: {e}")))?,
                    *first,
                    first + count - 1,
                );
                sec.native_t0.push(insn.word());
                sec.last_role = 0;
            }
            Op::TT(TTOp::Setrwc { dst }) => {
                use tt_isa::isa::generated::encode;
                let row = match sec.k.at(*dst) {
                    Op::Storage { scope: MemScope::Register, .. } => sec.reg_slot(*dst)?,
                    _ => return Err(sec.err(format!("setrwc {id:?} dst {dst:?} is not a Register storage"))),
                };
                let insn = encode::Setrwc::ZERO
                    .src_a(1)
                    .src_a_val(crate::backend::tenstorrent::kern::matmul::SRC_A_ROW as u32)
                    .src_b(1)
                    .src_b_val(crate::backend::tenstorrent::kern::matmul::SRC_B_ROW as u32)
                    .dst(1)
                    .dst_val(row)
                    .fidelity(1)
                    .encode()
                    .map_err(|e| sec.err(format!("setrwc {id:?}: {e:?}")))?;
                sec.native.push(insn.word());
                sec.last_role = 1;
            }
            Op::TT(TTOp::Zeroacc { mode, imm10 }) => {
                if *mode == 3 && sec.compute_ops > 0 {
                    // `CLR_ALL` wipes every DST row, including live SSA
                    // values from earlier compute ops in a fused kernel.
                    return Err(sec.err(format!(
                        "zeroacc {id:?}: CLR_ALL after {} compute op(s) would destroy live DST rows",
                        sec.compute_ops
                    )));
                }
                let insn = tt_isa::isa::generated::encode::zeroacc(*mode as u32, 0, 0, *imm10)
                    .map_err(|e| sec.err(format!("zeroacc {id:?}: {e:?}")))?;
                sec.native.push(insn.word());
                sec.last_role = 1;
            }
            Op::TT(TTOp::Mvmul { dst_row, addr_mod, flip_a, flip_b, broadcast_b }) => {
                let insn = tt_isa::isa::generated::encode::Mvmul::ZERO
                    .dst_row(*dst_row)
                    .addr_mod(*addr_mod as u32)
                    .flip_src_a(u32::from(*flip_a))
                    .flip_src_b(u32::from(*flip_b))
                    .broadcast_src_b_row(u32::from(*broadcast_b))
                    .encode()
                    .map_err(|e| sec.err(format!("mvmul {id:?}: {e:?}")))?;
                sec.native.push(insn.word());
                sec.last_role = 1;
                sec.compute_ops += 1;
            }
            Op::TT(TTOp::Setc16 { field, value }) => {
                // A whole `ThreadConfig` word: several of the matmul's address
                // modifiers pack three fields into one entry, so the caller
                // composes the word and this writes it as given (`0` leaves
                // every other field of the entry clear). `set_thread_entry`
                // rejects an address outside the thread-config space.
                let addr = u16::try_from(*field).map_err(|_| sec.err(format!("setc16 {id:?}: {field:#x} is not a ThreadConfig word")))?;
                let insn = tt_isa::backend::set_thread_entry(addr, *value).map_err(|e| sec.err(format!("setc16 {id:?}: {e:?}")))?;
                sec.native.push(insn.word());
                sec.last_role = 1;
            }
            Op::TT(TTOp::Setcfg { addr, value }) => {
                // A whole `Config` word: `seed` leaves every bit as given, so
                // the caller owns the word exactly (no field masks involved).
                let mut words = tt_isa::backend::ConfigWords::new();
                words
                    .seed(u16::try_from(*addr).map_err(|_| sec.err(format!("setcfg {id:?}: {addr:#x} is not a Config word")))?, *value)
                    .map_err(|e| sec.err(format!("setcfg {id:?}: {e:?}")))?;
                let program = crate::backend::tenstorrent::kern::datapath::config_program(&words);
                sec.native.extend(program.iter().map(|i| i.word()));
                sec.last_role = 1;
            }
            Op::TT(TTOp::Post { sem, role }) => {
                let sem = tt_sem(*sem).map_err(|e| sec.err(format!("post {id:?}: {e}")))?;
                // Drain the posting role's unit first, then post
                // (`sync::post_after`): the consumer must not see the post
                // before the work it stands for is done.
                let unit = tt_unit(*role).ok_or_else(|| sec.err(format!("post {id:?}: no stream {role}")))?;
                for insn in tt_isa::sync::post_after(unit, sem) {
                    sec.push_role_word_for(*role, insn.word());
                }
            }
            Op::TT(TTOp::Wait { sem, before, role }) => {
                let sem = tt_sem(*sem).map_err(|e| sec.err(format!("wait {id:?}: {e}")))?;
                let mask = tt_before(*before).ok_or_else(|| sec.err(format!("wait {id:?}: unknown before mask {before}")))?;
                if tt_unit(*role).is_none() {
                    return Err(sec.err(format!("wait {id:?}: no stream {role}")));
                }
                // A take: wait for the post, then zero the semaphore, so the
                // next round of the same hand-off starts from zero.
                for insn in tt_isa::sync::take(sem, mask) {
                    sec.push_role_word_for(*role, insn.word());
                }
            }
            Op::TT(TTOp::Semset { sem, value }) => {
                // Not an instruction: the run's semaphore init list.
                let init = TTSemaphoreInit { index: *sem, value: *value, max: tt_isa::sync::MAX_VALUE };
                if sec.sems.iter().any(|s| s.index == init.index) {
                    return Err(sec.err(format!("semset {id:?}: semaphore {} initialised twice", init.index)));
                }
                sec.sems.push(init);
            }
            Op::TT(TTOp::Sfpu { kind, a, b, slot }) => {
                // Generic SFPU op: one metalium math program on the actual
                // DST rows. Unpacking is the `Copy` before it; there is no
                // CB operand and no pack inside the op.
                let a_row = match sec.k.at(*a) {
                    Op::Storage { scope: MemScope::Register, .. } => sec.reg_slot(*a)?,
                    _ => return Err(sec.err(format!("Sfpu {id:?} first operand {a:?} is not a Register storage"))),
                };
                let b_row = if b.is_null() {
                    None
                } else {
                    match sec.k.at(*b) {
                        Op::Storage { scope: MemScope::Register, .. } => Some(sec.reg_slot(*b)?),
                        _ => return Err(sec.err(format!("Sfpu {id:?} second operand {b:?} is not a Register storage"))),
                    }
                };
                let slot_row = sec.reg_slot(*slot)?;
                sec.slots.insert(id, slot_row);
                let code = tt_sfpu_program(*kind, a_row, b_row, slot_row)?;
                for insn in code {
                    sec.native.push(insn.word());
                }
                sec.compute_ops += 1;
            }
            Op::TT(TTOp::EndReader) | Op::TT(TTOp::EndCompute) => {}
            Op::Barrier => return Err(sec.err("stray barrier in a TT kernel".to_string())),
            _ => return Err(sec.err(format!("{id:?} is not fully lowered (SSA remains)"))),
        }
    }
    if !loop_stack.is_empty() {
        return Err(sec.err("section ends inside a loop".to_string()));
    }
    if sec.math_open || sec.pack_open {
        return Err(sec.err("section ends with DST locked".to_string()));
    }
    Ok(())
}

/// Place every Circular storage in the L1 data arena, first touch order,
/// 64-byte aligned, `len` bytes each (the storage already counts bytes).
/// Returns one address per CB index, in the same order as `cb_order`.
fn tt_place_cbs(k: &Kernel, cb_order: &[OpId]) -> Result<Vec<u32>, BackendError> {
    let mut at = tt_isa::l1::DATA.base;
    let mut placed: Vec<u32> = Vec::with_capacity(cb_order.len());
    for st in cb_order {
        let Op::Storage { dtype, len, .. } = k.at(*st) else {
            return Err(tt_err(format!("tenstorrent: cb entry {st} is not a storage op")));
        };
        // `len` is a datum count, so a 16-bit CB is half the bytes of an FP32
        // one. Sizing by `len` alone overlaps the tiles of consecutive CBs.
        let el = u64::from(dtype.bit_size() as u32 / 8);
        let len = (*len as u64).checked_mul(el).ok_or_else(|| tt_err(format!("tenstorrent: CB {st} size overflows")))?;
        let base = (at + 63) & !63;
        let end = base + len;
        if end > tt_isa::l1::DATA.end {
            return Err(tt_err(format!(
                "tenstorrent: L1 data arena exhausted: CB {st} needs {len} bytes at {base:#x}, arena ends at {:#x}",
                tt_isa::l1::DATA.end
            )));
        }
        let addr = u32::try_from(base).map_err(|_| tt_err(format!("tenstorrent: L1 address {base:#x} does not fit u32")))?;
        placed.push(addr);
        at = end;
    }
    Ok(placed)
}

/// The DRAM boundary of the native program: one move per reader publish
/// (param -> CB) and per writer drain (CB -> param), in program order. The
/// DRAM side is symbolic (param ordinal); the launch resolves
/// channel/port/offset from the chunk's `DramRange` because `Pool::TT`
/// placements are only known then.
/// The write-back conversion a writer move needs: the L1 tile is FP32, the
/// DRAM param is the kernel's declared dtype. Only 16-bit floats differ in
/// width; everything else moves the same bit pattern.
fn tt_narrow(cb: DType, param: DType) -> u8 {
    match (cb, param) {
        (DType::F32, DType::BF16) => TT_NARROW_BF16,
        (DType::F32, DType::F16) => TT_NARROW_F16,
        // Same-width (or non-float) pairs: the FP32 tile's bytes are the
        // param's bytes.
        (DType::F32, DType::F32 | DType::I32 | DType::U32) => TT_NARROW_NONE,
        (cb, param) => {
            debug_assert_eq!(cb, param, "tenstorrent: writer tile {cb:?} cannot land in a {param:?} param");
            TT_NARROW_NONE
        }
    }
}

fn tt_boundary_moves(
    k: &Kernel,
    cbs: &Map<OpId, u32>,
    l1: &[u32],
    param_ordinal_of: &Map<OpId, u32>,
) -> Result<(Vec<TTDmMove>, Vec<TTDmMove>), BackendError> {
    let mut reader: Vec<TTDmMove> = Vec::new();
    let mut writer: Vec<TTDmMove> = Vec::new();
    let mut section = 0u8;
    let mut op_id = k.head;
    while !op_id.is_null() {
        match k.at(op_id) {
            Op::TT(TTOp::EndReader) => section = 1,
            Op::TT(TTOp::EndCompute) => section = 2,
            Op::TT(TTOp::Copy { src, dst, bytes, .. }) if section != 1 => {
                let (param, cb) = match section {
                    0 => (*src, *dst),
                    2 => (*dst, *src),
                    _ => unreachable!("section is 0 or 2 in this arm"),
                };
                let ord = param_ordinal_of
                    .get(&param)
                    .copied()
                    .ok_or_else(|| tt_err(format!("tenstorrent: boundary move {op_id:?} has an unnumbered param {param}")))?;
                let cb_num = cbs
                    .get(&cb)
                    .copied()
                    .ok_or_else(|| tt_err(format!("tenstorrent: boundary move {op_id:?} has an unplaced CB {cb}")))?;
                // A writer CB is FP32 while a 16-bit param needs narrowing;
                // the read direction always moves the param's own bytes.
                let narrow = match section {
                    0 => TT_NARROW_NONE,
                    2 => tt_narrow(k.dtype(cb), k.dtype(param)),
                    _ => unreachable!("section is 0 or 2 in this arm"),
                };
                let move_ = TTDmMove { param: ord, l1: l1[cb_num as usize], bytes: *bytes, narrow };
                match section {
                    0 => reader.push(move_),
                    2 => writer.push(move_),
                    _ => unreachable!("section is 0 or 2 in this arm"),
                }
            }
            _ => {}
        }
        op_id = k.next_op(op_id);
    }
    Ok((reader, writer))
}

/// Top-level render: split sections, collect per-section params
/// (transitive data deps of section ops, head order — the backend
/// passes exactly these as runtime args), emit preamble + body per
/// section. Pure 1:1 emission: placement (locks, sync, inits, pops)
/// is all in the IR; CB numbers come from `cbs`, slots are assigned
/// here (numbers, not placement). F32 circular buffers are rejected
/// (broken 32-bit CB path in tt-metal); any F32 tile selects 32-bit
/// DST mode.
/// L1 address to hand the unpacker as a tile image's base.
///
/// The unpacker reads a tile's datums at `(Base_address + 1) * 16`
/// (`UNPACR_Regular.md`: the `+1` steps over the 16-byte tile header), so the
/// base names the start of an image *with* its header. zyx CBs hold the datum
/// stream alone — the host tilize writes no header, and the packer writes the
/// datums at the CB's own base — so the base is the 16 bytes before the
/// datums. Passing the CB base itself would make the unpacker read the first
/// eight BF16 datums of the tile as a header and shift every face by eight.
fn tt_src_base(cb: u64) -> u64 {
    cb - tt_isa::tile::TileImage::ALIGNMENT as u64
}

/// Unpacker index -> the metalium `Unpacker` (0 = SrcA, 1 = SrcB).
fn tt_unpacker(unit: u8) -> Result<crate::backend::tenstorrent::kern::datapath::Unpacker, String> {
    match unit {
        0 => Ok(crate::backend::tenstorrent::kern::datapath::Unpacker::SrcA),
        1 => Ok(crate::backend::tenstorrent::kern::datapath::Unpacker::SrcB),
        other => Err(format!("unpacker {other} is neither SrcA (0) nor SrcB (1)")),
    }
}

/// A role stream (0 = T0, 1 = T1, 2 = T2) as the unit a post drains.
fn tt_unit(role: u8) -> Option<tt_isa::sync::Unit> {
    match role {
        0 => Some(tt_isa::sync::Unit::Unpacker0),
        1 => Some(tt_isa::sync::Unit::Matrix),
        2 => Some(tt_isa::sync::Unit::Packer),
        _ => None,
    }
}

/// Semaphore index -> the metalium handle (`0..8`).
fn tt_sem(index: u8) -> Result<tt_isa::sync::Semaphore, String> {
    tt_isa::sync::Semaphore::new(index).ok_or_else(|| format!("semaphore {index} is out of range 0..8"))
}

/// `before` code -> the `Before` consumer mask. The encoding is the order of
/// the metalium constants; `Before` has no public constructor from bits.
fn tt_before(code: u8) -> Option<tt_isa::backend::Before> {
    use tt_isa::backend::Before;
    match code {
        0 => Some(Before::UNPACKER),
        1 => Some(Before::PACKER),
        2 => Some(Before::MATRIX),
        3 => Some(Before::SFPU),
        4 => Some(Before::CONFIG),
        5 => Some(Before::SCALAR),
        6 => Some(Before::MOVER),
        7 => Some(Before::EVERYTHING),
        _ => None,
    }
}

/// `ThreadConfig` field address -> the typed field. The reverse lookup
/// `ThreadConfigField` does not offer, limited to the fields zyx emits.
/// T0, the unpack role: the thread config first (state id plus the unpacker
/// overrides — `src_thread_config` when the body unpacks into `SrcA`/`SrcB`,
/// the `Dst` path's `thread_config` otherwise), then one self-contained
/// unpack per `Copy`/`Unpacr`, then a drain so the role is complete when the
/// firmware reports it done.
fn tt_unpack_role(body: Vec<u32>, src_mode: bool) -> Vec<u32> {
    use tt_isa::backend::{Before, wait_for_unpacker0, wait_for_unpacker1};
    if body.is_empty() {
        return Vec::new();
    }
    let config = if src_mode {
        crate::backend::tenstorrent::kern::datapath::src_thread_config()
    } else {
        crate::backend::tenstorrent::kern::datapath::thread_config()
    };
    let mut role: Vec<u32> = config.iter().map(|i| i.word()).collect();
    role.extend(body);
    // Both unpackers can have run (a Src matmul unpacks `SrcA` and `SrcB`),
    // so both are drained; `matmul_items` ends the same two waits.
    role.push(wait_for_unpacker0(Before::EVERYTHING).expect("wait_for_unpacker0(EVERYTHING) is encodable").word());
    if src_mode {
        role.push(wait_for_unpacker1(Before::EVERYTHING).expect("wait_for_unpacker1(EVERYTHING) is encodable").word());
    }
    role
}

/// T1, the math role: the state id, then the per-op math programs in
/// emission order, then an SFPU drain.
fn tt_math_role(body: Vec<u32>) -> Vec<u32> {
    use tt_isa::backend::{Before, wait_for_sfpu};
    if body.is_empty() {
        return Vec::new();
    }
    let mut role: Vec<u32> = vec![crate::backend::tenstorrent::kern::datapath::state_id().word()];
    role.extend(body);
    role.push(wait_for_sfpu(Before::EVERYTHING).expect("wait_for_sfpu(EVERYTHING) is encodable").word());
    role
}

/// T2, the pack role: the state id, then one self-contained DST -> CB pack
/// per `Store`, then a packer drain (the thread unblocks once the packer
/// accepts work, not when it finishes).
fn tt_pack_role(body: Vec<u32>) -> Vec<u32> {
    use tt_isa::backend::{Before, wait_for_packer};
    if body.is_empty() {
        return Vec::new();
    }
    let mut role: Vec<u32> = vec![crate::backend::tenstorrent::kern::datapath::state_id().word()];
    role.extend(body);
    role.push(wait_for_packer(Before::EVERYTHING).expect("wait_for_packer(EVERYTHING) is encodable").word());
    role
}

fn render(
    k: &Kernel,
    param_ordinal_of: &Map<OpId, u32>,
    cbs: &Map<OpId, u32>,
    native: &mut TTNativeProgram,
) -> Result<(String, String, String, Vec<u32>, Vec<u32>, Vec<u32>, bool, Vec<u32>), BackendError> {
    let mut sections: [Vec<OpId>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut section = 0usize;
    let mut op_id = k.head;
    while !op_id.is_null() {
        match k.at(op_id) {
            Op::TT(TTOp::EndReader) => section = 1,
            Op::TT(TTOp::EndCompute) => section = 2,
            _ => sections[section].push(op_id),
        }
        op_id = k.next_op(op_id);
    }

    // Per-section params: transitive data deps of section traffic
    // and compute, head order. Seeds are effect/structure ops only —
    // bare definitions (params, storages, consts) seed nothing;
    // they join through their users.
    let mut params: [Vec<OpId>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for s in 0..3 {
        let mut seen: Set<OpId> = Set::default();
        let mut stack: Vec<OpId> = sections[s]
            .iter()
            .copied()
            .filter(|id| !matches!(k.at(*id), Op::Param { .. } | Op::Storage { .. } | Op::Const { .. }))
            .collect();
        while let Some(id) = stack.pop() {
            if id.is_null() || !seen.insert(id) {
                continue;
            }
            if let Op::Param { .. } = k.at(id) {
                params[s].push(id);
            }
            for p in k.at(id).parameters() {
                stack.push(p);
            }
        }
        // Walk order is head order; the DFS above is not — restore it.
        let mut order: Map<OpId, usize> = Map::default();
        let mut idx = 0usize;
        let mut scan = k.head;
        while !scan.is_null() {
            order.insert(scan, idx);
            idx += 1;
            scan = k.next_op(scan);
        }
        params[s].sort_by_key(|id| order[id]);
        // DRAM discipline (old accessor rule): reader takes Global +
        // GlobalMut, writer takes GlobalMut only, compute takes none;
        // every section takes the Variables its index math reads
        // (group lengths arrive through Range transitive deps — the
        // preamble emits their arg reads like compute's).
        for p in params[s].iter() {
            if let Op::Param { kind, .. } = k.at(*p) {
                match (s, kind) {
                    (0, ParamKind::Global) | (0, ParamKind::GlobalMut) | (2, ParamKind::GlobalMut) => {}
                    (_, ParamKind::Variable) => {}
                    _ => {
                        return Err(tt_err(format!("tenstorrent section {s}: param {p:?} ({kind:?}) has no accessor there")));
                    }
                }
            }
        }
    }

    // Users: the circular-Load arm (below) reads it.
    let mut users: Map<OpId, Vec<OpId>> = Map::default();
    let mut scan = k.head;
    while !scan.is_null() {
        for p in k.at(scan).parameters() {
            if !p.is_null() {
                users.entry(p).or_default().push(scan);
            }
        }
        scan = k.next_op(scan);
    }

    let mut srcs = [String::new(), String::new(), String::new()];
    let mut lists: [Vec<u32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut native_words: Vec<u32> = Vec::new();
    let mut fp32 = false; // 32-bit DST mode: any F32 tile, or any F8 circular (Blackhole
    // mandate). F32 circulars are accepted (format 0), like the old backend.
    let mut scan = k.head;
    while !scan.is_null() {
        if let Op::Storage { dtype, scope, .. } = k.at(scan) {
            if *dtype == DType::F32 || (*dtype == DType::F8E4M3 && *scope == MemScope::Circular) {
                fp32 = true;
            }
        }
        scan = k.next_op(scan);
    }
    let budget = if fp32 { 8 } else { 16 };

    for s in 0..3 {
        // Reachable names: loop counters (encounter order at their
        // Loop), grid coords and Variable args (in the closure, same
        // definition-excluding seeds as section params).
        let mut reachable: Set<OpId> = Set::default();
        let mut stack: Vec<OpId> = sections[s]
            .iter()
            .copied()
            .filter(|id| !matches!(k.at(*id), Op::Param { .. } | Op::Storage { .. } | Op::Const { .. }))
            .collect();
        while let Some(id) = stack.pop() {
            if id.is_null() || !reachable.insert(id) {
                continue;
            }
            for p in k.at(id).parameters() {
                stack.push(p);
            }
        }
        let mut loop_names: Map<OpId, String> = Map::default();
        let mut range_names: Map<OpId, String> = Map::default();
        let mut var_names: Map<OpId, String> = Map::default();
        let mut li = 0u32;
        for id in sections[s].iter().copied() {
            if matches!(k.at(id), Op::Loop { .. }) && !loop_names.contains_key(&id) {
                loop_names.insert(id, format!("i{li}"));
                li += 1;
            }
        }
        for id in reachable.iter().copied() {
            match k.at(id) {
                Op::Range { axis, kind, .. } => {
                    if matches!(kind, RangeKind::Group(_)) {
                        if *axis > 1 {
                            return Err(tt_err(format!("tenstorrent section {s}: grid axis {axis} (X=0/Y=1 only)")));
                        }
                        range_names.entry(id).or_insert_with(|| format!("g{axis}"));
                    }
                }
                Op::Param { kind, .. } => {
                    if matches!(kind, ParamKind::Variable) {
                        let ord = param_ordinal_of
                            .get(&id)
                            .copied()
                            .ok_or_else(|| tt_err(format!("tenstorrent: param {id:?} has no ordinal")))?;
                        var_names.entry(id).or_insert_with(|| format!("v{ord}"));
                    }
                }
                _ => {}
            }
        }

        let mut sec = TtSection {
            k,
            cbs,
            ordinals: param_ordinal_of,
            users: &users,
            section: s as u8,
            params: params[s].clone(),
            loop_names,
            range_names,
            var_names,
            slots: Map::default(),
            next_slot: 0,
            budget,
            transfer: 0,
            indent: String::from("  "),
            math_open: false,
            pack_open: false,
            out: String::new(),
            native: Vec::new(),
            l1: &native.l1,
            native_t0: Vec::new(),
            native_t2: Vec::new(),
            last_role: 1,
            sems: Vec::new(),
            compute_ops: 0,
            src_unpacker_ready: [false, false],
            has_src_unpack: false,
        };
        // Preamble: includes + kernel_main + CB declares + grid/var
        // reads + DRAM accessors (front emission for both dataflow
        // sections — same programs, fewer rules than inline-first).
        match s {
            0 => {
                sec.out.push_str("#include <cstdint>\n");
                sec.out.push_str("#include \"api/dataflow/dataflow_api.h\"\n");
                sec.out.push_str("#include \"api/dataflow/noc.h\"\n");
                sec.out.push_str("#include \"api/dataflow/circular_buffer.h\"\n");
                sec.out.push_str("#include \"api/tensor/noc_traits.h\"\n");
                sec.out.push_str("#include \"api/debug/device_print.h\"\n");
            }
            1 => {
                sec.out.push_str("#include <cstdint>\n");
                sec.out.push_str("#include \"api/compute/common.h\"\n");
                sec.out.push_str("#include \"api/compute/compute_kernel_api.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_binary_sfpu.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/binop_with_scalar.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/left_shift.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/right_shift.h\"\n");
                sec.out.push_str("#include \"api/compute/tile_move_copy.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/eltwise_unary.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/trigonometry.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/exp.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/recip.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/rsqrt.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/sqrt.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/rounding.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/negative.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/bitwise_not.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/typecast.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/logical_not.h\"\n");
                sec.out.push_str("#include \"api/compute/binary_max_min.h\"\n");
                sec.out.push_str("#include \"api/compute/binary_shift.h\"\n");
                sec.out.push_str("#include \"api/compute/eltwise_unary/fill.h\"\n");
                sec.out.push_str("#include \"api/compute/matmul.h\"\n");
                sec.out.push_str("#include \"api/compute/bcast.h\"\n");
                sec.out.push_str("#include \"api/compute/reduce.h\"\n");
                sec.out.push_str("#include \"api/compute/transpose_wh.h\"\n");
                sec.out.push_str("#include \"api/compute/reconfig_data_format.h\"\n");
                sec.out.push_str("#include \"api/dataflow/circular_buffer.h\"\n");
                sec.out.push_str("#include \"api/debug/device_print.h\"\n");
            }
            _ => {
                sec.out.push_str("#include <cstdint>\n");
                sec.out.push_str("#include \"api/dataflow/dataflow_api.h\"\n");
                sec.out.push_str("#include \"api/dataflow/noc.h\"\n");
                sec.out.push_str("#include \"api/dataflow/circular_buffer.h\"\n");
                sec.out.push_str("#include \"api/tensor/noc_traits.h\"\n");
                sec.out.push_str("#include \"api/debug/dprint.h\"\n");
            }
        }
        sec.out.push_str("void kernel_main() {\n");
        let mut cb_nums: Vec<u32> = cbs.values().copied().collect();
        cb_nums.sort_unstable();
        for cb in cb_nums {
            sec.out.push_str(&format!("  CircularBuffer cb{cb}(tt::CBIndex::c_{cb});\n"));
        }
        let mut axes: Vec<u32> = sec.range_names.values().map(|g| g[1..].parse().unwrap_or(99)).collect();
        axes.sort_unstable();
        axes.dedup();
        for axis in axes {
            let ai = sec.params.len() + axis as usize;
            sec.out.push_str(&format!("  uint32_t g{axis} = get_arg_val<uint32_t>({ai});\n"));
        }
        for id in params[s].iter().copied() {
            if let Op::Param { kind: ParamKind::Variable, .. } = k.at(id) {
                let ai = sec.arg_of(id)?;
                let ord = sec.ord_of(id)?;
                sec.out.push_str(&format!("  uint32_t v{ord} = get_arg_val<uint32_t>({ai});\n"));
            }
        }
        // DRAM accessors, chained compile-time offsets (old shape).
        // Reader GlobalMut reads use the `dst` form; writer drains
        // use `out`/`p_out`.
        let mut cta: Option<String> = None;
        for id in params[s].iter().copied() {
            let Op::Param { kind, .. } = k.at(id) else { continue };
            if matches!(kind, ParamKind::Variable) {
                continue;
            }
            let ai = sec.arg_of(id)?;
            let ord = sec.ord_of(id)?;
            let chain = cta.clone().unwrap_or_else(|| "0".to_string());
            match (s, kind) {
                (0, ParamKind::Global) => {
                    sec.out.push_str(&format!("  uint32_t src{ord} = get_arg_val<uint32_t>({ai});\n"));
                    sec.out.push_str(&format!("  auto args{ord} = TensorAccessorArgs<{chain}>({ai});\n"));
                    sec.out.push_str(&format!(
                        "  auto p{ord} = TensorAccessor(args{ord}, src{ord}, {page});\n",
                        page = TT_DRAM_PAGE_BYTES
                    ));
                    cta = Some(format!("args{ord}.next_compile_time_args_offset()"));
                }
                (0, ParamKind::GlobalMut) => {
                    sec.out.push_str(&format!("  uint32_t dst{ord} = get_arg_val<uint32_t>({ai});\n"));
                    sec.out.push_str(&format!("  auto args{ord} = TensorAccessorArgs<{chain}>({ai});\n"));
                    sec.out.push_str(&format!(
                        "  auto p{ord} = TensorAccessor(args{ord}, dst{ord}, {page});\n",
                        page = TT_DRAM_PAGE_BYTES
                    ));
                    cta = Some(format!("args{ord}.next_compile_time_args_offset()"));
                }
                (2, ParamKind::GlobalMut) => {
                    sec.out.push_str(&format!("  uint32_t out{ord} = get_arg_val<uint32_t>({ai});\n"));
                    sec.out.push_str(&format!("  auto args_out{ord} = TensorAccessorArgs<{chain}>({ai});\n"));
                    sec.out.push_str(&format!(
                        "  auto p_out{ord} = TensorAccessor(args_out{ord}, out{ord}, {page});\n",
                        page = TT_DRAM_PAGE_BYTES
                    ));
                    cta = Some(format!("args_out{ord}.next_compile_time_args_offset()"));
                }
                _ => {}
            }
        }
        render_section(&mut sec, &sections[s])?;
        match s {
            0 => sec.out.push_str("  noc_async_read_barrier();\n"),
            2 => sec.out.push_str("  noc_async_write_barrier();\n"),
            _ => {}
        }
        sec.out.push_str("}\n");
        let section_native = std::mem::take(&mut sec.native);
        let section_t0 = std::mem::take(&mut sec.native_t0);
        let section_t2 = std::mem::take(&mut sec.native_t2);
        if s == 1 {
            native_words = section_native;
            // Role streams, T0/T1/T2 in order. The roles run in order
            // (`Schedule::InOrder`), so no semaphores: each role drains its
            // unit before the firmware reports it done.
            native.compute = [tt_unpack_role(section_t0, sec.has_src_unpack), tt_math_role(native_words.clone()), tt_pack_role(section_t2)];
            native.sems = std::mem::take(&mut sec.sems);
            // Physical DST rows: the emitter's slot table is the only place
            // a Register storage is bound to a row, so snapshot it here.
            native.dst = sec
                .slots
                .iter()
                .filter(|(id, _)| matches!(k.at(**id), Op::Storage { scope: MemScope::Register, .. }))
                .map(|(id, row)| TTDstRow { storage: id.0, row: *row })
                .collect();
            native.dst.sort_by_key(|d| d.row);
        } else {
            debug_assert!(section_native.is_empty(), "native compute words outside the compute section");
            debug_assert!(section_t0.is_empty(), "native unpack words outside the compute section");
            debug_assert!(section_t2.is_empty(), "native pack words outside the compute section");
        }
        srcs[s] = sec.out;
        lists[s] = params[s].iter().map(|id| param_ordinal_of[id]).collect();
    }

    Ok((
        srcs[0].clone(),
        srcs[1].clone(),
        srcs[2].clone(),
        lists[0].clone(),
        lists[1].clone(),
        lists[2].clone(),
        fp32,
        native_words,
    ))
}

/// Render a traffic `Copy`: publish (DRAM→CB, NOC read) or drain
/// (CB→DRAM, NOC write). A NOC barrier follows every transfer (old
/// per-transfer barrier). Accessors use global param ordinals
/// (`p<ord>`); arg reads use section positions. CB→CB moves never
/// reach here (`tt_storage` canonicalizes them to load+store); the
/// arm below is defensive only.
fn render_copy(sec: &mut TtSection, id: OpId, src: OpId, src_idx: OpId, dst: OpId, dst_idx: OpId) -> Result<(), BackendError> {
    let src_cb = circular_base(sec.k, src)?;
    let dst_cb = circular_base(sec.k, dst)?;
    match (src_cb, dst_cb) {
        (None, None) => Err(sec.err(format!("copy {id:?} touches no CB"))),
        (None, Some(dst_storage)) => {
            if sec.section != 0 {
                return Err(sec.err(format!("publish {id:?} outside the reader")));
            }
            let dcb = sec.cb_num(dst_storage)?;
            let (param, sidx) = dram_side(sec.k, src, src_idx)?;
            let ord = sec.ord_of(param)?;
            let idx = sec.expr(sidx)?;
            let eb = elem_bytes(sec.k, param)?;
            // All CB indices are elementwise (uniform with vector
            // publishes, scalar traffic, and vectorized loads): the
            // index counts elements, `bytes` carries the transfer
            // size. Anything else is malformed traffic.
            let tile_bytes = tt_tile_bytes(cb_dtype(sec.k, dst_storage)?)?;
            let (bytes, stride) = match sec.k.shape(id).as_slice() {
                [_, _] => (tile_bytes, eb),
                [size] => ((*size as u32) * eb, eb),
                [] => (eb, eb),
                _ => todo!("tt publish: rank>2 access"),
            };
            let slot = cb_slot_expr(sec, dst_idx, stride)?;
            let noc = sec.transfer;
            sec.transfer += 1;
            sec.out.push_str(&format!(
                "{}uint64_t rnoc{noc} = p{ord}.get_noc_addr((uint32_t)((( {idx} )*{eb})/{page}), (uint32_t)((( {idx} )*{eb})%{page}));\n",
                sec.indent,
                page = TT_DRAM_PAGE_BYTES
            ));
            sec.out.push_str(&format!("{}noc_async_read(rnoc{noc}, cb{dcb}.get_write_ptr(){slot}, {bytes});\n", sec.indent));
            sec.out.push_str(&format!("{}noc_async_read_barrier();\n", sec.indent));
            Ok(())
        }
        (Some(src_storage), None) => {
            if sec.section != 2 {
                return Err(sec.err(format!("drain {id:?} outside the writer")));
            }
            let scb = sec.cb_num(src_storage)?;
            let (param, didx) = dram_side(sec.k, dst, dst_idx)?;
            let ord = sec.ord_of(param)?;
            let idx = sec.expr(didx)?;
            let eb = elem_bytes(sec.k, param)?;
            // Transfer size follows the access shape (mirror of the
            // publish arm): whole tile, `size`-element vector, scalar.
            let bytes = match sec.k.shape(id).as_slice() {
                [_, _] => tt_tile_bytes(cb_dtype(sec.k, src_storage)?)?,
                [size] => (*size as u32) * eb,
                [] => eb,
                _ => todo!("tt drain: rank>2 access"),
            };
            // Drain indices are elementwise (like the publish side), so
            // the slot stride is `eb`, not tile bytes.
            let slot = cb_slot_expr(sec, src_idx, eb)?;
            let noc = sec.transfer;
            sec.transfer += 1;
            sec.out.push_str(&format!(
                "{}uint64_t wnoc{noc} = p_out{ord}.get_noc_addr((uint32_t)((( {idx} )*{eb})/{page}), (uint32_t)((( {idx} )*{eb})%{page}));\n",
                sec.indent,
                page = TT_DRAM_PAGE_BYTES
            ));
            sec.out.push_str(&format!("{}noc_async_write(cb{scb}.get_read_ptr(){slot}, wnoc{noc}, {bytes});\n", sec.indent));
            sec.out.push_str(&format!("{}noc_async_write_barrier();\n", sec.indent));
            Ok(())
        }
        (Some(src_storage), Some(dst_storage)) => {
            if sec.section != 1 {
                return Err(sec.err(format!("CB→CB copy {id:?} outside the compute section")));
            }
            if matches!(sec.k.shape(id).as_slice(), [_]) {
                return Err(sec.err(format!("vector CB→CB copy {id:?} is not supported")));
            }
            let scb = sec.cb_num(src_storage)?;
            let dcb = sec.cb_num(dst_storage)?;
            let s = sec.fresh_slot()?;
            sec.out.push_str(&format!("{}copy_tile({scb}, 0, {s});\n", sec.indent));
            sec.out.push_str(&format!("{}pack_tile({s}, {dcb});\n", sec.indent));
            Ok(())
        }
    }
}

/// The Circular storage under a traffic `Copy` side (`None` = DRAM
/// side): the side IS the base param/storage now.
fn circular_base(k: &Kernel, side: OpId) -> Result<Option<OpId>, BackendError> {
    if matches!(k.at(side), Op::Storage { scope: MemScope::Circular, .. }) {
        Ok(Some(side))
    } else {
        Ok(None)
    }
}

/// The DRAM param + tile index of a traffic `Copy` side.
fn dram_side(k: &Kernel, side: OpId, index: OpId) -> Result<(OpId, OpId), BackendError> {
    match k.at(side) {
        Op::Param { .. } => Ok((side, index)),
        _ => Err(tt_err(format!("tenstorrent: copy DRAM side {side:?} is not a param"))),
    }
}

/// Dtype of a CB storage.
fn cb_dtype(k: &Kernel, cb: OpId) -> Result<DType, BackendError> {
    match k.at(cb) {
        Op::Storage { dtype, .. } => Ok(*dtype),
        _ => Err(tt_err(format!("tenstorrent: CB {cb:?} is not a storage op"))),
    }
}

/// Element bytes of a DRAM param dtype.
fn elem_bytes(k: &Kernel, param: OpId) -> Result<u32, BackendError> {
    match k.at(param) {
        Op::Param { dtype, .. } => Ok(u32::from(dtype.bit_size()) / 8),
        _ => Err(tt_err(format!("tenstorrent: {param:?} is not a param"))),
    }
}

/// Render a tiled elementwise op: unary/cast/bitcast alias their
/// lane slot (in-place chains, old ownership transfer); tile-tile
/// binaries take a fresh slot; scalar binaries fold into the unary
/// form (old `TileBinScalar` calls); shifts take immediates. The
/// pass validated lanes and placed the inits; unknown combos are
/// loud errors (old "has no LLK/scalar call" errors).
fn render_tile_op(sec: &mut TtSection, id: OpId) -> Result<(), BackendError> {
    match sec.k.at(id) {
        Op::Unary { x, uop } => {
            let s = sec.operand_slot(*x)?;
            let name = match uop {
                UOp::Neg => "negative_tile",
                UOp::BitNot => "bitwise_not_tile",
                UOp::Exp2 => "exp2_tile",
                UOp::Log2 => "log_with_base_tile",
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
            if *uop == UOp::Log2 {
                sec.out.push_str(&format!("{}{name}({s}, 0x3fb8aa3b);\n", sec.indent));
            } else {
                sec.out.push_str(&format!("{}{name}({s});\n", sec.indent));
            }
            sec.slots.insert(id, s);
            Ok(())
        }
        Op::Cast { x, dtype } => {
            let s = sec.operand_slot(*x)?;
            let in_dtype = sec.k.dtype(*x);
            sec.out.push_str(&format!("{}typecast_tile<{}, {}>({s});\n", sec.indent, tt_fmt(in_dtype)?, tt_fmt(*dtype)?));
            sec.slots.insert(id, s);
            Ok(())
        }
        Op::Bitcast { x, .. } => {
            let s = sec.operand_slot(*x)?;
            sec.slots.insert(id, s);
            Ok(())
        }
        Op::Binary { x, y, bop } => {
            let lane = match (sec.k.shape(*x).len() == 2, sec.k.shape(*y).len() == 2) {
                (true, true) => None,
                (true, false) => Some((false, *y)),
                (false, true) => Some((true, *x)),
                (false, false) => return Err(sec.err(format!("tiled binary {id:?} with no tile lane"))),
            };
            match lane {
                None => {
                    let sx = sec.operand_slot(*x)?;
                    let sy = sec.operand_slot(*y)?;
                    let dst = sec.fresh_slot()?;
                    let in_dtype = sec.k.dtype(*x);
                    let (name, tmpl) = match bop {
                        BOp::Add => ("add_binary_tile", ""),
                        BOp::Sub => ("sub_binary_tile", ""),
                        BOp::Mul => ("mul_binary_tile", ""),
                        BOp::Div => ("div_binary_tile", ""),
                        BOp::Max => ("binary_max_tile", ""),
                        BOp::BitShiftLeft => ("binary_left_shift_tile", shift_tmpl(in_dtype)?),
                        BOp::BitShiftRight if in_dtype == DType::U32 => {
                            ("binary_logical_right_shift_tile", shift_tmpl(in_dtype)?)
                        }
                        BOp::BitShiftRight => ("binary_right_shift_tile", shift_tmpl(in_dtype)?),
                        _ => return Err(sec.err(format!("tiled binary {bop:?} has no LLK call"))),
                    };
                    sec.out.push_str(&format!("{}{name}{tmpl}({sx}, {sy}, {dst});\n", sec.indent));
                    sec.slots.insert(id, dst);
                    Ok(())
                }
                Some((left, s)) => {
                    let tile = if left { sec.operand_slot(*y)? } else { sec.operand_slot(*x)? };
                    match bop {
                        BOp::Add | BOp::Mul | BOp::Div => {
                            let name = match bop {
                                BOp::Add => "add_unary_tile",
                                BOp::Mul => "mul_unary_tile",
                                _ => "div_unary_tile",
                            };
                            let bits = match sec.k.resolve_const(s) {
                                Some(Constant::F32(b)) => f32::from_le_bytes(b).to_bits(),
                                Some(Constant::F16(b)) => f16(u16::from_le_bytes(b)).to_f32().to_bits(),
                                Some(Constant::BF16(b)) => bf16(u16::from_le_bytes(b)).to_f32().to_bits(),
                                _ => {
                                    return Err(sec.err(format!("scalar lane {s:?} is not a foldable float const")));
                                }
                            };
                            sec.out.push_str(&format!("{}{name}({tile}, 0x{bits:x});\n", sec.indent));
                            sec.slots.insert(id, tile);
                            Ok(())
                        }
                        BOp::BitShiftLeft | BOp::BitShiftRight => {
                            let name = if *bop == BOp::BitShiftLeft {
                                "left_shift_tile"
                            } else {
                                "right_shift_tile"
                            };
                            let amt = match sec.k.at(s) {
                                Op::Const(c) => c.as_dim().ok_or_else(|| sec.err(format!("shift amount {s:?} is not a u32")))?,
                                _ => return Err(sec.err(format!("shift amount {s:?} is not an int const"))),
                            };
                            sec.out.push_str(&format!("{}{name}({tile}, {amt});\n", sec.indent));
                            sec.slots.insert(id, tile);
                            Ok(())
                        }
                        _ => Err(sec.err(format!("scalar {bop:?} has no scalar call"))),
                    }
                }
            }
        }
        _ => Err(sec.err(format!("{id:?} is not a tile op"))),
    }
}

/// `DataFormat` template for tiled shifts (old shift-format rule).
fn shift_tmpl(dtype: DType) -> Result<&'static str, BackendError> {
    match dtype {
        DType::I32 => Ok("<DataFormat::Int32>"),
        DType::U32 => Ok("<DataFormat::UInt32>"),
        DType::U16 => Ok("<DataFormat::UInt16>"),
        dt => Err(tt_err(format!("tenstorrent: tiled shift on {dt:?} has no LLK format"))),
    }
}

/// Render a `Load`: a circular load consumed straight by pack
/// stores unpacks here (`copy_tile` into a fresh slot, aliased for
/// the stores — the lock pass holds MATH open over exactly these
/// loads, the init pass configured the unpack ahead of them). Every
/// other load emits nothing: SSA consumers copy at their own use
/// (`operand_slot`), LLK provenance loads are sync accounting, dead
/// loads are pops only.
fn render_load(sec: &mut TtSection, id: OpId, src: OpId) -> Result<(), BackendError> {
    // Validate the access resolves to a concrete shape.
    let _ = sec.k.shape(id);
    if !matches!(sec.k.at(src), Op::Storage { scope: MemScope::Circular, .. }) {
        return Ok(());
    }
    let all_pack = match sec.users.get(&id) {
        Some(us) => {
            us.iter().all(|u| matches!(sec.k.at(*u), Op::Store { .. }))
                && us.iter().any(|u| matches!(sec.k.at(*u), Op::Store { .. }))
        }
        None => false,
    };
    if !all_pack {
        return Ok(());
    }
    let cb = sec.cb_num(src)?;
    let s = sec.fresh_slot()?;
    sec.out.push_str(&format!("{}copy_tile({cb}, 0, {s});\n", sec.indent));
    sec.slots.insert(id, s);
    Ok(())
}

/// Render a `Store`: pack a tile into a CB (`pack_tile`), alias
/// a Register acc (no traffic — the slot mapping is the state), or
/// write one indexed element into a reserved CB page (a DRAM-loaded
/// scalar NOC-read straight to the page at the element offset).
/// The packed value must already hold a slot (SSA lowered earlier,
/// circular load unpacked at its own position); anything else is
/// malformed pack traffic.
/// CB pointer offsets (`cb_slot_expr`): empty for slot 0 (the old
/// unbatched form), ` + (<expr>)*<bytes>` otherwise.
fn cb_slot_expr(sec: &TtSection, idx: OpId, bytes: u32) -> Result<String, BackendError> {
    match sec.k.at(idx) {
        Op::Const(c) => match c.as_dim() {
            Some(0) => Ok(String::new()),
            _ => Ok(format!(" + ({})*{bytes}", sec.expr(idx)?)),
        },
        _ => Ok(format!(" + ({})*{bytes}", sec.expr(idx)?)),
    }
}
fn render_store(sec: &mut TtSection, id: OpId, x: OpId, dst: OpId) -> Result<(), BackendError> {
    match sec.k.at(dst) {
        Op::Storage { scope: MemScope::Circular, .. } => {
            // Scalar stores into a Circular buffer are rejected:
            // DRAM-CB movement is Copy-only, use
            // load_global_to_cb_scalar.
            if let Op::Load { .. } = sec.k.at(x) {
                if sec.k.shape(x).is_empty() {
                    return Err(
                        sec.err(format!("scalar CB store {id:?}: DRAM-CB movement is Copy-only, use load_global_to_cb_scalar"))
                    );
                }
            } else if sec.k.shape(x).is_empty() {
                return Err(
                    sec.err(format!("scalar CB store {id:?}: DRAM-CB movement is Copy-only, use load_global_to_cb_scalar"))
                );
            }
            let dcb = sec.cb_num(dst)?;
            let slot = sec.slot_of(x)?;
            // The packer writes the CB's own format, not the value's: the
            // writer CB is FP32 even for a BF16 output, and the boundary
            // narrows on the way back to DRAM (`TTDmMove::narrow`). Packing
            // BF16 directly out of `Dst` is the packer's F32->BF16 conversion
            // (`pack_fmt_conv_mode=0x105`), which the pinned ttsim refuses.
            let out_fmt = tt_fmt(sec.k.dtype(dst))?;
            use tt_isa::backend::{Before, block, cond, stallwait, wait_for_packer};
            use tt_isa::cfg::generated::{global, thcon};
            let l1 = sec.cb_l1(dst)?;
            let mut words = tt_isa::backend::ConfigWords::new();
            crate::backend::tenstorrent::kern::datapath::pack_config(&mut words, l1);
            words
                .set(global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset, slot)
                .map_err(|e| sec.err(format!("pack {id:?}: dst offset: {e:?}")))?;
            words
                .set(thcon::THCON_SEC0_REG1_Out_data_format, out_fmt)
                .map_err(|e| sec.err(format!("pack {id:?}: out format: {e:?}")))?;
            let mut code = vec![wait_for_packer(Before::CONFIG).map_err(|e| sec.err(format!("pack {id:?}: {e:?}")))?];
            code.extend(crate::backend::tenstorrent::kern::datapath::config_program(&words));
            code.push(stallwait(block::PACKER | block::CONFIG, cond::CONFIG_BUSY).map_err(|e| sec.err(format!("pack {id:?}: {e:?}")))?);
            code.extend(crate::backend::tenstorrent::kern::datapath::pack_rows(crate::backend::tenstorrent::kern::datapath::TILE_DST_ROWS));
            sec.native_t2.extend(code.iter().map(|i| i.word()));
            sec.last_role = 2;
            sec.out.push_str(&format!("{}pack_tile({slot}, {dcb});\n", sec.indent));
            Ok(())
        }
        Op::Storage { scope: MemScope::Register, .. } => {
            let slot = sec.slot_of(x)?;
            sec.slots.insert(dst, slot);
            Ok(())
        }
        _ => Err(sec.err(format!("store {id:?} dst {dst:?} ({:?}) is not a buffer", sec.k.at(dst)))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_program_targets_the_given_dst_rows() {
        let input = 128;
        let slot = 192;
        let code = tt_reduce_program(BOp::Add, TileDim::Row, input, slot, None).unwrap();
        let mut rows = Vec::new();
        for insn in code {
            let def = insn.def();
            if def.mnemonic() == "SFPLOAD" || def.mnemonic() == "SFPSTORE" {
                rows.push(def.field("Imm10").unwrap().extract(insn.word()));
            }
        }
        assert!(!rows.is_empty());
        for row in rows {
            assert!(
                (input..input + 64).contains(&row) || (slot..slot + 64).contains(&row),
                "row {row} is not in the input or output DST range"
            );
        }
    }
}
