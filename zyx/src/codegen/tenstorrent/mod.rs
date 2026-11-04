// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Tenstorrent codegen — emission only.
//!
//! Lowering lives in [`crate::kernel::tenstorrent`]: the kernel TT passes
//! (`tt_storage`, `tt_lock_dst`, ...) rewrite the kernel IR in place, and this
//! module is a pure consumer. It walks `Op` + the kernel
//! [`TTOp`](crate::kernel::TTOp) and emits three RISC-V section sources plus
//! the launch tables the backend needs.
//!
//! The instruction builders are here, not in the backend: SFPU programs
//! ([`sfpu`]), the unpacker/packer word builders ([`datapath`]), the L1/CB
//! plan ([`l1`]), the stored-program format ([`code`]) and its expander
//! ([`loops`]), and the matmul constants ([`matmul`]). The backend keeps only
//! device access and execution.
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

// The instruction builders are ported whole from `metalium-rs` and only the
// parts zyx emits are exercised, so their unused surface is dead until a later
// pass prunes it (the same allowance the backend's ports carry).
#[allow(dead_code, unused_imports, unused_variables)]
pub mod code;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod datapath;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod elem;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod kind;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod l1;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod loops;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod matmul;
#[allow(dead_code, unused_imports, unused_variables)]
pub mod sfpu;

use self::sfpu::kernel::{A_ROW, B_ROW, OUT_ROW};
use crate::DType;
use crate::Map;
use crate::Set;
use crate::backend::{GwsDim, gws_from_kernel};
use crate::dtype::Constant;
use crate::error::{BackendError, ErrorStatus};
use crate::kernel::{
    BOp, GPUOp, Kernel, MemScope, Op, OpId, ParamKind, RangeKind, SourceBlock, TTDmRead, TTDmTileRead, TTDmWrite, TTOp,
    TTSfpInsn, UOp,
};
use crate::scalar::{bf16, f16};
use crate::slab::Slab;
use nanoserde::SerBin;
use tt_isa::isa::Instruction;

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

/// Codegen output for the Tenstorrent backend, built from the kernel IR by
/// [`Kernel::generate_tenstorrent`] and rendered into ops by
/// [`Kernel::render_tt`].
pub struct TTProgram {
    /// Reader section source.
    pub reader_src: String,
    /// Compute section source (empty when the kernel is pure copy).
    pub compute_src: String,
    /// Writer section source.
    pub writer_src: String,
    /// The compute core's three programs (T0/T1/T2) as their instruction
    /// words, in emission order. These become [`TTOp::Unpack`], [`TTOp::Math`]
    /// and [`TTOp::Pack`] in the rendered kernel; the C++ source is only the
    /// not-yet-ported fallback.
    pub roles: [Vec<u32>; 3],
    /// Block repeats per role (`[start, len, count]` over `roles`), from row
    /// loop bodies longer than the replay buffer. They ride on
    /// [`TTOp::Math`] so the launch hands them to the runner.
    pub loops: [Vec<[u32; 3]>; 3],
    /// The DRAM boundary: one move per reader publish (param -> CB) and per
    /// writer drain (CB -> param), in program order. Render expands each into
    /// one [`TTOp::DmRead`]/[`TTOp::DmWrite`] per 4 KB page.
    pub reader: Vec<TTDmMove>,
    pub writer: Vec<TTDmMove>,
    /// Semaphore initial states, rendered as [`TTOp::Semset`].
    pub sems: Vec<TTSemaphoreInit>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One data-mover move of the native program: `bytes` between the DRAM
/// buffer of param `param` and L1 address `l1`.
pub struct TTDmMove {
    /// Global head-order param ordinal (the shim/launch arg index).
    pub param: u32,
    /// L1 byte address of the CB side of the move.
    pub l1: u32,
    /// Transfer size in bytes of the L1 side.
    pub bytes: u32,
    /// Writer-side narrowing: the L1 tile is FP32 and the param is 16-bit,
    /// so the host write-back converts (`TT_NARROW_*`). The packer cannot do
    /// it: F32 `Dst` -> BF16 L1 is `pack_fmt_conv_mode=0x105`, which the
    /// pinned ttsim refuses (`ttsim-divergence.md`, "BF16 late narrowing").
    pub narrow: u8,
    /// Byte offset of this move's slice inside the param's buffer at grid
    /// point `(0, 0)`.
    pub base: u32,
    /// Byte stride of the slice per grid index on axis 0 (gidx0, the row).
    pub gstride0: u32,
    /// Byte stride of the slice per grid index on axis 1 (gidx1, the column).
    pub gstride1: u32,
}

/// Byte offset of a move's slice for one grid point: the program is
/// tile-uniform, so this is the only thing a grid point changes.
pub fn tt_move_offset(mv: &TTDmMove, gidx0: u32, gidx1: u32) -> usize {
    mv.base as usize + gidx0 as usize * mv.gstride0 as usize + gidx1 as usize * mv.gstride1 as usize
}

/// `TTDmMove::narrow`: the L1 bytes are copied as they are.
pub const TT_NARROW_NONE: u8 = 0;
/// `TTDmMove::narrow`: FP32 L1 tile -> FP16 param, round to nearest even.
pub const TT_NARROW_F16: u8 = 1;
/// `TTDmMove::narrow`: FP32 L1 tile -> BF16 param, round to nearest even.
pub const TT_NARROW_BF16: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin)]
/// One semaphore's initial state, mirroring
/// `tt_kernels::runtime::SemaphoreInit` (`(Semaphore, value, max)`), stored
/// as plain bytes so the program table stays plain data.
pub struct TTSemaphoreInit {
    /// Semaphore index, `0..8`.
    pub index: u8,
    /// Initial value.
    pub value: u8,
    /// Maximum value.
    pub max: u8,
}

impl Kernel {
    /// Tenstorrent render: lower to a fully rendered kernel — head order
    /// `Params`, `TensixGrid`, the three role programs, the reader's `DmRead`
    /// pages, `EndReader`, the compute section, `EndCompute`, the `Semset`s,
    /// the writer's `DmWrite` pages. The section sources are the
    /// not-yet-ported fallback only. Lowering lives in `device_epilogue`
    /// (which verifies the lowered IR); `generate_tenstorrent` renders 1:1.
    /// Backend `compile` only walks these ops.
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
        let [unpack, math, pack] = program.roles;
        let [_, math_loops, _] = program.loops;
        let mut rendered =
            Kernel { ops: Slab::new(), head: OpId::NULL, tail: OpId::NULL, dev: self.dev, dev_info: self.dev_info.clone() };
        rendered.push_back(Op::GPU(Box::new(GPUOp::Params(params.into_boxed_slice()))));
        rendered.push_back(Op::TT(TTOp::TensixGrid(Box::new(grid))));
        rendered.push_back(Op::TT(TTOp::Unpack(unpack.into_boxed_slice())));
        rendered.push_back(Op::TT(TTOp::Math(Box::new(crate::kernel::TTRoleProgram {
            words: math.into_boxed_slice(),
            loops: math_loops.into_boxed_slice(),
        }))));
        rendered.push_back(Op::TT(TTOp::Pack(pack.into_boxed_slice())));
        // The reader list: one op per 4 KB page, so the stream's shape is
        // fixed here and the launch only substitutes addresses (NOTES §2).
        rendered.push_back(Op::Source(SourceBlock(program.reader_src.into_boxed_str())));
        for mv in &program.reader {
            // A `TILE_SLOT` publish is the mover's shaped read of one whole
            // tile image (`READ_BROADCAST_COL`): one entry, never paged.
            if mv.bytes == tt_isa::dm::TILE_SLOT as u32 {
                rendered.push_back(Op::TT(TTOp::DmReadBroadcastCol(Box::new(TTDmTileRead {
                    param: mv.param,
                    dst: mv.l1,
                    base: mv.base,
                    gstride0: mv.gstride0,
                    gstride1: mv.gstride1,
                }))));
                continue;
            }
            for page in tt_pages(mv) {
                rendered.push_back(Op::TT(TTOp::DmRead(Box::new(TTDmRead {
                    param: mv.param,
                    l1: page.l1,
                    len: page.len,
                    base: page.base,
                    gstride0: mv.gstride0,
                    gstride1: mv.gstride1,
                }))));
            }
        }
        rendered.push_back(Op::TT(TTOp::EndReader));
        rendered.push_back(Op::Source(SourceBlock(program.compute_src.into_boxed_str())));
        rendered.push_back(Op::TT(TTOp::EndCompute));
        for sem in &program.sems {
            rendered.push_back(Op::TT(TTOp::Semset { sem: sem.index, value: sem.value }));
        }
        for mv in &program.writer {
            for page in tt_pages(mv) {
                rendered.push_back(Op::TT(TTOp::DmWrite(Box::new(TTDmWrite {
                    param: mv.param,
                    l1: page.l1,
                    len: page.len,
                    base: page.base,
                    gstride0: mv.gstride0,
                    gstride1: mv.gstride1,
                    narrow: mv.narrow,
                }))));
            }
        }
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

        // Param ordinals, flat head order.
        let mut param_ordinal_of: Map<OpId, u32> = Map::default();
        let mut next_param = 0u32;
        let mut scan = k.head;
        for _ in 0..50_000 {
            if scan.is_null() {
                break;
            }
            if let Op::Param { .. } = k.at(scan) {
                param_ordinal_of.insert(scan, next_param);
                next_param += 1;
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
        let l1 = tt_place_cbs(k, &cb_order)?;
        let mut sems: Vec<TTSemaphoreInit> = Vec::new();
        let (reader_src, compute_src, writer_src, roles, loops) = render(k, &param_ordinal_of, &cbs, &l1, &mut sems)?;

        // The DRAM boundary: one move per reader publish and per writer drain,
        // in program order. `render_tt` splits each into page ops.
        let (reader, writer) = tt_boundary_moves(k, &cbs, &l1, &param_ordinal_of)?;

        Ok(TTProgram { reader_src, compute_src, writer_src, roles, loops, reader, writer, sems })
    }
}

/// One 4 KB page of a boundary move: its L1 side, its byte length, and its
/// byte offset inside the param's slice. The grid strides move the whole
/// slice, so they stay on the move.
struct TtPage {
    l1: u32,
    len: u32,
    base: u32,
}

/// Split a boundary move into the pages codegen emits one op each for: the
/// DRAM page is the unit that is contiguous on both sides. A shaped tile read
/// (`READ_BROADCAST_COL`, one whole `TILE_SLOT`) is one entry by definition.
fn tt_pages(mv: &TTDmMove) -> Vec<TtPage> {
    if mv.bytes == tt_isa::dm::TILE_SLOT as u32 {
        return vec![TtPage { l1: mv.l1, len: mv.bytes, base: mv.base }];
    }
    let mut pages = Vec::with_capacity(mv.bytes.div_ceil(TT_DRAM_PAGE_BYTES) as usize);
    let mut at = 0u32;
    while at < mv.bytes {
        let len = (mv.bytes - at).min(TT_DRAM_PAGE_BYTES);
        pages.push(TtPage { l1: mv.l1 + at, len, base: mv.base + at });
        at += len;
    }
    pages
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

/// Encode one row-level SFPU instruction: look its form up by ISA opcode and
/// place the operand words into the form's fields, in the ISA's own order
/// (high bits first, as `InstructionDef::fields` lists them).
/// The ISA form an `Sfp` opcode names: the Blackhole layout when the opcode
/// exists on both, never the superseded Wormhole one (`tt_sfp_encode`'s rule).
pub(crate) fn tt_sfp_def(op: u8) -> Option<&'static tt_isa::isa::InstructionDef> {
    tt_isa::isa::generated::ALL
        .iter()
        .copied()
        .find(|d| d.opcode() == op && !matches!(d.provenance(), tt_isa::isa::Provenance::SupersededOnBlackhole { .. }))
        .or_else(|| tt_isa::isa::generated::ALL.iter().copied().find(|d| d.opcode() == op))
}

pub(crate) fn tt_sfp_encode(insn: &TTSfpInsn) -> Result<Instruction, BackendError> {
    // `args` are the form's operands in the ISA's own field order, which is
    // the order `InstructionDef::fields` lists them in; the array is fixed at
    // five words, so a form with fewer fields reads the leading entries and
    // leaves the rest unused.
    //
    // `ALL` carries both layouts where Blackhole replaced a Wormhole form, and
    // the Wormhole one keeps the plain key (`SFPAND`, `SFPPUSHC`, ...) while
    // Blackhole's is suffixed (`SFPAND_BH`). The superseded layout is never the
    // one to emit: it drops fields this IR carries (`SFPAND`'s `VB`, for one).
    let def = tt_sfp_def(insn.op).ok_or_else(|| tt_err(format!("tenstorrent: no ISA instruction has opcode {:#x}", insn.op)))?;
    let fields = def.fields();
    let mut word = def.skeleton();
    for (field, value) in fields.iter().zip(insn.args.iter()) {
        // Every SFPU field is at most 16 bits wide (the widest is `Imm16`);
        // a wider field would be silently truncated by the `u16` args array.
        debug_assert!(
            field.width() <= 16,
            "tenstorrent: {:#x} ({}) field {} is {} bits wide, wider than the u16 args",
            insn.op,
            def.mnemonic(),
            field.name(),
            field.width()
        );
        let value = u32::from(*value);
        if !field.fits(value) {
            return Err(tt_err(format!(
                "tenstorrent: {:#x} ({}) field {} cannot hold {value}",
                insn.op,
                def.mnemonic(),
                field.name()
            )));
        }
        word |= field.place(value);
    }
    Ok(Instruction::new(word, def))
}

/// The native math program for one generic SFPU op: `a_row` (and `b_row`) in,
/// `slot_row` out, every other DST row untouched. The metalium opcode is the
/// `kind` the lowering stored; the arity must match the operands the zyx op
/// carries, otherwise the program would read a row that was never written.
fn tt_sfpu_program(kind: u32, a_row: u32, b_row: Option<u32>, slot_row: u32) -> Result<Vec<Instruction>, BackendError> {
    use crate::codegen::tenstorrent::sfpu::kernel::Operands;
    use crate::codegen::tenstorrent::sfpu::ops::{Broadcast, program_for};
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
    /// A `Replay` body still being emitted: `(first word, word count,
    /// repeats)`. The body is the next `len` words pushed to T1.
    replay: Option<(usize, u16, u16)>,
    /// `true` while the pending row loop is a *block repeat* rather than a
    /// hardware `REPLAY`: its body is longer than the replay buffer, so the
    /// runner pushes it `count` times instead (`mailbox::loops`).
    replay_block: bool,
    /// Which role stream the last emitted op went to (0 = T0, 1 = T1,
    /// 2 = T2): a semaphore op joins the stream of the instruction before it.
    last_role: u8,
    /// Block repeats per role, `[start, len, count]` over that role's words:
    /// a row loop body longer than the replay buffer is pushed this way.
    loops: [Vec<[u32; 3]>; 3],
    /// Semaphore init list from `Semset` ops (the run's concurrent schedule).
    sems: Vec<TTSemaphoreInit>,
    /// Compute value ops emitted so far: `ZEROACC` mode 3 (`CLR_ALL`) is only
    /// safe before any of them, or it would wipe live DST rows.
    compute_ops: u32,
    /// The DST row base of the current matmul accumulator, set by `Setrwc`:
    /// `Mvmul`'s `dst_row` is tile-relative and is offset by this.
    acc_row: u32,
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
        // A Register storage *is* its slot: the allocator stamped the
        // allocation index on it, so the file needs no numbering pass here.
        if let Op::Storage { scope: MemScope::Register, id, .. } = self.k.at(storage) {
            return Ok(*id);
        }
        if let Some(s) = self.slots.get(&storage) {
            return Ok(*s);
        }
        let s = self.fresh_slot()?;
        self.slots.insert(storage, s);
        Ok(s)
    }

    /// DST **row** base of a Register storage: its slot times the rows one
    /// tile occupies (64 for a 32-bit datum, 32 for a 16-bit one). The row,
    /// not the slot, is what every native instruction names (`DstVal`,
    /// `SFPLOAD`/`SFPSTORE` rows, unpack destination, pack offset).
    fn dst_row(&mut self, storage: OpId) -> Result<u32, BackendError> {
        let slot = self.reg_slot(storage)?;
        let rows = if self.k.dtype(storage).bit_size() == 16 { 32 } else { 64 };
        Ok(slot * rows)
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
    #[allow(dead_code)] // kept for callers that follow the last emitted role; `push_role_word_for` is the explicit form
    fn push_role_word(&mut self, word: u32) {
        match self.last_role {
            0 => self.native_t0.push(word),
            1 => self.native.push(word),
            2 => self.native_t2.push(word),
            other => unreachable!("role {other} is not 0..2"),
        }
    }

    /// Begin a row loop: set the row address modifier up and remember that the
    /// next `len` T1 words are the body. A body that fits the replay buffer is
    /// recorded in hardware (`REPLAY`, executed once while being recorded); a
    /// longer one is a *block repeat*: the body is emitted once and the runner
    /// pushes it `count` times, so its length is not bounded by the buffer.
    fn begin_replay(&mut self, len: u16, count: u16) -> Result<(), BackendError> {
        use tt_isa::isa::generated::encode;
        if self.replay.is_some() {
            return Err(self.err("replay: a body cannot contain another".to_string()));
        }
        if count < 2 {
            return Err(self.err(format!("replay: {count} is not a repeat count")));
        }
        if u32::from(len) >= 1 << tt_isa::mailbox::loops::LEN_BITS {
            return Err(self.err(format!(
                "replay: a {len}-word body does not fit a block repeat's {} length bits",
                tt_isa::mailbox::loops::LEN_BITS
            )));
        }
        use crate::codegen::tenstorrent::datapath::{addr_mod_entry, thread_entry};
        let row = addr_mod_entry(0);
        let step = addr_mod_entry(7);
        self.native.push(thread_entry(row.dst_incr, 0).word());
        self.native.push(thread_entry(step.dst_incr, 2).word());
        self.native.push(setrwc_clear());
        self.replay_block = u32::from(len) > tt_isa::frontend::REPLAY_BUFFER;
        if !self.replay_block {
            let record = encode::replay(0, u32::from(len), 1, 1).map_err(|e| self.err(format!("replay: {e:?}")))?;
            self.native.push(record.word());
        }
        self.replay = Some((self.native.len(), len, count));
        self.last_role = 1;
        Ok(())
    }

    /// Close the pending `Replay` once its body is complete: put the stepping
    /// address modifier on the body's last `Dst` access, then emit the
    /// repeats. Does nothing while the body is still short.
    fn finish_replay(&mut self) -> Result<(), BackendError> {
        use tt_isa::isa::generated::{defs, encode};
        let Some((start, len, count)) = self.replay else {
            return Ok(());
        };
        let pushed = self.native.len() - start;
        if pushed < usize::from(len) {
            return Ok(());
        }
        if pushed > usize::from(len) {
            return Err(self.err(format!("replay: body overran {len} words with {pushed}")));
        }
        let (index, field) = (start..start + usize::from(len))
            .rev()
            .find_map(|i| {
                let opcode = (self.native[i] >> 24) as u8;
                if opcode == defs::SFPLOAD.opcode() {
                    Some((i, defs::SFPLOAD.field("AddrMod").expect("SFPLOAD has an AddrMod")))
                } else if opcode == defs::SFPSTORE.opcode() {
                    Some((i, defs::SFPSTORE.field("AddrMod").expect("SFPSTORE has an AddrMod")))
                } else {
                    None
                }
            })
            .ok_or_else(|| self.err("replay: the body never touches Dst".to_string()))?;
        self.native[index] |= field.place(7);
        if self.replay_block {
            // The runner pushes `words[start..start + len]` `count` times.
            self.loops[1].push([start as u32, u32::from(len), u32::from(count)]);
        } else {
            for _ in 1..count {
                let again = encode::replay(0, u32::from(len), 0, 0).map_err(|e| self.err(format!("replay: {e:?}")))?;
                self.native.push(again.word());
            }
        }
        self.native.push(setrwc_clear());
        self.replay = None;
        self.replay_block = false;
        Ok(())
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
                        // A Register source is the pack (DST tile -> L1 CB).
                        // A DRAM param is the reader publish. Both are `Copy`;
                        // the source side tells them apart. The pack is a tile
                        // move, not an SSA store, and it lives in the compute
                        // section while its words join the T2 pack role.
                        if matches!(sec.k.at(*src), Op::Storage { scope: MemScope::Register, .. }) {
                            if sec.section != 1 {
                                return Err(sec.err(format!("pack {id:?} outside the compute section")));
                            }
                            let row = sec.dst_row(*src)?;
                            let out_fmt = tt_fmt(sec.k.dtype(*dst))?;
                            use tt_isa::backend::{Before, block, cond, stallwait, wait_for_packer};
                            use tt_isa::cfg::generated::{global, thcon};
                            let l1 = sec.cb_l1(dcb)?;
                            let mut words = tt_isa::backend::ConfigWords::new();
                            crate::codegen::tenstorrent::datapath::pack_config(&mut words, l1);
                            words
                                .set(global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset, row)
                                .map_err(|e| sec.err(format!("pack {id:?}: dst offset: {e:?}")))?;
                            words
                                .set(thcon::THCON_SEC0_REG1_Out_data_format, out_fmt)
                                .map_err(|e| sec.err(format!("pack {id:?}: out format: {e:?}")))?;
                            let mut code =
                                vec![wait_for_packer(Before::CONFIG).map_err(|e| sec.err(format!("pack {id:?}: {e:?}")))?];
                            code.extend(crate::codegen::tenstorrent::datapath::config_program(&words));
                            code.push(
                                stallwait(block::PACKER | block::CONFIG, cond::CONFIG_BUSY)
                                    .map_err(|e| sec.err(format!("pack {id:?}: {e:?}")))?,
                            );
                            code.extend(crate::codegen::tenstorrent::datapath::pack_rows(
                                crate::codegen::tenstorrent::datapath::TILE_DST_ROWS,
                            ));
                            sec.native_t2.extend(code.iter().map(|i| i.word()));
                            sec.last_role = 2;
                            // A pack is one op, not the end of the section:
                            // the register hand-off (`tt_reg_sync`) can put
                            // another tile's pack after this one.
                            continue;
                        }
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
                                sec.dst_row(*dst)?
                            }
                            _ => return Err(sec.err(format!("unpack {id:?} dst is not a Register storage"))),
                        };
                        // The unpacker reads datums at `base + 16` (one tile
                        // header), and zyx CBs hold the datum stream alone, so
                        // the DST path needs the same header compensation the
                        // Src path does (`tt_src_base`). Without it the tile is
                        // shifted by one 16-byte unit (four F32 datums).
                        let l1 = tt_src_base(sec.cb_l1(scb)?);
                        let words = crate::codegen::tenstorrent::datapath::unpack_tile_to_dst(l1, dst);
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
            Op::TT(TTOp::Unpacr { src, unit, count, flip }) => {
                use crate::codegen::tenstorrent::datapath::{Unpacker, config_program, set_adc_x, tile_base_units};
                use tt_isa::cfg::generated::{alu, thcon};
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
                        DType::F32 => crate::codegen::tenstorrent::matmul::SrcRoute::Tf32FromFp32,
                        DType::BF16 => crate::codegen::tenstorrent::matmul::SrcRoute::Bf16FromBf16,
                        dt => {
                            return Err(sec.err(format!("unpacr {id:?}: no Src route from {dt:?} (F32 and BF16 are wired up)")));
                        }
                    };
                    let (in_fmt, out_fmt) = route.formats();
                    crate::codegen::tenstorrent::datapath::unpack_src_config(
                        &mut words,
                        unpacker,
                        crate::codegen::tenstorrent::matmul::tile_descriptor(in_fmt),
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
                    words.set(field, tile_base_units(l1)).map_err(|e| sec.err(format!("unpacr {id:?}: base address: {e:?}")))?;
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
                    code.push(set_adc_x(unpacker, 0, crate::codegen::tenstorrent::matmul::FACE_DATUMS - 1));
                    sec.src_unpacker_ready[*unit as usize] = true;
                }
                code.push(crate::codegen::tenstorrent::datapath::unpack_src_instruction(unpacker, *flip));
                sec.native_t0.extend(code.iter().map(|i| i.word()));
                sec.last_role = 0;
                sec.has_src_unpack = true;
            }
            Op::TT(TTOp::Setadczw { unit, face }) => {
                use tt_isa::isa::generated::encode;
                let z = encode::Setadczw::ZERO.z0(1).z0_val(*face);
                let insn = match tt_unpacker(*unit).map_err(|e| sec.err(format!("setadczw {id:?}: {e}")))? {
                    crate::codegen::tenstorrent::datapath::Unpacker::SrcA => z.u0(1),
                    crate::codegen::tenstorrent::datapath::Unpacker::SrcB => z.u1(1),
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
                let insn = crate::codegen::tenstorrent::datapath::set_adc_x(
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
                    Op::Storage { scope: MemScope::Register, .. } => sec.dst_row(*dst)?,
                    _ => return Err(sec.err(format!("setrwc {id:?} dst {dst:?} is not a Register storage"))),
                };
                sec.acc_row = row;
                let insn = encode::Setrwc::ZERO
                    .src_a(1)
                    .src_a_val(crate::codegen::tenstorrent::matmul::SRC_A_ROW as u32)
                    .src_b(1)
                    .src_b_val(crate::codegen::tenstorrent::matmul::SRC_B_ROW as u32)
                    .dst(1)
                    // `SETRWC`'s `DstVal` is four bits, so one unit is a 32-row
                    // half-tile, not a row: a `Dst` row cannot be encoded at
                    // all. Pinned by `setrwc_names_the_accumulator_in_dst_halves`.
                    .dst_val(0)
                    .fidelity(1)
                    .encode()
                    .map_err(|e| sec.err(format!("setrwc {id:?}: {e:?}")))?;
                sec.native.push(insn.word());
                sec.last_role = 1;
            }
            Op::TT(TTOp::Zeroacc { mode, imm10 }) => {
                if *mode == 3 && sec.compute_ops > 0 {
                    // `CLR_ALL` wipes every `Dst` row, including live SSA
                    // values from earlier compute ops in a fused kernel (two
                    // matmuls feeding one add, say). Clear only this
                    // accumulator's own tile instead: 64 `Dst` rows from the
                    // `Setrwc` base, eight F32 rows per `CLR_16` block
                    // (`UseDst32b = 0`, block = `row / 8`). That is the recipe
                    // tt-metal measured on silicon (`step97`: one-row clears
                    // invalidate earlier rows; sixteen-physical-row blocks at
                    // `dst_row / 8` survive).
                    for block in 0..8 {
                        let insn = tt_isa::isa::generated::encode::zeroacc(1, 0, 0, sec.acc_row / 8 + block)
                            .map_err(|e| sec.err(format!("zeroacc {id:?}: {e:?}")))?;
                        sec.native.push(insn.word());
                    }
                } else {
                    let insn = tt_isa::isa::generated::encode::zeroacc(*mode as u32, 0, 0, *imm10)
                        .map_err(|e| sec.err(format!("zeroacc {id:?}: {e:?}")))?;
                    sec.native.push(insn.word());
                }
                sec.last_role = 1;
            }
            Op::TT(TTOp::Mvmul { dst_row, addr_mod, flip_a, flip_b, broadcast_b }) => {
                let insn = tt_isa::isa::generated::encode::Mvmul::ZERO
                    .dst_row(*dst_row + sec.acc_row)
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
                let addr = u16::try_from(*field)
                    .map_err(|_| sec.err(format!("setc16 {id:?}: {field:#x} is not a ThreadConfig word")))?;
                let insn =
                    tt_isa::backend::set_thread_entry(addr, *value).map_err(|e| sec.err(format!("setc16 {id:?}: {e:?}")))?;
                sec.native.push(insn.word());
                sec.last_role = 1;
            }
            Op::TT(TTOp::Setcfg { addr, value }) => {
                // A whole `Config` word: `seed` leaves every bit as given, so
                // the caller owns the word exactly (no field masks involved).
                let mut words = tt_isa::backend::ConfigWords::new();
                words
                    .seed(
                        u16::try_from(*addr).map_err(|_| sec.err(format!("setcfg {id:?}: {addr:#x} is not a Config word")))?,
                        *value,
                    )
                    .map_err(|e| sec.err(format!("setcfg {id:?}: {e:?}")))?;
                let program = crate::codegen::tenstorrent::datapath::config_program(&words);
                sec.native.extend(program.iter().map(|i| i.word()));
                sec.last_role = 1;
            }
            Op::TT(TTOp::Post { sem, role }) => {
                let sem = tt_sem(*sem).map_err(|e| sec.err(format!("post {id:?}: {e}")))?;
                // A post announces that the posting stream's work is done, so
                // the unit that produced it must have drained before the
                // consumer sees it. T1 owns two DST-writing units (Matrix and
                // SFPU) and a role names the stream, not the unit: drain both.
                // Draining a unit with no work costs nothing.
                if *role == 1 {
                    use tt_isa::backend::{Before, wait_for_matrix, wait_for_sfpu};
                    for insn in [wait_for_matrix(Before::EVERYTHING), wait_for_sfpu(Before::EVERYTHING)] {
                        let insn = insn.map_err(|e| sec.err(format!("post {id:?}: {e:?}")))?;
                        sec.push_role_word_for(*role, insn.word());
                    }
                    sec.push_role_word_for(*role, tt_isa::sync::post(sem).word());
                } else {
                    let unit = tt_unit(*role).ok_or_else(|| sec.err(format!("post {id:?}: no stream {role}")))?;
                    for insn in tt_isa::sync::post_after(unit, sem) {
                        sec.push_role_word_for(*role, insn.word());
                    }
                }
            }
            Op::TT(TTOp::WaitBelowMax { sem, before, role }) => {
                // The producer's back-pressure: hold until the counter has
                // room, so a 4-bit hand-off cannot wrap when one role runs
                // ahead. No take: the matching `Post` follows.
                let sem = tt_sem(*sem).map_err(|e| sec.err(format!("wait below max {id:?}: {e}")))?;
                let mask =
                    tt_before(*before).ok_or_else(|| sec.err(format!("wait below max {id:?}: unknown before mask {before}")))?;
                if tt_unit(*role).is_none() {
                    return Err(sec.err(format!("wait below max {id:?}: no stream {role}")));
                }
                let insn = tt_isa::sync::wait_below_max(sem, mask);
                sec.push_role_word_for(*role, insn.word());
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
                    Op::Storage { scope: MemScope::Register, .. } => sec.dst_row(*a)?,
                    _ => return Err(sec.err(format!("Sfpu {id:?} first operand {a:?} is not a Register storage"))),
                };
                let b_row = if b.is_null() {
                    None
                } else {
                    match sec.k.at(*b) {
                        Op::Storage { scope: MemScope::Register, .. } => Some(sec.dst_row(*b)?),
                        _ => return Err(sec.err(format!("Sfpu {id:?} second operand {b:?} is not a Register storage"))),
                    }
                };
                let slot_row = sec.dst_row(*slot)?;
                let slot_idx = sec.reg_slot(*slot)?;
                sec.slots.insert(id, slot_idx);
                let code = tt_sfpu_program(*kind, a_row, b_row, slot_row)?;
                for insn in code {
                    sec.native.push(insn.word());
                }
                sec.compute_ops += 1;
            }
            Op::TT(TTOp::SfpLoad(load)) => {
                // One row-level load: the tile's row base plus the address
                // inside the tile.
                let at = sec.dst_row(load.slot)? + u32::from(load.at);
                let insn = tt_isa::sfpu::load(u32::from(load.reg), u32::from(load.fmt), 0, at)
                    .map_err(|e| sec.err(format!("sfp_load {id:?}: {e:?}")))?;
                sec.native.push(insn.word());
                sec.compute_ops += 1;
            }
            Op::TT(TTOp::SfpStore(store)) => {
                let at = sec.dst_row(store.slot)? + u32::from(store.at);
                let insn = tt_isa::sfpu::store(u32::from(store.reg), u32::from(store.fmt), 0, at)
                    .map_err(|e| sec.err(format!("sfp_store {id:?}: {e:?}")))?;
                sec.native.push(insn.word());
                sec.compute_ops += 1;
            }
            Op::TT(TTOp::SfpConst(c)) => {
                // `SFPLOADI` when the low half is zero, the two-instruction
                // `load_f32` form otherwise (as the builder does).
                let reg = u32::from(c.reg);
                if c.bits & 0xffff == 0 {
                    let insn = tt_isa::sfpu::loadi(reg, tt_isa::sfpu::loadi_mode::FLOATB, c.bits >> 16)
                        .map_err(|e| sec.err(format!("sfp_const {id:?}: {e:?}")))?;
                    sec.native.push(insn.word());
                } else {
                    for insn in tt_isa::sfpu::load_f32(reg, c.bits).map_err(|e| sec.err(format!("sfp_const {id:?}: {e:?}")))? {
                        sec.native.push(insn.word());
                    }
                }
                sec.compute_ops += 1;
            }
            Op::TT(TTOp::Sfp(insn)) => {
                let encoded = tt_sfp_encode(insn)?;
                sec.native.push(encoded.word());
                sec.compute_ops += 1;
            }
            Op::TT(TTOp::Replay { len, count }) => sec.begin_replay(*len, *count)?,
            Op::TT(TTOp::EndReader) | Op::TT(TTOp::EndCompute) => {}
            Op::Barrier => return Err(sec.err("stray barrier in a TT kernel".to_string())),
            _ => return Err(sec.err(format!("{id:?} is not fully lowered (SSA remains)"))),
        }
        sec.finish_replay()?;
    }
    if sec.replay.is_some() {
        return Err(sec.err("a replay body is truncated by the section end".to_string()));
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
        // Every CB starts 16 bytes *into* its alignment block: the datums then
        // sit at ≡16 mod 64, which is where a tile image's datum block lives
        // (`tt_isa::dm::TILE_DATA`). A plain read copies that datum block, and
        // a shaped read (`READ_BROADCAST_COL`) writes header+datums at
        // `base - 16`, so the 16 bytes of headroom must be inside the arena.
        let base = ((at + 63) & !63) + tt_isa::dm::TILE_DATA;
        let end = base + len;
        if end > tt_isa::l1::DATA.end || base < tt_isa::l1::DATA.base {
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

/// Byte offset and per-grid-axis byte strides of a boundary move's slice.
///
/// The base is the index's constant term. For an index `tt_reindex` already
/// rewrote to physical tile units, the grid strides are the index's own (a
/// grid step there is a whole number of tiles: the reduce loop makes a row
/// group advance `ntc` tiles, which the extent-only map cannot know).
/// Otherwise the index is a *logical* one and the strides come from the
/// physical tile map: one grid point owns exactly one tile (the unit phase 1
/// sizes CBs, slots and byte counts in), so a group step advances by 1024
/// datums times the extents of the higher-axis groups — the same map
/// `strip_unrolls` uses to place reader and writer slices. Trusting a logical
/// folded index here reads a 32-tile stride where one tile is meant.
fn tt_slice(
    k: &Kernel,
    idx: OpId,
    elem_bytes: u32,
    bind: &Map<OpId, i64>,
) -> Result<(u32, u32, u32), BackendError> {
    let strides = k.get_strides(idx).ok_or_else(|| tt_err(format!("tenstorrent: boundary index {idx:?} is not affine")))?;
    for var in strides.keys() {
        if *var == OpId::NULL || bind.contains_key(var) {
            continue;
        }
        let is_group = matches!(k.at(*var), Op::Range { kind: RangeKind::Group(_), .. });
        if !is_group {
            return Err(tt_err(format!(
                "tenstorrent: boundary index {idx:?} keeps a variable {var:?} that is neither a grid axis nor a bound loop"
            )));
        }
    }
    // A bound loop variable is one concrete iteration, so its term joins the
    // base instead of becoming a grid stride.
    let mut base = strides.get(&OpId::NULL).map_or(0, |(_, s)| *s);
    for (var, (_, stride)) in strides.iter() {
        if let Some(value) = bind.get(var) {
            base = value
                .checked_mul(*stride)
                .and_then(|t| base.checked_add(t))
                .ok_or_else(|| tt_err("tenstorrent: a bound loop's boundary offset overflows".to_string()))?;
        }
    }
    let mut per_axis = [0i64; 2];
    if k.tt_index_is_tile_units(idx) {
        // The rewritten index is the truth: each grid term is its physical
        // tile stride.
        for (var, (_, stride)) in strides.iter() {
            if *var == OpId::NULL || bind.contains_key(var) {
                continue;
            }
            let Op::Range { axis, kind: RangeKind::Group(_), .. } = k.at(*var) else {
                return Err(tt_err(format!("tenstorrent: boundary index {idx:?} keeps a non-grid variable {var:?}")));
            };
            let slot = per_axis
                .get_mut(*axis as usize)
                .ok_or_else(|| tt_err(format!("tenstorrent: grid axis {axis} is not dispatched")))?;
            *slot = *stride;
        }
    } else {
        // Only the grid axes **this index names** move the slice: a `[32]`
        // row vector is addressed by the row group alone, so the column grid
        // axis must not add a stride (that overshot the param by a whole
        // row of tiles). Unnamed axes keep a zero stride.
        let mut axes: Vec<(u32, i64)> = Vec::new();
        for var in strides.keys() {
            if *var == OpId::NULL || bind.contains_key(var) {
                continue;
            }
            let Op::Range { axis, kind: RangeKind::Group(len), .. } = k.at(*var) else {
                return Err(tt_err(format!("tenstorrent: boundary index {idx:?} keeps a non-grid variable {var:?}")));
            };
            let extent = k.resolve_const(*len).and_then(Constant::as_dim).ok_or_else(|| {
                tt_err("tenstorrent: a grid extent is symbolic; boundary slices need a concrete grid".to_string())
            })?;
            axes.push((*axis, extent));
        }
        axes.sort();
        let mut suffix = 1024i64;
        for (axis, extent) in axes.iter().rev() {
            let slot = per_axis
                .get_mut(*axis as usize)
                .ok_or_else(|| tt_err(format!("tenstorrent: grid axis {axis} is not dispatched")))?;
            *slot = suffix;
            suffix = suffix.checked_mul(*extent).ok_or_else(|| tt_err("tenstorrent: grid slice stride overflows".to_string()))?;
        }
    }
    // Element offsets are whole tiles here (`strip_unrolls` zeroes the lane
    // axes), so they scale by the *tile image* pitch, not the datum width:
    // one tile advances `tt_isa::dm::TILE_SLOT`, and the datum block starts
    // `tt_isa::dm::TILE_DATA` bytes into it. The caller adds the datum offset;
    // this returns the slot base (what a shaped read starts at).
    let to_bytes = |v: i64| -> Result<u32, BackendError> {
        if v % 1024 != 0 {
            return Err(tt_err(format!("tenstorrent: boundary offset {v} is not a whole tile")));
        }
        let slot_bytes = tt_isa::dm::TILE_SLOT as i64;
        let b =
            (v / 1024).checked_mul(slot_bytes).ok_or_else(|| tt_err(format!("tenstorrent: boundary offset {v} overflows")))?;
        let _ = elem_bytes;
        u32::try_from(b).map_err(|_| tt_err(format!("tenstorrent: boundary offset {b} bytes does not fit u32")))
    };
    Ok((to_bytes(base)?, to_bytes(per_axis[0])?, to_bytes(per_axis[1])?))
}

/// The L1 byte offset a boundary copy's buffer-side index names, for one
/// iteration of every bound loop.
///
/// The buffer side never moves with the grid: a grid axis there would mean a
/// per-core slot, which the plan does not express. So every variable the index
/// names must be a bound loop (or the constant term).
fn tt_cb_offset(k: &Kernel, idx: OpId, elem_bytes: u32, bind: &Map<OpId, i64>) -> Result<u32, BackendError> {
    if idx.is_null() {
        return Ok(0);
    }
    let strides = k.get_strides(idx).ok_or_else(|| tt_err(format!("tenstorrent: buffer index {idx:?} is not affine")))?;
    let mut elems = strides.get(&OpId::NULL).map_or(0, |(_, s)| *s);
    for (var, (_, stride)) in strides.iter() {
        if *var == OpId::NULL {
            continue;
        }
        let Some(value) = bind.get(var) else {
            return Err(tt_err(format!(
                "tenstorrent: buffer index {idx:?} names {var:?}, which is neither the constant term nor a bound loop"
            )));
        };
        elems = value
            .checked_mul(*stride)
            .and_then(|t| elems.checked_add(t))
            .ok_or_else(|| tt_err("tenstorrent: a buffer slot offset overflows".to_string()))?;
    }
    let bytes = elems
        .checked_mul(i64::from(elem_bytes))
        .ok_or_else(|| tt_err("tenstorrent: a buffer slot offset overflows".to_string()))?;
    u32::try_from(bytes).map_err(|_| tt_err(format!("tenstorrent: buffer offset {bytes} does not fit u32")))
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
    // A boundary copy inside a constant-trip loop is one entry per iteration:
    // the mover list is a static array, so a loop there is unrolled here.
    let mut loops: Vec<(OpId, i64)> = Vec::new();
    let mut op_id = k.head;
    while !op_id.is_null() {
        match k.at(op_id) {
            Op::TT(TTOp::EndReader) => section = 1,
            Op::TT(TTOp::EndCompute) => section = 2,
            Op::Loop { len } => {
                let trip = k
                    .resolve_const(*len)
                    .and_then(Constant::as_dim)
                    .filter(|t| *t > 0)
                    .ok_or_else(|| tt_err(format!("tenstorrent: boundary loop {op_id:?} needs a positive constant trip count")))?;
                loops.push((op_id, trip));
            }
            Op::EndLoop => {
                loops.pop();
            }
            Op::TT(TTOp::Copy { src, src_idx, dst, dst_idx, bytes }) if section != 1 => {
                let (param, cb, idx, cb_idx) = match section {
                    0 => (*src, *dst, *src_idx, *dst_idx),
                    2 => (*dst, *src, *dst_idx, *src_idx),
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
                // Which slice of the param this move touches, and how it
                // moves with the grid: the program is tile-uniform, so this
                // is the only per-core difference.
                let elem_bytes = k.dtype(param).bit_size() as u32 / 8;
                let tile_bytes = elem_bytes * 1024;
                let cb_elem_bytes = k.dtype(cb).bit_size() as u32 / 8;
                let slot = tt_isa::dm::TILE_SLOT as u32;
                let data = tt_isa::dm::TILE_DATA as u32;
                // One DMA entry carries one *tile image*: tiles are not
                // contiguous on the DRAM side (16-byte header, 4160 pitch), so
                // a block publish/drain is one entry per tile. A `TILE_SLOT`
                // publish is the mover's shaped read of one whole image
                // (`READ_BROADCAST_COL`) and stays one entry.
                let shaped = *bytes == slot;
                let n = if shaped {
                    1
                } else {
                    bytes.checked_div(tile_bytes).ok_or_else(|| tt_err("tenstorrent: zero-length boundary move".to_string()))?
                };
                if !shaped && n * tile_bytes != *bytes {
                    return Err(tt_err(format!(
                        "tenstorrent: boundary move {op_id:?} is {bytes} bytes, not a whole number of {tile_bytes}-byte tiles"
                    )));
                }
                // One entry per combination of the enclosing boundary loops'
                // iterations; no loops means the one empty binding.
                let mut combos: Vec<Map<OpId, i64>> = vec![Map::default()];
                for (var, trip) in &loops {
                    let mut next = Vec::with_capacity(combos.len().saturating_mul(*trip as usize));
                    for bind in &combos {
                        for i in 0..*trip {
                            let mut b = bind.clone();
                            b.insert(*var, i);
                            next.push(b);
                        }
                    }
                    combos = next;
                }
                for bind in &combos {
                    let (base, gstride0, gstride1) = tt_slice(k, idx, elem_bytes, bind)?;
                    // The buffer side is a slot inside the CB, never a grid
                    // slice: only a bound loop moves it.
                    let cb_off = if loops.is_empty() { 0 } else { tt_cb_offset(k, cb_idx, cb_elem_bytes, bind)? };
                    for t in 0..n {
                        // A shaped read writes the header too, so its L1 base is
                        // the slot start, 16 bytes below the CB's datums.
                        let l1 = if shaped {
                            l1[cb_num as usize] - data
                        } else {
                            l1[cb_num as usize] + cb_off + t * tile_bytes
                        };
                        let base = if shaped { base } else { base + t * slot + data };
                        let bytes = if shaped { slot } else { tile_bytes };
                        let move_ = TTDmMove { param: ord, l1, bytes, narrow, base, gstride0, gstride1 };
                        match section {
                            0 => reader.push(move_),
                            2 => writer.push(move_),
                            _ => unreachable!("section is 0 or 2 in this arm"),
                        }
                        let entries = if section == 0 { reader.len() } else { writer.len() };
                        if entries > tt_isa::dm::LIST_MAX as usize {
                            return Err(tt_err(format!(
                                "tenstorrent: {entries} boundary entries exceed the mover list (LIST_MAX {})",
                                tt_isa::dm::LIST_MAX
                            )));
                        }
                    }
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
fn tt_unpacker(unit: u8) -> Result<crate::codegen::tenstorrent::datapath::Unpacker, String> {
    match unit {
        0 => Ok(crate::codegen::tenstorrent::datapath::Unpacker::SrcA),
        1 => Ok(crate::codegen::tenstorrent::datapath::Unpacker::SrcB),
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
/// `SETRWC` clearing the issuing thread's `Dst` row counter (`tt_reg_sync`'s
/// row loops clear it before the first body and after the last repeat).
fn setrwc_clear() -> u32 {
    tt_isa::isa::generated::encode::Setrwc::ZERO.dst(1).encode().unwrap().word()
}

fn tt_unpack_role(body: Vec<u32>, src_mode: bool) -> Vec<u32> {
    use tt_isa::backend::{Before, wait_for_unpacker0, wait_for_unpacker1};
    if body.is_empty() {
        return Vec::new();
    }
    let config = if src_mode {
        crate::codegen::tenstorrent::datapath::src_thread_config()
    } else {
        crate::codegen::tenstorrent::datapath::thread_config()
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
/// emission order, then an SFPU drain. Also returns how many words precede the
/// body: a block repeat's `start` indexes this array, not the body.
fn tt_math_role(body: Vec<u32>) -> (Vec<u32>, u32) {
    use tt_isa::backend::{Before, wait_for_sfpu};
    if body.is_empty() {
        return (Vec::new(), 0);
    }
    let mut role: Vec<u32> = vec![crate::codegen::tenstorrent::datapath::state_id().word()];
    role.extend(body);
    role.push(wait_for_sfpu(Before::EVERYTHING).expect("wait_for_sfpu(EVERYTHING) is encodable").word());
    (role, 1)
}

/// T2, the pack role: the state id, then one self-contained DST -> CB pack
/// per `Store`, then a packer drain (the thread unblocks once the packer
/// accepts work, not when it finishes).
fn tt_pack_role(body: Vec<u32>) -> Vec<u32> {
    use tt_isa::backend::{Before, wait_for_packer};
    if body.is_empty() {
        return Vec::new();
    }
    let mut role: Vec<u32> = vec![crate::codegen::tenstorrent::datapath::state_id().word()];
    role.extend(body);
    role.push(wait_for_packer(Before::EVERYTHING).expect("wait_for_packer(EVERYTHING) is encodable").word());
    role
}

fn render(
    k: &Kernel,
    param_ordinal_of: &Map<OpId, u32>,
    cbs: &Map<OpId, u32>,
    l1: &[u32],
    sems: &mut Vec<TTSemaphoreInit>,
) -> Result<(String, String, String, [Vec<u32>; 3], [Vec<[u32; 3]>; 3]), BackendError> {
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
    // The compute core's three programs, filled by the compute section below.
    let mut roles: [Vec<u32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut loops: [Vec<[u32; 3]>; 3] = [Vec::new(), Vec::new(), Vec::new()];
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
            l1,
            native_t0: Vec::new(),
            native_t2: Vec::new(),
            replay: None,
            replay_block: false,
            last_role: 1,
            loops: [Vec::new(), Vec::new(), Vec::new()],
            sems: Vec::new(),
            compute_ops: 0,
            acc_row: 0,
            src_unpacker_ready: [false, false],
            has_src_unpack: false,
        };
        // The DST register file is the ordered list of physical Register
        // storages the allocator declared in the prologue: slot `i` is the
        // i-th declaration. Encoding it here (rather than on first use) is
        // what makes the IR's register file authoritative — ops name a
        // storage by OpId, and codegen resolves that to the same row on
        // every section and every launch.
        let mut scan = k.head;
        for _ in 0..250_000 {
            if scan.is_null() {
                break;
            }
            if matches!(k.at(scan), Op::Storage { scope: MemScope::Register, .. }) {
                if sec.next_slot >= sec.budget {
                    return Err(tt_err(format!(
                        "tenstorrent: DST register file holds {} tiles, budget is {}",
                        sec.next_slot + 1,
                        sec.budget
                    )));
                }
                sec.slots.insert(scan, sec.next_slot);
                sec.next_slot += 1;
            }
            scan = k.next_op(scan);
        }
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
            // Role streams, T0/T1/T2 in order. The roles run in order
            // (`Schedule::InOrder`), so no semaphores: each role drains its
            // unit before the firmware reports it done. The three streams
            // become `TTOp::Unpack`/`Math`/`Pack`: codegen is the encoder.
            let (math, preamble) = tt_math_role(section_native);
            roles = [tt_unpack_role(section_t0, sec.has_src_unpack), math, tt_pack_role(section_t2)];
            *sems = std::mem::take(&mut sec.sems);
            loops = std::mem::take(&mut sec.loops);
            for entry in &mut loops[1] {
                entry[0] += preamble;
            }
        } else {
            debug_assert!(section_native.is_empty(), "native compute words outside the compute section");
            debug_assert!(section_t0.is_empty(), "native unpack words outside the compute section");
            debug_assert!(section_t2.is_empty(), "native pack words outside the compute section");
        }
        srcs[s] = sec.out;
    }

    Ok((srcs[0].clone(), srcs[1].clone(), srcs[2].clone(), roles, loops))
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
            // The packer's offset names a DST **row**; the C++ fallback text
            // below still names a slot, so both are resolved here.
            let (slot, row) = match sec.k.at(x) {
                Op::Storage { scope: MemScope::Register, .. } => (sec.reg_slot(x)?, sec.dst_row(x)?),
                _ => {
                    let s = sec.slot_of(x)?;
                    (s, s * 64)
                }
            };
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
            crate::codegen::tenstorrent::datapath::pack_config(&mut words, l1);
            words
                .set(global::DEST_TARGET_REG_CFG_PACK_SEC0_Offset, row)
                .map_err(|e| sec.err(format!("pack {id:?}: dst offset: {e:?}")))?;
            words
                .set(thcon::THCON_SEC0_REG1_Out_data_format, out_fmt)
                .map_err(|e| sec.err(format!("pack {id:?}: out format: {e:?}")))?;
            let mut code = vec![wait_for_packer(Before::CONFIG).map_err(|e| sec.err(format!("pack {id:?}: {e:?}")))?];
            code.extend(crate::codegen::tenstorrent::datapath::config_program(&words));
            code.push(
                stallwait(block::PACKER | block::CONFIG, cond::CONFIG_BUSY)
                    .map_err(|e| sec.err(format!("pack {id:?}: {e:?}")))?,
            );
            code.extend(crate::codegen::tenstorrent::datapath::pack_rows(crate::codegen::tenstorrent::datapath::TILE_DST_ROWS));
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

    /// `SETRWC`'s `DstVal` is **four bits**, so one unit is a 32-row half of a
    /// 32-bit `Dst` tile: an accumulator at allocation index 1 (rows 64..128)
    /// encodes 2. The raw row does not fit the field, which is why the codegen
    /// divides — and why a head-room assumption here must be a test, not a
    /// comment.
    #[test]
    fn setrwc_names_the_accumulator_in_dst_halves() {
        use crate::backend::{DTypeCapability, DeviceInfo};
        use crate::shape::Dim;
        use std::sync::Arc;
        let mut k = Kernel::new(crate::Dev::Auto);
        // `generate_tenstorrent` reads one device property (the CB count); the
        // test only needs a bound device, not a real one.
        k.dev_info = Some(Arc::new(DeviceInfo {
            compute: 1_000_000,
            max_global_work_dims: vec![Dim::from(1_000_000_000); 3],
            max_local_threads: 1024,
            max_local_work_dims: vec![1024, 1024, 1024],
            preferred_vector_size: 8,
            local_mem_size: 8192,
            max_register_bytes: 256_000,
            tensor_cores: false,
            warp_size: 1,
            cc: [0, 0],
            dtype_capability: [DTypeCapability::all(); DType::N_DTYPES],
            has_native_exp2: false,
            supported_vec_lens: vec![2, 4, 8, 16],
            vector_depth: 1,
            tenstorrent: true,
            tile: [Dim::from(32), Dim::from(32)],
            tile_sizes: vec![[Dim::from(32), Dim::from(32)]],
            mma_variants: vec![],
            num_circular_buffers: 4,
            has_openmp: false,
        }));
        let storage = k.push_back(Op::Storage { dtype: DType::F32, scope: MemScope::Register, len: 1024, id: 1 });
        k.push_back(Op::TT(TTOp::EndReader));
        k.push_back(Op::TT(TTOp::Setrwc { dst: storage }));
        k.push_back(Op::TT(TTOp::EndCompute));
        let program = k.generate_tenstorrent().expect("a minimal math-only kernel renders");
        let mut setrwc = program.roles[1]
            .iter()
            .map(|&word| tt_isa::isa::Instruction::new(word, &tt_isa::isa::generated::defs::SETRWC))
            .filter(|insn| insn.def().mnemonic() == "SETRWC");
        let insn = setrwc.find(|insn| insn.operand("Dst") == Some(1)).expect("the accumulator pointer is in the math stream");
        // `DstVal` must stay zero: each `MVMUL` carries its accumulator row
        // absolutely in `DstRow`, and the hardware adds `DstVal` on top (ttsim
        // adds it raw, and a second accumulator at row 64 failed as
        // `dst_row=66` when both encoded the base). `tt-metal`'s own
        // `math_prelude` sets `dst_val(0)` for the same reason.
        assert_eq!(insn.operand("DstVal"), Some(0));
        // The row itself cannot be encoded anyway: `DstVal` is 0..15.
        assert!(tt_isa::isa::generated::encode::Setrwc::ZERO.dst(1).dst_val(64).encode().is_err());
    }
}
