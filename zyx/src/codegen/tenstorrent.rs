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
use crate::backend::{GwsDim, gws_from_kernel};
use crate::dtype::Constant;
use crate::error::{BackendError, ErrorStatus};
use crate::kernel::{
    BOp, GPUOp, Kernel, MemLayout, MemScope, Op, OpId, ParamKind, RangeKind, SourceBlock, TTCbConfig, TTOp, TTProgramDesc,
    TileDim, UOp,
};
use crate::scalar::{bf16, f16};
use crate::slab::Slab;

/// DRAM page size in bytes.
pub(crate) const TT_DRAM_PAGE_BYTES: u32 = 4096;

/// TT `DataFormat` code for a dtype on the tile path (the
/// `typecast_tile_init<in, out>` template args). Distinct from the
/// CB descriptor codes in `generate_tenstorrent` below: the two
/// numberings differ.
fn tt_fmt(dtype: DType) -> Result<u32, BackendError> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 | DType::BF16 => Ok(5),
        DType::I32 => Ok(8),
        DType::U16 => Ok(9),
        DType::I8 => Ok(14),
        DType::U32 => Ok(24),
        DType::F8E4M3 => Ok(26),
        DType::U8 => Ok(30),
        dt => Err(BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("tenstorrent: dtype {dt:?} has no tt tile format").into(),
        }),
    }
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
    /// `EndCompute` markers. Lowering + emission runs once here via
    /// `generate_tenstorrent` (which verifies the lowered IR);
    /// backend `compile` only decodes positions and drives the shim.
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
            fp32: program.fp32,
        };
        let mut rendered = Kernel {
            ops: Slab::new(),
            head: OpId::NULL,
            tail: OpId::NULL,
            dev: self.dev,
            dev_info: self.dev_info.clone(),
            shape_cache: Map::default(),
        };
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
        let mut k = self.clone();
        // Lowering order: fused LLK claiming (sigmoid/silu composites
        // become opaque calls), storage (compute ops become LLK calls over
        // bare storages, broadcast fusion), locks, duplicate-publish
        // elimination, engine-config inits + startup + reduce cones, CB
        // sync, pop placement + FIFO check. Render consumes the result 1:1.
        k.tt_fuse_llks();
        k.tt_storage();
        k.tt_lock_dst();
        k.tt_dedup_pushes();
        k.tt_init_math()?;
        k.tt_sync_cbs();
        k.tt_place_pops();
        k.verify();
        k.debug();

        // Param ordinals + input/output dtypes, flat head order.
        let mut param_ordinal_of: Map<OpId, u32> = Map::default();
        let mut next_param = 0u32;
        let mut input_dtypes: Vec<DType> = Vec::new();
        let mut output_dtypes: Vec<DType> = Vec::new();
        let mut scan = k.head;
        for _ in 0..10_000 {
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
                context: "tenstorrent: param scan did not finish in 10000 steps".into(),
            });
        }

        // CB allocation: first touch over loads/stores/copies of
        // Circular storages (chased through their GEPs), with the
        // legacy validity checks.
        let mut cbs: Map<OpId, u32> = Map::default();
        let mut cb_order: Vec<OpId> = Vec::new();
        let mut scan = k.head;
        for _ in 0..10_000 {
            if scan.is_null() {
                break;
            }
            let touched = match k.at(scan) {
                Op::Load { src, .. } | Op::Copy { src, .. } => Some(k.tt_storage_of(*src)),
                Op::Store { dst, .. } => Some(k.tt_storage_of(*dst)),
                _ => None,
            };
            if let Some(st) = touched
                && matches!(k.at(st), Op::Storage { scope: MemScope::Circular, .. })
                && !cbs.contains_key(&st)
            {
                cbs.insert(st, cb_order.len() as u32);
                cb_order.push(st);
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
        let (reader_src, compute_src, writer_src, reader_params, compute_params, writer_params, fp32) =
            render(&k, &param_ordinal_of, &cbs)?;

        // CB config per CB, in index order.
        let mut cb_config: Vec<(u32, u32, u32)> = Vec::new();
        for st in &cb_order {
            let Op::Storage { dtype, len, .. } = k.at(*st) else {
                return Err(BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("tenstorrent: cb entry {st} is not a storage op").into(),
                });
            };
            let (fmt, tb) = match dtype {
                DType::F32 => (0, 4096),
                DType::F16 => (1, 2048),
                DType::BF16 => (2, 2048),
                DType::U16 => (3, 2048),
                DType::F8E4M3 => (4, 1024),
                DType::U8 => (5, 1024),
                DType::I8 => (6, 1024),
                DType::U32 => (7, 4096),
                DType::I32 => (8, 4096),
                dt => {
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("tenstorrent: CB dtype {dt:?} has no tt format").into(),
                    });
                }
            };
            cb_config.push((fmt, tb, (len / 1024) as u32));
            debug_assert_eq!(cb_config.len() - 1, cbs[st] as usize, "tenstorrent: CB config out of sync with allocation");
        }

        Ok(TTProgram {
            reader_src,
            compute_src,
            writer_src,
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
        if let Op::Load { src } = self.k.at(v) {
            if let Op::GEP { x: base, .. } = self.k.at(*src) {
                match self.k.at(*base) {
                    Op::Storage { scope: MemScope::Circular, .. } => {
                        let cb = self.cbs.get(base).copied().ok_or_else(|| self.err(format!("CB {base:?} has no number")))?;
                        let s = self.fresh_slot()?;
                        self.out.push_str(&format!("{}copy_tile({cb}, 0, {s});\n", self.indent));
                        self.slots.insert(v, s);
                        return Ok(s);
                    }
                    Op::Storage { scope: MemScope::Register, .. } => return self.reg_slot(*base),
                    _ => {}
                }
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
        if let Op::Load { src } = self.k.at(v) {
            if let Op::GEP { x: base, .. } = self.k.at(*src) {
                if matches!(self.k.at(*base), Op::Storage { scope: MemScope::Register, .. }) {
                    return self.reg_slot(*base);
                }
            }
        }
        Err(self.err(format!("{v:?} has no tile slot")))
    }

    /// CB number of a Circular storage (sync ops carry storages).
    fn cb_num(&self, cb: OpId) -> Result<u32, BackendError> {
        self.cbs.get(&cb).copied().ok_or_else(|| self.err(format!("CB {cb:?} has no number")))
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
                    if let Op::Load { src } = chained {
                        if let Op::GEP { x: base, .. } = self.k.at(*src) {
                            if matches!(self.k.at(*base), Op::Storage { scope: MemScope::Circular, .. }) {
                                self.cbs.get(base).copied().ok_or_else(|| self.err(format!("CB {base:?} has no number")))?
                            } else if matches!(self.k.at(*base), Op::Storage { scope: MemScope::Register, .. }) {
                                self.reg_slot(*base)?
                            } else {
                                return Err(self.err(format!("asm operand {o:?} is not a CB or live tile")));
                            }
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
            Op::Const { .. } | Op::Param { .. } | Op::Storage { .. } | Op::GEP { .. } => {}
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
            Op::Copy { src, dst } => render_copy(sec, id, *src, *dst)?,
            Op::Load { src } => render_load(sec, id, *src)?,
            Op::Store { src: x, dst } => render_store(sec, id, *x, *dst)?,
            Op::Unary { .. } | Op::Binary { .. } | Op::Cast { .. } | Op::Bitcast { .. } => {
                if matches!(sec.k.layout(id), MemLayout::Tile { .. }) {
                    render_tile_op(sec, id)?;
                }
            }
            Op::Asm { asm, ops } => {
                let ops_vec: Vec<OpId> = ops.iter().copied().collect();
                let text = sec.substitute(asm.as_str(), &ops_vec)?;
                sec.out.push_str(&format!("{}{text}\n", sec.indent));
            }
            Op::TT(TTOp::LLK { asm, ops }) => {
                let text = asm.as_str().to_string();
                let ops_vec: Vec<OpId> = ops.iter().copied().collect();
                // Leading operands substitute; trailing provenance
                // loads are sync accounting (render ignores them).
                // Template arity: matmul/reduce carry a Register slot
                // at {2}; fused broadcast carries its CBs plus a NULL
                // dst filled with a fresh slot; transpose carries a
                // NULL dst filled with a fresh slot.
                let lead = if text.starts_with("matmul_tiles(") || text.starts_with("reduce_tile<") || text.contains("_bcast_") {
                    3
                } else if text.starts_with("transpose_wh_tile(") {
                    2
                } else if text.starts_with("sigmoid_tile(") || text.starts_with("silu_tile(") || text.starts_with("exp_tile(") {
                    // Fused unary: single in-place tile. The operand is
                    // the feeder value (a circular load copied in fresh
                    // at the call, exactly like the unfused `exp_tile(s)`
                    // path), never a bare storage.
                    debug_assert_eq!(ops_vec.len(), 1);
                    let s = sec.operand_slot(ops_vec[0])?;
                    let rendered = text.replace("{0}", &s.to_string());
                    if rendered.contains('{') {
                        return Err(sec.err(format!("LLK left placeholders unsubstituted: {rendered}")));
                    }
                    sec.slots.insert(id, s);
                    sec.out.push_str(&format!("{}{rendered}\n", sec.indent));
                    continue;
                } else {
                    return Err(sec.err(format!("LLK {id:?} is not a storage template")));
                };
                let mut rendered = text.clone();
                for (i, o) in ops_vec.iter().copied().enumerate().take(lead) {
                    let num = if o.is_null() {
                        sec.fresh_slot()?
                    } else {
                        match sec.k.at(o) {
                            Op::Storage { scope: MemScope::Circular, .. } => {
                                sec.cbs.get(&o).copied().ok_or_else(|| sec.err(format!("CB {o:?} has no number")))?
                            }
                            Op::Storage { scope: MemScope::Register, .. } => sec.reg_slot(o)?,
                            _ => return Err(sec.err(format!("LLK {id:?} operand {o:?} is not a storage"))),
                        }
                    };
                    rendered = rendered.replace(&format!("{{{i}}}"), &num.to_string());
                    if i == lead - 1 {
                        sec.slots.insert(id, num);
                    }
                }
                if rendered.contains('{') {
                    return Err(sec.err(format!("LLK left placeholders unsubstituted: {rendered}")));
                }
                sec.out.push_str(&format!("{}{rendered}\n", sec.indent));
            }
            Op::TT(TTOp::LLKReduce { rop, kind, cb_in, cb_sc, slot, .. }) => {
                // Structured lowered reduce: rebuild the compute
                // template from the fields (never parsed back).
                let op_name = match rop {
                    BOp::Max => "PoolType::MAX",
                    BOp::Add => "PoolType::SUM",
                    _ => return Err(sec.err(format!("LLKReduce {id:?} op {rop:?} has no LLK call"))),
                };
                let dim_name = match kind {
                    TileDim::Row => "ReduceDim::REDUCE_ROW",
                    TileDim::Col => "ReduceDim::REDUCE_COL",
                    TileDim::Scalar => "ReduceDim::REDUCE_SCALAR",
                };
                let template = format!("reduce_tile<{op_name}, {dim_name}>({{0}}, {{1}}, 0, 0, {{2}});");
                let mut rendered = template;
                for (i, o) in [*cb_in, *cb_sc, *slot].iter().copied().enumerate() {
                    let num = match sec.k.at(o) {
                        Op::Storage { scope: MemScope::Circular, .. } => {
                            sec.cbs.get(&o).copied().ok_or_else(|| sec.err(format!("CB {o:?} has no number")))?
                        }
                        Op::Storage { scope: MemScope::Register, .. } => sec.reg_slot(o)?,
                        _ => return Err(sec.err(format!("LLKReduce {id:?} operand {o:?} is not a storage"))),
                    };
                    rendered = rendered.replace(&format!("{{{i}}}"), &num.to_string());
                    if i == 2 {
                        sec.slots.insert(id, num);
                    }
                }
                if rendered.contains('{') {
                    return Err(sec.err(format!("LLKReduce left placeholders unsubstituted: {rendered}")));
                }
                sec.out.push_str(&format!("{}{rendered}\n", sec.indent));
            }
            Op::TT(TTOp::LLKBcast { bop, kind, cb_a, cb_b, .. }) => {
                // Structured lowered fused broadcast: same rebuild; the
                // result slot is DST (fresh slot, like the LLK NULL).
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
                    _ => return Err(sec.err(format!("LLKBcast {id:?} ({bop:?}, {kind:?}) has no fused call"))),
                };
                let template = if matches!(kind, TileDim::Row) {
                    format!("{name}({{0}}, {{1}}, 0, 0, {{2}}, 0);")
                } else {
                    format!("{name}({{0}}, {{1}}, 0, 0, {{2}});")
                };
                let mut rendered = template;
                for (i, o) in [*cb_a, *cb_b].iter().copied().enumerate() {
                    let num = match sec.k.at(o) {
                        Op::Storage { scope: MemScope::Circular, .. } => {
                            sec.cbs.get(&o).copied().ok_or_else(|| sec.err(format!("CB {o:?} has no number")))?
                        }
                        _ => return Err(sec.err(format!("LLKBcast {id:?} operand {o:?} is not a circular storage"))),
                    };
                    rendered = rendered.replace(&format!("{{{i}}}"), &num.to_string());
                }
                let dst = sec.fresh_slot()?;
                rendered = rendered.replace("{2}", &dst.to_string());
                sec.slots.insert(id, dst);
                if rendered.contains('{') {
                    return Err(sec.err(format!("LLKBcast left placeholders unsubstituted: {rendered}")));
                }
                sec.out.push_str(&format!("{}{rendered}\n", sec.indent));
            }
            Op::TT(TTOp::MatmulTile { .. }) | Op::TT(TTOp::ReduceTile { .. }) | Op::TT(TTOp::TransposeTile { .. }) => {
                return Err(sec.err(format!("{id:?} is not fully lowered (SSA remains)")));
            }
            // Fused broadcast marker: dead after `tt_storage` fused its
            // binary into an `LLK` (the call consumes the CBs straight
            // from the buffers). Sync accounting only — emits nothing
            // (old: markers ignored downstream).
            Op::TT(TTOp::BroadcastTile { .. }) => {}
            Op::TT(TTOp::ReserveBack { cb, n }) => {
                let c = sec.cb_num(*cb)?;
                sec.out.push_str(&format!("{}cb{c}.reserve_back({n});\n", sec.indent));
            }
            Op::TT(TTOp::PushBack { cb, n }) => {
                let c = sec.cb_num(*cb)?;
                sec.out.push_str(&format!("{}cb{c}.push_back({n});\n", sec.indent));
            }
            Op::TT(TTOp::WaitFront { cb, n }) => {
                let c = sec.cb_num(*cb)?;
                sec.out.push_str(&format!("{}cb{c}.wait_front({n});\n", sec.indent));
            }
            Op::TT(TTOp::PopFront { cb, n }) => {
                let c = sec.cb_num(*cb)?;
                sec.out.push_str(&format!("{}cb{c}.pop_front({n});\n", sec.indent));
            }
            Op::TT(TTOp::MathLock) => {
                if sec.section != 1 {
                    return Err(sec.err("DST lock outside compute".to_string()));
                }
                if sec.math_open {
                    return Err(sec.err("acquire with DST already held".to_string()));
                }
                sec.math_open = true;
                sec.out.push_str(&format!("{}tile_regs_acquire();\n", sec.indent));
            }
            Op::TT(TTOp::MathUnlock) => {
                if !sec.math_open {
                    return Err(sec.err("commit without MATH lock".to_string()));
                }
                sec.math_open = false;
                sec.out.push_str(&format!("{}tile_regs_commit();\n", sec.indent));
            }
            Op::TT(TTOp::PackLock) => {
                if sec.section != 1 {
                    return Err(sec.err("DST lock outside compute".to_string()));
                }
                if sec.pack_open {
                    return Err(sec.err("pack wait without release".to_string()));
                }
                sec.pack_open = true;
                sec.out.push_str(&format!("{}tile_regs_wait();\n", sec.indent));
            }
            Op::TT(TTOp::PackUnlock) => {
                if !sec.pack_open {
                    return Err(sec.err("release without PACK lock".to_string()));
                }
                sec.pack_open = false;
                sec.out.push_str(&format!("{}tile_regs_release();\n", sec.indent));
            }
            Op::TT(TTOp::NocReadBarrier) => sec.out.push_str(&format!("{}noc_async_read_barrier();\n", sec.indent)),
            Op::TT(TTOp::NocWriteBarrier) => sec.out.push_str(&format!("{}noc_async_write_barrier();\n", sec.indent)),
            Op::TT(TTOp::ReduceUninit) => sec.out.push_str(&format!("{}reduce_uninit();\n", sec.indent)),
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

/// Top-level render: split sections, collect per-section params
/// (transitive data deps of section ops, head order — the backend
/// passes exactly these as runtime args), emit preamble + body per
/// section. Pure 1:1 emission: placement (locks, sync, inits, pops)
/// is all in the IR; CB numbers come from `cbs`, slots are assigned
/// here (numbers, not placement). F32 circular buffers are rejected
/// (broken 32-bit CB path in tt-metal); any F32 tile selects 32-bit
/// DST mode.
fn render(
    k: &Kernel,
    param_ordinal_of: &Map<OpId, u32>,
    cbs: &Map<OpId, u32>,
) -> Result<(String, String, String, Vec<u32>, Vec<u32>, Vec<u32>, bool), BackendError> {
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
    // bare definitions (params, storages, consts, GEPs) seed nothing;
    // they join through their users.
    let mut params: [Vec<OpId>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for s in 0..3 {
        let mut seen: Set<OpId> = Set::default();
        let mut stack: Vec<OpId> = sections[s]
            .iter()
            .copied()
            .filter(|id| !matches!(k.at(*id), Op::Param { .. } | Op::Storage { .. } | Op::Const { .. } | Op::GEP { .. }))
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
            .filter(|id| !matches!(k.at(*id), Op::Param { .. } | Op::Storage { .. } | Op::Const { .. } | Op::GEP { .. }))
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
        srcs[s] = sec.out;
        lists[s] = params[s].iter().map(|id| param_ordinal_of[id]).collect();
    }

    Ok((srcs[0].clone(), srcs[1].clone(), srcs[2].clone(), lists[0].clone(), lists[1].clone(), lists[2].clone(), fp32))
}

/// Render a traffic `Copy`: publish (DRAM→CB, NOC read) or drain
/// (CB→DRAM, NOC write). A NOC barrier follows every transfer (old
/// per-transfer barrier). Accessors use global param ordinals
/// (`p<ord>`); arg reads use section positions. CB→CB moves never
/// reach here (`tt_storage` canonicalizes them to load+store); the
/// arm below is defensive only.
fn render_copy(sec: &mut TtSection, id: OpId, src: OpId, dst: OpId) -> Result<(), BackendError> {
    let src_cb = circular_base(sec.k, src)?;
    let dst_cb = circular_base(sec.k, dst)?;
    match (src_cb, dst_cb) {
        (None, None) => Err(sec.err(format!("copy {id:?} touches no CB"))),
        (None, Some(dst_storage)) => {
            if sec.section != 0 {
                return Err(sec.err(format!("publish {id:?} outside the reader")));
            }
            let dcb = sec.cb_num(dst_storage)?;
            let (param, sidx) = dram_side(sec.k, src)?;
            let ord = sec.ord_of(param)?;
            let idx = sec.expr(sidx)?;
            let eb = elem_bytes(sec.k, param)?;
            let bytes = tt_tile_bytes(cb_dtype(sec.k, dst_storage)?)?;
            let slot = cb_slot_expr(sec, dst, bytes)?;
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
            let (param, didx) = dram_side(sec.k, dst)?;
            let ord = sec.ord_of(param)?;
            let idx = sec.expr(didx)?;
            let eb = elem_bytes(sec.k, param)?;
            let bytes = tt_tile_bytes(cb_dtype(sec.k, src_storage)?)?;
            let slot = cb_slot_expr(sec, src, bytes)?;
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
            let scb = sec.cb_num(src_storage)?;
            let dcb = sec.cb_num(dst_storage)?;
            let s = sec.fresh_slot()?;
            sec.out.push_str(&format!("{}copy_tile({scb}, 0, {s});\n", sec.indent));
            sec.out.push_str(&format!("{}pack_tile({s}, {dcb});\n", sec.indent));
            Ok(())
        }
    }
}

/// The Circular storage under a traffic `GEP` (`None` = DRAM side).
fn circular_base(k: &Kernel, gep: OpId) -> Result<Option<OpId>, BackendError> {
    match k.at(gep) {
        Op::GEP { x, .. } => {
            if matches!(k.at(*x), Op::Storage { scope: MemScope::Circular, .. }) {
                Ok(Some(*x))
            } else {
                Ok(None)
            }
        }
        _ => Err(tt_err(format!("tenstorrent: copy side {gep:?} is not a GEP"))),
    }
}

/// The DRAM param + tile index under a traffic `GEP`.
fn dram_side(k: &Kernel, gep: OpId) -> Result<(OpId, OpId), BackendError> {
    match k.at(gep) {
        Op::GEP { x, index, .. } => match k.at(*x) {
            Op::Param { .. } => Ok((*x, *index)),
            _ => Err(tt_err(format!("tenstorrent: copy DRAM side {gep:?} is not a param"))),
        },
        _ => Err(tt_err(format!("tenstorrent: copy side {gep:?} is not a GEP"))),
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
            let lane =
                match (matches!(sec.k.layout(*x), MemLayout::Tile { .. }), matches!(sec.k.layout(*y), MemLayout::Tile { .. })) {
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
    let Op::GEP { x: base, .. } = sec.k.at(src) else {
        return Err(sec.err(format!("load {id:?} src is not a GEP")));
    };
    if !matches!(sec.k.at(*base), Op::Storage { scope: MemScope::Circular, .. }) {
        return Ok(());
    }
    let all_pack = match sec.users.get(&id) {
        Some(us) => {
            us.iter().all(|u| {
                matches!(sec.k.at(*u), Op::TT(TTOp::BroadcastTile { .. }))
                    || matches!(sec.k.at(*u), Op::Store { src: x, .. } if *x == id)
            }) && us.iter().any(|u| matches!(sec.k.at(*u), Op::Store { .. }))
        }
        None => false,
    };
    if !all_pack {
        return Ok(());
    }
    let cb = sec.cb_num(*base)?;
    let s = sec.fresh_slot()?;
    sec.out.push_str(&format!("{}copy_tile({cb}, 0, {s});\n", sec.indent));
    sec.slots.insert(id, s);
    Ok(())
}

/// Render a `Store`: pack a tile into a CB (`pack_tile`), or alias
/// a Register acc (no traffic — the slot mapping is the state).
/// The packed value must already hold a slot (SSA lowered earlier,
/// circular load unpacked at its own position); anything else is
/// malformed pack traffic.
/// CB pointer offsets (`cb_slot_expr`): empty for slot 0 (the old
/// unbatched form), ` + (<expr>)*<bytes>` otherwise.
fn cb_slot_expr(sec: &TtSection, gep: OpId, bytes: u32) -> Result<String, BackendError> {
    match sec.k.at(gep) {
        Op::GEP { index, .. } => match sec.k.at(*index) {
            Op::Const(c) => match c.as_dim() {
                Some(0) => Ok(String::new()),
                _ => Ok(format!(" + ({})*{bytes}", sec.expr(*index)?)),
            },
            _ => Ok(format!(" + ({})*{bytes}", sec.expr(*index)?)),
        },
        _ => Err(sec.err(format!("{gep:?} is not a GEP"))),
    }
}
fn render_store(sec: &mut TtSection, id: OpId, x: OpId, dst: OpId) -> Result<(), BackendError> {
    let Op::GEP { x: base, .. } = sec.k.at(dst) else {
        return Err(sec.err(format!("store {id:?} dst is not a GEP")));
    };
    match sec.k.at(*base) {
        Op::Storage { scope: MemScope::Circular, .. } => {
            let dcb = sec.cb_num(*base)?;
            let slot = sec.slot_of(x)?;
            sec.out.push_str(&format!("{}pack_tile({slot}, {dcb});\n", sec.indent));
            Ok(())
        }
        Op::Storage { scope: MemScope::Register, .. } => {
            let slot = sec.slot_of(x)?;
            sec.slots.insert(*base, slot);
            Ok(())
        }
        _ => Err(sec.err(format!("store {id:?} dst is not a buffer"))),
    }
}
