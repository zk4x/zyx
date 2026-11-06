// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

/// Debug utilities for kernel IR inspection.
///
/// This module provides debugging utilities for inspecting kernel IR,
/// including:
///
/// - Pretty-printed IR output
/// - Bounds computation for value range analysis
/// - Color-coded output (disabled when AGENT=1 or stdout is not a terminal)
///
/// Debug output is useful for:
///
/// - Understanding kernel transformations
/// - Identifying optimization opportunities
/// - Debugging kernel compilation issues
///
/// Usage:
///
/// ```text
/// ZYX_DEBUG=8 cargo run  # Print IR during kernel compilation
/// ZYX_DEBUG=16 cargo run # Print generated assembly
/// ```
use crate::kernel::{BOp, IDX_T, RangeKind, UOp};
use crate::slab::SlabId;
use crate::{
    BLUE, BOLD, CYAN, DType, GREEN, GREY, MAGENTA, Map, ORANGE, RED, RESET, YELLOW,
    kernel::{GPUOp, Kernel, Op, OpId, PTXOp, SpirvOp, TTOp},
    shape::Dim,
};
use std::fmt::{Display, Formatter};

impl Kernel {
    /// Print debug information for the kernel.
    ///
    /// Output is color-coded for readability when printed to an
    /// interactive terminal, but color is disabled when running with
    /// `AGENT=1` or when stdout is not a terminal (it will not colorize
    /// captured or logged output).
    ///
    /// # Example
    ///
    /// ```text
    /// ZYX_DEBUG=8 cargo run  # Print IR during kernel compilation
    /// ZYX_DEBUG=16 cargo run # Print generated assembly
    /// ```
    pub fn debug(&self) {
        println!("{self}")
    }

    fn shape_suffix(&self, op_id: OpId, shapes: &Map<OpId, (DType, Vec<Dim>)>) -> String {
        match shapes.get(&op_id).map(|(_, dims)| dims.as_slice()) {
            None | Some([]) => String::new(),
            Some([dim]) => format!("[{dim}]"),
            Some([x, y]) => format!("[{x}x{y}]"),
            Some(dims) => format!("[{}]", dims.iter().map(ToString::to_string).collect::<Vec<_>>().join("x")),
        }
    }

    pub(crate) fn format(&self, f: &mut Formatter<'_>, colorless: bool) -> std::fmt::Result {
        let remap_ids = false;
        let mut indent = String::from(" ");
        let bounds = self.compute_bounds();
        // Value shapes for display: single map computation up front, no
        // per-op shape walks. Ops with no value (barriers, embedded
        // source) simply have no entry and print without a suffix.
        let (shapes, _) = self.compute_dtypes_and_rcs();
        let mut dtypes: Map<OpId, DType> = Map::default();
        let mut op_id = self.head;
        let mut id_map = Map::default();
        let mut max_id = OpId::ZERO;
        let (bold, blue, cyan, green, grey, magenta, orange, red, reset, yellow) = if colorless {
            ("", "", "", "", "", "", "", "", "", "")
        } else {
            (BOLD, BLUE, CYAN, GREEN, GREY, MAGENTA, ORANGE, RED, RESET, YELLOW)
        };
        while !op_id.is_null() {
            max_id.inc();
            let out_id = if remap_ids {
                id_map.insert(op_id, max_id);
                max_id
            } else {
                id_map.insert(op_id, op_id);
                op_id
            };
            match self.ops[op_id].op {
                Op::After { .. } | Op::ToDevice { .. } | Op::Contiguous { .. } | Op::Program { .. } | Op::Kernel(_) => {
                    todo!()
                }
                // Graph-only marker, erased at kernelization: never present.
                Op::Detach { .. } => {
                    todo!()
                }
                Op::Reduce { x, rop, reduce_axis } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let x = id_map[&x];
                    let reduce_axis = id_map.get(&reduce_axis).unwrap_or(&reduce_axis);
                    writeln!(
                        f,
                        "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {red}reduce {}{reset} r{x} over r{reduce_axis}",
                        match rop {
                            BOp::Add => "sum",
                            BOp::Max => "max",
                            BOp::Mul => "prod",
                            _ => unreachable!(),
                        },
                    )
                    .unwrap();
                }
                Op::Unroll { axis, len } => {
                    dtypes.insert(op_id, IDX_T);
                    let shp = self.shape_suffix(op_id, &shapes);
                    writeln!(f, "{indent}r{out_id}{grey}: {IDX_T}{shp}{reset} = {red}unroll{reset} axis{axis} lanes {len}")
                        .unwrap();
                }
                Op::TT(TTOp::EndReader) => {
                    writeln!(f, "{indent}{blue}end_reader{reset}").unwrap();
                }
                Op::TT(TTOp::Unpack(ref words)) => {
                    writeln!(f, "{indent}{blue}unpack{reset} {} words", words.len()).unwrap();
                }
                Op::TT(TTOp::Math(ref program)) => {
                    writeln!(f, "{indent}{blue}math{reset} {} words, {} loops", program.words.len(), program.loops.len())
                        .unwrap();
                }
                Op::TT(TTOp::Pack(ref words)) => {
                    writeln!(f, "{indent}{blue}pack{reset} {} words", words.len()).unwrap();
                }
                Op::TT(TTOp::DmRead(ref page)) => {
                    writeln!(
                        f,
                        "{indent}{blue}dm_read{reset} param:{} l1:{:#x} len:{} base:{}",
                        page.param, page.l1, page.len, page.base
                    )
                    .unwrap();
                }
                Op::TT(TTOp::DmWrite(ref page)) => {
                    writeln!(
                        f,
                        "{indent}{blue}dm_write{reset} param:{} l1:{:#x} len:{} base:{} narrow:{}",
                        page.param, page.l1, page.len, page.base, page.narrow
                    )
                    .unwrap();
                }
                Op::TT(TTOp::DmReadTransposed(ref t)) => {
                    writeln!(f, "{indent}{blue}dm_read_transposed{reset} param:{} dst:{:#x} base:{}", t.param, t.dst, t.base)
                        .unwrap();
                }
                Op::TT(TTOp::DmReadBroadcastCol(ref t)) => {
                    writeln!(f, "{indent}{blue}dm_read_broadcast_col{reset} param:{} dst:{:#x} base:{}", t.param, t.dst, t.base)
                        .unwrap();
                }
                Op::TT(TTOp::DmFill(ref fill)) => {
                    writeln!(
                        f,
                        "{indent}{blue}dm_fill{reset} value:{} bounds:{:#x} dst:{:#x}",
                        fill.value, fill.bounds, fill.dst
                    )
                    .unwrap();
                }
                Op::TT(TTOp::DmCopyWords(ref copy)) => {
                    writeln!(f, "{indent}{blue}dm_copy_words{reset} {:#x}->{:#x} count:{}", copy.src, copy.dst, copy.count)
                        .unwrap();
                }
                Op::TT(TTOp::DmIndexPick(ref pick)) => {
                    writeln!(
                        f,
                        "{indent}{blue}dm_index_pick{reset} idx:{:#x} src:{:#x} out:{:#x} bound:{}",
                        pick.indices, pick.source, pick.output, pick.bound
                    )
                    .unwrap();
                }
                Op::TT(TTOp::DmKernel(ref kernel)) => {
                    writeln!(
                        f,
                        "{indent}{blue}dm_kernel{reset} generation:{} at:{:?} len:{:?}",
                        kernel.generation, kernel.at, kernel.len
                    )
                    .unwrap();
                }
                Op::TT(TTOp::DmWait) => {
                    writeln!(f, "{indent}{blue}dm_wait{reset}").unwrap();
                }
                Op::TT(TTOp::SfpLoad(ref load)) => {
                    writeln!(
                        f,
                        "{indent}{blue}sfp_load{reset} r{} <- slot:{} at:{} fmt:{}",
                        load.reg, load.slot, load.at, load.fmt
                    )
                    .unwrap();
                }
                Op::TT(TTOp::SfpStore(ref store)) => {
                    writeln!(
                        f,
                        "{indent}{blue}sfp_store{reset} r{} -> slot:{} at:{} fmt:{}",
                        store.reg, store.slot, store.at, store.fmt
                    )
                    .unwrap();
                }
                Op::TT(TTOp::SfpConst(ref c)) => {
                    writeln!(f, "{indent}{blue}sfp_const{reset} r{} = {:#x}", c.reg, c.bits).unwrap();
                }
                Op::TT(TTOp::Sfp(ref insn)) => {
                    writeln!(f, "{indent}{blue}sfp{reset} op:{:#x} args:{:?}", insn.op, insn.args).unwrap();
                }
                Op::TT(TTOp::Replay { len, count }) => {
                    writeln!(f, "{indent}{blue}replay{reset} {count} x {len} words").unwrap();
                }
                Op::TT(TTOp::TensixGrid(ref grid)) => {
                    writeln!(f, "{indent}{blue}tensix_grid{reset} {grid:?}").unwrap();
                }
                Op::TT(TTOp::EndCompute) => {
                    writeln!(f, "{indent}{blue}end_compute{reset}").unwrap();
                }
                Op::TT(TTOp::Unpacr { src, unit, count, .. }) => {
                    let dtype = dtypes.get(&src).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let map = |o: OpId| id_map.get(&o).copied().unwrap_or(OpId::NULL);
                    writeln!(
                        f,
                        "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {orange}unpacr{reset} u{unit} r{count} r{}",
                        map(src)
                    )
                    .unwrap();
                }
                Op::TT(TTOp::Setrwc { dst }) => {
                    let dtype = dtypes.get(&dst).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let map = |o: OpId| id_map.get(&o).copied().unwrap_or(OpId::NULL);
                    writeln!(f, "{indent}r{out_id}{grey}: {orange}setrwc{reset} r{}", map(dst)).unwrap();
                }
                Op::TT(TTOp::Setadczw { unit, face }) => {
                    writeln!(f, "{indent}{orange}setadczw{reset} u{unit} face{face}").unwrap();
                }
                Op::TT(TTOp::Setadcx { unit, first, count }) => {
                    writeln!(f, "{indent}{orange}setadcx{reset} u{unit} {first}..{}", first + count).unwrap();
                }
                Op::TT(TTOp::Zeroacc { mode, imm10 }) => {
                    writeln!(f, "{indent}{orange}zeroacc{reset} mode{mode} n{imm10}").unwrap();
                }
                Op::TT(TTOp::Mvmul { dst_row, addr_mod, .. }) => {
                    writeln!(f, "{indent}{orange}mvmul{reset} row{dst_row} am{addr_mod}").unwrap();
                }
                Op::TT(TTOp::Setc16 { field, value }) => {
                    writeln!(f, "{indent}{orange}setc16{reset} {field:#x} = {value}").unwrap();
                }
                Op::TT(TTOp::Setcfg { addr, value }) => {
                    writeln!(f, "{indent}{orange}setcfg{reset} {addr:#x} = {value:#x}").unwrap();
                }
                Op::TT(TTOp::Post { sem, role }) => {
                    writeln!(f, "{indent}{orange}sempost{reset} s{sem} on T{role}").unwrap();
                }
                Op::TT(TTOp::Wait { sem, before, role }) => {
                    writeln!(f, "{indent}{orange}semtake{reset} s{sem} before{before} on T{role}").unwrap();
                }
                Op::TT(TTOp::WaitBelowMax { sem, before, role }) => {
                    writeln!(f, "{indent}{orange}semwait_below{reset} s{sem} before{before} on T{role}").unwrap();
                }
                Op::TT(TTOp::Semset { sem, value }) => {
                    writeln!(f, "{indent}{orange}semset{reset} s{sem} = {value}").unwrap();
                }
                Op::TT(TTOp::Sfpu { kind, a, b, slot }) => {
                    let dtype = dtypes.get(&a).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let map = |o: OpId| id_map.get(&o).copied().unwrap_or(OpId::NULL);
                    if b.is_null() {
                        writeln!(
                            f,
                            "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {orange}sfpu_0x{kind:x}{reset} r{} -> r{}",
                            map(a),
                            map(slot),
                        )
                        .unwrap();
                    } else {
                        writeln!(
                            f,
                            "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {orange}sfpu_0x{kind:x}{reset} r{}, r{} -> r{}",
                            map(a),
                            map(b),
                            map(slot),
                        )
                        .unwrap();
                    }
                }
                Op::TT(TTOp::Copy { src, src_idx, dst, dst_idx, bytes }) => {
                    let shp = self.shape_suffix(op_id, &shapes);
                    let map = |o: OpId| id_map.get(&o).copied().unwrap_or(OpId::NULL);
                    writeln!(
                        f,
                        "{indent}r{out_id}{grey}: {shp}{reset} = {orange}tt_copy{reset} r{}, r{} -> r{}, r{} ({bytes}B)",
                        map(src),
                        map(src_idx),
                        map(dst),
                        map(dst_idx),
                    )
                    .unwrap();
                }
                Op::Param { dtype, kind, shape, .. } => {
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    if shape.is_null() {
                        writeln!(
                            f,
                            "{indent}{red}r{out_id}{reset}{grey}: {dtype}{shp}{reset} = {yellow}param{reset} {kind} shape=NULL"
                        )
                        .unwrap();
                    } else {
                        writeln!(
                            f,
                            "{indent}{red}r{out_id}{reset}{grey}: {dtype}{shp}{reset} = {yellow}param{reset} {kind} shape=r{shape}"
                        )
                        .unwrap();
                    }
                }
                Op::Storage { dtype, scope, len, .. } => {
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    writeln!(
                        f,
                        "{indent}{red}r{out_id}{reset}{grey}: {dtype}{shp}{reset} = {yellow}storage{reset} {scope}, len={len:?}"
                    )
                    .unwrap();
                }
                Op::Const(value) => {
                    let dtype = value.dtype();
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {magenta}{value}{reset}").unwrap();
                }
                Op::Load { src, src_idx, .. } => {
                    let dtype = dtypes.get(&src).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let src = id_map.get(&src).copied().unwrap_or(src);
                    let src_idx = id_map.get(&src_idx).copied().unwrap_or(src_idx);
                    let (lb, ub) = bounds.get(&src_idx).copied().unwrap_or((0, 0));
                    match self.shape(op_id).as_slice() {
                        [] => {
                            writeln!(
                                f,
                                "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {green}load{reset} r{src}[r{src_idx}]    // {lb}..={ub}",
                            )
                            .unwrap();
                        }
                        [len] => {
                            writeln!(
                                f,
                                "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {green}load{reset} vec{len} r{src}[r{src_idx}]    // {lb}..={ub}",
                            )
                            .unwrap();
                        }
                        dims => {
                            let t = dims.iter().map(|d| format!("{d}")).collect::<Vec<_>>().join("x");
                            writeln!(
                                f,
                                "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {green}load{reset} tile {t} r{src}[r{src_idx}]    // {lb}..={ub}",
                            )
                            .unwrap();
                        }
                    }
                }
                Op::Store { dst, dst_idx, src: x, .. } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let dst = id_map.get(&dst).copied().unwrap_or(dst);
                    let dst_idx = id_map.get(&dst_idx).copied().unwrap_or(dst_idx);
                    let x = id_map.get(&x).copied().unwrap_or(x);
                    let idx_txt = if dst_idx.is_null() {
                        String::new()
                    } else {
                        format!("[r{dst_idx}]")
                    };
                    let (lb, ub) = bounds.get(&dst_idx).copied().unwrap_or((0, 0));
                    let kind = match self.shape(op_id).as_slice() {
                        [] => String::from("store"),
                        [len] => format!("store vec{len}"),
                        [x, y] => format!("store tile {x}x{y}"),
                        dims => format!("store {}", dims.iter().map(|d| format!("{d}")).collect::<Vec<_>>().join("x")),
                    };
                    writeln!(f, "{indent}{red}{kind}{reset} r{dst}{idx_txt} = r{x}    // {lb}..={ub}").unwrap();
                }
                Op::Copy { src, src_idx, dst, dst_idx, .. } => {
                    let dtype = dtypes.get(&src).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let src = id_map.get(&src).copied().unwrap_or(src);
                    let src_idx = id_map.get(&src_idx).copied().unwrap_or(src_idx);
                    let dst = id_map.get(&dst).copied().unwrap_or(dst);
                    let dst_idx = id_map.get(&dst_idx).copied().unwrap_or(dst_idx);
                    let kind = match self.shape(op_id).as_slice() {
                        [] => String::from("copy"),
                        [len] => format!("copy vec{len}"),
                        dims => format!("copy tile {}", dims.iter().map(|d| format!("{d}")).collect::<Vec<_>>().join("x")),
                    };
                    writeln!(f, "{indent}{red}{kind}{reset} r{dst}[r{dst_idx}] = r{src}[r{src_idx}]").unwrap();
                }
                Op::Cast { x, dtype } => {
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let x = id_map.get(&x).copied().unwrap_or(x);
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {dtype}(r{x})    // {lb}..={ub}").unwrap();
                    } else {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {dtype}(r{x})").unwrap();
                    }
                }
                Op::Bitcast { x, dtype } => {
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let x = id_map.get(&x).copied().unwrap_or(x);
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = bits({dtype})r{x}    // {lb}..={ub}")
                            .unwrap();
                    } else {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = bits({dtype})r{x}").unwrap();
                    }
                }
                Op::Unary { x, uop, .. } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let (op1, op2) = match uop {
                        UOp::Neg => ("-", ""),
                        UOp::Not => ("!", ""),
                        UOp::BitNot => ("~", ""),
                        UOp::Exp2 => ("exp2(", ")"),
                        UOp::Log2 => ("log2(", ")"),
                        UOp::Reciprocal => ("1/", ""),
                        UOp::Sqrt => ("sqrt(", ")"),
                        UOp::Rsqrt => ("rsqrt(", ")"),
                        UOp::Sin => ("sin(", ")"),
                        UOp::Cos => ("cos(", ")"),
                        UOp::Floor => ("floor(", ")"),
                        UOp::Trunc => ("trunc(", ")"),
                        UOp::Abs => ("abs(", ")"),
                    };
                    let x = id_map.get(&x).copied().unwrap_or(x);
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {op1}r{x}{op2}    // {lb}..={ub}").unwrap();
                    } else {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {op1}r{x}{op2}").unwrap();
                    }
                }
                Op::Binary { x, y, bop, .. } => {
                    let dtype = if bop.returns_bool() {
                        DType::Bool
                    } else {
                        dtypes.get(&x).copied().unwrap_or(DType::U8)
                    };
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let (op1, op2, op3) = match bop {
                        BOp::Add => ("", " + ", ""),
                        BOp::Sub => ("", " - ", ""),
                        BOp::Mul => ("", " * ", ""),
                        BOp::Div => ("", " / ", ""),
                        BOp::Pow => ("pow(", ", ", ")"),
                        BOp::Mod => ("", " % ", ""),
                        BOp::Cmplt => ("", " < ", ""),
                        BOp::Cmpgt => ("", " > ", ""),
                        BOp::Cmpge => ("", " >= ", ""),
                        BOp::Max => ("max(", ", ", ")"),
                        BOp::Or => ("", " || ", ""),
                        BOp::And => ("", " && ", ""),
                        BOp::BitXor => ("", " ^ ", ""),
                        BOp::BitOr => ("", " | ", ""),
                        BOp::BitAnd => ("", " & ", ""),
                        BOp::BitShiftLeft => ("", " << ", ""),
                        BOp::BitShiftRight => ("", " >> ", ""),
                        BOp::NotEq => ("", " != ", ""),
                        BOp::Eq => ("", " == ", ""),
                    };
                    let x = id_map.get(&x).copied().unwrap_or(x);
                    let y = id_map.get(&y).copied().unwrap_or(y);
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {op1}r{x}{op2}r{y}{op3}    // {lb}..={ub}")
                            .unwrap();
                    } else {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {op1}r{x}{op2}r{y}{op3}").unwrap();
                    }
                }
                Op::Select { cond, if_true, if_false } => {
                    let dtype = dtypes.get(&if_true).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let cond = id_map.get(&cond).copied().unwrap_or(cond);
                    let if_true = id_map.get(&if_true).copied().unwrap_or(if_true);
                    let if_false = id_map.get(&if_false).copied().unwrap_or(if_false);
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        writeln!(
                            f,
                            "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = r{cond} ? r{if_true} : r{if_false}    // {lb}..={ub}"
                        )
                        .unwrap();
                    } else {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = r{cond} ? r{if_true} : r{if_false}").unwrap();
                    }
                }
                Op::Wmma { dims, layout, dtype, c, a, b } => {
                    let cdtype = dtypes.get(&c).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, cdtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let a = id_map.get(&a).copied().unwrap_or(a);
                    let b = id_map.get(&b).copied().unwrap_or(b);
                    let c = id_map.get(&c).copied().unwrap_or(c);
                    writeln!(f, "{indent}r{out_id}{grey}: {cdtype}{shp}{reset} = {orange}wmma{reset}.{dims:?}.{layout:?}.{dtype:?}(c={c}, a={a}, b={b})").unwrap();
                }
                Op::Range { axis, kind } => {
                    dtypes.insert(op_id, IDX_T);
                    match kind {
                        RangeKind::Group(len) => {
                            let ub = self
                                .resolve_const(len)
                                .and_then(crate::dtype::Constant::as_dim)
                                .unwrap_or(Dim::MAX)
                                .saturating_sub(1);
                            let len = id_map.get(&len).copied().unwrap_or(len);
                            writeln!(f, "{indent}r{out_id}{grey}: {IDX_T}{reset} = {blue}group_range({axis}){reset} over r{len}    // 0..={ub}").unwrap();
                        }
                        RangeKind::Local(len) => {
                            writeln!(
                                f,
                                "{indent}r{out_id}{grey}: {IDX_T}{reset} = {blue}local_range({axis}){reset}    // 0..={}",
                                len - 1
                            )
                            .unwrap();
                        }
                    }
                }
                Op::Loop { len } => {
                    let dtype = dtypes.get(&len).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let len = id_map.get(&len).copied().unwrap_or(OpId::NULL);
                    if dtype == DType::Bool {
                        writeln!(f, "{indent}{bold}if{reset} r{len} {{").unwrap();
                    } else if let Some((l, u)) = bounds.get(&op_id) {
                        writeln!(f, "{indent}{bold}for{reset} r{out_id} in 0..r{len} {{    // {l}..={}", u).unwrap();
                    } else {
                        writeln!(f, "{indent}{bold}for{reset} r{out_id} in 0..r{len} {{").unwrap();
                    }
                    indent += "  ";
                }
                Op::EndLoop => {
                    if indent.len() > 1 {
                        indent.pop();
                        indent.pop();
                    }
                    writeln!(f, "{indent}}}").unwrap();
                }
                Op::Asm { ref asm, ref ops } => {
                    // Operand-free Asm is an effect-only call: U8 is a
                    // display placeholder, never a value dtype.
                    let dtype = ops.iter().next().and_then(|x| dtypes.get(x)).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let ops: Vec<OpId> = ops.iter().map(|x| id_map.get(x).copied().unwrap_or(OpId::NULL)).collect();
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {orange}asm{reset} {asm} {ops:?}").unwrap();
                }
                Op::Stack { ref ops } => {
                    let dtype = dtypes.get(&ops[0]).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let ops: Vec<OpId> = ops.iter().map(|x| id_map.get(x).copied().unwrap_or(OpId::NULL)).collect();
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        writeln!(
                            f,
                            "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {orange}stack{reset}{ops:?}    // {lb}..={ub}"
                        )
                        .unwrap();
                    } else {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {orange}stack{reset}{ops:?}").unwrap();
                    }
                }
                Op::Index { vec, idx } => {
                    let dtype = dtypes.get(&vec).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let vec = id_map.get(&vec).copied().unwrap_or(OpId::NULL);
                    if let Some((l, u)) = bounds.get(&op_id) {
                        writeln!(
                            f,
                            "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = r{vec}{orange}.s{idx}{reset}    // {l}..={u}",
                        )
                        .unwrap();
                    } else {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = r{vec}{orange}.s{idx}{reset}").unwrap();
                    }
                }
                Op::Reshape { x, shape, .. } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    let shape = id_map.get(&shape).copied().unwrap_or(shape);
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {cyan}reshape{reset} r{x} -> {shape:?}").unwrap();
                }
                Op::Expand { x, shape } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    let shape = id_map.get(&shape).copied().unwrap_or(shape);
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {cyan}expand{reset} r{x} -> {shape:?}").unwrap();
                }
                Op::Permute { x, ref axes } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {cyan}permute{reset} r{x} axes={axes:?}")
                        .unwrap();
                }
                Op::Pad { x, ref axis, lp, len } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    let lp = id_map.get(&lp).copied().unwrap_or(lp);
                    let len = id_map.get(&len).copied().unwrap_or(len);
                    writeln!(
                        f,
                        "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {cyan}pad{reset} r{x} axis={axis} lp=r{lp} len=r{len}"
                    )
                    .unwrap();
                }
                Op::Narrow { x, axis, start, len } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    let start = id_map.get(&start).copied().unwrap_or(start);
                    let len = id_map.get(&len).copied().unwrap_or(len);
                    writeln!(
                        f,
                        "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {cyan}narrow{reset} r{x} axis={axis} start=r{start} len=r{len}"
                    )
                    .unwrap();
                }
                Op::Flip { x, ref axes } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let shp = self.shape_suffix(op_id, &shapes);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{shp}{reset} = {cyan}flip{reset} r{x} axes={axes:?}").unwrap();
                }
                Op::Barrier => {
                    writeln!(f, "{indent}barrier").unwrap();
                }
                Op::Source(ref src) => {
                    writeln!(f, "{}", src.as_str()).unwrap();
                }
                Op::GPU(ref gpu) => match gpu.as_ref() {
                    GPUOp::LocalWorkSize(size) => {
                        writeln!(f, "{indent}local_work_size=[{}, {}, {}]", size[0], size[1], size[2]).unwrap();
                    }
                    GPUOp::Grid(gws) => {
                        writeln!(f, "{indent}grid={gws:?}").unwrap();
                    }
                    GPUOp::Params(params) => {
                        writeln!(f, "{indent}params={params:?}").unwrap();
                    }
                },
                Op::Spirv(ref spirv) => match spirv.as_ref() {
                    SpirvOp::WordBytes(words) => {
                        writeln!(f, "{indent}spirv_words={}", words.len()).unwrap();
                    }
                    SpirvOp::PushConstants(size) => {
                        writeln!(f, "{indent}spirv_push_constants={size}").unwrap();
                    }
                },
                Op::PTX(ref ptx) => match ptx.as_ref() {
                    PTXOp::Bytes(bytes) => {
                        writeln!(f, "{indent}ptx_bytes={}", bytes.len()).unwrap();
                    }
                    PTXOp::LdMatrix { frags, src, addr, trans } => {
                        writeln!(f, "{indent}ldmatrix x{frags} trans={trans} from OpId({}) at OpId({})", src.0, addr.0)
                            .unwrap();
                    }
                },
            }
            op_id = self.ops[op_id].next;
        }
        writeln!(f)
    }
}

#[cfg(feature = "viz")]
pub(crate) struct KernelDebug<'a> {
    pub(crate) kernel: &'a Kernel,
    pub(crate) colorless: bool,
}

#[cfg(feature = "viz")]
impl<'a> std::fmt::Display for KernelDebug<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.kernel.format(f, self.colorless)
    }
}

impl Display for Kernel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Auto-infer colorless: `AGENT=1` disables it, and so does non-terminal stdout.
        let colorless = std::env::var("AGENT").is_ok_and(|v| v == "1") || !std::io::IsTerminal::is_terminal(&std::io::stdout());
        self.format(f, colorless)
    }
}
