// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

/// Debug utilities for kernel IR inspection.
///
/// This module provides debugging utilities for inspecting kernel IR,
/// including:
///
/// - Pretty-printed IR output
/// - Bounds computation for value range analysis
/// - Color-coded output (disabled when AGENT=1)
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
    /// Output is color-coded for readability, but color is disabled
    /// when running with `AGENT=1` (for cleaner log output).
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
}

impl Display for Kernel {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let remap_ids = false;
        let mut indent = String::from(" ");
        let bounds = self.compute_bounds();
        let mut dtypes: Map<OpId, DType> = Map::default();
        let mut op_id = self.head;
        let mut has_loops = false;
        let mut id_map = Map::default();
        let mut max_id = OpId::ZERO;
        let colorless = std::env::var("AGENT").is_ok_and(|v| v == "1");
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
                Op::Reduce { x, rop, reduce_axis } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let x = id_map[&x];
                    let reduce_axis = id_map.get(&reduce_axis).unwrap_or(&reduce_axis);
                    if has_loops {
                        indent.pop();
                        indent.pop();
                    }
                    writeln!(
                        f,
                        "{indent}r{out_id}{grey}: {dtype}{reset} = {red}reduce {}{reset} r{x} over r{reduce_axis}",
                        match rop {
                            BOp::Add => "sum",
                            BOp::Max => "max",
                            BOp::Mul => "prod",
                            _ => unreachable!(),
                        },
                    )
                    .unwrap();
                }
                Op::TT(TTOp::ReduceTile { x, scaler, acc, rop, .. }) => {
                    let dtype = dtypes.get(&acc).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let x = id_map[&x];
                    let scaler = id_map[&scaler];
                    let acc = id_map[&acc];
                    writeln!(
                        f,
                        "{indent}r{out_id}: {dtype}{grey}: {dtype}{reset} = {red}reduce_tile_{}{reset} r{x}, r{scaler}, r{acc}",
                        match rop {
                            BOp::Add => "sum",
                            BOp::Max => "max",
                            BOp::Mul => "prod",
                            _ => unreachable!(),
                        },
                    )
                    .unwrap();
                }
                Op::TT(TTOp::MatmulTile { x, y, acc }) => {
                    let dtype = dtypes.get(&acc).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let x = id_map[&x];
                    let y = id_map[&y];
                    let acc = id_map[&acc];
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {red}matmul_tile{reset} r{x}, r{y}, r{acc}").unwrap();
                }
                Op::TT(TTOp::TransposeTile { x }) => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let x = id_map[&x];
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {red}transpose_tile{reset} r{x}").unwrap();
                }
                Op::TT(TTOp::BroadcastTile { x, kind }) => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let x = id_map[&x];
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {red}broadcast_tile_{kind:?}{reset} r{x}").unwrap();
                }
                Op::TT(TTOp::ReserveBack { cb, n }) => {
                    let dtype = dtypes.get(&cb).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let cb = id_map.get(&cb).copied().unwrap_or(cb);
                    writeln!(f, "{indent}{red}reserve_back{reset} r{cb}, {n}").unwrap();
                }
                Op::TT(TTOp::PushBack { cb, n }) => {
                    let dtype = dtypes.get(&cb).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let cb = id_map.get(&cb).copied().unwrap_or(cb);
                    writeln!(f, "{indent}{red}push_back{reset} r{cb}, {n}").unwrap();
                }
                Op::TT(TTOp::WaitFront { cb, n }) => {
                    let dtype = dtypes.get(&cb).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let cb = id_map.get(&cb).copied().unwrap_or(cb);
                    writeln!(f, "{indent}{red}wait_front{reset} r{cb}, {n}").unwrap();
                }
                Op::TT(TTOp::PopFront { cb, n }) => {
                    let dtype = dtypes.get(&cb).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let cb = id_map.get(&cb).copied().unwrap_or(cb);
                    writeln!(f, "{indent}{red}pop_front{reset} r{cb}, {n}").unwrap();
                }
                Op::TT(TTOp::MathLock) => {
                    writeln!(f, "{indent}{red}math_lock{reset}").unwrap();
                }
                Op::TT(TTOp::MathUnlock) => {
                    writeln!(f, "{indent}{red}math_unlock{reset}").unwrap();
                }
                Op::TT(TTOp::PackLock) => {
                    writeln!(f, "{indent}{red}pack_lock{reset}").unwrap();
                }
                Op::TT(TTOp::PackUnlock) => {
                    writeln!(f, "{indent}{red}pack_unlock{reset}").unwrap();
                }
                Op::TT(TTOp::NocReadBarrier) => {
                    writeln!(f, "{indent}{red}noc_read_barrier{reset}").unwrap();
                }
                Op::TT(TTOp::NocWriteBarrier) => {
                    writeln!(f, "{indent}{red}noc_write_barrier{reset}").unwrap();
                }
                Op::TT(TTOp::ReduceUninit) => {
                    writeln!(f, "{indent}{red}reduce_uninit{reset}").unwrap();
                }
                Op::TT(TTOp::EndReader) => {
                    writeln!(f, "{indent}{blue}end_reader{reset}").unwrap();
                }
                Op::TT(TTOp::ProgramDesc(ref desc)) => {
                    writeln!(
                        f,
                        "{indent}{blue}program_desc{reset} reader:{} compute:{} writer:{} params:{} in:{} out:{} cbs:{} fp32:{}",
                        desc.reader_params.len(),
                        desc.compute_params.len(),
                        desc.writer_params.len(),
                        desc.n_params,
                        desc.n_inputs,
                        desc.n_outputs,
                        desc.cb_config.len(),
                        desc.fp32,
                    )
                    .unwrap();
                }
                Op::TT(TTOp::TensixGrid(ref grid)) => {
                    writeln!(f, "{indent}{blue}tensix_grid{reset} {grid:?}").unwrap();
                }
                Op::TT(TTOp::EndCompute) => {
                    writeln!(f, "{indent}{blue}end_compute{reset}").unwrap();
                }
                Op::TT(TTOp::LLK { ref asm, ref ops }) => {
                    // Opaque LLK call template: U8 is a display
                    // placeholder like Asm (effect-only, never a value).
                    let dtype = ops.iter().next().and_then(|x| dtypes.get(x)).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let ops: Vec<OpId> = ops.iter().map(|x| id_map.get(x).copied().unwrap_or(OpId::NULL)).collect();
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {orange}llk{reset} {asm} {ops:?}").unwrap();
                }
                Op::TT(TTOp::LLKReduce { rop, kind, cb_in, cb_sc, slot, x, scaler }) => {
                    let dtype = dtypes.get(&cb_in).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let map = |o: OpId| id_map.get(&o).copied().unwrap_or(OpId::NULL);
                    writeln!(
                        f,
                        "{indent}r{out_id}{grey}: {dtype}{reset} = {orange}llk_reduce_{rop:?}_{kind:?}{reset} r{}, r{}, r{}, r{}, r{}",
                        map(cb_in),
                        map(cb_sc),
                        map(slot),
                        map(x),
                        map(scaler),
                    )
                    .unwrap();
                }
                Op::TT(TTOp::LLKBcast { bop, kind, cb_a, cb_b, mx, plain }) => {
                    let dtype = dtypes.get(&cb_a).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let map = |o: OpId| id_map.get(&o).copied().unwrap_or(OpId::NULL);
                    writeln!(
                        f,
                        "{indent}r{out_id}{grey}: {dtype}{reset} = {orange}llk_bcast_{bop:?}_{kind:?}{reset} r{}, r{}, r{}, r{}",
                        map(cb_a),
                        map(cb_b),
                        map(mx),
                        map(plain),
                    )
                    .unwrap();
                }
                Op::Param { dtype, kind, shape, .. } => {
                    dtypes.insert(op_id, dtype);
                    if shape.is_null() {
                        writeln!(
                            f,
                            "{indent}{red}r{out_id}{reset}{grey}: {dtype}{reset} = {yellow}param{reset} {kind} shape=NULL"
                        )
                        .unwrap();
                    } else {
                        writeln!(
                            f,
                            "{indent}{red}r{out_id}{reset}{grey}: {dtype}{reset} = {yellow}param{reset} {kind} shape=r{shape}"
                        )
                        .unwrap();
                    }
                }
                Op::Storage { dtype, scope, len } => {
                    dtypes.insert(op_id, dtype);
                    writeln!(
                        f,
                        "{indent}{red}r{out_id}{reset}{grey}: {dtype}{reset} = {yellow}storage{reset} {scope}, len={len:?}"
                    )
                    .unwrap();
                }
                Op::Const(value) => {
                    let dtype = value.dtype();
                    dtypes.insert(op_id, dtype);
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {magenta}{value}{reset}").unwrap();
                }
                Op::GEP { x, index, layout } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let (lb, ub) = bounds.get(&index).copied().unwrap_or((0, 0));
                    let x = id_map.get(&x).copied().unwrap_or(x);
                    let index = id_map.get(&index).copied().unwrap_or(index);
                    writeln!(
                        f,
                        "{indent}r{out_id}{grey}: {dtype}{reset} = {red}r{x}{reset}[r{index} @ {layout}]    // {lb}..={ub} {green}gep{reset}"
                    )
                    .unwrap();
                }
                Op::Load { src } => {
                    let dtype = dtypes.get(&src).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let src = id_map.get(&src).copied().unwrap_or(src);
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {green}load{reset} r{src}").unwrap();
                }
                Op::Store { dst, src: x } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let dst = id_map.get(&dst).copied().unwrap_or(dst);
                    let x = id_map.get(&x).copied().unwrap_or(x);
                    writeln!(f, "{indent}{red}store{reset} r{dst} = r{x}").unwrap();
                }
                Op::Copy { src, dst } => {
                    let dtype = dtypes.get(&src).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let src = id_map.get(&src).copied().unwrap_or(src);
                    let dst = id_map.get(&dst).copied().unwrap_or(dst);
                    writeln!(f, "{indent}r{dst} = r{src}    // {red}copy{reset}").unwrap();
                }
                Op::Cast { x, dtype } => {
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(x);
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {dtype}(r{x})    // {lb}..={ub}").unwrap();
                    } else {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {dtype}(r{x})").unwrap();
                    }
                }
                Op::Bitcast { x, dtype } => {
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(x);
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = bits({dtype})r{x}    // {lb}..={ub}").unwrap();
                    } else {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = bits({dtype})r{x}").unwrap();
                    }
                }
                Op::Unary { x, uop, .. } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
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
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {op1}r{x}{op2}    // {lb}..={ub}").unwrap();
                    } else {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {op1}r{x}{op2}").unwrap();
                    }
                }
                Op::Binary { x, y, bop, .. } => {
                    let dtype = if bop.returns_bool() {
                        DType::Bool
                    } else {
                        dtypes.get(&x).copied().unwrap_or(DType::U8)
                    };
                    dtypes.insert(op_id, dtype);
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
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {op1}r{x}{op2}r{y}{op3}    // {lb}..={ub}")
                            .unwrap();
                    } else {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {op1}r{x}{op2}r{y}{op3}").unwrap();
                    }
                }
                Op::Wmma { dims, layout, dtype, c, a, b } => {
                    let cdtype = dtypes.get(&c).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, cdtype);
                    let a = id_map.get(&a).copied().unwrap_or(a);
                    let b = id_map.get(&b).copied().unwrap_or(b);
                    let c = id_map.get(&c).copied().unwrap_or(c);
                    writeln!(f, "{indent}r{out_id}{grey}: {cdtype}{reset} = {orange}wmma{reset}.{dims:?}.{layout:?}.{dtype:?}(c={c}, a={a}, b={b})").unwrap();
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
                            writeln!(f, "{indent}r{out_id}{grey}: {IDX_T}{reset} = {blue}{kind}_index({axis}){reset} over r{len}    // 0..={ub}").unwrap();
                        }
                        RangeKind::Local(len) => {
                            writeln!(
                                f,
                                "{indent}r{out_id}{grey}: {IDX_T}{reset} = {blue}{kind}_index({axis}){reset}    // 0..={}",
                                len - 1
                            )
                            .unwrap();
                        }
                        RangeKind::Warp(local_id) => {
                            let local_id = id_map.get(&local_id).copied().unwrap_or(local_id);
                            writeln!(
                                f,
                                "{indent}r{out_id}{grey}: {IDX_T}{reset} = {blue}{kind}_index({axis}){reset} over r{local_id}    // lane id"
                            )
                            .unwrap();
                        }
                    }
                }
                Op::Loop { len } => {
                    has_loops = true;
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
                    let ops: Vec<OpId> = ops.iter().map(|x| id_map.get(x).copied().unwrap_or(OpId::NULL)).collect();
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {orange}asm{reset} {asm} {ops:?}").unwrap();
                }
                Op::Stack { ref ops } => {
                    let dtype = dtypes.get(&ops[0]).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let ops: Vec<OpId> = ops.iter().map(|x| id_map.get(x).copied().unwrap_or(OpId::NULL)).collect();
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {orange}stack{reset}{ops:?}    // {lb}..={ub}")
                            .unwrap();
                    } else {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {orange}stack{reset}{ops:?}").unwrap();
                    }
                }
                Op::Index { vec, idx } => {
                    let dtype = dtypes.get(&vec).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let vec = id_map.get(&vec).copied().unwrap_or(OpId::NULL);
                    if let Some((l, u)) = bounds.get(&op_id) {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = r{vec}{orange}.s{idx}{reset}    // {l}..={u}",)
                            .unwrap();
                    } else {
                        writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = r{vec}{orange}.s{idx}{reset}").unwrap();
                    }
                }
                Op::Reshape { x, shape, .. } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    let shape = id_map.get(&shape).copied().unwrap_or(shape);
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {cyan}reshape{reset} r{x} -> {shape:?}").unwrap();
                }
                Op::Expand { x, shape } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    let shape = id_map.get(&shape).copied().unwrap_or(shape);
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {cyan}expand{reset} r{x} -> {shape:?}").unwrap();
                }
                Op::Permute { x, ref axes } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {cyan}permute{reset} r{x} axes={axes:?}").unwrap();
                }
                Op::Pad { x, ref axis, lp, len } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    let lp = id_map.get(&lp).copied().unwrap_or(lp);
                    let len = id_map.get(&len).copied().unwrap_or(len);
                    writeln!(
                        f,
                        "{indent}r{out_id}{grey}: {dtype}{reset} = {cyan}pad{reset} r{x} axis={axis} lp=r{lp} len=r{len}"
                    )
                    .unwrap();
                }
                Op::Narrow { x, axis, start, len } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    let start = id_map.get(&start).copied().unwrap_or(start);
                    let len = id_map.get(&len).copied().unwrap_or(len);
                    writeln!(
                        f,
                        "{indent}r{out_id}{grey}: {dtype}{reset} = {cyan}narrow{reset} r{x} axis={axis} start=r{start} len=r{len}"
                    )
                    .unwrap();
                }
                Op::Flip { x, ref axes } => {
                    let dtype = dtypes.get(&x).copied().unwrap_or(DType::U8);
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    writeln!(f, "{indent}r{out_id}{grey}: {dtype}{reset} = {cyan}flip{reset} r{x} axes={axes:?}").unwrap();
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
                },
            }
            op_id = self.ops[op_id].next;
        }
        writeln!(f)
    }
}
