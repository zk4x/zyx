// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

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
/// ZYX_DEBUG=16 cargo run # Print generated CUDA assembly
/// ```
use crate::kernel::{BOp, IDX_T, MoveOp, Scope, UOp};
use crate::slab::SlabId;
use crate::{BLUE, BOLD, CYAN, GREEN, GREY, MAGENTA, ORANGE, RED, RESET, YELLOW};
use crate::{
    DType, Map,
    kernel::{Kernel, Op, OpId},
};

impl Kernel {
    /// Print debug information for the kernel.
    ///
    /// This method prints detailed information about the kernel IR,
    /// including:
    ///
    /// - Loaded and stored tensor IDs
    /// - Output tensor IDs
    /// - Operation bounds (value ranges)
    /// - Operation dtypes
    /// - Loop information
    ///
    /// Output is color-coded for readability, but color is disabled
    /// when running with AGENT=1 (for cleaner log output).
    ///
    /// # Example
    ///
    /// ```text
    /// ZYX_DEBUG=8 cargo run  # Print IR during kernel compilation
    /// ZYX_DEBUG=16 cargo run # Print generated CUDA assembly
    /// ```
    pub fn debug(&self) {
        let remap_ids = false;
        println!("\nloads={:?}", self.loads);
        println!("stores={:?}", self.stores);
        println!("outputs={:?}", self.outputs);
        let mut indent = String::from(" ");
        let bounds = self.compute_bounds();
        let mut dtypes: Map<OpId, DType> = Map::default();
        let mut op_id = self.head;
        let mut has_loops = false;
        let mut id_map = Map::default();
        let mut max_id = OpId::ZERO;
        let colorless = std::env::var("AGENT").map_or(false, |v| v == "1");
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
            match *self.at(op_id) {
                Op::ConstView(ref x) => {
                    let value = x.0;
                    let view = &x.1;
                    let dtype = value.dtype();
                    dtypes.insert(op_id, dtype);
                    println!("{indent}r{out_id}{grey}: {dtype}{reset} = {cyan}{value}{reset} {view}");
                }
                Op::LoadView(ref x) => {
                    let dtype = x.0;
                    let view = &x.1;
                    dtypes.insert(op_id, dtype);
                    println!("{indent}r{out_id}{grey}: {dtype}{reset} = {cyan}load{reset} {view}");
                }
                Op::StoreView { src, dtype, .. } => {
                    let src = id_map[&src];
                    dtypes.insert(op_id, dtype);
                    println!("{indent}{cyan}store{reset} r{src}");
                }
                Op::Reduce { x, rop, n_axes, .. } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let x = id_map[&x];
                    if has_loops {
                        for _ in 0..n_axes * 2 {
                            indent.pop();
                        }
                    }
                    println!(
                        "{indent}r{out_id}{grey}: {dtype}{reset} = {red}reduce {}{reset} r{x}, dims={n_axes:?} {}",
                        match rop {
                            BOp::Add => "sum",
                            BOp::Max => "max",
                            BOp::Mul => "prod",
                            _ => unreachable!(),
                        },
                        dtypes[&op_id]
                    );
                }
                Op::Define { dtype, scope, ro, len, .. } => {
                    dtypes.insert(op_id, dtype);
                    let ro = if ro { "" } else { "mut " };
                    println!("{indent}{red}r{out_id}{reset}{grey}: {dtype}{reset} = {yellow}def {ro}{reset}{scope}, len={len}");
                }
                Op::Const(value) => {
                    let dtype = value.dtype();
                    dtypes.insert(op_id, dtype);
                    println!("{indent}r{out_id}{grey}: {dtype}{reset} = {magenta}{value}{reset}");
                }
                Op::Load { src, index, layout } => {
                    let dtype = dtypes[&src];
                    dtypes.insert(op_id, dtype);
                    let (lb, ub) = bounds.get(&index).copied().unwrap_or((0, 0));
                    let src = id_map.get(&src).copied().unwrap_or(OpId::NULL);
                    let index = id_map.get(&index).copied().unwrap_or(OpId::NULL);
                    println!(
                        "{indent}r{out_id}{grey}: {dtype}{reset} = {red}r{src}{reset}[r{index} @ {layout}]    // {lb}..={ub} {green}load{reset}"
                    );
                }
                Op::Store { dst, x, index, layout } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let (lb, ub) = bounds.get(&index).copied().unwrap_or((0, 0));
                    let dst = id_map.get(&dst).copied().unwrap_or(OpId::NULL);
                    let index = id_map.get(&index).copied().unwrap_or(OpId::NULL);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    println!("{indent}{red}r{dst}{reset}[r{index} @ {layout}] = r{x}    // {lb}..={ub} {red}store{reset}");
                }
                Op::Cast { x, dtype } => {
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}{grey}: {dtype}{reset} = {dtype}(r{x})    // {lb}..={ub}");
                    } else {
                        println!("{indent}r{out_id}{grey}: {dtype}{reset} = {dtype}(r{x})");
                    }
                }
                Op::Unary { x, uop, .. } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let (op1, op2) = match uop {
                        UOp::Neg => ("-", ""),
                        UOp::BitNot => ("~", ""),
                        UOp::Exp => ("exp(", ")"),
                        UOp::Exp2 => ("exp2(", ")"),
                        UOp::Ln => ("ln(", ")"),
                        UOp::Log2 => ("log2(", ")"),
                        UOp::Reciprocal => ("1/", ""),
                        UOp::Sqrt => ("sqrt(", ")"),
                        UOp::Sin => ("sin(", ")"),
                        UOp::Cos => ("cos(", ")"),
                        UOp::Floor => ("floor(", ")"),
                        UOp::Trunc => ("trunc(", ")"),
                        UOp::Abs => ("abs(", ")"),
                    };
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}{grey}: {dtype}{reset} = {op1}r{x}{op2}    // {lb}..={ub}");
                    } else {
                        println!("{indent}r{out_id}{grey}: {dtype}{reset} = {op1}r{x}{op2}");
                    }
                }
                Op::Binary { x, y, bop, .. } => {
                    let dtype = dtypes[&x];
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
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    let y = id_map.get(&y).copied().unwrap_or(OpId::NULL);
                    let x = if let Op::Const(c) = self.ops[x].op {
                        format!("{c}")
                    } else {
                        format!("r{x}")
                    };
                    let y = if let Op::Const(c) = self.ops[y].op {
                        format!("{c}")
                    } else {
                        format!("r{y}")
                    };
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}{grey}: {dtype}{reset} = {op1}{x}{op2}{y}{op3}    // {lb}..={ub}");
                    } else {
                        println!("{indent}r{out_id}{grey}: {dtype}{reset} = {op1}{x}{op2}{y}{op3}");
                    }
                }
                Op::Mad { x, y, z } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    let y = id_map.get(&y).copied().unwrap_or(OpId::NULL);
                    let z = id_map.get(&z).copied().unwrap_or(OpId::NULL);
                    if let Some((l, u)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}{grey}: {dtype}{reset} = r{x} * r{y} + r{z}    // {l}..={u}");
                    } else {
                        println!("{indent}r{out_id}{grey}: {dtype}{reset} = r{x} * r{y} + r{z}");
                    }
                }
                Op::Wmma { dims, layout, dtype, c, a, b } => {
                    let cdtype = dtypes[&c];
                    dtypes.insert(op_id, cdtype);
                    let a = id_map.get(&a).copied().unwrap_or(OpId::NULL);
                    let b = id_map.get(&b).copied().unwrap_or(OpId::NULL);
                    let c = id_map.get(&c).copied().unwrap_or(OpId::NULL);
                    println!(
                        "{indent}r{out_id}{grey}: {cdtype}{reset} = {orange}wmma{reset}.{dims:?}.{layout:?}.{dtype:?}(c={c}, a={a}, b={b})",
                    );
                }
                Op::Index { len, scope, axis } => {
                    dtypes.insert(op_id, IDX_T);
                    let ub = len - 1;
                    let scope = match scope {
                        Scope::Global => "g",
                        Scope::Local => "l",
                        Scope::Register => unreachable!(),
                    };
                    println!("{indent}r{out_id}{grey}: {IDX_T}{reset} = {blue}{scope}idx{axis}{reset}    // 0..={ub}");
                }
                Op::Loop { len } => {
                    has_loops = true;
                    dtypes.insert(op_id, IDX_T);
                    println!("{indent}{bold}for{reset} r{out_id} in 0..{len} {{");
                    indent += "  ";
                }
                Op::If { condition } => {
                    let condition = id_map.get(&condition).copied().unwrap_or(OpId::NULL);
                    println!("{indent}{bold}if{reset} r{condition} {{");
                    indent += "  ";
                }
                Op::EndIf | Op::EndLoop => {
                    if indent.len() > 1 {
                        indent.pop();
                        indent.pop();
                    }
                    println!("{indent}}}");
                }
                Op::Vectorize { ref ops } => {
                    let dtype = dtypes[&ops[0]];
                    dtypes.insert(op_id, dtype);
                    let ops: Vec<OpId> = ops.iter().map(|x| id_map.get(x).copied().unwrap_or(OpId::NULL)).collect();
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}{grey}: {dtype}{reset} = {orange}vec{reset}{ops:?}    // {lb}..={ub}");
                    } else {
                        println!("{indent}r{out_id}{grey}: {dtype}{reset} = {orange}vec{reset}{ops:?}");
                    }
                }
                Op::Devectorize { vec, idx } => {
                    let dtype = dtypes[&vec];
                    dtypes.insert(op_id, dtype);
                    let vec = id_map.get(&vec).copied().unwrap_or(OpId::NULL);
                    if let Some((l, u)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}{grey}: {dtype}{reset} = r{vec}{orange}.s{idx}{reset}    // {l}..={u}",);
                    } else {
                        println!("{indent}r{out_id}{grey}: {dtype}{reset} = r{vec}{orange}.s{idx}{reset}");
                    }
                }
                Op::Move { x, ref mop } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    match mop.as_ref() {
                        MoveOp::Reshape { shape } => {
                            println!("{indent}r{out_id}{grey}: {dtype}{reset} = {cyan}reshape{reset} r{x} -> {shape:?}");
                        }
                        MoveOp::Expand { shape } => {
                            println!("{indent}r{out_id}{grey}: {dtype}{reset} = {cyan}expand{reset} r{x} -> {shape:?}");
                        }
                        MoveOp::Permute { axes, shape } => {
                            println!(
                                "{indent}r{out_id}{grey}: {dtype}{reset} = {cyan}permute{reset} r{x} axes={axes:?} -> {shape:?}",
                            );
                        }
                        MoveOp::Pad { padding, shape } => {
                            println!(
                                "{indent}r{out_id}{grey}: {dtype}{reset} = {cyan}pad{reset} r{x} padding={padding:?} -> {shape:?}",
                            );
                        }
                    }
                }
                Op::Barrier { scope } => {
                    println!("{indent}barrier {scope}");
                }
            }
            op_id = self.ops[op_id].next;
        }
    }
}
