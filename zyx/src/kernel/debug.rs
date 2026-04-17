// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::kernel::{BOp, IDX_T, MoveOp, Scope, UOp};
use crate::slab::SlabId;
use crate::{BLUE, BOLD, CYAN, GREEN, GREY, MAGENTA, ORANGE, RED, RESET, YELLOW};
use crate::{
    DType, Map,
    kernel::{Kernel, Op, OpId},
};

impl Kernel {
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
                    println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {CYAN}{value}{RESET} {view}");
                }
                Op::LoadView(ref x) => {
                    let dtype = x.0;
                    let view = &x.1;
                    dtypes.insert(op_id, dtype);
                    println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {CYAN}load{RESET} {view}");
                }
                Op::StoreView { src, dtype, .. } => {
                    let src = id_map[&src];
                    dtypes.insert(op_id, dtype);
                    println!("{indent}{CYAN}store{RESET} r{src}");
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
                        "{indent}r{out_id}{GREY}: {dtype}{RESET} = {RED}reduce {}{RESET} r{x}, dims={n_axes:?} {}",
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
                    println!("{indent}{RED}r{out_id}{RESET}{GREY}: {dtype}{RESET} = {YELLOW}def {ro}{RESET}{scope}, len={len}");
                }
                Op::Const(value) => {
                    let dtype = value.dtype();
                    dtypes.insert(op_id, dtype);
                    println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {MAGENTA}{value}{RESET}");
                }
                Op::Load { src, index, vlen: len } => {
                    let dtype = dtypes[&src];
                    dtypes.insert(op_id, dtype);
                    let (lb, ub) = bounds.get(&index).copied().unwrap_or((0, 0));
                    let src = id_map[&src];
                    let index = id_map[&index];
                    if len > 1 {
                        println!(
                            "{indent}r{out_id}{GREY}: {dtype}{RESET} = {RED}r{src}{RESET}[r{index}..+{len}]    // {lb}..={ub} {GREEN}load{RESET}"
                        );
                    } else {
                        println!(
                            "{indent}r{out_id}{GREY}: {dtype}{RESET} = {RED}r{src}{RESET}[r{index}]    // {lb}..={ub} {GREEN}load{RESET}"
                        );
                    }
                }
                Op::Store { dst, x, index, vlen: len } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let (lb, ub) = bounds.get(&index).copied().unwrap_or((0, 0));
                    let dst = id_map[&dst];
                    let index = id_map[&index];
                    let x = id_map[&x];
                    if len > 1 {
                        println!("{indent}{RED}r{dst}{RESET}[r{index}..+len] = r{x}    // {lb}..={ub} {RED}store{RESET}",);
                    } else {
                        println!("{indent}{RED}r{dst}{RESET}[r{index}] = r{x}    // {lb}..={ub} {RED}store{RESET}",);
                    }
                }
                Op::Cast { x, dtype } => {
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {dtype}(r{x})    // {lb}..={ub}");
                    } else {
                        println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {dtype}(r{x})");
                    }
                }
                Op::Unary { x, uop, .. } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let (op1, op2) = match uop {
                        UOp::Neg => ("-", ""),
                        UOp::BitNot => ("~", ""),
                        UOp::Exp2 => ("exp2(", ")"),
                        UOp::Log2 => ("log2(", ")"),
                        UOp::Reciprocal => ("1/", ""),
                        UOp::Sqrt => ("sqrt(", ")"),
                        UOp::Sin => ("sin(", ")"),
                        UOp::Cos => ("cos(", ")"),
                        UOp::Floor => ("floor(", ")"),
                        UOp::Trunc => ("trunc(", ")"),
                    };
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {op1}r{x}{op2}    // {lb}..={ub}");
                    } else {
                        println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {op1}r{x}{op2}");
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
                    let x = id_map[&x];
                    let y = id_map[&y];
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {op1}r{x}{op2}r{y}{op3}    // {lb}..={ub}",);
                    } else {
                        println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {op1}r{x}{op2}r{y}{op3}",);
                    }
                }
                Op::Mad { x, y, z } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    let y = id_map.get(&y).copied().unwrap_or(OpId::NULL);
                    let z = id_map.get(&z).copied().unwrap_or(OpId::NULL);
                    if let Some((l, u)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = r{x} * r{y} + r{z}    // {l}..={u}");
                    } else {
                        println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = r{x} * r{y} + r{z}");
                    }
                }
                Op::WMMA { dims, layout, dtype, c, a, b } => {
                    let cdtype = dtypes[&c];
                    dtypes.insert(op_id, cdtype);
                    let a = id_map.get(&a).copied().unwrap_or(OpId::NULL);
                    let b = id_map.get(&b).copied().unwrap_or(OpId::NULL);
                    let c = id_map.get(&c).copied().unwrap_or(OpId::NULL);
                    println!(
                        "{indent}r{out_id}{GREY}: {cdtype}{RESET} = {ORANGE}wmma{RESET}.{dims:?}.{layout:?}.{dtype:?}(c={c}, a={a}, b={b})",
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
                    println!("{indent}r{out_id}{GREY}: {IDX_T}{RESET} = {BLUE}{scope}idx{axis}{RESET}    // 0..={ub}",);
                }
                Op::Loop { len } => {
                    has_loops = true;
                    dtypes.insert(op_id, IDX_T);
                    println!("{indent}{BOLD}for{RESET} r{out_id} in 0..{len} {{");
                    indent += "  ";
                }
                Op::If { condition } => {
                    let condition = id_map.get(&condition).copied().unwrap_or(OpId::NULL);
                    println!("{indent}{BOLD}if{RESET} r{condition} {{");
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
                        println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {ORANGE}vectorize{RESET}{ops:?}    // {lb}..={ub}",);
                    } else {
                        println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {ORANGE}vectorize{RESET}{ops:?}");
                    }
                }
                Op::Devectorize { vec, idx } => {
                    let dtype = dtypes[&vec];
                    dtypes.insert(op_id, dtype);
                    let vec = id_map.get(&vec).copied().unwrap_or(OpId::NULL);
                    if let Some((l, u)) = bounds.get(&op_id) {
                        println!(
                            "{indent}r{out_id}{GREY}: {dtype}{RESET} = {ORANGE}devectorize{RESET} r{vec}[{idx}]    // {l}..={u}",
                        );
                    } else {
                        println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {ORANGE}DEVECTORIZE{RESET} r{vec}[{idx}]",);
                    }
                }
                Op::Move { x, ref mop } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    match mop.as_ref() {
                        MoveOp::Reshape { shape } => {
                            println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {CYAN}reshape{RESET} r{x} -> {shape:?}",);
                        }
                        MoveOp::Expand { shape } => {
                            println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {CYAN}expand{RESET} r{x} -> {shape:?}");
                        }
                        MoveOp::Permute { axes, shape } => {
                            println!(
                                "{indent}r{out_id}{GREY}: {dtype}{RESET} = {CYAN}permute{RESET} r{x} axes={axes:?} -> {shape:?}",
                            );
                        }
                        MoveOp::Pad { padding, shape } => {
                            println!(
                                "{indent}r{out_id}{GREY}: {dtype}{RESET} = {CYAN}pad{RESET} r{x} padding={padding:?} -> {shape:?}",
                            );
                        }
                    };
                }
                Op::Barrier { scope } => {
                    println!("{indent}barrier {scope}");
                }
            }
            op_id = self.ops[op_id].next;
        }
    }

    #[allow(unused)]
    pub fn debug_colorless(&self) {
        println!(); // Just an empty line for readability
        let remap_ids = false;
        let mut indent = String::from(" ");
        let bounds = self.compute_bounds();
        let mut dtypes: Map<OpId, DType> = Map::default();
        let mut op_id = self.head;
        let mut has_loops = false;
        let mut id_map = Map::default();
        let mut max_id = OpId::ZERO;
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
                    println!("{indent}r{out_id}: {dtype} = {value} {view}");
                }
                Op::LoadView(ref x) => {
                    let dtype = x.0;
                    let view = &x.1;
                    dtypes.insert(op_id, dtype);
                    println!("{indent}r{out_id}: {dtype} = load {view}");
                }
                Op::StoreView { src, dtype, .. } => {
                    let src = id_map[&src];
                    dtypes.insert(op_id, dtype);
                    println!("{indent}store r{src}");
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
                        "{indent}r{out_id}: {dtype} = reduce {} r{x}, dims={n_axes:?} {}",
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
                    println!("{indent}r{out_id}: {dtype} = def {ro}{scope}, len={len}");
                }
                Op::Const(value) => {
                    let dtype = value.dtype();
                    dtypes.insert(op_id, dtype);
                    println!("{indent}r{out_id}: {dtype} = {value}");
                }
                Op::Load { src, index, vlen: len } => {
                    let dtype = dtypes[&src];
                    dtypes.insert(op_id, dtype);
                    let (lb, ub) = bounds.get(&index).copied().unwrap_or((0, 0));
                    let src = id_map[&src];
                    let index = id_map[&index];
                    if len > 1 {
                        println!("{indent}r{out_id}: {dtype} = r{src}[r{index}..+{len}]    // {lb}..={ub} load");
                    } else {
                        println!("{indent}r{out_id}: {dtype} = r{src}[r{index}]    // {lb}..={ub} load");
                    }
                }
                Op::Store { dst, x, index, vlen: len } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let (lb, ub) = bounds.get(&index).copied().unwrap_or((0, 0));
                    let dst = id_map[&dst];
                    let index = id_map[&index];
                    let x = id_map[&x];
                    if len > 1 {
                        println!("{indent}r{dst}[r{index}..+len] = r{x}    // {lb}..={ub} store",);
                    } else {
                        println!("{indent}r{dst}[r{index}] = r{x}    // {lb}..={ub} store",);
                    }
                }
                Op::Barrier { scope } => {
                    println!("{indent}barrier {scope}");
                }
                Op::Cast { x, dtype } => {
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}: {dtype} = {dtype}(r{x})    // {lb}..={ub}");
                    } else {
                        println!("{indent}r{out_id}: {dtype} = {dtype}(r{x})");
                    }
                }
                Op::Unary { x, uop, .. } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let (op1, op2) = match uop {
                        UOp::Neg => ("-", ""),
                        UOp::BitNot => ("~", ""),
                        UOp::Exp2 => ("exp2(", ")"),
                        UOp::Log2 => ("log2(", ")"),
                        UOp::Reciprocal => ("1/", ""),
                        UOp::Sqrt => ("sqrt(", ")"),
                        UOp::Sin => ("sin(", ")"),
                        UOp::Cos => ("cos(", ")"),
                        UOp::Floor => ("floor(", ")"),
                        UOp::Trunc => ("trunc(", ")"),
                    };
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}: {dtype} = {op1}r{x}{op2}    // {lb}..={ub}");
                    } else {
                        println!("{indent}r{out_id}: {dtype} = {op1}r{x}{op2}");
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
                    let x = id_map[&x];
                    let y = id_map[&y];
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}: {dtype} = {op1}r{x}{op2}r{y}{op3}    // {lb}..={ub}",);
                    } else {
                        println!("{indent}r{out_id}: {dtype} = {op1}r{x}{op2}r{y}{op3}",);
                    }
                }
                Op::Mad { x, y, z } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    let y = id_map.get(&y).copied().unwrap_or(OpId::NULL);
                    let z = id_map.get(&z).copied().unwrap_or(OpId::NULL);
                    if let Some((l, u)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}: {dtype} = r{x} * r{y} + r{z}    // {l}..={u}");
                    } else {
                        println!("{indent}r{out_id}: {dtype} = r{x} * r{y} + r{z}");
                    }
                }
                Op::WMMA { dims, layout, dtype, c, a, b } => {
                    let cdtype = dtypes[&c];
                    dtypes.insert(op_id, cdtype);
                    let a = id_map.get(&a).copied().unwrap_or(OpId::NULL);
                    let b = id_map.get(&b).copied().unwrap_or(OpId::NULL);
                    let c = id_map.get(&c).copied().unwrap_or(OpId::NULL);
                    println!("{indent}r{out_id}: {cdtype} = wmma.{dims:?}.{layout:?}.{dtype:?}(c={c}, a={a}, b={b})",);
                }
                Op::Index { len, scope, axis } => {
                    dtypes.insert(op_id, IDX_T);
                    let ub = len - 1;
                    let scope = match scope {
                        Scope::Global => "g",
                        Scope::Local => "l",
                        Scope::Register => unreachable!(),
                    };
                    println!("{indent}r{out_id}: {IDX_T} = {scope}idx{axis}    // 0..={ub}",);
                }
                Op::Loop { len } => {
                    has_loops = true;
                    dtypes.insert(op_id, IDX_T);
                    println!("{indent}for r{out_id} in 0..{len} {{");
                    indent += "  ";
                }
                Op::If { condition } => {
                    let condition = id_map.get(&condition).copied().unwrap_or(OpId::NULL);
                    println!("{indent}if r{condition} {{");
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
                        println!("{indent}r{out_id}: {dtype} = vectorize{:?}    // {}..={}", ops, lb, ub,);
                    } else {
                        println!("{indent}r{out_id}: {dtype} = vectorize{:?}", ops);
                    }
                }
                Op::Devectorize { vec, idx } => {
                    let dtype = dtypes[&vec];
                    dtypes.insert(op_id, dtype);
                    let vec = id_map.get(&vec).copied().unwrap_or(OpId::NULL);
                    if let Some((l, u)) = bounds.get(&op_id) {
                        println!("{indent}r{out_id}: {dtype} = devectorize r{vec}[{idx}]    // {}..={}", l, u,);
                    } else {
                        println!("{indent}r{out_id}: {dtype} = DEVECTORIZE r{vec}[{idx}]",);
                    }
                }
                Op::Move { x, ref mop } => {
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let x = id_map.get(&x).copied().unwrap_or(OpId::NULL);
                    match mop.as_ref() {
                        MoveOp::Reshape { shape } => {
                            println!("{indent}r{out_id}: {dtype} = reshape r{x} -> {shape:?}",);
                        }
                        MoveOp::Expand { shape } => {
                            println!("{indent}r{out_id}: {dtype} = expand r{x} -> {shape:?}");
                        }
                        MoveOp::Permute { axes, shape } => {
                            println!("{indent}r{out_id}: {dtype} = permute r{x} axes={axes:?} -> {shape:?}",);
                        }
                        MoveOp::Pad { padding, shape } => {
                            println!("{indent}r{out_id}: {dtype} = pad r{x} padding={padding:?} -> {shape:?}",);
                        }
                    };
                }
            }
            op_id = self.ops[op_id].next;
        }
    }
}
