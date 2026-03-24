// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

use crate::dtype::Constant;
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
        //println!("Kernel shape {:?}", self.shape);
        let mut indent = String::from(" ");
        let mut bounds: Map<OpId, (u32, u32)> = Map::default();
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
                    println!(
                        "{indent}{RED}r{out_id}{RESET}{GREY}: {dtype}{RESET} = {YELLOW}def {ro}{RESET}{scope}, len={len}"
                    );
                }
                Op::Const(value) => {
                    let dtype = value.dtype();
                    dtypes.insert(op_id, dtype);
                    if value.is_positive() {
                        let Constant::U32(v) = value.cast(DType::U32) else { unreachable!() };
                        bounds.insert(op_id, (v, v));
                    }
                    println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {MAGENTA}{value}{RESET}");
                }
                Op::Load { src, index, vlen: len } => {
                    let dtype = dtypes[&src];
                    dtypes.insert(op_id, dtype);
                    let (lb, ub) = bounds[&index];
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
                    let (lb, ub) = bounds[&index];
                    let dst = id_map[&dst];
                    let index = id_map[&index];
                    let x = id_map[&x];
                    if len > 1 {
                        println!(
                            "{indent}{RED}r{dst}{RESET}[r{index}..+len] = r{x}    // {lb}..={ub} {RED}store{RESET}",
                        );
                    } else {
                        println!("{indent}{RED}r{dst}{RESET}[r{index}] = r{x}    // {lb}..={ub} {RED}store{RESET}",);
                    }
                }
                Op::Cast { x, dtype } => {
                    dtypes.insert(op_id, dtype);
                    if let Some((l, u)) = bounds.get(&x) {
                        bounds.insert(op_id, (*l, *u));
                    }
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
                    if let Some((lb, ub)) = bounds.get(&x) {
                        bounds.insert(op_id, (*lb, *ub));
                    }
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
                    if let Some(&(xl, xu)) = bounds.get(&x)
                        && let Some(&(yl, yu)) = bounds.get(&y)
                    {
                        bounds.insert(
                            op_id,
                            match bop {
                                BOp::Add => (xl.wrapping_add(yl), xu.wrapping_add(yu)),
                                BOp::Sub => (xl.wrapping_sub(yl), xu.wrapping_sub(yu)),
                                BOp::Mul => (xl.wrapping_mul(yl), xu.wrapping_mul(yu)),
                                BOp::Div => (xl / yl, xu / yu),
                                BOp::Mod => (xl % yl, xu % yu),
                                BOp::Eq => ((xl == yl) as u32, (xu == yu) as u32),
                                BOp::NotEq => ((xl != yl) as u32, (xu != yu) as u32),
                                BOp::Cmpgt => ((xl > yl) as u32, (xu > yu) as u32),
                                BOp::Cmplt => ((xl < yl) as u32, (xu < yu) as u32),
                                BOp::And => ((xl == 1 && yl == 1) as u32, (xu == 1 && yu == 1) as u32),
                                BOp::BitShiftLeft => (xl << yl, xu << yu),
                                BOp::BitShiftRight => (xl >> yl, xu >> yu),
                                BOp::Pow => (xl.pow(yl as u32), xu.pow(yu as u32)),
                                BOp::Max => todo!(),
                                BOp::Or => todo!(),
                                BOp::BitXor => todo!(),
                                BOp::BitOr => todo!(),
                                BOp::BitAnd => todo!(),
                            },
                        );
                    }
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
                    if let Some(&(xl, xu)) = bounds.get(&x)
                        && let Some(&(yl, yu)) = bounds.get(&y)
                        && let Some(&(zl, zu)) = bounds.get(&z)
                    {
                        bounds.insert(
                            op_id,
                            (
                                xl.wrapping_mul(yl).wrapping_add(zl),
                                xu.wrapping_mul(yu).wrapping_add(zu),
                            ),
                        );
                    }
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
                    bounds.insert(op_id, (0, ub as u32));
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
                    bounds.insert(op_id, (0, len as u32 - 1));
                    println!("{indent}{BOLD}for{RESET} r{out_id} in 0..{len} {{");
                    indent += "  ";
                }
                Op::EndLoop => {
                    if indent.len() > 1 {
                        indent.pop();
                        indent.pop();
                    }
                    println!("{indent}}}");
                }
                Op::Vectorize { ref ops } => {
                    let dtype = dtypes[&ops[0]];
                    dtypes.insert(op_id, dtype);
                    let mut r = None;
                    for x in ops {
                        if let Some(&(xl, xu)) = bounds.get(x) {
                            if let Some((l, u)) = r {
                                r = Some((xl.min(l), xu.max(u)));
                            } else {
                                r = Some((xl, xu));
                            }
                        }
                    }
                    let ops: Vec<OpId> = ops.iter().map(|x| id_map.get(x).copied().unwrap_or(OpId::NULL)).collect();
                    if let Some((lb, ub)) = r {
                        bounds.insert(op_id, (lb, ub));
                        println!(
                            "{indent}r{out_id}{GREY}: {dtype}{RESET} = {ORANGE}vectorize{RESET}{ops:?}    // {lb}..={ub}",
                        );
                    } else {
                        println!("{indent}r{out_id}{GREY}: {dtype}{RESET} = {ORANGE}vectorize{RESET}{ops:?}");
                    }
                }
                Op::Devectorize { vec, idx } => {
                    let dtype = dtypes[&vec];
                    dtypes.insert(op_id, dtype);
                    if let Some((l, u)) = bounds.get(&vec) {
                        bounds.insert(op_id, (*l, *u));
                    }
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
                            println!(
                                "{indent}r{out_id}{GREY}: {dtype}{RESET} = {CYAN}reshape{RESET} r{x} -> {shape:?}",
                            );
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
            }
            op_id = self.ops[op_id].next;
        }
    }
}
