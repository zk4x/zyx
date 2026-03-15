use crate::dtype::Constant;
use crate::kernel::{BOp, IDX_T, Scope, UOp, MoveOp};
use crate::{BLUE, BOLD, CYAN, GREEN, GREY, MAGENTA, ORANGE, RED, RESET, YELLOW};
use crate::{
    DType, Map,
    kernel::{Kernel, Op, OpId},
    shape::Dim,
};

impl Kernel {
    pub fn debug(&self) {
        println!("\nloads={:?}", self.loads);
        println!("stores={:?}", self.stores);
        println!("outputs={:?}", self.outputs);
        //println!("Kernel shape {:?}", self.shape);
        let mut indent = String::from(" ");
        let mut bounds: Map<OpId, (Dim, Dim)> = Map::default();
        let mut dtypes: Map<OpId, DType> = Map::default();
        let mut op_id = self.head;
        let mut has_loops = false;
        let mut id_map = Map::default();
        let mut max_id = 0;
        while !op_id.is_null() {
            //println!("op_id={op_id}");
            max_id += 1;
            match *self.at(op_id) {
                Op::ConstView(ref x) => {
                    id_map.insert(op_id, max_id);
                    let value = x.0;
                    let view = &x.1;
                    let dtype = value.dtype();
                    dtypes.insert(op_id, dtype);
                    println!("{indent}r{max_id}{GREY}: {dtype}{RESET} = {CYAN}{value}{RESET} {view}");
                }
                Op::LoadView(ref x) => {
                    id_map.insert(op_id, max_id);
                    let dtype = x.0;
                    let view = &x.1;
                    dtypes.insert(op_id, dtype);
                    println!("{indent}r{max_id}{GREY}: {dtype}{RESET} = {CYAN}load{RESET} {view}");
                }
                Op::StoreView { src, dtype, .. } => {
                    id_map.insert(op_id, max_id);
                    let src = id_map[&src];
                    dtypes.insert(op_id, dtype);
                    println!("{indent}{CYAN}store{RESET} r{src}");
                }
                Op::Reduce { x, rop, n_axes, .. } => {
                    id_map.insert(op_id, max_id);
                    let dtype = dtypes[&x];
                    dtypes.insert(op_id, dtype);
                    let x = id_map[&x];
                    if has_loops {
                        indent.pop();
                        indent.pop();
                    }
                    println!(
                        "{indent}r{max_id}{GREY}: {dtype}{RESET} = {RED}reduce {}{RESET} r{x}, dims={n_axes:?} {}",
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
                    id_map.insert(op_id, max_id);
                    dtypes.insert(op_id, dtype);
                    let ro = if ro { "" } else { "mut " };
                    println!(
                        "{indent}{RED}r{max_id}{RESET}{GREY}: {dtype}{RESET} = {YELLOW}def {ro}{RESET}{scope}, len={len}"
                    );
                }
                Op::Const(value) => {
                    id_map.insert(op_id, max_id);
                    let dtype = value.dtype();
                    dtypes.insert(op_id, dtype);
                    if value.is_positive() {
                        let Constant::U64(v) = value.cast(DType::U64) else { unreachable!() };
                        let v = usize::from_le_bytes(v);
                        bounds.insert(op_id, (v, v));
                    }
                    println!("{indent}r{max_id}{GREY}: {dtype}{RESET} = {MAGENTA}{value}{RESET}");
                }
                Op::Load { src, index, vlen: len } => {
                    id_map.insert(op_id, max_id);
                    let dtype = dtypes[&src];
                    dtypes.insert(op_id, dtype);
                    let (lb, ub) = bounds[&index];
                    let src = id_map[&src];
                    let index = id_map[&index];
                    if len > 1 {
                        println!(
                            "{indent}r{max_id}{GREY}: {dtype}{RESET} = {RED}r{src}{RESET}[r{index}..r{index}+{len}]    // {lb}..={ub} {GREEN}load{RESET}"
                        );
                    } else {
                        println!(
                            "{indent}r{max_id}{GREY}: {dtype}{RESET} = {RED}r{src}{RESET}[r{index}]    // {lb}..={ub} {GREEN}load{RESET}"
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
                            "{indent}{RED}r{dst}{RESET}[r{index}..r{index}+len] = r{x}    // {lb}..={ub} {RED}store{RESET}",
                        );
                    } else {
                        println!("{indent}{RED}r{dst}{RESET}[r{index}] = r{x}    // {lb}..={ub} {RED}store{RESET}",);
                    }
                }
                Op::Cast { x, dtype } => {
                    id_map.insert(op_id, max_id);
                    dtypes.insert(op_id, dtype);
                    if let Some((l, u)) = bounds.get(&x) {
                        bounds.insert(op_id, (*l, *u));
                    }
                    let x = id_map[&x];
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        println!("{indent}r{max_id}{GREY}: {dtype}{RESET} = {dtype}(r{x})    // {lb}..={ub}");
                    } else {
                        println!("{indent}r{max_id}{GREY}: {dtype}{RESET} = {dtype}(r{x})");
                    }
                }
                Op::Unary { x, uop, .. } => {
                    id_map.insert(op_id, max_id);
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
                    let x = id_map[&x];
                    if let Some((lb, ub)) = bounds.get(&op_id) {
                        println!("{indent}r{max_id}{GREY}: {dtype}{RESET} = {op1}r{x}{op2}    // {lb}..={ub}");
                    } else {
                        println!("{indent}r{max_id}{GREY}: {dtype}{RESET} = {op1}r{x}{op2}");
                    }
                }
                Op::Binary { x, y, bop, .. } => {
                    id_map.insert(op_id, max_id);
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
                                BOp::Eq => ((xl == yl) as usize, (xu == yu) as usize),
                                BOp::NotEq => ((xl != yl) as usize, (xu != yu) as usize),
                                BOp::Cmpgt => ((xl > yl) as usize, (xu > yu) as usize),
                                BOp::Cmplt => ((xl < yl) as usize, (xu < yu) as usize),
                                BOp::And => ((xl == 1 && yl == 1) as usize, (xu == 1 && yu == 1) as usize),
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
                        println!("{indent}r{max_id}{GREY}: {dtype}{RESET} = {op1}r{x}{op2}r{y}{op3}    // {lb}..={ub}",);
                    } else {
                        println!("{indent}r{max_id}{GREY}: {dtype}{RESET} = {op1}r{x}{op2}r{y}{op3}",);
                    }
                }
                Op::Mad { x, y, z } => {
                    id_map.insert(op_id, max_id);
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
                    let x = id_map[&x];
                    let y = id_map[&y];
                    let z = id_map[&z];
                    if let Some((l, u)) = bounds.get(&op_id) {
                        println!("{indent}r{max_id}{GREY}: {dtype}{RESET} = {x} * {y} + {z}  // {l}..={u}");
                    } else {
                        println!("{indent}r{max_id}{GREY}: {dtype}{RESET} = {x} * {y} + {z}");
                    }
                }
                Op::WMMA { dims, layout, dtype, c, a, b } => {
                    dtypes.insert(op_id, dtypes[&c]);
                    println!(
                        "{op_id:>5}{indent}{ORANGE}WMMA{RESET} {} {dims:?}.{layout:?}.{dtype:?} c={c} a={a} b={b}",
                        dtypes[&op_id]
                    );
                }
                Op::Index { len, scope, axis } => {
                    id_map.insert(op_id, max_id);
                    dtypes.insert(op_id, IDX_T);
                    let ub = len - 1;
                    bounds.insert(op_id, (0, ub));
                    let scope = match scope {
                        Scope::Global => "g",
                        Scope::Local => "l",
                        Scope::Register => unreachable!(),
                    };
                    println!("{indent}r{max_id}{GREY}: {IDX_T}{RESET} = {BLUE}{scope}idx{axis}{RESET}  0..={ub}",);
                }
                Op::Loop { len, axis } => {
                    id_map.insert(op_id, max_id);
                    has_loops = true;
                    dtypes.insert(op_id, IDX_T);
                    bounds.insert(op_id, (0, len - 1));
                    println!("{indent}{BOLD}for{RESET} r{max_id} in 0..{len} {{ // {BLUE}ridx{axis}{RESET}");
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
                    id_map.insert(op_id, max_id);
                    dtypes.insert(op_id, dtypes[&ops[0]]);
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
                    if let Some((xl, xu)) = r {
                        bounds.insert(op_id, (xl, xu));
                        println!(
                            "{op_id:>5}{indent}{ORANGE}VECTORIZE{RESET} {} {ops:?}    {xl}..={xu}",
                            dtypes[&op_id]
                        );
                    } else {
                        println!("{op_id:>5}{indent}{ORANGE}VECTORIZE{RESET} {} {ops:?}", dtypes[&op_id]);
                    }
                }
                Op::Devectorize { vec, idx } => {
                    id_map.insert(op_id, max_id);
                    dtypes.insert(op_id, dtypes[&vec]);
                    if let Some((l, u)) = bounds.get(&vec) {
                        bounds.insert(op_id, (*l, *u));
                    }
                    if let Some((l, u)) = bounds.get(&op_id) {
                        println!(
                            "{op_id:>5}{indent}{ORANGE}DEVECTORIZE{RESET} {} {vec}[{idx}]    {l}..={u}",
                            dtypes[&op_id]
                        );
                    } else {
                        println!(
                            "{op_id:>5}{indent}{ORANGE}DEVECTORIZE{RESET} {} {vec}[{idx}]",
                            dtypes[&op_id]
                        );
                    }
                }
                Op::Move { x, ref mop } => {
                    dtypes.insert(op_id, dtypes[&x]);
                    match mop.as_ref() {
                        MoveOp::Reshape { shape } => {
                            println!(
                                "{op_id:>5}{indent}{CYAN}RESHAPE{RESET} {} {x} -> {shape:?}",
                                dtypes[&op_id]
                            );
                        }
                        MoveOp::Expand { shape } => {
                            println!(
                                "{op_id:>5}{indent}{CYAN}EXPAND{RESET} {} {x} -> {shape:?}",
                                dtypes[&op_id]
                            );
                        }
                        MoveOp::Permute { axes, shape } => {
                            println!(
                                "{op_id:>5}{indent}{CYAN}PERMUTE{RESET} {} {x} axes={axes:?} -> {shape:?}",
                                dtypes[&op_id]
                            );
                        }
                        MoveOp::Pad { padding, shape } => {
                            println!(
                                "{op_id:>5}{indent}{CYAN}PAD{RESET} {} {x} padding={padding:?} -> {shape:?}",
                                dtypes[&op_id]
                            );
                        }
                    };
                }
            }
            op_id = self.ops[op_id].next;
        }
    }
}
