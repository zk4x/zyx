use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, Op, OpId},
};

impl Kernel {
    pub fn verify(&self) {
        let mut stack = Vec::new();
        stack.push(Set::default());
        let check = |op_id, x: OpId, stack: &[Set<OpId>]| {
            if !stack.iter().any(|set| set.contains(&x)) {
                println!(
                    "{op_id} {:?} uses {x} -> {:?} before declaration.",
                    self.ops[op_id].op, self.ops[x].op
                );
                self.debug();
                panic!();
            }
        };

        let mut op_id = self.head;
        let mut prev: OpId;
        let mut dtypes: Map<OpId, DType> = Map::default();
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::ConstView(ref x) => {
                    dtypes.insert(op_id, x.0.dtype());
                }
                Op::LoadView(ref x) => {
                    dtypes.insert(op_id, x.0);
                }
                Op::StoreView { src, .. } => {
                    check(op_id, src, &stack);
                    dtypes.insert(op_id, dtypes[&src]);
                }
                Op::Store { dst, x, index, vlen: _ } => {
                    check(op_id, dst, &stack);
                    check(op_id, x, &stack);
                    check(op_id, index, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Cast { x, dtype } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtype);
                }
                Op::Reduce { x, n_axes, .. } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                    if stack.len() > 1 {
                        for _ in 0..n_axes {
                            stack.pop();
                        }
                    }
                }
                Op::Unary { x, .. } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Binary { x, y, bop } => {
                    check(op_id, x, &stack);
                    check(op_id, y, &stack);
                    if dtypes[&x] != dtypes[&y] {
                        println!("Binary dtype mismatch on op={op_id}.");
                        self.debug();
                        panic!();
                    }
                    if bop.returns_bool() {
                        dtypes.insert(op_id, DType::Bool);
                    } else {
                        dtypes.insert(op_id, dtypes[&x]);
                    }
                }
                Op::Vectorize { ref ops } => {
                    let dtype = dtypes[&ops[0]];
                    for &x in ops {
                        check(op_id, x, &stack);
                        if dtypes[&x] != dtype {
                            println!("Vectorize dtype mismatch on op={op_id}.");
                            self.debug();
                            panic!();
                        }
                    }
                    dtypes.insert(op_id, dtype);
                }
                Op::Devectorize { .. } => todo!(),
                Op::WMMA { c, a, b, .. } => {
                    let dtype = dtypes[&c];
                    check(op_id, c, &stack);
                    check(op_id, a, &stack);
                    check(op_id, b, &stack);
                    if dtypes[&a] != dtypes[&b] {
                        println!("MMA dtype mismatch on op={op_id}.");
                        self.debug();
                        panic!();
                    }
                    dtypes.insert(op_id, dtype);
                }
                Op::Mad { x, y, z } => {
                    check(op_id, x, &stack);
                    check(op_id, y, &stack);
                    check(op_id, z, &stack);
                    if dtypes[&x] != dtypes[&y] || dtypes[&x] != dtypes[&z] {
                        println!("Mad dtype mismatch on op={op_id}.");
                        self.debug();
                        panic!();
                    }
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Const(v) => {
                    dtypes.insert(op_id, v.dtype());
                }
                Op::Define { dtype, .. } => {
                    dtypes.insert(op_id, dtype);
                }
                Op::Load { src, index, .. } => {
                    check(op_id, src, &stack);
                    check(op_id, index, &stack);
                    dtypes.insert(op_id, dtypes[&src]);
                }
                Op::Index { .. } => {
                    dtypes.insert(op_id, IDX_T);
                }
                Op::Loop { .. } => {
                    stack.push(Set::default());
                    dtypes.insert(op_id, IDX_T);
                }
                Op::EndLoop => {
                    if stack.is_empty() {
                        println!("Endloop without matching loop.");
                        self.debug();
                        panic!();
                    }
                    stack.pop();
                }
                Op::Move { x, .. } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                }
            }
            stack.last_mut().unwrap().insert(op_id);
            prev = op_id;
            op_id = self.ops[op_id].next;
            if !op_id.is_null() && self.ops[op_id].prev != prev {
                println!("Inconsistency in prev.");
                self.debug();
                panic!()
            }
        }
        if stack.len() != 1 {
            println!("Wrong {} closing endloops.", stack.len());
            self.debug();
            panic!();
        }
        self.check_oob();
    }

    pub fn check_oob(&self) {
        use std::collections::HashMap;
        let mut ids: Map<OpId, (usize, usize)> = HashMap::default();
        let mut defines = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Const(x) => {
                    if x.is_positive() {
                        let Constant::U64(x) = x.cast(DType::U64) else { unreachable!() };
                        let v = usize::from_le_bytes(x);
                        ids.insert(op_id, (v, v));
                    }
                }
                Op::Define { len, .. } => {
                    defines.insert(op_id, len);
                }
                Op::Cast { x, .. } => {
                    if let Some((l, u)) = ids.get(&x) {
                        ids.insert(op_id, (*l, *u));
                    }
                }
                Op::Binary { x, y, bop } => {
                    if let Some(&(xl, xu)) = ids.get(&x)
                        && let Some(&(yl, yu)) = ids.get(&y)
                    {
                        ids.insert(
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
                                op => todo!("{:?}", op),
                            },
                        );
                    }
                }
                Op::Mad { x, y, z } => {
                    if let Some(&(xl, xu)) = ids.get(&x)
                        && let Some(&(yl, yu)) = ids.get(&y)
                        && let Some(&(zl, zu)) = ids.get(&z)
                    {
                        ids.insert(
                            op_id,
                            (
                                xl.wrapping_mul(yl).wrapping_add(zl),
                                xu.wrapping_mul(yu).wrapping_add(zu),
                            ),
                        );
                    }
                }
                Op::Index { len: dim, .. } => {
                    ids.insert(op_id, (0, dim - 1));
                }
                Op::Loop { len: dim, .. } => {
                    ids.insert(op_id, (0, dim - 1));
                }
                Op::Load { src, index, .. } => {
                    if !ids.contains_key(&index) {
                        self.debug();
                        panic!("Missing index={index} for op_id={op_id} -> {:?}", self.ops[op_id]);
                    }
                    let idx_range = ids[&index];
                    //println!("Max idx range: {}, define {}", idx_range.1, defines[src]);
                    if idx_range.1 > defines[&src] - 1 {
                        self.debug();
                        panic!(
                            "OOB detected in op {}: index {:?} exceeds buffer length {:?}",
                            op_id, idx_range, defines[&src]
                        );
                    }
                }
                Op::Store { dst, index, .. } => {
                    if !ids.contains_key(&index) {
                        panic!("Missing index={index} for op_id={op_id} -> {:?}", self.ops[op_id]);
                    }
                    let idx_range = ids[&index];
                    //println!("Max idx range: {}, define {}", idx_range.1, defines[src]);
                    if idx_range.1 > defines[&dst] - 1 {
                        self.debug();
                        panic!(
                            "OOB detected in op {}: index {:?} exceeds buffer length {:?}",
                            op_id, idx_range, defines[&dst]
                        );
                    }
                }
                Op::Vectorize { ref ops } => {
                    let mut r = None;
                    for x in ops {
                        if let Some(&(xl, xu)) = ids.get(x) {
                            if let Some((l, u)) = r {
                                r = Some((xl.min(l), xu.max(u)));
                            } else {
                                r = Some((xl, xu));
                            }
                        }
                    }
                    if let Some((xl, xu)) = r {
                        ids.insert(op_id, (xl, xu));
                    }
                }
                _ => {}
            }
            op_id = self.ops[op_id].next;
        }
    }
}
