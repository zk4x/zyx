// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, Op, OpId},
    shape::Dim,
};

impl Kernel {
    pub fn verify(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        let mut stack = Vec::new();
        stack.push(Set::default());
        let check = |op_id, x: OpId, stack: &[Set<OpId>]| {
            if !stack.iter().any(|set| set.contains(&x)) {
                println!(
                    "{op_id} {:?} uses {x} -> {:?} before declaration.",
                    self.ops[op_id].op, self.ops[x].op
                );
                self.debug_colorless();
                panic!();
            }
        };

        let mut defines = Map::default();

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
                    if !defines.contains_key(&dst) {
                        println!("store={op_id} is trying to store to undefined variable");
                        self.debug_colorless();
                        panic!();
                    }
                    debug_assert_eq!(dtypes[&index], IDX_T);
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
                        self.debug_colorless();
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
                            self.debug_colorless();
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
                        self.debug_colorless();
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
                        self.debug_colorless();
                        panic!();
                    }
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Const(v) => {
                    dtypes.insert(op_id, v.dtype());
                }
                Op::Define { dtype, scope, ro, len } => {
                    defines.insert(op_id, (scope, ro, len));
                    dtypes.insert(op_id, dtype);
                }
                Op::Load { src, index, .. } => {
                    if !defines.contains_key(&src) {
                        println!("load={op_id} is trying to load from undefined variable");
                        self.debug_colorless();
                        panic!();
                    }
                    debug_assert_eq!(dtypes[&index], IDX_T);
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
                        self.debug_colorless();
                        panic!();
                    }
                    stack.pop();
                }
                Op::If { condition } => {
                    if dtypes[&condition] != DType::Bool {
                        println!("If condition={condition} must be a boolean");
                        self.debug_colorless();
                        panic!();
                    }
                    stack.push(Set::default());
                }
                Op::EndIf => {
                    stack.pop();
                }
                Op::Move { x, .. } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Barrier { .. } => {}
            }
            stack.last_mut().unwrap().insert(op_id);
            prev = op_id;
            op_id = self.ops[op_id].next;
            if !op_id.is_null() && self.ops[op_id].prev != prev {
                println!("Inconsistency in prev.");
                self.debug_colorless();
                panic!()
            }
        }
        if stack.len() != 1 {
            println!("Wrong {} closing endloops.", stack.len());
            self.debug_colorless();
            panic!();
        }
        self.check_oob();
    }

    pub fn is_masked_index(&self, index: OpId, bounds: &Map<OpId, (Dim, Dim)>) -> bool {
        let mut stack = vec![index];
        let mut visited = Set::default();
        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }
            if let Some(&(l, u)) = bounds.get(&id) {
                if l == 0 && u == 1 {
                    return true;
                }
            }
            match self.ops[id].op {
                Op::Binary { x, y, .. } => {
                    stack.push(x);
                    stack.push(y);
                }
                Op::Mad { x, y, z } => {
                    stack.push(x);
                    stack.push(y);
                    stack.push(z);
                }
                Op::Cast { x, .. } => {
                    stack.push(x);
                }
                _ => {}
            }
        }
        false
    }

    pub fn check_oob(&self) {
        let bounds = self.compute_bounds();
        let mut defines = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Define { len, .. } => {
                    defines.insert(op_id, len);
                }
                Op::Load { src, index, .. } => {
                    if let Some(&idx_range) = bounds.get(&index) {
                        if idx_range.1 > defines[&src] - 1 {
                            if !self.is_masked_index(index, &bounds) {
                                self.debug_colorless();
                                panic!(
                                    "OOB detected in op {}: index {:?} exceeds buffer length {:?}",
                                    op_id, idx_range, defines[&src]
                                );
                            }
                        }
                    }
                }
                Op::Store { dst, index, .. } => {
                    if let Some(&idx_range) = bounds.get(&index) {
                        if idx_range.1 > defines[&dst] - 1 {
                            if !self.is_masked_index(index, &bounds) {
                                self.debug_colorless();
                                panic!(
                                    "OOB detected in op {}: index {:?} exceeds buffer length {:?}",
                                    op_id, idx_range, defines[&dst]
                                );
                            }
                        }
                    }
                }
                _ => {}
            }
            op_id = self.ops[op_id].next;
        }
    }

    pub fn compute_bounds(&self) -> Map<OpId, (Dim, Dim)> {
        let mut bounds: Map<OpId, (Dim, Dim)> = Map::default();
        let mut bounds_stack: Vec<Map<OpId, (Dim, Dim)>> = vec![Map::default()];
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Const(x) => {
                    let b = bounds_stack.last_mut().unwrap();
                    if x.is_positive() {
                        let Constant::U64(x) = x.cast(DType::U64) else { unreachable!() };
                        let v = u64::from_le_bytes(x);
                        b.insert(op_id, (v, v));
                    }
                }
                Op::Define { .. } => {}
                Op::Cast { x, .. } => {
                    let b = bounds_stack.last_mut().unwrap();
                    if let Some((l, u)) = b.get(&x) {
                        b.insert(op_id, (*l, *u));
                    }
                }
                Op::Binary { x, y, bop } => {
                    let b = bounds_stack.last_mut().unwrap();
                    if let Some(&(min_x, max_x)) = b.get(&x)
                        && let Some(&(min_y, max_y)) = b.get(&y)
                    {
                        let range = match bop {
                            BOp::Add => (min_x.wrapping_add(min_y), max_x.wrapping_add(max_y)),
                            BOp::Sub => (min_x.wrapping_sub(min_y), max_x.wrapping_sub(max_y)),
                            BOp::Mul => (min_x.wrapping_mul(min_y), max_x.wrapping_mul(max_y)),
                            BOp::Div | BOp::Mod if min_y == 0 || max_y == 0 => (0, Dim::MAX),
                            BOp::Div => (min_x / min_y, max_x / max_y),
                            BOp::Mod => (min_x % min_y, max_x % max_y),
                            BOp::BitShiftLeft => (min_x << min_y, max_x << max_y),
                            BOp::BitShiftRight => (min_x >> min_y, max_x >> max_y),
                            BOp::Pow => (min_x.pow(min_y as u32), max_x.pow(max_y as u32)),
                            BOp::Eq => {
                                // x == y
                                let always = (min_x == max_x) && (min_y == max_y) && (min_x == min_y);
                                let maybe = !(max_x < min_y || max_y < min_x) && !always;
                                let lower = if always { 1 } else { 0 };
                                let upper = if always || maybe { 1 } else { 0 };
                                (lower, upper)
                            }
                            BOp::NotEq => {
                                // x != y
                                let always = max_x < min_y || max_y < min_x; // disjoint ranges → always true
                                let maybe = !(min_x == max_x && min_y == max_y && min_x == min_y) && !always;
                                let lower = if always { 1 } else { 0 };
                                let upper = if always || maybe { 1 } else { 0 };
                                (lower, upper)
                            }
                            BOp::Cmpgt => {
                                // x > y
                                let always = min_x > max_y; // min(x) > max(y) → always true
                                let never = max_x <= min_y; // max(x) <= min(y) → always false
                                let maybe = !always && !never;
                                let lower = if always { 1 } else { 0 };
                                let upper = if always || maybe { 1 } else { 0 };
                                (lower, upper)
                            }
                            BOp::Cmplt => {
                                // x < y
                                let always = max_x < min_y; // max(x) < min(y) → always true
                                let never = max_y <= min_x; // max(y) <= min(x) → always false
                                let maybe = !always && !never;
                                let lower = if always { 1 } else { 0 };
                                let upper = if always || maybe { 1 } else { 0 };
                                (lower, upper)
                            }
                            BOp::And => {
                                // x & y
                                let always = min_x == 1 && max_x == 1 && min_y == 1 && max_y == 1;
                                let maybe = max_x >= 1 && max_y >= 1;
                                let lower = if always { 1 } else { 0 };
                                let upper = if always || maybe { 1 } else { 0 };
                                (lower, upper)
                            }
                            BOp::Or => {
                                // x | y
                                let always = max_x == 1 && max_y == 1;
                                let maybe = min_x == 1 || min_y == 1 || max_x == 1 || max_y == 1;
                                let lower = if always {
                                    1
                                } else if maybe {
                                    0
                                } else {
                                    0
                                };
                                let upper = if always || maybe { 1 } else { 0 };
                                (lower, upper)
                            }
                            BOp::Max => (min_x.min(min_y), max_x.max(max_y)),
                            _ => (0, 0),
                        };
                        b.insert(op_id, range);
                    }
                }
                Op::Mad { x, y, z } => {
                    let b = bounds_stack.last_mut().unwrap();
                    if let Some(&(xl, xu)) = b.get(&x)
                        && let Some(&(yl, yu)) = b.get(&y)
                        && let Some(&(zl, zu)) = b.get(&z)
                    {
                        b.insert(
                            op_id,
                            (xl.wrapping_mul(yl).wrapping_add(zl), xu.wrapping_mul(yu).wrapping_add(zu)),
                        );
                    }
                }
                Op::If { condition } => {
                    let mut prev = bounds_stack.last().unwrap().clone();
                    let mut params = Vec::new();
                    params.push(condition);
                    while let Some(param) = params.pop() {
                        if let Op::Binary { x, y, bop } = self.at(param) {
                            match bop {
                                BOp::Eq => {
                                    if let Some((yl, yu)) = prev.get(y) {
                                        if yl == yu {
                                            if let Some((_xl, _xu)) = prev.get(x) {
                                                prev.insert(*x, (*yl, *yu));
                                            }
                                        }
                                    }
                                }
                                BOp::Cmplt => {
                                    if let Some((yl, yu)) = prev.get(y) {
                                        if yl == yu {
                                            if let Some((xl, _xu)) = prev.get(x) {
                                                prev.insert(*x, (*xl, yl.saturating_sub(1)));
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        params.extend(self.ops[param].op.parameters());
                    }
                    bounds_stack.push(prev);
                }
                Op::EndIf => {
                    bounds_stack.pop();
                }
                Op::Index { len, .. } | Op::Loop { len } => {
                    let b = bounds_stack.last_mut().unwrap();
                    b.insert(op_id, (0, len as Dim - 1));
                }
                Op::Vectorize { ref ops } => {
                    let b = bounds_stack.last_mut().unwrap();
                    let mut r = None;
                    for x in ops {
                        if let Some(&(xl, xu)) = b.get(x) {
                            if let Some((l, u)) = r {
                                r = Some((xl.min(l), xu.max(u)));
                            } else {
                                r = Some((xl, xu));
                            }
                        }
                    }
                    if let Some((xl, xu)) = r {
                        b.insert(op_id, (xl, xu));
                    }
                }
                _ => {}
            }
            // Merge current scope bounds into the global bounds map
            if let Some(scope_bounds) = bounds_stack.last() {
                for (&k, &v) in scope_bounds {
                    bounds.insert(k, v);
                }
            }
            op_id = self.ops[op_id].next;
        }
        bounds
    }
}
