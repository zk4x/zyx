// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use std::ops::RangeInclusive;

use crate::{
    DType, Map, Set,
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
                self.debug();
                panic!();
            }
        };

        let mut gids = Set::default();
        let mut lids = Set::default();

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
                Op::Store { dst, x, index, .. } => {
                    if !defines.contains_key(&dst) {
                        println!("store={op_id} is trying to store to undefined variable");
                        self.debug();
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
                Op::Unary { x, .. } | Op::Move { x, .. } => {
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
                Op::Wmma { c, a, b, .. } => {
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
                Op::Define { dtype, scope, ro, len } => {
                    defines.insert(op_id, (scope, ro, len));
                    dtypes.insert(op_id, dtype);
                }
                Op::Load { src, index, .. } => {
                    if !defines.contains_key(&src) {
                        println!("load={op_id} is trying to load from undefined variable");
                        self.debug();
                        panic!();
                    }
                    debug_assert_eq!(dtypes[&index], IDX_T);
                    check(op_id, src, &stack);
                    check(op_id, index, &stack);
                    dtypes.insert(op_id, dtypes[&src]);
                }
                Op::Index { axis, scope, .. } => {
                    match scope {
                        super::Scope::Global => {
                            if !gids.insert(axis) {
                                println!("index={op_id} is using global axis={axis} for the second time");
                                self.debug();
                                panic!();
                            }
                        }
                        super::Scope::Local => {
                            if !lids.insert(axis) {
                                println!("index={op_id} is using local axis={axis} for the second time");
                                self.debug();
                                panic!();
                            }
                        }
                        super::Scope::Register => unreachable!(),
                    }

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
                Op::If { condition } => {
                    if dtypes[&condition] != DType::Bool {
                        println!("If condition={condition} must be a boolean");
                        self.debug();
                        panic!();
                    }
                    stack.push(Set::default());
                }
                Op::EndIf => {
                    stack.pop();
                }
                Op::Barrier { .. } => {}
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
        let mut defines = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Define { len, .. } => {
                    defines.insert(op_id, len);
                }
                Op::Load { src, index, .. } => {
                    let idx_range = Self::get_bounds(index);
                    if let Some(range) = idx_range {
                        if *range.end() >= defines[&src] {
                            self.debug();
                            panic!(
                                "OOB detected in op {}: index {:?} exceeds buffer length {:?}",
                                op_id, range, defines[&src]
                            );
                        }
                    }
                }
                Op::Store { dst, index, .. } => {
                    let idx_range = Self::get_bounds(index);
                    if let Some(range) = idx_range {
                        if *range.start() > defines[&dst] + 1 {
                            self.debug();
                            panic!(
                                "OOB detected in op {}: index {:?} exceeds buffer length {:?}",
                                op_id, range, defines[&dst]
                            );
                        }
                    }
                }
                _ => {}
            }
            op_id = self.ops[op_id].next;
        }
    }
}

impl Kernel {
    #[allow(clippy::match_same_arms)]
    pub fn compute_bounds(&self) -> Map<OpId, (Dim, Dim)> {
        let mut bounds: Map<OpId, (Dim, Dim)> = Map::default();
        let mut bounds_stack: Vec<Map<OpId, (Dim, Dim)>> = vec![Map::default()];
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Const(x) => {
                    let b = bounds_stack.last_mut().unwrap();
                    if let Some(v) = x.as_dim() {
                        b.insert(op_id, (v, v));
                    }
                }
                Op::Define { .. } => {}
                Op::Cast { .. } | Op::Binary { .. } | Op::Mad { .. } => {
                    let b = bounds_stack.last_mut().unwrap();
                    self.rederive_bounds(b, op_id);
                }
                Op::If { condition } => {
                    let mut prev = bounds_stack.last().unwrap().clone();
                    let mut skip_rederive = Set::default();
                    let mut params = Vec::new();
                    params.push(condition);
                    while let Some(param) = params.pop() {
                        if let Op::Binary { x, y, bop } = self.at(param) {
                            match bop {
                                BOp::Eq => {
                                    if let Some((yl, yu)) = prev.get(y) {
                                        if yl == yu {
                                            if let Some((_xl, _xu)) = prev.get(x) {
                                                let x_id = *x;
                                                let yl = *yl;
                                                let yu = *yu;
                                                prev.insert(x_id, (yl, yu));
                                                self.backward_constrain(x_id, yl, yu, &mut prev, &mut skip_rederive);
                                            }
                                        }
                                    }
                                }
                                BOp::Cmplt => {
                                    if let Some((yl, yu)) = prev.get(y) {
                                        if yl == yu {
                                            if let Some((xl, _xu)) = prev.get(x) {
                                                let x_id = *x;
                                                let xl = *xl;
                                                let new_upper = yl.saturating_sub(1);
                                                prev.insert(x_id, (xl, new_upper));
                                                // Don't add x_id to skip_rederive — the re-derive will
                                                // recompute it from the backward-constrained operands
                                                // correctly (and possibly tighter).
                                                self.backward_constrain(x_id, xl, new_upper, &mut prev, &mut skip_rederive);
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        params.extend(self.ops[param].op.parameters());
                    }
                    // Re-derive bounds for all ops up to this point in case any depend
                    // on the newly constrained variables (e.g. pad_index wraps a store in
                    // Op::If but the store index was computed before the If and used the
                    // unconstrained range).  Skip variables that were just hand-constrained
                    // — re-derive would overwrite them using stale operand bounds.
                    let mut scan = self.head;
                    while scan != op_id {
                        if !skip_rederive.contains(&scan) {
                            self.rederive_bounds(&mut prev, scan);
                        }
                        scan = self.ops[scan].next;
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
            // Merge current scope bounds into the global bounds map.
            // Skip at EndIf — parent scope entries are stale inside the If body
            // and would overwrite the refined bounds that were already merged
            // from the If scope during body processing.
            if !matches!(*self.at(op_id), Op::EndIf) {
                if let Some(scope_bounds) = bounds_stack.last() {
                    for (&k, &v) in scope_bounds {
                        bounds.insert(k, v);
                    }
                }
            }
            op_id = self.ops[op_id].next;
        }
        bounds
    }

    /// Propagate constraint backward from v to its operands (one level, no recursion).
    /// When v is constrained to (`new_lower`, `new_upper`) and v = f(operand, constant),
    /// the operand's upper bound can be narrowed accordingly.
    fn backward_constrain(
        &self,
        v: OpId,
        _new_lower: Dim,
        new_upper: Dim,
        prev: &mut Map<OpId, (Dim, Dim)>,
        skip_rederive: &mut Set<OpId>,
    ) {
        match &self.ops[v].op {
            Op::Binary { x, y, bop: BOp::Mul } => {
                let xc = prev.get(x).filter(|(l, u)| l == u).copied();
                let yc = prev.get(y).filter(|(l, u)| l == u).copied();
                let operand_k = match (xc, yc) {
                    (None, Some((k, _))) => Some((*x, k)),
                    (Some((k, _)), None) => Some((*y, k)),
                    _ => None,
                };
                if let Some((operand, k)) = operand_k {
                    if let Some(upper) = new_upper.checked_div(k) {
                        if let Some(&(ol, ou)) = prev.get(&operand) {
                            if upper < ou {
                                prev.insert(operand, (ol, upper));
                                skip_rederive.insert(operand);
                            }
                        }
                    }
                }
            }
            Op::Binary { x, y, bop: BOp::Add } => {
                let xc = prev.get(x).filter(|(l, u)| l == u).copied();
                let yc = prev.get(y).filter(|(l, u)| l == u).copied();
                let operand_k = match (xc, yc) {
                    (None, Some((k, _))) => Some((*x, k)),
                    (Some((k, _)), None) => Some((*y, k)),
                    _ => None,
                };
                if let Some((operand, k)) = operand_k {
                    if new_upper >= k {
                        let upper = new_upper - k;
                        if let Some(&(ol, ou)) = prev.get(&operand) {
                            if upper < ou {
                                prev.insert(operand, (ol, upper));
                                skip_rederive.insert(operand);
                            }
                        }
                    }
                }
            }
            Op::Cast { x, .. } => {
                if let Some(&(cl, cu)) = prev.get(x) {
                    if new_upper < cu {
                        prev.insert(*x, (cl, new_upper));
                        skip_rederive.insert(*x);
                    }
                }
            }
            _ => {}
        }
    }

    fn rederive_bounds(&self, prev: &mut Map<OpId, (Dim, Dim)>, op_id: OpId) {
        match *self.at(op_id) {
            Op::Cast { x, .. } => {
                if let Some(&b) = prev.get(&x) {
                    prev.insert(op_id, b);
                }
            }
            Op::Binary { x, y, bop } => {
                let Some(&(min_x, max_x)) = prev.get(&x) else { return };
                let Some(&(min_y, max_y)) = prev.get(&y) else { return };
                let range = match bop {
                    BOp::Add => (min_x.wrapping_add(min_y), max_x.wrapping_add(max_y)),
                    BOp::Sub => (min_x.wrapping_sub(max_y), max_x.wrapping_sub(min_y)),
                    BOp::Mul => (min_x.wrapping_mul(min_y), max_x.wrapping_mul(max_y)),
                    BOp::Div | BOp::Mod if min_y == 0 || max_y == 0 => (0, Dim::MAX),
                    BOp::Div => (min_x / min_y, max_x / max_y),
                    BOp::Mod => (0, max_y - 1),
                    BOp::BitShiftLeft => (min_x << min_y, max_x << max_y),
                    BOp::BitShiftRight => (min_x >> min_y, max_x >> max_y),
                    BOp::Pow => {
                        let min_val = if min_y == 0 {
                            1
                        } else if min_x == 0 {
                            0
                        } else {
                            min_x.pow(min_y as u32)
                        };
                        let max_val = if max_y == 0 {
                            1
                        } else if max_x == 0 {
                            0
                        } else {
                            max_x.pow(max_y as u32)
                        };
                        (min_val, max_val)
                    }
                    BOp::Eq => {
                        let always = (min_x == max_x) && (min_y == max_y) && (min_x == min_y);
                        let maybe = !(max_x < min_y || max_y < min_x || always);
                        let lower = u64::from(always);
                        let upper = u64::from(always || maybe);
                        (lower, upper)
                    }
                    BOp::NotEq => {
                        let always = max_x < min_y || max_y < min_x;
                        let maybe = !(always || min_x == max_x && min_y == max_y && min_x == min_y);
                        let lower = u64::from(always);
                        let upper = u64::from(always || maybe);
                        (lower, upper)
                    }
                    BOp::Cmpgt => {
                        let always = min_x > max_y;
                        let never = max_x <= min_y;
                        let maybe = !always && !never;
                        let lower = u64::from(always);
                        let upper = u64::from(always || maybe);
                        (lower, upper)
                    }
                    BOp::Cmplt => {
                        let always = max_x < min_y;
                        let never = max_y <= min_x;
                        let maybe = !always && !never;
                        let lower = u64::from(always);
                        let upper = u64::from(always || maybe);
                        (lower, upper)
                    }
                    BOp::And => {
                        let always = (min_x == 1 && max_x == 1) && (min_y == 1 && max_y == 1);
                        let maybe = (max_x >= 1) && (max_y >= 1);
                        (u64::from(always), u64::from(always || maybe))
                    }
                    BOp::Or => {
                        let always = (min_x == 1 && max_x == 1) || (min_y == 1 && max_y == 1);
                        let maybe = (min_x == 1) || (min_y == 1) || (max_x == 1) || (max_y == 1);
                        (Dim::from(always), Dim::from(always || maybe))
                    }
                    BOp::Max => (min_x.max(min_y), max_x.max(max_y)),
                    BOp::BitAnd => (0, max_x.min(max_y)),
                    BOp::BitOr => (min_x | min_y, max_x | max_y),
                    BOp::BitXor => (0, max_x.max(max_y)),
                };
                prev.insert(op_id, range);
            }
            Op::Mad { x, y, z } => {
                let Some(&(xl, xu)) = prev.get(&x) else { return };
                let Some(&(yl, yu)) = prev.get(&y) else { return };
                let Some(&(zl, zu)) = prev.get(&z) else { return };
                prev.insert(
                    op_id,
                    (xl.wrapping_mul(yl).wrapping_add(zl), xu.wrapping_mul(yu).wrapping_add(zu)),
                );
            }
            _ => {}
        }
    }
}

impl Kernel {
    const fn get_bounds(_op_id: OpId) -> Option<RangeInclusive<Dim>> {
        None
    }
}
