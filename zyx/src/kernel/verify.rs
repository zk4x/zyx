// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use std::ops::RangeInclusive;

use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, IdxKind, Kernel, MemScope, Op, OpId, ParamKind},
    shape::Dim,
};

impl Kernel {
    /// Verify the kernel IR.
    ///
    /// Validates that the kernel has correct operation ordering
    /// (no uses before declarations) and proper data type propagation.
    /// This is an internal method used during kernel compilation.
    pub fn verify(&self) {
        if !cfg!(debug_assertions) {
            return;
        }

        // Detect the kernel's linearization state from its stores — the rule
        // that always holds: a pre-linearize store writes a whole view and has
        // a NULL index; post-linearize the store carries the actual index op.
        // (Param shapes can NOT be used for detection: `Variable` params have
        // null shapes even pre-linearization.)
        let mut null_index_stores = 0u32;
        let mut indexed_stores = 0u32;
        let mut has_post_linearize_ops = false;
        let mut has_move_or_reduce = false;
        {
            let mut scan = self.head;
            while !scan.is_null() {
                match self.at(scan) {
                    // No compile-time NaN may enter the IR: a folded NaN is
                    // always a bug (invalid folding or invalid input data).
                    Op::Const(c) => {
                        let is_nan = match c {
                            Constant::F32(x) => f32::from_le_bytes(*x).is_nan(),
                            Constant::F64(x) => f64::from_le_bytes(*x).is_nan(),
                            Constant::BF16(x) => u16::from_le_bytes(*x) & 0x7fff > 0x7f80,
                            Constant::F16(x) => {
                                let b = u16::from_le_bytes(*x);
                                b & 0x7c00 == 0x7c00 && b & 0x03ff != 0
                            }
                            _ => false,
                        };
                        if is_nan {
                            self.debug();
                        }
                        debug_assert!(!is_nan, "kernel contains a NaN constant at op {scan:?}");
                    }
                    Op::Store { index, .. } => {
                        if index.is_null() {
                            null_index_stores += 1;
                        } else {
                            indexed_stores += 1;
                        }
                    }
                    Op::Load { .. }
                    | Op::Storage { .. }
                    | Op::Index { .. }
                    | Op::Loop { .. }
                    | Op::EndLoop
                    | Op::If { .. }
                    | Op::EndIf
                    | Op::Mad { .. }
                    | Op::Devectorize { .. }
                    | Op::Barrier
                    | Op::Wmma { .. }
                    | Op::ReduceTile { .. }
                    | Op::MatmulTile { .. }
                    | Op::TransposeTile { .. }
                    | Op::Asm { .. } => has_post_linearize_ops = true,
                    Op::Move { .. } | Op::Reduce { .. } => has_move_or_reduce = true,
                    _ => {}
                }
                scan = self.next_op(scan);
            }
        }
        debug_assert!(null_index_stores + indexed_stores > 0, "kernel must contain at least one store");
        if null_index_stores > 0 && indexed_stores > 0 {
            println!("Invalid mixed kernel: stores with both NULL and actual indices.");
            self.debug();
            panic!();
        }

        // Verify param/storage ordering: global params (RO) → GlobalMut params → local storages → everything else.
        // Only meaningful post-linearization; skipped for pre-linearize DAGs.
        if null_index_stores == 0 {
            debug_assert!(!has_move_or_reduce, "post-linearize kernel must not contain Move/Reduce ops");
            #[derive(PartialEq, Eq)]
            enum Phase {
                GlobalRo,
                GlobalRw,
                LocalRo,
                LocalRw,
                Done,
            }
            let mut phase = Phase::GlobalRo;
            let mut scan = self.head;
            while !scan.is_null() {
                match self.at(scan) {
                    Op::Param { kind, .. } => match kind {
                        ParamKind::Variable | ParamKind::Global => {
                            if phase != Phase::GlobalRo {
                                println!("Global read-only params must come first.");
                                self.debug();
                                panic!();
                            }
                        }
                        ParamKind::GlobalMut => {
                            if phase == Phase::GlobalRo {
                                phase = Phase::GlobalRw;
                            }
                            if phase != Phase::GlobalRw {
                                println!("Global read-write params must come before local storages.");
                                self.debug();
                                panic!();
                            }
                        }
                    },
                    Op::Storage { scope: MemScope::Local, .. } | Op::Storage { scope: MemScope::Circular, .. } => {
                        if phase == Phase::GlobalRo || phase == Phase::GlobalRw || phase == Phase::LocalRo {
                            phase = Phase::LocalRw;
                        }
                        if phase != Phase::LocalRw {
                            println!("Local read-write storages must come after local read-only storages.");
                            self.debug();
                            panic!();
                        }
                    }
                    _ => {
                        if phase != Phase::Done {
                            phase = Phase::Done;
                        }
                    }
                }
                scan = self.next_op(scan);
            }
        } else {
            // Pre-linearize DAG: no lowered memory/control ops may exist.
            debug_assert!(!has_post_linearize_ops, "pre-linearize kernel must not contain Load/Storage/Index/Loop ops");
        }

        let mut stack = Vec::new();
        stack.push(Set::default());
        let check = |op_id, x: OpId, stack: &[Set<OpId>]| {
            if !stack.iter().any(|set| set.contains(&x)) {
                println!("{op_id} {:?} uses {x} -> {:?} before declaration.", self.ops[op_id].op, self.ops[x].op);
                self.debug();
                panic!();
            }
        };

        let mut gids = Set::default();
        let mut lids = Set::default();

        let mut params: Map<OpId, ParamKind> = Map::default();
        let mut storages: Map<OpId, (MemScope, Dim)> = Map::default();

        let mut op_id = self.head;
        let mut prev: OpId;
        let mut dtypes: Map<OpId, DType> = Map::default();
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Store { dst, src: x, index, .. } => {
                    if !params.contains_key(&dst) && !storages.contains_key(&dst) {
                        println!("store={op_id} is trying to store to undefined variable");
                        self.debug();
                        panic!();
                    }
                    check(op_id, dst, &stack);
                    check(op_id, x, &stack);
                    // Pre-linearize stores have a NULL index (whole-view write).
                    if !index.is_null() {
                        debug_assert_eq!(dtypes[&index], IDX_T, "store index must be {IDX_T}");
                        check(op_id, index, &stack);
                    }
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Cast { x, dtype } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtype);
                }
                Op::Reduce { x, .. } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                    if stack.len() > 1 {
                        stack.pop();
                    }
                }
                Op::ReduceTile { x, .. } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::MatmulTile { x, y } => {
                    check(op_id, x, &stack);
                    check(op_id, y, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::TransposeTile { x } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
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
                Op::Asm { ref ops, .. } => {
                    let dtype = dtypes[&ops[0]];
                    for &x in ops.iter() {
                        check(op_id, x, &stack);
                        if dtypes[&x] != dtype {
                            println!("Vectorize dtype mismatch on op={op_id}.");
                            self.debug();
                            panic!();
                        }
                    }
                    dtypes.insert(op_id, dtype);
                }
                Op::Stack { ref ops } => {
                    let dtype = dtypes[&ops[0]];
                    for &x in ops.iter() {
                        check(op_id, x, &stack);
                        if dtypes[&x] != dtype {
                            println!("Vectorize dtype mismatch on op={op_id}.");
                            self.debug();
                            panic!();
                        }
                    }
                    dtypes.insert(op_id, dtype);
                }
                Op::Devectorize { vec, .. } => {
                    let dtype = dtypes[&vec];
                    dtypes.insert(op_id, dtype);
                }
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
                Op::Param { dtype, kind, shape } => {
                    params.insert(op_id, kind);
                    dtypes.insert(op_id, dtype);
                    if shape != OpId::NULL {
                        check(op_id, shape, &stack);
                    }
                }
                Op::Storage { dtype, scope, len } => {
                    storages.insert(op_id, (scope, len));
                    dtypes.insert(op_id, dtype);
                }
                Op::Load { src, index, .. } => {
                    if !params.contains_key(&src) && !storages.contains_key(&src) {
                        println!("load={op_id} is trying to load from undefined variable");
                        self.debug();
                        panic!();
                    }
                    debug_assert_eq!(dtypes[&index], IDX_T);
                    check(op_id, src, &stack);
                    check(op_id, index, &stack);
                    dtypes.insert(op_id, dtypes[&src]);
                }
                Op::Index { axis, kind: scope, .. } => {
                    match scope {
                        IdxKind::Group(_) => {
                            if !gids.insert(axis) {
                                println!("index={op_id} is using {scope} axis={axis} for the second time");
                                self.debug();
                                panic!();
                            }
                        }
                        IdxKind::Local(_) => {
                            if !lids.insert(axis) {
                                println!("index={op_id} is using {scope} axis={axis} for the second time");
                                self.debug();
                                panic!();
                            }
                        }
                        IdxKind::Warp(_) => todo!(),
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
                Op::Barrier => {}
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

    pub(crate) fn check_oob(&self) {
        let mut storages = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Storage { len, .. } => {
                    storages.insert(op_id, len);
                }
                Op::Load { src, index, .. } => {
                    let idx_range = Self::get_bounds(index);
                    if let Some(range) = idx_range
                        && *range.end() >= storages[&src]
                    {
                        self.debug();
                        panic!("OOB detected in op {}: index {:?} exceeds buffer length {:?}", op_id, range, storages[&src]);
                    }
                }
                Op::Store { dst, index, .. } => {
                    let idx_range = Self::get_bounds(index);
                    if let Some(range) = idx_range
                        && *range.start() > storages[&dst] + 1
                    {
                        self.debug();
                        panic!("OOB detected in op {}: index {:?} exceeds buffer length {:?}", op_id, range, storages[&dst]);
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
    pub(crate) fn compute_bounds(&self) -> Map<OpId, (Dim, Dim)> {
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
                Op::Storage { .. } => {}
                Op::Loop { .. } | Op::Unary { .. } | Op::Cast { .. } | Op::Binary { .. } | Op::Mad { .. } => {
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
                                    if let Some((yl, yu)) = prev.get(y)
                                        && yl == yu
                                        && let Some((_xl, _xu)) = prev.get(x)
                                    {
                                        let x_id = *x;
                                        let yl = *yl;
                                        let yu = *yu;
                                        prev.insert(x_id, (yl, yu));
                                        self.backward_constrain(x_id, yl, yu, &mut prev, &mut skip_rederive);
                                    }
                                }
                                BOp::Cmplt => {
                                    if let Some((yl, yu)) = prev.get(y)
                                        && yl == yu
                                        && let Some((xl, _xu)) = prev.get(x)
                                    {
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
                Op::Index { kind: scope, .. } => {
                    let b = bounds_stack.last_mut().unwrap();
                    let len = match scope {
                        IdxKind::Group(len) => self.resolve_const(len).and_then(crate::dtype::Constant::as_dim),
                        IdxKind::Local(len) => Some(i64::from(len)),
                        IdxKind::Warp(len) => Some(i64::from(len)),
                    };
                    // An unresolved (dynamic) group length is UNKNOWN: no
                    // bounds must be fabricated for it. A huge sentinel here
                    // would wrap around in downstream arithmetic and produce
                    // false tight ranges (-> provably-false guards -> wrong
                    // constant folding).
                    if let Some(len) = len {
                        b.insert(op_id, (0, len.saturating_sub(1)));
                    }
                }
                Op::Asm { ref ops, .. } => {
                    let b = bounds_stack.last_mut().unwrap();
                    let mut r = None;
                    for x in ops.iter() {
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
                Op::Stack { ref ops } => {
                    let b = bounds_stack.last_mut().unwrap();
                    let mut r = None;
                    for x in ops.iter() {
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
            if !matches!(*self.at(op_id), Op::EndIf)
                && let Some(scope_bounds) = bounds_stack.last()
            {
                for (&k, &v) in scope_bounds {
                    bounds.insert(k, v);
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
                if let Some((operand, k)) = operand_k
                    && let Some(upper) = new_upper.checked_div(k)
                    && let Some(&(ol, ou)) = prev.get(&operand)
                    && upper < ou
                {
                    prev.insert(operand, (ol, upper));
                    skip_rederive.insert(operand);
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
                if let Some((operand, k)) = operand_k
                    && new_upper >= k
                {
                    let upper = new_upper - k;
                    if let Some(&(ol, ou)) = prev.get(&operand)
                        && upper < ou
                    {
                        prev.insert(operand, (ol, upper));
                        skip_rederive.insert(operand);
                    }
                }
            }
            Op::Cast { x, .. } => {
                if let Some(&(cl, cu)) = prev.get(x)
                    && new_upper < cu
                {
                    prev.insert(*x, (cl, new_upper));
                    skip_rederive.insert(*x);
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
                    // Saturating, never wrapping: an overflow must not
                    // fabricate a small upper bound out of huge ones.
                    BOp::Add => (min_x.saturating_add(min_y), max_x.saturating_add(max_y)),
                    BOp::Sub => (min_x.saturating_sub(max_y), max_x.saturating_sub(min_y)),
                    BOp::Mul => (min_x.saturating_mul(min_y), max_x.saturating_mul(max_y)),
                    BOp::Div | BOp::Mod if min_y == 0 || max_y == 0 => (0, Dim::MAX),
                    BOp::Div => (min_x / min_y, max_x / max_y),
                    BOp::Mod => (0, max_y - 1),
                    BOp::BitShiftLeft => (min_x << min_y.min(63), max_x << max_y.min(63)),
                    BOp::BitShiftRight => (min_x >> min_y.min(63), max_x >> max_y.min(63)),
                    BOp::Pow => {
                        let min_val = if min_y == 0 {
                            1
                        } else if min_x == 0 {
                            0
                        } else {
                            min_x.saturating_pow(min_y.min(u32::MAX  as i64) as u32)
                        };
                        let max_val = if max_y == 0 {
                            1
                        } else if max_x == 0 {
                            0
                        } else {
                            max_x.saturating_pow(max_y.min(u32::MAX  as i64) as u32)
                        };
                        (min_val, max_val)
                    }
                    BOp::Eq => {
                        let always = (min_x == max_x) && (min_y == max_y) && (min_x == min_y);
                        let maybe = !(max_x < min_y || max_y < min_x || always);
                        let lower = Dim::from(always as u8);
                        let upper = Dim::from((always || maybe) as u8);
                        (lower, upper)
                    }
                    BOp::NotEq => {
                        let always = max_x < min_y || max_y < min_x;
                        let maybe = !(always || min_x == max_x && min_y == max_y && min_x == min_y);
                        let lower = Dim::from(always as u8);
                        let upper = Dim::from((always || maybe) as u8);
                        (lower, upper)
                    }
                    BOp::Cmpgt => {
                        let always = min_x > max_y;
                        let never = max_x <= min_y;
                        let maybe = !always && !never;
                        let lower = Dim::from(always as u8);
                        let upper = Dim::from((always || maybe) as u8);
                        (lower, upper)
                    }
                    BOp::Cmpge => {
                        let always = min_x >= max_y;
                        let never = max_x < min_y;
                        let maybe = !always && !never;
                        let lower = Dim::from(always as u8);
                        let upper = Dim::from((always || maybe) as u8);
                        (lower, upper)
                    }
                    BOp::Cmplt => {
                        let always = max_x < min_y;
                        let never = max_y <= min_x;
                        let maybe = !always && !never;
                        let lower = Dim::from(always as u8);
                        let upper = Dim::from((always || maybe) as u8);
                        (lower, upper)
                    }
                    BOp::And => {
                        let always = (min_x == 1 && max_x == 1) && (min_y == 1 && max_y == 1);
                        let maybe = (max_x >= 1) && (max_y >= 1);
                        (Dim::from(always as u8), Dim::from((always || maybe) as u8))
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
            Op::Loop { len } => {
                if let Some(&(_, upper)) = prev.get(&len) {
                    prev.insert(op_id, (0, upper.saturating_sub(1)));
                }
            }
            Op::Mad { x, y, z } => {
                let Some(&(xl, xu)) = prev.get(&x) else { return };
                let Some(&(yl, yu)) = prev.get(&y) else { return };
                let Some(&(zl, zu)) = prev.get(&z) else { return };
                prev.insert(op_id, (xl.saturating_mul(yl).saturating_add(zl), xu.saturating_mul(yu).saturating_add(zu)));
            }
            _ => {}
        }
    }
}

impl Kernel {
    const fn get_bounds(_op_id: OpId) -> Option<RangeInclusive<Dim>> {
        // TODO
        None
    }
}
