// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use std::ops::RangeInclusive;

use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, MemScope, Op, OpId, ParamKind, RangeKind},
    shape::Dim,
};

impl Kernel {
    /// Verify the kernel IR.
    ///
    /// Validates that the kernel has correct operation ordering
    /// (no uses before declarations) and proper data type propagation.
    /// This is an internal method used during kernel compilation.
    pub fn verify(&self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("verify");
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
                    | Op::Range { .. }
                    | Op::Loop { .. }
                    | Op::EndLoop
                    | Op::If { .. }
                    | Op::EndIf
                    | Op::Mad { .. }
                    | Op::Index { .. }
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
                    Op::Storage { scope: MemScope::Local, .. }
                    | Op::Storage { scope: MemScope::Circular, .. } => {
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
                Op::Bitcast { x, dtype } => {
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
                Op::Index { vec, .. } => {
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
                Op::Range { axis, kind: scope, .. } => {
                    match scope {
                        RangeKind::Group(len) => {
                            if !gids.insert(axis) {
                                println!("index={op_id} is using {scope} axis={axis} for the second time");
                                self.debug();
                                panic!();
                            }
                            if let Some(d) = self.resolve_const(len).and_then(Constant::as_dim)
                                && d < 0
                            {
                                println!("Group index length resolves to negative constant {d} at op {op_id:?}");
                                self.debug();
                                panic!();
                            }
                        }
                        RangeKind::Local(_) => {
                            if !lids.insert(axis) {
                                println!("index={op_id} is using {scope} axis={axis} for the second time");
                                self.debug();
                                panic!();
                            }
                        }
                        RangeKind::Warp(local_id) => {
                            // A warp is a view over a local range on the same axis: the
                            // local range owns the axis, so only the reference is validated.
                            match self.ops[local_id].op {
                                Op::Range { axis: ref_axis, kind: RangeKind::Local(_) } if ref_axis == axis => {}
                                _ => {
                                    println!("index={op_id} warp references op {local_id}, which is not a local range on axis {axis}");
                                    self.debug();
                                    panic!();
                                }
                            }
                        }
                    }
                    dtypes.insert(op_id, IDX_T);
                }
                Op::Loop { len } => {
                    if let Some(d) = self.resolve_const(len).and_then(Constant::as_dim)
                        && d < 0
                    {
                        println!("Loop length resolves to negative constant {d} at op {op_id:?}");
                        self.debug();
                        panic!();
                    }
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
    /// Compute value-range bounds for every operation in the kernel.
    ///
    /// # Invariant
    ///
    /// `compute_bounds` can **never be precise**. It always returns
    /// **conservative (over-approximating)** bounds: for every op, its true
    /// runtime value is contained in `[lb, ub]`. The single guarantee we MUST
    /// uphold is that bounds are **never too tight** — they must never
    /// *under*-approximate the true range. Being wider than reality is always
    /// safe; being tighter than reality is the only forbidden failure mode,
    /// because it would let a constant fold assume a value the op can never take.
    ///
    /// # Why it is always imprecise
    ///
    /// The imprecision is fundamental and expected, not a bug: variables are
    /// bounded **independently**, which ignores *correlations* between them. For
    /// example, `x` and `y` may always satisfy `x <= y` at runtime, but their
    /// independent ranges are derived separately and will overlap / be wider
    /// than the true joint set of reachable values. The resulting range is
    /// therefore wider than reality — that is correct and intended. Precision
    /// can never be recovered without tracking joint constraints, which this
    /// pass deliberately does not do.
    ///
    /// Because the bounds never under-approximate, they are safe to use for
    /// proving a comparison or boolean op is statically constant (if the
    /// conservative range already forces the comparison to one result, that
    /// result holds for every concrete value). They are **NOT** safe for
    /// replacing an op with a specific non-constant value, only for deciding
    /// constant outcomes.
    #[allow(clippy::match_same_arms)]
    pub(crate) fn compute_bounds(&self) -> Map<OpId, (Dim, Dim)> {
        // Single linear walk, O(number of ops). Bounds are ALWAYS conservative
        // (wide): we never narrow from guard conditions and never narrow across
        // scopes, so a single global map suffices — no scope stack, no cloning,
        // no per-op merge. Each op's bound is derived once from its
        // (already-processed) operands. This is intentionally not precise (see
        // the doc comment above): variables are bounded independently and
        // correlations are ignored, so ranges are wider than reality, which is
        // correct and required.
        let mut bounds: Map<OpId, (Dim, Dim)> = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Const(x) => {
                    if let Some(v) = x.as_dim() {
                        bounds.insert(op_id, (v, v));
                    }
                }
                Op::Storage { .. } => {}
                Op::Loop { .. } | Op::Unary { .. } | Op::Cast { .. } | Op::Binary { .. } | Op::Mad { .. } => {
                    self.rederive_bounds(&mut bounds, op_id);
                }
                Op::If { .. } | Op::EndIf => {}
                Op::Range { kind: scope, .. } => {
                    let len = match scope {
                        RangeKind::Group(len) => self.resolve_const(len).and_then(crate::dtype::Constant::as_dim),
                        RangeKind::Local(len) => Some(i64::from(len)),
                        // A warp's value is the lane id: bounded by the warp size.
                        RangeKind::Warp(_) => Some(i64::from(self.dev_info().warp_size)),
                    };
                    // An unresolved (dynamic) group length is UNKNOWN: no bounds
                    // must be fabricated for it. A huge sentinel here would wrap
                    // around in downstream arithmetic and produce false tight
                    // ranges (-> provably-false guards -> wrong constant folding).
                    if let Some(len) = len {
                        bounds.insert(op_id, (0, len.saturating_sub(1)));
                    }
                }
                Op::Asm { ref ops, .. } => {
                    let mut r = None;
                    for x in ops.iter() {
                        if let Some(&(xl, xu)) = bounds.get(x) {
                            r = Some(match r {
                                Some((l, u)) => (xl.min(l), xu.max(u)),
                                None => (xl, xu),
                            });
                        }
                    }
                    if let Some((xl, xu)) = r {
                        bounds.insert(op_id, (xl, xu));
                    }
                }
                Op::Stack { ref ops } => {
                    let mut r = None;
                    for x in ops.iter() {
                        if let Some(&(xl, xu)) = bounds.get(x) {
                            r = Some(match r {
                                Some((l, u)) => (xl.min(l), xu.max(u)),
                                None => (xl, xu),
                            });
                        }
                    }
                    if let Some((xl, xu)) = r {
                        bounds.insert(op_id, (xl, xu));
                    }
                }
                _ => {}
            }
            op_id = self.ops[op_id].next;
        }
        bounds
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
                    BOp::Mul => {
                        // The true range is the min/max over the four corner
                        // products; the naive (min_x*min_y, max_x*max_y) is only
                        // valid for non-negative operands and under-approximates
                        // (non-conservative) when signs mix.
                        let p1 = min_x.saturating_mul(min_y);
                        let p2 = min_x.saturating_mul(max_y);
                        let p3 = max_x.saturating_mul(min_y);
                        let p4 = max_x.saturating_mul(max_y);
                        (p1.min(p2).min(p3).min(p4), p1.max(p2).max(p3).max(p4))
                    }
                    BOp::Div | BOp::Mod if min_y == 0 || max_y == 0 => (Dim::MIN, Dim::MAX),
                    BOp::Div => {
                        // x / y over the rectangle: min/max of the four corner
                        // quotients (saturating — a divisor near zero would
                        // otherwise fabricate a tiny bound).
                        let q1 = min_x.saturating_div(min_y);
                        let q2 = min_x.saturating_div(max_y);
                        let q3 = max_x.saturating_div(min_y);
                        let q4 = max_x.saturating_div(max_y);
                        (q1.min(q2).min(q3).min(q4), q1.max(q2).max(q3).max(q4))
                    }
                    BOp::Mod => {
                        // zyx integer remainder has the same sign as the
                        // dividend (truncated division), so `|x % y| < |y|` and
                        // the sign follows `x`. When the dividend is known
                        // non-negative the remainder lies in `[0, |y|-1]`, and
                        // when it is known non-positive in `[-(|y|-1), 0]`.
                        // These are sound (conservative) tightenings of the
                        // sign-agnostic `[-(|y|-1), |y|-1]`.
                        let mag = max_y.unsigned_abs().max(min_y.unsigned_abs());
                        if mag == 0 {
                            (Dim::MIN, Dim::MAX)
                        } else {
                            let m = mag as i64 - 1;
                            if min_x >= 0 {
                                (0, m)
                            } else if max_x <= 0 {
                                (-m, 0)
                            } else {
                                (-m, m)
                            }
                        }
                    }
                    BOp::BitShiftLeft => (min_x << min_y.min(63), max_x << max_y.min(63)),
                    BOp::BitShiftRight => (min_x >> min_y.min(63), max_x >> max_y.min(63)),
                    BOp::Pow => {
                        let min_val = if min_y == 0 {
                            1
                        } else if min_x == 0 {
                            0
                        } else {
                            min_x.saturating_pow(min_y.min(u32::MAX as i64) as u32)
                        };
                        let max_val = if max_y == 0 {
                            1
                        } else if max_x == 0 {
                            0
                        } else {
                            max_x.saturating_pow(max_y.min(u32::MAX as i64) as u32)
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
