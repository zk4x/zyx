// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Algebraic simplification for kernel optimization.
//!
//! This module provides algebraic simplification techniques for kernels,
//! including:
//!
//! - Div/mod simplification with constant divisors
//! - Bitwise identity simplification
//! - Shift-left/shift-right roundtrip simplification
//! - Pattern matching for common algebraic expressions
//!
//! These optimizations reduce instruction count and improve performance.

use crate::{
    DType, Map,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, Pat, VExpr},
    shape::Dim,
};

impl Kernel {
    /// Apply algebraic simplification to the kernel.
    ///
    /// This method simplifies algebraic expressions in the kernel IR,
    /// including:
    ///
    /// 1. Div/mod simplification with constant divisors
    /// 2. Bitwise identity simplification (e.g., x & 0xFFFF_FFFF = x)
    /// 3. Shift-left/shift-right roundtrip simplification
    /// 4. Dead code elimination and verification
    ///
    /// The simplification uses bounds analysis to determine when
    /// algebraic patterns can be simplified safely.
    pub fn algebraic_simplifications(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("algebraic_simplification");

        // Single fused walk applying all per-op simplifications (shl_shr
        // roundtrip, bitwise identities, div/mod, zero shifts, constant
        // comparisons) at once instead of one full-IR walk per check. Bounds are
        // computed once and are conservative (see `compute_bounds`), so reusing
        // them across these value-preserving checks is sound.
        let bounds = self.compute_bounds();
        self.simplify_fused(&bounds);
        self.simplify_mod_shift_sequences(&bounds);
        self.simplify_mod_div_identity(&bounds);
        self.simplify_demux_roundtrip(&bounds);

        self.dead_code_elimination();
        self.verify();
    }

    /// Single linear walk applying every per-op simplification check. This fuses
    /// what used to be several separate full-IR walks (`simplify_shl_shr_roundtrips`,
    /// `simplify_bitwise_identities`, the div/mod simplification,
    /// `simplify_zero_shifts`, `simplify_constant_comparisons`) into one O(N)
    /// pass. `dead_code_elimination`, `verify`, and the structural
    /// passes `simplify_mod_shift_sequences` / `simplify_demux_roundtrip` remain
    /// separate.
    fn simplify_fused(&mut self, bounds: &Map<OpId, (Dim, Dim)>) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("simplify_fused");
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            // shl_shr roundtrip: ((y << d) + rest) >> d -> y when
            // 0 <= rest < 2^d: the residue shifts out exactly. Both shift
            // amounts share one value name; dim_lt keeps the < 64 range
            // in-pattern; retry covers the Add order.
            if let Some(m) = {
                self.match_pat(
                    op_id,
                    ((Pat::bind('y') << Pat::bind_const('d').dim_lt(64)) + Pat::bind('r')) >> Pat::bind_const('d'),
                )
            } {
                let (y, rest, dv) = (m.op('y'), m.op('r'), m.dim('d'));
                if bounds.get(&rest).is_some_and(|&(lo, hi)| lo >= 0 && hi < (1 << dv)) {
                    self.remap(op_id, y);
                }
            } else if let Some(replacement) = {
                // x & MAX -> x and x | 0 -> x as one alternation: a single
                // match binds x either way; failed alternatives leave nothing.
                let x = Pat::bind('x');
                self.match_pat(op_id, &Pat::any([&x & Pat::const_if(Constant::is_max), &x | 0])).map(|m| m.op('x'))
            } {
                self.remap(op_id, replacement);
            } else if let Some(m) = {
                // ((a*d)+b)/d → a when |b| < |dv|: truncated division's
                // correction term vanishes. Integer operands only (float
                // division is never exact here); dv == 0 cannot fire.
                self.match_pat(op_id, (Pat::bind('a').int() * Pat::bind_const('d') + Pat::bind('b').int()) / Pat::bind_const('d'))
            } {
                let (a, b, dv) = (m.op('a'), m.op('b'), m.dim('d'));
                if bounds.get(&b).is_some_and(|&(lo, hi)| {
                    let ad = dv.unsigned_abs();
                    lo.unsigned_abs() < ad && hi.unsigned_abs() < ad
                }) {
                    self.remap(op_id, a);
                }
            } else if let Some(m) = {
                // ((a<<k)+b)/d → a when d == 2^k and |b| < d. The pow2
                // relation is a first-class comparison; the bounds guard
                // is the same truncated-division correction as above.
                self.match_pat(
                    op_id,
                    Pat::all([
                        (Pat::bind('a').int() << Pat::bind_const('k') + Pat::bind('b').int()) / Pat::bind_const('d'),
                        Pat::ge('k', 0),
                        Pat::lt('k', 64),
                        Pat::eq('d', VExpr::shl(1, 'k')),
                    ]),
                )
            } {
                let (a, b, dv) = (m.op('a'), m.op('b'), m.dim('d'));
                if bounds.get(&b).is_some_and(|&(lo, hi)| {
                    let ad = dv.unsigned_abs();
                    lo.unsigned_abs() < ad && hi.unsigned_abs() < ad
                }) {
                    self.remap(op_id, a);
                }
            } else if let Some(m) = { self.match_pat(op_id, Pat::bind('x').int() / Pat::bind_const('d')) } {
                // x/d → 0 when 0 <= x < d (dv > 0 follows). Integer only:
                // float division has no truncating correction (1.0/2 = 0.5).
                let (x, dv) = (m.op('x'), m.dim('d'));
                if let Some(&(lo, xu)) = bounds.get(&x)
                    && lo >= 0
                    && xu < dv
                {
                    self.ops[op_id].op = Op::Const(self.dtype(m.op('d')).zero_constant());
                }
            } else if let Some(m) = { self.match_pat(op_id, Pat::bind('x') % Pat::bind_const('d')) } {
                // `dv` is the divisor value, `m["d"]` its op. Each acted
                // rule ends with `op_id = next; continue`: skip patterns 3-5,
                // advance the walk. Fall-through leaves the op untouched.
                let (x, d, dv) = (m.op('x'), m.op('d'), m.dim('d'));

                // Pattern 1: x % d when 0 <= x < d -> x.
                if let Some(&(lo_x, max_x)) = bounds.get(&x)
                    && lo_x >= 0
                    && max_x < dv
                {
                    self.remap(op_id, x);
                    op_id = next;
                    continue;
                }
                // Resolve (a*c)+b or (a<<k)+b (any add/mul order) to
                // (a, cval, b): a shape disjunction over two Pat matches
                // sharing one body below. Integer operands only (float
                // remainder is never exact here). Commutativity retry covers
                // the orderings; the shl form computes its 2^k multiple.
                let mul_like = self
                    .match_pat(x, (Pat::bind('a').int() * Pat::bind_const('c')) + Pat::bind('b').int())
                    .map(|ma| (ma.op('a'), ma.dim('c'), ma.op('b')))
                    .or_else(|| {
                        self.match_pat(x, (Pat::bind('a').int() << Pat::bind_const('k')) + Pat::bind('b').int()).and_then(|ma| {
                            let kv = ma.dim('k');
                            (kv >= 0 && kv < 64).then(|| (ma.op('a'), 1i64 << kv, ma.op('b')))
                        })
                    });
                if let Some((a, c, b)) = mul_like {
                    // Pattern 2: (a*c + b) % c -> b % c (because (a*c) % c = 0)
                    // Math: (a*c + b) % c = ((a*c) % c + b % c) % c = (0 + b % c) % c = b % c
                    // Since c == dv: result = b % dv
                    if c == dv {
                        self.ops[op_id].op = Op::Binary { x: b, y: d, bop: BOp::Mod };
                        // Pattern 1 on result: if 0 <= b < dv, b % dv = b
                        if let Some(&(lo_b, max_b)) = bounds.get(&b)
                            && lo_b >= 0
                            && max_b < dv
                        {
                            self.remap(op_id, b);
                        }
                        op_id = next;
                        continue;
                    }
                    // Pattern 2b: (a*c + b) % d when c % d == 1 -> (a + b) % d
                    // Math: (a*c + b) % d = ((a*(c%d) + b) % d) = ((a*1 + b) % d) = (a + b) % d
                    // checked_rem never fires (instead of panicking) when dv == 0.
                    if c.checked_rem(dv) == Some(1) {
                        let a_plus_b = self.insert_before(op_id, Op::Binary { x: a, y: b, bop: BOp::Add });
                        self.ops[op_id].op = Op::Binary { x: a_plus_b, y: d, bop: BOp::Mod };
                        // Pattern 1 on result: if 0 <= a+b < dv, (a+b) % dv = a+b
                        if let Some(&(min_a, max_a)) = bounds.get(&a)
                            && let Some(&(min_b, max_b)) = bounds.get(&b)
                            && min_a.saturating_add(min_b) >= 0
                            && max_a.saturating_add(max_b) < dv
                        {
                            self.remap(op_id, a_plus_b);
                        }
                        op_id = next;
                        continue;
                    }
                    // Patterns 2c/2d (unsound `(a*c+b) % d -> b` family) are
                    // gone: their sound cases are subsumed by Pattern 1,
                    // which matches x = (a*c+b) directly with x's own bounds.
                }
            } else if let Some(m) = {
                // Pattern 3: (a + b) % d when min_a > 0, min_b > 0, max(a+b) < d.
                // Both positive and sum < d: no wraparound, result = a + b.
                self.match_pat(op_id, (Pat::bind('a') + Pat::bind('b')) % Pat::bind_const('d'))
            } {
                let (x, (a, b), dv) = (m.op('x'), (m.op('a'), m.op('b')), m.dim('d'));
                if let Some(&(min_a, max_a)) = bounds.get(&a)
                    && let Some(&(min_b, max_b)) = bounds.get(&b)
                    && min_a > 0
                    && min_b > 0
                {
                    let sum = max_a.saturating_add(max_b);
                    if sum < dv && sum > 0 {
                        self.remap(op_id, x);
                    }
                }
            } else if let Some(m) = {
                // Pattern 4: (a * c) % d -> reduce c modulo d.
                // Math: (a * c) % d = (a * (c % d)) % d.
                self.match_pat(op_id, (Pat::bind('a') * Pat::bind_const('c')) % Pat::bind_const('d'))
            } {
                let (x, (a, c), dv) = (m.op('x'), (m.op('a'), m.dim('c')), m.dim('d'));
                // checked_rem never fires (instead of panicking) when dv == 0.
                if let Some(c_reduced) = c.checked_rem(dv)
                    && c_reduced != c
                    && c_reduced > 0
                    && let Some(&(min_a, max_a)) = bounds.get(&a)
                    && min_a > 0
                {
                    let prod = max_a.saturating_mul(c_reduced);
                    if prod < dv && prod > 0 {
                        self.remap(op_id, x);
                    }
                }
            } else if let Some(m) = {
                // Pattern 5: (a + C) % d where C is const and 0 <= a+C < d.
                // No wraparound, result = a + C. Saturating adds never fire
                // (instead of overflowing) on huge bounds.
                self.match_pat(op_id, (Pat::bind('a') + Pat::bind_const('y')) % Pat::bind_const('d'))
            } {
                let (x, (a, y), dv) = (m.op('x'), (m.op('a'), m.dim('y')), m.dim('d'));
                if let Some(&(min_a, max_a)) = bounds.get(&a)
                    && min_a.saturating_add(y) >= 0
                    && max_a.saturating_add(y) < dv
                {
                    self.remap(op_id, x);
                }
            } else if let Some(m) = {
                // x >> k -> 0 when 0 <= x < 2^k. dim_lt keeps the range in-pattern.
                self.match_pat(op_id, Pat::bind('x') >> Pat::bind_const('k').dim_lt(64))
            } {
                let (x, kv) = (m.op('x'), m.dim('k'));
                if let Some(&(lo, xu)) = bounds.get(&x)
                    && lo >= 0
                    && xu < (1i64 << kv)
                {
                    let dtype = self.dtype(x);
                    self.ops[op_id].op = Op::Const(dtype.zero_constant());
                }
            } else if let Some(m) = {
                // `(a - b) == 0` equals `a == b` for integers, but only if
                // `a - b` cannot overflow (otherwise wrapping breaks the
                // equivalence). Sub is not commutative: no operand swap.
                self.match_pat(op_id, Pat::binary(BOp::Eq, Pat::bind('a').int() - Pat::bind('b').int(), 0))
            } {
                let (a, b) = (m.op('a'), m.op('b'));
                if bounds.get(&a).zip(bounds.get(&b)).is_some_and(|(&(a_lb, a_ub), &(b_lb, b_ub))| {
                    a_ub.saturating_sub(b_lb) != i64::MAX && a_lb.saturating_sub(b_ub) != i64::MIN
                }) {
                    self.ops[op_id].op = Op::Binary { x: a, y: b, bop: BOp::Eq };
                }
            } else if let Some(m) = {
                // `(a - b) != 0` equals `a != b`. Same overflow guard.
                self.match_pat(op_id, Pat::binary(BOp::NotEq, Pat::bind('a').int() - Pat::bind('b').int(), 0))
            } {
                let (a, b) = (m.op('a'), m.op('b'));
                if bounds.get(&a).zip(bounds.get(&b)).is_some_and(|(&(a_lb, a_ub), &(b_lb, b_ub))| {
                    a_ub.saturating_sub(b_lb) != i64::MAX && a_lb.saturating_sub(b_ub) != i64::MIN
                }) {
                    self.ops[op_id].op = Op::Binary { x: a, y: b, bop: BOp::NotEq };
                }
            } else if let Some(m) = {
                // `(a - b) >= 0` equals `a >= b`. Same overflow guard.
                self.match_pat(op_id, Pat::binary(BOp::Cmpge, Pat::bind('a').int() - Pat::bind('b').int(), 0))
            } {
                let (a, b) = (m.op('a'), m.op('b'));
                if bounds.get(&a).zip(bounds.get(&b)).is_some_and(|(&(a_lb, a_ub), &(b_lb, b_ub))| {
                    a_ub.saturating_sub(b_lb) != i64::MAX && a_lb.saturating_sub(b_ub) != i64::MIN
                }) {
                    self.ops[op_id].op = Op::Binary { x: a, y: b, bop: BOp::Cmpge };
                }
            } else if let Some(m) = {
                // `(a - b) > 0` equals `a > b`. Same overflow guard.
                self.match_pat(op_id, Pat::binary(BOp::Cmpgt, Pat::bind('a').int() - Pat::bind('b').int(), 0))
            } {
                let (a, b) = (m.op('a'), m.op('b'));
                if bounds.get(&a).zip(bounds.get(&b)).is_some_and(|(&(a_lb, a_ub), &(b_lb, b_ub))| {
                    a_ub.saturating_sub(b_lb) != i64::MAX && a_lb.saturating_sub(b_ub) != i64::MIN
                }) {
                    self.ops[op_id].op = Op::Binary { x: a, y: b, bop: BOp::Cmpgt };
                }
            } else if let Some(m) = {
                // `(a - b) < 0` equals `a < b`. Same overflow guard.
                self.match_pat(op_id, Pat::binary(BOp::Cmplt, Pat::bind('a').int() - Pat::bind('b').int(), 0))
            } {
                let (a, b) = (m.op('a'), m.op('b'));
                if bounds.get(&a).zip(bounds.get(&b)).is_some_and(|(&(a_lb, a_ub), &(b_lb, b_ub))| {
                    a_ub.saturating_sub(b_lb) != i64::MAX && a_lb.saturating_sub(b_ub) != i64::MIN
                }) {
                    self.ops[op_id].op = Op::Binary { x: a, y: b, bop: BOp::Cmplt };
                }
            } else if let &Op::Binary { x, y, bop } = self.at(op_id)
                && !self.dtype(x).is_float()
            {
                let folded: Option<Constant> = match (self.at(x).clone(), self.at(y).clone()) {
                    (Op::Const(cx), Op::Const(cy)) => Some(Constant::binary(cx, cy, bop)),
                    (Op::Const(cx), _) => bounds
                        .get(&y)
                        .and_then(|&(lb, ub)| cx.as_dim().and_then(|cv| fold_cmp(bop, lb, ub, cv, true).map(Constant::Bool))),
                    (_, Op::Const(cy)) => bounds
                        .get(&x)
                        .and_then(|&(lb, ub)| cy.as_dim().and_then(|cv| fold_cmp(bop, lb, ub, cv, false).map(Constant::Bool))),
                    _ => None,
                };
                if let Some(c) = folded {
                    self.ops[op_id].op = Op::Const(c);
                }
            }
            op_id = next;
        }
    }

    /// Collapse `(a % n) + n * (a / n)` (and its commutation) back to `a`, the
    /// general integer round-trip identity `a = n * (a / n) + (a % n)`. This
    /// fires for non-power-of-two strides where `simplify_demux_roundtrip`
    /// (which requires power-of-two strides) cannot simplify the chain. Guarded
    /// to non-negative `a` and positive `n` (truncated div/mod), using the
    /// conservative bounds.
    fn simplify_mod_div_identity(&mut self, bounds: &Map<OpId, (Dim, Dim)>) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("simplify_mod_div_identity");
        // `(a % n) + (n * (a / n))` (in either order at both levels): the
        // truncated-division round-trip, valid for non-negative `a` and
        // positive `n`. Operand orders are handled by the matcher's
        // commutativity retry; the repeated `a`/`n` binders assert the
        // shared operands (replacing `match_div_a_n`'s `==` checks); the
        // integer binders replace the float-dtype guard.
        let a = Pat::bind('a').int();
        let n = Pat::bind('n').int();
        let pat = (&a % &n) + (&n * (&a / &n));
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Some(m) = self.match_pat(op_id, &pat) {
                let (a, n) = (m.op('a'), m.op('n'));
                // Default to the widest possible (negative) bound so a missing entry in
                // the conservative bounds map does NOT silently satisfy `a >= 0`. `n`
                // defaults to 0, which fails the `n > 0` check, so an unbounded `n` is
                // also refused.
                let a_lb = bounds.get(&a).map_or(Dim::MIN, |&(lb, _)| lb);
                let n_lb = bounds.get(&n).map_or(0, |&(lb, _)| lb);
                if a_lb >= 0 && n_lb > 0 {
                    self.remap(op_id, a);
                }
            }
            op_id = next;
        }
    }

    /// Try to recognize an expression as `root << K + constant` where the
    /// expression extracts disjoint bit slices of `root` via div/mod/shr,
    /// shifts each to a new position via mul/shl, and sums them (a round-trip
    /// after merge_nested_loops + constant folding).
    fn simplify_demux_roundtrip(&mut self, bounds: &Map<OpId, (Dim, Dim)>) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("simplify_demux_roundtrip");
        /// A slice of a variable extracted via div/mod/shr then shifted back.
        #[derive(Clone)]
        struct Slice {
            root: OpId,
            lo: Dim,
            width: Dim,
            shift: Dim,
        }

        /// Returns (slices derived from a loop root, constant expression not derived from root).
        fn collect_slices_inner(k: &mut Kernel, op_id: OpId) -> (Vec<Slice>, Option<OpId>) {
            match *k.at(op_id) {
                Op::Binary { x, y, bop: BOp::Add } => {
                    let (mut ls, lc) = collect_slices_inner(k, x);
                    let (rs, rc) = collect_slices_inner(k, y);
                    // Try to merge slices; if roots differ, non-root side becomes constant
                    let slices = if !ls.is_empty() && !rs.is_empty() && ls[0].root != rs[0].root {
                        // One side's root is not the loop — treat the whole other
                        // operand as an opaque constant term (it may carry its own
                        // scale, e.g. `4*loop`), never just its bare loop root.
                        if matches!(k.at(ls[0].root), Op::Loop { .. }) {
                            return (ls, Some(y));
                        } else {
                            return (rs, Some(x));
                        }
                    } else {
                        if ls.is_empty() {
                            ls = rs;
                        } else if !rs.is_empty() {
                            ls.extend(rs);
                        }
                        ls
                    };
                    // Merge constant terms
                    let constant = match (lc, rc) {
                        (Some(a), Some(b)) => Some(k.insert_before(op_id, Op::Binary { x: a, y: b, bop: BOp::Add })),
                        (Some(a), None) => Some(a),
                        (None, Some(b)) => Some(b),
                        (None, None) => None,
                    };
                    (slices, constant)
                }
                Op::Binary { x, y, bop: BOp::BitShiftLeft } if is_const(k, y) => {
                    let c = match const_u64(k, y) {
                        Some(c) => c,
                        None => return (vec![], None),
                    };
                    let (mut slices, constant) = collect_slices_inner(k, x);
                    for s in &mut slices {
                        s.shift += c;
                    }
                    (slices, constant)
                }
                Op::Binary { x, y, bop: BOp::Mul } if is_const(k, y) => {
                    let c = match const_u64(k, y) {
                        Some(c) => c,
                        None => return (vec![], Some(op_id)),
                    };
                    if !(c > 0 && c & (c - 1) == 0) {
                        // Not a slice-changing multiply — keep the term as a
                        // non-derived constant so it survives the roundtrip.
                        return (vec![], Some(op_id));
                    }
                    let kk = c.ilog2() as i64;
                    let (mut slices, constant) = collect_slices_inner(k, x);
                    for s in &mut slices {
                        s.shift += kk;
                    }
                    (slices, constant)
                }
                Op::Binary { x, y, bop: BOp::Div } if is_const(k, y) => {
                    let c = match const_u64(k, y) {
                        Some(c) => c,
                        None => return (vec![], Some(op_id)),
                    };
                    if !(c > 0 && c & (c - 1) == 0) {
                        return (vec![], Some(op_id));
                    }
                    let kk = c.ilog2() as i64;
                    let (mut slices, constant) = collect_slices_inner(k, x);
                    for s in &mut slices {
                        s.lo += kk;
                    }
                    (slices, constant)
                }
                Op::Binary { x, y, bop: BOp::BitShiftRight } if is_const(k, y) => {
                    let c = match const_u64(k, y) {
                        Some(c) => c,
                        None => return (vec![], None),
                    };
                    let (mut slices, constant) = collect_slices_inner(k, x);
                    for s in &mut slices {
                        s.lo += c;
                    }
                    (slices, constant)
                }
                Op::Binary { x, y, bop: BOp::Mod } if is_const(k, y) => {
                    let c = match const_u64(k, y) {
                        Some(c) => c,
                        None => return (vec![], Some(op_id)),
                    };
                    if !(c > 0 && c & (c - 1) == 0) {
                        return (vec![], Some(op_id));
                    }
                    let width = c.ilog2() as i64;
                    let (mut slices, constant) = collect_slices_inner(k, x);
                    for s in &mut slices {
                        s.width = s.width.min(width);
                    }
                    (slices, constant)
                }
                _ => {
                    if matches!(k.at(op_id), Op::Loop { .. }) {
                        (vec![Slice { root: op_id, lo: 0, width: Dim::MAX, shift: 0 }], None)
                    } else {
                        // Not a loop root — treat entire expression as constant
                        (vec![], Some(op_id))
                    }
                }
            }
        }

        fn const_u64(k: &Kernel, op_id: OpId) -> Option<Dim> {
            match k.at(op_id) {
                Op::Const(c) => c.as_dim(),
                _ => None,
            }
        }
        fn is_const(k: &Kernel, op_id: OpId) -> bool {
            matches!(k.at(op_id), Op::Const(_))
        }

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            let (x, y) = match self.at(op_id) {
                &Op::Binary { x, y, bop: BOp::Add } => (x, y),
                _ => {
                    op_id = next;
                    continue;
                }
            };

            // Skip if either operand is a constant
            if is_const(self, x) || is_const(self, y) {
                op_id = next;
                continue;
            }

            let ((x_slices, x_const), (y_slices, y_const)) = (collect_slices_inner(self, x), collect_slices_inner(self, y));

            let mut slices;
            let constant_term;
            match (x_slices.is_empty(), y_slices.is_empty()) {
                (true, true) => {
                    op_id = next;
                    continue;
                }
                (false, true) => {
                    slices = x_slices;
                    // y has no slices, so it is a constant term itself. If x also
                    // carried a non-derived constant, both must be preserved.
                    constant_term = if let Some(a) = x_const {
                        self.insert_before(op_id, Op::Binary { x: a, y, bop: BOp::Add })
                    } else {
                        y
                    };
                }
                (true, false) => {
                    slices = y_slices;
                    constant_term = if let Some(b) = y_const {
                        self.insert_before(op_id, Op::Binary { x, y: b, bop: BOp::Add })
                    } else {
                        x
                    };
                }
                (false, false) => {
                    if x_slices[0].root == y_slices[0].root {
                        slices = x_slices;
                        slices.extend(y_slices);
                        constant_term = match (x_const, y_const) {
                            (None, None) => OpId::NULL,
                            (Some(a), None) => a,
                            (None, Some(b)) => b,
                            (Some(a), Some(b)) => self.insert_before(op_id, Op::Binary { x: a, y: b, bop: BOp::Add }),
                        };
                    } else {
                        op_id = next;
                        continue;
                    }
                }
            }

            let root = slices[0].root;
            if slices.iter().any(|s| s.root != root) {
                op_id = next;
                continue;
            }

            let k_val = slices[0].shift.wrapping_sub(slices[0].lo);
            if slices.iter().any(|s| s.shift.wrapping_sub(s.lo) != k_val) {
                op_id = next;
                continue;
            }

            let root_width = bounds.get(&root).map_or(64, |&(_, max)| if max == 0 { 1 } else { (max.ilog2() + 1) as i64 });

            // Filter zero-width slices (e.g. x%1 == 0) — they are constant 0, not a partition piece.
            slices.retain(|s| s.width != 0);
            if slices.is_empty() {
                op_id = next;
                continue;
            }
            // Re-derive k_val after filtering (in case the zero-width slice was the first)
            let k_val = slices[0].shift.wrapping_sub(slices[0].lo);
            if slices.iter().any(|s| s.shift.wrapping_sub(s.lo) != k_val) {
                op_id = next;
                continue;
            }

            // Sort by lo, fill in MAX widths from bounds, verify partition
            slices.sort_by_key(|s| s.lo);
            let mut cursor = 0i64;
            let mut ok = true;
            for s in &slices {
                if s.lo != cursor {
                    ok = false;
                    break;
                }
                let w = if s.width == Dim::MAX {
                    root_width.saturating_sub(s.lo)
                } else {
                    s.width
                };
                cursor = cursor.saturating_add(w);
            }
            if !ok || cursor < root_width {
                op_id = next;
                continue;
            }

            // Only simplify true demux/roundtrip patterns (multiple slices).
            // A single slice is just an identity or shift — no roundtrip to collapse.
            if slices.len() < 2 {
                op_id = next;
                continue;
            }

            // Replace with root << k_val + constant
            let shift_const = self.insert_before(op_id, Op::Const(Constant::idx(k_val)));
            let shl = self.insert_before(op_id, Op::Binary { x: root, y: shift_const, bop: BOp::BitShiftLeft });
            if !constant_term.is_null() {
                self.ops[op_id].op = Op::Binary { x: shl, y: constant_term, bop: BOp::Add };
            } else {
                self.remap(op_id, shl);
            }

            op_id = next;
        }
    }

    /// Simplify modulo and shift sequences using valid algebraic identities.
    ///
    /// Two sweeps run in order:
    ///
    /// 1. Ceiling identity (k = 1): `(x >> 1) + (x & 1)` collapses to `(x + 1) >> 1`
    ///    since `floor(x/2) + (x mod 2) = ceil(x/2)`.
    /// 2. Distribution: `(a + c*b) % m` -> `a % m` when `m | c`, and
    ///    `(a + c*b) >> k` / `(a + c*b) / c` -> `(a >> k) + b` / `(a / c) + b`.
    ///
    /// Every rewrite is guarded by conservative bounds so it only fires when
    /// provably valid (no overflow, non-negative operands).
    fn simplify_mod_shift_sequences(&mut self, bounds: &Map<OpId, (Dim, Dim)>) {
        // `(x >> 1) + (x & 1)` -> `(x + 1) >> 1` (also `(x % 2)` residue).
        // Requires non-negative `x` (unsigned dtype) and `x + 1` not
        // overflowing. Restricted to `k == 1`: for `k >= 2` the sum
        // `floor(x/2^k) + (x mod 2^k)` is not `ceil(x/2^k)`. The repeated
        // `r` asserts the shared root structurally; the Add commutativity
        // retry covers both operand orders. `dtype` comes from the root
        // (shifts/residues preserve it; the old code read whichever Add
        // operand came first).
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Some(m) = self.match_pat(
                op_id,
                Pat::any([
                    Pat::all([
                        (Pat::bind('r') >> Pat::bind_const('k')) + (Pat::bind('r') % Pat::bind_const('s')),
                        Pat::eq('k', 1),
                        Pat::any([Pat::eq('s', 1), Pat::eq('s', 2)]),
                    ]),
                    Pat::all([
                        (Pat::bind('r') >> Pat::bind_const('k')) + (Pat::bind('r') & Pat::bind_const('s')),
                        Pat::eq('k', 1),
                        Pat::any([Pat::eq('s', 1), Pat::eq('s', 2)]),
                    ]),
                ]),
            ) {
                let r = m.op('r');
                let dtype = self.dtype(r);
                if is_unsigned(dtype)
                    && let Some(&(_, max_root)) = bounds.get(&r)
                    && max_root.saturating_add(1) <= dtype_max(dtype) as Dim
                {
                    let add_const = self.insert_before(op_id, Op::Const(Constant::from_le_bytes(&1i64.to_le_bytes(), dtype)));
                    let plus = self.insert_before(op_id, Op::Binary { x: r, y: add_const, bop: BOp::Add });
                    let k_const = self.insert_before(op_id, Op::Const(Constant::from_le_bytes(&1i64.to_le_bytes(), dtype)));
                    let result = self.insert_before(op_id, Op::Binary { x: plus, y: k_const, bop: BOp::BitShiftRight });
                    self.remap(op_id, result);
                }
            }
            op_id = next;
        }

        // Distribution: `(a + c*b) % m` -> `a % m` when `m | c`, and
        // `(a + c*b) >> k` / `(a + c*b) / c` -> `(a >> k) + b` / `(a / c) + b`.
        // Inlined single walk, no helper indirection; the multiple `c*b`
        // comes from `match_const_multiple`. All sides are unsigned, so
        // minima are 0 and upper bounds suffice.
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let &Op::Binary { x, y, bop } = self.at(op_id) {
                match bop {
                    BOp::Mod => {
                        // `(a + c*b) % m` -> `a % m` when `m` divides `c`.
                        // Guarded against overflow since a wrapping sum
                        // would break the identity.
                        if let Some(m) = self.match_pat(y, Pat::bind_const('m')).map(|m| m.dim('m'))
                            && m != 0
                            && let Op::Binary { x: a, y: mult, bop: BOp::Add } = self.ops[x].op
                        {
                            for (a_side, mult_side) in [(a, mult), (mult, a)] {
                                if !is_unsigned(self.dtype(a_side)) {
                                    continue;
                                }
                                let Some((b, c)) = self.match_const_multiple(mult_side) else {
                                    continue;
                                };
                                // c <= 0 never occurred under as_dim; the max
                                // reasoning below needs c >= 0 (c == 0 skips
                                // exactly as before).
                                if c <= 0 || c % m != 0 {
                                    continue;
                                }
                                let (Some(&(_, max_a)), Some(&(_, max_b))) = (bounds.get(&a_side), bounds.get(&b)) else {
                                    continue;
                                };
                                if max_a.saturating_add((c as Dim).saturating_mul(max_b)) > dtype_max(self.dtype(a_side)) as Dim {
                                    continue;
                                }
                                self.ops[op_id].op = Op::Binary { x: a_side, y, bop: BOp::Mod };
                                if max_a < m {
                                    self.remap(op_id, a_side);
                                }
                                break;
                            }
                        }
                    }
                    BOp::BitShiftRight | BOp::Div => {
                        // `(a + c*b) >> k` -> `(a >> k) + b` and
                        // `(a + c*b) / c` -> `(a / c) + b`. Requires the
                        // multiple to match the shift/divisor, unsigned
                        // operands, and no overflow of the original sum.
                        if let Some(amount) = self.match_pat(y, Pat::bind_const('a')).map(|m| m.dim('a')) {
                            // amount < 0 never occurred under as_dim; the
                            // shift below would panic on it.
                            let c = match bop {
                                BOp::BitShiftRight if amount >= 0 && amount < 64 => 1i64 << amount,
                                BOp::Div if amount > 0 => amount,
                                _ => continue,
                            };
                            if let Op::Binary { x: a, y: mult, bop: BOp::Add } = self.ops[x].op {
                                for (a_side, mult_side) in [(a, mult), (mult, a)] {
                                    let dtype = self.dtype(a_side);
                                    if !is_unsigned(dtype) {
                                        continue;
                                    }
                                    let Some((b, mult_c)) = self.match_const_multiple(mult_side) else {
                                        continue;
                                    };
                                    if mult_c != c {
                                        continue;
                                    }
                                    let (Some(&(_, max_a)), Some(&(_, max_b))) = (bounds.get(&a_side), bounds.get(&b)) else {
                                        continue;
                                    };
                                    if max_a.saturating_add(c.saturating_mul(max_b)) > dtype_max(dtype) as Dim {
                                        continue;
                                    }
                                    let amount_const = self
                                        .insert_before(op_id, Op::Const(Constant::from_le_bytes(&amount.to_le_bytes(), dtype)));
                                    let a_op = self.insert_before(op_id, Op::Binary { x: a_side, y: amount_const, bop });
                                    let result = self.insert_before(op_id, Op::Binary { x: a_op, y: b, bop: BOp::Add });
                                    self.remap(op_id, result);
                                    break;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            op_id = next;
        }
    }

    /// Matches a term that is `c * b` for a compile-time constant `c`, from
    /// `b + b` (c = 2, same op twice via the repeated binder), `b << k`
    /// (c = 2^k), or `b * const` (c = const, either order via retry).
    fn match_const_multiple(&self, op_id: OpId) -> Option<(OpId, Dim)> {
        if let Some(m) = self.match_pat(op_id, Pat::bind('b') + Pat::bind('b')) {
            return Some((m.op('b'), 2));
        }
        if let Some(m) = self.match_pat(op_id, Pat::bind('b') << Pat::bind_const('k')) {
            let kv = m.dim('k');
            if kv >= 0 && kv < 64 {
                return Some((m.op('b'), 1i64 << kv));
            }
        }
        if let Some(m) = self.match_pat(op_id, Pat::bind('b') * Pat::bind_const('c')) {
            return Some((m.op('b'), m.dim('c')));
        }
        None
    }
}

fn is_unsigned(dtype: DType) -> bool {
    matches!(dtype, DType::U8 | DType::U16 | DType::U32 | DType::U64)
}

fn dtype_max(dtype: DType) -> u64 {
    match dtype {
        DType::U8 => u64::from(u8::MAX),
        DType::U16 => u64::from(u16::MAX),
        DType::U32 => u64::from(u32::MAX),
        DType::U64 => u64::MAX,
        DType::I8 => i8::MAX as u64,
        DType::I16 => i16::MAX as u64,
        DType::I32 => i32::MAX as u64,
        DType::I64 => i64::MAX as u64,
        _ => u64::MAX,
    }
}

/// Statically evaluate a comparison / equality op `x <op> c` (or `c <op> x`)
/// when `x` has a conservative integer bound `(lb, ub)` and `c` is a
/// compile-time constant. Returns `Some(true)` / `Some(false)` when the bound
/// forces a single outcome, `None` when it cannot be decided from the bound
/// alone. `const_is_left` distinguishes `c <op> x` (`true`) from `x <op> c`
/// (`false`). See `compute_bounds` for the meaning of "conservative".
fn fold_cmp(bop: BOp, lb: Dim, ub: Dim, c: Dim, const_is_left: bool) -> Option<bool> {
    match bop {
        BOp::Cmpge => {
            if const_is_left {
                if c >= ub {
                    Some(true)
                } else if c < lb {
                    Some(false)
                } else {
                    None
                }
            } else if lb >= c {
                Some(true)
            } else if ub < c {
                Some(false)
            } else {
                None
            }
        }
        BOp::Cmpgt => {
            if const_is_left {
                if c > ub {
                    Some(true)
                } else if c <= lb {
                    Some(false)
                } else {
                    None
                }
            } else if lb > c {
                Some(true)
            } else if ub <= c {
                Some(false)
            } else {
                None
            }
        }
        BOp::Cmplt => {
            if const_is_left {
                if c < lb {
                    Some(true)
                } else if c >= ub {
                    Some(false)
                } else {
                    None
                }
            } else if ub < c {
                Some(true)
            } else if lb >= c {
                Some(false)
            } else {
                None
            }
        }
        BOp::Eq => {
            if lb == ub {
                Some(lb == c)
            } else if c < lb || c > ub {
                Some(false)
            } else {
                None
            }
        }
        BOp::NotEq => {
            if lb == ub {
                Some(lb != c)
            } else if c < lb || c > ub {
                Some(true)
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Dev, MemScope};

    /// Build the cumsum-window mask kernel exactly as linearize produces it
    /// for the gather_f32_dtype one-hot reduce: thread index r47 (outer loop)
    /// and reduce index r81 (inner loop) are packed as `r47 + 4*r81`, split
    /// back into (row, col) via >>2 / %4, repacked as `col + 8*row`, then
    /// masked with `% 7 > 2`. Returns the kernel and the mask cmpgt op.
    fn make_mask_kernel() -> (Kernel, OpId) {
        let mut k = Kernel::new(Dev::Auto);

        let r72 = k.param(DType::I32);
        let r65 = k.param(DType::F32);
        let r41 = k.param_mut(DType::F32);

        let c0 = k.const_idx(0u32);
        let c1 = k.const_idx(1u32);
        let c2 = k.const_idx(2u32);
        let c3 = k.const_idx(3u32);
        let c4 = k.const_idx(4u32);
        let c7 = k.const_idx(7u32);

        let r37 = k.group_range(0, c4);

        // Outer loop r47 (0..4), inner loop r81 (0..4).
        let mut mask = OpId::NULL;
        k.loop_over(c4, |k, r47| {
            let r78 = k.storage(DType::I64, MemScope::Register, 1);
            let r77 = k.const_val(0i64);
            k.store(r78, r77, c0);
            k.loop_over(c4, |k, r81| {
                let r92 = k.bit_shift_left(r81, c2);
                let r93 = k.add(r47, r92);
                let r95 = k.bit_shift_right(r93, c2);
                let r96 = k.mod_(r93, c4);
                let _r98 = k.div(r96, c1);
                let r99 = k.mod_(r93, c1);
                let r104 = k.bit_shift_left(r95, c2);
                let r105 = k.add(r96, r104);
                let r106 = k.add(r99, r105);
                let r108 = k.bit_shift_right(r106, c2);
                let r109 = k.mod_(r106, c4);
                let r113 = k.bit_shift_left(r108, c3);
                let r114 = k.add(r109, r113);
                let r120 = k.mod_(r114, c7);
                let r129 = k.cmpgt(r120, c2);
                mask = r129;

                // Keep the mask alive via an accumulate that feeds a store.
                let r131 = k.cast(r129, DType::I64);
                let r85 = k.load(r78, c0);
                let r86 = k.add(r131, r85);
                k.store(r78, r86, c0);
            });

            let r14 = k.load(r78, c0);
            let r21 = k.cast(r14, DType::I32);
            let r23 = k.load(r72, r37);
            let r25 = k.equal(r23, r21);
            let r26 = k.cast(r25, DType::F32);
            let r27 = k.load(r65, r47);
            let r32 = k.mul(r26, r27);
            k.store(r41, r32, r37);
        });

        (k, mask)
    }

    /// Evaluate the mask (r129) for every (r47, r81) pair using the kernel's op
    /// graph. Returns a 4x4 truth table.
    fn eval_mask(k: &Kernel, mask: OpId) -> [[bool; 4]; 4] {
        use std::collections::HashMap;
        let mut table = [[false; 4]; 4];
        for outer in 0i64..4i64 {
            for inner in 0i64..4i64 {
                let mut vals: HashMap<usize, Dim> = HashMap::new();
                let mut op_id = k.head;
                let mut loop_idx = 0usize;
                while !op_id.is_null() {
                    let next = k.next_op(op_id);
                    let id = op_id.0 as usize;
                    match k.at(op_id) {
                        Op::Loop { .. } => {
                            vals.insert(id, if loop_idx == 0 { outer } else { inner });
                            loop_idx += 1;
                        }
                        Op::Const(c) => {
                            if let Some(v) = c.as_dim() {
                                vals.insert(id, v);
                            } else if let crate::dtype::Constant::Bool(v) = c {
                                vals.insert(id, *v as i64);
                            }
                        }
                        Op::Binary { x, y, bop } => {
                            if let (Some(&a), Some(&b)) = (vals.get(&(x.0 as usize)), vals.get(&(y.0 as usize))) {
                                let v = match bop {
                                    BOp::Add => a.wrapping_add(b),
                                    BOp::Sub => a.wrapping_sub(b),
                                    BOp::Mul => a.wrapping_mul(b),
                                    BOp::Div => a.wrapping_div(b),
                                    BOp::Mod => a.wrapping_rem(b),
                                    BOp::Cmpgt => (a > b) as i64,
                                    BOp::Cmplt => (a < b) as i64,
                                    BOp::Eq => (a == b) as i64,
                                    BOp::BitShiftLeft => a << b,
                                    BOp::BitShiftRight => a >> b,
                                    BOp::And => (a != 0 && b != 0) as i64,
                                    _ => continue,
                                };
                                vals.insert(id, v);
                            }
                        }
                        Op::Unary { x, uop } => {
                            if let Some(&a) = vals.get(&(x.0 as usize)) {
                                vals.insert(
                                    id,
                                    match uop {
                                        crate::kernel::UOp::BitNot => !a,
                                        crate::kernel::UOp::Not => (a == 0) as i64,
                                        _ => a,
                                    },
                                );
                            }
                        }
                        _ => {}
                    }
                    op_id = next;
                }
                table[outer as usize][inner as usize] = vals.get(&(mask.0 as usize)).copied().unwrap_or(0) != 0;
            }
        }
        table
    }

    fn expected_mask() -> [[bool; 4]; 4] {
        // mask = (r47 + 8*r81) % 7 > 2 == (r47 + r81) % 7 > 2 == (r47 + r81) > 2
        // since r47, r81 in 0..4 and r47+r81 <= 6 < 7.
        let mut t = [[false; 4]; 4];
        for (i, row) in t.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = i + j > 2;
            }
        }
        t
    }

    #[test]
    fn mask_survives_algebraic_simplification() {
        let (mut k, mask) = make_mask_kernel();
        let before = eval_mask(&k, mask);
        assert_eq!(before, expected_mask(), "mask must be correct before simplification");

        k.move_constants_to_beginning();
        k.algebraic_simplifications();

        let after = eval_mask(&k, mask);
        assert_eq!(after, expected_mask(), "mask must stay correct after algebraic_simplification");
    }
}
