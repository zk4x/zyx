// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

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
    kernel::{BOp, Kernel, Op, OpId},
    shape::Dim,
    slab::SlabId,
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

        self.unfuse_mad();
        self.simplify_shl_shr_roundtrips();
        self.simplify_bitwise_identities();

        let bounds = self.compute_bounds();

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);

            if let &Op::Binary { x, y, bop } = self.at(op_id)
                && matches!(bop, BOp::Div | BOp::Mod)
                && let Op::Const(divisor) = self.at(y)
            {
                let dtype = divisor.dtype();
                if let Some(divisor) = divisor.as_dim() {
                    match bop {
                        BOp::Mod => self.simplify_mod(op_id, x, y, dtype, &bounds),
                        BOp::Div => self.simplify_div(op_id, x, divisor, dtype, &bounds),
                        _ => {}
                    }
                }
            }

            op_id = next;
        }

        self.simplify_mod_shift_sequences(&bounds);
        self.simplify_zero_shifts(&bounds);
        self.simplify_demux_roundtrip(&bounds);
        self.dead_code_elimination();
        self.verify();
    }

    /// Fold `x >> k` to constant zero when the upper bound of `x` is below `2^k`.
    fn simplify_zero_shifts(&mut self, bounds: &Map<OpId, (Dim, Dim)>) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let &Op::Binary { x, y, bop: BOp::BitShiftRight } = self.at(op_id)
                && let Op::Const(shift) = self.at(y)
                && let Some(k) = shift.as_dim()
                && k < 64
                && let Some(&(_, xu)) = bounds.get(&x)
                && xu < (1u64 << k)
            {
                let dtype = self.dtype(x);
                self.ops[op_id].op = Op::Const(dtype.zero_constant());
            }
            op_id = next;
        }
    }

    /// Try to recognize an expression as `root << K + constant` where the
    /// expression extracts disjoint bit slices of `root` via div/mod/shr,
    /// shifts each to a new position via mul/shl, and sums them (a round-trip
    /// after merge_nested_loops + constant folding).
    fn simplify_demux_roundtrip(&mut self, bounds: &Map<OpId, (Dim, Dim)>) {
        /// A slice of a variable extracted via div/mod/shr then shifted back.
        #[derive(Clone)]
        struct Slice {
            root: OpId,
            lo: u64,
            width: u64,
            shift: u64,
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
                    if !c.is_power_of_two() {
                        // Not a slice-changing multiply — keep the term as a
                        // non-derived constant so it survives the roundtrip.
                        return (vec![], Some(op_id));
                    }
                    let kk = c.ilog2() as u64;
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
                    if !c.is_power_of_two() {
                        return (vec![], Some(op_id));
                    }
                    let kk = c.ilog2() as u64;
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
                    if !c.is_power_of_two() {
                        return (vec![], Some(op_id));
                    }
                    let width = c.ilog2() as u64;
                    let (mut slices, constant) = collect_slices_inner(k, x);
                    for s in &mut slices {
                        s.width = s.width.min(width);
                    }
                    (slices, constant)
                }
                _ => {
                    if matches!(k.at(op_id), Op::Loop { .. }) {
                        (vec![Slice { root: op_id, lo: 0, width: u64::MAX, shift: 0 }], None)
                    } else {
                        // Not a loop root — treat entire expression as constant
                        (vec![], Some(op_id))
                    }
                }
            }
        }

        fn const_u64(k: &Kernel, op_id: OpId) -> Option<u64> {
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

            let root_width = bounds.get(&root).map_or(64, |&(_, max)| if max == 0 { 1 } else { (max.ilog2() + 1) as u64 });

            // Sort by lo, fill in MAX widths from bounds, verify partition
            slices.sort_by_key(|s| s.lo);
            let mut cursor = 0u64;
            let mut ok = true;
            for s in &slices {
                if s.lo != cursor {
                    ok = false;
                    break;
                }
                let w = if s.width == u64::MAX {
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

    fn simplify_shl_shr_roundtrips(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Some(y) = self.match_shl_shr_roundtrip(op_id) {
                self.remap(op_id, y);
            }
            op_id = next;
        }
        self.dead_code_elimination();
    }

    fn match_shl_shr_roundtrip(&self, op_id: OpId) -> Option<OpId> {
        let Op::Binary { x: add_op, y: shift_amount, bop: BOp::BitShiftRight } = self.at(op_id) else {
            return None;
        };
        let Op::Const(cst) = self.at(*shift_amount) else { return None };
        let n = cst.as_dim()?;
        if n >= 64 {
            return None;
        }
        let Op::Binary { x: add_x, y: add_y, bop: BOp::Add } = self.at(*add_op) else {
            return None;
        };
        for candidate in [add_x, add_y] {
            if let Op::Binary { x: y, y: s, bop: BOp::BitShiftLeft } = self.at(*candidate)
                && let Op::Const(c) = self.at(*s)
                && c.as_dim() == Some(n)
            {
                return Some(*y);
            }
        }
        None
    }

    fn simplify_bitwise_identities(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let Some(replacement) = self.match_bitwise_identity(op_id) {
                self.remap(op_id, replacement);
            }
            op_id = next;
        }
        self.dead_code_elimination();
    }

    fn match_bitwise_identity(&self, op_id: OpId) -> Option<OpId> {
        if let Op::Binary { x, y, bop: BOp::BitAnd } = self.at(op_id) {
            for candidate in [(*x, *y), (*y, *x)] {
                if let Op::Const(c) = self.at(candidate.0)
                    && c.is_max()
                {
                    return Some(candidate.1);
                }
            }
        }
        if let Op::Binary { x, y, bop: BOp::BitOr } = self.at(op_id) {
            for candidate in [(*x, *y), (*y, *x)] {
                if let Op::Const(c) = self.at(candidate.0)
                    && c.as_dim() == Some(0)
                {
                    return Some(candidate.1);
                }
            }
        }
        None
    }

    #[allow(unused)]
    fn const_dim(&self, op_id: OpId) -> Option<Dim> {
        let Op::Const(c) = self.ops[op_id].op else { return None };
        c.as_dim()
    }

    fn simplify_div(&mut self, op_id: OpId, x: OpId, divisor: Dim, dtype: DType, bounds: &Map<OpId, (Dim, Dim)>) {
        if let Some((a, c, _)) = mul_add(self, x)
            && c == divisor
        {
            self.remap(op_id, a);
            return;
        }

        if let Some((a, c, _)) = mad(self, x)
            && c == divisor
        {
            self.remap(op_id, a);
            return;
        }

        let Some(&(_, xu)) = bounds.get(&x) else { return };
        if xu < divisor {
            self.ops[op_id].op = Op::Const(dtype.zero_constant());
        }
    }

    fn simplify_mod(&mut self, op_id: OpId, x: OpId, divisor_const: OpId, _dtype: DType, bounds: &Map<OpId, (Dim, Dim)>) {
        let Op::Const(divisor) = self.ops[divisor_const].op else {
            return;
        };
        let Some(divisor) = divisor.as_dim() else { return };

        //self.debug();

        // Pattern 1: x % divisor when 0 <= x < divisor -> x
        if let Some(&(_, max_x)) = bounds.get(&x)
            && max_x < divisor
        {
            self.remap(op_id, x);
            return;
        }

        if let Some((a, c, b)) = mul_add(self, x) {
            // Pattern 2: (a*c + b) % c -> b % c (because (a*c) % c = 0)
            // Math: (a*c + b) % c = ((a*c) % c + b % c) % c = (0 + b % c) % c = b % c
            // Since c == divisor: result = b % divisor
            if c == divisor {
                self.ops[op_id].op = Op::Binary { x: b, y: divisor_const, bop: BOp::Mod };
                // Pattern 1 on result: if b < divisor, b % divisor = b
                if let Some(&(_, max_b)) = bounds.get(&b)
                    && max_b < divisor
                {
                    self.remap(op_id, b);
                }
                return;
            }
            // Pattern 2b: (a*c + b) % d when c % d == 1 -> (a + b) % d
            // Math: (a*c + b) % d = ((a*(c%d) + b) % d) = ((a*1 + b) % d) = (a + b) % d
            if c % divisor == 1 {
                let a_plus_b = self.insert_before(op_id, Op::Binary { x: a, y: b, bop: BOp::Add });
                self.ops[op_id].op = Op::Binary { x: a_plus_b, y: divisor_const, bop: BOp::Mod };
                // Pattern 1 on result: if max(a) + max(b) < divisor, (a+b) % divisor = a+b
                if let Some(&(_, max_a)) = bounds.get(&a)
                    && let Some(&(_, max_b)) = bounds.get(&b)
                    && max_a.saturating_add(max_b) < divisor
                {
                    self.remap(op_id, a_plus_b);
                }
                return;
            }
            // Pattern 2c: (a*c + b) % d when max(a*c + b) < d -> b % d
            // Need: min_b == 0 AND max(a*c) + max_b < divisor
            if let Some(&(_min_a, max_a)) = bounds.get(&a) {
                let max_a_c = max_a.saturating_mul(c);
                if let Some(&(min_b, max_b)) = bounds.get(&b)
                    && min_b == 0
                    && max_a_c.saturating_add(max_b) < divisor
                {
                    self.ops[op_id].op = Op::Binary { x: b, y: divisor_const, bop: BOp::Mod };
                    // Pattern 1 on result: if b < divisor, b % divisor = b
                    if max_b < divisor {
                        self.remap(op_id, b);
                    }
                    return;
                }
            }
            // Pattern 2d: (a*c + b) % d when d = c*k and max(a*c+b) < d -> b
            // Need: min_b == 0 AND max(a*c) + max_b < divisor
            // When max(a*c + b) < divisor, (a*c + b) % divisor = a*c + b, so if max < divisor -> result = b
            if divisor > c
                && divisor.is_multiple_of(c)
                && let Some(&(_min_a, max_a)) = bounds.get(&a)
                && let Some(&(min_b, max_b)) = bounds.get(&b)
            {
                let max_ac = max_a.saturating_mul(c);
                if min_b == 0 && max_ac.saturating_add(max_b) < divisor {
                    self.remap(op_id, b);
                    return;
                }
            }
        }

        // Pattern 3: (a + b) % divisor when min_a > 0, min_b > 0, max(a+b) < divisor
        // If both are positive and sum < divisor, no wraparound, so result = a + b
        if let Op::Binary { x: a, y: b, bop: BOp::Add } = self.ops[x].op
            && let Some(&(min_a, max_a)) = bounds.get(&a)
            && let Some(&(min_b, max_b)) = bounds.get(&b)
            && min_a > 0
            && min_b > 0
        {
            let sum = max_a.saturating_add(max_b);
            if sum < divisor && sum > 0 {
                self.remap(op_id, x);
                return;
            }
        }

        // Pattern 4: (a * c) % divisor -> reduce c modulo divisor
        // Math: (a * c) % d = (a * (c % d)) % d
        if let Op::Binary { x: a, y: c, bop: BOp::Mul } = self.ops[x].op
            && let Op::Const(y) = self.ops[c].op
            && let Some(c) = y.as_dim()
        {
            let c_reduced = c % divisor;
            if c_reduced != c
                && c_reduced > 0
                && let Some(&(min_a, max_a)) = bounds.get(&a)
                && min_a > 0
            {
                let prod = max_a.saturating_mul(c_reduced);
                if prod < divisor && prod > 0 {
                    self.remap(op_id, x);
                    return;
                }
            }
        }

        // Pattern 5: (a + C) % divisor where C is constant and max(a) + C < divisor
        // If max(a) + C < divisor, no wraparound, so result = a + C
        if let Op::Binary { x: a, y: b, bop: BOp::Add } = self.ops[x].op
            && let Op::Const(y) = self.ops[b].op
            && let Some(y) = y.as_dim()
            && let Some(&(_, max_a)) = bounds.get(&a)
            && max_a + y < divisor
        {
            self.remap(op_id, x);
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
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let &Op::Binary { x, y, bop: BOp::Add } = self.at(op_id) {
                self.simplify_ceil_add(op_id, x, y, bounds);
            }
            op_id = next;
        }

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if let &Op::Binary { x, y, bop } = self.at(op_id) {
                match bop {
                    BOp::Mod => self.simplify_mod_distribute(op_id, x, y, bounds),
                    BOp::BitShiftRight | BOp::Div => self.simplify_shift_distribute(op_id, x, y, bop, bounds),
                    _ => {}
                }
            }
            op_id = next;
        }
    }

    /// `(x >> 1) + (x & 1)` -> `(x + 1) >> 1`.
    ///
    /// Also matches `(x % 2)` as the residue term. Requires non-negative `x`
    /// (unsigned dtype) and `x + 1` not overflowing. Restricted to `k == 1`:
    /// for `k >= 2` the sum `floor(x/2^k) + (x mod 2^k)` is not `ceil(x/2^k)`.
    fn simplify_ceil_add(&mut self, op_id: OpId, x: OpId, y: OpId, bounds: &Map<OpId, (Dim, Dim)>) {
        let dtype = self.dtype(x);
        if !is_unsigned(dtype) {
            return;
        }
        let (shr_op, rem_op) = match (self.at(x), self.at(y)) {
            (
                &Op::Binary { x: sx, y: sy, bop: BOp::BitShiftRight },
                &Op::Binary { x: rx, y: ry, bop: BOp::Mod | BOp::BitAnd },
            ) => ((sx, sy), (rx, ry)),
            (
                &Op::Binary { x: rx, y: ry, bop: BOp::Mod | BOp::BitAnd },
                &Op::Binary { x: sx, y: sy, bop: BOp::BitShiftRight },
            ) => ((sx, sy), (rx, ry)),
            _ => return,
        };
        let ((shr_x, shr_y), (rem_root, rem_y)) = (shr_op, rem_op);
        if shr_x != rem_root {
            return;
        }
        let Op::Const(k) = self.ops[shr_y].op else { return };
        let Some(k) = k.as_dim() else { return };
        if k != 1 || k >= 64 {
            return;
        }
        let modulus = 1u64 << k;
        let Op::Const(residue) = self.ops[rem_y].op else { return };
        let Some(residue) = residue.as_dim() else { return };
        if residue != modulus - 1 && residue != modulus {
            return;
        }
        let Some(&(_, max_root)) = bounds.get(&shr_x) else { return };
        if max_root.saturating_add(modulus - 1) > dtype_max(dtype) {
            return;
        }
        let add_const = self.insert_before(op_id, Op::Const(Constant::from_le_bytes(&(modulus - 1).to_le_bytes(), dtype)));
        let plus = self.insert_before(op_id, Op::Binary { x: shr_x, y: add_const, bop: BOp::Add });
        let k_const = self.insert_before(op_id, Op::Const(Constant::from_le_bytes(&k.to_le_bytes(), dtype)));
        let result = self.insert_before(op_id, Op::Binary { x: plus, y: k_const, bop: BOp::BitShiftRight });
        self.remap(op_id, result);
    }

    /// `(a + c*b) % m` -> `a % m` when `m` is a constant that divides `c`.
    ///
    /// The multiple `c*b` is recognized from `b << k`, `b * const`, or `b + b`.
    /// Guarded against overflow since a wrapping `(a + c*b)` would break the identity.
    fn simplify_mod_distribute(&mut self, op_id: OpId, x: OpId, y: OpId, bounds: &Map<OpId, (Dim, Dim)>) {
        let Op::Const(m) = self.ops[y].op else { return };
        let Some(m) = m.as_dim() else { return };
        if m == 0 {
            return;
        }
        let Op::Binary { x: a, y: mult, bop: BOp::Add } = self.ops[x].op else {
            return;
        };
        for (a_side, mult_side) in [(a, mult), (mult, a)] {
            if !is_unsigned(self.dtype(a_side)) {
                continue;
            }
            let Some((b, c)) = self.match_const_multiple(mult_side) else {
                continue;
            };
            if c == 0 || c % m != 0 {
                continue;
            }
            let (Some(&(_, max_a)), Some(&(_, max_b))) = (bounds.get(&a_side), bounds.get(&b)) else {
                continue;
            };
            if max_a.saturating_add(c.saturating_mul(max_b)) > dtype_max(self.dtype(a_side)) {
                continue;
            }
            self.ops[op_id].op = Op::Binary { x: a_side, y, bop: BOp::Mod };
            if max_a < m {
                self.remap(op_id, a_side);
            }
            return;
        }
    }

    /// `(a + c*b) >> k` -> `(a >> k) + b` and `(a + c*b) / c` -> `(a / c) + b`.
    ///
    /// Requires the multiple constant to match the shift/divisor, non-negative
    /// operands (unsigned dtype), and no overflow of the original sum.
    fn simplify_shift_distribute(&mut self, op_id: OpId, x: OpId, y: OpId, bop: BOp, bounds: &Map<OpId, (Dim, Dim)>) {
        let Op::Const(amount) = self.ops[y].op else { return };
        let Some(amount) = amount.as_dim() else { return };
        let c = match bop {
            BOp::BitShiftRight if amount < 64 => 1u64 << amount,
            BOp::Div if amount > 0 => amount,
            _ => return,
        };
        let Op::Binary { x: a, y: mult, bop: BOp::Add } = self.ops[x].op else {
            return;
        };
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
            if max_a.saturating_add(c.saturating_mul(max_b)) > dtype_max(dtype) {
                continue;
            }
            let amount_const = self.insert_before(op_id, Op::Const(Constant::from_le_bytes(&amount.to_le_bytes(), dtype)));
            let a_op = self.insert_before(op_id, Op::Binary { x: a_side, y: amount_const, bop });
            let result = self.insert_before(op_id, Op::Binary { x: a_op, y: b, bop: BOp::Add });
            self.remap(op_id, result);
            return;
        }
    }

    /// Matches a term that is `c * b` for a compile-time constant `c`, from
    /// `b << k` (c = 2^k), `b * const` (c = const), or `b + b` (c = 2).
    fn match_const_multiple(&self, op_id: OpId) -> Option<(OpId, u64)> {
        if let Op::Binary { x, y, bop: BOp::Add } = self.ops[op_id].op
            && x == y
        {
            return Some((x, 2));
        }
        if let Op::Binary { x, y, bop: BOp::BitShiftLeft } = self.ops[op_id].op
            && let Op::Const(c) = self.ops[y].op
            && let Some(k) = c.as_dim()
            && k < 64
        {
            return Some((x, 1u64 << k));
        }
        if let Op::Binary { x, y, bop: BOp::Mul } = self.ops[op_id].op {
            for (a, b) in [(x, y), (y, x)] {
                if let Op::Const(c) = self.ops[a].op
                    && let Some(v) = c.as_dim()
                {
                    return Some((b, v));
                }
            }
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

fn mul_add(k: &Kernel, x: OpId) -> Option<(OpId, u64, OpId)> {
    if let Some(x) = mad(k, x) {
        return Some(x);
    }
    // Case 1: (a * c) + b  (also (a << c) + b for constant c)
    let Op::Binary { x: mul, y: add, bop: BOp::Add } = k.at(x) else {
        return None;
    };
    if let Some((a, cval)) = match_mul_or_shl(k, *mul) {
        return Some((a, cval, *add));
    }
    // Case 2: b + (a * c)  (also b + (a << c) for constant c)
    let Op::Binary { x: b, y: mul, bop: BOp::Add } = k.at(x) else {
        return None;
    };
    if let Some((a, cval)) = match_mul_or_shl(k, *mul) {
        return Some((a, cval, *b));
    }
    None
}

fn match_mul_or_shl(k: &Kernel, op: OpId) -> Option<(OpId, u64)> {
    if let Op::Binary { x: a, y: c, bop: BOp::Mul } = k.at(op)
        && let Op::Const(cst) = k.at(*c)
        && let Some(cval) = cst.as_dim()
    {
        return Some((*a, cval));
    }
    if let Op::Binary { x: a, y: c, bop: BOp::BitShiftLeft } = k.at(op)
        && let Op::Const(cst) = k.at(*c)
        && let Some(cval) = cst.as_dim()
        && cval < 64
    {
        return Some((*a, 1u64 << cval));
    }
    None
}

fn mad(k: &Kernel, x: OpId) -> Option<(OpId, u64, OpId)> {
    let Op::Mad { x: a, y: c, z: b } = k.at(x) else { return None };
    let Op::Const(cst) = k.at(*c) else { return None };
    let cval = cst.as_dim()?;
    Some((*a, cval, *b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{DeviceId, MemLayout, MemScope, ParamKind};

    /// Build the cumsum-window mask kernel exactly as linearize produces it
    /// for the gather_f32_dtype one-hot reduce: thread index r47 (outer loop)
    /// and reduce index r81 (inner loop) are packed as `r47 + 4*r81`, split
    /// back into (row, col) via >>2 / %4, repacked as `col + 8*row`, then
    /// masked with `% 7 > 2`. Returns the kernel and the mask cmpgt op.
    fn make_mask_kernel() -> (Kernel, OpId) {
        let mut k = Kernel::new(DeviceId::AUTO);

        let r72_shape = k.const_idx(4u32);
        let r65_shape = k.const_idx(4u32);
        let r41_shape = k.const_idx(4u32);
        let r72 = k.param(DType::I32, ParamKind::Global, r72_shape);
        let r65 = k.param(DType::F32, ParamKind::GlobalMut, r65_shape);
        let r41 = k.param(DType::F32, ParamKind::Global, r41_shape);

        let c0 = k.const_idx(0u32);
        let c1 = k.const_idx(1u32);
        let c2 = k.const_idx(2u32);
        let c3 = k.const_idx(3u32);
        let c4 = k.const_idx(4u32);
        let c7 = k.const_idx(7u32);

        let r37 = k.group_index(0, c4);

        // Outer loop r47 (0..4), inner loop r81 (0..4).
        let r47 = k.loop_(c4);
        let r78 = k.storage(DType::I64, MemScope::Register, 1);
        let r77 = k.const_val(0i64);
        k.store(r78, r77, c0, MemLayout::Scalar);
        let r81 = k.loop_(c4);

        let r92 = k.binary(r81, c2, BOp::BitShiftLeft);
        let r93 = k.binary(r47, r92, BOp::Add);
        let r95 = k.binary(r93, c2, BOp::BitShiftRight);
        let r96 = k.binary(r93, c4, BOp::Mod);
        let _r98 = k.binary(r96, c1, BOp::Div);
        let r99 = k.binary(r93, c1, BOp::Mod);
        let r104 = k.binary(r95, c2, BOp::BitShiftLeft);
        let r105 = k.binary(r96, r104, BOp::Add);
        let r106 = k.binary(r99, r105, BOp::Add);
        let r108 = k.binary(r106, c2, BOp::BitShiftRight);
        let r109 = k.binary(r106, c4, BOp::Mod);
        let r113 = k.binary(r108, c3, BOp::BitShiftLeft);
        let r114 = k.binary(r109, r113, BOp::Add);
        let r120 = k.binary(r114, c7, BOp::Mod);
        let r129 = k.binary(r120, c2, BOp::Cmpgt);

        // Keep the mask alive via an accumulate that feeds a store.
        let r131 = k.cast(r129, DType::I64);
        let r85 = k.load(r78, c0, MemLayout::Scalar);
        let r86 = k.binary(r131, r85, BOp::Add);
        k.store(r78, r86, c0, MemLayout::Scalar);
        k.end_loop();

        let r14 = k.load(r78, c0, MemLayout::Scalar);
        let r21 = k.cast(r14, DType::I32);
        let r23 = k.load(r72, r37, MemLayout::Scalar);
        let r25 = k.binary(r23, r21, BOp::Eq);
        let r26 = k.cast(r25, DType::F32);
        let r27 = k.load(r65, r47, MemLayout::Scalar);
        let r32 = k.binary(r26, r27, BOp::Mul);
        k.store(r41, r32, r37, MemLayout::Scalar);
        k.end_loop();

        (k, r129)
    }

    /// Evaluate the mask (r129) for every (r47, r81) pair using the kernel's op
    /// graph. Returns a 4x4 truth table.
    fn eval_mask(k: &Kernel, mask: OpId) -> [[bool; 4]; 4] {
        use std::collections::HashMap;
        let mut table = [[false; 4]; 4];
        for outer in 0..4u64 {
            for inner in 0..4u64 {
                let mut vals: HashMap<usize, u64> = HashMap::new();
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
                                vals.insert(id, *v as u64);
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
                                    BOp::Cmpgt => (a > b) as u64,
                                    BOp::Cmplt => (a < b) as u64,
                                    BOp::Eq => (a == b) as u64,
                                    BOp::BitShiftLeft => a << b,
                                    BOp::BitShiftRight => a >> b,
                                    BOp::And => (a != 0 && b != 0) as u64,
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
