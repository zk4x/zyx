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
    pub fn algebraic_simplification(&mut self) {
        #[cfg(feature = "time")]
        let _timer = crate::Timer::new("algebraic_simplification");

        self.unfuse_mad();
        self.simplify_shl_shr_roundtrips();
        self.simplify_bitwise_identities();

        let bounds = self.compute_bounds();

        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);

            if let &Op::Binary { x, y, bop } = self.at(op_id) {
                if matches!(bop, BOp::Div | BOp::Mod) {
                    if let Op::Const(divisor) = self.at(y) {
                        let dtype = divisor.dtype();
                        if let Some(divisor) = divisor.as_dim() {
                            match bop {
                                BOp::Mod => self.simplify_mod(op_id, x, y, dtype, &bounds),
                                BOp::Div => self.simplify_div(op_id, x, divisor, dtype, &bounds),
                                _ => {}
                            }
                        }
                    }
                }
            }

            op_id = next;
        }

        self.simplify_demux_roundtrip(&bounds);
        self.dead_code_elimination();
        self.verify();
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
            match k.at(op_id) {
                &Op::Binary { x, y, bop: BOp::Add } => {
                    let (mut ls, lc) = collect_slices_inner(k, x);
                    let (rs, rc) = collect_slices_inner(k, y);
                    // Try to merge slices; if roots differ, non-root side becomes constant
                    let slices = if !ls.is_empty() && !rs.is_empty() && ls[0].root != rs[0].root {
                        // One side's root is not the loop — treat as constant
                        if matches!(k.at(ls[0].root), Op::Loop { .. }) {
                            // ls is from the loop, rs is not
                            return (ls, Some(rs[0].root));
                        } else {
                            let const_term = ls[0].root;
                            return (rs, Some(const_term));
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
                        (Some(a), Some(b)) => Some(k.insert_before(
                            op_id,
                            Op::Binary {
                                x: a,
                                y: b,
                                bop: BOp::Add,
                            },
                        )),
                        (Some(a), None) => Some(a),
                        (None, Some(b)) => Some(b),
                        (None, None) => None,
                    };
                    (slices, constant)
                }
                &Op::Binary {
                    x,
                    y,
                    bop: BOp::BitShiftLeft,
                } if is_const(k, y) => {
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
                &Op::Binary { x, y, bop: BOp::Mul } if is_const(k, y) => {
                    let c = match const_u64(k, y) {
                        Some(c) => c,
                        None => return (vec![], None),
                    };
                    if !c.is_power_of_two() {
                        return (vec![], None);
                    }
                    let kk = c.ilog2() as u64;
                    let (mut slices, constant) = collect_slices_inner(k, x);
                    for s in &mut slices {
                        s.shift += kk;
                    }
                    (slices, constant)
                }
                &Op::Binary { x, y, bop: BOp::Div } if is_const(k, y) => {
                    let c = match const_u64(k, y) {
                        Some(c) => c,
                        None => return (vec![], None),
                    };
                    if !c.is_power_of_two() {
                        return (vec![], None);
                    }
                    let kk = c.ilog2() as u64;
                    let (mut slices, constant) = collect_slices_inner(k, x);
                    for s in &mut slices {
                        s.lo += kk;
                    }
                    (slices, constant)
                }
                &Op::Binary {
                    x,
                    y,
                    bop: BOp::BitShiftRight,
                } if is_const(k, y) => {
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
                &Op::Binary { x, y, bop: BOp::Mod } if is_const(k, y) => {
                    let c = match const_u64(k, y) {
                        Some(c) => c,
                        None => return (vec![], None),
                    };
                    if !c.is_power_of_two() {
                        return (vec![], None);
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
                        (
                            vec![Slice {
                                root: op_id,
                                lo: 0,
                                width: u64::MAX,
                                shift: 0,
                            }],
                            None,
                        )
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
                    constant_term = y_const.unwrap_or(y);
                }
                (true, false) => {
                    slices = y_slices;
                    constant_term = x_const.unwrap_or(x);
                }
                (false, false) => {
                    if x_slices[0].root == y_slices[0].root {
                        slices = x_slices;
                        slices.extend(y_slices);
                        constant_term = x_const.or(y_const).unwrap_or(OpId::NULL);
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

            let root_width = bounds
                .get(&root)
                .map_or(64, |&(_, max)| if max == 0 { 1 } else { (max.ilog2() + 1) as u64 });

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
            let shl = self.insert_before(
                op_id,
                Op::Binary {
                    x: root,
                    y: shift_const,
                    bop: BOp::BitShiftLeft,
                },
            );
            if !constant_term.is_null() {
                self.ops[op_id].op = Op::Binary {
                    x: shl,
                    y: constant_term,
                    bop: BOp::Add,
                };
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
        let Op::Binary {
            x: add_op,
            y: shift_amount,
            bop: BOp::BitShiftRight,
        } = self.at(op_id)
        else {
            return None;
        };
        let Op::Const(cst) = self.at(*shift_amount) else { return None };
        let n = cst.as_dim()?;
        if n >= 64 {
            return None;
        }
        let Op::Binary {
            x: add_x,
            y: add_y,
            bop: BOp::Add,
        } = self.at(*add_op)
        else {
            return None;
        };
        for candidate in [add_x, add_y] {
            if let Op::Binary {
                x: y,
                y: s,
                bop: BOp::BitShiftLeft,
            } = self.at(*candidate)
            {
                if let Op::Const(c) = self.at(*s) {
                    if c.as_dim() == Some(n) {
                        return Some(*y);
                    }
                }
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
                if let Op::Const(c) = self.at(candidate.0) {
                    if c.is_max() {
                        return Some(candidate.1);
                    }
                }
            }
        }
        if let Op::Binary { x, y, bop: BOp::BitOr } = self.at(op_id) {
            for candidate in [(*x, *y), (*y, *x)] {
                if let Op::Const(c) = self.at(candidate.0) {
                    if c.as_dim() == Some(0) {
                        return Some(candidate.1);
                    }
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
        if let Some((a, c, _)) = mul_add(self, x) {
            if c == divisor {
                self.remap(op_id, a);
                return;
            }
        }

        if let Some((a, c, _)) = mad(self, x) {
            if c == divisor {
                self.remap(op_id, a);
                return;
            }
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
        if let Some(&(_, max_x)) = bounds.get(&x) {
            if max_x < divisor {
                self.remap(op_id, x);
                return;
            }
        }

        if let Some((a, c, b)) = mul_add(self, x) {
            // Pattern 2: (a*c + b) % c -> b % c (because (a*c) % c = 0)
            // Math: (a*c + b) % c = ((a*c) % c + b % c) % c = (0 + b % c) % c = b % c
            // Since c == divisor: result = b % divisor
            if c == divisor {
                self.ops[op_id].op = Op::Binary {
                    x: b,
                    y: divisor_const,
                    bop: BOp::Mod,
                };
                // Pattern 1 on result: if b < divisor, b % divisor = b
                if let Some(&(_, max_b)) = bounds.get(&b) {
                    if max_b < divisor {
                        self.remap(op_id, b);
                    }
                }
                return;
            }
            // Pattern 2b: (a*c + b) % d when c % d == 1 -> (a + b) % d
            // Math: (a*c + b) % d = ((a*(c%d) + b) % d) = ((a*1 + b) % d) = (a + b) % d
            if c % divisor == 1 {
                let a_plus_b = self.insert_before(
                    op_id,
                    Op::Binary {
                        x: a,
                        y: b,
                        bop: BOp::Add,
                    },
                );
                self.ops[op_id].op = Op::Binary {
                    x: a_plus_b,
                    y: divisor_const,
                    bop: BOp::Mod,
                };
                // Pattern 1 on result: if max(a) + max(b) < divisor, (a+b) % divisor = a+b
                if let Some(&(_, max_a)) = bounds.get(&a)
                    && let Some(&(_, max_b)) = bounds.get(&b)
                {
                    if max_a.saturating_add(max_b) < divisor {
                        self.remap(op_id, a_plus_b);
                    }
                }
                return;
            }
            // Pattern 2c: (a*c + b) % d when max(a*c + b) < d -> b % d
            // Need: min_b == 0 AND max(a*c) + max_b < divisor
            if let Some(&(_min_a, max_a)) = bounds.get(&a) {
                let max_a_c = max_a.saturating_mul(c);
                if let Some(&(min_b, max_b)) = bounds.get(&b) {
                    if min_b == 0 && max_a_c.saturating_add(max_b) < divisor {
                        self.ops[op_id].op = Op::Binary {
                            x: b,
                            y: divisor_const,
                            bop: BOp::Mod,
                        };
                        // Pattern 1 on result: if b < divisor, b % divisor = b
                        if max_b < divisor {
                            self.remap(op_id, b);
                        }
                        return;
                    }
                }
            }
            // Pattern 2d: (a*c + b) % d when d = c*k and max(a*c+b) < d -> b
            // Need: min_b == 0 AND max(a*c) + max_b < divisor
            // When max(a*c + b) < divisor, (a*c + b) % divisor = a*c + b, so if max < divisor -> result = b
            if divisor > c && divisor.is_multiple_of(c) {
                if let Some(&(_min_a, max_a)) = bounds.get(&a)
                    && let Some(&(min_b, max_b)) = bounds.get(&b)
                {
                    let max_ac = max_a.saturating_mul(c);
                    if min_b == 0 && max_ac.saturating_add(max_b) < divisor {
                        self.remap(op_id, b);
                        return;
                    }
                }
            }
        }

        // Pattern 3: (a + b) % divisor when min_a > 0, min_b > 0, max(a+b) < divisor
        // If both are positive and sum < divisor, no wraparound, so result = a + b
        if let Op::Binary {
            x: a,
            y: b,
            bop: BOp::Add,
        } = self.ops[x].op
        {
            if let Some(&(min_a, max_a)) = bounds.get(&a) {
                if let Some(&(min_b, max_b)) = bounds.get(&b) {
                    if min_a > 0 && min_b > 0 {
                        let sum = max_a.saturating_add(max_b);
                        if sum < divisor && sum > 0 {
                            self.remap(op_id, x);
                            return;
                        }
                    }
                }
            }
        }

        // Pattern 4: (a * c) % divisor -> reduce c modulo divisor
        // Math: (a * c) % d = (a * (c % d)) % d
        if let Op::Binary {
            x: a,
            y: c,
            bop: BOp::Mul,
        } = self.ops[x].op
        {
            if let Op::Const(y) = self.ops[c].op {
                if let Some(c) = y.as_dim() {
                    let c_reduced = c % divisor;
                    if c_reduced != c && c_reduced > 0 {
                        if let Some(&(min_a, max_a)) = bounds.get(&a) {
                            if min_a > 0 {
                                let prod = max_a.saturating_mul(c_reduced);
                                if prod < divisor && prod > 0 {
                                    self.remap(op_id, x);
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Pattern 5: (a + C) % divisor where C is constant and max(a) + C < divisor
        // If max(a) + C < divisor, no wraparound, so result = a + C
        if let Op::Binary {
            x: a,
            y: b,
            bop: BOp::Add,
        } = self.ops[x].op
        {
            if let Op::Const(y) = self.ops[b].op {
                if let Some(y) = y.as_dim() {
                    if let Some(&(_, max_a)) = bounds.get(&a) {
                        if max_a + y < divisor {
                            self.remap(op_id, x);
                            return;
                        }
                    }
                }
            }
        }
    }
}

fn mul_add(k: &Kernel, x: OpId) -> Option<(OpId, u64, OpId)> {
    if let Some(x) = mad(k, x) {
        return Some(x);
    }
    // Case 1: (a * c) + b  (also (a << c) + b for constant c)
    let Op::Binary {
        x: mul,
        y: add,
        bop: BOp::Add,
    } = k.at(x)
    else {
        return None;
    };
    if let Some((a, cval)) = match_mul_or_shl(k, *mul) {
        return Some((a, cval, *add));
    }
    // Case 2: b + (a * c)  (also b + (a << c) for constant c)
    let Op::Binary {
        x: b,
        y: mul,
        bop: BOp::Add,
    } = k.at(x)
    else {
        return None;
    };
    if let Some((a, cval)) = match_mul_or_shl(k, *mul) {
        return Some((a, cval, *b));
    }
    None
}

fn match_mul_or_shl(k: &Kernel, op: OpId) -> Option<(OpId, u64)> {
    if let Op::Binary {
        x: a,
        y: c,
        bop: BOp::Mul,
    } = k.at(op)
    {
        if let Op::Const(cst) = k.at(*c) {
            if let Some(cval) = cst.as_dim() {
                return Some((*a, cval));
            }
        }
    }
    if let Op::Binary {
        x: a,
        y: c,
        bop: BOp::BitShiftLeft,
    } = k.at(op)
    {
        if let Op::Const(cst) = k.at(*c) {
            if let Some(cval) = cst.as_dim() {
                if cval < 64 {
                    return Some((*a, 1u64 << cval));
                }
            }
        }
    }
    None
}

fn mad(k: &Kernel, x: OpId) -> Option<(OpId, u64, OpId)> {
    let Op::Mad { x: a, y: c, z: b } = k.at(x) else { return None };
    let Op::Const(cst) = k.at(*c) else { return None };
    let cval = cst.as_dim()?;
    Some((*a, cval, *b))
}
