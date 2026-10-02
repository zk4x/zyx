// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use crate::{
    Map,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, UOp},
    scalar::{bf16, f16, f8e4m3, f8e5m2},
};

/// Pattern for matching kernel (and graph) IR shapes.
///
/// `Pat` is immutable: patterns are built once from constructors and
/// operators, then matched by shared reference. No method takes `&mut Pat`.
#[derive(Clone, Debug)]
pub enum Pat {
    /// Matches any op and binds it to `name`. A repeated name asserts
    /// `OpId` equality with the first binding.
    Bind(&'static str),
    /// Matches any compile-time constant (`Op::Const`).
    Const,
    /// Matches a constant with this exact value.
    Value(Constant),
    /// Dtypeless numeric constant (0, 1, -1, ...). Matches any `Op::Const`
    /// with a numerically equal value, regardless of dtype. This gets its
    /// own matcher arm: `as_dim` alone cannot express it (it returns `None`
    /// for negatives and floats).
    Num(i64),
    /// Matches a unary op with this exact `UOp` and an operand matching `x`.
    Unary {
        /// Pattern for the operand.
        x: Box<Pat>,
        /// Required unary op.
        uop: UOp,
    },
    /// Matches a binary op with this exact `BOp` and operands matching
    /// `x`/`y`. Commutative ops also try the swapped order.
    Binary {
        /// Pattern for the left operand.
        x: Box<Pat>,
        /// Pattern for the right operand.
        y: Box<Pat>,
        /// Required binary op.
        bop: BOp,
    },
    /// Matches if any alternative matches. Failed alternatives leave no bindings.
    Any(Box<[Pat]>),
    /// Matches if all alternatives match the same op. Bindings merge;
    /// a name bound twice must agree.
    All(Box<[Pat]>),
}

impl<P: Into<Pat>> std::ops::Add<P> for &Pat {
    type Output = Pat;
    fn add(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self.clone()), y: Box::new(rhs.into()), bop: BOp::Add }
    }
}

impl<P: Into<Pat>> std::ops::Add<P> for Pat {
    type Output = Pat;
    fn add(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self), y: Box::new(rhs.into()), bop: BOp::Add }
    }
}

impl<P: Into<Pat>> std::ops::Div<P> for &Pat {
    type Output = Pat;
    fn div(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self.clone()), y: Box::new(rhs.into()), bop: BOp::Div }
    }
}

impl<P: Into<Pat>> std::ops::Div<P> for Pat {
    type Output = Pat;
    fn div(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self), y: Box::new(rhs.into()), bop: BOp::Div }
    }
}

impl Pat {
    /// Matches anything and binds it to `name`. A repeated name asserts
    /// `OpId` equality with the first binding.
    pub fn bind(name: &'static str) -> Pat {
        Pat::Bind(name)
    }

    /// Negation of this pattern.
    pub fn neg(self) -> Pat {
        Pat::Unary { x: Box::new(self), uop: UOp::Neg }
    }

    /// Exponential of this pattern.
    pub fn exp(self) -> Pat {
        Pat::Unary { x: Box::new(self), uop: UOp::Exp }
    }
}

impl From<&Pat> for Pat {
    fn from(p: &Pat) -> Pat {
        p.clone()
    }
}

impl From<i64> for Pat {
    fn from(v: i64) -> Pat {
        Pat::Num(v)
    }
}

impl From<Constant> for Pat {
    fn from(c: Constant) -> Pat {
        Pat::Value(c)
    }
}

impl Kernel {
    /// Match the IR cone above `root` against `pat`.
    ///
    /// Returns the bound subexpression ids on success. A repeated [`Pat::Bind`]
    /// name asserts `OpId` equality; commutative [`BOp`]s try both operand
    /// orders with full binding backtracking. Only the cone *above* `root`
    /// (definitions, never uses) is inspected — use-counts, layouts and
    /// bounds stay in the calling pass.
    pub fn match_pat(&self, root: OpId, pat: impl Into<Pat>) -> Option<Map<&'static str, OpId>> {
        let pat = pat.into();
        let mut bindings = Map::default();
        if self.match_node(&pat, root, &mut bindings) { Some(bindings) } else { None }
    }

    /// Match one pattern node. Returns `false` with `bindings` untouched.
    fn match_node(&self, pat: &Pat, id: OpId, bindings: &mut Map<&'static str, OpId>) -> bool {
        match pat {
            Pat::Bind(name) => match bindings.get(name) {
                Some(&bound) => bound == id,
                None => {
                    bindings.insert(*name, id);
                    true
                }
            },
            Pat::Const => matches!(self.at(id), Op::Const(_)),
            Pat::Value(v) => matches!(self.at(id), Op::Const(c) if c == v),
            Pat::Num(n) => matches!(self.at(id), Op::Const(c) if const_eq_num(*c, *n)),
            Pat::Unary { x, uop } => match self.at(id) {
                Op::Unary { x: xid, uop: u } => *u == *uop && self.match_node(x, *xid, bindings),
                _ => false,
            },
            Pat::Binary { x, y, bop } => match self.at(id) {
                Op::Binary { x: xid, y: yid, bop: b } if *b == *bop => {
                    let snap = bindings.clone();
                    let ok = (self.match_node(x, *xid, bindings) && self.match_node(y, *yid, bindings))
                        || (bop.is_commutative() && {
                            *bindings = snap.clone();
                            self.match_node(x, *yid, bindings) && self.match_node(y, *xid, bindings)
                        });
                    if !ok {
                        *bindings = snap;
                    }
                    ok
                }
                _ => false,
            },
            Pat::Any(ps) => {
                let snap = bindings.clone();
                for p in ps.iter() {
                    *bindings = snap.clone();
                    if self.match_node(p, id, bindings) {
                        return true;
                    }
                }
                *bindings = snap;
                false
            }
            Pat::All(ps) => {
                let snap = bindings.clone();
                for p in ps.iter() {
                    if !self.match_node(p, id, bindings) {
                        *bindings = snap;
                        return false;
                    }
                }
                true
            }
        }
    }
}

/// Dtypeless numeric equality for [`Pat::Num`]: true when `c` has the same
/// numeric value as `n`, whatever its dtype. Integers and bool compare
/// exactly; floats must be integral (`1.0 == 1`, `1.5` never matches).
fn const_eq_num(c: Constant, n: i64) -> bool {
    match c {
        Constant::U8(v) => i128::from(v) == n as i128,
        Constant::U16(v) => i128::from(v) == n as i128,
        Constant::U32(v) => i128::from(v) == n as i128,
        Constant::U64(v) => u64::from_le_bytes(v) as i128 == n as i128,
        Constant::I8(v) => i128::from(v) == n as i128,
        Constant::I16(v) => i128::from(v) == n as i128,
        Constant::I32(v) => i128::from(v) == n as i128,
        Constant::I64(v) => i64::from_le_bytes(v) as i128 == n as i128,
        Constant::Bool(v) => i64::from(v) == n,
        Constant::F32(v) => float_eq_num(f32::from_le_bytes(v) as f64, n),
        Constant::F64(v) => float_eq_num(f64::from_le_bytes(v), n),
        Constant::BF16(v) => float_eq_num(bf16::from_le_bytes(v).to_f64(), n),
        Constant::F16(v) => float_eq_num(f16::from_le_bytes(v).to_f64(), n),
        Constant::F8E4M3(v) => float_eq_num(f8e4m3::from_le_bytes([v]).to_f64(), n),
        Constant::F8E5M2(v) => float_eq_num(f8e5m2::from_le_bytes([v]).to_f64(), n),
    }
}

/// Exact integral-float comparison: `fract == 0` plus a strict range check
/// so the `as i64` conversion below is exact (no saturation edge at `MAX`).
fn float_eq_num(v: f64, n: i64) -> bool {
    v.fract() == 0.0 && v >= -9223372036854775808.0 && v < 9223372036854775808.0 && v as i64 == n
}
