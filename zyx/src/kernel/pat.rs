// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use crate::{
    DType, Map,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, UOp},
    scalar::{bf16, f16, f8e4m3, f8e5m2},
};

/// Dtype class constraining a [`Pat::Bind`] binder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtypeClass {
    /// Any dtype.
    Any,
    /// Floating-point dtypes (`is_float`).
    Float,
    /// Integer dtypes (`is_int`: signed and unsigned, but not bool).
    Int,
    /// Unsigned integer dtypes (`is_uint`).
    Unsigned,
}

impl DtypeClass {
    fn matches(self, dtype: DType) -> bool {
        match self {
            DtypeClass::Any => true,
            DtypeClass::Float => dtype.is_float(),
            DtypeClass::Int => dtype.is_int(),
            DtypeClass::Unsigned => dtype.is_uint(),
        }
    }
}

/// Pattern for matching kernel (and graph) IR shapes.
///
/// `Pat` is immutable: patterns are built once from constructors and
/// operators, then matched by shared reference. No method takes `&mut Pat`.
#[derive(Clone, Debug)]
pub enum Pat {    /// Matches anything and binds it to `name`. A repeated name asserts
    /// `OpId` equality with the first binding.
    Bind {
        /// Binder name.
        name: &'static str,
        /// Required dtype class of the matched op.
        class: DtypeClass,
    },
    /// Matches any compile-time constant (`Op::Const`).
    Const,
    /// Matches a constant with this exact value.
    Value(Constant),
    /// Matches a constant satisfying this predicate (e.g. `Constant::is_max`).
    /// Value checks live in-pattern so commutativity retry engages on them;
    /// a failing predicate is a structural mismatch like any other.
    ConstIf(fn(Constant) -> bool),
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

impl<P: Into<Pat>> std::ops::Sub<P> for &Pat {
    type Output = Pat;
    fn sub(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self.clone()), y: Box::new(rhs.into()), bop: BOp::Sub }
    }
}

impl<P: Into<Pat>> std::ops::Sub<P> for Pat {
    type Output = Pat;
    fn sub(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self), y: Box::new(rhs.into()), bop: BOp::Sub }
    }
}

impl<P: Into<Pat>> std::ops::Mul<P> for &Pat {
    type Output = Pat;
    fn mul(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self.clone()), y: Box::new(rhs.into()), bop: BOp::Mul }
    }
}

impl<P: Into<Pat>> std::ops::Mul<P> for Pat {
    type Output = Pat;
    fn mul(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self), y: Box::new(rhs.into()), bop: BOp::Mul }
    }
}

impl<P: Into<Pat>> std::ops::Rem<P> for &Pat {
    type Output = Pat;
    fn rem(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self.clone()), y: Box::new(rhs.into()), bop: BOp::Mod }
    }
}

impl<P: Into<Pat>> std::ops::Rem<P> for Pat {
    type Output = Pat;
    fn rem(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self), y: Box::new(rhs.into()), bop: BOp::Mod }
    }
}

impl<P: Into<Pat>> std::ops::BitAnd<P> for &Pat {
    type Output = Pat;
    fn bitand(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self.clone()), y: Box::new(rhs.into()), bop: BOp::BitAnd }
    }
}

impl<P: Into<Pat>> std::ops::BitAnd<P> for Pat {
    type Output = Pat;
    fn bitand(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self), y: Box::new(rhs.into()), bop: BOp::BitAnd }
    }
}

impl<P: Into<Pat>> std::ops::BitOr<P> for &Pat {
    type Output = Pat;
    fn bitor(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self.clone()), y: Box::new(rhs.into()), bop: BOp::BitOr }
    }
}

impl<P: Into<Pat>> std::ops::BitOr<P> for Pat {
    type Output = Pat;
    fn bitor(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self), y: Box::new(rhs.into()), bop: BOp::BitOr }
    }
}

impl<P: Into<Pat>> std::ops::BitXor<P> for &Pat {
    type Output = Pat;
    fn bitxor(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self.clone()), y: Box::new(rhs.into()), bop: BOp::BitXor }
    }
}

impl<P: Into<Pat>> std::ops::BitXor<P> for Pat {
    type Output = Pat;
    fn bitxor(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self), y: Box::new(rhs.into()), bop: BOp::BitXor }
    }
}

impl<P: Into<Pat>> std::ops::Shl<P> for &Pat {
    type Output = Pat;
    fn shl(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self.clone()), y: Box::new(rhs.into()), bop: BOp::BitShiftLeft }
    }
}

impl<P: Into<Pat>> std::ops::Shl<P> for Pat {
    type Output = Pat;
    fn shl(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self), y: Box::new(rhs.into()), bop: BOp::BitShiftLeft }
    }
}

impl<P: Into<Pat>> std::ops::Shr<P> for &Pat {
    type Output = Pat;
    fn shr(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self.clone()), y: Box::new(rhs.into()), bop: BOp::BitShiftRight }
    }
}

impl<P: Into<Pat>> std::ops::Shr<P> for Pat {
    type Output = Pat;
    fn shr(self, rhs: P) -> Pat {
        Pat::Binary { x: Box::new(self), y: Box::new(rhs.into()), bop: BOp::BitShiftRight }
    }
}

impl std::ops::Neg for &Pat {
    type Output = Pat;
    fn neg(self) -> Pat {
        Pat::Unary { x: Box::new(self.clone()), uop: UOp::Neg }
    }
}

impl std::ops::Neg for Pat {
    type Output = Pat;
    fn neg(self) -> Pat {
        Pat::Unary { x: Box::new(self), uop: UOp::Neg }
    }
}

impl std::ops::Not for &Pat {
    type Output = Pat;
    fn not(self) -> Pat {
        Pat::Unary { x: Box::new(self.clone()), uop: UOp::Not }
    }
}

impl std::ops::Not for Pat {
    type Output = Pat;
    fn not(self) -> Pat {
        Pat::Unary { x: Box::new(self), uop: UOp::Not }
    }
}

impl Pat {
    /// Matches anything and binds it to `name`. A repeated name asserts
    /// `OpId` equality with the first binding.
    pub fn bind(name: &'static str) -> Pat {
        Pat::Bind { name, class: DtypeClass::Any }
    }

    /// Matches if any alternative matches. Failed alternatives leave no bindings.
    pub fn any(pats: Vec<Pat>) -> Pat {
        Pat::Any(pats.into_boxed_slice())
    }

    /// Matches if all alternatives match the same op. Bindings merge.
    pub fn all(pats: Vec<Pat>) -> Pat {
        Pat::All(pats.into_boxed_slice())
    }

    /// Matches a constant satisfying this predicate (e.g. `Constant::is_max`).
    pub fn const_if(f: fn(Constant) -> bool) -> Pat {
        Pat::ConstIf(f)
    }

    /// Constrain this binder to floating-point dtypes. Panics on non-binders.
    pub fn float(self) -> Pat {
        self.with_class(DtypeClass::Float)
    }

    /// Constrain this binder to integer dtypes. Panics on non-binders.
    pub fn int(self) -> Pat {
        self.with_class(DtypeClass::Int)
    }

    /// Constrain this binder to unsigned integer dtypes. Panics on non-binders.
    pub fn uint(self) -> Pat {
        self.with_class(DtypeClass::Unsigned)
    }

    fn with_class(self, class: DtypeClass) -> Pat {
        match self {
            Pat::Bind { name, .. } => Pat::Bind { name, class },
            _ => panic!("dtype class applies to binders only"),
        }
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
    ///
    /// The pattern's own root is the prefilter: a `Binary`/`Unary` pattern
    /// rejects a mismatched root op with a single enum compare before any
    /// recursion or allocation, so call sites need no manual root check.
    pub fn match_pat(&self, root: OpId, pat: &Pat) -> Option<Map<&'static str, OpId>> {
        let mut bindings = Map::default();
        if self.match_node(pat, root, &mut bindings) { Some(bindings) } else { None }
    }

    /// Match one pattern node. Returns `false` with `bindings` untouched.
    fn match_node(&self, pat: &Pat, id: OpId, bindings: &mut Map<&'static str, OpId>) -> bool {
        match pat {
            Pat::Bind { name, class } => {
                if !class.matches(self.dtype(id)) {
                    return false;
                }
                match bindings.get(name) {
                    Some(&bound) => bound == id,
                    None => {
                        bindings.insert(*name, id);
                        true
                    }
                }
            }
            Pat::Const => matches!(self.at(id), Op::Const(_)),
            Pat::Value(v) => matches!(self.at(id), Op::Const(c) if c == v),
            Pat::ConstIf(f) => matches!(self.at(id), Op::Const(c) if f(*c)),
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
