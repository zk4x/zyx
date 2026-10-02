// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use crate::{
    DType, Map,
    dtype::Constant,
    kernel::{BOp, Kernel, Op, OpId, UOp},
    scalar::{bf16, f8e4m3, f8e5m2, f16},
    shape::Dim,
};
use std::borrow::Borrow;

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

/// Value expression over bound dims, evaluated after the structural match.
/// The only computation is [`VExpr::Shl`] (the `d == 1<<k` relation);
/// anything richer stays in the calling pass until a rule needs it.
///
/// Evaluation returns `None` on overflow (never panics, so guard order
/// cannot crash a match) and on genuinely unrepresentable shifts; an
/// unbound name panics via [`Bindings::val`] (a comparison placed before
/// its binders is a bug, fail loud).
#[derive(Clone, Debug)]
pub enum VExpr {
    /// Value bound by [`Pat::bind_int`] under this name.
    Dim(&'static str),
    /// Constant integer.
    Const(Dim),
    /// `a << b` (e.g. `Shl(1, k)` for `2^k`).
    Shl(Box<VExpr>, Box<VExpr>),
}

impl VExpr {
    /// Bound dim value.
    pub fn dim(name: &'static str) -> VExpr {
        VExpr::Dim(name)
    }

    /// Constant integer.
    pub fn num(v: i64) -> VExpr {
        VExpr::Const(v)
    }

    /// `a << b`.
    pub fn shl(a: impl Into<VExpr>, b: impl Into<VExpr>) -> VExpr {
        VExpr::Shl(Box::new(a.into()), Box::new(b.into()))
    }

    /// Evaluate against bound values. `None` on overflow or out-of-range
    /// shift; unbound names panic.
    fn eval(&self, bindings: &Bindings) -> Option<i64> {
        match self {
            VExpr::Dim(name) => Some(bindings.val(name)),
            VExpr::Const(v) => Some(*v),
            VExpr::Shl(a, b) => {
                let (av, bv) = (a.eval(bindings)?, b.eval(bindings)?);
                if !(0..64).contains(&bv) {
                    return None;
                }
                i64::try_from((av as i128) << (bv as u32)).ok()
            }
        }
    }
}

impl From<&'static str> for VExpr {
    fn from(name: &'static str) -> VExpr {
        VExpr::Dim(name)
    }
}

impl From<i64> for VExpr {
    fn from(v: i64) -> VExpr {
        VExpr::Const(v)
    }
}

/// Pattern for matching kernel (and graph) IR shapes.
///
/// `Pat` is immutable: patterns are built once from constructors and
/// operators, then matched by shared reference. No method takes `&mut Pat`.
#[derive(Clone, Debug)]
pub enum Pat {
    /// Matches anything and binds it to `name`. A repeated name asserts
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
    Int(i64),
    /// Float, dtypeless
    Float(f64),
    /// Matches a constant whose `as_dim` value is less than this bound.
    /// Unlike [`Pat::Num`], floats and negatives never match (`as_dim`
    /// semantics): the in-pattern form of a `< bound` value guard.
    DimLt(Dim),
    /// Matches a constant whose `as_dim` value equals this bound. The
    /// in-pattern form of a `const_dim(id) == Some(v)` value guard.
    DimEq(Dim),
    /// Value equality between two [`VExpr`]s (bound dims, constants, shifts).
    /// Evaluated after the structural match; overflow evaluates to no-match.
    /// Compose trailing in `all()`, after the binders: `all([pat, Pat::eq("d", VExpr::shl(1, "k"))])`.
    Eq {
        /// Left-hand value.
        a: VExpr,
        /// Right-hand value.
        b: VExpr,
    },
    /// Value inequality between two [`VExpr`]s.
    Ne {
        /// Left-hand value.
        a: VExpr,
        /// Right-hand value.
        b: VExpr,
    },
    /// Strict `a < b` between two [`VExpr`]s.
    Lt {
        /// Left-hand value.
        a: VExpr,
        /// Right-hand value.
        b: VExpr,
    },
    /// `a <= b` between two [`VExpr`]s.
    Le {
        /// Left-hand value.
        a: VExpr,
        /// Right-hand value.
        b: VExpr,
    },
    /// `a > b` between two [`VExpr`]s.
    Gt {
        /// Left-hand value.
        a: VExpr,
        /// Right-hand value.
        b: VExpr,
    },
    /// `a >= b` between two [`VExpr`]s.
    Ge {
        /// Left-hand value.
        a: VExpr,
        /// Right-hand value.
        b: VExpr,
    },
    /// Binds a constant's general integer value (the value-binder: value, not
    /// node). First occurrence matches any integer `Op::Const` (every int
    /// dtype, both signs, via [`Constant::as_integer`]) and records both
    /// the op and its value; a repeated name asserts *value* equality, so
    /// distinct const ops with equal values match. Floats never match.
    BindInt {
        /// Binder name.
        name: &'static str,
        /// Required dtype class of the matched op.
        class: DtypeClass,
    },
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

/// Bindings produced by [`Kernel::match_pat`]: matched subexpression ids
/// plus bound integer values.
///
/// Every [`Pat::Bind`] and [`Pat::BindInt`] records its op (read with
/// `m["x"]`); value binders additionally record the general integer value
/// (read with [`Bindings::val`]). A repeated plain name asserts `OpId`
/// equality; a repeated value name asserts *value* equality across
/// distinct const ops.
#[derive(Clone, Debug, Default)]
pub struct Bindings {
    ops: Map<&'static str, OpId>,
    vals: Map<&'static str, Dim>,
}

impl Bindings {
    /// Read a value bound by [`Pat::bind_int`]. Panics on unbound names.
    pub fn val(&self, name: &'static str) -> Dim {
        self.vals.get(name).copied().expect("unbound value name")
    }
}

impl std::ops::Index<&'static str> for Bindings {
    type Output = OpId;
    fn index(&self, name: &'static str) -> &OpId {
        &self.ops[name]
    }
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

    /// Binds a constant's `as_dim` value. A repeated name asserts *value*
    /// equality, so distinct const ops with equal values match.
    pub fn bind_int(name: &'static str) -> Pat {
        Pat::BindInt { name, class: DtypeClass::Any }
    }

    /// Matches if any alternative matches. Failed alternatives leave no bindings.
    pub fn any<const N: usize>(pats: [Pat; N]) -> Pat {
        Pat::Any(Box::new(pats))
    }

    /// Matches if all alternatives match the same op. Bindings merge.
    pub fn all<const N: usize>(pats: [Pat; N]) -> Pat {
        Pat::All(Box::new(pats))
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
            Pat::BindInt { name, .. } => Pat::BindInt { name, class },
            _ => panic!("dtype class applies to binders only"),
        }
    }

    /// Constrain this binder to compile-time constants. Panics on non-binders.
    pub fn const_(self) -> Pat {
        match self {
            Pat::Bind { .. } | Pat::BindInt { .. } => Pat::all([self, Pat::Const]),
            _ => panic!("const constraint applies to binders only"),
        }
    }

    /// Constrain this binder to constants with `as_dim` below `bound`.
    /// Panics on non-binders.
    pub fn dim_lt(self, bound: Dim) -> Pat {
        match self {
            Pat::Bind { .. } | Pat::BindInt { .. } => Pat::all([self, Pat::DimLt(bound)]),
            _ => panic!("dim constraint applies to binders only"),
        }
    }

    /// Constrain this binder to constants with `as_dim` equal to `v`.
    /// Panics on non-binders.
    pub fn dim_eq(self, v: Dim) -> Pat {
        match self {
            Pat::Bind { .. } | Pat::BindInt { .. } => Pat::all([self, Pat::DimEq(v)]),
            _ => panic!("dim constraint applies to binders only"),
        }
    }

    /// Value equality between two value expressions (bound names coerce from
    /// `&str`, constants from `i64`). Evaluated after the structural match.
    pub fn eq(a: impl Into<VExpr>, b: impl Into<VExpr>) -> Pat {
        Pat::Eq { a: a.into(), b: b.into() }
    }

    /// Value inequality between two value expressions.
    pub fn ne(a: impl Into<VExpr>, b: impl Into<VExpr>) -> Pat {
        Pat::Ne { a: a.into(), b: b.into() }
    }

    /// Strict `a < b` between two value expressions.
    pub fn lt(a: impl Into<VExpr>, b: impl Into<VExpr>) -> Pat {
        Pat::Lt { a: a.into(), b: b.into() }
    }

    /// `a <= b` between two value expressions.
    pub fn le(a: impl Into<VExpr>, b: impl Into<VExpr>) -> Pat {
        Pat::Le { a: a.into(), b: b.into() }
    }

    /// `a > b` between two value expressions.
    pub fn gt(a: impl Into<VExpr>, b: impl Into<VExpr>) -> Pat {
        Pat::Gt { a: a.into(), b: b.into() }
    }

    /// `a >= b` between two value expressions.
    pub fn ge(a: impl Into<VExpr>, b: impl Into<VExpr>) -> Pat {
        Pat::Ge { a: a.into(), b: b.into() }
    }

    /// Binary op with explicit [`BOp`] (for operators without an
    /// operator-overload spelling, e.g. comparisons). Operands coerce.
    pub fn binary(bop: BOp, x: impl Into<Pat>, y: impl Into<Pat>) -> Pat {
        Pat::Binary { x: Box::new(x.into()), y: Box::new(y.into()), bop }
    }

    /// Negation of this pattern.
    pub fn neg(self) -> Pat {
        Pat::Unary { x: Box::new(self), uop: UOp::Neg }
    }

    /// Exponential of this pattern (`2^x`).
    pub fn exp2(self) -> Pat {
        Pat::Unary { x: Box::new(self), uop: UOp::Exp2 }
    }

    /// Exponential of this pattern (`2^x`).
    pub fn exp(self) -> Pat {
        (self * std::f64::consts::LOG2_E).exp2()
    }
}

impl From<&Pat> for Pat {
    fn from(p: &Pat) -> Pat {
        p.clone()
    }
}

impl From<i64> for Pat {
    fn from(v: i64) -> Pat {
        Pat::Int(v)
    }
}

impl From<f64> for Pat {
    fn from(v: f64) -> Pat {
        Pat::Float(v)
    }
}

impl From<Constant> for Pat {
    fn from(c: Constant) -> Pat {
        Pat::Value(c)
    }
}

impl Kernel {
    /// Match the IR cone above `root` against `pat`. Takes anything borrowing
    /// a pattern: a fresh owned temporary (`match_pat(id, Pat::bind("x") / n)`)
    /// or a shared reference to a reused pattern (`match_pat(id, &pat)`).
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
    pub fn match_pat(&self, root: OpId, pat: impl Borrow<Pat>) -> Option<Bindings> {
        let mut bindings = Bindings::default();
        if self.match_node(pat.borrow(), root, &mut bindings) {
            Some(bindings)
        } else {
            None
        }
    }

    /// Match one pattern node. Returns `false` with `bindings` untouched.
    fn match_node(&self, pat: &Pat, id: OpId, bindings: &mut Bindings) -> bool {
        match pat {
            Pat::Bind { name, class } => {
                if !class.matches(self.dtype(id)) {
                    return false;
                }
                match bindings.ops.get(name) {
                    Some(&bound) => bound == id,
                    None => {
                        bindings.ops.insert(*name, id);
                        true
                    }
                }
            }
            Pat::BindInt { name, class } => {
                if !class.matches(self.dtype(id)) {
                    return false;
                }
                match self.at(id) {
                    Op::Const(c) => match c.as_integer() {
                        Some(v) => match bindings.vals.get(name) {
                            Some(&bound) => bound == v,
                            None => {
                                bindings.vals.insert(*name, v);
                                bindings.ops.insert(*name, id);
                                true
                            }
                        },
                        None => false,
                    },
                    _ => false,
                }
            }
            Pat::Const => matches!(self.at(id), Op::Const(_)),
            Pat::Value(v) => matches!(self.at(id), Op::Const(c) if c == v),
            Pat::ConstIf(f) => matches!(self.at(id), Op::Const(c) if f(*c)),
            Pat::Int(n) => matches!(self.at(id), Op::Const(c) if const_eq_int(*c, *n)),
            Pat::Float(n) => matches!(self.at(id), Op::Const(c) if const_eq_float(*c, *n)),
            Pat::DimLt(n) => matches!(self.at(id), Op::Const(c) if c.as_dim().is_some_and(|k| k < *n)),
            Pat::DimEq(n) => matches!(self.at(id), Op::Const(c) if c.as_dim() == Some(*n)),
            Pat::Eq { a, b } => matches!((a.eval(bindings), b.eval(bindings)), (Some(x), Some(y)) if x == y),
            Pat::Ne { a, b } => matches!((a.eval(bindings), b.eval(bindings)), (Some(x), Some(y)) if x != y),
            Pat::Lt { a, b } => matches!((a.eval(bindings), b.eval(bindings)), (Some(x), Some(y)) if x < y),
            Pat::Le { a, b } => matches!((a.eval(bindings), b.eval(bindings)), (Some(x), Some(y)) if x <= y),
            Pat::Gt { a, b } => matches!((a.eval(bindings), b.eval(bindings)), (Some(x), Some(y)) if x > y),
            Pat::Ge { a, b } => matches!((a.eval(bindings), b.eval(bindings)), (Some(x), Some(y)) if x >= y),
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
fn const_eq_int(c: Constant, n: i64) -> bool {
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
        Constant::F32(v) => float_eq_int(f32::from_le_bytes(v) as f64, n),
        Constant::F64(v) => float_eq_int(f64::from_le_bytes(v), n),
        Constant::BF16(v) => float_eq_int(bf16::from_le_bytes(v).to_f64(), n),
        Constant::F16(v) => float_eq_int(f16::from_le_bytes(v).to_f64(), n),
        Constant::F8E4M3(v) => float_eq_int(f8e4m3::from_le_bytes([v]).to_f64(), n),
        Constant::F8E5M2(v) => float_eq_int(f8e5m2::from_le_bytes([v]).to_f64(), n),
    }
}

fn const_eq_float(c: Constant, n: f64) -> bool {
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
        Constant::F32(v) => float_eq_int(f32::from_le_bytes(v) as f64, n),
        Constant::F64(v) => float_eq_int(f64::from_le_bytes(v), n),
        Constant::BF16(v) => float_eq_int(bf16::from_le_bytes(v).to_f64(), n),
        Constant::F16(v) => float_eq_int(f16::from_le_bytes(v).to_f64(), n),
        Constant::F8E4M3(v) => float_eq_int(f8e4m3::from_le_bytes([v]).to_f64(), n),
        Constant::F8E5M2(v) => float_eq_int(f8e5m2::from_le_bytes([v]).to_f64(), n),
    }
}

/// Exact integral-float comparison: `fract == 0` plus a strict range check
/// so the `as i64` conversion below is exact (no saturation edge at `MAX`).
fn float_eq_int(v: f64, n: i64) -> bool {
    v.fract() == 0.0 && v >= -9223372036854775808.0 && v < 9223372036854775808.0 && v as i64 == n
}

/// Exact -float comparison: `fract == 0` plus a strict range check
/// so the `as i64` conversion below is exact (no saturation edge at `MAX`).
fn float_eq_float(v: f64, n: f64) -> bool {
    todo!()
}
