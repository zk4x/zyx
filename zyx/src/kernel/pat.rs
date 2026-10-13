// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use crate::{
    DType,
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
/// unbound name panics via [`Bindings::dim`] (a comparison placed before
/// its binders is a bug, fail loud).
#[derive(Clone, Debug)]
pub enum VExpr {
    /// Value bound by [`Pat::bind_int`] under this name.
    Dim(char),
    /// Constant integer.
    Const(Dim),
    /// `a << b` (e.g. `Shl(1, k)` for `2^k`).
    Shl(Box<VExpr>, Box<VExpr>),
}

impl VExpr {
    /// Bound dim value.
    pub fn dim(name: char) -> VExpr {
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
            &VExpr::Dim(name) => Some(bindings.int(name)),
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

impl From<char> for VExpr {
    fn from(name: char) -> VExpr {
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
        name: char,
        /// Required dtype class of the matched op.
        class: DtypeClass,
    },
    /// Binds a constant's general integer value (the value-binder: value, not
    /// node). First occurrence matches any integer `Op::Const` (every int
    /// dtype, both signs, via [`Constant::as_integer`]) and records both
    /// the op and its value; a repeated name asserts *value* equality, so
    /// distinct const ops with equal values match. Floats never match.
    BindConst {
        /// Binder name.
        name: char,
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
    /// Matches a group index definition (`Op::Range`).
    Range,
    /// Matches a loop index definition (`Op::Loop`).
    Loop,
    /// Value equality between two [`VExpr`]s (bound dims, constants, shifts).
    /// Evaluated after the structural match; overflow evaluates to no-match.
    /// Compose trailing in `all()`, after the binders: `all([pat, Pat::eq('d', VExpr::shl(1, 'k'))])`.
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
    /// Matches a vector lane (`Op::Index`): the source vector matches
    /// `vec` and the lane number binds as an integer value under `idx`
    /// (read with [`Bindings::int`]).
    Index {
        /// Pattern for the source vector.
        vec: Box<Pat>,
        /// Binder name for the lane number.
        idx: char,
    },
    /// Matches a cast op (`Op::Cast`) to this exact dtype with an operand
    /// matching `x`.
    Cast {
        /// Pattern for the operand.
        x: Box<Pat>,
        /// Required dtype.
        dtype: DType,
    },
    /// Matches a bitcast op (`Op::Bitcast`) to this exact dtype with an
    /// operand matching `x`.
    Bitcast {
        /// Pattern for the operand.
        x: Box<Pat>,
        /// Required dtype.
        dtype: DType,
    },
    /// Matches a vector reduce (`Op::VecReduce`) with an operand matching
    /// `x`, binding the reduce `bop` under `bop` (read with
    /// [`Bindings::bop`]). A repeated name asserts `bop` equality.
    VecReduce {
        /// Pattern for the operand.
        x: Box<Pat>,
        /// Binder name for the reduce op.
        bop: char,
    },
    /// Fold over a `bop` spine of vector lanes. Matches when the cone is a
    /// chain of `bop` links (either operand order per link, `bop` being
    /// commutative) whose leaves are all `leaf`-matched lanes of ONE
    /// vector covering lanes `0..n` exactly, with `n` in `widths`.
    /// Self-validating: a match IS the whole reduction, no follow-up
    /// coverage check needed. The leaf's vector binders bind once (and
    /// every leaf must name the same vector op); the leaf's lane-number
    /// binder is set-collected, never bound — reading it panics.
    Seq {
        /// Fold op of the spine links.
        bop: BOp,
        /// Leaf pattern (v1: must be [`Pat::Index`]).
        leaf: Box<Pat>,
        /// Allowed vector widths.
        widths: Box<[u8]>,
    },
}

/// Bindings produced by [`Kernel::match_pat`]: matched subexpression ids
/// plus bound integer values.
///
/// Every [`Pat::Bind`] and [`Pat::BindInt`] records its op (read with
/// `m.op('x')`); value binders additionally record the general integer value
/// (read with [`Bindings::val`]). A repeated plain name asserts `OpId`
/// equality; a repeated value name asserts *value* equality across
/// distinct const ops.
#[derive(Clone, Debug, Default)]
pub struct Bindings {
    ops: [(char, OpId, Binding); 5],
}

#[derive(Clone, Debug, Default)]
enum Binding {
    Op,
    Int(i64),
    Float(f64),
    Bop(BOp),
    #[default]
    None,
}

impl PartialEq for Binding {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Binding::Op, Binding::Op) => true,
            (Binding::Int(a), Binding::Int(b)) => a == b,
            (Binding::Float(a), Binding::Float(b)) => a.to_bits() == b.to_bits(),
            (Binding::Bop(a), Binding::Bop(b)) => a == b,
            (Binding::None, Binding::None) => true,
            _ => false,
        }
    }
}

impl Bindings {
    /// Read a value bound by [`Pat::bind_int`]. Panics on unbound names.
    pub fn int(&self, name: char) -> i64 {
        for (n, _, binding) in &self.ops {
            if *n == name {
                if let Binding::Int(dim) = binding {
                    return *dim;
                }
                panic!("unbound value name");
            }
        }
        panic!("unbound value name");
    }

    /// Read a value bound by [`Pat::bind_float`]. Panics on unbound names.
    pub fn float(&self, name: char) -> f64 {
        for (n, _, binding) in &self.ops {
            if *n == name {
                if let Binding::Float(v) = binding {
                    return *v;
                }
                panic!("unbound value name");
            }
        }
        panic!("unbound value name");
    }

    /// Read an OpId bound by pattern
    pub fn op(&self, name: char) -> OpId {
        for (n, op, binding) in &self.ops {
            if *n == name {
                match binding {
                    Binding::None => panic!("unbound value name"),
                    Binding::Float(_) | Binding::Int(_) | Binding::Bop(_) | Binding::Op => return *op,
                }
            }
        }
        panic!("unbound value name");
    }

    /// Read a `BOp` bound by [`Pat::vecreduce`]. Panics on unbound names.
    pub fn bop(&self, name: char) -> BOp {
        for (n, _, binding) in &self.ops {
            if *n == name {
                match binding {
                    Binding::Bop(b) => return *b,
                    Binding::None | Binding::Float(_) | Binding::Int(_) | Binding::Op => panic!("unbound value name"),
                }
            }
        }
        panic!("unbound value name");
    }

    /// Insert with replacement. Returns `false` if the name is already
    /// bound to a *different* op or value (the pattern is then
    /// inconsistent — two distinct ops sharing one binder name) — the
    /// caller (`match_node`) rolls back the snapshot and rejects the
    /// whole match. A rebind to the identical op/value is a no-op
    /// success (the same binder reused across a pattern).
    fn try_insert(&mut self, name: char, id: OpId, value: Binding) -> bool {
        for (n, op, binding) in &mut self.ops {
            if *n == name {
                // A repeated plain binder (`Binding::Op`) asserts OpId
                // equality: two distinct ops sharing one name is an
                // inconsistent pattern. A repeated const binder asserts
                // value equality across distinct const ops (the same
                // integer may live in two different Const ops).
                return match (&value, binding) {
                    (Binding::Op, Binding::Op) => *op == id,
                    (Binding::Int(a), Binding::Int(b)) => a == b,
                    (Binding::Float(a), Binding::Float(b)) => a.to_bits() == b.to_bits(),
                    (Binding::Bop(a), Binding::Bop(b)) => a == b,
                    _ => false,
                };
            }
            match binding {
                Binding::None => {
                    *n = name;
                    *op = id;
                    *binding = value;
                    return true;
                }
                _ => {}
            }
        }
        panic!("too many bindings, keep bindings under 5 values");
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
    pub fn bind(name: char) -> Pat {
        Pat::Bind { name, class: DtypeClass::Any }
    }

    /// Binds a constant's `as_dim` value. A repeated name asserts *value*
    /// equality, so distinct const ops with equal values match.
    pub fn bind_const(name: char) -> Pat {
        Pat::BindConst { name, class: DtypeClass::Any }
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
            Pat::BindConst { name, .. } => Pat::BindConst { name, class },
            _ => panic!("dtype class applies to binders only"),
        }
    }

    /// Constrain this binder to compile-time constants. Panics on non-binders.
    pub fn const_(self) -> Pat {
        match self {
            Pat::Bind { .. } | Pat::BindConst { .. } => Pat::all([self, Pat::Const]),
            _ => panic!("const constraint applies to binders only"),
        }
    }

    /// Constrain this binder to constants with `as_dim` below `bound`.
    /// Panics on non-binders.
    pub fn dim_lt(self, bound: Dim) -> Pat {
        match self {
            Pat::Bind { .. } | Pat::BindConst { .. } => Pat::all([self, Pat::DimLt(bound)]),
            _ => panic!("dim constraint applies to binders only"),
        }
    }

    /// Constrain this binder to constants with `as_dim` equal to `v`.
    /// Panics on non-binders.
    pub fn dim_eq(self, v: Dim) -> Pat {
        match self {
            Pat::Bind { .. } | Pat::BindConst { .. } => Pat::all([self, Pat::DimEq(v)]),
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

    /// Vector lane of the `vec` pattern, binding the lane number as an
    /// integer value under `idx` (read with [`Bindings::int`]).
    pub fn index(vec: impl Into<Pat>, idx: char) -> Pat {
        Pat::Index { vec: Box::new(vec.into()), idx }
    }

    /// Cast to `dtype` of the `x` pattern.
    pub fn cast(x: impl Into<Pat>, dtype: DType) -> Pat {
        Pat::Cast { x: Box::new(x.into()), dtype }
    }

    /// Bitcast to `dtype` of the `x` pattern.
    pub fn bitcast(x: impl Into<Pat>, dtype: DType) -> Pat {
        Pat::Bitcast { x: Box::new(x.into()), dtype }
    }

    /// Fold over a `bop` spine of `leaf`-matched vector lanes covering one
    /// of `widths` exactly (see [`Pat::Seq`]).
    pub fn seq(bop: BOp, leaf: impl Into<Pat>, widths: &[u8]) -> Pat {
        Pat::Seq { bop, leaf: Box::new(leaf.into()), widths: widths.into() }
    }

    /// Vector reduce of the `x` pattern, binding the reduce op under `bop`.
    pub fn vecreduce(x: impl Into<Pat>, bop: char) -> Pat {
        Pat::VecReduce { x: Box::new(x.into()), bop }
    }

    /// Negation of this pattern.
    pub fn neg(&self) -> Pat {
        Pat::Unary { x: Box::new(self.clone()), uop: UOp::Neg }
    }

    /// Exponential of this pattern (`2^x`).
    pub fn exp2(self) -> Pat {
        Pat::Unary { x: Box::new(self), uop: UOp::Exp2 }
    }

    /// Reciprocal of this pattern (`1/x`).
    pub fn recip(self) -> Pat {
        Pat::Unary { x: Box::new(self), uop: UOp::Reciprocal }
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

/// One agreed cover group from [`Kernel::match_cover`]: the shared base
/// bindings (lane binder excluded), plus the roots and their lane numbers
/// in kernel order.
#[derive(Clone, Debug)]
pub struct CoverGroup {
    /// Agreed bindings (tile, bop, ...).
    pub bindings: Bindings,
    /// Root op per lane, kernel order.
    pub roots: Vec<OpId>,
    /// Lane number per root, parallel to `roots`.
    pub lanes: Vec<i64>,
}

impl Kernel {
    /// Multi-root cover over the existing leaf machinery: match `root`
    /// against every op (an arbitrary op sequence over `Index` lanes of
    /// one base); group successes by all non-lane binders (op, bop and
    /// value equality — the agreement `try_insert` asserts within one
    /// match, reused across roots). The single int binder is the lane
    /// number and must form one exact `{0, d, 2d, ..}` progression — a
    /// single stride, `Seq`-style self-validating — with the group size
    /// in `widths`. Returns one group per agreed base. Shapes and
    /// use-counts stay in the calling pass; only the final mutation
    /// lives there.
    pub fn match_cover(&self, root: &Pat, widths: &[u8]) -> Vec<CoverGroup> {
        let mut hits: Vec<(OpId, Bindings)> = Vec::new();
        let mut id = self.head;
        while !id.is_null() {
            let mut b = Bindings::default();
            if self.match_node(root, id, &mut b) {
                hits.push((id, b));
            }
            id = self.next_op(id);
        }
        // Key = agreed binders, lane = the single int binder.
        let mut keyed: Vec<(Vec<(char, OpId, Binding)>, Vec<OpId>, Vec<i64>)> = Vec::new();
        for (id, b) in &hits {
            let mut lane: Option<(char, i64)> = None;
            let mut key = Vec::new();
            let mut bad = false;
            for (n, op, binding) in &b.ops {
                match binding {
                    Binding::None => {}
                    Binding::Int(v) => {
                        if lane.is_some() {
                            bad = true;
                        }
                        lane = Some((*n, *v));
                    }
                    Binding::Op | Binding::Bop(_) | Binding::Float(_) => key.push((*n, *op, binding.clone())),
                }
            }
            if bad {
                continue;
            }
            if lane.is_none() {
                continue;
            }
            let l = lane.unwrap().1;
            if let Some(g) = keyed.iter_mut().find(|(k, _, _)| *k == key) {
                g.1.push(*id);
                g.2.push(l);
            } else {
                keyed.push((key, vec![*id], vec![l]));
            }
        }
        let mut out = Vec::new();
        for (key, roots, lanes) in keyed {
            let mut sorted = lanes.clone();
            sorted.sort_unstable();
            sorted.dedup();
            let n = sorted.len();
            if roots.len() != n {
                continue;
            }
            if u8::try_from(n).ok().is_none_or(|w| !widths.contains(&w)) {
                continue;
            }
            if sorted.is_empty() || sorted[0] != 0 {
                continue;
            }
            let mut ok = true;
            if n > 1 {
                let d = sorted[1] - sorted[0];
                if d <= 0 {
                    continue;
                }
                for (i, k) in sorted.iter().enumerate() {
                    if *k != i as i64 * d {
                        ok = false;
                    }
                }
            }
            if !ok {
                continue;
            }
            let mut agreed = Bindings::default();
            for (name, op, binding) in &key {
                debug_assert!(agreed.try_insert(*name, *op, binding.clone()));
            }
            out.push(CoverGroup { bindings: agreed, roots, lanes });
        }
        out
    }
    /// Match the IR cone above `root` against `pat`. Takes anything borrowing
    /// a pattern: a fresh owned temporary (`match_pat(id, Pat::bind("x") / n)`)
    /// or a shared reference to a reused pattern (`match_pat(id, &pat)`).
    ///
    /// Returns the bound subexpression ids on success. A repeated [`Pat::Bind`]
    /// name asserts `OpId` equality; commutative [`BOp`]s try both operand
    /// orders with full binding backtracking. Only the cone *above* `root`
    /// (definitions, never uses) is inspected — use-counts, shapes and
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
                bindings.try_insert(*name, id, Binding::Op)
            }
            Pat::BindConst { name, class } => {
                if !class.matches(self.dtype(id)) {
                    return false;
                }
                match self.at(id) {
                    Op::Const(c) => {
                        if let Some(v) = c.as_integer() {
                            bindings.try_insert(*name, id, Binding::Int(v))
                        } else if let Some(v) = c.as_float() {
                            bindings.try_insert(*name, id, Binding::Float(v))
                        } else {
                            false
                        }
                    }
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
            Pat::Range => matches!(self.at(id), Op::Range { .. }),
            Pat::Loop => matches!(self.at(id), Op::Loop { .. }),
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
            Pat::Index { vec, idx } => match self.at(id) {
                Op::Index { vec: vid, idx: i } => {
                    let snap = bindings.clone();
                    if !self.match_node(vec, *vid, bindings) {
                        *bindings = snap;
                        return false;
                    }
                    let Some(v) = i64::try_from(*i).ok() else {
                        *bindings = snap;
                        return false;
                    };
                    if !bindings.try_insert(*idx, id, Binding::Int(v)) {
                        *bindings = snap;
                        return false;
                    }
                    true
                }
                _ => false,
            },
            Pat::Cast { x, dtype } => match self.at(id) {
                Op::Cast { x: xid, dtype: d } => *d == *dtype && self.match_node(x, *xid, bindings),
                _ => false,
            },
            Pat::Bitcast { x, dtype } => match self.at(id) {
                Op::Bitcast { x: xid, dtype: d } => *d == *dtype && self.match_node(x, *xid, bindings),
                _ => false,
            },
            Pat::VecReduce { x, bop } => match self.at(id) {
                Op::VecReduce { x: xid, bop: b } => {
                    let snap = bindings.clone();
                    if self.match_node(x, *xid, bindings) {
                        if bindings.try_insert(*bop, id, Binding::Bop(*b)) {
                            return true;
                        }
                        *bindings = snap;
                    } else {
                        *bindings = snap;
                    }
                    false
                }
                _ => false,
            },
            Pat::Seq { bop, leaf, widths } => {
                // Lane-per-side matching is order-insensitive: only
                // commutative folds are sound (an ordered fold would need
                // lane order in the contract). Anything else never matches.
                if !bop.is_commutative() {
                    return false;
                }
                // v1: sequence leaves are vector lanes; anything else
                // never matches. Bindings untouched on this path.
                let Pat::Index { vec: vec_pat, .. } = leaf.as_ref() else { return false };
                let snap = bindings.clone();
                // Lane or spine-continuation per operand. A lane matches
                // the leaf's vector pattern with the shared bindings (so
                // the leaf's vector binders must agree across leaves) and
                // contributes its number to the collected set; anything
                // else continues the spine only as a same-`bop` link.
                let mut lanes: Vec<i64> = Vec::new();
                let mut first_vec: Option<OpId> = None;
                let mut cur = id;
                let walked = loop {
                    let (x, y) = match self.at(cur) {
                        Op::Binary { x, y, bop: b } if *b == *bop => (*x, *y),
                        _ => break false,
                    };
                    let lane = |o: OpId, bindings: &mut Bindings| match self.at(o) {
                        Op::Index { vec: vid, idx }
                            if self.match_node(vec_pat, *vid, bindings) =>
                        {
                            i64::try_from(*idx).ok().map(|k| (*vid, k))
                        }
                        _ => None,
                    };
                    let xl = lane(x, &mut *bindings);
                    let yl = lane(y, &mut *bindings);
                    match (xl, yl) {
                        // Bottom link: two lanes of the same vector.
                        (Some((vx, kx)), Some((vy, ky))) => {
                            if vx == vy && first_vec.map_or(true, |w| w == vx) {
                                first_vec = Some(vx);
                                lanes.push(kx);
                                lanes.push(ky);
                                break true;
                            }
                            break false;
                        }
                        // Spine link: one lane, the other side continues.
                        (Some((v, k)), None) | (None, Some((v, k))) => {
                            if first_vec.map_or(false, |w| w != v) {
                                break false;
                            }
                            first_vec = Some(v);
                            lanes.push(k);
                            let next = if xl.is_some() { y } else { x };
                            if matches!(self.at(next), Op::Binary { bop: b, .. } if *b == *bop) {
                                cur = next;
                            } else {
                                break false;
                            }
                        }
                        (None, None) => break false,
                    }
                };
                // Self-validating: the collected set must be exactly
                // lanes 0..n of a vector whose width is allowed.
                if walked && let Some(v) = first_vec {
                    let shape = self.shape(v);
                    if let [w] = shape.as_slice()
                        && let Ok(w8) = u8::try_from(*w)
                        && widths.contains(&w8)
                    {
                        lanes.sort_unstable();
                        lanes.dedup();
                        if lanes.len() == *w as usize && lanes.iter().enumerate().all(|(i, &k)| k == i as i64) {
                            return true;
                        }
                    }
                }
                *bindings = snap;
                false
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

/// Dtype-respective float equality for [`Pat::Float`]: true when `c` has
/// the same value as `n` within a tolerance of a few ulps of `c`'s own
/// dtype at the magnitude of `n`. Integers and bool compare exactly.
fn const_eq_float(c: Constant, n: f64) -> bool {
    match c {
        // Integer and boolean constants: exact numeric equality.
        Constant::U8(v) => (v as f64) == n,
        Constant::U16(v) => (v as f64) == n,
        Constant::U32(v) => (v as f64) == n,
        Constant::U64(v) => (u64::from_le_bytes(v) as f64) == n,
        Constant::I8(v) => (v as f64) == n,
        Constant::I16(v) => (v as f64) == n,
        Constant::I32(v) => (v as f64) == n,
        Constant::I64(v) => (i64::from_le_bytes(v) as f64) == n,
        Constant::Bool(v) => v == (n != 0.),
        // Float constants: compare at a tolerance of a few ulps of the
        // constant's own dtype at the magnitude of `n`.
        Constant::F64(v) => f64::from_le_bytes(v) == n,
        Constant::F32(v) => {
            let val = f32::from_le_bytes(v) as f64;
            (val - n).abs() <= n.abs().max(1.0) * f32::EPSILON as f64 * 8.0
        }
        Constant::F16(v) => {
            let val = f16::from_le_bytes(v).to_f64();
            (val - n).abs() <= n.abs().max(1.0) * 2.0_f64.powi(-10) * 8.0
        }
        Constant::BF16(v) => {
            let val = bf16::from_le_bytes(v).to_f64();
            (val - n).abs() <= n.abs().max(1.0) * 2.0_f64.powi(-8) * 8.0
        }
        Constant::F8E4M3(v) => {
            let val = f8e4m3::from_le_bytes([v]).to_f64();
            (val - n).abs() <= n.abs().max(1.0) * 2.0_f64.powi(-4) * 4.0
        }
        Constant::F8E5M2(v) => {
            let val = f8e5m2::from_le_bytes([v]).to_f64();
            (val - n).abs() <= n.abs().max(1.0) * 2.0_f64.powi(-3) * 4.0
        }
    }
}

/// Exact integral-float comparison: `fract == 0` plus a strict range check
/// so the `as i64` conversion below is exact (no saturation edge at `MAX`).
fn float_eq_int(v: f64, n: i64) -> bool {
    v.fract() == 0.0 && v >= -9223372036854775808.0 && v < 9223372036854775808.0 && v as i64 == n
}
