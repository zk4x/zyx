// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Hashconsed symbolic expressions: shapes, dims, and scalar constants.
//!
//! Shapes used to live in the tensors slab as dedicated `TensorData` variants,
//! which tangled shape lifetimes with value lifetimes (a shape died with its
//! last value edge even while still referenced). Expressions live in their own
//! append-only slab instead: interning returns the same [`ExprId`] for equal
//! expressions, so structural equality is a single integer compare and nothing
//! is ever freed (no lifecycle bugs by construction).
//!
//! `Variable` is layout-identical to `Constant`; only the tag differs. The tag
//! tells the graph and the kernels not to bake the value into cache keys — the
//! per-launch value is bound from backend variable slots instead.

use std::hash::BuildHasherDefault;

use crate::{
    Map,
    dtype::{Constant, DType},
    graph::{ClassId, GraphId, Node},
    kernel::{BOp, IDX_T, Op, OpId, UOp},
    runtime::{KernelId, ResolvedDim, Runtime, TensorData},
    shape::Dim,
    slab::SlabId,
    tensor::TensorId,
};

/// Identifier of an interned [`Expr`]: stable forever (append-only slab).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprId(pub u32);

impl ExprId {
    /// First valid index (0).
    pub const ZERO: Self = Self(0);
    /// Sentinel for "no expression".
    pub const NULL: Self = Self(u32::MAX);
    /// Whether this is [`ExprId::NULL`].
    pub const fn is_null(self) -> bool {
        self.0 == u32::MAX
    }
}

impl From<usize> for ExprId {
    fn from(value: usize) -> Self {
        Self(value as u32)
    }
}

impl From<ExprId> for usize {
    fn from(value: ExprId) -> usize {
        value.0 as usize
    }
}

impl SlabId for ExprId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

/// A symbolic scalar or shape: an interned, hashconsed expression tree.
///
/// Leaves are [`Expr::Constant`] (baked into cache keys) and [`Expr::Variable`]
/// (bound per launch, excluded from cache keys). Shapes are `Stack*` nodes
/// over dim expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    /// Baked constant: value participates in hashing and cache keys.
    Constant {
        /// The value (carries its own dtype).
        value: Constant,
    },
    /// Launch-bound variable: same layout as [`Expr::Constant`], but the value
    /// is read from backend variable slots per launch and never baked into
    /// cache keys.
    Variable {
        /// The default/fallback value (carries its own dtype).
        value: Constant,
    },
    /// Dtype conversion.
    Cast {
        /// Input expression.
        x: ExprId,
        /// Target dtype.
        dtype: DType,
    },
    /// Element-wise unary op.
    Unary {
        /// Input expression.
        x: ExprId,
        /// The op.
        uop: UOp,
    },
    /// Element-wise binary op.
    Binary {
        /// Left input expression.
        x: ExprId,
        /// Right input expression.
        y: ExprId,
        /// The op.
        bop: BOp,
    },
    /// Heap-allocated stack of expressions (shapes of rank 6+).
    Stack {
        /// Child expressions.
        exprs: Box<[ExprId]>,
    },
    /// Inline stack of 2 expressions.
    Stack2 {
        /// Child expressions.
        exprs: [ExprId; 2],
    },
    /// Inline stack of 3 expressions.
    Stack3 {
        /// Child expressions.
        exprs: [ExprId; 3],
    },
    /// Inline stack of 4 expressions.
    Stack4 {
        /// Child expressions.
        exprs: [ExprId; 4],
    },
    /// Inline stack of 5 expressions.
    Stack5 {
        /// Child expressions.
        exprs: [ExprId; 5],
    },
}

// Budget: 16 bytes target, 24 accepted (Box<[ExprId]> is 16 alone).
const _: () = assert!(core::mem::size_of::<Expr>() <= 24);

impl Runtime {
    /// Hashcons interning: returns the existing [`ExprId`] for a structurally
    /// equal expression, or pushes and returns a fresh one. Append-only —
    /// ids are stable forever and nothing is freed.
    pub fn intern(&mut self, expr: Expr) -> ExprId {
        if let Some(&id) = self.expr_hash.get(&expr) {
            return id;
        }
        let id = self.exprs.push(expr.clone());
        self.expr_hash.insert(expr, id);
        id
    }

    /// Host-side evaluation of a scalar expression: folds the interned
    /// expression tree (`Constant` / `Variable` / `Cast` / `Unary` /
    /// `Binary`) into a single `Constant`, preserving dtype. Variables
    /// evaluate to the value stored in their (immutable) node — a changed
    /// value is a different node, so evaluation never reads anything
    /// outside the expression table. Returns `None` for non-symbolic
    /// tensors. Stack shapes are not scalars and return `None`.
    pub(crate) fn resolve_symbolic(&self, x: TensorId) -> Option<Constant> {
        match self.tensors[x] {
            TensorData::Symbolic { expr, .. } => {
                let mut memo: Map<ExprId, (Constant, bool)> = Map::with_hasher(BuildHasherDefault::new());
                Some(self.fold_dim(expr, &mut memo).0)
            }
            _ => None,
        }
    }

    /// Fold one scalar dim expression: returns the folded value AND whether
    /// the tree contains a [`Expr::Variable`], in a single pass. One `memo`
    /// shared across every dim of a shape means each interned node is
    /// visited exactly once, no matter how many dims share subexpressions.
    /// Variables fold to the value stored in their (immutable) node — a
    /// changed value is a different node, so evaluation never reads
    /// anything outside the expression table. Total over the scalar closed
    /// set; stacks and anything else panic.
    pub(crate) fn fold_dim(&self, root: ExprId, memo: &mut Map<ExprId, (Constant, bool)>) -> (Constant, bool) {
        if let Some(&v) = memo.get(&root) {
            return v;
        }
        let v = match self.exprs[root] {
            Expr::Constant { value } => (value, false),
            Expr::Variable { value } => (value, true),
            Expr::Cast { x: a, dtype } => {
                let (v, tainted) = self.fold_dim(a, memo);
                (v.cast(dtype), tainted)
            }
            Expr::Unary { x: a, uop } => {
                let (v, tainted) = self.fold_dim(a, memo);
                (v.unary(uop), tainted)
            }
            Expr::Binary { x: a, y: b, bop } => {
                let (va, ta) = self.fold_dim(a, memo);
                let (vb, tb) = self.fold_dim(b, memo);
                (Constant::binary(va, vb, bop), ta || tb)
            }
            ref e => panic!("dim expression {root:?} is not a scalar expression: {e:?}"),
        };
        memo.insert(root, v);
        v
    }

    /// Dtype of an interned scalar expression: walks to the
    /// `Constant`/`Variable`/`Cast` leaf that determines it. Stack shapes
    /// have no dtype and panic.
    pub(crate) fn expr_dtype(&self, root: ExprId) -> DType {
        match self.exprs[root] {
            Expr::Constant { value, .. } | Expr::Variable { value, .. } => value.dtype(),
            Expr::Cast { dtype, .. } => dtype,
            Expr::Unary { x, .. } => self.expr_dtype(x),
            Expr::Binary { x, bop, .. } => {
                if bop.returns_bool() {
                    DType::Bool
                } else {
                    self.expr_dtype(x)
                }
            }
            ref e => panic!("expression {root:?} has no scalar dtype: {e:?}"),
        }
    }

    /// Dim expressions of a shape expression: a `Stack*` yields one dim per
    /// element; a bare scalar expression (1d shapes skip the `Stack` node)
    /// is a single dim; null is rank zero.
    pub(crate) fn shape_expr_ids(&self, shape_id: ExprId) -> Vec<ExprId> {
        if shape_id.is_null() {
            return Vec::new();
        }
        match &self.exprs[shape_id] {
            Expr::Stack { exprs } => exprs.to_vec(),
            Expr::Stack2 { exprs } => exprs.to_vec(),
            Expr::Stack3 { exprs } => exprs.to_vec(),
            Expr::Stack4 { exprs } => exprs.to_vec(),
            Expr::Stack5 { exprs } => exprs.to_vec(),
            _ => vec![shape_id],
        }
    }

    /// Dim expressions of tensor `x` without minting handles: unwraps the
    /// tensor's `shape_id` via [`Runtime::shape_expr_ids`]. A `Symbolic`
    /// tensor holding a `Stack` is a shape value — its elements are the dims.
    /// Any other symbolic is a scalar VALUE (constant, variable, arithmetic)
    /// with no dims — its own expression is never a dim.
    pub(crate) fn tensor_shape_exprs(&self, x: TensorId) -> Vec<ExprId> {
        let shape_id = match self.tensors[x] {
            TensorData::Eager { shape_id, .. }
            | TensorData::Graph { shape_id, .. }
            | TensorData::Promoted { shape_id, .. }
            | TensorData::PendingLeaf { shape_id, .. }
            | TensorData::GraphLeaf { shape_id, .. }
            | TensorData::Leaf { shape_id, .. } => shape_id,
            TensorData::Symbolic { expr, .. } => match &self.exprs[expr] {
                Expr::Stack { exprs } => return exprs.to_vec(),
                Expr::Stack2 { exprs } => return exprs.to_vec(),
                Expr::Stack3 { exprs } => return exprs.to_vec(),
                Expr::Stack4 { exprs } => return exprs.to_vec(),
                Expr::Stack5 { exprs } => return exprs.to_vec(),
                _ => return Vec::new(),
            },
        };
        self.shape_expr_ids(shape_id)
    }

    /// Concrete (resolved) shape of tensor `x`: evaluates every dim expression
    /// to a `Dim` by folding the interned expression (variables evaluate to
    /// the value stored in their node).
    ///
    /// # Convention
    /// `shape` (the symbolic variant) should be used EVERYWHERE in kernel
    /// construction — shapes must stay symbolic (variable-backed) so that
    /// consumers sharing a dim reference the same op and symbolic dims survive
    /// to launch time. `resolve_shape` is mostly for debug checks, assertions
    /// and user-facing messages where a concrete value is genuinely needed;
    /// resolving a variable into a const shape in kernel IR silently breaks
    /// symbolic-dim consumers (they end up with a different op than the rest
    /// of the graph for the same dim).
    pub(crate) fn resolve_shape(&self, x: TensorId) -> Vec<Dim> {
        // A Symbolic tensor whose expression is a Stack describes a shape
        // value: its ACTUAL shape is [len] — not its elements' values. Dims
        // described by a shape expression are `resolve_symbolic_dims`'s job.
        if let TensorData::Symbolic { expr, .. } = self.tensors[x] {
            match &self.exprs[expr] {
                Expr::Stack { exprs } => return vec![exprs.len() as Dim],
                Expr::Stack2 { exprs } => return vec![exprs.len() as Dim],
                Expr::Stack3 { exprs } => return vec![exprs.len() as Dim],
                Expr::Stack4 { exprs } => return vec![exprs.len() as Dim],
                Expr::Stack5 { exprs } => return vec![exprs.len() as Dim],
                _ => {}
            }
        }
        self.tensor_shape_exprs(x)
            .into_iter()
            .map(|e| {
                let mut memo = Map::default();
                self.fold_dim(e, &mut memo).0.as_dim().expect("dim expression does not evaluate to an integer")
            })
            .collect()
    }

    /// Dims described by a SHAPE EXPRESSION (a `shape_id`): a `Stack` (any
    /// arity) yields one dim per element; a bare dim expr is a single dim —
    /// 1d shapes skip the Stack node, so their shape_id IS the dim expr.
    /// Panics on data tensors: those are values, not shape expressions —
    /// use [`Runtime::resolve_shape`] for their actual shape.
    pub(crate) fn resolve_symbolic_dims(&self, shape_id: ExprId) -> Vec<Dim> {
        self.shape_expr_ids(shape_id)
            .into_iter()
            .map(|e| {
                let mut memo = Map::default();
                self.fold_dim(e, &mut memo).0.as_dim().expect("dim expression does not evaluate to an integer")
            })
            .collect()
    }

    /// A dimension resolved for merge-compatibility checking (see
    /// [`Runtime::resolve_shape_without_variables`]).
    /// Shape of tensor `x` with variable-backed dims left symbolic: a dim
    /// whose expression tree contains any [`TensorData::Variable`] resolves
    /// to [`ResolvedDim::Symbolic`] (its root dim tensor), everything else
    /// evaluates to [`ResolvedDim::Static`]. Unlike `resolve_shape`, variable
    /// slots are never read — this is the PROVABILITY view of a shape: what
    /// can be checked for equality without depending on the variable's
    /// current bound value.
    ///
    /// Used by the merge-time compatibility checks in `binary` and `assign`:
    /// two shapes may only merge if every dim is provably equal — same
    /// constant, or the SAME symbolic dim tensor in both operands. If only
    /// the bound values agree, the merge is rejected with an error instead.
    ///
    /// TODO: the same-TensorId identity rule is a conservative proxy for
    /// algebraic equality of dim expressions. A factored normal form (split
    /// each dim expr into a multiset of irreducible atoms — every additive
    /// subexpression is one atom, constants folded — then cancel numerator
    /// against denominator atoms) would make e.g. `numel / divisor` Div nodes
    /// provably equal to the original symbolic dim, without requiring
    /// construction sites to preserve the exact same TensorId (see the `-1`
    /// inference in `Tensor::reshape` and llama's `repeat_kv`).
    pub(crate) fn resolve_shape_without_variables(&self, x: TensorId) -> Vec<ResolvedDim> {
        let mut dims = Vec::new();
        let mut memo = Map::default();
        for root in self.tensor_shape_exprs(x) {
            let (value, tainted) = self.fold_dim(root, &mut memo);
            if tainted {
                dims.push(ResolvedDim::Symbolic(root));
                continue;
            }
            dims.push(ResolvedDim::Static(value.as_dim().expect("dim expression does not evaluate to an integer")));
        }
        dims
    }

    /// Symbolic shape of tensor `x` as dim handles: one fresh
    /// [`TensorData::Symbolic`] tensor (rc = 1, caller-owned) per dimension.
    /// Scalar tensors (including scalar symbolic handles) have an empty
    /// shape. A symbolic handle whose expression is a `Stack` describes a
    /// shape value — its dims are the stack elements.
    ///
    /// # Convention
    /// This is the DEFAULT way to read a shape when building kernels — use it
    /// everywhere. `resolve_shape` (concrete evaluation) is mostly for debug
    /// checks and messages; resolving dims to consts in kernel IR breaks
    /// symbolic-dim consumers.
    pub fn shape(&mut self, x: TensorId) -> Vec<TensorId> {
        let exprs = self.tensor_shape_exprs(x);
        // Scalars have no dims — but a Stack expression IS a shape value.
        if let TensorData::Symbolic { expr, .. } = self.tensors[x] {
            match &self.exprs[expr] {
                Expr::Stack { .. } | Expr::Stack2 { .. } | Expr::Stack3 { .. } | Expr::Stack4 { .. } | Expr::Stack5 { .. } => {}
                _ => return Vec::new(),
            }
        }
        exprs.into_iter().map(|expr| self.tensors.push(TensorData::Symbolic { expr, rc: 1 })).collect()
    }

    /// Push a symbolic scalar expression (a tensors-slab tree: Constant /
    /// Variable / Unary / Binary / Stack) into `kernel` as concrete ops and
    /// return its root `OpId` together with the variable tids the expression
    /// loads (each retained once — the kernel now holds an edge to them).
    /// Constants and variables keep their own dtype; linearize's autocast
    /// handles mixing with the surrounding expression. Appends the expression's
    /// variable loads to the kernel and retains each (the kernel holds an edge
    /// to them). A null `shape` is a scalar: returns `OpId::NULL`.
    /// Replay a symbolic scalar/shape expression (a tensors-slab tree) into
    /// kernel IR, returning the root `OpId`.
    ///
    /// This is one third of the symbolic-shapes story; the other two live in
    /// [`Runtime::replay_symbolic_into_graph`] (slab → egraph) and the
    /// kernelizer's graph-side replay (egraph → kernel IR, see
    /// `Graph::replay_symbolic_into_kernel`). All three follow the same laws,
    /// which is what makes them interchangeable representations of the same
    /// expression tree.
    ///
    /// # The symbolic closed set
    ///
    /// Every shape and every dimension everywhere in zyx is a value built
    /// from exactly these slab variants — nothing else participates in a
    /// shape expression, ever:
    ///
    /// - `Constant` — baked at construction; carries its own dtype and is
    ///   emitted verbatim as `Op::Const` (linearize's autocast reconciles
    ///   dtype mixing with surrounding expression ops).
    /// - `Variable` — resolved at execution time from `variable_map`; each
    ///   occurrence becomes a `Param { kind: Variable }` define registered in
    ///   the owning kernel's `loads` under its originating `TensorId`.
    /// - `Cast`, `Unary`, `Binary` over already-mapped operands — replayed as
    ///   real kernel ops.
    /// - `Stack`, `Stack2`–`Stack5` — grouped into a single `Op::Stack`
    ///   (the fixed-arity `Stack2`–`Stack5` variants are the inline-storage
    ///   forms for 2–5 element shapes).
    ///
    /// Anything else reaching this walk is a bug and panics loudly here. No
    /// fallback, no fabricated dims, no folding: constants are NOT folded
    /// into precomputed values because linearize and verify reason about the
    /// symbolic structure itself.
    ///
    /// # Positional binding (the args law)
    ///
    /// All `Param` defines of a kernel — global buffers and scalar variables
    /// alike — appear in flat head order in the kernel IR, and the launch-time
    /// args slice binds positionally over exactly that sequence. Deduplicating
    /// repeated variable tids via `op_map` therefore preserves correctness:
    /// fewer defines, and each surviving define still maps to the same slot in
    /// whatever positional binding the caller passes. See also
    /// `kernel::verify`'s checks and the gws section of AGENTS.md.
    ///
    /// # Why registration rides on `loads`
    ///
    /// Each `Variable` leaf adds `tid` to `KernelData::loads` and takes an rc.
    /// This is what ties an abstract define back to the pooled value at
    /// launch time without any parallel bookkeeping structure: `n_params ==
    /// loads.len()` remains an enforced invariant.
    pub fn replay_symbolic_into_kernel(&mut self, kid: KernelId, shape: TensorId) -> OpId {
        if shape.is_null() {
            return OpId::NULL;
        }
        let root = match self.tensors[shape] {
            TensorData::Symbolic { expr, .. } => expr,
            ref t => panic!("replay_symbolic_into_kernel: tid {shape} is not symbolic: {t:?}"),
        };

        // Flatten the tree post-order: every node lands after its operands,
        // so the flat emit loop below always finds children already mapped.
        // Duplicated from `replay_expr` by design (no shared abstraction).
        fn flatten(rt: &Runtime, x: ExprId, order: &mut Vec<ExprId>) {
            if order.contains(&x) {
                return;
            }
            match &rt.exprs[x] {
                Expr::Constant { .. } | Expr::Variable { .. } => (),
                Expr::Cast { x: a, .. } => flatten(rt, *a, order),
                Expr::Unary { x: a, .. } => flatten(rt, *a, order),
                Expr::Binary { x: a, y: b, .. } => {
                    flatten(rt, *a, order);
                    flatten(rt, *b, order);
                }
                Expr::Stack { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
                Expr::Stack2 { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
                Expr::Stack3 { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
                Expr::Stack4 { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
                Expr::Stack5 { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
            }
            order.push(x);
        }
        let mut order = Vec::new();
        flatten(self, root, &mut order);

        let mut op_map: Map<ExprId, OpId> = Map::with_hasher(BuildHasherDefault::new());
        let mut var_handles: Map<ExprId, TensorId> = Map::with_hasher(BuildHasherDefault::new());
        let mut root_op = OpId::NULL;
        for eid in order {
            let node = self.exprs[eid].clone();
            let op_id = match node {
                Expr::Constant { value } => self.kernels[kid].kernel.push_back(Op::Const(value)),
                Expr::Variable { value } => {
                    let op_id = self.kernels[kid].kernel.variable(value.dtype());
                    // Load-owned handle (rc = 1 is the load edge; no user
                    // handle). Released with the kernel's loads.
                    let tid = self.tensors.push(TensorData::Symbolic { expr: eid, rc: 1 });
                    self.kernels[kid].loads.push(tid);
                    var_handles.insert(eid, tid);
                    op_id
                }
                Expr::Cast { x, dtype } => {
                    let a = op_map[&x];
                    self.kernels[kid].kernel.cast(a, dtype)
                }
                Expr::Unary { x, uop } => {
                    let a = op_map[&x];
                    self.kernels[kid].kernel.unary(a, uop)
                }
                Expr::Binary { x, y, bop } => {
                    let (a, b) = (op_map[&x], op_map[&y]);
                    self.kernels[kid].kernel.binary(a, b, bop)
                }
                Expr::Stack { exprs } => {
                    let ops: Vec<OpId> = exprs.iter().map(|e| op_map[e]).collect();
                    self.kernels[kid].kernel.stack(&ops)
                }
                Expr::Stack2 { exprs } => {
                    let ops: Vec<OpId> = exprs.iter().map(|e| op_map[e]).collect();
                    self.kernels[kid].kernel.stack(&ops)
                }
                Expr::Stack3 { exprs } => {
                    let ops: Vec<OpId> = exprs.iter().map(|e| op_map[e]).collect();
                    self.kernels[kid].kernel.stack(&ops)
                }
                Expr::Stack4 { exprs } => {
                    let ops: Vec<OpId> = exprs.iter().map(|e| op_map[e]).collect();
                    self.kernels[kid].kernel.stack(&ops)
                }
                Expr::Stack5 { exprs } => {
                    let ops: Vec<OpId> = exprs.iter().map(|e| op_map[e]).collect();
                    self.kernels[kid].kernel.stack(&ops)
                }
            };
            op_map.insert(eid, op_id);
            root_op = op_id;
        }
        root_op
    }

    /// Lower an interned symbolic expression into kernel ops (the `ExprId`
    /// entry point; `replay_symbolic_into_kernel` is the `TensorId` wrapper).
    /// Walk duplicated from `replay_symbolic_into_kernel` by design.
    pub fn replay_expr(&mut self, kid: KernelId, root: ExprId) -> OpId {
        if root.is_null() {
            return OpId::NULL;
        }

        fn flatten(rt: &Runtime, x: ExprId, order: &mut Vec<ExprId>) {
            if order.contains(&x) {
                return;
            }
            match &rt.exprs[x] {
                Expr::Constant { .. } | Expr::Variable { .. } => (),
                Expr::Cast { x: a, .. } => flatten(rt, *a, order),
                Expr::Unary { x: a, .. } => flatten(rt, *a, order),
                Expr::Binary { x: a, y: b, .. } => {
                    flatten(rt, *a, order);
                    flatten(rt, *b, order);
                }
                Expr::Stack { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
                Expr::Stack2 { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
                Expr::Stack3 { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
                Expr::Stack4 { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
                Expr::Stack5 { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
            }
            order.push(x);
        }
        let mut order = Vec::new();
        flatten(self, root, &mut order);

        let mut op_map: Map<ExprId, OpId> = Map::with_hasher(BuildHasherDefault::new());
        let mut root_op = OpId::NULL;
        for eid in order {
            let node = self.exprs[eid].clone();
            let op_id = match node {
                Expr::Constant { value } => self.kernels[kid].kernel.push_back(Op::Const(value)),
                Expr::Variable { value } => {
                    let op_id = self.kernels[kid].kernel.variable(value.dtype());
                    let tid = self.tensors.push(TensorData::Symbolic { expr: eid, rc: 1 });
                    self.kernels[kid].loads.push(tid);
                    op_id
                }
                Expr::Cast { x, dtype } => {
                    let a = op_map[&x];
                    self.kernels[kid].kernel.cast(a, dtype)
                }
                Expr::Unary { x, uop } => {
                    let a = op_map[&x];
                    self.kernels[kid].kernel.unary(a, uop)
                }
                Expr::Binary { x, y, bop } => {
                    let (a, b) = (op_map[&x], op_map[&y]);
                    self.kernels[kid].kernel.binary(a, b, bop)
                }
                Expr::Stack { exprs } => {
                    let ops: Vec<OpId> = exprs.iter().map(|e| op_map[e]).collect();
                    self.kernels[kid].kernel.stack(&ops)
                }
                Expr::Stack2 { exprs } => {
                    let ops: Vec<OpId> = exprs.iter().map(|e| op_map[e]).collect();
                    self.kernels[kid].kernel.stack(&ops)
                }
                Expr::Stack3 { exprs } => {
                    let ops: Vec<OpId> = exprs.iter().map(|e| op_map[e]).collect();
                    self.kernels[kid].kernel.stack(&ops)
                }
                Expr::Stack4 { exprs } => {
                    let ops: Vec<OpId> = exprs.iter().map(|e| op_map[e]).collect();
                    self.kernels[kid].kernel.stack(&ops)
                }
                Expr::Stack5 { exprs } => {
                    let ops: Vec<OpId> = exprs.iter().map(|e| op_map[e]).collect();
                    self.kernels[kid].kernel.stack(&ops)
                }
            };
            op_map.insert(eid, op_id);
            root_op = op_id;
        }
        root_op
    }

    /// Lower a symbolic scalar expression (a tensors-slab tree: Constant /
    /// Variable / Unary / Binary / Stack) into graph nodes and return its
    /// root class. Constants become `Const` classes, variables become
    /// `IDX_T` leaves — the same lowering `promote_to_graph` uses for dim
    /// expressions.
    /// Replay a symbolic scalar/shape expression from the tensors slab into
    /// egraph classes, returning its root class.
    ///
    /// Middle stage of the symbolic-shapes pipeline (see
    /// [`Runtime::replay_symbolic_into_kernel`] for the full contract):
    /// - `Constant` → `Const` class (merged by value — see [`Node::Const`]).
    /// - `Variable` → a fresh `IDX_T` **dim-variable leaf**: a `Node::Leaf`
    ///   with `shape == NULL`. Leaves hashcons but never merge (fresh
    ///   `cons_id` every time), so the same logical variable appearing under
    ///   two tensors' shapes yields two distinct classes. This duplication is
    ///   deliberate for now: classes carry identity, not value identity, and
    ///   execution-time binding resolves through `variable_map` outside the
    ///   egraph entirely. TensorIds must NOT enter the egraph — that would
    ///   poison graph hashing and cross-replay plan caching.
    /// - `Cast` / `Unary` / `Binary` → corresponding nodes over operand
    ///   classes (hashconsed normally).
    /// - `Stack` / `Stack2`–`Stack5` → a `Stack` node (or folded away for
    ///   len < 2).
    ///
    /// The same closed-set rule applies: anything else panics here.
    pub(crate) fn replay_symbolic_into_graph(&mut self, graph_id: GraphId, shape: TensorId) -> ClassId {
        // DFS post-order flatten: every node lands after its operands.
        fn flatten(rt: &Runtime, x: ExprId, order: &mut Vec<ExprId>) {
            if order.contains(&x) {
                return;
            }
            match &rt.exprs[x] {
                Expr::Constant { .. } | Expr::Variable { .. } => (),
                Expr::Cast { x: a, .. } => flatten(rt, *a, order),
                Expr::Unary { x: a, .. } => flatten(rt, *a, order),
                Expr::Binary { x: a, y: b, .. } => {
                    flatten(rt, *a, order);
                    flatten(rt, *b, order);
                }
                Expr::Stack { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
                Expr::Stack2 { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
                Expr::Stack3 { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
                Expr::Stack4 { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
                Expr::Stack5 { exprs } => {
                    for &e in exprs.iter() {
                        flatten(rt, e, order);
                    }
                }
            }
            order.push(x);
        }
        let root = match self.tensors[shape] {
            TensorData::Symbolic { expr, .. } => expr,
            ref t => panic!("replay_symbolic_into_graph: shape tid {shape} is not symbolic: {t:?}"),
        };
        let mut order = Vec::new();
        flatten(self, root, &mut order);

        let mut class_map: Map<ExprId, ClassId> = Map::with_hasher(BuildHasherDefault::new());
        let mut root_class = ClassId::NULL;
        for eid in order {
            let class_id = match self.exprs[eid].clone() {
                Expr::Constant { value } => self.push_const(graph_id, value),
                Expr::Variable { .. } => {
                    // A variable in a shape expression is an input, not
                    // structure: register its leaf so the plan binds it via
                    // the tensors slab and value changes never force
                    // recompilation.
                    let (_, cid) = self.push_leaf_node(graph_id, IDX_T, ClassId::NULL);
                    let var_tid = self.tensors.push(TensorData::Symbolic { expr: eid, rc: 1 });
                    self.graphs[graph_id].leaf_map.insert(cid, var_tid);
                    self.retain(var_tid);
                    self.graphs[graph_id].leaf_classes.push(cid);
                    self.graphs[graph_id].ref_count += 1;
                    cid
                }
                Expr::Cast { x, dtype } => {
                    let a = class_map[&x];
                    self.push_node(graph_id, Node::Cast { x: a, dtype }).1
                }
                Expr::Unary { x, uop } => {
                    let a = class_map[&x];
                    self.push_node(graph_id, Node::Unary { x: a, uop }).1
                }
                Expr::Binary { x, y, bop } => {
                    let a = class_map[&x];
                    let b = class_map[&y];
                    self.push_binary_node(graph_id, a, b, bop)
                }
                Expr::Stack { ref exprs } => {
                    let ops: Vec<ClassId> = exprs.iter().map(|e| class_map[e]).collect();
                    match ops.len() {
                        0 => ClassId::NULL,
                        1 => ops[0],
                        _ => self.push_node(graph_id, Node::Stack { ops: ops.into_boxed_slice() }).1,
                    }
                }
                Expr::Stack2 { ref exprs } => {
                    let ops: Vec<ClassId> = exprs.iter().map(|e| class_map[e]).collect();
                    match ops.len() {
                        0 => ClassId::NULL,
                        1 => ops[0],
                        _ => self.push_node(graph_id, Node::Stack { ops: ops.into_boxed_slice() }).1,
                    }
                }
                Expr::Stack3 { ref exprs } => {
                    let ops: Vec<ClassId> = exprs.iter().map(|e| class_map[e]).collect();
                    match ops.len() {
                        0 => ClassId::NULL,
                        1 => ops[0],
                        _ => self.push_node(graph_id, Node::Stack { ops: ops.into_boxed_slice() }).1,
                    }
                }
                Expr::Stack4 { ref exprs } => {
                    let ops: Vec<ClassId> = exprs.iter().map(|e| class_map[e]).collect();
                    match ops.len() {
                        0 => ClassId::NULL,
                        1 => ops[0],
                        _ => self.push_node(graph_id, Node::Stack { ops: ops.into_boxed_slice() }).1,
                    }
                }
                Expr::Stack5 { ref exprs } => {
                    let ops: Vec<ClassId> = exprs.iter().map(|e| class_map[e]).collect();
                    match ops.len() {
                        0 => ClassId::NULL,
                        1 => ops[0],
                        _ => self.push_node(graph_id, Node::Stack { ops: ops.into_boxed_slice() }).1,
                    }
                }
            };
            class_map.insert(eid, class_id);
            root_class = class_id;
        }
        root_class
    }
}
