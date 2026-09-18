// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Runtime: the eager tensor executor.
//!
//! The `Runtime` holds the per-process tensor slab, kernel pool, variable slots, and the
//! graph state for the active tape. Most callers go through [`Tensor`], which acquires the
//! process-wide runtime via a `Mutex` (`RT.lock()`).
//!
//! # Concurrency and the runtime lock
//!
//! The runtime mutex (`src/mutex.rs`) is a hand-rolled spinlock that is **not reentrant**.
//! Deadlock prevention lives in the lock itself (a bounded spin with a `debug_assert!` that
//! fires when the spin count exceeds the configured limit instead of hanging forever).
//!
//! Two lock-related footguns recur; both are avoidable by construction:
//!
//! 1. **Drop-order trap.** An assignment like
//!    `n = Tensor { id: RT.lock().binary(n.id, ...) };`
//!    evaluates the RHS first — the `MutexGuard` temporary is still alive when the
//!    assignment then drops the **old** `n`. `Tensor::drop` calls `RT.lock().release`, and
//!    the still-held guard deadlocks on its own lock. Always bind the result of
//!    `RT.lock()` to a `let` first, so the guard drops at the end of that statement
//!    before the assignment drops the old handle:
//!    `let id = RT.lock().binary(...); n = Tensor { id };`.
//!
//! 2. **Held-guard / nested-call trap.** Never call a method that takes `RT.lock()`
//!    (e.g. `Tensor::symbolic_shape`, `Tensor::shape`, `Tensor::stack`, `Tensor::expand`,
//!    `Tensor::reshape`) while already holding the guard. Scope every `let rt = RT.lock()`
//!    tightly so the guard is released before the next call.
//!
//! # The reference counting model
//!
//! Every tensor's liveness is a plain refcount (`rc`) on its slab entry, and **every edge
//! to a tensor counts**. There are exactly four edge kinds; each increments the target's
//! `rc` when created and decrements it exactly once when the edge dies:
//!
//! 1. **User handle.** The caller's `Tensor` value. Created by tensor construction /
//!    `Clone`, released by `Tensor::drop` → [`Runtime::release`].
//! 2. **Symbolic tensor edges.** The pure-slab symbolic nodes reference each other:
//!    `Cast.x`, `Unary.x`, `Binary { x, y }`, `Stack.tensors`, `Stack2–5.tensors`. Each held child is one
//!    edge: retained when the node is created (see `stack` / `binary` symbolic arms),
//!    released when the node dies ([`Runtime::release`]'s death path).
//! 3. **Kernel load edges.** `KernelData.loads` lists the tensors a kernel reads; each
//!    entry is one edge counted in the loaded tensor's `rc`. A tensor may appear more
//!    than once in the same kernel's `loads` (one entry per read occurrence) and in
//!    several kernels at once. Edges are created when the read is introduced (kernel
//!    build, fusion, symbolic replay) and released when the reader's interest ends —
//!    at op-chain prune ([`Runtime::release`] death path) or at kernel death, when the
//!    kernel launches (`materialize_kernel`) and has consumed its loads. Moving ops
//!    between kernels (`merge_kernel`) moves the load entries with them — an ownership
//!    transfer, **no rc change**.
//! 4. **Graph leaf edges.** A leaf promoted into a tape's graph is retained by the tape
//!    (`promote_to_graph`); the tape releases the edge when it dies.
//!
//! `shape_id` is also an edge: a kernel-backed tensor holds an edge to the symbolic
//! expression describing its shape, released on the tensor's death.
//!
//! **Fresh stacks own their result.** `self.stack(&dims)` returns a new `shape_id` with
//! `rc = 1` which is the result's ownership. Callers must **not** `retain` a freshly built
//! stack — that double-counts and leaks. Retain only when *sharing* an existing shape_id
//! (e.g. `flip`, `bitcast`, eager `cast`, eager `binary`).
//!
//! A tensor's `rc` is therefore: user handles + symbolic children it is referenced by +
//! kernel load entries + (for graph leaves) the tape's edge. Every variant dies purely on
//! `rc == 0`; nothing else keeps a kernel-backed tensor alive.
//!
//! # Kernel / tensor lifecycle
//!
//! - A kernel's `outputs` set tracks what the kernel still **owes** (unrealized results a
//!   user may still be waiting for). `outputs` membership is **not** an edge and does not
//!   keep the tensor alive. `stores` lists already-realized outputs.
//! - A kernel-backed tensor is born with `rc = 2`: the user's handle plus its producer
//!   kernel's load edge (`KernelData.loads` contains the tensor itself — the kernel may
//!   read what it writes). Exactly one kernel holds it: its `kernel_id`, which lists the
//!   tensor in `outputs` and/or `loads`.
//! - **Disown (Eager only).** When a user handle drops and the tensor's remaining `rc`
//!   is exactly its producer kernel's load entries, but the kernel owes pending stores
//!   (or other outputs), the tensor is disowned: removed from the kernel's `outputs`, but
//!   kept alive (rc unchanged) as input for the kernel's still-pending computations.
//!   Its buffer is still needed — pending store chains read it. The disowned state is
//!   load-affiliated (`loads` contains it) but not outputs-affiliated; `eagerify` and
//!   eager-kernel promotion must both tolerate that. For a **Promoted** tensor the same
//!   rc condition means death instead (see below): a graph tensor with no user handle
//!   cannot be resurrected eagerly, so lingering disowned is pointless.
//! - **Breaker.** When a handle drops, the remaining rc is exactly the producer kernel's
//!   load entries, the kernel's `outputs` is exactly `{x}` **and the kernel owes no
//!   pending stores** — the kernel's only remaining purpose was producing x, so the
//!   tensor↔kernel cycle is real and nothing else can ever reference the tensor — the
//!   cycle is broken: x dies and takes the (now purposeless) kernel with it. Without this
//!   the pair would deadlock: x kept alive by its kernel's load edge, the kernel kept
//!   alive by x being its last output. The pending-stores constraint is essential: store
//!   chains *consume* x's load, so x's buffer must survive until the stores launch. A
//!   kernel with stores but outputs `{x}` falls through to disown instead.
//! - **Death (rc == 0).** The tensor detaches from its producer kernel: removed from
//!   `outputs`, its now-unreferenced op chain pruned, the pruned ops' load edges released.
//!   If the kernel's `outputs` are then empty the kernel is materialized (launching its
//!   pending stores), and if both `outputs` and `stores` end up empty the kernel is
//!   dropped and its remaining load edges are released, which recursively kills tensors
//!   nothing else reads (including pending producers). Then: free the buffer, remove
//!   the slab entry, release the `shape_id` edge.
//! - **Promoted death = full death, never disown.** A Promoted tensor whose user handle
//!   drops while `rc` is exactly its producer kernel's load entries dies immediately. The
//!   graph is still alive at that point, but no user handle will ever return; the death
//!   path detaches, prunes, and materializes the producer kernel if its stores are still
//!   owed with no outputs left. Keeping it "disowned" would leave a tensor that is in
//!   neither `outputs` nor owned by any handle — a state later classification (e.g.
//!   `Tape::drop`'s visit loop) cannot reason about.
//! - **Graph tensors may hold a buffer — one exception.** A `Graph` tensor is normally
//!   unrealized (its value lives only in the graph). The exception: a **disowned eager
//!   tensor promoted as a graph leaf** (`promote_to_graph`'s kernel-replay load branch).
//!   The promotion materializes its producing kernel, so the buffer exists and the graph
//!   reads it as a leaf input — but the user handle is gone, so the eager side
//!   (`kernel_id`/`op_id`) is dropped and the tensor becomes pure `Graph`. Its buffer is
//!   freed by the `Graph` death path: no user can ever use it eagerly after the tape
//!   dies, so the buffer's lifetime is exactly the tensor's.
//! - **Kernel death tolerates `kernel_id == NULL`.** Before a kernel is dropped or
//!   materialized, every load tensor whose `kernel_id` points at it has that field
//!   nulled: the kernel is gone, its edges are being released, and a later death must not
//!   dereference the dead kernel. (This also matters because `materialize_kernel` uses
//!   `remove_and_return` — a `swap_remove` — so `kernel_id` values pointing at the
//!   removed kernel would silently alias a different kernel afterwards.)
//! - `depends_on` is never released directly: it dies through the same recursion when
//!   the loading kernel's loads are released.
//!
//! # Graph affiliation ledger
//!
//! `Graph::ref_count` counts live affiliations: one increment per tensor that gains a
//! graph identity (every new graph node tensor, plus each `leaf_map` occurrence), one
//! decrement per destroyed affiliation. Each `leaf_map` occurrence is an edge: created →
//! `retain` + increment, destroyed → decrement. The decrement happens exactly once per
//! affiliation, owned by whichever step deletes it:
//!
//! - `eagerify` (including `Tape::realize`'s output conversion): the tensor stops
//!   pointing at the graph — decrement.
//! - `Tape::drop`'s visit loop: for tensors still `Graph`/`Promoted` at collection, plus
//!   tensors re-homed to `Eager` by add_store cascades **mid-drop** (the `Eager` arm).
//!   Leafs that were **already `Eager` at collection** were eagerified during `realize`
//!   (an output that is also a leaf) — their edge was deleted then, so the visit loop
//!   must skip their decrement (`eager_leafs` in `Tape::drop`).
//! - death paths: a dying tensor with a graph affiliation decrements before the graph
//!   teardown asserts the inventory (`assert_graph_inventory`: `ref_count` == number of
//!   live tensors pointing at the graph).
//!
//! # Recursive materialization
//!
//! Materialization is driven by `add_store`, never called directly on a kernel with
//! unstored outputs (see `materialize_kernel`'s convention note):
//!
//! 1. `add_store(x)` removes x from its producer's `outputs`, appends a `GlobalMut`
//!    store op to that kernel, and re-homes x onto a **fresh load kernel** (a `Global`
//!    param reading the stored buffer; x becomes that kernel's only output and carries
//!    its birth load edge). One `retain(x)` pays for the new kernel's edge.
//! 2. When the producer's `outputs` become empty, `materialize_kernel` runs:
//!    a dtype pre-pass over `loads ∪ stores` (resolved while every involved kernel is
//!    still in the slab), the kernel is removed from the slab, then the **recursive
//!    phase**: every load's `depends_on` producer is materialized the same way
//!    (`add_store` on all of its outputs) so the load buffers exist before launch.
//!    Depth is bounded by the `depends_on` chain length, which is acyclic by
//!    construction: each pending load's `depends_on` points at the kernel
//!    whose `stores` produce it — following it strictly descends toward
//!    already-realized buffers, so a cycle would mean a kernel needs its
//!    own result as input.
//! 3. After the launch, the kernel's load edges are released. That release cascade can
//!    kill tensors, which kills their kernels, which may themselves still have pending
//!    stores and so materialize recursively — the release recursion below.
//!
//! During the recursive phase an already-removed (consumer) kernel's load may still be
//! alive and needed by the *next* materialization (as a store of the producer being
//! materialized). This is why tensor metadata must never be derived from the kernel
//! slab: `dtype` lives in `TensorData` precisely so those queries stay valid while the
//! producing kernel is already consumed.
//!
//! # Recursive release
//!
//! `release(x)` decrements x's `rc`; the interesting work happens at `rc == 0` (the
//! death path), which recurses through three mechanisms:
//!
//! 1. **Symbolic edges.** A dead `Cast`/`Unary`/`Binary`/`Stack` (any arity) releases its operand
//!    tensors; a dead kernel-backed tensor releases its `shape_id` expression tree.
//!    Each released operand may itself hit `rc == 0` and recurse.
//! 2. **Kernel detachment.** The death path removes x from its producer's `outputs`,
//!    prunes x's now-unreferenced op chain from the kernel IR, and releases the load
//!    edges the prune dropped. If `outputs` and `stores` are then empty the kernel is
//!    dropped and *all* its remaining load edges are released — recursively killing
//!    tensors that nothing else reads (their `depends_on` producers die the same way).
//!    If `stores` remain (the kernel's results are still wanted), the death path
//!    materializes the kernel instead, chaining into the materialization recursion
//!    above.
//! 3. **Materialization releases.** `materialize_kernel` releases the consumed load
//!    edges after its launch, feeding back into mechanism 1/2.
//!
//! Termination: every recursive step strictly decreases a well-founded measure — the
//! total rc sum (edges), the kernel count (drops), or moves a kernel from "pending" to
//! "materialized" (each kernel materializes at most once; results are cached in
//! `kernel_map`/`programs`). Nothing in the cascade can resurrect an edge, so the
//! recursion is finite.
//!
//! # Invariants carried into the kernelizer
//!
//! The graph-side kernelizer mirrors these eager contracts (see
//! `graph::kernelize` for the full list and the shape-replay rule):
//!
//! - `duplicate_or_store` always returns a **store-free, outputs-empty** kernel — this is
//!   why `narrow`'s "input must have empty outputs" assertion holds unconditionally.
//! - `narrow` requires its input kernel's `outputs` to be empty (no other pending outputs).
//! - `assign` requires its `dst` kernel to be movement-only with no other outputs and no
//!   stores, and removes the kernel after the in-place store. Shape equality is proved
//!   per dim (same const, or the same symbolic dim tensor).
//! - `merge_kernel` requires the merge kernel to be store-free (callers must `add_store`
//!   first if it isn't).

// ----- ASYNC RULES -----
//
// Launches and cross-pool copies are fire-and-forget: the backend retains
// every buffer a submission touches and releases it once the work completes.
// pool_to_host is the sync point — it flushes all pending work first, so a
// value read back is always up to date.
// -----------------------------

use std::{collections::BTreeSet, hash::BuildHasherDefault, path::Path};

#[cfg(feature = "viz")]
use crate::viz::Viz;
use crate::{
    DType, Dev, Map, Scalar, Set, ZyxError,
    backend::{Buffer, DTypeCapability, DeviceProgramId, LaunchArg, Pool, ProgramId},
    dtype::Constant,
    graph::{ClassId, ExecPlan, Graph, GraphId, Node},
    kernel::{BOp, Kernel, MoveOp, Op, OpId, ParamKind, UOp},
    rng::Rng,
    scalar::{bf16, f8e4m3, f8e5m2, f16},
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    symbolic::{Expr, ExprId},
    tensor::TensorId,
};

/// Loads present in `old` but not in `new`, counting multiplicities.
pub fn loads_dropped_by_prune(old: &[TensorId], new: &[TensorId]) -> Vec<TensorId> {
    let mut dropped = Vec::new();
    let mut seen: Set<TensorId> = Set::default();
    for &tid in old {
        if !seen.insert(tid) {
            continue;
        }
        let old_c = old.iter().filter(|&&t| t == tid).count();
        let new_c = new.iter().filter(|&&t| t == tid).count();
        dropped.extend(std::iter::repeat_n(tid, old_c - new_c));
    }
    dropped
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct KernelId(u16);

impl From<usize> for KernelId {
    fn from(value: usize) -> Self {
        KernelId(value as u16)
    }
}

impl From<KernelId> for usize {
    fn from(value: KernelId) -> Self {
        value.0 as usize
    }
}

impl SlabId for KernelId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u16::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

/// A dimension resolved for merge-compatibility checking (see
/// [`Runtime::resolve_shape_without_variables`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResolvedDim {
    /// Concrete dimension: the expression contains no variables, so its
    /// value is a compile-time constant.
    Static(Dim),
    /// Symbolic dimension: identified by the ROOT dim expression.
    /// Two dims are provably equal ONLY if they are structurally identical
    /// (same [`ExprId`] — hashconsing makes this canonical): different
    /// expressions over equal-valued variables are NOT proof (a variable
    /// node is immutable, but a later step may use a different node).
    Symbolic(ExprId),
}

#[derive(Debug)]
pub enum TensorData {
    Leaf {
        shape_id: ExprId,
        dtype: DType,
        buffer: Buffer,
        rc: u16,
    },
    PendingLeaf {
        old_buffer: Option<Buffer>,
        depends_on: KernelId,
        shape_id: ExprId,
        dtype: DType,
        rc: u16,
    },
    GraphLeaf {
        class_id: ClassId,
        graph_id: GraphId,
        shape_id: ExprId,
        dtype: DType,
        rc: u16,
        buffer: Buffer,
    },
    // Eager only
    //
    /// # Field semantics
    ///
    /// - `kernel_id`: the kernel that holds this tensor in its `outputs` (or
    ///   `stores`, once stored). Exactly one kernel lists it — see
    ///   [`KernelData`] for the inventory invariants.
    Eager {
        kernel_id: KernelId,
        op_id: OpId,
        shape_id: ExprId,
        dtype: DType,
        rc: u16,
    },
    // Graph only
    //
    // Normally unrealized (the value lives only in the graph). The one
    // exception: a disowned eager tensor promoted as a graph leaf keeps its
    // materialized buffer, and no user handle ever returns — so the buffer is
    // freed by this variant's death path. See the module docs
    // ("Graph tensors may hold a buffer — one exception").
    Graph {
        class_id: ClassId,
        graph_id: GraphId,
        shape_id: ExprId,
        dtype: DType,
        rc: u16,
    },
    /// So that when graph is dropped, eager version remains.
    ///
    /// A promoted tensor has NO `depends_on` by construction: promotion
    /// realizes the tensor's producing kernel immediately, so every load of
    /// `kernel_id` is already realized — present in `buffer_map` or
    /// `variable_map`. No pending producer can remain, hence nothing to
    /// record in `depends_on`.
    Promoted {
        kernel_id: KernelId,
        op_id: OpId,
        class_id: ClassId,
        graph_id: GraphId,
        shape_id: ExprId,
        dtype: DType,
        rc: u16,
    },
    /// Symbolic scalar, dim, or shape: an interned [`Expr`]. Replaces the old
    /// Constant/Variable/Cast/Unary/Binary/Stack* tensor variants — shapes no
    /// longer live in the tensors slab. The expression slab is append-only,
    /// so a `Symbolic` holds no edges: retain/release touch only its own
    /// `rc`, never expression children.
    Symbolic {
        /// The interned expression (stable forever).
        expr: ExprId,
        /// User handles + kernel load entries referencing this handle.
        rc: u16,
    },
}

#[derive(Debug)]
pub struct KernelData {
    /// Tensors this kernel must produce.
    ///
    /// # Fields
    ///
    /// - `outputs`: the set of tensors the kernel **produces** and which can
    ///   still take part in further fusion. They are NOT realized (no
    ///   buffer), nor pending realization — they are simply produced here.
    /// - `stores`: tensors whose StoreView the kernel holds. They are
    ///   **finished**: materializing the kernel must realize every one of
    ///   them (allocate their buffers), because other kernels already use
    ///   them as loads. A `GlobalMut` define's buffer slot is carried here,
    ///   NOT in `loads`: an in-place assign turns dst from a load into a pure
    ///   store, so the target must not appear in `loads`.
    /// - `loads`: tensors this kernel reads, aligned to its non-store
    ///   `Param` defines (`Global` buffers and scalar `Variable` dim params)
    ///   in head order. Launch args bind positionally: load slots first
    ///   (`loads`), then store slots (`stores`) — never shuffled.
    ///
    /// # Fusion break (`add_store`)
    ///
    /// A tensor is moved from `outputs` to `stores` when fusion breaks. In
    /// that step the tensor is re-homed to `LeafPending` and its `depends_on`
    /// points at the kernel whose `stores` hold it. Moving such a kernel
    /// around afterwards — including merging it into later kernels — is legal
    /// and DESIRED: it is the core fusion principle of the eager fusion machinery.
    ///
    /// # Inventory invariants
    ///
    /// - A tensor appears in **exactly one** kernel's `outputs` — precisely
    ///   the kernel equal to its `kernel_id` (`Eager`, `Promoted`). Listing a
    ///   live tensor in several kernels' `outputs` is ILLEGAL. A tensor in
    ///   more than one `outputs` (or in an `outputs` of a kernel other than
    ///   its `kernel_id`) is an inventory desync bug.
    /// - [`Runtime::release`] removes a dying tensor from the `outputs` OR
    ///   `stores` of its `kernel_id`. Afterwards: if both `stores` and
    ///   `outputs` are empty the kernel is dropped; if `outputs` is empty but
    ///   `stores` are not, the kernel is materialized immediately (its stores
    ///   must still run to feed consumers).
    /// - Edge refcounting: every reference to a tensor is an edge — the
    ///   caller's handle or one load in some kernel. A kernel's lifetime is
    ///   carried by its loads: when a kernel dies, its `loads` are released,
    ///   which recursively tears down whatever those loads depended on.
    ///   `depends_on` is never released directly.
    /// - Load/store correspondence: the tensors that appear as `loads` of a
    ///   kernel K are stored by the `stores` of K's producer — for a pending
    ///   leaf that producer is recorded in `TensorData::LeafPending::depends_on`.
    /// - `merge_kernel(keep, merge)` repoints every tensor whose `kernel_id`
    ///   was `merge` to `keep`: after the merge each such tensor is in
    ///   `keep.outputs` and NOWHERE else.
    pub outputs: Set<TensorId>,
    pub loads: Vec<TensorId>,
    pub stores: Vec<TensorId>,
    pub kernel: Kernel,
}

pub struct Runtime {
    pub graphs: Slab<GraphId, Graph>,
    pub tensors: Slab<TensorId, TensorData>,
    pub kernels: Slab<KernelId, KernelData>,
    kernel_map: Map<Kernel, KernelId>,
    programs: Map<KernelId, DeviceProgramId>,
    timings: Map<ProgramId, u64>,
    /// Hashcons table for [`Expr`]: structural equality is one [`ExprId`] compare.
    pub(crate) expr_hash: Map<Expr, ExprId>,
    /// Append-only interned expressions. Ids are stable forever; nothing is freed.
    pub exprs: Slab<ExprId, Expr>,
    pub rng: Rng,
    pub implicit_casts: bool,
    pub training: bool,
    pub plan_cache: Map<u64, ExecPlan>,
    #[cfg(feature = "viz")]
    pub viz: Viz,
}

impl Runtime {
    /// Cache key for the plan cache: the graph's content key (structure +
    /// outputs) folded together with the pool each leaf class's buffer lives
    /// in at call time. The compiled plan bakes pool-dependent bindings
    /// (`ExecPlan::leaf_pools`, cross-pool alias handling), so two realizations
    /// of the same graph shape may only share a plan when the leaf pool layout
    /// matches; otherwise the plan recompiles.
    pub(crate) fn plan_cache_key(&self, graph_id: GraphId, outputs: &BTreeSet<ClassId>) -> u64 {
        use std::hash::{Hash, Hasher};
        let graph = &self.graphs[graph_id];
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        graph.cache_key(outputs).hash(&mut hasher);
        for &cid in &graph.leaf_classes {
            let &tid = graph.leaf_map.get(&cid).unwrap();
            self.leaf_buffer(tid).map(|b| b.pool).hash(&mut hasher);
        }
        hasher.finish()
    }

    pub const fn new() -> Self {
        Runtime {
            graphs: Slab::new(),
            tensors: Slab::new(),
            kernels: Slab::new(),
            kernel_map: Map::with_hasher(BuildHasherDefault::new()),
            programs: Map::with_hasher(BuildHasherDefault::new()),
            timings: Map::with_hasher(BuildHasherDefault::new()),
            expr_hash: Map::with_hasher(BuildHasherDefault::new()),
            exprs: Slab::new(),
            rng: Rng::seed_from_u64(42069),
            implicit_casts: true,
            training: false,
            plan_cache: Map::with_hasher(BuildHasherDefault::new()),
            #[cfg(feature = "viz")]
            viz: Viz::new(),
        }
    }

    pub fn dtype(&self, x: TensorId) -> DType {
        match self.tensors[x] {
            // Dtype lives in the tensor entry itself — never derived from the
            // kernel. A kernel can be materialized (removed) while the tensor
            // is still alive and queried (e.g. a producer kernel materializing
            // needs the dtype of a store whose producing kernel was already
            // consumed). See `materialize_kernel`.
            TensorData::Eager { dtype, .. }
            | TensorData::Promoted { dtype, .. }
            | TensorData::Graph { dtype, .. }
            | TensorData::PendingLeaf { dtype, .. }
            | TensorData::Leaf { dtype, .. }
            | TensorData::GraphLeaf { dtype, .. } => dtype,
            // Symbolic dtype comes from the expression table: walk to the
            // Constant/Variable/Cast leaf that determines it.
            TensorData::Symbolic { expr, .. } => self.expr_dtype(expr),
        }
    }

    /// Realized: the tensor's value is available without further execution —
    /// it is a `Leaf` (backing buffer on the variant), or it resolves to a
    /// constant (a variable/scalar expression bound at launch).
    pub fn is_realized(&self, x: TensorId) -> bool {
        self.leaf_buffer(x).is_some() || self.resolve_symbolic(x).is_some()
    }

    // True if x is currently a graph tensor (class_id set and its graph alive).
    // A promoted non-realized tensor whose graph has died is treated as eager
    // (its kernel_id is still valid), so is_graph returns false in that case.
    pub(crate) fn is_graph(&self, x: TensorId) -> bool {
        match self.tensors[x] {
            TensorData::GraphLeaf { .. } | TensorData::Graph { .. } | TensorData::Promoted { .. } => true,
            TensorData::Eager { .. } | TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } | TensorData::Symbolic { .. } => {
                false
            }
        }
    }

    /// Returns operation capabilities for a dtype across all devices.
    pub fn supports_dtype(&mut self, dtype: DType) -> DTypeCapability {
        let mut caps = DTypeCapability::none();
        for dev in Dev::all() {
            caps = caps.include(dev.info().supports_dtype(dtype));
        }
        caps
    }

    /// Tensor lifetime architecture.
    ///
    /// A tensor lives in one of three states:
    /// - **eager-only** (`class_id` null): the tensor is an output op of its
    ///   producing kernel (`kernel_id`/`op_id`). Its lifetime is tied to its
    ///   refcount (`rc` = handles + kernel loads referencing it).
    /// - **graph-only** (`class_id` set, `kernel_id` null): created directly as
    ///   a graph tensor directly (`TensorData::Graph` push).
    /// - **both** (promoted): an eager tensor that entered a tape scope via
    ///   `promote_to_graph`. It keeps its `kernel_id`/`op_id` AND gains a
    ///   `class_id`. The graph has precedence while alive; the eager kernel is
    ///   left completely untouched (rc/outputs still count this tensor's
    ///   handles). This is what makes graph death seamless:
    ///
    ///   ```text
    ///   y = x.exp()            // eager kernel: Param -> Exp -> y
    ///   promote(y)             // replays the kernel into graph nodes;
    ///                          // y keeps kernel_id, x becomes a leaf
    ///   ... graph dies ...
    ///   ```
    ///
    ///   Without the kept kernel, y would be dead once the graph dies. Instead
    ///   y simply reverts to eager mode with zero issues: its kernel and buffer
    ///   are exactly as they were before promotion.
    ///
    /// Death path (in `release`, matched per `TensorData` variant):
    /// - eager/promoted: detach from the producer kernel's outputs (pruning ops
    ///   no surviving output needs), free the buffer, drop the kernel if it was
    ///   the last live output.
    /// - graph: remove the slab entry; the graph's `ref_count` tracks
    ///   affiliated tensors and tears the graph down when it hits zero.
    /// - constant/variable/unary/binary/stack: free the slot and drop the
    ///   edges to children.
    ///
    /// Invariants:
    /// - while a tensor exists in the slab, every non-null `kernel_id` it holds
    ///   lists that tensor in the kernel's `outputs` **and/or** `loads` (a
    ///   disowned tensor — user handle dropped, kept alive as input — is listed
    ///   in `loads` only);
    /// - a promoted ("both") tensor is removed only through its graph branch,
    ///   which must therefore also restore eager consistency for any surviving
    ///   siblings of its producer kernel.
    pub fn retain(&mut self, x: TensorId) {
        if !x.is_null() {
            match &mut self.tensors[x] {
                TensorData::Eager { rc, .. }
                | TensorData::PendingLeaf { rc, .. }
                | TensorData::Leaf { rc, .. }
                | TensorData::GraphLeaf { rc, .. }
                | TensorData::Graph { rc, .. }
                | TensorData::Promoted { rc, .. }
                | TensorData::Symbolic { rc, .. } => {
                    *rc += 1;
                    #[cfg(feature = "debug_tensor_op")]
                    println!("rc::retain({x}) -> {rc}");
                }
            };
        }
    }

    /// Buffer backing a realized `Leaf`, or a `LeafPending` whose buffer
    /// already exists (assign-pending in-place write) — `None` for
    /// not-yet-allocated pending tensors and non-Leaf tensors.
    pub(crate) fn leaf_buffer(&self, x: TensorId) -> Option<Buffer> {
        match self.tensors[x] {
            TensorData::Leaf { buffer, .. } | TensorData::GraphLeaf { buffer, .. } => Some(buffer),
            _ => None,
        }
    }

    /// Death path: detach from the producer kernel's outputs (pruning ops no
    /// surviving output needs), free the buffer, drop the kernel if it was the
    /// last live output. Graph tensors are kept by their graph instead; its
    /// `ref_count` tears it down when it hits zero.
    pub fn release(&mut self, x: TensorId) {
        #[cfg(feature = "debug_tensor_op")]
        {
            let desc: String = match &self.tensors[x] {
                TensorData::Eager { kernel_id, op_id, .. } => format!("eager kernel={kernel_id:?} op={op_id:?}"),
                TensorData::PendingLeaf { shape_id, .. } => format!("pending shape={shape_id:?}"),
                TensorData::Leaf { shape_id, buffer, .. } => {
                    format!("leaf shape={shape_id:?} buffer={buffer:?}")
                }
                TensorData::GraphLeaf { shape_id, buffer, .. } => {
                    format!("graphleaf shape={shape_id:?} buffer={buffer:?}")
                }
                TensorData::Graph { class_id, graph_id, .. } => format!("graph class={class_id:?} graph={graph_id:?}"),
                TensorData::Promoted { kernel_id, class_id, graph_id, .. } => {
                    format!("promoted kernel={kernel_id:?} class={class_id:?} graph={graph_id:?}")
                }
                TensorData::Symbolic { expr, .. } => format!("symbolic {expr:?}"),
            };
            println!("runtime::release(tid={x}) kind={desc}");
        }

        // Drop one reference. Handles and edges (kernel loads, symbolic-node
        // children) all count through here. No disown/breaker logic: a kernel
        // that both outputs and loads the same tensor is unconstructible (a
        // pure-buffer value is a `Leaf`, which has no kernel), so every
        // variant dies purely on its refcount.
        let rc = {
            match &mut self.tensors[x] {
                TensorData::Promoted { rc, .. }
                | TensorData::Eager { rc, .. }
                | TensorData::Leaf { rc, .. }
                | TensorData::PendingLeaf { rc, .. }
                | TensorData::GraphLeaf { rc, .. }
                | TensorData::Graph { rc, .. }
                | TensorData::Symbolic { rc, .. } => {
                    *rc -= 1;
                    *rc
                }
            }
        };

        #[cfg(feature = "debug_tensor_op")]
        println!("rc::release({x}) -> rc={rc}");
        // A still-positive count means another live reference (handle, kernel
        // load edge) keeps the entry alive.
        if rc != 0 {
            return;
        }

        match self.tensors[x] {
            TensorData::Symbolic { .. } => {
                // Append-only expr slab: no children, no edges. Drop the handle only.
                self.tensors.remove(x);
            }
            TensorData::GraphLeaf { buffer: buffer_id, .. } => {
                // A realized Leaf owns a buffer (or borrows its `view_of`
                // source's buffer). `free_buffer` deallocates owners and
                // releases the source of views.
                buffer_id.pool.release(buffer_id.buffer_id);
                self.tensors.remove(x);
            }
            TensorData::Leaf { buffer: buffer_id, .. } => {
                // A realized Leaf owns a buffer (or borrows its `view_of`
                // source's buffer). `free_buffer` deallocates owners and
                // releases the source of views.
                buffer_id.pool.release(buffer_id.buffer_id);
                self.tensors.remove(x);
            }
            TensorData::PendingLeaf { depends_on, old_buffer, .. } => {
                // A pending Leaf owns no buffer yet, only a pending store in
                // `depends_on`. When it dies, every consumer kernel holding a
                // what brought rc to 0), so the buffer — and any pending
                // store writing into it — must go too.
                // A kept (assign in-place) buffer must never die here: assign
                // force-materializes before returning, so a Some here means
                // the store kernel never launched.
                debug_assert!(
                    old_buffer.is_none(),
                    "release: PendingLeaf {x} dies with a kept buffer — its store kernel never launched"
                );
                if !depends_on.is_null() && self.kernels.contains_id(depends_on) {
                    let mut_params: Vec<OpId> = {
                        let kd = &self.kernels[depends_on];
                        let mut mut_params: Vec<OpId> = Vec::new();
                        let mut i = kd.kernel.head;
                        for _ in 0..100_000 {
                            if i.is_null() {
                                break;
                            }
                            if matches!(kd.kernel.ops[i].op, Op::Param { kind: ParamKind::GlobalMut, .. }) {
                                mut_params.push(i);
                            }
                            i = kd.kernel.next_op(i);
                        }
                        debug_assert_eq!(
                            mut_params.len(),
                            kd.stores.len(),
                            "GlobalMut params and stores vec diverged in {depends_on:?}"
                        );
                        mut_params
                    };
                    let dead_params: Vec<OpId> = {
                        let kd = &self.kernels[depends_on];
                        mut_params.iter().enumerate().filter(|(idx, _)| kd.stores[*idx] == x).map(|(_, op)| *op).collect()
                    };
                    if !dead_params.is_empty() {
                        // Keep-alive for the chain prune: the value roots of
                        // the surviving stores and every op owned by a live
                        // tensor affiliated with this kernel.
                        let mut keep_alive: Vec<OpId> = {
                            let kd = &self.kernels[depends_on];
                            mut_params
                                .iter()
                                .enumerate()
                                .filter(|&(idx, _)| kd.stores[idx] != x)
                                .flat_map(|(_, param)| {
                                    let mut stores_to_param: Vec<OpId> = Vec::new();
                                    let mut i = kd.kernel.head;
                                    for _ in 0..100_000 {
                                        if i.is_null() {
                                            break;
                                        }
                                        if let Op::Store { dst, .. } = kd.kernel.ops[i].op {
                                            if dst == *param {
                                                stores_to_param.push(i);
                                            }
                                        }
                                        i = kd.kernel.next_op(i);
                                    }
                                    debug_assert!(!stores_to_param.is_empty(), "store entry without store op in {depends_on:?}");
                                    stores_to_param
                                })
                                .collect()
                        };
                        {
                            let kd = &self.kernels[depends_on];
                            for &tid in kd.outputs.iter().chain(kd.loads.iter()) {
                                if let TensorData::Eager { op_id, .. } | TensorData::Promoted { op_id, .. } = self.tensors[tid] {
                                    if kd.kernel.ops.contains_id(op_id) {
                                        keep_alive.push(op_id);
                                    }
                                }
                            }
                        }
                        let mut loads = self.kernels[depends_on].loads.clone();
                        for &param in &dead_params {
                            // Delete every store op writing to this param, then
                            // prune each one's now-unreachable value chain.
                            while let Some((store_op, src)) = {
                                let kd = &self.kernels[depends_on];
                                let mut found = None;
                                let mut i = kd.kernel.head;
                                for _ in 0..100_000 {
                                    if i.is_null() {
                                        break;
                                    }
                                    if let Op::Store { dst, src, .. } = kd.kernel.ops[i].op {
                                        if dst == param {
                                            found = Some((i, src));
                                            break;
                                        }
                                    }
                                    i = kd.kernel.next_op(i);
                                }
                                found
                            } {
                                self.kernels[depends_on].kernel.remove_op(store_op);
                                loads = self.kernels[depends_on].kernel.remove_unused_chain(src, &keep_alive, &loads);
                            }
                            // The store target param itself is now unused.
                            self.kernels[depends_on].kernel.remove_op(param);
                        }
                        let kd = &mut self.kernels[depends_on];
                        kd.stores.retain(|&t| t != x);
                        kd.loads = loads;
                    }
                }
                self.tensors.remove(x);
            }
            TensorData::Graph { graph_id, .. } => {
                // A graph tensor normally never holds a buffer (pure graph
                // tensors are unrealized by construction). The exception is a
                // disowned tensor promoted as a graph leaf
                // (`promote_to_graph`'s load branch): its buffer was
                // materialized by the promotion and no user handle exists, so
                // the buffer dies with the tensor here.
                self.tensors.remove(x);
                if !graph_id.is_null() {
                    self.graphs[graph_id].ref_count -= 1;
                    if self.graphs[graph_id].ref_count == 0 {
                        self.remove_dead_graph(graph_id);
                    }
                }
            }
            TensorData::Eager { kernel_id, op_id, .. } => {
                // Detach from the producer kernel (inlined; the former
                // `detach_from_kernel`, duplicated per death arm by design).
                if !kernel_id.is_null() {
                    debug_assert!(!op_id.is_null());
                    debug_assert!(!self.kernels[kernel_id].stores.contains(&x));
                    self.kernels[kernel_id].outputs.remove(&x);
                    if self.kernels[kernel_id].outputs.is_empty() && self.kernels[kernel_id].stores.is_empty() {
                        // The kernel dies. Null out the kernel_ids of its loads
                        // first (they point at this kernel): their death paths,
                        // triggered by the load releases below, must not
                        // dereference the dead kernel.
                        for &tid in &self.kernels[kernel_id].loads {
                            if let TensorData::Eager { kernel_id: k, .. } | TensorData::Promoted { kernel_id: k, .. } =
                                &mut self.tensors[tid]
                            {
                                if *k == kernel_id {
                                    *k = KernelId::NULL;
                                }
                            }
                        }
                        let mut loads = std::mem::take(&mut self.kernels[kernel_id].loads);
                        self.kernels.remove(kernel_id);
                        #[cfg(feature = "debug_tensor_op")]
                        eprintln!("KDROP {kernel_id:?} dying={x} all_loads={loads:?}");
                        loads.retain(|&id| id != x);
                        for tid in loads {
                            self.release(tid);
                        }
                    } else {
                        // Keep-alive set for the prune: every op owned by a
                        // live tensor affiliated with this kernel (outputs ∪
                        // loads). A load-affiliated tensor's op can point into
                        // this kernel (e.g. after a merge re-point); pruning it
                        // would orphan the tensor underneath its owner.
                        let (out_ops, loads) = {
                            let kd = &self.kernels[kernel_id];
                            let mut out_ops: Vec<OpId> = kd
                                .outputs
                                .iter()
                                .map(|&tid| match self.tensors[tid] {
                                    TensorData::Eager { op_id, .. } | TensorData::Promoted { op_id, .. } => op_id,
                                    ref t => panic!("kernel output tid {tid} has unexpected tensor data {t:?}"),
                                })
                                .collect();
                            for &tid in &kd.loads {
                                if let TensorData::Eager { op_id, .. } | TensorData::Promoted { op_id, .. } = self.tensors[tid] {
                                    if kd.kernel.ops.contains_id(op_id) {
                                        out_ops.push(op_id);
                                    }
                                }
                            }
                            (out_ops, kd.loads.clone())
                        };
                        let new_loads = self.kernels[kernel_id].kernel.remove_unused_chain(op_id, &out_ops, &loads);
                        let pruned = loads_dropped_by_prune(&loads, &new_loads);
                        for load in pruned {
                            // Skip x itself: its own load entry's count is already
                            // consumed by this death (releasing again would
                            // underflow a zero rc).
                            if load != x {
                                self.release(load);
                            }
                        }
                        self.kernels[kernel_id].loads = new_loads;
                        if self.kernels[kernel_id].outputs.is_empty() {
                            // Materialize, which removes the kernel
                            self.materialize_kernel(kernel_id).expect("materialization in tensor detach from kernel failed");
                        }
                    }
                }
                self.tensors.remove(x);
            }
            TensorData::Promoted { kernel_id, op_id, graph_id, .. } => {
                // Detach from the producer kernel (inlined; the former
                // `detach_from_kernel`, duplicated per death arm by design).
                if !kernel_id.is_null() {
                    debug_assert!(!op_id.is_null());
                    debug_assert!(!self.kernels[kernel_id].stores.contains(&x));
                    self.kernels[kernel_id].outputs.remove(&x);
                    if self.kernels[kernel_id].outputs.is_empty() && self.kernels[kernel_id].stores.is_empty() {
                        // The kernel dies. Null out the kernel_ids of its loads
                        // first (they point at this kernel): their death paths,
                        // triggered by the load releases below, must not
                        // dereference the dead kernel.
                        for &tid in &self.kernels[kernel_id].loads {
                            if let TensorData::Eager { kernel_id: k, .. } | TensorData::Promoted { kernel_id: k, .. } =
                                &mut self.tensors[tid]
                            {
                                if *k == kernel_id {
                                    *k = KernelId::NULL;
                                }
                            }
                        }
                        let mut loads = std::mem::take(&mut self.kernels[kernel_id].loads);
                        self.kernels.remove(kernel_id);
                        #[cfg(feature = "debug_tensor_op")]
                        eprintln!("KDROP {kernel_id:?} dying={x} all_loads={loads:?}");
                        loads.retain(|&id| id != x);
                        for tid in loads {
                            self.release(tid);
                        }
                    } else {
                        // Keep-alive set for the prune: every op owned by a
                        // live tensor affiliated with this kernel (outputs ∪
                        // loads). A load-affiliated tensor's op can point into
                        // this kernel (e.g. after a merge re-point); pruning it
                        // would orphan the tensor underneath its owner.
                        let (out_ops, loads) = {
                            let kd = &self.kernels[kernel_id];
                            let mut out_ops: Vec<OpId> = kd
                                .outputs
                                .iter()
                                .map(|&tid| match self.tensors[tid] {
                                    TensorData::Eager { op_id, .. } | TensorData::Promoted { op_id, .. } => op_id,
                                    ref t => panic!("kernel output tid {tid} has unexpected tensor data {t:?}"),
                                })
                                .collect();
                            for &tid in &kd.loads {
                                if let TensorData::Eager { op_id, .. } | TensorData::Promoted { op_id, .. } = self.tensors[tid] {
                                    if kd.kernel.ops.contains_id(op_id) {
                                        out_ops.push(op_id);
                                    }
                                }
                            }
                            (out_ops, kd.loads.clone())
                        };
                        let new_loads = self.kernels[kernel_id].kernel.remove_unused_chain(op_id, &out_ops, &loads);
                        let pruned = loads_dropped_by_prune(&loads, &new_loads);
                        for load in pruned {
                            // Skip x itself: its own load entry's count is already
                            // consumed by this death (releasing again would
                            // underflow a zero rc).
                            if load != x {
                                self.release(load);
                            }
                        }
                        self.kernels[kernel_id].loads = new_loads;
                        if self.kernels[kernel_id].outputs.is_empty() {
                            // Materialize, which removes the kernel
                            self.materialize_kernel(kernel_id).expect("materialization in tensor detach from kernel failed");
                        }
                    }
                }
                self.tensors.remove(x);
                if !graph_id.is_null() {
                    self.graphs[graph_id].ref_count -= 1;
                    if self.graphs[graph_id].ref_count == 0 {
                        self.remove_dead_graph(graph_id);
                    }
                }
            }
        }
    }

    pub(crate) fn remove_dead_graph(&mut self, graph_id: GraphId) {
        self.graphs.remove(graph_id);
    }

    /// Assert the kernel-affiliation invariants for every tensor (debug builds
    /// only, called at the entry of every tensor op):
    ///
    /// - `Eager`/`Promoted`: non-NULL `kernel_id` (a kernel-backed tensor
    ///   always has a real producer — pure-buffer state is a `Leaf`),
    ///   membership in that kernel's `outputs`/`loads`, live reachable
    ///   `op_id`.
    /// - `Leaf`: either realized (`depends_on` NULL ⇒ buffer in
    ///   `buffer_map`) or pending (`depends_on` live and its `stores` list
    ///   the Leaf).
    /// - Every kernel: `outputs`, `loads` and `stores` are pairwise
    ///   disjoint; a kernel with non-empty `outputs` always ends in a real
    ///   compute op — never a bare `Op::Param` (a pure-load kernel is not an
    ///   operation; that state is a `Leaf`).
    pub(crate) fn verify_tensor_invariants(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        for (tid, td) in self.tensors.iter() {
            match td {
                TensorData::Eager { kernel_id, op_id, .. } | TensorData::Promoted { kernel_id, op_id, .. } => {
                    let (kernel_id, op_id) = (*kernel_id, *op_id);
                    assert!(!kernel_id.is_null(), "verify: kernel-backed tensor {tid} has NULL kernel_id");
                    assert!(self.kernels.contains_id(kernel_id), "verify: tensor {tid} points at deleted kernel {kernel_id:?}");
                    let kd = &self.kernels[kernel_id];
                    assert!(
                        kd.outputs.contains(&tid) || kd.loads.contains(&tid),
                        "verify: tensor {tid} has kernel_id {kernel_id:?} but is neither in its outputs nor loads"
                    );
                    assert!(!op_id.is_null(), "verify: tensor {tid} has NULL op_id with live kernel {kernel_id:?}");
                    assert!(
                        kd.kernel.ops.contains_id(op_id),
                        "verify: tensor {tid} op {op_id:?} is not in kernel {kernel_id:?}'s op slab"
                    );
                    let mut reachable = false;
                    let mut i = kd.kernel.head;
                    for _ in 0..100_000 {
                        if i.is_null() {
                            break;
                        }
                        if i == op_id {
                            reachable = true;
                            break;
                        }
                        i = kd.kernel.next_op(i);
                    }
                    assert!(reachable, "verify: tensor {tid} op {op_id:?} is not reachable from kernel {kernel_id:?}'s op list");
                }
                TensorData::Leaf { .. } => {}
                TensorData::PendingLeaf { depends_on, .. } => {
                    let depends_on = *depends_on;
                    if !depends_on.is_null() {
                        assert!(
                            self.kernels.contains_id(depends_on) && self.kernels[depends_on].stores.contains(&tid),
                            "verify: pending Leaf {tid} points at depends_on {depends_on:?} which does not store it"
                        );
                    }
                }
                _ => {}
            }
        }
        for (kid, kd) in self.kernels.iter() {
            for &tid in &kd.outputs {
                assert!(!kd.loads.contains(&tid), "verify: kernel {kid:?} both outputs and loads tid {tid}");
                assert!(!kd.stores.contains(&tid), "verify: kernel {kid:?} both outputs and stores tid {tid}");
            }
            for &tid in &kd.loads {
                assert!(!kd.stores.contains(&tid), "verify: kernel {kid:?} both loads and stores tid {tid}");
            }
            if !kd.outputs.is_empty() {
                match &kd.kernel.ops[kd.kernel.tail].op {
                    Op::Param { .. } => {
                        panic!("verify: kernel {kid:?} with outputs ends in a bare Param op (pure-load kernel — use a Leaf)");
                    }
                    _ => {}
                }
            }
        }
    }

    /// Assert the graph affiliation invariant: `graph.ref_count` equals the
    /// number of live tensors (rc > 0) whose `graph_id` points at `graph_id`.
    ///
    /// Any desync means an increment/decrement was missed somewhere (promotion,
    /// conversion, eagerify, orphaning, death) — i.e. the graph will either
    /// never be removed (leak) or was torn down early.
    pub fn assert_graph_inventory(&self, graph_id: GraphId) {
        let live = self
            .tensors
            .iter()
            .filter(|(_, td)| match td {
                TensorData::Graph { graph_id: g, rc, .. } | TensorData::Promoted { graph_id: g, rc, .. } if *g == graph_id => {
                    *rc > 0
                }
                _ => false,
            })
            .count();
        assert_eq!(
            live as u64, self.graphs[graph_id].ref_count,
            "graph {graph_id:?} affiliation desync: {live} live affiliated tensors but ref_count = {}",
            self.graphs[graph_id].ref_count
        );
    }

    /// Creates a **LeafPending**: a tensor with no producing kernel and no
    /// buffer yet. The caller transitions it to a realized `Leaf` by storing
    /// its buffer on the variant (or it arrives later via a pending
    /// `depends_on` store). The Leaf carries no kernel and is never listed
    /// in any kernel's `outputs` — consumers mint their own load kernels via
    /// [`Runtime::new_kernel_from_leaf`], which is why no self-referencing "tensor is
    /// its own kernel's load" cycle can exist anymore (the old rc==2
    /// handle+self-load construction is gone).
    pub fn new_eager_tensor(&mut self, shape: TensorId, dtype: DType, buffer_id: Buffer) -> TensorId {
        // Shape handles are Symbolic (or NULL for scalar); the Leaf stores the
        // interned ExprId (append-only slab, no retain needed).
        let shape_id = if shape == TensorId::NULL {
            ExprId::NULL
        } else {
            match self.tensors[shape] {
                TensorData::Symbolic { expr, .. } => expr,
                ref t => panic!("new_eager_tensor: shape tid {shape} is not symbolic: {t:?}"),
            }
        };
        let tid = self.tensors.push(TensorData::Leaf { shape_id, dtype, buffer: buffer_id, rc: 1 });
        #[cfg(feature = "debug_tensor_op")]
        println!("rc::new_eager_tensor -> tid={tid} Leaf shape_id={shape_id} rc=1 (handle only)");
        tid
    }

    /// Fresh kernel with a single Global load `Param` reading realized Leaf
    /// `x`'s buffer. The kernel has empty `outputs` — the caller appends its
    /// compute ops and owns the results. `x` gains one reference for the
    /// kernel's load edge (released when the kernel dies or materializes).
    pub(crate) fn new_kernel_from_leaf(&mut self, x: TensorId) -> (KernelId, OpId) {
        let (shape_id, dtype) = match self.tensors[x] {
            TensorData::Leaf { shape_id, dtype, .. }
            | TensorData::GraphLeaf { shape_id, dtype, .. }
            | TensorData::PendingLeaf { shape_id, dtype, .. } => (shape_id, dtype),
            TensorData::Eager { .. } | TensorData::Graph { .. } | TensorData::Promoted { .. } | TensorData::Symbolic { .. } => {
                unreachable!("new_kernel_from_leaf: {:?}", self.tensors[x])
            }
        };
        // The kernel binds Dev::Auto (unbound placeholder): the launch device
        // is resolved at compile time and materialize moves the buffers to the
        // resolved device's pool. Auto carries no info — without taking the
        // RT lock (not reentrant).
        let kernel_id = self.kernels.push(KernelData {
            outputs: Set::default(),
            loads: Vec::new(),
            stores: Vec::new(),
            kernel: Kernel::from_device_id(Dev::Auto, None),
        });
        let shape = self.replay_expr(kernel_id, shape_id);
        let op_id = self.kernels[kernel_id].kernel.push_back(Op::Param { dtype, kind: ParamKind::Global, shape });
        self.kernels[kernel_id].loads.push(x);
        self.retain(x);
        (kernel_id, op_id)
    }

    pub fn new_constant_tensor(&mut self, value: Constant) -> TensorId {
        // Constants are pure slab entries: value lives in the interned
        // expression slab, no kernel is allocated. Consumers replay the
        // value into their own kernels via Op::Const when needed.
        let expr = self.intern(Expr::Constant { value });
        self.tensors.push(TensorData::Symbolic { expr, rc: 1 })
    }

    pub fn new_full(&mut self, shape: TensorId, value: Constant) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::new_full(shape={shape:?}, value={value:?})");
        let x = self.new_constant_tensor(value);
        if shape.is_null() {
            return x;
        }
        let expanded = self.expand(x, shape).unwrap();
        self.release(x);
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={expanded}, {:?}", self.tensors[expanded]);
        expanded
    }

    pub fn new_variable_tensor<T: Scalar>(&mut self, x: T) -> TensorId {
        // Variables are pure slab entries — the value lives in the
        // interned `Expr::Variable` itself; no kernel, no buffer_map entry.
        // Kernels replay them as Param { Variable } loads
        // (see replay_shape_into_kernel).
        let value = Constant::new(x);
        let expr = self.intern(Expr::Variable { value });
        self.tensors.push(TensorData::Symbolic { expr, rc: 1 })
    }

    // Creates new tensor in host memory
    pub fn new_host_tensor<T: Scalar>(&mut self, shape: TensorId, data: Box<[T]>) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::new_host_tensor(shape={shape:?})");

        if data.len() == 1 && shape.is_null() {
            let tid = self.new_constant_tensor(Constant::new(data[0]));
            return Ok(tid);
        }

        let dtype = T::dtype();

        let bytes = (data.len() * dtype.bit_size() as usize).div_ceil(8);
        debug_assert_eq!(data.len() * std::mem::size_of::<T>(), bytes);

        // Allocate one element extra so masked store writes to the trash
        // element stay within bounds (eager tensors can become store
        // targets, e.g. in-place assign).
        let alloc_bytes = bytes + dtype.bit_size() as usize / 8;
        // Store to Host memory
        let free_bytes = Pool::Host.free_bytes();
        if alloc_bytes as Dim > free_bytes {
            return Err(ZyxError::AllocationError(
                format!("Attempted to allocate {alloc_bytes} B on host, but it only has {free_bytes} B free").into(),
            ));
        }

        let mut buf = vec![0u8; alloc_bytes].into_boxed_slice();
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), bytes) };
        buf[..bytes].copy_from_slice(src);

        let buffer_id = Buffer { pool: Pool::Host, buffer_id: Pool::Host.insert_host(buf) };

        // The caller keeps its own handle on `shape`; the Leaf stores the
        // interned ExprId (append-only slab, no retain needed).
        let shape_id = match self.tensors[shape] {
            TensorData::Symbolic { expr, .. } => expr,
            ref t => panic!("new_host_tensor: shape tid {shape} is not symbolic: {t:?}"),
        };
        let tid = self.tensors.push(TensorData::Leaf { shape_id, dtype, buffer: buffer_id, rc: 1 });

        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, shape={:?} dtype={}", self.shape(tid), self.dtype(tid));
        Ok(tid)
    }

    // Creates new tensor in disk
    pub fn new_disk_tensor(
        &mut self,
        shape: TensorId,
        dtype: DType,
        path: &Path,
        offset_bytes: u64,
    ) -> Result<TensorId, ZyxError> {
        // The caller keeps its own handle on `shape`; the Leaf stores the
        // interned ExprId (append-only slab, no retain needed).
        let shape_id = match self.tensors[shape] {
            TensorData::Symbolic { expr, .. } => expr,
            ref t => panic!("new_disk_tensor: shape tid {shape} is not symbolic: {t:?}"),
        };
        let resolved = self.resolve_symbolic_dims(shape_id);
        let bytes: Dim = ((resolved.iter().product::<Dim>() * dtype.bit_size() as Dim) + 7) / 8;

        let buffer_id = Buffer { pool: Pool::Disk, buffer_id: Pool::Disk.disk_buffer_from_path(bytes, path, offset_bytes) };
        let tid = self.tensors.push(TensorData::Leaf { shape_id, dtype, buffer: buffer_id, rc: 1 });
        Ok(tid)
    }

    pub fn cast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::cast(x={x}, dtype={dtype:?})");

        match self.tensors[x] {
            TensorData::Symbolic { expr, .. } => match self.exprs[expr].clone() {
                Expr::Constant { value } => self.new_constant_tensor(value.cast(dtype)),
                Expr::Variable { .. }
                | Expr::Cast { .. }
                | Expr::Unary { .. }
                | Expr::Binary { .. }
                | Expr::Stack { .. }
                | Expr::Stack2 { .. }
                | Expr::Stack3 { .. }
                | Expr::Stack4 { .. }
                | Expr::Stack5 { .. } => {
                    let nested = self.intern(Expr::Cast { x: expr, dtype });
                    let tid = self.tensors.push(TensorData::Symbolic { expr: nested, rc: 1 });
                    // The cast node holds an edge to x.
                    self.retain(x);
                    tid
                }
            },
            TensorData::Eager { kernel_id, op_id, shape_id, .. } => {
                let op_id = self.kernels[kernel_id].kernel.cast(op_id, dtype);
                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, shape_id, dtype, rc: 1 });
                self.kernels[kernel_id].outputs.insert(tid);
                // The cast shares the input's shape expression.

                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            TensorData::PendingLeaf { shape_id, .. } | TensorData::Leaf { shape_id, .. } => {
                // A Leaf has no kernel to extend: mint a fresh load kernel
                // for its buffer, then cast in it.
                let (kernel_id, op_id) = self.new_kernel_from_leaf(x);
                let op_id = self.kernels[kernel_id].kernel.cast(op_id, dtype);
                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, shape_id, dtype, rc: 1 });
                self.kernels[kernel_id].outputs.insert(tid);
                // The cast shares the input's shape expression.

                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            TensorData::GraphLeaf { class_id, graph_id, shape_id, .. }
            | TensorData::Graph { class_id, graph_id, shape_id, .. }
            | TensorData::Promoted { class_id, graph_id, shape_id, .. } => {
                self.assert_graph_alive(graph_id);
                let (_, class_id) = self.push_node(graph_id, Node::Cast { x: class_id, dtype });
                self.graphs[graph_id].ref_count += 1;
                // Shape-preserving op: share the input's shape expression, like eager.
                debug_assert!(!shape_id.is_null(), "cast: input graph tensor {x} has no shape expression");

                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> graph: tid={tid}, graph_id={graph_id:?}, class_id={class_id:?}");
                tid
            }
        }
    }

    pub fn bitcast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::bitcast(x={x}, dtype={dtype:?})");
        debug_assert_eq!(self.dtype(x).bit_size(), dtype.bit_size(), "bitcast requires equal bit widths");

        match self.tensors[x] {
            TensorData::Symbolic { .. } => {
                todo!("bitcast of pure-symbolic tensors")
            }
            TensorData::Eager { kernel_id, op_id, shape_id, .. } => {
                let op_id = self.kernels[kernel_id].kernel.bitcast(op_id, dtype);
                // The bitcast shares the input's shape expression.

                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, shape_id, dtype, rc: 1 });
                self.kernels[kernel_id].outputs.insert(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            TensorData::PendingLeaf { shape_id, .. } | TensorData::Leaf { shape_id, .. } => {
                // A Leaf has no kernel to extend: mint a fresh load kernel
                // for its buffer, then bitcast in it.
                let (kernel_id, op_id) = self.new_kernel_from_leaf(x);
                let op_id = self.kernels[kernel_id].kernel.bitcast(op_id, dtype);
                // The bitcast shares the input's shape expression.

                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, shape_id, dtype, rc: 1 });
                self.kernels[kernel_id].outputs.insert(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            TensorData::GraphLeaf { class_id, graph_id, shape_id, .. }
            | TensorData::Graph { class_id, graph_id, shape_id, .. }
            | TensorData::Promoted { class_id, graph_id, shape_id, .. } => {
                self.assert_graph_alive(graph_id);
                let (_, class_id) = self.push_node(graph_id, Node::Bitcast { x: class_id, dtype });
                self.graphs[graph_id].ref_count += 1;
                // Shape-preserving op: share the input's shape expression, like eager.
                debug_assert!(!shape_id.is_null(), "bitcast: input graph tensor {x} has no shape expression");

                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> graph: tid={tid}, graph_id={graph_id:?}, class_id={class_id:?}");
                tid
            }
        }
    }

    pub fn unary(&mut self, x: TensorId, uop: UOp) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::unary(x={x}, uop={uop:?})");
        self.verify_tensor_invariants();

        match self.tensors[x] {
            TensorData::Symbolic { expr, .. } => match self.exprs[expr].clone() {
                Expr::Constant { value } => self.new_constant_tensor(value.unary(uop)),
                _ => {
                    let root = self.intern(Expr::Unary { x: expr, uop });
                    self.tensors.push(TensorData::Symbolic { expr: root, rc: 1 })
                }
            },
            TensorData::Eager { kernel_id, op_id, shape_id, dtype, .. } => {
                let op_id = self.kernels[kernel_id].kernel.unary(op_id, uop);
                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, shape_id, dtype, rc: 1 });
                self.kernels[kernel_id].outputs.insert(tid);
                // The unary shares the input's shape expression.

                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            TensorData::PendingLeaf { shape_id, dtype, .. } | TensorData::Leaf { shape_id, dtype, .. } => {
                // A Leaf has no kernel to extend: mint a fresh load kernel
                // for its buffer, then apply the unary op in it.
                let (kernel_id, op_id) = self.new_kernel_from_leaf(x);
                let op_id = self.kernels[kernel_id].kernel.unary(op_id, uop);
                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, shape_id, dtype, rc: 1 });
                self.kernels[kernel_id].outputs.insert(tid);
                // The unary shares the input's shape expression.

                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            TensorData::GraphLeaf { class_id, graph_id, shape_id, dtype, .. }
            | TensorData::Graph { class_id, graph_id, shape_id, dtype, .. }
            | TensorData::Promoted { class_id, graph_id, shape_id, dtype, .. } => {
                self.assert_graph_alive(graph_id);
                let (_node_id, class_id) = self.push_node(graph_id, Node::Unary { x: class_id, uop });
                self.graphs[graph_id].ref_count += 1;
                // Shape-preserving op: share the input's shape expression, like eager.
                debug_assert!(!shape_id.is_null(), "unary: input graph tensor {x} has no shape expression");

                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> graph: tid={tid}, graph_id={graph_id:?}, nid={_node_id:?}, cid={class_id:?}");
                tid
            }
        }
    }

    pub fn binary(&mut self, x: TensorId, y: TensorId, bop: BOp) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::binary(x={x}, y={y}, bop={bop:?})");
        self.verify_tensor_invariants();
        // A pending operand still owes its value to a live store kernel
        // (e.g. a contiguous() cast shim whose producer still has other
        // outputs): flush the producer first so the operand is a Leaf, then
        // take the Leaf path below. Same idiom as assign/to_device/load.
        for tid in [x, y] {
            if let TensorData::PendingLeaf { depends_on, .. } = self.tensors[tid] {
                // depends_on is never null: a PendingLeaf always names the
                // live kernel that owes its store.
                debug_assert!(!depends_on.is_null(), "binary: PendingLeaf {tid} with null depends_on");
                let seen: Set<TensorId> = self.kernels[depends_on].outputs.iter().copied().collect();
                for out in seen {
                    self.add_store(out)?;
                }
            }
        }
        // Scalars broadcast implicitly. Non-scalar operands must already be
        // broadcast to equal shapes by the time they reach a binary op: any
        // non-scalar broadcasting is performed upstream by `Tensor::broadcast`.
        // `Node::Binary` / `Kernel::binary` do NOT broadcast.
        let rx = self.resolve_shape(x).len();
        let ry = self.resolve_shape(y).len();
        if !(rx == 0 || ry == 0) {
            debug_assert_eq!(
                self.resolve_shape(x),
                self.resolve_shape(y),
                "binary operands must be broadcast to equal shapes before runtime.binary (broadcasting is performed upstream by Tensor::broadcast)"
            );
        }
        // Pure-slab operands: the result is a slab Binary node (symbolic
        // scalar computation), no kernel or graph involved.
        let x_sym = matches!(self.tensors[x], TensorData::Symbolic { .. });
        let y_sym = matches!(self.tensors[y], TensorData::Symbolic { .. });
        if x_sym && y_sym {
            let ex = match self.tensors[x] {
                TensorData::Symbolic { expr, .. } => expr,
                ref t => panic!("binary: symbolic operand tid {x} is not Symbolic: {t:?}"),
            };
            let ey = match self.tensors[y] {
                TensorData::Symbolic { expr, .. } => expr,
                ref t => panic!("binary: symbolic operand tid {y} is not Symbolic: {t:?}"),
            };
            let root = self.intern(Expr::Binary { x: ex, y: ey, bop });
            let tid = self.tensors.push(TensorData::Symbolic { expr: root, rc: 1 });
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> symbolic: tid={tid}");
            return Ok(tid);
        }

        // Result shape: NULL when both sides are scalar, otherwise the
        // non-scalar side's shape expression (x wins when both are non-scalar;
        // they are asserted equal above).
        fn result_shape(rt: &Runtime, a: TensorId, b: TensorId) -> ExprId {
            let sa = match rt.tensors[a] {
                TensorData::Eager { shape_id, .. }
                | TensorData::Leaf { shape_id, .. }
                | TensorData::PendingLeaf { shape_id, .. }
                | TensorData::GraphLeaf { shape_id, .. }
                | TensorData::Graph { shape_id, .. }
                | TensorData::Promoted { shape_id, .. } => shape_id,
                _ => ExprId::NULL,
            };
            let sb = match rt.tensors[b] {
                TensorData::Eager { shape_id, .. }
                | TensorData::Leaf { shape_id, .. }
                | TensorData::PendingLeaf { shape_id, .. }
                | TensorData::GraphLeaf { shape_id, .. }
                | TensorData::Graph { shape_id, .. }
                | TensorData::Promoted { shape_id, .. } => shape_id,
                _ => ExprId::NULL,
            };
            if sa.is_null() { sb } else { sa }
        }

        let x_is_graph = self.is_graph(x);
        let y_is_graph = self.is_graph(y);
        if x_is_graph || y_is_graph {
            let graph_id = if x_is_graph {
                match self.tensors[x] {
                    TensorData::Graph { graph_id, .. }
                    | TensorData::Promoted { graph_id, .. }
                    | TensorData::GraphLeaf { graph_id, .. } => graph_id,
                    ref t => unreachable!("{t:?}"),
                }
            } else {
                match self.tensors[y] {
                    TensorData::Graph { graph_id, .. }
                    | TensorData::Promoted { graph_id, .. }
                    | TensorData::GraphLeaf { graph_id, .. } => graph_id,
                    ref t => unreachable!("{t:?}"),
                }
            };
            self.assert_graph_alive(graph_id);
            if !x_is_graph && !x_sym {
                self.promote_to_graph(x, graph_id)?;
            }
            if !y_is_graph && !y_sym {
                self.promote_to_graph(y, graph_id)?;
            }
            let cx = match self.tensors[x] {
                TensorData::Graph { class_id, .. }
                | TensorData::Promoted { class_id, .. }
                | TensorData::GraphLeaf { class_id, .. } => class_id,
                TensorData::Symbolic { expr, .. } => match self.exprs[expr].clone() {
                    Expr::Constant { value } => self.push_const(graph_id, value),
                    ref e => todo!("promote symbolic scalar tid {x} ({e:?}) into a graph"),
                },
                ref t => unreachable!("unreachable after promote: {t:?}"),
            };
            let cy = match self.tensors[y] {
                TensorData::Graph { class_id, .. }
                | TensorData::Promoted { class_id, .. }
                | TensorData::GraphLeaf { class_id, .. } => class_id,
                TensorData::Symbolic { expr, .. } => match self.exprs[expr].clone() {
                    Expr::Constant { value } => self.push_const(graph_id, value),
                    ref e => todo!("promote symbolic scalar tid {y} ({e:?}) into a graph"),
                },
                ref t => unreachable!("unreachable after promote: {t:?}"),
            };
            let class_id = self.push_binary_node(graph_id, cx, cy, bop);

            {
                let shape_id = result_shape(self, x, y);
                debug_assert!(!shape_id.is_null(), "binary: non-scalar graph operands {x}/{y} have no shape expression");

                self.graphs[graph_id].ref_count += 1;
                let dtype = if bop.returns_bool() { DType::Bool } else { self.dtype(x) };
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                Ok(tid)
            }
        } else if x_sym || y_sym {
            // Exactly one side is a pure-slab scalar: replay it into the
            // eager side's kernel.
            debug_assert!(!x_sym || !y_sym);
            let sym = if x_sym { x } else { y };
            let data = if x_sym { y } else { x };
            let shape_id = result_shape(self, x, y);
            // The result shares the operand's shape expression; take our own
            // reference instead of stealing the operand's.

            let (kid, data_op) = match self.tensors[data] {
                TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                ref t => panic!("binary: non-slab operand tid {data} is not an eager tensor: {t:?}"),
            };
            let sym_op = self.replay_symbolic_into_kernel(kid, sym);
            let op_id = if x_sym {
                self.kernels[kid].kernel.binary(sym_op, data_op, bop)
            } else {
                self.kernels[kid].kernel.binary(data_op, sym_op, bop)
            };
            let dtype = if bop.returns_bool() { DType::Bool } else { self.dtype(data) };
            let tid = self.tensors.push(TensorData::Eager { kernel_id: kid, op_id, shape_id, dtype, rc: 1 });
            self.kernels[kid].outputs.insert(tid);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> eager: tid={tid}, kid={kid:?}, op_id={op_id:?}");
            Ok(tid)
        } else {
            // Merge-time shape-compatibility rule: non-scalar operands must be
            // PROVABLY equal — per dim, the same constant or the SAME symbolic
            // dim tensor. A variable dim that only agrees with the other side
            // by its currently bound value is not proof (the slot may change
            // before launch), so the merge is rejected with an error. Code
            // with dynamic shapes must propagate the same dim tensor into
            // both operands' shapes (e.g. llama propagating the kv-cache len).
            let sx = self.resolve_shape_without_variables(x);
            let sy = self.resolve_shape_without_variables(y);
            if !sx.is_empty() && !sy.is_empty() && sx != sy {
                return Err(ZyxError::shape_error(
                    format!(
                        "binary: cannot prove operand shapes are equal: {sx:?} vs {sy:?} — a symbolic dim must be the same dim tensor in both operands, or concrete in both"
                    )
                    .into(),
                ));
            }
            let shape_id = result_shape(self, x, y);
            // The result shares the operand's shape expression; take our own
            // reference instead of stealing the operand's.

            let (mut kid_x, mut op_id_x) = match self.tensors[x] {
                TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => self.new_kernel_from_leaf(x),
                TensorData::Graph { .. } | TensorData::GraphLeaf { .. } | TensorData::Promoted { .. } | TensorData::Symbolic { .. } => {
                    panic!("binary: operand tid {x} is not an eager tensor: {:?}", self.tensors[x])
                }
            };
            let (mut kid_y, mut op_id_y) = match self.tensors[y] {
                TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => self.new_kernel_from_leaf(y),
                TensorData::Graph { .. } | TensorData::GraphLeaf { .. } | TensorData::Promoted { .. } | TensorData::Symbolic { .. } => {
                    panic!("binary: operand tid {y} is not an eager tensor: {:?}", self.tensors[y])
                }
            };

            let (kernel_id, op_id) = if kid_x == kid_y {
                let op_id = self.kernels[kid_x].kernel.binary(op_id_x, op_id_y, bop);
                (kid_x, op_id)
            } else {
                let x_stores = !self.kernels[kid_x].stores.is_empty();
                let y_stores = !self.kernels[kid_y].stores.is_empty();
                match (x_stores, y_stores) {
                    (true, true) => {
                        self.add_store(x)?;
                        self.add_store(y)?;
                    }
                    (true, false) => self.add_store(x)?,
                    (false, true) => self.add_store(y)?,
                    (false, false) => {}
                }
                // add_store may have re-created the operands as Leafs (a
                // stored operand is a buffer now) — re-resolve via new_kernel_from_leaf.
                (kid_x, op_id_x) = match self.tensors[x] {
                    TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                    TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => self.new_kernel_from_leaf(x),
                    TensorData::Graph { .. }
                    | TensorData::GraphLeaf { .. }
                    | TensorData::Promoted { .. }
                    | TensorData::Symbolic { .. } => unreachable!("add_store turned operand into unexpected data: {:?}", self.tensors[x]),
                };
                (kid_y, op_id_y) = match self.tensors[y] {
                    TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                    TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => self.new_kernel_from_leaf(y),
                    TensorData::Graph { .. }
                    | TensorData::GraphLeaf { .. }
                    | TensorData::Promoted { .. }
                    | TensorData::Symbolic { .. } => unreachable!("add_store turned operand into unexpected data: {:?}", self.tensors[y]),
                };

                let swap = self.kernels[kid_y].kernel.is_reduce() && !self.kernels[kid_x].kernel.is_reduce();
                let (keep_kid, merge_kid, keep_op, merge_op) = if swap {
                    (kid_y, kid_x, op_id_y, op_id_x)
                } else {
                    (kid_x, kid_y, op_id_x, op_id_y)
                };

                let op_map = self.merge_kernel(keep_kid, merge_kid)?;

                let op_id = if swap {
                    self.kernels[keep_kid].kernel.binary(op_map[&merge_op], keep_op, bop)
                } else {
                    self.kernels[keep_kid].kernel.binary(keep_op, op_map[&merge_op], bop)
                };
                (keep_kid, op_id)
            };

            let dtype = if bop.returns_bool() { DType::Bool } else { self.dtype(x) };
            let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, shape_id, dtype, rc: 1 });
            self.kernels[kernel_id].outputs.insert(tid);

            #[cfg(feature = "debug_tensor_op")]
            println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            Ok(tid)
        }
    }

    /// Returns the device the tensor lives on.
    ///
    /// Realized tensors map through their buffer's pool to the owning device;
    /// unrealized eager tensors use their kernel's device; graph/slab tensors
    /// have no device and return [`Dev::Auto`](Dev::Auto).
    ///
    /// # Panics
    ///
    /// If the tensor's buffer pool has no devices attached.
    pub fn device(&self, x: TensorId) -> Dev {
        if let Some(buf_id) = self.leaf_buffer(x) {
            // The host pool is shared by the C and Cblas devices; report C.
            // The disk pool has no device; report Auto.
            return match buf_id.pool {
                Pool::Host => Dev::C,
                Pool::Disk => Dev::Auto,
                pool => Dev::all().into_iter().find(|d| d.pool() == pool).unwrap_or_else(|| {
                    panic!(
                        "device: tensor {x} lives in pool {pool:?}, which has no devices attached. The backend operating this pool was likely configured out or never initialized."
                    )
                }),
            };
        }
        match self.tensors[x] {
            TensorData::Eager { kernel_id, .. } => self.kernels[kernel_id].kernel.dev,
            // A kept buffer reports its pool's device, same mapping as
            // realized tensors above; otherwise the producer kernel's device.
            TensorData::PendingLeaf { old_buffer: Some(buf_id), .. } => match buf_id.pool {
                Pool::Host => Dev::C,
                Pool::Disk => Dev::Auto,
                pool => Dev::all().into_iter().find(|d| d.pool() == pool).unwrap_or_else(|| {
                    panic!(
                        "device: tensor {x} lives in pool {pool:?}, which has no devices attached. The backend operating this pool was likely configured out or never initialized."
                    )
                }),
            },
            TensorData::PendingLeaf { depends_on, .. } if !depends_on.is_null() => {
                self.kernels[depends_on].kernel.dev
            }
            _ => Dev::Auto,
        }
    }

    #[allow(clippy::wrong_self_convention)] // naming convention from GPU API, not a conversion method
    pub fn to_device(&mut self, x: TensorId, device: Dev) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::to_device(x={x}, device={device:?})");
        let dst_pool = device.pool();
        // Fast path: tensor already lives in the destination pool. Compares
        // pools (not devices via device(x)): the source pool may have no
        // devices attached at all (e.g. disk), which device(x) panics on.
        if let Some(buf_id) = self.leaf_buffer(x) {
            if buf_id.pool == dst_pool {
                self.retain(x);
                return Ok(x);
            }
        }
        match self.tensors[x] {
            TensorData::Leaf { shape_id, dtype, .. } => {
                let buf_id = self.leaf_buffer(x).expect("to_device: realized Leaf has no buffer");
                // Materialization may have placed x in the destination pool
                // already (e.g. an unrealized eager tensor on the target
                // device). Copying pool-to-itself is unsupported: skip it.
                if buf_id.pool == dst_pool {
                    self.retain(x);
                    return Ok(x);
                }
                let shape = self.resolve_shape(x);
                let bytes = ((shape.iter().product::<Dim>() * dtype.bit_size() as Dim) + 7) / 8;
                let alloc_bytes = bytes + dtype.bit_size() as Dim / 8;
                let dst_buf = dst_pool.allocate(alloc_bytes)?;
                let dst_id = Buffer { pool: dst_pool, buffer_id: dst_buf };
                // Drain pending events on the source buffer before the copy.
                dst_pool.pool_to_pool(buf_id.pool, buf_id.buffer_id, dst_id.buffer_id)?;
                debug_assert!(!shape_id.is_null(), "to_device: eager tensor {x} has no shape expression");

                let tid = self.tensors.push(TensorData::Leaf { shape_id, dtype, buffer: dst_id, rc: 1 });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid} (cross-pool copy {buf_id:?} -> {dst_id:?})");
                Ok(tid)
            }
            TensorData::PendingLeaf { depends_on: kernel_id, shape_id, dtype, .. }
            | TensorData::Eager { kernel_id, shape_id, dtype, .. } => {
                if !kernel_id.is_null() {
                    let outputs: Vec<TensorId> = self.kernels[kernel_id].outputs.iter().copied().collect();
                    for out in outputs {
                        self.add_store(out)?;
                    }
                }
                let buf_id = self.leaf_buffer(x).expect("to_device: tensor {x} was not materialized");
                // Materialization may have placed x in the destination pool
                // already (e.g. an unrealized eager tensor on the target
                // device). Copying pool-to-itself is unsupported: skip it.
                if buf_id.pool == dst_pool {
                    self.retain(x);
                    return Ok(x);
                }
                let shape = self.resolve_shape(x);
                let bytes = ((shape.iter().product::<Dim>() * dtype.bit_size() as Dim) + 7) / 8;
                let alloc_bytes = bytes + dtype.bit_size() as Dim / 8;
                let dst_buf = dst_pool.allocate(alloc_bytes)?;
                let dst_id = Buffer { pool: dst_pool, buffer_id: dst_buf };
                // Drain pending events on the source buffer before the copy.
                dst_pool.pool_to_pool(buf_id.pool, buf_id.buffer_id, dst_id.buffer_id)?;
                debug_assert!(!shape_id.is_null(), "to_device: eager tensor {x} has no shape expression");

                let tid = self.tensors.push(TensorData::Leaf { shape_id, dtype, buffer: dst_id, rc: 1 });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid} (cross-pool copy {buf_id:?} -> {dst_id:?})");
                Ok(tid)
            }
            TensorData::GraphLeaf { class_id, graph_id, shape_id, .. }
            | TensorData::Graph { class_id, graph_id, shape_id, .. }
            | TensorData::Promoted { class_id, graph_id, shape_id, .. } => {
                assert!(!self.graphs[graph_id].dead, "tape scope has ended (tensor belongs to a dead tape scope");
                // TODO measure actual time by running a test copy
                let (_node_id, cid) = self.push_node(graph_id, Node::ToDevice { x: class_id, device, time: 0 });
                self.graphs[graph_id].ref_count += 1;
                // Shape-preserving op: share the input's shape expression.
                debug_assert!(!shape_id.is_null(), "to_device: input graph tensor {x} has no shape expression");

                let dtype = self.dtype(x);
                let tid = self.tensors.push(TensorData::Graph { class_id: cid, graph_id, shape_id, dtype, rc: 1 });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid}, nid={_node_id:?}, cid={cid:?}");
                Ok(tid)
            }
            TensorData::Symbolic { .. } => {
                self.retain(x);
                Ok(x)
            }
        }
    }

    /// Forces a contiguous, materialized view of `x` (breaks aliasing / forces a fresh buffer).
    ///
    /// Structure:
    /// - **Already realized** (in `buffer_map`): a no-op — the tensor is a load from its own
    ///   contiguous buffer. `x` is retained once up front; the match arms that return `x`
    ///   itself (symbolic slab variants, `Leaf`, realized Graph/Promoted/Eager) rely on that.
    /// - **Graph** (unrealized): pushes a `Node::Contiguous` and shares `x`'s shape
    ///   expression. The kernelizer's `Node::Contiguous` arm applies a same-dtype `Cast`
    ///   (value identity) and stores the new class — giving it a distinct op and its own
    ///   backing buffer instead of aliasing `x`'s load op.
    /// - **Eager (unrealized)**: a cast-shim — emit a same-dtype `Cast` op in `x`'s kernel
    ///   and store the cast tensor as a new handle, leaving `x` itself unfused in its producer.
    pub fn contiguous(&mut self, x: TensorId) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::contiguous(x={x})");
        self.verify_tensor_invariants();

        match self.tensors[x] {
            TensorData::Symbolic { .. }
            | TensorData::Leaf { .. }
            | TensorData::GraphLeaf { .. }
            | TensorData::PendingLeaf { .. } => {
                // Pure-slab, realized, or pending value: nothing to
                // materialize — `x` is already the contiguous value.
                self.retain(x);
                Ok(x)
            }
            TensorData::Graph { class_id, graph_id, shape_id, dtype, .. }
            | TensorData::Promoted { class_id, graph_id, shape_id, dtype, .. } => {
                self.assert_graph_alive(graph_id);
                let (_node_id, cid) = self.push_node(graph_id, Node::Contiguous { x: class_id });
                self.graphs[graph_id].ref_count += 1;
                // Shape-preserving op: share the input's shape expression.
                debug_assert!(!shape_id.is_null(), "contiguous: input graph tensor {x} has no shape expression");

                let tid = self.tensors.push(TensorData::Graph { class_id: cid, graph_id, shape_id, dtype, rc: 1 });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={tid}, nid={_node_id:?}, cid={cid:?}");
                Ok(tid)
            }
            TensorData::Eager { .. } => {
                // Cast-shim semantics: contiguous adds a same-dtype Cast (a value
                // identity) to x's kernel and stores THAT as a new tensor. The
                // cast tid is the returned handle — it gets its own buffer_map
                // entry whenever the store materializes (immediately, or lazily
                // via `depends_on` when the producer kernel still has other
                // outputs), while x itself stays unfused in its producer.
                let cast_tid = self.cast(x, self.dtype(x));
                self.add_store(cast_tid)?;
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> tid={cast_tid} (cast shim stored)");
                Ok(cast_tid)
            }
        }
    }

    pub fn reduce(&mut self, x: TensorId, mut axes: Vec<UAxis>, rop: BOp) -> Result<TensorId, ZyxError> {
        self.verify_tensor_invariants();
        let rank = self.shape(x).len();
        debug_assert!(!axes.is_empty(), "reduce must specify at least one axis");
        debug_assert!(axes.iter().all(|&a| (a as usize) < rank), "reduce axis {axes:?} out of bounds for rank {rank}");
        debug_assert!(
            axes.len() == axes.iter().collect::<std::collections::BTreeSet<_>>().len(),
            "reduce axes must be unique: {axes:?}"
        );
        axes.sort_unstable();

        match self.tensors[x] {
            TensorData::Graph { class_id, graph_id, dtype, .. } | TensorData::Promoted { class_id, graph_id, dtype, .. } => {
                self.assert_graph_alive(graph_id);
                // Result shape mirrors the eager arm: surviving dim
                // expressions, reduced axes skipped; a full reduction keeps a
                // single dim of size 1. Computed first since `axes` moves into
                // the node below.
                let mut dims = self.shape(x).to_vec();
                debug_assert!(!dims.is_empty(), "reduce: input graph tensor {x} has no shape expression");
                for axis in axes.iter().rev() {
                    dims.remove(*axis as usize);
                }
                let shape_id = if dims.is_empty() {
                    let one_const = self.new_constant_tensor(Constant::idx(1i64));
                    let stacked = self.stack(&[one_const])?;
                    self.release(one_const);
                    let expr = match self.tensors[stacked] {
                        TensorData::Symbolic { expr, .. } => expr,
                        ref t => panic!("reduce: shape tid {stacked} is not symbolic: {t:?}"),
                    };
                    self.release(stacked);
                    expr
                } else {
                    let stacked = self.stack(&dims)?;
                    let expr = match self.tensors[stacked] {
                        TensorData::Symbolic { expr, .. } => expr,
                        ref t => panic!("reduce: shape tid {stacked} is not symbolic: {t:?}"),
                    };
                    self.release(stacked);
                    expr
                };
                let (_node_id, class_id) =
                    self.push_node(graph_id, Node::Reduce { x: class_id, rop, axes: axes.into_boxed_slice() });
                self.graphs[graph_id].ref_count += 1;

                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                Ok(tid)
            }
            TensorData::Eager { dtype, .. } | TensorData::Leaf { dtype, .. } => {
                // Reduce one axis at a time, permuting each to be last. Reduce the
                // highest axis first so lower indices stay valid as the rank shrinks.
                let mut cur = x;
                // Ownership: `owns_cur` tells whether reduce holds exactly one
                // reference on `cur` that it must release before overwriting
                // it. Entering the loop, `cur` is the caller's `x` — reduce
                // holds nothing on it.
                let mut owns_cur = false;
                let n_axes = axes.len();
                axes.sort_unstable_by(|a, b| b.cmp(a));
                let mut dims = self.shape(x);
                for axis in axes {
                    let rank = self.resolve_shape(cur).len();
                    let permute_axes: Vec<UAxis> = (0..rank as UAxis).filter(|&i| i != axis).chain([axis]).collect();
                    let prev = cur;
                    let prev_owned = owns_cur;
                    cur = self.permute(cur, permute_axes);
                    // `permute` grants one reference on its result — including
                    // the identity fast path, which retains and returns the
                    // same tid.
                    if prev_owned {
                        self.release(prev);
                    }

                    let (kid, op_id) = self.duplicate_or_store(cur, false)?;
                    let dims_ops = self.kernels[kid].kernel.shape_ids(op_id);
                    debug_assert!(!dims_ops.is_empty(), "reduce of scalar");
                    let reduce_axis = *dims_ops.last().unwrap();
                    let op_id = self.kernels[kid].kernel.push_back(Op::Reduce { x: op_id, rop, reduce_axis });

                    // Result shape: surviving dim expressions, reduced axis skipped.
                    let mut kept_dims = dims.clone();
                    kept_dims.remove(axis);
                    let shape_id = if kept_dims.is_empty() {
                        ExprId::NULL
                    } else {
                        let stacked = self.stack(&kept_dims)?;
                        let expr = match self.tensors[stacked] {
                            TensorData::Symbolic { expr, .. } => expr,
                            ref t => panic!("reduce: shape tid {stacked} is not symbolic: {t:?}"),
                        };
                        self.release(stacked);
                        expr
                    };

                    let tid = self.tensors.push(TensorData::Eager { kernel_id: kid, op_id, shape_id, dtype, rc: 1 });
                    dims = kept_dims;

                    debug_assert_eq!(self.kernels[kid].outputs.len(), 0, "input into reduce must have empty outputs");
                    self.kernels[kid].outputs.insert(tid);
                    // Overwrite `cur` with the reduce result: release the
                    // reference reduce holds on the permuted intermediate
                    // (granted by `permute` above).
                    self.release(cur);
                    owns_cur = true;
                    cur = tid;
                }

                if rank == n_axes {
                    let (kid, op_id) = match self.tensors[cur] {
                        TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                        ref t => unreachable!("{t:?}"),
                    };
                    // Full reduction keeps a single dim of size 1.
                    let one_const = self.new_constant_tensor(Constant::idx(1i64));
                    let stacked = self.stack(&[one_const])?;
                    self.release(one_const);
                    let shape_id = match self.tensors[stacked] {
                        TensorData::Symbolic { expr, .. } => expr,
                        ref t => panic!("reduce: shape tid {stacked} is not symbolic: {t:?}"),
                    };
                    self.release(stacked);
                    let one = self.kernels[kid].kernel.const_idx(1);
                    let op_id = self.kernels[kid].kernel.reshape(op_id, one);
                    match &mut self.tensors[cur] {
                        TensorData::Eager { op_id: slot, shape_id: slot_shape, .. } => {
                            *slot = op_id;
                            *slot_shape = shape_id;
                        }
                        ref t => unreachable!("{t:?}"),
                    }
                }

                #[cfg(feature = "debug_tensor_op")]
                println!(
                    "  -> eager: tid={cur}, op_id={:?}",
                    match self.tensors[cur] {
                        TensorData::Eager { op_id, .. } => op_id,
                        ref t => unreachable!("{t:?}"),
                    }
                );
                Ok(cur)
            }
            ref t => todo!("reduce of pure-slab tensor {t:?}"),
        }
    }

    pub(super) fn stack(&mut self, tensors: &[TensorId]) -> Result<TensorId, ZyxError> {
        debug_assert!(!tensors.is_empty(), "stack: empty");
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::stack(tensors={tensors:?})");

        let dtype = self.dtype(tensors[0]);

        // All pure-slab operands: the result is a slab Stack node (used both
        // for data-less symbolic stacks, e.g. shape expressions, and nothing
        // else — data stacking needs a kernel or graph below).
        if tensors.iter().all(|&t| matches!(self.tensors[t], TensorData::Symbolic { .. })) {
            // 1d shapes skip the Stack node entirely: the shape_id IS the
            // single dim tensor (shared, rc'd like any shape expression).
            if tensors.len() == 1 {
                self.retain(tensors[0]);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> symbolic: tid={} (1d shape, no stack node)", tensors[0]);
                return Ok(tensors[0]);
            }
            // Arity dispatch: 2-5 element shapes avoid the Box<[ExprId]>
            // allocation of the generic Stack node.
            let exprs: Vec<ExprId> = tensors
                .iter()
                .map(|&t| match self.tensors[t] {
                    TensorData::Symbolic { expr, .. } => expr,
                    ref t => panic!("stack: operand tid is not symbolic: {t:?}"),
                })
                .collect();
            let expr = match exprs.len() {
                2 => self.intern(Expr::Stack2 { exprs: [exprs[0], exprs[1]] }),
                3 => self.intern(Expr::Stack3 { exprs: [exprs[0], exprs[1], exprs[2]] }),
                4 => self.intern(Expr::Stack4 { exprs: [exprs[0], exprs[1], exprs[2], exprs[3]] }),
                5 => self.intern(Expr::Stack5 { exprs: [exprs[0], exprs[1], exprs[2], exprs[3], exprs[4]] }),
                _ => self.intern(Expr::Stack { exprs: exprs.into_boxed_slice() }),
            };
            let tid = self.tensors.push(TensorData::Symbolic { expr, rc: 1 });
            // The stack node holds an edge to every element.
            for &t in tensors {
                self.retain(t);
            }
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> symbolic: tid={tid}");
            return Ok(tid);
        }

        if tensors.iter().any(|&t| self.is_graph(t)) {
            let graph_id = tensors
                .iter()
                .find(|&&t| self.is_graph(t))
                .map(|&t| match self.tensors[t] {
                    TensorData::Graph { graph_id, .. } | TensorData::Promoted { graph_id, .. } => graph_id,
                    ref t => unreachable!("{t:?}"),
                })
                .unwrap();
            self.assert_graph_alive(graph_id);
            for &t in tensors {
                let is_pure_const = match self.tensors[t] {
                    TensorData::Symbolic { expr, .. } => matches!(self.exprs[expr], Expr::Constant { .. }),
                    _ => false,
                };
                if !self.is_graph(t) && !is_pure_const {
                    self.promote_to_graph(t, graph_id)?;
                }
            }
            let mut ops = Vec::with_capacity(tensors.len());
            for &t in tensors {
                ops.push(match self.tensors[t] {
                    TensorData::Graph { class_id, .. } | TensorData::Promoted { class_id, .. } => class_id,
                    TensorData::Symbolic { expr, .. } => match self.exprs[expr].clone() {
                        Expr::Constant { value } => self.push_const(graph_id, value),
                        ref e => todo!("stack: promote symbolic scalar tid {t} ({e:?}) into a graph"),
                    },
                    ref t => todo!("stack: promote symbolic scalar tid {t:?} into a graph"),
                });
            }
            let (_, class_id) = self.push_node(graph_id, Node::Stack { ops: ops.into_boxed_slice() });
            {
                // Result shape mirrors the eager arm: [len] ++ first operand's
                // dims, as a slab stack.
                let len_const = self.new_constant_tensor(Constant::idx(tensors.len() as i64));
                let mut shape_dims = Vec::with_capacity(tensors.len() + 1);
                shape_dims.push(len_const);
                shape_dims.extend(self.shape(tensors[0]));
                let stacked = self.stack(&shape_dims)?;
                self.release(len_const);
                let shape_id = match self.tensors[stacked] {
                    TensorData::Symbolic { expr, .. } => expr,
                    ref t => panic!("stack: shape tid {stacked} is not symbolic: {t:?}"),
                };
                self.release(stacked);
                self.graphs[graph_id].ref_count += 1;
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                Ok(tid)
            }
        } else {
            let keep_kid = match self.tensors[tensors[0]] {
                TensorData::Eager { kernel_id, .. } => kernel_id,
                TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => self.new_kernel_from_leaf(tensors[0]).0,
                TensorData::Graph { .. } | TensorData::GraphLeaf { .. } | TensorData::Promoted { .. } | TensorData::Symbolic { .. } => {
                    panic!("stack: operand tid {} is not an eager tensor: {:?}", tensors[0], self.tensors[tensors[0]])
                }
            };
            let mut ops = Vec::with_capacity(tensors.len());
            for &t in tensors {
                let (mut kid, mut op) = match self.tensors[t] {
                    TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                    TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => self.new_kernel_from_leaf(t),
                    TensorData::Graph { .. } | TensorData::GraphLeaf { .. } | TensorData::Promoted { .. } | TensorData::Symbolic { .. } => {
                        panic!("stack: operand is not an eager tensor: {:?}", self.tensors[t])
                    }
                };
                if kid != keep_kid {
                    if !self.kernels[kid].stores.is_empty() {
                        self.add_store(t)?;
                        (kid, op) = match self.tensors[t] {
                            TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                            TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => self.new_kernel_from_leaf(t),
                            TensorData::Graph { .. }
                            | TensorData::GraphLeaf { .. }
                            | TensorData::Promoted { .. }
                            | TensorData::Symbolic { .. } => unreachable!("{:?}", self.tensors[t]),
                        };
                    }
                    if kid != keep_kid {
                        let op_map = self.merge_kernel(keep_kid, kid)?;
                        op = op_map[&op];
                    }
                }
                ops.push(op);
            }
            let op_id = self.kernels[keep_kid].kernel.stack(&ops);

            // Result shape: [len] ++ first operand's dims, as a slab stack.
            let len_const = self.new_constant_tensor(Constant::idx(tensors.len() as i64));
            let mut shape_dims = Vec::with_capacity(tensors.len() + 1);
            shape_dims.push(len_const);
            shape_dims.extend(self.shape(tensors[0]));
            let stacked = self.stack(&shape_dims)?;
            self.release(len_const);
            let shape_id = match self.tensors[stacked] {
                TensorData::Symbolic { expr, .. } => expr,
                ref t => panic!("stack: shape tid {stacked} is not symbolic: {t:?}"),
            };
            self.release(stacked);

            let tid = self.tensors.push(TensorData::Eager { kernel_id: keep_kid, op_id, shape_id, dtype, rc: 1 });
            self.kernels[keep_kid].outputs.insert(tid);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> eager: tid={tid}, kid={keep_kid:?}, op_id={op_id:?}");
            Ok(tid)
        }
    }

    pub(super) fn reshape(&mut self, x: TensorId, shape_id: TensorId) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::reshape(x={x}, shape={shape_id:?})");
        // The shape operand is a TensorId handle; the result stores the
        // interned ExprId (append-only slab, no retain needed).
        let shape_expr = if shape_id == TensorId::NULL {
            ExprId::NULL
        } else {
            match self.tensors[shape_id] {
                TensorData::Symbolic { expr, .. } => expr,
                TensorData::Eager { shape_id, .. }
                | TensorData::Leaf { shape_id, .. }
                | TensorData::PendingLeaf { shape_id, .. }
                | TensorData::GraphLeaf { shape_id, .. }
                | TensorData::Graph { shape_id, .. }
                | TensorData::Promoted { shape_id, .. } => shape_id,
            }
        };
        // Shapes are always resolvable on the tensor side (closed expressions
        // over variable_map), so this is a total check.
        debug_assert_eq!(
            self.resolve_shape(x).iter().product::<Dim>(),
            self.resolve_symbolic_dims(shape_expr).iter().product::<Dim>(),
            "reshape element count mismatch"
        );

        let dtype = self.dtype(x);

        if self.is_graph(x) || self.is_graph(shape_id) {
            let graph_id = if self.is_graph(x) {
                match self.tensors[x] {
                    TensorData::Graph { graph_id, .. }
                    | TensorData::GraphLeaf { graph_id, .. }
                    | TensorData::Promoted { graph_id, .. } => graph_id,
                    ref t => unreachable!("{t:?}"),
                }
            } else {
                match self.tensors[shape_id] {
                    TensorData::Graph { graph_id, .. }
                    | TensorData::GraphLeaf { graph_id, .. }
                    | TensorData::Promoted { graph_id, .. } => graph_id,
                    ref t => unreachable!("{t:?}"),
                }
            };
            self.assert_graph_alive(graph_id);
            if !self.is_graph(x) {
                self.promote_to_graph(x, graph_id)?;
            }
            let x_class = match self.tensors[x] {
                TensorData::Graph { class_id, .. }
                | TensorData::GraphLeaf { class_id, .. }
                | TensorData::Promoted { class_id, .. } => class_id,
                ref t => unreachable!("{t:?}"),
            };
            // The target shape enters the graph: a graph-affiliated shape is
            // used directly (same scope asserted); a slab-side symbolic
            // expression is promoted node by node.
            let shape_class = match self.tensors[shape_id] {
                TensorData::Graph { class_id, graph_id: g, .. }
                | TensorData::GraphLeaf { class_id, graph_id: g, .. }
                | TensorData::Promoted { class_id, graph_id: g, .. } => {
                    assert!(g == graph_id, "reshape: shape belongs to a different tape scope");
                    class_id
                }
                _ => self.replay_symbolic_into_graph(graph_id, shape_id),
            };
            let (_, class_id) = self.push_node(graph_id, Node::Reshape { x: x_class, shape: shape_class });
            {
                self.graphs[graph_id].ref_count += 1;

                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id: shape_expr, dtype, rc: 1 });
                Ok(tid)
            }
        } else {
            // If x is realized, the result is a **Leaf** sharing x's buffer:
            // a view is not an operation, so no kernel is created and nothing
            // is listed in any kernel's outputs — consumers mint their own
            // load kernels via `new_kernel_from_leaf`. This avoids copying data for a
            // view-only reshape. The view retains x, so x (the owner)
            // outlives all its views and deallocates the buffer on death.
            if let Some(buf_id) = self.leaf_buffer(x) {
                if !shape_expr.is_null() {}
                let dtype = self.dtype(x);
                self.retain(x);
                // The view is a second owner of the pool buffer: pool-level
                // retain pairs with the release in the Leaf death path, so a
                // dying view never frees the owner's buffer early.
                buf_id.pool.retain(buf_id.buffer_id);
                let tid = self.tensors.push(TensorData::Leaf { shape_id: shape_expr, dtype, buffer: buf_id, rc: 1 });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid} (Leaf, shares buffer with x={x})");
                return Ok(tid);
            }

            let (kernel_id, op_id) = self.duplicate_or_store(x, false)?;

            debug_assert_eq!(
                self.kernels[kernel_id].outputs.len(),
                0,
                "input into reshape must have empty outputs before the shape kernel is merged"
            );
            let shape_op = self.replay_symbolic_into_kernel(kernel_id, shape_id);
            let op_id = self.kernels[kernel_id].kernel.reshape(op_id, shape_op);
            if !shape_expr.is_null() {}
            let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, shape_id: shape_expr, dtype, rc: 1 });

            debug_assert_eq!(self.kernels[kernel_id].outputs.contains(&tid), false);
            self.kernels[kernel_id].outputs.insert(tid);

            #[cfg(feature = "debug_tensor_op")]
            println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            Ok(tid)
        }
    }

    pub fn expand(&mut self, x: TensorId, shape_id: TensorId) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::expand(x={x}, shape={shape_id:?})");
        // The shape operand is a TensorId handle; the result stores the
        // interned ExprId (append-only slab, no retain needed).
        let shape_expr = if shape_id == TensorId::NULL {
            ExprId::NULL
        } else {
            match self.tensors[shape_id] {
                TensorData::Symbolic { expr, .. } => expr,
                TensorData::Eager { shape_id, .. }
                | TensorData::Leaf { shape_id, .. }
                | TensorData::PendingLeaf { shape_id, .. }
                | TensorData::GraphLeaf { shape_id, .. }
                | TensorData::Graph { shape_id, .. }
                | TensorData::Promoted { shape_id, .. } => shape_id,
            }
        };
        let dtype = self.dtype(x);
        let sh = self.resolve_shape(x);
        let target = self.resolve_symbolic_dims(shape_expr);
        debug_assert!(
            sh.len() <= target.len(),
            "expand: input rank {} > target rank {}: {:?} -> {:?}",
            sh.len(),
            target.len(),
            sh,
            target
        );
        for (old, new) in sh.iter().copied().rev().zip(target.iter().copied().rev()) {
            debug_assert!(old == new || old == 1, "expand: incompatible dims: {old} vs {new} in {:?} -> {:?}", sh, target);
        }

        match self.tensors[x] {
            TensorData::Graph { class_id: x_class, graph_id, .. }
            | TensorData::GraphLeaf { class_id: x_class, graph_id, .. }
            | TensorData::Promoted { class_id: x_class, graph_id, .. } => {
                self.assert_graph_alive(graph_id);
                let shape_class = match self.tensors[shape_id] {
                    TensorData::Graph { class_id, graph_id: g, .. }
                    | TensorData::GraphLeaf { class_id, graph_id: g, .. }
                    | TensorData::Promoted { class_id, graph_id: g, .. } => {
                        assert!(g == graph_id, "expand: shape belongs to a different tape scope");
                        class_id
                    }
                    TensorData::Eager { .. }
                    | TensorData::Leaf { .. }
                    | TensorData::PendingLeaf { .. }
                    | TensorData::Symbolic { .. } => self.replay_symbolic_into_graph(graph_id, shape_id),
                };
                let (_, class_id) = self.push_node(graph_id, Node::Expand { x: x_class, shape: shape_class });
                {
                    self.graphs[graph_id].ref_count += 1;

                    let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id: shape_expr, dtype, rc: 1 });
                    Ok(tid)
                }
            }
            TensorData::Symbolic { .. } => {
                // Pure-slab operand (e.g. a broadcast scalar): materialize it into
                // a fresh eager kernel that replays the slab expression and
                // expands it to the target shape.
                let kid = self.kernels.push(KernelData {
                    outputs: Set::default(),
                    loads: Vec::new(),
                    stores: Vec::new(),
                    kernel: Kernel::from_device_id(Dev::Auto, None),
                });
                let val_op = self.replay_symbolic_into_kernel(kid, x);
                let shape_op = self.replay_symbolic_into_kernel(kid, shape_id);
                let op_id = self.kernels[kid].kernel.expand(val_op, shape_op);
                if !shape_expr.is_null() {}
                let tid = self.tensors.push(TensorData::Eager { kernel_id: kid, op_id, shape_id: shape_expr, dtype, rc: 1 });
                self.kernels[kid].outputs.insert(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("runtime::expand(x={x}) -> eager from slab: tid={tid}, kid={kid:?}, op_id={op_id:?}");
                Ok(tid)
            }
            TensorData::Eager { .. } | TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => {
                let force_store = match self.tensors[x] {
                    TensorData::Eager { kernel_id, op_id, .. } => self.kernels[kernel_id].kernel.is_preceded_by_compute(op_id),
                    TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => false,
                    TensorData::Graph { .. } | TensorData::GraphLeaf { .. } | TensorData::Promoted { .. } | TensorData::Symbolic { .. } => {
                        panic!("expand: operand tid {x} is not an eager tensor: {:?}", self.tensors[x])
                    }
                };
                let (kernel_id, op_id) = self.duplicate_or_store(x, force_store)?;

                debug_assert_eq!(
                    self.kernels[kernel_id].outputs.len(),
                    0,
                    "input into expand must have empty outputs before the shape kernel is merged"
                );
                let shape_op = self.replay_symbolic_into_kernel(kernel_id, shape_id);
                let op_id = self.kernels[kernel_id].kernel.expand(op_id, shape_op);

                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, shape_id: shape_expr, dtype, rc: 1 });

                debug_assert_eq!(self.kernels[kernel_id].outputs.contains(&tid), false);
                self.kernels[kernel_id].outputs.insert(tid);

                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                Ok(tid)
            }
        }
    }

    pub fn permute(&mut self, x: TensorId, axes: Vec<UAxis>) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::permute(x={x}, axes={axes:?})");
        self.verify_tensor_invariants();
        let sh = self.resolve_shape(x).to_vec();
        debug_assert_eq!(axes.len(), sh.len(), "permute: axes length {} != rank {}", axes.len(), sh.len());
        {
            let mut sorted = axes.clone();
            sorted.sort();
            debug_assert!(
                sorted.iter().copied().eq(0..sh.len() as UAxis),
                "permute: axes not a valid permutation: {axes:?} for rank {}",
                sh.len()
            );
        }
        if axes.iter().copied().eq(0..sh.len() as UAxis) {
            self.retain(x);
            return x;
        }

        // Result shape: x's dims in the new axis order. The stack's expr is
        // stored (append-only slab); the transient handle is released.
        let shape_id = {
            let dims = self.shape(x);
            let permuted = crate::shape::permute(&dims, &axes);
            if permuted.is_empty() {
                ExprId::NULL
            } else {
                let stacked = self.stack(&permuted).expect("permute: failed to build shape stack");
                let expr = match self.tensors[stacked] {
                    TensorData::Symbolic { expr, .. } => expr,
                    TensorData::Eager { .. }
                    | TensorData::Leaf { .. }
                    | TensorData::PendingLeaf { .. }
                    | TensorData::Graph { .. }
                    | TensorData::GraphLeaf { .. }
                    | TensorData::Promoted { .. } => {
                        panic!("permute: shape tid {stacked} is not symbolic: {:?}", self.tensors[stacked])
                    }
                };
                self.release(stacked);
                expr
            }
        };

        match self.tensors[x] {
            TensorData::Graph { class_id, graph_id, dtype, .. }
            | TensorData::GraphLeaf { class_id, graph_id, dtype, .. }
            | TensorData::Promoted { class_id, graph_id, dtype, .. } => {
                self.assert_graph_alive(graph_id);
                let (_, class_id) = self.push_node(graph_id, Node::Permute { x: class_id, axes: axes.into_boxed_slice() });
                self.graphs[graph_id].ref_count += 1;
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> graph: tid={tid}, graph_id={graph_id:?}, class_id={class_id:?}");
                tid
            }
            TensorData::Eager { dtype, .. } | TensorData::Leaf { dtype, .. } | TensorData::PendingLeaf { dtype, .. } => {
                let (kernel_id, op_id) = self.duplicate_or_store(x, false).unwrap();
                let op_id = self.kernels[kernel_id]
                    .kernel
                    .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes: axes.into() }) });
                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, shape_id, dtype, rc: 1 });
                debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into permute must have empty outputs");
                self.kernels[kernel_id].outputs.insert(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            TensorData::Symbolic { .. } => todo!("permute of symbolic scalar"),
        }
    }

    /// Pad axis `axis` with zeros: `lp` zeros on the left, up to total
    /// length `len`; right padding is `len - lp - orig_len`. `lp` and
    /// `len` are scalar tensors.
    pub fn pad_zeros(&mut self, x: TensorId, axis: UAxis, lp: TensorId, len: TensorId) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::pad_zeros(x={x}, axis={axis}, lp={lp}, len={len})");
        self.verify_tensor_invariants();
        let rank = self.resolve_shape(x).len();
        debug_assert!((axis as usize) < rank, "pad_zeros axis {axis} out of bounds for rank {rank}");
        debug_assert!(
            self.resolve_shape(lp).is_empty() || self.resolve_shape(lp) == [1],
            "pad_zeros lp must be scalar, got {:?}",
            self.resolve_shape(lp)
        );
        debug_assert!(
            self.resolve_shape(len).is_empty() || self.resolve_shape(len) == [1],
            "pad_zeros len must be scalar, got {:?}",
            self.resolve_shape(len)
        );
        // Dtypes are fully static: shape descriptors must be integer-typed.
        debug_assert!(
            self.dtype(lp).is_int() && self.dtype(len).is_int(),
            "pad_zeros bounds must be integer-typed, got lp={:?} len={:?}",
            self.dtype(lp),
            self.dtype(len)
        );

        // Result shape: x's dims with the padded axis replaced by `len`
        // directly (total-length semantics). The stack's expr is stored
        // (append-only slab); the transient handle is released.
        let shape_id = {
            let mut dims = self.shape(x);
            dims[axis as usize] = len;
            self.retain(len);
            let stacked = self.stack(&dims).expect("pad_zeros: failed to build shape stack");
            let expr = match self.tensors[stacked] {
                TensorData::Symbolic { expr, .. } => expr,
                TensorData::Eager { .. }
                | TensorData::Leaf { .. }
                | TensorData::PendingLeaf { .. }
                | TensorData::Graph { .. }
                | TensorData::GraphLeaf { .. }
                | TensorData::Promoted { .. } => {
                    panic!("pad_zeros: shape tid {stacked} is not symbolic: {:?}", self.tensors[stacked])
                }
            };
            self.release(stacked);
            expr
        };

        match self.tensors[x] {
            TensorData::Graph { class_id, graph_id, dtype, .. }
            | TensorData::GraphLeaf { class_id, graph_id, dtype, .. }
            | TensorData::Promoted { class_id, graph_id, dtype, .. } => {
                self.assert_graph_alive(graph_id);
                let lp_class = match self.tensors[lp] {
                    TensorData::Graph { class_id, graph_id: g, .. }
                    | TensorData::GraphLeaf { class_id, graph_id: g, .. }
                    | TensorData::Promoted { class_id, graph_id: g, .. } => {
                        assert!(g == graph_id, "pad_zeros: lp belongs to a different tape scope");
                        class_id
                    }
                    TensorData::Eager { .. }
                    | TensorData::Leaf { .. }
                    | TensorData::PendingLeaf { .. }
                    | TensorData::Symbolic { .. } => self.replay_symbolic_into_graph(graph_id, lp),
                };
                let len_class = match self.tensors[len] {
                    TensorData::Graph { class_id, graph_id: g, .. }
                    | TensorData::GraphLeaf { class_id, graph_id: g, .. }
                    | TensorData::Promoted { class_id, graph_id: g, .. } => {
                        assert!(g == graph_id, "pad_zeros: len belongs to a different tape scope");
                        class_id
                    }
                    TensorData::Eager { .. }
                    | TensorData::Leaf { .. }
                    | TensorData::PendingLeaf { .. }
                    | TensorData::Symbolic { .. } => self.replay_symbolic_into_graph(graph_id, len),
                };
                let (_, class_id) = self.push_node(graph_id, Node::Pad { x: class_id, axis, lp: lp_class, len: len_class });
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                self.graphs[graph_id].ref_count += 1;
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> graph: tid={tid}, graph_id={graph_id:?}, class_id={class_id:?}");
                tid
            }
            TensorData::Eager { dtype, .. } | TensorData::Leaf { dtype, .. } | TensorData::PendingLeaf { dtype, .. } => {
                // Duplicate only when the pad actually grows the tensor AND
                // compute precedes it in the kernel (conv layers need this).
                let len_const = self
                    .resolve_symbolic(len)
                    .expect("pad_zeros: eager-arm len bound must be a resolvable scalar")
                    .as_dim()
                    .expect("pad_zeros: len bound does not evaluate to an integer");
                let grows = len_const > self.resolve_shape(x)[axis as usize];
                let force_store = match self.tensors[x] {
                    TensorData::Eager { kernel_id, op_id, .. } => {
                        grows && self.kernels[kernel_id].kernel.is_preceded_by_compute(op_id)
                    }
                    TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } | TensorData::Promoted { .. } => false,
                    TensorData::Graph { .. } | TensorData::GraphLeaf { .. } | TensorData::Symbolic { .. } => {
                        unreachable!("{:?}", self.tensors[x])
                    }
                };
                let (kernel_id, op_id) = self.duplicate_or_store(x, force_store).unwrap();

                debug_assert_eq!(
                    self.kernels[kernel_id].outputs.len(),
                    0,
                    "input into pad must have empty outputs before the bound kernels are merged"
                );
                let lp_op = self.replay_symbolic_into_kernel(kernel_id, lp);
                let len_op = self.replay_symbolic_into_kernel(kernel_id, len);

                let op_id = self.kernels[kernel_id]
                    .kernel
                    .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Pad { axis, lp: lp_op, len: len_op }) });
                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, shape_id, dtype, rc: 1 });
                self.kernels[kernel_id].outputs.insert(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            TensorData::Symbolic { .. } => todo!("pad_zeros of symbolic scalar"),
        }
    }

    /// Narrow `x` along `axis` to `[start, start + len)`.
    ///
    /// # Contract
    ///
    /// On the eager path, `x` is consumed from a kernel whose `outputs` list is **empty** at
    /// the moment of the narrow — `x` is alone in its kernel with no pending stores. This is
    /// why the kernelizer's `Node::Narrow` arm asserts the same condition on the graph side
    /// (after `consume(x)` the kernel's `outputs` must be empty). `start` and `len` are
    /// scalar integer dim-expressions; their values may be symbolic (variable-backed).
    /// Bounds are replayed symbolically into the producing kernel.
    pub fn narrow(&mut self, x: TensorId, axis: UAxis, start: TensorId, len: TensorId) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::narrow(x={x}, axis={axis}, start={start}, len={len})");
        self.verify_tensor_invariants();
        // Dtypes are fully static: shape descriptors must be integer-typed.
        debug_assert!(
            self.dtype(start).is_int() && self.dtype(len).is_int(),
            "narrow bounds must be integer-typed, got start={:?} len={:?}",
            self.dtype(start),
            self.dtype(len)
        );
        debug_assert!(
            self.resolve_shape(start).is_empty() || self.resolve_shape(start) == [1],
            "narrow start must be scalar, got {:?}",
            self.resolve_shape(start)
        );
        debug_assert!(
            self.resolve_shape(len).is_empty() || self.resolve_shape(len) == [1],
            "narrow len must be scalar, got {:?}",
            self.resolve_shape(len)
        );

        let sh = self.resolve_shape(x).to_vec();
        debug_assert!(axis < sh.len() as UAxis, "narrow: axis {axis} out of range for rank {}", sh.len());

        // Result shape: x's dims with the narrowed axis replaced by `len`.
        // The stack's expr is stored (append-only slab); the transient
        // handle is released.
        let shape_id = {
            let mut dims = self.shape(x);
            dims[axis as usize] = len;
            self.retain(len);
            let stacked = self.stack(&dims).expect("narrow: failed to build shape stack");
            let expr = match self.tensors[stacked] {
                TensorData::Symbolic { expr, .. } => expr,
                ref t => panic!("narrow: shape tid {stacked} is not symbolic: {t:?}"),
            };
            self.release(stacked);
            expr
        };

        match self.tensors[x] {
            TensorData::Graph { class_id, graph_id, dtype, .. }
            | TensorData::GraphLeaf { class_id, graph_id, dtype, .. }
            | TensorData::Promoted { class_id, graph_id, dtype, .. } => {
                self.assert_graph_alive(graph_id);
                let start_class = match self.tensors[start] {
                    TensorData::Graph { class_id, graph_id: g, .. }
                    | TensorData::GraphLeaf { class_id, graph_id: g, .. }
                    | TensorData::Promoted { class_id, graph_id: g, .. } => {
                        assert!(g == graph_id, "narrow: start belongs to a different tape scope");
                        class_id
                    }
                    TensorData::Eager { .. }
                    | TensorData::Leaf { .. }
                    | TensorData::PendingLeaf { .. }
                    | TensorData::Symbolic { .. } => self.replay_symbolic_into_graph(graph_id, start),
                };
                let len_class = match self.tensors[len] {
                    TensorData::Graph { class_id, graph_id: g, .. }
                    | TensorData::GraphLeaf { class_id, graph_id: g, .. }
                    | TensorData::Promoted { class_id, graph_id: g, .. } => {
                        assert!(g == graph_id, "narrow: len belongs to a different tape scope");
                        class_id
                    }
                    TensorData::Eager { .. }
                    | TensorData::Leaf { .. }
                    | TensorData::PendingLeaf { .. }
                    | TensorData::Symbolic { .. } => self.replay_symbolic_into_graph(graph_id, len),
                };
                let (_, class_id) =
                    self.push_node(graph_id, Node::Narrow { x: class_id, axis, start: start_class, len: len_class });
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                self.graphs[graph_id].ref_count += 1;
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> graph: tid={tid}, graph_id={graph_id:?}, class_id={class_id:?}");
                tid
            }
            TensorData::Eager { dtype, .. } | TensorData::Leaf { dtype, .. } | TensorData::PendingLeaf { dtype, .. } => {
                let (kernel_id, op_id) = self.duplicate_or_store(x, false).unwrap();
                debug_assert_eq!(
                    self.kernels[kernel_id].outputs.len(),
                    0,
                    "input into narrow must have empty outputs before the bound kernels are merged"
                );
                let start_op = self.replay_symbolic_into_kernel(kernel_id, start);
                let len_op = self.replay_symbolic_into_kernel(kernel_id, len);

                let op_id = self.kernels[kernel_id]
                    .kernel
                    .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Narrow { axis, start: start_op, len: len_op }) });
                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, shape_id, dtype, rc: 1 });
                self.kernels[kernel_id].outputs.insert(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            TensorData::Symbolic { .. } => todo!("narrow of symbolic scalar"),
        }
    }

    /// Flip tensor along axes.
    ///
    /// # Errors
    /// Returns shape error if the axes list is empty.
    pub fn flip(&mut self, x: TensorId, mut axes: Vec<UAxis>) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::flip(x={x}, axes={axes:?})");
        self.verify_tensor_invariants();

        let sh = self.resolve_shape(x).to_vec();
        if axes.is_empty() {
            return Err(ZyxError::shape_error(format!("flip: axes must not be empty for tensor of shape {sh:?}").into()));
        }
        for &axis in &axes {
            if axis >= sh.len() {
                return Err(ZyxError::shape_error(format!("Axis {axis} is out of range of rank {}", sh.len()).into()));
            }
        }
        axes.sort_unstable();
        axes.dedup();

        // Shape-preserving: the result shares x's shape expression.
        let shape_id = match self.tensors[x] {
            TensorData::Eager { shape_id, .. }
            | TensorData::Leaf { shape_id, .. }
            | TensorData::Graph { shape_id, .. }
            | TensorData::GraphLeaf { shape_id, .. }
            | TensorData::Promoted { shape_id, .. } => shape_id,
            ref t => todo!("flip of pure-slab tensor {t:?}"),
        };
        if shape_id != ExprId::NULL {}

        match self.tensors[x] {
            TensorData::Graph { class_id, graph_id, dtype, .. }
            | TensorData::GraphLeaf { class_id, graph_id, dtype, .. }
            | TensorData::Promoted { class_id, graph_id, dtype, .. } => {
                self.assert_graph_alive(graph_id);
                let (_, class_id) = self.push_node(graph_id, Node::Flip { x: class_id, axes: axes.into_boxed_slice() });
                self.graphs[graph_id].ref_count += 1;
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> graph: tid={tid}, graph_id={graph_id:?}, class_id={class_id:?}");
                Ok(tid)
            }
            TensorData::Eager { dtype, .. } | TensorData::Leaf { dtype, .. } => {
                let (kernel_id, op_id) = self.duplicate_or_store(x, false).unwrap();
                let op_id = self.kernels[kernel_id].kernel.flip(op_id, &axes);
                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, shape_id, dtype, rc: 1 });
                debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into flip must have empty outputs");
                self.kernels[kernel_id].outputs.insert(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                Ok(tid)
            }
            ref t => unreachable!("shape extraction already rejected non-slab shapes: {t:?}"),
        }
    }

    // Data can be smaller or equal lenght as number of elements in tensor.
    // If data is smaller, only first elements in tensor will be loaded.
    pub fn load<T: Scalar>(&mut self, x: TensorId, data: &mut [T]) -> Result<(), ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::load(x={x})");
        self.verify_tensor_invariants();

        // Symbolic (slab) tensors carry no buffer; resolve their value directly,
        // no kernel launch needed. They are constants/broadcast scalars, so the
        // single resolved value is written to every element of `data`.
        if self.leaf_buffer(x).is_none() {
            if let Some(c) = self.resolve_symbolic(x) {
                let v = match c.cast(T::dtype()) {
                    Constant::BF16(v) => T::from_bf16(bf16::from_le_bytes(v)),
                    Constant::F16(v) => T::from_f16(f16::from_le_bytes(v)),
                    Constant::F32(v) => T::from_f32(f32::from_le_bytes(v)),
                    Constant::F64(v) => T::from_f64(f64::from_le_bytes(v)),
                    Constant::F8E4M3(v) => T::from_f32(f8e4m3::from_bits(v).to_f32()),
                    Constant::F8E5M2(v) => T::from_f32(f8e5m2::from_bits(v).to_f32()),
                    Constant::U8(v) => T::from_u8(v),
                    Constant::U16(v) => T::from_u16(v),
                    Constant::U32(v) => T::from_u32(v),
                    Constant::U64(v) => T::from_u64(u64::from_le_bytes(v)),
                    Constant::I8(v) => T::from_i8(v),
                    Constant::I16(v) => T::from_i16(v),
                    Constant::I32(v) => T::from_i32(v),
                    Constant::I64(v) => T::from_i64(i64::from_le_bytes(v)),
                    Constant::Bool(v) => T::from_bool(v),
                };
                for d in data.iter_mut() {
                    *d = v;
                }
                return Ok(());
            }
        }

        let dt = self.dtype(x);
        if dt != T::dtype() {
            return Err(ZyxError::DTypeError(format!("loading dtype {}, but the data has dtype {dt}", T::dtype()).into()));
        }

        let shape_numel: Dim = self.resolve_shape(x).iter().product();
        if (data.len() as Dim) > shape_numel {
            return Err(ZyxError::AllocationError(
                format!("load buffer of {} elements is larger than tensor with {shape_numel} elements", data.len()).into(),
            ));
        }

        // Fast path: scalars (constants, variables and fully-symbolic
        // expressions over them) live in the tensors slab — no buffer, no
        // pool storage.
        if let Some(value) = self.resolve_symbolic(x) {
            let bytes = (data.len() * T::bit_size() as usize).div_ceil(8);
            let byte_slice = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), bytes) };
            let value_bytes = value.to_le_bytes();
            byte_slice[..value_bytes.len()].copy_from_slice(&value_bytes);
            return Ok(());
        }

        // Fast path: already realized. A pending Leaf may hold a buffer with
        // an in-place store still owed (e.g. assign into a Leaf) — run the
        // pending producer kernel before reading.
        if let TensorData::PendingLeaf { depends_on, .. } = self.tensors[x] {
            let depends_on = depends_on;
            if !depends_on.is_null() && self.kernels.contains_id(depends_on) {
                let outputs: Set<TensorId> = self.kernels[depends_on].outputs.iter().copied().collect();
                for tid in outputs {
                    self.add_store(tid)?;
                }
                if self.kernels.contains_id(depends_on)
                    && self.kernels[depends_on].outputs.is_empty()
                    && !self.kernels[depends_on].stores.is_empty()
                {
                    self.materialize_kernel(depends_on)?;
                }
            }
        }
        let Some(mut buffer_id) = self.leaf_buffer(x) else {
            let this = &mut *self;
            let pending = match this.tensors[x] {
                TensorData::PendingLeaf { depends_on, .. } => depends_on,
                // An `Eager` tensor has no producer edge: its kernel's loads
                // are leaves, each carrying its own pending producer. The
                // `kernel_id` flush below composes with that.
                TensorData::Eager { .. } => KernelId::NULL,
                TensorData::Graph { .. } => return Err(ZyxError::graph_tensor_not_realized(x)),
                // Leaf and GraphLeaf always carry a buffer (`leaf_buffer`
                // returns Some for both), so reaching this bufferless
                // fallback with either is impossible.
                TensorData::Leaf { .. } | TensorData::GraphLeaf { .. } => {
                    unreachable!("load: buffered tensor {x} has no buffer: {:?}", self.tensors[x])
                }
                TensorData::Promoted { .. } | TensorData::Symbolic { .. } => {
                    panic!("load: tensor {x} has no buffer and cannot be materialized: {:?}", self.tensors[x])
                }
            };
            if !pending.is_null() {
                let outputs: Set<TensorId> = this.kernels[pending].outputs.iter().copied().collect();
                for tid in outputs {
                    this.add_store(tid)?;
                }
            }
            // The pending store above may have realized x (a pending Leaf
            // becomes a buffer-backed Leaf). Only kernel-backed eager tensors
            // still need their producer's outputs flushed.
            if this.leaf_buffer(x).is_none() {
                let kid = match this.tensors[x] {
                    TensorData::Eager { kernel_id, .. } => kernel_id,
                    TensorData::Graph { .. } | TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => {
                        return Err(ZyxError::graph_tensor_not_realized(x));
                    }
                    ref t => panic!("load: tensor {x} has no buffer and cannot be materialized: {t:?}"),
                };
                let seen: Set<TensorId> = this.kernels[kid].outputs.iter().copied().collect();
                for tid in seen {
                    this.add_store(tid)?;
                }
            }
            let bytes = (data.len() * T::bit_size() as usize).div_ceil(8);
            let byte_slice = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), bytes) };
            let buffer_id = this.leaf_buffer(x).expect("load: tensor has no buffer after materialization");
            buffer_id.pool.pool_to_host(buffer_id.buffer_id, byte_slice)?;
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> x={x}, {:?}", self.tensors[x]);
            return Ok(());
        };

        // A store may still be pending on this tensor (assign wrote into
        // this buffer in place). Run the pending producer kernel first so
        // the buffer is up to date, then re-fetch the buffer id (the store
        // may have moved it to a device pool). Realized Leafs have no
        // pending store — they use their buffer directly.
        if let TensorData::PendingLeaf { depends_on, .. } = self.tensors[x] {
            debug_assert!(!depends_on.is_null(), "load: PendingLeaf {x} with null depends_on");
            if !depends_on.is_null() {
                let seen: Set<TensorId> = self.kernels[depends_on].outputs.iter().copied().collect();
                for tid in seen {
                    self.add_store(tid)?;
                }
                buffer_id = self.leaf_buffer(x).ok_or_else(|| {
                    ZyxError::AllocationError(format!("load: tensor {x} lost its buffer during pending store").into())
                })?;
            }
        }
        let bytes = (data.len() * T::bit_size() as usize).div_ceil(8);
        let byte_slice = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), bytes) };
        buffer_id.pool.pool_to_host(buffer_id.buffer_id, byte_slice)?;
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> x={x}, {:?}", self.tensors[x]);
        Ok(())
    }

    /// In-place assignment of `src` into `dst`.
    ///
    /// # Eager contract
    ///
    /// - `dst` must be a movement-only kernel with no other pending outputs and no stores;
    ///   its single output is `dst` itself. The kernel is **removed** and its base buffer is
    ///   re-pointed through the in-place store.
    /// - `dst`'s kernel's `loads` mixes the owning buffer with IDX_T scalar dim-variables.
    ///   Exactly one entry may be a buffer; anything else must be a dim-variable.
    ///   Zero buffers means the base was never materialized into pool storage (e.g. a
    ///   const-fill) — the assign is rejected with a `ShapeError` asking the user to
    ///   `.contiguous()` the base first.
    /// - `src` and `dst` may not share a kernel; `src` may not load `dst`'s buffer (data race).
    /// - Shape compatibility is **proved** per dim: the same constant, or the **same**
    ///   symbolic dim tensor in both operands. A variable that only agrees with the other
    ///   side by its currently bound value is not proof and is rejected.
    ///
    /// # Graph mirror
    ///
    /// `Node::Assign` in the kernelizer replays `dst`'s movement chain into `src`'s kernel
    /// and emits an in-place store of `src`'s value into `dst`'s base buffer. `dst`'s
    /// remaining consumers are re-pointed at a **fresh load kernel** for `dst` (the same
    /// contract `add_store` uses for every stored class) so that any later consumer whose
    /// `force_store` would otherwise trigger a second in-place store still finds `dst`
    /// waiting on a load, not buried inside the storing kernel.
    ///
    /// # Errors
    ///
    /// Returns [`ZyxError::DTypeError`] if the dtypes do not match.
    ///
    /// Returns [`ZyxError::ShapeError`] if the shapes do not match.
    ///
    /// Returns [`ZyxError::GraphTensorNotRealized`] if `dst` is a
    /// graph tensor that has not been realized yet.
    pub fn assign(&mut self, dst: TensorId, src: TensorId) -> Result<(), ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::assign(dst={dst}, src={src})");
        if src == dst {
            return Err(ZyxError::shape_error(format!("assign: src and dst are the same tensor {dst}").into()));
        }
        self.verify_tensor_invariants();

        let dst_dtype = self.dtype(dst);
        let src_dtype = self.dtype(src);
        if dst_dtype != src_dtype {
            return Err(ZyxError::DTypeError(format!("assign dtype mismatch: dst={dst_dtype}, src={src_dtype}").into()));
        }
        let dst_shape = self.resolve_shape(dst);
        let src_shape = self.resolve_shape(src);
        if dst_shape != src_shape {
            return Err(ZyxError::shape_error(format!("assign shape mismatch: dst={dst_shape:?}, src={src_shape:?}").into()));
        }
        // Merge-time shape-compatibility rule (same as `binary`): dst and src
        // must be PROVABLY equal shapes — per dim, the same constant or the
        // SAME symbolic dim tensor. A variable dim that only agrees with the
        // other side by its currently bound value is not proof (the slot may
        // change before launch), so the assign is rejected with an error.
        // Dynamic-shape code must propagate the same dim tensor into both
        // operands' shapes (e.g. llama propagating the kv-cache len).
        let dst_syms = self.resolve_shape_without_variables(dst);
        let src_syms = self.resolve_shape_without_variables(src);
        if !dst_syms.is_empty() && !src_syms.is_empty() && dst_syms != src_syms {
            return Err(ZyxError::shape_error(
                format!(
                    "assign: cannot prove dst and src shapes are equal: {dst_syms:?} vs {src_syms:?} — a symbolic dim must be the same dim tensor in both operands, or concrete in both"
                )
                .into(),
            ));
        }
        match self.tensors[dst] {
            TensorData::Graph { class_id: dst_cid, graph_id, .. }
            | TensorData::GraphLeaf { class_id: dst_cid, graph_id, .. }
            | TensorData::Promoted { class_id: dst_cid, graph_id, .. } => {
                // Graph-mode in-place assign: record a Node::Assign inside the tape
                // graph. The plan writes src's value into dst's buffer in-place; dst
                // is either a realized (promoted) leaf tensor or a movement view over
                // one (e.g. a slice), whose movement chain the kernelizer replays
                // into src's kernel so the store lands at the view's position.
                if dst == src {
                    return Err(ZyxError::ShapeError("assign: dst equals src (self-assign)".into()));
                }
                self.assert_graph_alive(graph_id);
                match self.tensors[src] {
                    TensorData::Graph { graph_id: g, .. }
                    | TensorData::GraphLeaf { graph_id: g, .. }
                    | TensorData::Promoted { graph_id: g, .. } => {
                        if g != graph_id {
                            panic!("tensor belongs to a different tape scope");
                        }
                    }
                    TensorData::Eager { .. }
                    | TensorData::Leaf { .. }
                    | TensorData::PendingLeaf { .. }
                    | TensorData::Symbolic { .. } => {
                        self.promote_to_graph(src, graph_id)?;
                    }
                }
                let mut dst_leaf_cid = dst_cid;
                // Walk graph to find the source of the lvalue
                let graph = &self.graphs[graph_id];
                loop {
                    match graph.nodes[graph.classes[dst_leaf_cid].nodes[0]].node {
                        Node::Pad { x, .. }
                        | Node::Flip { x, .. }
                        | Node::Expand { x, .. }
                        | Node::Reshape { x, .. }
                        | Node::Narrow { x, .. }
                        | Node::Permute { x, .. } => dst_leaf_cid = x,
                        Node::After { .. } | Node::Leaf { .. } => break,
                        ref op => unreachable!("{op:?}"),
                    }
                }
                // Resolve the base leaf through any After chain (a previous assign on
                // the same buffer) to find the base tensor. The After for this assign
                // threads onto the previous After, not the original buffer.
                let mut leaf_cid = dst_leaf_cid;
                while let Node::After { x, .. } = &graph.nodes[graph.classes[leaf_cid].nodes[0]].node {
                    leaf_cid = *x;
                }
                let dst_leaf = graph.leaf_map[&leaf_cid];

                // The Assign node keeps the ORIGINAL dst-chain and src classes; the
                // output class cid is what any later use of dst or src resolves to,
                // so both tensors are re-pointed at it.
                let src_cid = match self.tensors[src] {
                    TensorData::Graph { class_id, .. }
                    | TensorData::GraphLeaf { class_id, .. }
                    | TensorData::Promoted { class_id, .. } => class_id,
                    TensorData::Eager { .. }
                    | TensorData::Leaf { .. }
                    | TensorData::PendingLeaf { .. }
                    | TensorData::Symbolic { .. } => {
                        unreachable!("{:?}", self.tensors[src])
                    }
                };
                let (_node_id, assign_cid) = self.push_node(graph_id, Node::Assign { dst: dst_cid, src: src_cid });
                let leaf_class = self.push_node(graph_id, Node::After { x: dst_leaf_cid, dep: assign_cid }).1;
                let dst_class = self.push_node(graph_id, Node::After { x: dst_cid, dep: assign_cid }).1;
                for (tid, class_id) in [(dst_leaf, leaf_class), (dst, dst_class)] {
                    match &mut self.tensors[tid] {
                        TensorData::Graph { class_id: c, .. }
                        | TensorData::GraphLeaf { class_id: c, .. }
                        | TensorData::Promoted { class_id: c, .. } => *c = class_id,
                        TensorData::Eager { .. }
                        | TensorData::Leaf { .. }
                        | TensorData::PendingLeaf { .. }
                        | TensorData::Symbolic { .. } => {
                            panic!("assign: tensor {tid} has no graph class to re-point: {:?}", self.tensors[tid])
                        }
                    }
                }
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> assign_cid={assign_cid:?}");
                return Ok(());
            }
            TensorData::Eager { .. } | TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } | TensorData::Symbolic { .. } => {}
        }
        // Merge dst's (movement-only) kernel ops into src's kernel, then store
        // src's value into dst's base buffer in-place. A Leaf src has no
        // kernel: mint a fresh load kernel for its buffer — the same clean
        // placement the old load-kernel path produced.
        let (src_kid, src_op) = match self.tensors[src] {
            TensorData::Eager { kernel_id, op_id, .. } | TensorData::Promoted { kernel_id, op_id, .. } => (kernel_id, op_id),
            TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => self.new_kernel_from_leaf(src),
            TensorData::Graph { .. } | TensorData::GraphLeaf { .. } | TensorData::Symbolic { .. } => {
                panic!("assign: src {src} is not an eager/promoted tensor: {:?}", self.tensors[src])
            }
        };
        // A pending dst still owes its value to a live store kernel: run the
        // producer first so dst has a buffer, then take the Leaf path below.
        // depends_on is never null by state-machine invariant (a PendingLeaf
        // is always created with and consumed by a live kernel).
        if let TensorData::PendingLeaf { depends_on, .. } = self.tensors[dst] {
            debug_assert!(!depends_on.is_null(), "assign: PendingLeaf {dst} with null depends_on");
            let seen: Set<TensorId> = self.kernels[depends_on].outputs.iter().copied().collect();
            for tid in seen {
                self.add_store(tid)?;
            }
        }
        // A bare Leaf dst (realized buffer, no view kernel): write src's value
        // into dst's buffer in-place. A fresh kernel loads src (Leaf src) and
        // stores into a GlobalMut param bound to dst's buffer; an eager src
        // extends its own kernel instead. dst becomes a pending Leaf — reads
        // materialize the store first.
        if let TensorData::Leaf { shape_id: dst_shape_id, buffer: dst_buf, rc: dst_rc, .. } = self.tensors[dst] {
            let dtype = self.dtype(dst);
            let (kernel_id, src_op) = match self.tensors[src] {
                TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => {
                    let (kid, op) = self.new_kernel_from_leaf(src);
                    (kid, op)
                }
                TensorData::Eager { kernel_id, op_id, .. } | TensorData::Promoted { kernel_id, op_id, .. } => (kernel_id, op_id),
                TensorData::Graph { .. } | TensorData::GraphLeaf { .. } | TensorData::Symbolic { .. } => {
                    panic!("assign: src {src} is not an eager/promoted tensor: {:?}", self.tensors[src])
                }
            };
            let dst_shape_op = self.replay_expr(kernel_id, dst_shape_id);
            let mut_param =
                self.kernels[kernel_id].kernel.push_back(Op::Param { dtype, kind: ParamKind::GlobalMut, shape: dst_shape_op });
            self.kernels[kernel_id].kernel.store(mut_param, src_op, OpId::NULL);
            self.kernels[kernel_id].stores.push(dst);
            // dst becomes pending: the store kernel owns the value now, mutating
            // the kept buffer in place — nothing is released. Materialize
            // passes the kept buffer into the kernel as the store target.
            self.tensors[dst] = TensorData::PendingLeaf {
                old_buffer: Some(dst_buf),
                depends_on: kernel_id,
                shape_id: dst_shape_id,
                dtype,
                rc: dst_rc,
            };
            return Ok(());
        }
        let (dst_kid, dst_op) = match self.tensors[dst] {
            TensorData::Eager { kernel_id, op_id, .. } | TensorData::Promoted { kernel_id, op_id, .. } => (kernel_id, op_id),
            ref t => panic!("assign: dst {dst} is not an eager/promoted tensor: {t:?}"),
        };
        // The destination must be a movement-only view kernel with no outputs
        // other than dst itself (dst may appear multiple times, once per
        // cloned handle).
        if self.kernels[dst_kid].outputs.iter().any(|&e| e != dst) {
            return Err(ZyxError::ShapeError(
                format!("assign: dst kernel {dst_kid:?} has other outputs {:?}, only dst allowed", self.kernels[dst_kid].outputs)
                    .into(),
            ));
        }
        for op in self.kernels[dst_kid].kernel.ops.values() {
            if !matches!(op.op, Op::Param { .. } | Op::Move { .. } | Op::Const(_) | Op::Stack { .. }) {
                return Err(ZyxError::ShapeError(
                    format!("assign: dst kernel {dst_kid:?} has unsupported op {:?}, only movement ops allowed", op.op).into(),
                ));
            }
        }
        if src_kid == dst_kid {
            return Err(ZyxError::ShapeError(
                format!("assign: src and dst share kernel {dst_kid:?}; dst must be a separate movement-only kernel").into(),
            ));
        }
        if !self.kernels[dst_kid].stores.is_empty() {
            return Err(ZyxError::ShapeError(
                format!("assign: dst kernel {dst_kid:?} has stores {}; expected none", self.kernels[dst_kid].stores.len()).into(),
            ));
        }
        if self.kernels[src_kid].loads.contains(&dst) {
            return Err(ZyxError::ShapeError(
                format!("assign: src kernel {dst_kid:?} loads dst tensor, not allowed to avoid data races").into(),
            ));
        }

        // Validate the dst kernel's loads before using them: a failed
        // validation must leave the kernel intact so the dst view tensor (still
        // pointing at dst_kid) stays valid and its later drop does not index a
        // deleted kernel.
        let dst_kernel_loads = self.kernels[dst_kid].loads.clone();
        let dst_org = {
            let mut buffer_loads =
                dst_kernel_loads.iter().copied().filter(|&t| {
                    !matches!(self.tensors[t], TensorData::Symbolic { expr, .. } if matches!(self.exprs[expr], Expr::Variable { .. }))
                });
            match (buffer_loads.next(), buffer_loads.next()) {
                (Some(t), None) => t,
                (None, _) => {
                    return Err(ZyxError::ShapeError(
                        "assign: dst kernel has no backing buffer; its base was never materialized \
                     into pool storage — call `.contiguous()` on it before assign"
                            .into(),
                    ));
                }
                _ => return Err(ZyxError::ShapeError(
                    "assign: dst kernel contains more than one buffer load; dst must be a movement-only view of exactly one base"
                        .into(),
                )),
            }
        };

        // The base's backing store may still be deferred (e.g. `contiguous`
        // marks the store but leaves it unmaterialized for fusion). Assign
        // writes in-place, so the storage must exist NOW: force-materialize
        // the producer kernel that holds the pending store (after the
        // replayed store is in place, see below).
        match self.tensors[dst_org] {
            TensorData::PendingLeaf { depends_on, .. } if !depends_on.is_null() => {
                assert!(
                    depends_on != src_kid,
                    "assign: dst base {dst_org} is pending on src's kernel {src_kid:?}; assign would interleave with its own store"
                );
            }
            TensorData::Eager { .. } | TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => {}
            ref t => panic!("assign: dst base {dst_org} in unexpected state {t:?}"),
        }
        // dst's movement-only kernel STAYS ALIVE: dst remains a live view over
        // dst_org and keeps reading the base buffer through it. The replay
        // below only reads the kernel's ops; the in-place store lands in
        // src's kernel instead. Nothing about dst or its kernel is mutated,
        // and no refcounts change — after the store is force-realized the
        // view simply observes the post-assign buffer on its next read.
        let kernel = self.kernels[dst_kid].kernel.clone();
        let loads = self.kernels[dst_kid].loads.clone();
        for t in &loads {
            assert!(
                *t == dst_org
                    || matches!(self.tensors[*t], TensorData::Symbolic { expr, .. } if matches!(self.exprs[expr], Expr::Variable { .. })),
                "assign: dst kernel load {t} is neither the buffer nor a known variable"
            );
        }
        // `loads` is positionally aligned with the kernel's Param defines
        // (invariant: len == number of defines); verify before replaying.
        {
            let mut n_params = 0usize;
            let mut p = kernel.head;
            while !p.is_null() {
                if matches!(&kernel.ops[p].op, Op::Param { .. }) {
                    n_params += 1;
                }
                p = kernel.next_op(p);
            }
            assert_eq!(n_params, loads.len(), "assign: dst kernel param/loads count mismatch");
        }
        let mut dst_param = dst_op;
        for _ in 0..100 {
            match kernel.ops[dst_param].op {
                Op::Move { x, .. } => {
                    dst_param = x;
                }
                Op::Param { .. } => {
                    break;
                }
                _ => {}
            }
        }

        // Replay dst's movement chain into src's kernel. The replayed base
        // param becomes the (mutable) store target; the last replayed
        // movement op yields dst's final value within src's kernel.
        let mut op_map = Map::default();
        // Load classes for src's kernel, aligned to every define replayed
        // below, in define order. Variable defines keep their variable tid;
        // the base buffer param gets dst_org.
        let mut new_def_loads: Vec<TensorId> = Vec::new();
        // Pass A: transitive dependency closure over the removed kernel's ops
        // via `parameters()` — this pulls in every referenced id, including
        // `Param { shape }` descriptors and `MoveOp` internals (narrow
        // start/len, pad lp/len, reshape/expand shapes). Nothing may be left
        // dangling: ids from the removed kernel would silently collide with
        // unrelated ops in src's kernel.
        let mut required: Set<OpId> = Set::default();
        {
            let mut stack: Vec<OpId> = Vec::new();
            let mut oid = kernel.head;
            while !oid.is_null() {
                stack.push(oid);
                oid = kernel.next_op(oid);
            }
            while let Some(id) = stack.pop() {
                if !required.insert(id) {
                    continue;
                }
                stack.extend(kernel.ops[id].op.parameters());
            }
            debug_assert!(stack.is_empty(), "assign replay: dependency walk did not finish");
        }
        // Pass B: copy in ORIGINAL head order. This preserves the head-order
        // relation between defines and `loads` (params appended in the same
        // order they appear in `loads`) and guarantees every dependency is
        // copied before its user.
        let mut def_i = 0usize;
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            if required.contains(&op_id) {
                let mut op = kernel.ops[op_id].op.clone();
                if let Op::Move { x, .. } = &mut op {
                    if op_map.get(x).is_none() {
                        // this is the move on the load
                        *x = op_map[&dst_param];
                    }
                }
                // Single remap pass: `parameters_mut` covers the Move's `x`
                // AND its `MoveOp` internals (reshape/expand shapes, pad
                // lp/len, narrow start/len) — no second mop.remap, that would
                // look up already-remapped ids.
                for p in op.parameters_mut() {
                    *p =
                        op_map.get(p).copied().expect("assign replay: dependency was not copied before its user despite closure");
                }
                let mut new_def_load: Option<TensorId> = None;
                match &mut op {
                    Op::Param { kind, .. } => {
                        // Assign turns dst's base from a load into a PURE
                        // STORE: it must NOT register in loads — its buffer
                        // slot comes via `stores` instead (see KernelData docs).
                        if op_id == dst_param {
                            *kind = ParamKind::GlobalMut;
                        }
                        assert!(
                            matches!(kind, ParamKind::GlobalMut | ParamKind::Variable),
                            "assign: unexpected param kind {kind:?} in dst movement kernel"
                        );
                        if *kind != ParamKind::GlobalMut {
                            new_def_load = Some(loads[def_i]);
                        }
                        def_i += 1;
                    }
                    _ => {}
                }
                let new_id = self.kernels[src_kid].kernel.push_back(op);
                if let Some(load) = new_def_load {
                    new_def_loads.push(load);
                }
                op_map.insert(op_id, new_id);
            }
            op_id = kernel.next_op(op_id);
        }

        let dst_op = op_map.get(&dst_op).copied().unwrap_or(op_map[&dst_param]);
        // Store src's value into dst's base buffer through the replayed chain.
        self.kernels[src_kid].kernel.store(dst_op, src_op, OpId::NULL);
        debug_assert!(
            matches!(self.tensors[dst_org], TensorData::Leaf { .. } | TensorData::PendingLeaf { .. }),
            "assign: dst base {dst_org} is not a leaf/pending leaf"
        );
        self.kernels[src_kid].stores.push(dst_org);
        // Register every replayed define's load in define order. Variables
        // only — the GlobalMut base is a PURE STORE now and must not appear
        // in loads (its buffer slot comes via `stores`). dst's kernel stays
        // alive and keeps its own load edges, so these are NEW edges on
        // src's kernel — each holds one rc (released when src's kernel dies
        // or materializes), same convention as `new_kernel_from_leaf`.
        for load in new_def_loads {
            debug_assert!(
                matches!(
                    self.tensors[load],
                    TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } | TensorData::Symbolic { .. }
                ),
                "assign: replayed load {load} is not a leaf/pending leaf/symbolic"
            );
            self.kernels[src_kid].loads.push(load);
            self.retain(load);
        }
        #[cfg(debug_assertions)]
        {
            let kd = &self.kernels[src_kid];
            assert!(!kd.loads.contains(&dst_org), "assign: GlobalMut store target {dst_org} leaked into loads");
            // loads ↔ non-mut defines, aligned in head order.
            let mut n_non_mut = 0usize;
            let mut p = kd.kernel.head;
            while !p.is_null() {
                if let Op::Param { kind, .. } = &kd.kernel.ops[p].op {
                    if *kind != ParamKind::GlobalMut {
                        n_non_mut += 1;
                    }
                }
                p = kd.kernel.next_op(p);
            }
            assert_eq!(n_non_mut, kd.loads.len(), "assign: loads/defines alignment broken for kernel {:?}", src_kid);
        }

        // The base's backing store may still be deferred (e.g. `contiguous`
        // marks the store but leaves it unmaterialized for fusion). Assign
        // writes in-place, so the storage must exist NOW: force-materialize
        // the producer kernel that holds the pending store, then assert the
        // buffer actually exists.
        match self.tensors[dst_org] {
            TensorData::PendingLeaf { depends_on, .. } if !depends_on.is_null() => {
                for out in self.kernels[depends_on].outputs.clone() {
                    self.add_store(out)?;
                }
            }
            TensorData::Eager { .. } | TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => {}
            ref t => panic!("assign: dst base {dst_org} in unexpected state {t:?}"),
        }
        debug_assert!(self.leaf_buffer(dst_org).is_some(), "assign: dst base {dst_org} has no buffer after materialization");
        // dst_org is a realized Leaf now: re-home it as a pending store. Its
        // buffer is kept as src's kernel's in-place mutation target, owned by
        // src's kernel from here until materialize turns it back into a Leaf.
        let (dst_shape_id, dst_dtype, dst_rc, dst_buf) = match self.tensors[dst_org] {
            TensorData::Leaf { shape_id, dtype, rc, buffer } => (shape_id, dtype, rc, buffer),
            ref t => panic!("assign: dst base {dst_org} is not a realized leaf: {t:?}"),
        };
        self.tensors[dst_org] = TensorData::PendingLeaf {
            old_buffer: Some(dst_buf),
            depends_on: src_kid,
            shape_id: dst_shape_id,
            dtype: dst_dtype,
            rc: dst_rc,
        };
        // Torch semantics: the in-place write is a completed fact before
        // assign returns — force-realize the store kernel NOW so every view
        // of the base (dst and any other view over dst_org) observes the
        // write on its next read, with no per-view ordering bookkeeping.
        let outputs: Vec<TensorId> = self.kernels[src_kid].outputs.iter().copied().collect();
        if outputs.is_empty() {
            self.materialize_kernel(src_kid)?;
        } else {
            for tid in outputs {
                self.add_store(tid)?;
            }
        }

        Ok(())
    }

    /// This function deinitializes the whole runtime, deallocates all allocated memory and deallocates all caches
    /// It does not reset the rng and it does not change search and training fields
    #[allow(unused)]
    pub fn deinitialize(&mut self) {
        #[cfg(feature = "time")]
        {
            let lock = crate::ET.lock();
            let mut timings: Vec<_> = lock.iter().map(|(name, &(total_us, count))| (name.clone(), total_us, count)).collect();
            timings.sort_by_key(|a| std::cmp::Reverse(a.1));
            println!("\n=== Timing Info (sorted by total time, descending) ===");
            for (name, total_us, count) in timings {
                let per_call = total_us.checked_div(count).unwrap_or(0);
                println!("{name}: {total_us}us total, {per_call}us/call ({count} calls)");
            }
        }
        //println!("DEINIT runtime");
        self.tensors = Slab::new();
        self.kernels = Slab::new();
    }

    pub const fn manual_seed(&mut self, seed: u64) {
        self.rng = Rng::seed_from_u64(seed);
    }

    /// Returns the maximum free bytes available across all memory pools.
    pub fn free_memory(&mut self) -> Dim {
        Dev::all().iter().map(|d| d.pool().free_bytes()).max().unwrap_or(0)
    }
}

impl Runtime {
    /// Ensures `x` lives alone in a **store-free** kernel and returns its `(kid, op_id)` in
    /// that kernel — the canonical "clean" placement for a tensor that another op (e.g.
    /// `narrow`, `permute`, `transpose`) is about to merge into.
    ///
    /// Steps:
    /// 1. If the producer kernel already has stores, or `x`'s op is preceded by a reduce,
    ///    or `force_store` was set: call [`add_store`] so `x` lands in a fresh load kernel.
    /// 2. **Duplicate** `x`'s dependency chain into a brand-new kernel with empty
    ///    `outputs` and no stores, retargeting the variables/loads correctly.
    ///
    /// The returned kernel is therefore a **fresh, store-free, outputs-empty** kernel that
    /// contains `x` and nothing else pending — this is the contract that makes
    /// `Runtime::narrow`'s "input into narrow must have empty outputs" assertion hold
    /// unconditionally, and that the kernelizer's `Node::Narrow` arm mirrors.
    ///
    /// # Runtime duplication semantics
    ///
    /// Runtime ops never *consume*: they only create new tensors. The original
    /// kernel is left completely untouched — it keeps all its ops, all its
    /// loads and `x` stays in its `outputs` (the single producer). The fresh
    /// kernel **recomputes** `x`'s chain from the shared input loads; its
    /// `outputs` stays empty because it produces values only for the op the
    /// caller is about to build on it. The only bookkeeping is reference
    /// sharing: every load the fresh kernel reads (one `new_loads` entry per
    /// occurrence) gains one reference.
    fn duplicate_or_store(&mut self, x: TensorId, force_store: bool) -> Result<(KernelId, OpId), ZyxError> {
        let (mut kid, mut op_id) = match self.tensors[x] {
            TensorData::Eager { kernel_id, op_id, .. } | TensorData::Promoted { kernel_id, op_id, .. } => (kernel_id, op_id),
            // A Leaf or PendingLeaf has no kernel: mint a fresh load kernel
            // for its buffer — the clean placement contract. The load edge
            // resolves a pending producer at launch via the recursive flush.
            TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => {
                return Ok(self.new_kernel_from_leaf(x));
            }
            TensorData::Graph { .. } | TensorData::GraphLeaf { .. } | TensorData::Symbolic { .. } => {
                panic!("duplicate_or_store: tensor {x} is not an eager/promoted tensor: {:?}", self.tensors[x])
            }
        };

        let contains_stores = self.kernels[kid].kernel.contains_stores();
        let preceded_by_reduce = self.kernels[kid].kernel.is_preceded_by_reduce(op_id);
        if force_store || contains_stores || preceded_by_reduce {
            self.add_store(x)?;
            // add_store re-homes x onto... nothing: x becomes a **Leaf**.
            // Mint the fresh load kernel directly — it is already the clean
            // placement (store-free, outputs-empty), no duplication needed.
            // A (pending/graph) leaf is a leaf: the load edge resolves the
            // pending producer at launch via the recursive flush.
            (kid, op_id) = match self.tensors[x] {
                TensorData::Eager { kernel_id, op_id, .. } | TensorData::Promoted { kernel_id, op_id, .. } => (kernel_id, op_id),
                TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } | TensorData::GraphLeaf { .. } => {
                    return Ok(self.new_kernel_from_leaf(x));
                }
                TensorData::Graph { .. } | TensorData::Symbolic { .. } => {
                    panic!("duplicate_or_store: tensor {x} is not an eager/promoted tensor: {:?}", self.tensors[x])
                }
            };
        }

        debug_assert!(self.kernels[kid].stores.is_empty(), "duplicated kernel must not have stores");

        let old_loads = self.kernels[kid].loads.clone();
        // Pure duplication: the original kernel (ops, loads, outputs) is
        // untouched; the fresh kernel clones x's chain and recomputes it.
        let (kernel, op_id, new_loads) = self.kernels[kid].kernel.duplicate_subkernel(op_id, &old_loads);

        // Each `new_loads` occurrence is an additional reader of the load
        // tensor: one extra reference per occurrence.
        for &tid in &new_loads {
            self.retain(tid);
        }

        kid = self.kernels.push(KernelData { outputs: Set::default(), loads: new_loads, stores: Vec::new(), kernel });

        Ok((kid, op_id))
    }

    /// Merge `merge_kid`'s kernel into `keep_kid`'s kernel, repointing every
    /// tensor and op that referenced `merge_kid` at its remapped `keep_kid`
    /// equivalents, and folding `merge_kid`'s outputs/loads/stores into the
    /// keep kernel's. Returns the op-id map (merge-kernel op -> keep-kernel op)
    /// so callers can remap op ids they captured before the merge.
    ///
    /// The merge kernel must be store-free: any kernel with stores must be
    /// realized (`add_store`) by the caller *before* merging. `keep_kid` and
    /// `merge_kid` must differ.
    fn merge_kernel(&mut self, keep_kid: KernelId, merge_kid: KernelId) -> Result<Map<OpId, OpId>, ZyxError> {
        debug_assert_ne!(keep_kid, merge_kid, "merge_kernel: cannot merge a kernel into itself");
        debug_assert!(
            self.kernels[merge_kid].stores.is_empty(),
            "merge_kernel: merge kernel {merge_kid:?} has stores; add_store them before merging"
        );

        let KernelData { outputs: merge_outputs, loads: merge_loads, stores: merge_stores, kernel } =
            unsafe { self.kernels.remove_and_return(merge_kid) };
        let Kernel { ops: merge_ops, head: merge_head, .. } = kernel;

        // Seed the Const map from keep's existing ops (chain walk — slab
        // order is NOT chain order after prunes, so a slab walk could hand
        // out dead ops as survivors). Merged Consts then collapse onto
        // keep's or earlier-merged identical values instead of piling up.
        let mut const_map: Map<Constant, OpId> = Map::with_hasher(BuildHasherDefault::new());
        {
            let keep = &self.kernels[keep_kid].kernel;
            let mut i = keep.head;
            while !i.is_null() {
                if let Op::Const(c) = keep.ops[i].op {
                    const_map.insert(c, i);
                }
                i = keep.ops[i].next;
            }
        }

        let mut op_map: Map<OpId, OpId> = Map::with_hasher(BuildHasherDefault::new());
        let mut i = merge_head;
        while !i.is_null() {
            let mut op = merge_ops[i].op.clone();
            for param in op.parameters_mut() {
                if let Some(&new_param) = op_map.get(param) {
                    *param = new_param;
                }
            }
            // Const dedup: an identical Const (value + dtype) already in
            // keep or earlier in this merge is reused — its consumers and
            // any tensor holding it as op_id repoint to the survivor below.
            if let Op::Const(c) = op {
                if let Some(&survivor) = const_map.get(&c) {
                    op_map.insert(i, survivor);
                    i = merge_ops[i].next;
                    continue;
                }
            }
            let new_op_id = self.kernels[keep_kid].kernel.push_back(op);
            if let Op::Const(c) = merge_ops[i].op {
                const_map.insert(c, new_op_id);
            }
            op_map.insert(i, new_op_id);
            i = merge_ops[i].next;
        }

        // Repoint every tensor whose producer was the merge kernel.
        for (_tid, t_data) in self.tensors.iter_mut() {
            let (kernel_id, op_id) = match t_data {
                TensorData::Eager { kernel_id, op_id, .. } | TensorData::Promoted { kernel_id, op_id, .. } => (kernel_id, op_id),
                _ => continue,
            };
            if *kernel_id == merge_kid {
                debug_assert_ne!(keep_kid, merge_kid);
                debug_assert!(op_map.contains_key(op_id), "merge_kernel: holder {op_id:?} not in merge kernel op list");
                *kernel_id = keep_kid;
                *op_id = op_map[op_id];
            }
        }

        // Stores folded into `keep_kid` now live there: a pending store's
        // `depends_on` (the producer kernel that owes it) must follow. Without
        // this, the recursive materializer follows the stale `depends_on`
        // (the removed merge kernel) and the tensor is never realized.
        let store_tids: Vec<TensorId> = merge_stores.clone();
        let keep_data = &mut self.kernels[keep_kid];
        keep_data.outputs.extend(merge_outputs);
        // Load entries move with their ops — an ownership transfer between
        // kernels. The edges (and their rc counts) persist unchanged; a retain
        // here would double-count every moved entry.
        keep_data.loads.extend(merge_loads.iter().copied());
        keep_data.stores.extend(merge_stores);
        for &tid in &store_tids {
            if let TensorData::PendingLeaf { depends_on, .. } = &mut self.tensors[tid] {
                if *depends_on == merge_kid {
                    *depends_on = keep_kid;
                }
            }
        }

        // Inventory invariant: after the merge every repointed tensor is in
        // `keep_kid`'s `outputs` and NOWHERE else — and each listing agrees
        // with the tensor's own `kernel_id`.
        #[cfg(debug_assertions)]
        {
            let outputs: Vec<TensorId> = self.kernels[keep_kid].outputs.iter().copied().collect();
            for tid in &outputs {
                match self.tensors.get(*tid) {
                    Some(TensorData::Eager { kernel_id, .. }) | Some(TensorData::Promoted { kernel_id, .. }) => {
                        debug_assert_eq!(
                            *kernel_id, keep_kid,
                            "merged output tid {tid} has kernel_id {kernel_id:?}, not keep {keep_kid:?}"
                        );
                    }
                    Some(t) => panic!("merge_kernel: keep kernel output tid {tid} has unexpected tensor data {t:?}"),
                    None => panic!("merge_kernel: keep kernel output tid {tid} was deleted from the slab (stale outputs entry)"),
                }
                let count = self.kernels.values().filter(|kd| kd.outputs.contains(tid)).count();
                debug_assert!(count <= 1, "merged output tid {tid} is listed in {count} kernels' outputs");
            }
        }

        Ok(op_map)
    }

    /// Materializes `x`'s value into its kernel's storage and re-exposes it as a
    /// **Leaf** for any remaining consumers (mirrors the kernelizer's `add_store` contract).
    ///
    /// After this call, `x`'s kernel has the new entry in `stores`, and `x`
    /// becomes a `Leaf` — no load kernel is created. This is the canonical
    /// split point: a class that has been stored is no longer fused into a
    /// producer kernel — any subsequent consumer that would otherwise have to
    /// materialize again loads the Leaf's buffer via [`Runtime::new_kernel_from_leaf`].
    ///
    /// Called by `duplicate_or_store` (when the producer kernel already stores or its op is
    /// preceded by a reduce) and by `contiguous`'s cast-shim.
    pub fn add_store(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let (kid, op_id, pending) = match self.tensors[x] {
            TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id, KernelId::NULL),
            TensorData::Promoted { kernel_id, op_id, .. } => (kernel_id, op_id, KernelId::NULL),
            ref t => panic!("add_store: tensor {x} is not an eager/promoted tensor: {t:?}"),
        };

        // Remove x from the kernel's outputs (it is being stored).
        debug_assert!(self.kernels[kid].outputs.contains(&x), "add_store called for tid not in outputs");
        self.kernels[kid].outputs.remove(&x);

        // Only add StoreView if x isn't already realized or pending
        let dtype = self.dtype(x);
        let add_store = self.leaf_buffer(x).is_none() && pending.is_null();
        let pending = if add_store {
            // Invariant: a kernel must never both load and store the same tensor
            debug_assert!(!self.kernels[kid].loads.contains(&x), "kernel {kid:?} both loads and stores tid {x}");

            let store_shape_id = self.kernels[kid].kernel.stack_shape_dims(op_id);
            let dst_id =
                self.kernels[kid].kernel.push_back(Op::Param { dtype, kind: ParamKind::GlobalMut, shape: store_shape_id });
            self.kernels[kid].kernel.store(dst_id, op_id, OpId::NULL);
            self.kernels[kid].stores.push(x);
            kid
        } else {
            pending
        };
        let outputs_empty = self.kernels[kid].outputs.is_empty();

        // x becomes a **LeafPending**: the stored value lives in the
        // (possibly pending) buffer — no load kernel, no outputs
        // registration. If the store is still pending (producer not launched
        // yet), `depends_on` records the kernel that owes it; the buffer is
        // stored on the variant when that kernel materializes.
        let (shape_id, rc) = match self.tensors[x] {
            TensorData::Eager { shape_id, rc, .. }
            | TensorData::Graph { shape_id, rc, .. }
            | TensorData::Promoted { shape_id, rc, .. } => {
                // NOTE: no `retain(shape_id)` here — the re-home keeps the same
                // slab entry and the same `shape_id` field, so the existing
                // edge count persists unchanged. A retain would double-count
                // the shape edge and orphan it at this tensor's death.
                (shape_id, rc)
            }
            ref t => panic!("add_store: tensor {x} is not a kernel-backed tensor: {t:?}"),
        };
        self.tensors[x] = TensorData::PendingLeaf { old_buffer: None, depends_on: pending, shape_id, dtype, rc };

        if outputs_empty {
            self.materialize_kernel(kid)?;
        }
        Ok(())
    }

    /// Autotune (or fetch from cache) a compiled program for `kernel`.
    ///
    /// `buffers` are the complete launch arguments from the caller,
    /// positionally bound: read-only defines (`Global` buffers and scalar
    /// `Variable` values) in head order, then `GlobalMut` stores in head
    /// order. Every buffer is preallocated (eager path) or freshly
    /// allocated by the graph caller; every variable carries its actual
    /// runtime value. Nothing is invented here.
    pub fn get_or_autotune(&mut self, kernel: Kernel, buffers: &[LaunchArg]) -> Result<(DeviceProgramId, u64), ZyxError> {
        let kernel_id = if let Some(&cached_kid) = self.kernel_map.get(&kernel) {
            if let Some(&program_id) = self.programs.get(&cached_kid) {
                let pid = ProgramId { dev: kernel.dev, program_id };
                let timing = self.timings.get(&pid).copied().unwrap_or(10_000_000_000);
                return Ok((program_id, timing));
            }
            // Kernel cached but program gone: re-run the search.
            cached_kid
        } else {
            let kernel_id =
                KernelId::from(self.kernel_map.values().copied().max().map_or(0, |id| usize::from(id).checked_add(1).unwrap()));
            let newly_inserted = self.kernel_map.insert(kernel.clone(), kernel_id).is_none();
            assert!(newly_inserted);
            kernel_id
        };

        if crate::debug_mask().sched() {
            kernel.debug();
        }

        let device_id = kernel.dev;
        #[cfg(feature = "viz")]
        let sched_kernel = kernel.clone();

        // Seed preparation happens OUTSIDE the beam search: linearize + the
        // basic post-linearize passes, then the epilogue runs 3x so loop
        // folding converges before the search starts.
        {
            let mut n_params = 0usize;
            let mut op_id = kernel.head;
            while !op_id.is_null() {
                if matches!(kernel.ops[op_id].op, Op::Param { .. }) {
                    n_params += 1;
                }
                op_id = kernel.next_op(op_id);
            }
            debug_assert_eq!(buffers.len(), n_params, "caller arg count must match kernel param count");
        }
        let dev_info = device_id.info();
        let mut base = kernel;
        base.linearize();
        base.common_subexpression_elimination();
        base.dead_code_elimination();
        base.instruction_schedule();
        {
            let global_indices = base.get_group_indices();
            let max_global_dims = dev_info.max_global_work_dims.len();
            if global_indices.len() > max_global_dims {
                let n = global_indices.len() + 1 - max_global_dims;
                let indices: Vec<OpId> = global_indices.values().copied().take(n).collect();
                base.merge_indices(&indices);
            }
            base.renumber_indices();
            base.verify();
        }
        base.delete_zero_len_indices();
        base.renumber_indices();
        for _ in 0..3 {
            base.default_epilogue();
        }

        let beam_search = crate::backend::autotune_config();
        let (winner, timing) = beam_search.run_(
            self,
            [base],
            buffers,
            &Kernel::default_optimizations(),
            Kernel::default_epilogue,
            Kernel::base_cost,
        )?;

        let program_id = device_id.compile(&winner, crate::debug_mask().asm())?;
        self.programs.insert(kernel_id, program_id);
        self.timings.insert(ProgramId { dev: device_id, program_id }, timing);

        #[cfg(feature = "viz")]
        {
            let kc = {
                crate::viz::KernelCapture {
                    sched_kernel,
                    winner: winner.clone(),
                    dev_info: dev_info.clone(),
                    device_label: device_id.name(),
                    cc: match device_id {
                        Dev::Cuda(_) => Some(dev_info.cc),
                        _ => None,
                    },
                    has_openmp: dev_info.has_openmp,
                }
            };
            self.viz.record(ProgramId { dev: device_id, program_id }, kc);
        }

        Ok((program_id, timing))
    }

    /// Materializes a kernel by compiling, launching, then creating load kernels
    /// for each output so the tensors remain usable in further graph construction.
    /// The kernel is consumed (removed from the slab) and cached in
    /// `kernel_map`/`programs` for reuse.
    ///
    /// # Convention
    /// The way to materialize a kernel is NOT to call this method directly, but
    /// to `add_store` all of the kernel's outputs: `add_store` moves each tid
    /// out of `outputs` and, once the last one is stored (`outputs` empty),
    /// materializes the kernel automatically. Calling `materialize_kernel`
    /// directly on a kernel that still has unstored outputs trips the
    /// `all outputs must be stored` debug_assert below.
    ///
    /// # Invariant
    /// A kernel must never both load and store the same tensor (prevents aliasing).
    /// Picks the fastest device (by compute) that has at least `bytes` free.
    fn pick_device(&self, bytes: Dim) -> Result<Dev, ZyxError> {
        let mut devs: Vec<Dev> =
            Dev::all().into_iter().filter(|&dev| !dev.aot_only() && dev.pool().free_bytes() >= bytes).collect();
        if devs.is_empty() {
            return Err(ZyxError::AllocationError(format!("no device with {bytes} bytes free").into()));
        }
        devs.sort_unstable_by_key(|&dev| dev.free_compute());
        devs.reverse();
        Ok(devs[0])
    }

    pub(crate) fn materialize_kernel(&mut self, kid: KernelId) -> Result<(), ZyxError> {
        // Flush pending loads first: a PendingLeaf's value is owed by a live
        // kernel, so launch it (via add_store over its outputs, the runtime
        // custom) before anything below touches buffers. Recursive: the owed
        // kernel's own materialization flushes its pending loads the same way.
        let pending_kids: Vec<KernelId> = self.kernels[kid]
            .loads
            .iter()
            .filter_map(|&tid| match self.tensors[tid] {
                TensorData::PendingLeaf { depends_on, .. } => {
                    debug_assert!(!depends_on.is_null(), "materialize: PendingLeaf {tid} with null depends_on");
                    Some(depends_on)
                }
                _ => None,
            })
            .collect();
        for pending_kid in pending_kids {
            let seen: Set<TensorId> = self.kernels[pending_kid].outputs.iter().copied().collect();
            for tid in seen {
                self.add_store(tid)?;
            }
        }
        // Every non-scalar load now has a buffer.
        for &tid in &self.kernels[kid].loads {
            assert!(
                matches!(self.tensors[tid], TensorData::Leaf { .. } | TensorData::GraphLeaf { .. } | TensorData::Symbolic { .. }),
                "materialize: load {tid} has no buffer after pending flush: {:?}",
                self.tensors[tid]
            );
        }
        // Resolve the dtypes of the loads and stores now, while this kernel (and any
        // tensor whose dtype resolves through it) is still alive. After remove_and_return
        // below, self.dtype on those tensors would panic on the removed kernel.
        let dtypes: Map<TensorId, DType> =
            self.kernels[kid].loads.iter().chain(&self.kernels[kid].stores).map(|&tid| (tid, self.dtype(tid))).collect();
        // Null out kernel_ids of disowned loads pointing at this kernel: it dies
        // below, and `remove_and_return` is a swap_remove — a stale kernel_id
        // would alias a *different* kernel afterwards. Their edges are released
        // after the launch; the NULL kernel_id ends that recursion cleanly.
        for &tid in &self.kernels[kid].loads {
            if let TensorData::Eager { kernel_id: k, .. } | TensorData::Promoted { kernel_id: k, .. } = &mut self.tensors[tid] {
                if *k == kid {
                    *k = KernelId::NULL;
                }
            }
        }
        let KernelData { outputs, loads, stores, mut kernel } = unsafe { self.kernels.remove_and_return(kid) };

        debug_assert!(outputs.is_empty(), "all outputs must be stored before materialize");
        // Stores are ALL pending leafs: add_store re-homes every stored
        // tensor, assign re-homes its in-place target. Outputs and stores
        // never overlap — add_store removes from outputs when pushing to
        // stores, and outputs is empty here anyway.
        debug_assert!(
            stores.iter().all(|&tid| matches!(self.tensors[tid], TensorData::PendingLeaf { .. })),
            "materialize: not all stores are pending leafs"
        );

        if stores.is_empty() {
            // Nothing to launch, but the kernel is still being removed: its
            // load edges must be released exactly as in the launch path, or
            // the counts orphan and the tensors leak.
            for &tid in &loads {
                self.release(tid);
            }
            return Ok(());
        }

        for &tid in &loads {
            assert!(
                self.leaf_buffer(tid).is_some()
                    || outputs.contains(&tid)
                    || self.kernels.values().any(|kd| kd.outputs.contains(&tid) || kd.stores.contains(&tid))
                    || self.resolve_symbolic(tid).is_some(),
                "load tid {tid} not realized, not in outputs, not in any kernel; kernels loading it: {:?}",
                self.kernels.iter().filter(|(_, kd)| kd.loads.contains(&tid)).map(|(k, _)| k).collect::<Vec<_>>(),
            );
        }

        // Debug: ensure each store tid is in exactly one kernel's outputs
        // (count may be 0 if add_store removed it and triggered this materialization)
        #[cfg(debug_assertions)]
        {
            for &tid in &stores {
                let count = self.kernels.values().filter(|kd| kd.outputs.contains(&tid)).count();
                debug_assert!(count <= 1, "store tid={tid} is in {count} kernels' outputs");
            }
            // Inventory invariant: NO tensor may be listed in more than one
            // kernel's `outputs` — a live tensor's only listing is the kernel
            // equal to its `kernel_id` (which is 0 here for this kernel: it
            // was removed above). A tensor appearing in several `outputs`
            // sets (or in one that isn't its `kernel_id`) means an inventory
            // desync — `release` will clean up the wrong kernel and leave
            // stale entries behind.
            let mut listed_in_multiple = Vec::new();
            let mut counted: Map<TensorId, usize> = Map::with_hasher(BuildHasherDefault::new());
            for kd in self.kernels.values() {
                for &tid in &kd.outputs {
                    *counted.entry(tid).or_insert(0) += 1;
                }
            }
            for (&tid, &count) in &counted {
                if count > 1 {
                    listed_in_multiple.push(tid);
                }
            }
            debug_assert!(
                listed_in_multiple.is_empty(),
                "inventory desync: tensors listed in multiple kernels' outputs: {listed_in_multiple:?}"
            );
        }

        // Recursive materialization: realize every load, first via each
        // unrealized load's `depends_on` producer (add_store-ing its outputs
        // launches the pending store that produces the load), then via
        // `add_store` on whatever is still unrealized. Scalars (constants/
        // variables) live in the tensors slab instead of buffer_map —
        // `resolve_symbolic` covers them.
        for &load in &loads {
            if self.leaf_buffer(load).is_some() || self.resolve_symbolic(load).is_some() {
                continue;
            }
            // A pending `Leaf` records its producer in `depends_on`
            // (add_store re-homed it without a kernel).
            let pending = match self.tensors[load] {
                TensorData::PendingLeaf { depends_on, .. } => depends_on,
                _ => KernelId::NULL,
            };
            if pending.is_null() {
                continue;
            }
            let outputs: Set<TensorId> = self.kernels[pending].outputs.iter().copied().collect();
            if outputs.is_empty() {
                // The producer's outputs were all add_store'd away while its
                // stores are still pending — the only way to realize the load
                // is to launch the producer directly (its precondition "all
                // outputs stored" holds: outputs is empty).
                self.materialize_kernel(pending)?;
                continue;
            }
            for output in outputs {
                self.add_store(output)?;
            }
        }
        for &load in &loads {
            if self.leaf_buffer(load).is_some() || self.resolve_symbolic(load).is_some() {
                continue;
            }
            if matches!(
                self.tensors[load],
                TensorData::Eager { .. } | TensorData::Promoted { .. } | TensorData::PendingLeaf { .. }
            ) {
                self.add_store(load)?;
            }
        }

        debug_assert!(
            loads.iter().all(|&tid| self.leaf_buffer(tid).is_some() || self.resolve_symbolic(tid).is_some()),
            "all loads must be realized after recursive materialization"
        );

        // If stores already have buffers (assign writes in-place into a kept
        // buffer), a kernel can only touch memory of one pool, so those
        // buffers dictate the pool — and hence the device. Stores spanning
        // multiple pools is an error. Without existing store buffers (or if
        // no device shares their pool), fall back to the freest device and
        // move the buffers.
        let mut store_pools: BTreeSet<Pool> = BTreeSet::new();
        for &tid in &stores {
            if let TensorData::PendingLeaf { old_buffer: Some(buf_id), .. } = self.tensors[tid] {
                store_pools.insert(buf_id.pool);
            }
        }
        // Bytes needed for all outputs that don't already have buffers.
        let out_bytes: Dim = stores
            .iter()
            .filter(|&&tid| matches!(self.tensors[tid], TensorData::PendingLeaf { old_buffer: None, .. }))
            .map(|&tid| {
                let dtype = dtypes[&tid];
                (self.resolve_shape(tid).iter().product::<Dim>() * dtype.bit_size() as Dim + 7) / 8
            })
            .sum();
        // A pinned device (kernel built with a concrete `Dev`) wins over
        // everything — when the user moves a tensor to a device, that has to
        // take effect. Only a `Dev::Auto` kernel gets auto-picked.
        let (dev_id, pool_id) = if kernel.dev != Dev::Auto {
            let dev_id = kernel.dev;
            if store_pools.len() > 1 || !store_pools.iter().all(|&pool| pool == dev_id.pool()) {
                return Err(ZyxError::AllocationError(
                    format!(
                        "stores {store_pools:?} do not match the kernel's pinned device {dev_id:?} and cannot span multiple pools",
                    )
                    .into(),
                ));
            }
            (dev_id, dev_id.pool())
        } else if store_pools.len() == 1 {
            let pool_id = *store_pools.iter().next().unwrap();
            let dev_id = Dev::all().into_iter().find(|dev| dev.pool() == pool_id);
            match dev_id {
                Some(dev_id) => (dev_id, pool_id),
                None => {
                    let dev_id = self.pick_device(out_bytes)?;
                    (dev_id, dev_id.pool())
                }
            }
        } else if store_pools.is_empty() {
            // Pick the device where most loaded bytes reside, provided it has
            // enough memory for the outputs; otherwise the fastest device
            // with enough memory.
            let mut loaded_bytes: Map<Pool, Dim> = Map::default();
            for &tid in &loads {
                if let Some(buf_id) = self.leaf_buffer(tid) {
                    let dtype = dtypes[&tid];
                    *loaded_bytes.entry(buf_id.pool).or_insert(0) +=
                        (self.resolve_shape(tid).iter().product::<Dim>() * dtype.bit_size() as Dim + 7) / 8;
                }
            }
            let mut best: Option<(Dev, Pool)> = None;
            for dev_id in Dev::all() {
                if dev_id.aot_only() {
                    continue;
                }
                let pool_id = dev_id.pool();
                if pool_id.free_bytes() < out_bytes {
                    continue;
                }
                let bytes = loaded_bytes.get(&pool_id).copied().unwrap_or(0);
                if best.is_none_or(|(_, best_pool)| loaded_bytes.get(&best_pool).copied().unwrap_or(0) < bytes) {
                    best = Some((dev_id, pool_id));
                }
            }
            match best {
                Some((dev_id, pool_id)) => (dev_id, pool_id),
                None => {
                    let dev_id = self.pick_device(out_bytes)?;
                    (dev_id, dev_id.pool())
                }
            }
        } else {
            return Err(ZyxError::AllocationError(
                format!("stores span multiple pools {store_pools:?}; a kernel can only touch memory of a single pool").into(),
            ));
        };
        kernel.dev = dev_id;
        kernel.dev_info = Some(dev_id.info());

        // Ensure loads are in target pool. Variables and symbolic leaves are
        // not backed by any buffer — they bind at launch from `variable_map`.
        for &tid in &loads {
            let Some(buf_id) = self.leaf_buffer(tid) else { continue };
            if buf_id.pool != pool_id {
                let src = buf_id.buffer_id;
                let bytes =
                    (self.resolve_shape(tid).iter().product::<Dim>() as usize * dtypes[&tid].bit_size() as usize).div_ceil(8);
                let alloc_bytes = bytes + dtypes[&tid].bit_size() as usize / 8;

                let dst = pool_id.allocate(alloc_bytes as Dim)?;
                let dst_global = Buffer { pool: pool_id, buffer_id: dst };
                debug_assert_ne!(buf_id.pool, pool_id, "pool_to_pool across the same pool is disallowed");
                pool_id.pool_to_pool(buf_id.pool, src, dst)?;
                buf_id.pool.release(src);
                // Record the new location in place: leaf_buffer reads the slab.
                match &mut self.tensors[tid] {
                    TensorData::Leaf { buffer: buffer_id, .. } | TensorData::GraphLeaf { buffer: buffer_id, .. } => {
                        *buffer_id = dst_global;
                    }
                    ref t => panic!("materialize: moved load {tid} has no buffer field: {t:?}"),
                }
            }
        }

        // Ensure stores are in target pool (assign writes in-place into a kept
        // buffer, which may live in a different pool). Bufferless stores are
        // skipped here — they get fresh buffers below.
        for &tid in &stores {
            let Some(buf_id) = (match self.tensors[tid] {
                TensorData::PendingLeaf { old_buffer, .. } => old_buffer,
                ref t => panic!("materialize: store {tid} is not a pending leaf: {t:?}"),
            }) else {
                continue;
            };
            if buf_id.pool != pool_id {
                let src = buf_id.buffer_id;
                let bytes =
                    (self.resolve_shape(tid).iter().product::<Dim>() as usize * dtypes[&tid].bit_size() as usize).div_ceil(8);
                let alloc_bytes = bytes as Dim + Dim::from(dtypes[&tid].bit_size() / 8);

                let dst = pool_id.allocate(alloc_bytes)?;
                let dst_global = Buffer { pool: pool_id, buffer_id: dst };
                debug_assert_ne!(buf_id.pool, pool_id, "pool_to_pool across the same pool is disallowed");
                pool_id.pool_to_pool(buf_id.pool, src, dst)?;
                buf_id.pool.release(src);
                // Record the new location in place on the pending store.
                match &mut self.tensors[tid] {
                    TensorData::PendingLeaf { old_buffer: Some(buffer_id), .. } => {
                        *buffer_id = dst_global;
                    }
                    ref t => panic!("materialize: moved store {tid} is not a pending leaf: {t:?}"),
                }
            }
        }

        // Collect store buffers: a pending store with a kept buffer (assign
        // in-place target) is passed into the kernel to be mutated; the rest
        // get fresh buffers, written back onto the tensor as its kept buffer.
        let mut kernel_buffers = BTreeSet::new();
        for &tid in &loads {
            // Scalars (`Symbolic` handles over an `Expr::Variable`) are
            // launch-time values, not pool storage — they have no buffer.
            if matches!(self.tensors[tid], TensorData::Symbolic { expr, .. } if matches!(self.exprs[expr], Expr::Variable { .. }))
            {
                continue;
            }
            kernel_buffers.insert(self.leaf_buffer(tid).expect("materialize: load without buffer after pending flush"));
        }
        for &tid in &stores {
            if let TensorData::PendingLeaf { old_buffer: Some(buf), .. } = self.tensors[tid] {
                kernel_buffers.insert(buf);
                continue;
            }
            let bytes = (self.resolve_shape(tid).iter().product::<Dim>() as usize * dtypes[&tid].bit_size() as usize).div_ceil(8);
            let alloc_bytes = bytes as Dim + Dim::from(dtypes[&tid].bit_size() / 8);
            let buf = pool_id.allocate(alloc_bytes)?;
            let global_id = Buffer { pool: pool_id, buffer_id: buf };
            kernel_buffers.insert(global_id);
            match &mut self.tensors[tid] {
                TensorData::PendingLeaf { old_buffer: slot @ None, .. } => {
                    *slot = Some(global_id);
                }
                ref t => panic!("materialize: store {tid} is not a pending leaf: {t:?}"),
            }
        }
        // Materialization must realize EVERY store: every pending store now
        // carries its (kept or freshly allocated) buffer.
        for &tid in &stores {
            debug_assert!(
                matches!(self.tensors[tid], TensorData::PendingLeaf { old_buffer: Some(_), .. }),
                "materialize: store tid {tid} has no buffer after realization"
            );
        }

        // Build args: load buffers/variables first, then store buffers.
        // Law: `loads` ↔ Global+Variable defines, `stores` carries the
        // GlobalMut store targets — args bind positionally over exactly this
        // concatenation, so both sides must stay aligned and unshuffled.
        #[cfg(debug_assertions)]
        {
            let (mut n_non_mut, mut n_mut) = (0usize, 0usize);
            let mut p = kernel.head;
            while !p.is_null() {
                if let Op::Param { kind, .. } = &kernel.ops[p].op {
                    match kind {
                        ParamKind::GlobalMut => n_mut += 1,
                        ParamKind::Global | ParamKind::Variable => n_non_mut += 1,
                    }
                }
                p = kernel.next_op(p);
            }
            assert_eq!(n_non_mut, loads.len(), "materialize: {} non-store defines but {} load entries", n_non_mut, loads.len());
            assert!(n_mut <= stores.len(), "materialize: {} GlobalMut defines but only {} stores", n_mut, stores.len());
        }
        let mut buffers: Vec<LaunchArg> = Vec::new();
        for &tid in &loads {
            // Variables live only in the expr slab — they never have a
            // buffer or pool storage; the value is bound at launch.
            let var_value = match self.tensors[tid] {
                TensorData::Symbolic { expr, .. } => match self.exprs[expr] {
                    Expr::Variable { value } => Some(value),
                    ref e => panic!("materialize: symbolic load {tid} is not a variable: {e:?}"),
                },
                _ => None,
            };
            if let Some(value) = var_value {
                buffers.push(LaunchArg::Variable(value));
            } else {
                buffers.push(LaunchArg::Buffer(self.leaf_buffer(tid).expect("materialize: load without buffer").buffer_id));
            }
        }
        for &tid in &stores {
            let buf = match self.tensors[tid] {
                TensorData::PendingLeaf { old_buffer: Some(buf), .. } => buf,
                ref t => panic!("materialize: store {tid} has no buffer after realization: {t:?}"),
            };
            buffers.push(LaunchArg::Buffer(buf.buffer_id));
        }

        // Compile and launch (caches in kernel_map / programs)
        let (dev_prog, _timing) = self.get_or_autotune(kernel, &buffers)?;

        dev_id.launch(dev_prog, &buffers)?;

        // The kernel launch gives new buffers: ALL stores are turned into
        // Leafs over their (kept or freshly allocated) buffer at once.
        for &tid in &stores {
            let (shape_id, dtype, rc, buf) = match self.tensors[tid] {
                TensorData::PendingLeaf { shape_id, dtype, rc, old_buffer: Some(buf), .. } => (shape_id, dtype, rc, buf),
                ref t => panic!("materialize: store {tid} is not a realized pending leaf: {t:?}"),
            };
            self.tensors[tid] = TensorData::Leaf { shape_id, dtype, buffer: buf, rc };
        }

        // The kernel has consumed its loads. Release the load references so
        // dead load tensors and their buffers are reclaimed. Buffers still in
        // use keep rc > 0 via other kernels' load references or handles.
        for &tid in &loads {
            self.release(tid);
        }

        Ok(())
    }

    /// Number of live slab entries entries.
    ///
    /// Unit-test surface (runtime is not publicly exported): after a full
    /// create/operate/drop cycle both must be zero — anything else is a leak.
    #[cfg(test)]
    #[allow(unused)]
    pub fn live_inventory(&self) -> usize {
        self.tensors.iter().count()
    }
}

/*#[cfg(test)]
mod leak_tests {
    use super::*;
    use crate::{RT, Tape, Tensor};
    use std::sync::{Mutex, OnceLock};

    /// RT is process-global: serialize the inventory tests so parallel unit
    /// tests never observe each other's live tensors.
    fn test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    /// After a full create/operate/drop cycle the runtime must have drained
    /// completely: no live slab entries, no live buffers.
    fn assert_drained() {
        let rt = RT.lock();
        let (t, b) = rt.live_inventory();
        if (t, b) != (0, 0) {
            // Ledger check (handles are 0 at drain time): expected rc per tid =
            // kernel load entries + symbolic children edges. Any excess is an
            // orphan count (retain whose edge/entry no longer exists).
            let mut expected: Map<TensorId, usize> = Map::with_hasher(BuildHasherDefault::new());
            for kd in rt.kernels.values() {
                for &l in &kd.loads {
                    *expected.entry(l).or_insert(0) += 1;
                }
            }
            for (tid, td) in rt.tensors.iter() {
                match td {
                    TensorData::Cast { x, .. } | TensorData::Unary { x, .. } => {
                        *expected.entry(*x).or_insert(0) += 1;
                    }
                    TensorData::Binary { x, y, .. } => {
                        *expected.entry(*x).or_insert(0) += 1;
                        *expected.entry(*y).or_insert(0) += 1;
                    }
                    TensorData::Stack { tensors, .. } => {
                        for t in tensors.iter() {
                            *expected.entry(*t).or_insert(0) += 1;
                        }
                    }
                    TensorData::Stack2 { tensors, .. } => {
                        for t in tensors.iter() {
                            *expected.entry(*t).or_insert(0) += 1;
                        }
                    }
                    TensorData::Stack3 { tensors, .. } => {
                        for t in tensors.iter() {
                            *expected.entry(*t).or_insert(0) += 1;
                        }
                    }
                    TensorData::Stack4 { tensors, .. } => {
                        for t in tensors.iter() {
                            *expected.entry(*t).or_insert(0) += 1;
                        }
                    }
                    TensorData::Stack5 { tensors, .. } => {
                        for t in tensors.iter() {
                            *expected.entry(*t).or_insert(0) += 1;
                        }
                    }
                    TensorData::Eager { shape_id, .. }
                    | TensorData::Graph { shape_id, .. }
                    | TensorData::Promoted { shape_id, .. } => {
                        if !shape_id.is_null() {
                            *expected.entry(*shape_id).or_insert(0) += 1;
                        }
                    }
                    _ => {}
                }
                let _ = tid;
            }
            for (tid, td) in rt.tensors.iter() {
                let rc = match td {
                    TensorData::Eager { rc, .. }
                    | TensorData::Graph { rc, .. }
                    | TensorData::Promoted { rc, .. }
                    | TensorData::Constant { rc, .. }
                    | TensorData::Variable { rc, .. }
                    | TensorData::Cast { rc, .. }
                    | TensorData::Unary { rc, .. }
                    | TensorData::Binary { rc, .. }
                    | TensorData::Stack { rc, .. }
                    | TensorData::Stack2 { rc, .. }
                    | TensorData::Stack3 { rc, .. }
                    | TensorData::Stack4 { rc, .. }
                    | TensorData::Stack5 { rc, .. } => *rc as usize,
                };
                let exp = expected.get(&tid).copied().unwrap_or(0);
                if rc != exp {
                    eprintln!("LEDGER tid={tid} rc={rc} expected={exp} orphan={}", rc as isize - exp as isize);
                }
                eprintln!("ITER leak-check tid={tid} rc={rc} {td:?}");
                if let TensorData::Eager { kernel_id, .. } | TensorData::Promoted { kernel_id, .. } = td {
                    if !kernel_id.is_null() {
                        if let Some(kd) = rt.kernels.get(*kernel_id) {
                            eprintln!(
                                "   kernel {kernel_id:?}: outputs={:?} loads={:?} stores={:?}",
                                kd.outputs, kd.loads, kd.stores
                            );
                        } else {
                            eprintln!("   kernel {kernel_id:?}: REMOVED from slab (stale kernel_id)");
                        }
                    }
                }
            }
        }
        assert_eq!((t, b), (0, 0), "runtime did not drain: leaked tensors or buffers");
    }

    #[test]
    fn eager_ops_drain_inventory() -> Result<(), ZyxError> {
        let _guard = test_lock().lock().unwrap();
        for _ in 0..8 {
            {
                let x = Tensor::from([[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
                let y = Tensor::from([1.0f32, 2.0, 3.0, 4.0]);
                let z = x + y; // broadcast + binary fusion
                let w = z * 2.0; // scalar binary
                let e = w.exp(); // unary
                let r = e.reshape([2, 4])?; // movement
                let p = r.t(); // permute (duplicate_or_store path)
                let s = p.sum_all(); // reduce
                let v: Vec<f32> = s.try_into()?; // force execution
                assert_eq!(v.len(), 1);
                // Breaker repro: a standalone buffer-backed tensor that is
                // immediately dropped. Under the Leaf design no self-load
                // kernel exists, so this drains cleanly; without it this
                // leaks (the old tensor↔kernel self-load cycle).
                drop(Tensor::from([2, 3, 4]));
            } // every handle drops here
            assert_drained();
        }
        Ok(())
    }

    #[test]
    fn eager_and_tape_drain_inventory() -> Result<(), ZyxError> {
        let _guard = test_lock().lock().unwrap();
        for _ in 0..8 {
            {
                // Buffer-backed leaf (stays eager-side, promoted as leaf).
                let x = Tensor::from([[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]);
                let w = Tensor::from([1.0f32, 2.0, 3.0, 4.0]);
                {
                    let tape = Tape::new([&x, &w])?;
                    // Non-leaf promoted intermediates: freely droppable, the
                    // graph replays their computation.
                    let y = x.exp() + w; // x promoted, exp replayed, binary
                    let z = y * 2.0;
                    let r = z.reshape([2, 4])?;
                    tape.realize([&r])?;
                    let v: Vec<f32> = r.try_into()?;
                    assert_eq!(v.len(), 8);
                } // tape drops here: revert loop + graph teardown
            }
            assert_drained();
        }
        Ok(())
    }
}*/
