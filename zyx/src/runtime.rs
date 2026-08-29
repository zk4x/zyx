// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

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
//!    `Cast.x`, `Unary.x`, `Binary { x, y }`, `Stack.tensors`. Each held child is one
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
//! - **Disown.** When a user handle drops and the tensor's remaining `rc` is exactly its
//!   producer kernel's load edge, the tensor is disowned: removed from the kernel's
//!   `outputs`, but kept alive (rc stays 1) as input for the kernel's still-pending
//!   computations. Other surviving outputs may still read its buffer.
//! - **Breaker.** When a handle drops, the remaining rc is 1 and the producer kernel's
//!   `outputs` is exactly `{x}` — the kernel's only remaining purpose was producing x, so
//!   the tensor↔kernel cycle is real and nothing else can ever reference the tensor — the
//!   cycle is broken: x dies and takes the (now purposeless) kernel with it. Without this
//!   the pair would deadlock: x kept alive by its kernel's load edge, the kernel kept
//!   alive by x being its last output.
//! - **Death (rc == 0).** The tensor detaches from its producer kernel: removed from
//!   `outputs`, its now-unreferenced op chain pruned, the pruned ops' load edges released.
//!   If the kernel's `outputs` and `stores` are then empty the kernel is dropped and its
//!   remaining load edges are released, which recursively kills tensors nothing else
//!   reads (including `depends_on` producers). Then: free the buffer, remove the slab
//!   entry, release the `shape_id` edge.
//! - **Kernel death tolerates `kernel_id == NULL`.** Before a kernel is dropped or
//!   materialized, every load tensor whose `kernel_id` points at it has that field
//!   nulled: the kernel is gone, its edges are being released, and a later death must not
//!   dereference the dead kernel. (This also matters because `materialize_kernel` uses
//!   `remove_and_return` — a `swap_remove` — so `kernel_id` values pointing at the
//!   removed kernel would silently alias a different kernel afterwards.)
//! - `depends_on` is never released directly: it dies through the same recursion when
//!   the loading kernel's loads are released.
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
//!    construction: `depends_on` points at the kernel whose `stores` feed `kernel_id`'s
//!    loads — following it strictly descends toward already-realized buffers, so a
//!    cycle would mean a kernel needs its own result as input.
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
//! 1. **Symbolic edges.** A dead `Cast`/`Unary`/`Binary`/`Stack` releases its operand
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

// ----- ASYNC EVENT RULES -----
//
// host_to_pool is async: the host-side source buffer must stay valid
// until the returned event is synced via sync_events.
//
// Kernel launch is async: ALL buffers used by the kernel (both loads
// and stores) must stay valid until the kernel's event is consumed.
//
// Events are tracked in self.events: Map<BTreeSet<BufferId>, Event>.
// The key set must include every buffer the kernel touches, so future
// operations can find and wait on the event before reusing the buffer.
//
// When extracting a pending event for a buffer: iterate events.keys(),
// find the set containing the buffer, remove the event, and add it to
// the wait list passed to the next launch. A removed event is consumed.
// -----------------------------

use std::{
    collections::BTreeSet,
    env,
    hash::BuildHasherDefault,
    path::{Path, PathBuf},
};

use nanoserde::DeJson;

#[cfg(feature = "viz")]
use crate::viz::Viz;
use crate::{
    DType, DebugMask, Map, Scalar, Set, ZyxError,
    backend::{
        AutotuneConfig, BufferId, Config, DTypeCapability, Device, DeviceInfo, DeviceProgramId, Event, LaunchArg, MemoryPool,
        PoolId, ProgramId,
    },
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    graph::{ClassId, ExecPlan, Graph, GraphId, Node, plan::drain_events_for_buf},
    kernel::{BOp, DeviceId, IDX_T, Kernel, MemLayout, MoveOp, Op, OpId, ParamKind, UOp, autotune::OptSeq},
    rng::Rng,
    scalar::{bf16, f16},
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
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

#[derive(Debug, Copy, Clone, Hash, PartialEq, PartialOrd, Eq, Ord)]
pub struct ShapeId(u16);

impl From<usize> for ShapeId {
    fn from(value: usize) -> Self {
        ShapeId(value as u16)
    }
}

impl From<ShapeId> for usize {
    fn from(value: ShapeId) -> Self {
        value.0 as usize
    }
}

impl SlabId for ShapeId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u16::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub(crate) struct DeviceInfoId(u32);

impl From<usize> for DeviceInfoId {
    fn from(value: usize) -> Self {
        DeviceInfoId(value as u32)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub(crate) struct KernelId(u16);

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
    /// Symbolic dimension: identified by the ROOT dim-expression tensor.
    /// Two dims are provably equal ONLY if they are the same tensor —
    /// different expressions over the same variable, or a variable whose
    /// slot happens to hold the same value as a constant, are NOT proof
    /// (the slot can change before launch).
    Symbolic(TensorId),
}

#[derive(Debug)]
pub enum TensorData {
    // Eager only
    //
    /// # Field semantics
    ///
    /// - `kernel_id`: the kernel that holds this tensor in its `outputs` (or
    ///   `stores`, once stored). Exactly one kernel lists it — see
    ///   [`KernelData`] for the inventory invariants.
    /// - `depends_on`: the producer kernel whose `stores` store the tensors
    ///   that appear as `loads` of `kernel_id`. In other words, to run
    ///   `kernel_id` you must first materialize `depends_on`. It is never
    ///   released directly; when a kernel dies its `loads` are released and
    ///   that handles it recursively.
    Eager {
        kernel_id: KernelId,
        op_id: OpId,
        depends_on: KernelId,
        shape_id: TensorId,
        dtype: DType,
        rc: u16,
    },
    // Graph only
    Graph {
        class_id: ClassId,
        graph_id: GraphId,
        shape_id: TensorId,
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
        shape_id: TensorId,
        dtype: DType,
        rc: u16,
    },
    // Baked into kernel
    Constant {
        value: Constant,
        rc: u16,
    },
    // Kernel argument
    Variable {
        value: Constant,
        rc: u16,
    },
    // Symbolic on variables and constants
    Cast {
        x: TensorId,
        dtype: DType,
        rc: u16,
    },
    // Symbolic on variables and constants
    Unary {
        x: TensorId,
        uop: UOp,
        rc: u16,
    },
    // Symbolic on variables and constants
    Binary {
        x: TensorId,
        y: TensorId,
        bop: BOp,
        rc: u16,
    },
    Stack {
        tensors: Box<[TensorId]>,
        rc: u16,
    },
}

#[derive(Debug)]
pub(crate) struct KernelData {
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
    /// that step a new *load kernel* is created (loads=[tensor],
    /// outputs={tensor}) and the tensor's `depends_on` points at the kernel
    /// whose `stores` hold it. Moving such a load kernel around afterwards —
    /// including merging it into later kernels — is legal and DESIRED: it is
    /// the core fusion principle of the eager fusion machinery.
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
    ///   kernel K are stored by the `stores` of K's producer — for an eager
    ///   tensor that producer is recorded in `TensorData::Eager::depends_on`.
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
    optimizations: Map<(KernelId, DeviceInfoId), OptSeq>,
    device_infos: Map<DeviceInfo, DeviceInfoId>,
    programs: Map<KernelId, DeviceProgramId>,
    timings: Map<ProgramId, u64>,
    pub devices: Slab<DeviceId, Device>,
    // Pool 0 is always host, pool 1 is disk if disk is present
    pub pools: Slab<PoolId, MemoryPool>,
    config_dir: Option<PathBuf>,
    pub buffer_map: Map<TensorId, BufferId>,
    pub variable_map: Map<TensorId, Constant>,
    pub events: Map<BTreeSet<BufferId>, Event>,
    pub rng: Rng,
    autotune_config: AutotuneConfig,
    pub implicit_casts: bool,
    pub training: bool,
    pub debug: DebugMask,
    pub plan_cache: Map<u64, ExecPlan>,
    #[cfg(feature = "viz")]
    pub viz: Viz,
}

impl Runtime {
    pub const fn new() -> Self {
        Runtime {
            graphs: Slab::new(),
            tensors: Slab::new(),
            kernels: Slab::new(),
            kernel_map: Map::with_hasher(BuildHasherDefault::new()),
            device_infos: Map::with_hasher(BuildHasherDefault::new()),
            devices: Slab::new(),
            pools: Slab::new(),
            programs: Map::with_hasher(BuildHasherDefault::new()),
            timings: Map::with_hasher(BuildHasherDefault::new()),
            config_dir: None,
            optimizations: Map::with_hasher(BuildHasherDefault::new()),
            buffer_map: Map::with_hasher(BuildHasherDefault::new()),
            variable_map: Map::with_hasher(BuildHasherDefault::new()),
            events: Map::with_hasher(BuildHasherDefault::new()),
            rng: Rng::seed_from_u64(42069),
            autotune_config: AutotuneConfig::new(),
            implicit_casts: true,
            training: false,
            debug: DebugMask::new(0),
            plan_cache: Map::with_hasher(BuildHasherDefault::new()),
            #[cfg(feature = "viz")]
            viz: Viz::new(),
        }
    }

    /// Concrete shape of tensor `x`, fully evaluated: every symbolic dimension
    /// is resolved to a concrete `Dim` by folding const expressions and reading
    /// variable slots. Panics if a dimension cannot be resolved (an unboundable
    /// variable is a bug), never returns a sentinel.
    /// Host-side evaluation of a scalar expression: folds the tensors-slab
    /// tree (`Constant` / `Variable` / `Unary` / `Binary`) into a single
    /// `Constant` using the same rules as `resolve_shape`, preserving
    /// dtype. Returns `None` for non-symbolic tensors.
    pub(crate) fn resolve_symbolic(&self, x: TensorId) -> Option<Constant> {
        match &self.tensors[x] {
            TensorData::Constant { value, .. } | TensorData::Variable { value, .. } => Some(*value),
            TensorData::Cast { x: a, dtype, .. } => self.resolve_symbolic(*a).map(|v| v.cast(*dtype)),
            TensorData::Unary { x: a, uop, .. } => self.resolve_symbolic(*a).map(|v| v.unary(*uop)),
            TensorData::Binary { x: a, y: b, bop, .. } => {
                Some(Constant::binary(self.resolve_symbolic(*a)?, self.resolve_symbolic(*b)?, *bop))
            }
            _ => None,
        }
    }

    /// Concrete (resolved) shape of tensor `x`: evaluates every dim expression
    /// to a `Dim`, reading variable slots from `variable_map`.
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
        // DFS post-order flatten (same traversal as replay_shape_into_kernel):
        // every node lands after its operands, so one flat pass evaluates.
        fn flatten(rt: &Runtime, x: TensorId, order: &mut Vec<TensorId>) {
            match &rt.tensors[x] {
                TensorData::Constant { .. } | TensorData::Variable { .. } => (),
                TensorData::Cast { x: a, .. } => flatten(rt, *a, order),
                TensorData::Unary { x: a, .. } => flatten(rt, *a, order),
                TensorData::Binary { x: a, y: b, .. } => {
                    flatten(rt, *a, order);
                    flatten(rt, *b, order);
                }
                TensorData::Stack { tensors, .. } => {
                    for &t in tensors.iter() {
                        flatten(rt, t, order);
                    }
                }
                t => panic!(
                    "shape expression tid {x} contains non-symbolic tensor data {t:?}; shapes must be built from Dim constants, variables and dim ops"
                ),
            }
            order.push(x);
        }

        let mut dims = Vec::new();
        for root in self.shape(x) {
            let mut order = Vec::new();
            flatten(self, root, &mut order);
            let mut vals: Map<TensorId, Constant> = Map::with_hasher(BuildHasherDefault::new());
            let mut value = Constant::I64(0i64.to_le_bytes());
            for tid in order {
                value = match self.tensors[tid] {
                    TensorData::Constant { value, .. } => value,
                    TensorData::Variable { .. } => self.variable_map[&tid],
                    TensorData::Cast { x: a, dtype, .. } => vals[&a].cast(dtype),
                    TensorData::Unary { x: a, uop, .. } => vals[&a].unary(uop),
                    TensorData::Binary { x: a, y: b, bop, .. } => Constant::binary(vals[&a], vals[&b], bop),
                    ref t => panic!("dimension tid {tid} is not a dim expression: {t:?}"),
                };
                vals.insert(tid, value);
            }
            dims.push(value.as_dim().expect("dim expression does not evaluate to an integer"));
        }
        dims
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
    pub(crate) fn resolve_shape_without_variables(&self, x: TensorId) -> Vec<ResolvedDim> {
        // Post-order check: does this dim expression contain a Variable anywhere?
        fn contains_variable(rt: &Runtime, x: TensorId) -> bool {
            match &rt.tensors[x] {
                TensorData::Variable { .. } => true,
                TensorData::Cast { x: a, .. } | TensorData::Unary { x: a, .. } => contains_variable(rt, *a),
                TensorData::Binary { x: a, y: b, .. } => contains_variable(rt, *a) || contains_variable(rt, *b),
                TensorData::Stack { tensors, .. } => tensors.iter().any(|&t| contains_variable(rt, t)),
                TensorData::Constant { .. } => false,
                t => panic!(
                    "dim expression tid {x} contains non-symbolic tensor data {t:?}; shapes must be built from Dim constants, variables and dim ops"
                ),
            }
        }

        // Same DFS post-order flatten + evaluation as `resolve_shape`, but a
        // variable-tainted root short-circuits to Symbolic (no slot reads).
        fn flatten(rt: &Runtime, x: TensorId, order: &mut Vec<TensorId>) {
            match &rt.tensors[x] {
                TensorData::Constant { .. } | TensorData::Variable { .. } => (),
                TensorData::Cast { x: a, .. } => flatten(rt, *a, order),
                TensorData::Unary { x: a, .. } => flatten(rt, *a, order),
                TensorData::Binary { x: a, y: b, .. } => {
                    flatten(rt, *a, order);
                    flatten(rt, *b, order);
                }
                TensorData::Stack { tensors, .. } => {
                    for &t in tensors.iter() {
                        flatten(rt, t, order);
                    }
                }
                t => panic!(
                    "shape expression tid {x} contains non-symbolic tensor data {t:?}; shapes must be built from Dim constants, variables and dim ops"
                ),
            }
            order.push(x);
        }

        let mut dims = Vec::new();
        for root in self.shape(x) {
            if contains_variable(self, root) {
                dims.push(ResolvedDim::Symbolic(root));
                continue;
            }
            let mut order = Vec::new();
            flatten(self, root, &mut order);
            let mut vals: Map<TensorId, Constant> = Map::with_hasher(BuildHasherDefault::new());
            let mut value = Constant::I64(0i64.to_le_bytes());
            for tid in order {
                value = match self.tensors[tid] {
                    TensorData::Constant { value, .. } => value,
                    TensorData::Variable { .. } => unreachable!("variable inside a dim checked as variable-free"),
                    TensorData::Cast { x: a, dtype, .. } => vals[&a].cast(dtype),
                    TensorData::Unary { x: a, uop, .. } => vals[&a].unary(uop),
                    TensorData::Binary { x: a, y: b, bop, .. } => Constant::binary(vals[&a], vals[&b], bop),
                    ref t => panic!("dimension tid {tid} is not a dim expression: {t:?}"),
                };
                vals.insert(tid, value);
            }
            dims.push(ResolvedDim::Static(value.as_dim().expect("dim expression does not evaluate to an integer")));
        }
        dims
    }

    /// Symbolic shape of tensor `x` as dim tensors: one scalar IDX_T tensor
    /// per dimension — a `Constant` for static dims, a variable-backed
    /// expression for dynamic ones. Dim-expression tensors themselves are
    /// scalars: their shape is empty.
    ///
    /// # Convention
    /// This is the DEFAULT way to read a shape when building kernels — use it
    /// everywhere. `resolve_shape` (concrete evaluation) is mostly for debug
    /// checks and messages; resolving dims to consts in kernel IR breaks
    /// symbolic-dim consumers.
    pub fn shape(&self, x: TensorId) -> Vec<TensorId> {
        let shape_id = match self.tensors[x] {
            TensorData::Eager { shape_id, .. } | TensorData::Graph { shape_id, .. } | TensorData::Promoted { shape_id, .. } => {
                shape_id
            }
            TensorData::Constant { .. }
            | TensorData::Variable { .. }
            | TensorData::Cast { .. }
            | TensorData::Unary { .. }
            | TensorData::Binary { .. } => {
                return Vec::new();
            }
            TensorData::Stack { ref tensors, .. } => return tensors.to_vec(),
        };
        if shape_id.is_null() {
            return Vec::new();
        }
        match &self.tensors[shape_id] {
            TensorData::Stack { tensors, .. } => tensors.to_vec(),
            _ => vec![shape_id],
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
            | TensorData::Cast { dtype, .. } => dtype,
            TensorData::Constant { value, .. } | TensorData::Variable { value, .. } => value.dtype(),
            TensorData::Unary { x, .. } => self.dtype(x),
            TensorData::Binary { x, bop, .. } => {
                if bop.returns_bool() {
                    DType::Bool
                } else {
                    self.dtype(x)
                }
            }
            TensorData::Stack { .. } => IDX_T,
        }
    }

    pub fn is_realized(&self, x: TensorId) -> bool {
        self.buffer_map.contains_key(&x)
    }

    // True if x is currently a graph tensor (class_id set and its graph alive).
    // A promoted non-realized tensor whose graph has died is treated as eager
    // (its kernel_id is still valid), so is_graph returns false in that case.
    pub(crate) fn is_graph(&self, x: TensorId) -> bool {
        match self.tensors[x] {
            TensorData::Graph { .. } | TensorData::Promoted { .. } => true,
            TensorData::Eager { .. }
            | TensorData::Constant { .. }
            | TensorData::Variable { .. }
            | TensorData::Cast { .. }
            | TensorData::Unary { .. }
            | TensorData::Binary { .. }
            | TensorData::Stack { .. } => false,
        }
    }

    /// Returns operation capabilities for a dtype across all devices.
    pub fn supports_dtype(&mut self, dtype: DType) -> DTypeCapability {
        self.initialize_backends();
        let mut caps = DTypeCapability::none();
        for (_id, dev) in self.devices.iter() {
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
                | TensorData::Graph { rc, .. }
                | TensorData::Promoted { rc, .. }
                | TensorData::Constant { rc, .. }
                | TensorData::Variable { rc, .. }
                | TensorData::Cast { rc, .. }
                | TensorData::Unary { rc, .. }
                | TensorData::Binary { rc, .. }
                | TensorData::Stack { rc, .. } => {
                    *rc += 1;
                    eprintln!("$$$ RETAIN {x} -> rc={rc}");
                    #[cfg(feature = "debug_tensor_op")]
                    println!("rc::retain({x}) -> {rc}");
                }
            };
        }
    }

    fn free_buffer(&mut self, x: TensorId) {
        if let Some(buf_id) = self.buffer_map.remove(&x) {
            let still_used = self.buffer_map.values().any(|b| b.pool == buf_id.pool && b.buffer == buf_id.buffer);
            if !still_used {
                let wait_list = drain_events_for_buf(&mut self.events, buf_id);
                self.pools[buf_id.pool].deallocate(buf_id.buffer, wait_list);
            }
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
                TensorData::Graph { class_id, graph_id, .. } => format!("graph class={class_id:?} graph={graph_id:?}"),
                TensorData::Promoted { kernel_id, class_id, graph_id, .. } => {
                    format!("promoted kernel={kernel_id:?} class={class_id:?} graph={graph_id:?}")
                }
                TensorData::Constant { value, .. } => format!("constant {value:?}"),
                TensorData::Variable { value, .. } => format!("variable {value:?}"),
                TensorData::Unary { x: a, uop, .. } => format!("unary {uop:?}({a})"),
                TensorData::Binary { x: a, y: b, bop, .. } => format!("binary {bop:?}({a},{b})"),
                TensorData::Stack { tensors, .. } => format!("stack len={}", tensors.len()),
                TensorData::Cast { x: a, dtype, .. } => format!("cast {a} -> {dtype:?}"),
            };
            println!("runtime::release(tid={x}) kind={desc}");
        }

        // Drop one reference. Handles and edges (kernel loads, symbolic-node
        // children) all count through here.
        #[cfg(feature = "debug_tensor_op")]
        println!("rc::release({x}) pre: {:?}", self.tensors[x]);
        eprintln!("$$$ RELEASE {x} pre={:?}", self.tensors.get(x));
        let mut disown = false;
        let rc = {
            match &mut self.tensors[x] {
                TensorData::Promoted { rc, kernel_id, .. } | TensorData::Eager { rc, kernel_id, .. } => {
                    *rc -= 1;
                    if !kernel_id.is_null() {
                        // Entries of x in its own kernel — each holds one rc
                        // count. `rc == n` (with n >= 1) means every remaining
                        // reference is such an entry: no handles, no other
                        // kernels' entries, no symbolic edges.
                        let n = self.kernels[*kernel_id].loads.iter().filter(|&&t| t == x).count() as u16;
                        debug_assert!(*rc >= n, "rc {rc} below own-kernel entry count {n} for {x}: ledger desync");
                        if n > 0 && *rc == n {
                            let outs = &self.kernels[*kernel_id].outputs;
                            // The kernel must owe no pending stores: their
                            // value chains consume x's load — x's buffer is
                            // still needed until they launch. A kernel with
                            // only x as output and no stores has no other
                            // purpose: the cycle is real.
                            if outs.len() == 1 && outs.contains(&x) && self.kernels[*kernel_id].stores.is_empty() {
                                // Breaker: the kernel's only remaining purpose was
                                // producing x — the tensor↔kernel cycle is real and
                                // nothing else can ever reference x. Consume ALL n
                                // entry counts; the death path drops exactly those
                                // entries.
                                #[cfg(feature = "debug_tensor_op")]
                                println!(
                                    "rc::release({x}) BREAKER fires: rc == {n} entries, kernel outputs={{x}} — cycle broken"
                                );
                                *rc = 0;
                                // The n consumed counts ARE these load entries:
                                // remove them NOW. Anything triggered by the death
                                // path below (e.g. materialize_kernel's post-launch
                                // release loop) would otherwise release them again.
                                self.kernels[*kernel_id].loads.retain(|&t| t != x);
                                // Consume the outputs edge too (here it is exactly
                                // {x}): with kernel_id nulled below, the death
                                // path's detach can no longer remove it.
                                self.kernels[*kernel_id].outputs.remove(&x);
                                // The edge is fully consumed — the pointer must go
                                // with it, so no observer sees the half-consumed
                                // state (kernel_id set, no membership, rc 0).
                                *kernel_id = KernelId::NULL;
                            } else {
                                disown = true;
                            }
                        }
                    }
                    *rc
                }
                TensorData::Graph { rc, .. }
                | TensorData::Constant { rc, .. }
                | TensorData::Variable { rc, .. }
                | TensorData::Cast { rc, .. }
                | TensorData::Unary { rc, .. }
                | TensorData::Binary { rc, .. }
                | TensorData::Stack { rc, .. } => {
                    *rc -= 1;
                    *rc
                }
            }
        };

        #[cfg(feature = "debug_tensor_op")]
        println!("rc::release({x}) -> rc={rc}");
        // Every variant dies purely on its refcount. `rc` is decremented above;
        // a still-positive count means another live reference (handle, kernel
        // self-load edge, or symbolic-node child) keeps the entry alive.
        if rc != 0 {
            if disown {
                // The user handle is gone; the remaining rc is this kernel's load
                // edge. The kernel still owes other outputs whose computations may
                // read x — disown it: drop it from `outputs` (the kernel no longer
                // owes x to anyone) but keep it alive as input. Its load edge is
                // released when the kernel dies or materializes.
                let kernel_id = match self.tensors[x] {
                    TensorData::Eager { kernel_id, .. } | TensorData::Promoted { kernel_id, .. } => kernel_id,
                    ref t => unreachable!("disown on non-kernel-backed tensor: {t:?}"),
                };
                debug_assert!(!kernel_id.is_null());
                debug_assert!(
                    !(self.kernels[kernel_id].outputs.len() == 1 && self.kernels[kernel_id].outputs.contains(&x))
                        || !self.kernels[kernel_id].stores.is_empty(),
                    "disown: kernel outputs must have other members (or pending stores)"
                );
                self.kernels[kernel_id].outputs.remove(&x);
            }
            return;
        }

        match self.tensors[x] {
            TensorData::Constant { .. } => {
                self.tensors.remove(x);
            }
            TensorData::Variable { .. } => {
                self.variable_map.remove(&x);
                self.tensors.remove(x);
            }
            TensorData::Cast { x: a, .. } => {
                self.tensors.remove(x);
                self.release(a);
            }
            TensorData::Unary { x: a, .. } => {
                self.tensors.remove(x);
                self.release(a);
            }
            TensorData::Binary { x: a, y: b, .. } => {
                self.tensors.remove(x);
                self.release(a);
                self.release(b);
            }
            TensorData::Stack { ref tensors, .. } => {
                let children: Vec<TensorId> = tensors.to_vec();
                self.tensors.remove(x);
                for t in children {
                    self.release(t);
                }
            }
            TensorData::Graph { graph_id, shape_id, .. } => {
                debug_assert!(!self.buffer_map.contains_key(&x), "dead non-leaf graph tensor holds a buffer");
                self.tensors.remove(x);
                if !shape_id.is_null() {
                    self.release(shape_id);
                }
                if !graph_id.is_null() {
                    self.graphs[graph_id].ref_count -= 1;
                    if self.graphs[graph_id].ref_count == 0 {
                        self.remove_dead_graph(graph_id);
                    }
                }
            }
            TensorData::Eager { kernel_id, op_id, depends_on, shape_id, .. } => {
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
                                    eprintln!(">>> NULLING (death path) tid={tid} dying={x} kernel={kernel_id:?}");
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
                        {
                            let live_ops: Set<OpId> = self.kernels[kernel_id].kernel.ops.ids().collect();
                            for (tid, td) in self.tensors.iter() {
                                if let TensorData::Eager { kernel_id: k, op_id: o, .. } | TensorData::Promoted { kernel_id: k, op_id: o, .. } = td {
                                    if *k == kernel_id && !live_ops.contains(o) {
                                        eprintln!("$$$ PRUNE-ORPHAN dying={x} kernel={kernel_id:?} orphan-tid={tid} orphan-op={o:?} dying-op={op_id:?}");
                                    }
                                }
                            }
                        }
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
                // x may still be pending a store in its `depends_on` kernel:
                // `add_store` pushed the store there and re-pointed x onto a
                // fresh load kernel, so the entry in the producer's `stores`
                // outlives the re-point. x is dying: consume the store edge.
                // The GlobalMut params pair positionally with `stores` (this
                // is the same order kernel launch binds buffers in), so find
                // x's param, delete every store writing to it, and prune the
                // now-dead value chains — the kernel must never launch a
                // write into x's freed buffer.
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
                        debug_assert_eq!(mut_params.len(), kd.stores.len(), "GlobalMut params and stores vec diverged in {depends_on:?}");
                        mut_params
                    };
                    let dead_params: Vec<OpId> = {
                        let kd = &self.kernels[depends_on];
                        mut_params
                            .iter()
                            .enumerate()
                            .filter(|(idx, _)| kd.stores[*idx] == x)
                            .map(|(_, op)| *op)
                            .collect()
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
                self.free_buffer(x);
                self.tensors.remove(x);
                if !shape_id.is_null() {
                    self.release(shape_id);
                }
            }
            TensorData::Promoted { kernel_id, op_id, graph_id, shape_id, .. } => {
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
                                    eprintln!(">>> NULLING (death path) tid={tid} dying={x} kernel={kernel_id:?}");
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
                self.free_buffer(x);
                self.tensors.remove(x);
                if !shape_id.is_null() {
                    self.release(shape_id);
                }
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

    /// Number of live slab entries and live buffer_map entries.
    ///
    /// Unit-test surface (runtime is not publicly exported): after a full
    /// create/operate/drop cycle both must be zero — anything else is a leak.
    pub fn live_inventory(&self) -> (usize, usize) {
        (self.tensors.iter().count(), self.buffer_map.len())
    }

    /// Assert the kernel-affiliation invariants for every eager/promoted
    /// tensor: a non-NULL `kernel_id` must reference a live kernel, the tensor
    /// must be listed in that kernel's `outputs`, and its `op_id` must be live
    /// in the kernel's op slab AND reachable from the op list head.
    ///
    /// Temporary debugging aid: called at the entry of every tensor op.
    pub(crate) fn verify_tensor_invariants(&self) {
        if !cfg!(debug_assertions) {
            return;
        }
        for (tid, td) in self.tensors.iter() {
            let (kernel_id, op_id) = match td {
                TensorData::Eager { kernel_id, op_id, .. } | TensorData::Promoted { kernel_id, op_id, .. } => (*kernel_id, *op_id),
                _ => continue,
            };
            if kernel_id.is_null() {
                continue;
            }
            assert!(
                self.kernels.contains_id(kernel_id),
                "verify: tensor {tid} points at deleted kernel {kernel_id:?}"
            );
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
    /// from exactly these six slab variants — nothing else participates in a
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
    /// - `Stack` — grouped into a single `Op::Stack`.
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

        // Flatten the tree post-order: every node lands after its operands,
        // so the flat emit loop below always finds children already mapped.
        fn flatten(rt: &Runtime, x: TensorId, order: &mut Vec<TensorId>) {
            match &rt.tensors[x] {
                TensorData::Constant { .. } | TensorData::Variable { .. } => (),
                TensorData::Cast { x: a, .. } => flatten(rt, *a, order),
                TensorData::Unary { x: a, .. } => flatten(rt, *a, order),
                TensorData::Binary { x: a, y: b, .. } => {
                    flatten(rt, *a, order);
                    flatten(rt, *b, order);
                }
                TensorData::Stack { tensors, .. } => {
                    for &t in tensors.iter() {
                        flatten(rt, t, order);
                    }
                }
                t => panic!(
                    "symbolic expression tid {x} contains non-symbolic tensor data {t:?}; scalars must be built from constants, variables and dim ops"
                ),
            }
            order.push(x);
        }
        let mut order = Vec::new();
        flatten(self, shape, &mut order);

        let mut op_map: Map<TensorId, OpId> = Map::with_hasher(BuildHasherDefault::new());
        let mut root = OpId::NULL;
        for tid in order {
            // Copy the node's fields out so the slab borrow ends before we
            // touch the kernel.
            let op_id = match self.tensors[tid] {
                // Constants keep their own dtype; linearize's autocast handles
                // mixing with the surrounding expression.
                TensorData::Constant { value, .. } => self.kernels[kid].kernel.push_back(Op::Const(value)),
                TensorData::Variable { value, .. } => {
                    let op_id = self.kernels[kid].kernel.param(value.dtype(), ParamKind::Variable, OpId::NULL);
                    self.kernels[kid].loads.push(tid);
                    self.retain(tid);
                    op_id
                }
                TensorData::Cast { x, dtype, .. } => {
                    let a = op_map[&x];
                    self.kernels[kid].kernel.cast(a, dtype)
                }
                TensorData::Unary { x, uop, .. } => {
                    let a = op_map[&x];
                    self.kernels[kid].kernel.unary(a, uop)
                }
                TensorData::Binary { x, y, bop, .. } => {
                    let (a, b) = (op_map[&x], op_map[&y]);
                    self.kernels[kid].kernel.binary(a, b, bop)
                }
                TensorData::Stack { ref tensors, .. } => {
                    let ops: Vec<OpId> = tensors.iter().map(|t| op_map[t]).collect();
                    self.kernels[kid].kernel.stack(&ops)
                }
                ref t => unreachable!("flatten rejected non-symbolic data {t:?}"),
            };
            op_map.insert(tid, op_id);
            root = op_id;
        }
        root
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
    /// - `Constant` → `Const` class (never merged — see [`Node::Const`]).
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
    /// - `Stack` → a `Stack` node (or folded away for len < 2).
    ///
    /// The same closed-set rule applies: anything else panics here.
    fn replay_symbolic_into_graph(&mut self, graph_id: GraphId, shape: TensorId) -> ClassId {
        // DFS post-order flatten: every node lands after its operands.
        fn flatten(rt: &Runtime, x: TensorId, order: &mut Vec<TensorId>) {
            match &rt.tensors[x] {
                TensorData::Constant { .. } | TensorData::Variable { .. } => (),
                TensorData::Cast { x: a, .. } => flatten(rt, *a, order),
                TensorData::Unary { x: a, .. } => flatten(rt, *a, order),
                TensorData::Binary { x: a, y: b, .. } => {
                    flatten(rt, *a, order);
                    flatten(rt, *b, order);
                }
                TensorData::Stack { tensors, .. } => {
                    for &t in tensors.iter() {
                        flatten(rt, t, order);
                    }
                }
                t => panic!("symbolic expression tid {x} contains non-symbolic tensor data {t:?}"),
            }
            order.push(x);
        }
        let mut order = Vec::new();
        flatten(self, shape, &mut order);

        let mut class_map: Map<TensorId, ClassId> = Map::with_hasher(BuildHasherDefault::new());
        let mut root = ClassId::NULL;
        for tid in order {
            let class_id = match self.tensors[tid] {
                TensorData::Constant { value, .. } => self.push_const(graph_id, value),
                TensorData::Variable { .. } => self.push_leaf_node(graph_id, IDX_T, ClassId::NULL).1,
                TensorData::Cast { x, dtype, .. } => {
                    let a = class_map[&x];
                    self.push_node(graph_id, Node::Cast { x: a, dtype }).1
                }
                TensorData::Unary { x, uop, .. } => {
                    let a = class_map[&x];
                    self.push_node(graph_id, Node::Unary { x: a, uop }).1
                }
                TensorData::Binary { x, y, bop, .. } => {
                    let a = class_map[&x];
                    let b = class_map[&y];
                    self.push_binary_node(graph_id, a, b, bop)
                }
                TensorData::Stack { ref tensors, .. } => {
                    let ops: Vec<ClassId> = tensors.iter().map(|t| class_map[t]).collect();
                    match ops.len() {
                        0 => ClassId::NULL,
                        1 => ops[0],
                        _ => self.push_node(graph_id, Node::Stack { ops: ops.into_boxed_slice() }).1,
                    }
                }
                ref t => unreachable!("flatten rejected non-symbolic data {t:?}"),
            };
            class_map.insert(tid, class_id);
            root = class_id;
        }
        root
    }

    pub fn new_eager_tensor(&mut self, shape_id: TensorId, dtype: DType, kind: ParamKind) -> TensorId {
        let kernel_id = self.kernels.push(KernelData {
            outputs: Set::default(),
            loads: Vec::new(),
            stores: Vec::new(),
            kernel: Kernel::new(DeviceId::AUTO),
        });
        let shape_op = self.replay_symbolic_into_kernel(kernel_id, shape_id);
        let op_id = self.kernels[kernel_id].kernel.push_back(Op::Param { dtype, kind, shape: shape_op });
        // rc: 2 — one reference for the caller's handle, one for the kernel's
        // own self-load edge (see `release`'s eager arm, which releases that
        // edge through the op-chain prune).
        let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 2 });
        #[cfg(feature = "debug_tensor_op")]
        println!("rc::new_eager_tensor -> tid={tid} kernel_id={kernel_id:?} shape_id={shape_id} rc=2 (handle + self-load)");
        self.kernels[kernel_id].loads.push(tid);
        self.kernels[kernel_id].outputs.insert(tid);
        tid
    }

    pub fn new_constant_tensor(&mut self, value: Constant) -> TensorId {
        // Constants are pure slab entries: value lives in TensorData, no
        // kernel is allocated. Consumers replay the value into their own
        // kernels via Op::Const when needed.
        self.tensors.push(TensorData::Constant { value, rc: 1 })
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
        // Variables are pure slab entries + one slot in variable_map; no
        // kernel, no buffer_map entry. Kernels replay them as
        // Param { Variable } loads (see replay_shape_into_kernel).
        let value = Constant::new(x);
        let tid = self.tensors.push(TensorData::Variable { value, rc: 1 });
        self.variable_map.insert(tid, value);
        tid
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
        self.initialize_backends();

        let bytes = (data.len() * dtype.bit_size() as usize).div_ceil(8);
        debug_assert_eq!(data.len() * std::mem::size_of::<T>(), bytes);

        // Allocate one element extra so masked store writes to the trash
        // element stay within bounds (eager tensors can become store
        // targets, e.g. in-place assign).
        let alloc_bytes = bytes + dtype.bit_size() as usize / 8;
        // Store to Host memory
        let MemoryPool::Host(ref mut pool) = self.pools[PoolId::HOST] else {
            unreachable!("Host must exist.")
        };
        let free_bytes = pool.free_bytes();
        if alloc_bytes as Dim > free_bytes {
            return Err(ZyxError::AllocationError(
                format!("Attempted to allocate {alloc_bytes} B on host, but it only has {free_bytes} B free").into(),
            ));
        }

        let mut buf = vec![0u8; alloc_bytes].into_boxed_slice();
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), bytes) };
        buf[..bytes].copy_from_slice(src);

        let buffer_id = BufferId { pool: PoolId::HOST, buffer: pool.insert(buf) };

        // The caller keeps its own handle on `shape`; new_eager_tensor
        // consumes one reference.
        self.retain(shape);
        let tid = self.new_eager_tensor(shape, dtype, ParamKind::Global);

        self.buffer_map.insert(tid, buffer_id);

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
        self.initialize_backends();
        let resolved = self.resolve_shape(shape);
        let bytes: Dim = ((resolved.iter().product::<Dim>() * dtype.bit_size() as Dim) + 7) / 8;

        let pool = self.pools[PoolId::DISK]
            .disk_pool()
            .ok_or(BackendError { status: ErrorStatus::Initialization, context: "[disk] not available.".into() })?;
        let buffer_id = BufferId { pool: PoolId::DISK, buffer: pool.buffer_from_path(bytes, path, offset_bytes) };

        // The caller keeps its own handle on `shape`; new_eager_tensor
        // consumes one reference.
        self.retain(shape);
        let tid = self.new_eager_tensor(shape, dtype, ParamKind::Global);
        self.buffer_map.insert(tid, buffer_id);
        Ok(tid)
    }

    pub fn cast(&mut self, x: TensorId, dtype: DType) -> TensorId {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::cast(x={x}, dtype={dtype:?})");

        match self.tensors[x] {
            TensorData::Constant { value, .. } => self.new_constant_tensor(value.cast(dtype)),
            TensorData::Variable { .. }
            | TensorData::Cast { .. }
            | TensorData::Unary { .. }
            | TensorData::Binary { .. }
            | TensorData::Stack { .. } => {
                let tid = self.tensors.push(TensorData::Cast { x, dtype, rc: 1 });
                // The cast node holds an edge to x.
                self.retain(x);
                tid
            }
            TensorData::Eager { kernel_id, op_id, shape_id, .. } => {
                let op_id = self.kernels[kernel_id].kernel.cast(op_id, dtype);
                let tid =
                    self.tensors.push(TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 1 });
                self.kernels[kernel_id].outputs.insert(tid);
                // The cast shares the input's shape expression.
                self.retain(shape_id);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            TensorData::Graph { class_id, graph_id, shape_id, .. }
            | TensorData::Promoted { class_id, graph_id, shape_id, .. } => {
                self.assert_graph_alive(graph_id);
                let (_, class_id) = self.push_node(graph_id, Node::Cast { x: class_id, dtype });
                self.graphs[graph_id].ref_count += 1;
                // Shape-preserving op: share the input's shape expression, like eager.
                debug_assert!(!shape_id.is_null(), "cast: input graph tensor {x} has no shape expression");
                self.retain(shape_id);
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

        match self.tensors[x] {
            TensorData::Constant { .. }
            | TensorData::Variable { .. }
            | TensorData::Cast { .. }
            | TensorData::Unary { .. }
            | TensorData::Binary { .. }
            | TensorData::Stack { .. } => {
                todo!("bitcast of pure-symbolic tensors")
            }
            TensorData::Eager { kernel_id, op_id, shape_id, .. } => {
                let op_id = self.kernels[kernel_id].kernel.bitcast(op_id, dtype);
                // The bitcast shares the input's shape expression.
                self.retain(shape_id);
                let tid =
                    self.tensors.push(TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 1 });
                self.kernels[kernel_id].outputs.insert(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            TensorData::Graph { class_id, graph_id, shape_id, .. }
            | TensorData::Promoted { class_id, graph_id, shape_id, .. } => {
                self.assert_graph_alive(graph_id);
                let (_, class_id) = self.push_node(graph_id, Node::Cast { x: class_id, dtype });
                self.graphs[graph_id].ref_count += 1;
                // Shape-preserving op: share the input's shape expression, like eager.
                debug_assert!(!shape_id.is_null(), "bitcast: input graph tensor {x} has no shape expression");
                self.retain(shape_id);
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
        debug_assert!(!self.resolve_shape(x).is_empty(), "unary input must have at least one dim");

        match self.tensors[x] {
            TensorData::Constant { value, .. } => self.new_constant_tensor(value.unary(uop)),
            TensorData::Variable { .. }
            | TensorData::Cast { .. }
            | TensorData::Unary { .. }
            | TensorData::Binary { .. }
            | TensorData::Stack { .. } => {
                let tid = self.tensors.push(TensorData::Unary { x, uop, rc: 1 });
                // The unary node holds an edge to x.
                self.retain(x);
                tid
            }
            TensorData::Eager { kernel_id, op_id, shape_id, dtype, .. } => {
                let op_id = self.kernels[kernel_id].kernel.unary(op_id, uop);
                let tid =
                    self.tensors.push(TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 1 });
                self.kernels[kernel_id].outputs.insert(tid);
                // The unary shares the input's shape expression.
                self.retain(shape_id);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            TensorData::Graph { class_id, graph_id, shape_id, dtype, .. }
            | TensorData::Promoted { class_id, graph_id, shape_id, dtype, .. } => {
                self.assert_graph_alive(graph_id);
                let (_node_id, class_id) = self.push_node(graph_id, Node::Unary { x: class_id, uop });
                self.graphs[graph_id].ref_count += 1;
                // Shape-preserving op: share the input's shape expression, like eager.
                debug_assert!(!shape_id.is_null(), "unary: input graph tensor {x} has no shape expression");
                self.retain(shape_id);
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
        let x_sym = matches!(
            self.tensors[x],
            TensorData::Constant { .. }
                | TensorData::Variable { .. }
                | TensorData::Cast { .. }
                | TensorData::Unary { .. }
                | TensorData::Binary { .. }
                | TensorData::Stack { .. }
        );
        let y_sym = matches!(
            self.tensors[y],
            TensorData::Constant { .. }
                | TensorData::Variable { .. }
                | TensorData::Cast { .. }
                | TensorData::Unary { .. }
                | TensorData::Binary { .. }
                | TensorData::Stack { .. }
        );
        if x_sym && y_sym {
            let tid = self.tensors.push(TensorData::Binary { x, y, bop, rc: 1 });
            // The node holds an edge to both operands.
            self.retain(x);
            self.retain(y);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> symbolic: tid={tid}");
            return Ok(tid);
        }

        // Result shape: NULL when both sides are scalar, otherwise the
        // non-scalar side's shape expression (x wins when both are non-scalar;
        // they are asserted equal above).
        fn result_shape(rt: &Runtime, a: TensorId, b: TensorId) -> TensorId {
            let sa = match rt.tensors[a] {
                TensorData::Eager { shape_id, .. }
                | TensorData::Graph { shape_id, .. }
                | TensorData::Promoted { shape_id, .. } => shape_id,
                _ => TensorId::NULL,
            };
            let sb = match rt.tensors[b] {
                TensorData::Eager { shape_id, .. }
                | TensorData::Graph { shape_id, .. }
                | TensorData::Promoted { shape_id, .. } => shape_id,
                _ => TensorId::NULL,
            };
            if sa.is_null() { sb } else { sa }
        }

        let x_is_graph = self.is_graph(x);
        let y_is_graph = self.is_graph(y);
        if x_is_graph || y_is_graph {
            let graph_id = if x_is_graph {
                match self.tensors[x] {
                    TensorData::Graph { graph_id, .. } | TensorData::Promoted { graph_id, .. } => graph_id,
                    ref t => unreachable!("{t:?}"),
                }
            } else {
                match self.tensors[y] {
                    TensorData::Graph { graph_id, .. } | TensorData::Promoted { graph_id, .. } => graph_id,
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
                TensorData::Graph { class_id, .. } | TensorData::Promoted { class_id, .. } => class_id,
                // Pure-slab scalars enter the graph as nodes.
                TensorData::Constant { value, .. } => self.push_const(graph_id, value),
                ref t
                    if matches!(
                        t,
                        TensorData::Variable { .. }
                            | TensorData::Cast { .. }
                            | TensorData::Unary { .. }
                            | TensorData::Binary { .. }
                            | TensorData::Stack { .. }
                    ) =>
                {
                    todo!("promote symbolic scalar tid {x} ({t:?}) into a graph")
                }
                ref t => unreachable!("unreachable after promote: {t:?}"),
            };
            let cy = match self.tensors[y] {
                TensorData::Graph { class_id, .. } | TensorData::Promoted { class_id, .. } => class_id,
                TensorData::Constant { value, .. } => self.push_const(graph_id, value),
                ref t
                    if matches!(
                        t,
                        TensorData::Variable { .. }
                            | TensorData::Cast { .. }
                            | TensorData::Unary { .. }
                            | TensorData::Binary { .. }
                            | TensorData::Stack { .. }
                    ) =>
                {
                    todo!("promote symbolic scalar tid {y} ({t:?}) into a graph")
                }
                ref t => unreachable!("unreachable after promote: {t:?}"),
            };
            let class_id = self.push_binary_node(graph_id, cx, cy, bop);

            {
                let shape_id = result_shape(self, x, y);
                debug_assert!(!shape_id.is_null(), "binary: non-scalar graph operands {x}/{y} have no shape expression");
                self.retain(shape_id);
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
            self.retain(shape_id);
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
            let tid = self.tensors.push(TensorData::Eager {
                kernel_id: kid,
                op_id,
                depends_on: KernelId::NULL,
                shape_id,
                dtype,
                rc: 1,
            });
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
            self.retain(shape_id);
            let (mut kid_x, mut op_id_x) = match self.tensors[x] {
                TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                ref t => panic!("binary: operand tid {x} is not an eager tensor: {t:?}"),
            };
            let (mut kid_y, mut op_id_y) = match self.tensors[y] {
                TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                ref t => panic!("binary: operand tid {y} is not an eager tensor: {t:?}"),
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
                // add_store may have re-created the operands as load params.
                (kid_x, op_id_x) = match self.tensors[x] {
                    TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                    ref t => unreachable!("add_store turned operand into non-eager data: {t:?}"),
                };
                (kid_y, op_id_y) = match self.tensors[y] {
                    TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                    ref t => unreachable!("add_store turned operand into non-eager data: {t:?}"),
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
            let tid =
                self.tensors.push(TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 1 });
            self.kernels[kernel_id].outputs.insert(tid);

            #[cfg(feature = "debug_tensor_op")]
            println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            Ok(tid)
        }
    }

    #[allow(clippy::wrong_self_convention)] // naming convention from GPU API, not a conversion method
    pub fn to_device(&mut self, x: TensorId, device_id: DeviceId) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::to_device(x={x}, device_id={device_id:?})");
        let (class_id, graph_id, shape_id) = match self.tensors[x] {
            TensorData::Graph { class_id, graph_id, shape_id, .. }
            | TensorData::Promoted { class_id, graph_id, shape_id, .. } => (class_id, graph_id, shape_id),
            ref t => panic!("to_device: tensor tid {x} is not graph-affiliated (graph-only op for now): {t:?}"),
        };
        assert!(!self.graphs[graph_id].dead, "tape scope has ended (tensor belongs to a dead tape scope");
        // TODO measure actual time by running a test copy
        let (_node_id, cid) = self.push_node(graph_id, Node::ToDevice { x: class_id, device: device_id, time: 0 });
        self.graphs[graph_id].ref_count += 1;
        // Shape-preserving op: share the input's shape expression.
        debug_assert!(!shape_id.is_null(), "to_device: input graph tensor {x} has no shape expression");
        self.retain(shape_id);
        let dtype = self.dtype(x);
        let tid = self.tensors.push(TensorData::Graph { class_id: cid, graph_id, shape_id, dtype, rc: 1 });
        #[cfg(feature = "debug_tensor_op")]
        println!("  -> tid={tid}, nid={_node_id:?}, cid={cid:?}");
        Ok(tid)
    }

    /// Forces a contiguous, materialized view of `x` (breaks aliasing / forces a fresh buffer).
    ///
    /// Three branches, matching the tensor's storage kind:
    /// - **Graph** (`is_graph(x)`): pushes a `Node::Contiguous` and shares `x`'s shape
    ///   expression. The kernelizer's `Node::Contiguous` arm applies a same-dtype `Cast`
    ///   (value identity) and stores the new class — giving it a distinct op and its own
    ///   backing buffer instead of aliasing `x`'s load op.
    /// - **Already realized** (in `buffer_map`): a no-op — the tensor is a load from its own
    ///   contiguous buffer, so the returned handle is `x` (retained).
    /// - **Eager (unrealized)**: a cast-shim — emit a same-dtype `Cast` op in `x`'s kernel
    ///   and store the cast tensor as a new handle, leaving `x` itself unfused in its producer.
    pub fn contiguous(&mut self, x: TensorId) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::contiguous(x={x})");
        self.verify_tensor_invariants();

        if self.is_graph(x) {
            let (class_id, graph_id, shape_id) = match self.tensors[x] {
                TensorData::Graph { class_id, graph_id, shape_id, .. }
                | TensorData::Promoted { class_id, graph_id, shape_id, .. } => (class_id, graph_id, shape_id),
                ref t => unreachable!("{t:?}"),
            };
            let (_node_id, cid) = self.push_node(graph_id, Node::Contiguous { x: class_id });
            self.graphs[graph_id].ref_count += 1;
            // Shape-preserving op: share the input's shape expression.
            debug_assert!(!shape_id.is_null(), "contiguous: input graph tensor {x} has no shape expression");
            self.retain(shape_id);
            let dtype = self.dtype(x);
            let tid = self.tensors.push(TensorData::Graph { class_id: cid, graph_id, shape_id, dtype, rc: 1 });
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> tid={tid}, nid={_node_id:?}, cid={cid:?}");
            Ok(tid)
        } else if self.buffer_map.contains_key(&x) {
            // Already realized: the tensor is a load from its own contiguous
            // buffer, so this is a no-op. Mirror reshape's already-resized path.
            self.retain(x);
            Ok(x)
        } else {
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
                    stacked
                } else {
                    self.stack(&dims)?
                };
                let (_node_id, class_id) =
                    self.push_node(graph_id, Node::Reduce { x: class_id, rop, axes: axes.into_boxed_slice() });
                self.graphs[graph_id].ref_count += 1;
                self.retain(shape_id);
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                Ok(tid)
            }
            TensorData::Eager { dtype, .. } => {
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
                        TensorId::NULL
                    } else {
                        self.stack(&kept_dims)?
                    };

                    let tid = self.tensors.push(TensorData::Eager {
                        kernel_id: kid,
                        op_id,
                        depends_on: KernelId::NULL,
                        shape_id,
                        dtype,
                        rc: 1,
                    });
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
                    let shape_id = self.stack(&[one_const])?;
                    self.release(one_const);
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
        if tensors.iter().all(|&t| {
            matches!(
                self.tensors[t],
                TensorData::Constant { .. }
                    | TensorData::Variable { .. }
                    | TensorData::Cast { .. }
                    | TensorData::Unary { .. }
                    | TensorData::Binary { .. }
                    | TensorData::Stack { .. }
            )
        }) {
            let tid = self.tensors.push(TensorData::Stack { tensors: tensors.into(), rc: 1 });
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
                if !self.is_graph(t) && !matches!(self.tensors[t], TensorData::Constant { .. }) {
                    self.promote_to_graph(t, graph_id)?;
                }
            }
            let mut ops = Vec::with_capacity(tensors.len());
            for &t in tensors {
                ops.push(match self.tensors[t] {
                    TensorData::Graph { class_id, .. } | TensorData::Promoted { class_id, .. } => class_id,
                    TensorData::Constant { value, .. } => self.push_const(graph_id, value),
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
                let shape_id = self.stack(&shape_dims)?;
                self.release(len_const);
                self.graphs[graph_id].ref_count += 1;
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                Ok(tid)
            }
        } else {
            let keep_kid = match self.tensors[tensors[0]] {
                TensorData::Eager { kernel_id, .. } => kernel_id,
                ref t => panic!("stack: operand tid {} is not an eager tensor: {t:?}", tensors[0]),
            };
            let mut ops = Vec::with_capacity(tensors.len());
            for &t in tensors {
                let (mut kid, mut op) = match self.tensors[t] {
                    TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                    ref t => panic!("stack: operand is not an eager tensor: {t:?}"),
                };
                if kid != keep_kid {
                    if !self.kernels[kid].stores.is_empty() {
                        self.add_store(t)?;
                        (kid, op) = match self.tensors[t] {
                            TensorData::Eager { kernel_id, op_id, .. } => (kernel_id, op_id),
                            ref t => unreachable!("{t:?}"),
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
            let shape_id = self.stack(&shape_dims)?;
            self.release(len_const);

            let tid =
                self.tensors.push(TensorData::Eager { kernel_id: keep_kid, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 1 });
            self.kernels[keep_kid].outputs.insert(tid);
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> eager: tid={tid}, kid={keep_kid:?}, op_id={op_id:?}");
            Ok(tid)
        }
    }

    pub(super) fn reshape(&mut self, x: TensorId, shape_id: TensorId) -> Result<TensorId, ZyxError> {
        #[cfg(feature = "debug_tensor_op")]
        println!("runtime::reshape(x={x}, shape={shape_id:?})");
        // Shapes are always resolvable on the tensor side (closed expressions
        // over variable_map), so this is a total check.
        debug_assert_eq!(
            self.resolve_shape(x).iter().product::<Dim>(),
            self.resolve_shape(shape_id).iter().product::<Dim>(),
            "reshape element count mismatch"
        );

        let dtype = self.dtype(x);

        if self.is_graph(x) || self.is_graph(shape_id) {
            let graph_id = if self.is_graph(x) {
                match self.tensors[x] {
                    TensorData::Graph { graph_id, .. } | TensorData::Promoted { graph_id, .. } => graph_id,
                    ref t => unreachable!("{t:?}"),
                }
            } else {
                match self.tensors[shape_id] {
                    TensorData::Graph { graph_id, .. } | TensorData::Promoted { graph_id, .. } => graph_id,
                    ref t => unreachable!("{t:?}"),
                }
            };
            self.assert_graph_alive(graph_id);
            if !self.is_graph(x) {
                self.promote_to_graph(x, graph_id)?;
            }
            let x_class = match self.tensors[x] {
                TensorData::Graph { class_id, .. } | TensorData::Promoted { class_id, .. } => class_id,
                ref t => unreachable!("{t:?}"),
            };
            // The target shape enters the graph: a graph-affiliated shape is
            // used directly (same scope asserted); a slab-side symbolic
            // expression is promoted node by node.
            let shape_class = match self.tensors[shape_id] {
                TensorData::Graph { class_id, graph_id: g, .. } | TensorData::Promoted { class_id, graph_id: g, .. } => {
                    assert!(g == graph_id, "reshape: shape belongs to a different tape scope");
                    class_id
                }
                _ => self.replay_symbolic_into_graph(graph_id, shape_id),
            };
            let (_, class_id) = self.push_node(graph_id, Node::Reshape { x: x_class, shape: shape_class });
            {
                self.graphs[graph_id].ref_count += 1;
                self.retain(shape_id);
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                Ok(tid)
            }
        } else {
            // If x is realized, create a load kernel with the target shape.
            // The result shares x's buffer (set in buffer_map), so add_store
            // won't add a StoreView for it. This avoids copying data for a
            // view-only reshape.
            if let Some(&buf_id) = self.buffer_map.get(&x) {
                let kernel_id = self.kernels.push(KernelData {
                    outputs: Set::default(),
                    loads: Vec::new(),
                    stores: Vec::new(),
                    kernel: Kernel::new(DeviceId::AUTO),
                });
                let shape_op = self.replay_symbolic_into_kernel(kernel_id, shape_id);
                let dtype = self.dtype(x);
                let op_id = self.kernels[kernel_id].kernel.param(dtype, ParamKind::Global, shape_op);
                if !shape_id.is_null() {
                    self.retain(shape_id);
                }
                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 2 });
                eprintln!(">>> NEW_EAGER tid={tid} kernel_id={kernel_id:?}");
                self.kernels[kernel_id].loads.push(tid);
                self.kernels[kernel_id].outputs.insert(tid);
                self.buffer_map.insert(tid, buf_id);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid} (load kernel, shares buffer with x={x})");
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
            if !shape_id.is_null() {
                self.retain(shape_id);
            }
            let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 1 });

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
        let dtype = self.dtype(x);
        let sh = self.resolve_shape(x);
        let target = self.resolve_shape(shape_id);
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

        if self.is_graph(x) {
            let graph_id = match self.tensors[x] {
                TensorData::Graph { graph_id, .. } | TensorData::Promoted { graph_id, .. } => graph_id,
                ref t => unreachable!("{t:?}"),
            };
            self.assert_graph_alive(graph_id);
            if !self.is_graph(x) {
                self.promote_to_graph(x, graph_id)?;
            }
            let x_class = match self.tensors[x] {
                TensorData::Graph { class_id, .. } | TensorData::Promoted { class_id, .. } => class_id,
                ref t => unreachable!("{t:?}"),
            };
            let shape_class = match self.tensors[shape_id] {
                TensorData::Graph { class_id, graph_id: g, .. } | TensorData::Promoted { class_id, graph_id: g, .. } => {
                    assert!(g == graph_id, "expand: shape belongs to a different tape scope");
                    class_id
                }
                _ => self.replay_symbolic_into_graph(graph_id, shape_id),
            };
            let (_, class_id) = self.push_node(graph_id, Node::Expand { x: x_class, shape: shape_class });
            {
                self.graphs[graph_id].ref_count += 1;
                self.retain(shape_id);
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                Ok(tid)
            }
        } else if matches!(
            self.tensors[x],
            TensorData::Constant { .. }
                | TensorData::Variable { .. }
                | TensorData::Cast { .. }
                | TensorData::Unary { .. }
                | TensorData::Binary { .. }
                | TensorData::Stack { .. }
        ) {
            // Pure-slab operand (e.g. a broadcast scalar): materialize it into
            // a fresh eager kernel that replays the slab expression and
            // expands it to the target shape.
            let kid = self.kernels.push(KernelData {
                outputs: Set::default(),
                loads: Vec::new(),
                stores: Vec::new(),
                kernel: Kernel::new(DeviceId::AUTO),
            });
            let val_op = self.replay_symbolic_into_kernel(kid, x);
            let shape_op = self.replay_symbolic_into_kernel(kid, shape_id);
            let op_id = self.kernels[kid].kernel.expand(val_op, shape_op);
            if !shape_id.is_null() {
                self.retain(shape_id);
            }
            let tid = self.tensors.push(TensorData::Eager { kernel_id: kid, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 1 });
            self.kernels[kid].outputs.insert(tid);
            #[cfg(feature = "debug_tensor_op")]
            println!("runtime::expand(x={x}) -> eager from slab: tid={tid}, kid={kid:?}, op_id={op_id:?}");
            Ok(tid)
        } else {
            let force_store = match self.tensors[x] {
                TensorData::Eager { kernel_id, op_id, .. } => self.kernels[kernel_id].kernel.is_preceded_by_compute(op_id),
                ref t => panic!("expand: operand tid {x} is not an eager tensor: {t:?}"),
            };
            let (kernel_id, op_id) = self.duplicate_or_store(x, force_store)?;

            debug_assert_eq!(
                self.kernels[kernel_id].outputs.len(),
                0,
                "input into expand must have empty outputs before the shape kernel is merged"
            );
            let shape_op = self.replay_symbolic_into_kernel(kernel_id, shape_id);
            let op_id = self.kernels[kernel_id].kernel.expand(op_id, shape_op);
            self.retain(shape_id);
            let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 1 });

            debug_assert_eq!(self.kernels[kernel_id].outputs.contains(&tid), false);
            self.kernels[kernel_id].outputs.insert(tid);

            #[cfg(feature = "debug_tensor_op")]
            println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
            Ok(tid)
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

        // Result shape: x's dims in the new axis order. The stack's rc is
        // transferred into the result tensor.
        let shape_id = {
            let dims = self.shape(x);
            let permuted = crate::shape::permute(&dims, &axes);
            if permuted.is_empty() {
                TensorId::NULL
            } else {
                self.stack(&permuted).expect("permute: failed to build shape stack")
            }
        };

        match self.tensors[x] {
            TensorData::Graph { class_id, graph_id, dtype, .. } | TensorData::Promoted { class_id, graph_id, dtype, .. } => {
                self.assert_graph_alive(graph_id);
                let (_, class_id) = self.push_node(graph_id, Node::Permute { x: class_id, axes: axes.into_boxed_slice() });
                self.graphs[graph_id].ref_count += 1;
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> graph: tid={tid}, graph_id={graph_id:?}, class_id={class_id:?}");
                tid
            }
            TensorData::Eager { dtype, .. } => {
                let (kernel_id, op_id) = self.duplicate_or_store(x, false).unwrap();
                let op_id = self.kernels[kernel_id]
                    .kernel
                    .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes: axes.into() }) });
                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 1 });
                debug_assert_eq!(self.kernels[kernel_id].outputs.len(), 0, "input into permute must have empty outputs");
                self.kernels[kernel_id].outputs.insert(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            ref t => todo!("permute of pure-slab tensor {t:?}"),
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
        // directly (total-length semantics).
        let shape_id = {
            let mut dims = self.shape(x);
            dims[axis as usize] = len;
            self.retain(len);
            self.stack(&dims).expect("pad_zeros: failed to build shape stack")
        };

        match self.tensors[x] {
            TensorData::Graph { class_id, graph_id, dtype, .. } | TensorData::Promoted { class_id, graph_id, dtype, .. } => {
                self.assert_graph_alive(graph_id);
                let lp_class = match self.tensors[lp] {
                    TensorData::Graph { class_id, graph_id: g, .. } | TensorData::Promoted { class_id, graph_id: g, .. } => {
                        assert!(g == graph_id, "pad_zeros: lp belongs to a different tape scope");
                        class_id
                    }
                    _ => self.replay_symbolic_into_graph(graph_id, lp),
                };
                let len_class = match self.tensors[len] {
                    TensorData::Graph { class_id, graph_id: g, .. } | TensorData::Promoted { class_id, graph_id: g, .. } => {
                        assert!(g == graph_id, "pad_zeros: len belongs to a different tape scope");
                        class_id
                    }
                    _ => self.replay_symbolic_into_graph(graph_id, len),
                };
                let (_, class_id) = self.push_node(graph_id, Node::Pad { x: class_id, axis, lp: lp_class, len: len_class });
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                self.graphs[graph_id].ref_count += 1;
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> graph: tid={tid}, graph_id={graph_id:?}, class_id={class_id:?}");
                tid
            }
            TensorData::Eager { kernel_id: xkid, op_id: xop, dtype, .. } => {
                // Duplicate only when the pad actually grows the tensor AND
                // compute precedes it in the kernel (conv layers need this).
                let len_const = self
                    .resolve_symbolic(len)
                    .expect("pad_zeros: eager-arm len bound must be a resolvable scalar")
                    .as_dim()
                    .expect("pad_zeros: len bound does not evaluate to an integer");
                let grows = len_const > self.resolve_shape(x)[axis as usize];
                let force_store = grows && self.kernels[xkid].kernel.is_preceded_by_compute(xop);
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
                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 1 });
                self.kernels[kernel_id].outputs.insert(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            ref t => todo!("pad_zeros of pure-slab tensor {t:?}"),
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
        let shape_id = {
            let mut dims = self.shape(x);
            dims[axis as usize] = len;
            self.retain(len);
            self.stack(&dims).expect("narrow: failed to build shape stack")
        };

        match self.tensors[x] {
            TensorData::Graph { class_id, graph_id, dtype, .. } | TensorData::Promoted { class_id, graph_id, dtype, .. } => {
                self.assert_graph_alive(graph_id);
                let start_class = match self.tensors[start] {
                    TensorData::Graph { class_id, graph_id: g, .. } | TensorData::Promoted { class_id, graph_id: g, .. } => {
                        assert!(g == graph_id, "narrow: start belongs to a different tape scope");
                        class_id
                    }
                    _ => self.replay_symbolic_into_graph(graph_id, start),
                };
                let len_class = match self.tensors[len] {
                    TensorData::Graph { class_id, graph_id: g, .. } | TensorData::Promoted { class_id, graph_id: g, .. } => {
                        assert!(g == graph_id, "narrow: len belongs to a different tape scope");
                        class_id
                    }
                    _ => self.replay_symbolic_into_graph(graph_id, len),
                };
                let (_, class_id) =
                    self.push_node(graph_id, Node::Narrow { x: class_id, axis, start: start_class, len: len_class });
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                self.graphs[graph_id].ref_count += 1;
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> graph: tid={tid}, graph_id={graph_id:?}, class_id={class_id:?}");
                tid
            }
            TensorData::Eager { dtype, .. } => {
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
                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 1 });
                self.kernels[kernel_id].outputs.insert(tid);
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> eager: tid={tid}, kid={kernel_id:?}, op_id={op_id:?}");
                tid
            }
            ref t => todo!("narrow of pure-slab tensor {t:?}"),
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
            TensorData::Eager { shape_id, .. } | TensorData::Graph { shape_id, .. } | TensorData::Promoted { shape_id, .. } => {
                shape_id
            }
            ref t => todo!("flip of pure-slab tensor {t:?}"),
        };
        if shape_id != TensorId::NULL {
            self.retain(shape_id);
        }

        match self.tensors[x] {
            TensorData::Graph { class_id, graph_id, dtype, .. } | TensorData::Promoted { class_id, graph_id, dtype, .. } => {
                self.assert_graph_alive(graph_id);
                let (_, class_id) = self.push_node(graph_id, Node::Flip { x: class_id, axes: axes.into_boxed_slice() });
                self.graphs[graph_id].ref_count += 1;
                let tid = self.tensors.push(TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: 1 });
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> graph: tid={tid}, graph_id={graph_id:?}, class_id={class_id:?}");
                Ok(tid)
            }
            TensorData::Eager { dtype, .. } => {
                let (kernel_id, op_id) = self.duplicate_or_store(x, false).unwrap();
                let op_id = self.kernels[kernel_id].kernel.flip(op_id, &axes);
                let tid = self.tensors.push(TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc: 1 });
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
        if !self.buffer_map.contains_key(&x) {
            if let Some(c) = self.resolve_symbolic(x) {
                let v = match c.cast(T::dtype()) {
                    Constant::BF16(v) => T::from_bf16(bf16::from_le_bytes(v)),
                    Constant::F16(v) => T::from_f16(f16::from_le_bytes(v)),
                    Constant::F32(v) => T::from_f32(f32::from_le_bytes(v)),
                    Constant::F64(v) => T::from_f64(f64::from_le_bytes(v)),
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

        // Fast path: variables (and fully-symbolic expressions over them)
        // live only in `variable_map` — no buffer, no pool storage.
        if let Some(value) = self.variable_map.get(&x).copied().or_else(|| self.resolve_symbolic(x)) {
            let bytes = (data.len() * T::bit_size() as usize).div_ceil(8);
            let byte_slice = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), bytes) };
            let value_bytes = value.to_le_bytes();
            byte_slice[..value_bytes.len()].copy_from_slice(&value_bytes);
            return Ok(());
        }

        // Fast path: already realized
        let Some(mut buffer_id) = self.buffer_map.get(&x).copied() else {
            let this = &mut *self;
            this.initialize_backends();
            let pending = match this.tensors[x] {
                TensorData::Eager { depends_on, .. } => depends_on,
                TensorData::Graph { .. } => return Err(ZyxError::graph_tensor_not_realized(x)),
                ref t => panic!("load: tensor {x} has no buffer and cannot be materialized: {t:?}"),
            };
            if !pending.is_null() {
                let outputs: Set<TensorId> = this.kernels[pending].outputs.iter().copied().collect();
                for tid in outputs {
                    this.add_store(tid)?;
                }
            }
            let kid = match this.tensors[x] {
                TensorData::Eager { kernel_id, .. } => kernel_id,
                TensorData::Graph { .. } => return Err(ZyxError::graph_tensor_not_realized(x)),
                ref t => panic!("load: tensor {x} has no buffer and cannot be materialized: {t:?}"),
            };
            let seen: Set<TensorId> = this.kernels[kid].outputs.iter().copied().collect();
            for tid in seen {
                this.add_store(tid)?;
            }
            let bytes = (data.len() * T::bit_size() as usize).div_ceil(8);
            let byte_slice = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), bytes) };
            let buffer_id = this.buffer_map[&x];
            for buffers in this.events.keys() {
                if buffers.contains(&buffer_id) {
                    let buffers = buffers.clone();
                    let event = this.events.remove(&buffers).unwrap();
                    this.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, vec![event])?;
                    #[cfg(feature = "debug_tensor_op")]
                    println!("  -> x={x}, {:?}", self.tensors[x]);
                    return Ok(());
                }
            }
            this.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, Vec::new())?;
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> x={x}, {:?}", self.tensors[x]);
            return Ok(());
        };

        // A store may still be pending on this tensor (assign wrote into
        // this buffer in place). Run the pending producer kernel first so
        // the buffer is up to date, then re-fetch the buffer id (the store
        // may have moved it to a device pool).
        let kid = match self.tensors[x] {
            TensorData::Eager { depends_on, .. } => depends_on,
            ref t => panic!("load: pending store check on non-kernel tensor {x}: {t:?}"),
        };
        if !kid.is_null() {
            let seen: Set<TensorId> = self.kernels[kid].outputs.iter().copied().collect();
            for tid in seen {
                self.add_store(tid)?;
            }
            buffer_id = self.buffer_map.get(&x).copied().ok_or_else(|| {
                ZyxError::AllocationError(format!("load: tensor {x} lost its buffer during pending store").into())
            })?;
        }
        let bytes = (data.len() * T::bit_size() as usize).div_ceil(8);
        let byte_slice = unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), bytes) };
        for buffers in self.events.keys() {
            if buffers.contains(&buffer_id) {
                let buffers = buffers.clone();
                let event = self.events.remove(&buffers).unwrap();
                self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, vec![event])?;
                #[cfg(feature = "debug_tensor_op")]
                println!("  -> x={x}, {:?}", self.tensors[x]);
                return Ok(());
            }
        }
        self.pools[buffer_id.pool].pool_to_host(buffer_id.buffer, byte_slice, Vec::new())?;
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
        if self.is_graph(dst) {
            // Graph-mode in-place assign: record a Node::Assign inside the tape
            // graph. The plan writes src's value into dst's buffer in-place; dst
            // is either a realized (promoted) leaf tensor or a movement view over
            // one (e.g. a slice), whose movement chain the kernelizer replays
            // into src's kernel so the store lands at the view's position.
            if dst == src {
                return Err(ZyxError::ShapeError("assign: dst equals src (self-assign)".into()));
            }
            let (dst_cid, graph_id) = match self.tensors[dst] {
                TensorData::Graph { class_id, graph_id, .. } | TensorData::Promoted { class_id, graph_id, .. } => {
                    (class_id, graph_id)
                }
                ref t => unreachable!("assign: is_graph(dst) guaranteed a graph-affiliated tensor: {t:?}"),
            };
            self.assert_graph_alive(graph_id);
            if self.is_graph(src) {
                let src_graph_id = match self.tensors[src] {
                    TensorData::Graph { graph_id, .. } | TensorData::Promoted { graph_id, .. } => graph_id,
                    ref t => unreachable!("{t:?}"),
                };
                if src_graph_id != graph_id {
                    panic!("tensor belongs to a different tape scope");
                }
            } else {
                self.promote_to_graph(src, graph_id)?;
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
                    _ => unreachable!(),
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
                TensorData::Graph { class_id, .. } | TensorData::Promoted { class_id, .. } => class_id,
                ref t => unreachable!("{t:?}"),
            };
            let (_node_id, assign_cid) = self.push_node(graph_id, Node::Assign { dst: dst_cid, src: src_cid });
            let leaf_class = self.push_node(graph_id, Node::After { x: dst_leaf_cid, dep: assign_cid }).1;
            let dst_class = self.push_node(graph_id, Node::After { x: dst_cid, dep: assign_cid }).1;
            for (tid, class_id) in [(dst_leaf, leaf_class), (dst, dst_class)] {
                match &mut self.tensors[tid] {
                    TensorData::Graph { class_id: c, .. } | TensorData::Promoted { class_id: c, .. } => *c = class_id,
                    ref t => panic!("assign: tensor {tid} has no graph class to re-point: {t:?}"),
                }
            }
            #[cfg(feature = "debug_tensor_op")]
            println!("  -> assign_cid={assign_cid:?}");
            return Ok(());
        }
        // Merge dst's (movement-only) kernel into src's kernel, then store src's
        // value into dst's base buffer in-place.
        let (src_kid, src_op) = match self.tensors[src] {
            TensorData::Eager { kernel_id, op_id, .. } | TensorData::Promoted { kernel_id, op_id, .. } => (kernel_id, op_id),
            ref t => panic!("assign: src {src} is not an eager/promoted tensor: {t:?}"),
        };
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

        // Validate the dst kernel's loads BEFORE removing it: a failed
        // validation must leave the kernel intact so the dst view tensor (still
        // pointing at dst_kid) stays valid and its later drop does not index a
        // deleted kernel.
        let dst_kernel_loads = self.kernels[dst_kid].loads.clone();
        let dst_org = {
            let mut buffer_loads = dst_kernel_loads.iter().copied().filter(|&t| !self.variable_map.contains_key(&t));
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
        // the producer kernel that holds the pending store.
        match self.tensors[dst_org] {
            TensorData::Eager { depends_on, .. } if !depends_on.is_null() => {
                assert!(
                    depends_on != src_kid,
                    "assign: dst base {dst_org} is pending on src's kernel {src_kid:?}; assign would interleave with its own store"
                );
            }
            _ => {}
        }
        // Remove the dst (movement-only) kernel; its base buffer is dst_org.
        // The removed kernel held a kernel-load reference on dst_org.
        let KernelData { kernel, loads, .. } = unsafe { self.kernels.remove_and_return(dst_kid) };
        {
            let holders: Vec<TensorId> = self
                .tensors
                .iter()
                .filter_map(|(tid, td)| match td {
                    TensorData::Eager { kernel_id: k, .. } | TensorData::Promoted { kernel_id: k, .. } if *k == dst_kid => Some(tid),
                    _ => None,
                })
                .collect();
            eprintln!("$$$ ASSIGN removing dst kernel {dst_kid:?} holders={holders:?}");
        }
        for t in &loads {
            assert!(
                *t == dst_org || self.variable_map.contains_key(t),
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
                Op::Storage { .. } => {
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
        self.kernels[src_kid].kernel.store(dst_op, src_op, OpId::NULL, MemLayout::Scalar);
        self.kernels[src_kid].stores.push(dst_org);
        // Register every replayed define's load in define order. Variables
        // only — the GlobalMut base is a PURE STORE now and must not appear
        // in loads (its buffer slot comes via `stores`).
        self.kernels[src_kid].loads.extend(new_def_loads);
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

        // Re-point dst BEFORE any cascade-triggering call below. The removal of
        // dst_kid left dst's `kernel_id` stale (pointing at a deleted kernel);
        // the force-materialize / release cascades must never observe it.
        // - dst owns the target buffer (dst == dst_org): keep it valid but
        //   pending on the store kernel so a read of dst runs the in-place
        //   write first. Re-point dst onto src_kid (as add_store does) so
        //   clone drops / releases target a live kernel instead of the store
        //   kernel that gets consumed on materialization.
        // - dst is a movement view (dst != dst_org): it has no buffer and is
        //   invalid after the in-place write — null it out immediately.
        let rehomed = dst == dst_org;
        if rehomed {
            self.kernels[src_kid].outputs.insert(dst);
        }
        match &mut self.tensors[dst] {
            TensorData::Eager { kernel_id, op_id, depends_on, .. } => {
                if rehomed {
                    *kernel_id = src_kid;
                    *op_id = dst_op;
                    *depends_on = src_kid;
                } else {
                    *kernel_id = KernelId::NULL;
                    *op_id = OpId::NULL;
                    *depends_on = KernelId::NULL;
                }
            }
            TensorData::Promoted { kernel_id, op_id, .. } => {
                if rehomed {
                    *kernel_id = src_kid;
                    *op_id = dst_op;
                } else {
                    *kernel_id = KernelId::NULL;
                    *op_id = OpId::NULL;
                }
            }
            ref t => panic!("assign: dst {dst} is not an eager/promoted tensor: {t:?}"),
        }

        // The base's backing store may still be deferred (e.g. `contiguous`
        // marks the store but leaves it unmaterialized for fusion). Assign
        // writes in-place, so the storage must exist NOW: force-materialize
        // the producer kernel that holds the pending store.
        let pending_kid = match self.tensors[dst_org] {
            TensorData::Eager { depends_on, .. } if !depends_on.is_null() => Some(depends_on),
            _ => None,
        };
        if let Some(kid) = pending_kid {
            // Convention (see `materialize_kernel`): a kernel is materialized by
            // `add_store`-ing all of its outputs — the last call materializes it
            // automatically. `materialize_kernel` must not be called directly on
            // a kernel with unstored outputs.
            for out in self.kernels[kid].outputs.clone() {
                self.add_store(out)?;
            }
        }
        // Drop the kernel-load edge the removed kernel held on dst_org; the
        // user's handle keeps it alive.
        self.release(dst_org);

        if rehomed {
            self.add_store(dst)?;
        } else {
            let seen: Set<TensorId> = self.kernels[src_kid].outputs.iter().copied().collect();
            for tid in seen {
                self.add_store(tid)?;
            }
        }

        Ok(())
    }

    /// Initializes all available devices, creating a device for each compute
    /// device and a memory pool for each physical memory.
    /// Does nothing if devices were already initialized.
    pub fn initialize_backends(&mut self) {
        if !self.pools.is_empty() {
            return;
        }

        // Set env vars
        if let Ok(x) = env::var("ZYX_DEBUG")
            && let Ok(x) = x.parse::<u32>()
        {
            self.debug = DebugMask(x);
        }

        // Search through config directory and find zyx/backend_config.json
        // If not found or failed to parse, use defaults.

        let config_file = env::var_os("XDG_CONFIG_HOME")
            .and_then(|path| {
                let path = PathBuf::from(path);
                if path.is_absolute() { Some(path) } else { None }
            })
            .or_else(|| env::home_dir().map(|home| home.join(".config")))
            .map(|path| path.join("zyx/config.json"))
            .and_then(|mut path| {
                if let Ok(file) = std::fs::read_to_string(&path) {
                    path.pop();
                    self.config_dir = Some(path);
                    Some(file)
                } else {
                    None
                }
            });

        let config = config_file
            .and_then(|file| {
                DeJson::deserialize_json(&file)
                    .map_err(|e| {
                        if self.debug.dev() {
                            println!("Failed to parse config.json, {e}");
                        }
                    })
                    .ok()
            })
            .inspect(|_| {
                if self.debug.dev() {
                    println!("Device config successfully read and parsed.");
                }
            })
            .unwrap_or_else(|| {
                if self.debug.dev() {
                    println!("Failed to get device config, using defaults.");
                }
                Config::default()
            });

        // Load optimizer cache from disk if it exists
        /*if let Some(mut path) = self.config_dir.clone() {
            path.push("cached_kernels");
            if let Ok(mut file) = std::fs::File::open(path) {
                use std::io::Read;
                let mut buf = Vec::new();
                file.read_to_end(&mut buf).unwrap();
                if let Ok(cache) = nanoserde::DeBin::deserialize_bin(&buf) {
                    self.kernel_cache = cache;
                }
            }
        }*/

        crate::backend::initialize_backends(&config, &mut self.pools, &mut self.devices, self.debug.dev());

        self.autotune_config = config.autotune;
        //println!("INIT runtime");
    }

    /// This function deinitializes the whole runtime, deallocates all allocated memory and deallocates all caches
    /// It does not reset the rng and it does not change debug, search, training and `config_dir` fields
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
        if self.pools.is_empty() {
            self.initialize_backends();
        }
        self.pools.iter().map(|(_, p)| p.free_bytes()).max().unwrap_or(0)
    }
}

#[allow(clippy::similar_names)]
pub fn get_perf(flop: Dim, bytes_read: u64, bytes_written: u64, nanos: u64) -> String {
    const fn value_unit(x: u64) -> (u64, &'static str) {
        match x {
            0..1000 => (x * 100, ""),
            1_000..1_000_000 => (x / 10, "k"),
            1_000_000..1_000_000_000 => (x / 10_000, "M"),
            1_000_000_000..1_000_000_000_000 => (x / 10_000_000, "G"),
            1_000_000_000_000..1_000_000_000_000_000 => (x / 10_000_000_000, "T"),
            1_000_000_000_000_000..1_000_000_000_000_000_000 => (x / 10_000_000_000_000, "P"),
            1_000_000_000_000_000_000.. => (x / 10_000_000_000_000_000, "E"),
        }
    }

    if nanos == u64::MAX {
        return "INF time taken".to_string();
    }

    let (t, t_u) = match nanos {
        0..1_000 => (nanos * 10, "ns"),
        1_000..1_000_000 => (nanos / 100, "μs"),
        1_000_000..1_000_000_000 => (nanos / 100_000, "ms"),
        1_000_000_000..1_000_000_000_000 => (nanos / 100_000_000, "s"),
        1_000_000_000_000.. => (nanos / 6_000_000_000, "min"),
    };

    let (fs, f_us) = value_unit(flop as u64 * 1_000_000 / nanos * 1000);
    let (brs, br_us) = value_unit(bytes_read * 1_000_000_000 / nanos);
    let (bws, bw_us) = value_unit(bytes_written * 1_000_000_000 / nanos);

    format!(
        "{}.{} {t_u} ~ {}.{:02} {f_us}FLOP/s, {}.{:02} {br_us}B/s r, {}.{:02} {bw_us}B/s w",
        t / 10,
        t % 10,
        fs / 100,
        fs % 100,
        brs / 100,
        brs % 100,
        bws / 100,
        bws % 100,
    )
}

impl Runtime {
    /// Ensures `x` lives alone in a **store-free** kernel and returns its `(kid, op_id)` in
    /// that kernel — the canonical "clean" placement for a tensor that another op (e.g.
    /// `narrow`, `permute`, `transpose`) is about to merge into.
    ///
    /// Steps:
    /// 1. If the producer kernel already has stores, or `x`'s op is preceded by a reduce,
    ///    or `force_store` was set: call [`add_store`] so `x` lands in a fresh load kernel.
    /// 2. Extract `x`'s op (and any other outputs) into a **brand-new** kernel with empty
    ///    `outputs` and no stores, retargeting the variables/loads correctly.
    ///
    /// The returned kernel is therefore a **fresh, store-free, outputs-empty** kernel that
    /// contains `x` and nothing else pending — this is the contract that makes
    /// `Runtime::narrow`'s "input into narrow must have empty outputs" assertion hold
    /// unconditionally, and that the kernelizer's `Node::Narrow` arm mirrors.
    fn duplicate_or_store(&mut self, x: TensorId, force_store: bool) -> Result<(KernelId, OpId), ZyxError> {
        fn eager_ids(rt: &Runtime, x: TensorId) -> (KernelId, OpId) {
            match rt.tensors[x] {
                TensorData::Eager { kernel_id, op_id, .. } | TensorData::Promoted { kernel_id, op_id, .. } => (kernel_id, op_id),
                ref t => panic!("duplicate_or_store: tensor {x} is not an eager/promoted tensor: {t:?}"),
            }
        }
        let (mut kid, mut op_id) = eager_ids(self, x);

        let contains_stores = self.kernels[kid].kernel.contains_stores();
        let preceded_by_reduce = self.kernels[kid].kernel.is_preceded_by_reduce(op_id);
        if force_store || contains_stores || preceded_by_reduce {
            self.add_store(x)?;
            (kid, op_id) = eager_ids(self, x);
            // We need to duplicate the new load kernel too, which we do below
        }

        debug_assert!(self.kernels[kid].stores.is_empty(), "duplicated kernel must not have stores");

        let old_loads = self.kernels[kid].loads.clone();
        // Keep-alive set for the split: every op owned by a live tensor
        // affiliated with this kernel (outputs ∪ loads) must stay in the old
        // kernel — pruning a load-affiliated tensor's op orphans it.
        let out_op_ids: Vec<OpId> = {
            let kd = &self.kernels[kid];
            let mut out_op_ids: Vec<OpId> = kd
                .outputs
                .iter()
                .map(|&tid| match &self.tensors[tid] {
                    TensorData::Eager { op_id, .. } | TensorData::Promoted { op_id, .. } => *op_id,
                    t => panic!("kernel output tid {tid} has unexpected tensor data {t:?}"),
                })
                .collect();
            for &tid in &kd.loads {
                if let TensorData::Eager { op_id, .. } | TensorData::Promoted { op_id, .. } = self.tensors[tid] {
                    if kd.kernel.ops.contains_id(op_id) {
                        out_op_ids.push(op_id);
                    }
                }
            }
            out_op_ids
        };
        let pre_live_ops: Set<OpId> = self.kernels[kid].kernel.ops.ids().collect();
        let (kernel, op_id, self_loads, new_loads) = self.kernels[kid].kernel.extract_subkernel(op_id, &out_op_ids, &old_loads);
        self.kernels[kid].loads = self_loads.clone();
        {
            let live_ops: Set<OpId> = self.kernels[kid].kernel.ops.ids().collect();
            for (tid, td) in self.tensors.iter() {
                if let TensorData::Eager { kernel_id: k, op_id: o, .. } | TensorData::Promoted { kernel_id: k, op_id: o, .. } = td {
                    if *k == kid && !live_ops.contains(o) {
                        eprintln!(
                            ">>> SPLIT-ORPHAN kid={kid:?} orphan-tid={tid} orphan-op={o:?} root-op={op_id:?} outputs={:?} pre_had_op={} pre_in_outputs={}",
                            self.kernels[kid].outputs,
                            pre_live_ops.contains(o),
                            self.kernels[kid].outputs.contains(&tid),
                        );
                    }
                }
            }
        }

        // Each kernel-load occurrence carries its own rc reference. The split
        // may duplicate a load into both kernels (an extra ref) or drop it
        // (release the ref).
        let mut seen: Set<TensorId> = Set::default();
        for &tid in old_loads.iter().chain(self_loads.iter()).chain(new_loads.iter()) {
            if !seen.insert(tid) {
                continue;
            }
            let old_c = old_loads.iter().filter(|&&t| t == tid).count();
            let self_c = self_loads.iter().filter(|&&t| t == tid).count();
            let new_c = self_c + new_loads.iter().filter(|&&t| t == tid).count();
            let delta = (new_c as i64) - (old_c as i64);
            #[cfg(feature = "debug_tensor_op")]
            eprintln!("DUP split kid={kid:?}: tid={tid} old={old_c} self={self_c} new={new_c} delta={delta}");
            for _ in 0..delta {
                self.retain(tid);
            }
            for _ in 0..(-delta) {
                self.release(tid);
            }
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
        {
            let holders: Vec<TensorId> = self
                .tensors
                .iter()
                .filter_map(|(tid, td)| match td {
                    TensorData::Eager { kernel_id: k, .. } | TensorData::Promoted { kernel_id: k, .. } if *k == merge_kid => Some(tid),
                    _ => None,
                })
                .collect();
            eprintln!("$$$ MERGE removing {merge_kid:?} holders={holders:?}");
        }
        let Kernel { ops: merge_ops, head: merge_head, .. } = kernel;

        let mut op_map: Map<OpId, OpId> = Map::with_hasher(BuildHasherDefault::new());
        let mut i = merge_head;
        while !i.is_null() {
            let mut op = merge_ops[i].op.clone();
            for param in op.parameters_mut() {
                if let Some(&new_param) = op_map.get(param) {
                    *param = new_param;
                }
            }
            let new_op_id = self.kernels[keep_kid].kernel.push_back(op);
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
                if !op_map.contains_key(op_id) {
                    let in_slab = merge_ops.contains_id(*op_id);
                    let in_list = {
                        let mut j = merge_head;
                        let mut found = false;
                        for _ in 0..100_000 {
                            if j.is_null() {
                                break;
                            }
                            if j == *op_id {
                                found = true;
                                break;
                            }
                            j = merge_ops[j].next;
                        }
                        found
                    };
                    eprintln!(
                        ">>> MERGE-MISS holder={_tid:?} op_id={op_id:?} in_slab={in_slab} in_list={in_list} merge_kid={merge_kid:?} keep_kid={keep_kid:?}"
                    );
                }
                *kernel_id = keep_kid;
                *op_id = op_map[op_id];
            }
        }

        // Stores folded into `keep_kid` now live there: their `depends_on`
        // (the producer kernel that holds the StoreView) must follow, even
        // though the tensor's `kernel_id` already points at its load kernel
        // (≠ merge_kid). Without this, the recursive materializer follows the
        // stale `depends_on` (the removed merge kernel) and the tensor is
        // never realized.
        let store_tids: Vec<TensorId> = merge_stores.clone();
        let keep_data = &mut self.kernels[keep_kid];
        keep_data.outputs.extend(merge_outputs);
        // Load entries move with their ops — an ownership transfer between
        // kernels. The edges (and their rc counts) persist unchanged; a retain
        // here would double-count every moved entry.
        keep_data.loads.extend(merge_loads.iter().copied());
        keep_data.stores.extend(merge_stores);
        for &tid in &store_tids {
            if let TensorData::Eager { depends_on, .. } = &mut self.tensors[tid] {
                *depends_on = keep_kid;
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

    /// Materializes `x`'s value into its kernel's storage and re-exposes it via a fresh load
    /// kernel for any remaining consumers (mirrors the kernelizer's `add_store` contract).
    ///
    /// After this call, `x`'s kernel has the new entry in `stores`, and `x`'s value is
    /// available to any later consumer through a reload (the freshly pushed load kernel
    /// holds `x`'s op as its only load). This is the canonical split point: a class that has
    /// been stored is no longer fused into a producer kernel — any subsequent consumer that
    /// would otherwise have to materialize again gets a clean reload.
    ///
    /// Called by `duplicate_or_store` (when the producer kernel already stores or its op is
    /// preceded by a reduce) and by `contiguous`'s cast-shim.
    pub fn add_store(&mut self, x: TensorId) -> Result<(), ZyxError> {
        eprintln!(">>> ADD_STORE x={x} data={:?}", self.tensors.get(x));
        let (kid, op_id, pending) = match self.tensors[x] {
            TensorData::Eager { kernel_id, op_id, depends_on, .. } => (kernel_id, op_id, depends_on),
            TensorData::Promoted { kernel_id, op_id, .. } => (kernel_id, op_id, KernelId::NULL),
            ref t => panic!("add_store: tensor {x} is not an eager/promoted tensor: {t:?}"),
        };

        // Remove x from the kernel's outputs (it is being stored).
        debug_assert!(self.kernels[kid].outputs.contains(&x), "add_store called for tid not in outputs");
        eprintln!(
            "$$$ ADD_STORE {x}: kid={kid:?} pre_outputs={:?} pre_loads={:?} pre_stores={:?}",
            self.kernels[kid].outputs, self.kernels[kid].loads, self.kernels[kid].stores
        );
        self.kernels[kid].outputs.remove(&x);

        // Only add StoreView if x isn't already realized or pending
        let dtype = self.dtype(x);
        let add_store = !self.buffer_map.contains_key(&x) && pending.is_null();
        let pending = if add_store {
            // Invariant: a kernel must never both load and store the same tensor
            debug_assert!(!self.kernels[kid].loads.contains(&x), "kernel {kid:?} both loads and stores tid {x}");

            let store_shape_id = self.kernels[kid].kernel.stack_shape_dims(op_id);
            let dst_id = self.kernels[kid].kernel.param(dtype, ParamKind::GlobalMut, store_shape_id);
            self.kernels[kid].kernel.store(dst_id, op_id, OpId::NULL, MemLayout::Scalar);
            self.kernels[kid].stores.push(x);
            kid
        } else {
            pending
        };
        let outputs_empty = self.kernels[kid].outputs.is_empty();

        // Create load kernel so the tensor remains usable (visited must point to a live kernel)
        let mut kernel = Kernel::new(DeviceId::AUTO);
        let dims: Vec<OpId> = self.resolve_shape(x).iter().map(|&d| kernel.const_idx(d)).collect();
        let load_shape = match dims.len() {
            0 => OpId::NULL,
            1 => dims[0],
            _ => kernel.stack(&dims),
        };
        let load_op_id = kernel.param(dtype, ParamKind::Global, load_shape);
        let load_kid = self.kernels.push(KernelData { outputs: Set::from_iter([x]), loads: vec![x], stores: Vec::new(), kernel });
        eprintln!(
            "$$$ ADD_STORE {x}: new load kernel {load_kid:?}, old kid={kid:?} post_outputs={:?} post_loads={:?}",
            self.kernels[kid].outputs, self.kernels[kid].loads
        );
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
        self.tensors[x] = TensorData::Eager { kernel_id: load_kid, op_id: load_op_id, depends_on: pending, shape_id, dtype, rc };
        self.retain(x);

        if outputs_empty {
            self.materialize_kernel(kid)?;
        }
        Ok(())
    }

    pub fn get_or_autotune(
        &mut self,
        mut kernel: Kernel,
        pool_id: PoolId,
        flop: Dim,
        read: u64,
        write: u64,
        buffers: &[LaunchArg],
    ) -> Result<(DeviceProgramId, OptSeq, u64), ZyxError> {
        let kernel_id = if let Some(&cached_kid) = self.kernel_map.get(&kernel) {
            if let Some(&program_id) = self.programs.get(&cached_kid) {
                let pid = ProgramId { device: kernel.device_id, program: program_id };
                let timing = self.timings.get(&pid).copied().unwrap_or(10_000_000_000);
                let dev_info = self.devices[kernel.device_id].info().clone();
                let dev_info_id = self.get_or_add_dev_info(&dev_info);
                let opt_seq = self.optimizations.get(&(cached_kid, dev_info_id)).cloned().unwrap_or_default();
                return Ok((program_id, opt_seq, timing));
            }

            let dev_info = self.devices[kernel.device_id].info().clone();
            let dev_info_id = self.get_or_add_dev_info(&dev_info);

            if let Some(opt_seq) = self.optimizations.get(&(cached_kid, dev_info_id)) {
                kernel.linearize();
                kernel.common_subexpression_elimination();
                kernel.dead_code_elimination();
                kernel.instruction_schedule();
                {
                    let global_indices = kernel.get_group_indices();
                    let max_global_dims = self.devices[kernel.device_id].info().max_global_work_dims.len();
                    if global_indices.len() > max_global_dims {
                        let n = global_indices.len() + 1 - max_global_dims;
                        let indices: Vec<OpId> = global_indices.values().copied().take(n).collect();
                        kernel.merge_indices(&indices);
                    }
                    kernel.renumber_indices();
                    kernel.verify();
                }
                let opt_seq = opt_seq.clone();
                opt_seq.apply(&mut kernel, &dev_info);
                let program_id = {
                    let device = &mut self.devices[kernel.device_id];
                    device.compile(&kernel, self.debug.asm())?
                };
                self.programs.insert(cached_kid, program_id);
                return Ok((program_id, opt_seq, 0));
            }
            cached_kid
        } else {
            let kernel_id =
                KernelId::from(self.kernel_map.values().copied().max().map_or(0, |id| usize::from(id).checked_add(1).unwrap()));
            let newly_inserted = self.kernel_map.insert(kernel.clone(), kernel_id).is_none();
            assert!(newly_inserted);
            kernel_id
        };

        let dev_info = self.devices[kernel.device_id].info().clone();
        let dev_info_id = self.get_or_add_dev_info(&dev_info);

        if self.debug.sched() {
            kernel.debug();
        }

        // The kernel is handed to autotune_ PRE-linearization: alloc_buffers
        // resolves buffer sizes from the param shape stacks, which linearize
        // nulls. Linearization and the post-linearize passes now live in
        // autotune_.

        let (program_id, opts, timing) = kernel.autotune_(
            &mut self.devices[kernel.device_id],
            &mut self.pools[pool_id],
            &self.autotune_config,
            flop,
            read,
            write,
            self.debug,
            buffers,
        )?;

        self.programs.insert(kernel_id, program_id);
        self.optimizations.insert((kernel_id, dev_info_id), opts.clone());
        self.timings.insert(ProgramId { device: kernel.device_id, program: program_id }, timing);

        Ok((program_id, opts, timing))
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
    /// The debug_assert in the recursive materialization loop enforces this.
    pub fn materialize_kernel(&mut self, kid: KernelId) -> Result<(), ZyxError> {
        // Temporary debug instrumentation (do not commit).
        let stale: Vec<TensorId> = self
            .tensors
            .iter()
            .filter_map(|(tid, td)| match td {
                TensorData::Eager { kernel_id: k, .. } | TensorData::Promoted { kernel_id: k, .. } if *k == kid => Some(tid),
                _ => None,
            })
            .collect();
        eprintln!("$$$ MATERL {kid:?}: outputs={:?} loads={:?} stores={:?} holders={:?}", self.kernels[kid].outputs, self.kernels[kid].loads, self.kernels[kid].stores, stale);
        // Resolve the dtypes of the loads and stores now, while this kernel (and any
        // tensor whose dtype resolves through it) is still alive. After remove_and_return
        // below, self.dtype on those tensors would panic on the removed kernel.
        for &tid in &self.kernels[kid].loads.iter().chain(&self.kernels[kid].stores).collect::<Vec<_>>() {
            eprintln!(">>> MATERL dtype-loop kid={kid:?} tid={tid} data={:?}", self.tensors[*tid]);
        }
        let dtypes: Map<TensorId, DType> =
            self.kernels[kid].loads.iter().chain(&self.kernels[kid].stores).map(|&tid| (tid, self.dtype(tid))).collect();
        // Null out kernel_ids of disowned loads pointing at this kernel: it dies
        // below, and `remove_and_return` is a swap_remove — a stale kernel_id
        // would alias a *different* kernel afterwards. Their edges are released
        // after the launch; the NULL kernel_id ends that recursion cleanly.
        for &tid in &self.kernels[kid].loads {
            if let TensorData::Eager { kernel_id: k, .. } | TensorData::Promoted { kernel_id: k, .. } = &mut self.tensors[tid] {
                if *k == kid {
                    eprintln!(">>> NULLING (materialize) tid={tid} kid={kid:?}");
                    *k = KernelId::NULL;
                }
            }
        }
        let KernelData { outputs, loads, stores, mut kernel } = unsafe { self.kernels.remove_and_return(kid) };

        debug_assert!(outputs.is_empty(), "all outputs must be stored before materialize");

        if stores.is_empty() {
            // Nothing to launch, but the kernel is still being removed: its
            // load edges must be released exactly as in the launch path, or
            // the counts orphan and the tensors leak.
            for &tid in &loads {
                eprintln!(">>> MATERL early-return release tid={tid} data={:?}", self.tensors.get(tid));
                self.release(tid);
            }
            return Ok(());
        }

        for &tid in &loads {
            assert!(
                self.buffer_map.contains_key(&tid)
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
        // `add_store` on whatever is still unrealized. Scalars may live in
        // variable_map instead of buffer_map.
        for &load in &loads {
            if self.buffer_map.contains_key(&load)
                || self.variable_map.contains_key(&load)
                || self.resolve_symbolic(load).is_some()
            {
                continue;
            }
            // An `Eager` tensor never carries a graph class, so its
            // depends_on is the pending producer.
            let pending = match self.tensors[load] {
                TensorData::Eager { depends_on, .. } => depends_on,
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
            if self.buffer_map.contains_key(&load)
                || self.variable_map.contains_key(&load)
                || self.resolve_symbolic(load).is_some()
            {
                continue;
            }
            if matches!(self.tensors[load], TensorData::Eager { .. } | TensorData::Promoted { .. }) {
                self.add_store(load)?;
            }
        }

        debug_assert!(
            loads
                .iter()
                .all(|&tid| self.buffer_map.contains_key(&tid)
                    || self.variable_map.contains_key(&tid)
                    || self.resolve_symbolic(tid).is_some()),
            "all loads must be realized after recursive materialization"
        );

        // Pick device and pool
        self.initialize_backends();

        // If stores already have buffers (e.g. assign writes in-place), a
        // kernel can only touch memory of one pool, so those buffers dictate
        // the pool — and hence the device. Stores spanning multiple pools is
        // an error. Without existing store buffers (or if no device shares
        // their pool), fall back to the freest device and move the buffers.
        let mut store_pools: BTreeSet<PoolId> = BTreeSet::new();
        for &tid in &stores {
            if let Some(buf_id) = self.buffer_map.get(&tid) {
                store_pools.insert(buf_id.pool);
            }
        }
        let (dev_id, pool_id) = if store_pools.len() == 1 {
            let pool_id = *store_pools.iter().next().unwrap();
            let dev_id = self.devices.ids().find(|&dev_id| self.devices[dev_id].memory_pool_id() == pool_id);
            match dev_id {
                Some(dev_id) => (dev_id, pool_id),
                None => {
                    let mut dev_ids: Vec<DeviceId> = self.devices.ids().collect();
                    dev_ids.sort_unstable_by_key(|&dev_id| self.devices[dev_id].free_compute());
                    dev_ids.reverse();
                    let dev_id = *dev_ids.first().ok_or_else(|| ZyxError::AllocationError("no available device".into()))?;
                    (dev_id, self.devices[dev_id].memory_pool_id())
                }
            }
        } else if store_pools.is_empty() {
            let mut dev_ids: Vec<DeviceId> = self.devices.ids().collect();
            dev_ids.sort_unstable_by_key(|&dev_id| self.devices[dev_id].free_compute());
            dev_ids.reverse();
            let dev_id = *dev_ids.first().ok_or_else(|| ZyxError::AllocationError("no available device".into()))?;
            (dev_id, self.devices[dev_id].memory_pool_id())
        } else {
            return Err(ZyxError::AllocationError(
                format!("stores span multiple pools {store_pools:?}; a kernel can only touch memory of a single pool").into(),
            ));
        };
        kernel.device_id = dev_id;

        // Ensure loads are in target pool. Variables and symbolic leaves are
        // not backed by any buffer — they bind at launch from `variable_map`.
        let mut event_wait_list = Vec::new();
        for &tid in &loads {
            let Some(&buf_id) = self.buffer_map.get(&tid) else { continue };
            if buf_id.pool != pool_id {
                let src = buf_id.buffer;
                let bytes =
                    (self.resolve_shape(tid).iter().product::<Dim>() as usize * dtypes[&tid].bit_size() as usize).div_ceil(8);
                let alloc_bytes = bytes + dtypes[&tid].bit_size() as usize / 8;

                // Gather the events that the source buffer depends on (prior
                // writers), so the copy waits for them.
                let mut wait_list = Vec::new();
                for buffers in self.events.keys() {
                    if buffers.contains(&buf_id) {
                        let buffers = buffers.clone();
                        let event = self.events.remove(&buffers).unwrap();
                        wait_list.push(event);
                        break;
                    }
                }

                let (dst, alloc_ev) = self.pools[pool_id].allocate(alloc_bytes as Dim)?;
                let dst_global = BufferId { pool: pool_id, buffer: dst };
                debug_assert_ne!(buf_id.pool, pool_id, "pool_to_pool across the same pool is disallowed");
                let src_pool_ptr: *mut MemoryPool = &mut self.pools[buf_id.pool];
                let copy_ev = self.pools[pool_id].pool_to_pool(unsafe { &mut *src_pool_ptr }, src, dst, {
                    wait_list.push(alloc_ev);
                    wait_list
                })?;
                self.pools[pool_id].sync_events(vec![copy_ev])?;

                // Remove and deallocate the old buffer only AFTER pool_to_pool
                // has finished reading it.
                self.buffer_map.remove(&tid);
                if !self.buffer_map.values().any(|b| b.buffer == src) {
                    self.pools[buf_id.pool].deallocate(src, vec![]);
                }
                self.buffer_map.insert(tid, dst_global);
            } else {
                for buffers in self.events.keys() {
                    if buffers.contains(&buf_id) {
                        let buffers = buffers.clone();
                        let event = self.events.remove(&buffers).unwrap();
                        event_wait_list.push(event);
                        break;
                    }
                }
            }
        }

        // Ensure stores are in target pool (assign writes in-place into an
        // existing buffer, which may live in a different pool).
        for &tid in &stores {
            let Some(buf_id) = self.buffer_map.get(&tid).copied() else {
                continue;
            };
            if buf_id.pool != pool_id {
                let src = buf_id.buffer;
                let bytes =
                    (self.resolve_shape(tid).iter().product::<Dim>() as usize * dtypes[&tid].bit_size() as usize).div_ceil(8);
                let alloc_bytes = bytes as Dim + Dim::from(dtypes[&tid].bit_size() / 8);
                let mut byte_slice = vec![0u8; bytes];

                let mut ev = Vec::new();
                for buffers in self.events.keys() {
                    if buffers.contains(&buf_id) {
                        let buffers = buffers.clone();
                        let event = self.events.remove(&buffers).unwrap();
                        ev.push(event);
                        break;
                    }
                }
                self.pools[buf_id.pool].pool_to_host(src, &mut byte_slice, ev)?;
                self.buffer_map.remove(&tid);
                if !self.buffer_map.values().any(|b| b.buffer == src) {
                    self.pools[buf_id.pool].deallocate(src, vec![]);
                }

                let (dst, event) = self.pools[pool_id].allocate(alloc_bytes)?;
                let dst_global = BufferId { pool: pool_id, buffer: dst };
                let event = self.pools[pool_id].host_to_pool(&byte_slice, dst, vec![event])?;
                self.pools[pool_id].sync_events(vec![event])?;
                self.buffer_map.insert(tid, dst_global);
            }
        }

        // Collect existing store buffers for already-realized store tensors,
        // allocate new buffers for the rest.
        let mut kernel_buffers = BTreeSet::new();
        for &tid in &loads {
            // Scalars bound via `variable_map` are launch-time values, not
            // pool storage — they have no buffer and no event dependency.
            if self.variable_map.contains_key(&tid) {
                continue;
            }
            kernel_buffers.insert(self.buffer_map[&tid]);
        }
        for &tid in &stores {
            if let Some(&buf_id) = self.buffer_map.get(&tid) {
                kernel_buffers.insert(buf_id);
                if let TensorData::Eager { depends_on, .. } = &mut self.tensors[tid] {
                    *depends_on = KernelId::NULL;
                }
            } else {
                let bytes =
                    (self.resolve_shape(tid).iter().product::<Dim>() as usize * dtypes[&tid].bit_size() as usize).div_ceil(8);
                let alloc_bytes = bytes as Dim + Dim::from(dtypes[&tid].bit_size() / 8);
                let (buf, event) = self.pools[pool_id].allocate(alloc_bytes)?;
                let global_id = BufferId { pool: pool_id, buffer: buf };
                self.buffer_map.insert(tid, global_id);
                if let TensorData::Eager { depends_on, .. } = &mut self.tensors[tid] {
                    *depends_on = KernelId::NULL;
                }
                kernel_buffers.insert(global_id);
                event_wait_list.push(event);
            }
        }
        // Materialization must realize EVERY store: stores are finished
        // tensors other kernels already use as loads.
        for &tid in &stores {
            debug_assert!(self.buffer_map.contains_key(&tid), "materialize: store tid {tid} has no buffer after realization");
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
            if let Some(&value) = self.variable_map.get(&tid) {
                // Variables live only in variable_map — they never have a
                // buffer or pool storage; the value is bound at launch.
                buffers.push(LaunchArg::Variable(value));
            } else {
                buffers.push(LaunchArg::Buffer(self.buffer_map[&tid].buffer));
            }
        }
        for &tid in &stores {
            buffers.push(LaunchArg::Buffer(self.buffer_map[&tid].buffer));
        }

        // Compile and launch (caches in kernel_map / programs)
        let (flop, read, write) = kernel.flop_mem_rw();
        let (dev_prog, _opts, _timing) = self.get_or_autotune(kernel, pool_id, flop, read, write, &buffers)?;

        let event = self.devices[dev_id].launch(dev_prog, &mut self.pools[pool_id], &buffers, event_wait_list)?;
        self.events.insert(kernel_buffers, event);

        // The kernel has consumed its loads. Release the load references so
        // dead load tensors and their buffers are reclaimed. Buffers still in
        // use keep rc > 0 via other kernels' load references or handles.
        for &tid in &loads {
            eprintln!(">>> MATERL post-launch release tid={tid} data={:?}", self.tensors.get(tid));
            self.release(tid);
        }

        Ok(())
    }

    fn get_or_add_dev_info(&mut self, device_info: &DeviceInfo) -> DeviceInfoId {
        if let Some(&dev_info_id) = self.device_infos.get(device_info) {
            dev_info_id
        } else {
            let dev_info_id =
                DeviceInfoId(self.device_infos.values().copied().max().map_or(0, |id| id.0.checked_add(1).unwrap()));
            let newly_inserted = self.device_infos.insert(device_info.clone(), dev_info_id).is_none();
            assert!(newly_inserted);
            dev_info_id
        }
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
                    TensorData::Eager { shape_id, .. } | TensorData::Graph { shape_id, .. }
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
                    TensorData::Eager { rc, .. } | TensorData::Graph { rc, .. } | TensorData::Promoted { rc, .. }
                    | TensorData::Constant { rc, .. } | TensorData::Variable { rc, .. } | TensorData::Cast { rc, .. }
                    | TensorData::Unary { rc, .. } | TensorData::Binary { rc, .. } | TensorData::Stack { rc, .. } => *rc as usize,
                };
                let exp = expected.get(&tid).copied().unwrap_or(0);
                if rc != exp {
                    eprintln!("LEDGER tid={tid} rc={rc} expected={exp} orphan={}", rc as isize - exp as isize);
                }
                eprintln!("ITER leak-check tid={tid} rc={rc} {td:?}");
                if let TensorData::Eager { kernel_id, .. } | TensorData::Promoted { kernel_id, .. } = td {
                    if !kernel_id.is_null() {
                        if let Some(kd) = rt.kernels.get(*kernel_id) {
                            eprintln!("   kernel {kernel_id:?}: outputs={:?} loads={:?} stores={:?}", kd.outputs, kd.loads, kd.stores);
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
            } // every handle drops here
            assert_drained();
        }
        Ok(())
    }

    #[test]
    fn eager_and_tape_drain_inventory() -> Result<(), ZyxError> {
        let _guard = test_lock().lock().unwrap();
        for iteration in 0..8 {
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
