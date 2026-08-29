//! E-graph for tensor operation equivalence and optimization.
//!
//! The graph supports rewrites that produce equivalent forms of a computation:
//! - **CSE** (common subexpression elimination) via hashconsing
//! - **Algebraic rewrites** like transpose fusion: `transpose(A) @ transpose(B)` ↔ `(B @ A).transpose()`
//! - **Layout rewrites**: a matmul can be realized as transposed or un-transposed,
//!   with the transpose either fused into the kernel or materialized as a separate
//!   pre-processing step
//! - **Shape rewrites**: reshape and padding can be fused into adjacent ops or
//!   split out as separate nodes
//!
//! Each equivalence class (`EClass`) holds all equivalent node forms. A cost
//! model selects the cheapest extraction for kernel compilation.

use std::collections::BTreeSet;

use crate::{
    DType, Map, Set, ZyxError,
    backend::{BufferId, Device, PoolId, ProgramId},
    dtype::Constant,
    kernel::{BOp, DeviceId, IDX_T, Kernel, MoveOp, Op, OpId, ParamKind, UOp},
    runtime::{KernelData, KernelId, Runtime, TensorData},
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
};

mod autograd;
mod kernelizer;
pub(crate) mod plan;
pub use plan::ExecPlan;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

impl From<usize> for NodeId {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}
impl From<NodeId> for usize {
    fn from(v: NodeId) -> usize {
        v.0 as usize
    }
}

impl SlabId for NodeId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GraphId(pub u16);

impl From<usize> for GraphId {
    fn from(v: usize) -> Self {
        Self(v as u16)
    }
}
impl From<GraphId> for usize {
    fn from(v: GraphId) -> usize {
        v.0 as usize
    }
}

impl SlabId for GraphId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u16::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClassId(pub u32);

impl From<usize> for ClassId {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}
impl From<ClassId> for usize {
    fn from(v: ClassId) -> usize {
        v.0 as usize
    }
}

impl SlabId for ClassId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Node {
    /// A compile-time constant.
    ///
    /// `cons_id` is **mandatory** and must never be removed. It is assigned by
    /// `push_node` (constructors pass 0) so that every const is *structurally
    /// unique*: hashconsing compares it, two consts never compare equal, and
    /// therefore two consts never merge into one e-class.
    ///
    /// # Why this id exists (bug history — read before deleting!)
    ///
    /// This bug has already happened **twice**:
    ///
    /// 1. Leaves used to be hashconsed without such an id, so two buffers with
    ///    identical dtype+shape collapsed into one class; `leaf_id` was added
    ///    to fix it, but the documentation did not explain the underlying
    ///    invariant, and the same mistake was then repeated for `Const`.
    /// 2. Consts without an id were hashconsed, so e.g. a narrow's `start=2`
    ///    and another op's `len=2` merged into ONE class. The class gets pinned
    ///    to whichever kernel materialized it first; the second consumer then
    ///    inherits that placement and the kernelizer materializes the constant
    ///    into a foreign kernel (or skips duplication), producing invalid or
    ///    silently wrong kernels. Worse, `Graph::cache_key` hashes the
    ///    hashcons map — without const nodes in it, two graphs differing only
    ///    in const values (f32[3] vs f32[10] relu) collide on the same plan
    ///    cache entry and execute with the first graph's allocation sizes.
    ///
    /// The invariant: **a class must have exactly one creation site.** Value
    /// nodes (consts) and buffer nodes (leaves) have value semantics — equal
    /// values must still stay distinct classes, because classes carry
    /// identity: placement (which kernel materialized them), refcounts, and
    /// plan/kernel cache keys all assume it.
    Const {
        cons_id: u32,
        value: Constant,
    },
    /// A realized input buffer. See [`Node::Const`] for why `cons_id` exists
    /// and must not be removed — the exact same reasoning applies here.
    Leaf {
        cons_id: u32,
        dtype: DType,
        /// Shape of the leaf as a class: a Stack of dim classes (Const dims or
        /// symbolic dim leaves). `ClassId::NULL` for scalars (`[]` shape).
        ///
        /// Two leaf kinds are distinguished purely by `(dtype, shape)`:
        /// buffer leaves carry a data dtype and a (possibly NULL for scalars)
        /// shape stack; **dim-variable leaves** are `dtype == IDX_T` with
        /// `shape == ClassId::NULL` — they represent a dynamic dimension
        /// value, created by `replay_symbolic_into_graph`, never merged with
        /// any other class, and bound at execution time via `variable_map`.
        shape: ClassId,
    },
    Expand {
        x: ClassId,
        shape: ClassId,
    },
    Permute {
        x: ClassId,
        axes: Box<[UAxis]>,
    },
    Reshape {
        x: ClassId,
        shape: ClassId,
    },
    Pad {
        x: ClassId,
        axis: UAxis,
        /// Left padding amount, as a dim class.
        lp: ClassId,
        /// Total padded length of `axis` (`orig_len + lp + rp`), as a dim
        /// class (tinygrad convention). Right padding is `len - lp - orig_len`.
        len: ClassId,
    },
    Flip {
        x: ClassId,
        axes: Box<[UAxis]>,
    },
    Narrow {
        x: ClassId,
        axis: UAxis,
        start: ClassId,
        len: ClassId,
    },
    Stack {
        ops: Box<[ClassId]>,
    },
    Reduce {
        x: ClassId,
        rop: BOp,
        axes: Box<[UAxis]>,
    },
    Cast {
        x: ClassId,
        dtype: DType,
    },
    Unary {
        x: ClassId,
        uop: UOp,
    },
    Binary {
        x: ClassId,
        y: ClassId,
        bop: BOp,
    },
    Assign {
        dst: ClassId,
        src: ClassId,
    },
    After {
        x: ClassId,
        dep: ClassId,
    },
    ToDevice {
        x: ClassId,
        device: DeviceId,
        time: u64,
    },
    Contiguous {
        x: ClassId,
    },
    Kernel {
        inputs: Box<[ClassId]>,
        outputs: Box<[ClassId]>,
        program_id: ProgramId,
        time: u64,
    },
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Const { cons_id: a, value: av }, Self::Const { cons_id: b, value: bv }) => a == b && av == bv,
            (Self::Leaf { cons_id: a, .. }, Self::Leaf { cons_id: b, .. }) => a == b,
            (Self::Expand { x: a, shape: as_ }, Self::Expand { x: b, shape: bs }) => a == b && as_ == bs,
            (Self::Permute { x: a, axes: aa }, Self::Permute { x: b, axes: ba }) => a == b && aa == ba,
            (Self::Reshape { x: a, shape: as_, .. }, Self::Reshape { x: b, shape: bs, .. }) => a == b && as_ == bs,
            (Self::Pad { x: a, axis: aa, lp: al, len: aln }, Self::Pad { x: b, axis: ba, lp: bl, len: bln }) => {
                a == b && aa == ba && al == bl && aln == bln
            }
            (Self::Flip { x: a, axes: aa }, Self::Flip { x: b, axes: ba }) => a == b && aa == ba,
            (Self::Reduce { x: a, rop: ar, axes: aa }, Self::Reduce { x: b, rop: br, axes: ba }) => {
                a == b && ar == br && aa == ba
            }
            (Self::Cast { x: a, dtype: ad }, Self::Cast { x: b, dtype: bd }) => a == b && ad == bd,
            (Self::Unary { x: a, uop: au }, Self::Unary { x: b, uop: bu }) => a == b && au == bu,
            (Self::Binary { x: a, y: ay, bop: ab }, Self::Binary { x: b, y: by, bop: bb }) => a == b && ay == by && ab == bb,
            (Self::Assign { dst: a, src: as_ }, Self::Assign { dst: b, src: bs }) => a == b && as_ == bs,
            (Self::ToDevice { x: a, device: ad, .. }, Self::ToDevice { x: b, device: bd, .. }) => a == b && ad == bd,
            (Self::Contiguous { x: a }, Self::Contiguous { x: b }) => a == b,
            (
                Self::Kernel { inputs: ai, outputs: ao, program_id: ap, .. },
                Self::Kernel { inputs: bi, outputs: bo, program_id: bp, .. },
            ) => ai == bi && ao == bo && ap == bp,
            _ => false,
        }
    }
}

impl Eq for Node {}

impl std::hash::Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Self::Const { cons_id, value } => {
                0u8.hash(state);
                cons_id.hash(state);
                value.hash(state);
            }
            Self::Leaf { cons_id, dtype, shape } => {
                1u8.hash(state);
                cons_id.hash(state);
                dtype.hash(state);
                shape.hash(state);
            }
            Self::Expand { x, shape } => {
                2u8.hash(state);
                x.hash(state);
                shape.hash(state);
            }
            Self::Permute { x, axes } => {
                3u8.hash(state);
                x.hash(state);
                axes.hash(state);
            }
            Self::Reshape { x, shape, .. } => {
                4u8.hash(state);
                x.hash(state);
                shape.hash(state);
            }
            Self::Pad { x, axis, lp, len } => {
                5u8.hash(state);
                x.hash(state);
                axis.hash(state);
                lp.hash(state);
                len.hash(state);
            }
            Self::Stack { ops } => {
                13u8.hash(state);
                ops.hash(state);
            }
            Self::Flip { x, axes } => {
                12u8.hash(state);
                x.hash(state);
                axes.hash(state);
            }
            Self::Narrow { x, axis, start, len } => {
                16u8.hash(state);
                x.hash(state);
                axis.hash(state);
                start.hash(state);
                len.hash(state);
            }
            Self::Reduce { x, rop: bop, axes } => {
                6u8.hash(state);
                x.hash(state);
                bop.hash(state);
                axes.hash(state);
            }
            Self::Cast { x, dtype } => {
                7u8.hash(state);
                x.hash(state);
                dtype.hash(state);
            }
            Self::Unary { x, uop } => {
                8u8.hash(state);
                x.hash(state);
                uop.hash(state);
            }
            Self::Binary { x, y, bop } => {
                9u8.hash(state);
                x.hash(state);
                y.hash(state);
                bop.hash(state);
            }
            Self::ToDevice { x, device, .. } => {
                10u8.hash(state);
                x.hash(state);
                device.hash(state);
            }
            Self::Contiguous { x } => {
                14u8.hash(state);
                x.hash(state);
            }
            Self::Assign { dst, src } => {
                13u8.hash(state);
                dst.hash(state);
                src.hash(state);
            }
            Self::After { x, dep } => {
                15u8.hash(state);
                x.hash(state);
                dep.hash(state);
            }
            Self::Kernel { inputs, outputs, program_id, .. } => {
                11u8.hash(state);
                inputs.hash(state);
                outputs.hash(state);
                program_id.hash(state);
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct NodeData {
    pub(crate) node: Node,
    pub(crate) class_of: ClassId,
}

#[derive(Debug)]
pub struct EClass {
    pub nodes: Vec<NodeId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JitKernelId(pub u32);

impl From<usize> for JitKernelId {
    fn from(v: usize) -> Self {
        Self(v as u32)
    }
}
impl From<JitKernelId> for usize {
    fn from(v: JitKernelId) -> usize {
        v.0 as usize
    }
}
impl SlabId for JitKernelId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);
    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone)]
/// A jit kernel under construction by the kernelizer.
///
/// # Field contracts
///
/// - `kernel`: the kernel IR. All `Param` defines — global buffer params and
///   scalar `Param { kind: Variable }` dim params alike — sit in flat head
///   order, and launch-time args bind **positionally** over exactly that
///   sequence (see the gws section of AGENTS.md). No define may be inserted,
///   removed, or reordered after ops referencing it exist: that would silently
///   re-bind every arg.
/// - `loads`: every class this kernel reads, aligned to the kernel's
///   **non-store** `Param` defines (`Global` buffers and scalar
///   `Param { kind: Variable }` dim params) in head order — each entry
///   corresponds to exactly one such define: a global buffer class
///   (`Node::Leaf` with data dtype) for a `Global` param, or a dim-variable
///   class (`Node::Leaf { dtype: IDX_T, shape: NULL }`) for a `Variable`
///   param. **A `GlobalMut` store target must NOT appear here**: an in-place
///   assign turns dst from a load into a pure store; its buffer slot is
///   carried by `stores` instead. Invariant
///   `loads.len() == number of Global+Variable defines` is asserted at
///   extraction. Never shuffle; consumers (exec plan, tape) map entries to
///   pooled values via `buffer_map` / `variable_map` keyed by the originating
///   tensor id resolved through `leaf_map`.
/// - `outputs`: classes whose value this kernel produces; one slot per rc so
///   multi-consumer reloads work.
/// - `stores`: classes written to storage.
///
/// Known pending fix: `assign` handling assumed `loads[0]` was the destination
/// buffer — with variables now also present in `loads`, it must trace the
/// actual buffer class instead of assuming position 0.
pub struct JitKernelData {
    pub(crate) kernel: Kernel,
    pub(crate) outputs: Vec<ClassId>,
    pub(crate) loads: Vec<ClassId>,
    pub(crate) stores: Vec<ClassId>,
}

#[derive(Debug)]
pub struct Graph {
    pub(crate) hashcons: Map<Node, NodeId>,
    pub(crate) nodes: Slab<NodeId, NodeData>,
    pub(crate) classes: Slab<ClassId, EClass>,
    pub(crate) jit_kernels: Slab<JitKernelId, JitKernelData>,
    pub(crate) leaf_classes: Vec<ClassId>,
    pub(crate) leaf_map: Map<ClassId, TensorId>,
    // Number of alive graph tensors (TensorState::Graph) referencing this graph.
    // Incremented at every graph-tensor birth, decremented when a tensor dies
    // (release), is eagerified, or is dropped.
    pub(crate) ref_count: u64,
    // Tape scope has ended (Tape::drop ran); no new ops may use this graph.
    // The graph is removed from the slab only when dead && ref_count == 0, which
    // guarantees no stale tensor ever observes a reused GraphId.
    pub(crate) dead: bool,
    /// Allocator for [`Node::Const`]/[`Node::Leaf`] `cons_id`s.
    pub(crate) max_cons_id: u32,
}

impl Node {
    /// Classes this node references: operands, and metadata like shape
    /// descriptors (a leaf's shape is a parameter of the leaf).
    fn class_params(&self) -> impl Iterator<Item = ClassId> {
        let v = match self {
            Self::Const { .. } => vec![],
            Self::Leaf { shape, .. } => vec![*shape],
            Self::Expand { x, shape } => vec![*x, *shape],
            Self::Permute { x, .. } => vec![*x],
            Self::Reshape { x, shape, .. } => vec![*x, *shape],
            Self::Pad { x, lp, len, .. } => vec![*x, *lp, *len],
            Self::Narrow { x, axis: _, start, len } => vec![*x, *start, *len],
            Self::Flip { x, .. } => vec![*x],
            Self::Stack { ops } => ops.to_vec(),
            Self::Reduce { x, .. } => vec![*x],
            Self::Cast { x, .. } => vec![*x],
            Self::Unary { x, .. } => vec![*x],
            Self::Binary { x, y, .. } => vec![*x, *y],
            Self::Assign { dst, src } => vec![*dst, *src],
            Self::After { x, dep } => vec![*x, *dep],
            Self::ToDevice { x, .. } => vec![*x],
            Self::Contiguous { x, .. } => vec![*x],
            Self::Kernel { inputs, .. } => inputs.to_vec(),
        };
        v.into_iter()
    }
}

impl Graph {
    /// Cleanup - when graph is no longer needed, but cannot be dropped yet, so it's marked dead
    pub fn mark_dead(&mut self) {
        self.dead = true;
        self.hashcons = Map::default();
        self.nodes = Slab::new();
        self.classes = Slab::new();
        self.jit_kernels = Slab::new();
        self.leaf_map = Map::default();
    }

    pub fn new() -> Self {
        Self {
            hashcons: Map::default(),
            nodes: Slab::new(),
            classes: Slab::new(),
            jit_kernels: Slab::new(),
            leaf_map: Map::default(),
            leaf_classes: Vec::new(),
            ref_count: 0,
            dead: false,
            max_cons_id: 0,
        }
    }

    pub fn is_leaf(&self, class_id: ClassId) -> bool {
        self.classes[class_id].nodes.iter().any(|&nid| matches!(&self.nodes[nid].node, Node::Leaf { .. }))
    }

    /// Walks back through single-input movement nodes until reaching dst's base
    /// leaf class (a key of `leaf_map`). Used to find which leaf buffer an
    /// [`ExecNode`] class's store aliases.
    pub(crate) fn base_leaf(&self, mut c: ClassId) -> ClassId {
        loop {
            if self.leaf_map.contains_key(&c) {
                return c;
            }
            let mut next = None;
            for nid in &self.classes[c].nodes {
                match &self.nodes[*nid].node {
                    Node::Expand { x, .. }
                    | Node::Permute { x, .. }
                    | Node::Reshape { x, .. }
                    | Node::Pad { x, .. }
                    | Node::Flip { x, .. }
                    | Node::Narrow { x, .. }
                    | Node::ToDevice { x, .. }
                    | Node::After { x, .. } => next = Some(*x),
                    _ => {}
                }
            }
            c = next.unwrap_or_else(|| panic!("assign dst class {c:?} must be a realized leaf or a movement chain over one"));
        }
    }

    /// Whether `class_id` is the output of an in-place `assign` — a class whose
    /// value lives in (aliases) dst's realized leaf buffer.
    pub fn is_after(&self, class_id: ClassId) -> bool {
        self.classes[class_id].nodes.iter().any(|&nid| matches!(&self.nodes[nid].node, Node::After { .. }))
    }

    pub fn push_to_device(&mut self, x: ClassId, device: DeviceId, time: u64) -> ClassId {
        let node = Node::ToDevice { x, device, time };
        if let Some(&nid) = self.hashcons.get(&node) {
            return self.nodes[nid].class_of;
        }
        let nid = self.nodes.push(NodeData { node: node.clone(), class_of: ClassId::NULL });
        let cid = self.classes.push(EClass { nodes: vec![nid] });
        self.nodes[nid].class_of = cid;
        self.hashcons.insert(node, nid);
        cid
    }

    /// Topologically sorts the classes reachable from `outputs` (consumers
    /// first, the returned vector is reversed into dependency order).
    ///
    /// With `WITHOUT_KERNELS`, [`Node::Kernel`] nodes are ignored when
    /// collecting dependencies and the walk stops at the classes in `inputs`
    /// (a boundary input contributes only its non-boundary kernel inputs).
    /// Used when iterating the structural graph — e.g. fusing remaining ops
    /// into kernels — where kernel nodes would add spurious input
    /// dependencies between classes and boundary classes must not be walked
    /// through into other regions. When `allowed` is `Some`, the walk never
    /// leaves that set.
    ///
    /// # Why boundary shape classes are absent from the order
    ///
    /// Because `deps` prunes a boundary class's `class_params`, a boundary
    /// leaf's shape stack never enters the returned order — by design, not by
    /// accident: shapes are purely symbolic metadata, never values flowing
    /// between kernels ("a shape dimension is a result of a kernel" was
    /// abandoned). Load kernels re-materialize their shapes themselves via
    /// `replay_symbolic_into_kernel`, exactly as the eager path does with
    /// `Runtime::replay_symbolic_into_kernel`. Consequently a missing shape
    /// class here must NOT be treated as a lost dependency; conversely, if a
    /// load kernel ever needs to consume a *computed* dim class, that is an
    /// invariant violation and panics inside replay rather than being fed
    /// through this sort.
    pub fn topo_sort_classes<const WITHOUT_KERNELS: bool>(
        &self,
        inputs: &Set<ClassId>,
        outputs: &BTreeSet<ClassId>,
        allowed: Option<&Set<ClassId>>,
    ) -> Vec<ClassId> {
        // Dead classes (unconsumed, not an output) are harmless: traversal
        // never reaches them, so they neither appear in `rcs` nor stall
        // anything. What must NEVER happen is a *reachable* class failing
        // to emit — that would mean its consumers' token accounting is
        // broken and everything depending on it silently drops out of the
        // order. Checked only in the global sort: region-restricted walks
        // legitimately cannot see consumers outside their boundary.
        let mut rcs: Map<ClassId, u32> = Map::default();
        let mut stack: Vec<ClassId> = outputs.iter().copied().collect();
        while let Some(cid) = stack.pop() {
            rcs.entry(cid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                let deps = self.deps::<WITHOUT_KERNELS>(inputs, cid);
                stack.extend(deps.into_iter().filter(|d| allowed.is_none_or(|a| a.contains(d))));
                1
            });
        }

        let mut order = Vec::new();
        let mut internal_rcs: Map<ClassId, u32> = Map::default();
        let mut stack: Vec<ClassId> = outputs.iter().copied().collect();
        while let Some(cid) = stack.pop() {
            if let Some(&rc) = rcs.get(&cid) {
                let visited = internal_rcs.entry(cid).and_modify(|c| *c += 1).or_insert(1);
                if rc == *visited {
                    order.push(cid);
                    let deps = self.deps::<WITHOUT_KERNELS>(inputs, cid);
                    stack.extend(deps.into_iter().filter(|d| allowed.is_none_or(|a| a.contains(d))));
                }
            }
        }
        if cfg!(debug_assertions) && !WITHOUT_KERNELS && allowed.is_none() {
            for (cid, &rc) in rcs.iter() {
                let visited = internal_rcs.get(cid).copied().unwrap_or(0);
                if visited == rc {
                    continue;
                }
                let mut report = String::new();
                let mut frontier = vec![*cid];
                let mut seen: Set<ClassId> = Set::default();
                while let Some(c) = frontier.pop() {
                    if !seen.insert(c) {
                        continue;
                    }
                    let v = internal_rcs.get(&c).copied().unwrap_or(0);
                    let r = rcs.get(&c).copied().unwrap_or(0);
                    let types: Vec<String> = self.classes[c]
                        .nodes
                        .iter()
                        .map(|n| format!("{:?}", self.nodes[*n].node))
                        .map(|s| s.split(" NodeId").next().unwrap_or(&s).to_string())
                        .collect();
                    report.push_str(&format!("\n  {c:?} rc={r} visited={v} types={types:?}"));
                    for p in self.classes.ids().filter(|p| {
                        self.classes[*p].nodes.iter().any(|n| self.nodes[*n].node.class_params().any(|q| q == c))
                            && rcs.contains_key(p)
                    }) {
                        let pv = internal_rcs.get(&p).copied().unwrap_or(0);
                        let pr = rcs.get(&p).copied().unwrap_or(0);
                        report.push_str(&format!(" <- {p:?}(rc={pr},visited={pv})"));
                        if pv != pr && !seen.contains(&p) {
                            frontier.push(p);
                        }
                    }
                }
                panic!(
                    "topo sort: reachable class {cid:?} did not emit (visited {visited} of rc {rc}) — token accounting broken. Chain:{report}"
                );
            }
        }
        order.reverse();
        order
    }

    /// Verifies that the class dependency graph under the extraction view
    /// ([`Self::extract_deps`]) is acyclic. A cycle means some kernel stores a
    /// value whose structural descendants are consumed by earlier kernels —
    /// no valid execution order exists.
    ///
    /// # Panics
    ///
    /// Panics with the offending dependency chain if a cycle is found, or if
    /// the walk exceeds 10 000 steps.
    pub(crate) fn verify(&self) {
        // Iterative colored DFS: 1 = on stack (gray), 2 = done (black).
        let mut color: Map<ClassId, u8> = Map::default();
        let mut parent: Map<ClassId, ClassId> = Map::default();
        for root in self.classes.ids() {
            let mut steps = 0;
            let mut stack = vec![(root, false)];
            while let Some((cid, processed)) = stack.pop() {
                steps += 1;
                if steps > 10_000 {
                    panic!("graph::verify did not finish in 10000 steps");
                }
                if processed {
                    color.insert(cid, 2);
                    continue;
                }
                match color.get(&cid).copied() {
                    Some(2) => continue,
                    // 1 = gray: the class is on the current DFS path — cycle.
                    Some(1) | Some(_) => {
                        let mut chain = vec![cid];
                        let mut cur = cid;
                        while let Some(&p) = parent.get(&cur) {
                            chain.push(p);
                            cur = p;
                            if cur == cid || chain.len() > 100 {
                                break;
                            }
                        }
                        panic!("graph::verify: dependency cycle through classes {chain:?}");
                    }
                    None => {}
                }
                color.insert(cid, 1);
                stack.push((cid, true));
                for d in self.extract_deps(cid) {
                    if !d.is_null() && color.get(&d) != Some(&2) {
                        parent.insert(d, cid);
                        stack.push((d, false));
                    }
                }
            }
        }
    }

    /// Dependencies of class `cid` for [`Self::topo_sort_classes`].
    ///
    /// With `WITHOUT_KERNELS`, [`Node::Kernel`] nodes are ignored and a
    /// boundary class (in `inputs`) contributes only its non-boundary kernel
    /// inputs; otherwise every node's [`Node::class_params`] is used.
    fn deps<const WITHOUT_KERNELS: bool>(&self, inputs: &Set<ClassId>, cid: ClassId) -> Vec<ClassId> {
        let mut deps = Vec::new();
        for nid in &self.classes[cid].nodes {
            match &self.nodes[*nid].node {
                Node::Kernel { inputs: kin, .. } => {
                    if WITHOUT_KERNELS && !inputs.contains(&cid) {
                        continue;
                    }
                    for p in kin.iter() {
                        if !deps.contains(p) && !(WITHOUT_KERNELS && inputs.contains(p)) {
                            deps.push(*p);
                        }
                    }
                }
                node => {
                    if WITHOUT_KERNELS && inputs.contains(&cid) {
                        continue;
                    }
                    for p in node.class_params() {
                        if !deps.contains(&p) {
                            deps.push(p);
                        }
                    }
                }
            }
        }
        deps
    }

    /// Dependencies of class `cid` under the **extraction** view: once a
    /// class is produced by a jit/AOT kernel (or [`Node::ToDevice`]), its
    /// scheduling dependencies are exactly those producers' inputs. The
    /// e-graph keeps competing derivations on the class, and following them
    /// alongside kernel inputs creates false cycles (a kernel may recompute
    /// a value whose structural path runs through another kernel's stores).
    ///
    /// [`Node::After`] and [`Node::Assign`] edges are kept regardless: they
    /// encode store *ordering* between in-place writes, not an alternative
    /// derivation.
    ///
    /// Used by [`Self::topo_sort_for_extract`] and [`Self::verify`].
    fn extract_deps(&self, cid: ClassId) -> Vec<ClassId> {
        let mut kdeps: Vec<ClassId> = Vec::new();
        for nid in &self.classes[cid].nodes {
            match &self.nodes[*nid].node {
                Node::Kernel { inputs, .. } => {
                    for p in inputs.iter() {
                        if !kdeps.contains(p) {
                            kdeps.push(*p);
                        }
                    }
                }
                Node::ToDevice { x, .. } => {
                    if !kdeps.contains(x) {
                        kdeps.push(*x);
                    }
                }
                _ => {}
            }
        }
        if kdeps.is_empty() {
            return self.deps::<false>(&Set::default(), cid);
        }
        for nid in &self.classes[cid].nodes {
            if let Node::After { x, dep } = &self.nodes[*nid].node {
                for p in [x, dep] {
                    if !kdeps.contains(p) {
                        kdeps.push(*p);
                    }
                }
            }
            if let Node::Assign { dst, src } = &self.nodes[*nid].node {
                for p in [dst, src] {
                    if !kdeps.contains(p) {
                        kdeps.push(*p);
                    }
                }
            }
        }
        kdeps
    }

    /// Topological order of classes for [`Self::extract`]: like
    /// [`Self::topo_sort_classes`] but using the extraction view
    /// ([`Self::extract_deps`]) for dependencies.
    fn topo_sort_for_extract(&self, outputs: &BTreeSet<ClassId>) -> Vec<ClassId> {
        let mut rcs: Map<ClassId, u32> = Map::default();
        let mut stack: Vec<ClassId> = outputs.iter().copied().collect();
        while let Some(cid) = stack.pop() {
            rcs.entry(cid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                stack.extend(self.extract_deps(cid));
                1
            });
        }

        let mut order = Vec::new();
        let mut internal_rcs: Map<ClassId, u32> = Map::default();
        let mut stack: Vec<ClassId> = outputs.iter().copied().collect();
        while let Some(cid) = stack.pop() {
            if let Some(&rc) = rcs.get(&cid) {
                let visited = internal_rcs.entry(cid).and_modify(|c| *c += 1).or_insert(1);
                if rc == *visited {
                    order.push(cid);
                    stack.extend(self.extract_deps(cid));
                }
            }
        }
        if cfg!(debug_assertions) {
            for (cid, &rc) in rcs.iter() {
                let visited = internal_rcs.get(cid).copied().unwrap_or(0);
                assert_eq!(visited, rc, "extraction topo: reachable class {cid:?} did not emit (visited {visited} of rc {rc})");
            }
        }
        order.reverse();
        order
    }

    pub fn debug(&self) {
        let line = "─".repeat(60);
        println!("\n{}", line);
        println!("  E-Graph");
        println!("{}", line);
        for cid in self.classes.ids() {
            let class = &self.classes[cid];
            let shape_str = format!("{:?}", self.shape(cid));
            let dtype_str = format!("{:?}", self.dtype(cid));
            println!("Class {:?} shape={} dtype={}", cid, shape_str, dtype_str);
            for &nid in &class.nodes {
                let kind = &self.nodes[nid].node;
                let inputs: Vec<ClassId> = match kind {
                    Node::Kernel { inputs, .. } => inputs.to_vec(),
                    _ => kind.class_params().collect(),
                };
                let name = match kind {
                    Node::Reduce { rop: bop, .. } => format!("Reduce {:?}", bop),
                    Node::Binary { bop, .. } => format!("Binary {:?}", bop),
                    Node::Assign { .. } => "Assign".into(),
                    Node::After { .. } => "After".into(),
                    Node::Unary { uop, .. } => format!("Unary {:?}", uop),
                    Node::Cast { dtype, .. } => format!("Cast {:?}", dtype),
                    Node::Kernel { program_id, time, .. } => format!("Kernel prog={:?} time={}", program_id, time),
                    Node::Expand { .. } => "Expand".into(),
                    Node::Permute { axes, .. } => format!("Permute {:?}", axes),
                    Node::Reshape { shape, .. } => format!("Reshape shape={shape:?}"),
                    Node::Pad { axis, lp, len, .. } => format!("Pad axis={axis:?} lp={lp:?} len={len:?}"),
                    Node::Narrow { axis, start, len, x } => format!("Narrow {x:?} axis={axis:?} start={start:?} len={len:?}"),
                    Node::Flip { axes, .. } => format!("Flip {:?}", axes),
                    Node::Stack { ops } => format!("Stack {:?}", ops),
                    Node::ToDevice { device, time, .. } => format!("ToDevice {:?} time={}", device, time),
                    Node::Contiguous { .. } => "Contiguous".into(),
                    Node::Const { value: v, .. } => format!("Const {:?}", v),
                    Node::Leaf { dtype, .. } => format!("Leaf {:?}", dtype),
                };
                println!("  {name} {nid:?}: inputs={inputs:?}");
            }
        }
        println!("{}\n", line);
    }

    /// For each kernel node, it needs to go over inputs. Inputs are either realized or other kernel nodes.
    /// Debug assert that. Then for each input, if that input comes from kernel on different device
    /// or if it's in buffer_map on different device, add EGraph::ToDevice node that moves it
    /// to the device of the Node::Kernel.
    // TODO Clean up this method, it's a mess
    pub fn add_memory_ops(&mut self, devices: &Slab<DeviceId, Device>, buffer_map: &Map<TensorId, BufferId>) {
        let class_ids: Vec<ClassId> = self.classes.ids().collect();
        for cid in class_ids {
            let node_ids: Vec<NodeId> = self.classes[cid].nodes.to_vec();
            for &nid in &node_ids {
                let (device_id, inputs) = match &self.nodes[nid].node {
                    Node::Kernel { program_id, inputs, .. } => {
                        debug_assert_ne!(program_id.device, DeviceId::NULL);
                        (program_id.device, inputs.clone())
                    }
                    _ => continue,
                };
                let dev_pool = devices[device_id].memory_pool_id();

                let class_of = self.nodes[nid].class_of;
                let mut new_inputs: Option<Box<[ClassId]>> = None;
                for (i, &input_cid) in inputs.iter().enumerate() {
                    let mut same_device = false;
                    let mut from_kernel = false;
                    for &inid in &self.classes[input_cid].nodes {
                        if let Node::Kernel { program_id, .. } = &self.nodes[inid].node {
                            from_kernel = true;
                            if program_id.device == device_id {
                                same_device = true;
                                break;
                            }
                        }
                    }
                    if from_kernel {
                        if !same_device {
                            let to_cid = self.push_to_device(input_cid, device_id, 0);
                            if to_cid != class_of {
                                let new_inputs = new_inputs.get_or_insert_with(|| inputs.clone());
                                new_inputs[i] = to_cid;
                            }
                        }
                    } else {
                        let already_on_device = self.classes[input_cid]
                            .nodes
                            .iter()
                            .any(|&inid| matches!(&self.nodes[inid].node, Node::ToDevice { device: d, .. } if *d == device_id));
                        if !already_on_device {
                            let is_leaf = self.classes[input_cid]
                                .nodes
                                .iter()
                                .any(|&inid| matches!(&self.nodes[inid].node, Node::Leaf { .. }));
                            if is_leaf {
                                let tid = self.leaf_map.get(&input_cid).copied().unwrap_or_else(|| {
                                    let leaf_cid = self.classes[input_cid]
                                        .nodes
                                        .iter()
                                        .find_map(|&inid| {
                                            if matches!(&self.nodes[inid].node, Node::Leaf { .. }) {
                                                Some(self.nodes[inid].class_of)
                                            } else {
                                                None
                                            }
                                        })
                                        .expect("already checked is_leaf");
                                    self.leaf_map[&leaf_cid]
                                });
                                let leaf_pool = buffer_map[&tid].pool;
                                if leaf_pool != dev_pool {
                                    let to_cid = self.push_to_device(input_cid, device_id, 0);
                                    if to_cid != cid && to_cid != class_of {
                                        let new_inputs = new_inputs.get_or_insert_with(|| inputs.clone());
                                        new_inputs[i] = to_cid;
                                    }
                                }
                            }
                        }
                    }
                }
                if let Some(new_inputs) = new_inputs
                    && let Node::Kernel { inputs: node_inputs, .. } = &mut self.nodes[nid].node
                {
                    *node_inputs = new_inputs;
                }
            }
        }
    }

    /// Hash of the graph structure (hashcons), output classes, and the shape
    /// and dtype of every class. Deterministic across equivalent graphs — used
    /// as a cache key for compiled plans.
    ///
    /// Shape and dtype must be part of the key: two graphs with the same node
    /// structure but different shapes/dtypes (e.g. an `f32[10]` sin vs an
    /// `f32[3]` sin) would otherwise share a plan with wrong allocation sizes.
    #[must_use]
    pub fn cache_key(&self, outputs: &BTreeSet<ClassId>) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        for (node, &id) in &self.hashcons {
            id.hash(&mut hasher);
            node.hash(&mut hasher);
        }

        // TODO - is this even needed?
        for cid in self.classes.ids() {
            cid.hash(&mut hasher);
        }

        for &cid in outputs {
            cid.hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Returns the set of Kernel/ToDevice nodes forming the cheapest valid computation from leaves
    /// to all outputs.
    ///
    /// # Cost model
    ///
    /// Only [`Node::Kernel`] and [`Node::ToDevice`] carry real costs (execution time in nanoseconds).
    /// All other node types (Expand, Reshape, Cast, Binary, Unary, etc.) are structural/fusing
    /// artifacts — they represent intermediate graph transformations that must be fused into kernels
    /// by [`kernelize`](self::kernelizer::Graph::kernelize) before extraction.
    ///
    /// # Invariant
    ///
    /// A path composed exclusively of [`Node::Kernel`] and [`Node::ToDevice`] nodes must exist
    /// from leaves (the only realized classes) to every output class. Without this path the output
    /// cannot be computed, because non-Kernel/ToDevice nodes have no associated runtime cost.
    ///
    /// Dead graph regions (classes with no kernel path) are harmless as long as they don't appear
    /// on output computation paths. [`kernelize`](self::kernelizer::Graph::kernelize) is responsible for ensuring every output
    /// class satisfies this invariant by fusing enough nodes into kernels.
    ///
    /// # Panics
    ///
    /// Panics if any output class lacks a producer path through Kernel or ToDevice nodes.
    #[must_use]
    pub fn extract(&self, outputs: &BTreeSet<ClassId>) -> Vec<NodeId> {
        let order = self.topo_sort_for_extract(outputs);

        let n = self.classes.ids().count();
        let is_leaf: Vec<bool> = (0..n)
            .map(|i| {
                let cid = ClassId(i as u32);
                self.classes[cid].nodes.iter().any(|&nid| matches!(&self.nodes[nid].node, Node::Leaf { .. }))
            })
            .collect();

        // Candidate producer nodes per class: Kernel and ToDevice nodes. Multiple
        // kernels may produce the same class (different fusions compete in
        // extraction); a leaf class is already realized and never needs one.
        #[derive(Clone, Copy)]
        struct Cand {
            nid: NodeId,
            time: u64,
        }
        let mut cands: Vec<Vec<Cand>> = vec![Vec::new(); n];
        let nn = self.nodes.ids().count();
        let mut node_in: Vec<Vec<ClassId>> = vec![Vec::new(); nn];
        let mut node_out: Vec<Vec<ClassId>> = vec![Vec::new(); nn];
        let mut node_time: Vec<u64> = vec![0; nn];
        for &cid in &order {
            for &nid in &self.classes[cid].nodes {
                let (time, inputs, outputs) = match &self.nodes[nid].node {
                    Node::Kernel { inputs, outputs, time, .. } => (*time, inputs.to_vec(), outputs.to_vec()),
                    Node::ToDevice { x, time, .. } => {
                        let outputs = vec![self.nodes[nid].class_of];
                        (*time, vec![*x], outputs)
                    }
                    _ => continue,
                };
                node_time[nid.0 as usize] = time;
                node_in[nid.0 as usize] = inputs.clone();
                node_out[nid.0 as usize] = outputs.clone();
                cands[cid.0 as usize].push(Cand { nid, time });
            }
        }

        // After classes alias their base leaf buffer; their value comes from the
        // assign writing in-place over the previous version of that buffer.
        // Needing an After class forces its whole assign chain to run (every
        // earlier After plus the assign classes) — otherwise the in-place store
        // kernels of chained assigns get dropped. Mirrors the backward walk below.
        let mut after_chain: Vec<Vec<ClassId>> = vec![Vec::new(); n];
        for &cid in &order {
            let mut chain = Vec::new();
            let mut cur = cid;
            while let Some(&nid2) =
                self.classes[cur].nodes.iter().find(|&&nid| matches!(&self.nodes[nid].node, Node::After { .. }))
            {
                let Node::After { x, dep } = &self.nodes[nid2].node else {
                    unreachable!()
                };
                chain.push(*x);
                chain.push(*dep);
                if *x == cur {
                    break;
                }
                cur = *x;
            }
            after_chain[cid.0 as usize] = chain;
        }

        struct Ctx<'a> {
            outputs: &'a BTreeSet<ClassId>,
            order: &'a [ClassId],
            cands: &'a [Vec<Cand>],
            node_in: &'a [Vec<ClassId>],
            node_out: &'a [Vec<ClassId>],
            node_time: &'a [u64],
            after_chain: &'a [Vec<ClassId>],
            is_leaf: &'a [bool],
        }

        impl Ctx<'_> {
            /// The classes that still must be produced (`pending`, in topological
            /// order) and the classes already produced, derived from `selected`.
            fn pending_and_produced(&self, selected: &Set<NodeId>) -> (Vec<ClassId>, Set<ClassId>) {
                let mut produced: Set<ClassId> = Set::default();
                let mut requested: Set<ClassId> = self.outputs.iter().copied().collect();
                for &nid in selected {
                    for &o in &self.node_out[nid.0 as usize] {
                        produced.insert(o);
                    }
                    for &i in &self.node_in[nid.0 as usize] {
                        requested.insert(i);
                    }
                }
                loop {
                    let mut add: Vec<ClassId> = Vec::new();
                    for &c in &requested {
                        for &r in &self.after_chain[c.0 as usize] {
                            if !requested.contains(&r) {
                                add.push(r);
                            }
                        }
                    }
                    if add.is_empty() {
                        break;
                    }
                    for r in add {
                        requested.insert(r);
                    }
                }
                let mut pending = Vec::new();
                for &c in self.order {
                    if !produced.contains(&c)
                        && requested.contains(&c)
                        && !self.is_leaf[c.0 as usize]
                        && !self.cands[c.0 as usize].is_empty()
                    {
                        pending.push(c);
                    }
                }
                (pending, produced)
            }

            fn plan_cost(&self, selected: &Set<NodeId>) -> u64 {
                selected.iter().map(|&nid| self.node_time[nid.0 as usize]).sum()
            }

            /// A feasible plan that selects the cheapest producer of each pending
            /// class in topological order. Always terminates; provides the upper
            /// bound for the search and a safe fallback.
            fn greedy(&self) -> Set<NodeId> {
                let mut selected: Set<NodeId> = Set::default();
                loop {
                    let (pending, _) = self.pending_and_produced(&selected);
                    if pending.is_empty() {
                        return selected;
                    }
                    let c = pending[0];
                    let cand = self.cands[c.0 as usize].iter().min_by_key(|k| k.time).expect("pending class has no candidates");
                    selected.insert(cand.nid);
                }
            }

            /// Branch-and-bound DFS over producer sets. `selected` is the current
            /// set, `cost` the cost so far, `best` the best total cost seen
            /// (prunes branches that cannot improve it). Returns the cheapest
            /// completion from this state and the nodes it selects.
            fn search(&self, selected: &mut Set<NodeId>, cost: u64, best: &mut u64) -> Option<(u64, Vec<NodeId>)> {
                let (pending, _) = self.pending_and_produced(selected);
                if pending.is_empty() {
                    return Some((0, Vec::new()));
                }
                let c = pending[0];
                let mut ordered: Vec<&Cand> = self.cands[c.0 as usize].iter().collect();
                ordered.sort_by_key(|k| k.time);
                let mut best_res: Option<(u64, Vec<NodeId>)> = None;
                for cand in ordered {
                    if selected.contains(&cand.nid) {
                        continue;
                    }
                    let new_cost = cost + cand.time;
                    if new_cost >= *best {
                        continue;
                    }
                    selected.insert(cand.nid);
                    if let Some((rest, mut nodes)) = self.search(selected, new_cost, best) {
                        let total = cand.time + rest;
                        nodes.push(cand.nid);
                        if best_res.as_ref().is_none_or(|(b, _)| total < *b) {
                            best_res = Some((total, nodes));
                            *best = (*best).min(cost + total);
                        }
                    }
                    selected.remove(&cand.nid);
                }
                best_res
            }
        }

        let ctx = Ctx {
            outputs,
            order: &order,
            cands: &cands,
            node_in: &node_in,
            node_out: &node_out,
            node_time: &node_time,
            after_chain: &after_chain,
            is_leaf: &is_leaf,
        };

        let greedy_plan = ctx.greedy();
        let greedy_cost = ctx.plan_cost(&greedy_plan);

        // Output classes must have a producer path through Kernel/ToDevice
        // nodes. Leaves are already realized and need none.
        let (_, produced) = ctx.pending_and_produced(&greedy_plan);
        for &ocid in outputs {
            if !is_leaf[ocid.0 as usize] && !produced.contains(&ocid) {
                panic!("class {ocid:?} has no valid producer path through Kernel or ToDevice nodes");
            }
        }

        // Cheapest closed producer set: the search improves on greedy when a
        // cheaper closure exists, otherwise greedy is already optimal.
        let mut best = greedy_cost;
        let mut winning = greedy_plan.clone();
        if let Some((total, nodes)) = ctx.search(&mut Set::default(), 0, &mut best)
            && total < greedy_cost
        {
            winning = nodes.into_iter().collect();
        }

        // Producer of each class in the winning plan (multi-output kernels
        // produce several classes at once).
        let mut producer: Vec<Option<NodeId>> = vec![None; n];
        for &nid in &winning {
            for &oc in &node_out[nid.0 as usize] {
                producer[oc.0 as usize] = Some(nid);
            }
        }

        // Mark every class needed to compute the outputs by walking backward from
        // the outputs through the selected producers. The winning plan's selected
        // set is already closed under its producers, so this is a no-op on the
        // pure kernel graph — it exists to (a) thread the After/assign chains
        // below and (b) emit the producers in class-topological order.
        let mut needed: Vec<bool> = vec![false; n];
        let mut stack: Vec<ClassId> = outputs.iter().copied().collect();
        loop {
            while let Some(cid) = stack.pop() {
                if !needed[cid.0 as usize] {
                    needed[cid.0 as usize] = true;
                    if let Some(nid) = producer[cid.0 as usize] {
                        match &self.nodes[nid].node {
                            Node::Kernel { inputs, .. } => stack.extend(inputs.iter().copied()),
                            Node::ToDevice { x, .. } => stack.push(*x),
                            _ => {}
                        }
                    }
                    // After classes alias their base leaf buffer, and their
                    // value comes from dep (the assign) writing over x's
                    // version of that buffer. If the post-assign value is
                    // needed, the assign that wrote it and every earlier
                    // After in the chain are needed too — otherwise extract
                    // drops the in-place store kernels of chained assigns.
                    if let Some(&nid2) =
                        self.classes[cid].nodes.iter().find(|&&nid| matches!(&self.nodes[nid].node, Node::After { .. }))
                        && let Node::After { x, dep } = &self.nodes[nid2].node
                    {
                        stack.push(*x);
                        stack.push(*dep);
                    }
                }
            }
            // In-place assigns write into the realized buffer of their dst's base
            // leaf class, so the store kernel is a side effect on that buffer
            // rather than a producer on the read path — nothing consumes the
            // assign class, so the backward walk above never reaches it. Run the
            // store whenever the buffer it writes is needed.
            let mut add: Vec<ClassId> = Vec::new();
            for &cid in &order {
                if !needed[cid.0 as usize]
                    && self.classes[cid].nodes.iter().any(|&nid| {
                        matches!(&self.nodes[nid].node, Node::Assign { dst, .. } if needed[self.base_leaf(*dst).0 as usize])
                    })
                {
                    add.push(cid);
                }
            }
            if add.is_empty() {
                break;
            }
            for &cid in &add {
                needed[cid.0 as usize] = true;
            }
            stack.extend(add);
        }

        let mut result = Vec::new();
        let mut seen: Set<NodeId> = Set::default();
        for &cid in &order {
            if !needed[cid.0 as usize] {
                continue;
            }
            if let Some(nid) = producer[cid.0 as usize]
                && seen.insert(nid)
            {
                result.push(nid);
            }
        }
        result
    }

    pub fn rank(&self, class: ClassId) -> UAxis {
        self.shape(class).len() as UAxis
    }

    /// Shape of a class as dim classes (tinygrad-style symbolic shapes):
    /// each element is a class evaluating to a dimension value — a `Const`
    /// for static dims or a symbolic dim leaf otherwise. Empty vec for
    /// scalars.
    pub fn shape(&self, class: ClassId) -> Vec<ClassId> {
        match &self.nodes[self.classes[class].nodes[0]].node {
            Node::Const { .. } | Node::Stack { .. } => Vec::new(),
            Node::Leaf { shape, .. } => self.dims(*shape),
            Node::Expand { shape, .. } | Node::Reshape { shape, .. } => self.dims(*shape),
            Node::Permute { x, axes } => {
                let s = self.shape(*x);
                axes.iter().map(|&a| s[a as usize]).collect()
            }
            Node::Pad { x, axis, len, .. } => {
                let mut s = self.shape(*x);
                s[*axis as usize] = *len;
                s
            }
            Node::Narrow { x, axis, len, .. } => {
                let mut s = self.shape(*x);
                s[*axis as usize] = *len;
                s
            }
            Node::Flip { x, .. }
            | Node::Cast { x, .. }
            | Node::Unary { x, .. }
            | Node::After { x, .. }
            | Node::ToDevice { x, .. }
            | Node::Contiguous { x } => self.shape(*x),
            // Scalars broadcast implicitly (see `push_binary_node`): the
            // result takes the shape of the non-scalar operand. Both scalars
            // → rank 0.
            Node::Binary { x, y, .. } => {
                let sx = self.shape(*x);
                if !sx.is_empty() { sx } else { self.shape(*y) }
            }
            Node::Reduce { x, axes, .. } => {
                let s = self.shape(*x);
                s.into_iter().enumerate().filter(|(i, _)| !axes.contains(&*i)).map(|(_, d)| d).collect()
            }
            Node::Assign { dst, .. } => self.shape(*dst),
            Node::Kernel { outputs, .. } => self.shape(outputs[0]),
        }
    }

    /// Interpret a shape class: `NULL` is `[]`, a `Stack` of dim classes is
    /// its ops, anything else is a single bare dim class (rank-1 convention).
    pub fn dims(&self, shape: ClassId) -> Vec<ClassId> {
        if shape.is_null() {
            return Vec::new();
        }
        match &self.nodes[self.classes[shape].nodes[0]].node {
            Node::Stack { ops } => ops.to_vec(),
            _ => vec![shape],
        }
    }

    /// Replay a symbolic shape expression (egraph classes) into kernel IR.
    ///
    /// Graph-side counterpart of [`Runtime::replay_symbolic_into_kernel`]
    /// (slab → kernel) — see its doc for the shared contract. Differences
    /// forced by living on the egraph:
    ///
    /// - Operands are `ClassId`s, never TensorIds. TensorIds must not appear
    ///   inside the egraph or anything derived from it (graph hashing, replay,
    ///   plan caching all depend on this).
    /// - Dim variables are `Node::Leaf { dtype: IDX_T, shape: NULL }` classes;
    ///   each distinct class becomes exactly one `Param { kind: Variable }`
    ///   define plus one entry in `jit_kernels[kid].loads` (registered at mint
    ///   time so define order == load order and positional binding holds).
    /// - `dims` is the already-decomposed list of top-level dim classes (the
    ///   result of [`Graph::dims`]). Each is replayed as a full expression;
    ///   dedupe of shared subexpressions happens within this call via the
    ///   class map. Note this decomposition loses no structure: a dim
    ///   expression is always a scalar tree, only the outermost Stack layer is
    ///   flattened here, which re-emerges as a single `Op::Stack`.
    ///
    /// Panics loudly on any node outside the symbolic closed set — in
    /// particular on computed dims (`Reduce` results feeding shapes). Shapes
    /// are purely symbolic; a shape dimension may never be produced by a
    /// kernel (jax/inductor/tinygrad convention adopted repo-wide).
    pub(crate) fn replay_symbolic_into_kernel(&mut self, kid: JitKernelId, dims: &[ClassId]) -> OpId {
        // Post-order flatten: every class lands after its operands, so one
        // flat pass emits with operands already mapped.
        fn flatten(graph: &Graph, cid: ClassId, order: &mut Vec<ClassId>) {
            let nodes = &graph.classes[cid].nodes;
            debug_assert!(nodes.len() == 1, "symbolic dim class must have exactly one node, got {}", nodes.len());
            let node = &graph.nodes[nodes[0]].node;
            match node {
                Node::Const { .. } | Node::Leaf { .. } => (),
                Node::Cast { x, .. } | Node::Unary { x, .. } => flatten(graph, *x, order),
                Node::Binary { x, y, .. } => {
                    flatten(graph, *x, order);
                    flatten(graph, *y, order);
                }
                Node::Stack { ops } => {
                    for op in ops.iter() {
                        flatten(graph, *op, order);
                    }
                }
                n => panic!(
                    "shape expression contains non-symbolic node {:?}; shapes are purely symbolic and must never be computed by kernels",
                    n
                ),
            }
            order.push(cid);
        }

        let mut class_map: Map<ClassId, OpId> = Map::default();
        let mut dim_ops: Vec<OpId> = Vec::with_capacity(dims.len());
        for &cid in dims {
            if cid.is_null() {
                continue;
            }
            let mut order = Vec::new();
            flatten(self, cid, &mut order);
            let mut root = OpId::NULL;
            for c in order {
                if let Some(&mapped) = class_map.get(&c) {
                    root = mapped;
                    continue;
                }
                let nodes = self.classes[c].nodes.clone();
                debug_assert!(nodes.len() == 1, "symbolic dim class must have exactly one node");
                let node = self.nodes[nodes[0]].node.clone();
                let op_id = match node {
                    Node::Const { value, .. } => self.jit_kernels[kid].kernel.push_back(Op::Const(value)),
                    Node::Leaf { dtype, shape, .. } => {
                        debug_assert!(shape.is_null(), "dim-variable leaf must be scalar, got shape {:?}", shape);
                        debug_assert!(dtype == IDX_T, "dim-variable leaf must be {:?}-typed, got {:?}", IDX_T, dtype);
                        let op_id = self.jit_kernels[kid].kernel.param(IDX_T, ParamKind::Variable, OpId::NULL);
                        self.jit_kernels[kid].loads.push(c);
                        op_id
                    }
                    Node::Cast { x, dtype } => {
                        let a = class_map[&x];
                        self.jit_kernels[kid].kernel.cast(a, dtype)
                    }
                    Node::Unary { x, uop } => {
                        let a = class_map[&x];
                        self.jit_kernels[kid].kernel.unary(a, uop)
                    }
                    Node::Binary { x, y, bop } => {
                        let (a, b) = (class_map[&x], class_map[&y]);
                        self.jit_kernels[kid].kernel.binary(a, b, bop)
                    }
                    n => unreachable!("flatten rejected non-symbolic data {n:?}"),
                };
                class_map.insert(c, op_id);
                root = op_id;
            }
            dim_ops.push(root);
        }

        match dim_ops.len() {
            0 => OpId::NULL,
            1 => *dim_ops.last().unwrap(),
            _ => self.jit_kernels[kid].kernel.stack(&dim_ops),
        }
    }

    /// Replays a shape-descriptor class (a `Reshape`/`Expand` shape, a `Pad`
    /// `lp`/`len` bound, a `Narrow` `start`/`len` bound) directly into kernel
    /// `kid` and returns the root op of the replayed expression.
    ///
    /// Shape descriptors are pure symbolic metadata: the kernelizer never
    /// materializes kernels for them — each consumer replays the expression
    /// on demand (the graph-side mirror of eager's
    /// `Runtime::replay_symbolic_into_kernel`). A `Stack` class replays as a
    /// stack of its dim elements; any other class replays as a single dim
    /// expression. Read-only over the egraph: no graph or kernel mutation
    /// beyond emitting the expression's ops into `kid`.
    pub(crate) fn replay_shape_into_kernel(&mut self, kid: JitKernelId, shape: ClassId) -> OpId {
        if shape.is_null() {
            return OpId::NULL;
        }
        match &self.nodes[self.classes[shape].nodes[0]].node {
            Node::Stack { ops } => {
                let ops: Vec<ClassId> = ops.iter().copied().collect();
                self.replay_symbolic_into_kernel(kid, &ops)
            }
            _ => self.replay_symbolic_into_kernel(kid, &[shape]),
        }
    }

    pub fn dtype(&self, class: ClassId) -> DType {
        match &self.nodes[self.classes[class].nodes[0]].node {
            Node::Const { value: c, .. } => c.dtype(),
            Node::Leaf { dtype, .. } => *dtype,
            Node::Cast { dtype, .. } => *dtype,
            Node::Assign { dst, .. } => self.dtype(*dst),
            Node::Kernel { outputs, .. } => self.dtype(outputs[0]),
            Node::Stack { ops } => self.dtype(ops[0]),
            Node::Expand { x, .. }
            | Node::Permute { x, .. }
            | Node::Reshape { x, .. }
            | Node::Pad { x, .. }
            | Node::Flip { x, .. }
            | Node::Narrow { x, .. }
            | Node::Reduce { x, .. }
            | Node::Unary { x, .. }
            | Node::After { x, .. }
            | Node::ToDevice { x, .. }
            | Node::Contiguous { x }
            | Node::Binary { x, .. } => self.dtype(*x),
        }
    }

    /// Tries to resolve the value of a scalar class by walking its const
    /// expression: `Const` leaves evaluated through `Cast`, `Unary` and
    /// `Binary` nodes (iteratively, no recursion). Returns `None` if the
    /// class is not a scalar, the walk exceeds 10 000 steps, or any leaf
    /// is not a `Const`.
    pub(crate) fn resolve_const(&self, class: ClassId) -> Option<Constant> {
        // Preorder of the const-expression subgraph reachable through
        // `Cast`, `Unary` and `Binary`; non-const leaves abort.
        let mut visited: Set<NodeId> = Set::default();
        let mut order: Vec<NodeId> = Vec::new();
        let mut stack = vec![class];
        for _ in 0..10_000 {
            let Some(id) = stack.pop() else { break };
            let node_id = self.classes[id].nodes[0];
            if !visited.insert(node_id) {
                continue;
            }
            match &self.nodes[node_id].node {
                Node::Cast { x, .. } => stack.push(*x),
                Node::Unary { x, .. } => stack.push(*x),
                Node::Binary { x, y, .. } => {
                    stack.push(*y);
                    stack.push(*x);
                }
                Node::Const { .. } => {}
                // Every other variant is a non-scalar / dynamic leaf: not
                // resolvable to a constant.
                Node::Leaf { .. }
                | Node::Expand { .. }
                | Node::Permute { .. }
                | Node::Reshape { .. }
                | Node::Pad { .. }
                | Node::Flip { .. }
                | Node::Narrow { .. }
                | Node::Stack { .. }
                | Node::Reduce { .. }
                | Node::Assign { .. }
                | Node::After { .. }
                | Node::ToDevice { .. }
                | Node::Contiguous { .. }
                | Node::Kernel { .. } => return None,
            }
            order.push(node_id);
        }
        if !stack.is_empty() {
            panic!("resolve_const did not finish in 10000 steps");
        }
        // Evaluate bottom-up: `order` is a preorder (parents before their
        // operands), so reversing it evaluates every operand before its
        // consumer.
        let mut values: Map<NodeId, Constant> = Map::default();
        for &node_id in order.iter().rev() {
            let value = match &self.nodes[node_id].node {
                Node::Const { value, .. } => *value,
                Node::Cast { x, dtype } => values[&self.classes[*x].nodes[0]].cast(*dtype),
                Node::Unary { x, uop } => values[&self.classes[*x].nodes[0]].unary(*uop),
                Node::Binary { x, y, bop } => {
                    Constant::binary(values[&self.classes[*x].nodes[0]], values[&self.classes[*y].nodes[0]], *bop)
                }
                _ => unreachable!("non-expression node in const walk"),
            };
            values.insert(node_id, value);
        }
        Some(values[&self.classes[class].nodes[0]])
    }
}

impl Runtime {
    pub fn promote_to_graph(&mut self, tid: TensorId, graph_id: GraphId) -> Result<ClassId, ZyxError> {
        let (class_id, gid) = match self.tensors[tid] {
            TensorData::Graph { class_id, graph_id, .. } | TensorData::Promoted { class_id, graph_id, .. } => {
                (class_id, graph_id)
            }
            _ => (ClassId::NULL, GraphId::NULL),
        };
        if !class_id.is_null() {
            if !self.graphs[gid].dead {
                if graph_id == gid {
                    return Ok(class_id);
                } else {
                    panic!("tensor belongs to a different tape scope");
                }
            }
            // Graph is dead: the tensor reverts to eager (its kernel_id is still
            // valid since we never mutated the eager kernel). Clear the graph
            // affiliation before promoting it into a new scope. Its pending
            // store is gone because promotion materializes pending stores.
            match &mut self.tensors[tid] {
                TensorData::Promoted { kernel_id, op_id, shape_id, rc, dtype, .. } => {
                    let (kernel_id, op_id, shape_id, rc, dtype) = (*kernel_id, *op_id, *shape_id, *rc, *dtype);
                    self.tensors[tid] = TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc };
                }
                ref t => panic!("promote_to_graph: dead-graph tensor {tid} has no eager side to revert to: {t:?}"),
            }
            self.graphs[gid].ref_count -= 1;
            if self.graphs[gid].dead && self.graphs[gid].ref_count == 0 {
                self.remove_dead_graph(gid);
            }
        }

        let (kernel_id, my_op_id) = match self.tensors[tid] {
            TensorData::Eager { kernel_id, op_id, .. } | TensorData::Promoted { kernel_id, op_id, .. } => (kernel_id, op_id),
            ref t => panic!("promote_to_graph: tensor {tid} has no eager kernel: {t:?}"),
        };
        if !self.kernels[kernel_id].outputs.contains(&tid) {
            eprintln!(">>> PROMOTE tid={tid} kernel={kernel_id:?} NOT in outputs, data={:?}", self.tensors[tid]);
        }

        // Already realized eager tensors promote to the graph as leaves directly.
        // Their buffer is read by the plan as an input; the value is preserved and
        // not recomputed. The eager kernel is left untouched (rc/outputs already
        // count the handles), so the tensor reverts to eager when the graph dies.
        if self.buffer_map.contains_key(&tid) {
            let dtype = self.dtype(tid);
            // Build the leaf's symbolic shape class from the eager kernel's
            // own Param shape stack: const dims become Const classes, dynamic
            // dims (`Param { kind: Variable }`) become symbolic dim leaves.
            let shape_op = match self.kernels[kernel_id].kernel.ops[my_op_id].op {
                Op::Param { shape, .. } => shape,
                ref op => unreachable!("promote_to_graph: realized tensor op {op:?} is not a Param"),
            };
            let dim_entries: Vec<OpId> = if shape_op.is_null() {
                Vec::new()
            } else {
                match &self.kernels[kernel_id].kernel.ops[shape_op].op {
                    Op::Stack { ops } => ops.as_ref().to_vec(),
                    _ => vec![shape_op],
                }
            };
            let mut dim_classes = Vec::with_capacity(dim_entries.len());
            for entry in dim_entries {
                dim_classes.push(match self.kernels[kernel_id].kernel.ops[entry].op {
                    Op::Const(c) => self.push_const(graph_id, c),
                    Op::Param { kind: ParamKind::Variable, .. } => self.push_leaf_node(graph_id, IDX_T, ClassId::NULL).1,
                    ref op => unreachable!("promote_to_graph: dim op {op:?} in param shape stack"),
                });
            }
            let shape_class = match dim_classes.len() {
                0 => ClassId::NULL,
                1 => dim_classes[0],
                _ => self.push_node(graph_id, Node::Stack { ops: dim_classes.into_boxed_slice() }).1,
            };
            let (_, class_id) = self.push_leaf_node(graph_id, dtype, shape_class);
            self.graphs[graph_id].leaf_map.insert(class_id, tid);
            self.retain(tid);
            self.graphs[graph_id].leaf_classes.push(class_id);
            self.graphs[graph_id].ref_count += 1;
            match &mut self.tensors[tid] {
                TensorData::Graph { class_id: c, .. } | TensorData::Promoted { class_id: c, .. } => *c = class_id,
                TensorData::Eager { .. } => {
                    let (kernel_id, op_id, shape_id, rc, dtype) = match self.tensors[tid] {
                        TensorData::Eager { kernel_id, op_id, depends_on, shape_id, rc, dtype } => {
                            debug_assert!(depends_on.is_null(), "promoting unrealized tensor {tid} with pending store");
                            (kernel_id, op_id, shape_id, rc, dtype)
                        }
                        ref t => unreachable!("{t:?}"),
                    };
                    self.tensors[tid] = TensorData::Promoted { kernel_id, op_id, class_id, graph_id, shape_id, dtype, rc };
                }
                ref t => panic!("promote_to_graph: cannot attach tensor {tid} to the graph: {t:?}"),
            }
            return Ok(class_id);
        }

        debug_assert!(self.kernels[kernel_id].outputs.contains(&tid));

        let relevant = {
            let kernel = &self.kernels[kernel_id].kernel;
            let mut relevant: Set<OpId> = Set::default();
            let mut stack = vec![my_op_id];
            while let Some(oid) = stack.pop() {
                if !relevant.insert(oid) {
                    continue;
                }
                match &kernel.ops[oid].op {
                    Op::Storage { .. } | Op::Const(_) | Op::Param { .. } => {}
                    Op::Unary { x, .. } => stack.push(*x),
                    Op::Binary { x, y, .. } => {
                        stack.push(*x);
                        stack.push(*y);
                    }
                    Op::Cast { x, .. } => stack.push(*x),
                    Op::Reduce { x, .. } => stack.push(*x),
                    Op::Move { x, mop } => {
                        stack.push(*x);
                        match mop.as_ref() {
                            MoveOp::Reshape { shape } | MoveOp::Expand { shape } => stack.push(*shape),
                            MoveOp::Pad { lp, len, .. } => {
                                stack.push(*lp);
                                stack.push(*len);
                            }
                            MoveOp::Narrow { start, len, .. } => {
                                stack.push(*start);
                                stack.push(*len);
                            }
                            MoveOp::Permute { .. } | MoveOp::Flip { .. } => {}
                        }
                    }
                    Op::Stack { ops } => stack.extend(ops.iter().copied()),
                    op => unreachable!("promote_to_graph: eager kernel op {op:?}"),
                }
            }
            relevant
        };

        let loads = self.kernels[kernel_id].loads.clone();
        let mut op_to_class: Map<OpId, ClassId> = Map::default();
        let mut storage_idx = 0;
        let mut op_id = self.kernels[kernel_id].kernel.head;
        while !op_id.is_null() {
            if relevant.contains(&op_id) {
                let class_id = match self.kernels[kernel_id].kernel.ops[op_id].op {
                    Op::Param { shape, dtype, .. } => {
                        let load_tid = loads[storage_idx];
                        if !self.buffer_map.contains_key(&load_tid) {
                            // An `Eager` tensor never carries a graph class,
                            // so its depends_on is the pending producer.
                            let pending = match &self.tensors[load_tid] {
                                TensorData::Eager { depends_on, .. } => *depends_on,
                                TensorData::Graph { .. } | TensorData::Promoted { .. } => KernelId::NULL,
                                ref t => panic!("promote_to_graph: load tid {load_tid} is not a kernel tensor: {t:?}"),
                            };
                            debug_assert!(!pending.is_null());
                            let outputs: Vec<TensorId> = self.kernels[pending].outputs.iter().copied().collect();
                            for &otid in &outputs {
                                self.add_store(otid)?;
                            }
                        }

                        let load_is_leaf = match &self.tensors[load_tid] {
                            TensorData::Graph { class_id: c, graph_id: g, .. }
                            | TensorData::Promoted { class_id: c, graph_id: g, .. } => {
                                !c.is_null() && *g == graph_id && !self.graphs[graph_id].dead
                            }
                            _ => false,
                        };
                        if load_is_leaf {
                            // load_tid is already a leaf of this graph: reuse its class.
                            match &self.tensors[load_tid] {
                                TensorData::Graph { class_id: c, .. } | TensorData::Promoted { class_id: c, .. } => *c,
                                ref t => unreachable!("{t:?}"),
                            }
                        } else {
                            // Create load_tid's leaf, with the symbolic shape
                            // class built from this Param's own shape stack
                            // (const dims → Const classes, dynamic dims →
                            // symbolic dim leaves).
                            let dim_entries: Vec<OpId> = if shape.is_null() {
                                Vec::new()
                            } else {
                                match &self.kernels[kernel_id].kernel.ops[shape].op {
                                    Op::Stack { ops } => ops.as_ref().to_vec(),
                                    _ => vec![shape],
                                }
                            };
                            let mut dim_classes = Vec::with_capacity(dim_entries.len());
                            for entry in dim_entries {
                                dim_classes.push(match self.kernels[kernel_id].kernel.ops[entry].op {
                                    Op::Const(c) => self.push_const(graph_id, c),
                                    Op::Param { kind: ParamKind::Variable, .. } => {
                                        self.push_leaf_node(graph_id, IDX_T, ClassId::NULL).1
                                    }
                                    ref op => unreachable!("promote_to_graph: dim op {op:?} in param shape stack"),
                                });
                            }
                            let shape_class = match dim_classes.len() {
                                0 => ClassId::NULL,
                                1 => dim_classes[0],
                                _ => self.push_node(graph_id, Node::Stack { ops: dim_classes.into_boxed_slice() }).1,
                            };
                            let (_, class_id) = self.push_leaf_node(graph_id, dtype, shape_class);
                            self.graphs[graph_id].leaf_map.insert(class_id, load_tid);
                            self.retain(load_tid);
                            self.graphs[graph_id].leaf_classes.push(class_id);
                            self.graphs[graph_id].ref_count += 1;
                            match &mut self.tensors[load_tid] {
                                TensorData::Graph { class_id: c, .. } | TensorData::Promoted { class_id: c, .. } => *c = class_id,
                                TensorData::Eager { .. } => {
                                    // A disowned load (user handle gone, not in
                                    // its producer's `outputs`) has no eager
                                    // future: after the tape dies nobody can use
                                    // it eagerly, so drop the eager side and
                                    // make it a pure graph leaf. Its buffer (the
                                    // Param branch just materialized it) stays
                                    // alive through the leaf edge and is freed
                                    // by its death path.
                                    let (kernel_id, op_id, shape_id, rc, dtype) = match self.tensors[load_tid] {
                                        TensorData::Eager { kernel_id, op_id, depends_on, shape_id, rc, dtype } => {
                                            debug_assert!(
                                                depends_on.is_null(),
                                                "promoting unrealized tensor {load_tid} with pending store"
                                            );
                                            (kernel_id, op_id, shape_id, rc, dtype)
                                        }
                                        ref t => unreachable!("{t:?}"),
                                    };
                                    if self.kernels[kernel_id].outputs.contains(&load_tid) {
                                        self.tensors[load_tid] =
                                            TensorData::Promoted { kernel_id, op_id, class_id, graph_id, shape_id, dtype, rc };
                                    } else {
                                        self.tensors[load_tid] = TensorData::Graph { class_id, graph_id, shape_id, dtype, rc };
                                    }
                                }
                                ref t => panic!("promote_to_graph: cannot attach load tensor {load_tid} to the graph: {t:?}"),
                            }
                            class_id
                        }
                    }
                    Op::Const(x) => {
                        let class_id = self.push_const(graph_id, x);
                        class_id
                    }
                    Op::Unary { x, uop } => {
                        let x_class = op_to_class[&x];
                        let (_, class_id) = self.push_node(graph_id, Node::Unary { x: x_class, uop });
                        class_id
                    }
                    Op::Binary { x, y, bop } => {
                        let x_class = op_to_class[&x];
                        let y_class = op_to_class[&y];
                        self.push_binary_node(graph_id, x_class, y_class, bop)
                    }
                    Op::Cast { x, dtype } => {
                        let x_class = op_to_class[&x];
                        let (_, class_id) = self.push_node(graph_id, Node::Cast { x: x_class, dtype });
                        class_id
                    }
                    Op::Stack { ref ops } => {
                        let ops: Box<[ClassId]> = ops.iter().map(|o| op_to_class[o]).collect();
                        let (_, class_id) = self.push_node(graph_id, Node::Stack { ops });
                        class_id
                    }
                    Op::Reduce { x, rop, .. } => {
                        let x_class = op_to_class[&x];
                        let rank = self.graphs[graph_id].rank(x_class);
                        debug_assert!(rank >= 1, "Reduce: input rank must be >= 1");
                        let (_, class_id) =
                            self.push_node(graph_id, Node::Reduce { x: x_class, rop, axes: vec![rank - 1].into() });
                        class_id
                    }
                    Op::Move { x, ref mop } => {
                        let x_class = op_to_class[&x];
                        let in_shape = self.graphs[graph_id].shape(x_class);
                        match mop.as_ref() {
                            MoveOp::Reshape { shape } => {
                                let shape = op_to_class[&shape];
                                let (_, class_id) = self.push_node(graph_id, Node::Reshape { x: x_class, shape });
                                class_id
                            }
                            MoveOp::Expand { shape } => {
                                let shape = op_to_class[&shape];
                                let (_, class_id) = self.push_node(graph_id, Node::Expand { x: x_class, shape });
                                class_id
                            }
                            MoveOp::Permute { axes } => {
                                debug_assert_eq!(
                                    axes.len(),
                                    in_shape.len(),
                                    "Permute: axes length {} != input rank {} (shape {:?})",
                                    axes.len(),
                                    in_shape.len(),
                                    in_shape
                                );
                                /*debug_assert_eq!(
                                    shape.len(),
                                    in_shape.len(),
                                    "Permute: output shape rank {} != input rank {} (shape {:?})",
                                    shape.len(),
                                    in_shape.len(),
                                    in_shape
                                );*/
                                let axes = axes.clone().into();
                                let (_, class_id) = self.push_node(graph_id, Node::Permute { x: x_class, axes });
                                class_id
                            }
                            MoveOp::Pad { axis, lp, len } => {
                                let lp = op_to_class[&lp];
                                let len = op_to_class[&len];
                                let (_, class_id) = self.push_node(graph_id, Node::Pad { x: x_class, axis: *axis, lp, len });
                                class_id
                            }
                            MoveOp::Narrow { axis, start, len } => {
                                let start = op_to_class[&start];
                                let len = op_to_class[&len];
                                let (_, class_id) =
                                    self.push_node(graph_id, Node::Narrow { x: x_class, axis: *axis, start, len });
                                class_id
                            }
                            MoveOp::Flip { axes } => {
                                debug_assert!(
                                    !axes.is_empty(),
                                    "Flip: axes must not be empty (rank {} shape {:?})",
                                    in_shape.len(),
                                    in_shape
                                );
                                let axes = axes.clone().into();
                                let (_, class_id) = self.push_node(graph_id, Node::Flip { x: x_class, axes });
                                class_id
                            }
                        }
                    }
                    _ => unreachable!(),
                };
                op_to_class.insert(op_id, class_id);
            }

            if matches!(self.kernels[kernel_id].kernel.at(op_id), Op::Storage { .. }) {
                storage_idx += 1;
            }
            op_id = self.kernels[kernel_id].kernel.next_op(op_id);
        }

        let class_id = op_to_class[&my_op_id];
        self.graphs[graph_id].ref_count += 1;
        match &mut self.tensors[tid] {
            TensorData::Graph { class_id: c, .. } | TensorData::Promoted { class_id: c, .. } => *c = class_id,
            TensorData::Eager { .. } => {
                let (kernel_id, op_id, shape_id, rc, dtype) = match self.tensors[tid] {
                    TensorData::Eager { kernel_id, op_id, depends_on, shape_id, rc, dtype } => {
                        debug_assert!(depends_on.is_null(), "promoting unrealized tensor {tid} with pending store");
                        (kernel_id, op_id, shape_id, rc, dtype)
                    }
                    ref t => unreachable!("{t:?}"),
                };
                self.tensors[tid] = TensorData::Promoted { kernel_id, op_id, class_id, graph_id, shape_id, dtype, rc };
            }
            ref t => panic!("promote_to_graph: cannot attach tensor {tid} to the graph: {t:?}"),
        }
        Ok(class_id)
    }

    pub fn autotune_jit_kernels(&mut self, graph_id: GraphId) -> Result<(), ZyxError> {
        println!("Autotuning");
        let device_ids: Vec<DeviceId> = self.devices.ids().collect();

        let jit_kernels: *const Slab<JitKernelId, JitKernelData> = &self.graphs[graph_id].jit_kernels;
        let jit_kernels: &Slab<JitKernelId, JitKernelData> = unsafe { &*jit_kernels };
        let total = jit_kernels.len().0 as i64 * device_ids.len() as i64;
        let mut bar = crate::progress::ProgressBar::new(total as u64);
        for ek in jit_kernels.values() {
            let (flop, read, write) = ek.kernel.flop_mem_rw();
            let class_of = ek.stores.first().copied().unwrap();

            for &dev_id in device_ids.iter() {
                // AOT-only devices (e.g. cblas) never compile generic zyx kernels
                if self.devices[dev_id].aot_only() {
                    continue;
                }
                let pool_id = self.devices[dev_id].memory_pool_id();
                let mut kernel = ek.kernel.clone();
                kernel.device_id = dev_id;
                bar.inc(1, &format!("autotune {} on dev={}", kernel.name(), dev_id.0));
                let (dev_prog, _opts, timing) = self.get_or_autotune(kernel, pool_id, flop, read, write, &[])?;
                let prog = ProgramId { device: dev_id, program: dev_prog };
                #[cfg(feature = "viz")]
                {
                    let sched_kernel = ek.kernel.clone();
                    let dev = &self.devices[dev_id];
                    let kc = crate::viz::KernelCapture {
                        sched_kernel,
                        opt_seq: _opts,
                        dev_info: dev.info().clone(),
                        device_label: dev.name(),
                        cc: dev.compute_capability(),
                        has_openmp: dev.has_openmp(),
                    };
                    self.viz.record(prog, kc);
                }

                let knid = self.graphs[graph_id].nodes.push(NodeData {
                    node: Node::Kernel {
                        inputs: ek.loads.clone().into(),
                        outputs: ek.stores.clone().into(),
                        program_id: prog,
                        time: timing,
                    },
                    class_of,
                });

                for &ocid in &*ek.stores {
                    self.graphs[graph_id].classes[ocid].nodes.push(knid);
                }
                if !ek.stores.contains(&class_of) {
                    self.graphs[graph_id].classes[class_of].nodes.push(knid);
                }
            }
        }

        if cfg!(debug_assertions) {
            let mut seen: Set<NodeId> = Set::default();
            for cid in self.graphs[graph_id].classes.ids() {
                for &nid in &self.graphs[graph_id].classes[cid].nodes {
                    if !seen.insert(nid) {
                        continue;
                    }
                    if let Node::Kernel { time, .. } = &self.graphs[graph_id].nodes[nid].node {
                        debug_assert!(*time > 0, "Kernel node {nid:?} has zero cost after autotune");
                    }
                }
            }
        }

        Ok(())
    }

    pub(crate) fn debug_assert_pre_realize(&self, graph_id: GraphId) {
        if cfg!(debug_assertions) {
            // I2: all leaves realized. A leaf is either a directly-promoted
            // realized tensor (Graph state) or the load tensor of a promoted
            // kernel (Eager state) — both carry a buffer.
            for &tid in self.graphs[graph_id].leaf_map.values() {
                debug_assert!(
                    self.buffer_map.contains_key(&tid) | self.variable_map.contains_key(&tid),
                    "leaf {tid} not realized"
                );
                let affiliated = match self.tensors[tid] {
                    TensorData::Graph { graph_id: g, .. } | TensorData::Promoted { graph_id: g, .. } => g == graph_id,
                    ref t => panic!("leaf {tid} is not a graph tensor: {t:?}"),
                };
                debug_assert!(affiliated, "leaf {tid} belongs to another graph");
            }
            // I2: no non-leaf graph tensor is realized — except in-place assign
            // targets, whose value lives in the (realized) leaf buffer they alias.
            for (tid, td) in self.tensors.iter() {
                let (affiliated, class_id) = match td {
                    TensorData::Graph { class_id: c, graph_id: g, .. }
                    | TensorData::Promoted { class_id: c, graph_id: g, .. } => (*g == graph_id, *c),
                    _ => continue,
                };
                if affiliated && !self.graphs[graph_id].is_leaf(class_id) && !self.graphs[graph_id].is_after(class_id) {
                    debug_assert!(!self.buffer_map.contains_key(&tid), "non-leaf graph tensor {tid} realized before realize");
                }
            }
        }
    }

    /// Compiles the graph into an [`ExecPlan`]: pattern-matches AOT kernels,
    /// kernelizes the remaining structural nodes, autotunes the fused kernels,
    /// extracts the cheapest kernel path, and returns the resulting plan.
    pub(crate) fn compile_graph(&mut self, graph_id: GraphId, output_set: &BTreeSet<ClassId>) -> Result<ExecPlan, ZyxError> {
        debug_assert!(self.graphs.contains_id(graph_id));
        self.debug_assert_pre_realize(graph_id);

        if self.debug.egraph() {
            self.graphs[graph_id].debug();
        }

        for cid in self.graphs[graph_id].classes.ids() {
            let has_leaf = self.graphs[graph_id].classes[cid]
                .nodes
                .iter()
                .any(|&nid| matches!(&self.graphs[graph_id].nodes[nid].node, Node::Leaf { .. }));
            if has_leaf {
                let &tid = self.graphs[graph_id].leaf_map.get(&cid).expect("class {cid:?} has Leaf node but not in leaf_map");
                assert!(self.buffer_map.contains_key(&tid), "leaf class {cid:?} tid {tid:?} not in buffer_map");
            } else {
                assert!(!self.graphs[graph_id].leaf_map.contains_key(&cid), "class {cid:?} has no Leaf node but is in leaf_map");
            }
        }

        // Pattern match specialized AOT kernels (e.g. matmul -> cblas) so they can
        // compete with the fused zyx kernels in extraction.
        // SAFETY: devices, graphs and shapes are separate fields of Runtime, no aliasing, rust is stupid
        let dev_ids: Vec<DeviceId> = self.devices.ids().collect();
        let graph_ptr: *mut Graph = &mut self.graphs[graph_id];
        for dev_id in dev_ids {
            self.devices[dev_id].match_graph(unsafe { &mut *graph_ptr }, output_set);
        }

        // AOT kernel output classes, grouped by the memory pool they run in.
        let mut pool_kernel_outputs: Map<PoolId, Set<ClassId>> = Map::default();
        for cid in self.graphs[graph_id].classes.ids() {
            for nid in &self.graphs[graph_id].classes[cid].nodes {
                if let Node::Kernel { program_id, .. } = &self.graphs[graph_id].nodes[*nid].node {
                    let pool = self.devices[program_id.device].memory_pool_id();
                    pool_kernel_outputs.entry(pool).or_default().insert(cid);
                }
            }
        }

        // Pass 1: fill every gap between all AOT kernels, ignoring devices.
        let all_kernel_outputs: Set<ClassId> = pool_kernel_outputs.values().flatten().copied().collect();
        self.graphs[graph_id].fill_gaps(&all_kernel_outputs, output_set);

        // Pass 2: for each memory pool, fill the gaps between only that pool's
        // kernels — other pools' kernels are ignored, giving single-pool paths.
        for active_outputs in pool_kernel_outputs.values() {
            self.graphs[graph_id].fill_gaps(active_outputs, output_set);
        }

        // Autotunes custom zyx kernels for all devices and adds kernel nodes for all of them
        self.autotune_jit_kernels(graph_id)?;
        self.graphs[graph_id].verify();

        // After all kernels nodes are added, this adds movement ops so extract can pick fastest path
        let devices_ptr: *const Slab<DeviceId, Device> = &self.devices;
        let buffer_map_ptr: *const Map<TensorId, BufferId> = &self.buffer_map;
        self.graphs[graph_id].add_memory_ops(unsafe { &*devices_ptr }, unsafe { &*buffer_map_ptr });

        let nodes = self.graphs[graph_id].extract(output_set);

        // Leaf pools at compile time — the plan bakes the alias binding (and
        // any cross-pool copy) into its ExecNodes, so leaves must stay put.
        let mut leaf_pools: Map<ClassId, PoolId> = Map::default();
        for (&cid, &tid) in &self.graphs[graph_id].leaf_map {
            leaf_pools.insert(cid, self.buffer_map[&tid].pool);
        }
        let plan = ExecPlan::new(&self.graphs[graph_id], &nodes, output_set, &self.devices, &leaf_pools);
        if self.debug.egraph() {
            plan.debug();
        }
        #[cfg(feature = "viz")]
        self.viz.snapshot(&self.graphs[graph_id], &plan);

        Ok(plan)
    }

    pub fn eagerify(&mut self, tid: TensorId) {
        let realized = self.buffer_map.contains_key(&tid);
        let (old_kernel_id, _, graph_id, shape_id) = match self.tensors[tid] {
            TensorData::Graph { graph_id, shape_id, .. } => (KernelId::NULL, OpId::NULL, graph_id, shape_id),
            TensorData::Promoted { kernel_id, op_id, graph_id, shape_id, .. } => (kernel_id, op_id, graph_id, shape_id),
            // Already eager or a pure-slab value: nothing to do.
            _ => return,
        };

        if !realized {
            match self.tensors[tid] {
                TensorData::Promoted { kernel_id, op_id, shape_id, rc, dtype, .. } => {
                    // Unrealized promoted tensor: the eager producer kernel was
                    // never mutated, so just demote in place.
                    self.tensors[tid] = TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc };
                }
                TensorData::Graph { .. } => {
                    // Unrealized graph-only tensor: its value can only be
                    // recomputed by the (dropping) graph; keep it as a dead
                    // handle that panics on use.
                    return;
                }
                ref t => unreachable!("eagerify: unexpected variant after affiliation match: {t:?}"),
            }
        } else {
            // Realized: detach from any old eager producer (promoted only),
            // then build a fresh load kernel sharing the existing buffer —
            // no data moves. The slab-side `shape_id` is replayed into the
            // new kernel.
            let mut old_load_duplicates = 0usize;
            if !old_kernel_id.is_null() {
                // Live detach: this tensor is still alive (rc > 0), only its
                // affiliation moves to a fresh load kernel below. A kernel can
                // be used by MULTIPLE tensors, so the old producer survives
                // intact — just remove this tensor from its outputs. Its fate
                // is settled later by whoever truly kills it (`release` →
                // `on_rc_zero`); no death bookkeeping here.
                let removed = self.kernels[old_kernel_id].outputs.remove(&tid);
                debug_assert!(removed, "eagerify: tid {tid} not listed in outputs of producer {old_kernel_id:?}");
                old_load_duplicates = self.kernels[old_kernel_id].loads.iter().filter(|&&t| t == tid).count();
            }
            let dtype = self.dtype(tid);
            let kernel_id = self.kernels.push(KernelData {
                outputs: Set::from_iter([tid]),
                loads: Vec::new(),
                stores: Vec::new(),
                kernel: Kernel::new(DeviceId::AUTO),
            });
            let shape_op = self.replay_symbolic_into_kernel(kernel_id, shape_id);
            let op_id = self.kernels[kernel_id].kernel.push_back(Op::Param { dtype, kind: ParamKind::Global, shape: shape_op });
            self.kernels[kernel_id].loads.push(tid);
            let (rc, dtype) = match self.tensors[tid] {
                TensorData::Graph { rc, dtype, .. } | TensorData::Promoted { rc, dtype, .. } => (rc, dtype),
                ref t => unreachable!("eagerify: {t:?}"),
            };
            self.tensors[tid] = TensorData::Eager { kernel_id, op_id, depends_on: KernelId::NULL, shape_id, dtype, rc };
            // The new kernel-load occurrence carries its own rc reference.
            self.retain(tid);
            // Fully detach from the old producer: its load entries on tid are
            // released — the fresh kernel's entry (retained above) replaces
            // them. Without this the old kernel keeps a stale edge whose count
            // pins tid above the breaker threshold forever.
            if old_load_duplicates > 0 {
                self.kernels[old_kernel_id].loads.retain(|&t| t != tid);
                for _ in 0..old_load_duplicates {
                    self.release(tid);
                }
                if self.kernels[old_kernel_id].outputs.is_empty() && self.kernels[old_kernel_id].stores.is_empty() {
                    // The old producer is dead wood: drop it and release its
                    // remaining load edges (same recursion as release's
                    // kernel-drop branch).
                    for &t in &self.kernels[old_kernel_id].loads {
                        if let TensorData::Eager { kernel_id: k, .. } | TensorData::Promoted { kernel_id: k, .. } =
                            &mut self.tensors[t]
                        {
                            if *k == old_kernel_id {
                                *k = KernelId::NULL;
                            }
                        }
                    }
                    let loads = std::mem::take(&mut self.kernels[old_kernel_id].loads);
                    self.kernels.remove(old_kernel_id);
                    for t in loads {
                        self.release(t);
                    }
                }
            }
        }

        self.graphs[graph_id].ref_count -= 1;
    }

    pub fn assert_graph_alive(&self, graph_id: GraphId) {
        assert!(!graph_id.is_null(), "tape scope has ended (tensor belongs to a dead tape scope)");
        assert!(!self.graphs[graph_id].dead, "tape scope has ended (tensor belongs to a dead tape scope");
    }

    /// Pushes a constant node into the graph and returns its class.
    ///
    /// The preferred way to create consts: `push_node` assigns the fresh
    /// `cons_id` that keeps every const in its own class (see
    /// [`Node::Const`] for why that is load-bearing).
    pub fn push_const(&mut self, graph_id: GraphId, value: Constant) -> ClassId {
        self.push_node(graph_id, Node::Const { cons_id: 0, value }).1
    }

    pub fn push_leaf_node(&mut self, graph_id: GraphId, dtype: DType, shape: ClassId) -> (NodeId, ClassId) {
        // Fresh cons_id: leaves hashcons but never merge (each buffer keeps
        // its own class).
        let cons_id = self.graphs[graph_id].max_cons_id;
        self.graphs[graph_id].max_cons_id += 1;
        let node = Node::Leaf { cons_id, dtype, shape };
        let g = &mut self.graphs[graph_id];
        let nid = g.nodes.push(NodeData { node: node.clone(), class_of: ClassId::NULL });
        let cid = g.classes.push(EClass { nodes: vec![nid] });
        g.nodes[nid].class_of = cid;
        g.hashcons.insert(node, nid);
        (nid, cid)
    }

    /// Numeric shape of a class for the runtime's `shapes` cache: static dim
    pub fn push_node(&mut self, graph_id: GraphId, node: Node) -> (NodeId, ClassId) {
        match node {
            Node::Permute { .. } => {
                /*let in_shape = &self.shapes[self.graphs[graph_id].classes[x].shape];
                assert_eq!(
                    axes.len(),
                    in_shape.len(),
                    "Permute: axes length {} != input rank {} (shape {:?})",
                    axes.len(),
                    in_shape.len(),
                    in_shape
                );*/
            }
            Node::Reshape { .. } => {
                /*let in_shape = &self.shapes[self.graphs[graph_id].classes[x].shape];
                let out_shape = &self.shapes[out_shape_id];
                assert_eq!(
                    in_shape.iter().product::<Dim>(),
                    out_shape.iter().product::<Dim>(),
                    "Reshape: element count mismatch {:?} -> {:?}",
                    in_shape,
                    out_shape
                );*/
            }
            Node::Expand { .. } => { /* shape dims not yet resolved (Stack). Re-enable once shape() resolves Stack. */ }
            Node::Pad { x, axis, .. } => {
                let in_rank = self.graphs[graph_id].rank(x);
                assert!(axis < in_rank, "Pad: axis {} out of range for input rank {}", axis, in_rank);
            }
            _ => {}
        }
        let g = &mut self.graphs[graph_id];
        // Const and Leaf carry a fresh cons_id (assigned here; constructors
        // pass 0): they hashcons like everything else, but the unique id means
        // two consts/leaves never merge into one class — no class is ever
        // shared between creation sites, so none gets pinned to whichever
        // kernel materialized it first.
        let node = match node {
            Node::Const { value, .. } => {
                let cons_id = g.max_cons_id;
                g.max_cons_id += 1;
                Node::Const { cons_id, value }
            }
            node => node,
        };
        if let Some(&nid) = g.hashcons.get(&node) {
            return (nid, g.nodes[nid].class_of);
        }
        let nid = g.nodes.push(NodeData { node: node.clone(), class_of: ClassId::NULL });
        let cid = g.classes.push(EClass { nodes: vec![nid] });
        g.nodes[nid].class_of = cid;
        g.hashcons.insert(node, nid);
        (nid, cid)
    }

    pub fn push_binary_node(&mut self, graph_id: GraphId, x: ClassId, y: ClassId, bop: BOp) -> ClassId {
        // With symbolic shapes we can only check rank — dim classes may differ
        // yet resolve equal (e.g. dims built from user tensors). Numeric
        // broadcastability is validated upstream by Tensor::broadcast.
        let (rx, ry) = (self.graphs[graph_id].rank(x), self.graphs[graph_id].rank(y));
        debug_assert!(
            rx == ry || rx == 0 || ry == 0,
            "binary operand ranks must match (scalars broadcast implicitly): {rx} vs {ry}"
        );
        // Scalars broadcast implicitly — make the expand an explicit graph node
        let (x, y) = match (rx, ry) {
            (_, 0) if rx > 0 => {
                let shape = self.shape_class(graph_id, self.graphs[graph_id].shape(x));
                let y = self.push_node(graph_id, Node::Expand { x: y, shape }).1;
                (x, y)
            }
            (0, _) if ry > 0 => {
                let shape = self.shape_class(graph_id, self.graphs[graph_id].shape(y));
                let x = self.push_node(graph_id, Node::Expand { x, shape }).1;
                (x, y)
            }
            _ => (x, y),
        };
        // After scalar broadcasting the two operands must already have the same
        // shape: any non-scalar broadcasting is performed upstream by
        // `Tensor::broadcast` (and the eager binary path must call it before
        // reaching here). `Node::Binary` in the kernelizer does NOT broadcast.
        // Shapes are symbolic `Vec<ClassId>`; compare their *concrete* dims
        // (unresolved/dynamic dims are `-1` and skipped) so that two operands
        // with the same concrete shape but distinct dim classes still compare
        // equal.
        let concrete = |s: &[ClassId]| -> Vec<Dim> {
            s.iter().map(|&d| self.graphs[graph_id].resolve_const(d).and_then(Constant::as_dim).unwrap_or(-1)).collect()
        };
        let sx = self.graphs[graph_id].shape(x);
        let sy = self.graphs[graph_id].shape(y);
        debug_assert_eq!(
            concrete(&sx),
            concrete(&sy),
            "binary operands must be broadcast to equal shapes before Node::Binary (broadcasting is performed upstream); got {sx:?} vs {sy:?}"
        );
        self.push_node(graph_id, Node::Binary { x, y, bop }).1
    }
}
