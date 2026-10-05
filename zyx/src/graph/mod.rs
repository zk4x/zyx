// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0
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
use std::sync::Arc;

use crate::{
    DType, Map, Set, ZyxError,
    backend::{Cmd, CmdQueue, Dev, LaunchArg, Placement, Plan, PlanDim, Pool, ProgramId, Shard},
    dtype::Constant,
    kernel::{BOp, IDX_T, Kernel, Op, OpId, ParamKind, TTOp},
    runtime::{KernelId, Runtime, TensorData},
    scalar::{bf16, f8e4m3, f8e5m2, f16},
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    symbolic::{Expr, ExprId},
    tensor::TensorId,
};

mod autograd;
mod kernelizer;

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

#[derive(Debug)]
pub(crate) struct OpNode {
    pub(crate) op: Op,
    pub(crate) class_of: OpId,
    /// Next node of the same e-class (intrusive chain), or `OpId::NULL` if
    /// this is the last variant. Chains preserve insertion order: a class's
    /// first node is its oldest, and later variants (e.g. lowered Kernel
    /// twins) are appended at the tail.
    pub(crate) next_in_class: OpId,
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
///   (`Op::Param` with data dtype) for a `Global` param, or a dim-variable
///   class (`Op::Param { dtype: IDX_T, shape: NULL }`) for a `Variable`
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
#[derive(Debug, Clone)]
pub struct JitKernelData {
    pub(crate) kernel: Kernel,
    pub(crate) outputs: Vec<OpId>,
    pub(crate) loads: Vec<OpId>,
    pub(crate) stores: Vec<OpId>,
}

#[derive(Debug)]
pub struct Graph {
    pub(crate) hashcons: Map<Op, OpId>,
    pub(crate) ops: Slab<OpId, OpNode>,
    pub(crate) jit_kernels: Slab<JitKernelId, JitKernelData>,
    pub(crate) leaf_classes: Vec<OpId>,
    pub(crate) leaf_map: Map<OpId, TensorId>,
    // Number of alive graph tensors (TensorState::Graph) referencing this graph.
    // Incremented at every graph-tensor birth, decremented when a tensor dies
    // (release), is eagerified, or is dropped.
    pub(crate) ref_count: u64,
    // Tape scope has ended (Tape::drop ran); no new ops may use this graph.
    // The graph is removed from the slab only when dead && ref_count == 0, which
    // guarantees no stale tensor ever observes a reused GraphId.
    pub(crate) dead: bool,
    /// Allocator for [`Op::Param`] `cons_id`s.
    pub(crate) max_cons_id: u32,
}

impl Graph {
    /// Cleanup - when graph is no longer needed, but cannot be dropped yet, so it's marked dead
    pub fn mark_dead(&mut self) {
        self.dead = true;
        self.hashcons = Map::default();
        self.ops = Slab::new();
        self.jit_kernels = Slab::new();
        self.leaf_map = Map::default();
    }

    pub fn new() -> Self {
        Self {
            hashcons: Map::default(),
            ops: Slab::new(),
            jit_kernels: Slab::new(),
            leaf_map: Map::default(),
            leaf_classes: Vec::new(),
            ref_count: 0,
            dead: false,
            max_cons_id: 0,
        }
    }

    pub fn push_op(&mut self, op: Op) -> OpId {
        if let Some(&nid) = self.hashcons.get(&op) {
            return self.ops[nid].class_of;
        }
        let nid = self.ops.push(OpNode { op: op.clone(), class_of: OpId::NULL, next_in_class: OpId::NULL });
        self.ops[nid].class_of = nid;
        self.hashcons.insert(op, nid);
        nid
    }

    /// Iterates a class's variant nodes in insertion order (oldest first) by
    /// walking the intrusive `next_in_class` chain.
    pub(crate) fn class_nodes(&self, cid: OpId) -> impl Iterator<Item = OpId> + '_ {
        let mut cur = cid;
        std::iter::from_fn(move || {
            if cur.is_null() {
                return None;
            }
            let nid = cur;
            cur = self.ops[cur].next_in_class;
            Some(nid)
        })
    }

    /// Appends a variant node to a class's intrusive chain (insertion order).
    pub(crate) fn class_push(&mut self, cid: OpId, nid: OpId) {
        debug_assert_eq!(self.ops[nid].next_in_class, OpId::NULL);
        let mut cur = cid;
        while !self.ops[cur].next_in_class.is_null() {
            cur = self.ops[cur].next_in_class;
        }
        self.ops[cur].next_in_class = nid;
    }

    /// Mints `node` as a new variant of `class_of`: pushes it into the nodes
    /// slab, registers it in the hashcons map and appends it to the class's
    /// intrusive chain.
    ///
    /// A node joins exactly one class (its `class_of`); additional outputs of
    /// multi-output nodes are referenced through the node's own fields, never
    /// through extra chain membership.
    pub(crate) fn mint_node(&mut self, node: Op, class_of: OpId) -> OpId {
        let nid = self.ops.push(OpNode { op: node.clone(), class_of, next_in_class: OpId::NULL });
        self.hashcons.insert(node, nid);
        self.class_push(class_of, nid);
        nid
    }

    pub fn is_leaf(&self, class_id: OpId) -> bool {
        self.class_nodes(class_id).any(|nid| matches!(&self.ops[nid].op, Op::Param { .. }))
    }

    /// Walks back through single-input movement nodes until reaching dst's base
    /// leaf class (a key of `leaf_map`). Used to find which leaf buffer an
    /// `After` class's store aliases.
    pub(crate) fn base_leaf(&self, mut c: OpId) -> OpId {
        loop {
            if self.leaf_map.contains_key(&c) {
                return c;
            }
            let mut next = None;
            for nid in self.class_nodes(c) {
                match &self.ops[nid].op {
                    Op::Reshape { x, .. }
                    | Op::Expand { x, .. }
                    | Op::Permute { x, .. }
                    | Op::Flip { x, .. }
                    | Op::Pad { x, .. }
                    | Op::Narrow { x, .. }
                    | Op::ToDevice { x, .. }
                    | Op::After { x, .. } => next = Some(*x),
                    _ => {}
                }
            }
            c = next.unwrap_or_else(|| panic!("assign dst class {c:?} must be a realized leaf or a movement chain over one"));
        }
    }

    /// Whether `class_id` is the output of an in-place `assign` — a class whose
    /// value lives in (aliases) dst's realized leaf buffer.
    pub fn is_after(&self, class_id: OpId) -> bool {
        self.class_nodes(class_id).any(|nid| matches!(&self.ops[nid].op, Op::After { .. }))
    }

    pub fn push_to_device(&mut self, x: OpId, device: Dev, time: u64) -> OpId {
        let node = Op::ToDevice { x, device, time };
        if let Some(&nid) = self.hashcons.get(&node) {
            return self.ops[nid].class_of;
        }
        let nid = self.ops.push(OpNode { op: node.clone(), class_of: OpId::NULL, next_in_class: OpId::NULL });
        self.ops[nid].class_of = nid;
        self.hashcons.insert(node, nid);
        nid
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
    /// Because `deps` prunes a boundary class's `parameters`, a boundary
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
        inputs: &Set<OpId>,
        outputs: &BTreeSet<OpId>,
        allowed: Option<&Set<OpId>>,
    ) -> Vec<OpId> {
        // Dead classes (unconsumed, not an output) are harmless: traversal
        // never reaches them, so they neither appear in `rcs` nor stall
        // anything. What must NEVER happen is a *reachable* class failing
        // to emit — that would mean its consumers' token accounting is
        // broken and everything depending on it silently drops out of the
        // order. Checked only in the global sort: region-restricted walks
        // legitimately cannot see consumers outside their boundary.
        let mut rcs: Map<OpId, u32> = Map::default();
        let mut stack: Vec<OpId> = outputs.iter().copied().collect();
        while let Some(cid) = stack.pop() {
            rcs.entry(cid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                let deps = self.deps::<WITHOUT_KERNELS>(inputs, cid);
                stack.extend(deps.into_iter().filter(|d| allowed.is_none_or(|a| a.contains(d))));
                1
            });
        }

        let mut order = Vec::new();
        let mut internal_rcs: Map<OpId, u32> = Map::default();
        let mut stack: Vec<OpId> = outputs.iter().copied().collect();
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
                let mut seen: Set<OpId> = Set::default();
                while let Some(c) = frontier.pop() {
                    if !seen.insert(c) {
                        continue;
                    }
                    let v = internal_rcs.get(&c).copied().unwrap_or(0);
                    let r = rcs.get(&c).copied().unwrap_or(0);
                    let types: Vec<String> = self
                        .class_nodes(c)
                        .map(|n| format!("{:?}", self.ops[n].op))
                        .map(|s| s.split(" OpId").next().unwrap_or(&s).to_string())
                        .collect();
                    report.push_str(&format!("\n  {c:?} rc={r} visited={v} types={types:?}"));
                    let mut parents: Set<OpId> = Set::default();
                    for (_, nd) in self.ops.iter() {
                        if nd.op.parameters().any(|q| q == c) && rcs.contains_key(&nd.class_of) {
                            parents.insert(nd.class_of);
                        }
                    }
                    for p in parents {
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
        let mut color: Map<OpId, u8> = Map::default();
        let mut parent: Map<OpId, OpId> = Map::default();
        for root in self.ops.iter().filter(|(id, nd)| nd.class_of == *id).map(|(id, _)| id) {
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
    /// inputs; otherwise every op's [`Op::parameters`] is used.
    fn deps<const WITHOUT_KERNELS: bool>(&self, inputs: &Set<OpId>, cid: OpId) -> Vec<OpId> {
        let mut deps = Vec::new();
        for nid in self.class_nodes(cid) {
            match self.ops[nid].op {
                Op::Kernel { inputs: kin, .. } => {
                    if WITHOUT_KERNELS && !inputs.contains(&cid) {
                        continue;
                    }
                    let Op::Stack { ref ops } = self.ops[kin].op else {
                        unreachable!()
                    };
                    for p in ops.iter() {
                        if !deps.contains(p) && !(WITHOUT_KERNELS && inputs.contains(p)) {
                            deps.push(*p);
                        }
                    }
                }
                Op::Custom(ref inner) => {
                    if WITHOUT_KERNELS && !inputs.contains(&cid) {
                        continue;
                    }
                    for p in inner.inputs.iter() {
                        if !deps.contains(p) && !(WITHOUT_KERNELS && inputs.contains(p)) {
                            deps.push(*p);
                        }
                    }
                }
                ref node => {
                    if WITHOUT_KERNELS && inputs.contains(&cid) {
                        continue;
                    }
                    for p in node.parameters() {
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
    fn extract_deps(&self, cid: OpId) -> Vec<OpId> {
        let mut kdeps: Vec<OpId> = Vec::new();
        for nid in self.class_nodes(cid) {
            match self.ops[nid].op {
                Op::Kernel { inputs, .. } => {
                    let Op::Stack { ref ops } = self.ops[inputs].op else {
                        unreachable!()
                    };
                    for p in ops.iter() {
                        if !kdeps.contains(p) {
                            kdeps.push(*p);
                        }
                    }
                }
                Op::ToDevice { x, .. } => {
                    if !kdeps.contains(&x) {
                        kdeps.push(x);
                    }
                }
                _ => {}
            }
        }
        if kdeps.is_empty() {
            return self.deps::<false>(&Set::default(), cid);
        }
        for nid in self.class_nodes(cid) {
            if let Op::After { x, dep } = &self.ops[nid].op {
                for p in [x, dep] {
                    if !kdeps.contains(p) {
                        kdeps.push(*p);
                    }
                }
            }
            if let Op::Store { dst, src } = &self.ops[nid].op {
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
    fn topo_sort_for_extract(&self, outputs: &BTreeSet<OpId>) -> Vec<OpId> {
        let mut rcs: Map<OpId, u32> = Map::default();
        let mut stack: Vec<OpId> = outputs.iter().copied().collect();
        while let Some(cid) = stack.pop() {
            rcs.entry(cid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                stack.extend(self.extract_deps(cid));
                1
            });
        }

        let mut order = Vec::new();
        let mut internal_rcs: Map<OpId, u32> = Map::default();
        let mut stack: Vec<OpId> = outputs.iter().copied().collect();
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
        for cid in self.ops.iter().filter(|(id, nd)| nd.class_of == *id).map(|(id, _)| id) {
            let shape_str = format!("{:?}", self.shape(cid));
            let dtype_str = format!("{:?}", self.dtype(cid));
            println!("Class {:?} shape={} dtype={}", cid, shape_str, dtype_str);
            for nid in self.class_nodes(cid) {
                let inputs: Vec<OpId> = self.ops[nid].op.parameters().collect();
                let name = match self.ops[nid].op {
                    Op::Reduce { rop, .. } => format!("Reduce {:?}", rop),
                    Op::Binary { bop, .. } => format!("Binary {:?}", bop),
                    Op::Store { .. } => "Store".into(),
                    Op::After { .. } => "After".into(),
                    Op::Unary { uop, .. } => format!("Unary {:?}", uop),
                    Op::Cast { dtype, .. } => format!("Cast {:?}", dtype),
                    Op::Bitcast { dtype, .. } => format!("Bitcast {:?}", dtype),
                    Op::Kernel { .. } => format!("Kernel"),
                    Op::Custom { .. } => format!("Custom"),
                    Op::Reshape { .. } => "Reshape".into(),
                    Op::Expand { .. } => "Expand".into(),
                    Op::Permute { .. } => "Permute".into(),
                    Op::Flip { .. } => "Flip".into(),
                    Op::Pad { .. } => "Pad".into(),
                    Op::Narrow { .. } => "Narrow".into(),
                    Op::Stack { ref ops } => format!("Stack {:?}", ops),
                    Op::Index { vec, idx } => format!("Index {vec:?}[{idx}]"),
                    Op::ToDevice { device, time, .. } => format!("ToDevice {:?} time={}", device, time),
                    Op::Contiguous { .. } => "Contiguous".into(),
                    Op::Const(v) => format!("Const {:?}", v),
                    Op::Param { dtype, .. } => format!("Param {:?}", dtype),
                    _ => todo!(),
                };
                println!("  {name} {nid:?}: inputs={inputs:?}");
            }
        }
        println!("{}\n", line);
    }

    /// After extraction, inserts [`Node::ToDevice`] transfers on the extracted
    /// path wherever a chosen kernel consumes a class placed on a different
    /// device, and returns the repaired node list in topological order.
    ///
    /// Only the extracted producer/consumer pairs are considered: a class may
    /// hold kernels on several devices (every fusion is autotuned on all of
    /// them), and no transfer is needed when extraction chose the same-device
    /// producer. A transfer is added only on a real mismatch between the
    /// consumer kernel's device and the placement of its input on the
    /// extracted path (chosen kernel output, chosen transfer output, or
    /// realized leaf buffer). User-inserted [`Node::ToDevice`] nodes are kept
    /// as-is and reused through hashconsing.
    pub fn add_memory_ops(&mut self, buffer_map: &Map<TensorId, Arc<Placement>>, chosen: &[OpId]) -> Vec<OpId> {
        // Pool each class lives in on the extracted path. Chosen kernel
        // outputs live in their kernel's pool, chosen transfers in their
        // target pool, realized leaves in their buffer pool. Variable leaves
        // have no buffer and no placement — they bind at launch.
        let mut pool_of: Map<OpId, Pool> = Map::default();
        for (&cid, &tid) in &self.leaf_map {
            if let Some(buf) = buffer_map.get(&tid) {
                let [Shard { pool, .. }] = &buf.shards[..] else {
                    todo!("multi-shard leaf in add_memory_ops")
                };
                pool_of.insert(cid, *pool);
            }
        }

        let mut repaired: Vec<OpId> = Vec::with_capacity(chosen.len());
        let mut emitted: Set<OpId> = Set::default();
        for &nid in chosen {
            let (device_id, inputs, class_of) = match &self.ops[nid].op {
                Op::Kernel { info, inputs, .. } => {
                    debug_assert_ne!(info.0.dev, Dev::Auto);
                    (info.0.dev, inputs.clone(), self.ops[nid].class_of)
                }
                Op::ToDevice { device, .. } => {
                    // Pool is always derived from the device, never the reverse.
                    pool_of.insert(self.ops[nid].class_of, device.pool());
                    if emitted.insert(nid) {
                        repaired.push(nid);
                    }
                    continue;
                }
                _ => unreachable!("add_memory_ops runs on extracted nodes, which are only Kernel/ToDevice"),
            };
            let dev_pool = device_id.pool();
            if let Op::Kernel { outputs, .. } = self.ops[nid].op {
                let Op::Stack { ref ops } = self.ops[outputs].op else {
                    unreachable!()
                };
                for &oc in ops {
                    pool_of.insert(oc, dev_pool);
                }
            }
            let Op::Stack { ref ops } = self.ops[inputs].op else {
                unreachable!("add_memory_ops: kernel inputs must be a Stack class, got {:?}", self.ops[inputs].op)
            };
            let inputs: Vec<OpId> = ops.to_vec();
            let mut new_inputs: Option<Vec<OpId>> = None;
            for (i, &input_cid) in inputs.iter().enumerate() {
                if pool_of.get(&input_cid) == Some(&dev_pool) {
                    continue;
                }
                if !pool_of.contains_key(&input_cid) {
                    // No buffer on the extracted path (variable leaf bound at
                    // launch) — nothing to transfer.
                    continue;
                }
                let to_cid = self.push_to_device(input_cid, device_id, 0);
                if to_cid != class_of {
                    let tnode = Op::ToDevice { x: input_cid, device: device_id, time: 0 };
                    let tnid = *self.hashcons.get(&tnode).expect("push_to_device just inserted the transfer");
                    pool_of.insert(to_cid, dev_pool);
                    if emitted.insert(tnid) {
                        repaired.push(tnid);
                    }
                    let new_inputs = new_inputs.get_or_insert_with(|| inputs.clone());
                    new_inputs[i] = to_cid;
                }
            }
            if let Some(new_inputs) = new_inputs {
                let stack = self.push_op(Op::Stack { ops: new_inputs.into_boxed_slice() });
                if let Op::Kernel { inputs: node_inputs, .. } = &mut self.ops[nid].op {
                    *node_inputs = stack;
                }
            }
            if emitted.insert(nid) {
                repaired.push(nid);
            }
        }
        repaired
    }

    /// Hash of the graph structure (hashcons), output classes, and the shape
    /// and dtype of every class. Deterministic across equivalent graphs — used
    /// as a cache key for compiled plans.
    ///
    /// Shape and dtype must be part of the key: two graphs with the same node
    /// structure but different shapes/dtypes (e.g. an `f32[10]` sin vs an
    /// `f32[3]` sin) would otherwise share a plan with wrong allocation sizes.
    #[must_use]
    pub fn cache_key(&self, outputs: &BTreeSet<OpId>) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        for (node, &id) in &self.hashcons {
            id.hash(&mut hasher);
            node.hash(&mut hasher);
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
    pub fn extract(&self, outputs: &BTreeSet<OpId>) -> Vec<OpId> {
        let order = self.topo_sort_for_extract(outputs);

        let n = self.ops.ids().count();
        let is_leaf: Vec<bool> = (0..n)
            .map(|i| {
                let cid = OpId(i as u32);
                self.class_nodes(cid).any(|nid| matches!(&self.ops[nid].op, Op::Param { .. }))
            })
            .collect();

        // Candidate producer nodes per class: Kernel and ToDevice nodes. Multiple
        // kernels may produce the same class (different fusions compete in
        // extraction); a leaf class is already realized and never needs one.
        #[derive(Clone, Copy)]
        struct Cand {
            nid: OpId,
            time: u64,
        }
        let mut cands: Vec<Vec<Cand>> = vec![Vec::new(); n];
        let nn = n;
        let mut node_in: Vec<Vec<OpId>> = vec![Vec::new(); nn];
        let mut node_out: Vec<Vec<OpId>> = vec![Vec::new(); nn];
        let mut node_time: Vec<u64> = vec![0; nn];
        for &cid in &order {
            for nid in self.class_nodes(cid) {
                let (time, inputs, outputs) = match &self.ops[nid].op {
                    Op::Kernel { inputs, outputs, info, .. } => {
                        let time = info.1;
                        let Op::Stack { ref ops } = self.ops[*inputs].op else {
                            unreachable!("extract: kernel inputs must be a Stack class, got {:?}", self.ops[*inputs].op)
                        };
                        let ins = ops.to_vec();
                        let Op::Stack { ref ops } = self.ops[*outputs].op else {
                            unreachable!("extract: kernel outputs must be a Stack class, got {:?}", self.ops[*outputs].op)
                        };
                        (time, ins, ops.to_vec())
                    }
                    Op::ToDevice { x, time, .. } => {
                        let outputs = vec![self.ops[nid].class_of];
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
        let mut after_chain: Vec<Vec<OpId>> = vec![Vec::new(); n];
        for &cid in &order {
            let mut chain = Vec::new();
            let mut cur = cid;
            while let Some(nid2) = self.class_nodes(cur).find(|&nid| matches!(&self.ops[nid].op, Op::After { .. })) {
                let Op::After { x, dep } = &self.ops[nid2].op else {
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
            outputs: &'a BTreeSet<OpId>,
            order: &'a [OpId],
            cands: &'a [Vec<Cand>],
            node_in: &'a [Vec<OpId>],
            node_out: &'a [Vec<OpId>],
            node_time: &'a [u64],
            after_chain: &'a [Vec<OpId>],
            is_leaf: &'a [bool],
        }

        impl Ctx<'_> {
            /// The classes that still must be produced (`pending`, in topological
            /// order) and the classes already produced, derived from `selected`.
            fn pending_and_produced(&self, selected: &Set<OpId>) -> (Vec<OpId>, Set<OpId>) {
                let mut produced: Set<OpId> = Set::default();
                let mut requested: Set<OpId> = self.outputs.iter().copied().collect();
                for &nid in selected {
                    for &o in &self.node_out[nid.0 as usize] {
                        produced.insert(o);
                    }
                    for &i in &self.node_in[nid.0 as usize] {
                        requested.insert(i);
                    }
                }
                loop {
                    let mut add: Vec<OpId> = Vec::new();
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

            fn plan_cost(&self, selected: &Set<OpId>) -> u64 {
                selected.iter().map(|&nid| self.node_time[nid.0 as usize]).sum()
            }

            /// A feasible plan that selects the cheapest producer of each pending
            /// class in topological order. Always terminates; provides the upper
            /// bound for the search and a safe fallback.
            fn greedy(&self) -> Set<OpId> {
                let mut selected: Set<OpId> = Set::default();
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
            fn search(&self, selected: &mut Set<OpId>, cost: u64, best: &mut u64) -> Option<(u64, Vec<OpId>)> {
                let (pending, _) = self.pending_and_produced(selected);
                if pending.is_empty() {
                    return Some((0, Vec::new()));
                }
                let c = pending[0];
                let mut ordered: Vec<&Cand> = self.cands[c.0 as usize].iter().collect();
                ordered.sort_by_key(|k| k.time);
                let mut best_res: Option<(u64, Vec<OpId>)> = None;
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
        let mut producer: Vec<Option<OpId>> = vec![None; n];
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
        let mut stack: Vec<OpId> = outputs.iter().copied().collect();
        loop {
            while let Some(cid) = stack.pop() {
                if !needed[cid.0 as usize] {
                    needed[cid.0 as usize] = true;
                    if let Some(nid) = producer[cid.0 as usize] {
                        match &self.ops[nid].op {
                            Op::Kernel { inputs, .. } => {
                                let Op::Stack { ref ops } = self.ops[*inputs].op else {
                                    unreachable!("extract: kernel inputs must be a Stack class, got {:?}", self.ops[*inputs].op)
                                };
                                stack.extend(ops.iter().copied())
                            }
                            Op::ToDevice { x, .. } => stack.push(*x),
                            _ => {}
                        }
                    }
                    // After classes alias their base leaf buffer, and their
                    // value comes from dep (the assign) writing over x's
                    // version of that buffer. If the post-assign value is
                    // needed, the assign that wrote it and every earlier
                    // After in the chain are needed too — otherwise extract
                    // drops the in-place store kernels of chained assigns.
                    if let Some(nid2) = self.class_nodes(cid).find(|&nid| matches!(&self.ops[nid].op, Op::After { .. }))
                        && let Op::After { x, dep } = &self.ops[nid2].op
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
            let mut add: Vec<OpId> = Vec::new();
            for &cid in &order {
                if !needed[cid.0 as usize]
                    && self
                        .class_nodes(cid)
                        .any(|nid| matches!(&self.ops[nid].op, Op::Store { dst, .. } if needed[self.base_leaf(*dst).0 as usize]))
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
        let mut seen: Set<OpId> = Set::default();
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

    pub fn rank(&self, class: OpId) -> UAxis {
        self.shape(class).len() as UAxis
    }

    /// Shape of a class as dim classes:
    /// each element is a class evaluating to a dimension value — a `Const`
    /// for static dims or a symbolic dim leaf otherwise. Empty vec for
    /// scalars.
    pub fn shape(&self, class: OpId) -> Vec<OpId> {
        match &self.ops[class].op {
            Op::Const(_) | Op::Stack { .. } => Vec::new(),
            Op::Index { vec, idx } => match &self.ops[*vec].op {
                Op::Stack { ops } => self.shape(ops[*idx]),
                // Projection of a multi-output kernel: the shape metadata of
                // output `idx` lives in the Custom node's descriptor.
                Op::Custom(inner) => self.dims(inner.outputs[*idx].1),
                n => panic!("Index vec must be a Stack or Custom class, got {n:?}"),
            },
            Op::Param { shape, .. } => self.dims(*shape),
            Op::Expand { shape, .. } | Op::Reshape { shape, .. } => self.dims(*shape),
            Op::Permute { x, axes } => {
                let s = self.shape(*x);
                axes.iter().map(|&a| s[a as usize]).collect()
            }
            Op::Pad { x, axis, len, .. } => {
                let mut s = self.shape(*x);
                s[*axis as usize] = *len;
                s
            }
            Op::Narrow { x, axis, len, .. } => {
                let mut s = self.shape(*x);
                s[*axis as usize] = *len;
                s
            }
            Op::Flip { x, .. }
            | Op::Cast { x, .. }
            | Op::Bitcast { x, .. }
            | Op::Unary { x, .. }
            | Op::After { x, .. }
            | Op::ToDevice { x, .. }
            | Op::Contiguous { x } => self.shape(*x),
            // Scalars broadcast implicitly (see `push_binary_node`): the
            // result takes the shape of the non-scalar operand. Both scalars
            // → rank 0.
            Op::Binary { x, y, .. } => {
                let sx = self.shape(*x);
                if !sx.is_empty() { sx } else { self.shape(*y) }
            }
            Op::Reduce { x, reduce_axis, .. } => {
                let mut s = self.shape(*x);
                debug_assert!(
                    reduce_axis.is_null() || *s.last().expect("Reduce of scalar") == *reduce_axis,
                    "Reduce axis must be trailing"
                );
                s.pop().expect("Reduce of scalar");
                s
            }
            Op::Store { dst, .. } => self.shape(*dst),
            Op::Kernel { outputs, .. } => {
                let Op::Stack { ref ops } = self.ops[*outputs].op else {
                    unreachable!("shape: kernel outputs must be a Stack class, got {:?}", self.ops[*outputs].op)
                };
                self.shape(ops[0])
            }
            // A Custom node is a member of every one of its output classes, so
            // the queried class selects the matching output's shape metadata.
            Op::Custom(inner) => {
                let (_, shape, _) =
                    inner.outputs.iter().find(|(c, _, _)| *c == class).expect("Custom node queried outside its output classes");
                self.dims(*shape)
            }
            Op::Storage { .. }
            | Op::GEP { .. }
            | Op::Load { .. }
            | Op::Copy { .. }
            | Op::Range { .. }
            | Op::Loop { .. }
            | Op::EndLoop
            | Op::Barrier
            | Op::Wmma { .. }
            | Op::TT { .. }
            | Op::Asm { .. } => unreachable!("shape: kernel-internal op never appears in graph classes"),
        }
    }

    /// Interpret a shape class: `NULL` is `[]`, a `Stack` of dim classes is
    /// its ops, anything else is a single bare dim class (rank-1 convention).
    pub fn dims(&self, shape: OpId) -> Vec<OpId> {
        if shape.is_null() {
            return Vec::new();
        }
        match &self.ops[shape].op {
            Op::Stack { ops } => ops.to_vec(),
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
    /// - Dim variables are `Op::Param { dtype: IDX_T, shape: NULL }` classes;
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
    /// kernel.
    pub(crate) fn replay_symbolic_into_kernel(&mut self, kid: JitKernelId, dims: &[OpId]) -> OpId {
        // Post-order flatten: every class lands after its operands, so one
        // flat pass emits with operands already mapped.
        fn flatten(graph: &Graph, cid: OpId, order: &mut Vec<OpId>) {
            debug_assert!(graph.class_nodes(cid).count() == 1, "symbolic dim class must have exactly one node");
            let node = &graph.ops[cid].op;
            match node {
                Op::Const(_) | Op::Param { .. } => (),
                Op::Cast { x, .. } | Op::Unary { x, .. } => flatten(graph, *x, order),
                Op::Binary { x, y, .. } => {
                    flatten(graph, *x, order);
                    flatten(graph, *y, order);
                }
                Op::Stack { ops } => {
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

        let mut class_map: Map<OpId, OpId> = Map::default();
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
                debug_assert!(self.class_nodes(c).count() == 1, "symbolic dim class must have exactly one node");
                let node = self.ops[c].op.clone();
                let op_id = match node {
                    Op::Const(value) => self.jit_kernels[kid].kernel.push_back(Op::Const(value)),
                    Op::Param { dtype, shape, .. } => {
                        debug_assert!(shape.is_null(), "dim-variable leaf must be scalar, got shape {:?}", shape);
                        debug_assert!(dtype == IDX_T, "dim-variable leaf must be {:?}-typed, got {:?}", IDX_T, dtype);
                        let op_id = self.jit_kernels[kid].kernel.variable(IDX_T);
                        self.jit_kernels[kid].loads.push(c);
                        op_id
                    }
                    Op::Cast { x, dtype } => {
                        let a = class_map[&x];
                        self.jit_kernels[kid].kernel.cast(a, dtype)
                    }
                    Op::Unary { x, uop } => {
                        let a = class_map[&x];
                        self.jit_kernels[kid].kernel.unary(a, uop)
                    }
                    Op::Binary { x, y, bop } => {
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
    pub(crate) fn replay_shape_into_kernel(&mut self, kid: JitKernelId, shape: OpId) -> OpId {
        if shape.is_null() {
            return OpId::NULL;
        }
        match &self.ops[shape].op {
            Op::Stack { ops } => {
                let ops: Vec<OpId> = ops.iter().copied().collect();
                self.replay_symbolic_into_kernel(kid, &ops)
            }
            _ => self.replay_symbolic_into_kernel(kid, &[shape]),
        }
    }

    pub fn dtype(&self, class: OpId) -> DType {
        match &self.ops[class].op {
            Op::Const(c) => c.dtype(),
            Op::Index { vec, idx } => match &self.ops[*vec].op {
                Op::Stack { ops } => self.dtype(ops[*idx]),
                // Projection of a multi-output kernel: the dtype of output
                // `idx` lives in the Custom node's descriptor.
                Op::Custom(inner) => inner.outputs[*idx].2,
                n => panic!("Index vec must be a Stack or Custom class, got {n:?}"),
            },
            Op::Param { dtype, .. } => *dtype,
            Op::Cast { dtype, .. } => *dtype,
            Op::Bitcast { dtype, .. } => *dtype,
            Op::Store { dst, .. } => self.dtype(*dst),
            Op::Kernel { outputs, .. } => {
                let Op::Stack { ref ops } = self.ops[*outputs].op else {
                    unreachable!("dtype: kernel outputs must be a Stack class, got {:?}", self.ops[*outputs].op)
                };
                self.dtype(ops[0])
            }
            Op::Custom(inner) => {
                let (_, _, dtype) =
                    inner.outputs.iter().find(|(c, ..)| *c == class).expect("Custom node queried outside its output classes");
                *dtype
            }
            Op::Stack { ops } => self.dtype(ops[0]),
            Op::Expand { x, .. }
            | Op::Permute { x, .. }
            | Op::Reshape { x, .. }
            | Op::Pad { x, .. }
            | Op::Flip { x, .. }
            | Op::Narrow { x, .. }
            | Op::Reduce { x, .. }
            | Op::Unary { x, .. }
            | Op::After { x, .. }
            | Op::ToDevice { x, .. }
            | Op::Contiguous { x }
            | Op::Binary { x, .. } => self.dtype(*x),
            Op::Storage { .. }
            | Op::GEP { .. }
            | Op::Load { .. }
            | Op::Copy { .. }
            | Op::Range { .. }
            | Op::Loop { .. }
            | Op::EndLoop
            | Op::Barrier
            | Op::Wmma { .. }
            | Op::TT { .. }
            | Op::Asm { .. } => unreachable!("dtype: kernel-internal op never appears in graph classes"),
        }
    }

    /// Tries to resolve the value of a scalar class by walking its const
    /// expression: `Const` leaves evaluated through `Cast`, `Unary` and
    /// `Binary` nodes (iteratively, no recursion). Returns `None` if the
    /// class is not a scalar, the walk exceeds 10 000 steps, or any leaf
    /// is not a `Const`.
    pub(crate) fn resolve_const(&self, class: OpId) -> Option<Constant> {
        // Preorder of the const-expression subgraph reachable through
        // `Cast`, `Unary` and `Binary`; non-const leaves abort.
        let mut visited: Set<OpId> = Set::default();
        let mut order: Vec<OpId> = Vec::new();
        let mut stack = vec![class];
        for _ in 0..10_000 {
            let Some(id) = stack.pop() else { break };
            let node_id = id;
            if !visited.insert(node_id) {
                continue;
            }
            match &self.ops[node_id].op {
                Op::Cast { x, .. } => stack.push(*x),
                Op::Bitcast { x, .. } => stack.push(*x),
                Op::Unary { x, .. } => stack.push(*x),
                Op::Binary { x, y, .. } => {
                    stack.push(*y);
                    stack.push(*x);
                }
                Op::Index { vec, idx } => match &self.ops[*vec].op {
                    Op::Stack { ops } => stack.push(ops[*idx]),
                    _ => return None,
                },
                Op::Const(_) => {}
                // Every other variant is a non-scalar / dynamic leaf: not
                // resolvable to a constant.
                Op::Param { .. }
                | Op::Expand { .. }
                | Op::Permute { .. }
                | Op::Reshape { .. }
                | Op::Pad { .. }
                | Op::Flip { .. }
                | Op::Narrow { .. }
                | Op::Stack { .. }
                | Op::Reduce { .. }
                | Op::Storage { .. }
                | Op::GEP { .. }
                | Op::Load { .. }
                | Op::Copy { .. }
                | Op::Store { .. }
                | Op::Range { .. }
                | Op::Loop { .. }
                | Op::EndLoop
                | Op::Barrier
                | Op::Wmma { .. }
                | Op::TT(TTOp::ReduceTile { .. })
                | Op::TT(TTOp::MatmulTile { .. })
                | Op::TT(TTOp::TransposeTile { .. })
                | Op::TT(TTOp::BroadcastTile { .. })
                | Op::TT(TTOp::ReserveBack { .. })
                | Op::TT(TTOp::PushBack { .. })
                | Op::TT(TTOp::WaitFront { .. })
                | Op::TT(TTOp::PopFront { .. })
                | Op::TT(TTOp::MathLock)
                | Op::TT(TTOp::MathUnlock)
                | Op::TT(TTOp::PackLock)
                | Op::TT(TTOp::PackUnlock)
                | Op::TT(TTOp::NocReadBarrier)
                | Op::TT(TTOp::NocWriteBarrier)
                | Op::TT(TTOp::ReduceUninit)
                | Op::TT(TTOp::EndReader)
                | Op::TT(TTOp::EndCompute)
                | Op::TT(TTOp::LLK { .. })
                | Op::TT(TTOp::LLKReduce { .. })
                | Op::TT(TTOp::LLKBcast { .. })
                | Op::Asm { .. }
                | Op::After { .. }
                | Op::ToDevice { .. }
                | Op::Contiguous { .. }
                | Op::Kernel { .. }
                | Op::Custom { .. } => return None,
            }
            order.push(node_id);
        }
        if !stack.is_empty() {
            panic!("resolve_const did not finish in 10000 steps");
        }
        // Evaluate bottom-up: `order` is a preorder (parents before their
        // operands), so reversing it evaluates every operand before its
        // consumer.
        let mut values: Map<OpId, Constant> = Map::default();
        for &node_id in order.iter().rev() {
            let value = match &self.ops[node_id].op {
                Op::Const(value) => *value,
                Op::Cast { x, dtype } => values[x].cast(*dtype),
                Op::Bitcast { x, dtype } => values[x].bitcast(*dtype),
                Op::Unary { x, uop } => values[x].unary(*uop),
                Op::Index { vec, idx } => match &self.ops[*vec].op {
                    Op::Stack { ops } => values[&ops[*idx]].clone(),
                    n => unreachable!("Index vec must be a Stack class, got {n:?}"),
                },
                Op::Binary { x, y, bop } => Constant::binary(values[x], values[y], *bop),
                _ => unreachable!("non-expression node in const walk"),
            };
            values.insert(node_id, value);
        }
        Some(values[&class])
    }
}

impl Runtime {
    pub fn promote_to_graph(&mut self, tid: TensorId, graph_id: GraphId) -> Result<OpId, ZyxError> {
        let (class_id, gid) = match self.tensors[tid] {
            TensorData::Graph { class_id, graph_id, .. } | TensorData::Promoted { class_id, graph_id, .. } => {
                (class_id, graph_id)
            }
            _ => (OpId::NULL, GraphId::NULL),
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
                    self.tensors[tid] = TensorData::Eager { kernel_id, op_id, shape_id, dtype, rc };
                }
                ref t => panic!("promote_to_graph: dead-graph tensor {tid} has no eager side to revert to: {t:?}"),
            }
            self.graphs[gid].ref_count -= 1;
            if self.graphs[gid].dead && self.graphs[gid].ref_count == 0 {
                self.remove_dead_graph(gid);
            }
        }

        // A **Leaf** is a buffer-backed value with no kernel. It promotes as a
        // pure leaf: its shape class is replayed from the slab-side shape
        // expression, the class binds to the tid via `leaf_map` (the plan
        // reads its buffer), and the tensor becomes `TensorData::GraphLeaf` —
        // affiliated with the graph (ref_count + rc incremented), so
        // `Tape::drop`'s visit loop handles it; the buffer stays on the
        // variant (a graph leaf is a leaf) and the drop/eagerify arms revert
        // buffer-backed graph tensors back to `Leaf` (the value is preserved,
        // not tombstoned).
        if matches!(self.tensors[tid], TensorData::Leaf { .. }) {
            let (shape_id, dtype, rc) = match self.tensors[tid] {
                TensorData::Leaf { shape_id, dtype, rc, .. } => (shape_id, dtype, rc),
                ref t => unreachable!("{t:?}"),
            };
            let buffer = self.leaf_buffer(tid).expect("promote_to_graph: Leaf {tid} has no buffer (pending store not realized)");
            debug_assert!(
                self.leaf_buffer(tid).is_some(),
                "promote_to_graph: Leaf {tid} has no buffer (pending store not realized)"
            );
            let shape_class = if shape_id.is_scalar() {
                OpId::NULL
            } else {
                // replay_symbolic_into_graph takes a TensorId handle: mint a
                // transient Symbolic handle for the shape expression and
                // release it after the replay (the slab expr is append-only;
                // only the handle has a refcount).
                let shape_tid = self.tensors.push(TensorData::Symbolic { expr: shape_id, rc: 1 });
                let shape_class = self.replay_symbolic_into_graph(graph_id, shape_tid);
                self.release(shape_tid);
                shape_class
            };
            let (_, class_id) = self.push_leaf_node(graph_id, dtype, shape_class);
            self.graphs[graph_id].leaf_map.insert(class_id, tid);
            self.retain(tid);
            self.graphs[graph_id].leaf_classes.push(class_id);
            self.graphs[graph_id].ref_count += 1;
            self.tensors[tid] = TensorData::GraphLeaf { class_id, graph_id, shape_id, dtype, rc: rc + 1, buffer };
            return Ok(class_id);
        }

        // Pure-slab symbolic tensors (dim expressions: constants, variables,
        // dim arithmetic, shape stacks) have no eager kernel. They promote by
        // replaying the slab expression into the egraph
        // (`replay_symbolic_into_graph`): constants become Const classes,
        // variables become IDX_T leaf inputs (registered in `leaf_map` by the
        // replay itself), and arithmetic replays as graph nodes. No buffer is
        // involved. NOTE: a tape dropped without `realize` cannot revert
        // these to their slab state — the drop arm panics loudly for them.
        if matches!(self.tensors[tid], TensorData::Symbolic { .. }) {
            let (expr, dtype, rc) = match self.tensors[tid] {
                TensorData::Symbolic { expr, rc, .. } => (expr, self.dtype(tid), rc),
                ref t => unreachable!("{t:?}"),
            };
            // Rank: dim exprs are scalars; shape stacks are rank-1 with one
            // dim per element. A bare const is a valid 1d shape expression.
            let rank = match &self.exprs[expr] {
                Expr::Stack { exprs } => exprs.len(),
                Expr::Stack2 { .. } => 2,
                Expr::Stack3 { .. } => 3,
                Expr::Stack4 { .. } => 4,
                Expr::Stack5 { .. } => 5,
                _ => 0,
            };
            let class_id = self.replay_symbolic_into_graph(graph_id, tid);
            let shape_id = if rank == 0 {
                ExprId::SCALAR
            } else {
                let stacked = self.new_constant_tensor(Constant::idx(rank as i64));
                let shape_expr = match self.tensors[stacked] {
                    TensorData::Symbolic { expr, .. } => expr,
                    ref t => panic!("promote_to_graph: shape tid {stacked} is not symbolic: {t:?}"),
                };
                self.release(stacked);
                shape_expr
            };
            self.graphs[graph_id].ref_count += 1;
            self.tensors[tid] = TensorData::Graph { class_id, graph_id, shape_id, dtype, rc: rc + 1 };
            return Ok(class_id);
        }

        let (kernel_id, my_op_id) = match self.tensors[tid] {
            TensorData::Eager { kernel_id, op_id, .. } | TensorData::Promoted { kernel_id, op_id, .. } => (kernel_id, op_id),
            // A NULL-ids Graph variant is the tombstone left by `Tape::drop`
            // for a graph tensor that still had a user handle when its tape
            // died without `realize`. Its value was never computed and cannot
            // be recomputed (the graph is gone), so it can never be promoted
            // into a new tape. This is intended behaviour, not a bug.
            TensorData::Graph { class_id: OpId::NULL, .. } => panic!(
                "tensor {tid} is bound to a tape that was dropped without `Tape::realize`: its \
                 graph is gone, the value was never computed and cannot be recomputed, so the \
                 tensor is permanently invalid.\n\
                 This is a caller mistake, not a zyx bug: a tape must be realized before it is \
                 dropped or consumed.\n\
                 How to fix: end every tape scope with `tape.realize(outputs)?` (the training-loop \
                 pattern: tape.gradient → optim.update → tape.realize(params)), and do not keep \
                 using tensors traced by a tape after it is gone — rebuild the computation inside \
                 a fresh tape instead."
            ),
            ref t => panic!("promote_to_graph: tensor {tid} has no eager kernel: {t:?}"),
        };
        debug_assert!(
            self.kernels[kernel_id].outputs.contains(&tid),
            "promote_to_graph: tensor {tid} kernel {kernel_id:?} not in outputs"
        );

        // Already realized eager tensors promote to the graph as leaves directly.
        // Their buffer is read by the plan as an input; the value is preserved and
        // not recomputed. The eager kernel is left untouched (rc/outputs already
        // count the handles), so the tensor reverts to eager when the graph dies.
        if self.leaf_buffer(tid).is_some() {
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
            // `loads` is parallel to the kernel's Global|Variable Params in
            // head order (see `Kernel::duplicate_subkernel`); map each such
            // Param op to its load index so shape-stack variable dims can be
            // resolved to their tensors.
            let loads = self.kernels[kernel_id].loads.clone();
            let mut load_of_param: Map<OpId, usize> = Map::default();
            let mut load_idx = 0;
            let mut p = self.kernels[kernel_id].kernel.head;
            while !p.is_null() {
                if let Op::Param { kind: ParamKind::Global | ParamKind::Variable, .. } = self.kernels[kernel_id].kernel.ops[p].op
                {
                    load_of_param.insert(p, load_idx);
                    load_idx += 1;
                }
                p = self.kernels[kernel_id].kernel.next_op(p);
            }
            let mut dim_classes = Vec::with_capacity(dim_entries.len());
            for entry in dim_entries {
                dim_classes.push(match self.kernels[kernel_id].kernel.ops[entry].op {
                    Op::Const(c) => self.push_const(graph_id, c),
                    Op::Param { kind: ParamKind::Variable, .. } => {
                        // A variable dim is an input, not structure: register
                        // its leaf so the plan binds it via the tensors slab
                        // and value changes never force recompilation.
                        let var_tid = loads[load_of_param[&entry]];
                        debug_assert!(
                            matches!(self.tensors[var_tid], TensorData::Symbolic { expr, .. } if matches!(self.exprs[expr], Expr::Variable { .. })),
                            "promote_to_graph: dim variable {var_tid} is not a symbolic variable"
                        );
                        let (_, dim_cid) = self.push_leaf_node(graph_id, IDX_T, OpId::NULL);
                        self.graphs[graph_id].leaf_map.insert(dim_cid, var_tid);
                        self.retain(var_tid);
                        self.graphs[graph_id].leaf_classes.push(dim_cid);
                        self.graphs[graph_id].ref_count += 1;
                        dim_cid
                    }
                    ref op => unreachable!("promote_to_graph: dim op {op:?} in param shape stack"),
                });
            }
            let shape_class = match dim_classes.len() {
                0 => OpId::NULL,
                1 => dim_classes[0],
                _ => self.push_op(graph_id, Op::Stack { ops: dim_classes.into_boxed_slice() }),
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
                        TensorData::Eager { kernel_id, op_id, shape_id, rc, dtype } => (kernel_id, op_id, shape_id, rc, dtype),
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
                    Op::Storage { .. } | Op::Const(_) => {}
                    Op::Param { shape, .. } => {
                        // The Param's shape stack is part of its structure:
                        // dim expressions feeding it must be replayed too.
                        if !shape.is_null() {
                            stack.push(*shape);
                        }
                    }
                    Op::Unary { x, .. } => stack.push(*x),
                    Op::Binary { x, y, .. } => {
                        stack.push(*x);
                        stack.push(*y);
                    }
                    Op::Cast { x, .. } => stack.push(*x),
                    Op::Bitcast { x, .. } => stack.push(*x),
                    Op::Reduce { x, .. } => stack.push(*x),
                    Op::Reshape { x, shape } | Op::Expand { x, shape } => {
                        stack.push(*x);
                        stack.push(*shape);
                    }
                    Op::Pad { x, lp, len, .. } => {
                        stack.push(*x);
                        stack.push(*lp);
                        stack.push(*len);
                    }
                    Op::Narrow { x, start, len, .. } => {
                        stack.push(*x);
                        stack.push(*start);
                        stack.push(*len);
                    }
                    Op::Permute { x, .. } | Op::Flip { x, .. } => {
                        stack.push(*x);
                    }
                    Op::Stack { ops } => stack.extend(ops.iter().copied()),
                    Op::Store { dst, src } => {
                        stack.push(*dst);
                        stack.push(*src);
                    }
                    Op::TT(TTOp::ReduceTile { x, scaler, acc, .. }) => {
                        stack.push(*x);
                        stack.push(*scaler);
                        stack.push(*acc);
                    }
                    Op::EndLoop
                    | Op::Barrier
                    | Op::Range { .. }
                    | Op::Loop { .. }
                    | Op::GEP { .. }
                    | Op::Load { .. }
                    | Op::Copy { .. }
                    | Op::Asm { .. }
                    | Op::Index { .. }
                    | Op::Wmma { .. }
                    | Op::TT { .. }
                    | Op::After { .. }
                    | Op::ToDevice { .. }
                    | Op::Contiguous { .. }
                    | Op::Kernel { .. }
                    | Op::Custom(_) => {
                        unreachable!("promote_to_graph: eager kernel op {oid:?}")
                    }
                }
            }
            relevant
        };

        let loads = self.kernels[kernel_id].loads.clone();
        // Map each Global|Variable Param op to its index in `loads` (parallel
        // lists in head order, see `Kernel::duplicate_subkernel`) so shape-stack
        // variable dims resolve to their tensors.
        let mut load_of_param: Map<OpId, usize> = Map::default();
        let mut load_idx = 0;
        let mut p = self.kernels[kernel_id].kernel.head;
        while !p.is_null() {
            if let Op::Param { kind: ParamKind::Global | ParamKind::Variable, .. } = self.kernels[kernel_id].kernel.ops[p].op {
                load_of_param.insert(p, load_idx);
                load_idx += 1;
            }
            p = self.kernels[kernel_id].kernel.next_op(p);
        }
        let mut op_to_class: Map<OpId, OpId> = Map::default();
        let mut op_id = self.kernels[kernel_id].kernel.head;
        while !op_id.is_null() {
            if relevant.contains(&op_id) {
                let class_id = match self.kernels[kernel_id].kernel.ops[op_id].op {
                    Op::Param { shape, dtype, .. } => {
                        let load_tid = loads[load_of_param[&op_id]];
                        if self.leaf_buffer(load_tid).is_none() {
                            // Loads without a buffer are pending: the producer
                            // is recorded on the tensor. A `Variable` scalar
                            // has no buffer and no producer — its value comes
                            // from the variable slots at launch; it is
                            // registered as a leaf below.
                            let pending = match &self.tensors[load_tid] {
                                TensorData::PendingLeaf { depends_on, .. } => *depends_on,
                                TensorData::Eager { .. }
                                | TensorData::Graph { .. }
                                | TensorData::Promoted { .. }
                                | TensorData::Symbolic { .. } => KernelId::NULL,
                                ref t => panic!("promote_to_graph: load tid {load_tid} is not a kernel tensor: {t:?}"),
                            };
                            if !pending.is_null() {
                                let outputs: Vec<TensorId> = self.kernels[pending].outputs.iter().copied().collect();
                                for &otid in &outputs {
                                    self.add_store(otid)?;
                                }
                            }
                        }

                        let load_is_leaf = match &self.tensors[load_tid] {
                            TensorData::Graph { class_id: c, graph_id: g, .. }
                            | TensorData::GraphLeaf { class_id: c, graph_id: g, .. }
                            | TensorData::Promoted { class_id: c, graph_id: g, .. } => {
                                !c.is_null() && *g == graph_id && !self.graphs[graph_id].dead
                            }
                            _ => false,
                        };
                        if load_is_leaf {
                            // load_tid is already a leaf of this graph: reuse its class.
                            match &self.tensors[load_tid] {
                                TensorData::Graph { class_id: c, .. }
                                | TensorData::GraphLeaf { class_id: c, .. }
                                | TensorData::Promoted { class_id: c, .. } => *c,
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
                                        // A variable dim is an input, not structure: register
                                        // its leaf so the plan binds it via the tensors slab
                                        // and value changes never force recompilation.
                                        let var_tid = loads[load_of_param[&entry]];
                                        debug_assert!(
                                            matches!(self.tensors[var_tid], TensorData::Symbolic { expr, .. } if matches!(self.exprs[expr], Expr::Variable { .. })),
                                            "promote_to_graph: dim variable {var_tid} is not a symbolic variable"
                                        );
                                        let (_, dim_cid) = self.push_leaf_node(graph_id, IDX_T, OpId::NULL);
                                        self.graphs[graph_id].leaf_map.insert(dim_cid, var_tid);
                                        self.retain(var_tid);
                                        self.graphs[graph_id].leaf_classes.push(dim_cid);
                                        self.graphs[graph_id].ref_count += 1;
                                        dim_cid
                                    }
                                    // Computed dim expressions (rope half, mean
                                    // divisor, ...) are parameters of the Param
                                    // op, so the replay above already mapped
                                    // them to graph classes — just reuse.
                                    Op::Binary { .. } | Op::Unary { .. } | Op::Cast { .. } | Op::Stack { .. } => {
                                        op_to_class[&entry]
                                    }
                                    Op::Bitcast { .. } => {
                                        unreachable!("promote_to_graph: bitcast in param shape stack")
                                    }
                                    Op::Param { kind: ParamKind::Global, .. } | Op::Param { kind: ParamKind::GlobalMut, .. } => {
                                        unreachable!("promote_to_graph: buffer param as dim in param shape stack")
                                    }
                                    Op::Storage { .. }
                                    | Op::EndLoop
                                    | Op::Barrier
                                    | Op::Range { .. }
                                    | Op::Loop { .. }
                                    | Op::Reshape { .. }
                                    | Op::Expand { .. }
                                    | Op::Permute { .. }
                                    | Op::Flip { .. }
                                    | Op::Pad { .. }
                                    | Op::Narrow { .. }
                                    | Op::Reduce { .. }
                                    | Op::Store { .. }
                                    | Op::GEP { .. }
                                    | Op::Load { .. }
                                    | Op::Copy { .. }
                                    | Op::Asm { .. }
                                    | Op::Index { .. }
                                    | Op::Wmma { .. }
                                    | Op::TT { .. }
                                    | Op::After { .. }
                                    | Op::ToDevice { .. }
                                    | Op::Contiguous { .. }
                                    | Op::Kernel { .. }
                                    | Op::Custom(_) => {
                                        unreachable!("promote_to_graph: dim op {entry:?} in param shape stack")
                                    }
                                });
                            }
                            let shape_class = match dim_classes.len() {
                                0 => OpId::NULL,
                                1 => dim_classes[0],
                                _ => self.push_op(graph_id, Op::Stack { ops: dim_classes.into_boxed_slice() }),
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
                                        TensorData::Eager { kernel_id, op_id, shape_id, rc, dtype } => {
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
                                TensorData::Symbolic { expr, .. } => {
                                    let expr = *expr;
                                    if !matches!(self.exprs[expr], Expr::Variable { .. }) {
                                        panic!(
                                            "promote_to_graph: symbolic load {load_tid} is not a variable: {:?}",
                                            self.exprs[expr]
                                        );
                                    }
                                    // A scalar variable stays `Symbolic`: its
                                    // value is bound at launch — `leaf_map`
                                    // binds the leaf class to this tid,
                                    // nothing else to attach.
                                }
                                TensorData::Leaf { shape_id, dtype, rc, buffer, .. } => {
                                    // A Leaf load becomes a **GraphLeaf**:
                                    // affiliated (ref_count + this rc edge),
                                    // buffer carried on the variant (a graph
                                    // leaf is a leaf), class bound via
                                    // `leaf_map`. Its death path decrements
                                    // the affiliation — so a Leaf dropped
                                    // before the tape still keeps the
                                    // inventory consistent.
                                    let (shape_id, dtype, rc, buffer) = (*shape_id, *dtype, *rc, buffer.clone());
                                    self.tensors[load_tid] =
                                        TensorData::GraphLeaf { class_id, graph_id, shape_id, dtype, rc, buffer };
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
                        let class_id = self.push_op(graph_id, Op::Unary { x: x_class, uop });
                        class_id
                    }
                    Op::Binary { x, y, bop } => {
                        let x_class = op_to_class[&x];
                        let y_class = op_to_class[&y];
                        self.push_binary_node(graph_id, x_class, y_class, bop)
                    }
                    Op::Cast { x, dtype } => {
                        let x_class = op_to_class[&x];
                        let class_id = self.push_op(graph_id, Op::Cast { x: x_class, dtype });
                        class_id
                    }
                    Op::Bitcast { x, dtype } => {
                        let x_class = op_to_class[&x];
                        let class_id = self.push_op(graph_id, Op::Bitcast { x: x_class, dtype });
                        class_id
                    }
                    Op::Stack { ref ops } => {
                        let ops: Box<[OpId]> = ops.iter().map(|o| op_to_class[o]).collect();
                        let class_id = self.push_op(graph_id, Op::Stack { ops });
                        class_id
                    }
                    Op::Reduce { x, rop, reduce_axis } => {
                        let x_class = op_to_class[&x];
                        let reduce_axis = if reduce_axis.is_null() {
                            *self.graphs[graph_id].shape(x_class).last().expect("Reduce of scalar")
                        } else {
                            op_to_class[&reduce_axis]
                        };
                        let class_id = self.push_op(graph_id, Op::Reduce { x: x_class, rop, reduce_axis });
                        class_id
                    }
                    Op::Reshape { x, shape } => {
                        let x_class = op_to_class[&x];
                        let shape = op_to_class[&shape];
                        let class_id = self.push_op(graph_id, Op::Reshape { x: x_class, shape });
                        class_id
                    }
                    Op::Expand { x, shape } => {
                        let x_class = op_to_class[&x];
                        let shape = op_to_class[&shape];
                        let class_id = self.push_op(graph_id, Op::Expand { x: x_class, shape });
                        class_id
                    }
                    Op::Permute { x, ref axes } => {
                        let x_class = op_to_class[&x];
                        let in_shape = self.graphs[graph_id].shape(x_class);
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
                        let axes = axes.clone();
                        let class_id = self.push_op(graph_id, Op::Permute { x: x_class, axes });
                        class_id
                    }
                    Op::Pad { x, axis, lp, len } => {
                        let x_class = op_to_class[&x];
                        let lp = op_to_class[&lp];
                        let len = op_to_class[&len];
                        let class_id = self.push_op(graph_id, Op::Pad { x: x_class, axis, lp, len });
                        class_id
                    }
                    Op::Narrow { x, axis, start, len } => {
                        let x_class = op_to_class[&x];
                        let start = op_to_class[&start];
                        let len = op_to_class[&len];
                        let class_id = self.push_op(graph_id, Op::Narrow { x: x_class, axis, start, len });
                        class_id
                    }
                    Op::Flip { x, ref axes } => {
                        let x_class = op_to_class[&x];
                        let in_shape = self.graphs[graph_id].shape(x_class);
                        debug_assert!(
                            !axes.is_empty(),
                            "Flip: axes must not be empty (rank {} shape {:?})",
                            in_shape.len(),
                            in_shape
                        );
                        let axes = axes.clone();
                        let class_id = self.push_op(graph_id, Op::Flip { x: x_class, axes });
                        class_id
                    }
                    _ => unreachable!(),
                };
                op_to_class.insert(op_id, class_id);
            }

            op_id = self.kernels[kernel_id].kernel.next_op(op_id);
        }

        let class_id = op_to_class[&my_op_id];
        self.graphs[graph_id].ref_count += 1;
        match &mut self.tensors[tid] {
            TensorData::Graph { class_id: c, .. } | TensorData::Promoted { class_id: c, .. } => *c = class_id,
            TensorData::Eager { .. } => {
                let (kernel_id, op_id, shape_id, rc, dtype) = match self.tensors[tid] {
                    TensorData::Eager { kernel_id, op_id, shape_id, rc, dtype } => (kernel_id, op_id, shape_id, rc, dtype),
                    ref t => unreachable!("{t:?}"),
                };
                self.tensors[tid] = TensorData::Promoted { kernel_id, op_id, class_id, graph_id, shape_id, dtype, rc };
            }
            ref t => panic!("promote_to_graph: cannot attach tensor {tid} to the graph: {t:?}"),
        }
        Ok(class_id)
    }

    /// Fold a symbolic dim-expression class of a graph to a `Constant`,
    /// mirroring [`Kernel::resolve_const`] over graph nodes: `Const` folds
    /// directly; a dim-variable `Leaf` resolves through `leaf_map` into the
    /// tensors slab (variables always carry concrete values, so this never
    /// invents any); `Cast`/`Unary`/`Binary` fold bottom-up with the same
    /// dtype rules. Iterative postorder with dedup, so shared subexpressions
    /// evaluate before every parent referencing them. Anything outside the
    /// symbolic closed set (kernels, movement, data ops, shapes) is not a
    /// scalar dim and resolves to `None`.
    pub(crate) fn resolve_symbolic_class(&self, graph_id: GraphId, cid: OpId) -> Option<Constant> {
        let graph = &self.graphs[graph_id];
        if cid.is_null() {
            return None;
        }
        let mut seen: Set<OpId> = Set::default();
        let mut order: Vec<OpId> = Vec::new();
        let mut stack = vec![(cid, false)];
        for _ in 0..10_000 {
            let Some((id, emit)) = stack.pop() else { break };
            if id.is_null() {
                continue;
            }
            if emit {
                order.push(id);
                continue;
            }
            if !seen.insert(id) {
                continue;
            }
            stack.push((id, true));
            match &graph.ops[id].op {
                Op::Cast { x, .. } | Op::Unary { x, .. } => stack.push((*x, false)),
                Op::Binary { x, y, .. } => {
                    stack.push((*x, false));
                    stack.push((*y, false));
                }
                Op::Index { vec, idx } => {
                    if let Op::Stack { ops } = &graph.ops[*vec].op {
                        stack.push((ops[*idx], false));
                    }
                }
                _ => {}
            }
        }
        if !stack.is_empty() {
            panic!("resolve_symbolic_class did not finish in 10000 steps");
        }

        let mut values: Map<OpId, Option<Constant>> = Map::default();
        for &id in &order {
            let v = match &graph.ops[id].op {
                Op::Const(value) => Some(*value),
                Op::Param { .. } => {
                    let tid = graph.leaf_map.get(&id)?;
                    self.resolve_symbolic(*tid)
                }
                Op::Index { vec, idx } => match &graph.ops[*vec].op {
                    Op::Stack { ops } => values.get(&ops[*idx]).copied().flatten(),
                    _ => None,
                },
                Op::Cast { x, dtype } => values.get(x).copied().flatten().map(|v| v.cast(*dtype)),
                Op::Unary { x, uop } => values.get(x).copied().flatten().map(|v| v.unary(*uop)),
                Op::Binary { x, y, bop } => {
                    values.get(x).copied().flatten().zip(values.get(y).copied().flatten()).map(|(a, b)| {
                        let dt = a.dtype().least_upper_dtype(b.dtype());
                        Constant::binary(a.cast(dt), b.cast(dt), *bop)
                    })
                }
                _ => None,
            };
            values.insert(id, v);
        }
        values[&cid].clone()
    }

    pub fn autotune_jit_kernels(&mut self, graph_id: GraphId) -> Result<(), ZyxError> {
        println!("Autotuning");
        let device_ids: Vec<Dev> = Dev::all();

        let jit_kernels: *const Slab<JitKernelId, JitKernelData> = &self.graphs[graph_id].jit_kernels;
        let jit_kernels: &Slab<JitKernelId, JitKernelData> = unsafe { &*jit_kernels };
        let total = jit_kernels.len().0 as i64 * device_ids.len() as i64;
        let mut progress_bar = crate::progress::ProgressBar::new(total as u64);
        for ek in jit_kernels.values() {
            let class_of = ek.stores.first().copied().unwrap();

            // Timing launch args, bound positionally: read-only defines
            // (`Global` buffers and scalar `Variable` dims) in head order,
            // then `GlobalMut` stores in head order. Every variable carries
            // its actual runtime value — variables are never unknown, so
            // nothing is substituted. Buffer lengths resolve from true
            // graph shapes (never const-folded, never substituted).
            // `ek.loads` parallels the non-store defines and `ek.stores`
            // the mut defines; both invariants are asserted below.
            // `args` holds only Variable values in Param order (no
            // placeholders: buffer slots are bound per-device below).
            let mut args: Vec<LaunchArg> = Vec::new();
            let mut ro_lens: Vec<Dim> = Vec::new();
            let mut mut_lens: Vec<Dim> = Vec::new();
            // True length in elements of a buffer class. Scalar (empty
            // shape) holds one element.
            let resolve_len = |cid: OpId| -> Dim {
                let mut len: Dim = 1;
                for &d in &self.graphs[graph_id].shape(cid) {
                    let v = match self.resolve_symbolic_class(graph_id, d) {
                        Some(v) => v,
                        None => unreachable!("buffer dim class {d:?} does not resolve to a value"),
                    };
                    let dv = match v.as_dim() {
                        Some(dv) => dv,
                        None => unreachable!("buffer dim class {d:?} is not a non-negative integer: {v:?}"),
                    };
                    len = match len.checked_mul(dv) {
                        Some(len) => len,
                        None => unreachable!("buffer dim product overflows"),
                    };
                }
                len
            };
            {
                let mut load_idx = 0usize;
                let mut store_idx = 0usize;
                let mut p = ek.kernel.head;
                while !p.is_null() {
                    match ek.kernel.ops[p].op {
                        Op::Param { kind: ParamKind::Variable, .. } => {
                            let value = match self.resolve_symbolic_class(graph_id, ek.loads[load_idx]) {
                                Some(v) => v,
                                None => unreachable!("dim variable class {:?} does not resolve to a value", ek.loads[load_idx]),
                            };
                            load_idx += 1;
                            args.push(LaunchArg::Variable(value));
                        }
                        Op::Param { kind: ParamKind::Global, .. } => {
                            ro_lens.push(resolve_len(ek.loads[load_idx]));
                            load_idx += 1;
                        }
                        Op::Param { kind: ParamKind::GlobalMut, .. } => {
                            mut_lens.push(resolve_len(ek.stores[store_idx]));
                            store_idx += 1;
                        }
                        _ => {}
                    }
                    p = ek.kernel.next_op(p);
                }
                debug_assert_eq!(load_idx, ek.loads.len(), "loads must parallel Global|Variable defines");
                debug_assert_eq!(store_idx, ek.stores.len(), "stores must parallel GlobalMut defines");
            }

            for &dev_id in device_ids.iter() {
                // AOT-only devices (e.g. cblas) never compile generic zyx kernels
                if dev_id.aot_only() {
                    continue;
                }
                let mut kernel = ek.kernel.clone();
                kernel.dev = dev_id;
                kernel.dev_info = Some(dev_id.info()?);
                progress_bar.inc(1, &format!("autotune {} on dev={dev_id:?}", kernel.name()));
                // Allocate fresh timing buffers in this device's pool for
                // every NULL slot, pre-filled with ones like eager inputs.
                let pool_id = dev_id.pool();
                let mut full_args: Vec<LaunchArg> = Vec::with_capacity(args.len());
                let mut full_mut: Vec<LaunchArg> = Vec::with_capacity(mut_lens.len());
                let mut fresh: Vec<Arc<Placement>> = Vec::new();
                {
                    let (mut vi, mut rli, mut mli) = (0usize, 0usize, 0usize);
                    let mut p = kernel.head;
                    while !p.is_null() {
                        if let Op::Param { kind, dtype, .. } = kernel.ops[p].op {
                            match kind {
                                ParamKind::Variable => {
                                    full_args.push(args[vi].clone());
                                    vi += 1;
                                }
                                ParamKind::Global | ParamKind::GlobalMut => {
                                    let (len, is_mut) = if kind == ParamKind::GlobalMut {
                                        let len = mut_lens[mli];
                                        mli += 1;
                                        (len, true)
                                    } else {
                                        let len = ro_lens[rli];
                                        rli += 1;
                                        (len, false)
                                    };
                                    let bytes_alloc = (dtype.bit_size() as Dim * (len + 1)) / 8;
                                    if !is_mut {
                                        // Fill with dtype ONE, element by
                                        // element, directly into a HOST-POOL
                                        // staging buffer — never a Vec (tensors
                                        // can be tens of GB). Doubling fill:
                                        // write the pattern, then repeatedly
                                        // copy the filled prefix over itself.
                                        let elem: Vec<u8> = match dtype {
                                            DType::BF16 => bf16::ONE.to_le_bytes().to_vec(),
                                            DType::F16 => f16::ONE.to_le_bytes().to_vec(),
                                            DType::F32 => 1f32.to_le_bytes().to_vec(),
                                            DType::F64 => 1f64.to_le_bytes().to_vec(),
                                            DType::U8 | DType::I8 | DType::Bool => vec![1],
                                            DType::F8E4M3 => vec![f8e4m3::ONE.to_bits()],
                                            DType::F8E5M2 => vec![f8e5m2::ONE.to_bits()],
                                            DType::U16 | DType::I16 => 1u16.to_le_bytes().to_vec(),
                                            DType::U32 | DType::I32 => 1u32.to_le_bytes().to_vec(),
                                            DType::U64 | DType::I64 => 1i64.to_le_bytes().to_vec(),
                                        };
                                        let one_len = elem.len();
                                        let fill_bytes = (dtype.bit_size() as usize / 8) * len as usize;
                                        let host_buf = Pool::Host.allocate(fill_bytes as Dim)?;
                                        {
                                            let dst = Pool::Host.buffer_ptr_mut(host_buf);
                                            unsafe {
                                                std::ptr::copy_nonoverlapping(elem.as_ptr(), dst, one_len);
                                                let mut filled = one_len;
                                                while filled < fill_bytes {
                                                    let chunk = filled.min(fill_bytes - filled);
                                                    std::ptr::copy_nonoverlapping(dst, dst.add(filled), chunk);
                                                    filled += chunk;
                                                }
                                            }
                                        }
                                        // Upload through a one-copy queue: slot 0 is
                                        // the host staging placement, slot 1
                                        // the destination, allocated by replay
                                        // from its byte size.
                                        let mut queue = CmdQueue::new();
                                        queue.push(Cmd::Copy {
                                            src: OpId::from(0),
                                            dst: OpId::from(1),
                                            dst_pool: pool_id,
                                            dst_dtype: 1,
                                            dst_dims: vec![PlanDim::Const(bytes_alloc)],
                                        });
                                        let mut boundary = Map::default();
                                        boundary.insert(
                                            OpId::from(0),
                                            Arc::new(Placement {
                                                shards: vec![Shard { pool: Pool::Host, chunk: host_buf, offset: 0, len: fill_bytes }],
                                            }),
                                        );
                                        let out = {
                                            let mut outputs = Set::default();
                                            outputs.insert(OpId::from(1));
                                            queue.schedule(&outputs).replay(boundary, &Map::default())?
                                        };
                                        // No manual release: the boundary Arc
                                        // in `out` frees the staging chunk on
                                        // drop (Placement Drop semantics).
                                        let placed = Arc::clone(&out[&OpId::from(1)]);
                                        fresh.push(Arc::clone(&placed));
                                        let [shard] = &placed.shards[..] else {
                                            todo!("multi-shard staging upload in graph launch")
                                        };
                                        full_args.push(LaunchArg::Buffer {
                                            chunk: shard.chunk,
                                            offset: shard.offset,
                                            len: shard.len,
                                        });
                                    } else {
                                        // Mut timing buffers are written by the
                                        // kernel: allocate directly, no upload.
                                        let buf = pool_id.allocate(bytes_alloc)?;
                                        let placed = Arc::new(Placement {
                                            shards: vec![Shard { pool: pool_id, chunk: buf, offset: 0, len: bytes_alloc as usize }],
                                        });
                                        fresh.push(Arc::clone(&placed));
                                        full_mut.push(LaunchArg::Buffer { chunk: buf, offset: 0, len: bytes_alloc as usize });
                                    }
                                }
                            }
                        }
                        p = kernel.next_op(p);
                    }
                }
                full_args.extend(full_mut);
                let (dev_prog, timing) = self.get_or_autotune(kernel, &full_args)?;
                // Dropping the fresh placements returns every timing buffer
                // to its pool's free list (last-owner Drop).
                drop(fresh);
                let prog = ProgramId { dev: dev_id, program_id: dev_prog };

                let g = &mut self.graphs[graph_id];
                let inputs = g.push_op(Op::Stack { ops: ek.loads.clone().into() });
                let outputs = g.push_op(Op::Stack { ops: ek.stores.clone().into() });
                g.mint_node(Op::Kernel { inputs, outputs, info: Box::new((prog, timing)) }, class_of);
            }
        }

        if cfg!(debug_assertions) {
            let mut seen: Set<OpId> = Set::default();
            for cid in self.graphs[graph_id].ops.iter().filter(|(id, nd)| nd.class_of == *id).map(|(id, _)| id) {
                for nid in self.graphs[graph_id].class_nodes(cid) {
                    if !seen.insert(nid) {
                        continue;
                    }
                    if let Op::Kernel { info, .. } = &self.graphs[graph_id].ops[nid].op {
                        debug_assert!(info.1 > 0, "Kernel node {nid:?} has zero cost after autotune");
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
                    self.leaf_buffer(tid).is_some()
                        | matches!(self.tensors[tid], TensorData::Symbolic { expr, .. } if matches!(self.exprs[expr], Expr::Variable { .. })),
                    "leaf {tid} not realized"
                );
                let affiliated = match self.tensors[tid] {
                    TensorData::Graph { graph_id: g, .. }
                    | TensorData::Promoted { graph_id: g, .. }
                    | TensorData::GraphLeaf { graph_id: g, .. } => g == graph_id,
                    // A variable leaf is a shared input: it carries no graph
                    // affiliation in its TensorData — nothing to check.
                    TensorData::Symbolic { .. } => continue,
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
                    debug_assert!(self.leaf_buffer(tid).is_none(), "non-leaf graph tensor {tid} realized before realize");
                }
            }
        }
    }

    /// Compiles the graph into a backend [`Plan`]: pattern-matches AOT kernels,
    /// kernelizes the remaining structural nodes, autotunes the fused kernels,
    /// extracts the cheapest kernel path, lowers it to a [`CmdQueue`], and
    /// schedules the resulting plan.
    pub(crate) fn compile_graph(&mut self, graph_id: GraphId, output_set: &BTreeSet<OpId>) -> Result<Plan, ZyxError> {
        debug_assert!(self.graphs.contains_id(graph_id));
        self.debug_assert_pre_realize(graph_id);

        if crate::debug_mask().egraph() {
            self.graphs[graph_id].debug();
        }

        for cid in self.graphs[graph_id].ops.iter().filter(|(id, nd)| nd.class_of == *id).map(|(id, _)| id) {
            let has_leaf =
                self.graphs[graph_id].class_nodes(cid).any(|nid| matches!(&self.graphs[graph_id].ops[nid].op, Op::Param { .. }));
            if has_leaf {
                let &tid = self.graphs[graph_id].leaf_map.get(&cid).expect("class {cid:?} has Leaf node but not in leaf_map");
                assert!(
                    self.leaf_buffer(tid).is_some()
                        || matches!(self.tensors[tid], TensorData::Symbolic { expr, .. } if matches!(self.exprs[expr], Expr::Variable { .. })),
                    "leaf class {cid:?} tid {tid:?} neither in buffer_map nor a variable"
                );
            } else {
                assert!(!self.graphs[graph_id].leaf_map.contains_key(&cid), "class {cid:?} has no Leaf node but is in leaf_map");
            }
        }

        // Pattern match specialized AOT kernels (e.g. matmul -> cblas) so they can
        // compete with the fused zyx kernels in extraction.
        // SAFETY: graphs borrow ends before the match call, rust is stupid
        let dev_ids: Vec<Dev> = Dev::all();
        let graph_ptr: *mut Graph = &mut self.graphs[graph_id];
        for dev_id in dev_ids {
            dev_id.match_graph(unsafe { &mut *graph_ptr }, output_set);
        }

        // Lower user custom kernels into Kernel twins so the pool grouping and
        // gap filling below see them alongside the AOT kernels.
        self.graphs[graph_id].lower_custom_kernels();

        // AOT kernel output classes, grouped by the memory pool they run in.
        let mut pool_kernel_outputs: Map<Pool, Set<OpId>> = Map::default();
        for cid in self.graphs[graph_id].ops.iter().filter(|(id, nd)| nd.class_of == *id).map(|(id, _)| id) {
            for nid in self.graphs[graph_id].class_nodes(cid) {
                if let Op::Kernel { info, .. } = &self.graphs[graph_id].ops[nid].op {
                    let pool = info.0.dev.pool();
                    pool_kernel_outputs.entry(pool).or_default().insert(cid);
                }
            }
        }

        // Pass 1: fill every gap between all AOT kernels, ignoring devices.
        let all_kernel_outputs: Set<OpId> = pool_kernel_outputs.values().flatten().copied().collect();
        self.graphs[graph_id].fill_gaps(&all_kernel_outputs, output_set);

        // Pass 2: for each memory pool, fill the gaps between only that pool's
        // kernels — other pools' kernels are ignored, giving single-pool paths.
        for active_outputs in pool_kernel_outputs.values() {
            self.graphs[graph_id].fill_gaps(active_outputs, output_set);
        }

        // Autotunes custom zyx kernels for all devices and adds kernel nodes for all of them
        self.autotune_jit_kernels(graph_id)?;
        self.graphs[graph_id].verify();

        let nodes = self.graphs[graph_id].extract(output_set);

        // Transfers between the extracted producer/consumer pairs that live
        // on different devices. Only the extracted path is considered: a
        // class holding kernels on several devices needs no transfer when
        // extraction chose the same-device producer.
        // Leaf buffers collected into an owned map so the immutable borrow
        // ends before the &mut add call below.
        let buffer_map: Map<TensorId, Arc<Placement>> =
            self.graphs[graph_id].leaf_map.values().filter_map(|&tid| self.leaf_buffer(tid).map(|buf| (tid, buf))).collect();
        let nodes = self.graphs[graph_id].add_memory_ops(&buffer_map, &nodes);

        // Leaf pools at compile time — cross-pool aliases copy through the
        // kernel pool, so leaves must stay put across replays.
        let mut leaf_pools: Map<OpId, Pool> = Map::default();
        for (&cid, &tid) in &self.graphs[graph_id].leaf_map {
            // Variable leaves have no buffer and no pool — they bind per
            // replay from the tensors slab, so no pool invariant applies.
            if let Some(buf) = self.leaf_buffer(tid) {
                let [Shard { pool, .. }] = &buf.shards[..] else {
                    todo!("multi-shard leaf in compile-time leaf pools")
                };
                leaf_pools.insert(cid, *pool);
            }
        }

        // Lower the extracted nodes to a command queue: kernels become
        // launches (outputs sized symbolically for replay), transfers become
        // copies, After outputs alias their base leaf buffer. Replay
        // allocates every unbound def and resolves aliases through the
        // boundary — no allocation or liveness decisions are made here.
        fn dim_expr(graph: &Graph, dim: OpId) -> PlanDim {
            match graph.ops[dim].op {
                Op::Const(c) => PlanDim::Const(c.as_dim().unwrap_or_else(|| panic!("dim class {dim:?} is not a constant"))),
                Op::Param { .. } => PlanDim::Leaf(dim),
                Op::Binary { x, y, bop } => {
                    PlanDim::Binary { x: Box::new(dim_expr(graph, x)), y: Box::new(dim_expr(graph, y)), bop }
                }
                Op::Cast { x, dtype } => PlanDim::Cast { x: Box::new(dim_expr(graph, x)), dtype },
                ref op => unreachable!("alloc dim class {dim:?} must be a dim over Const/leaf leaves, got {op:?}"),
            }
        }
        fn alloc_spec(graph: &Graph, class: OpId) -> (Dim, Vec<PlanDim>) {
            let dtype_size = Dim::from(graph.dtype(class).bit_size() / 8);
            let dims = graph.shape(class).iter().map(|&d| dim_expr(graph, d)).collect();
            (dtype_size, dims)
        }
        let graph = &self.graphs[graph_id];

        // After output classes alias the buffer of x's base leaf class: the
        // assign writes the new buffer version in-place into that leaf
        // buffer, so an After class shares the leaf's buffer. A cross-pool
        // leaf needs one kernel-pool copy of itself shared by every alias
        // of that leaf — chained assigns must write the same physical
        // buffer or the intermediate writes are lost.
        let mut aliases: Vec<(OpId, OpId, Dim, Vec<PlanDim>)> = Vec::new();
        for (cid, nd) in graph.ops.iter().filter(|(id, nd)| nd.class_of == *id) {
            if let Op::After { x, .. } = nd.op {
                let base = graph.base_leaf(x);
                let (dtype_size, dims) = alloc_spec(graph, cid);
                aliases.push((cid, base, dtype_size, dims));
            }
        }

        // Pool of the kernel that stores each alias class.
        let mut store_pool: Map<OpId, Pool> = Map::default();
        for &nid in &nodes {
            if let Op::Kernel { outputs, ref info, .. } = graph.ops[nid].op {
                let Op::Stack { ops: outputs } = &graph.ops[outputs].op else {
                    unreachable!()
                };
                let pool = info.0.dev.pool();
                for &oc in outputs {
                    store_pool.insert(oc, pool);
                }
            }
        }

        let mut queue = CmdQueue::new();
        let mut leaf_copy: Map<OpId, OpId> = Map::default();
        for &(class, to, dtype_size, ref dims) in &aliases {
            match store_pool.get(&class) {
                Some(pool) if leaf_pools[&to] != *pool => {
                    let owner = *leaf_copy.entry(to).or_insert_with(|| {
                        queue.push(Cmd::Copy {
                            src: to,
                            dst: class,
                            dst_pool: *pool,
                            dst_dtype: dtype_size,
                            dst_dims: dims.clone(),
                        });
                        class
                    });
                    if owner != class {
                        queue.push(Cmd::Alias { class, to: owner });
                    }
                }
                _ => queue.push(Cmd::Alias { class, to }),
            }
        }

        // One launch per kernel node (args in head order: loads then
        // stores), one copy per transfer. A repeated output class keeps a
        // single allocation spec — replay resolves the duplicate def to the
        // first placement, mirroring the old single-buffer behavior.
        let mut emitted: Set<OpId> = Set::default();
        for &nid in &nodes {
            match graph.ops[nid].op {
                Op::Kernel { inputs, outputs, ref info, .. } => {
                    let Op::Stack { ops: inputs } = &graph.ops[inputs].op else {
                        unreachable!()
                    };
                    let Op::Stack { ops: outputs } = &graph.ops[outputs].op else {
                        unreachable!()
                    };
                    let mut args: Vec<OpId> = inputs.to_vec();
                    let mut specs = Vec::new();
                    for &oc in outputs {
                        args.push(oc);
                        if emitted.insert(oc) {
                            let (dtype_size, dims) = alloc_spec(graph, oc);
                            specs.push((oc, dtype_size, dims));
                        }
                    }
                    queue.push(Cmd::Launch { program: info.0, args, outputs: specs });
                }
                Op::ToDevice { x, device, .. } => {
                    // Pool is always derived from the device, never the reverse.
                    let pool = device.pool();
                    let class_of = graph.ops[nid].class_of;
                    let (dtype_size, dims) = alloc_spec(graph, class_of);
                    queue.push(Cmd::Copy { src: x, dst: class_of, dst_pool: pool, dst_dtype: dtype_size, dst_dims: dims });
                }
                _ => unreachable!(),
            }
        }

        #[cfg(feature = "viz")]
        self.viz.snapshot(&self.graphs[graph_id], &queue.cmds);
        let plan = queue.schedule(&output_set.iter().copied().collect());
        Ok(plan)
    }

    pub fn eagerify(&mut self, tid: TensorId, new_buffer: Option<Arc<Placement>>) {
        // `None` = demote-only (no realization buffer); `Some` = the plan
        // computed this class into the placement.
        // Snapshot the replacement (borrows end here); the slot is assigned below.
        let (graph_id, next) = match &self.tensors[tid] {
            TensorData::Promoted { kernel_id, op_id, graph_id, shape_id, rc, dtype, .. } => {
                // Unrealized promoted tensor: the eager producer kernel was
                // never mutated, so just demote in place.
                (
                    *graph_id,
                    TensorData::Eager { kernel_id: *kernel_id, op_id: *op_id, shape_id: *shape_id, dtype: *dtype, rc: *rc },
                )
            }
            // Realized graph output: the plan computed this class into
            // `new_buffer` — leave the graph as a buffer-backed Leaf
            // (normal plan execution; no special launch). Only reached
            // from realize's output loop, which always passes a placement —
            // drop never eagerifies Graph tensors.
            TensorData::Graph { graph_id, shape_id, dtype, rc, .. } => {
                let Some(new_buffer) = new_buffer else {
                    panic!("eagerify: realized graph tensor {tid} given no buffer");
                };
                (*graph_id, TensorData::Leaf { shape_id: *shape_id, dtype: *dtype, buffer: new_buffer, rc: *rc })
            }
            TensorData::GraphLeaf { graph_id, shape_id, dtype, rc, .. } => {
                let Some(new_buffer) = new_buffer else {
                    panic!("eagerify: realized graph-leaf tensor {tid} given no buffer");
                };
                // Realized: re-point at the realization's placement. No
                // producer to detach from (GraphLeaf carries no kernel_id).
                // The overwritten placement drops with the slot: its final
                // clone runs `Placement::drop`. When both are the SAME
                // placement (an assign's After class aliases the base leaf's
                // buffer), the count never hits zero — nothing is released.
                (*graph_id, TensorData::Leaf { shape_id: *shape_id, dtype: *dtype, buffer: new_buffer, rc: *rc })
            }
            // Already-realized leaves carry no graph affiliation and eagerify
            // is only called on graph tensors: reaching them is a bug.
            TensorData::Leaf { .. } | TensorData::PendingLeaf { .. } => {
                unreachable!("eagerify: {tid} is already a realized leaf")
            }
            // Already eager or a pure-slab value: nothing to do.
            TensorData::Eager { .. } | TensorData::Symbolic { .. } => return,
        };
        self.tensors[tid] = next;

        self.graphs[graph_id].ref_count -= 1;
    }

    pub fn assert_graph_alive(&self, graph_id: GraphId) {
        assert!(!graph_id.is_null(), "tape scope has ended (tensor belongs to a dead tape scope)");
        assert!(!self.graphs[graph_id].dead, "tape scope has ended (tensor belongs to a dead tape scope");
    }

    /// Pushes a constant node into the graph and returns its class.
    ///
    /// Consts hashcons by value: pushing an equal constant twice returns the
    /// same class (see [`Node::Const`] for why that is sound).
    pub fn push_const(&mut self, graph_id: GraphId, value: Constant) -> OpId {
        self.push_op(graph_id, Op::Const(value))
    }

    pub fn push_leaf_node(&mut self, graph_id: GraphId, dtype: DType, shape: OpId) -> (OpId, OpId) {
        // Fresh cons_id: leaves hashcons but never merge (each buffer keeps
        // its own class).
        let cons_id = self.graphs[graph_id].max_cons_id;
        self.graphs[graph_id].max_cons_id += 1;
        let node = Op::Param { dtype, kind: ParamKind::Global, shape, cons_id };
        let g = &mut self.graphs[graph_id];
        let nid = g.ops.push(OpNode { op: node.clone(), class_of: OpId::NULL, next_in_class: OpId::NULL });
        let cid = nid;
        g.ops[nid].class_of = cid;
        g.hashcons.insert(node, nid);
        (nid, cid)
    }

    // TODO delete this method
    /// Numeric shape of a class for the runtime's `shapes` cache: static dim
    pub fn push_op(&mut self, graph_id: GraphId, op: Op) -> OpId {
        self.graphs[graph_id].push_op(op)
    }

    pub fn push_binary_node(&mut self, graph_id: GraphId, x: OpId, y: OpId, bop: BOp) -> OpId {
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
                let y = self.push_op(graph_id, Op::Expand { x: y, shape });
                (x, y)
            }
            (0, _) if ry > 0 => {
                let shape = self.shape_class(graph_id, self.graphs[graph_id].shape(y));
                let x = self.push_op(graph_id, Op::Expand { x, shape });
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
        let concrete = |s: &[OpId]| -> Vec<Dim> {
            s.iter().map(|&d| self.graphs[graph_id].resolve_const(d).and_then(Constant::as_dim).unwrap_or(-1)).collect()
        };
        let sx = self.graphs[graph_id].shape(x);
        let sy = self.graphs[graph_id].shape(y);
        debug_assert_eq!(
            concrete(&sx),
            concrete(&sy),
            "binary operands must be broadcast to equal shapes before Node::Binary (broadcasting is performed upstream); got {sx:?} vs {sy:?}"
        );
        self.push_op(graph_id, Op::Binary { x, y, bop })
    }
}
