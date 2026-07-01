// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! E-graph-based kernel search.
//!
//! Builds an e-graph from the computation graph, enumerates kernel alternatives
//! via pattern matching, then extracts the cheapest execution plan.

use crate::{
    DType, DebugMask, Map, Set, ZyxError,
    backend::{AutotuneConfig, Device, DeviceId, MemoryPool, PoolId, ProgramId},
    dtype::Constant,
    graph::{
        Node,
        compiled::{BufferSlot, CompiledNode},
    },
    kernel::{BOp, Kernel, UOp},
    kernel_cache::KernelCache,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
};

use super::Graph;

// ── IDs ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct NodeId(pub u32);

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
pub(crate) struct ClassId(pub u32);

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

// ── ENode —

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum ENode {
    Const(Constant),
    Leaf(DType),
    Expand(ClassId),
    Permute(ClassId, Box<[UAxis]>),
    Reshape(ClassId, Box<[Dim]>),
    Pad(ClassId, Box<[(i64, i64)]>),
    Reduce(ClassId, BOp),
    Cast(ClassId, DType),
    Unary(ClassId, UOp),
    Binary(ClassId, ClassId, BOp),
    ToDevice(ClassId, DeviceId),
    Kernel(Box<[ClassId]>, Box<[ClassId]>, ProgramId),
}

impl ENode {
    pub(crate) fn child_classes(&self) -> Vec<ClassId> {
        match self {
            ENode::Const(_) | ENode::Leaf(_) => vec![],
            ENode::Expand(x) => vec![*x],
            ENode::Permute(x, _) => vec![*x],
            ENode::Reshape(x, _) => vec![*x],
            ENode::Pad(x, _) => vec![*x],
            ENode::Reduce(x, _) => vec![*x],
            ENode::Cast(x, _) => vec![*x],
            ENode::Unary(x, _) => vec![*x],
            ENode::ToDevice(x, _) => vec![*x],
            ENode::Binary(x, y, _) => vec![*x, *y],
            ENode::Kernel(inputs, _, _) => inputs.to_vec(),
        }
    }

    fn child_classes_mut(&mut self) -> Vec<&mut ClassId> {
        match self {
            ENode::Const(_) | ENode::Leaf(_) => vec![],
            ENode::Expand(x) => vec![x],
            ENode::Permute(x, _) => vec![x],
            ENode::Reshape(x, _) => vec![x],
            ENode::Pad(x, _) => vec![x],
            ENode::Reduce(x, _) => vec![x],
            ENode::Cast(x, _) => vec![x],
            ENode::Unary(x, _) => vec![x],
            ENode::ToDevice(x, _) => vec![x],
            ENode::Binary(x, y, _) => vec![x, y],
            ENode::Kernel(inputs, _, _) => inputs.iter_mut().collect(),
        }
    }

    pub(crate) fn is_kernel(&self) -> bool {
        matches!(self, ENode::Kernel(..))
    }
}

// ── EClass ─────────────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct EClass {
    pub nodes: Vec<NodeId>,
    pub parents: Vec<(NodeId, usize)>,
    pub shape: Option<Box<[Dim]>>,
    pub dtype: Option<DType>,
}

impl EClass {
    fn new(node: NodeId) -> Self {
        Self {
            nodes: vec![node],
            parents: vec![],
            shape: None,
            dtype: None,
        }
    }
}

// ── EGraph ─────────────────────────────────────────────────────

pub(crate) struct EGraph {
    pub(crate) nodes: Slab<NodeId, ENode>,
    pub(crate) classes: Slab<ClassId, EClass>,
    pub(crate) class_of: Vec<ClassId>,
    pub(crate) class_parent: Vec<ClassId>,
    pub(crate) class_rank: Vec<u8>,
    hashcons: Map<ENode, NodeId>,
    pub(crate) costs: Map<NodeId, u64>,
    pub(crate) kernel_irs: Map<NodeId, Kernel>,
}

impl EGraph {
    pub(crate) fn new() -> Self {
        Self {
            nodes: Slab::new(),
            classes: Slab::new(),
            class_of: Vec::new(),
            class_parent: Vec::new(),
            class_rank: Vec::new(),
            hashcons: Map::default(),
            costs: Map::default(),
            kernel_irs: Map::default(),
        }
    }

    // ── Union-Find ─────────────────────────────────────────

    pub(crate) fn find_class(&mut self, cid: ClassId) -> ClassId {
        // Iterative find with path compression.
        let mut root = cid;
        while root != self.class_parent[root.0 as usize] {
            root = self.class_parent[root.0 as usize];
        }
        let mut cur = cid;
        while cur != root {
            let next = self.class_parent[cur.0 as usize];
            self.class_parent[cur.0 as usize] = root;
            cur = next;
        }
        root
    }

    pub(crate) fn find(&mut self, nid: NodeId) -> ClassId {
        let cid = self.class_of[nid.0 as usize];
        self.find_class(cid)
    }

    pub(crate) fn grow_uf_arrays(&mut self, idx: usize) {
        if idx >= self.class_of.len() {
            let cid = ClassId(self.class_of.len() as u32);
            self.class_of.resize(idx + 1, cid);
            self.class_parent.resize(idx + 1, cid);
            self.class_rank.resize(idx + 1, 0);
        }
        // Ensure class_parent[idx] self-points for new UF entries.
        if idx < self.class_parent.len() && self.class_parent[idx] != ClassId(idx as u32) {
            self.class_parent[idx] = ClassId(idx as u32);
        }
    }

    // ── Hashcons + creation ────────────────────────────────

    pub(crate) fn make(&mut self, mut kind: ENode) -> (NodeId, ClassId) {
        for c in kind.child_classes_mut() {
            *c = self.find_class(*c);
        }

        if let Some(&nid) = self.hashcons.get(&kind) {
            let cid = self.find(nid);
            return (nid, cid);
        }

        let children: Vec<ClassId> = kind.child_classes();

        let nid = self.nodes.push(kind.clone());
        let cid = self.classes.push(EClass::new(nid));

        let idx = nid.0 as usize;
        self.grow_uf_arrays(idx);
        self.class_of[idx] = cid;
        let cidx = cid.0 as usize;
        self.grow_uf_arrays(cidx);
        self.class_parent[cidx] = cid;

        self.hashcons.insert(kind, nid);

        for (child_idx, &child) in children.iter().enumerate() {
            let child_root = self.find_class(child);
            self.classes[child_root].parents.push((nid, child_idx));
        }

        (nid, cid)
    }

    // ── Add enode to an existing class ─────────────────────

    pub(crate) fn add_to_class(&mut self, nid: NodeId, cid: ClassId) {
        let target = self.find_class(cid);
        let old_cid = self.find(nid);
        if old_cid == target {
            return;
        }
        // Remove from old class, add to target.
        self.classes[old_cid].nodes.retain(|&n| n != nid);
        self.class_of[nid.0 as usize] = target;
        self.classes[target].nodes.push(nid);
        // Link UF: old root → target root (no draining).
        self.link_into(old_cid, target);

        let children: Vec<ClassId> = self.nodes[nid].child_classes();
        for (child_idx, &child) in children.iter().enumerate() {
            let child_root = self.find_class(child);
            self.classes[child_root].parents.push((nid, child_idx));
        }
    }

    /// Link UF structure: merge class `a` into class `b`.
    /// Propagates a's parents to b so reverse edges remain reachable.
    fn link_into(&mut self, a: ClassId, b: ClassId) {
        let a = self.find_class(a);
        let b = self.find_class(b);
        if a != b {
            // Move parents from a to b (the surviving root).
            let a_parents = std::mem::take(&mut self.classes[a].parents);
            self.classes[b].parents.extend(a_parents);

            // Union by rank to keep trees shallow; always make b the logical root.
            let rank_a = self.class_rank[a.0 as usize];
            let rank_b = self.class_rank[b.0 as usize];
            self.class_parent[a.0 as usize] = b;
            if rank_a >= rank_b {
                self.class_rank[b.0 as usize] = rank_a + 1;
            }
        }
    }

    // ── Rebuild / path compression ─────────────────────────

    pub(crate) fn compress_paths(&mut self) {
        let cids: Vec<ClassId> = self.classes.ids().collect();
        for &cid in &cids {
            self.find_class(cid);
        }
    }

    /// Canonicalize all enode children, rebuild the hashcons,
    /// and rebuild parent lists from scratch.
    /// Call after a batch of merges to keep the e-graph consistent.
    pub(crate) fn rebuild(&mut self) {
        let all_nids: Vec<NodeId> = self.nodes.ids().collect();

        // 1. Canonicalize every enode's child ClassIds.
        for &nid in &all_nids {
            let old: Vec<ClassId> = self.nodes[nid].child_classes();
            let canonical: Vec<ClassId> = old.iter().map(|&c| self.find_class(c)).collect();
            for (i, c) in self.nodes[nid].child_classes_mut().into_iter().enumerate() {
                *c = canonical[i];
            }
        }

        // 2. Rebuild hashcons with canonicalized keys.
        self.hashcons.clear();
        for &nid in &all_nids {
            self.hashcons.insert(self.nodes[nid].clone(), nid);
        }

        // 3. Rebuild parent lists from scratch.
        let cids: Vec<ClassId> = self.classes.ids().collect();
        for &cid in &cids {
            self.classes[cid].parents.clear();
        }
        for &nid in &all_nids {
            let children = self.nodes[nid].child_classes();
            for (idx, &child) in children.iter().enumerate() {
                let child_root = self.find_class(child);
                self.classes[child_root].parents.push((nid, idx));
            }
        }
    }

    // ── Build from Graph ─────────────────────────────────

    pub(crate) fn build_from_graph(&mut self, order: &[TensorId], graph: &Graph) -> Map<TensorId, ClassId> {
        let mut tensor_to_cid: Map<TensorId, ClassId> = Map::default();

        for &tid in order {
            let kind = self.node_to_enkind(tid, &tensor_to_cid, graph);
            let (nid, cid) = self.make(kind);
            let cid = self.find_class(cid);

            // For Custom nodes, fix the Kernel enode's output class (it was
            // unknown at construction time) and set a heuristic cost so extraction
            // can find it.  No kernel_irs entry — these are pre-compiled programs
            // and autotune_all_kernels will skip them.
            if matches!(&graph[tid], Node::Custom(_)) {
                if let ENode::Kernel(_, outputs, _) = &mut self.nodes[nid] {
                    outputs[0] = cid;
                }
                self.costs.insert(nid, 1000);
            }

            let shape: Box<[Dim]> = graph.shape(tid).to_vec().into_boxed_slice();
            self.classes[cid].shape = Some(shape);
            self.classes[cid].dtype = Some(graph.dtype(tid));
            tensor_to_cid.insert(tid, cid);
        }

        tensor_to_cid
    }

    fn node_to_enkind(&self, tid: TensorId, map: &Map<TensorId, ClassId>, graph: &Graph) -> ENode {
        match &graph[tid] {
            Node::Const { value } => ENode::Const(*value),
            Node::Leaf { dtype } => ENode::Leaf(*dtype),
            Node::Expand { x } => ENode::Expand(map[x]),
            Node::Permute { x } => {
                let axes = graph.axes(tid).to_vec().into_boxed_slice();
                ENode::Permute(map[x], axes)
            }
            Node::Reshape { x } => {
                let shape = graph.shape(tid).to_vec().into_boxed_slice();
                ENode::Reshape(map[x], shape)
            }
            Node::Pad { x } => {
                let padding = graph.padding(tid).to_vec().into_boxed_slice();
                ENode::Pad(map[x], padding)
            }
            Node::Reduce { x, rop } => ENode::Reduce(map[x], *rop),
            Node::Cast { x, dtype } => ENode::Cast(map[x], *dtype),
            Node::Unary { x, uop } => ENode::Unary(map[x], *uop),
            Node::Binary { x, y, bop } => ENode::Binary(map[x], map[y], *bop),
            Node::Custom(ck) => {
                let inputs: Vec<ClassId> = ck.inputs.iter().map(|tid| map[tid]).collect();
                // Output class is unknown yet — fixed up in build_from_graph.
                let outputs = vec![ClassId::NULL].into_boxed_slice();
                ENode::Kernel(inputs.into_boxed_slice(), outputs, ck.program)
            }
            Node::ToDevice { x, device } => ENode::ToDevice(map[x], *device),
        }
    }

    // ── Kernel enumeration ────────────────────────────────

    pub(crate) fn saturate(&mut self) {
        // Each fuser tries to match enodes in e-classes and insert kernel
        // alternatives. Run to fixpoint since one match may enable another.
        loop {
            let mut added = false;
            if self.try_fuse_matmul() {
                added = true;
            }
            if !added {
                break;
            }
        }
        // Rebuild after saturation so children, hashcons, and parents are
        // fully canonicalised before extraction or further passes.
        self.rebuild();
    }

    /// Try to match all e-classes for `reduce_sum(mul(expand(A), expand(permute(B))))`
    /// and insert `MatmulKernel` alternatives.
    fn try_fuse_matmul(&mut self) -> bool {
        let mut added = false;
        let classes: Vec<ClassId> = self.classes.ids().collect();

        for cid in classes {
            if !self.classes.contains_key(cid) {
                continue;
            }
            // Collect enodes from this class (clone to avoid borrow issues)
            let enodes: Vec<NodeId> = self.classes[cid].nodes.clone();
            for &nid in &enodes {
                // Pattern: Reduce(Add, mul_class)
                let mul_class = match &self.nodes[nid] {
                    ENode::Reduce(x, BOp::Add) => *x,
                    _ => continue,
                };

                let mul_class_root = self.find_class(mul_class);
                // Check the reduce is over the last dimension.
                let shape = match &self.classes[cid].shape {
                    Some(s) => s.clone(),
                    None => continue,
                };
                let mul_shape = match &self.classes[mul_class_root].shape {
                    Some(s) => s.clone(),
                    None => continue,
                };
                // Reduce must contract one axis — the last axis of the mul output.
                if mul_shape.len() < 2 {
                    continue;
                }
                // shape after reduction should be mul_shape without the contracted dim
                if shape.len() != mul_shape.len() - 1 {
                    continue;
                }

                // Check mul_class contains Binary(Mul, ...)
                let mul_nodes: Vec<NodeId> = self.classes[mul_class_root].nodes.clone();
                let found_mul = mul_nodes
                    .iter()
                    .any(|&mn| matches!(&self.nodes[mn], ENode::Binary(_, _, BOp::Mul)));
                if !found_mul {
                    continue;
                }

                // Try both orderings for A and B sides.
                // We need: left = Expand(A_raw), right = Expand(Permute[1,0](B_raw))
                let mul_nid = mul_nodes
                    .iter()
                    .find(|&&mn| matches!(&self.nodes[mn], ENode::Binary(_, _, BOp::Mul)))
                    .copied()
                    .unwrap();
                let (left, right) = match &self.nodes[mul_nid] {
                    ENode::Binary(a, b, BOp::Mul) => (*a, *b),
                    _ => unreachable!(),
                };

                for &(expand_a, expand_b) in &[(left, right), (right, left)] {
                    let expand_a_root = self.find_class(expand_a);
                    let expand_b_root = self.find_class(expand_b);

                    let a_raw = match self.classes[expand_a_root].nodes.iter().find_map(|&n| match &self.nodes[n] {
                        ENode::Expand(x) => Some(*x),
                        _ => None,
                    }) {
                        Some(x) => x,
                        None => continue,
                    };

                    // B side: Expand → Reshape → Permute[1,0] → Leaf
                    let reshape_class = match self.classes[expand_b_root].nodes.iter().find_map(|&n| match &self.nodes[n] {
                        ENode::Expand(x) => Some(*x),
                        _ => None,
                    }) {
                        Some(x) => x,
                        None => continue,
                    };

                    let reshape_class_root = self.find_class(reshape_class);
                    let permute_class = match self.classes[reshape_class_root]
                        .nodes
                        .iter()
                        .find_map(|&n| match &self.nodes[n] {
                            ENode::Reshape(x, _) => Some(*x),
                            _ => None,
                        }) {
                        Some(x) => x,
                        None => continue,
                    };

                    let permute_class_root = self.find_class(permute_class);
                    let b_raw = match self.classes[permute_class_root]
                        .nodes
                        .iter()
                        .find_map(|&n| match &self.nodes[n] {
                            ENode::Permute(x, axes) if axes.len() == 2 && axes[0] == 1 && axes[1] == 0 => Some(*x),
                            _ => None,
                        }) {
                        Some(x) => x,
                        None => continue,
                    };

                    let a_raw_root = self.find_class(a_raw);
                    let b_raw_root = self.find_class(b_raw);

                    // Verify inputs are leaves
                    if !self.classes[a_raw_root]
                        .nodes
                        .iter()
                        .any(|&n| matches!(&self.nodes[n], ENode::Leaf(_)))
                    {
                        continue;
                    }
                    if !self.classes[b_raw_root]
                        .nodes
                        .iter()
                        .any(|&n| matches!(&self.nodes[n], ENode::Leaf(_)))
                    {
                        continue;
                    }

                    // Create the MatmulKernel enode
                    let inputs: Box<[ClassId]> = vec![a_raw, b_raw].into_boxed_slice();
                    let outputs: Box<[ClassId]> = vec![cid].into_boxed_slice();
                    let (knid, _own_class) = self.make(ENode::Kernel(inputs, outputs, ProgramId::NULL));
                    self.add_to_class(knid, cid);
                    added = true;
                    break;
                }
            }
        }

        added
    }

    // ── Extraction (DP) ───────────────────────────────────

    /// Extract the cheapest all-kernel plan from the e-graph.
    ///
    /// Returns the selected kernels as `(NodeId, ENode)` pairs in topological order.
    /// Every enode in the returned plan is a `ENode::Kernel` — raw operator enodes
    /// cannot be executed directly, only compiled kernels can.
    ///
    /// # Invariant
    /// Every class reachable from an output must contain at least one `ENode::Kernel`
    /// alternative (or be a `Leaf` base case — `Const` is always compiled into kernels
    /// as a literal, never loaded from memory). `kernelize_all` guarantees this.
    /// Violation panics in `extract_dp`.
    pub(crate) fn extract(&mut self, outputs: &[ClassId]) -> Vec<(NodeId, ENode)> {
        let mut cost_map: Map<ClassId, u64> = Map::default();
        let mut choice: Map<ClassId, NodeId> = Map::default();

        for &out in outputs {
            self.extract_dp(out, &mut cost_map, &mut choice);
        }

        // Build plan: walk outputs, follow choices, emit only reachable kernels.
        let mut plan: Vec<(NodeId, ENode)> = Vec::new();
        let mut emitted: Set<ClassId> = Set::default();
        for &out in outputs {
            self.emit_plan(out, &choice, &mut plan, &mut emitted);
        }
        plan
    }

    /// Pick the cheapest kernel alternative for `cid`.
    ///
    /// Only considers `ENode::Kernel` enodes — non-kernel enodes (operator nodes
    /// like Binary, Cast, Expand, etc.) are not valid execution targets because
    /// the plan must be a sequence of compilable kernels, not raw graph operations.
    ///
    /// # Invariant
    /// This function panics if `cid` has no kernel or Leaf alternatives
    /// (`Const` is always compiled into kernels as a literal, never loaded from memory).
    /// Every class on the execution path must be kernelized — see `kernelize_all`.
    fn extract_dp(&mut self, cid: ClassId, cost_map: &mut Map<ClassId, u64>, choice: &mut Map<ClassId, NodeId>) -> u64 {
        let cid = self.find_class(cid);
        if let Some(&cost) = cost_map.get(&cid) {
            return cost;
        }

        let enodes: Vec<NodeId> = self.classes[cid].nodes.clone();
        let mut best_cost: u64 = u64::MAX;
        let mut best_nid: Option<NodeId> = None;

        for &nid in &enodes {
            // Non-kernel enodes are not executable — skip them.
            if !self.nodes[nid].is_kernel() {
                continue;
            }

            let children: Vec<ClassId> = self.nodes[nid].child_classes();
            let mut total = self.costs.get(&nid).copied().unwrap_or(u64::MAX);
            for &child in &children {
                let child_cost = self.extract_dp(child, cost_map, choice);
                total = total.saturating_add(child_cost);
            }

            if total < best_cost {
                best_cost = total;
                best_nid = Some(nid);
            }
        }

        if let Some(nid) = best_nid {
            cost_map.insert(cid, best_cost);
            choice.insert(cid, nid);
            return best_cost;
        }

        // The plan can only contain compiled kernels — non-kernel enodes (operator
        // nodes like Binary, Cast, Expand, etc.) are not executable targets.
        // Fall back to Leaf base cases which cost nothing to load.
        // (Const is never reached here — it is always compiled into kernels as a literal.)
        if self.classes[cid]
            .nodes
            .iter()
            .any(|&n| matches!(&self.nodes[n], ENode::Leaf(_) | ENode::Const(_)))
        {
            cost_map.insert(cid, 0);
            return 0;
        }

        // PANIC: This class is reachable from an output but has no kernel alternative.
        // Every class on the execution path must have at least one compiled kernel.
        // This means `kernelize_all` missed this class — it must be fixed.
        panic!(
            "extract_dp: class {cid:?} has no Kernel or Leaf alternative. \
             Non-kernel enodes cannot appear in the extracted plan — only compiled \
             kernels are executable. (Const is always compiled into kernels as a \
             literal, so it should never be missing a kernel.) Fix: kernelize_all \
             must create at least one kernel enode for this class."
        );
    }

    /// Walk choices from output classes, emitting kernels bottom-up
    /// (children before parents).
    ///
    /// # Invariant
    /// All enodes in `choice` are `ENode::Kernel` — non-kernel enodes are never
    /// selected by `extract_dp`. The returned plan is 100% compilable kernels.
    fn emit_plan(
        &mut self,
        cid: ClassId,
        choice: &Map<ClassId, NodeId>,
        plan: &mut Vec<(NodeId, ENode)>,
        emitted: &mut Set<ClassId>,
    ) {
        let cid = self.find_class(cid);
        if !emitted.insert(cid) {
            return;
        }
        let &nid = match choice.get(&cid) {
            Some(nid) => nid,
            None => return, // external input (Leaf), no kernel to emit
        };
        let children: Vec<ClassId> = self.nodes[nid].child_classes();
        for &child in &children {
            self.emit_plan(child, choice, plan, emitted);
        }
        plan.push((nid, self.nodes[nid].clone()));
    }

    // ── Compile (orchestrate build → saturate → autotune → extract) ─

    pub(crate) fn compile(
        inputs: &[TensorId],
        output_order: &[TensorId],
        order: &[TensorId],
        graph: &Graph,
        devices: &mut Slab<DeviceId, Device>,
        pools: &mut Slab<PoolId, MemoryPool>,
        cache: &mut KernelCache,
        config: &AutotuneConfig,
        debug: DebugMask,
    ) -> (Vec<CompiledNode>, Vec<BufferSlot>) {
        let mut eg = Self::new();
        let tensor_to_cid = eg.build_from_graph(order, graph);
        eg.saturate();
        eg.compress_paths();
        eg.kernelize_all();

        // Autotune every kernel variant on every available device.
        let _ = eg.autotune_all_kernels(devices, pools, cache, config, debug);

        let output_classes: Vec<ClassId> = output_order.iter().filter_map(|tid| tensor_to_cid.get(tid).copied()).collect();
        debug_assert!(!output_classes.is_empty(), "compile: no output tensors found in e-graph");

        eg.debug_print();

        // Extract: pick cheapest all-kernel plan
        let plan = eg.extract(&output_classes);
        if debug.sched() {
            eg.debug_print_plan(&plan);
        }
        eg.plan_to_compiled(inputs, output_order, &tensor_to_cid, &plan, devices)
    }

    /// Convert an extracted kernel plan to a sequence of [`CompiledNode`]s with
    /// buffer slot management.
    ///
    /// The plan is already in bottom-up topological order (children before
    /// parents).  Each kernel references input/output [`ClassId`]s.  We assign
    /// a [`BufferSlot`] to every class that appears in the plan, then emit:
    ///
    ///   • [`CompiledNode::Leaf`] for each input tensor (in caller order),
    ///   • [`CompiledNode::Allocate`] for every intermediate/output class,
    ///   • [`CompiledNode::LaunchProgram`] for each kernel.
    ///
    /// Returns the compiled nodes plus the output [`BufferSlot`]s in the same
    /// order as `output_order` (no TensorIds — the caller matches by position).
    fn plan_to_compiled(
        &mut self,
        inputs: &[TensorId],
        output_order: &[TensorId],
        tensor_to_cid: &Map<TensorId, ClassId>,
        plan: &[(NodeId, ENode)],
        devices: &Slab<DeviceId, Device>,
    ) -> (Vec<CompiledNode>, Vec<BufferSlot>) {
        // 1. Collect every class referenced in the plan.
        let mut all_classes: Vec<ClassId> = Vec::new();
        let mut seen: Set<ClassId> = Set::default();
        for (_, enode) in plan {
            if let ENode::Kernel(inputs, outputs, _) = enode {
                for &cid in inputs.iter() {
                    if seen.insert(cid) {
                        all_classes.push(cid);
                    }
                }
                for &cid in outputs.iter() {
                    if seen.insert(cid) {
                        all_classes.push(cid);
                    }
                }
            }
        }

        // 2. Identify input classes (Leaf/Const enodes — not produced by a kernel).
        let input_cids: Set<ClassId> = inputs.iter().filter_map(|tid| tensor_to_cid.get(tid).copied()).collect();

        // 3. Assign a BufferSlot to every class.
        let mut class_to_slot: Map<ClassId, BufferSlot> = Map::default();
        for (i, &cid) in all_classes.iter().enumerate() {
            class_to_slot.insert(cid, BufferSlot(i as u32));
        }

        let mut nodes: Vec<CompiledNode> = Vec::new();
        let mut allocated: Set<ClassId> = Set::default();
        let mut output_slots: Vec<BufferSlot> = Vec::new();

        // 4. Leaf nodes for input tensors (in caller order, one per input).
        for &tid in inputs {
            if let Some(&cid) = tensor_to_cid.get(&tid) {
                let cid = self.find_class(cid);
                if let Some(&slot) = class_to_slot.get(&cid) {
                    nodes.push(CompiledNode::Leaf {
                        pool: PoolId::from(0usize),
                        slot,
                    });
                }
            }
        }

        // 5. Process each kernel in plan order.
        for (_, enode) in plan {
            let (kernel_inputs, kernel_outputs, prog) = match enode {
                ENode::Kernel(inputs, outputs, prog) => (inputs, outputs, *prog),
                _ => continue,
            };

            // Allocate outputs that haven't been allocated yet.
            for &out_cid in kernel_outputs.iter() {
                let out_cid = self.find_class(out_cid);
                if !input_cids.contains(&out_cid) && allocated.insert(out_cid) {
                    let shape = self.classes[out_cid].shape.as_ref().unwrap();
                    let dtype = self.classes[out_cid].dtype.unwrap();
                    let elem_count: Dim = shape.iter().product();
                    let bytes = elem_count * Dim::from(dtype.bit_size() / 8);
                    let pool_id = devices[prog.device].memory_pool_id();
                    nodes.push(CompiledNode::Allocate {
                        size: bytes,
                        pool: pool_id,
                        slot: class_to_slot[&out_cid],
                    });
                }
            }

            // Build launch args: inputs then outputs.
            let mut args: Vec<BufferSlot> = Vec::with_capacity(kernel_inputs.len() + kernel_outputs.len());
            for &in_cid in kernel_inputs.iter() {
                let in_cid = self.find_class(in_cid);
                args.push(class_to_slot[&in_cid]);
            }
            for &out_cid in kernel_outputs.iter() {
                let out_cid = self.find_class(out_cid);
                args.push(class_to_slot[&out_cid]);
            }
            nodes.push(CompiledNode::LaunchProgram { program: prog, args });
        }

        // 6. Output slots in the same order as output_order.
        for &tid in output_order {
            if let Some(&cid) = tensor_to_cid.get(&tid) {
                let cid = self.find_class(cid);
                if let Some(&slot) = class_to_slot.get(&cid) {
                    if !input_cids.contains(&cid) {
                        output_slots.push(slot);
                    }
                }
            }
        }

        (nodes, output_slots)
    }

    /// Compile every kernel enode on every available device.
    /// Inserts device-specific kernel enodes into each class with real timing costs.
    fn autotune_all_kernels(
        &mut self,
        devices: &mut Slab<DeviceId, Device>,
        pools: &mut Slab<PoolId, MemoryPool>,
        cache: &mut KernelCache,
        config: &AutotuneConfig,
        debug: DebugMask,
    ) -> Result<(), ZyxError> {
        let kernel_enodes: Vec<(NodeId, ClassId, Box<[ClassId]>, Box<[ClassId]>)> = {
            let mut v = Vec::new();
            for cid in self.classes.ids() {
                for &nid in &self.classes[cid].nodes {
                    if let ENode::Kernel(inputs, outputs, _) = &self.nodes[nid] {
                        v.push((nid, cid, inputs.clone(), outputs.clone()));
                    }
                }
            }
            v
        };

        for (nid, cid, inputs, outputs) in &kernel_enodes {
            let Some(kernel) = self.kernel_irs.remove(nid) else {
                // Pre-compiled kernel (custom kernel) — keep it in its class
                // and use the heuristic cost already set in build_from_graph.
                continue;
            };

            // Remove the un-compiled original from its class.
            if let Some(class) = self.classes.get_mut(*cid) {
                class.nodes.retain(|&n| n != *nid);
            }
            let heuristic_cost = self.costs.get(nid).copied().unwrap_or(u64::MAX);

            let device_ids: Vec<DeviceId> = devices.ids().collect();
            for dev_id in device_ids {
                let device = &mut devices[dev_id];
                let pool_id = device.memory_pool_id();
                let pool = &mut pools[pool_id];

                let mut kernel = kernel.clone();
                let debug_kernel = kernel.clone();
                let (flop, read, write) = kernel.flop_mem_rw();

                if let Ok((device_prog, timing)) =
                    cache.get_or_autotune(dev_id, &mut kernel, device, pool, config, flop, read, write, debug)
                {
                    let prog = ProgramId {
                        device: dev_id,
                        program: device_prog,
                    };
                    let (new_nid, _) = self.make(ENode::Kernel(inputs.clone(), outputs.clone(), prog));
                    let cost = if timing > 0 { timing } else { heuristic_cost };
                    self.costs.insert(new_nid, cost);
                    self.kernel_irs.insert(new_nid, debug_kernel);
                    self.add_to_class(new_nid, *cid);
                }
            }
        }

        Ok(())
    }

    pub(crate) fn debug_print_plan(&self, plan: &[(NodeId, ENode)]) {
        let line = "─".repeat(60);
        println!("\n{}", line);
        println!("  Extracted Plan with {} auto kernels", self.kernel_irs.len());
        println!("{}", line);
        for (i, (nid, enode)) in plan.iter().enumerate() {
            if let ENode::Kernel(inputs, outputs, prog) = enode {
                let cost = self.costs.get(nid).copied().unwrap_or(0);
                println!(
                    "Kernel {} n{} d{}:  cost={}, inputs={:?} outputs={:?}",
                    i, nid.0, prog.device.0, cost, inputs, outputs
                );
                if let Some(kernel) = self.kernel_irs.get(nid) {
                    kernel.debug();
                }
            } else {
                println!("  Step {}: {:?} (non-kernel)", i, enode);
            }
        }
        if plan.is_empty() {
            println!("  (empty plan)");
        }
        println!("{}\n", line);
    }

    pub(crate) fn debug_print(&self) {
        let line = "─".repeat(60);
        println!("\n{}", line);
        println!("  E-Graph");
        println!("{}", line);
        for cid in self.classes.ids() {
            // Skip dead classes left by `make` (not a UF root).
            if self.class_parent[cid.0 as usize] != cid {
                continue;
            }
            let eclass = &self.classes[cid];
            let shape_str = match &eclass.shape {
                Some(s) => format!("{:?}", s),
                None => "?".to_string(),
            };
            let dtype_str = match &eclass.dtype {
                Some(dt) => format!("{:?}", dt),
                None => "?".to_string(),
            };
            println!("Class {:?} shape={} dtype={}", cid, shape_str, dtype_str);
            for &nid in &eclass.nodes {
                let kind = &self.nodes[nid];
                let inputs = kind.child_classes();
                let name = match kind {
                    ENode::Reduce(_, rop) => format!("Reduce {:?}", rop),
                    ENode::Binary(_, _, bop) => format!("Binary {:?}", bop),
                    ENode::Unary(_, uop) => format!("Unary {:?}", uop),
                    ENode::Cast(_, dt) => format!("Cast {:?}", dt),
                    ENode::Kernel(_, _, p) => format!("Kernel prog={:?}", p),
                    ENode::Expand(_) => format!("Expand"),
                    ENode::Permute(_, a) => format!("Permute {:?}", a),
                    ENode::Reshape(_, s) => format!("Reshape {:?}", s),
                    ENode::Pad(_, p) => format!("Pad {:?}", p),
                    ENode::ToDevice(_, d) => format!("ToDevice {:?}", d),
                    ENode::Const(v) => format!("Const {:?}", v),
                    ENode::Leaf(dt) => format!("Leaf {:?}", dt),
                };
                println!("  {name} {nid:?}: inputs={inputs:?}");
            }
        }
        println!("{}\n", line);
    }
}
