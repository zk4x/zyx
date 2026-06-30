// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! E-graph-based kernel search.
//!
//! Builds an e-graph from the computation graph, enumerates kernel alternatives
//! via pattern matching, then extracts the cheapest execution plan.

#![allow(unused)]

use crate::{
    DType, DebugMask, Map, Set,
    backend::ProgramId,
    dtype::Constant,
    graph::{Node, compiled::CompiledNode},
    kernel::{BOp, DeviceId, UOp},
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
};

use super::Graph;

// ── IDs ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct NodeId(pub u32);

impl NodeId {
    pub(crate) const fn null() -> Self {
        Self(u32::MAX)
    }
}

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

impl ClassId {
    pub(crate) const fn null() -> Self {
        Self(u32::MAX)
    }
}

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

// ── ENode — the e-graph language ──────────────────────────────

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

    pub(crate) fn is_transform(&self) -> bool {
        matches!(
            self,
            ENode::Expand(_) | ENode::Permute(_, _) | ENode::Reshape(_, _) | ENode::Pad(_, _) | ENode::Cast(_, _)
        )
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
    class_of: Vec<ClassId>,
    class_parent: Vec<ClassId>,
    class_rank: Vec<u8>,
    hashcons: Map<ENode, NodeId>,
    pub(crate) costs: Map<NodeId, u64>,
    pub(crate) ops_count: Map<NodeId, u32>,
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
            ops_count: Map::default(),
        }
    }

    // ── Union-Find ─────────────────────────────────────────

    pub(crate) fn find_class(&self, cid: ClassId) -> ClassId {
        let parent = self.class_parent[cid.0 as usize];
        if parent != cid { self.find_class(parent) } else { cid }
    }

    pub(crate) fn find(&self, nid: NodeId) -> ClassId {
        let cid = self.class_of[nid.0 as usize];
        self.find_class(cid)
    }

    fn compress_class_paths(&mut self, cid: ClassId) -> ClassId {
        let parent = self.class_parent[cid.0 as usize];
        if parent != cid {
            let root = self.compress_class_paths(parent);
            self.class_parent[cid.0 as usize] = root;
            root
        } else {
            cid
        }
    }

    pub(crate) fn compress_paths(&mut self) {
        let cids: Vec<ClassId> = self.classes.ids().collect();
        for &cid in &cids {
            self.compress_class_paths(cid);
        }
    }

    fn grow_uf_arrays(&mut self, idx: usize) {
        if idx >= self.class_of.len() {
            let cid = ClassId(self.class_of.len() as u32);
            self.class_of.resize(idx + 1, cid);
            self.class_parent.resize(idx + 1, cid);
            self.class_rank.resize(idx + 1, 0);
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
        let cid = self.find_class(cid);
        let old_cid = self.find(nid);
        if old_cid == cid {
            return;
        }
        self.class_of[nid.0 as usize] = cid;
        self.classes[cid].nodes.push(nid);

        let children: Vec<ClassId> = self.nodes[nid].child_classes();
        for (child_idx, &child) in children.iter().enumerate() {
            let child_root = self.find_class(child);
            self.classes[child_root].parents.push((nid, child_idx));
        }
    }

    // ── Union two e-classes ───────────────────────────────

    pub(crate) fn union(&mut self, a: ClassId, b: ClassId) {
        let a = self.find_class(a);
        let b = self.find_class(b);
        if a == b {
            return;
        }

        let (keep, drop) = if self.class_rank[a.0 as usize] >= self.class_rank[b.0 as usize] {
            (a, b)
        } else {
            (b, a)
        };

        self.class_parent[drop.0 as usize] = keep;
        if self.class_rank[keep.0 as usize] == self.class_rank[drop.0 as usize] {
            self.class_rank[keep.0 as usize] += 1;
        }

        let drop_nodes: Vec<NodeId> = self.classes[drop].nodes.drain(..).collect();
        for &nid in &drop_nodes {
            self.class_of[nid.0 as usize] = keep;
            self.classes[keep].nodes.push(nid);
        }

        let drop_parents = std::mem::take(&mut self.classes[drop].parents);
        if !drop_parents.is_empty() {
            self.classes[keep].parents.extend(drop_parents);
        }

        if self.classes[keep].shape.is_none() {
            self.classes[keep].shape = self.classes[drop].shape.take();
        }
        if self.classes[keep].dtype.is_none() {
            self.classes[keep].dtype = self.classes[drop].dtype.take();
        }
    }

    // ── Build from Graph ─────────────────────────────────

    pub(crate) fn build_from_graph(&mut self, order: &[TensorId], graph: &Graph) -> Map<TensorId, ClassId> {
        let mut tensor_to_cid: Map<TensorId, ClassId> = Map::default();

        for &tid in order {
            let kind = self.node_to_enkind(tid, &tensor_to_cid, graph);
            let (_, cid) = self.make(kind);
            let cid = self.find_class(cid);
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
            Node::Custom(_) => panic!("Custom nodes not supported in e-graph"),
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

                // Check the reduce is over the last dimension.
                let shape = match &self.classes[cid].shape {
                    Some(s) => s.clone(),
                    None => continue,
                };
                let mul_shape = match &self.classes[self.find_class(mul_class)].shape {
                    Some(s) => s.clone(),
                    None => continue,
                };
                // Reduce must contract one axis — the last axis of the mul output.
                if mul_shape.len() < 2 {
                    continue;
                }
                let contracted = mul_shape.len() - 1;
                // shape after reduction should be mul_shape without the contracted dim
                if shape.len() != mul_shape.len() - 1 {
                    continue;
                }

                // Check mul_class contains Binary(Mul, ...)
                let mul_nodes: Vec<NodeId> = self.classes[self.find_class(mul_class)].nodes.clone();
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
                    let a_raw = match self.classes[self.find_class(expand_a)]
                        .nodes
                        .iter()
                        .find_map(|&n| match &self.nodes[n] {
                            ENode::Expand(x) => Some(*x),
                            _ => None,
                        }) {
                        Some(x) => x,
                        None => continue,
                    };

                    // B side: Expand → Reshape → Permute[1,0] → Leaf
                    let reshape_class =
                        match self.classes[self.find_class(expand_b)]
                            .nodes
                            .iter()
                            .find_map(|&n| match &self.nodes[n] {
                                ENode::Expand(x) => Some(*x),
                                _ => None,
                            }) {
                            Some(x) => x,
                            None => continue,
                        };

                    let permute_class = match self.classes[self.find_class(reshape_class)]
                        .nodes
                        .iter()
                        .find_map(|&n| match &self.nodes[n] {
                            ENode::Reshape(x, _) => Some(*x),
                            _ => None,
                        }) {
                        Some(x) => x,
                        None => continue,
                    };

                    let b_raw =
                        match self.classes[self.find_class(permute_class)]
                            .nodes
                            .iter()
                            .find_map(|&n| match &self.nodes[n] {
                                ENode::Permute(x, axes) if axes.len() == 2 && axes[0] == 1 && axes[1] == 0 => Some(*x),
                                _ => None,
                            }) {
                            Some(x) => x,
                            None => continue,
                        };

                    // Verify inputs are leaves
                    if !self.classes[self.find_class(a_raw)]
                        .nodes
                        .iter()
                        .any(|&n| matches!(&self.nodes[n], ENode::Leaf(_)))
                    {
                        continue;
                    }
                    if !self.classes[self.find_class(b_raw)]
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
    /// Returns the selected kernel enodes in topological order.
    /// Panics if any class on the path has no Kernel alternative.
    pub(crate) fn extract(&mut self, outputs: &[ClassId]) -> Vec<ENode> {
        let mut cost_map: Map<ClassId, u64> = Map::default();
        let mut choice: Map<ClassId, NodeId> = Map::default();

        for &out in outputs {
            self.extract_dp(out, &mut cost_map, &mut choice);
        }

        // Build plan: walk outputs, follow choices, emit only reachable kernels.
        let mut plan: Vec<ENode> = Vec::new();
        let mut emitted: Set<ClassId> = Set::default();
        for &out in outputs {
            self.emit_plan(out, &choice, &mut plan, &mut emitted);
        }
        plan
    }

    fn extract_dp(
        &mut self,
        cid: ClassId,
        cost_map: &mut Map<ClassId, u64>,
        choice: &mut Map<ClassId, NodeId>,
    ) -> u64 {
        let cid = self.find_class(cid);
        if let Some(&cost) = cost_map.get(&cid) {
            return cost;
        }

        let enodes: Vec<NodeId> = self.classes[cid].nodes.clone();
        let mut best_cost: u64 = u64::MAX;
        let mut best_nid: Option<NodeId> = None;

        for &nid in &enodes {
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

        let nid = best_nid.expect(
            &format!("extract_dp: class {cid:?} has no Kernel alternative — kernelizer must fix this")
        );
        cost_map.insert(cid, best_cost);
        choice.insert(cid, nid);
        best_cost
    }

    /// Walk choices from output classes, emitting kernels bottom-up
    /// (children before parents).
    fn emit_plan(
        &self,
        cid: ClassId,
        choice: &Map<ClassId, NodeId>,
        plan: &mut Vec<ENode>,
        emitted: &mut Set<ClassId>,
    ) {
        let cid = self.find_class(cid);
        if !emitted.insert(cid) {
            return;
        }
        let &nid = choice.get(&cid).expect("extract: no kernel chosen for class");
        let children: Vec<ClassId> = self.nodes[nid].child_classes();
        for &child in &children {
            self.emit_plan(child, choice, plan, emitted);
        }
        plan.push(self.nodes[nid].clone());
    }

    // ── Compile (orchestrate build → saturate → extract) ─

    pub(crate) fn compile(
        inputs: &[TensorId],
        to_eval: &Set<TensorId>,
        order: &[TensorId],
        graph: &Graph,
        debug: DebugMask,
    ) -> Vec<CompiledNode> {
        let mut eg = Self::new();
        let tensor_to_cid = eg.build_from_graph(order, graph);
        eg.saturate();
        eg.kernelize_all();

        let output_classes: Vec<ClassId> = to_eval.iter().filter_map(|tid| tensor_to_cid.get(tid).copied()).collect();
        if output_classes.is_empty() {
            return Vec::new();
        }

        // Extract: pick cheapest all-kernel plan
        let plan = eg.extract(&output_classes);
        if debug.sched() {
            eg.debug_print();
            eg.debug_print_plan(&plan);
        }
        // TODO: convert plan (Vec<ENode>) to Vec<CompiledNode> with buffer slot management
        Vec::new()
    }

    pub(crate) fn debug_print_plan(&self, plan: &[ENode]) {
        let line = "─".repeat(60);
        println!("\n{}", line);
        println!("  Extracted Plan");
        println!("{}", line);
        for (i, enode) in plan.iter().enumerate() {
            if let ENode::Kernel(inputs, outputs, prog) = enode {
                let nid = self.hashcons.get(enode).copied().unwrap_or(NodeId::NULL);
                let ops = self.ops_count.get(&nid).copied().unwrap_or(0);
                let cost = self.costs.get(&nid).copied().unwrap_or(0);
                println!(
                    "  Kernel {}: {} fused ops, cost={}, inputs={:?} outputs={:?}",
                    i, ops, cost, inputs, outputs
                );
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
                let (name, extra) = match kind {
                    ENode::Reduce(_, rop) => ("Reduce", format!("{:?}", rop)),
                    ENode::Binary(_, _, bop) => ("Binary", format!("{:?}", bop)),
                    ENode::Unary(_, uop) => ("Unary", format!("{:?}", uop)),
                    ENode::Cast(_, dt) => ("Cast", format!("{:?}", dt)),
                    ENode::Kernel(_, _, p) => ("Kernel", format!("prog={:?}", p)),
                    ENode::Expand(_) => ("Expand", String::new()),
                    ENode::Permute(_, a) => ("Permute", format!("{:?}", a)),
                    ENode::Reshape(_, s) => ("Reshape", format!("{:?}", s)),
                    ENode::Pad(_, p) => ("Pad", format!("{:?}", p)),
                    ENode::ToDevice(_, d) => ("ToDevice", format!("{:?}", d)),
                    ENode::Const(v) => ("Const", format!("{:?}", v)),
                    ENode::Leaf(dt) => ("Leaf", format!("{:?}", dt)),
                };
                println!("  {name} {:?}: inputs={:?} {}", nid, inputs, extra);
            }
        }
        println!("{}\n", line);
    }
}


