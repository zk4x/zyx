// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! E-graph kernelizer.
//!
//! This module MUST mirror the fusion/launch strategy of `kernelize.rs`,
//! operating on e-graph classes (`ClassId`) instead of graph tensors
//! (`TensorId`).  Below is how `kernelize.rs` works — this module must
//! do the same, adapted to the e-graph.
//!
//! ── How kernelize.rs works ──────────────────────────────────
//!
//! The kernelizer processes tensors in topological order (children
//! before parents).  It maintains two key data structures:
//!
//! * `visited: Map<TensorId, (KMKernelId, OpId)>`
//!   Every tensor is inserted into `visited` after being processed.
//!   Maps each tensor → (which kernel it lives in, the OpId of its
//!   result within that kernel).
//!
//! * `pending_stores: Set<TensorId>`
//!   Tracks tensors that have been stored (have a Store op somewhere).
//!   NOT the same as `visited` — a tensor can be in `visited` without
//!   being in `pending_stores` (its computation is in a kernel but has
//!   not yet been stored to memory).
//!
//! Each kernel has an `outputs: Vec<TensorId>` field.  When a tensor is
//! added to a kernel (as the result of a compute op), it is appended to
//! `outputs` (once for each reference count).  When a tensor is used as
//! an input to a later op, `remove_first_output` removes one occurrence.
//! A tensor still in `outputs` is "live" — its value exists only as a
//! register in the kernel, not yet stored to global memory.
//!
//! ## Processing a tensor (pseudocode)
//!
//! ```text
//! for each nid in order:
//!     if nid is in pending_stores:
//!         create_load_kernel(nid)   // new kernel with just a Load
//!     else:
//!         match graph[nid]:
//!             Leaf/Const → handled directly
//!             Unary  → add_unary_op(nid, x, uop)
//!             Binary → add_binary_op(nid, x, y, bop)
//!             Expand → add_expand_op(nid, x)
//!             ...
//!
//!     if nid is output (to_eval) and not realized:
//!         add_store(nid)             // persist to memory
//! ```
//!
//! ## Kernel merging (the key to fusion)
//!
//! When `add_binary_op` processes a binary op, the operands x and y
//! may live in different kernels.  Instead of loading between them,
//! the kernels are **merged**: all ops from y's kernel are copied into
//! x's kernel (with OpIds remapped), and `visited` entries pointing to
//! y's kernel are updated to point to x's kernel.  After merging, both
//! operands and the result are in the same kernel.  The binary op is
//! appended, and the result tensor is added to `visited`.
//!
//! This is what produces fused kernels — single-use intermediate
//! tensors stay in the same kernel as their consumer; no load/store
//! boundary is created between them.
//!
//! ## Storing and launching
//!
//! `add_store(nid)` is called when a tensor needs to be persisted:
//!   - It is an output (`to_eval`)
//!   - It is used by multiple consumers (refcount > 1)
//!   - A Reduce needs its input materialised (reduction boundary)
//!   - Other heuristics from `duplicate_or_store`
//!
//! `add_store` removes the tensor from `visited`, adds a Store op,
//! inserts it into `pending_stores`, and removes it from the kernel's
//! `outputs` list.  If the kernel's `outputs` becomes empty AND all its
//! loads are already realized tensors, the kernel is launched
//! immediately — it cannot be enlarged any more because every
//! intermediate result has been stored.
//!
//! ## Adaptation to e-graph
//!
//! In this module, the SAME kernelizer architecture from `kernelize.rs` is
//! followed as closely as the e-graph allows:
//!
//!   - `ClassId` replaces `TensorId`
//!   - `visited: Map<ClassId, (KMKernelId, OpId)>` — EVERY class is
//!     inserted after processing.  Maps each class → (which kernel it
//!     lives in, the OpId of its result within that kernel).  Keyed by
//!     `ClassId`, not `NodeId`, because enodes reference child classes
//!     (e.g. `ENode::Binary(a_cid, b_cid, Add)`) and all enodes in a
//!     class are equivalent — the kernelizer just needs the child's
//!     kernel+op regardless of which variant produced it.  The e-graph
//!     makes decisions among kernel enodes; non-kernel enodes are just
//!     IR and don't need per-variant tracking.
//!   - Kernels live in `EGraph::kernel_irs: Map<KMKernelId, Kernel>`,
//!     a slab indexed by `KMKernelId`.  A kernel accumulates ops across
//!     MULTIPLE classes (not one per class).
//!   - When a parent class needs a child's value, it looks up
//!     `visited[child] = (kid, op_id)`.  If `kid == parent_kid`, the
//!     result register is already in the same kernel — use `op_id`
//!     directly.  If `kid != parent_kid`, the child's kernel is merged
//!     into the parent's (ops copied, OpIds remapped, visited updated).
//!     This is exactly how `kernelize.rs`'s `add_binary_op` merges
//!     kernels when `kid != kidy`.  This merge-or-same-kernel check
//!     is ONLY about kernel identity — it has nothing to do with
//!     outputs or whether the kernel is "done".
//!   - An `outputs: Map<KMKernelId, ClassId>` tracks which classes
//!     have been computed in each kernel but not yet stored.  When a
//!     child class is consumed by a parent in the same kernel
//!     (`kid == parent_kid`), it is removed from outputs.  When a
//!     class is stored (output, multi-consumer, reduce boundary), it
//!     is also removed from outputs.  When no more entries exist for
//!     a kernel, no new ops can be fused into it — it is finalized,
//!     added as `ENode::Kernel` to the e-graph, and removed from the
//!     builder slab.  Outputs are about fusion lifetime, not about
//!     merge decisions.
//!   - `EGraph::kernel_map: Map<NodeId, KMKernelId>` links each
//!     kernel enode (`ENode::Kernel`) back to its `KMKernelId` in the
//!     kernel builder slab.
//!   - Each class is processed ONCE and only once — no recursion.
//!     Processing mirrors the `add_*_op` methods of `kernelize.rs`:
//!     look up children in `visited`, merge if needed, emit the op
//!     into the kernel, update `outputs`, add self to `visited`,
//!     store if needed.
//!   - Reference counts (`rcs: Map<ClassId, u32>`) are computed directly
//!     from the e-graph, using the same topo-sort order used for
//!     processing.  Iterate the order and count each child occurrence
//!     across all enodes.  This mirrors how `kernelize.rs` computes
//!     `rcs` from the graph order.  `rcs` is decremented each time a
//!     child class is consumed as input by a parent op.  When the
//!     processing loop finishes and `rcs[cid] > 0`, the class still
//!     has consumers and needs to be stored (output, multi-consumer).
//!   - When a class has multiple enode variants (from rewrites), the
//!     accumulated kernel state (children's ops already merged in) is
//!     CLONED once per variant.  Each clone gets the variant's ops
//!     emitted into it, becomes its own `KMKernelId` and its own
//!     `ENode::Kernel` in the class.  Each clone updates
//!     `visited[class]` — the last one wins.  Since all variants
//!     compute the same value, any one works when a parent class
//!     looks up the child's kernel via `visited[child_class]`.
//!     The extractor picks the cheapest among all kernel enodes
//!     in the class.
//!   - `pending_stores` (from `kernelize.rs`) is NOT present in the
//!     e-graph version.  In `kernelize.rs` it prevents double-stores
//!     during merge — but in the e-graph a class is stored once and
//!     then removed from the builder slab; no second `add_store` is
//!     possible.  If a problem arises from not tracking pending
//!     stores, it will be fixed on the fly.

use crate::{
    Map, Set,
    backend::ProgramId,
    graph::search::{ClassId, EGraph, ENode, NodeId},
    kernel::{DeviceId, Kernel, MoveOp, Op, OpId},
    kernelize::KMKernelId,
    shape::{Dim, UAxis},
};

const HIGH_COST: u64 = 1_000_000;

/// Cost for a kernel based on number of compute operations.
fn kernel_cost(ops_count: u32) -> u64 {
    ((1.0 + ops_count as f64).ln() * HIGH_COST as f64) as u64
}

// ── Public entry point ─────────────────────────────────────

impl EGraph {
    /// Build fused kernels for classes in topological order.
    ///
    /// Mirrors `kernelize.rs`: walk classes in topological order,
    /// accumulating ops into kernels shared by MULTIPLE classes.
    /// Kernels are added as `ENode::Kernel` to the e-graph only when
    /// their output list becomes empty.
    pub(crate) fn kernelize_all(&mut self) {
        let order = topo_sort_classes(self);

        // Reference counts: how many times each class appears as a child.
        let mut rcs: Map<ClassId, u32> = Map::default();
        // Collect child classes first to avoid borrow conflicts.
        let child_lists: Vec<Vec<ClassId>> = order
            .iter()
            .map(|&cid| {
                let mut all = Vec::new();
                for &nid in &self.classes[cid].nodes {
                    all.extend(self.nodes[nid].child_classes());
                }
                all
            })
            .collect();
        for children in &child_lists {
            for &child in children {
                let root = self.find_class(child);
                *rcs.entry(root).or_default() += 1;
            }
        }

        let mut visited: Map<ClassId, (KMKernelId, OpId)> = Map::default();
        let mut outputs: Map<KMKernelId, Vec<ClassId>> = Map::default();
        let mut kernel_id_counter: u32 = 0;
        // Track which classes are loaded/stored by each kernel.
        let mut kernel_loads: Map<KMKernelId, Vec<ClassId>> = Map::default();
        let mut kernel_stores: Map<KMKernelId, Vec<ClassId>> = Map::default();

        for cid in order {
            // Leaf & Const classes: create a load kernel so parents can
            // reference their value.
            if self.is_leaf_or_const(cid) {
                let rc = rcs.get(&cid).copied().unwrap_or(0);
                if rc > 0 {
                    let kid = self.new_load_kernel(cid, &mut kernel_id_counter, &mut kernel_loads);
                    visited.insert(cid, (kid, kid_first_op(kid, &self.kernel_irs)));
                    outputs.entry(kid).or_default().push(cid);
                }
                continue;
            }

            // Gather non-Kernel, non-Leaf, non-Const enodes.
            let enodes: Vec<NodeId> = self.classes[cid]
                .nodes
                .iter()
                .copied()
                .filter(|&nid| {
                    !matches!(
                        self.nodes[nid],
                        ENode::Leaf(_) | ENode::Const(_) | ENode::Kernel(..)
                    )
                })
                .collect();

            if enodes.is_empty() {
                continue;
            }

            // ── Process the first enode variant (extends accumulated kernel) ──
            let mut child_kids: Vec<(KMKernelId, OpId)> = Vec::new();
            let children: Vec<ClassId> = self.nodes[enodes[0]].child_classes();
            // Find roots first to avoid borrow conflicts.
            let child_roots: Vec<ClassId> = children.iter().map(|&c| self.find_class(c)).collect();
            for &root in &child_roots {
                if let Some(&(kid, op)) = visited.get(&root) {
                    child_kids.push((kid, op));
                } else {
                    // Child not in visited (shouldn't happen for a valid
                    // topo order), create load kernel on the fly.
                    let kid = self.new_load_kernel(root, &mut kernel_id_counter, &mut kernel_loads);
                    let op = kid_first_op(kid, &self.kernel_irs);
                    visited.insert(root, (kid, op));
                    child_kids.push((kid, op));
                }
            }

            // Merge all child kernels into one.
            let target_kid = child_kids[0].0;
            for &(ckid, _) in &child_kids[1..] {
                if ckid != target_kid {
                    self.merge_kernels(ckid, target_kid, &mut visited, &mut kernel_loads, &mut kernel_stores);
                }
            }
            // After merge, re-read from visited (merge_kernels remapped OpIds).
            child_kids.clear();
            for &root in &child_roots {
                let &(kid, op) = visited.get(&root).unwrap_or_else(|| {
                    panic!("child root {root:?} not in visited after merge")
                });
                child_kids.push((kid, op));
            }

            // Now target_kid contains all children. Decrement rcs and outputs.
            for &root in &child_roots {
                if let Some(rc) = rcs.get_mut(&root) {
                    *rc = rc.saturating_sub(1);
                }
                if let Some(vec) = outputs.get_mut(&target_kid) {
                    if let Some(pos) = vec.iter().position(|&x| x == root) {
                        vec.remove(pos);
                    }
                }
            }

            // Emit the operation for this enode.
            let result_op = self.emit_enode(
                enodes[0],
                Some(cid),
                &children,
                &child_kids,
                target_kid,
            );
            debug_assert!(
                !result_op.is_null(),
                "emit_enode returned NULL op for cid={cid:?}, nid={:?}",
                enodes[0]
            );

            // Add result to outputs and visited.
            visited.insert(cid, (target_kid, result_op));
            outputs.entry(target_kid).or_default().push(cid);

            // Check if the class needs storage (output, multi, reduce).
            let remaining_rc = rcs.get(&cid).copied().unwrap_or(0);
            let is_output = remaining_rc > 0;
            let is_reduce = matches!(self.nodes[enodes[0]], ENode::Reduce(..));
            let same_kid_consumers: usize = child_kids.iter().filter(|&&(k, _)| k == target_kid).count();
            let multi_in_same_kid = same_kid_consumers > 1;
            let is_multi = child_kids.len() > 1 || multi_in_same_kid;

            if is_output || is_multi || is_reduce {
                self.add_store(cid, target_kid, result_op, &mut visited, &mut outputs, &mut kernel_id_counter, &kernel_loads, &mut kernel_stores);
            }

            // ── Process subsequent enode variants (cloned per variant) ──
            // TODO: implement after first-variant path is working.
        }
    }

    // ── Helpers ─────────────────────────────────────────────

    fn is_leaf_or_const(&self, cid: ClassId) -> bool {
        self.classes[cid].nodes.iter().any(|&nid| {
            matches!(self.nodes[nid], ENode::Leaf(_) | ENode::Const(_))
        })
    }

    fn new_load_kernel(
        &mut self,
        cid: ClassId,
        counter: &mut u32,
        kernel_loads: &mut Map<KMKernelId, Vec<ClassId>>,
    ) -> KMKernelId {
        let mut kernel = Kernel::new(DeviceId::AUTO);
        let shape: Vec<Dim> = self.classes[cid].shape.to_vec();
        let _load_op = kernel.load_contiguous(self.classes[cid].dtype, &shape);
        // Outputs tracked separately via the `outputs` map.
        let kid = KMKernelId::from(*counter as usize);
        *counter += 1;
        self.kernel_irs.insert(kid, kernel);
        kernel_loads.entry(kid).or_default().push(cid);
        kid
    }

    fn merge_kernels(
        &mut self,
        src: KMKernelId,
        dst: KMKernelId,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        kernel_loads: &mut Map<KMKernelId, Vec<ClassId>>,
        kernel_stores: &mut Map<KMKernelId, Vec<ClassId>>,
    ) {
        debug_assert!(src != dst, "merge_kernels: self-merge src={src:?}==dst");
        debug_assert!(
            self.kernel_irs.contains_key(&dst),
            "merge_kernels: dst={dst:?} not in kernel_irs"
        );
        debug_assert!(
            self.kernel_irs.contains_key(&src),
            "merge_kernels: src={src:?} not in kernel_irs (removed already?)"
        );
        let src_kernel = self.kernel_irs.remove(&src).unwrap();
        // Merge load/store tracking.
        if let Some(loads) = kernel_loads.remove(&src) {
            kernel_loads.entry(dst).or_default().extend(loads);
        }
        if let Some(stores) = kernel_stores.remove(&src) {
            kernel_stores.entry(dst).or_default().extend(stores);
        }
        let mut op_map: Map<OpId, OpId> = Map::default();
        let mut i = src_kernel.head;
        while !i.is_null() {
            let mut op = src_kernel.ops[i].op.clone();
            for param in op.parameters_mut() {
                if !param.is_null() {
                    *param = op_map[param];
                }
            }
            let new_id = self.kernel_irs.get_mut(&dst).unwrap().push_back(op);
            op_map.insert(i, new_id);
            i = src_kernel.ops[i].next;
        }
        // Update visited entries that pointed to src
        for (_, (kid, op_id)) in visited.iter_mut() {
            if *kid == src {
                *kid = dst;
                if let Some(&new_op) = op_map.get(op_id) {
                    *op_id = new_op;
                }
            }
        }
    }

    fn emit_enode(
        &mut self,
        nid: NodeId,
        cid_override: Option<ClassId>,
        children: &[ClassId],
        child_results: &[(KMKernelId, OpId)],
        target_kid: KMKernelId,
    ) -> OpId {
        let kind = self.nodes[nid].clone();
        // Pre-compute values that need &self before borrowing kernel_irs mutably.
        let expand_shape = if matches!(kind, ENode::Expand(..)) {
            let cid = cid_override.unwrap_or_else(|| self.find(nid));
            Some(self.classes[cid].shape.to_vec())
        } else {
            None
        };
        let (permute_shape, permute_axes) = if let ENode::Permute(_, ref axes) = kind {
            let cid = cid_override.unwrap_or_else(|| self.find(nid));
            (Some(self.classes[cid].shape.to_vec()), Some(axes.clone().into_vec()))
        } else {
            (None, None)
        };
        let pad_shape_and_padding = if let ENode::Pad(_, ref padding) = kind {
            let cid = cid_override.unwrap_or_else(|| self.find(nid));
            Some((self.classes[cid].shape.to_vec(), padding.clone().into_vec()))
        } else {
            None
        };
        let reduce_data = if let ENode::Reduce(_, rop) = &kind {
            let in_root = self.find_class(children[0]);
            let in_shape: Vec<Dim> = self.classes[in_root].shape.to_vec();
            let cid = cid_override.unwrap_or_else(|| self.find(nid));
            let out_shape: Vec<Dim> = self.classes[cid].shape.to_vec();
            let n_axes = in_shape.len().saturating_sub(out_shape.len());
            Some((*rop, in_shape, out_shape, n_axes))
        } else {
            None
        };
        // Now borrow kernel mutably.
        let kernel = self.kernel_irs.get_mut(&target_kid).unwrap();
        match kind {
            ENode::Expand(..) => {
                let shape = expand_shape.unwrap();
                let x = child_results[0].1;
                kernel.push_back(Op::Move {
                    x,
                    mop: Box::new(MoveOp::Expand { shape }),
                })
            }
            ENode::Permute(..) => {
                let shape = permute_shape.unwrap();
                let axes = permute_axes.unwrap();
                let x = child_results[0].1;
                kernel.push_back(Op::Move {
                    x,
                    mop: Box::new(MoveOp::Permute { axes, shape }),
                })
            }
            ENode::Reshape(_, shape) => {
                let shape: Vec<Dim> = shape.into_vec();
                let x = child_results[0].1;
                kernel.push_back(Op::Move {
                    x,
                    mop: Box::new(MoveOp::Reshape { shape }),
                })
            }
            ENode::Pad(..) => {
                let (shape, padding) = pad_shape_and_padding.unwrap();
                let x = child_results[0].1;
                kernel.push_back(Op::Move {
                    x,
                    mop: Box::new(MoveOp::Pad { padding, shape }),
                })
            }
            ENode::Reduce(..) => {
                let (rop, in_shape, out_shape, n_axes) = reduce_data.unwrap();
                let x = child_results[0].1;
                let permuted = try_permute_reduce_axes(kernel, x, &in_shape, &out_shape, n_axes);
                let r = kernel.push_back(Op::Reduce {
                    x: permuted,
                    rop,
                    n_axes: n_axes as UAxis,
                });
                if out_shape.len() == 1 && n_axes > 0 && in_shape.len() > 1 {
                    kernel.reshape(r, &out_shape);
                }
                r
            }
            ENode::Cast(_, dt) => {
                let x = child_results[0].1;
                kernel.cast(x, dt)
            }
            ENode::Unary(_, uop) => {
                let x = child_results[0].1;
                kernel.push_back(Op::Unary { x, uop })
            }
            ENode::Binary(_, _, bop) => {
                let lhs = child_results[0].1;
                let rhs = child_results[1].1;
                kernel.binary(lhs, rhs, bop)
            }
            _ => unreachable!(),
        }
    }

    fn add_store(
        &mut self,
        cid: ClassId,
        kid: KMKernelId,
        op_id: OpId,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        outputs: &mut Map<KMKernelId, Vec<ClassId>>,
        _counter: &mut u32,
        kernel_loads: &Map<KMKernelId, Vec<ClassId>>,
        kernel_stores: &mut Map<KMKernelId, Vec<ClassId>>,
    ) {
        debug_assert!(self.kernel_irs.contains_key(&kid), "add_store: kid={kid:?} not in kernel_irs");
        debug_assert!(!op_id.is_null(), "add_store: NULL op_id for cid={cid:?}");
        let dtype = self.classes[cid].dtype;
        let kernel = self.kernel_irs.get_mut(&kid).unwrap();
        kernel.store_contiguous(op_id, dtype);
        kernel_stores.entry(kid).or_default().push(cid);

        // Remove from visited — stored classes are loaded, not merged.
        visited.remove(&cid);

        // Remove from outputs.
        if let Some(vec) = outputs.get_mut(&kid) {
            vec.retain(|&x| x != cid);
            if vec.is_empty() {
                // Kernel is done — add as ENode::Kernel to the e-graph.
                // Remove ALL visited entries pointing to this kernel so
                // their stale (kid, op) pairs don't get reused later.
                visited.retain(|_, &mut (k, _)| k != kid);
                let owned_kernel = self.kernel_irs.remove(&kid).unwrap();
                let input_cids = kernel_loads.get(&kid).cloned().unwrap_or_default();
                let output_cids = kernel_stores.get(&kid).cloned().unwrap_or_default();
                debug_assert!(
                    !input_cids.iter().any(|&c| c.0 == u32::MAX),
                    "add_store: NULL class in kernel_loads[{kid:?}]"
                );
                debug_assert!(
                    !output_cids.iter().any(|&c| c.0 == u32::MAX),
                    "add_store: NULL class in kernel_stores[{kid:?}]"
                );
                let inputs: Box<[ClassId]> = input_cids.into_boxed_slice();
                let outputs_box: Box<[ClassId]> = output_cids.into_boxed_slice();
                let compute_ops = owned_kernel
                    .ops
                    .values()
                    .filter(|n| {
                        matches!(
                            n.op,
                            Op::Unary { .. }
                                | Op::Binary { .. }
                                | Op::Cast { .. }
                                | Op::Reduce { .. }
                                | Op::Mad { .. }
                        )
                    })
                    .count() as u32;
                let kind = ENode::Kernel(inputs, outputs_box, ProgramId::NULL);
                let knid = self.nodes.push(kind);
                let idx = knid.0 as usize;
                self.grow_uf_arrays(idx);
                self.class_of[idx] = cid; // last stored class "owns" the enode
                self.classes[cid].nodes.push(knid);
                self.costs.insert(knid, kernel_cost(compute_ops.max(1)));
                self.kernel_irs.insert(kid, owned_kernel); // keep for autotune
                self.kernel_map.insert(knid, kid);
            }
        }
    }
}

/// Get the first (and usually only) op of a freshly created load kernel.
fn kid_first_op(kid: KMKernelId, kernel_irs: &Map<KMKernelId, Kernel>) -> OpId {
    let kernel = &kernel_irs[&kid];
    kernel.head
}

// ── Topological sort ──────────────────────────────────────

/// Topologically sort e-graph classes (children before parents)
/// using Kahn's algorithm on the enode dependency DAG.
///
/// All classes are included in the result.  Classes with no children
/// (Leaf, Const) come first; classes whose children have all been
/// emitted come next.
pub(crate) fn topo_sort_classes(eg: &EGraph) -> Vec<ClassId> {
    // Skip merged (non-root) classes — only process each equivalence
    // class root once.
    let mut children_of: Map<ClassId, Set<ClassId>> = Map::default();
    for (cid, class) in eg.classes.iter() {
        if eg.class_parent[cid.0 as usize] != cid {
            continue;
        }
        let mut children: Set<ClassId> = Set::default();
        for &nid in &class.nodes {
            for &child in eg.nodes[nid].child_classes().iter() {
                let root = eg.class_parent[child.0 as usize];
                children.insert(root);
            }
        }
        children_of.insert(cid, children);
    }

    // in_degree: how many children each class depends on.
    // dependents: for each child class, which classes depend on it.
    let mut in_degree: Map<ClassId, u32> = Map::default();
    let mut dependents: Map<ClassId, Vec<ClassId>> = Map::default();
    for (&cid, children) in &children_of {
        let deg = children.len() as u32;
        in_degree.insert(cid, deg);
        for &child in children {
            dependents.entry(child).or_default().push(cid);
        }
    }

    // Start queue with classes that have no dependencies.
    let mut queue: Vec<ClassId> = Vec::new();
    for (&cid, &deg) in &in_degree {
        if deg == 0 {
            queue.push(cid);
        }
    }

    // Process in topological order.
    let mut order = Vec::new();
    while let Some(cid) = queue.pop() {
        order.push(cid);
        if let Some(deps) = dependents.get(&cid) {
            for &parent in deps {
                if let Some(deg) = in_degree.get_mut(&parent) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        queue.push(parent);
                    }
                }
            }
        }
    }

    order
}

// ── Reduce-axis permutation ───────────────────────────────

/// If the reduction axes are not the trailing dimensions, insert a
/// permute before the reduce op to move them there.
/// Returns the (possibly permuted) input op id.
fn try_permute_reduce_axes(
    k: &mut Kernel,
    x: OpId,
    in_shape: &[Dim],
    out_shape: &[Dim],
    n_axes: usize,
) -> OpId {
    if n_axes == 0 {
        return x;
    }

    // Walk in_shape and out_shape together to find which axes are reduced.
    let mut kept = Vec::new();
    let mut reduced = Vec::new();
    let mut j = 0;
    for (i, &dim) in in_shape.iter().enumerate() {
        if j < out_shape.len() && dim == out_shape[j] {
            kept.push(i as UAxis);
            j += 1;
        } else {
            reduced.push(i as UAxis);
        }
    }

    // If reduced axes are already trailing, no permute needed.
    let kept_end = kept.len();
    let already_trailing = reduced.iter().enumerate().all(|(idx, &ax)| {
        ax as usize == kept_end + idx
    });
    if already_trailing {
        return x;
    }

    // Build permute: kept axes first, then reduced axes.
    let mut permute = kept;
    permute.extend(reduced);
    let shape = crate::shape::permute(in_shape, &permute);
    k.push_back(Op::Move {
        x,
        mop: Box::new(MoveOp::Permute { axes: permute, shape }),
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        graph::search::{ENode, ClassId},
        kernel::{BOp, UOp},
        DType,
    };

    use super::EGraph;

    /// Simple chain: Leaf -> Expand -> Expand
    #[test]
    fn topo_chain() {
        let mut eg = EGraph::new();
        let (_, c0) = eg.make(ENode::Leaf(DType::F32), Box::new([]), DType::F32);
        let (_, c1) = eg.make(ENode::Expand(c0), Box::new([]), DType::F32);
        let (_, c2) = eg.make(ENode::Expand(c1), Box::new([]), DType::F32);

        let order = super::topo_sort_classes(&eg);
        assert_eq!(order.len(), 3, "expected 3 classes, got {}", order.len());

        // Children before parents: c0 (Leaf) -> c1 -> c2
        let pos = |cid: ClassId| order.iter().position(|&x| x == cid).unwrap();
        assert!(pos(c0) < pos(c1), "c0 should come before c1");
        assert!(pos(c1) < pos(c2), "c1 should come before c2");
    }

    /// Diamond:      leaf
    ///              /    \
    ///           Neg     Abs
    ///              \    /
    ///               Add
    #[test]
    fn topo_diamond() {
        let mut eg = EGraph::new();
        let (_, leaf) = eg.make(ENode::Leaf(DType::F32), Box::new([]), DType::F32);
        let (_, neg) = eg.make(ENode::Unary(leaf, UOp::Neg), Box::new([]), DType::F32);
        let (_, abs) = eg.make(ENode::Unary(leaf, UOp::Abs), Box::new([]), DType::F32);
        let (_, add) = eg.make(ENode::Binary(neg, abs, BOp::Add), Box::new([]), DType::F32);

        let order = super::topo_sort_classes(&eg);
        assert_eq!(order.len(), 4, "expected 4 classes, got {}", order.len());

        let pos = |cid: ClassId| order.iter().position(|&x| x == cid).unwrap();
        assert!(pos(leaf) < pos(neg), "leaf should come before neg");
        assert!(pos(leaf) < pos(abs), "leaf should come before abs");
        assert!(pos(neg) < pos(add), "neg should come before add");
        assert!(pos(abs) < pos(add), "abs should come before add");
    }

    /// Disjoint classes: two independent chains
    #[test]
    fn topo_disjoint() {
        let mut eg = EGraph::new();
        let (_, l0) = eg.make(ENode::Leaf(DType::F32), Box::new([]), DType::F32);
        let (_, l1) = eg.make(ENode::Leaf(DType::F64), Box::new([]), DType::F64);
        let (_, e0) = eg.make(ENode::Expand(l0), Box::new([]), DType::F32);

        let order = super::topo_sort_classes(&eg);
        assert_eq!(order.len(), 3, "expected 3 classes, got {}", order.len());

        let pos = |cid: ClassId| order.iter().position(|&x| x == cid).unwrap();
        // Both leaves before expand
        assert!(pos(l0) < pos(e0));
        // l1 can be anywhere, just check it's in order
        assert!(order.contains(&l1));
    }
}
