// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! E-graph kernelizer.
//!
//! **CRITICAL RULE: This module MUST produce the EXACT SAME kernels as
//! `src/kernelize.rs`.**  Every kernel built here (op sequence, load/store
//! structure, output tensors/classes, duplication boundaries) must match
//! what kernelize.rs would produce for the same computation.  If the
//! kernels differ, test failures are introduced here, not in kernelize.rs.
//!
//! **To compare kernels**: set `USE_EGRAPH` below to `false`.  This
//! falls back to `src/kernelize.rs`'s `realize_with_order` and prints
//! both kernels side by side.  Set back to `true` and compare IR dumps
//! (ZYX_DEBUG=8) to find differences.
//!
//! Below is how `kernelize.rs` works — this module must do the same,
//! adapted to the e-graph.
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
    DType, Map, Set,
    backend::ProgramId,
    graph::search::{ClassId, EGraph, ENode},
    kernel::{BOp, DeviceId, Kernel, MoveOp, Op, OpId, UOp},
    kernelize::KMKernelId,
    shape::{Dim, UAxis},
    tensor::TensorId,
    view::View,
};

type KernelData = (Vec<ClassId>, Vec<ClassId>, Vec<ClassId>);

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
    pub(crate) fn kernelize_all(&mut self, output_classes: &Set<ClassId>) {
        let order = topo_sort_classes(self, output_classes);

        // Reference counts: how many times each class appears as a child.
        let mut rcs: Map<ClassId, u32> = Map::default();
        {
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
        }

        let mut pending_stores: Set<ClassId> = Set::default();
        let mut visited: Map<ClassId, (KMKernelId, OpId)> = Map::default();
        let mut kernel_data: Map<KMKernelId, KernelData> = Map::default();
        let mut kernel_id_counter: u32 = 0;

        for cid in order {
            debug_assert!(!visited.contains_key(&cid), "class {cid:?} already visited");

            // Classes with only Kernel enodes are pre-compiled programs
            // (CustomKernel). Skip — they don't need kernelization.
            let has_non_kernel = self.classes[cid]
                .nodes
                .iter()
                .any(|&nid| !matches!(self.nodes[nid], ENode::Kernel(..)));
            if !has_non_kernel {
                continue;
            }

            // Leaf & Const classes: create load kernel so parents can
            // reference their value.
            if self.is_leaf_or_const(cid) {
                let rc = rcs.get(&cid).copied().unwrap_or(0);
                if rc > 0 {
                    let kid = self.new_load_kernel(cid, &mut kernel_id_counter, &mut kernel_data);
                    let n_consumers = rc as usize;
                    for _ in 0..n_consumers {
                        kernel_data.entry(kid).or_default().0.push(cid);
                    }
                    visited.insert(cid, (kid, kid_first_op(kid, &self.kernel_irs)));
                }
                continue;
            }

            // Pick the first non-Kernel enode (all are equivalent in a class).
            let nid = match self.classes[cid]
                .nodes
                .iter()
                .copied()
                .find(|&nid| !matches!(self.nodes[nid], ENode::Kernel(..)))
            {
                Some(nid) => nid,
                None => panic!("There shouldn't be empty classes. This is internal bug."),
            };

            match self.nodes[nid] {
                ENode::Unary(child, uop) => {
                    self.add_unary(cid, child, uop, &mut visited, &mut kernel_data, &mut kernel_id_counter, &rcs)
                }
                ENode::Cast(child, dtype) => self.add_cast(
                    cid,
                    child,
                    dtype,
                    &mut visited,
                    &mut kernel_data,
                    &mut kernel_id_counter,
                    &rcs,
                ),
                ENode::Binary(lhs, rhs, bop) => self.add_binary(
                    cid,
                    lhs,
                    rhs,
                    bop,
                    &mut visited,
                    &mut kernel_data,
                    &mut kernel_id_counter,
                    &rcs,
                    &mut pending_stores,
                ),
                ENode::Reduce(child, rop, ref axes) => {
                    let axes: Vec<UAxis> = axes.to_vec();
                    self.add_reduce(
                        cid,
                        child,
                        rop,
                        axes,
                        &mut visited,
                        &mut kernel_data,
                        &mut kernel_id_counter,
                        &rcs,
                        &mut pending_stores,
                    )
                }
                ENode::Expand(child) => self.add_expand(
                    cid,
                    child,
                    &mut visited,
                    &mut kernel_data,
                    &mut kernel_id_counter,
                    &rcs,
                    &mut pending_stores,
                ),
                ENode::Permute(child, ref axes) => {
                    let axes: Vec<UAxis> = axes.clone().into_vec();
                    self.add_permute(
                        cid,
                        child,
                        axes,
                        &mut visited,
                        &mut kernel_data,
                        &mut kernel_id_counter,
                        &rcs,
                        &mut pending_stores,
                    )
                }
                ENode::Reshape(child, ref shape) => {
                    let shape: Vec<Dim> = shape.clone().into_vec();
                    self.add_reshape(
                        cid,
                        child,
                        shape,
                        &mut visited,
                        &mut kernel_data,
                        &mut kernel_id_counter,
                        &rcs,
                        &mut pending_stores,
                    )
                }
                ENode::Pad(child, ref padding) => {
                    let padding: Vec<(i64, i64)> = padding.clone().into_vec();
                    self.add_pad(
                        cid,
                        child,
                        padding,
                        &mut visited,
                        &mut kernel_data,
                        &mut kernel_id_counter,
                        &rcs,
                        &mut pending_stores,
                    )
                }
                ENode::ToDevice(child, _dev) => {
                    // ToDevice forces storage at the boundary.
                    let (child_kid, child_op) = match visited.get(&child) {
                        Some(&v) => v,
                        None => {
                            let kid = self.new_load_kernel(child, &mut kernel_id_counter, &mut kernel_data);
                            let op = kid_first_op(kid, &self.kernel_irs);
                            visited.insert(child, (kid, op));
                            (kid, op)
                        }
                    };
                    self.add_store(
                        child,
                        child_kid,
                        child_op,
                        &mut visited,
                        &mut kernel_data,
                        &mut pending_stores,
                    );
                    let (kid, op_id) = match visited.get(&child) {
                        Some(&v) => v,
                        None => unreachable!(),
                    };
                    visited.insert(cid, (kid, op_id));
                }
                ENode::Const(_) | ENode::Leaf(_) => {
                    // Handled by leaf_or_const check above.
                    unreachable!()
                }
                ENode::Kernel(..) => continue,
            }

            // Post-processing: store if output, final, or reduce boundary.
            if !visited.contains_key(&cid) {
                // add_* method stored this class already.
                continue;
            }
            let remaining_rc = rcs.get(&cid).copied().unwrap_or(0);
            let is_output = remaining_rc == 0 || output_classes.contains(&cid);
            let is_reduce = self.classes[cid]
                .nodes
                .iter()
                .any(|&n| matches!(&self.nodes[n], ENode::Reduce(..)));

            if is_output || is_reduce {
                let (kid, op_id) = visited[&cid];
                self.add_store(cid, kid, op_id, &mut visited, &mut kernel_data, &mut pending_stores);
                // If this output still has consumers, reload it in a new kernel.
                if output_classes.contains(&cid) && remaining_rc > 0 {
                    let new_kid = self.new_load_kernel(cid, &mut kernel_id_counter, &mut kernel_data);
                    let new_op = kid_first_op(new_kid, &self.kernel_irs);
                    visited.insert(cid, (new_kid, new_op));
                }
            }

            // ── Process subsequent enode variants (cloned per variant) ──
            // TODO: implement after first-variant path is working.
        }

        // Force-seal remaining kernels that still have output classes.
        let kid_list: Vec<KMKernelId> = self.kernel_irs.keys().copied().collect();
        for &kid in &kid_list {
            if !self.kernel_irs.contains_key(&kid) {
                continue;
            }
            let remaining: Vec<ClassId> = kernel_data
                .get(&kid)
                .map(|(outputs, _, _)| outputs.iter().copied().filter(|&c| output_classes.contains(&c)).collect())
                .unwrap_or_default();
            for &cid in &remaining {
                let op_id = match visited.get(&cid) {
                    Some(&(_, op)) => op,
                    None => continue,
                };
                self.add_store(cid, kid, op_id, &mut visited, &mut kernel_data, &mut pending_stores);
            }
        }
    }

    // ── Helpers ─────────────────────────────────────────────

    fn is_leaf_or_const(&self, cid: ClassId) -> bool {
        self.classes[cid]
            .nodes
            .iter()
            .any(|&nid| matches!(self.nodes[nid], ENode::Leaf(_) | ENode::Const(_)))
    }

    fn new_load_kernel(&mut self, cid: ClassId, counter: &mut u32, kernel_data: &mut Map<KMKernelId, KernelData>) -> KMKernelId {
        let mut kernel = Kernel::new(DeviceId::AUTO);
        let shape: Vec<Dim> = self.classes[cid].shape.to_vec();
        // Check if this class holds a Const — if so, embed the value inline.
        let is_const = self.classes[cid]
            .nodes
            .iter()
            .any(|&nid| matches!(self.nodes[nid], ENode::Const(..)));
        if is_const {
            let value = self.classes[cid]
                .nodes
                .iter()
                .copied()
                .find_map(|nid| {
                    if let ENode::Const(v) = &self.nodes[nid] {
                        Some(*v)
                    } else {
                        None
                    }
                })
                .unwrap();
            kernel.push_back(Op::ConstView(Box::new((value, View::contiguous(&[1])))));
        } else {
            kernel.load_contiguous(self.classes[cid].dtype, &shape);
        };
        let kid = KMKernelId::from(*counter as usize);
        *counter += 1;
        self.kernel_irs.insert(kid, kernel);
        if !is_const {
            kernel_data.entry(kid).or_default().1.push(cid);
        }
        kid
    }

    fn merge_kernels(
        &mut self,
        src: KMKernelId,
        dst: KMKernelId,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        kernel_data: &mut Map<KMKernelId, KernelData>,
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
        if let Some((outputs, loads, stores)) = kernel_data.remove(&src) {
            let (dst_outputs, dst_loads, dst_stores) = kernel_data.entry(dst).or_default();
            dst_outputs.extend(outputs);
            dst_loads.extend(loads);
            dst_stores.extend(stores);
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

    // ── Per-op methods (mirror kernelize.rs add_*_op) ──────

    /// If `child` lives in a kernel that has stores, store it and create a fresh
    /// load kernel.  If the kernel has multiple outputs, duplicate the kernel
    /// (mirrors kernelize.rs duplicate_or_store + duplicate_kernel exactly).
    fn duplicate_or_store(
        &mut self,
        child: ClassId,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        counter: &mut u32,
        kernel_data: &mut Map<KMKernelId, KernelData>,
        pending_stores: &mut Set<ClassId>,
    ) -> (KMKernelId, OpId) {
        let (kid, op_id) = match visited.get(&child) {
            Some(&v) => v,
            None => return self.child_to_kid(child, visited, counter, kernel_data),
        };

        // Phase 1: if kernel has stores, store child and create fresh load kernel
        let has_stores = self.kernel_irs.get(&kid).is_some_and(|k| k.contains_stores());
        let (mut kid, op_id) = if has_stores {
            self.add_store(child, kid, op_id, visited, kernel_data, pending_stores);
            let new_kid = self.new_load_kernel(child, counter, kernel_data);
            let new_op = kid_first_op(new_kid, &self.kernel_irs);
            visited.insert(child, (new_kid, new_op));
            (new_kid, new_op)
        } else {
            (kid, op_id)
        };

        // Phase 2: if kernel has multiple outputs, duplicate it so child
        // gets its own copy (avoids sharing LoadViews between Move chains).
        let n_outputs = kernel_data.get(&kid).map(|(o, _, _)| o.len()).unwrap_or(0);
        if n_outputs > 1 {
            // Check if this op is preceded by a reduce (complex case → store+reload)
            let preceded_by_reduce = self.kernel_irs.get(&kid).is_some_and(|k| k.is_preceded_by_reduce(op_id));
            if preceded_by_reduce {
                self.add_store(child, kid, op_id, visited, kernel_data, pending_stores);
                let new_kid = self.new_load_kernel(child, counter, kernel_data);
                let new_op = kid_first_op(new_kid, &self.kernel_irs);
                visited.insert(child, (new_kid, new_op));
                kid = new_kid;
                let n_outputs2 = kernel_data.get(&kid).map(|(o, _, _)| o.len()).unwrap_or(0);
                if n_outputs2 > 1 {
                    kid = self.duplicate_kernel(child, kid, op_id, visited, kernel_data, counter);
                }
            } else {
                kid = self.duplicate_kernel(child, kid, op_id, visited, kernel_data, counter);
            }
        }

        (kid, op_id)
    }

    /// Duplicate the kernel so `child` gets its own kernel with only its ops.
    /// Returns the new kernel id; updates visited and kernel_data (mirrors
    /// kernelize.rs duplicate_kernel).
    fn duplicate_kernel(
        &mut self,
        child: ClassId,
        kid: KMKernelId,
        op_id: OpId,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        kernel_data: &mut Map<KMKernelId, KernelData>,
        counter: &mut u32,
    ) -> KMKernelId {
        let orig_loads: Vec<TensorId> = kernel_data
            .get(&kid)
            .map(|(_, l, _)| l.iter().map(|c| TensorId(c.0)).collect())
            .unwrap_or_default();
        // Clone kernel IR — new kernel gets only child's ops
        let mut clone = self.kernel_irs[&kid].clone();
        let clone_loads = clone.drop_unused_ops_by_params(vec![op_id], &orig_loads);
        let new_kid = KMKernelId::from(*counter as usize);
        *counter += 1;
        self.kernel_irs.insert(new_kid, clone);

        // Original kernel: remove ONE copy of child from outputs, drop ops
        // only needed by child (mirrors kernelize.rs duplicate_kernel).
        Self::remove_first_output(kernel_data, kid, child);
        let remaining_op_ids: Vec<OpId> = kernel_data
            .get(&kid)
            .map(|(outputs, _, _)| outputs.iter().filter_map(|c| visited.get(c).map(|&(_, oid)| oid)).collect())
            .unwrap_or_default();
        let orig_kernel = self.kernel_irs.get_mut(&kid).unwrap();
        let new_loads = orig_kernel.drop_unused_ops_by_params(remaining_op_ids, &orig_loads);
        // Sync kernel_data loads with rebuilt loads
        if let Some((_, loads, _)) = kernel_data.get_mut(&kid) {
            *loads = new_loads.iter().map(|t| ClassId(t.0)).collect();
        }

        // New kernel gets ONE copy of child as output, with loads matching
        // the subset of LoadView ops that survived drop_unused_ops_by_params.
        let (_, kid_stores) = kernel_data
            .get(&kid)
            .map(|(_, l, s)| (l.clone(), s.clone()))
            .unwrap_or_default();
        let (new_outputs, new_loads, new_stores) = kernel_data.entry(new_kid).or_default();
        new_outputs.push(child);
        *new_loads = clone_loads.iter().map(|&t| ClassId(t.0)).collect();
        *new_stores = kid_stores;

        new_kid
    }

    fn child_to_kid(
        &mut self,
        child: ClassId,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        counter: &mut u32,
        kernel_data: &mut Map<KMKernelId, KernelData>,
    ) -> (KMKernelId, OpId) {
        match visited.get(&child) {
            Some(&v) => v,
            None => {
                let kid = self.new_load_kernel(child, counter, kernel_data);
                let op = kid_first_op(kid, &self.kernel_irs);
                visited.insert(child, (kid, op));
                (kid, op)
            }
        }
    }

    fn remove_first_output(kernel_data: &mut Map<KMKernelId, KernelData>, kid: KMKernelId, cid: ClassId) {
        if let Some((outputs, _, _)) = kernel_data.get_mut(&kid) {
            if let Some(pos) = outputs.iter().position(|&x| x == cid) {
                outputs.remove(pos);
            }
        }
    }

    fn add_unary(
        &mut self,
        cid: ClassId,
        child: ClassId,
        uop: UOp,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        kernel_data: &mut Map<KMKernelId, KernelData>,
        counter: &mut u32,
        rcs: &Map<ClassId, u32>,
    ) {
        let (kid, op_id) = self.child_to_kid(child, visited, counter, kernel_data);
        Self::remove_first_output(kernel_data, kid, child);
        let kernel = self.kernel_irs.get_mut(&kid).unwrap();
        let result_op = kernel.push_back(Op::Unary { x: op_id, uop });
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            kernel_data.entry(kid).or_default().0.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_cast(
        &mut self,
        cid: ClassId,
        child: ClassId,
        dtype: DType,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        kernel_data: &mut Map<KMKernelId, KernelData>,
        counter: &mut u32,
        rcs: &Map<ClassId, u32>,
    ) {
        let (kid, op_id) = self.child_to_kid(child, visited, counter, kernel_data);
        Self::remove_first_output(kernel_data, kid, child);
        let kernel = self.kernel_irs.get_mut(&kid).unwrap();
        let result_op = kernel.cast(op_id, dtype);
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            kernel_data.entry(kid).or_default().0.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_binary(
        &mut self,
        cid: ClassId,
        lhs: ClassId,
        rhs: ClassId,
        bop: BOp,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        kernel_data: &mut Map<KMKernelId, KernelData>,
        counter: &mut u32,
        rcs: &Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
    ) {
        let (mut kid, mut op_id) = self.child_to_kid(lhs, visited, counter, kernel_data);
        let (mut kidy, op_idy) = self.child_to_kid(rhs, visited, counter, kernel_data);

        let kid_stores = self.kernel_irs.get(&kid).is_some_and(|k| k.contains_stores());
        let kidy_stores = self.kernel_irs.get(&kidy).is_some_and(|k| k.contains_stores());

        if kid == kidy {
            Self::remove_first_output(kernel_data, kid, lhs);
            Self::remove_first_output(kernel_data, kid, rhs);
            let kernel = self.kernel_irs.get_mut(&kid).unwrap();
            let result_op = kernel.binary(op_id, op_idy, bop);
            let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
            for _ in 0..n_consumers {
                kernel_data.entry(kid).or_default().0.push(cid);
            }
            visited.insert(cid, (kid, result_op));
        } else {
            match (kid_stores, kidy_stores) {
                (true, true) => {
                    self.add_store(lhs, kid, op_id, visited, kernel_data, pending_stores);
                    let new_kid = self.new_load_kernel(lhs, counter, kernel_data);
                    let new_op = kid_first_op(new_kid, &self.kernel_irs);
                    visited.insert(lhs, (new_kid, new_op));
                    (kid, op_id) = (new_kid, new_op);

                    self.add_store(rhs, kidy, op_idy, visited, kernel_data, pending_stores);
                    let new_kid = self.new_load_kernel(rhs, counter, kernel_data);
                    let new_op = kid_first_op(new_kid, &self.kernel_irs);
                    visited.insert(rhs, (new_kid, new_op));
                    (kidy, _) = (new_kid, new_op);
                }
                (true, false) => {
                    self.add_store(lhs, kid, op_id, visited, kernel_data, pending_stores);
                    let new_kid = self.new_load_kernel(lhs, counter, kernel_data);
                    let new_op = kid_first_op(new_kid, &self.kernel_irs);
                    visited.insert(lhs, (new_kid, new_op));
                    (kid, op_id) = (new_kid, new_op);
                }
                (false, true) => {
                    self.add_store(rhs, kidy, op_idy, visited, kernel_data, pending_stores);
                    let new_kid = self.new_load_kernel(rhs, counter, kernel_data);
                    let new_op = kid_first_op(new_kid, &self.kernel_irs);
                    visited.insert(rhs, (new_kid, new_op));
                    (kidy, _) = (new_kid, new_op);
                }
                (false, false) => {}
            }

            // Merge rhs kernel into lhs kernel.
            self.merge_kernels(kidy, kid, visited, kernel_data);
            let (_, op_idy) = visited[&rhs];
            Self::remove_first_output(kernel_data, kid, lhs);
            Self::remove_first_output(kernel_data, kid, rhs);
            let kernel = self.kernel_irs.get_mut(&kid).unwrap();
            let result_op = kernel.binary(op_id, op_idy, bop);
            let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
            for _ in 0..n_consumers {
                kernel_data.entry(kid).or_default().0.push(cid);
            }
            visited.insert(cid, (kid, result_op));
        }
    }

    fn add_reduce(
        &mut self,
        cid: ClassId,
        child: ClassId,
        rop: BOp,
        axes: Vec<UAxis>,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        kernel_data: &mut Map<KMKernelId, KernelData>,
        counter: &mut u32,
        rcs: &Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
    ) {
        let n_axes = axes.len() as UAxis;
        let (kid, op_id) = self.duplicate_or_store(child, visited, counter, kernel_data, pending_stores);
        Self::remove_first_output(kernel_data, kid, child);

        // Permute reduce axes to be trailing (mirrors kernelize.rs).
        let in_root = self.find_class(child);
        let in_shape: Vec<Dim> = self.classes[in_root].shape.to_vec();
        let kernel = self.kernel_irs.get_mut(&kid).unwrap();
        let permuted = {
            let n = in_shape.len();
            let max_axis = *axes.last().unwrap() as usize;
            let mut permute_axes = Vec::with_capacity(n);
            let mut ai = 0;
            for i in 0..=max_axis {
                if axes[ai] as usize == i {
                    ai += 1;
                } else {
                    permute_axes.push(i as UAxis);
                }
            }
            permute_axes.extend((max_axis + 1..n).map(|i| i as UAxis));
            permute_axes.extend_from_slice(&axes);
            if !permute_axes.iter().copied().eq(0..permute_axes.len() as UAxis) {
                let shape = crate::shape::permute(&in_shape, &permute_axes);
                kernel.push_back(Op::Move {
                    x: op_id,
                    mop: Box::new(MoveOp::Permute {
                        axes: permute_axes,
                        shape,
                    }),
                })
            } else {
                op_id
            }
        };
        let result_op = kernel.push_back(Op::Reduce {
            x: permuted,
            rop,
            n_axes: n_axes as UAxis,
        });
        // If all dimensions are reduced, reshape to [1] (mirrors kernelize.rs behavior).
        let result_op = if in_shape.len() == n_axes as usize {
            kernel.reshape(result_op, &vec![1])
        } else {
            result_op
        };

        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            kernel_data.entry(kid).or_default().0.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_expand(
        &mut self,
        cid: ClassId,
        child: ClassId,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        kernel_data: &mut Map<KMKernelId, KernelData>,
        counter: &mut u32,
        rcs: &Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
    ) {
        let (kid, op_id) = self.duplicate_or_store(child, visited, counter, kernel_data, pending_stores);
        Self::remove_first_output(kernel_data, kid, child);
        let shape: Vec<Dim> = self.classes[cid].shape.to_vec();
        let kernel = self.kernel_irs.get_mut(&kid).unwrap();
        let result_op = kernel.push_back(Op::Move {
            x: op_id,
            mop: Box::new(MoveOp::Expand { shape }),
        });
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            kernel_data.entry(kid).or_default().0.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_permute(
        &mut self,
        cid: ClassId,
        child: ClassId,
        axes: Vec<UAxis>,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        kernel_data: &mut Map<KMKernelId, KernelData>,
        counter: &mut u32,
        rcs: &Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
    ) {
        let (kid, op_id) = self.duplicate_or_store(child, visited, counter, kernel_data, pending_stores);
        Self::remove_first_output(kernel_data, kid, child);
        let shape: Vec<Dim> = self.classes[cid].shape.to_vec();
        let kernel = self.kernel_irs.get_mut(&kid).unwrap();
        let result_op = kernel.push_back(Op::Move {
            x: op_id,
            mop: Box::new(MoveOp::Permute { axes, shape }),
        });
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            kernel_data.entry(kid).or_default().0.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_reshape(
        &mut self,
        cid: ClassId,
        child: ClassId,
        shape: Vec<Dim>,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        kernel_data: &mut Map<KMKernelId, KernelData>,
        counter: &mut u32,
        rcs: &Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
    ) {
        let (kid, op_id) = self.duplicate_or_store(child, visited, counter, kernel_data, pending_stores);
        Self::remove_first_output(kernel_data, kid, child);

        // Permute reduce axes to be trailing (mirrors kernelize.rs).
        let kernel = self.kernel_irs.get_mut(&kid).unwrap();
        let result_op = kernel.push_back(Op::Move {
            x: op_id,
            mop: Box::new(MoveOp::Reshape { shape }),
        });
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            kernel_data.entry(kid).or_default().0.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_pad(
        &mut self,
        cid: ClassId,
        child: ClassId,
        padding: Vec<(i64, i64)>,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        kernel_data: &mut Map<KMKernelId, KernelData>,
        counter: &mut u32,
        rcs: &Map<ClassId, u32>,
        pending_stores: &mut Set<ClassId>,
    ) {
        // If the pad EXPANDS the element count (result shape has more elements
        // than the input), force a store boundary to prevent a single kernel
        // from having outputs with different shapes. This mirrors the
        // kernelize.rs behavior where block-level movement ops that change size
        // are never inlined.
        let child_root = self.find_class(child);
        let child_n: Dim = self.classes[child_root].shape.iter().product();
        let pad_n: Dim = self.classes[cid].shape.iter().product();
        let expands = pad_n > child_n;

        let kid;
        let op_id;
        if expands {
            if let Some(&(ckid, cop_id)) = visited.get(&child) {
                self.add_store(child, ckid, cop_id, visited, kernel_data, pending_stores);
            }
            (kid, op_id) = self.child_to_kid(child, visited, counter, kernel_data);
        } else {
            (kid, op_id) = self.duplicate_or_store(child, visited, counter, kernel_data, pending_stores);
        }
        Self::remove_first_output(kernel_data, kid, child);
        let shape: Vec<Dim> = self.classes[cid].shape.to_vec();
        let kernel = self.kernel_irs.get_mut(&kid).unwrap();
        let result_op = kernel.push_back(Op::Move {
            x: op_id,
            mop: Box::new(MoveOp::Pad { padding, shape }),
        });
        let n_consumers = rcs.get(&cid).copied().unwrap_or(0) as usize;
        for _ in 0..n_consumers {
            kernel_data.entry(kid).or_default().0.push(cid);
        }
        visited.insert(cid, (kid, result_op));
    }

    fn add_store(
        &mut self,
        cid: ClassId,
        kid: KMKernelId,
        op_id: OpId,
        visited: &mut Map<ClassId, (KMKernelId, OpId)>,
        kernel_data: &mut Map<KMKernelId, KernelData>,
        pending_stores: &mut Set<ClassId>,
    ) {
        debug_assert!(!op_id.is_null(), "add_store: NULL op_id for cid={cid:?}");
        if pending_stores.contains(&cid) {
            // Already stored — just remove from visited/outputs.
            visited.remove(&cid);
            if let Some((outputs, _, _)) = kernel_data.get_mut(&kid) {
                outputs.retain(|&x| x != cid);
            }
            return;
        }
        debug_assert!(self.kernel_irs.contains_key(&kid), "add_store: kid={kid:?} not in kernel_irs");
        pending_stores.insert(cid);
        let dtype = self.classes[cid].dtype;
        let kernel = self.kernel_irs.get_mut(&kid).unwrap();
        kernel.store_contiguous(op_id, dtype);
        kernel_data.entry(kid).or_default().2.push(cid);

        // Remove from visited — stored classes are loaded, not merged.
        visited.remove(&cid);

        // Remove from outputs; seal the kernel if no outputs remain.
        let outputs_empty = match kernel_data.get_mut(&kid) {
            Some((outputs, _, _)) => {
                outputs.retain(|&x| x != cid);
                outputs.is_empty()
            }
            None => true,
        };
        if outputs_empty {
            // Kernel is done — add as ENode::Kernel to the e-graph.
            // Remove ALL visited entries pointing to this kernel so
            // their stale (kid, op) pairs don't get reused later.
            visited.retain(|_, &mut (k, _)| k != kid);
            let owned_kernel = self.kernel_irs.remove(&kid).unwrap();
            let (_, mut input_cids, output_cids) = kernel_data.remove(&kid).unwrap_or_default();
            // Remove from inputs any class that is also an output — this prevents
            // cycles where a class appears as both kernel input and output
            // (e.g., when a stored value is re-loaded in the same kernel after a merge).
            {
                let output_set: Set<ClassId> = output_cids.iter().copied().collect();
                input_cids.retain(|c| !output_set.contains(c));
            }
            let inputs: Box<[ClassId]> = input_cids.into_boxed_slice();
            let outputs_box: Box<[ClassId]> = output_cids.clone().into_boxed_slice();
            let compute_ops = owned_kernel
                .ops
                .values()
                .filter(|n| {
                    matches!(
                        n.op,
                        Op::Unary { .. } | Op::Binary { .. } | Op::Cast { .. } | Op::Reduce { .. } | Op::Mad { .. }
                    )
                })
                .count() as u32;
            let kind = ENode::Kernel(inputs, outputs_box, ProgramId::NULL);
            let knid = self.nodes.push(kind);
            let idx = knid.0 as usize;
            self.grow_uf_arrays(idx);
            self.class_of[idx] = cid;
            // Add the kernel enode to ALL classes it computes (all stored outputs).
            for &scid in &output_cids {
                let sci = scid.0 as usize;
                self.grow_uf_arrays(sci);
                self.classes[scid].nodes.push(knid);
            }
            if !output_cids.contains(&cid) {
                self.classes[cid].nodes.push(knid);
            }
            self.costs.insert(knid, kernel_cost(compute_ops.max(1)));
            self.kernel_irs.insert(kid, owned_kernel); // keep for autotune
            self.kernel_map.insert(knid, kid);
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
/// emitted come next.  Output classes are prioritized over non-outputs
/// when siblings become available simultaneously — this matches the
/// processing order in kernelize.rs and prevents non-output siblings
/// from being fused into a kernel before its output siblings are stored.
pub(crate) fn topo_sort_classes(eg: &EGraph, output_classes: &Set<ClassId>) -> Vec<ClassId> {
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
    // Use a deque: push outputs at front so they are processed first
    // when siblings become ready at the same time.
    use std::collections::VecDeque;
    let mut queue: VecDeque<ClassId> = VecDeque::new();
    for (&cid, &deg) in &in_degree {
        if deg == 0 {
            if output_classes.contains(&cid) {
                queue.push_front(cid);
            } else {
                queue.push_back(cid);
            }
        }
    }

    // Process in topological order.
    let mut order = Vec::new();
    while let Some(cid) = queue.pop_front() {
        order.push(cid);
        if let Some(deps) = dependents.get(&cid) {
            for &parent in deps {
                if let Some(deg) = in_degree.get_mut(&parent) {
                    *deg = deg.saturating_sub(1);
                    if *deg == 0 {
                        if output_classes.contains(&parent) {
                            queue.push_front(parent);
                        } else {
                            queue.push_back(parent);
                        }
                    }
                }
            }
        }
    }

    order
}

#[cfg(test)]
mod tests {
    use crate::{
        DType, Set,
        graph::search::{ClassId, ENode, NodeId},
        kernel::{BOp, UOp},
    };

    use super::EGraph;

    /// Simple chain: Leaf -> Expand -> Expand
    #[test]
    fn topo_chain() {
        let mut eg = EGraph::new();
        let (_, c0) = eg.make(ENode::Leaf(DType::F32), Box::new([]), DType::F32);
        let (_, c1) = eg.make(ENode::Expand(c0), Box::new([]), DType::F32);
        let (_, c2) = eg.make(ENode::Expand(c1), Box::new([]), DType::F32);

        let order = super::topo_sort_classes(&eg, &Set::default());
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

        let order = super::topo_sort_classes(&eg, &Set::default());
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

        let order = super::topo_sort_classes(&eg, &Set::default());
        assert_eq!(order.len(), 3, "expected 3 classes, got {}", order.len());

        let pos = |cid: ClassId| order.iter().position(|&x| x == cid).unwrap();
        // Both leaves before expand
        assert!(pos(l0) < pos(e0));
        // l1 can be anywhere, just check it's in order
        assert!(order.contains(&l1));
    }

    /// Test that classes with only Kernel enodes reach the `None` case at line 234
    #[test]
    fn kernel_only_classes() {
        let mut eg = EGraph::new();

        // Create a simple class with a non-Kernel enode
        let (_, leaf) = eg.make(
            crate::graph::search::ENode::Leaf(crate::DType::F32),
            Box::new([]),
            crate::DType::F32,
        );

        // Now create a Kernel enode and move it to the leaf's class
        // This simulates what happens in `add_to_class` when moving Kernel enodes
        let kernel_nid: NodeId = eg
            .make(
                crate::graph::search::ENode::Kernel(vec![leaf].into_boxed_slice(), Box::new([]), crate::backend::ProgramId::NULL),
                Box::new([]),
                crate::DType::F32,
            )
            .0;

        // Move the kernel to the leaf's class
        eg.add_to_class(kernel_nid, leaf);

        // The leaf's class should now have two enodes: the leaf and the kernel
        // The kernel's original class should be empty

        // Check if the leaf's class has both enodes
        let leaf_class = eg.find_class(leaf);
        assert_eq!(eg.classes[leaf_class].nodes.len(), 2, "Leaf's class should have 2 enodes");

        // Now try to kernelize - there should be no empty classes
        eg.kernelize_all(&Set::default());

        // This should not panic
    }
}
