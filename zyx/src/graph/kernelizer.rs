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
    shape::{Dim, UAxis},
    view::View,
};

const HIGH_COST: u64 = 1_000_000;

/// Cost for a kernel based on number of compute operations.
fn kernel_cost(ops_count: u32) -> u64 {
    ((1.0 + ops_count as f64).ln() * HIGH_COST as f64) as u64
}

// ── Shared visited map ────────────────────────────────────

/// Maps a class whose value was already computed (multi-consumer →
/// stored) to the kernel enode that stores it and the result `OpId`
/// within that kernel.
type Visited = Map<ClassId, (NodeId, OpId)>;

// ── Public entry point ─────────────────────────────────────

impl EGraph {
    /// Build fused kernels for classes in topological order.
    ///
    /// Mirrors `kernelize.rs`: walk classes in topological order and
    /// build at least one kernel enode for every class that has non-trivial
    /// enodes.  After this pass, every class reachable from an output contains
    /// at least one `ENode::Kernel` or is a `Leaf`/`Const` base case.
    ///
    /// `visited` tracks which classes have already been assigned to a kernel
    /// (analogous to `visited` in kernelize.rs).  When building a kernel for
    /// class C, children already in `visited` are loaded from global memory;
    /// children not yet visited are inlined via `build_enode_forward`
    /// (the e-graph equivalent of kernelize.rs's per-op methods).
    pub(crate) fn kernelize_all(&mut self, output_classes: &Set<ClassId>) {
        let order = topo_sort_classes(self);

        // Mirrors kernelize.rs `visited`: maps each computed class to
        // the (kernel_enode_id, result_op_id) that produces its value.
        let mut visited: Visited = Map::default();

        for cid in order {
            for &nid in self.classes[cid].nodes.clone().iter() {
                let mut k = Kernel::new(DeviceId::AUTO);
                let mut loaded: Vec<ClassId> = Vec::new();

                let result =
                    if let Some(op) = build_enode_forward(self, nid, Some(cid), &mut k, &mut visited, &mut loaded) {
                        op
                    } else {
                        continue;
                    };

                if k.tail.is_null() {
                    continue;
                }

                k.store_contiguous(result, self.classes[cid].dtype);

                let knid = self.make_kernel_enode(cid, &loaded, &k);
                self.kernel_irs.insert(knid, k);

                // Only register in visited if the class needs its own storage
                // (output, multi-consumer, or Reduce boundary).  Mirroring
                // kernelize.rs: single-consumer tensors don't get add_store —
                // their parent's kernel inlines their computation.
                let parent_count = self.classes[cid].parents.len();
                let is_output = output_classes.contains(&cid);
                let is_multi = parent_count > 1;
                let is_reduce = self.classes[cid].nodes.iter().any(|n| {
                    matches!(&self.nodes[*n], ENode::Reduce(..))
                }) || self.classes[cid].parents.iter().any(|(pn, _)| {
                    matches!(&self.nodes[*pn], ENode::Reduce(..))
                });
                if is_output || is_multi || is_reduce {
                    visited.insert(cid, (knid, result));
                }
            }
        }
    }

    fn make_kernel_enode(
        &mut self,
        cid: ClassId,
        inputs: &[ClassId],
        kernel: &Kernel,
    ) -> NodeId {
        let compute_ops = kernel
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

        let inputs: Box<[ClassId]> = inputs.to_vec().into_boxed_slice();
        let outputs: Box<[ClassId]> = vec![cid].into_boxed_slice();
        let kind = ENode::Kernel(inputs, outputs, ProgramId::NULL);
        let knid = self.nodes.push(kind);
        let idx = knid.0 as usize;
        self.grow_uf_arrays(idx);
        self.class_of[idx] = cid;
        self.classes[cid].nodes.push(knid);
        self.costs.insert(knid, kernel_cost(compute_ops.max(1)));
        knid
    }
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

// ── Forward fused kernel builder ──────────────────────────

/// Resolve a node's class, using `cid_override` when supplied (root enode)
/// or falling back to `eg.find(nid)` for inlined children.
fn find_class_of(eg: &mut EGraph, nid: NodeId, cid_override: Option<ClassId>) -> ClassId {
    cid_override.unwrap_or_else(|| eg.find(nid))
}

/// Emit a `LoadView` for a class and return the resulting `OpId`.
/// Records `root` in `loaded` for building accurate kernel input metadata.
fn load_child(eg: &mut EGraph, cid: ClassId, k: &mut Kernel, loaded: &mut Vec<ClassId>) -> Option<OpId> {
    let root = eg.find_class(cid);
    loaded.push(root);
    let dtype = eg.classes[root].dtype;
    let shape: Vec<Dim> = eg.classes[root].shape.to_vec();
    Some(k.load_contiguous(dtype, &shape))
}

/// Resolve a child class to an `OpId` for use in the current kernel.
///
/// If the child is multi-consumer (in `visited`) it was already stored
/// to global memory — emit a `LoadView`.  If it is a single-consumer
/// operator (not in `visited`) inline its computation by recursing into
/// its enode.  Leaf and Const are handled  directly  (load / literal).
fn resolve_child(
    eg: &mut EGraph,
    child_cid: ClassId,
    k: &mut Kernel,
    visited: &Visited,
    loaded: &mut Vec<ClassId>,
) -> Option<OpId> {
    let root = eg.find_class(child_cid);

    // Multi-consumer class that was already computed and stored → load.
    if visited.contains_key(&root) {
        return load_child(eg, root, k, loaded);
    }

    // Single-consumer — inline or handle base cases.
    // Collect first to avoid borrow conflicts with build_enode_forward.
    let enodes: Vec<NodeId> = eg.classes[root].nodes.clone();
    for &nid in &enodes {
        match &eg.nodes[nid] {
            ENode::Const(v) => {
                let op = Op::ConstView(Box::new((v.clone(), View::contiguous(&[1]))));
                return Some(k.push_back(op));
            }
            ENode::Leaf(_) => {
                return load_child(eg, root, k, loaded);
            }
            ENode::Kernel(..) => {
                // Already compiled kernel → load.
                return load_child(eg, root, k, loaded);
            }
            _ => {
                // First non-trivial enode — inline it.
                return build_enode_forward(eg, nid, None, k, visited, loaded);
            }
        }
    }

    // No enode produced a value (shouldn't happen for a valid graph).
    None
}

/// Emit the ops for a single enode, resolving children via the shared
/// `visited` map.
///
/// `cid_override` is `Some` for the root enode (the one we are kernelizing),
/// `None` for inlined children.
fn build_enode_forward(
    eg: &mut EGraph,
    nid: NodeId,
    cid_override: Option<ClassId>,
    k: &mut Kernel,
    visited: &Visited,
    loaded: &mut Vec<ClassId>,
) -> Option<OpId> {
    let kind = eg.nodes[nid].clone();
    let children: Vec<ClassId> = kind.child_classes();

    match kind {
        ENode::Const(v) => {
            let op = Op::ConstView(Box::new((v, View::contiguous(&[1]))));
            Some(k.push_back(op))
        }
        ENode::Leaf(_) | ENode::Kernel(..) | ENode::ToDevice(..) => None,
        ENode::Expand(..) => {
            let cid = find_class_of(eg, nid, cid_override);
            let shape: Vec<Dim> = eg.classes[cid].shape.to_vec();
            let x = resolve_child(eg, children[0], k, visited, loaded)?;
            Some(k.push_back(Op::Move {
                x,
                mop: Box::new(MoveOp::Expand { shape }),
            }))
        }
        ENode::Permute(_, axes) => {
            let cid = find_class_of(eg, nid, cid_override);
            let shape: Vec<Dim> = eg.classes[cid].shape.to_vec();
            let axes: Vec<UAxis> = axes.into_vec();
            let x = resolve_child(eg, children[0], k, visited, loaded)?;
            Some(k.push_back(Op::Move {
                x,
                mop: Box::new(MoveOp::Permute { axes, shape }),
            }))
        }
        ENode::Reshape(_, shape) => {
            let shape: Vec<Dim> = shape.into_vec();
            let x = resolve_child(eg, children[0], k, visited, loaded)?;
            Some(k.push_back(Op::Move {
                x,
                mop: Box::new(MoveOp::Reshape { shape }),
            }))
        }
        ENode::Pad(_, padding) => {
            let cid = find_class_of(eg, nid, cid_override);
            let shape: Vec<Dim> = eg.classes[cid].shape.to_vec();
            let padding: Vec<(i64, i64)> = padding.into_vec();
            let x = resolve_child(eg, children[0], k, visited, loaded)?;
            Some(k.push_back(Op::Move {
                x,
                mop: Box::new(MoveOp::Pad { padding, shape }),
            }))
        }
        ENode::Reduce(_, rop) => {
            // Reductions always load their input (fusion boundary).
            let in_shape: Vec<Dim> = {
                let root = eg.find_class(children[0]);
                eg.classes[root].shape.clone()
            }
            .to_vec();
            let cid = find_class_of(eg, nid, cid_override);
            let out_shape: Vec<Dim> = eg.classes[cid].shape.to_vec();
            let n_axes = in_shape.len().saturating_sub(out_shape.len());

            let x = {
                let root = eg.find_class(children[0]);
                load_child(eg, root, k, loaded)?
            };

            let permuted = try_permute_reduce_axes(k, x, &in_shape, &out_shape, n_axes);

            let r = k.push_back(Op::Reduce {
                x: permuted,
                rop,
                n_axes: n_axes as UAxis,
            });

            // Full reduction -> reshape to [1] so the output shape matches.
            if out_shape.len() == 1 && n_axes > 0 && in_shape.len() > 1 {
                k.reshape(r, &out_shape);
            }
            Some(r)
        }
        ENode::Cast(_, dt) => {
            let x = resolve_child(eg, children[0], k, visited, loaded)?;
            Some(k.cast(x, dt))
        }
        ENode::Unary(_, uop) => {
            let x = resolve_child(eg, children[0], k, visited, loaded)?;
            Some(k.push_back(Op::Unary { x, uop }))
        }
        ENode::Binary(_, _, bop) => {
            let lhs = resolve_child(eg, children[0], k, visited, loaded)?;
            let rhs = resolve_child(eg, children[1], k, visited, loaded)?;
            Some(k.binary(lhs, rhs, bop))
        }
    }
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
