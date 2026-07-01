// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! E-graph kernelizer.
//!
//! Walks the egraph and builds fused kernel IRs by greedily inlining
//! operations into kernels, stopping at reduction/physical boundaries.
//! This mirrors the fusion strategy from the old kernelize.rs but works
//! on the egraph's class/enode structure.

use crate::{
    DType, Map, Set,
    backend::ProgramId,
    graph::search::{ClassId, EGraph, ENode, NodeId},
    kernel::{DeviceId, Kernel, MoveOp, Op, OpId},
    shape::{Dim, UAxis},
    view::View,
};
use std::hash::BuildHasherDefault;

const HIGH_COST: u64 = 1_000_000;

/// Cost for a kernel based on number of compute operations.
fn kernel_cost(ops_count: u32) -> u64 {
    ((1.0 + ops_count as f64).ln() * HIGH_COST as f64) as u64
}

/// Maximum number of parent classes before we stop inlining.
const MAX_INLINE_PARENTS: usize = 1;

// ── Inline-or-load frontier ────────────────────────────────

/// Result of resolving a child class: either we inlined its
/// computation into the current kernel, or we must load it.
enum InlineResult {
    Op(OpId),
    Load(OpId),
}

type InlineCache = crate::Map<ClassId, InlineResult>;

// ── Public entry point ─────────────────────────────────────

impl EGraph {
    /// Build fused kernel IR for every non-trivial enode, inserting kernel
    /// alternatives into their e-classes.
    ///
    /// Each kernel stores its result — this makes it a candidate that later
    /// classes can either inline (by picking the original enode) or load from
    /// (by picking the kernel enode).  The extract phase chooses the cheapest
    /// combination.
    ///
    /// Leaf, Kernel, and Const enodes are skipped (they have no computation to emit).
    ///
    /// For each class we create:
    ///   • Always — a store (load-everything) variant, so extraction can
    ///     materialise this class's value if a parent picks the store path.
    ///   • Only for 0-parent (output) and multi-parent classes — a fused
    ///     (duplicate) variant that inlines children.  Single-parent classes
    ///     are always inlined by their parent's fused variant, so a fused
    ///     variant here would never be cheaper than inlining into the parent.
    ///
    /// # Invariant
    /// After `kernelize_all`, every class reachable from an output must contain
    /// at least one `ENode::Kernel` or be a `Leaf`/`Const` base case.  The
    /// extracted plan can only contain compiled kernels — non-kernel operator
    /// enodes (Binary, Cast, Expand, etc.) are not executable.  `extract_dp`
    /// panics if a class has no kernel alternative.
    ///
    /// # Future work
    /// The parent-count heuristic is a coarse approximation of the
    /// `duplicate_or_store` logic from `kernelize.rs`.  It should be refined
    /// to also consider reduction boundaries and computation depth — if a
    /// fused variant would be identical to the store variant (nothing to
    /// inline), skip it.  Likewise, classes with many parents and cheap
    /// computation might still benefit from a fused variant even if a Reduce
    /// is present.
    pub(crate) fn kernelize_all(&mut self) {
        let to_kernelize: Vec<(NodeId, ClassId)> = {
            let mut v = Vec::new();
            for cid in self.classes.ids() {
                for &nid in &self.classes[cid].nodes {
                    if matches!(&self.nodes[nid], ENode::Leaf(_) | ENode::Kernel(..) | ENode::Const(_)) {
                        continue;
                    }
                    v.push((nid, cid));
                }
            }
            v
        };

        for (nid, cid) in to_kernelize {
            let out_dtype = match self.classes[cid].dtype {
                Some(dt) => dt,
                None => continue,
            };

            // Build one kernel per enode using forward fusion: inline children
            // that can be inlined (single-parent, non-reduce, non-kernel),
            // load the rest from global buffers.  Constants are always inlined
            // as ConstView — they have no buffer allocation.
            if let Some((kernel, loaded_classes)) = build_kernel(self, nid, cid, out_dtype) {
                let knid = self.make_kernel_enode(cid, &loaded_classes, &kernel);
                self.kernel_irs.insert(knid, kernel);
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
        // Push directly without `make` — we don't want parent-list pollution
        // from kernel alternatives interfering with can_inline decisions.
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
    // Map each class to its set of unique child classes (dependencies).
    let mut children_of: Map<ClassId, Set<ClassId>> = Map::default();
    for (cid, class) in eg.classes.iter() {
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

// ── Fused kernel builder ──────────────────────────────────

/// Build a fused kernel for the computation represented by `nid` in class `cid`.
///
/// Starting from the given enode, the builder recursively inlines fusible
/// input classes until it hits a boundary (Reduce, multi-consumer, Leaf, Const).
/// At boundaries a `LoadView` is emitted instead.
///
/// Returns `(Kernel, Vec<ClassId>)` where the vec lists every class whose
/// data is loaded from global memory in the kernel.
fn build_kernel(
    eg: &mut EGraph,
    nid: NodeId,
    cid: ClassId,
    out_dtype: DType,
) -> Option<(Kernel, Vec<ClassId>)> {
    let mut k = Kernel::new(DeviceId::AUTO);
    let mut cache: InlineCache = crate::Map::with_capacity_and_hasher(8, BuildHasherDefault::new());
    let mut loaded: Vec<ClassId> = Vec::new();

    let result = build_enode(eg, nid, Some(cid), &mut k, &mut cache, &mut loaded)?;

    if k.tail.is_null() {
        return None;
    }
    k.store_contiguous(result, out_dtype);
    Some((k, loaded))
}

// ── Per-enode builder ─────────────────────────────────────

/// Resolve a node's class, using `cid_override` when supplied (root enode)
/// or falling back to `eg.find(nid)` for inlined children.
fn find_class_of(eg: &mut EGraph, nid: NodeId, cid_override: Option<ClassId>) -> ClassId {
    cid_override.unwrap_or_else(|| eg.find(nid))
}

/// Emit the ops for a single enode, inlining child classes where possible.
/// `cid_override` is `Some` for the root enode (the one we're kernelizing),
/// `None` for inlined children.
/// `loaded` accumulates every class that gets loaded from global memory.
fn build_enode(
    eg: &mut EGraph,
    nid: NodeId,
    cid_override: Option<ClassId>,
    k: &mut Kernel,
    cache: &mut InlineCache,
    loaded: &mut Vec<ClassId>,
) -> Option<OpId> {
    // Clone the enode and extract children so we can borrow eg mutably later.
    let kind = eg.nodes[nid].clone();
    let children: Vec<ClassId> = kind.child_classes();

    match kind {
        ENode::Const(v) => {
            let op = Op::ConstView(Box::new((v, View::contiguous(&[1]))));
            Some(k.push_back(op))
        }
        ENode::Leaf(_) => None,
        ENode::Expand(..) => {
            let cid = find_class_of(eg, nid, cid_override);
            let shape: Vec<Dim> = eg.classes[cid].shape.clone()?.to_vec();
            let x = inline_or_load(eg, children[0], k, cache, loaded)?;
            Some(k.push_back(Op::Move {
                x,
                mop: Box::new(MoveOp::Expand { shape }),
            }))
        }
        ENode::Permute(_, axes) => {
            let cid = find_class_of(eg, nid, cid_override);
            let shape: Vec<Dim> = eg.classes[cid].shape.clone()?.to_vec();
            let axes: Vec<UAxis> = axes.into_vec();
            let x = inline_or_load(eg, children[0], k, cache, loaded)?;
            Some(k.push_back(Op::Move {
                x,
                mop: Box::new(MoveOp::Permute { axes, shape }),
            }))
        }
        ENode::Reshape(_, shape) => {
            let shape: Vec<Dim> = shape.into_vec();
            let x = inline_or_load(eg, children[0], k, cache, loaded)?;
            Some(k.push_back(Op::Move {
                x,
                mop: Box::new(MoveOp::Reshape { shape }),
            }))
        }
        ENode::Pad(_, padding) => {
            let cid = find_class_of(eg, nid, cid_override);
            let shape: Vec<Dim> = eg.classes[cid].shape.clone()?.to_vec();
            let padding: Vec<(i64, i64)> = padding.into_vec();
            let x = inline_or_load(eg, children[0], k, cache, loaded)?;
            Some(k.push_back(Op::Move {
                x,
                mop: Box::new(MoveOp::Pad { padding, shape }),
            }))
        }
        ENode::Reduce(_, rop) => {
            // Reductions materialize their input (no inlining through a reduce).
            let in_shape: Vec<Dim> = {
                let root = eg.find_class(children[0]);
                eg.classes[root].shape.clone()?
            }
            .to_vec();
            let cid = find_class_of(eg, nid, cid_override);
            let out_shape: Vec<Dim> = eg.classes[cid].shape.clone()?.to_vec();
            let n_axes = in_shape.len().saturating_sub(out_shape.len());

            let x = load_class(eg, children[0], k, loaded)?;

            // If the reduced axes are not the trailing dimensions, insert a
            // permute to move them there (required by unfold_reduces).
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
            let x = inline_or_load(eg, children[0], k, cache, loaded)?;
            Some(k.cast(x, dt))
        }
        ENode::Unary(_, uop) => {
            let x = inline_or_load(eg, children[0], k, cache, loaded)?;
            Some(k.push_back(Op::Unary { x, uop }))
        }
        ENode::Binary(_, _, bop) => {
            let lhs = inline_or_load(eg, children[0], k, cache, loaded)?;
            let rhs = inline_or_load(eg, children[1], k, cache, loaded)?;
            Some(k.binary(lhs, rhs, bop))
        }
        ENode::ToDevice(..) => None,
        ENode::Kernel(..) => {
            // A pre-existing Kernel enode is a materialized result — load it.
            let cid = find_class_of(eg, nid, cid_override);
            load_class(eg, cid, k, loaded)
        }
    }
}

// ── Inline-or-load decision ───────────────────────────────

/// Resolve a child class to an `OpId` – either by inlining its
/// computation into the current kernel, or by emitting a load.
fn inline_or_load(
    eg: &mut EGraph,
    child_cid: ClassId,
    k: &mut Kernel,
    cache: &mut InlineCache,
    loaded: &mut Vec<ClassId>,
) -> Option<OpId> {
    let root = eg.find_class(child_cid);

    if let Some(cached) = cache.get(&root) {
        return Some(match cached {
            InlineResult::Op(op) => *op,
            InlineResult::Load(op) => *op,
        });
    }

    let result = if can_inline(eg, root) {
        // Pick the first non-Leaf, non-Kernel enode (Const is inlinable as literal).
        let inline_nid = eg.classes[root]
            .nodes
            .iter()
            .copied()
            .find(|&n| {
                !matches!(&eg.nodes[n], ENode::Leaf(_) | ENode::Kernel(..))
            })?;
        InlineResult::Op(build_enode(eg, inline_nid, None, k, cache, loaded)?)
    } else {
        InlineResult::Load(load_class(eg, root, k, loaded)?)
    };

    let op_id = match result {
        InlineResult::Op(op) => op,
        InlineResult::Load(op) => op,
    };
    cache.insert(root, match result {
        InlineResult::Op(_) => InlineResult::Op(op_id),
        InlineResult::Load(_) => InlineResult::Load(op_id),
    });
    Some(op_id)
}

/// A class can be inlined if it has few enough parents and contains
/// an enode that can be computed inline (anything other than Leaf or Kernel).
/// Leaf and Kernel are terminal — they must be loaded from global memory.
/// Const is fine — it emits as a literal (`Op::ConstView`), not a global load.
/// Reduce is handled specially in `build_enode` (it loads its input).
fn can_inline(eg: &EGraph, root: ClassId) -> bool {
    if eg.classes[root].parents.len() > MAX_INLINE_PARENTS {
        return false;
    }
    eg.classes[root].nodes.iter().any(|&n| {
        !matches!(&eg.nodes[n], ENode::Leaf(_) | ENode::Kernel(..))
    })
}

/// Emit a `LoadView` for a class and return the resulting `OpId`.
/// Records `root` in `loaded` for building accurate kernel input metadata.
fn load_class(eg: &mut EGraph, cid: ClassId, k: &mut Kernel, loaded: &mut Vec<ClassId>) -> Option<OpId> {
    let root = eg.find_class(cid);
    loaded.push(root);
    let dtype = eg.classes[root].dtype?;
    let shape: Vec<Dim> = eg.classes[root].shape.clone()?.to_vec();
    Some(k.load_contiguous(dtype, &shape))
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
        let (_, c0) = eg.make(ENode::Leaf(DType::F32));
        let (_, c1) = eg.make(ENode::Expand(c0));
        let (_, c2) = eg.make(ENode::Expand(c1));

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
        let (_, leaf) = eg.make(ENode::Leaf(DType::F32));
        let (_, neg) = eg.make(ENode::Unary(leaf, UOp::Neg));
        let (_, abs) = eg.make(ENode::Unary(leaf, UOp::Abs));
        let (_, add) = eg.make(ENode::Binary(neg, abs, BOp::Add));

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
        let (_, l0) = eg.make(ENode::Leaf(DType::F32));
        let (_, l1) = eg.make(ENode::Leaf(DType::F64));
        let (_, e0) = eg.make(ENode::Expand(l0));

        let order = super::topo_sort_classes(&eg);
        assert_eq!(order.len(), 3, "expected 3 classes, got {}", order.len());

        let pos = |cid: ClassId| order.iter().position(|&x| x == cid).unwrap();
        // Both leaves before expand
        assert!(pos(l0) < pos(e0));
        // l1 can be anywhere, just check it's in order
        assert!(order.contains(&l1));
    }
}
