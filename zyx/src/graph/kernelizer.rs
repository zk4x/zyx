// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! E-graph kernelizer.
//!
//! Walks the egraph and builds fused kernel IRs by greedily inlining
//! operations into kernels, stopping at reduction/physical boundaries.
//! This mirrors the fusion strategy from the old kernelize.rs but works
//! on the egraph's class/enode structure.

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
    /// Only classes with multiple consumers (or  consumers that
    /// are all output classes) get their own kernel — single-consumer
    /// classes are inlined into their parent's kernel.
    ///
    /// After `kernelize_all`, every class reachable from an output must
    /// contain at least one `ENode::Kernel` or be a  `Leaf`/`Const` base
    /// case (the extracted plan can only execute kernels).
    ///
    /// # Invariant
    /// After `kernelize_all`, every class reachable from an output must contain
    /// at least one `ENode::Kernel` or be a `Leaf`/`Const` base case.  The
    /// extracted plan can only contain compiled kernels — non-kernel operator
    /// enodes (Binary, Cast, Expand, etc.) are not executable.  `extract_dp`
    /// panics if a class has no kernel alternative.
    pub(crate) fn kernelize_all(&mut self) {
        let order = topo_sort_classes(self);

        // Shared map: which classes have been computed and stored.
        // When a class is in `visited`, its kernel is registered and
        // consumers must  load  its value from global memory.
        let mut visited: Visited = Map::default();

        // Determine which classes need their own kernel.
        // A class needs a kernel if:
        //   • it is an output (0 parents) — no consumer to inline into
        //   • it has multiple consumers — store once, load by each
        //   • it is the input of a Reduce — Reduce always loads its input
        let mut needs_kernel: Set<ClassId> = Set::default();
        for (cid, class) in self.classes.iter() {
            let has_compute = class.nodes.iter().any(|nid| {
                !matches!(&self.nodes[*nid], ENode::Leaf(_) | ENode::Kernel(..) | ENode::Const(_))
            });
            if !has_compute {
                continue;
            }
            if class.parents.len() != 1 {
                // Multi-consumer (parents > 1) or output (parents == 0)
                needs_kernel.insert(cid);
                continue;
            }
            // Single-consumer: check if this class itself is a Reduce
            // or its parent is a Reduce.  Reduce is always a fusion boundary.
            let is_reduce = class.nodes.iter().any(|nid| {
                matches!(&self.nodes[*nid], ENode::Reduce(..))
            });
            let has_reduce_parent = class.parents.iter().any(|(parent_nid, _)| {
                matches!(&self.nodes[*parent_nid], ENode::Reduce(..))
            });
            if is_reduce || has_reduce_parent {
                needs_kernel.insert(cid);
            }
        }

        for cid in order {
            if !needs_kernel.contains(&cid) {
                continue;
            }

            let out_dtype = match self.classes[cid].dtype {
                Some(dt) => dt,
                None => continue,
            };

            // Build a kernel for each non-trivial enode in this class.
            let nids: Vec<NodeId> = self.classes[cid]
                .nodes
                .iter()
                .copied()
                .filter(|&nid| {
                    !matches!(&self.nodes[nid], ENode::Leaf(_) | ENode::Kernel(..) | ENode::Const(_))
                })
                .collect();

            for &nid in &nids {
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

                k.store_contiguous(result, out_dtype);

                let knid = self.make_kernel_enode(cid, &loaded, &k);
                self.kernel_irs.insert(knid, k);
                visited.insert(cid, (knid, result));
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
    let dtype = eg.classes[root].dtype?;
    let shape: Vec<Dim> = eg.classes[root].shape.clone()?.to_vec();
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
            let shape: Vec<Dim> = eg.classes[cid].shape.clone()?.to_vec();
            let x = resolve_child(eg, children[0], k, visited, loaded)?;
            Some(k.push_back(Op::Move {
                x,
                mop: Box::new(MoveOp::Expand { shape }),
            }))
        }
        ENode::Permute(_, axes) => {
            let cid = find_class_of(eg, nid, cid_override);
            let shape: Vec<Dim> = eg.classes[cid].shape.clone()?.to_vec();
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
            let shape: Vec<Dim> = eg.classes[cid].shape.clone()?.to_vec();
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
                eg.classes[root].shape.clone()?
            }
            .to_vec();
            let cid = find_class_of(eg, nid, cid_override);
            let out_shape: Vec<Dim> = eg.classes[cid].shape.clone()?.to_vec();
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
