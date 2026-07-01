// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! E-graph kernelizer.
//!
//! Walks the egraph and builds fused kernel IRs by greedily inlining
//! operations into kernels, stopping at reduction/physical boundaries.
//! This mirrors the fusion strategy from the old kernelize.rs but works
//! on the egraph's class/enode structure.

use crate::{
    DType,
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
    /// Walk every class and build a fused kernel IR for each non-trivial
    /// enode.  Each kernel stores its result — this makes it a candidate
    /// that later classes can either inline (by picking the original enode)
    /// or load from (by picking the kernel enode).  The extract phase
    /// chooses the cheapest combination.
    ///
    /// Leaf, Kernel, and Const enodes are skipped (they have no computation
    /// to emit).
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

            // Variant 1: fused kernel — inline single-parent children,
            // load multi-parent ones.  This avoids materialising
            // intermediates when the child has only one consumer.
            if let Some((kernel, loaded_classes)) = build_fused_kernel(self, nid, cid, out_dtype) {
                let knid = self.make_kernel_enode(cid, &loaded_classes, &kernel);
                self.kernel_irs.insert(knid, kernel);
            }

            // Variant 2: load-everything kernel — every child is loaded
            // from global memory.  This is the "store" alternative:
            // intermediates are always materialised.
            if let Some((kernel, loaded_classes)) = build_load_kernel(self, nid, cid, out_dtype) {
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

// ── Fused kernel builder ──────────────────────────────────

/// Build a fused kernel for the computation represented by `nid` in class `cid`.
///
/// Starting from the given enode, the builder recursively inlines fusible
/// input classes until it hits a boundary (Reduce, multi-consumer, Leaf, Const).
/// At boundaries a `LoadView` is emitted instead.
///
/// Returns `(Kernel, Vec<ClassId>)` where the vec lists every class whose
/// data is loaded from global memory in the kernel.
fn build_fused_kernel(
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

/// Build a "load everything" kernel for `nid` — every child class is loaded
/// from global memory and then the single operation for this enode is applied.
/// This is the "store" alternative: intermediates are always materialised.
fn build_load_kernel(
    eg: &mut EGraph,
    nid: NodeId,
    cid: ClassId,
    out_dtype: DType,
) -> Option<(Kernel, Vec<ClassId>)> {
    let mut k = Kernel::new(DeviceId::AUTO);
    let mut loaded: Vec<ClassId> = Vec::new();
    let kind = eg.nodes[nid].clone();
    let children: Vec<ClassId> = kind.child_classes();

    match kind {
        ENode::Const(v) => {
            k.push_back(Op::ConstView(Box::new((v, View::contiguous(&[1])))));
            k.store_contiguous(k.tail, out_dtype);
            return Some((k, loaded));
        }
        ENode::Expand(..) => {
            let in_cid = find_class_of(eg, nid, Some(cid));
            let shape: Vec<Dim> = eg.classes[in_cid].shape.clone()?.to_vec();
            let x = load_class(eg, children[0], &mut k, &mut loaded)?;
            k.push_back(Op::Move { x, mop: Box::new(MoveOp::Expand { shape }) });
        }
        ENode::Permute(_, ref axes) => {
            let in_cid = find_class_of(eg, nid, Some(cid));
            let shape: Vec<Dim> = eg.classes[in_cid].shape.clone()?.to_vec();
            let axes: Vec<UAxis> = axes.clone().into_vec();
            let x = load_class(eg, children[0], &mut k, &mut loaded)?;
            k.push_back(Op::Move { x, mop: Box::new(MoveOp::Permute { axes, shape }) });
        }
        ENode::Reshape(_, ref shape) => {
            let shape: Vec<Dim> = shape.clone().into_vec();
            let x = load_class(eg, children[0], &mut k, &mut loaded)?;
            k.push_back(Op::Move { x, mop: Box::new(MoveOp::Reshape { shape }) });
        }
        ENode::Pad(_, ref padding) => {
            let in_cid = find_class_of(eg, nid, Some(cid));
            let shape: Vec<Dim> = eg.classes[in_cid].shape.clone()?.to_vec();
            let padding: Vec<(i64, i64)> = padding.clone().into_vec();
            let x = load_class(eg, children[0], &mut k, &mut loaded)?;
            k.push_back(Op::Move { x, mop: Box::new(MoveOp::Pad { padding, shape }) });
        }
        ENode::Reduce(_, rop) => {
            let x = load_class(eg, children[0], &mut k, &mut loaded)?;
            let in_root = eg.find_class(children[0]);
            let in_shape: Vec<Dim> = eg.classes[in_root].shape.clone()?.to_vec();
            let out_cid = find_class_of(eg, nid, Some(cid));
            let out_shape: Vec<Dim> = eg.classes[out_cid].shape.clone()?.to_vec();
            let n_axes = in_shape.len().saturating_sub(out_shape.len());
            let permuted = try_permute_reduce_axes(&mut k, x, &in_shape, &out_shape, n_axes);
            let r = k.push_back(Op::Reduce { x: permuted, rop, n_axes: n_axes as UAxis });
            if out_shape.len() == 1 && n_axes > 0 && in_shape.len() > 1 {
                k.reshape(r, &out_shape);
            }
        }
        ENode::Cast(_, dt) => {
            let x = load_class(eg, children[0], &mut k, &mut loaded)?;
            k.cast(x, dt);
        }
        ENode::Unary(_, uop) => {
            let x = load_class(eg, children[0], &mut k, &mut loaded)?;
            k.push_back(Op::Unary { x, uop });
        }
        ENode::Binary(_, _, bop) => {
            let lhs = load_class(eg, children[0], &mut k, &mut loaded)?;
            let rhs = load_class(eg, children[1], &mut k, &mut loaded)?;
            k.binary(lhs, rhs, bop);
        }
        ENode::ToDevice(..) | ENode::Kernel(..) | ENode::Leaf(_) => return None,
    }

    if k.tail.is_null() {
        return None;
    }
    k.store_contiguous(k.tail, out_dtype);
    Some((k, loaded))
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
