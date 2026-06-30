// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! E-graph kernelizer.
//!
//! Walks the e-graph's e-classes and inserts `ENode::Kernel` variants for
//! different fusion granularities, building real `Kernel` IR for each variant.

use crate::{
    backend::ProgramId,
    graph::search::{ClassId, EGraph, ENode, NodeId},
    kernel::{DeviceId, Kernel, MoveOp, Op},
    shape::{Dim, UAxis},
    view::View,
};

const HIGH_COST: u64 = 1_000_000;

fn kernel_cost(ops_count: u32) -> u64 {
    ((1.0 + ops_count as f64).ln() * HIGH_COST as f64) as u64
}

impl EGraph {
    /// Walk all e-classes, inserting `ENode::Kernel` alternatives and building
    /// the corresponding `Kernel` IR (without stores — stores are added at
    /// launch time, matching `kernelize.rs`).
    pub(crate) fn kernelize_all(&mut self) {
        let classes: Vec<ClassId> = self.classes.ids().collect();

        // First pass: create a kernel for every eligible enode.
        for &cid in &classes {
            if !self.classes.contains_key(cid) {
                continue;
            }
            let nodes: Vec<NodeId> = self.classes[cid].nodes.clone();
            for &nid in &nodes {
                if self.nodes[nid].is_kernel() || self.nodes[nid].is_transform() || matches!(&self.nodes[nid], ENode::Leaf(_)) {
                    continue;
                }
                let mut inputs: Vec<ClassId> = self.nodes[nid].child_classes().to_vec();
                inputs.retain(|&c| {
                    !self.classes[self.find_class(c)]
                        .nodes
                        .iter()
                        .any(|&n| matches!(&self.nodes[n], ENode::Leaf(_)))
                });
                inputs.sort();
                let inputs: Box<[ClassId]> = inputs.into_boxed_slice();
                let outputs: Box<[ClassId]> = vec![cid].into_boxed_slice();
                let (knid, _) = self.make(ENode::Kernel(inputs, outputs, ProgramId::NULL));
                self.costs.insert(knid, kernel_cost(1));
                self.ops_count.insert(knid, 1);
                self.add_to_class(knid, cid);

                if let Some(kernel) = self.build_kernel(nid) {
                    self.kernel_irs.insert(knid, kernel);
                }
            }
        }

        // Fixpoint: extend kernels forward through ops (transforms + compute).
        for _ in 0..5 {
            let mut changed = false;
            let classes: Vec<ClassId> = self.classes.ids().collect();

            for &cid in &classes {
                if !self.classes.contains_key(cid) {
                    continue;
                }
                let nodes: Vec<NodeId> = self.classes[cid].nodes.clone();
                for &nid in &nodes {
                    let op = &self.nodes[nid];
                    if op.is_kernel() {
                        continue;
                    }

                    let child_classes: Vec<ClassId> = op.child_classes();
                    if child_classes.is_empty() {
                        continue;
                    }

                    for &child in &child_classes {
                        if !self.classes.contains_key(child) {
                            continue;
                        }
                        let child_nodes: Vec<NodeId> = self.classes[child].nodes.clone();
                        for &cnid in &child_nodes {
                            let ENode::Kernel(child_inputs, child_outputs, _) = &self.nodes[cnid] else {
                                continue;
                            };
                            if !child_outputs.contains(&child) {
                                continue;
                            }

                            let mut new_inputs: Vec<ClassId> = child_inputs.to_vec();
                            for &other_child in &child_classes {
                                if other_child != child {
                                    if self.classes[self.find_class(other_child)]
                                        .nodes
                                        .iter()
                                        .any(|&n| matches!(&self.nodes[n], ENode::Leaf(_)))
                                    {
                                        continue;
                                    }
                                    new_inputs.push(other_child);
                                }
                            }
                            new_inputs.sort();
                            new_inputs.dedup();
                            let outputs: Vec<ClassId> = vec![cid];
                            let (knid, _) = self.make(ENode::Kernel(
                                new_inputs.into_boxed_slice(),
                                outputs.clone().into_boxed_slice(),
                                ProgramId::NULL,
                            ));

                            let child_ops = self.ops_count.get(&cnid).copied().unwrap_or(1);
                            let total_ops = 1 + child_ops;
                            self.costs.insert(knid, kernel_cost(total_ops));
                            self.ops_count.insert(knid, total_ops);

                            for &out in &outputs {
                                self.add_to_class(knid, out);
                            }
                            changed = true;

                            if let Some(kernel) = self.extend_kernel(cnid, nid, child, &child_classes) {
                                self.kernel_irs.insert(knid, kernel);
                            }
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }
    }

    /// Build a kernel for a single enode (no fusion). No stores — matches
    /// `kernelize.rs` which creates kernels without stores.
    fn build_kernel(&self, nid: NodeId) -> Option<Kernel> {
        let op = &self.nodes[nid];
        let inputs: Vec<ClassId> = op.child_classes().to_vec();
        let mut kernel = Kernel::new(DeviceId::AUTO);

        match op {
            ENode::Binary(_, _, bop) => {
                let lhs_dtype = self.classes[inputs[0]].dtype?;
                let lhs_shape: Vec<Dim> = self.classes[inputs[0]].shape.clone()?.to_vec();
                let rhs_dtype = self.classes[inputs[1]].dtype?;
                let rhs_shape: Vec<Dim> = self.classes[inputs[1]].shape.clone()?.to_vec();
                let lhs = kernel.load_contiguous(lhs_dtype, &lhs_shape);
                let rhs = kernel.load_contiguous(rhs_dtype, &rhs_shape);
                kernel.binary(lhs, rhs, *bop);
                Some(kernel)
            }
            ENode::Unary(_, uop) => {
                let in_dtype = self.classes[inputs[0]].dtype?;
                let in_shape: Vec<Dim> = self.classes[inputs[0]].shape.clone()?.to_vec();
                let load = kernel.load_contiguous(in_dtype, &in_shape);
                kernel.push_back(Op::Unary { x: load, uop: *uop });
                Some(kernel)
            }
            ENode::Cast(_, dt) => {
                let in_dtype = self.classes[inputs[0]].dtype?;
                let in_shape: Vec<Dim> = self.classes[inputs[0]].shape.clone()?.to_vec();
                let load = kernel.load_contiguous(in_dtype, &in_shape);
                kernel.cast(load, *dt);
                Some(kernel)
            }
            ENode::Reduce(_, rop) => {
                let in_dtype = self.classes[inputs[0]].dtype?;
                let in_shape: Vec<Dim> = self.classes[inputs[0]].shape.clone()?.to_vec();
                let out_shape: Vec<Dim> = self.classes[self.find(nid)].shape.clone()?.to_vec();
                let load = kernel.load_contiguous(in_dtype, &in_shape);

                let n_axes = in_shape.len().saturating_sub(out_shape.len());
                let result = kernel.push_back(Op::Reduce {
                    x: load,
                    rop: *rop,
                    n_axes,
                });

                if out_shape.len() == 1 && n_axes > 0 && in_shape.len() > 1 {
                    kernel.reshape(result, &out_shape);
                }
                Some(kernel)
            }
            ENode::Const(v) => {
                kernel.push_back(Op::ConstView(Box::new((*v, View::contiguous(&[1])))));
                Some(kernel)
            }
            _ => None,
        }
    }

    /// Extend a child kernel (`cnid`) forward through the enode at `nid`.
    /// No stores — same as `kernelize.rs` which extends first and stores later.
    fn extend_kernel(&self, cnid: NodeId, nid: NodeId, child_class: ClassId, child_classes: &[ClassId]) -> Option<Kernel> {
        let mut kernel = self.kernel_irs.get(&cnid)?.clone();

        // The last op in the child kernel is the last compute result.
        let child_result = kernel.tail;
        if child_result.is_null() {
            return None;
        }

        let op = &self.nodes[nid];
        let out_shape: Vec<Dim> = self.classes[self.find(nid)].shape.clone()?.to_vec();

        match op {
            ENode::Expand(_) => {
                kernel.push_back(Op::Move {
                    x: child_result,
                    mop: Box::new(MoveOp::Expand { shape: out_shape }),
                });
            }
            ENode::Permute(_, axes) => {
                let axes: Vec<UAxis> = axes.to_vec();
                kernel.push_back(Op::Move {
                    x: child_result,
                    mop: Box::new(MoveOp::Permute { axes, shape: out_shape }),
                });
            }
            ENode::Reshape(_, shape) => {
                let shape: Vec<Dim> = shape.to_vec();
                kernel.push_back(Op::Move {
                    x: child_result,
                    mop: Box::new(MoveOp::Reshape { shape }),
                });
            }
            ENode::Pad(_, padding) => {
                let padding: Vec<(i64, i64)> = padding.to_vec();
                kernel.push_back(Op::Move {
                    x: child_result,
                    mop: Box::new(MoveOp::Pad {
                        padding,
                        shape: out_shape,
                    }),
                });
            }
            ENode::Cast(_, dt) => {
                kernel.cast(child_result, *dt);
            }
            ENode::Unary(_, uop) => {
                kernel.push_back(Op::Unary {
                    x: child_result,
                    uop: *uop,
                });
            }
            ENode::Reduce(_, rop) => {
                let in_shape_len = self.classes[self.find_class(child_class)].shape.as_ref()?.len();
                let n_axes = in_shape_len.saturating_sub(out_shape.len());
                kernel.push_back(Op::Reduce {
                    x: child_result,
                    rop: *rop,
                    n_axes,
                });
            }
            ENode::Binary(_, _, bop) => {
                let other_class = child_classes.iter().find(|&&c| c != child_class)?;
                let other_cid = self.find_class(*other_class);

                let other_is_const = self.classes[other_cid]
                    .nodes
                    .iter()
                    .any(|&n| matches!(&self.nodes[n], ENode::Const(_)));

                let other_load = if other_is_const {
                    let const_val = self.classes[other_cid].nodes.iter().find_map(|&n| {
                        if let ENode::Const(c) = &self.nodes[n] {
                            Some(*c)
                        } else {
                            None
                        }
                    })?;
                    kernel.push_back(Op::ConstView(Box::new((const_val, View::contiguous(&[1])))))
                } else {
                    let other_dtype = self.classes[other_cid].dtype?;
                    let other_shape: Vec<Dim> = self.classes[other_cid].shape.clone()?.to_vec();
                    kernel.load_contiguous(other_dtype, &other_shape)
                };

                let (lhs, rhs) = match &self.nodes[nid] {
                    ENode::Binary(a, _, _) if *a == child_class => (child_result, other_load),
                    _ => (other_load, child_result),
                };
                kernel.binary(lhs, rhs, *bop);
            }
            _ => {}
        }

        Some(kernel)
    }
}
