// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! E-graph kernelizer.
//!
//! Creates one `ENode::Kernel` per enode, each backed by a standalone `Kernel`
//! IR: load inputs → op → store.  No fusion.  Fusion is a separate pass.

use crate::{
    DType,
    backend::ProgramId,
    graph::search::{ClassId, EGraph, ENode, NodeId},
    kernel::{DeviceId, Kernel, MoveOp, Op, OpId},
    shape::{Dim, UAxis},
    view::View,
};

const HIGH_COST: u64 = 1_000_000;

fn kernel_cost(ops_count: u32) -> u64 {
    ((1.0 + ops_count as f64).ln() * HIGH_COST as f64) as u64
}

impl EGraph {
    /// Walk all enodes, insert a `Kernel` enode for each, and build its IR.
    pub(crate) fn kernelize_all(&mut self) {
        let classes: Vec<ClassId> = self.classes.ids().collect();

        for &cid in &classes {
            if !self.classes.contains_key(cid) {
                continue;
            }
            let nodes: Vec<NodeId> = self.classes[cid].nodes.clone();
            for &nid in &nodes {
                let op = &self.nodes[nid];
                if matches!(op, ENode::Leaf(_) | ENode::Kernel(..)) {
                    continue;
                }

                let out_dtype = match self.classes[cid].dtype {
                    Some(dt) => dt,
                    None => continue,
                };

                let child_classes: Vec<ClassId> = op.child_classes().to_vec();
                let kernel_knid = self.make_kernel_enode(cid, &child_classes);
                if let Some(kernel) = build_kernel(self, nid, &child_classes, out_dtype) {
                    self.kernel_irs.insert(kernel_knid, kernel);
                }
            }
        }
    }

    fn make_kernel_enode(&mut self, cid: ClassId, inputs: &[ClassId]) -> NodeId {
        let inputs: Box<[ClassId]> = inputs.to_vec().into_boxed_slice();
        let outputs: Box<[ClassId]> = vec![cid].into_boxed_slice();
        let (knid, _) = self.make(ENode::Kernel(inputs, outputs, ProgramId::NULL));
        self.costs.insert(knid, kernel_cost(1));
        self.ops_count.insert(knid, 1);
        self.add_to_class(knid, cid);
        knid
    }
}

fn build_kernel(eg: &EGraph, nid: NodeId, inputs: &[ClassId], out_dtype: DType) -> Option<Kernel> {
    let op = &eg.nodes[nid];
    let mut k = Kernel::new(DeviceId::AUTO);

    match op {
        ENode::Binary(_, _, bop) => {
            let lhs = load(&mut k, eg, inputs[0])?;
            let rhs = load(&mut k, eg, inputs[1])?;
            k.binary(lhs, rhs, *bop);
        }
        ENode::Unary(_, uop) => {
            let x = load(&mut k, eg, inputs[0])?;
            k.push_back(Op::Unary { x, uop: *uop });
        }
        ENode::Cast(_, dt) => {
            let x = load(&mut k, eg, inputs[0])?;
            k.cast(x, *dt);
        }
        ENode::Reduce(_, rop) => {
            let x = load(&mut k, eg, inputs[0])?;
            let in_shape: Vec<Dim> = eg.classes[eg.find_class(inputs[0])].shape.clone()?.to_vec();
            let out_shape: Vec<Dim> = eg.classes[eg.find(nid)].shape.clone()?.to_vec();
            let n_axes = in_shape.len().saturating_sub(out_shape.len());
            let r = k.push_back(Op::Reduce { x, rop: *rop, n_axes });
            if out_shape.len() == 1 && n_axes > 0 && in_shape.len() > 1 {
                k.reshape(r, &out_shape);
            }
        }
        ENode::Const(v) => {
            k.push_back(Op::ConstView(Box::new((*v, View::contiguous(&[1])))));
        }
        ENode::Expand(_) => {
            let x = load(&mut k, eg, inputs[0])?;
            let shape: Vec<Dim> = eg.classes[eg.find(nid)].shape.clone()?.to_vec();
            k.push_back(Op::Move {
                x,
                mop: Box::new(MoveOp::Expand { shape }),
            });
        }
        ENode::Permute(_, axes) => {
            let x = load(&mut k, eg, inputs[0])?;
            let shape: Vec<Dim> = eg.classes[eg.find(nid)].shape.clone()?.to_vec();
            let axes: Vec<UAxis> = axes.to_vec();
            k.push_back(Op::Move {
                x,
                mop: Box::new(MoveOp::Permute { axes, shape }),
            });
        }
        ENode::Reshape(_, shape) => {
            let x = load(&mut k, eg, inputs[0])?;
            let shape: Vec<Dim> = shape.to_vec();
            k.push_back(Op::Move {
                x,
                mop: Box::new(MoveOp::Reshape { shape }),
            });
        }
        ENode::Pad(_, padding) => {
            let x = load(&mut k, eg, inputs[0])?;
            let shape: Vec<Dim> = eg.classes[eg.find(nid)].shape.clone()?.to_vec();
            let padding: Vec<(i64, i64)> = padding.to_vec();
            k.push_back(Op::Move {
                x,
                mop: Box::new(MoveOp::Pad { padding, shape }),
            });
        }
        ENode::ToDevice(_, _) => return None,
        _ => return None,
    }

    if k.tail.is_null() {
        return None;
    }
    k.store_contiguous(k.tail, out_dtype);
    Some(k)
}

fn load(k: &mut Kernel, eg: &EGraph, cid: ClassId) -> Option<OpId> {
    let dtype = eg.classes[eg.find_class(cid)].dtype?;
    let shape: Vec<Dim> = eg.classes[eg.find_class(cid)].shape.clone()?.to_vec();
    Some(k.load_contiguous(dtype, &shape))
}
