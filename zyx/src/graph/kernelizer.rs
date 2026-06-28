// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Creates kernels from graph order, without launching or tuning.
//! Eventually replaces `kernelize.rs` once the full pipeline works.

use crate::{
    DType,
    graph::{Graph, Node},
    kernel::{BOp, Kernel, Op, OpId, UOp},
    slab::Slab,
    tensor::TensorId,
    view::View,
};
use crate::{Map, Set};
use std::hash::BuildHasherDefault;

pub use crate::kernelize::KMKernelId;

pub struct Kernelizer<'a> {
    order: &'a [TensorId],
    graph: &'a Graph,
    pub kernels: Slab<KMKernelId, Kernel>,
    visited: Map<TensorId, (KMKernelId, OpId)>,
    rcs: Map<TensorId, u32>,
    pending_stores: Set<TensorId>,
}

impl<'a> Kernelizer<'a> {
    pub fn new(order: &'a [TensorId], graph: &'a Graph, realized: &[TensorId]) -> Self {
        let mut rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(order.len(), BuildHasherDefault::new());
        for &nid in order {
            if !realized.contains(&nid) {
                for param in graph[nid].parameters() {
                    *rcs.entry(param).or_insert(0) += 1;
                }
            }
        }
        for &nid in order {
            rcs.entry(nid).or_insert(1);
        }
        Self {
            order,
            graph,
            kernels: Slab::with_capacity(30),
            visited: Map::with_capacity_and_hasher(100, BuildHasherDefault::new()),
            rcs,
            pending_stores: realized.iter().copied().collect(),
        }
    }

    fn has_pending_store(&self, nid: TensorId) -> bool {
        self.pending_stores.contains(&nid)
    }

    fn create_load_kernel(&mut self, nid: TensorId) -> (KMKernelId, OpId) {
        let shape = self.graph.shape(nid);
        let dtype = self.graph.dtype(nid);
        let mut kernel = Kernel {
            outputs: vec![nid; self.rcs[&nid] as usize],
            loads: vec![nid],
            stores: Vec::new(),
            ops: Slab::with_capacity(100),
            head: OpId::NULL,
            tail: OpId::NULL,
            device_id: crate::kernel::DeviceId::AUTO,
            custom_kernel_id: None,
        };
        let op_id = kernel.load_contiguous(dtype, shape);
        let kid = self.kernels.push(kernel);
        self.visited.insert(nid, (kid, op_id));
        (kid, op_id)
    }

    fn create_const_kernel(&mut self, nid: TensorId, value: crate::dtype::Constant) {
        let mut kernel = Kernel {
            outputs: vec![nid; self.rcs[&nid] as usize],
            loads: Vec::new(),
            stores: Vec::new(),
            ops: Slab::with_capacity(100),
            head: OpId::NULL,
            tail: OpId::NULL,
            device_id: crate::kernel::DeviceId::AUTO,
            custom_kernel_id: None,
        };
        let op_id = kernel.push_back(Op::ConstView(Box::new((value, View::contiguous(&[1])))));
        let kid = self.kernels.push(kernel);
        self.visited.insert(nid, (kid, op_id));
    }

    fn duplicate_or_store(&mut self, x: TensorId) -> (KMKernelId, OpId) {
        let (mut kid, mut op_id) = self.visited[&x];

        if self.kernels[kid].contains_stores() {
            self.add_store(x);
            (kid, op_id) = self.create_load_kernel(x);
            if self.kernels[kid].outputs.len() > 1 {
                kid = self.duplicate_kernel(x, kid);
            }
        }

        if self.kernels[kid].outputs.len() > 1 {
            let reduce_dims_big = self.kernels[kid].is_preceded_by_reduce(op_id);
            if reduce_dims_big {
                self.add_store(x);
                (kid, op_id) = self.create_load_kernel(x);
                if self.kernels[kid].outputs.len() > 1 {
                    kid = self.duplicate_kernel(x, kid);
                }
            } else {
                kid = self.duplicate_kernel(x, kid);
            }
        }

        (kid, op_id)
    }

    fn duplicate_kernel(&mut self, x: TensorId, kid: KMKernelId) -> KMKernelId {
        let mut kernel = self.kernels[kid].clone();
        kernel.outputs = vec![x];
        kernel.drop_unused_ops(&self.visited);
        self.kernels[kid].remove_first_output(x);
        self.kernels[kid].drop_unused_ops(&self.visited);
        self.kernels.push(kernel)
    }

    fn add_store(&mut self, x: TensorId) {
        let (kid, op_id) = self.visited[&x];
        if self.pending_stores.contains(&x) {
            self.visited.remove(&x).unwrap();
            self.kernels[kid].outputs.retain(|&elem| elem != x);
        } else {
            self.visited.remove(&x).unwrap();
            self.pending_stores.insert(x);
            let dtype = self.graph.dtype(x);
            self.kernels[kid].store_contiguous(op_id, dtype);
            self.kernels[kid].stores.push(x);
            self.kernels[kid].outputs.retain(|&elem| elem != x);
        }
    }

    fn add_cast_op(&mut self, nid: TensorId, x: TensorId, dtype: DType) {
        let (kid, op_id) = self.visited[&x];
        let kernel = &mut self.kernels[kid];
        let op_id = kernel.cast(op_id, dtype);
        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        self.visited.insert(nid, (kid, op_id));
    }

    fn add_unary_op(&mut self, nid: TensorId, x: TensorId, uop: UOp) {
        let (kid, op_id) = self.visited[&x];
        let kernel = &mut self.kernels[kid];
        let op_id = kernel.push_back(Op::Unary { x: op_id, uop });
        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        self.visited.insert(nid, (kid, op_id));
    }

    fn add_expand_op(&mut self, nid: TensorId, x: TensorId) {
        let (mut kid, mut op_id) = self.visited[&x];

        if self.kernels[kid].contains_stores() | self.kernels[kid].is_preceded_by_compute(op_id) {
            self.add_store(x);
            (kid, op_id) = self.create_load_kernel(x);
            if self.kernels[kid].outputs.len() > 1 {
                kid = self.duplicate_kernel(x, kid);
            }
        }

        if self.kernels[kid].outputs.len() > 1 {
            let reduce_dims_big = self.kernels[kid].is_preceded_by_reduce(op_id);
            if reduce_dims_big {
                self.add_store(x);
                (kid, op_id) = self.create_load_kernel(x);
                if self.kernels[kid].outputs.len() > 1 {
                    kid = self.duplicate_kernel(x, kid);
                }
            } else {
                kid = self.duplicate_kernel(x, kid);
            }
        }

        let shape = self.graph.shape(nid);
        let kernel = &mut self.kernels[kid];

        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(crate::kernel::MoveOp::Expand { shape: shape.into() }) });

        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        self.visited.insert(nid, (kid, op_id));
    }

    fn add_reshape_op(&mut self, nid: TensorId, x: TensorId) {
        debug_assert!(self.visited.contains_key(&x), "Missing tensor {x} in visited.");
        let (kid, op_id) = self.duplicate_or_store(x);
        let shape = self.graph.shape(nid);
        let kernel = &mut self.kernels[kid];

        let op_id =
            kernel.push_back(Op::Move { x: op_id, mop: Box::new(crate::kernel::MoveOp::Reshape { shape: shape.into() }) });

        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        self.visited.insert(nid, (kid, op_id));
    }

    fn add_permute_op(&mut self, nid: TensorId, x: TensorId) {
        let (kid, op_id) = self.duplicate_or_store(x);
        let axes: Vec<_> = self.graph.axes(nid).into();
        let kernel = &mut self.kernels[kid];

        let shape = self.graph.shape(nid).into();
        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(crate::kernel::MoveOp::Permute { axes, shape }) });

        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        self.visited.insert(nid, (kid, op_id));
    }

    fn add_pad_op(&mut self, nid: TensorId, x: TensorId) {
        let (kid, op_id) = self.duplicate_or_store(x);
        let padding = self.graph.padding(nid).into();
        let kernel = &mut self.kernels[kid];

        let shape = self.graph.shape(nid).into();
        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(crate::kernel::MoveOp::Pad { padding, shape }) });

        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        self.visited.insert(nid, (kid, op_id));
    }

    fn add_reduce_op(&mut self, nid: TensorId, x: TensorId, rop: BOp) {
        let axes = self.graph.axes(nid);
        let shape = self.graph.shape(x);

        let (mut kid, mut op_id) = self.visited[&x];
        if self.kernels[kid].contains_stores() | self.kernels[kid].is_preceded_by_reduce(op_id) {
            self.add_store(x);
            (kid, op_id) = self.create_load_kernel(x);
            if self.kernels[kid].outputs.len() > 1 {
                kid = self.duplicate_kernel(x, kid);
            }
        }

        if self.kernels[kid].outputs.len() > 1 {
            let reduce_dims_big = self.kernels[kid].is_preceded_by_reduce(op_id);
            if reduce_dims_big {
                self.add_store(x);
                (kid, op_id) = self.create_load_kernel(x);
                if self.kernels[kid].outputs.len() > 1 {
                    kid = self.duplicate_kernel(x, kid);
                }
            } else {
                kid = self.duplicate_kernel(x, kid);
            }
        }

        {
            let n = shape.len();
            let mut permute_axes = Vec::with_capacity(n);
            let max_axis = *axes.last().unwrap();
            let mut ai = 0;
            for i in 0..=max_axis {
                if axes[ai] == i {
                    ai += 1;
                } else {
                    permute_axes.push(i);
                }
            }
            permute_axes.extend(max_axis + 1..n);
            permute_axes.extend_from_slice(axes);

            if !permute_axes.iter().copied().eq(0..permute_axes.len()) {
                let shape = crate::shape::permute(self.graph.shape(x), &permute_axes);
                op_id = self.kernels[kid].push_back(Op::Move {
                    x: op_id,
                    mop: Box::new(crate::kernel::MoveOp::Permute { axes: permute_axes, shape }),
                });
            }
        }

        let kernel = &mut self.kernels[kid];
        op_id = kernel.push_back(Op::Reduce { x: op_id, rop, n_axes: axes.len() });
        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;

        if shape.len() == axes.len() {
            op_id = self.kernels[kid]
                .push_back(Op::Move { x: op_id, mop: Box::new(crate::kernel::MoveOp::Reshape { shape: vec![1] }) });
        }

        self.visited.insert(nid, (kid, op_id));
    }

    fn add_binary_op(&mut self, nid: TensorId, mut x: TensorId, mut y: TensorId, bop: BOp) {
        let (mut kid, mut op_id) = self.visited[&x];
        let (mut kidy, mut op_idy) = self.visited[&y];

        let kid_stores = !self.kernels[kid].stores.is_empty();
        let kidy_stores = !self.kernels[kidy].stores.is_empty();

        let new_op_id = if kid == kidy {
            let kernel = &mut self.kernels[kid];
            kernel.remove_first_output(x);
            kernel.remove_first_output(y);
            kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
            kernel.binary(op_id, op_idy, bop)
        } else {
            match (kid_stores, kidy_stores) {
                (true, true) => {
                    self.add_store(x);
                    (kid, op_id) = self.create_load_kernel(x);
                    if self.kernels[kid].outputs.len() > 1 {
                        kid = self.duplicate_kernel(x, kid);
                        self.kernels[kid].outputs.push(x);
                    }
                    self.add_store(y);
                    (kidy, op_idy) = self.create_load_kernel(y);
                    if self.kernels[kidy].outputs.len() > 1 {
                        kidy = self.duplicate_kernel(y, kidy);
                        self.kernels[kidy].outputs.push(y);
                    }
                }
                (true, false) => {
                    self.add_store(x);
                    (kid, op_id) = self.create_load_kernel(x);
                    if self.kernels[kid].outputs.len() > 1 {
                        kid = self.duplicate_kernel(x, kid);
                        self.kernels[kid].outputs.push(x);
                    }
                }
                (false, true) => {
                    self.add_store(y);
                    (kidy, op_idy) = self.create_load_kernel(y);
                    if self.kernels[kidy].outputs.len() > 1 {
                        kidy = self.duplicate_kernel(y, kidy);
                        self.kernels[kidy].outputs.push(y);
                    }
                }
                (false, false) => {}
            }

            let swapped_xy = if self.kernels[kidy].is_reduce() && !self.kernels[kid].is_reduce() {
                std::mem::swap(&mut kid, &mut kidy);
                std::mem::swap(&mut op_id, &mut op_idy);
                std::mem::swap(&mut x, &mut y);
                true
            } else {
                false
            };

            self.kernels[kidy].remove_first_output(y);
            let Kernel { outputs, loads, stores, ops, head, tail: _, device_id: _, custom_kernel_id: _ } =
                unsafe { self.kernels.remove_and_return(kidy) };

            let mut y_ops_map = Map::with_capacity_and_hasher(5, BuildHasherDefault::new());

            let mut i = head;
            while !i.is_null() {
                let mut op = ops[i].op.clone();
                for param in op.parameters_mut() {
                    *param = y_ops_map[param];
                }
                let new_op_id = self.kernels[kid].push_back(op);
                y_ops_map.insert(i, new_op_id);
                i = ops[i].next;
            }

            for (kidm, op_id) in self.visited.values_mut() {
                if *kidm == kidy {
                    *kidm = kid;
                    if let Some(new_op_id) = y_ops_map.get(op_id) {
                        *op_id = *new_op_id;
                    }
                }
            }

            self.kernels[kid].loads.extend(loads);
            self.kernels[kid].stores.extend(stores);

            self.kernels[kid].remove_first_output(x);
            self.kernels[kid].outputs.extend(outputs);
            self.kernels[kid].outputs.extend(vec![nid; self.rcs[&nid] as usize]);

            if swapped_xy {
                self.kernels[kid].binary(y_ops_map[&op_idy], op_id, bop)
            } else {
                self.kernels[kid].binary(op_id, y_ops_map[&op_idy], bop)
            }
        };

        *self.rcs.get_mut(&x).unwrap() -= 1;
        *self.rcs.get_mut(&y).unwrap() -= 1;
        self.visited.insert(nid, (kid, new_op_id));
    }

    /// Walk through `order` and create kernels for every node.
    pub fn kernelize(mut self, to_eval: &Set<TensorId>) -> Slab<KMKernelId, Kernel> {
        for &nid in self.order {
            /*println!(
                "{}{nid} x {} -> {:?}  {}  {:?}",
                if self.has_pending_store(nid) { "LOAD " } else { "" },
                self.rcs[&nid],
                self.graph[nid],
                self.graph.dtype(nid),
                self.graph.shape(nid)
            );*/
            if self.has_pending_store(nid) {
                self.create_load_kernel(nid);
            } else {
                match self.graph[nid] {
                    Node::Leaf { .. } => unreachable!(),
                    Node::Const { value } => self.create_const_kernel(nid, value),
                    Node::Cast { x, dtype } => self.add_cast_op(nid, x, dtype),
                    Node::Unary { x, uop } => self.add_unary_op(nid, x, uop),
                    Node::Expand { x } => self.add_expand_op(nid, x),
                    Node::Permute { x } => self.add_permute_op(nid, x),
                    Node::Reshape { x } => self.add_reshape_op(nid, x),
                    Node::Pad { x } => self.add_pad_op(nid, x),
                    Node::Reduce { x, rop } => self.add_reduce_op(nid, x, rop),
                    Node::Binary { x, y, bop } => self.add_binary_op(nid, x, y, bop),
                    Node::ToDevice { x, .. } => {
                        self.add_store(x);
                        let (kid, _) = self.create_load_kernel(x);
                        self.kernels[kid].device_id = crate::kernel::DeviceId::AUTO;
                        self.add_cast_op(nid, x, self.graph.dtype(x));
                    }
                    Node::Custom(_) => {
                        // Custom kernels not yet handled in graph kernelizer
                    }
                }
            }

            if to_eval.contains(&nid) && !self.has_pending_store(nid) {
                self.add_store(nid);
                *self.rcs.get_mut(&nid).unwrap() -= 1;
                if self.rcs[&nid] > 0 {
                    self.create_load_kernel(nid);
                }
            }
        }
        self.kernels
    }
}
