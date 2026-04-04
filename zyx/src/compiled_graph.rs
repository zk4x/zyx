// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: AGPL-3.0-only

//! Compiled graph caching layer.

use crate::{
    backend::{BufferId, ProgramId},
    cache::DeviceId,
    dtype::Constant,
    graph::Node,
    kernel::{BOp, Kernel, MoveOp, Op, OpId, OpNode, UOp},
    kernelize::KMKernelId,
    runtime::Runtime,
    shape::{Dim, UAxis},
    slab::Slab,
    tensor::TensorId,
    view::View,
    DType, Map, Set, ZyxError,
};
use std::collections::BTreeMap;
use std::hash::BuildHasherDefault;

/// Cached result of compiling a graph, ready for replay.
#[allow(dead_code)]
pub struct CompiledGraph {
    pub nodes: Vec<CompiledNode>,
    pub buffer_slots: Vec<BufferId>,
}

/// Index into CompiledGraph::buffer_slots, used instead of raw BufferId
/// so the compiled graph is stable across runs with different buffer IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferSlot(pub u32);

/// A single step in a compiled graph execution.
#[derive(Debug, Clone)]
pub enum CompiledNode {
    Allocate {
        pool: usize,
        size: usize,
        slot: BufferSlot,
    },
    Deallocate {
        pool: usize,
        slot: BufferSlot,
    },
    CopyMemory {
        src_pool: usize,
        src_buffer: BufferSlot,
        dst_pool: usize,
        dst_buffer: BufferSlot,
    },
    LaunchProgram {
        program: ProgramId,
        device: DeviceId,
    },
}

/// Compact representation of a graph, used as cache key.
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct CompactedGraph {
    pub nodes: Vec<Node>,
    pub shapes: BTreeMap<TensorId, Box<[Dim]>>,
    pub paddings: BTreeMap<TensorId, Box<[(i32, i32)]>>,
    pub axes: BTreeMap<TensorId, Box<[UAxis]>>,
}

struct GraphKernelizer {
    realized_nodes: Set<TensorId>,
    virt_realized_nodes: Set<TensorId>,
    kernels: Slab<KMKernelId, Kernel>,
    visited: Map<TensorId, (KMKernelId, OpId)>,
    rcs: Map<TensorId, u32>,
    compacted: CompactedGraph,
}

impl GraphKernelizer {
    fn new(compacted: CompactedGraph, rcs: Map<TensorId, u32>, realized_nodes: Set<TensorId>) -> Self {
        Self {
            realized_nodes,
            virt_realized_nodes: Set::with_hasher(BuildHasherDefault::new()),
            kernels: Slab::with_capacity(30),
            visited: Map::with_capacity_and_hasher(100, BuildHasherDefault::new()),
            rcs,
            compacted,
        }
    }

    fn shape(&self, tensor_id: TensorId) -> Vec<Dim> {
        self.compacted.shapes[&tensor_id].to_vec()
    }

    fn dtype(&self, tensor_id: TensorId) -> DType {
        let mut tid = tensor_id;
        for _ in 0..100_000 {
            match self.compacted.nodes[usize::from(tid)] {
                Node::Const { value } => return value.dtype(),
                Node::Leaf { dtype } | Node::Cast { dtype, .. } => return dtype,
                Node::Binary { bop, .. } if bop.returns_bool() => return DType::Bool,
                _ => tid = self.compacted.nodes[usize::from(tid)].param1(),
            }
        }
        panic!("DType not found")
    }

    fn padding(&self, tensor_id: TensorId) -> Vec<(i32, i32)> {
        self.compacted.paddings[&tensor_id].to_vec()
    }

    fn axes(&self, tensor_id: TensorId) -> Vec<UAxis> {
        self.compacted.axes[&tensor_id].to_vec()
    }

    fn duplicate_or_store(&mut self, x: TensorId) -> Result<(KMKernelId, OpId), ZyxError> {
        let (mut kid, mut op_id) = self.visited[&x];

        if self.kernels[kid].contains_stores() {
            self.add_store(x)?;
            (kid, op_id) = self.create_load_kernel(x);
            if self.kernels[kid].outputs.len() > 1 {
                kid = self.duplicate_kernel(x, kid);
            }
        }

        if self.kernels[kid].outputs.len() > 1 {
            let reduce_dims_big = self.kernels[kid].is_preceded_by_reduce(op_id);
            if reduce_dims_big {
                self.add_store(x)?;
                (kid, op_id) = self.create_load_kernel(x);
                if self.kernels[kid].outputs.len() > 1 {
                    kid = self.duplicate_kernel(x, kid);
                }
            } else {
                kid = self.duplicate_kernel(x, kid);
            }
        }

        Ok((kid, op_id))
    }

    fn duplicate_kernel(&mut self, x: TensorId, kid: KMKernelId) -> KMKernelId {
        let mut kernel = self.kernels[kid].clone();
        kernel.outputs = vec![x];
        kernel.drop_unused_ops(&self.visited);
        self.kernels[kid].remove_first_output(x);
        self.kernels[kid].drop_unused_ops(&self.visited);
        self.kernels.push(kernel)
    }

    fn create_load_kernel(&mut self, nid: TensorId) -> (KMKernelId, OpId) {
        let shape = self.shape(nid);
        let dtype = self.dtype(nid);
        let mut ops = Slab::with_capacity(100);
        let op = Op::LoadView(Box::new((dtype, View::contiguous(&shape))));
        let op_id = ops.push(OpNode { prev: OpId::NULL, next: OpId::NULL, op });
        let kernel = Kernel {
            outputs: vec![nid; self.rcs[&nid] as usize],
            loads: vec![nid],
            stores: Vec::new(),
            ops,
            head: op_id,
            tail: op_id,
        };
        let kid = self.kernels.push(kernel);
        self.visited.insert(nid, (kid, op_id));
        (kid, op_id)
    }

    fn create_const_kernel(&mut self, nid: TensorId, value: Constant) {
        let mut ops = Slab::with_capacity(100);
        let op = Op::ConstView(Box::new((value, View::contiguous(&[1]))));
        let op_id = ops.push(OpNode { prev: OpId::NULL, next: OpId::NULL, op });
        let kernel = Kernel {
            outputs: vec![nid; self.rcs[&nid] as usize],
            loads: Vec::new(),
            stores: Vec::new(),
            ops,
            head: op_id,
            tail: op_id,
        };
        let kid = self.kernels.push(kernel);
        self.visited.insert(nid, (kid, op_id));
    }

    fn add_expand_op(&mut self, nid: TensorId, x: TensorId) -> Result<(), ZyxError> {
        let (mut kid, mut op_id) = self.visited[&x];

        if self.kernels[kid].contains_stores() | self.kernels[kid].is_reduce() {
            self.add_store(x)?;
            (kid, op_id) = self.create_load_kernel(x);
            if self.kernels[kid].outputs.len() > 1 {
                kid = self.duplicate_kernel(x, kid);
            }
        }

        if self.kernels[kid].outputs.len() > 1 {
            let reduce_dims_big = self.kernels[kid].is_preceded_by_reduce(op_id);
            if reduce_dims_big {
                self.add_store(x)?;
                (kid, op_id) = self.create_load_kernel(x);
                if self.kernels[kid].outputs.len() > 1 {
                    kid = self.duplicate_kernel(x, kid);
                }
            } else {
                kid = self.duplicate_kernel(x, kid);
            }
        }

        let shape = self.shape(nid);
        let kernel = &mut self.kernels[kid];
        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Expand { shape }) });

        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_reshape_op(&mut self, nid: TensorId, x: TensorId) -> Result<(), ZyxError> {
        let (kid, op_id) = self.duplicate_or_store(x)?;
        let shape = self.shape(nid);
        let kernel = &mut self.kernels[kid];

        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Reshape { shape }) });

        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_permute_op(&mut self, nid: TensorId, x: TensorId) -> Result<(), ZyxError> {
        let (kid, op_id) = self.duplicate_or_store(x)?;
        let axes = self.axes(nid);
        let shape = self.shape(nid);
        let kernel = &mut self.kernels[kid];

        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes, shape }) });

        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_pad_op(&mut self, nid: TensorId, x: TensorId) -> Result<(), ZyxError> {
        let (kid, op_id) = self.duplicate_or_store(x)?;
        let padding = self.padding(nid);
        let shape = self.shape(nid);
        let kernel = &mut self.kernels[kid];

        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Pad { padding, shape }) });

        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_reduce_op(&mut self, nid: TensorId, x: TensorId, rop: BOp) -> Result<(), ZyxError> {
        let axes = self.axes(nid);
        let shape = self.shape(x);

        let (mut kid, mut op_id) = self.visited[&x];
        if self.kernels[kid].contains_stores() {
            self.add_store(x)?;
            (kid, op_id) = self.create_load_kernel(x);
            if self.kernels[kid].outputs.len() > 1 {
                kid = self.duplicate_kernel(x, kid);
            }
        }
        if self.kernels[kid].outputs.len() > 1 {
            let reduce_dims_big = self.kernels[kid].is_preceded_by_reduce(op_id);
            if reduce_dims_big {
                self.add_store(x)?;
                (kid, op_id) = self.create_load_kernel(x);
                if self.kernels[kid].outputs.len() > 1 {
                    kid = self.duplicate_kernel(x, kid);
                }
            } else {
                kid = self.duplicate_kernel(x, kid);
            }
        }

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
        permute_axes.extend_from_slice(&axes);

        if !permute_axes.iter().copied().eq(0..permute_axes.len()) {
            let permuted_shape = crate::shape::permute(&self.shape(x), &permute_axes);
            let kernel = &mut self.kernels[kid];
            op_id = kernel
                .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes: permute_axes, shape: permuted_shape }) });
        }

        let kernel = &mut self.kernels[kid];
        op_id = kernel.push_back(Op::Reduce { x: op_id, rop, n_axes: axes.len() });
        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;

        if shape.len() == axes.len() {
            op_id = self.kernels[kid].push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Reshape { shape: vec![1] }) });
        }

        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_cast_op(&mut self, nid: TensorId, x: TensorId, dtype: DType) {
        let (kid, op_id) = self.visited[&x];
        let kernel = &mut self.kernels[kid];
        let op_id = kernel.push_back(Op::Cast { x: op_id, dtype });
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

    fn add_binary_op(&mut self, nid: TensorId, mut x: TensorId, mut y: TensorId, bop: BOp) -> Result<(), ZyxError> {
        let (mut kid, mut op_id) = self.visited[&x];
        let (mut kidy, mut op_idy) = self.visited[&y];

        let kid_stores = !self.kernels[kid].stores.is_empty();
        let kidy_stores = !self.kernels[kidy].stores.is_empty();

        let new_op_id = if kid == kidy {
            let kernel = &mut self.kernels[kid];
            kernel.remove_first_output(x);
            kernel.remove_first_output(y);
            kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
            kernel.push_back(Op::Binary { x: op_id, y: op_idy, bop })
        } else {
            match (kid_stores, kidy_stores) {
                (true, true) => {
                    self.add_store(x)?;
                    (kid, op_id) = self.create_load_kernel(x);
                    if self.kernels[kid].outputs.len() > 1 {
                        kid = self.duplicate_kernel(x, kid);
                        self.kernels[kid].outputs.push(x);
                    }
                    self.add_store(y)?;
                    (kidy, op_idy) = self.create_load_kernel(y);
                    if self.kernels[kidy].outputs.len() > 1 {
                        kidy = self.duplicate_kernel(y, kidy);
                        self.kernels[kidy].outputs.push(y);
                    }
                }
                (true, false) => {
                    self.add_store(x)?;
                    (kid, op_id) = self.create_load_kernel(x);
                    if self.kernels[kid].outputs.len() > 1 {
                        kid = self.duplicate_kernel(x, kid);
                        self.kernels[kid].outputs.push(x);
                    }
                }
                (false, true) => {
                    self.add_store(y)?;
                    (kidy, op_idy) = self.create_load_kernel(y);
                    if self.kernels[kidy].outputs.len() > 1 {
                        kidy = self.duplicate_kernel(y, kidy);
                        self.kernels[kidy].outputs.push(y);
                    }
                }
                (false, false) => {}
            }

            let mut swapped_xy = false;
            if self.kernels[kidy].is_reduce() && !self.kernels[kid].is_reduce() {
                std::mem::swap(&mut kid, &mut kidy);
                std::mem::swap(&mut op_id, &mut op_idy);
                std::mem::swap(&mut x, &mut y);
                swapped_xy = true;
            }

            self.kernels[kidy].remove_first_output(y);
            let Kernel { outputs, loads, stores, ops, head, tail: _ } = unsafe { self.kernels.remove_and_return(kidy) };

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

            let op = if swapped_xy {
                Op::Binary { x: y_ops_map[&op_idy], y: op_id, bop }
            } else {
                Op::Binary { x: op_id, y: y_ops_map[&op_idy], bop }
            };
            self.kernels[kid].push_back(op)
        };

        *self.rcs.get_mut(&x).unwrap() -= 1;
        *self.rcs.get_mut(&y).unwrap() -= 1;
        self.visited.insert(nid, (kid, new_op_id));
        Ok(())
    }

    fn add_store(&mut self, x: TensorId) -> Result<(), ZyxError> {
        let (kid, op_id) = self.visited[&x];
        if self.virt_realized_nodes.contains(&x) {
            self.visited.remove(&x).unwrap();
            self.kernels[kid].outputs.retain(|&elem| elem != x);
        } else {
            self.visited.remove(&x).unwrap();
            self.virt_realized_nodes.insert(x);
            let dtype = self.dtype(x);
            self.kernels[kid].push_back(Op::StoreView { src: op_id, dtype });
            self.kernels[kid].stores.push(x);
            self.kernels[kid].outputs.retain(|&elem| elem != x);
        }
        Ok(())
    }

    fn run(mut self) {
        let mut kids: Vec<KMKernelId> = self.kernels.ids().collect();
        while let Some(kid) = kids
            .iter()
            .find(|&&kid| self.kernels[kid].loads.iter().all(|x| self.realized_nodes.contains(x)))
            .copied()
        {
            kids.retain(|x| *x != kid);
            let kernel = unsafe { self.kernels.remove_and_return(kid) };
            if !kernel.stores.is_empty() {
                // Kernel is ready
            }
        }
    }
}

impl Runtime {
    pub(crate) fn launch_or_store_graph_with_order(
        &mut self,
        rcs: Map<TensorId, u32>,
        realized_nodes: Set<TensorId>,
        order: &[TensorId],
        _to_eval: &Set<TensorId>,
    ) -> Result<(), ZyxError> {
        let mut compacted = CompactedGraph {
            nodes: Vec::with_capacity(order.len()),
            shapes: BTreeMap::new(),
            paddings: BTreeMap::new(),
            axes: BTreeMap::new(),
        };
        let mut id_map: Map<TensorId, TensorId> = Map::with_capacity_and_hasher(order.len(), BuildHasherDefault::new());

        for (i, &nid) in order.iter().enumerate() {
            let new_id = TensorId::from(i);
            let node = &self.graph[nid];
            let reindexed = match node {
                Node::Const { value } => Node::Const { value: *value },
                Node::Leaf { dtype } => Node::Leaf { dtype: *dtype },
                Node::Expand { x } => Node::Expand { x: id_map[x] },
                Node::Permute { x } => Node::Permute { x: id_map[x] },
                Node::Reshape { x } => Node::Reshape { x: id_map[x] },
                Node::Pad { x } => Node::Pad { x: id_map[x] },
                Node::Reduce { x, rop } => Node::Reduce { x: id_map[x], rop: *rop },
                Node::Cast { x, dtype } => Node::Cast { x: id_map[x], dtype: *dtype },
                Node::Unary { x, uop } => Node::Unary { x: id_map[x], uop: *uop },
                Node::Binary { x, y, bop } => Node::Binary { x: id_map[x], y: id_map[y], bop: *bop },
            };
            compacted.nodes.push(reindexed);
            id_map.insert(nid, new_id);

            if matches!(
                node,
                Node::Leaf { .. }
                    | Node::Expand { .. }
                    | Node::Permute { .. }
                    | Node::Reshape { .. }
                    | Node::Pad { .. }
                    | Node::Reduce { .. }
            ) {
                compacted.shapes.insert(new_id, self.graph.shape(nid).into());
            }
            if matches!(node, Node::Pad { .. }) {
                compacted.paddings.insert(new_id, self.graph.padding(nid).into());
            }
            if matches!(node, Node::Permute { .. } | Node::Reduce { .. }) {
                compacted.axes.insert(new_id, self.graph.axes(nid).into());
            }
        }

        if let Some(_cached_graph) = self.graph_cache.get(&compacted) {
            return Ok(());
        }

        let compacted_realized: Set<TensorId> = realized_nodes.iter().filter_map(|&tid| id_map.get(&tid).copied()).collect();

        let kernelizer = GraphKernelizer::new(compacted.clone(), rcs, compacted_realized);
        kernelizer.run();

        Ok(())
    }
}
