// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: GPL-2.0-only

//! Compiled graph caching layer.

use crate::{
    DType, Map, Set, ZyxError,
    backend::{BufferId, DeviceId, ProgramId},
    graph::Node,
    kernel::{Kernel, MoveOp, Op, OpId, OpNode},
    kernelize::KMKernelId,
    runtime::Runtime,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
    view::View,
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
    pub paddings: BTreeMap<TensorId, Box<[(i64, i64)]>>,
    pub axes: BTreeMap<TensorId, Box<[UAxis]>>,
}

impl Runtime {
    pub(crate) fn launch_or_store_graph_with_order(
        &mut self,
        rcs: Map<TensorId, u32>,
        realized_nodes: Set<TensorId>,
        order: &[TensorId],
        to_eval: &Set<TensorId>,
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

        let mut virt_realized_nodes = compacted_realized.clone();
        let mut kernels: Slab<KMKernelId, Kernel> = Slab::with_capacity(30);
        let mut visited: Map<TensorId, (KMKernelId, OpId)> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
        let mut rcs = rcs;

        for (idx, node) in compacted.nodes.iter().enumerate() {
            let new_id = TensorId::from(idx);
            let nid = order[idx];

            if virt_realized_nodes.contains(&new_id) {
                let shape = compacted.shapes[&new_id].to_vec();
                let dtype = Self::compacted_dtype(&compacted, new_id);
                let mut ops = Slab::with_capacity(100);
                let op = Op::LoadView(Box::new((dtype, View::contiguous(&shape))));
                let op_id = ops.push(OpNode { prev: OpId::NULL, next: OpId::NULL, op });
                let kernel = Kernel {
                    outputs: vec![new_id; rcs[&new_id] as usize],
                    loads: vec![new_id],
                    stores: Vec::new(),
                    ops,
                    head: op_id,
                    tail: op_id,
                };
                let kid = kernels.push(kernel);
                visited.insert(new_id, (kid, op_id));
            } else {
                match node {
                    Node::Leaf { .. } => unreachable!(),
                    Node::Const { value } => {
                        let mut ops = Slab::with_capacity(100);
                        let op = Op::ConstView(Box::new((*value, View::contiguous(&[1]))));
                        let op_id = ops.push(OpNode { prev: OpId::NULL, next: OpId::NULL, op });
                        let kernel = Kernel {
                            outputs: vec![new_id; rcs[&new_id] as usize],
                            loads: Vec::new(),
                            stores: Vec::new(),
                            ops,
                            head: op_id,
                            tail: op_id,
                        };
                        let kid = kernels.push(kernel);
                        visited.insert(new_id, (kid, op_id));
                    }
                    Node::Cast { x, dtype } => {
                        let (kid, op_id) = visited[x];
                        let kernel = &mut kernels[kid];
                        let op_id = kernel.push_back(Op::Cast { x: op_id, dtype: *dtype });
                        kernel.remove_first_output(*x);
                        kernel.outputs.extend(vec![new_id; rcs[&new_id] as usize]);
                        *rcs.get_mut(x).unwrap() -= 1;
                        visited.insert(new_id, (kid, op_id));
                    }
                    Node::Unary { x, uop } => {
                        let (kid, op_id) = visited[x];
                        let kernel = &mut kernels[kid];
                        let op_id = kernel.push_back(Op::Unary { x: op_id, uop: *uop });
                        kernel.remove_first_output(*x);
                        kernel.outputs.extend(vec![new_id; rcs[&new_id] as usize]);
                        *rcs.get_mut(x).unwrap() -= 1;
                        visited.insert(new_id, (kid, op_id));
                    }
                    Node::Expand { x } => {
                        let (mut kid, mut op_id) = visited[x];

                        if kernels[kid].contains_stores() | kernels[kid].is_reduce() {
                            Self::add_store(&compacted, &mut kernels, &mut visited, &mut virt_realized_nodes, *x)?;
                            (kid, op_id) = Self::create_load_kernel(&compacted, &mut kernels, &mut visited, &rcs, *x);
                            if kernels[kid].outputs.len() > 1 {
                                kid = Self::duplicate_kernel(&mut kernels, &visited, *x, kid);
                            }
                        }

                        if kernels[kid].outputs.len() > 1 {
                            let reduce_dims_big = kernels[kid].is_preceded_by_reduce(op_id);
                            if reduce_dims_big {
                                Self::add_store(&compacted, &mut kernels, &mut visited, &mut virt_realized_nodes, *x)?;
                                (kid, op_id) = Self::create_load_kernel(&compacted, &mut kernels, &mut visited, &rcs, *x);
                                if kernels[kid].outputs.len() > 1 {
                                    kid = Self::duplicate_kernel(&mut kernels, &visited, *x, kid);
                                }
                            } else {
                                kid = Self::duplicate_kernel(&mut kernels, &visited, *x, kid);
                            }
                        }

                        let shape: Vec<Dim> = compacted.shapes[&new_id].to_vec();
                        let kernel = &mut kernels[kid];
                        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Expand { shape }) });

                        kernel.remove_first_output(*x);
                        kernel.outputs.extend(vec![new_id; rcs[&new_id] as usize]);
                        *rcs.get_mut(x).unwrap() -= 1;
                        visited.insert(new_id, (kid, op_id));
                    }
                    Node::Permute { x } => {
                        let (kid, op_id) =
                            Self::duplicate_or_store(&compacted, &mut kernels, &mut visited, &mut virt_realized_nodes, &rcs, *x)?;
                        let axes: Vec<UAxis> = compacted.axes[&new_id].to_vec();
                        let shape: Vec<Dim> = compacted.shapes[&new_id].to_vec();
                        let kernel = &mut kernels[kid];

                        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes, shape }) });

                        kernel.remove_first_output(*x);
                        kernel.outputs.extend(vec![new_id; rcs[&new_id] as usize]);
                        *rcs.get_mut(x).unwrap() -= 1;
                        visited.insert(new_id, (kid, op_id));
                    }
                    Node::Reshape { x } => {
                        let (kid, op_id) =
                            Self::duplicate_or_store(&compacted, &mut kernels, &mut visited, &mut virt_realized_nodes, &rcs, *x)?;
                        let shape: Vec<Dim> = compacted.shapes[&new_id].to_vec();
                        let kernel = &mut kernels[kid];

                        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Reshape { shape }) });

                        kernel.remove_first_output(*x);
                        kernel.outputs.extend(vec![new_id; rcs[&new_id] as usize]);
                        *rcs.get_mut(x).unwrap() -= 1;
                        visited.insert(new_id, (kid, op_id));
                    }
                    Node::Pad { x } => {
                        let (kid, op_id) =
                            Self::duplicate_or_store(&compacted, &mut kernels, &mut visited, &mut virt_realized_nodes, &rcs, *x)?;
                        let padding: Vec<(i32, i32)> = compacted.paddings[&new_id].to_vec();
                        let shape: Vec<Dim> = compacted.shapes[&new_id].to_vec();
                        let kernel = &mut kernels[kid];

                        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Pad { padding, shape }) });

                        kernel.remove_first_output(*x);
                        kernel.outputs.extend(vec![new_id; rcs[&new_id] as usize]);
                        *rcs.get_mut(x).unwrap() -= 1;
                        visited.insert(new_id, (kid, op_id));
                    }
                    Node::Reduce { x, rop } => {
                        let axes: Vec<UAxis> = compacted.axes[&new_id].to_vec();
                        let shape: Vec<Dim> = compacted.shapes[x].to_vec();

                        let (mut kid, mut op_id) = visited[x];
                        if kernels[kid].contains_stores() {
                            Self::add_store(&compacted, &mut kernels, &mut visited, &mut virt_realized_nodes, *x)?;
                            (kid, op_id) = Self::create_load_kernel(&compacted, &mut kernels, &mut visited, &rcs, *x);
                            if kernels[kid].outputs.len() > 1 {
                                kid = Self::duplicate_kernel(&mut kernels, &visited, *x, kid);
                            }
                        }
                        if kernels[kid].outputs.len() > 1 {
                            let reduce_dims_big = kernels[kid].is_preceded_by_reduce(op_id);
                            if reduce_dims_big {
                                Self::add_store(&compacted, &mut kernels, &mut visited, &mut virt_realized_nodes, *x)?;
                                (kid, op_id) = Self::create_load_kernel(&compacted, &mut kernels, &mut visited, &rcs, *x);
                                if kernels[kid].outputs.len() > 1 {
                                    kid = Self::duplicate_kernel(&mut kernels, &visited, *x, kid);
                                }
                            } else {
                                kid = Self::duplicate_kernel(&mut kernels, &visited, *x, kid);
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
                            permute_axes.extend_from_slice(&axes);

                            if !permute_axes.iter().copied().eq(0..permute_axes.len()) {
                                let permuted_shape = crate::shape::permute(&compacted.shapes[x], &permute_axes);
                                op_id = kernels[kid].push_back(Op::Move {
                                    x: op_id,
                                    mop: Box::new(MoveOp::Permute { axes: permute_axes, shape: permuted_shape }),
                                });
                            }
                        }

                        let kernel = &mut kernels[kid];
                        op_id = kernel.push_back(Op::Reduce { x: op_id, rop: *rop, n_axes: axes.len() });
                        kernel.remove_first_output(*x);
                        kernel.outputs.extend(vec![new_id; rcs[&new_id] as usize]);
                        *rcs.get_mut(x).unwrap() -= 1;

                        if shape.len() == axes.len() {
                            op_id =
                                kernels[kid].push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Reshape { shape: vec![1] }) });
                        }

                        visited.insert(new_id, (kid, op_id));
                    }
                    Node::Binary { x, y, bop } => {
                        let (mut kid, mut op_id) = visited[x];
                        let (mut kidy, mut op_idy) = visited[y];

                        let kid_stores = !kernels[kid].stores.is_empty();
                        let kidy_stores = !kernels[kidy].stores.is_empty();

                        let new_op_id = if kid == kidy {
                            let kernel = &mut kernels[kid];
                            kernel.remove_first_output(*x);
                            kernel.remove_first_output(*y);
                            kernel.outputs.extend(vec![new_id; rcs[&new_id] as usize]);
                            kernel.push_back(Op::Binary { x: op_id, y: op_idy, bop: *bop })
                        } else {
                            match (kid_stores, kidy_stores) {
                                (true, true) => {
                                    Self::add_store(&compacted, &mut kernels, &mut visited, &mut virt_realized_nodes, *x)?;
                                    (kid, op_id) = Self::create_load_kernel(&compacted, &mut kernels, &mut visited, &rcs, *x);
                                    if kernels[kid].outputs.len() > 1 {
                                        kid = Self::duplicate_kernel(&mut kernels, &visited, *x, kid);
                                        kernels[kid].outputs.push(*x);
                                    }
                                    Self::add_store(&compacted, &mut kernels, &mut visited, &mut virt_realized_nodes, *y)?;
                                    (kidy, op_idy) = Self::create_load_kernel(&compacted, &mut kernels, &mut visited, &rcs, *y);
                                    if kernels[kidy].outputs.len() > 1 {
                                        kidy = Self::duplicate_kernel(&mut kernels, &visited, *y, kidy);
                                        kernels[kidy].outputs.push(*y);
                                    }
                                }
                                (true, false) => {
                                    Self::add_store(&compacted, &mut kernels, &mut visited, &mut virt_realized_nodes, *x)?;
                                    (kid, op_id) = Self::create_load_kernel(&compacted, &mut kernels, &mut visited, &rcs, *x);
                                    if kernels[kid].outputs.len() > 1 {
                                        kid = Self::duplicate_kernel(&mut kernels, &visited, *x, kid);
                                        kernels[kid].outputs.push(*x);
                                    }
                                }
                                (false, true) => {
                                    Self::add_store(&compacted, &mut kernels, &mut visited, &mut virt_realized_nodes, *y)?;
                                    (kidy, op_idy) = Self::create_load_kernel(&compacted, &mut kernels, &mut visited, &rcs, *y);
                                    if kernels[kidy].outputs.len() > 1 {
                                        kidy = Self::duplicate_kernel(&mut kernels, &visited, *y, kidy);
                                        kernels[kidy].outputs.push(*y);
                                    }
                                }
                                (false, false) => {}
                            }

                            let mut swapped_xy = false;
                            let mut xx = *x;
                            let mut yy = *y;
                            if kernels[kidy].is_reduce() && !kernels[kid].is_reduce() {
                                std::mem::swap(&mut kid, &mut kidy);
                                std::mem::swap(&mut op_id, &mut op_idy);
                                std::mem::swap(&mut xx, &mut yy);
                                swapped_xy = true;
                            }

                            kernels[kidy].remove_first_output(yy);
                            let Kernel { outputs, loads, stores, ops, head, tail: _ } =
                                unsafe { kernels.remove_and_return(kidy) };

                            let mut y_ops_map = Map::with_capacity_and_hasher(5, BuildHasherDefault::new());

                            let mut i = head;
                            while !i.is_null() {
                                let mut op = ops[i].op.clone();
                                for param in op.parameters_mut() {
                                    *param = y_ops_map[param];
                                }
                                let new_op_id = kernels[kid].push_back(op);
                                y_ops_map.insert(i, new_op_id);
                                i = ops[i].next;
                            }

                            for (kidm, op_id) in visited.values_mut() {
                                if *kidm == kidy {
                                    *kidm = kid;
                                    if let Some(new_op_id) = y_ops_map.get(op_id) {
                                        *op_id = *new_op_id;
                                    }
                                }
                            }

                            kernels[kid].loads.extend(loads);
                            kernels[kid].stores.extend(stores);

                            kernels[kid].remove_first_output(xx);
                            kernels[kid].outputs.extend(outputs);
                            kernels[kid].outputs.extend(vec![new_id; rcs[&new_id] as usize]);

                            let op = if swapped_xy {
                                Op::Binary { x: y_ops_map[&op_idy], y: op_id, bop: *bop }
                            } else {
                                Op::Binary { x: op_id, y: y_ops_map[&op_idy], bop: *bop }
                            };
                            kernels[kid].push_back(op)
                        };

                        *rcs.get_mut(x).unwrap() -= 1;
                        *rcs.get_mut(y).unwrap() -= 1;
                        visited.insert(new_id, (kid, new_op_id));
                    }
                }
            }

            if to_eval.contains(&nid) && !compacted_realized.contains(&new_id) {
                Self::add_store(&compacted, &mut kernels, &mut visited, &mut virt_realized_nodes, new_id)?;
                *rcs.get_mut(&new_id).unwrap() -= 1;
                if rcs[&new_id] > 0 {
                    Self::create_load_kernel(&compacted, &mut kernels, &mut visited, &rcs, new_id);
                }
            }
        }

        if kernels.len() > KMKernelId::ZERO {
            for (kid, kernel) in kernels.iter() {
                println!("Kernel {kid:?}:");
                kernel.debug_colorless();
            }
        }

        Ok(())
    }

    fn compacted_dtype(compacted: &CompactedGraph, tensor_id: TensorId) -> DType {
        let mut tid = tensor_id;
        for _ in 0..100_000 {
            match compacted.nodes[usize::from(tid)] {
                Node::Const { value } => return value.dtype(),
                Node::Leaf { dtype } | Node::Cast { dtype, .. } => return dtype,
                Node::Binary { bop, .. } if bop.returns_bool() => return DType::Bool,
                _ => tid = compacted.nodes[usize::from(tid)].param1(),
            }
        }
        panic!("DType not found")
    }

    fn create_load_kernel(
        compacted: &CompactedGraph,
        kernels: &mut Slab<KMKernelId, Kernel>,
        visited: &mut Map<TensorId, (KMKernelId, OpId)>,
        rcs: &Map<TensorId, u32>,
        nid: TensorId,
    ) -> (KMKernelId, OpId) {
        let shape = compacted.shapes[&nid].clone();
        let dtype = Self::compacted_dtype(compacted, nid);
        let mut ops = Slab::with_capacity(100);
        let op = Op::LoadView(Box::new((dtype, View::contiguous(&shape))));
        let op_id = ops.push(OpNode { prev: OpId::NULL, next: OpId::NULL, op });
        let kernel = Kernel {
            outputs: vec![nid; rcs[&nid] as usize],
            loads: vec![nid],
            stores: Vec::new(),
            ops,
            head: op_id,
            tail: op_id,
        };
        let kid = kernels.push(kernel);
        visited.insert(nid, (kid, op_id));
        (kid, op_id)
    }

    fn add_store(
        compacted: &CompactedGraph,
        kernels: &mut Slab<KMKernelId, Kernel>,
        visited: &mut Map<TensorId, (KMKernelId, OpId)>,
        virt_realized_nodes: &mut Set<TensorId>,
        x: TensorId,
    ) -> Result<(), ZyxError> {
        let (kid, op_id) = visited[&x];
        if virt_realized_nodes.contains(&x) {
            visited.remove(&x).unwrap();
            kernels[kid].outputs.retain(|&elem| elem != x);
        } else {
            visited.remove(&x).unwrap();
            virt_realized_nodes.insert(x);
            let dtype = Self::compacted_dtype(compacted, x);
            kernels[kid].push_back(Op::StoreView { src: op_id, dtype });
            kernels[kid].stores.push(x);
            kernels[kid].outputs.retain(|&elem| elem != x);
        }
        Ok(())
    }

    fn duplicate_kernel(
        kernels: &mut Slab<KMKernelId, Kernel>,
        visited: &Map<TensorId, (KMKernelId, OpId)>,
        x: TensorId,
        kid: KMKernelId,
    ) -> KMKernelId {
        let mut kernel = kernels[kid].clone();
        kernel.outputs = vec![x];
        kernel.drop_unused_ops(visited);
        kernels[kid].remove_first_output(x);
        kernels[kid].drop_unused_ops(visited);
        kernels.push(kernel)
    }

    fn duplicate_or_store(
        compacted: &CompactedGraph,
        kernels: &mut Slab<KMKernelId, Kernel>,
        visited: &mut Map<TensorId, (KMKernelId, OpId)>,
        virt_realized_nodes: &mut Set<TensorId>,
        rcs: &Map<TensorId, u32>,
        x: TensorId,
    ) -> Result<(KMKernelId, OpId), ZyxError> {
        let (mut kid, mut op_id) = visited[&x];

        if kernels[kid].contains_stores() {
            Self::add_store(compacted, kernels, visited, virt_realized_nodes, x)?;
            (kid, op_id) = Self::create_load_kernel(compacted, kernels, visited, rcs, x);
            if kernels[kid].outputs.len() > 1 {
                kid = Self::duplicate_kernel(kernels, visited, x, kid);
            }
        }

        if kernels[kid].outputs.len() > 1 {
            let reduce_dims_big = kernels[kid].is_preceded_by_reduce(op_id);
            if reduce_dims_big {
                Self::add_store(compacted, kernels, visited, virt_realized_nodes, x)?;
                (kid, op_id) = Self::create_load_kernel(compacted, kernels, visited, rcs, x);
                if kernels[kid].outputs.len() > 1 {
                    kid = Self::duplicate_kernel(kernels, visited, x, kid);
                }
            } else {
                kid = Self::duplicate_kernel(kernels, visited, x, kid);
            }
        }

        Ok((kid, op_id))
    }
}
