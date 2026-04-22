// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Converts graph to kernels and schedules them to devices

use crate::{
    DType, DebugMask, Map, Set, ZyxError,
    backend::{AutotuneConfig, BufferId, Device, DeviceId, PoolId},
    cache::Cache,
    dtype::Constant,
    graph::{Graph, Node},
    kernel::{BOp, Kernel, MoveOp, Op, OpId, OpNode, Scope, UOp},
    runtime::{Pool, Runtime, deallocate_tensors},
    schedule::schedule,
    slab::{Slab, SlabId},
    tensor::TensorId,
    view::View,
};
use std::{collections::BTreeMap, hash::BuildHasherDefault};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KMKernelId(u32);

impl SlabId for KMKernelId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl From<usize> for KMKernelId {
    fn from(value: usize) -> Self {
        KMKernelId(value as u32)
    }
}

impl From<KMKernelId> for usize {
    fn from(value: KMKernelId) -> Self {
        value.0 as usize
    }
}

struct Kernelizer<'a> {
    // TODO merge as many of these as possible. Perhaps start by mergins rcs and visited
    // Those nodes that have been store ops in some kernel, but those kernels may have not yet run (must be checked in realized_nodex).
    must_keep_nodes: Set<TensorId>,     // Nodes that were realized before kernelizer was created
    virt_realized_nodes: Set<TensorId>, // Nodes that appear in kernel stores, but are not realized yet
    realized_nodes: Set<TensorId>,      // Nodes that are realized
    // TODO later delete this and just directly use the runtime kernel cache
    kernels: Slab<KMKernelId, Kernel>,
    // We should remove either visited, or rcs
    visited: Map<TensorId, (KMKernelId, OpId)>,
    rcs: Map<TensorId, u32>,
    graph: &'a Graph,
    pools: &'a mut Slab<PoolId, Pool>,
    temp_data: &'a mut Map<BufferId, Box<[u8]>>,
    buffer_map: &'a mut Map<TensorId, BufferId>,
    devices: &'a mut Slab<DeviceId, Device>,
    cache: &'a mut Cache,
    autotune_config: &'a AutotuneConfig,
    debug: DebugMask,
    n_launches: u32,
}

impl<'a> Kernelizer<'a> {
    fn new(
        realized_nodes: Set<TensorId>,
        to_eval: &'a Set<TensorId>,
        rcs: Map<TensorId, u32>,
        graph: &'a Graph,
        pools: &'a mut Slab<PoolId, Pool>,
        temp_data: &'a mut Map<BufferId, Box<[u8]>>,
        buffer_map: &'a mut Map<TensorId, BufferId>,
        devices: &'a mut Slab<DeviceId, Device>,
        cache: &'a mut Cache,
        search_config: &'a AutotuneConfig,
        debug: DebugMask,
    ) -> Self {
        let mut must_keep_nodes = realized_nodes.clone();
        must_keep_nodes.extend(to_eval);
        Self {
            // Those nodes that have been store ops in some kernel, but those kernels may have not yet run (must be checked in realized_nodex).
            must_keep_nodes,
            virt_realized_nodes: realized_nodes.clone(),
            realized_nodes,
            kernels: Slab::with_capacity(30),
            visited: Map::with_capacity_and_hasher(100, BuildHasherDefault::new()),
            rcs,
            graph,
            pools,
            temp_data,
            buffer_map,
            devices,
            cache,
            autotune_config: search_config,
            debug,
            n_launches: 0,
        }
    }

    #[allow(unused)]
    fn debug(&self) {
        for kernel in self.kernels.values() {
            kernel.debug();
        }
        println!();
    }

    fn is_virt_realized(&self, nid: TensorId) -> bool {
        self.virt_realized_nodes.contains(&nid)
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

        // If values inside reduction need to be used elsewhere, we have to duplicate
        if self.kernels[kid].outputs.len() > 1 {
            //println!("Duplicating kernel");
            //self.kernels[kid].debug();
            //let split_reduce_dim = 32; // Can be tuned later, or hardware based, likely needs to be MUCH higher for streaming softmax
            //let reduce_dims_big = self.kernels[kid].cumulative_reduce_dim(op_id, 1) > split_reduce_dim;
            let reduce_dims_big = self.kernels[kid].is_preceded_by_reduce(op_id);
            if reduce_dims_big {
                //println!("Adding store for reduce with outputs");
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
        //println!("op_id={op_id}");
        //println!("Duplicating");
        //self.kernels[kid].debug();
        // Instead of copy of the whole kernel, copy only relevant ops
        // and remove these ops from the original if not needed.
        let mut kernel = self.kernels[kid].clone();
        kernel.outputs = vec![x];
        kernel.drop_unused_ops(&self.visited);
        self.kernels[kid].remove_first_output(x);
        self.kernels[kid].drop_unused_ops(&self.visited);
        self.kernels.push(kernel)
    }

    fn create_load_kernel(&mut self, nid: TensorId) -> (KMKernelId, OpId) {
        let shape = self.graph.shape(nid);
        let dtype = self.graph.dtype(nid);
        let mut ops = Slab::with_capacity(100);
        let op = Op::LoadView(Box::new((dtype, View::contiguous(shape))));
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
        // TODO instead of store add expand op that inserts loop in IR
        let (mut kid, mut op_id) = self.visited[&x];

        if self.kernels[kid].contains_stores() | self.kernels[kid].is_preceded_by_reduce(op_id) {
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

        let shape = self.graph.shape(nid);
        let kernel = &mut self.kernels[kid];

        //kernel.apply_movement(|view| view.expand(shape));
        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Expand { shape: shape.into() }) });

        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        debug_assert_eq!(self.graph.shape(nid), kernel.shape());
        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_reshape_op(&mut self, nid: TensorId, x: TensorId) -> Result<(), ZyxError> {
        debug_assert!(self.visited.contains_key(&x), "Missing tensor {x} in visited.");
        let (kid, op_id) = self.duplicate_or_store(x)?;
        let shape = self.graph.shape(nid);
        let kernel = &mut self.kernels[kid];

        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Reshape { shape: shape.into() }) });

        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        debug_assert_eq!(self.graph.shape(nid), kernel.shape());
        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_permute_op(&mut self, nid: TensorId, x: TensorId) -> Result<(), ZyxError> {
        // TODO instead of store add permute op that swaps indices in IR in unfold_movement_ops
        //let (kid, op_id) = self.visited[&x];
        let (kid, op_id) = self.duplicate_or_store(x)?;
        let axes: Vec<_> = self.graph.axes(nid).into();
        let kernel = &mut self.kernels[kid];

        let shape = self.graph.shape(nid).into();
        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes, shape }) });

        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        debug_assert_eq!(self.graph.shape(nid), kernel.shape());
        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_pad_op(&mut self, nid: TensorId, x: TensorId) -> Result<(), ZyxError> {
        // TODO instead of duplication add pad op that adds if statement into ir (e.g. if idx < padding) in unfold_movement_ops
        //let (kid, op_id) = self.visited[&x];
        let (kid, op_id) = self.duplicate_or_store(x)?;
        let padding = self.graph.padding(nid).into();
        let kernel = &mut self.kernels[kid];

        //let rank = self.graph.shape(nid).len();
        //kernel.apply_movement(|view| view.pad(rank, padding));
        let shape = self.graph.shape(nid).into();
        let op_id = kernel.push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Pad { padding, shape }) });

        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;
        debug_assert_eq!(self.graph.shape(nid), kernel.shape());
        self.visited.insert(nid, (kid, op_id));
        Ok(())
    }

    fn add_reduce_op(&mut self, nid: TensorId, x: TensorId, rop: BOp) -> Result<(), ZyxError> {
        let axes = self.graph.axes(nid);
        let shape = self.graph.shape(x);

        // If the kernel has more than one output, or rc of x is more than one,
        // we have to either copy it (if it is small), or store x (if kid is big)

        let (mut kid, mut op_id) = self.visited[&x];
        if self.kernels[kid].contains_stores() {
            self.add_store(x)?;
            (kid, op_id) = self.create_load_kernel(x);
            if self.kernels[kid].outputs.len() > 1 {
                kid = self.duplicate_kernel(x, kid);
            }
        }
        //let reduce_dims_product: usize = axes.iter().map(|&a| shape[a]).product();
        if self.kernels[kid].outputs.len() > 1 {
            // TODO
            // small reduces can be duplicated in the future
            //let split_reduce_dim = 32000;
            //println!("prev_reduce_dims={prev_reduce_dims}");
            //let is_duplicated_big_reduce = prev_reduce_dims > 32;
            //let is_big_reduce = reduce_dims_product * prev_reduce_dims > split_reduce_dim;
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

        /*for kernel in kernels.iter() {
            println!("{:?}, {}", kernel.0, kernel.1.n_outputs);
        }*/

        #[cfg(debug_assertions)]
        {
            use crate::shape::UAxis;
            let mut sorted_axes: Vec<UAxis> = axes.into();
            sorted_axes.sort_unstable();
            debug_assert_eq!(axes, sorted_axes, "Reduce axes must be sorted.");
        }

        //kernels[kid].debug();
        //println!("{axes:?}");

        {
            // Permute before reduce so that reduce axes are last
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

            //self.kernels[kid].apply_movement(|v| v.permute(&permute_axes));
            if !permute_axes.iter().copied().eq(0..permute_axes.len()) {
                let shape = crate::shape::permute(self.graph.shape(x), &permute_axes);
                op_id = self.kernels[kid]
                    .push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Permute { axes: permute_axes, shape }) });
            }
        }

        let kernel = &mut self.kernels[kid];
        op_id = kernel.push_back(Op::Reduce { x: op_id, rop, n_axes: axes.len() });
        kernel.remove_first_output(x);
        kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
        *self.rcs.get_mut(&x).unwrap() -= 1;

        // If all dims are reduced
        if shape.len() == axes.len() {
            //self.kernels[kid].apply_movement(|v| v.reshape(0..1, &[1, shape[0]]));
            op_id = self.kernels[kid].push_back(Op::Move { x: op_id, mop: Box::new(MoveOp::Reshape { shape: vec![1] }) });
        }

        debug_assert_eq!(self.graph.shape(nid), self.kernels[kid].shape());
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

        //self.kernels[kid].debug();
        //self.kernels[kidy].debug();

        let kid_stores = !self.kernels[kid].stores.is_empty();
        let kidy_stores = !self.kernels[kidy].stores.is_empty();

        let new_op_id = if kid == kidy {
            //println!("Same kernels for binary");
            let kernel = &mut self.kernels[kid];
            kernel.remove_first_output(x);
            kernel.remove_first_output(y);
            kernel.outputs.extend(vec![nid; self.rcs[&nid] as usize]);
            kernel.push_back(Op::Binary { x: op_id, y: op_idy, bop })
        } else {
            //println!("Different kernels for binary");
            // TODO later use this, but this requires global memory sync inside of the kernel
            // as it loads and stores from the same kernel
            //if kid_stores && kidy_stores {
            match (kid_stores, kidy_stores) {
                (true, true) => {
                    //println!("Adding store for binary");
                    self.add_store(x)?;
                    (kid, op_id) = self.create_load_kernel(x);
                    if self.kernels[kid].outputs.len() > 1 {
                        kid = self.duplicate_kernel(x, kid);
                        self.kernels[kid].outputs.push(x);
                    }
                    self.add_store(y)?;
                    (kidy, op_idy) = self.create_load_kernel(y);
                    //println!("kidy={:?}", kidy);
                    if self.kernels[kidy].outputs.len() > 1 {
                        kidy = self.duplicate_kernel(y, kidy);
                        self.kernels[kidy].outputs.push(y);
                    }
                    //println!("kidy={:?}", kidy);
                }
                (true, false) => {
                    //println!("Adding store for binary 1");
                    self.add_store(x)?;
                    (kid, op_id) = self.create_load_kernel(x);
                    if self.kernels[kid].outputs.len() > 1 {
                        kid = self.duplicate_kernel(x, kid);
                        self.kernels[kid].outputs.push(x);
                    }
                }
                (false, true) => {
                    //println!("Adding store for binary 2");
                    self.add_store(y)?;
                    (kidy, op_idy) = self.create_load_kernel(y);
                    //println!("kidy={:?}", kidy);
                    if self.kernels[kidy].outputs.len() > 1 {
                        kidy = self.duplicate_kernel(y, kidy);
                        self.kernels[kidy].outputs.push(y);
                    }
                    //println!("kidy={:?}", kidy);
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
            let Kernel { outputs, loads, stores, ops, head, tail: _ } = unsafe { self.kernels.remove_and_return(kidy) };

            // Extend x kernel with y ops
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

            // Fix visited
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
        //println!("Binary output");
        //self.kernels[kid].debug();
        Ok(())
    }

    /// Stores x and returns remaining reference count to x
    fn add_store(&mut self, x: TensorId) -> Result<(), ZyxError> {
        //println!("Adding store.");
        let (kid, op_id) = self.visited[&x];
        //self.kernels[kid].debug();
        if self.virt_realized_nodes.contains(&x) {
            self.visited.remove(&x).unwrap();
            self.kernels[kid].outputs.retain(|&elem| elem != x);
        } else {
            self.visited.remove(&x).unwrap();
            self.virt_realized_nodes.insert(x);
            let dtype = self.graph.dtype(x);
            self.kernels[kid].push_back(Op::StoreView { src: op_id, dtype });
            self.kernels[kid].stores.push(x);

            // remove all references to x
            self.kernels[kid].outputs.retain(|&elem| elem != x);
        }

        if self.kernels[kid].outputs.is_empty() && self.kernels[kid].loads.iter().all(|x| self.realized_nodes.contains(x)) {
            let kernel = unsafe { self.kernels.remove_and_return(kid) };
            let loads = kernel.loads.clone();
            let stores = kernel.stores.clone();
            self.launch_kernel(kernel)?;
            self.realized_nodes.extend(stores);
            // Delete unneeded intermediate tensors from memory pools
            let mut to_remove = Set::with_capacity_and_hasher(1, BuildHasherDefault::new());
            for tid in loads {
                if !self.kernels.values().any(|kernel| kernel.loads.contains(&tid)) && !self.must_keep_nodes.contains(&tid) {
                    to_remove.insert(tid);
                }
            }
            deallocate_tensors(&to_remove, self.pools, self.temp_data, self.buffer_map);
        }
        //println!("ADDED STORE for {x} x {xrc_rem}");
        Ok(())
    }

    fn launch_kernel(&mut self, mut kernel: Kernel) -> Result<(), ZyxError> {
        if kernel.stores.is_empty() {
            println!("Empty stores in this kernel:");
            kernel.debug();
            panic!("Empty stores in this kernel:");
        }
        debug_assert!(!kernel.stores.is_empty());
        debug_assert!(!kernel.ops.is_empty());

        self.n_launches += 1;

        //let time_w = std::time::Instant::now();
        let (dev_id, pool_id, event_wait_list, output_buffers, args) = schedule(
            &kernel.loads,
            &kernel.stores,
            self.graph,
            self.devices,
            self.pools,
            self.buffer_map,
        )?;

        /***** CACHE and OPTIMIZATION SEARCH *****/

        let device = &mut self.devices[dev_id];
        // ...
        let pool = &mut self.pools[pool_id];

        let dev_info_id = self.cache.get_or_add_dev_info(device.info());

        // Launch if it is in cache
        if let Some(&kid) = self.cache.kernels.get(&kernel) {
            // If it has been compiled for the device
            if let Some(&program_id) = self.cache.programs.get(&(kid, dev_id)) {
                if self.debug.kmd() {
                    println!("Kernel launch from memory pool {pool_id:?} with args: {args:?}");
                }
                let event = device.launch(program_id, &mut pool.pool, &args, event_wait_list)?;
                self.pools[pool_id].events.insert(output_buffers, event);
                //println!("Elapsed during kernel launch {:?}", time_w.elapsed());
                return Ok(());
            }

            // The kernel was optimized and is cached in disk, but was not compiled as of this run
            if let Some(opt_seq) = self.cache.optimizations.get(&(kid, dev_info_id)) {
                opt_seq.apply(&mut kernel);
                let program_id = device.compile(&kernel, self.debug.asm())?;
                let event = device.launch(program_id, &mut pool.pool, &args, event_wait_list)?;
                self.pools[pool_id].events.insert(output_buffers, event);
                return Ok(());
            }
        }

        // If it is not in cache, we just get new empty kernel id where we insert the kernel
        let kernel_id = self.cache.insert_kernel(kernel.clone());

        if self.debug.sched() {
            kernel.debug();
        }

        let (flop, read, write) = kernel.flop_mem_rw();

        // Fix kernels for movement ops and if they have too many dims
        kernel.unfold_movement_ops();
        let global_indices = kernel.get_global_indices();
        let max_global_dims = device.info().max_global_work_dims.len();
        if global_indices.len() > max_global_dims {
            let n = global_indices.len() + 1 - max_global_dims;
            let loops: Vec<OpId> = global_indices.values().copied().take(n).collect();
            kernel.merge_indices(&loops);
        }
        // Reset indices after merges
        {
            let mut indices = BTreeMap::new();
            indices.insert(Scope::Global, BTreeMap::new());
            indices.insert(Scope::Local, BTreeMap::new());
            for (op_id, op_node) in kernel.ops.iter() {
                if let Op::Index { scope, axis, .. } = op_node.op {
                    indices.get_mut(&scope).unwrap().insert(axis, op_id);
                }
            }
            for (_, scoped_indices) in indices {
                let mut ax = 0;
                for &idx_id in scoped_indices.values() {
                    let Op::Index { axis, .. } = &mut kernel.ops[idx_id].op else { unreachable!() };
                    *axis = ax;
                    ax += 1;
                }
            }

            kernel.verify();
        }
        //kernel.run_always_on_optimizations();
        //kernel.debug();

        let (program_id, opts) = kernel.autotune(
            &args,
            device,
            &mut pool.pool,
            self.autotune_config,
            flop,
            read,
            write,
            self.debug,
        )?;
        self.cache.programs.insert((kernel_id, dev_id), program_id);
        self.cache.optimizations.insert((kernel_id, dev_info_id), opts);
        let event = device.launch(program_id, &mut pool.pool, &args, event_wait_list)?;
        self.pools[pool_id].events.insert(output_buffers, event);

        Ok(())
    }
}

impl Runtime {
    pub(crate) fn realize_with_order(
        &mut self,
        rcs: Map<TensorId, u32>,
        realized_nodes: Set<TensorId>,
        order: &[TensorId],
        to_eval: &Set<TensorId>,
    ) -> Result<(), ZyxError> {
        #[cfg(debug_assertions)]
        {
            let mut rcs2 = Map::with_hasher(BuildHasherDefault::default());
            for &nid in order {
                if !realized_nodes.contains(&nid) {
                    for nid in self.graph[nid].parameters() {
                        rcs2.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                    }
                }
            }
            for &nid in to_eval {
                rcs2.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
            }
            assert_eq!(rcs2, rcs, "rcs are incorrect, rcs: {rcs:?}\nrcs2: {rcs2:?}");
        }

        //println!("{rcs:?}");
        //println!("realized_nodes: {realized_nodes:?}");
        //println!("to_eval: {to_eval:?}");

        let begin = std::time::Instant::now();

        let mut kernelizer = Kernelizer::new(
            realized_nodes,
            to_eval,
            rcs,
            &self.graph,
            &mut self.pools,
            &mut self.temp_data,
            &mut self.buffer_map,
            &mut self.devices,
            &mut self.cache,
            &self.autotune_config,
            self.debug,
        );

        for &nid in order {
            /*use crate::{RED, RESET};
            println!(
                "{RED}{}{nid} x {} -> {:?}  {}  {:?}{RESET}",
                if kernelizer.is_virt_realized(nid) { "LOAD " } else { "" },
                kernelizer.rcs[&nid],
                self.graph[nid],
                self.graph.dtype(nid),
                self.graph.shape(nid)
            );*/
            if kernelizer.is_virt_realized(nid) {
                kernelizer.create_load_kernel(nid);
            } else {
                match self.graph[nid] {
                    Node::Leaf { .. } => unreachable!(),
                    Node::Const { value } => kernelizer.create_const_kernel(nid, value),
                    Node::Cast { x, dtype } => kernelizer.add_cast_op(nid, x, dtype),
                    Node::Unary { x, uop } => kernelizer.add_unary_op(nid, x, uop),
                    Node::Expand { x } => kernelizer.add_expand_op(nid, x)?,
                    Node::Permute { x } => kernelizer.add_permute_op(nid, x)?,
                    Node::Reshape { x } => kernelizer.add_reshape_op(nid, x)?,
                    Node::Pad { x } => kernelizer.add_pad_op(nid, x)?,
                    Node::Reduce { x, rop } => kernelizer.add_reduce_op(nid, x, rop)?,
                    Node::Binary { x, y, bop } => kernelizer.add_binary_op(nid, x, y, bop)?,
                    Node::Custom(_) => todo!(),
                }
            }

            // verify kernelizer.visited
            /*#[cfg(debug_assertions)]
            {
                let mut kernel_op_ids: Map<KMKernelId, Set<OpId>> = Map::default();
                for (kid, kernel) in kernelizer.kernels.iter() {
                    kernel_op_ids.insert(kid, kernel.kernel.ops.ids().collect());
                }
                for (kid, op_id) in kernelizer.visited.values() {
                    if !kernel_op_ids[kid].contains(op_id) {
                        kernelizer.kernels[*kid].debug();
                        panic!("Missing op_id={op_id} in kernel {kid:?}");
                    }
                    if !kernelizer.kernels.contains_key(*kid) {
                        panic!("Missing kid");
                    }
                }
            }*/

            if to_eval.contains(&nid) && !kernelizer.realized_nodes.contains(&nid) {
                //println!("Adding store due to to_eval for {nid}");
                kernelizer.add_store(nid)?;
                *kernelizer.rcs.get_mut(&nid).unwrap() -= 1;
                if kernelizer.rcs[&nid] > 0 {
                    kernelizer.create_load_kernel(nid);
                }
            }

            //kernelizer.debug();
        }

        if kernelizer.kernels.len() > KMKernelId(0) {
            let mut kids: Vec<KMKernelId> = kernelizer.kernels.ids().collect();
            while let Some(kid) = kids
                .iter()
                .find(|&&kid| {
                    kernelizer.kernels[kid]
                        .loads
                        .iter()
                        .all(|x| kernelizer.realized_nodes.contains(x))
                })
                .copied()
            {
                kids.retain(|x| *x != kid);
                let kernel = unsafe { kernelizer.kernels.remove_and_return(kid) };

                let loads = kernel.loads.clone();
                if !kernel.stores.is_empty() {
                    let stores = kernel.stores.clone();
                    kernelizer.launch_kernel(kernel)?;
                    kernelizer.realized_nodes.extend(stores);
                }

                // Delete unneeded intermediate tensors in memory pools
                let mut to_remove = Set::with_capacity_and_hasher(1, BuildHasherDefault::new());
                for tid in loads {
                    if !kernelizer.kernels.values().any(|kernel| kernel.loads.contains(&tid))
                        && !kernelizer.must_keep_nodes.contains(&tid)
                    {
                        to_remove.insert(tid);
                    }
                }
                deallocate_tensors(&to_remove, kernelizer.pools, kernelizer.temp_data, kernelizer.buffer_map);
            }
        }

        #[cfg(debug_assertions)]
        {
            assert!(kernelizer.kernels.len() <= KMKernelId(0));
            debug_assert!(to_eval.is_subset(&kernelizer.realized_nodes));
            //println!("Realized nodes in kernelizer: {:?}", kernelizer.realized_nodes);
        }

        let elapsed = begin.elapsed();
        if self.debug.perf() {
            println!(
                "Kernelizer took {} μs for {} kernels",
                elapsed.as_micros(),
                kernelizer.n_launches
            );
        }
        Ok(())
    }
}
