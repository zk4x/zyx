//! Converts graph to kernels and schedules them to devices

use crate::{
    Map, Set, ZyxError,
    cache::{Kernel, Op},
    graph::Node,
    runtime::Runtime,
    slab::{Slab, SlabId},
    tensor::TensorId,
};
use std::{collections::VecDeque, hash::BuildHasherDefault};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct KernelId(u32);

impl SlabId for KernelId {
    const ZERO: Self = Self(0);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl From<usize> for KernelId {
    fn from(value: usize) -> Self {
        KernelId(value as u32)
    }
}

impl From<KernelId> for usize {
    fn from(value: KernelId) -> Self {
        value.0 as usize
    }
}

impl Runtime {
    pub fn realize(&mut self, to_eval: &Set<TensorId>) -> Result<(), ZyxError> {
        let begin = std::time::Instant::now();
        let realized_nodes: Set<TensorId> =
            self.pools.iter().flat_map(|pool| pool.buffer_map.keys()).copied().collect();

        let mut kernels: Slab<KernelId, Kernel> = Slab::with_capacity(300);
        let mut kids: VecDeque<(TensorId, KernelId)> = VecDeque::with_capacity(500);

        let mut visited = Map::with_capacity_and_hasher(
            self.graph.nodes.len().into(),
            BuildHasherDefault::default(),
        );

        for &nid in to_eval {
            let op = Op::Store { x: 1 };
            let kid = kernels.len();
            kernels.push(Kernel { ops: vec![op], next_id: 1 });
            kids.push_back((nid, kid));
        }

        while let Some((nid, kid)) = kids.pop_front() {
            // Let's say for now if we find reduce on a later expanded kernel,
            // that is our kernel split, so no expand on reduced kernels.

            let op = if realized_nodes.contains(&nid) {
                let shape = self.graph.shape(nid).to_vec().into_boxed_slice();
                let dtype = self.graph.dtype(nid);
                Op::Load { dtype, shape }
            } else {
                match self.graph[nid] {
                    Node::Leaf { .. } => unreachable!(),
                    Node::Const { value } => Op::Const { value },
                    Node::Cast { x, dtype } => {
                        let x = get_next_id(&mut kernels, &mut kids, &mut visited, x, kid);
                        Op::Cast { x, dtype }
                    }
                    Node::Unary { x, uop } => {
                        let x = get_next_id(&mut kernels, &mut kids, &mut visited, x, kid);
                        Op::Unary { x, uop }
                    }
                    Node::Binary { x, y, bop } => {
                        // With BFS, we can just do x + 1 for y
                        let x = get_next_id(&mut kernels, &mut kids, &mut visited, x, kid);
                        let y = get_next_id(&mut kernels, &mut kids, &mut visited, y, kid);
                        Op::Binary { x, y, bop }
                    }
                    Node::Reduce { x, rop } => {
                        // TODO permute, but not unnecessary
                        let x = get_next_id(&mut kernels, &mut kids, &mut visited, x, kid);
                        let axes = self.graph.axes(nid);
                        Op::Reduce { x, rop, axes: axes.to_vec().into_boxed_slice() }
                    }
                    Node::Permute { x } => {
                        let x = get_next_id(&mut kernels, &mut kids, &mut visited, x, kid);
                        let axes = self.graph.axes(nid);
                        Op::Permute { x, axes: axes.to_vec().into_boxed_slice() }
                    }
                    Node::Reshape { x } => {
                        let x = get_next_id(&mut kernels, &mut kids, &mut visited, x, kid);
                        let shape = self.graph.shape(nid);
                        Op::Reshape { x, shape: shape.to_vec().into_boxed_slice() }
                    }
                    Node::Pad { x } => {
                        let x = get_next_id(&mut kernels, &mut kids, &mut visited, x, kid);
                        let padding = self.graph.padding(nid);
                        Op::Pad { x, padding: padding.to_vec().into_boxed_slice() }
                    }
                    Node::Expand { x } => {
                        // Look ahead if sooner there is another expand or reduce.
                        // If it is reduce, split kernels
                        /*{
                            let mut params = Vec::with_capacity(20);
                            params.push(x);
                            while let Some(param) = params.pop() {
                                param
                            }
                        }*/
                        let x = get_next_id(&mut kernels, &mut kids, &mut visited, x, kid);
                        let shape = self.graph.shape(nid);
                        Op::Expand { x, shape: shape.to_vec().into_boxed_slice() }
                    }
                }
            };
            kernels[kid].ops.push(op);
        }

        let elapsed = begin.elapsed();
        if self.debug.perf() {
            println!("Kernelizer took {} us", elapsed.as_micros(),);
        }
        if self.debug.sched() {
            for kernel in kernels.values() {
                kernel.debug();
            }
        }
        todo!();
        Ok(())
    }

    // 1. gets a set of tensors which need to be processed and in which order
    // 2. generates kernels from them
    // 3. assigns those kernels to devices, compiles and launches them
    /*#[allow(clippy::cognitive_complexity)]
    pub fn realize2(&mut self, to_eval: &Set<TensorId>) -> Result<(), ZyxError> {
        let (realized_nodes, order, rcs, new_leafs, mut to_delete) = {
            let begin = std::time::Instant::now();
            let realized_nodes: Set<TensorId> =
                self.pools.iter().flat_map(|pool| pool.buffer_map.keys()).copied().collect();
            let mut to_eval: Set<TensorId> = to_eval.difference(&realized_nodes).copied().collect();
            if to_eval.is_empty() {
                return Ok(());
            }
            if self.devices.is_empty() {
                self.initialize_devices()?;
            }

            let (order, to_delete, new_leafs, rcs) = if self.graph.gradient_tape.is_some() {
                self.graph_order_with_gradient(&realized_nodes, &mut to_eval)
            } else {
                self.graph_order(&realized_nodes, &mut to_eval)
            };
            let elapsed = begin.elapsed();
            if self.debug.perf() {
                println!(
                    "Runtime realize graph order took {} us for {}/{} tensors with gradient_tape={}",
                    elapsed.as_micros(),
                    order.len(),
                    usize::from(self.graph.nodes.len()),
                    self.graph.gradient_tape.is_some(),
                );
            }
            (realized_nodes, order, rcs, new_leafs, to_delete)
        };

        {
            let rcs = if rcs.is_empty() {
                let mut rcs = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
                // to_eval are not in rcs
                for &nid in &order {
                    if !realized_nodes.contains(&nid) {
                        for nid in self.graph[nid].parameters() {
                            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                        }
                    }
                }
                rcs
            } else {
                rcs
            };

            #[cfg(debug_assertions)]
            {
                let mut rcs2 = Map::with_hasher(BuildHasherDefault::default());
                // to_eval are not in rcs
                for &nid in &order {
                    if !realized_nodes.contains(&nid) {
                        for nid in self.graph[nid].parameters() {
                            rcs2.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                        }
                    }
                }
                if rcs2 != rcs {
                    println!("Realized nodes: {realized_nodes:?}");
                    for &nid in &order {
                        println!(
                            "ID({nid:?}): {:?}, sh: {:?}, rcs: {}, rcs actual: {}",
                            self.graph[nid],
                            self.graph.shape(nid),
                            rcs.get(&nid).copied().unwrap_or(0),
                            rcs2.get(&nid).copied().unwrap_or(0),
                        );
                    }
                    panic!("rcs are incorrect, rcs: {rcs:?}\nrcs2: {rcs2:?}");
                }
            }
        }
        todo!();

        // All ops that have not been evaluated yet will be evaluated here
        // All kernels with the same or sufficiently similar shapes should be merged,
        // with possibly some heuristics later to determine which kernels should not be merged.
        // For now we won't merge any kernels.
        /*for (nid, op) in ops {
            println!("{nid} -> {op:?}");

            // Iterate over all memory pools ordered by device speed.
            // Then select first fastest device that has associated memory pool which fits all tensors used
            // as arguments for the kernel that are not yet allocated on that memory pool.
            let loads: Vec<TensorId> = loads[&nid].iter().copied().collect();

            // For NOW there is only one store
            let required_kernel_memory: Dim = [nid]
                .iter()
                .map(|&tid| self.shape(tid).iter().product::<Dim>() * self.dtype(tid) as Dim)
                .sum::<Dim>()
                + loads
                    .iter()
                    .map(|&tid| self.shape(tid).iter().product::<Dim>() * self.dtype(tid) as Dim)
                    .sum::<Dim>();
            let mut dev_ids: Vec<usize> = (0..self.devices.len()).collect();
            dev_ids.sort_unstable_by_key(|&dev_id| self.devices[dev_id].free_compute());
            let mut device_id = None;
            for dev_id in dev_ids {
                let mpid = self.devices[dev_id].memory_pool_id() as usize;
                // Check if kernel arguments fit into associated memory pool
                let free_memory = self.pools[mpid].pool.free_bytes();
                // required memory is lowered by the amount of tensors already stored in that memory pool
                let required_memory = required_kernel_memory
                    - self.pools[mpid]
                        .buffer_map
                        .keys()
                        .map(|&tid| {
                            self.shape(tid).iter().product::<Dim>() * self.dtype(tid) as Dim
                        })
                        .sum::<Dim>();
                if free_memory > required_memory {
                    device_id = Some(dev_id);
                    break;
                }
            }
            // else
            let Some(dev_id) = device_id else { return Err(ZyxError::AllocationError) };
            let mpid = self.devices[dev_id].memory_pool_id() as usize;

            let mut event_wait_list = Vec::new();
            // Move all loads to that pool if they are not there already.
            for tid in &loads {
                if !self.pools[mpid].buffer_map.contains_key(tid) {
                    let tid = *tid;
                    #[allow(clippy::map_entry)]
                    if !self.pools[mpid].buffer_map.contains_key(&tid) {
                        // Check where the tensor is
                        let mut old_mpid = usize::MAX;
                        for (i, pool) in self.pools.iter().enumerate() {
                            if pool.buffer_map.contains_key(&tid) {
                                old_mpid = i;
                                break;
                            }
                        }
                        debug_assert_ne!(old_mpid, usize::MAX);

                        let bytes = self.graph.shape(tid).iter().product::<Dim>()
                            * self.graph.dtype(tid).byte_size() as Dim;
                        // No need to initialize here, other than rust is bad.
                        let mut byte_slice = vec![0u8; bytes as usize];
                        let src = self.pools[old_mpid].buffer_map[&tid];

                        // Move the tensor from old pool into temporary in RAM
                        // TODO later we can implement direct GPU to GPU movement, it's easy here,
                        // a bit harder for the backends.
                        // Pool to host blocks on event, so we can remove that event.
                        let mut event_wait_list = Vec::new();
                        for buffers in self.pools[old_mpid].events.keys() {
                            if buffers.contains(&src) {
                                let buffers = buffers.clone();
                                // Pool to host blocks on event, so we can remove that event.
                                let event = self.pools[old_mpid].events.remove(&buffers).unwrap();
                                event_wait_list.push(event);
                                break;
                            }
                        }
                        self.pools[old_mpid].pool.pool_to_host(
                            src,
                            &mut byte_slice,
                            event_wait_list,
                        )?;

                        // Delete the tensor from the old pool
                        self.pools[old_mpid].pool.deallocate(src, vec![]);
                        self.pools[old_mpid].buffer_map.remove(&tid);
                        //println!("{byte_slice:?}");

                        let (dst, event) = self.pools[mpid].pool.allocate(bytes)?;
                        let event =
                            self.pools[mpid].pool.host_to_pool(&byte_slice, dst, vec![event])?;
                        // We have to sync here, because byte_slice does not exist any more.
                        // The other solution would be to put this into temp_data.
                        // But perhaps we should figure some better async.
                        self.pools[mpid].pool.sync_events(vec![event])?;
                        self.pools[mpid].buffer_map.insert(tid, dst);
                        //memory_pools[mpid].events.insert(BTreeSet::from([dst]), event);
                    }
                }
            }

            // Allocate space for all stores (outputs)
            let mut output_buffers = BTreeSet::new();
            for &tid in &[nid] {
                let bytes =
                    self.shape(tid).iter().product::<Dim>() * self.dtype(tid).byte_size() as Dim;
                let (buffer_id, event) = self.pools[mpid].pool.allocate(bytes)?;
                self.pools[mpid].buffer_map.insert(tid, buffer_id);
                event_wait_list.push(event);
                output_buffers.insert(buffer_id);
            }

            // Get a list of all args. These must be specifically in order as they are mentioned in kernel ops
            let mut args = Vec::new();
            for tid in &loads {
                args.push(self.pools[mpid].buffer_map[tid]);
            }
            for tid in &[nid] {
                args.push(self.pools[mpid].buffer_map[tid]);
            }

            // Send the kernel to kernel cache.
            if let Some(event) = self.cache.launch(
                &op,
                u32::try_from(dev_id).unwrap(),
                &mut self.devices[dev_id],
                &mut self.pools[mpid],
                &args,
                event_wait_list,
                self.search_iterations,
                self.debug,
            )? {
                self.pools[mpid].events.insert(output_buffers, event.clone());
            }

            // TODO Deallocate loads that are not used by any other kernel
        }*/

        // Deallocate tensors from devices, new_leafs can be deallocated too
        self.deallocate_tensors(&to_delete);

        // Remove evaluated part of graph unless needed for backpropagation
        for tensor in new_leafs {
            self.graph.add_shape(tensor);
            self.graph[tensor] = Node::Leaf { dtype: self.graph.dtype(tensor) };
            to_delete.remove(&tensor);
        }
        // Delete nodes, but do not use release function, just remove it from graph.nodes
        self.graph.delete_tensors(&to_delete);

        Ok(())
    }

    fn graph_order(
        &mut self,
        realized_nodes: &Set<TensorId>,
        to_eval: &mut Set<TensorId>,
    ) -> (
        Vec<TensorId>,
        Set<TensorId>,
        Set<TensorId>,
        Map<TensorId, u32>,
    ) {
        let old_to_eval = to_eval.clone();
        let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
        let mut rcs: Map<TensorId, u32> =
            Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        while let Some(nid) = params.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                if !realized_nodes.contains(&nid) {
                    params.extend(self.graph.nodes[nid].1.parameters());
                }
                1
            });
        }
        // Order them using rcs reference counts
        let mut to_delete = Set::with_capacity_and_hasher(100, BuildHasherDefault::default());
        let mut new_leafs = Set::with_capacity_and_hasher(10, BuildHasherDefault::default());
        let mut order = Vec::new();
        let mut internal_rcs: Map<TensorId, u32> =
            Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
        while let Some(nid) = params.pop() {
            if let Some(&rc) = rcs.get(&nid) {
                if rc == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                    order.push(nid);
                    let node = &self.graph.nodes[nid];
                    if node.0 > rc {
                        new_leafs.insert(nid);
                        if !realized_nodes.contains(&nid) {
                            to_eval.insert(nid);
                        }
                    } else if !to_eval.contains(&nid) {
                        to_delete.insert(nid);
                    } else {
                        new_leafs.insert(nid);
                    }
                    if !realized_nodes.contains(&nid) {
                        params.extend(node.1.parameters());
                    }
                }
            }
        }
        order.reverse();
        for x in &old_to_eval {
            *rcs.get_mut(x).unwrap() -= 1;
            if *rcs.get(x).unwrap() == 0 {
                rcs.remove(x);
            }
        }
        //println!("Order {order:?}");
        //println!("ToEval {to_eval:?}");
        //println!("ToDelete {to_delete:?}");
        //println!("NewLeafs {new_leafs:?}");
        (order, to_delete, new_leafs, rcs)
    }

    fn graph_order_with_gradient(
        &mut self,
        realized_nodes: &Set<TensorId>,
        to_eval: &mut Set<TensorId>,
    ) -> (
        Vec<TensorId>,
        Set<TensorId>,
        Set<TensorId>,
        Map<TensorId, u32>,
    ) {
        // Get order for evaluation using DFS with ref counting to resolve
        // nodes with more than one parent.
        let (outside_nodes, mut order) = {
            let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
            let mut rcs: Map<TensorId, u32> =
                Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
            while let Some(nid) = params.pop() {
                rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                    params.extend(self.graph.nodes[nid].1.parameters());
                    1
                });
            }
            // Order them using rcs reference counts
            let mut order = Vec::new();
            let mut internal_rcs: Map<TensorId, u32> =
                Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
            let mut outside_nodes =
                Set::with_capacity_and_hasher(100, BuildHasherDefault::default());
            let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
            while let Some(nid) = params.pop() {
                if let Some(&rc) = rcs.get(&nid) {
                    if rc == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                        order.push(nid);
                        let node = &self.graph.nodes[nid];
                        params.extend(node.1.parameters());
                        if node.0 > rc {
                            outside_nodes.insert(nid);
                        }
                    }
                }
            }
            outside_nodes.extend(to_eval.clone());
            order.reverse();
            (outside_nodes, order)
        };
        //println!("Outside nodes: {outside_nodes:?}");
        // Constant folding and deleting unused parts of graph
        let mut new_leafs = Set::with_capacity_and_hasher(100, BuildHasherDefault::default());
        let mut to_delete = Set::with_capacity_and_hasher(100, BuildHasherDefault::default());
        for &nid in &order {
            let node = &self.graph.nodes[nid].1;
            match node.num_parameters() {
                0 => {
                    if !outside_nodes.contains(&nid) {
                        to_delete.insert(nid);
                    }
                }
                1 => {
                    let x = node.param1();
                    if to_delete.contains(&x) {
                        if outside_nodes.contains(&nid) {
                            to_eval.insert(nid);
                            new_leafs.insert(nid);
                        } else {
                            to_delete.insert(nid);
                        }
                    }
                }
                2 => {
                    let (x, y) = node.param2();
                    let xc = to_delete.contains(&x);
                    let yc = to_delete.contains(&y);
                    match (xc, yc) {
                        (true, true) => {
                            if outside_nodes.contains(&nid) {
                                to_eval.insert(nid);
                                new_leafs.insert(nid);
                            } else {
                                to_delete.insert(nid);
                            }
                        }
                        (true, false) => {
                            to_eval.insert(nid);
                            new_leafs.insert(x);
                        }
                        (false, true) => {
                            to_eval.insert(nid);
                            new_leafs.insert(y);
                        }
                        (false, false) => {}
                    }
                }
                _ => unreachable!(),
            }
        }
        //println!("To eval: {to_eval:?}");
        //println!("New leafs: {new_leafs:?}");
        //println!("To delete: {to_delete:?}");
        let to_eval: Set<TensorId> = to_eval.difference(realized_nodes).copied().collect();
        let mut rcs: Map<TensorId, u32> =
            Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
        while let Some(nid) = params.pop() {
            if let Some(rc) = rcs.get_mut(&nid) {
                *rc += 1;
            } else {
                rcs.insert(nid, 1);
                if !realized_nodes.contains(&nid) {
                    params.extend(self.graph.nodes[nid].1.parameters());
                }
            }
        }
        /*for x in &to_eval {
            *rcs.get_mut(x).unwrap() -= 1;
            if *rcs.get(x).unwrap() == 0 {
                rcs.remove(x);
            }
        }*/
        order.retain(|x| rcs.contains_key(x));
        // Currently rcs with gradient tape cannot be used by scheduler, so we give it empty ids
        (
            order,
            to_delete,
            new_leafs,
            Map::with_hasher(BuildHasherDefault::default()),
        )
    }*/
}

fn get_next_id(
    kernels: &mut Slab<KernelId, Kernel>,
    kids: &mut VecDeque<(TensorId, KernelId)>,
    visited: &mut Map<TensorId, (KernelId, usize)>,
    x: TensorId,
    kid: KernelId,
) -> usize {
    let xt = if let Some((kidy, op_id)) = visited.get(&x) {
        if kid == *kidy {
            *op_id
        } else {
            // merge kernels
            todo!()
        }
    } else {
        kids.push_back((x, kid));
        kernels[kid].next_op_id()
    };
    visited.insert(x, (kid, xt));
    xt
}
