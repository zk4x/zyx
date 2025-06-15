//! Converts graph to kernels and schedules them to devices

use crate::{graph::Node, kernel::Op, runtime::Runtime, shape::Dim, tensor::TensorId, Map, Set, ZyxError};
use std::{collections::BTreeSet, hash::BuildHasherDefault};

impl Runtime {
    /// 1. gets a set of tensors which need to be processed and in which order
    /// 2. generates kernels from them
    /// 3. assigns those kernels to devices, compiles and launches them
    #[allow(clippy::cognitive_complexity)]
    pub fn realize(&mut self, to_eval: &Set<TensorId>) -> Result<(), ZyxError> {
        let begin = std::time::Instant::now();

        let mut realized_nodes: Set<TensorId> =
            self.pools.iter().flat_map(|pool| pool.buffer_map.keys()).copied().collect();
        let mut to_eval: Set<TensorId> = to_eval.difference(&realized_nodes).copied().collect();
        if to_eval.is_empty() {
            return Ok(());
        }
        if self.devices.is_empty() {
            self.initialize_devices()?;
        }

        let (order, mut to_delete, new_leafs, rcs) = if self.graph.gradient_tape.is_some() {
            self.graph_order_with_gradient(&realized_nodes, &mut to_eval)
        } else {
            self.graph_order(&realized_nodes, &mut to_eval)
        };
        let elapsed = begin.elapsed();
        if self.debug.perf() {
            println!(
                "Runtime realize graph order took {} us for {}/{} tensors with gradient_tape = {}",
                elapsed.as_micros(),
                order.len(),
                self.graph.nodes.len(),
                self.graph.gradient_tape.is_some(),
            );
        }

        // All tensors from loads
        // When a tensor needs load that is not yet in realized tensors, we can immediatelly send it for execution
        let mut ops: Map<TensorId, Op> = Map::with_hasher(Default::default());
        let mut loads: Map<TensorId, Vec<TensorId>> = Map::with_hasher(Default::default());
        {
            let graph = &self.graph;
            let order: &[TensorId] = &order;
            let to_eval: &Set<TensorId> = &to_eval;
            let memory_pools = &self.pools;
            let realized_nodes: &Set<TensorId> = &realized_nodes;
            let mut rcs = if rcs.is_empty() {
                let mut rcs = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
                // to_eval are not in rcs
                for &nid in order {
                    if !realized_nodes.contains(&nid) {
                        for nid in graph[nid].parameters() {
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
                for &nid in order {
                    if !realized_nodes.contains(&nid) {
                        for nid in graph[nid].parameters() {
                            rcs2.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                        }
                    }
                }
                if rcs2 != rcs {
                    println!("Realized nodes: {realized_nodes:?}");
                    for &nid in order {
                        println!(
                            "ID({nid:?}): {:?}, sh: {:?}, rcs: {}, rcs actual: {}",
                            graph[nid],
                            graph.shape(nid),
                            rcs.get(&nid).copied().unwrap_or(0),
                            rcs2.get(&nid).copied().unwrap_or(0),
                        );
                    }
                    panic!("rcs are incorrect, rcs: {rcs:?}\nrcs2: {rcs2:?}");
                }
            }

            for &nid in order {
                match graph[nid] {
                    Node::Const { value } => {
                        ops.insert(nid, Op::constant(value));
                    }
                    Node::Leaf { dtype } => {
                        let shape = graph.shape(nid);
                        ops.insert(nid, Op::load(&shape, dtype));
                        loads.insert(nid, vec![nid]);
                    }
                    Node::Permute { x } => {
                        let axes = graph.axes(nid);
                        ops.get_mut(&x).unwrap().movement(|view| view.permute(axes));
                    }
                    Node::Reshape { x } => {
                        let shape = graph.shape(nid);
                        ops.get_mut(&x).unwrap().movement(|view| view.reshape(0..shape.len(), shape));
                    }
                    Node::Reduce { x, rop } => {
                        // First permute
                        // Then apply reduce
                        let op = ops.remove(&x).unwrap();
                        let axes = graph.axes(nid);
                        let shape = graph.shape(nid);
                        ops.insert(nid, Op::reduce(op, rop, axes, shape.len()));
                    }
                    Node::Cast { x, dtype } => {
                        let op = ops.remove(&x).unwrap();
                        ops.insert(nid, Op::cast(op, dtype));
                    }
                    Node::Unary { x, uop } => {
                        let op = ops.remove(&x).unwrap();
                        ops.insert(nid, Op::unary(op, uop));
                    }
                    Node::Binary { x, y, bop } => {
                        let opx = ops.remove(&x).unwrap();
                        let opy = ops.remove(&y).unwrap();
                        ops.insert(nid, Op::binary(opx, opy, bop));
                        let mut loadsx = loads.remove(&x).unwrap();
                        let loadsy = loads.remove(&x).unwrap();
                        loadsx.extend(loadsy);
                        loads.insert(nid, loadsx);
                    }
                    // Padding and Expand are only nodes that are better not fused under specific conditions,
                    // thus these separate kernels.
                    Node::Expand { x } => {
                        //let shape = graph.shape(nid);
                        //ops.get_mut(&x).unwrap().movement(|view| view.expand(shape));
                        todo!();
                    }
                    Node::Pad { x } => {
                        //let padding = graph.padding(nid);
                        //ops.get_mut(&x).unwrap().movement(|view| view.pad(padding));
                        todo!();
                    }
                }
                if to_eval.contains(&nid) {
                    let op = ops.remove(&nid).unwrap();
                    ops.insert(nid, Op::store(op, graph.shape(nid)));
                }
            }
        }

        // All ops that have not been evaluated yet will be evaluated here
        // All kernels with the same or sufficiently similar shapes should be merged,
        // with possibly some heuristics later to determine which kernels should not be merged.
        // For now we won't merge any kernels.
        for (nid, op) in ops {
            println!("{nid} -> {op:?}");

            // Iterate over all memory pools ordered by device speed.
            // Then select first fastest device that has associated memory pool which fits all tensors used
            // as arguments for the kernel that are not yet allocated on that memory pool.
            let loads: Set<TensorId> = loads[&nid].iter().copied().collect();
            let required_kernel_memory: Dim = kernel
                .stores
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
            for &tid in &kernel.stores {
                let bytes =
                    self.shape(tid).iter().product::<Dim>() * self.dtype(tid).byte_size() as Dim;
                let (buffer_id, event) = self.pools[mpid].pool.allocate(bytes)?;
                self.pools[mpid].buffer_map.insert(tid, buffer_id);
                event_wait_list.push(event);
                output_buffers.insert(buffer_id);
            }

            // Get a list of all args. These must be specifically in order as they are mentioned in kernel ops
            let mut args = Vec::new();
            for tid in &kernel.loads {
                args.push(self.pools[mpid].buffer_map[tid]);
            }
            for tid in &kernel.stores {
                args.push(self.pools[mpid].buffer_map[tid]);
            }

            // Send the kernel to kernel cache.
            let event = if let Some(event) = self.kernel_compiler.launch(
                &kernel,
                u32::try_from(dev_id).unwrap(),
                &mut self.devices[dev_id],
                &mut self.pools[mpid],
                &args,
                event_wait_list,
                self.search_iterations,
                self.debug,
            )? {
                self.pools[mpid].events.insert(output_buffers, event.clone());
                Some(event)
            } else {
                None
            };

            // TODO Deallocate loads that are not used by any other kernel
        }

        /*
        // Launch all kernels
        for kid in 0..kernels.len() {
            let kernel = &kernels[kid];
            #[cfg(debug_assertions)]
            if !kernel.has_stores() {
                kernel.debug();
                panic!("Trying to launch kernel without stores");
            }

            // Iterate over all memory pools ordered by device speed.
            // Then select first fastest device that has associated memory pool which fits all tensors used
            // as arguments for the kernel that are not yet allocated on that memory pool.
            let loads: Set<TensorId> = kernel.loads.iter().copied().collect();
            let required_kernel_memory: Dim = kernel
                .stores
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
            for &tid in &kernel.stores {
                let bytes =
                    self.shape(tid).iter().product::<Dim>() * self.dtype(tid).byte_size() as Dim;
                let (buffer_id, event) = self.pools[mpid].pool.allocate(bytes)?;
                self.pools[mpid].buffer_map.insert(tid, buffer_id);
                event_wait_list.push(event);
                output_buffers.insert(buffer_id);
            }

            // Get a list of all args. These must be specifically in order as they are mentioned in kernel ops
            let mut args = Vec::new();
            for tid in &kernel.loads {
                args.push(self.pools[mpid].buffer_map[tid]);
            }
            for tid in &kernel.stores {
                args.push(self.pools[mpid].buffer_map[tid]);
            }

            // Send the kernel to kernel cache.
            let event = if let Some(event) = self.kernel_compiler.launch(
                &kernel,
                u32::try_from(dev_id).unwrap(),
                &mut self.devices[dev_id],
                &mut self.pools[mpid],
                &args,
                event_wait_list,
                self.search_iterations,
                self.debug,
            )? {
                self.pools[mpid].events.insert(output_buffers, event.clone());
                Some(event)
            } else {
                None
            };

            // Deallocate loads that are not used by any other kernel
            let mut loads: Set<TensorId> = loads.difference(&realized_nodes).copied().collect();
            realized_nodes.extend(kernel.stores.iter());
            if kid < kernels.len() - 1 {
                for kid in kid + 1..kernels.len() {
                    for tid in &kernels[kid].loads {
                        loads.remove(tid);
                    }
                }
            }
            for tid in loads {
                for pool in &mut self.pools {
                    if let Some(buffer_id) = pool.buffer_map.remove(&tid) {
                        let mut events = Vec::new();
                        for buffers in pool.events.keys() {
                            if buffers.contains(&buffer_id) {
                                events.push(pool.events.remove(&buffers.clone()).unwrap());
                                break;
                            }
                        }
                        // Push event from the current kernel
                        if let Some(event) = &event {
                            events.push(event.clone());
                        }
                        pool.pool.deallocate(buffer_id, events);
                    }
                }
            }
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
    }
}
