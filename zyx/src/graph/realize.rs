//! Converts graph to kernels and schedules them to devices

use crate::{
    Map, Set, ZyxError, graph::kernel::Op, runtime::Runtime, shape::Dim, tensor::TensorId,
};
use std::{collections::BTreeSet, hash::BuildHasherDefault};

use super::{Node, kernel::kernelize};

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

        //let t = crate::Timer::new("realize_graph");
        let begin = std::time::Instant::now();
        let mut kernels = kernelize(
            &self.graph,
            &order,
            rcs,
            &to_eval,
            &self.pools,
            &realized_nodes,
        );

        let elapsed = begin.elapsed().as_micros();
        let mut min_ops = usize::MAX;
        let mut max_ops = 0;
        let mut avg_ops = 0;
        for kernel in &kernels {
            let n = kernel.ops.len();
            if n > max_ops {
                max_ops = n;
            }
            if n < min_ops {
                min_ops = n;
            }
            avg_ops += n;
        }
        let kernels_len = kernels.len();
        if self.debug.perf() {
            println!(
                "Scheduled {kernels_len} kernels, scheduling took {elapsed}us, ops per kernel: min: {min_ops}, max: {max_ops}, avg: {}",
                avg_ops / kernels_len
            );
        }
        //println!("Expand clones: {expa_u}, reshape clones: {resh_u}, pad clones: {pad_u}, permute clones: {perm_u}, reduce clones: {red_u}");
        // Timer
        /*for (name, (time, iters)) in crate::ET.lock().iter() {
            println!(
                "Timer {name} took {time}us for {iters} iterations, {}us/iter",
                time / iters
            );
        }*/

        // Check for small kernels (to improve performance)
        /*for kernel in kernels.values() {
            if kernel.ops.len() < 20 {
                kernel.debug();
            }
        }*/

        realized_nodes.extend(to_eval);

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
            let loads: Set<TensorId> = kernel.loads.values().copied().collect();
            let required_kernel_memory: Dim = kernel
                .stores
                .keys()
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
            for &tid in kernel.stores.keys() {
                let bytes =
                    self.shape(tid).iter().product::<Dim>() * self.dtype(tid).byte_size() as Dim;
                let (buffer_id, event) = self.pools[mpid].pool.allocate(bytes)?;
                self.pools[mpid].buffer_map.insert(tid, buffer_id);
                event_wait_list.push(event);
                output_buffers.insert(buffer_id);
            }

            // Get a list of all args. These must be specifically in order as they are mentioned in kernel ops
            let mut args = Vec::new();
            for (i, op) in kernel.ops.iter().enumerate() {
                if let Op::Load { .. } = op {
                    let tid = kernel.loads[&i];
                    args.push(self.pools[mpid].buffer_map[&tid]);
                }
            }
            for (i, op) in kernel.ops.iter().enumerate() {
                if let Op::Store { .. } = op {
                    let tid = kernel.stores.iter().find(|(_, x)| **x == i).unwrap().0;
                    args.push(self.pools[mpid].buffer_map[tid]);
                }
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
            realized_nodes.extend(kernel.stores.keys());
            if kid < kernels.len() - 1 {
                for kid in kid+1..kernels.len() {
                    for tid in kernels[kid].loads.values() {
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
        }

        // Deallocate tensors from devices, new_leafs can be deallocated too
        self.deallocate_tensors(&to_delete);

        // Remove evaluated part of graph unless needed for backpropagation
        for tensor in new_leafs {
            self.graph.add_shape(tensor);
            self.graph[tensor] = Node::Leaf { dtype: self.graph.dtype(tensor) };
            to_delete.remove(&tensor);
        }
        // Delete the node, but do not use release function, just remove it from graph.nodes
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
