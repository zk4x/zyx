//! Converts graph to kernels and schedules them to devices

use crate::{
    DType, Map, Set, ZyxError,
    backend::ProgramId,
    error::BackendError,
    graph::Node,
    kernel::{Kernel, Op, OpId, get_perf},
    optimizer::Optimizer,
    prog_bar::ProgressBar,
    runtime::Runtime,
    shape::Dim,
    slab::{Slab, SlabId},
    tensor::TensorId,
    view::View,
};
use std::{collections::BTreeSet, hash::BuildHasherDefault};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId(u32);

impl SlabId for KernelId {
    const ZERO: Self = Self(0);

    fn inc(&mut self) { self.0 += 1; }
}

impl From<usize> for KernelId {
    fn from(value: usize) -> Self { KernelId(value as u32) }
}

impl From<KernelId> for usize {
    fn from(value: KernelId) -> Self { value.0 as usize }
}

impl Runtime {
    // 1. gets a set of tensors which need to be processed and in which order
    // 2. generates kernels from them
    // 3. assigns those kernels to devices, compiles and launches them
    #[allow(clippy::cognitive_complexity)]
    pub fn realize(&mut self, to_eval: &Set<TensorId>) -> Result<(), ZyxError> {
        let (mut realized_nodes, order, rcs, new_leafs, mut to_delete) = {
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

        // Those nodes that have been store ops in some kernel, but those kernels may have not yet run (must be checked in realized_nodex).
        let mut virt_realized_nodes = realized_nodes.clone();

        {
            let mut rcs = if rcs.is_empty() {
                let mut rcs = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
                // to_eval are not in rcs
                for &nid in &order {
                    if !virt_realized_nodes.contains(&nid) {
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
                    if !virt_realized_nodes.contains(&nid) {
                        for nid in self.graph[nid].parameters() {
                            rcs2.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                        }
                    }
                }
                if rcs2 != rcs {
                    println!("Realized nodes: {virt_realized_nodes:?}");
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

            for &nid in to_eval {
                rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
            }

            let begin = std::time::Instant::now();

            let mut kernels: Slab<KernelId, Kernel> = Slab::with_capacity(300);
            let mut visited: Map<TensorId, (KernelId, OpId)> =
                Map::with_capacity_and_hasher(order.len() + 10, BuildHasherDefault::new());
            let mut loads: Map<KernelId, Vec<TensorId>> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
            let mut stores: Map<KernelId, Vec<TensorId>> =
                Map::with_capacity_and_hasher(100, BuildHasherDefault::new());

            //println!("{rcs:?}");

            for nid in order {
                println!("{nid} -> {:?}", self.graph[nid]);
                let (kid, op_id) = if virt_realized_nodes.contains(&nid) {
                    let dtype = self.graph.dtype(nid);
                    let shape = self.graph.shape(nid);
                    /*let view = View::contiguous(shape);
                    let op = Op::LoadView { dtype, view };
                    let kernel = Kernel { ops: vec![op], n_outputs: rcs[&nid] };
                    let kid = kernels.push(kernel);
                    loads.insert(kid, vec![nid]);
                    (kid, 0)*/
                    add_load(nid, shape, dtype, &mut kernels, &mut loads)
                } else {
                    match self.graph[nid] {
                        Node::Leaf { .. } => unreachable!(),
                        Node::Const { value } => {
                            let view = View::contiguous(&[1]);
                            let op = Op::ConstView { value, view };
                            let kernel = Kernel { ops: vec![op], n_outputs: rcs[&nid] };
                            (kernels.push(kernel), 0)
                        }
                        Node::Expand { x } => {
                            let (mut kid, mut op_id) = self.get_visited(x, &mut kernels, &visited, &mut loads);
                            // TODO instead of sotre add expand op that inserts loop in IR
                            if kernels[kid].n_outputs > 1 || rcs[&x] > 1 {
                                if kernels[kid].is_reduce() || kernels[kid].contains_stores() {
                                    let dtype = self.graph.dtype(x);
                                    self.add_store(
                                        x,
                                        kid,
                                        op_id,
                                        dtype,
                                        &mut realized_nodes,
                                        &mut virt_realized_nodes,
                                        &mut kernels,
                                        &loads,
                                        &mut stores,
                                        &mut visited,
                                    )?;
                                    let shape = self.graph.shape(x);
                                    (kid, op_id) = add_load(x, shape, dtype, &mut kernels, &mut loads);
                                } else {
                                    duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                }
                            }
                            kernels[kid].n_outputs = rcs[&nid];
                            let shape = self.graph.shape(nid);
                            kernels[kid].apply_movement(|view| view.expand(shape));
                            debug_assert_eq!(self.graph.shape(nid), kernels[kid].shape());
                            (kid, op_id)
                        }
                        Node::Permute { x } => {
                            let (mut kid, mut op_id) = self.get_visited(x, &mut kernels, &visited, &mut loads);
                            // TODO instead of store add permute op that swaps indices in IR
                            if kernels[kid].n_outputs > 1 || rcs[&x] > 1 {
                                if kernels[kid].is_reduce() || kernels[kid].contains_stores() {
                                    let dtype = self.graph.dtype(x);
                                    self.add_store(
                                        x,
                                        kid,
                                        op_id,
                                        dtype,
                                        &mut realized_nodes,
                                        &mut virt_realized_nodes,
                                        &mut kernels,
                                        &loads,
                                        &mut stores,
                                        &mut visited,
                                    )?;
                                    let shape = self.graph.shape(x);
                                    (kid, op_id) = add_load(x, shape, dtype, &mut kernels, &mut loads);
                                } else {
                                    duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                }
                            }
                            kernels[kid].n_outputs = rcs[&nid];
                            let axes = self.graph.axes(nid);
                            kernels[kid].apply_movement(|view| view.permute(axes));
                            debug_assert_eq!(self.graph.shape(nid), kernels[kid].shape());
                            (kid, op_id)
                        }
                        Node::Reshape { x } => {
                            let (mut kid, mut op_id) = self.get_visited(x, &mut kernels, &visited, &mut loads);

                            // TODO duplicate or store only if this is not mergeable.
                            // Otherwise if it is like unsqueeze or splitting two dims
                            // or fusing two dims, it can be represented by a custom
                            // op that is unfoldable into indices, since it does not change
                            // global work size.

                            if kernels[kid].n_outputs > 1 || rcs[&x] > 1 {
                                if kernels[kid].is_reduce() || kernels[kid].contains_stores() {
                                    let dtype = self.graph.dtype(x);
                                    self.add_store(
                                        x,
                                        kid,
                                        op_id,
                                        dtype,
                                        &mut realized_nodes,
                                        &mut virt_realized_nodes,
                                        &mut kernels,
                                        &loads,
                                        &mut stores,
                                        &mut visited,
                                    )?;
                                    let shape = self.graph.shape(x);
                                    (kid, op_id) = add_load(x, shape, dtype, &mut kernels, &mut loads);
                                } else {
                                    duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                }
                            }
                            kernels[kid].n_outputs = rcs[&nid];
                            let n = self.graph.shape(x).len();
                            let shape = self.graph.shape(nid);
                            kernels[kid].apply_movement(|view| view.reshape(0..n, shape));
                            debug_assert_eq!(self.graph.shape(nid), kernels[kid].shape());
                            (kid, op_id)
                        }
                        Node::Pad { x } => {
                            let (mut kid, mut op_id) = self.get_visited(x, &mut kernels, &visited, &mut loads);

                            // TODO instead of duplication add pad op that add if statement into ir (e.g. if idx < padding)
                            if kernels[kid].n_outputs > 1 || rcs[&x] > 1 {
                                //}
                                if kernels[kid].is_reduce() || kernels[kid].contains_stores() {
                                    let dtype = self.graph.dtype(x);
                                    self.add_store(
                                        x,
                                        kid,
                                        op_id,
                                        dtype,
                                        &mut realized_nodes,
                                        &mut virt_realized_nodes,
                                        &mut kernels,
                                        &loads,
                                        &mut stores,
                                        &mut visited,
                                    )?;
                                    let shape = self.graph.shape(x);
                                    (kid, op_id) = add_load(x, shape, dtype, &mut kernels, &mut loads);
                                } else {
                                    duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                }
                            }
                            //println!("{loads:?}, {kid:?}");
                            kernels[kid].n_outputs = rcs[&nid];

                            let padding = self.graph.padding(nid);
                            kernels[kid].apply_movement(|view| view.pad(padding));
                            debug_assert_eq!(self.graph.shape(nid), kernels[kid].shape());
                            (kid, op_id)
                        }
                        Node::Reduce { x, rop } => {
                            // Don't apply reduce if the kernel already contains reduce
                            // and the resulting shape's dimension is less than 256
                            let (mut kid, mut op_id) = self.get_visited(x, &mut kernels, &visited, &mut loads);

                            // If the kernel has more than one output, or rc of x is more than one,
                            // we have to either copy it (if it is small), or store x (if kid is big)
                            if kernels[kid].n_outputs > 1 || rcs[&x] > 1 {
                                if kernels[kid].is_reduce() || kernels[kid].contains_stores() {
                                    let dtype = self.graph.dtype(x);
                                    self.add_store(
                                        x,
                                        kid,
                                        op_id,
                                        dtype,
                                        &mut realized_nodes,
                                        &mut virt_realized_nodes,
                                        &mut kernels,
                                        &loads,
                                        &mut stores,
                                        &mut visited,
                                    )?;
                                    let shape = self.graph.shape(x);
                                    (kid, op_id) = add_load(x, shape, dtype, &mut kernels, &mut loads);
                                } else {
                                    duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                }
                            }
                            kernels[kid].n_outputs = rcs[&nid];

                            /*for kernel in kernels.iter() {
                                println!("{:?}, {}", kernel.0, kernel.1.n_outputs);
                            }*/

                            let axes = self.graph.axes(nid);
                            #[cfg(debug_assertions)]
                            {
                                use crate::shape::Axis;
                                let mut sorted_axes: Vec<Axis> = axes.into();
                                sorted_axes.sort_unstable();
                                debug_assert_eq!(axes, sorted_axes, "Reduce axes must be sorted.");
                            }

                            //kernels[kid].debug();
                            //println!("{axes:?}");

                            let shape = self.graph.shape(x);
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
                                kernels[kid].apply_movement(|v| v.permute(&permute_axes));
                            }

                            let dims = axes.iter().map(|&a| shape[a]).collect();
                            if shape == dims {
                                kernels[kid].apply_movement(|v| v.reshape(0..1, &[1, shape[0]]));
                            }
                            let op = Op::Reduce { x: op_id, rop, dims };
                            kernels[kid].ops.push(op);
                            (kid, kernels[kid].ops.len() - 1)
                        }
                        Node::Cast { x, dtype } => {
                            let (kid, op_id) = self.get_visited(x, &mut kernels, &visited, &mut loads);
                            kernels[kid].ops.push(Op::Cast { x: op_id, dtype });
                            kernels[kid].n_outputs += rcs[&nid] - 1;
                            (kid, kernels[kid].ops.len() - 1)
                        }
                        Node::Unary { x, uop } => {
                            let (kid, op_id) = self.get_visited(x, &mut kernels, &visited, &mut loads);
                            kernels[kid].ops.push(Op::Unary { x: op_id, uop });
                            kernels[kid].n_outputs += rcs[&nid] - 1;
                            (kid, kernels[kid].ops.len() - 1)
                        }
                        Node::Binary { x, y, bop } => {
                            //let (kid, op_id) = visited[&x];
                            let (kid, op_id) = self.get_visited(x, &mut kernels, &visited, &mut loads);
                            let (kidy, op_idy) = self.get_visited(y, &mut kernels, &visited, &mut loads);
                            /*if nid.0 == 11 {
                                for (id, kernel) in kernels.iter() {
                                    println!("{id:?}");
                                    kernel.debug();
                                    println!();
                                }
                                println!("kid={kid:?}, kidy={kidy:?}");
                            }*/
                            /*println!("visited={visited:?}");
                            println!("loads={loads:?}");
                            println!("kid={kid:?}, kidy={kidy:?}");*/
                            if kid == kidy {
                                kernels[kid].n_outputs = kernels[kid].n_outputs + rcs[&nid] - 2;
                                let op = Op::Binary { x: op_id, y: op_idy, bop };
                                kernels[kid].ops.push(op);
                            } else if true {
                                // The issues is what to do if not mergeable (like both kernels have stores)
                                // and op_id is not the last op in that kernel, then we don't know if there was
                                // not some movement operation applied. Perhaps we have to check if RC > 1
                                // before applying movement ops and not apply movement op in that case.
                                // This is solved, because we never apply movement on kernel that contains ops
                                // that will be used elsewhere.

                                // TODO Binary must deduplicate loops, in reverse, those that are not reduced.

                                let mut kernely = unsafe { kernels.remove_and_return(kidy) };
                                let n = kernels[kid].ops.len();
                                for op in &mut kernely.ops {
                                    match op {
                                        Op::ConstView { .. } | Op::LoadView { .. } | Op::Loop { .. } => {}
                                        Op::StoreView { src: x, .. }
                                        | Op::Cast { x, .. }
                                        | Op::Unary { x, .. }
                                        | Op::Reduce { x, .. } => *x += n,
                                        Op::Binary { x, y, .. } => {
                                            *x += n;
                                            *y += n;
                                        }
                                        _ => unreachable!(),
                                    }
                                }
                                kernels[kid].ops.extend(kernely.ops);

                                if let Some(kidy_loads) = loads.remove(&kidy)
                                    && let Some(kid_loads) = loads.get_mut(&kid)
                                {
                                    kid_loads.extend(kidy_loads);
                                }

                                kernels[kid].n_outputs = kernels[kid].n_outputs + kernely.n_outputs + rcs[&nid] - 2;
                                let op = Op::Binary { x: op_id, y: op_idy + n, bop };
                                kernels[kid].ops.push(op);

                                // Fix visited
                                for (_, (kidm, op_id)) in &mut visited {
                                    if *kidm == kidy {
                                        *kidm = kid;
                                        *op_id += n;
                                    }
                                }
                            } else {
                                todo!()
                            }

                            (kid, kernels[kid].ops.len() - 1)
                        }
                    }
                };
                visited.insert(nid, (kid, op_id));

                /*println!();
                println!("{}, rc={}", kernels[kid].n_outputs, rcs[&nid]);
                kernels[kid].debug();
                println!("{loads:?}, {kid:?}");
                println!();*/

                if to_eval.contains(&nid) {
                    //println!();
                    //kernels[kid].debug();
                    //println!("{}", kernels[kid].n_outputs);
                    let dtype = self.graph.dtype(nid);
                    self.add_store(
                        nid,
                        kid,
                        op_id,
                        dtype,
                        &mut realized_nodes,
                        &mut virt_realized_nodes,
                        &mut kernels,
                        &loads,
                        &mut stores,
                        &mut visited,
                    )?;
                }
            }

            #[cfg(debug_assertions)]
            {
                if kernels.len() > KernelId(0) {
                    println!("Unrealized kernels:");
                    for (id, kernel) in kernels.iter() {
                        println!("{id:?}");
                        kernel.debug();
                        println!();
                    }
                }
            }

            let elapsed = begin.elapsed();
            if self.debug.perf() {
                println!("Kernelizer took {} μs", elapsed.as_micros());
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
        // Delete nodes, but do not use release function, just remove it from graph.nodes
        self.graph.delete_tensors(&to_delete);

        Ok(())
    }

    /// Either get visited kernel from visited, or it's realized kernel,
    /// so create a new load kernel for that.
    fn get_visited(
        &mut self,
        x: TensorId,
        kernels: &mut Slab<KernelId, Kernel>,
        visited: &Map<TensorId, (KernelId, usize)>,
        loads: &mut Map<KernelId, Vec<TensorId>>,
    ) -> (KernelId, usize) {
        if let Some(&(kid, op_id)) = visited.get(&x) {
            (kid, op_id)
        } else {
            let shape = self.graph.shape(x);
            let dtype = self.graph.dtype(x);
            add_load(x, shape, dtype, kernels, loads)
        }
    }

    fn add_store(
        &mut self,
        x: TensorId,
        kid: KernelId,
        op_id: usize,
        dtype: DType,
        realized_nodes: &mut Set<TensorId>,
        virt_realized_nodes: &mut Set<TensorId>,
        kernels: &mut Slab<KernelId, Kernel>,
        loads: &Map<KernelId, Vec<TensorId>>,
        stores: &mut Map<KernelId, Vec<TensorId>>,
        visited: &mut Map<TensorId, (KernelId, OpId)>,
    ) -> Result<(), ZyxError> {
        visited.remove(&x).unwrap();
        virt_realized_nodes.insert(x);
        kernels[kid].ops.push(Op::StoreView { src: op_id, dtype });
        stores.entry(kid).and_modify(|vec| vec.push(x)).or_insert_with(|| vec![x]);
        kernels[kid].n_outputs -= 1;
        if kernels[kid].n_outputs == 0 && loads[&kid].iter().all(|x| realized_nodes.contains(x)) {
            let kernel = unsafe { kernels.remove_and_return(kid) };
            self.launch_kernel(kernel, &*loads[&kid], &*stores[&kid])?;
            realized_nodes.extend(&*stores[&kid]);
        }
        Ok(())
    }

    fn graph_order(
        &self,
        realized_nodes: &Set<TensorId>,
        to_eval: &mut Set<TensorId>,
    ) -> (Vec<TensorId>, Set<TensorId>, Set<TensorId>, Map<TensorId, u32>) {
        let old_to_eval = to_eval.clone();
        let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
        let mut rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
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
        let mut internal_rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
        while let Some(nid) = params.pop()
            && let Some(&rc) = rcs.get(&nid)
        {
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
        &self,
        realized_nodes: &Set<TensorId>,
        to_eval: &mut Set<TensorId>,
    ) -> (Vec<TensorId>, Set<TensorId>, Set<TensorId>, Map<TensorId, u32>) {
        // Get order for evaluation using DFS with ref counting to resolve
        // nodes with more than one parent.
        let (outside_nodes, mut order) = {
            let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
            let mut rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
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
            let mut outside_nodes = Set::with_capacity_and_hasher(100, BuildHasherDefault::default());
            let mut params: Vec<TensorId> = to_eval.iter().copied().collect();
            while let Some(nid) = params.pop()
                && let Some(&rc) = rcs.get(&nid)
            {
                if rc == *internal_rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1) {
                    order.push(nid);
                    let node = &self.graph.nodes[nid];
                    params.extend(node.1.parameters());
                    if node.0 > rc {
                        outside_nodes.insert(nid);
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
        let mut rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
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

    fn launch_kernel(&mut self, mut kernel: Kernel, loads: &[TensorId], stores: &[TensorId]) -> Result<(), ZyxError> {
        // Iterate over all memory pools ordered by device speed.
        // Then select first fastest device that has associated memory pool which fits all tensors used
        // as arguments for the kernel that are not yet allocated on that memory pool.

        let required_kernel_memory: Dim = stores
            .iter()
            .map(|&tid| self.shape(tid).iter().product::<Dim>() * self.dtype(tid).byte_size() as Dim)
            .sum::<Dim>()
            + loads
                .iter()
                .map(|&tid| self.shape(tid).iter().product::<Dim>() * self.dtype(tid).byte_size() as Dim)
                .sum::<Dim>();
        //println!("Kernel requires {required_kernel_memory} B");
        let mut dev_ids: Vec<usize> = (0..self.devices.len()).collect();
        dev_ids.sort_unstable_by_key(|&dev_id| self.devices[dev_id].free_compute());
        let mut device_id = None;
        for dev_id in dev_ids {
            let mpid = self.devices[dev_id].memory_pool_id() as usize;
            // Check if kernel arguments fit into associated memory pool
            let free_memory = self.pools[mpid].pool.free_bytes();
            // required memory is lowered by the amount of tensors already stored in that memory pool
            let existing_memory = loads
                .iter()
                .map(|tid| {
                    if self.pools[mpid].buffer_map.contains_key(tid) {
                        self.shape(*tid).iter().product::<Dim>() * self.dtype(*tid).byte_size() as Dim
                    } else {
                        0
                    }
                })
                .sum::<Dim>();
            //println!("Existing memory {existing_memory} B");
            let required_memory = required_kernel_memory - existing_memory;
            if free_memory > required_memory {
                device_id = Some(dev_id);
                break;
            }
        }
        // else
        let Some(dev_id) = device_id else { return Err(ZyxError::AllocationError) };
        let _ = device_id;
        let mpid = self.devices[dev_id].memory_pool_id() as usize;

        let mut event_wait_list = Vec::new();
        // Move all loads to that pool if they are not there already.
        for tid in loads {
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

                    let bytes =
                        self.graph.shape(tid).iter().product::<Dim>() * self.graph.dtype(tid).byte_size() as Dim;
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
                    self.pools[old_mpid].pool.pool_to_host(src, &mut byte_slice, event_wait_list)?;

                    // Delete the tensor from the old pool
                    self.pools[old_mpid].pool.deallocate(src, vec![]);
                    self.pools[old_mpid].buffer_map.remove(&tid);
                    //println!("{byte_slice:?}");

                    let (dst, event) = self.pools[mpid].pool.allocate(bytes)?;
                    let event = self.pools[mpid].pool.host_to_pool(&byte_slice, dst, vec![event])?;
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
        for &tid in stores {
            let bytes = self.shape(tid).iter().product::<Dim>() * self.dtype(tid).byte_size() as Dim;
            let (buffer_id, event) = self.pools[mpid].pool.allocate(bytes)?;
            self.pools[mpid].buffer_map.insert(tid, buffer_id);
            event_wait_list.push(event);
            output_buffers.insert(buffer_id);
        }

        // Get a list of all arg buffers. These must be specifically in order as they are mentioned in kernel ops
        let mut args = Vec::new();
        for tid in loads {
            args.push(self.pools[mpid].buffer_map[tid]);
        }
        for tid in stores {
            args.push(self.pools[mpid].buffer_map[tid]);
        }

        /***** CACHE and OPTIMIZATION SEARCH *****/

        let device = &mut self.devices[dev_id];
        let pool = &mut self.pools[mpid];

        // Send the kernel to kernel cache.
        let dev_info_id = if let Some(&dev_info_id) = self.cache.device_infos.get(device.info()) {
            dev_info_id
        } else {
            let dev_info_id = self.cache.device_infos.values().max().map_or(0, |id| id.checked_add(1).unwrap());
            assert!(self.cache.device_infos.insert(device.info().clone(), dev_info_id).is_none());
            dev_info_id
        };

        // Launch if it is in cache
        let kernel_id;
        let mut optimizer;
        if let Some(&kid) = self.cache.kernels.get(&kernel) {
            // If it has been compiled for the device
            if let Some(&program_id) = self.cache.programs.get(&(kid, dev_info_id)) {
                let event = device.launch(program_id, &mut pool.pool, &args, event_wait_list)?;
                self.pools[mpid].events.insert(output_buffers, event);
                // TODO Deallocate loads that are not used by any other kernel
                return Ok(());
            } else if let Some(opt) = self.cache.optimizations.get(&(kid, dev_info_id)) {
                // Continue optimizing using optimizations cached to disk
                kernel_id = kid;
                optimizer = opt.clone();
            } else {
                unreachable!()
            }
        } else {
            // If it is not in cache, we just get new empty kernel id where we insert the kernel
            kernel_id = self.cache.kernels.values().copied().max().unwrap_or(0).checked_add(1).unwrap();
            assert!(self.cache.kernels.insert(kernel.clone(), kernel_id).is_none());
            optimizer = Optimizer::new(&kernel, device.info());
        }

        if self.debug.sched() {
            println!("Optimizing kernel, max iterations: {}", optimizer.max_iters());
            kernel.debug();
        }

        // If search_iters == 0, we use default optimizations
        if self.search_iterations == 0 {
            let optimization = optimizer.next_optimization(u128::MAX).unwrap();
            optimizer.apply_optimization(&mut kernel, optimization);

            if self.debug.ir() {
                println!("\nIR optimized kernel");
                kernel.debug();
                println!();
            }

            let program_id = device.compile(&kernel, self.debug.asm())?;
            let nanos = std::time::Instant::now();
            let event = device.launch(program_id, &mut pool.pool, &args, event_wait_list)?;
            pool.pool.sync_events(vec![event])?;
            let nanos = nanos.elapsed().as_nanos();
            //assert!(self.cache.programs.insert((kernel_id, dev_id as u32), program_id).is_none());
            if nanos < optimizer.best_time_nanos {
                self.cache.programs.insert((kernel_id, dev_id as u32), program_id);
            }
            if self.debug.perf() {
                let (flop, mem_read, mem_write) = kernel.flop_mem_rw();
                println!("{}", get_perf(flop, mem_read, mem_write, nanos));
            }
            optimizer.best_time_nanos = nanos;
        } else {
            pool.pool.sync_events(event_wait_list)?;

            let mut progress_bar = if self.debug.perf() {
                let (flop, mem_read, mem_write) = kernel.flop_mem_rw();
                Some((
                    ProgressBar::new(self.search_iterations as u64),
                    flop,
                    mem_read,
                    mem_write,
                ))
            } else {
                None
            };

            /*for &arg in &args {
                let mut data = [0f32; 10];
                Runtime::load_buffer(&mut data, pool, arg)?;
                println!("{data:?}");
            }*/

            let mut i = 0;
            let mut last_time_nanos = u128::MAX;
            while let Some(optimization) = optimizer.next_optimization(last_time_nanos)
                && i < self.search_iterations
            {
                let mut kernel = kernel.clone();
                optimizer.apply_optimization(&mut kernel, optimization);
                if self.debug.ir() {
                    println!("\nIR optimized kernel");
                    kernel.debug();
                    println!();
                }

                let res = (|| -> Result<(ProgramId, u128), BackendError> {
                    let program_id = device.compile(&kernel, self.debug.asm())?;
                    let begin = std::time::Instant::now();
                    let event = device.launch(program_id, &mut pool.pool, &args, Vec::new())?;
                    pool.pool.sync_events(vec![event])?;
                    Ok((program_id, begin.elapsed().as_nanos()))
                })();

                last_time_nanos = if let Ok((program_id, last_time_nanos)) = res {
                    if last_time_nanos < optimizer.best_time_nanos {
                        self.cache.programs.insert((kernel_id, dev_id as u32), program_id);
                    }
                    last_time_nanos
                } else {
                    u128::MAX
                };

                if let Some((prog_bar, flop, mem_read, mem_write)) = &mut progress_bar {
                    prog_bar.inc(
                        1,
                        &format!(
                            "{}, best={}μs",
                            get_perf(*flop, *mem_read, *mem_write, last_time_nanos),
                            if optimizer.best_time_nanos == u128::MAX {
                                "inf"
                            } else {
                                &(optimizer.best_time_nanos / 1000).to_string()
                            }
                        ),
                    );
                }
                i += 1;
            }
            if let Some((_, flop, mem_read, mem_write)) = &progress_bar {
                println!();
                println!(
                    "Best: {}",
                    get_perf(*flop, *mem_read, *mem_write, optimizer.best_time_nanos)
                );
            }
        }
        self.cache.optimizations.insert((kernel_id, dev_info_id), optimizer);

        // TODO Deallocate loads that are not used by any other kernel
        Ok(())
    }
}

fn add_load(
    x: TensorId,
    shape: &[Dim],
    dtype: DType,
    kernels: &mut Slab<KernelId, Kernel>,
    loads: &mut Map<KernelId, Vec<TensorId>>,
) -> (KernelId, OpId) {
    let view = View::contiguous(shape);
    let op = Op::LoadView { dtype, view };
    let kernel = Kernel { ops: vec![op], n_outputs: 1 };
    let kid = kernels.push(kernel);
    loads.insert(kid, vec![x]);
    (kid, 0)
}

fn duplicate_kernel(
    kid: &mut KernelId,
    kernels: &mut Slab<KernelId, Kernel>,
    loads: &mut Map<KernelId, Vec<TensorId>>,
    stores: &mut Map<KernelId, Vec<TensorId>>,
) {
    //println!("Duplicating");
    kernels[*kid].n_outputs -= 1;
    let kernel = kernels[*kid].clone();
    //kernel.n_outputs -= 1;
    let nkid = kernels.push(kernel);
    if let Some(loaded_tensors) = loads.get(kid) {
        loads.insert(nkid, loaded_tensors.clone());
    }
    if let Some(stored_tensors) = stores.get(kid) {
        stores.insert(nkid, stored_tensors.clone());
    }
    *kid = nkid;
}
