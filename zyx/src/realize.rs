//! Converts graph to kernels and schedules them to devices

use nanoserde::SerBin;

use crate::{
    DType, Map, Set, ZyxError,
    backend::ProgramId,
    error::{BackendError, ErrorStatus},
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
    // 1. gets a set of tensors which need to be processed and in which order
    // 2. generates kernels from them
    // 3. assigns those kernels to devices, compiles and launches them
    #[allow(clippy::cognitive_complexity)]
    pub fn realize(&mut self, to_eval: &Set<TensorId>) -> Result<(), ZyxError> {
        let (to_eval, mut realized_nodes, order, rcs, new_leafs, mut to_delete) = {
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
            (to_eval, realized_nodes, order, rcs, new_leafs, to_delete)
        };

        {
            let mut rcs = if rcs.is_empty() {
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

            for &nid in &to_eval {
                rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
            }

            let begin = std::time::Instant::now();

            // Those nodes that have been store ops in some kernel, but those kernels may have not yet run (must be checked in realized_nodex).
            let mut virt_realized_nodes = realized_nodes.clone();

            let mut kernels: Slab<KernelId, Kernel> = Slab::with_capacity(30);
            let mut visited: Map<TensorId, (KernelId, OpId)> =
                Map::with_capacity_and_hasher(order.len() + 2, BuildHasherDefault::new());
            let mut outputs: Map<KernelId, Vec<TensorId>> = Map::with_hasher(BuildHasherDefault::new());
            let mut loads: Map<KernelId, Vec<TensorId>> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());
            let mut stores: Map<KernelId, Vec<TensorId>> =
                Map::with_capacity_and_hasher(100, BuildHasherDefault::new());

            //println!("{rcs:?}");
            println!("{to_eval:?}");

            for nid in order {
                println!("{nid} x {} -> {:?}", rcs[&nid], self.graph[nid]);
                let (kid, op_id) = if virt_realized_nodes.contains(&nid) {
                    let dtype = self.graph.dtype(nid);
                    let shape = self.graph.shape(nid);
                    /*let view = View::contiguous(shape);
                    let op = Op::LoadView { dtype, view };
                    let kernel = Kernel { ops: vec![op], n_outputs: rcs[&nid] };
                    let kid = kernels.push(kernel);
                    loads.insert(kid, vec![nid]);
                    (kid, 0)*/
                    add_load(nid, shape, dtype, &mut kernels, &mut loads, &mut outputs, rcs[&nid])
                } else {
                    match self.graph[nid] {
                        Node::Leaf { .. } => unreachable!(),
                        Node::Const { value } => {
                            let view = View::contiguous(&[1]);
                            let op = Op::ConstView { value, view };
                            let kernel = Kernel { ops: vec![op] };
                            let kid = kernels.push(kernel);
                            outputs.insert(kid, vec![nid; rcs[&nid] as usize]);
                            (kid, 0)
                        }
                        Node::Expand { x } => {
                            let (mut kid, mut op_id) = visited[&x];
                            // TODO instead of sotre add expand op that inserts loop in IR
                            if outputs[&kid].len() > 1 {
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
                                        &mut loads,
                                        &mut stores,
                                        &mut visited,
                                        &mut outputs,
                                    )?;
                                    let shape = self.graph.shape(x);
                                    (kid, op_id) =
                                        add_load(x, shape, dtype, &mut kernels, &mut loads, &mut outputs, rcs[&x]);
                                    visited.insert(x, (kid, op_id));
                                    if outputs[&kid].len() > 1 {
                                        remove_first(x, kid, &mut outputs);
                                        duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                    }
                                } else {
                                    remove_first(x, kid, &mut outputs);
                                    duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                }
                            }
                            outputs.insert(kid, vec![nid; rcs[&nid] as usize]);
                            let shape = self.graph.shape(nid);
                            kernels[kid].apply_movement(|view| view.expand(shape));
                            *rcs.get_mut(&x).unwrap() -= 1;
                            debug_assert_eq!(self.graph.shape(nid), kernels[kid].shape());
                            (kid, op_id)
                        }
                        Node::Permute { x } => {
                            let (mut kid, mut op_id) = visited[&x];
                            // TODO instead of store add permute op that swaps indices in IR
                            if outputs[&kid].len() > 1 {
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
                                        &mut loads,
                                        &mut stores,
                                        &mut visited,
                                        &mut outputs,
                                    )?;
                                    let shape = self.graph.shape(x);
                                    (kid, op_id) =
                                        add_load(x, shape, dtype, &mut kernels, &mut loads, &mut outputs, rcs[&x]);
                                    visited.insert(x, (kid, op_id));
                                    if outputs[&kid].len() > 1 {
                                        remove_first(x, kid, &mut outputs);
                                        duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                    }
                                } else {
                                    remove_first(x, kid, &mut outputs);
                                    duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                }
                            }
                            outputs.insert(kid, vec![nid; rcs[&nid] as usize]);
                            let axes = self.graph.axes(nid);
                            kernels[kid].apply_movement(|view| view.permute(axes));
                            *rcs.get_mut(&x).unwrap() -= 1;
                            debug_assert_eq!(self.graph.shape(nid), kernels[kid].shape());
                            (kid, op_id)
                        }
                        Node::Reshape { x } => {
                            #[cfg(debug_assertions)]
                            if !visited.contains_key(&x) {
                                panic!("Missing tensor {x} in visited.");
                            }
                            let (mut kid, mut op_id) = visited[&x];

                            // TODO duplicate or store only if this is not mergeable.
                            // Otherwise if it is like unsqueeze or splitting two dims
                            // or fusing two dims, it can be represented by a custom
                            // op that is unfoldable into indices, since it does not change
                            // global work size.

                            if outputs[&kid].len() > 1 {
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
                                        &mut loads,
                                        &mut stores,
                                        &mut visited,
                                        &mut outputs,
                                    )?;
                                    let shape = self.graph.shape(x);
                                    (kid, op_id) =
                                        add_load(x, shape, dtype, &mut kernels, &mut loads, &mut outputs, rcs[&x]);
                                    visited.insert(x, (kid, op_id));
                                    if outputs[&kid].len() > 1 {
                                        remove_first(x, kid, &mut outputs);
                                        duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                    }
                                } else {
                                    remove_first(x, kid, &mut outputs);
                                    duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                }
                            }
                            outputs.insert(kid, vec![nid; rcs[&nid] as usize]);
                            let n = self.graph.shape(x).len();
                            let shape = self.graph.shape(nid);
                            kernels[kid].apply_movement(|view| view.reshape(0..n, shape));
                            *rcs.get_mut(&x).unwrap() -= 1;
                            debug_assert_eq!(self.graph.shape(nid), kernels[kid].shape());
                            (kid, op_id)
                        }
                        Node::Pad { x } => {
                            let (mut kid, mut op_id) = visited[&x];

                            // TODO instead of duplication add pad op that adds if statement into ir (e.g. if idx < padding)
                            if outputs[&kid].len() > 1 {
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
                                        &mut loads,
                                        &mut stores,
                                        &mut visited,
                                        &mut outputs,
                                    )?;
                                    let shape = self.graph.shape(x);
                                    (kid, op_id) =
                                        add_load(x, shape, dtype, &mut kernels, &mut loads, &mut outputs, rcs[&x]);
                                    visited.insert(x, (kid, op_id));
                                    if outputs[&kid].len() > 1 {
                                        remove_first(x, kid, &mut outputs);
                                        duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                    }
                                } else {
                                    remove_first(x, kid, &mut outputs);
                                    duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                }
                            }
                            outputs.insert(kid, vec![nid; rcs[&nid] as usize]);
                            let padding = self.graph.padding(nid);
                            let rank = self.graph.shape(nid).len();
                            kernels[kid].apply_movement(|view| view.pad(rank, padding));
                            *rcs.get_mut(&x).unwrap() -= 1;
                            debug_assert_eq!(self.graph.shape(nid), kernels[kid].shape());
                            (kid, op_id)
                        }
                        Node::Reduce { x, rop } => {
                            // Don't apply reduce if the kernel already contains reduce
                            // and the resulting shape's dimension is less than 256
                            let (mut kid, mut op_id) = visited[&x];

                            // If the kernel has more than one output, or rc of x is more than one,
                            // we have to either copy it (if it is small), or store x (if kid is big)
                            if outputs[&kid].len() > 1 {
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
                                        &mut loads,
                                        &mut stores,
                                        &mut visited,
                                        &mut outputs,
                                    )?;
                                    let shape = self.graph.shape(x);
                                    (kid, op_id) =
                                        add_load(x, shape, dtype, &mut kernels, &mut loads, &mut outputs, rcs[&x]);
                                    visited.insert(x, (kid, op_id));
                                    if outputs[&kid].len() > 1 {
                                        remove_first(x, kid, &mut outputs);
                                        duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                    }
                                } else {
                                    remove_first(x, kid, &mut outputs);
                                    duplicate_kernel(&mut kid, &mut kernels, &mut loads, &mut stores);
                                }
                            }
                            outputs.insert(kid, vec![nid; rcs[&nid] as usize]);

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
                            *rcs.get_mut(&x).unwrap() -= 1;
                            debug_assert_eq!(self.graph.shape(nid), kernels[kid].shape());
                            (kid, kernels[kid].ops.len() - 1)
                        }
                        Node::Cast { x, dtype } => {
                            let (kid, op_id) = visited[&x];
                            kernels[kid].ops.push(Op::Cast { x: op_id, dtype });
                            remove_first(x, kid, &mut outputs);
                            outputs.get_mut(&kid).unwrap().extend(vec![nid; rcs[&nid] as usize]);
                            *rcs.get_mut(&x).unwrap() -= 1;
                            (kid, kernels[kid].ops.len() - 1)
                        }
                        Node::Unary { x, uop } => {
                            let (kid, op_id) = visited[&x];
                            kernels[kid].ops.push(Op::Unary { x: op_id, uop });
                            remove_first(x, kid, &mut outputs);
                            outputs.get_mut(&kid).unwrap().extend(vec![nid; rcs[&nid] as usize]);
                            *rcs.get_mut(&x).unwrap() -= 1;
                            (kid, kernels[kid].ops.len() - 1)
                        }
                        Node::Binary { x, y, bop } => {
                            //let (kid, op_id) = visited[&x];
                            let (kid, op_id) = visited[&x];
                            let (kidy, op_idy) = visited[&y];
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
                                remove_first(x, kid, &mut outputs);
                                remove_first(y, kid, &mut outputs);
                                outputs.get_mut(&kid).unwrap().extend(vec![nid; rcs[&nid] as usize]);
                                let op = Op::Binary { x: op_id, y: op_idy, bop };
                                kernels[kid].ops.push(op);
                            } else if stores.get(&kid).map(|x| x.is_empty()).unwrap_or(true) {
                                // The issues is what to do if not mergeable (like both kernels have stores)
                                // and op_id is not the last op in that kernel, then we don't know if there was
                                // not some movement operation applied. Perhaps we have to check if RC > 1
                                // before applying movement ops and not apply movement op in that case.
                                // This is solved, because we never apply movement on kernel that contains ops
                                // that will be used elsewhere.

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

                                if let Some(kidy_loads) = loads.remove(&kidy) {
                                    if let Some(kid_loads) = loads.get_mut(&kid) {
                                        kid_loads.extend(kidy_loads);
                                    } else {
                                        loads.insert(kid, kidy_loads);
                                    }
                                }
                                if let Some(kidy_stores) = stores.remove(&kidy) {
                                    stores.insert(kid, kidy_stores);
                                }

                                remove_first(x, kid, &mut outputs);
                                remove_first(y, kidy, &mut outputs);
                                let youtputs = outputs.remove(&kidy).unwrap();
                                let xoutputs = outputs.get_mut(&kid).unwrap();
                                xoutputs.extend(youtputs);
                                xoutputs.extend(vec![nid; rcs[&nid] as usize]);

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

                            *rcs.get_mut(&x).unwrap() -= 1;
                            *rcs.get_mut(&y).unwrap() -= 1;
                            (kid, kernels[kid].ops.len() - 1)
                        }
                    }
                };
                visited.insert(nid, (kid, op_id));

                if to_eval.contains(&nid) {
                    let dtype = self.graph.dtype(nid);
                    self.add_store(
                        nid,
                        kid,
                        op_id,
                        dtype,
                        &mut realized_nodes,
                        &mut virt_realized_nodes,
                        &mut kernels,
                        &mut loads,
                        &mut stores,
                        &mut visited,
                        &mut outputs,
                    )?;
                    *rcs.get_mut(&nid).unwrap() -= 1;
                    if rcs[&nid] > 0 {
                        let shape = self.graph.shape(nid);
                        let (kid, op_id) =
                            add_load(nid, shape, dtype, &mut kernels, &mut loads, &mut outputs, rcs[&nid]);
                        visited.insert(nid, (kid, op_id));
                    }
                }

                /*for (kid, kernel) in kernels.iter() {
                    println!("outputs={:?}", outputs[&kid]);
                    kernel.debug();
                }*/
            }

            if kernels.len() > KernelId(0) {
                let kids: Vec<KernelId> = kernels.ids().collect();
                while let Some(kid) = kids
                    .iter()
                    .find(|&&kid| {
                        loads.get(&kid).map(|loads| loads.iter().all(|x| realized_nodes.contains(x))).unwrap_or(true)
                    })
                    .copied()
                {
                    let kernel = unsafe { kernels.remove_and_return(kid) };
                    //println!("outputs={:?}", outputs[&kid]);
                    //kernel.debug();
                    realized_nodes.extend(&*stores[&kid]);
                    let kstores = stores.remove(&kid).unwrap();
                    if let Some(kernel_loads) = loads.remove(&kid) {
                        self.launch_kernel(kernel, kernel_loads.clone(), kstores)?;

                        // Delete unneeded intermediate tensors in memory pools
                        /*for tid in kernel_loads {
                            if !loads.values().any(|loads| loads.contains(&tid)) {
                                // drop tid from memory pools
                                let mut to_remove = Set::with_capacity_and_hasher(1, BuildHasherDefault::new());
                                to_remove.insert(tid);
                                self.deallocate_tensors(&to_remove);
                            }
                        }*/
                    } else {
                        self.launch_kernel(kernel, Vec::new(), kstores)?;
                    }
                }
            }

            #[cfg(debug_assertions)]
            {
                if kernels.len() > KernelId(0) {
                    println!("Unrealized kernels:");
                    for (kid, kernel) in kernels.iter() {
                        println!("{kid:?}, outputs={:?}", outputs[&kid]);
                        kernel.debug();
                        println!();
                    }
                    panic!();
                }
                debug_assert!(to_eval.is_subset(&realized_nodes));
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
        // Delete nodes, but do not use release function (don't deallocate again),
        // only remove it from graph.nodes
        self.graph.delete_tensors(&to_delete);

        Ok(())
    }

    /// Stores x and returns remaining reference count to x
    fn add_store(
        &mut self,
        x: TensorId,
        kid: KernelId,
        op_id: usize,
        dtype: DType,
        realized_nodes: &mut Set<TensorId>,
        virt_realized_nodes: &mut Set<TensorId>,
        kernels: &mut Slab<KernelId, Kernel>,
        loads: &mut Map<KernelId, Vec<TensorId>>,
        stores: &mut Map<KernelId, Vec<TensorId>>,
        visited: &mut Map<TensorId, (KernelId, OpId)>,
        outputs: &mut Map<KernelId, Vec<TensorId>>,
    ) -> Result<(), ZyxError> {
        visited.remove(&x).unwrap();
        virt_realized_nodes.insert(x);
        kernels[kid].ops.push(Op::StoreView { src: op_id, dtype });
        /*if let Some(stores) = stores.get(&kid) {
            println!("\nStoring, stores: {stores:?}");
        }*/
        stores.entry(kid).and_modify(|vec| vec.push(x)).or_insert_with(|| vec![x]);
        //println!("Storing, stores: {:?}", stores[&kid]);

        // remove all references to x
        outputs.get_mut(&kid).unwrap().retain(|&elem| elem != x);

        //kernels[kid].debug();
        if outputs[&kid].is_empty()
            && loads.get(&kid).map(|loads| loads.iter().all(|x| realized_nodes.contains(x))).unwrap_or(true)
        {
            outputs.remove(&kid);
            let kernel = unsafe { kernels.remove_and_return(kid) };
            realized_nodes.extend(&*stores[&kid]);
            let stores = stores.remove(&kid).unwrap();
            if let Some(kernel_loads) = loads.remove(&kid) {
                self.launch_kernel(kernel, kernel_loads.clone(), stores)?;

                // Delete unneeded intermediate tensors in memory pools
                /*for tid in kernel_loads {
                    if !loads.values().any(|loads| loads.contains(&tid)) {
                        // drop tid from memory pools
                        let mut to_remove = Set::with_capacity_and_hasher(1, BuildHasherDefault::new());
                        to_remove.insert(tid);
                        self.deallocate_tensors(&to_remove);
                    }
                }*/
            } else {
                self.launch_kernel(kernel, Vec::new(), stores)?;
            }
        }
        //println!("ADDED STORE for {x} x {xrc_rem}");
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

    fn launch_kernel(
        &mut self,
        mut kernel: Kernel,
        loads: Vec<TensorId>,
        stores: Vec<TensorId>,
    ) -> Result<(), ZyxError> {
        // Iterate over all memory pools ordered by device speed.
        // Then select first fastest device that has associated memory pool which fits all tensors used
        // as arguments for the kernel that are not yet allocated on that memory pool.

        //println!("Loads: {loads:?}");
        //println!("Stores: {stores:?}");
        //println!("Kernel launch");

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
        dev_ids.reverse();
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
        for &tid in &loads {
            if !self.pools[mpid].buffer_map.contains_key(&tid) {
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
        for &tid in &stores {
            let bytes = self.shape(tid).iter().product::<Dim>() * self.dtype(tid).byte_size() as Dim;
            let (buffer_id, event) = self.pools[mpid].pool.allocate(bytes)?;
            self.pools[mpid].buffer_map.insert(tid, buffer_id);
            event_wait_list.push(event);
            output_buffers.insert(buffer_id);
        }

        // Get a list of all arg buffers. These must be specifically in order as they are mentioned in kernel ops
        let mut args = Vec::new();
        for tid in loads {
            args.push(self.pools[mpid].buffer_map[&tid]);
        }
        for tid in stores {
            args.push(self.pools[mpid].buffer_map[&tid]);
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
                optimizer = opt.clone();
            } else {
                // It was optimized for different device
                optimizer = Optimizer::new(&kernel, device.info());
            }
            kernel_id = kid;
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

        // Check if best optimization already found
        if optimizer.fully_optimized() {
            // done optimizing, loaded best from disk
            let opt_res = optimizer.apply_optimization(&mut kernel, optimizer.best_optimization());
            debug_assert!(opt_res);
            if self.debug.ir() {
                println!("\nIR optimized kernel");
                kernel.debug();
                println!();
            }
            let program_id = device.compile(&kernel, self.debug.asm())?;
            let event = device.launch(program_id, &mut pool.pool, &args, event_wait_list)?;
            self.pools[mpid].events.insert(output_buffers, event);
            return Ok(());
        }

        // If search_iters == 0, we use default optimizations
        if self.search_config.iterations == 0 {
            let mut okernel;
            loop {
                okernel = kernel.clone();
                let optimization =
                    optimizer.next_optimization(u128::MAX).unwrap_or_else(|| optimizer.best_optimization());
                if optimizer.apply_optimization(&mut okernel, optimization) {
                    break;
                }
            }

            if self.debug.ir() {
                println!("\nIR optimized kernel");
                okernel.debug();
                println!();
            }

            let program_id = device.compile(&okernel, self.debug.asm())?;
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
            let mut last_time_nanos = u128::MAX;

            pool.pool.sync_events(event_wait_list)?;

            let mut progress_bar = if self.debug.perf() {
                let (flop, mem_read, mem_write) = kernel.flop_mem_rw();
                Some((
                    ProgressBar::new(self.search_config.iterations as u64),
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
            while let Some(optimization) = optimizer.next_optimization(last_time_nanos)
                && i < self.search_config.iterations
            {
                i += 1;
                let mut kernel = kernel.clone();
                if !optimizer.apply_optimization(&mut kernel, optimization) {
                    continue;
                }
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
                    if let Err(err) = res {
                        match err.status {
                            ErrorStatus::KernelCompilation
                            | ErrorStatus::IncorrectKernelArg
                            | ErrorStatus::KernelLaunch
                            | ErrorStatus::KernelSync => {}
                            _ => {
                                println!();
                                return Err(ZyxError::BackendError(err));
                            }
                        }
                    }
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
        if self.search_config.save_to_disk {
            if let Some(mut path) = self.config_dir.as_ref().cloned() {
                path.push("cached_kernels");
                let ser_cache: Vec<u8> = self.cache.serialize_bin();
                std::fs::write(path, ser_cache)?;
            }
        }

        Ok(())
    }
}

fn add_load(
    x: TensorId,
    shape: &[Dim],
    dtype: DType,
    kernels: &mut Slab<KernelId, Kernel>,
    loads: &mut Map<KernelId, Vec<TensorId>>,
    outputs: &mut Map<KernelId, Vec<TensorId>>,
    rc: u32,
) -> (KernelId, OpId) {
    //println!("ADDING LOAD for {x} x {}", rc);
    let view = View::contiguous(shape);
    let op = Op::LoadView { dtype, view };
    let kernel = Kernel { ops: vec![op] };
    let kid = kernels.push(kernel);
    outputs.insert(kid, vec![x; rc as usize]);
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

fn remove_first(x: TensorId, kid: KernelId, outputs: &mut Map<KernelId, Vec<TensorId>>) {
    let outputs = outputs.get_mut(&kid).unwrap();
    outputs.iter().position(|elem| *elem == x).map(|i| outputs.remove(i));
}
