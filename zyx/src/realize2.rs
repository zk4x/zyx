//! Converts graph to kernels and schedules them to devices

use std::{collections::BTreeSet, hash::BuildHasherDefault, time::Instant};

use nanoserde::SerBin;

use crate::{
    DType, Map, Set, ZyxError,
    backend::ProgramId,
    error::{BackendError, ErrorStatus},
    graph::{Graph, Node},
    kernel::{Kernel, Op, OpId, get_perf},
    optimizer::Optimizer,
    prog_bar::ProgressBar,
    runtime::Runtime,
    shape::{Axis, Dim},
    slab::{Slab, SlabId},
    tensor::TensorId,
    view::View,
};

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
    fn from(value: KernelId) -> usize {
        value.0 as usize
    }
}

#[derive(Debug, Clone)]
enum MovementOp {
    Expand { shape: Vec<Dim> },
    Reshape { shape: Vec<Dim> },
    Permute { axes: Vec<Axis> },
    Pad { padding: Vec<(isize, isize)> },
}

struct KernelManager<'a> {
    graph: &'a Graph,
    to_eval: &'a Set<TensorId>,
    realized_nodes: Set<TensorId>,
    kernels: Slab<KernelId, Kernel>,
    visited: Map<TensorId, bool>, // tensor_id -> is_expensive node (contains reduce)
                                  //movement_ops: Map<TensorId, Vec<MovementOp>>,
}

impl<'a> KernelManager<'a> {
    fn new(runtime: &'a Runtime, to_eval: &'a Set<TensorId>) -> Self {
        Self {
            graph: &runtime.graph,
            to_eval,
            realized_nodes: runtime.pools.iter().flat_map(|pool| pool.buffer_map.keys()).copied().collect(),
            kernels: Slab::with_capacity(100),
            visited: Map::with_capacity_and_hasher(1000, BuildHasherDefault::new()),
            //movement_ops: Map::with_hasher(BuildHasherDefault::new()),
        }
    }

    fn parse_backward(
        &mut self,
        nid: TensorId,
        coming_from_store: bool,
        mut mops: Vec<MovementOp>,
    ) -> Result<(OpId, Kernel, bool), ZyxError> {
        println!(
            "{nid} -> {:?}  {}  {:?}",
            self.graph[nid],
            self.graph.dtype(nid),
            self.graph.shape(nid)
        );

        if let Some(expensive_node) = self.visited.get(&nid)
            && !coming_from_store
            && *expensive_node
        {
            todo!();
            // First check in which kernel it was already realized and add to that kernel if our ops that we
            // are adding are not too complex or the source did not apply movement ops that would invalidate it.
            // Otherwise we have to realize this tensor and somehow split the existing kernel.

            // Movement ops can be applied on the storeview side if there is expensive subgraph, otherwise they
            // are applied on loads.
        }

        if self.realized_nodes.contains(&nid) {
            let mut view = View::contiguous(self.graph.shape(nid));
            // Replay movement ops on the view
            for mop in mops.iter().rev() {
                match mop {
                    MovementOp::Expand { shape } => view.expand(shape),
                    MovementOp::Reshape { shape } => view.reshape(0..shape.len(), &shape),
                    MovementOp::Permute { axes } => view.permute(axes),
                    MovementOp::Pad { padding } => view.pad(padding.len(), padding),
                }
            }
            let dtype = self.graph.dtype(nid);
            let ops = vec![Op::LoadView { dtype, view }];
            let kernel = Kernel { ops };
            let src = 0;
            self.visited.insert(nid, false);
            return Ok((src, kernel, false));
        }

        if self.to_eval.contains(&nid) && !coming_from_store {
            let dtype = self.graph.dtype(nid);
            let (src, mut kernel, expensive) = self.parse_backward(nid, true, mops)?;
            let op = Op::StoreView { src, dtype };
            let src = kernel.ops.len();
            kernel.ops.push(op);
            self.visited.insert(nid, expensive);
            return Ok((src, kernel, expensive));
        } else {
            let node = self.graph.nodes[nid].1.clone();

            match node {
                Node::Const { value } => {}
                Node::Leaf { .. } => {
                    unreachable!();
                }
                Node::Unary { x, uop } => {
                    let (x, mut kernel, expensive) = self.parse_backward(x, false, mops)?;
                    kernel.ops.push(Op::Unary { x, uop });
                    self.visited.insert(nid, expensive);
                    return Ok((kernel.ops.len() - 1, kernel, expensive));
                }
                Node::Cast { x, dtype } => {
                    let (x, mut kernel, expensive) = self.parse_backward(x, false, mops)?;
                    kernel.ops.push(Op::Cast { x, dtype });
                    self.visited.insert(nid, expensive);
                    return Ok((kernel.ops.len() - 1, kernel, expensive));
                }
                Node::Binary { x, y, bop } => {
                    let (x, mut kernel, expensive) = self.parse_backward(x, false, mops.clone())?;
                    let (y, kernel_y, expensive_y) = self.parse_backward(y, false, mops)?;
                    let expensive = expensive || expensive_y;
                    // TODO increment kernel_y.ops and y
                    kernel.ops.extend(kernel_y.ops);
                    kernel.ops.push(Op::Binary { x, y, bop });
                    self.visited.insert(nid, expensive);
                    return Ok((kernel.ops.len() - 1, kernel, expensive));
                }
                Node::Reshape { x } => {
                    let shape = self.graph.shape(nid).into();
                    mops.push(MovementOp::Reshape { shape });
                    let (src, kernel, expensive) = self.parse_backward(x, false, mops)?;
                    self.visited.insert(nid, expensive);
                    return Ok((src, kernel, expensive));
                }
                Node::Permute { x } => {
                    let axes = self.graph.axes(nid).into();
                    mops.push(MovementOp::Permute { axes });
                    let (src, kernel, expensive) = self.parse_backward(x, false, mops)?;
                    self.visited.insert(nid, expensive);
                    return Ok((src, kernel, expensive));
                }
                Node::Expand { x } => {
                    let shape = self.graph.shape(nid).into();
                    mops.push(MovementOp::Expand { shape });
                    let (src, kernel, expensive) = self.parse_backward(x, false, mops)?;
                    self.visited.insert(nid, expensive);
                    return Ok((src, kernel, expensive));
                }
                Node::Pad { x } => {
                    let padding = self.graph.padding(nid).into();
                    mops.push(MovementOp::Pad { padding });
                    let (src, kernel, expensive) = self.parse_backward(x, false, mops)?;
                    self.visited.insert(nid, expensive);
                    return Ok((src, kernel, expensive));
                }
                Node::Reduce { x, rop } => {
                    let shape = self.graph.axes(x);
                    let axes = self.graph.axes(nid);
                    let dims = axes.iter().map(|&a| shape[a]).collect();
                    let (x, mut kernel, _) = self.parse_backward(x, false, mops)?;
                    kernel.ops.push(Op::Reduce { x, rop, dims });
                    self.visited.insert(nid, true);
                    return Ok((kernel.ops.len() - 1, kernel, true));
                }
            }
        }
        todo!()
    }
}

impl Runtime {
    pub fn realize(&mut self, to_eval: &Set<TensorId>) -> Result<(), ZyxError> {
        let time = Instant::now();

        let realized_nodes: Set<TensorId> =
            self.pools.iter().flat_map(|pool| pool.buffer_map.keys()).copied().collect();

        // Get ref counts
        let mut queue: Vec<TensorId> = to_eval.iter().copied().collect();
        let mut rcs: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        while let Some(nid) = queue.pop() {
            rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                queue.extend(self.graph.nodes[nid].1.parameters());
                1
            });
        }

        // kernel, parent -> child
        let mut kernels: Map<TensorId, (Kernel, OpId)> = Map::default();
        let mut queue: Vec<TensorId> = to_eval.iter().copied().collect();
        let mut rcs2: Map<TensorId, u32> = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());

        while let Some(nid) = queue.pop() {
            if *rcs2.entry(nid).and_modify(|rc| *rc += 1).or_insert_with(|| {
                queue.extend(self.graph.nodes[nid].1.parameters());
                1
            }) != rcs[&nid]
            {
                continue;
            }
            if rcs[&nid] > 1 {}

            if realized_nodes.contains(&nid) {
                let op = Op::LoadView { dtype: todo!(), view: todo!() };
                let kernel = Kernel { ops: vec![op] };
                kernels.insert(nid, (kernel, 0));
            } else {
                match self.graph[nid] {
                    Node::Const { value } => {
                        todo!();
                    }
                    Node::Leaf { dtype } => unreachable!(),
                    Node::Cast { x, dtype } => todo!(),
                    Node::Unary { x, uop } => {
                        let (kernel, x) = kernels.remove(&x).unwrap();
                        let op = Op::Unary { x, uop };
                        kernels.insert(nid, (kernel, kernel.ops.len() - 1));
                    }
                    Node::Binary { bop, x, y } => todo!(),
                    Node::Reduce { x, rop } => todo!(),
                    Node::Expand { x } => todo!(),
                    Node::Permute { x } => todo!(),
                    Node::Reshape { x } => todo!(),
                    Node::Pad { x } => todo!(),
                }
            }
        }

        panic!("Kernelizer took {}us", time.elapsed().as_micros());

        // Execute kernels
        for (_, kernel) in kernel_manager.kernels.iter() {
            let kernel = kernel.clone();
            let loads = Vec::new(); // For backward pass, we don't need loads
            let stores = Vec::new(); // For backward pass, we don't need stores
            println!("Executing kernel with {} ops", kernel.ops.len());
            self.launch_kernel(kernel, loads, stores)?;
        }

        Ok(())
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

        println!("Loads: {loads:?}");
        println!("Stores: {stores:?}");
        println!("Kernel launch");

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
        //println!("dev_ids={dev_ids:?}");
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
            //println!("Free memory {free_memory} B, existing memory {existing_memory} B");
            let required_memory = required_kernel_memory - existing_memory;
            if free_memory > required_memory {
                device_id = Some(dev_id);
                break;
            }
        }
        // else
        let Some(dev_id) = device_id else {
            return Err(ZyxError::AllocationError(
                format!("no device has enough memory to store {required_kernel_memory} B for intermedite tensors.")
                    .into(),
            ));
        };
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
        for tid in &loads {
            args.push(self.pools[mpid].buffer_map[tid]);
        }
        for tid in &stores {
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
                if self.debug.kmd() {
                    println!("Kernel launch from memory pool {} with args: {:?}", mpid, args);
                }
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
            println!(
                "Optimizing kernel stores {stores:?}, loads {loads:?}, max iterations: {}",
                optimizer.max_iters()
            );
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
            if self.debug.kmd() {
                println!("Kernel launch from memory pool {} with args: {:?}", mpid, args);
            }
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
            if self.debug.kmd() {
                println!("Kernel launch from memory pool {} with args: {:?}", mpid, args);
            }
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
