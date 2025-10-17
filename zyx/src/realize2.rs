//! Converts graph to kernels and schedules them to devices

use std::collections::BTreeSet;

use nanoserde::SerBin;

use crate::{
    backend::ProgramId, error::{BackendError, ErrorStatus}, graph::Node, kernel::{get_perf, Kernel, Op, OpId}, optimizer::Optimizer, prog_bar::ProgressBar, runtime::Runtime, shape::Dim, slab::{Slab, SlabId}, tensor::TensorId, view::View, Set, ZyxError
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

impl Runtime {
    pub fn realize(&mut self, to_eval: &Set<TensorId>) -> Result<(), ZyxError> {
        // Create a separate scope to handle the graph processing
        {
            let mut kernels = Slab::new();
            let mut visited = std::collections::HashSet::new();
            let mut movement_ops = std::collections::HashMap::new();
            let mut tensor_to_kernel = std::collections::HashMap::new();

            // Process tensors in reverse order
            for &tensor_id in to_eval {
                let (_, kernel) = self.parse_backward(
                    tensor_id,
                    &to_eval,
                    &mut kernels,
                    &mut visited,
                    &mut movement_ops,
                    &mut tensor_to_kernel,
                    false,
                )?;
                kernel.debug();
            }
            panic!();

            // Execute kernels
            for (_, kernel) in kernels.iter() {
                let kernel = kernel.clone();
                let loads = Vec::new(); // For backward pass, we don't need loads
                let stores = Vec::new(); // For backward pass, we don't need stores
                println!("Executing kernel with {} ops", kernel.ops.len());
                self.launch_kernel(kernel, loads, stores)?;
            }
        }

        Ok(())
    }

    fn parse_backward(
        &mut self,
        nid: TensorId,
        to_eval: &Set<TensorId>,
        kernels: &mut Slab<KernelId, Kernel>,
        visited: &mut std::collections::HashSet<TensorId>,
        movement_ops: &mut std::collections::HashMap<TensorId, Vec<View>>,
        tensor_to_kernel: &mut std::collections::HashMap<TensorId, (KernelId, usize)>,
        coming_from_store: bool,
    ) -> Result<(OpId, Kernel), ZyxError> {
        if visited.contains(&nid) {
            todo!()
        }

        println!(
            "{nid} -> {:?}  {}  {:?}",
            self.graph[nid],
            self.graph.dtype(nid),
            self.graph.shape(nid)
        );

        if to_eval.contains(&nid) && !coming_from_store {
            let dtype = self.dtype(nid);
            let (src, mut kernel) = self.parse_backward(nid, to_eval, kernels, visited, movement_ops, tensor_to_kernel, true)?;
            let op = Op::StoreView { src, dtype };
            let src = kernel.ops.len();
            kernel.ops.push(op);
            return Ok((src, kernel));
        } else {
            let node = self.graph.nodes[nid].1.clone();

            match node {
                Node::Const { value } => {}
                Node::Leaf { dtype } => {
                    let view = View::contiguous(self.shape(nid));
                    let ops = vec![Op::LoadView { dtype, view }];
                    let kernel = Kernel { ops };
                    let src = 0;
                    return Ok((src, kernel));
                }
                Node::Unary { x, uop } => {
                    let (x, mut kernel) = self.parse_backward(x, to_eval, kernels, visited, movement_ops, tensor_to_kernel, false)?;
                    let op = Op::Unary { x, uop };
                    let src = kernel.ops.len();
                    kernel.ops.push(op);
                    return Ok((src, kernel));
                }
                Node::Binary { x, y, bop } => {
                    let (x, mut kernel) = self.parse_backward(x, to_eval, kernels, visited, movement_ops, tensor_to_kernel, false)?;
                    let (y, kernel_y) = self.parse_backward(y, to_eval, kernels, visited, movement_ops, tensor_to_kernel, false)?;
                    // TODO increment kernel_y.ops and y
                    kernel.ops.extend(kernel_y.ops);
                    let op = Op::Binary { x, y, bop };
                    let src = kernel.ops.len();
                    kernel.ops.push(op);
                    return Ok((src, kernel));
                }
                Node::Cast { x, dtype } => {}
                Node::Reshape { x } => {}
                Node::Permute { x } => {}
                Node::Expand { x } => {}
                Node::Pad { x } => {}
                Node::Reduce { x, .. } => {}
            }
        }
        todo!()
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
