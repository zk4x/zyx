use super::{
    backend::MemoryPoolId,
    ir::IRKernel,
    node::{BOp, ROp},
    BufferId, DeviceId,
};
use crate::{
    index_map::Id,
    runtime::{graph::Graph, ir::Scope, node::Node, view::View, Runtime, ZyxError},
    tensor::TensorId,
};
pub(super) use optimizer::KernelOptimizer;
use std::collections::{BTreeMap, BTreeSet};
use vop::MOp;

// Export Kernel and VOp for IR
pub(super) use kernel::Kernel;
pub(super) use optimizer::KernelOptimization;
pub(super) use vop::VOp;

mod kernel;
// Kernel optimizer, multi device scheduler is optimized elsewhere
mod optimizer;
mod vop;

#[derive(Debug)]
pub(super) struct CompiledGraph {
    sched_graph: Vec<SchedulerOp>,
    flop: u128,
    bytes_read: u128,
    bytes_written: u128,
}

#[derive(Debug)]
enum SchedulerOp {
    // Async launch kernel on device
    Launch(VProgram),
    // Block for kernel to finish execution
    Finish(VProgram),
    // Copy part of tensor between devices
    // This is used for sharding, but can be used for other purposes too,
    // if found usefull
    Move {
        tensor_id: TensorId,
        view: View,
        dst: MemoryPoolId,
    },
    Allocate {
        tensor_id: TensorId,
        memory_pool_id: MemoryPoolId,
        bytes: usize,
        view: View,
    },
    Deallocate {
        tensor_id: TensorId,
        view: View,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct VProgram {
    pub(super) device_id: DeviceId,
    pub(super) program_id: Id,
    pub(super) args: Vec<(TensorId, View, bool)>,
}

impl Runtime {
    #[allow(clippy::cognitive_complexity)]
    pub(super) fn compile_graph(&mut self, mut graph: Graph) -> Result<CompiledGraph, ZyxError> {
        // get order of nodes and graph characteristics, some basic optimizations are node reordering are applied
        let (order, flop, bytes_read, bytes_written) = graph.execution_order();
        // create vop representation
        let mut kernels: Vec<Kernel> = generate_kernels(&graph, &order, self.debug_sched());
        //println!("{:?}", &*crate::ET.lock());
        //panic!();
        //println!("{:?}", self.tensor_buffer_map);
        if self.debug_sched() {
            //for kernel in &kernels { kernel.debug(); }
            println!("Scheduler generated {} kernels", kernels.len());
        }
        //panic!("Done");
        let mut sched_graph: Vec<SchedulerOp> = Vec::new();
        // Simulated device occupation. How many kernels are running on each device, if more than 5, finish first one before launching next one
        let mut device_program_map: BTreeMap<DeviceId, Vec<Id>> = (0..self.devices.len())
            .map(|device_id| (device_id, Vec::new()))
            .collect();
        // Simulated tensor buffer map
        let mut tensor_buffer_map: BTreeMap<(TensorId, View), MemoryPoolId> = self
            .tensor_buffer_map
            .iter()
            .map(|(x, &BufferId { memory_pool_id, .. })| (x.clone(), memory_pool_id))
            .collect();
        let mut temp_tensors: BTreeSet<TensorId> = BTreeSet::new();

        for kid in 0..kernels.len() {
            //for mut kernel in kernels {
            let kernel = &mut kernels[kid];
            let mut program_wait_list = Vec::new();
            // Which kernels we need to wait for before launching the next one?
            for i in (0..sched_graph.len()).rev() {
                if let SchedulerOp::Launch(vprogram) = &sched_graph[i] {
                    for (arg, _, read_only) in &vprogram.args {
                        if !read_only && kernel.inputs().contains(arg) {
                            device_program_map
                                .get_mut(&vprogram.device_id)
                                .unwrap()
                                .retain(|pid| *pid != vprogram.program_id);
                            program_wait_list.push(vprogram.clone());
                        }
                    }
                }
            }
            // Is this kernel shardable across multiple devices?
            let shard = if device_program_map.len() > 1 {
                if let Some((axis, dimension)) = kernel.shard_axis() {
                    Some((axis, dimension))
                } else {
                    None
                }
            } else {
                None
            };
            if let Some((axis, dimension)) = shard {
                let _ = axis;
                let _ = dimension;
                todo!()
            } else {
                // Find fastest device out of least occupied ones
                // Smallest number of programs
                let min_programs = device_program_map
                    .iter()
                    .min_by(|x, y| x.1.len().cmp(&y.1.len()))
                    .unwrap()
                    .1
                    .len();
                /*if min_programs > 32 {
                    // Finish some program before proceeding
                    sched_graph.push(SchedulerOp::Finish(()));
                }*/
                let min_devs: Vec<usize> = device_program_map
                    .iter()
                    .filter_map(|(dev_id, p)| {
                        if p.len() == min_programs {
                            Some(*dev_id)
                        } else {
                            None
                        }
                    })
                    .collect();
                let device_id = *device_program_map
                    .iter()
                    .filter(|(dev_id, _)| min_devs.contains(dev_id))
                    .max_by(|x, y| {
                        self.devices[*x.0]
                            .compute()
                            .cmp(&self.devices[*y.0].compute())
                    })
                    .unwrap()
                    .0;

                let memory_pool_id = self.devices[device_id].memory_pool_id();
                // Allocate memory for outputs
                for output in kernel.outputs() {
                    let key = (output, View::contiguous(graph.shape(output)));
                    if !tensor_buffer_map.contains_key(&key) {
                        let shape = graph.shape(output);
                        let view = View::contiguous(shape);
                        sched_graph.push(SchedulerOp::Allocate {
                            tensor_id: output,
                            memory_pool_id,
                            bytes: shape.iter().product::<usize>()
                                * graph.dtype(output).byte_size(),
                            view: view.clone(),
                        });
                        temp_tensors.insert(output);
                        tensor_buffer_map.insert((output, view), memory_pool_id);
                    }
                }
                // Move necessary inputs to memory pool associated with this device
                //kernel.debug();
                //println!("{tensor_buffer_map:?}");

                for input in &kernel.inputs() {
                    let view = View::contiguous(graph.shape(*input));
                    //println!("Tensor map tensor {input}");
                    let buf_mpid = tensor_buffer_map.remove(&(*input, view.clone())).unwrap();
                    //println!("From {memory_pool_id} to {buf_mpid} {}", self.memory_pools[memory_pool_id].free_bytes());
                    if buf_mpid != memory_pool_id {
                        sched_graph.push(SchedulerOp::Move {
                            tensor_id: *input,
                            dst: memory_pool_id,
                            view: view.clone(),
                        });
                    }
                    tensor_buffer_map.insert((*input, view), memory_pool_id);
                }
                // Finish kernels that contain this kernel's inputs
                for vprogram in program_wait_list {
                    sched_graph.push(SchedulerOp::Finish(vprogram));
                }
                // Prints unoptimized kernel
                if self.debug_sched() {
                    kernel.debug();
                }
                // Disk cached search, works across devices and platforms
                let optimization = self.search_kernel_optimization(kernel, device_id, &graph)?;
                /*let optimization = KernelOptimization {
                    //splits: vec![(3, vec![1, 3]), (0, vec![1, 2]), (2, vec![2, 1, 1]), (1, vec![1, 1, 2]), (0, vec![1, 1, 1])],
                    //splits: vec![(3, vec![1, 3]), (0, vec![1, 2]), (2, vec![1, 2, 1]), (1, vec![1, 2, 1]), (0, vec![1, 1, 1])],
                    splits: vec![(0, vec![1, 8]), (2, vec![8, 1]), (1, vec![8, 1]), (0, vec![1, 1])],
                    permutation: vec![0, 1, 2, 3, 4, 5, 6, 7],
                    local_tiles: false,
                };*/
                if self.debug_sched() {
                    println!("Using optimization {optimization}");
                }
                let (program_id, args) =
                    self.compile_cached(kernel, &optimization, device_id, &graph)?;
                // Since it is not sharded, sharding view is contiguous
                device_program_map
                    .get_mut(&device_id)
                    .unwrap()
                    .push(program_id);
                sched_graph.push(SchedulerOp::Launch(VProgram {
                    device_id,
                    program_id,
                    args,
                }));
                // Deallocate kernel inputs that will not be used by other kernels
                let mut unneeded_tensors = temp_tensors.clone();
                if kid + 1 < kernels.len() {
                    for kernel in &kernels[kid + 1..] {
                        for input in kernel.inputs() {
                            unneeded_tensors.remove(&input);
                        }
                    }
                }
                for tensor in &graph.to_eval {
                    unneeded_tensors.remove(tensor);
                }
                //println!("Unneeded tensors: {unneeded_tensors:?}, kernel inputs {:?} tensor_buffer_map {tensor_buffer_map:?}", &kernels[kid].inputs);
                for tensor_id in unneeded_tensors {
                    let view = View::contiguous(graph.shape(tensor_id));
                    sched_graph.push(SchedulerOp::Deallocate { tensor_id, view });
                }
            }
        }
        // Finish unfinished programs
        let mut unfinished_programs = BTreeSet::new();
        for sched_op in &sched_graph {
            match sched_op {
                SchedulerOp::Launch(program) => {
                    unfinished_programs.insert(program);
                }
                SchedulerOp::Finish(program) => {
                    unfinished_programs.remove(program);
                }
                _ => {}
            }
        }
        let unfinished_programs: BTreeSet<VProgram> =
            unfinished_programs.into_iter().cloned().collect();
        for program in unfinished_programs {
            sched_graph.push(SchedulerOp::Finish(program));
        }

        if self.debug_sched() {
            for sched_op in &sched_graph {
                match sched_op {
                    SchedulerOp::Launch(program) => {
                        println!("Launch kernel {}", self.devices[program.device_id]);
                    }
                    SchedulerOp::Finish(program) => println!(
                        "Finish kernel {} on device {}",
                        program.program_id, program.device_id
                    ),
                    SchedulerOp::Move {
                        tensor_id: tensor,
                        view,
                        dst,
                    } => println!("Move tensor {tensor} with {view} to memory pool {dst:?}"),
                    SchedulerOp::Allocate {
                        tensor_id,
                        bytes,
                        memory_pool_id,
                        view: _,
                    } => {
                        println!("Allocate tensor {tensor_id} on memory pool {memory_pool_id:?} with size {bytes:?} B");
                    }
                    SchedulerOp::Deallocate { tensor_id, .. } => {
                        println!("Deallocate tensor {tensor_id}");
                    }
                }
            }
        }

        Ok(CompiledGraph {
            sched_graph,
            flop,
            bytes_read,
            bytes_written,
        })
    }

    pub(super) fn launch_graph(&mut self, graph: &Graph) -> Result<(), ZyxError> {
        let mut queues: BTreeMap<VProgram, usize> = BTreeMap::new();
        let compiled_graph = &self.compiled_graph_cache[graph];
        //println!("Launching compiled graph: {compiled_graph:?}");
        let begin = std::time::Instant::now();
        for sched_op in &compiled_graph.sched_graph {
            match sched_op {
                SchedulerOp::Launch(vprogram) => {
                    let buffer_ids: Vec<Id> = vprogram
                        .args
                        .iter()
                        .map(|arg| {
                            //println!("Arg {} {}", arg.0, arg.1);
                            self.tensor_buffer_map[&(arg.0, arg.1.clone())].buffer_id
                        })
                        .collect();
                    let device = &mut self.devices[vprogram.device_id];
                    let mpid = device.memory_pool_id();
                    queues.insert(
                        vprogram.clone(),
                        device.launch(
                            vprogram.program_id,
                            &mut self.memory_pools[mpid],
                            &buffer_ids,
                        )?,
                    );
                }
                SchedulerOp::Finish(program) => {
                    for (vprogram, queue) in &mut queues {
                        if program == vprogram {
                            self.devices[program.device_id].sync(*queue)?;
                        }
                    }
                }
                SchedulerOp::Move {
                    tensor_id,
                    view,
                    dst,
                } => {
                    let bytes = view.numel() * graph.dtype(*tensor_id).byte_size();
                    let BufferId {
                        memory_pool_id,
                        buffer_id: src_buffer_id,
                    } = self.tensor_buffer_map[&(*tensor_id, view.clone())];
                    let (src_mps, dst_mps) = self.memory_pools.split_at_mut(*dst);
                    let src_mp = &mut src_mps[memory_pool_id];
                    let dst_mp = &mut dst_mps[0];
                    let dst_buffer_id = dst_mp.allocate(bytes)?;
                    src_mp.pool_to_pool(src_buffer_id, dst_mp, dst_buffer_id, bytes)?;
                    src_mp.deallocate(src_buffer_id)?;
                }
                SchedulerOp::Allocate {
                    tensor_id,
                    memory_pool_id,
                    bytes,
                    view,
                } => {
                    let buffer_id = self.memory_pools[*memory_pool_id].allocate(*bytes)?;
                    self.tensor_buffer_map.insert(
                        (*tensor_id, view.clone()),
                        BufferId {
                            memory_pool_id: *memory_pool_id,
                            buffer_id,
                        },
                    );
                }
                SchedulerOp::Deallocate { tensor_id, view } => {
                    let key = &(*tensor_id, view.clone());
                    if let Some(BufferId {
                        memory_pool_id,
                        buffer_id,
                    }) = self.tensor_buffer_map.get(key)
                    {
                        self.memory_pools[*memory_pool_id].deallocate(*buffer_id)?;
                        self.tensor_buffer_map.remove(key).unwrap();
                    }
                }
            }
        }
        if self.debug_perf() {
            let duration = begin.elapsed();
            print_perf(
                compiled_graph.flop,
                compiled_graph.bytes_read,
                compiled_graph.bytes_written,
                duration.as_nanos(),
            );
        }
        Ok(())
    }

    fn search_kernel_optimization(
        &mut self,
        kernel: &Kernel,
        device_id: DeviceId,
        graph: &Graph,
    ) -> Result<KernelOptimization, ZyxError> {
        let dev_info = self.devices[device_id].info().clone();
        let cache_key = (kernel.clone(), dev_info.clone());
        if let Some(KernelOptimizer::Optimized(optimizations, _)) =
            self.optimizer_cache.get(&cache_key)
        {
            Ok(optimizations.clone())
        } else {
            let search_iters = self.search_iterations;
            let debug_perf = self.debug_perf();
            let debug_sched = self.debug_sched();
            let debug_asm = self.debug_asm();
            // allocate space for inputs and outputs that are not allocated for this kernel
            let mut allocated_temps = Vec::new();
            let mpid = self.devices[device_id].memory_pool_id();
            let mut temp_ids: BTreeSet<TensorId> = kernel.inputs();
            temp_ids.extend(&kernel.outputs());
            for tid in temp_ids {
                let buffer_id = self.memory_pools[mpid].allocate(
                    graph.shape(tid).iter().product::<usize>() * graph.dtype(tid).byte_size(),
                )?;
                allocated_temps.push(buffer_id);
            }

            let flop_mem_rw = if debug_perf {
                Some(kernel.flop_mem_rw())
            } else {
                None
            };
            // Get search space of possible optimizations
            let optimizer = self
                .optimizer_cache
                .entry(cache_key.clone())
                .or_insert_with(|| kernel.new_optimizer(&dev_info));
            let rem_opts = optimizer.remaining();
            if debug_sched {
                println!(
                    "Searching over {search_iters} out of {rem_opts} remaining optimizations.",
                );
            }
            #[cfg(feature = "disk_cache")]
            let mut timer = std::time::Instant::now();
            for i in 0..search_iters {
                let optimizer = self.optimizer_cache.get_mut(&cache_key).unwrap();
                let Some(optimization_id) = optimizer.next() else {
                    if debug_sched {
                        println!(
                            "All optimizations were tried and fastest kernel has been selected."
                        );
                    }
                    #[cfg(feature = "disk_cache")]
                    store_optimizer_cache(
                        &self.optimizer_cache,
                        self.config_dir.clone(),
                        self.debug_sched(),
                    );
                    break;
                };
                if debug_sched {
                    println!(
                        "{:>6}/{} {}",
                        i + 1,
                        search_iters.min(rem_opts),
                        optimizer[optimization_id]
                    );
                }
                // Optimize and compile multiple kernels at once on different threads,
                // since compilation takes ~50ms,
                /*let optimized_kernel = if kernel.is_reduce() {
                    kernel.optimize(&KernelOptimization {
                        local_tiles: true,
                        permutation: vec![0, 1, 2, 3, 4, 5, 6, 7],
                        splits: vec![(3, vec![64, 16]), (0, vec![1, 1024]), (2, vec![8, 16, 8]), (1, vec![8, 16, 8]), (0, vec![1, 1, 1])],
                    })
                } else {
                    kernel.optimize(&optimizer[optimization_id])
                };*/
                let optimized_kernel = kernel.optimize(&optimizer[optimization_id]);
                //optimized_kernel.debug();
                //panic!();
                let (ir_kernel, _) = IRKernel::new(&optimized_kernel.ops);
                let program_id = self.devices[device_id].compile(&ir_kernel, debug_asm)?;
                // Launch kernel and measure it's performance
                let begin = std::time::Instant::now();
                let queue_id = match self.devices[device_id].launch(
                    program_id,
                    &mut self.memory_pools[mpid],
                    &allocated_temps,
                ) {
                    Ok(queue_id) => queue_id,
                    Err(e) => {
                        optimizer.set_exec_time(optimization_id, u128::MAX);
                        if debug_sched {
                            println!("Could not launch, {e:?}, skipping");
                        }
                        continue;
                    }
                };
                if let Err(e) = self.devices[device_id].sync(queue_id) {
                    optimizer.set_exec_time(optimization_id, u128::MAX);
                    if debug_sched {
                        println!("Could not sync, {e:?}, skipping");
                    }
                    continue;
                };
                let exec_time = begin.elapsed().as_nanos();
                let _ = self.devices[device_id].release_program(program_id);
                if let Some((f, mr, mw)) = flop_mem_rw {
                    print_perf(f, mr, mw, exec_time);
                }
                optimizer.set_exec_time(optimization_id, exec_time);
                #[cfg(feature = "disk_cache")]
                if timer.elapsed().as_secs() > 60 {
                    store_optimizer_cache(
                        &self.optimizer_cache,
                        self.config_dir.clone(),
                        debug_sched,
                    );
                    timer = std::time::Instant::now();
                }
            }
            #[cfg(feature = "disk_cache")]
            store_optimizer_cache(
                &self.optimizer_cache,
                self.config_dir.clone(),
                self.debug_sched(),
            );
            if debug_sched {
                println!("Optimization has been finished.\n");
            }
            // Deallocate inputs and outputs that are used only for beam search
            for buffer_id in allocated_temps {
                self.memory_pools[mpid].deallocate(buffer_id)?;
            }
            Ok(self.optimizer_cache.get(&cache_key).unwrap().best().clone())
        }
    }

    // Compiles kernel using given optimizations
    #[allow(clippy::type_complexity)]
    pub(super) fn compile_cached(
        &mut self,
        kernel: &Kernel,
        optimizations: &KernelOptimization,
        device_id: DeviceId,
        graph: &Graph,
    ) -> Result<(Id, Vec<(Id, View, bool)>), ZyxError> {
        let mut program_id = None;
        let kernel_cache_key = (kernel.clone(), optimizations.clone());
        if let Some((dev_id, prog_id)) = self.kernel_cache.get(&kernel_cache_key) {
            if *dev_id == device_id {
                program_id = Some(*prog_id);
            }
        }
        let optimized_kernel = kernel.optimize(optimizations);
        //println!("Compiling kernel with shape {:?}", optimized_kernel.shape);
        //optimized_kernel.debug();
        let (ir_kernel, ir_args) = IRKernel::new(&optimized_kernel.ops);
        if program_id.is_none() {
            if self.debug_ir() {
                ir_kernel.debug();
            }
            let debug_asm = self.debug_asm();
            program_id = Some(self.devices[device_id].compile(&ir_kernel, debug_asm)?);
            self.kernel_cache
                .insert(kernel_cache_key, (device_id, program_id.unwrap()));
        }
        Ok((
            program_id.unwrap(),
            ir_args
                .into_iter()
                .map(|arg| {
                    (
                        arg,
                        View::contiguous(graph.shape(arg)),
                        !kernel.outputs().contains(&arg),
                    )
                })
                .collect(),
        ))
    }
}

/// Generate kernels from graph. This function determines which ops will get fused together
/// and how many kernels will be created.
#[allow(clippy::similar_names)]
#[allow(clippy::cognitive_complexity)]
fn generate_kernels(graph: &Graph, order: &[TensorId], debug: bool) -> Vec<Kernel> {
    //let _t = crate::Timer::new("generate_kernels");
    // This function sorts nodes into smallest number of kernels that can be compiled on the device
    // This function defines loops, loads, stores and elementwise ops.
    // The aim is to sort nodes in such a way, that maximum performance is attained.
    // These kernels mostly keep shapes of original nodes.
    // Further optimization is done in optimize kernels function.
    //println!("Eval: {to_eval:?}");
    let mut kernels: Vec<Kernel> = Vec::new();
    for nid in order.iter().copied() {
        let node = &graph[nid];
        if debug {
            println!(
                "ID({nid})x{}: {node:?}, sh: {:?}",
                graph.rc(nid),
                graph.shape(nid)
            );
        }
        match node {
            Node::Const { value } => {
                let mut ops = shape_to_loops(&[1]);
                ops.push(VOp::Const {
                    z: nid,
                    value: *value,
                    view: View::contiguous(&[1]),
                });
                kernels.push(Kernel { ops });
            }
            Node::Leaf => {
                kernels.push(Kernel::load(nid, graph));
            }
            &Node::Expand { x } => {
                let shape = graph.shape(nid);
                let xshape = graph.shape(x);
                assert_eq!(shape.len(), xshape.len());
                let mut kernel = get_kernel(x, &mut kernels, graph);
                // For now no expand on reduce kernels or kernels that store something.
                // Later this can be done if the store or reduce is in different loop,
                // that is if we are expanding loop after reduce and if store is before
                // that expanded loop.
                if kernel.ops.iter().any(|op| matches!(op, VOp::Store { .. })) || kernel.is_reduce()
                {
                    // TODO not sure if this is perfectly correct. Can it contain x in outputs,
                    // but can it be x evaluated to different values, i.e. some intermediate?
                    if !kernel.outputs().contains(&x) {
                        kernel.store(x, View::contiguous(xshape), graph.dtype(x));
                    }
                    kernels.push(Kernel::load(x, graph));
                    kernel = kernels.last_mut().unwrap();
                }
                //println!("Expanding");
                //kernel.debug();
                assert_eq!(shape.len(), kernel.shape().len());
                let mut expand_axes = BTreeSet::new();
                for (a, d) in kernel.shape().into_iter().enumerate() {
                    if d != shape[a] {
                        assert_eq!(d, 1);
                        expand_axes.insert(a);
                    }
                }
                // We go over ops in reverse, increasing last loops dimension
                //println!("expand_axes = {expand_axes:?}");
                let mut done_expanding = BTreeSet::new();
                for op in kernel.ops.iter_mut().rev() {
                    match op {
                        VOp::Loop {
                            axis,
                            len: dimension,
                        } => {
                            if expand_axes.contains(axis) && done_expanding.insert(*axis) {
                                assert_eq!(*dimension, 1);
                                *dimension = shape[*axis];
                            }
                        }
                        VOp::Load { xview: view, .. } | VOp::Const { view, .. } => {
                            // Done expanding marks which loops are behind us,
                            // so we need to only adjust strides to 0 in axes for those axes that are not behind us yet.
                            for a in expand_axes.difference(&done_expanding) {
                                view.expand(*a, shape[*a]);
                            }
                        }
                        VOp::Store { .. } => {
                            unreachable!();
                            // TODO This will do multiple writes to the same index, so this would probably be better solved in different way,
                            // perhaps doing only single write during the whole loop using if condition, but that could also be added
                            // to View in VOp::Store as optimization when converting to IROps
                            //for a in expand_axes.difference(&done_expanding) { zview.expand(*a, shape[*a]); }
                        }
                        _ => {}
                    }
                }
                kernel.ops.push(VOp::Move {
                    z: nid,
                    x,
                    mop: MOp::Expa,
                });
                //kernel.debug();
                assert_eq!(kernel.shape(), shape);
                //println!("Into");
            }
            &Node::Permute { x } => {
                let axes = graph.axes(nid);
                // Permute shuffles load and store strides
                // It also changes the dimension of loops
                // and shape of kernel
                let kernel = get_kernel(x, &mut kernels, graph);
                kernel.permute(axes);
                kernel.ops.push(VOp::Move {
                    z: nid,
                    x,
                    mop: MOp::Perm,
                });
                assert_eq!(kernel.shape(), graph.shape(nid));
            }
            &Node::Reshape { x } => {
                // Reshape needs to add new loops to the end of the kernel if it is unsqueeze

                // If reshape comes after reduce, then if it just aplits axes, it can be merged,
                // otherwise we have to create new kernel.
                //for kernel in &kernels { kernel.debug(); }

                let shape = graph.shape(nid);
                //println!("Reshape node from {:?} to {:?}", graph.shape(x), shape);
                let mut kernel = get_kernel(x, &mut kernels, graph);
                if !kernel.reshape(&shape) {
                    // else create new kernel after storing results of previous kernel
                    let xdtype = graph.dtype(x);
                    kernel.store(x, View::contiguous(graph.shape(x)), xdtype);
                    let mut ops = shape_to_loops(shape);
                    ops.push(VOp::Load {
                        z: nid,
                        zscope: Scope::Register,
                        zview: View::none(),
                        x,
                        xscope: Scope::Global,
                        xview: View::contiguous(shape),
                        xdtype,
                    });
                    kernels.push(Kernel { ops });
                    kernel = kernels.last_mut().unwrap();
                }
                kernel.ops.push(VOp::Move {
                    z: nid,
                    x,
                    mop: MOp::Resh,
                });
                //println!("\nKernels {kernels:?}\n");
            }
            &Node::Pad { x } => {
                let padding = graph.padding(nid);
                // Pad shrinks or expands dimension of axes, this is ZERO padding
                let mut kernel = get_kernel(x, &mut kernels, graph);
                // Kernel cannot be padded if it containe max reduce.
                // For now kernel also won't be padded if it contains store,
                // but that can be changed.
                if !kernel.can_be_zero_padded() {
                    kernel.store(x, View::contiguous(graph.shape(x)), graph.dtype(x));
                    kernels.push(Kernel::load(x, graph));
                    kernel = kernels.last_mut().unwrap();
                }
                let rank = kernel.shape().len();
                // Get which axes are padded
                let mut padded_axes = BTreeMap::new();
                for (op, &p) in kernel.ops[..rank].iter().rev().zip(padding) {
                    let &VOp::Loop { axis, .. } = op else {
                        unreachable!()
                    };
                    padded_axes.insert(axis, p);
                }
                // Apply padding
                let mut num_paddings = padding.len();
                //println!("Padded axes: {padded_axes:?}");
                for op in &mut kernel.ops {
                    match op {
                        VOp::Loop { axis, len } => {
                            if let Some((lp, rp)) = padded_axes.get(axis) {
                                *len = usize::try_from(isize::try_from(*len).unwrap() + lp + rp)
                                    .unwrap();
                            }
                        }
                        VOp::EndLoop => {
                            num_paddings -= 1;
                            if num_paddings == 0 {
                                break;
                            }
                        }
                        VOp::Const { view, .. }
                        | VOp::Load { xview: view, .. }
                        | VOp::Store { zview: view, .. }
                        | VOp::Accumulator { view, .. } => {
                            for (&axis, &(lp, rp)) in &padded_axes {
                                view.pad(axis, lp, rp);
                            }
                        }
                        _ => {}
                    }
                }
                kernel.ops.push(VOp::Move {
                    z: nid,
                    x,
                    mop: MOp::Padd,
                });
                //kernel.debug();
                assert_eq!(kernel.shape(), graph.shape(nid));
            }
            &Node::Reduce { x, rop } => {
                let axes = graph.axes(nid);
                // TODO do not apply reduce on a previously fully reduced and expanded kernel, this
                // happens in softmax
                let kernel = get_kernel(x, &mut kernels, graph);
                // Reduce removes loops and adds accumulator before those loops that it removes
                //println!("Axes {axes:?}");
                // Permute the axes such that reduce loops are last
                // and keep the order of axes that are not reduced.
                let permute_axes: Vec<usize> = (0..graph.shape(x).len())
                    .filter(|a| !axes.contains(a))
                    .chain(axes.iter().copied())
                    .collect();
                //println!("Permute axes in reduce: {permute_axes:?}");
                kernel.permute(&permute_axes);

                // We can also just merge these reduce loops into single loop, since it gets removed
                // from the resulting shape either way, but only if there are no ops between those loops.

                // Add accumulator
                let num_axes = graph.shape(x).len();
                let mut looped_axes: BTreeSet<usize> = (num_axes - axes.len()..num_axes).collect();
                //println!("Looped axes: {looped_axes:?}");
                let acc_id = kernel.ops.len()
                    - kernel
                        .ops
                        .iter()
                        .rev()
                        .position(|op| {
                            if let VOp::Loop { axis, .. } = op {
                                looped_axes.remove(axis);
                            }
                            looped_axes.is_empty()
                        })
                        .unwrap()
                    - 1;
                //println!("Acc id: {acc_id}");
                kernel.ops.insert(
                    acc_id,
                    VOp::Accumulator {
                        z: nid,
                        rop,
                        view: View::none(),
                        dtype: graph.dtype(nid),
                    },
                );
                kernel.ops.push(VOp::Binary {
                    z: nid,
                    x,
                    y: nid,
                    bop: match rop {
                        ROp::Sum => BOp::Add,
                        ROp::Max => BOp::Max,
                    },
                });
                for _ in 0..axes.len() {
                    kernel.ops.push(VOp::EndLoop);
                }
                if kernel.shape().is_empty() {
                    kernel.insert_loop(0, 0);
                }
                // Optionally merge axes (if possible) for potentially better performance
                //kernel.merge_axes(acc_id + 1, axes.len());
            }
            Node::Unary { x, uop } => {
                let kernel = get_kernel(*x, &mut kernels, graph);
                kernel.ops.push(VOp::Unary {
                    z: nid,
                    x: *x,
                    uop: *uop,
                });
            }
            &Node::Binary { x, y, bop } => {
                // Binary ops may allow us to join two kernels together
                if let Some(id) = kernels
                    .iter_mut()
                    .position(|kernel| kernel.vars().is_superset(&[x, y].into()))
                {
                    // If both inputs are in the same kernel
                    let kernel = if kernels[id].shape() == graph.shape(x) {
                        &mut kernels[id]
                    } else {
                        // create new kernel using already predefined stores of both x and y
                        let mut kernel = Kernel::load(x, graph);
                        kernel.ops.push(VOp::Load {
                            z: y,
                            zscope: Scope::Register,
                            zview: View::none(),
                            x: y,
                            xscope: Scope::Global,
                            xview: View::contiguous(graph.shape(y)),
                            xdtype: graph.dtype(x),
                        });
                        kernels.push(kernel);
                        kernels.last_mut().unwrap()
                    };
                    kernel.ops.push(VOp::Binary { z: nid, x, y, bop });
                } else {
                    // If inputs are in different kernels
                    // TODO rewrite this, this is incorrect
                    //todo!();

                    let kernel_x_id = kernels
                        .iter()
                        .enumerate()
                        .filter(|(_, kernel)| kernel.vars().contains(&x))
                        .min_by_key(|(_, kernel)| kernel.ops.len())
                        .unwrap()
                        .0;
                    let kernel_y_id = kernels
                        .iter()
                        .enumerate()
                        .filter(|(_, kernel)| kernel.vars().contains(&y))
                        .min_by_key(|(_, kernel)| kernel.ops.len())
                        .unwrap()
                        .0;

                    // Check which kernel needs to be evaluated first
                    // TODO make sure that depends on is never needed
                    /*match (
                        depends_on(kernel_x_id, kernel_y_id, &kernels),
                        depends_on(kernel_y_id, kernel_x_id, &kernels),
                    ) {
                        (true, true) => {
                            // This should not be possible
                            panic!()
                        }
                        (true, false) => {
                            // kernel x depends on kernel y
                            // This is ok, nothing needs to be done
                        }
                        (false, true) => {
                            println!("Depends on caused changes");
                            // Here we need to do some reordering,
                            // or just swap ids.
                            (kernel_x_id, kernel_y_id) = (kernel_y_id, kernel_x_id);
                        }
                        (false, false) => {
                            // Nothing needs to be done
                        }
                    }*/

                    // Now we know that kernel x depends on kernel y or there is no dependence at all
                    // So kernel y must go first
                    let (kernel_y, kernel_x) = if kernel_x_id > kernel_y_id {
                        (kernels.remove(kernel_y_id), &mut kernels[kernel_x_id - 1])
                    } else {
                        (kernels.remove(kernel_y_id), &mut kernels[kernel_x_id])
                    };
                    let kernel_y_ops: Vec<VOp> = kernel_y
                        .ops
                        .into_iter()
                        .enumerate()
                        .skip_while(|(i, op)| {
                            matches!(op, VOp::Loop { .. }) && op == &kernel_x.ops[*i]
                        })
                        .map(|(_, op)| op)
                        .collect();
                    kernel_x.ops.extend(kernel_y_ops);
                    kernel_x.ops.push(VOp::Binary { z: nid, x, y, bop });
                    // if kernel is not last, then make it last
                    if kernel_y_id > kernel_x_id {
                        let kernel = kernels.remove(kernel_x_id);
                        kernels.push(kernel);
                    }
                }
            }
        }
        //println!("nid: {nid} to_eval {to_eval:?}");
        if graph.to_eval.contains(&nid) {
            if let Some(kernel) = kernels
                .iter_mut()
                .find(|kernel| kernel.vars().contains(&nid))
            {
                kernel.store(nid, View::contiguous(graph.shape(nid)), graph.dtype(nid));
            } else {
                unreachable!()
            }
        }
        // TODO only if this is not nid in user ids
        if graph.rc(nid) > 1 {
            if let Some(kernel) = kernels
                .iter_mut()
                .find(|kernel| kernel.vars().contains(&nid))
            {
                // if graph.rc(nid) > 1 then just copy that graph if it is not too big graph
                // and if the kernel does not have too big shape.
                // TODO beware of too many copies. We need to make sure that we are not doing
                // the same work twice.
                //if user_leafs.contains(&nid) {
                //kernel.store(nid, View::new(graph.shape(nid)));
                if (kernel.ops.len() > 10
                    && kernel.shape().into_iter().product::<usize>() < 1024 * 1024 * 1024)
                    || kernel.is_reduce()
                    || !kernel.outputs().is_empty()
                {
                    kernel.store(nid, View::contiguous(graph.shape(nid)), graph.dtype(nid));
                    kernels.push(Kernel::load(nid, graph));
                } else {
                    let kernel2 = kernel.clone();
                    kernels.push(kernel2);
                }
            } else {
                unreachable!()
            }
        }
    }
    // Remove unnecessary kernels (these should be only loads for user_rc > 1 kernels)
    kernels.retain(|kernel| !kernel.outputs().is_empty());
    // Remove unnecessary stores not for tensors moved across kernels
    // and not in to_eval that were inserted for rc > 1, but ops got merged,
    // and these stores were not used.
    let mut necessary_stores = graph.to_eval.clone();
    for kernel in &kernels {
        necessary_stores.extend(kernel.inputs().iter());
    }
    for kernel in &mut kernels {
        let mut i = 0;
        while i < kernel.ops.len() {
            if let VOp::Store { z, .. } = kernel.ops[i] {
                if !necessary_stores.contains(&z) {
                    kernel.ops.remove(i);
                }
            }
            i += 1;
        }
    }
    kernels
}

fn shape_to_loops(shape: &[usize]) -> Vec<VOp> {
    let mut res = Vec::with_capacity(20);
    for (axis, dimension) in shape.iter().copied().enumerate() {
        res.push(VOp::Loop {
            axis,
            len: dimension,
        });
    }
    res
}

// Checks if kernel_y needs to be evaluated before kernel_x
// This takes 96% of the execution time of the whole generate_kernels function.
// We need to either improve the speed of this function or do it in some other way.
/*fn depends_on(kernel_x_id: usize, kernel_y_id: usize, kernels: &[Kernel]) -> bool {
    let _t = crate::Timer::new("depends_on");
    let mut kernel_x_inputs = kernels[kernel_x_id].inputs();
    let kernel_y_outputs = &kernels[kernel_y_id].outputs();
    //println!("y outputs: {kernel_y_outputs:?}");
    //for kernel in kernels { kernel.debug(); }
    let mut visited = BTreeSet::new();
    while let Some(x) = kernel_x_inputs.pop_last() {
        if visited.insert(x) {
            if kernel_y_outputs.contains(&x) {
                return true;
            } else {
                'a: for kernel in kernels.iter().rev() {
                    if kernel.outputs().contains(&x) {
                        kernel_x_inputs.extend(kernel.inputs());
                        //println!("x inputs: {kernel_x_inputs:?}");
                        break 'a;
                    }
                }
            }
        }
    }
    false
}*/

fn get_kernel<'a>(x: TensorId, kernels: &'a mut Vec<Kernel>, graph: &Graph) -> &'a mut Kernel {
    //let _t = crate::Timer::new("get_kernel");
    // First if there is kernel which stores x, then just return new load kernel
    if kernels
        .iter()
        .rev()
        .any(|kernel| kernel.outputs().contains(&x))
    {
        kernels.push(Kernel::load(x, graph));
        return kernels.last_mut().unwrap();
    }
    kernels
        .iter_mut()
        .filter(|kernel| kernel.vars().contains(&x))
        .min_by_key(|kernel| kernel.ops.len())
        .unwrap()
}

#[allow(clippy::similar_names)]
fn print_perf(flop: u128, bytes_read: u128, bytes_written: u128, nanos: u128) {
    const fn value_unit(x: u128) -> (u128, &'static str) {
        match x {
            0..1000 => (x / 100, ""),
            1_000..1_000_000 => (x / 10, "k"),
            1_000_000..1_000_000_000 => (x / 10_000, "M"),
            1_000_000_000..1_000_000_000_000 => (x / 10_000_000, "G"),
            1_000_000_000_000..1_000_000_000_000_000 => (x / 10_000_000_000, "T"),
            1_000_000_000_000_000..1_000_000_000_000_000_000 => (x / 10_000_000_000_000, "P"),
            1_000_000_000_000_000_000.. => (x / 10_000_000_000_000_000, "E"),
        }
    }

    let (f, f_u) = value_unit(flop);
    let (br, br_u) = value_unit(bytes_read);
    let (bw, bw_u) = value_unit(bytes_written);
    let (t_d, t_u) = match nanos {
        0..1_000 => (1 / 10, "ns"),
        1_000..1_000_000 => (100, "μs"),
        1_000_000..1_000_000_000 => (100_000, "ms"),
        1_000_000_000..1_000_000_000_000 => (100_000_000, "s"),
        1_000_000_000_000.. => (6_000_000_000, "min"),
    };

    let (fs, f_us) = value_unit(flop * 1_000_000_000 / nanos);
    let (brs, br_us) = value_unit(bytes_read * 1_000_000_000 / nanos);
    let (bws, bw_us) = value_unit(bytes_written * 1_000_000_000 / nanos);

    println!("        {}.{} {t_u} ~ {}.{} {f_us}FLOP/s, {}.{} {br_us}B/s read, {}.{} {bw_us}B/s write, {f} {f_u}FLOP, {br} {br_u}B read, {bw} {bw_u}B write",
        nanos/(t_d*10),
        (nanos/t_d)%10,
        fs/100,
        fs%100,
        brs/100,
        brs%100,
        bws/100,
        bws%100,
    );
}

#[cfg(feature = "disk_cache")]
fn store_optimizer_cache(
    optimizer_cache: &BTreeMap<(Kernel, super::backend::DeviceInfo), KernelOptimizer>,
    config_dir: Option<std::path::PathBuf>,
    debug_sched: bool,
) {
    use std::io::Write;
    if let Some(mut path) = config_dir.clone() {
        path.push("cached_kernels");
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&bitcode::encode(optimizer_cache)).unwrap();
    } else if debug_sched {
        println!("Zyx config path was not found. Searched kernels won't be cached to disk.");
    }
}
