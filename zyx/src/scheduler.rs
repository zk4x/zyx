//! Scheduler schedules kernels to hardware devices with respect to memory allocation limits.

use crate::{
    backend::{BufferId, Device, DeviceId, DeviceInfo, MemoryPool, MemoryPoolId}, graph::Graph, index_map::Id, ir::IRKernel, kernel::Kernel, optimizer::{KernelOptimization, KernelOptimizer}, tensor::TensorId, view::View, ZyxError
};
use std::collections::{BTreeMap, BTreeSet};

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

    #[allow(clippy::cognitive_complexity)]
    pub(super) fn compile_graph(mut graph: Graph, memory_pools: &mut [MemoryPool], devices: &mut [Device], tensor_buffer_map: &BTreeMap<(TensorId, View), BufferId>,
        optimizer_cache: &mut BTreeMap<(Kernel, DeviceInfo), KernelOptimizer>,
        kernel_cache: &mut BTreeMap<IRKernel, (DeviceId, Id)>,
        search_iterations: usize,
        config_dir: Option<&std::path::Path>,
        debug_perf: bool,
        debug_sched: bool,
        debug_ir: bool,
        debug_asm: bool,
        ) -> Result<CompiledGraph, ZyxError> {
        // get order of nodes and graph characteristics, some basic optimizations are node reordering are applied
        let (order, flop, bytes_read, bytes_written) = graph.execution_order();
        // create vop representation
        let mut kernels: Vec<Kernel> = crate::generator::generate_kernels(&graph, &order, debug_sched);
        //println!("{:?}", &*crate::ET.lock());
        //panic!();
        //println!("{:?}", self.tensor_buffer_map);
        if debug_sched {
            //for kernel in &kernels { kernel.debug(); }
            println!("Scheduler generated {} kernels", kernels.len());
        }
        //panic!("Done");
        let mut sched_graph: Vec<SchedulerOp> = Vec::new();
        // Simulated device occupation. How many kernels are running on each device, if more than x, finish first one before launching next one
        let mut device_program_map: BTreeMap<DeviceId, Vec<Id>> = (0..devices.len() as u32)
            .map(|device_id| (device_id, Vec::new()))
            .collect();
        // Simulated tensor buffer map
        let mut tensor_buffer_map: BTreeMap<(TensorId, View), MemoryPoolId> = tensor_buffer_map
            .iter()
            .map(|(x, &BufferId { memory_pool_id, .. })| (x.clone(), memory_pool_id))
            .collect();
        let mut temp_tensors: BTreeSet<TensorId> = BTreeSet::new();

        let kernels_len = kernels.len();
        for kid in 0..kernels_len {
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
                let min_devs: Vec<DeviceId> = device_program_map
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
                        devices[*x.0 as usize]
                            .compute()
                            .cmp(&devices[*y.0 as usize].compute())
                    })
                    .unwrap()
                    .0;

                let memory_pool_id = devices[device_id as usize].memory_pool_id();
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
                //kernel.debug();
                //println!("{tensor_buffer_map:?}");

                // Move necessary inputs to memory pool associated with this device
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
                if debug_sched {
                    kernel.debug();
                }
                // Disk cached search, works across devices and platforms
                let optimization = search_kernel_optimization(kernel, device_id, &graph, memory_pools, devices,
                    optimizer_cache, search_iterations, config_dir, debug_perf, debug_sched, debug_asm)?;
                if debug_sched {
                    println!("Kernel {kid}/{kernels_len} using {optimization}");
                }
                // Compile and cache program
                let (program_id, args) = compile_cached(kernel, &optimization, device_id, &graph, kernel_cache, devices, debug_asm, debug_ir)?;
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

        if debug_sched {
            for sched_op in &sched_graph {
                match sched_op {
                    SchedulerOp::Launch(program) => {
                        println!(
                            "Launch kernel {} on device {} with args {:?}",
                            program.program_id,
                            program.device_id,
                            program.args.iter().map(|a| a.0).collect::<Vec<u32>>()
                        );
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

    pub(super) fn launch_graph(graph: &Graph, compiled_graph: &CompiledGraph, memory_pools: &mut [MemoryPool], devices: &mut [Device], tensor_buffer_map: &mut BTreeMap<(TensorId, View), BufferId>, debug_perf: bool) -> Result<(), ZyxError> {
        let mut queues: BTreeMap<VProgram, usize> = BTreeMap::new();
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
                            tensor_buffer_map[&(arg.0, arg.1.clone())].buffer_id
                        })
                        .collect();
                    let device = &mut devices[vprogram.device_id as usize];
                    let mpid = device.memory_pool_id();
                    queues.insert(
                        vprogram.clone(),
                        device.launch(
                            vprogram.program_id,
                            &mut memory_pools[mpid as usize],
                            &buffer_ids,
                        )?,
                    );
                }
                SchedulerOp::Finish(program) => {
                    for (vprogram, queue) in &mut queues {
                        if program == vprogram {
                            devices[program.device_id as usize].sync(*queue)?;
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
                    } = tensor_buffer_map[&(*tensor_id, view.clone())];
                    let (src_mps, dst_mps) = memory_pools.split_at_mut(*dst as usize);
                    let src_mp = &mut src_mps[memory_pool_id as usize];
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
                    let buffer_id = memory_pools[*memory_pool_id as usize].allocate(*bytes)?;
                    tensor_buffer_map.insert(
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
                    }) = tensor_buffer_map.get(key)
                    {
                        memory_pools[*memory_pool_id as usize].deallocate(*buffer_id)?;
                        tensor_buffer_map.remove(key).unwrap();
                    }
                }
            }
        }
        if debug_perf {
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
        kernel: &Kernel,
        device_id: DeviceId,
        graph: &Graph,
        memory_pools: &mut [MemoryPool],
        devices: &mut [Device],
        optimizer_cache: &mut BTreeMap<(Kernel, DeviceInfo), KernelOptimizer>,
        search_iterations: usize,
        config_dir: Option<&std::path::Path>,
        debug_perf: bool,
        debug_sched: bool,
        debug_asm: bool
    ) -> Result<KernelOptimization, ZyxError> {
        let dev_info = devices[device_id as usize].info().clone();
        let cache_key = (kernel.clone(), dev_info.clone());
        if let Some(KernelOptimizer::Optimized(optimizations, _)) =
            optimizer_cache.get(&cache_key)
        {
            Ok(optimizations.clone())
        } else {
            // allocate space for inputs and outputs that are not allocated for this kernel
            let mut allocated_temps = Vec::new();
            let mpid = devices[device_id as usize].memory_pool_id();
            let mut temp_ids: BTreeSet<TensorId> = kernel.inputs();
            temp_ids.extend(&kernel.outputs());
            for tid in temp_ids {
                let buffer_id = memory_pools[mpid as usize].allocate(
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
            let optimizer = optimizer_cache
                .entry(cache_key.clone())
                .or_insert_with(|| kernel.new_optimizer(&dev_info));
            let rem_opts = optimizer.remaining();
            if debug_sched {
                println!(
                    "Searching over {search_iterations} out of {rem_opts} remaining optimizations.",
                );
            }
            #[cfg(feature = "disk_cache")]
            let mut timer = std::time::Instant::now();
            for i in 0..search_iterations {
                let optimizer = optimizer_cache.get_mut(&cache_key).unwrap();
                let Some(optimization_id) = optimizer.next() else {
                    if debug_sched {
                        println!(
                            "All optimizations were tried and fastest kernel has been selected."
                        );
                    }
                    #[cfg(feature = "disk_cache")]
                    store_optimizer_cache(
                        &optimizer_cache,
                        config_dir,
                        debug_sched,
                    );
                    break;
                };
                if debug_sched {
                    println!(
                        "{:>6}/{} {}",
                        i + 1,
                        search_iterations.min(rem_opts),
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
                let program_id = devices[device_id as usize].compile(&ir_kernel, debug_asm)?;
                // Launch kernel and measure it's performance
                let begin = std::time::Instant::now();
                let queue_id = match devices[device_id as usize].launch(
                    program_id,
                    &mut memory_pools[mpid as usize],
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
                if let Err(e) = devices[device_id as usize].sync(queue_id) {
                    optimizer.set_exec_time(optimization_id, u128::MAX);
                    if debug_sched {
                        println!("Could not sync, {e:?}, skipping");
                    }
                    continue;
                };
                let exec_time = begin.elapsed().as_nanos();
                let _ = devices[device_id as usize].release_program(program_id);
                if let Some((f, mr, mw)) = flop_mem_rw {
                    print_perf(f, mr, mw, exec_time);
                }
                optimizer.set_exec_time(optimization_id, exec_time);
                #[cfg(feature = "disk_cache")]
                if timer.elapsed().as_secs() > 60 {
                    store_optimizer_cache(
                        &optimizer_cache,
                        config_dir.clone(),
                        debug_sched,
                    );
                    timer = std::time::Instant::now();
                }
            }
            #[cfg(feature = "disk_cache")]
            store_optimizer_cache(
                &optimizer_cache,
                config_dir.clone(),
                debug_sched,
            );
            if debug_sched {
                println!("Optimization has been finished.\n");
            }
            // Deallocate inputs and outputs that are used only for beam search
            for buffer_id in allocated_temps {
                memory_pools[mpid as usize].deallocate(buffer_id)?;
            }
            Ok(optimizer_cache.get(&cache_key).unwrap().best().clone())
        }
    }

    // Compiles kernel using given optimizations
    #[allow(clippy::type_complexity)]
    pub(super) fn compile_cached(
        kernel: &Kernel,
        optimization: &KernelOptimization,
        device_id: DeviceId,
        graph: &Graph,
        kernel_cache: &mut BTreeMap<IRKernel, (DeviceId, Id)>,
        devices: &mut [Device],
        debug_asm: bool,
        debug_ir: bool,
    ) -> Result<(Id, Vec<(Id, View, bool)>), ZyxError> {
        let mut program_id = None;
        let optimized_kernel = kernel.optimize(optimization);
        //println!("Compiling kernel with shape {:?}", optimized_kernel.shape);
        //optimized_kernel.debug();
        let (ir_kernel, ir_args) = IRKernel::new(&optimized_kernel.ops);
        if let Some((dev_id, prog_id)) = kernel_cache.get(&ir_kernel) {
            if *dev_id == device_id {
                program_id = Some(*prog_id);
            }
        }
        if program_id.is_none() {
            if debug_ir {
                ir_kernel.debug();
            }
            program_id = Some(devices[device_id as usize].compile(&ir_kernel, debug_asm)?);
            kernel_cache
                .insert(ir_kernel, (device_id, program_id.unwrap()));
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

#[cfg(feature = "disk_cache")]
fn store_optimizer_cache(
    optimizer_cache: &BTreeMap<(Kernel, super::backend::DeviceInfo), KernelOptimizer>,
    config_dir: Option<&std::path::Path>,
    debug_sched: bool,
) {
    use std::io::Write;
    if let Some(path) = config_dir {
        let mut path = path.to_path_buf();
        path.push("cached_kernels");
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(&bitcode::encode(optimizer_cache)).unwrap();
    } else if debug_sched {
        println!("Zyx config path was not found. Searched kernels won't be cached to disk.");
    }
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
