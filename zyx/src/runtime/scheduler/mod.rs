pub(super) use kernel::Kernel;
use vop::MOp;
pub(super) use vop::VOp;

use crate::{
    runtime::{graph::Graph, ir::Scope, node::Node, view::View, Runtime, ZyxError},
    tensor::TensorId,
};
use std::collections::{BTreeMap, BTreeSet};
use super::{backend::{cuda::CUDAEvent, hip::HIPEvent, opencl::OpenCLEvent}, BufferId, Device, DeviceId, MemoryPool, MemoryPoolId};

mod kernel;
mod optimizer;
mod vop;

#[derive(Debug)]
pub(super) struct CompiledGraph {
    sched_graph: Vec<SchedulerOp>,
    flop: u128,
    bytes_read: u128,
    bytes_written: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct VProgram {
    device_id: DeviceId,
    program_id: usize,
    args: Vec<(TensorId, View, bool)>,
}

// TODO this function could take &mut Runtime
impl Runtime {
    pub(super) fn compile_graph(
        &mut self,
        #[allow(unused_mut)] mut graph: Graph,
        to_eval: &BTreeSet<TensorId>,
    ) -> Result<CompiledGraph, ZyxError> {
        // get order of nodes and graph characteristics, some basic optimizations are node reordering are applied
        let (order, flop, bytes_read, bytes_written) = graph.execution_order(to_eval);
        // create vop representation
        let mut kernels: Vec<Kernel> = generate_kernels(&graph, &order, &to_eval);
        let mut sched_graph: Vec<SchedulerOp> = Vec::new();
        // Simulated device occupation. How many kernels are running on each device, if more than 5, finish first one before launching next one
        let mut device_program_map: BTreeMap<DeviceId, Vec<usize>> = (0..self.devices.len())
            .map(|device_id| (device_id, Vec::new()))
            .collect();
        // Simulated tensor buffer map
        let mut tensor_buffer_map: BTreeMap<(TensorId, View), MemoryPoolId> = self
            .graph
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
                        if !read_only && kernel.inputs.contains(&arg) {
                            device_program_map
                                .get_mut(&vprogram.device_id)
                                .unwrap()
                                .retain(|pid| *pid != vprogram.program_id);
                            program_wait_list.push((vprogram.device_id, vprogram.program_id));
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
                if min_programs > 5 {
                    // Finish some program before proceding
                    todo!()
                }
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

                // Prints unoptimized kernel
                if self.debug_sched() { kernel.debug(); }

                // TODO rerun this function and the kernel multiple times and cache optimizations to the disk
                let _optimizations = kernel.optimize(self.devices[device_id].info());


                let memory_pool_id = self.devices[device_id].memory_pool_id();
                // Allocate memory for outputs
                for output in &kernel.outputs {
                    let key = (*output, View::new(graph.shape(*output)));
                    if !tensor_buffer_map.contains_key(&key) {
                        let shape = graph.shape(*output);
                        let view = View::new(shape);
                        sched_graph.push(SchedulerOp::Allocate {
                            tensor_id: *output,
                            memory_pool_id,
                            bytes: shape.iter().product::<usize>()
                                * graph.dtype(*output).byte_size(),
                            view: view.clone(),
                        });
                        temp_tensors.insert(*output);
                        tensor_buffer_map.insert((*output, view), memory_pool_id);
                    }
                }
                // Move necessary inputs to memory pool associated with this device
                for input in &kernel.inputs {
                    let view = View::new(graph.shape(*input));
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
                for (device_id, program_id) in program_wait_list {
                    sched_graph.push(SchedulerOp::Finish { device_id, program_id });
                }
                let (ir_kernel, ir_args) = kernel.to_ir(&graph);
                let mut program_id = None;
                if let Some((dev_id, prog_id) ) = self.ir_kernel_cache.get(&ir_kernel) {
                    if *dev_id == device_id {
                        program_id = Some(*prog_id);
                    }
                }
                if program_id.is_none() {
                    if self.debug_ir() { ir_kernel.debug(); }
                    let debug_asm = self.debug_asm();
                    program_id = Some(match &mut self.devices[device_id] {
                        Device::CUDA {
                            device, programs, ..
                        } => {
                            programs.push(device.compile(&ir_kernel, debug_asm)?);
                            programs.len() - 1
                        }
                        Device::HIP {
                            device, programs, ..
                        } => {
                            programs.push(device.compile(&ir_kernel, debug_asm)?);
                            programs.len() - 1
                        }
                        Device::OpenCL {
                            device, programs, ..
                        } => {
                            programs.push(device.compile(&ir_kernel, debug_asm)?);
                            programs.len() - 1
                        }
                    });
                    self.ir_kernel_cache.insert(
                        ir_kernel,
                        (device_id, program_id.unwrap())
                    );
                }
                let program_id = program_id.unwrap();
                // Since it is not sharded, sharding view is contiguous
                device_program_map
                    .get_mut(&device_id)
                    .unwrap()
                    .push(program_id);
                sched_graph.push(SchedulerOp::Launch(VProgram {
                    device_id,
                    program_id,
                    args: ir_args.into_iter().map(|(arg, read_only)| (arg, View::new(graph.shape(arg)), read_only)).collect(),
                }));
                // Deallocate kernel inputs that will not be used by other kernels
                let mut unneeded_tensors = temp_tensors.clone();
                if kid + 1 < kernels.len() {
                    for kernel in &kernels[kid + 1..] {
                        for input in &kernel.inputs {
                            unneeded_tensors.remove(input);
                        }
                    }
                }
                for tensor in to_eval {
                    unneeded_tensors.remove(tensor);
                }
                //println!("Unneeded tensors: {unneeded_tensors:?}, kernel inputs {:?} tensor_buffer_map {tensor_buffer_map:?}", &kernels[kid].inputs);
                for tensor_id in unneeded_tensors {
                    let view = View::new(graph.shape(tensor_id));
                    let dtype = graph.dtype(tensor_id);
                    let Some(memory_pool_id) = tensor_buffer_map.get(&(tensor_id, view.clone()))
                    else {
                        panic!()
                    };
                    sched_graph.push(SchedulerOp::Deallocate {
                        tensor_id,
                        memory_pool_id: *memory_pool_id,
                        bytes: view.numel() * dtype.byte_size(),
                        view,
                    });
                }
            }
        }
        for (device_id, programs) in device_program_map.iter_mut() {
            for program_id in &mut *programs {
                sched_graph.push(SchedulerOp::Finish {
                    device_id: *device_id,
                    program_id: *program_id,
                });
            }
            programs.clear();
        }

        if self.debug_sched() {
            for sched_op in &sched_graph {
                match sched_op {
                    SchedulerOp::Launch(program_id) => {
                        println!("Launch kernel {}", self.devices[program_id.device_id])
                    }
                    SchedulerOp::Finish { device_id, program_id }=> println!("Finish kernel {program_id} on device {device_id}"),
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
                        println!("Allocate tensor {tensor_id} on memory pool {memory_pool_id:?} with size {bytes:?} B")
                    }
                    SchedulerOp::Deallocate {
                        tensor_id,
                        memory_pool_id,
                        bytes,
                        view: _,
                    } => {
                        println!("Deallocate tensor {tensor_id} on {memory_pool_id:?}, {bytes}")
                    }
                }
            }
        }

        return Ok(CompiledGraph {
            sched_graph,
            flop,
            bytes_read,
            bytes_written,
        });
    }

    pub(super) fn launch_graph(&mut self, graph: &Graph) -> Result<(), ZyxError> {
        let mut events: BTreeMap<VProgram, Event> = BTreeMap::new();
        let compiled_graph = &self.compiled_graph_cache[graph];
        //println!("Launching compiled graph: {compiled_graph:?}");
        let begin = std::time::Instant::now();
        for sched_op in &compiled_graph.sched_graph {
            match sched_op {
                SchedulerOp::Launch(vprogram) => {
                    // Same program can launch with different args. Thus in program map we also need args.
                    //println!("Launch {program_id:?} with args:");
                    let args: Vec<usize> = vprogram.args.iter()
                                        .map(|arg| {
                                            //println!("Arg {} {}", arg.0, arg.1);
                                            self.graph.tensor_buffer_map[&(arg.0, arg.1.clone())]
                                                .buffer_id
                                        })
                                        .collect();
                    events.insert(vprogram.clone(), match &mut self.devices[vprogram.device_id] {
                        Device::CUDA {
                            device: _,
                            memory_pool_id,
                            programs,
                        } => {
                            let MemoryPool::CUDA { buffers, .. } =
                                &mut self.memory_pools[*memory_pool_id]
                            else {
                                panic!()
                            };
                            Event::CUDA(programs[vprogram.program_id].launch(buffers, &args)?)
                        }
                        Device::HIP {
                            device: _,
                            memory_pool_id,
                            programs,
                        } => {
                            let MemoryPool::HIP { buffers, .. } =
                                &mut self.memory_pools[*memory_pool_id]
                            else {
                                panic!()
                            };
                            Event::HIP(programs[vprogram.program_id].launch(buffers, &args)?)
                        }
                        Device::OpenCL {
                            device: _,
                            memory_pool_id,
                            programs,
                        } => {
                            let MemoryPool::OpenCL { buffers, .. } =
                                &mut self.memory_pools[*memory_pool_id]
                            else {
                                panic!()
                            };
                            Event::OpenCL(programs[vprogram.program_id].launch(buffers, &args)?)
                        }
                    });
                }
                SchedulerOp::Finish { device_id: dev_id, program_id: prog_id } => {
                    // Dropping event finishes it
                    events.retain(|VProgram { device_id, program_id, .. }, _| *device_id != *dev_id || *program_id != *prog_id);
                }
                SchedulerOp::Move {
                    tensor_id,
                    view,
                    dst,
                } => {
                    let bytes = view.numel() * graph.dtype(*tensor_id).byte_size();
                    let BufferId {
                        memory_pool_id,
                        buffer_id: src_bid,
                    } = self.graph.tensor_buffer_map[&(*tensor_id, view.clone())];
                    let (src_mps, dst_mps) = self.memory_pools.split_at_mut(*dst);

                    macro_rules! cross_backend {
                        ($sm: expr, $sb: expr, $dm: expr, $db: expr) => {{
                            let dst_bid = $db.push($dm.allocate(bytes)?);
                            let mut data: Vec<u8> = Vec::with_capacity(bytes);
                            unsafe { data.set_len(bytes) };
                            $sm.pool_to_host(&$sb[src_bid], &mut data)?;
                            $sm.deallocate($sb.remove(src_bid).unwrap())?;
                            $dm.host_to_pool(&data, &$db[dst_bid])?;
                            self.graph.tensor_buffer_map.insert(
                                (*tensor_id, view.clone()),
                                BufferId {
                                    memory_pool_id: *dst,
                                    buffer_id: dst_bid,
                                },
                            );
                        }};
                    }

                    macro_rules! within_backend {
                        ($sm: expr, $sb: expr, $dm: expr, $db: expr) => {{
                            let dst_bid = $db.push($dm.allocate(bytes)?);
                            $dm.pool_to_pool(&$sb[src_bid], &$db[dst_bid])?;
                            $sm.deallocate($sb.remove(src_bid).unwrap())?;
                            self.graph.tensor_buffer_map.insert(
                                (*tensor_id, view.clone()),
                                BufferId {
                                    memory_pool_id: *dst,
                                    buffer_id: dst_bid,
                                },
                            );
                        }};
                    }

                    match (&mut src_mps[memory_pool_id], &mut dst_mps[0]) {
                        (
                            MemoryPool::CUDA {
                                memory_pool: sm,
                                buffers: sb,
                            },
                            MemoryPool::CUDA {
                                memory_pool: dm,
                                buffers: db,
                            },
                        ) => {
                            within_backend!(sm, sb, dm, db)
                        }
                        (
                            MemoryPool::CUDA {
                                memory_pool: sm,
                                buffers: sb,
                            },
                            MemoryPool::HIP {
                                memory_pool: dm,
                                buffers: db,
                            },
                        ) => {
                            cross_backend!(sm, sb, dm, db)
                        }
                        (
                            MemoryPool::CUDA {
                                memory_pool: sm,
                                buffers: sb,
                            },
                            MemoryPool::OpenCL {
                                memory_pool: dm,
                                buffers: db,
                            },
                        ) => {
                            cross_backend!(sm, sb, dm, db)
                        }
                        (
                            MemoryPool::HIP {
                                memory_pool: sm,
                                buffers: sb,
                            },
                            MemoryPool::CUDA {
                                memory_pool: dm,
                                buffers: db,
                            },
                        ) => {
                            cross_backend!(sm, sb, dm, db)
                        }
                        (
                            MemoryPool::HIP {
                                memory_pool: sm,
                                buffers: sb,
                            },
                            MemoryPool::HIP {
                                memory_pool: dm,
                                buffers: db,
                            },
                        ) => {
                            within_backend!(sm, sb, dm, db)
                        }
                        (
                            MemoryPool::HIP {
                                memory_pool: sm,
                                buffers: sb,
                            },
                            MemoryPool::OpenCL {
                                memory_pool: dm,
                                buffers: db,
                            },
                        ) => {
                            cross_backend!(sm, sb, dm, db)
                        }
                        (
                            MemoryPool::OpenCL {
                                memory_pool: sm,
                                buffers: sb,
                            },
                            MemoryPool::CUDA {
                                memory_pool: dm,
                                buffers: db,
                            },
                        ) => {
                            cross_backend!(sm, sb, dm, db)
                        }
                        (
                            MemoryPool::OpenCL {
                                memory_pool: sm,
                                buffers: sb,
                            },
                            MemoryPool::HIP {
                                memory_pool: dm,
                                buffers: db,
                            },
                        ) => {
                            cross_backend!(sm, sb, dm, db)
                        }
                        (
                            MemoryPool::OpenCL {
                                memory_pool: sm,
                                buffers: sb,
                            },
                            MemoryPool::OpenCL {
                                memory_pool: dm,
                                buffers: db,
                            },
                        ) => {
                            within_backend!(sm, sb, dm, db)
                        }
                    }
                }
                SchedulerOp::Allocate {
                    tensor_id,
                    memory_pool_id,
                    bytes,
                    view,
                } => match &mut self.memory_pools[*memory_pool_id] {
                    MemoryPool::CUDA {
                        memory_pool,
                        buffers,
                    } => {
                        let buffer = memory_pool.allocate(*bytes)?;
                        let buffer_id = buffers.push(buffer);
                        self.graph.tensor_buffer_map.insert(
                            (*tensor_id, view.clone()),
                            BufferId {
                                memory_pool_id: *memory_pool_id,
                                buffer_id,
                            },
                        );
                    }
                    MemoryPool::HIP {
                        memory_pool,
                        buffers,
                    } => {
                        let buffer = memory_pool.allocate(*bytes)?;
                        let buffer_id = buffers.push(buffer);
                        self.graph.tensor_buffer_map.insert(
                            (*tensor_id, view.clone()),
                            BufferId {
                                memory_pool_id: *memory_pool_id,
                                buffer_id,
                            },
                        );
                    }
                    MemoryPool::OpenCL {
                        memory_pool,
                        buffers,
                    } => {
                        let buffer = memory_pool.allocate(*bytes)?;
                        let buffer_id = buffers.push(buffer);
                        self.graph.tensor_buffer_map.insert(
                            (*tensor_id, view.clone()),
                            BufferId {
                                memory_pool_id: *memory_pool_id,
                                buffer_id,
                            },
                        );
                    }
                },
                SchedulerOp::Deallocate {
                    tensor_id,
                    memory_pool_id,
                    bytes,
                    view,
                } => match &mut self.memory_pools[*memory_pool_id] {
                    // TODO probably just add macro for this, 'cause rust can't do this without macro
                    MemoryPool::CUDA {
                        memory_pool,
                        buffers,
                    } => {
                        let _ = bytes;
                        let key = &(*tensor_id, view.clone());
                        if let Some(BufferId {
                            memory_pool_id: buf_mpid,
                            ..
                        }) = self.graph.tensor_buffer_map.get(key)
                        {
                            if buf_mpid == memory_pool_id {
                                let BufferId { buffer_id, .. } =
                                    self.graph.tensor_buffer_map.remove(key).unwrap();
                                memory_pool.deallocate(buffers.remove(buffer_id).unwrap())?;
                            }
                        }
                    }
                    MemoryPool::HIP {
                        memory_pool,
                        buffers,
                    } => {
                        let _ = bytes;
                        let key = &(*tensor_id, view.clone());
                        if let Some(BufferId {
                            memory_pool_id: buf_mpid,
                            ..
                        }) = self.graph.tensor_buffer_map.get(key)
                        {
                            if buf_mpid == memory_pool_id {
                                let BufferId { buffer_id, .. } =
                                    self.graph.tensor_buffer_map.remove(key).unwrap();
                                memory_pool.deallocate(buffers.remove(buffer_id).unwrap())?;
                            }
                        }
                    }
                    MemoryPool::OpenCL {
                        memory_pool,
                        buffers,
                    } => {
                        let _ = bytes;
                        let key = &(*tensor_id, view.clone());
                        if let Some(BufferId {
                            memory_pool_id: buf_mpid,
                            ..
                        }) = self.graph.tensor_buffer_map.get(key)
                        {
                            if buf_mpid == memory_pool_id {
                                let BufferId { buffer_id, .. } =
                                    self.graph.tensor_buffer_map.remove(key).unwrap();
                                memory_pool.deallocate(buffers.remove(buffer_id).unwrap())?;
                            }
                        }
                    }
                },
            }
        }
        if self.debug_perf() {
            let duration = begin.elapsed();
            let nanos = duration.as_nanos();

            fn value_unit(x: u128) -> (u128, &'static str) {
                match x {
                    0..1000 => (x, ""),
                    1000..1000000 => (x / 1000, "k"),
                    1000_000..1000000000 => (x / 1000_000, "M"),
                    1000_000_000..1000_000_000_000 => (x / 1000_000_000, "G"),
                    1000_000_000_000..1000_000_000_000_000 => (x / 1000_000_000_000, "T"),
                    1000_000_000_000_000..1000_000_000_000_000_000 => {
                        (x / 1000_000_000_000_000, "P")
                    }
                    1000_000_000_000_000_000.. => (x / 1000_000_000_000_000_000, "E"),
                }
            }

            let (f, f_u) = value_unit(compiled_graph.flop);
            let (br, br_u) = value_unit(compiled_graph.bytes_read);
            let (bw, bw_u) = value_unit(compiled_graph.bytes_written);
            let (t_d, t_u) = match nanos {
                0..1000 => (1, "ns"),
                1000..1000_000 => (1000, "μs"),
                1000_000..1000_000_000 => (1000_000, "ms"),
                1000_000_000..1000_000_000_000 => (1000_000_000, "s"),
                1000_000_000_000.. => (60_000_000_000, "min"),
            };

            let (fs, f_us) = value_unit(compiled_graph.flop * 1000_000_000 / nanos);
            let (brs, br_us) = value_unit(compiled_graph.bytes_read * 1000_000_000 / nanos);
            let (bws, bw_us) = value_unit(compiled_graph.bytes_written * 1000_000_000 / nanos);

            println!("Graph {f} {f_u}FLOP, {br} {br_u}B read, {bw} {bw_u}B write, took {} {t_u} ~ {fs} {f_us}FLOP/s, {brs} {br_us}B/s read, {bws} {bw_us}B/s write.", nanos/t_d);
        }
        Ok(())
    }
}

// Just read only, when event is dropped, it automatically finishes associated program
enum Event {
    #[allow(unused)]
    CUDA(CUDAEvent),
    #[allow(unused)]
    HIP(HIPEvent),
    #[allow(unused)]
    OpenCL(OpenCLEvent),
}

#[derive(Debug)]
enum SchedulerOp {
    // Async launch kernel on device
    Launch(VProgram),
    // Block for kernel to finish execution
    Finish {
        device_id: DeviceId,
        program_id: usize,
    },
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
        memory_pool_id: MemoryPoolId,
        bytes: usize,
        view: View,
    },
}

fn generate_kernels(
    graph: &Graph,
    order: &[TensorId],
    to_eval: &BTreeSet<TensorId>,
) -> Vec<Kernel> {
    // This function sorts nodes into smallest number of kernels that can be compiled on the device
    // This function defines loops, loads, stores and elementwise ops.
    // The aim is to sort nodes in such a way, that maximum performance is attained.
    // These kernels mostly keep shapes of original nodes.
    // Further optimization is done in optimize kernels function.
    //println!("Eval: {to_eval:?}");
    let mut kernels: Vec<Kernel> = Vec::new();
    for nid in order.iter().copied() {
        let node = &graph[nid];
        println!("ID({nid})x{}: {node:?}, sh: {:?}", graph.rc(nid), graph.shape(nid));
        match node {
            Node::Const { value } => {
                let const_op = VOp::Const {
                    z: nid,
                    value: *value,
                    view: View::new(&[1]),
                };
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.shape == [1]) {
                    kernel.ops.push(const_op);
                    kernel.inputs.insert(nid);
                    kernel.vars.insert(nid);
                } else {
                    let mut ops = shape_to_loops(&[1]);
                    ops.push(const_op);
                    kernels.push(Kernel {
                        shape: vec![1],
                        inputs: BTreeSet::new(),
                        outputs: BTreeSet::new(),
                        vars: BTreeSet::from([nid]),
                        ops,
                    })
                }
            }
            Node::Leaf => {
                let shape = graph.shape(nid);
                if let Some(kernel) = kernels.iter_mut().find(|kernel| &kernel.shape == shape) {
                    kernel.ops.push(VOp::Load {
                        z: nid,
                        zscope: Scope::Register,
                        x: nid,
                        xscope: Scope::Global,
                        view: View::new(shape),
                    });
                    kernel.inputs.insert(nid);
                    kernel.vars.insert(nid);
                } else {
                    kernels.push(Kernel::load(graph, nid));
                }
            }
            Node::Expand { x } => {
                let shape = graph.shape(nid);
                let kernel = get_kernel(*x, &mut kernels, graph);
                // Expand can just add loops
                // Expand means that global buffer is accessed multiple times. Thus we need to add caching (local, register) here.
                // Expand increases axes with dimension of 1 to bigger dimension
                // and sets strides in those axes to 0 for both loads and stores
                //println!("Expanding {kernel:?}");
                assert!(shape.len() >= kernel.shape.len());
                if shape.len() > kernel.shape.len() {
                    let mut dimensions: Vec<usize> = core::iter::repeat(1)
                        .take(shape.len() - kernel.shape.len())
                        .collect();
                    dimensions.push(kernel.shape[0]);
                    kernel.split_axis(0, &dimensions);
                }
                assert_eq!(kernel.shape.len(), shape.len());
                let mut expand_axes = BTreeSet::new();
                for a in 0..kernel.shape.len() {
                    if kernel.shape[a] != shape[a] {
                        assert_eq!(kernel.shape[a], 1);
                        kernel.shape[a] = shape[a];
                        expand_axes.insert(a);
                    }
                }
                // We go over ops in reverse, increasing last loops dimension
                let mut done_expanding = BTreeSet::new();
                for op in kernel.ops.iter_mut().rev() {
                    match op {
                        VOp::Loop { axis, dimension } => {
                            if expand_axes.contains(axis) && done_expanding.insert(*axis) {
                                assert_eq!(*dimension, 1);
                                *dimension = shape[*axis];
                            }
                        }
                        VOp::Load { view, .. } | VOp::Const { view, .. } => {
                            // Done expanding marks which loops are behind us,
                            // so we need to only adjust strides to 0 in axes for those axes that are not behind us yet.
                            for a in expand_axes.difference(&done_expanding) {
                                view.expand(*a, shape[*a]);
                            }
                        }
                        VOp::Store { view, .. } => {
                            // TODO This will do multiple writes to the same index, so this would probably be better solved in different way,
                            // perhaps doing only single write during the whole loop using if condition, but that could also be added
                            // to View in VOp::Store as optimization when converting to IROps
                            for a in expand_axes.difference(&done_expanding) {
                                view.expand(*a, shape[*a]);
                            }
                        }
                        _ => {}
                    }
                }
                kernel.ops.push(VOp::Move {
                    z: nid,
                    x: *x,
                    mop: MOp::Expa,
                });
                kernel.vars.insert(nid);
                kernel.shape = shape.into();
            }
            Node::Permute { x, axes, .. } => {
                // Permute shuffles load and store strides
                // It also changes the dimension of loops
                // and shape of kernel
                // TODO but what if it is permute after reduce?
                let kernel = get_kernel(*x, &mut kernels, graph);
                kernel.permute(&axes);
                kernel.ops.push(VOp::Move {
                    z: nid,
                    x: *x,
                    mop: MOp::Perm,
                });
                kernel.vars.insert(nid);
            }
            Node::Reshape { x } => {
                // If we really want, we can get reshape working with loads and stores
                // simply by using view for loads to have multiple reshapes in single view.
                // But for now it is much simpler to just add new kernel.

                // If reshape comes after reduce, then if it just aplits axes, it can be merged,
                // otherwise we have to create new kernel.

                let shape = graph.shape(nid);
                let kernel = get_kernel(*x, &mut kernels, graph);
                // If this is just a reshape of kernel with only unary ops and contiguous loads
                // and stores, we can remove old loops and replace them with new loops.
                if kernel.ops.iter().all(|op| match op {
                    VOp::Loop { .. }
                    | VOp::Unary { .. }
                    | VOp::Binary { .. }
                    | VOp::Move { .. } => true,
                    VOp::Load { view, .. } | VOp::Store { view, .. } | VOp::Const { view, .. } => {
                        view.is_contiguous()
                    }
                    VOp::Accumulator { .. } | VOp::Reduce { .. } => false,
                }) {
                    // Remove old loops
                    for _ in 0..kernel.shape.len() {
                        kernel.ops.remove(0);
                    }
                    // Put in new loops
                    for op in shape_to_loops(shape).into_iter().rev() {
                        kernel.ops.insert(0, op);
                    }
                    // Change Reshape loads and stores
                    for op in &mut kernel.ops {
                        match op {
                            VOp::Load { view, .. }
                            | VOp::Const { view, .. }
                            | VOp::Store { view, .. } => {
                                *view = View::new(shape);
                            }
                            _ => {}
                        }
                    }
                    kernel.shape = shape.into();
                    kernel.ops.push(VOp::Move {
                        z: nid,
                        x: *x,
                        mop: MOp::Resh,
                    });
                    kernel.vars.insert(nid);
                } else {
                    // TODO we could also merge axes if possible
                    let mut splits = Some(BTreeMap::new());
                    let prev_shape = graph.shape(*x);
                    if prev_shape.len() > shape.len() {
                        splits = None;
                    } else {
                        // Example split
                        //    2, 4,    4,    3
                        // 1, 2, 4, 2, 2, 1, 3
                        let mut dimensions = Vec::new();
                        let mut i = prev_shape.len() - 1;
                        let mut dim = 1;
                        for d in shape.iter().copied().rev() {
                            if i == 0 {
                                dimensions.insert(0, d);
                                continue;
                            }
                            if dim * d > prev_shape[i] {
                                if dim == prev_shape[i] {
                                    if dimensions.len() > 1 {
                                        splits.as_mut().unwrap().insert(i, dimensions);
                                    }
                                    dimensions = vec![d];
                                    dim = d;
                                    i -= 1;
                                } else if dim > prev_shape[i] {
                                    splits = None;
                                    break;
                                }
                            } else {
                                dimensions.insert(0, d);
                                dim *= d;
                            }
                        }
                        if dimensions.len() > 1 {
                            splits.as_mut().unwrap().insert(i, dimensions);
                        }
                    }
                    //println!("Splits: {splits:?}");
                    if let Some(mut splits) = splits {
                        let mut loop_id = kernel.shape.len() - 1;
                        let mut skip_loops = 0;
                        let mut split_ids = Vec::new();
                        for (id, vop) in kernel.ops.iter().enumerate().rev() {
                            match vop {
                                VOp::Reduce { num_axes, .. } => {
                                    skip_loops += num_axes;
                                }
                                VOp::Loop { dimension, .. } => {
                                    if skip_loops > 0 {
                                        skip_loops -= 1;
                                    } else {
                                        if loop_id < splits.len() {
                                            let dimensions = splits[&loop_id].clone();
                                            assert_eq!(*dimension, dimensions.iter().product::<usize>());
                                            split_ids.push(id);
                                        }
                                        if loop_id > 0 {
                                            loop_id -= 1;
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        for op_id in split_ids {
                            let Some((_, dimensions)) = splits.pop_last() else {
                                panic!()
                            };
                            //println!("Splitting at {op_id} to {dimensions:?}");
                            kernel.split_axis(op_id, &dimensions);
                        }
                        // TODO If last axes are unsqueezes with ones, add new loops to the end of the kernel.
                        // All unsqueezes can be adding new loops to the end of the kernel by permuting loops.
                        // However we also need to make sure all code can work with out of order loop ids.

                        kernel.shape = shape.into();
                        kernel.ops.push(VOp::Move {
                            z: nid,
                            x: *x,
                            mop: MOp::Resh,
                        });
                        kernel.vars.insert(nid);
                        //kernel.debug();
                    } else {
                        // else create new kernel after storing results of previous kernel
                        kernel.store(*x, graph);
                        let mut ops = shape_to_loops(shape);
                        ops.push(VOp::Load {
                            z: nid,
                            zscope: Scope::Register,
                            x: *x,
                            xscope: Scope::Global,
                            view: View::new(shape),
                        });
                        kernels.push(Kernel {
                            shape: shape.into(),
                            inputs: BTreeSet::from([*x]),
                            outputs: BTreeSet::new(),
                            vars: BTreeSet::from([nid]),
                            ops,
                        });
                    }
                }
                //println!("\nKernels {kernels:?}\n");
            }
            Node::Pad { x, padding } => {
                let shape = graph.shape(nid);
                // Pad shrinks or expands dimension of axes, but if there is store,
                // then it creates new kernel
                let mut kernel = get_kernel(*x, &mut kernels, graph);
                let axes: BTreeSet<usize> = (shape.len() - padding.len()..shape.len()).collect();
                //println!("Shape: {shape:?}, padding: {padding:?}, axes: {axes:?}");
                let mut padded_loops: BTreeSet<usize> = axes.clone();
                let mut padding_possible = true;
                'ops_loop: for op in kernel.ops.iter_mut().rev() {
                    match op {
                        VOp::Loop { axis, .. } => {
                            if axes.contains(axis) {
                                padded_loops.remove(axis);
                                if padded_loops.is_empty() {
                                    break 'ops_loop;
                                }
                            }
                        }
                        VOp::Store { .. } => {
                            padding_possible = false;
                            break 'ops_loop;
                        }
                        _ => {}
                    }
                }
                //println!("Padding possible: {padding_possible}");
                if !padding_possible {
                    kernel.store(*x, graph);
                    kernels.push(Kernel::load(graph, *x));
                    kernel = kernels.last_mut().unwrap();
                }
                let mut padded_loops: BTreeSet<usize> = axes.clone();
                'ops_loop: for op in kernel.ops.iter_mut().rev() {
                    match op {
                        VOp::Loop { axis, dimension } => {
                            if axes.contains(axis) {
                                *dimension = shape[*axis];
                                padded_loops.remove(axis);
                                if padded_loops.is_empty() {
                                    break 'ops_loop;
                                }
                            }
                        }
                        VOp::Load { view, .. } | VOp::Const { view, .. } => {
                            let n = view.rank();
                            let mut a = n;
                            for (lp, rp) in &padding[shape.len() - n..] {
                                a -= 1;
                                view.pad(a, *lp, *rp);
                            }
                        }
                        _ => {}
                    }
                }
                kernel.shape = shape.into();
                kernel.ops.push(VOp::Move {
                    z: nid,
                    x: *x,
                    mop: MOp::Padd,
                });
                kernel.vars.insert(nid);
            }
            Node::Reduce { x, axes, rop } => {
                let shape = graph.shape(nid);
                let kernel = get_kernel(*x, &mut kernels, graph);
                // Reduce removes loops and adds accumulator before those loops that it removes
                //println!("Axes {axes:?}");
                // Permute the axes such that reduce loops are last
                // and keep the order of axes that are not reduced.
                let permute_axes: Vec<usize> = (0..graph.shape(*x).len())
                    .filter(|a| !axes.contains(a))
                    .chain(axes.iter().copied())
                    .collect();
                //println!("Permute axes in reduce: {permute_axes:?}");
                kernel.permute(&permute_axes);

                // We can also just merge these reduce loops into single loop, since it gets removed
                // from the resulting shape either way, but only if there are no ops between those loops.

                // Add accumulator
                let num_axes = graph.shape(*x).len();
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
                        rop: *rop,
                        view: View::None,
                    },
                );
                // End loops
                kernel.ops.push(VOp::Reduce {
                    num_axes: axes.len(),
                    rop: *rop,
                    z: nid,
                    x: *x,
                });
                kernel.vars.insert(nid);
                kernel.shape = shape.into();

                if kernel.shape == [1] && !matches!(kernel.ops[0], VOp::Loop { .. }) {
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
                kernel.vars.insert(nid);
            }
            Node::Binary { x, y, bop } => {
                // Binary ops may allow us to join two kernels together
                if let Some(id) = kernels
                    .iter_mut()
                    .position(|kernel| kernel.vars.is_superset(&[*x, *y].into()))
                {
                    // If both inputs are in the same kernel
                    let kernel = if kernels[id].shape != graph.shape(*x) {
                        // create new kernel using already predefined stores of both x and y
                        let mut kernel = Kernel::load(graph, *x);
                        kernel.ops.push(VOp::Load {
                            z: *y,
                            zscope: Scope::Register,
                            x: *y,
                            xscope: Scope::Global,
                            view: View::new(graph.shape(*y)),
                        });
                        kernel.vars.insert(*y);
                        kernel.inputs.insert(*y);
                        kernels.push(kernel);
                        kernels.last_mut().unwrap()
                    } else {
                        &mut kernels[id]
                    };
                    kernel.ops.push(VOp::Binary {
                        z: nid,
                        x: *x,
                        y: *y,
                        bop: *bop,
                    });
                    kernel.vars.insert(nid);
                } else if let Some(mut kernel_x_id) =
                    kernels.iter().position(|kernel| kernel.vars.contains(x))
                {
                    if let Some(mut kernel_y_id) =
                        kernels.iter().position(|kernel| kernel.vars.contains(y))
                    {
                        // TODO check that swapping id's never changes order in the binary op itself,
                        // because some binary ops may not be commutative
                        //println!("Both inputs are in different kernels.");
                        // Two separate kernels contain our inputs, so we join them together

                        // We can not join kernels if say kernel x depends on kernel a
                        // and kernel a depends on kernel y. In that case we have to create a new kernel.
                        // However often we can reorder kernels if kernel a does not depend on kernel y,
                        // just put kernel a before kernel x and kernel y and we can join it normally.
                        match (
                            depends_on(kernel_x_id, kernel_y_id, &kernels),
                            depends_on(kernel_y_id, kernel_x_id, &kernels),
                        ) {
                            (true, true) => {
                                // This should not be possible
                                panic!()
                            }
                            (true, false) => {
                                // This is ok, nothing needs to be done
                            }
                            (false, true) => {
                                // Here we need to do some reordering,
                                // or just swap ids.
                                (kernel_x_id, kernel_y_id) = (kernel_y_id, kernel_x_id);
                            }
                            (false, false) => {
                                // Nothing needs to be done
                            }
                        }

                        let shape = graph.shape(*x);
                        if kernels[kernel_x_id].shape != shape {
                            kernels.push(Kernel::load(graph, *x));
                            kernel_x_id = kernels.len() - 1;
                        }

                        if kernels[kernel_y_id].shape != shape {
                            kernels.push(Kernel::load(graph, *y));
                            kernel_y_id = kernels.len() - 1
                        }

                        //println!("Kernel x");
                        //kernels[kernel_x_id].debug();
                        //println!("Kernel y");
                        //kernels[kernel_y_id].debug();
                        let (kernel_x, kernel_y) = if kernel_y_id > kernel_x_id {
                            let kernel_x = kernels.remove(kernel_x_id);
                            // we have just removed kernel before this one
                            kernel_y_id -= 1;
                            let kernel_y = &mut kernels[kernel_y_id];
                            (kernel_x, kernel_y)
                        } else {
                            let kernel_y = kernels.remove(kernel_y_id);
                            // we have just removed kernel before this one
                            kernel_x_id -= 1;
                            let kernel_x = &mut kernels[kernel_x_id];
                            (kernel_y, kernel_x)
                        };

                        assert_eq!(kernel_x.shape, kernel_y.shape);

                        // We cannot have both loops from kernel_x and kernel_y
                        // We have to remove one set of loops

                        let kernel_x_ops: Vec<VOp> = kernel_x
                            .ops
                            .into_iter()
                            .enumerate()
                            .skip_while(|(i, op)| {
                                matches!(op, VOp::Loop { .. }) && op == &kernel_y.ops[*i]
                            })
                            .map(|(_, op)| op)
                            .collect();
                        kernel_y.ops.extend(kernel_x_ops);
                        kernel_y.ops.push(VOp::Binary {
                            z: nid,
                            x: *x,
                            y: *y,
                            bop: *bop,
                        });
                        kernel_y.inputs.extend(kernel_x.inputs);
                        kernel_y.outputs.extend(kernel_x.outputs);
                        kernel_y.vars.extend(kernel_x.vars);
                        kernel_y.vars.insert(nid);
                    } else {
                        panic!()
                    }
                } else {
                    panic!()
                }
            }
        }
        if to_eval.contains(&nid)
            || (graph.rc(nid) > 1 && !matches!(graph[nid], Node::Leaf { .. } | Node::Const { .. }))
        {
            if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(&nid)) {
                kernel.store(nid, graph);
            } else {
                panic!()
            }
        }
    }
    // Remove unnecessary stores not for tensors moved across kernels
    // and not in to_eval that were inserted for rc > 1, but ops got merged,
    // and these stores were not used.
    let mut necessary_stores = to_eval.clone();
    for kernel in &kernels {
        necessary_stores.extend(kernel.inputs.iter());
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
    shape
        .iter()
        .copied()
        .enumerate()
        .map(|(axis, dimension)| VOp::Loop { axis, dimension })
        .collect()
}

// Checks if kernel_x depends on kernel_y
fn depends_on(kernel_x_id: usize, kernel_y_id: usize, kernels: &[Kernel]) -> bool {
    let mut kernel_x_inputs = kernels[kernel_x_id].inputs.clone();
    let kernel_y_outputs = &kernels[kernel_y_id].outputs;
    let mut visited = BTreeSet::new();
    while let Some(x) = kernel_x_inputs.pop_last() {
        if visited.insert(x) {
            if kernel_y_outputs.contains(&x) {
                return true;
            } else {
                for kernel in kernels.iter().rev() {
                    if kernel.outputs.contains(&x) {
                        kernel_x_inputs.extend(kernel.inputs.clone());
                        break;
                    }
                }
            }
        }
    }
    false
}

fn get_kernel<'a>(x: TensorId, kernels: &'a mut Vec<Kernel>, graph: &Graph) -> &'a mut Kernel {
    if let Some(id) = kernels
        .iter_mut()
        .position(|kernel| kernel.vars.contains(&x))
    {
        if kernels[id].shape != graph.shape(x) {
            // create new kernel using already predefined store
            kernels.push(Kernel::load(graph, x));
            kernels.last_mut().unwrap()
        } else {
            &mut kernels[id]
        }
    } else {
        panic!()
    }
}