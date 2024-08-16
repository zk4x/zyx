use crate::{
    dtype::Constant,
    runtime::{node::{BOp, Node, ROp, UOp}, backend::DeviceInfo, Runtime, ZyxError},
    runtime::{graph::Graph, view::View, DeviceId},
    shape::{Axis, Dimension},
    tensor::TensorId,
};
use std::{collections::{BTreeMap, BTreeSet}, fmt::Display};

use super::{BufferId, Device, MemoryPool, MemoryPoolId, ProgramId};

// In which order
pub(super) struct CompiledGraph {
    sched_graph: Vec<SchedulerOp>,
}

// TODO this function could take &mut Runtime
impl Runtime {
    pub(super) fn compile_graph(
        &mut self,
        #[allow(unused_mut)]
        mut graph: Graph,
        to_eval: &BTreeSet<TensorId>,
    ) -> Result<CompiledGraph, ZyxError> {
        let (order, flop, bytes_read, bytes_written) = graph.execution_order(to_eval);
        // create vop representation
        let mut kernels = generate_kernels(&graph, &order, &to_eval);
        // create graph of kernels, sharding tensors across devices, shard also kernels appropriatelly
        let mut sched_graph: Vec<SchedulerOp> = Vec::new();
        // Tensors that are being evaluated on devices, but still not finished
        // Each device can calculate only small part of the tensor
        // KernelId points into kernel in kernels
        let mut finished_programs: BTreeSet<ProgramId> = BTreeSet::new();
        let mut all_programs: BTreeSet<ProgramId> = BTreeSet::new();
        // All leafs are allocated in some memory pool (can be disk)
        let mut allocated_tensors: BTreeSet<TensorId> = order.iter().filter_map(|t| if let Node::Leaf { .. } = graph[*t] { Some(*t) } else { None }).collect();
        let mut unoccupied_devices: BTreeSet<usize> = (0..self.devices.len()).collect();
        let mut program_kernel_map: BTreeMap<ProgramId, usize> = BTreeMap::new();
        let mut kernel_id = 0;
        while kernel_id < kernels.len() {
            // check which kernels (if any) must be finished before launchibng the next kernel
            for i in (0..sched_graph.len()).rev() {
                if let SchedulerOp::Launch(program_id) = sched_graph[i] {
                    if !kernels[program_kernel_map[&program_id]].outputs.is_disjoint(&kernels[kernel_id].inputs)
                        && !finished_programs.contains(&program_id)
                    {
                        sched_graph.push(SchedulerOp::Finish(program_id));
                        finished_programs.insert(program_id);
                        unoccupied_devices.insert(program_id.device_id);
                        // TODO deallocate inputs of kernels[lkid] if they are not used elsewhere
                    }
                }
            }
            // assign kernel to devices based on available compute and whether the kernel can be sharded
            // This can be later improved by using better heuristics, but for now just check if kernel
            // can be sharded, if yes, shard across all available devices, otherwise assign it to most
            // powerfull device.
            // TODO also check if it even makes sense to shard. That is if we have at least
            // two devices and if flop required to realize the kernel is not less than
            // the time taken to launch the kernel and time needed to copy memory
            let shard = if unoccupied_devices.len() > 1 {
                if let Some((axis, dimension)) = kernels[kernel_id].shard_axis() {
                    Some((axis, dimension))
                } else {
                    None
                }
            } else {
                None
            };

            if let Some((axis, dimension)) = shard {
                // dimension is size of the sharded dimension
                let unoccupied_compute: usize =
                    unoccupied_devices.iter().map(|dev_id| self.devices[*dev_id].compute()).sum();
                let shard_sizes: Vec<Dimension> = unoccupied_devices
                    .iter()
                    .map(|dev_id| dimension * self.devices[*dev_id].compute() / unoccupied_compute)
                    .collect();
                let sharded_kernel_ids: Vec<KernelId> =
                    shard_kernels(&mut kernels, kernel_id, axis, shard_sizes);
                for (skid, dev) in sharded_kernel_ids
                    .into_iter()
                    .zip(unoccupied_devices.iter())
                {
                    // launch the kernel
                    sched_graph.push(SchedulerOp::Launch(ProgramId { device_id: *dev, program_id: skid }));
                }
                unoccupied_devices.clear();
            } else {
                // Find the fastest out of unoccupied devices
                let device_id = *unoccupied_devices
                    .iter()
                    .max_by(|x, y| self.devices[**x].compute().cmp(&self.devices[**y].compute()))
                    .unwrap();
                // launch the kernel
                let program_id = match &mut self.devices[device_id] {
                    Device::OpenCL { device, memory_pool_id, programs } => {
                        kernels[kernel_id].optimize(device.info());
                        // TODO make more efficient use of global variables by reusing the same allocation
                        // for temporary global variables which are not used through the whole running of the
                        // kernel (rc drops to 0)
                        println!();
                        #[cfg(feature = "debug_sched")]
                        for vop in &kernels[kernel_id].ops {
                            println!("{vop}");
                        }
                        println!();

                        // Move inputs to the device
                        for input in &kernels[kernel_id].inputs {
                            // move from device to this device if the device is not the same
                            // check on which memory pool it is stored
                            // TODO
                        }
                        // Allocate memory for outputs
                        for output in &kernels[kernel_id].outputs {
                            if !allocated_tensors.contains(output) {
                                let shape = graph.shape(*output);
                                sched_graph.push(SchedulerOp::Allocate { tensor_id: *output, memory_pool_id: *memory_pool_id, bytes: shape.iter().product::<usize>() * graph.dtype(*output).byte_size(), view: View::new(shape) });
                                allocated_tensors.insert(*output);
                            }
                        }
                        let (ir_kernel, args) = kernels[kernel_id].to_ir(&graph);
                        let program = self.opencl.as_mut().unwrap().compile_program(&ir_kernel, &device)?;
                        // Since it is not sharded, sharding view is contiguous
                        programs.push((program, args.into_iter().map(|arg| (arg, View::new(graph.shape(arg)))).collect()));
                        programs.len() - 1
                    }
                };
                let program_id = ProgramId { device_id, program_id };
                sched_graph.push(SchedulerOp::Launch(program_id));
                all_programs.insert(program_id);
                program_kernel_map.insert(program_id, kernel_id);
                unoccupied_devices.remove(&device_id);
            }
            kernel_id += 1;
        }
        for program_id in all_programs.difference(&finished_programs) {
            sched_graph.push(SchedulerOp::Finish(*program_id));
        }

        #[cfg(feature = "debug_sched")]
        for sched_op in &sched_graph {
            match sched_op {
                SchedulerOp::Launch(program_id) => println!("Launch kernel {:?}", self.devices[program_id.device_id]),
                SchedulerOp::Finish(program_id) => println!("Finish kernel {program_id:?}"),
                SchedulerOp::MemCopy {
                    tensor_id: tensor,
                    src,
                    dst,
                    view,
                } => println!("Copy tensor {tensor} from {src:?} to {dst:?} with {view:?}"),
                SchedulerOp::Allocate { tensor_id, bytes, memory_pool_id, view } => {
                    println!("Allocate tensor {tensor_id} on memory pool {memory_pool_id:?} with size {bytes:?} B")
                }
                SchedulerOp::Deallocate { tensor_id, memory_pool_id, bytes, view } => {
                    println!("Allocate tensor {tensor_id} on {memory_pool_id:?}")
                }
            }
        }

        Ok(CompiledGraph {
            sched_graph,
        })
    }

    pub(super) fn launch_graph(&mut self, graph: &Graph) -> Result<(), ZyxError> {
        let mut events = BTreeMap::new();
        for sched_op in &self.compiled_graphs[graph].sched_graph {
            match sched_op {
                SchedulerOp::Launch(program_id) => match &mut self.devices[program_id.device_id] {
                    Device::OpenCL { device: _, memory_pool_id, programs } => {
                        let (program, args) = &mut programs[program_id.program_id];
                        let args: Vec<usize> = args.iter().map(|arg| self.tensor_buffer_map[arg].buffer_id).collect();
                        let MemoryPool::OpenCL { buffers, .. } = &mut self.memory_pools[*memory_pool_id];
                        events.insert(*program_id, self.opencl.as_mut().unwrap().launch_program(program, buffers, &args)?);
                    }
                }
                SchedulerOp::Finish(program_id) => {
                    match &self.devices[program_id.device_id] {
                        Device::OpenCL { .. } => {
                            self.opencl.as_mut().unwrap().finish_event(events.remove(&program_id).unwrap())?;
                        },
                    }
                },
                SchedulerOp::MemCopy { tensor_id, src, dst, view } => todo!(),
                SchedulerOp::Allocate { tensor_id, memory_pool_id, bytes, view } => match &mut self.memory_pools[*memory_pool_id] {
                    MemoryPool::OpenCL { memory_pool, buffers } => {
                        let buffer = self.opencl.as_mut().unwrap().allocate_memory(*bytes, memory_pool)?;
                        let buffer_id = buffers.push(buffer);
                        self.tensor_buffer_map.insert((*tensor_id, view.clone()), BufferId { memory_pool_id: *memory_pool_id, buffer_id });
                    }
                }
                SchedulerOp::Deallocate { tensor_id, memory_pool_id: memory_pool, bytes, view } => todo!(),
            }
        }
        Ok(())
    }
}

type KernelId = usize;

enum SchedulerOp {
    // Async launch kernel on device
    Launch(ProgramId),
    // Block for kernel to finish execution
    Finish(ProgramId),
    // Copy part of tensor between devices
    // This is used for sharding, but can be used for other purposes too,
    // if found usefull
    MemCopy {
        tensor_id: TensorId,
        src: DeviceId,
        dst: DeviceId,
        view: View,
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

#[derive(Debug, PartialEq, Eq)]
pub(super) enum VOp {
    Const {
        z: TensorId,
        value: Constant,
    },
    Load {
        z: TensorId,
        x: TensorId,
        view: View,
    },
    Store {
        z: TensorId,
        view: View,
    },
    Loop {
        axis: Axis,
        dimension: Dimension,
    },
    Accumulator {
        z: TensorId,
        rop: ROp,
        view: View,
    },
    Reduce {
        z: TensorId,
        x: TensorId,
        num_axes: usize,
        rop: ROp,
    },
    Noop {
        z: TensorId,
        x: TensorId,
    },
    Unary {
        z: TensorId,
        x: TensorId,
        uop: UOp,
    },
    Binary {
        z: TensorId,
        x: TensorId,
        y: TensorId,
        bop: BOp,
    },
}

#[derive(Debug)]
pub(super) struct Kernel {
    // Current shape of the kernel after all current ops
    pub(super) shape: Vec<Dimension>,
    // Global loads
    pub(super) inputs: BTreeSet<TensorId>,
    // Global stores
    pub(super) outputs: BTreeSet<TensorId>,
    // Register variables
    vars: BTreeSet<TensorId>,
    pub(super) ops: Vec<VOp>,
}

impl Kernel {
    fn load(graph: &Graph, x: TensorId) -> Kernel {
        let shape: Vec<usize> = graph.shape(x).into();
        let mut ops: Vec<VOp> = shape_to_loops(&shape);
        ops.push(VOp::Load {
            z: x,
            x,
            view: View::new(&shape),
        });
        Kernel {
            shape,
            inputs: BTreeSet::from([x]),
            outputs: BTreeSet::new(),
            vars: BTreeSet::from([x]),
            ops,
        }
    }

    fn store(&mut self, z: TensorId, graph: &Graph) {
        let store_op = VOp::Store {
            z,
            view: View::new(graph.shape(z)),
        };
        if self.ops.last().unwrap() != &store_op {
            self.ops.push(store_op);
            self.outputs.insert(z);
        }
    }

    fn permute(&mut self, axes: &[usize]) {
        if axes.iter().zip(0..axes.len()).all(|(a, ca)| *a == ca) {
            // no permute
            return;
        }
        let shape: Vec<usize> = axes.iter().map(|a| self.shape[*a]).collect();
        let mut permuted_loops: BTreeSet<usize> = axes.iter().copied().collect();
        'ops_loop: for op in self.ops.iter_mut().rev() {
            match op {
                VOp::Loop { axis, dimension } => {
                    if axes.contains(axis) {
                        *dimension = shape[*axis];
                        permuted_loops.remove(axis);
                        if permuted_loops.is_empty() {
                            break 'ops_loop;
                        }
                    }
                }
                VOp::Load { view, .. } | VOp::Store { view, .. } => {
                    let n = view.rank();
                    let all_axes: Vec<usize> = if axes.len() < n {
                        axes.iter().copied().chain(axes.len()..n).collect()
                    } else {
                        axes.iter().copied().filter(|a| *a < n).collect()
                    };
                    view.permute(&all_axes);
                }
                _ => {}
            }
        }
        self.shape = shape.clone();
    }

    pub(super) fn split_axis(&mut self, op_id: usize, dimensions: &[usize]) {
        //println!("Splitting {op_id} into {dimensions:?}");
        // First split loop at op_id
        let VOp::Loop { axis, dimension } = &mut self.ops[op_id] else {
            panic!()
        };
        *dimension = dimensions[0];
        let axis = *axis;
        let mut temp_axis = axis;
        let mut id = op_id;
        for dim in dimensions[1..].iter() {
            id += 1;
            temp_axis += 1;
            self.ops.insert(
                id,
                VOp::Loop {
                    axis: temp_axis,
                    dimension: *dim,
                },
            )
        }
        let mut num_loops = 0;
        // Update shape
        self.shape.remove(axis);
        for dim in dimensions {
            self.shape.insert(axis, *dim);
        }
        // Update loops, loads and stores
        for i in id + 1..self.ops.len() {
            match &mut self.ops[i] {
                // Then change axis ids for all following loops
                VOp::Loop { axis, .. } => {
                    *axis += dimensions.len() - 1;
                    num_loops += 1;
                }
                VOp::Reduce { .. } => {
                    num_loops -= 1;
                    if num_loops == 0 {
                        break;
                    }
                    // TODO num_axes changes?
                }
                // Then change all load and store operations in this
                // loop in the same way.
                VOp::Load { view, .. } => {
                    //println!("Splitting {view:?}");
                    view.split_axis(axis, dimensions);
                }
                VOp::Store { view, .. } => {
                    // Example of axis split
                    // shape
                    //  2, 6,    2
                    //  2, 3, 2, 2
                    // strides
                    // 12, 2,    1
                    // 12, 4, 2, 1
                    view.split_axis(axis, dimensions);
                }
                _ => {}
            }
        }
    }

    /*fn merge_axes(&mut self, op_id: usize, num_loops: usize) {
        // Merges multiple consecutive loops (beginning with loop at op_id) into single loop
        // This function does not change shape of the kernel
        // When there are loads and stores with expanded strides in merged axes,
        // then merge is not possible unless we add multiple shapes to view
        let mut dim_size = 1;
        for id in op_id..op_id + num_loops {
            if let VOp::Loop { dimension, .. } = self.ops[id] {
                dim_size *= dimension;
            }
        }
        // Get which axis is kept
        let axis_id = if let VOp::Loop { dimension, axis } = &mut self.ops[op_id] {
            *dimension = dim_size;
            *axis
        } else {
            panic!()
        };
        // Remove unnecessary loops
        for _ in op_id..op_id + num_loops - 1 {
            self.ops.remove(op_id + 1);
        }
        // Merge strides and dimensions on loads and stores
        for op in &mut self.ops[op_id + 1..] {
            match op {
                VOp::Reduce { num_axes, .. } => {
                    *num_axes = 1;
                    break;
                }
                VOp::Load { view, .. } => {
                    let stride = view.0[axis_id + num_loops - 1].stride;
                    view.0[axis_id].dim = dim_size;
                    view.0[axis_id].stride = stride;
                    for _ in 0..num_loops - 1 {
                        view.0.remove(axis_id + 1);
                    }
                }
                VOp::Store { strides, .. } => {
                    let stride = strides[axis_id + num_loops - 1];
                    strides[axis_id] = stride;
                    for _ in 0..num_loops - 1 {
                        strides.remove(axis_id + 1);
                    }
                }
                _ => {}
            }
        }
    }*/

    fn shard_axis(&self) -> Option<(Axis, Dimension)> {
        // Shard axis is axis that is not gonna be locally cached,
        // which is usually the batch axis, but it can also be other axes.
        // Since we do not locally cache axis 0, we can for now always just return that
        Some((0, self.shape[0]))
    }

    fn optimize(&mut self, dev_info: &DeviceInfo) {
        // add per device optimizations to each kernel, local memory, accumulators, work per thread, tiling on many levels
        // Get the number of loops before any other operation
        let num_loops = self
            .ops
            .iter()
            .position(|kernel| !matches!(kernel, VOp::Loop { .. }))
            .unwrap();

        // If this is full reduce kernel
        if num_loops == 0 {
            // this should never happen, because we should use local and register memory
            // and always spread work across multiple threads
            todo!("Full reduce")
        }

        // If there is more loops than 3, pick first three loops as global loops,
        // rest is register loops.
        // So nothing needs to be done.
        // If there is less than three loops, add loops with dimension 1
        if num_loops < 3 {
            let dims: Vec<usize> = core::iter::repeat(1)
                .take(3 - num_loops)
                .chain([self.shape[0]])
                .collect();
            self.split_axis(0, &dims);
        }

        // Split first three loops into global and local loops.
        let mut gws = [1; 3];
        for op in &self.ops {
            if let VOp::Loop { axis, dimension } = op {
                if *axis > 2 {
                    break;
                }
                gws[*axis] = *dimension;
            }
        }

        // Reorder global loops from smallest to largest
        //gws.sort();
        // Get sort indices and permute both kernel and gws
        // by those indices

        // Determine the best possible work size
        let lws = best_local_work_size(gws, dev_info.max_work_group_size);
        gws[0] /= lws[0];
        gws[1] /= lws[1];
        gws[2] /= lws[2];

        self.split_axis(0, &[gws[0], lws[0]]);
        self.split_axis(2, &[gws[1], lws[1]]);
        self.split_axis(4, &[gws[2], lws[2]]);
    }
}

// Takes global work size (gws) and maximum work group size (mwgs)
fn best_local_work_size(mut gws: [usize; 3], mwgs: usize) -> [usize; 3] {
    let mut lws = [1; 3];
    //println!("Max {mwgs:?}");
    let rwgs = (mwgs as f64).sqrt() as usize;
    //println!("Root {rwgs:?}");

    let mut total = 1;
    let mut n = 1;
    while gws[1] % (n * 2) == 0 && n * 2 <= rwgs {
        n *= 2;
    }
    gws[1] /= n;
    lws[1] *= n;
    total *= n;
    // put the rest into third dimension
    let mut n = 1;
    while gws[2] % (n * 2) == 0 && n * 2 * total <= mwgs {
        n *= 2;
    }
    gws[2] /= n;
    lws[2] *= n;
    total *= n;
    // if third dimension was too small, put the rest into second dimension
    let mut n = 1;
    while gws[1] % (n * 2) == 0 && n * 2 * total <= mwgs {
        n *= 2;
    }
    gws[1] /= n;
    lws[1] *= n;

    return lws;
}

fn shard_kernels(
    kernels: &mut Vec<Kernel>,
    kid: KernelId,
    axis: Axis,
    shard_sizes: Vec<Dimension>,
) -> Vec<KernelId> {
    let _ = kernels;
    let _ = kid;
    let _ = axis;
    let _ = shard_sizes;
    todo!()
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
        //println!("ID({nid})x{}: {node:?}", graph.rc(nid));
        match node {
            Node::Const { value } => {
                let const_op = VOp::Const {
                    z: nid,
                    value: *value,
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
            Node::Leaf { shape, .. } => {
                if let Some(kernel) = kernels.iter_mut().find(|kernel| &kernel.shape == shape) {
                    kernel.ops.push(VOp::Load {
                        z: nid,
                        x: nid,
                        view: View::new(shape),
                    });
                    kernel.inputs.insert(nid);
                    kernel.vars.insert(nid);
                } else {
                    kernels.push(Kernel::load(graph, nid));
                }
            }
            Node::Expand { x, shape } => {
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
                        VOp::Load { view, .. } => {
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
                kernel.ops.push(VOp::Noop { z: nid, x: *x });
                kernel.vars.insert(nid);
                kernel.shape = shape.clone();
            }
            Node::Permute { x, axes, .. } => {
                // Permute shuffles load and store strides
                // It also changes the dimension of loops
                // and shape of kernel
                // TODO but what if it is permute after reduce?
                let kernel = get_kernel(*x, &mut kernels, graph);
                kernel.permute(&axes);
                kernel.ops.push(VOp::Noop { z: nid, x: *x });
                kernel.vars.insert(nid);
            }
            Node::Reshape { x, shape } => {
                // If we really want, we can get reshape working with loads and stores
                // simply by using view for loads to have multiple reshapes in single view.
                // But for now it is much simpler to just add new kernel.

                // If reshape comes after reduce, then if it just aplits axes, it can be merged,
                // otherwise we have to create new kernel.

                let kernel = get_kernel(*x, &mut kernels, graph);
                // If this is just a reshape of kernel with only unary ops and contiguous loads
                // and stores, we can remove old loops and replace them with new loops.
                if kernel.ops.iter().all(|op| match op {
                    VOp::Loop { .. }
                    | VOp::Unary { .. }
                    | VOp::Binary { .. }
                    | VOp::Const { .. }
                    | VOp::Noop { .. } => true,
                    VOp::Load { view, .. } | VOp::Store { view, .. } => view.is_contiguous(),
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
                            VOp::Load { view, .. } | VOp::Store { view, .. } => {
                                *view = View::new(shape);
                            }
                            _ => {}
                        }
                    }
                    kernel.shape = shape.clone();
                    kernel.ops.push(VOp::Noop { z: nid, x: *x });
                    kernel.vars.insert(nid);
                } else {
                    // TODO
                    // If we can split axes, split axes by replacing one loop with two loops.
                    // If last axes are unsqueezes with ones, add new loops to the end of the kernel.

                    // else create new kernel after storing results of previous kernel
                    kernel.store(*x, graph);
                    let mut ops = shape_to_loops(shape);
                    ops.push(VOp::Load {
                        z: nid,
                        x: *x,
                        view: View::new(shape),
                    });
                    kernels.push(Kernel {
                        shape: shape.clone(),
                        inputs: BTreeSet::from([*x]),
                        outputs: BTreeSet::new(),
                        vars: BTreeSet::from([nid]),
                        ops,
                    });
                }
                //println!("\nKernels {kernels:?}\n");
            }
            Node::Pad { x, padding, shape } => {
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
                        VOp::Load { view, .. } => {
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
                kernel.shape = shape.clone();
                kernel.ops.push(VOp::Noop { z: nid, x: *x });
                kernel.vars.insert(nid);
            }
            Node::Reduce {
                x,
                axes,
                rop,
                shape,
            } => {
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
                    num_axes: axes.len(), // Now we are merging them, without merging its axes.len(),
                    rop: *rop,
                    z: nid,
                    x: *x,
                });
                kernel.vars.insert(nid);
                kernel.shape = shape.clone();
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
                            x: *y,
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

                        // We know that kernel_y is the latest kernel,
                        // since this is the order in which ordering of nodes works.
                        assert_eq!(kernel_y_id, kernels.len() - 1);

                        let kernel_x = kernels.remove(kernel_x_id);
                        // we have just removed kernel before this one
                        kernel_y_id -= 1;

                        let kernel_y = &mut kernels[kernel_y_id];
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

#[cfg(feature = "debug_sched")]
impl Display for VOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use inline_colorization::*;
        match self {
            VOp::Const { z, value } => f.write_fmt(format_args!("{color_white}Const{color_reset}       {z} <- value: {value}")),
            VOp::Load { z, x, view } => f.write_fmt(format_args!("{color_yellow}Load{color_reset}        {z} <- {x}")),
            VOp::Store { z, view } => f.write_fmt(format_args!("{color_red}Store{color_reset}       {z}")),
            VOp::Loop { axis, dimension } => f.write_fmt(format_args!("{color_green}Loop{color_reset}        axis: {axis}, dimension: {dimension}")),
            VOp::Accumulator { z, rop, view } => f.write_fmt(format_args!("{color_blue}Accum{color_reset}.{rop:?}   {z} {:?}", view.shape())),
            VOp::Reduce { z, x, num_axes, rop } => f.write_fmt(format_args!("{color_magenta}Reduce{color_reset}.{rop:?}  {z} <- {x}, num_axes: {num_axes}")),
            VOp::Noop { z, x } => f.write_fmt(format_args!("{color_white}Noop{color_reset}        {z} <- {x}")),
            VOp::Unary { z, x, uop } => f.write_fmt(format_args!("{color_white}Unary{color_reset}.{uop:?}{} {z} <- {x}", core::iter::repeat(" ").take(5-format!("{uop:?}").len()).collect::<String>())),
            VOp::Binary { z, x, y, bop } => f.write_fmt(format_args!("{color_white}Binary{color_reset}.{bop:?}  {z} <- {x}, {y}")),
        }
    }
}
