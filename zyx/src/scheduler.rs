//! Converts graph to kernels and schedules them to devices

use crate::{
    backend::{BufferId, Device, MemoryPool},
    graph::Graph,
    index_map::Id,
    ir::{IRKernel, Scope},
    kernel::{Kernel, MOp, Op, TId},
    node::Node,
    optimizer::Optimizer,
    tensor::TensorId,
    view::View,
    DebugMask, ZyxError,
};
use std::collections::{BTreeMap, BTreeSet};

type KernelId = usize;

/// Convert graph into kernels and schedule them to devices.
/// This function needs to be optimized a lot, because it always needs to run faster than async launched kernels.
/// So no more than 10 microseconds per kernel.
/// If the scheduler is too slow more complex caching can be introduce.
pub(super) fn realize_graph(
    graph: &Graph,
    order: &[TensorId],
    to_eval: &BTreeSet<TensorId>,
    devices: &mut [Device],
    memory_pools: &mut [MemoryPool],
    tensor_buffer_map: &mut BTreeMap<TensorId, BufferId>,
    optimizer: &mut Optimizer,
    search_iters: usize,
    debug: DebugMask,
) -> Result<(), ZyxError> {
    // Unfinished kernels represented by ops
    let mut kernels: Vec<Kernel> = Vec::with_capacity(100);
    // Mapping from tensor ids to loads and stores in kernels
    let mut kernel_tensors: Vec<BTreeMap<TensorId, TId>> = Vec::with_capacity(100);
    // Mapping from tensor ids to unused tensors in kernels.
    // Once number of unused ids reaches zero, kernel gets scheduled to device.
    let mut kernel_outputs: Vec<BTreeMap<TensorId, TId>> = Vec::with_capacity(100);

    let mut free_kernels: BTreeSet<KernelId> = BTreeSet::new();

    for nid in order.iter().copied() {
        println!("ID({nid}): {:?}, sh: {:?}", graph[nid], graph.shape(nid));

        if free_kernels.is_empty() {
            kernels.push(Kernel::empty());
            kernel_tensors.push(BTreeMap::new());
            kernel_outputs.push(BTreeMap::new());
            free_kernels.insert(kernels.len() - 1);
        }

        let kid: KernelId = match graph[nid] {
            // All ops are merged except
            // Pad is not merged of kernel contains store
            // Reshape is not merged if reshaping reduce loops
            // Expand is not merged if expanding reduce kernel or kernel contains store
            // These rules will be later loosened using some heuristic
            Node::Const { value } => {
                let kid = free_kernels.pop_first().unwrap();
                kernels[kid] = Kernel::constant(value);
                kernel_outputs[kid] = BTreeMap::from([(nid, 0)]);
                kid
            }
            Node::Leaf => {
                let kid = free_kernels.pop_first().unwrap();
                kernels[kid] = Kernel::leaf(graph.shape(nid), graph.dtype(nid));
                kernel_tensors[kid] = BTreeMap::from([(nid, 0)]);
                kernel_outputs[kid] = BTreeMap::from([(nid, 0)]);
                kid
            }
            Node::Expand { x } => {
                let (xt, mut kid) = get_kernel(x, &kernel_outputs);

                let shape = graph.shape(nid);
                if !kernels[kid].expand(shape) {
                    let dtype = graph.dtype(x);
                    kernels[kid].store(xt, View::contiguous(graph.shape(x)), dtype);
                    kid = free_kernels.pop_first().unwrap();
                    kernels[kid] = Kernel::leaf(shape, dtype);
                }

                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;
                kernels[kid].ops.push(Op::Move {
                    z,
                    x: xt,
                    mop: MOp::Expa,
                });
                kernel_outputs[kid].remove(&x).unwrap();
                kernel_outputs[kid].insert(nid, z);
                kid
            }
            Node::Reshape { x } => {
                let (xt, mut kid) = get_kernel(x, &kernel_outputs);

                let shape = graph.shape(nid);
                //println!("Reshape node from {:?} to {:?}", graph.shape(x), shape);
                if !kernels[kid].reshape(shape) {
                    // else create new kernel after storing results of previous kernel
                    let dtype = graph.dtype(x);
                    kernels[kid].store(xt, View::contiguous(graph.shape(x)), dtype);
                    kid = free_kernels.pop_first().unwrap();
                    kernels[kid] = Kernel::leaf(shape, dtype);
                }

                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;
                kernels[kid].ops.push(Op::Move {
                    z,
                    x: xt,
                    mop: MOp::Resh,
                });
                kernel_outputs[kid].remove(&x).unwrap();
                kernel_outputs[kid].insert(nid, z);
                kid
            }
            Node::Pad { x } => {
                let (xt, mut kid) = get_kernel(x, &kernel_outputs);

                let padding = graph.padding(nid);
                if !kernels[kid].pad(padding) {
                    let dtype = graph.dtype(x);
                    kernels[kid].store(xt, View::contiguous(graph.shape(x)), dtype);
                    kid = free_kernels.pop_first().unwrap();
                    let shape = graph.shape(nid);
                    kernels[kid] = Kernel::leaf(shape, dtype);
                }

                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;
                kernels[kid].ops.push(Op::Move {
                    z,
                    x: xt,
                    mop: MOp::Expa,
                });
                kernel_outputs[kid].remove(&x).unwrap();
                kernel_outputs[kid].insert(nid, z);
                kid
            }
            Node::Permute { x } => {
                let (xt, kid) = get_kernel(x, &kernel_outputs);
                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;

                let axes = graph.axes(nid);
                kernels[kid].permute(axes);
                kernels[kid].ops.push(Op::Move {
                    z,
                    x: xt,
                    mop: MOp::Perm,
                });

                kernel_outputs[kid].remove(&x).unwrap();
                kernel_outputs[kid].insert(nid, z);
                kid
            }
            Node::Reduce { x, rop } => {
                let (xt, kid) = get_kernel(x, &kernel_outputs);
                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;
                let axes = graph.axes(nid);
                kernels[kid].reduce(xt, axes, rop);
                kernel_outputs[kid].remove(&x).unwrap();
                kernel_outputs[kid].insert(nid, z);
                kid
            }
            Node::Unary { x, uop } => {
                let (xt, kid) = get_kernel(x, &kernel_outputs);
                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;
                kernels[kid].ops.push(Op::Unary { z, x: xt, uop });
                kernel_outputs[kid].remove(&x).unwrap();
                kernel_outputs[kid].insert(nid, z);
                kid
            }
            Node::Binary { x, y, bop } => todo!(),
        };

        if to_eval.contains(&nid) {
            kernels[kid].store(
                kernel_outputs[kid][&nid],
                View::contiguous(graph.shape(nid)),
                graph.dtype(nid),
            );
            let id = kernel_outputs[kid].remove(&nid).unwrap();
            kernel_tensors[kid].insert(nid, id);
        }

        if kernel_outputs[kid].is_empty() {
            // Delete kernel and dispatch it to device
            let mut kernel = Kernel::empty();
            std::mem::swap(&mut kernel, &mut kernels[kid]);

            if debug.sched() {
                kernel.debug();
            }

            // Pick a device to run program
            // Find in which memory pool are most of input tensors stored
            let memory_pool_id = 0;
            let memory_pool = &mut memory_pools[memory_pool_id as usize];

            // Move all other tensors to that memory pool
            // and finish queues with this kernel's inputs

            // Get device which is associated with that memory pool
            let device = &mut devices[0];

            let buffer_ids: Vec<Id> = kernel_tensors[kid]
                .keys()
                .map(|&tensor_id| {
                    if let Some(BufferId { buffer_id, .. }) = tensor_buffer_map.get(&tensor_id) {
                        *buffer_id
                    } else {
                        // Allocate bytes for outputs
                        let buffer_id = memory_pool
                            .allocate(
                                graph.shape(tensor_id).iter().product::<usize>()
                                    * graph.dtype(tensor_id).byte_size(),
                            )
                            .unwrap();
                        tensor_buffer_map.insert(
                            tensor_id,
                            BufferId {
                                memory_pool_id,
                                buffer_id,
                            },
                        );
                        buffer_id
                    }
                })
                .collect();

            if device.is_cached(&kernel) {
                device.launch(&kernel, memory_pool, &buffer_ids)?;
            } else {
                let optimization = optimizer.search_optimization(
                    &kernel,
                    device,
                    memory_pool,
                    search_iters,
                    debug,
                )?;
                let optimized_kernel = kernel.optimize(optimization);
                let ir_kernel = IRKernel::new(&optimized_kernel.ops);
                device.compile(kernel.clone(), &ir_kernel, true)?;
                device.launch(&kernel, memory_pool, &buffer_ids)?;
            }

            // add load kernels for all outputs of this kernel
            for op in kernel.ops {
                if let Op::Store { z, .. } = op {
                    let shape = graph.shape(nid);
                    let dtype = graph.dtype(nid);
                    kernel_tensors.push(BTreeMap::from([(
                        kernel_tensors[kid]
                            .iter()
                            .find(|(_, tid)| **tid == z)
                            .unwrap()
                            .0
                            .clone(),
                        0,
                    )]));
                    kernel_outputs.push(BTreeMap::new());
                    kernels.push(Kernel::leaf(shape, dtype));
                }
            }
        }
    }

    Ok(())
}

fn get_kernel(x: TensorId, kernel_outputs: &[BTreeMap<TensorId, TId>]) -> (TId, KernelId) {
    for (kid, outputs) in kernel_outputs.iter().enumerate() {
        if let Some(&x_tid) = outputs.get(&x) {
            return (x_tid, kid);
        }
    }
    unreachable!()
}
