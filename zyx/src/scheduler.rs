//! Converts graph to kernels and schedules them to devices

use crate::{
    backend::{BufferId, Device, MemoryPool},
    graph::Graph,
    index_map::Id,
    ir::IRKernel,
    kernel::{Kernel, Op, TId},
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
    // Mapping from tensor ids to loads in kernels
    let mut kernel_tensors: Vec<BTreeMap<TensorId, TId>> = Vec::with_capacity(100);
    // Mapping from tensor ids to unused tensors in kernels.
    // Once number of unused ids reaches zero, kernel gets scheduled to device.
    let mut kernel_outputs: Vec<BTreeMap<TensorId, TId>> = Vec::with_capacity(100);

    let mut free_kernels: BTreeSet<KernelId> = BTreeSet::new();

    for nid in order.iter().copied() {
        if free_kernels.is_empty() {
            kernels.push(Kernel::empty());
            kernel_tensors.push(BTreeMap::new());
            kernel_outputs.push(BTreeMap::new());
            free_kernels.insert(kernels.len() - 1);
        }

        let kid: KernelId = match graph[nid] {
            Node::Const { value } => {
                let kid = free_kernels.pop_first().unwrap();
                Kernel::constant(value);
                kid
            }
            Node::Leaf => {
                let kid = free_kernels.pop_first().unwrap();
                kernels[kid] = Kernel::leaf(graph.shape(nid), graph.dtype(nid));
                kernel_tensors[kid] = BTreeMap::from([(nid, 0)]);
                kid
            }
            Node::Expand { x } => todo!(),
            Node::Permute { x } => todo!(),
            Node::Reshape { x } => todo!(),
            Node::Pad { x } => todo!(),
            Node::Reduce { x, rop } => todo!(),
            Node::Unary { x, uop } => {
                let (x, kid) = get_kernel(x, &kernel_outputs);
                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;
                kernels[kid].ops.push(Op::Unary { z, x, uop });
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
        }

        if kernel_outputs[kid].is_empty() {
            // Delete kernel and dispatch it to device
            let mut kernel = Kernel::empty();
            std::mem::swap(&mut kernel, &mut kernels[kid]);

            // Pick a device to run program
            // Find in which memory pool are most of input tensors stored
            //kernel_inputs[kid]
            let memory_pool = &mut memory_pools[0];

            // Move all other tensors to that memory pool
            // and finish queues with this kernel's inputs

            // Get device which is associated with that memory pool
            let device = &mut devices[0];

            let buffer_ids: Vec<Id> = kernel_tensors[kid]
                .keys()
                .map(|tensor| tensor_buffer_map[tensor].buffer_id)
                .collect();

            if device.is_cached(&kernel) {
                device.launch(&kernel, memory_pool, &buffer_ids)?;
            } else {
                let optimization =
                    optimizer.get_optimization(&kernel, device, memory_pool, search_iters);
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
