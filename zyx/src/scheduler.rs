//! Converts graph to kernels and schedules them to devices

use crate::{
    backend::{BufferId, Device, MemoryPool},
    graph::Graph,
    index_map::{Id, IndexMap},
    ir::IRKernel,
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
    // TODO perhaps we can get graph rcs from runtime realize frunction instead of this...
    let mut graph_rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
    let mut params = to_eval.clone();
    while let Some(param) = params.pop_last() {
        graph_rcs.entry(param).and_modify(|rc| *rc += 1).or_insert_with(|| {
            if !tensor_buffer_map.contains_key(&param) {
                params.extend(graph[param].parameters());
            }
            1
        });
    }

    // Unfinished kernels represented by ops
    let mut kernels: IndexMap<Kernel> = IndexMap::with_capacity(100);

    println!("To eval: {:?}", to_eval);

    for nid in order.iter().copied() {
        /*println!("Current kernels:");
        for kernel in &kernels {
            kernel.debug();
        }
        println!();*/
        println!("ID({nid}): {:?}, sh: {:?}", graph[nid], graph.shape(nid));

        // In case of kernels which delete outputs we need to keep reference count
        // and not delete tensors from outputs if rc > 1

        let (kidx, kidy) = match graph[nid] {
            // All ops are merged except
            // Pad is not merged of kernel contains store
            // Reshape is not merged if reshaping reduce loops
            // Expand is not merged if expanding reduce kernel or kernel contains store
            // These rules will be later loosened using some heuristic
            Node::Const { value } => (kernels.push(Kernel::constant(nid, value)), KernelId::MAX),
            Node::Leaf => (kernels.push(Kernel::leaf(nid, graph.shape(nid), graph.dtype(nid))), KernelId::MAX),
            Node::Expand { x } => {
                let (xt, kid) = get_kernel(x, &kernel_outputs);

                let shape = graph.shape(nid);
                if kernels[kid].expand(shape) {
                    kernels[kid].max_id += 1;
                    let z = kernels[kid].max_id;
                    kernels[kid].ops.push(Op::Move { z, x: xt, mop: MOp::Expa });
                    if let Some(rc) = graph_rcs.get_mut(&x) {
                        *rc -= 1;
                        if *rc == 0 {
                            graph_rcs.remove(&x);
                            kernel_outputs[kid].remove(&x).unwrap();
                        }
                    }
                    kernel_outputs[kid].insert(nid, z);
                    (kid, KernelId::MAX)
                } else {
                    // if it is not expandable, we need to store it and create new kernel
                    let dtype = graph.dtype(x);
                    kernels[kid].store(xt, View::contiguous(graph.shape(x)), dtype);
                    kernel_tensors[kid].insert(x, kernel_outputs[kid].remove(&x).unwrap());

                    let new_kid = free_kernels.pop_first().unwrap();
                    kernels[new_kid] = Kernel::leaf(shape, dtype);

                    assert!(kernels[new_kid].expand(shape));
                    let z = kernels[new_kid].max_id;
                    kernels[kid].max_id += 1;
                    kernels[kid].ops.push(Op::Move { z, x: 0, mop: MOp::Expa });

                    // new kernel now has both x and nid outputs
                    kernel_tensors[new_kid] = BTreeMap::from([(x, 0)]);
                    kernel_outputs[new_kid] = BTreeMap::from([(x, 0), (nid, 0)]);

                    if let Some(rc) = graph_rcs.get_mut(&x) {
                        *rc -= 1;
                        if *rc == 0 {
                            graph_rcs.remove(&x);
                            kernel_outputs[new_kid].remove(&x).unwrap();
                        }
                    }
                    (new_kid, kid)
                }
            }
            Node::Reshape { x } => {
                let (xt, mut kid) = get_kernel(x, &kernel_outputs);

                let shape = graph.shape(nid);
                //println!("Reshape node from {:?} to {:?}", graph.shape(x), shape);
                if !kernels[kid].reshape(shape) {
                    //println!("Could not be reshaped, storing");
                    // else create new kernel after storing results of previous kernel
                    let dtype = graph.dtype(x);
                    kernels[kid].store(xt, View::contiguous(graph.shape(x)), dtype);
                    kid = free_kernels.pop_first().unwrap();
                    kernels[kid] = Kernel::leaf(shape, dtype);
                }

                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;
                kernels[kid].ops.push(Op::Move { z, x: xt, mop: MOp::Resh });
                if let Some(rc) = graph_rcs.get_mut(&x) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&x);
                        kernel_outputs[kid].remove(&x).unwrap();
                    }
                }
                kernel_outputs[kid].insert(nid, z);
                println!("After reshape outputs: {:?}", kernel_outputs[kid]);
                (kid, KernelId::MAX)
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
                kernels[kid].ops.push(Op::Move { z, x: xt, mop: MOp::Padd });
                if let Some(rc) = graph_rcs.get_mut(&x) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&x);
                        kernel_outputs[kid].remove(&x).unwrap();
                    }
                }
                kernel_outputs[kid].insert(nid, z);
                (kid, KernelId::MAX)
            }
            Node::Permute { x } => {
                let (xt, kid) = get_kernel(x, &kernel_outputs);
                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;

                let axes = graph.axes(nid);
                kernels[kid].permute(axes);
                kernels[kid].ops.push(Op::Move { z, x: xt, mop: MOp::Perm });

                if let Some(rc) = graph_rcs.get_mut(&x) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&x);
                        kernel_outputs[kid].remove(&x).unwrap();
                    }
                }
                kernel_outputs[kid].insert(nid, z);
                (kid, KernelId::MAX)
            }
            Node::Reduce { x, rop } => {
                let (xt, kid) = get_kernel(x, &kernel_outputs);
                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;
                kernels[kid].reduce(xt, graph.shape(x), graph.axes(nid), graph.dtype(x), rop);
                if let Some(rc) = graph_rcs.get_mut(&x) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&x);
                        kernel_outputs[kid].remove(&x).unwrap();
                    }
                }
                kernel_outputs[kid].insert(nid, z);
                (kid, KernelId::MAX)
            }
            Node::Unary { x, uop } => {
                let (xt, kid) = get_kernel(x, &kernel_outputs);
                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;
                kernels[kid].ops.push(Op::Unary { z, x: xt, uop });
                if let Some(rc) = graph_rcs.get_mut(&x) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&x);
                        kernel_outputs[kid].remove(&x).unwrap();
                    }
                }
                kernel_outputs[kid].insert(nid, z);
                (kid, KernelId::MAX)
            }
            Node::Binary { x, y, bop } => {
                // x goes first, we delete y
                let (xt, kidx) = get_kernel(x, &kernel_outputs);
                kernels[kidx].max_id += 1;
                let (yt, kidy) = get_kernel(y, &kernel_outputs);

                // push ops from kernel y to kernel x, increasing
                // their ids by kernels[kidx].max_id and skipping
                // ops in both kernels
                let n = kernels[kidx].max_id;
                for op_i in 0..kernels[kidy].ops.len() {
                    if !matches!(kernels[kidy].ops[op_i], Op::Loop { .. })
                        || kernels[kidy].ops[op_i] != kernels[kidx].ops[op_i]
                    {
                        let new_op = match kernels[kidy].ops[op_i] {
                            Op::Loop { axis, len } => Op::Loop { axis, len },
                            Op::EndLoop => Op::EndLoop,
                            Op::Const { z, value, ref view } => {
                                Op::Const { z: z + n, value: value.clone(), view: view.clone() }
                            }
                            Op::Load { z, zscope, ref zview, xscope, ref xview, xdtype } => {
                                Op::Load {
                                    z: z + n,
                                    zscope,
                                    zview: zview.clone(),
                                    xscope,
                                    xview: xview.clone(),
                                    xdtype,
                                }
                            }
                            Op::Store { z, zscope, ref zview, zdtype, xscope, ref xview } => {
                                Op::Store {
                                    z: z + n,
                                    zscope,
                                    zview: zview.clone(),
                                    zdtype,
                                    xscope,
                                    xview: xview.clone(),
                                }
                            }
                            Op::Accumulator { z, rop, ref view, dtype } => {
                                Op::Accumulator { z: z + n, rop, view: view.clone(), dtype }
                            }
                            Op::Move { z, x, mop } => Op::Move { z: z + n, x: x + n, mop },
                            Op::Unary { z, x, uop } => Op::Unary { z: z + n, x: x + n, uop },
                            Op::Binary { z, x, y, bop } => {
                                Op::Binary { z: z + n, x: x + n, y: y + n, bop }
                            }
                            Op::Barrier { scope } => Op::Barrier { scope },
                        };
                        kernels[kidx].ops.push(new_op);
                    }
                }

                let y_tensors: BTreeMap<TensorId, TId> =
                    kernel_tensors[kidy].iter().map(|(t, id)| (*t, id + n)).collect();
                kernel_tensors[kidx].extend(y_tensors);

                kernels[kidx].max_id = kernels[kidx].max_id + kernels[kidy].max_id + 1;
                let z = kernels[kidx].max_id;
                kernels[kidx].ops.push(Op::Binary { z, x: xt, y: yt + n, bop });
                if let Some(rc) = graph_rcs.get_mut(&x) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&x);
                        kernel_outputs[kidx].remove(&x).unwrap();
                    }
                }
                if let Some(rc) = graph_rcs.get_mut(&y) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&y);
                        kernel_outputs[kidy].remove(&y).unwrap();
                        std::mem::swap(&mut Kernel::empty(), &mut kernels[kidy]);
                        free_kernels.insert(kidy);
                    }
                }
                // Notice we are not adding kernel outputs from kernel y to kernel x,
                // since those outputs may be used from kernel y, but not from kernel x.
                // This is why we keep kernel y alive.
                // Otherwise we could just delete kernel y and put everything into kernel x,
                // which seems like an interesting idea, but would it work? Likely not.
                kernel_outputs[kidx].insert(nid, z);
                (kidx, kidy)
            }
        };

        if to_eval.contains(&nid) {
            kernels[kidx].store(
                kernel_outputs[kidx][&nid],
                View::contiguous(graph.shape(nid)),
                graph.dtype(nid),
            );
            let id = kernel_outputs[kidx].remove(&nid).unwrap();
            kernel_tensors[kidx].insert(nid, id);
        }

        for kid in [kidy, kidx] {
            if kid != KernelId::MAX
                && !kernels[kid].ops.is_empty()
                && kernel_outputs[kid].is_empty()
            {
                // Delete kernel and dispatch it to device
                let mut kernel = Kernel::empty();
                std::mem::swap(&mut kernel, &mut kernels[kid]);
                free_kernels.insert(kid);

                if debug.sched() {
                    println!();
                    println!("Kernel tensors: {:?}", kernel_tensors[kid]);
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
                        if let Some(BufferId { buffer_id, .. }) = tensor_buffer_map.get(&tensor_id)
                        {
                            *buffer_id
                        } else {
                            // Allocate bytes for outputs
                            let buffer_id = memory_pool
                                .allocate(
                                    graph.shape(tensor_id).iter().product::<usize>()
                                        * graph.dtype(tensor_id).byte_size(),
                                )
                                .unwrap();
                            tensor_buffer_map
                                .insert(tensor_id, BufferId { memory_pool_id, buffer_id });
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
                    let ir_kernel = IRKernel::new(&optimized_kernel.ops, debug.asm());
                    device.compile(kernel.clone(), &ir_kernel, true)?;
                    device.launch(&kernel, memory_pool, &buffer_ids)?;
                }

                // add load kernels for all outputs of this kernel
                for op in kernel.ops {
                    if let Op::Store { z, .. } = op {
                        kernels.push(Kernel::leaf(graph.shape(nid), graph.dtype(nid)));
                        let tensor_id = kernel_tensors[kid]
                            .iter()
                            .find(|(_, tid)| **tid == z)
                            .unwrap()
                            .0
                            .clone();
                        let map = BTreeMap::from([(tensor_id, 0)]);
                        kernel_tensors.push(map.clone());
                        kernel_outputs.push(map);
                    }
                }
            }
        }
    }

    Ok(())
}

fn get_kernel(x: TensorId, kernel_outputs: &[BTreeMap<TensorId, TId>]) -> (TId, KernelId) {
    // perhaps chose kernel with fewest ops, or not? Is there any advantage to that?
    for (kid, outputs) in kernel_outputs.iter().enumerate() {
        if let Some(&x_tid) = outputs.get(&x) {
            return (x_tid, kid);
        }
    }
    unreachable!()
}
