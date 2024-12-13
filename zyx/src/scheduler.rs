//! Converts graph to kernels and schedules them to devices

use crate::{
    backend::{BufferId, Device, MemoryPool},
    graph::Graph,
    kernel::{Kernel, MOp, Op, TId},
    node::Node,
    optimizer::Optimizer,
    slab::{Id, Slab},
    tensor::TensorId,
    DebugMask, ZyxError,
};
use std::collections::{BTreeMap, BTreeSet};

type KernelId = Id;

/// Convert graph into kernels and schedule them to devices.
/// This function needs to be optimized a lot, because it always needs to run faster than async launched kernels.
/// So no more than 10 microseconds per kernel.
/// If the scheduler is too slow more complex caching can be introduce.
pub(super) fn realize_graph(
    graph: &Graph,
    order: &[TensorId],
    to_eval: &BTreeSet<TensorId>,
    devices: &mut [Device],
    mps: &mut [MemoryPool],
    tbm: &mut BTreeMap<TensorId, BufferId>,
    optimizer: &mut Optimizer,
    searches: usize,
    debug: DebugMask,
) -> Result<(), ZyxError> {
    // TODO perhaps we can get graph rcs from runtime realize frunction instead of this...
    let mut graph_rcs: BTreeMap<TensorId, u32> = BTreeMap::new();
    let mut params = to_eval.clone();
    while let Some(param) = params.pop_last() {
        graph_rcs.entry(param).and_modify(|rc| *rc += 1).or_insert_with(|| {
            if !tbm.contains_key(&param) {
                params.extend(graph[param].parameters());
            }
            1
        });
    }

    // Unfinished kernels represented by ops
    let mut kernels: Slab<Kernel> = Slab::with_capacity(100);

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
        let kid = match graph[nid] {
            // All ops are merged except
            // Pad is not merged of kernel contains store
            // Reshape is not merged if reshaping reduce loops
            // Expand is not merged if expanding reduce kernel or kernel contains store
            // These rules will be later loosened using some heuristic
            Node::Const { value } => kernels.push(Kernel::constant(nid, value)),
            Node::Leaf => kernels.push(Kernel::leaf(nid, graph.shape(nid), graph.dtype(nid))),
            // Expand, reshape and pad are not always mergeable and thus can finish kernels.
            // All other ops are always mergeable.
            Node::Expand { x } => {
                let (mut xt, mut kid) = get_kernel(x, &kernels);
                let shape = graph.shape(nid);
                if !kernels[kid].expand(shape) {
                    // if it is not expandable, we need to store it and create new kernel
                    let dtype = graph.dtype(x);
                    if let Some(tensors) = kernels[kid]
                        .store(nid, graph, devices, mps, tbm, optimizer, searches, debug)?
                    {
                        kernels.remove(kid).unwrap();
                        for tid in tensors {
                            kernels.push(Kernel::leaf(tid, graph.shape(tid), graph.dtype(tid)));
                        }
                    }
                    kid = kernels.push(Kernel::leaf(nid, shape, dtype));
                    assert!(kernels[kid].expand(shape));
                    xt = 0;
                }
                let z = kernels[kid].max_id;
                kernels[kid].max_id += 1;
                kernels[kid].ops.push(Op::Move { z, x: xt, mop: MOp::Expa });
                kernels[kid].outputs.insert(nid, z);
                if let Some(rc) = graph_rcs.get_mut(&x) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&x);
                        kernels[kid].outputs.remove(&x).unwrap();
                    }
                }
                kid
            }
            Node::Reshape { x } => {
                let (mut xt, mut kid) = get_kernel(x, &kernels);
                let shape = graph.shape(nid);
                if !kernels[kid].reshape(shape) {
                    // if it is not expandable, we need to store it and create new kernel
                    let dtype = graph.dtype(x);
                    if let Some(tensors) = kernels[kid]
                        .store(nid, graph, devices, mps, tbm, optimizer, searches, debug)?
                    {
                        kernels.remove(kid).unwrap();
                        for tid in tensors {
                            kernels.push(Kernel::leaf(tid, graph.shape(tid), graph.dtype(tid)));
                        }
                    }
                    kid = kernels.push(Kernel::leaf(nid, shape, dtype));
                    assert!(kernels[kid].reshape(shape));
                    xt = 0;
                }
                let z = kernels[kid].max_id;
                kernels[kid].max_id += 1;
                kernels[kid].ops.push(Op::Move { z, x: xt, mop: MOp::Resh });
                kernels[kid].outputs.insert(nid, z);
                if let Some(rc) = graph_rcs.get_mut(&x) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&x);
                        kernels[kid].outputs.remove(&x).unwrap();
                    }
                }
                kid
            }
            Node::Pad { x } => {
                let (mut xt, mut kid) = get_kernel(x, &kernels);
                let shape = graph.shape(nid);
                let padding = graph.padding(nid);
                if !kernels[kid].pad(padding) {
                    // if it is not expandable, we need to store it and create new kernel
                    let dtype = graph.dtype(x);
                    if let Some(tensors) = kernels[kid]
                        .store(nid, graph, devices, mps, tbm, optimizer, searches, debug)?
                    {
                        kernels.remove(kid).unwrap();
                        for tid in tensors {
                            kernels.push(Kernel::leaf(tid, graph.shape(tid), graph.dtype(tid)));
                        }
                    }
                    kid = kernels.push(Kernel::leaf(nid, shape, dtype));
                    assert!(kernels[kid].pad(padding));
                    xt = 0;
                }
                let z = kernels[kid].max_id;
                kernels[kid].max_id += 1;
                kernels[kid].ops.push(Op::Move { z, x: xt, mop: MOp::Padd });
                kernels[kid].outputs.insert(nid, z);
                if let Some(rc) = graph_rcs.get_mut(&x) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&x);
                        kernels[kid].outputs.remove(&x).unwrap();
                    }
                }
                kid
            }
            Node::Permute { x } => {
                let (xt, kid) = get_kernel(x, &kernels);
                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;

                let axes = graph.axes(nid);
                kernels[kid].permute(axes);
                kernels[kid].ops.push(Op::Move { z, x: xt, mop: MOp::Perm });

                if let Some(rc) = graph_rcs.get_mut(&x) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&x);
                        kernels[kid].outputs.remove(&x).unwrap();
                    }
                }
                kernels[kid].outputs.insert(nid, z);
                kid
            }
            Node::Reduce { x, rop } => {
                let (xt, kid) = get_kernel(x, &kernels);
                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;
                kernels[kid].reduce(xt, graph.shape(x), graph.axes(nid), graph.dtype(x), rop);
                if let Some(rc) = graph_rcs.get_mut(&x) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&x);
                        kernels[kid].outputs.remove(&x).unwrap();
                    }
                }
                kernels[kid].outputs.insert(nid, z);
                kid
            }
            Node::Unary { x, uop } => {
                let (xt, kid) = get_kernel(x, &kernels);
                kernels[kid].max_id += 1;
                let z = kernels[kid].max_id;
                kernels[kid].ops.push(Op::Unary { z, x: xt, uop });
                if let Some(rc) = graph_rcs.get_mut(&x) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&x);
                        kernels[kid].outputs.remove(&x).unwrap();
                    }
                }
                kernels[kid].outputs.insert(nid, z);
                kid
            }
            Node::Binary { x, y, bop } => {
                // x goes first, we delete y
                let (xt, kidx) = get_kernel(x, &kernels);
                kernels[kidx].max_id += 1;
                let (yt, kidy) = get_kernel(y, &kernels);

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
                    kernels[kidy].tensors.iter().map(|(t, id)| (*t, id + n)).collect();
                kernels[kidx].tensors.extend(y_tensors);

                kernels[kidx].max_id = kernels[kidx].max_id + kernels[kidy].max_id + 1;
                let z = kernels[kidx].max_id;
                kernels[kidx].ops.push(Op::Binary { z, x: xt, y: yt + n, bop });
                if let Some(rc) = graph_rcs.get_mut(&x) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&x);
                        kernels[kidx].outputs.remove(&x).unwrap();
                    }
                }
                if let Some(rc) = graph_rcs.get_mut(&y) {
                    *rc -= 1;
                    if *rc == 0 {
                        graph_rcs.remove(&y);
                        kernels[kidy].outputs.remove(&y).unwrap();
                        kernels.remove(kidy).unwrap();
                    }
                }
                // Notice we are not adding kernel outputs from kernel y to kernel x,
                // since those outputs may be used from kernel y, but not from kernel x.
                // This is why we keep kernel y alive.
                // Otherwise we could just delete kernel y and put everything into kernel x,
                // which seems like an interesting idea, but would it work? Likely not.
                kernels[kidx].outputs.insert(nid, z);
                kidx
            }
        };

        if to_eval.contains(&nid) {
            if let Some(tensors) =
                kernels[kid].store(nid, graph, devices, mps, tbm, optimizer, searches, debug)?
            {
                kernels.remove(kid).unwrap();
                for tid in tensors {
                    kernels.push(Kernel::leaf(tid, graph.shape(tid), graph.dtype(tid)));
                }
            }
        }
    }

    Ok(())
}

fn get_kernel(x: TensorId, kernels: &Slab<Kernel>) -> (TId, KernelId) {
    // perhaps chose kernel with fewest ops, or not? Is there any advantage to that?
    for (kid, kernel) in kernels.values().enumerate() {
        if let Some(&x_tid) = kernel.outputs.get(&x) {
            return (x_tid, kid.try_into().unwrap());
        }
    }
    unreachable!()
}
