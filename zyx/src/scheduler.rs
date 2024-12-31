//! Converts graph to kernels and schedules them to devices

use crate::{
    backend::Device,
    graph::Graph,
    kernel::{Kernel, MOp, Op, TId},
    node::Node,
    optimizer::Optimizer,
    runtime::Pool,
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
///
/// How it creates kernels?
/// Const and Leaf create new kernels.
/// Unary and movement ops are added to existing kernels,
/// expand, reshape, pad can store the old kernel and create new one if necessary
/// binary op fuses two kernels together and creates a new kernel, deleting the old kernels.
///
/// For now we remove tensor from outputs of a kernel immediatelly upon applying any op.
/// Later we can stop doing this and find a better way where we differentiate between ops
/// that require invalidation of kernel outputs, such as reduce ops and those that have no effect,
/// i. e. unary ops.
///
/// In the end check if tensor is in `to_eval` requiring immediate evaluation, if yes, then evaluate immediatelly.
///
/// If tensor is used elsewhere (rc > 1), then create rc - 1 copies of the kernel.
/// Potentially if this is expensive kernel or it requires many ops, then we might evaluate it immediatelly.
#[allow(clippy::cognitive_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn realize_graph(
    graph: &Graph,
    order: &[TensorId],
    to_eval: &BTreeSet<TensorId>,
    devs: &mut [Box<dyn Device>],
    mps: &mut [Pool],
    opt: &mut Optimizer,
    searches: usize,
    realized_nodes: &BTreeSet<TensorId>,
    debug: DebugMask,
) -> Result<(), ZyxError> {
    //let t = crate::Timer::new("realize_graph");

    // Unary and binary ops do not require duplication of kernels
    // TODO merge this with rcs in realize function
    let mut rcs = BTreeMap::new();
    for &nid in order {
        for p in graph[nid].parameters() {
            rcs.entry(p).and_modify(|rc| *rc += 1).or_insert(1u32);
        }
    }

    let mut num_kernels = 0;

    // Unfinished kernels represented by ops
    let mut kernels: Slab<Kernel> = Slab::with_capacity(30);

    if debug.sched() {
        println!("To eval: {to_eval:?}");
    }

    for nid in order.iter().copied() {
        if debug.sched() {
            println!(
                "ID({nid}): {:?}, sh: {:?}, rcs: {}, num kernels: {}",
                graph[nid],
                graph.shape(nid),
                rcs.get(&nid).copied().unwrap_or(0),
                kernels.len(),
            );
        }

        // In case of kernels which delete outputs we need to keep reference count
        // and not delete tensors from outputs if rc > 1
        let mut kid = if realized_nodes.contains(&nid) {
            kernels.push(Kernel::leaf(nid, graph.shape(nid), graph.dtype(nid)))
        } else {
            match graph[nid] {
                // All ops are merged except
                // Pad is not merged of kernel contains store
                // Reshape is not merged if reshaping reduce loops
                // Expand is not merged if expanding reduce kernel or kernel contains store
                // These rules can be later loosened using some heuristic.
                Node::Const { value } => kernels.push(Kernel::constant(nid, value)),
                Node::Leaf => unreachable!(),
                // Expand, reshape and pad are not always mergeable and thus can finish kernels.
                // All other ops are always mergeable.
                Node::Expand { x } => {
                    let (mut xt, mut kid) = get_kernel_min(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));
                    let shape = graph.shape(nid);
                    if kernels[kid].is_expandable(graph.shape(x)) {
                        // If it's gonna be used elsewhere, we need to copy this kernel,
                        // because expand invalidates outputs.
                        if kernels[kid].outputs.len() > 1 || rcs[&x] > 1 {
                            let mut new_kernel = kernels[kid].clone();
                            if rcs[&x] < 2 {
                                new_kernel.outputs.remove(&x);
                            }
                            kernels.push(new_kernel);
                        }
                    } else {
                        // if it is not expandable, we need to store it and create new kernel
                        sto(
                            &mut kernels,
                            &mut kid,
                            x,
                            graph,
                            devs,
                            mps,
                            opt,
                            searches,
                            &rcs,
                            debug,
                            &mut num_kernels,
                        )?;
                        if rcs[&x] > 1 {
                            kernels.push(kernels[kid].clone());
                        }
                        xt = 0;
                    }
                    kernels[kid].expand(shape);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(nid));
                    kernels[kid].max_id += 1;
                    let z = kernels[kid].max_id;
                    kernels[kid].ops.push(Op::Move { z, x: xt, mop: MOp::Expa });
                    kernels[kid].outputs.clear();
                    kernels[kid].outputs.insert(nid, z);
                    kid
                }
                Node::Reshape { x } => {
                    let (mut xt, mut kid) = get_kernel_min(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));
                    debug_assert!(kernels[kid].outputs.contains_key(&x));
                    //println!("Reshaping x = {x} to {nid}");
                    //kernels[kid].debug();
                    let shape = graph.shape(nid);
                    if kernels[kid].is_reshapable(shape) {
                        // If it's gonna be used elsewhere, we need to copy this kernel,
                        // because expand invalidates outputs.
                        if kernels[kid].outputs.len() > 1 || rcs[&x] > 1 {
                            let mut new_kernel = kernels[kid].clone();
                            if rcs[&x] < 2 {
                                new_kernel.outputs.remove(&x);
                            }
                            kernels.push(new_kernel);
                        }
                    } else {
                        // if it is not expandable, we need to store it and create new kernel
                        sto(
                            &mut kernels,
                            &mut kid,
                            x,
                            graph,
                            devs,
                            mps,
                            opt,
                            searches,
                            &rcs,
                            debug,
                            &mut num_kernels,
                        )?;
                        if rcs[&x] > 1 {
                            kernels.push(kernels[kid].clone());
                        }
                        xt = 0;
                    }
                    kernels[kid].reshape(shape);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(nid));
                    kernels[kid].max_id += 1;
                    let z = kernels[kid].max_id;
                    kernels[kid].ops.push(Op::Move { z, x: xt, mop: MOp::Resh });
                    kernels[kid].outputs.clear();
                    kernels[kid].outputs.insert(nid, z);
                    kid
                }
                Node::Pad { x } => {
                    let (mut xt, mut kid) = get_kernel_min(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));
                    let padding = graph.padding(nid);
                    if kernels[kid].is_paddable() {
                        // If it's gonna be used elsewhere, we need to copy this kernel,
                        // because expand invalidates outputs.
                        if kernels[kid].outputs.len() > 1 || rcs[&x] > 1 {
                            let mut new_kernel = kernels[kid].clone();
                            if rcs[&x] < 2 {
                                new_kernel.outputs.remove(&x);
                            }
                            kernels.push(new_kernel);
                        }
                    } else {
                        // if it is not paddable, we need to store it and create new kernel
                        sto(
                            &mut kernels,
                            &mut kid,
                            x,
                            graph,
                            devs,
                            mps,
                            opt,
                            searches,
                            &rcs,
                            debug,
                            &mut num_kernels,
                        )?;
                        if rcs[&x] > 1 {
                            kernels.push(kernels[kid].clone());
                        }
                        xt = 0;
                    }
                    kernels[kid].pad(padding);
                    kernels[kid].max_id += 1;
                    let z = kernels[kid].max_id;
                    kernels[kid].ops.push(Op::Move { z, x: xt, mop: MOp::Padd });
                    kernels[kid].outputs.clear();
                    kernels[kid].outputs.insert(nid, z);
                    kid
                }
                Node::Permute { x } => {
                    let (xt, kid) = get_kernel_min(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));
                    if kernels[kid].outputs.len() > 1 || rcs[&x] > 1 {
                        let mut new_kernel = kernels[kid].clone();
                        if rcs[&x] < 2 {
                            new_kernel.outputs.remove(&x);
                        }
                        kernels.push(new_kernel);
                    }
                    let axes = graph.axes(nid);
                    kernels[kid].permute(axes);
                    kernels[kid].max_id += 1;
                    let z = kernels[kid].max_id;
                    kernels[kid].ops.push(Op::Move { z, x: xt, mop: MOp::Perm });
                    kernels[kid].outputs.clear();
                    kernels[kid].outputs.insert(nid, z);
                    kid
                }
                Node::Reduce { x, rop } => {
                    let (xt, kid) = get_kernel_min(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));
                    if kernels[kid].outputs.len() > 1 || rcs[&x] > 1 {
                        let mut new_kernel = kernels[kid].clone();
                        if rcs[&x] < 2 {
                            new_kernel.outputs.remove(&x);
                        }
                        kernels.push(new_kernel);
                    }
                    kernels[kid].reduce(xt, graph.shape(x), graph.axes(nid), graph.dtype(x), rop);
                    let z = kernels[kid].max_id;
                    kernels[kid].outputs.clear();
                    kernels[kid].outputs.insert(nid, z);
                    kid
                }
                Node::Unary { x, uop } => {
                    let (xt, kid) = get_kernel_max(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));
                    kernels[kid].max_id += 1;
                    let z = kernels[kid].max_id;
                    kernels[kid].ops.push(Op::Unary { z, x: xt, uop });
                    if rcs[&x] < 2 {
                        kernels[kid].outputs.remove(&x);
                    }
                    kernels[kid].outputs.insert(nid, z);
                    kid
                }
                Node::Binary { x, y, bop } => {
                    // x goes first, delete y
                    let (xt, kidx) = get_kernel_max(x, &kernels);
                    let (yt, kidy) = get_kernel_max(y, &kernels);

                    debug_assert_eq!(kernels[kidx].shape(), graph.shape(x));
                    debug_assert_eq!(kernels[kidy].shape(), graph.shape(y));

                    #[allow(clippy::branches_sharing_code)]
                    let kid = if kidx == kidy {
                        kernels[kidx].max_id += 1;
                        let z = kernels[kidx].max_id;
                        kernels[kidx].ops.push(Op::Binary { z, x: xt, y: yt, bop });
                        kernels[kidx].outputs.insert(nid, z);
                        kidx
                    } else {
                        // we delete kidy (could by also kidx) and put everything in kidx
                        let n = kernels[kidx].max_id + 1;
                        let Kernel { ops, tensors, outputs, max_id } =
                            kernels.remove(kidy).unwrap();
                        for (i, op) in ops.into_iter().enumerate() {
                            if !(matches!(op, Op::Loop { .. }) && op == kernels[kidx].ops[i]) {
                                kernels[kidx].ops.push(match op {
                                    Op::Loop { axis, len } => Op::Loop { axis, len },
                                    Op::EndLoop => Op::EndLoop,
                                    Op::Const { z, value, ref view } => {
                                        Op::Const { z: z + n, value, view: view.clone() }
                                    }
                                    Op::Load {
                                        z,
                                        zscope,
                                        ref zview,
                                        xscope,
                                        ref xview,
                                        xdtype,
                                    } => Op::Load {
                                        z: z + n,
                                        zscope,
                                        zview: zview.clone(),
                                        xscope,
                                        xview: xview.clone(),
                                        xdtype,
                                    },
                                    Op::Store {
                                        z,
                                        zscope,
                                        ref zview,
                                        zdtype,
                                        xscope,
                                        ref xview,
                                    } => Op::Store {
                                        z: z + n,
                                        zscope,
                                        zview: zview.clone(),
                                        zdtype,
                                        xscope,
                                        xview: xview.clone(),
                                    },
                                    Op::Accumulator { z, rop, dtype } => {
                                        Op::Accumulator { z: z + n, rop, dtype }
                                    }
                                    Op::Move { z, x, mop } => Op::Move { z: z + n, x: x + n, mop },
                                    Op::Unary { z, x, uop } => {
                                        Op::Unary { z: z + n, x: x + n, uop }
                                    }
                                    Op::Binary { z, x, y, bop } => {
                                        Op::Binary { z: z + n, x: x + n, y: y + n, bop }
                                    }
                                    Op::Barrier { scope } => Op::Barrier { scope },
                                });
                            }
                        }
                        kernels[kidx].max_id += max_id + 2;
                        let z = kernels[kidx].max_id;
                        kernels[kidx].ops.push(Op::Binary { z, x: xt, y: yt + n, bop });
                        kernels[kidx]
                            .tensors
                            .extend(tensors.into_iter().map(|(tid, t)| (tid + n, t)));
                        kernels[kidx].outputs.extend(outputs.iter().map(|(t, tid)| (*t, tid + n)));
                        kernels[kidx].outputs.insert(nid, z);
                        kidx
                    };
                    if x == y && rcs[&x] < 3 {
                        kernels[kidx].outputs.remove(&x);
                    } else {
                        if rcs[&x] < 2 {
                            kernels[kidx].outputs.remove(&x);
                        }
                        if rcs[&y] < 2 {
                            kernels[kidx].outputs.remove(&y);
                        }
                    }
                    kid
                }
            }
        };

        //for kernel in kernels.values() { kernel.debug(); println!(); }

        for param in graph[nid].parameters() {
            if let Some(rc) = rcs.get_mut(&param) {
                *rc -= 1;
                if *rc == 0 {
                    rcs.remove(&param);
                }
            }
        }

        debug_assert_eq!(kernels[kid].shape(), graph.shape(nid));

        if to_eval.contains(&nid) {
            sto(
                &mut kernels,
                &mut kid,
                nid,
                graph,
                devs,
                mps,
                opt,
                searches,
                &rcs,
                debug,
                &mut num_kernels,
            )?;
        }
    }

    #[cfg(debug_assertions)]
    if kernels.len() != 0 {
        for kernel in kernels.values() {
            kernel.debug();
            println!();
        }
        panic!("Kernels in scheduler are not empty.");
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sto(
    kernels: &mut Slab<Kernel>,
    kid: &mut u32,
    x: u32,
    graph: &Graph,
    devices: &mut [Box<dyn Device>],
    mps: &mut [Pool],
    optimizer: &mut Optimizer,
    searches: usize,
    rcs: &BTreeMap<u32, u32>,
    debug: DebugMask,
    num_kernels: &mut usize,
) -> Result<(), ZyxError> {
    if let Some(tensors) = kernels[*kid].store(
        x,
        graph,
        devices,
        mps,
        optimizer,
        searches,
        debug,
        num_kernels,
    )? {
        kernels.remove(*kid).unwrap();
        for kernel in kernels.values_mut() {
            for tid in &tensors {
                kernel.outputs.remove(tid);
            }
        }
        let mut kid_ = 0u32;
        while kid_ < kernels.len().try_into().unwrap() {
            if let Some(kernel) = kernels.get(kid_) {
                if kernel.outputs.is_empty() {
                    kernels.remove(kid_).unwrap();
                }
            }
            kid_ += 1;
        }

        /*println!("After cleanup of stored {x}");
        for kernel in kernels.values() {
            kernel.debug();
            println!();
        }*/

        //println!("RCS: {rcs:?}");
        //println!("tensors: {tensors:?}");
        for tid in tensors {
            if rcs.contains_key(&tid) {
                let tkid = kernels.push(Kernel::leaf(tid, graph.shape(tid), graph.dtype(tid)));
                if tid == x {
                    *kid = tkid;
                }
            }
        }
        /*for kernel in kernels.values() {
            kernel.debug();
        }*/
    }
    Ok(())
}

// Choose kernel with most outputs (binary, unary)
fn get_kernel_max(x: TensorId, kernels: &Slab<Kernel>) -> (TId, KernelId) {
    // TODO perhaps we can optimize this more?
    kernels
        .iter()
        .filter(|(_, kernel)| kernel.outputs.contains_key(&x))
        .max_by_key(|(_, kernel)| kernel.outputs.len())
        .map_or_else(
            || {
                println!("Current kernels:");
                for kernel in kernels.values() {
                    println!();
                    kernel.debug();
                }
                println!();
                panic!();
            },
            |(kid, kernel)| (*kernel.outputs.get(&x).unwrap(), kid),
        )
}

// Choose kernel with fewest outputs (expand, reshape, pad, permute, reduce)
fn get_kernel_min(x: TensorId, kernels: &Slab<Kernel>) -> (TId, KernelId) {
    kernels
        .iter()
        .filter(|(_, kernel)| kernel.outputs.contains_key(&x))
        .min_by_key(|(_, kernel)| kernel.outputs.len())
        .map_or_else(
            || {
                println!("Current kernels:");
                for kernel in kernels.values() {
                    kernel.debug();
                }
                println!();
                panic!();
            },
            |(kid, kernel)| (*kernel.outputs.get(&x).unwrap(), kid),
        )
}
