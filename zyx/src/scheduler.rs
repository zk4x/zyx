//! Converts graph to kernels and schedules them to devices

use crate::{
    backend::Device, graph::Graph, ir::Scope, kernel::{Kernel, MOp, Op, TId}, node::Node, optimizer::Optimizer, runtime::Pool, tensor::TensorId, view::View, DebugMask, Timer, ZyxError
};
use std::collections::{BTreeMap, BTreeSet};

type KernelId = usize;

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
    devices: &mut [Box<dyn Device>],
    memory_pools: &mut [Pool],
    optimizer: &mut Optimizer,
    search_iters: usize,
    realized_nodes: &BTreeSet<TensorId>,
    debug: DebugMask,
) -> Result<(), ZyxError> {
    //let t = crate::Timer::new("realize_graph");
    let begin = std::time::Instant::now();

    // Unary and binary ops do not require duplication of kernels
    // TODO merge this with rcs in realize function
    let mut rcs = BTreeMap::new();
    for &nid in order {
        for p in graph[nid].parameters() {
            rcs.entry(p).and_modify(|rc| *rc += 1).or_insert(1u32);
        }
    }

    // Unfinished kernels represented by ops
    let mut kernels: Vec<Kernel> = Vec::with_capacity(100);

    /*if debug.sched() {
        println!("To eval: {to_eval:?}");
    }*/

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
        let mut kid: KernelId = if realized_nodes.contains(&nid) {
            kernels.push(Kernel::leaf(nid, graph.shape(nid), graph.dtype(nid)));
            kernels.len() - 1
        } else {
            match graph[nid] {
                // All ops are merged except
                // Pad is not merged of kernel contains store
                // Reshape is not merged if reshaping reduce loops
                // Expand is not merged if expanding reduce kernel or kernel contains store
                // These rules can be later loosened using some heuristic.
                Node::Const { value } => {
                    let _timer = Timer::new("const");
                    kernels.push(Kernel::constant(nid, value));
                    kernels.len() - 1
                }
                Node::Leaf => unreachable!(),
                // Expand, reshape and pad are not always mergeable and thus can finish kernels.
                // All other ops are always mergeable.
                Node::Expand { x } => {
                    let _timer = Timer::new("expand");
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
                        store(&mut kernels, &mut kid, x, graph, &rcs);
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
                    let _timer = Timer::new("reshape");
                    let (mut xt, mut kid) = get_kernel_min(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));
                    debug_assert!(kernels[kid].outputs.contains_key(&x));
                    //println!("Reshaping x = {x} to {nid}");
                    //kernels[kid].debug();
                    let shape = graph.shape(nid);
                    if kernels[kid].is_reshapable(shape) {
                        // If it's gonna be used elsewhere, we need to copy this kernel,
                        // because reshape invalidates outputs.
                        if kernels[kid].outputs.len() > 1 || rcs[&x] > 1 {
                            let mut new_kernel = kernels[kid].clone();
                            if rcs[&x] < 2 {
                                new_kernel.outputs.remove(&x);
                            }
                            kernels.push(new_kernel);
                        }
                    } else {
                        // if it is not expandable, we need to store it and create new kernel
                        store(&mut kernels, &mut kid, x, graph, &rcs);
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
                    let _timer = Timer::new("pad");
                    let (mut xt, mut kid) = get_kernel_min(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));
                    let padding = graph.padding(nid);
                    if kernels[kid].is_paddable() {
                        // If it's gonna be used elsewhere, we need to copy this kernel,
                        // because padding invalidates outputs.
                        if kernels[kid].outputs.len() > 1 || rcs[&x] > 1 {
                            // TODO we may want to evaluate this kernel instead of cloning
                            // if this kernels[kid] is too large.
                            let mut new_kernel = kernels[kid].clone();
                            if rcs[&x] < 2 {
                                new_kernel.outputs.remove(&x);
                            }
                            kernels.push(new_kernel);
                        }
                    } else {
                        // if it is not paddable, we need to store it and create new kernel
                        store(&mut kernels, &mut kid, x, graph, &rcs);
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
                    let _timer = Timer::new("permute");
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
                    let _timer = Timer::new("reduce");
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
                    let _timer = Timer::new("unary");
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
                    let _timer = Timer::new("binary");
                    // x goes first, delete y
                    let (xt, mut kidx) = get_kernel_max(x, &kernels);
                    let (yt, kidy) = get_kernel_max(y, &kernels);

                    debug_assert_eq!(kernels[kidx].shape(), graph.shape(x));
                    debug_assert_eq!(kernels[kidy].shape(), graph.shape(y));

                    debug_assert!(!(kernels[kidx].depends_on.contains(&kidy) && kernels[kidy].depends_on.contains(&kidx)));

                    /*if kernels[kidx].depends_on.contains(&kidy) {
                        store(&mut kernels, &mut kidy, y, &graph, &rcs);
                    } else if kernels[kidy].depends_on.contains(&kidx) {
                        store(&mut kernels, &mut kidx, x, &graph, &rcs);
                    }*/

                    // TODO test for cyclical dependency between kidx and kidy
                    // and realize the kernel with more ops in order to break that dependency cycle.

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
                        let Kernel { ops, tensors, outputs, max_id, depends_on } =
                            kernels.remove(kidy);

                        // After removing kidy, we have to decrease all depends_on
                        if kidx > kidy {
                            kidx -= 1;
                        }
                        for kernel in &mut kernels {
                            let depends_on = kernel.depends_on.clone();
                            for kid in depends_on {
                                if kid > kidy {
                                    kernel.depends_on.insert(kid-1);
                                    kernel.depends_on.remove(&kid);
                                }
                            }
                        }

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
                        kernels[kidx]
                            .tensors
                            .extend(tensors.into_iter().map(|(tid, t)| (tid + n, t)));
                        kernels[kidx].outputs.extend(outputs.iter().map(|(t, tid)| (*t, tid + n)));
                        kernels[kidx].depends_on.extend(depends_on);
                        let z = kernels[kidx].max_id;
                        kernels[kidx].ops.push(Op::Binary { z, x: xt, y: yt + n, bop });
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
            store(&mut kernels, &mut kid, nid, graph, &rcs);
        }
    }

    //if debug.sched() {
    let taken  = begin.elapsed().as_micros();
    println!("Scheduled {} kernels, scheduling took {taken}us", kernels.len());
    //}

    // Launch all kernels
    for _ in 0..kernels.len() {
        let mut evaluated = BTreeSet::new();
        for (kid, kernel) in kernels.iter().enumerate() {
            if kernel.depends_on.is_empty() {
                kernel.launch(graph, devices, memory_pools, optimizer, search_iters, debug)?;
                evaluated.insert(kid);
            }
        }
        if evaluated.is_empty() {
            break;
        }
        for kernel in &mut kernels {
            kernel.depends_on.retain(|x| !evaluated.contains(x));
        }
    }

    /*#[cfg(debug_assertions)]
    if kernels.len() != 0 {
        /*for kernel in kernels.values() {
            kernel.debug();
            println!();
        }*/
        panic!("Kernels in scheduler are not empty.");
    }*/

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn store(
    kernels: &mut Vec<Kernel>,
    kid: &mut KernelId,
    nid: TensorId,
    graph: &Graph,
    rcs: &BTreeMap<TensorId, u32>,
) {
    // Add store op to kernel
    let zview = View::contiguous(graph.shape(nid));
    let zdtype = graph.dtype(nid);
    let z = kernels[*kid].outputs[&nid];
    if let Some(&Op::Store { z: nz, zview: ref nzview, .. }) = kernels[*kid].ops.last() {
        if z == nz && &zview == nzview {
            unreachable!();
        }
    }
    debug_assert!(zview.numel() < 1024 * 1024 * 1024, "Too big store.");
    let store_op = Op::Store {
        z,
        zview,
        zscope: Scope::Global,
        zdtype,
        xscope: Scope::Register,
        xview: View::none(),
    };
    kernels[*kid].ops.push(store_op);
    let tid = kernels[*kid].outputs.remove(&nid).unwrap();
    kernels[*kid].tensors.insert(tid, nid);

    // Remove this output from all other kernels that produce it. It will be used as a load kernel from now on.
    for kernel in kernels.iter_mut() {
        kernel.outputs.remove(&nid);
    }

    // If this is non user defined realization (not in to_eval), then it will be needed by at least one other kernel
    if rcs.contains_key(&nid) {
        kernels.push(Kernel::leaf(nid, graph.shape(nid), graph.dtype(nid)));
        let nkid = kernels.len() - 1;
        kernels[nkid].depends_on.insert(*kid);
        *kid = nkid;
    }

    /*println!("After cleanup of stored {x}");
    for kernel in kernels.values() {
        kernel.debug();
        println!();
    }*/

    //println!("RCS: {rcs:?}");
    //println!("tensors: {tensors:?}");

    /*for kernel in kernels.values() {
        kernel.debug();
    }*/
}

// Choose kernel with most outputs (binary, unary)
fn get_kernel_max(x: TensorId, kernels: &[Kernel]) -> (TId, KernelId) {
    // TODO perhaps we can optimize this more?
    kernels
        .iter()
        .enumerate()
        .filter(|(_, kernel)| kernel.outputs.contains_key(&x))
        .max_by_key(|(_, kernel)| kernel.outputs.len())
        .map_or_else(
            || {
                println!("Current kernels:");
                for kernel in kernels {
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
fn get_kernel_min(x: TensorId, kernels: &[Kernel]) -> (TId, KernelId) {
    kernels
        .iter()
        .enumerate()
        .filter(|(_, kernel)| kernel.outputs.contains_key(&x))
        .min_by_key(|(_, kernel)| kernel.outputs.len())
        .map_or_else(
            || {
                println!("Current kernels:");
                for kernel in kernels {
                    kernel.debug();
                }
                println!();
                panic!();
            },
            |(kid, kernel)| (*kernel.outputs.get(&x).unwrap(), kid),
        )
}
