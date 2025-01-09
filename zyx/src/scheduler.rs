//! Converts graph to kernels and schedules them to devices

use crate::{
    backend::Device, graph::Graph, ir::Scope, kernel::{Kernel, Op, TId}, node::Node, optimizer::Optimizer, runtime::Pool, slab::{Id, Slab}, tensor::TensorId, view::View, DType, DebugMask, Map, Set, ZyxError
};
use std::collections::BTreeSet;

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
    // RCS are only ref counts from parameters, excluding ref counts from being in to_eval/user rcs
    mut rcs: Map<TensorId, u32>,
    to_eval: &Set<TensorId>,
    devices: &mut [Box<dyn Device>],
    memory_pools: &mut [Pool],
    optimizer: &mut Optimizer,
    search_iters: usize,
    mut realized_nodes: Set<TensorId>,
    debug: DebugMask,
) -> Result<(), ZyxError> {
    //let t = crate::Timer::new("realize_graph");
    let begin = std::time::Instant::now();

    // Unary and binary ops do not require duplication of kernels

    // Unfinished kernels represented by ops
    let mut kernels: Slab<Kernel> = Slab::with_capacity(500);

    if debug.sched() {
        println!("To schedule: {} tensors, to eval: {to_eval:?}", order.len());
    }

    /*let mut expa_u = 0;
    let mut resh_u = 0;
    let mut pad_u = 0;
    let mut red_u = 0;
    let mut perm_u = 0;*/
    /*for &nid in order {
        println!(
            "ID({nid}): {:?}, sh: {:?}, rcs: {}, num kernels: {}",
            graph[nid],
            graph.shape(nid),
            rcs.get(&nid).copied().unwrap_or(0),
            kernels.len(),
        );
    }*/

    for &nid in order {
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
        let kid: KernelId = if realized_nodes.contains(&nid) {
            //let _timer = Timer::new("leaf");
            //kernels.len() - 1
            kernels.push(Kernel::leaf(
                nid,
                graph.shape(nid),
                graph.dtype(nid),
                BTreeSet::new(),
            ))
        } else {
            match graph[nid] {
                // All ops are merged except
                // Pad is not merged of kernel contains store (could be merged eventually)
                // Expand is not merged if expanding reduce kernel or kernel contains store
                // These rules can be later loosened using some heuristic.
                Node::Const { value } => {
                    //let _timer = Timer::new("const");
                    //kernels.len() - 1
                    kernels.push(Kernel::constant(nid, value))
                }
                Node::Leaf { .. } => {
                    let realized_nodes: Set<TensorId> = memory_pools
                        .iter()
                        .map(|pool| pool.buffer_map.keys())
                        .flatten()
                        .copied()
                        .collect();
                    if !realized_nodes.contains(&nid) {
                        println!("tensor {nid} not in realized nodes and not in buffers");
                    }
                    unreachable!();
                }
                // Expand and pad are not always mergeable and thus can finish kernels.
                // All other ops are always mergeable.
                Node::Expand { x } => {
                    //let _timer = Timer::new("expand");
                    let (mut xt, mut kid) = get_kernel_min(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));
                    let x_shape = graph.shape(x);
                    if kernels[kid].is_expandable(x_shape) {
                        if kernels[kid].outputs.len() > 1 || rcs[&x] > 1 {
                            if kernels[kid].is_small() {
                                let mut new_kernel = kernels[kid].clone();
                                if rcs[&x] < 2 {
                                    new_kernel.outputs.remove(&x);
                                }
                                kernels.push(new_kernel);
                            } else {
                                let x_dtype = graph.dtype(x);
                                store(&mut kernels, kid, x, x_shape, x_dtype);
                                let nkid = kernels.push(Kernel::leaf(
                                    x,
                                    x_shape,
                                    x_dtype,
                                    BTreeSet::from([kid]),
                                ));
                                xt = 0;
                                kid = nkid;
                            }
                        }
                    } else {
                        let x_dtype = graph.dtype(x);
                        store(&mut kernels, kid, x, x_shape, x_dtype);
                        let nkid =
                            kernels.push(Kernel::leaf(x, x_shape, x_dtype, BTreeSet::from([kid])));
                        xt = 0;
                        kid = nkid;
                    }

                    let shape = graph.shape(nid);
                    kernels[kid].expand(shape);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(nid));
                    kernels[kid].outputs.clear();
                    kernels[kid].outputs.insert(nid, xt);
                    kid
                }
                Node::Pad { x } => {
                    //let _timer = Timer::new("pad");
                    let (mut xt, mut kid) = get_kernel_min(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));
                    let padding = graph.padding(nid);

                    // TODO make kernel always paddable by padding store view, thus removing the else branch
                    if kernels[kid].is_paddable() {
                        if kernels[kid].outputs.len() > 1 || rcs[&x] > 1 {
                            if kernels[kid].is_small() {
                                let mut new_kernel = kernels[kid].clone();
                                if rcs[&x] < 2 {
                                    new_kernel.outputs.remove(&x);
                                }
                                kernels.push(new_kernel);
                            } else {
                                let x_dtype = graph.dtype(x);
                                let x_shape = graph.shape(x);
                                store(&mut kernels, kid, x, x_shape, x_dtype);
                                let nkid = kernels.push(Kernel::leaf(
                                    x,
                                    x_shape,
                                    x_dtype,
                                    BTreeSet::from([kid]),
                                ));
                                xt = 0;
                                kid = nkid;
                            }
                        }
                    } else {
                        let x_dtype = graph.dtype(x);
                        let x_shape = graph.shape(x);
                        store(&mut kernels, kid, x, x_shape, x_dtype);
                        let nkid =
                            kernels.push(Kernel::leaf(x, x_shape, x_dtype, BTreeSet::from([kid])));
                        xt = 0;
                        kid = nkid;
                    }

                    kernels[kid].pad(padding);
                    kernels[kid].outputs.clear();
                    kernels[kid].outputs.insert(nid, xt);
                    kid
                }
                Node::Reshape { x } => {
                    //let _timer = Timer::new("reshape");
                    let (mut xt, mut kid) = get_kernel_min(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));
                    debug_assert!(kernels[kid].outputs.contains_key(&x));
                    //println!("Reshaping x = {x} to {nid}");
                    //kernels[kid].debug();
                    let shape = graph.shape(nid);

                    if kernels[kid].outputs.len() > 1 || rcs[&x] > 1 {
                        if kernels[kid].is_small() {
                            let mut new_kernel = kernels[kid].clone();
                            if rcs[&x] < 2 {
                                new_kernel.outputs.remove(&x);
                            }
                            kernels.push(new_kernel);
                        } else {
                            let x_shape = graph.shape(x);
                            let x_dtype = graph.dtype(x);
                            store(&mut kernels, kid, x, x_shape, x_dtype);
                            let nkid = kernels.push(Kernel::leaf(
                                x,
                                x_shape,
                                x_dtype,
                                BTreeSet::from([kid]),
                            ));
                            xt = 0;
                            kid = nkid;
                        }
                    }

                    kernels[kid].reshape_unchecked(shape);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(nid));
                    kernels[kid].outputs.clear();
                    kernels[kid].outputs.insert(nid, xt);
                    kid
                }
                Node::Permute { x } => {
                    //let _timer = Timer::new("permute");
                    let (mut xt, mut kid) = get_kernel_min(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));
                    if kernels[kid].outputs.len() > 1 || rcs[&x] > 1 {
                        if kernels[kid].is_small() {
                            let mut new_kernel = kernels[kid].clone();
                            if rcs[&x] < 2 {
                                new_kernel.outputs.remove(&x);
                            }
                            kernels.push(new_kernel);
                        } else {
                            let x_shape = graph.shape(x);
                            let x_dtype = graph.dtype(x);
                            store(&mut kernels, kid, x, x_shape, x_dtype);
                            let nkid = kernels.push(Kernel::leaf(
                                x,
                                x_shape,
                                x_dtype,
                                BTreeSet::from([kid]),
                            ));
                            xt = 0;
                            kid = nkid;
                        }
                    }

                    let axes = graph.axes(nid);
                    kernels[kid].permute(axes);
                    kernels[kid].outputs.clear();
                    kernels[kid].outputs.insert(nid, xt);
                    kid
                }
                Node::Reduce { x, rop } => {
                    //let _timer = Timer::new("reduce");
                    let (xt, kid) = get_kernel_min(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));

                    if kernels[kid].outputs.len() > 1 || rcs[&x] > 1 {
                        // If the kernel is not too complex, we can just clone it
                        let mut new_kernel = kernels[kid].clone();
                        if rcs[&x] < 2 {
                            new_kernel.outputs.remove(&x);
                        }
                        kernels.push(new_kernel);
                    }

                    kernels[kid].reduce(
                        nid,
                        xt,
                        graph.shape(x),
                        graph.axes(nid),
                        graph.dtype(x),
                        rop,
                    );
                    kid
                }
                Node::Unary { x, uop } => {
                    //let _timer = Timer::new("unary");
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
                    //let _timer = Timer::new("binary");
                    // x goes first, delete y
                    let (xt, kidx) = get_kernel_max(x, &kernels);
                    let (mut yt, mut kidy) = get_kernel_max(y, &kernels);

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

                        // can't remove kidy if any kernel depends on kidy
                        // store y and all outputs that are not stored yet in kidy
                        // clear kidy's outputs
                        // write a load kernel for y
                        // mark kidy as new kernel
                        if kernels[kidy].has_stores() && depends_on(&kernels, kidx, kidy) {
                            //if !kernels[kidy].depends_on.is_empty() {
                            //println!("kidx depends on kidy");
                            let outputs = kernels[kidy].outputs.clone();
                            for (nid, inner_x) in outputs {
                                if rcs.contains_key(&nid) {
                                    if !kernels[kidy].tensors.values().any(|id| *id == nid) {
                                        let nid_shape = graph.shape(nid);
                                        let nid_dtype = graph.dtype(nid);
                                        // if the kernel wasn't stored yet, because it wasn't processed yet
                                        let zview = View::contiguous(nid_shape);
                                        kernels[kidy].max_id += 1;
                                        let z = kernels[kidy].max_id;
                                        let store_op = Op::Store {
                                            z,
                                            zview,
                                            zscope: Scope::Global,
                                            zdtype: nid_dtype,
                                            x: inner_x,
                                            xscope: Scope::Register,
                                            xview: View::none(),
                                        };
                                        kernels[kidy].ops.push(store_op);
                                        kernels[kidy].tensors.insert(z, nid);

                                        let nkid = kernels.push(Kernel::leaf(
                                            nid,
                                            nid_shape,
                                            nid_dtype,
                                            BTreeSet::from([kidy]),
                                        ));
                                        if nid == y {
                                            yt = 0;
                                            kidy = nkid;
                                        }
                                    } else if nid == y {
                                        let nid_shape = graph.shape(nid);
                                        let nid_dtype = graph.dtype(nid);
                                        let nkid = kernels.push(Kernel::leaf(
                                            nid,
                                            nid_shape,
                                            nid_dtype,
                                            BTreeSet::from([kidy]),
                                        ));
                                        yt = 0;
                                        kidy = nkid;
                                    }
                                }
                            }
                            kernels[kidy].outputs.clear();
                            // TODO since no more ops will be added to kidy, we can immediatelly send it for
                            // evaluation. This can make the scheduler faster as we will have fewer kernels to work with.
                            debug_assert_eq!(yt, 0);
                        }

                        // DUe to the way nodes are handled it should not be possible for kidx to depend on kidy
                        #[cfg(debug_assertions)]
                        if depends_on(&kernels, kidy, kidx) {
                            panic!("Binary with kidy depending on kidx.")
                        }

                        let Kernel { ops, tensors, outputs, max_id, depends_on } =
                            kernels.remove(kidy).unwrap();

                        for (i, op) in ops.into_iter().enumerate() {
                            if !(matches!(op, Op::Loop { .. }) && op == kernels[kidx].ops[i]) {
                                let new_op = match op {
                                    Op::Loop { axis, len } => Op::Loop { axis, len },
                                    Op::EndLoop => Op::EndLoop,
                                    Op::Const { z, value, ref view } => {
                                        Op::Const { z: z + n, value, view: view.clone() }
                                    }
                                    Op::Load {
                                        z,
                                        zscope,
                                        ref zview,
                                        x,
                                        xscope,
                                        ref xview,
                                        xdtype,
                                    } => Op::Load {
                                        z: z + n,
                                        zscope,
                                        zview: zview.clone(),
                                        x: x + n,
                                        xscope,
                                        xview: xview.clone(),
                                        xdtype,
                                    },
                                    Op::Store {
                                        z,
                                        zscope,
                                        ref zview,
                                        zdtype,
                                        x,
                                        xscope,
                                        ref xview,
                                    } => Op::Store {
                                        z: z + n,
                                        zscope,
                                        zview: zview.clone(),
                                        zdtype,
                                        x,
                                        xscope,
                                        xview: xview.clone(),
                                    },
                                    Op::Accumulator { z, rop, dtype } => {
                                        Op::Accumulator { z: z + n, rop, dtype }
                                    }
                                    //Op::Move { z, x, mop } => Op::Move { z: z + n, x: x + n, mop },
                                    Op::Unary { z, x, uop } => {
                                        Op::Unary { z: z + n, x: x + n, uop }
                                    }
                                    Op::Binary { z, x, y, bop } => {
                                        Op::Binary { z: z + n, x: x + n, y: y + n, bop }
                                    }
                                    Op::Barrier { scope } => Op::Barrier { scope },
                                };
                                kernels[kidx].ops.push(new_op);
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

                    // Now we have to search all depends_on for all kernels, if any kernel depends on
                    // kidy, now it depends on kidx
                    for kernel in kernels.values_mut() {
                        let depends_on = kernel.depends_on.clone();
                        for kid in depends_on {
                            if kid == kidy {
                                kernel.depends_on.remove(&kidy);
                                kernel.depends_on.insert(kidx);
                            }
                        }
                    }

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
            let nid_shape = graph.shape(nid);
            let nid_dtype = graph.dtype(nid);
            store(&mut kernels, kid, nid, nid_shape, nid_dtype);
            /*if rcs.contains_key(&nid) {
                let nkid = kernels.push(Kernel::leaf(nid, nid_shape, nid_dtype));
                kernels[nkid].depends_on.insert(kid);
            }*/
        }
    }

    let elapsed = begin.elapsed().as_micros();
    let mut min_ops = usize::MAX;
    let mut max_ops = 0;
    let mut avg_ops = 0;
    for kernel in kernels.values() {
        let n = kernel.ops.len();
        if n > max_ops {
            max_ops = n;
        }
        if n < min_ops {
            min_ops = n;
        }
        avg_ops += n;
    }
    let kernels_len = kernels.len();
    println!("Scheduled {kernels_len} kernels, scheduling took {elapsed}us, ops per kernel: min: {min_ops}, max: {max_ops}, avg: {}", avg_ops/kernels_len);
    //println!("Expand clones: {expa_u}, reshape clones: {resh_u}, pad clones: {pad_u}, permute clones: {perm_u}, reduce clones: {red_u}");
    // Timer
    /*for (name, (time, iters)) in crate::ET.lock().iter() {
        println!(
            "Timer {name} took {time}us for {iters} iterations, {}us/iter",
            time / iters
        );
    }*/

    /*for kernel in kernels.values() {
        if kernel.ops.len() < 20 {
            kernel.debug();
        }
    }*/

    //panic!();

    realized_nodes.extend(to_eval);

    // Launch all kernels
    let mut ids: Vec<Id> = kernels.ids().collect();
    while !ids.is_empty() {
        let mut i = 0;
        while i < ids.len() {
            let kid = ids[i];
            if kernels[kid].depends_on.is_empty() {
                ids.remove(i);
                let mut kernel = kernels.remove(kid).unwrap();
                kernel.launch(graph, devices, memory_pools, optimizer, search_iters, debug)?;
                for kernel in kernels.values_mut() {
                    kernel.depends_on.remove(&kid);
                }
                let loads: Set<TensorId> = kernel.ops.iter().filter_map(|op| if let Op::Load { x, .. } = op { Some(kernel.tensors[x]) } else { None }).collect();
                let mut loads: Set<TensorId> = loads.difference(&realized_nodes).copied().collect();
                for kernel in kernels.values() {
                    for tensor in kernel.tensors.values() {
                        loads.remove(tensor);
                    }
                }
                for tensor in loads {
                    for pool in &mut *memory_pools {
                        if let Some(buffer_id) = pool.buffer_map.remove(&tensor) {
                            pool.pool.deallocate(buffer_id, vec![])?;
                        }
                    }
                }
            } else {
                i += 1
            }
        }
    }

    /*let mut num_evaluated = 0;
    #[cfg(debug_assertions)]
    let mut evaluated_tensors = BTreeSet::new();
    for _ in 0..kernels.len() + 1 {
        let mut evaluated = BTreeSet::new();
        for (kid, kernel) in kernels.iter_mut() {
            if kernel.depends_on.is_empty() {
                num_evaluated += 1;
                kernel.launch(graph, devices, memory_pools, optimizer, search_iters, debug)?;
                evaluated.insert(kid);

                #[cfg(debug_assertions)]
                evaluated_tensors.extend(kernel.ops.iter().filter_map(|op| {
                    if let Op::Store { z, .. } = op {
                        Some(kernel.tensors[z])
                    } else {
                        None
                    }
                }));
            }
        }
        //println!("Evaluated: {evaluated:?}");
        if evaluated.is_empty() {
            break;
        }
        for kid in &evaluated {
            kernels.remove(*kid);
        }
        for kernel in kernels.values_mut() {
            kernel.depends_on.retain(|x| !evaluated.contains(x));
        }
        // TODO delete input tensors that won't be used by other kernels,
        // that is those loads that won't be loaded by other kernels and are not in realized_nodes
    }
    if num_evaluated != kernels_len {
        println!("Evaluated {num_evaluated} kernels");
        for (id, kernel) in kernels.iter() {
            println!("\nKernel id = {id}");
            kernel.debug();
        }
        panic!();
    }*/

    /*#[cfg(debug_assertions)]
    if to_eval.difference(&evaluated_tensors).count() != 0 {
        let realized_nodes: BTreeSet<TensorId> =
            memory_pools.iter().map(|pool| pool.buffer_map.keys()).flatten().copied().collect();
        if !to_eval.is_subset(&realized_nodes) {
            let diff: BTreeSet<TensorId> = to_eval.difference(&realized_nodes).copied().collect();
            panic!("In to eval but not evaluated: {diff:?}",);
        }
    }*/

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

// Adds store to kernel and removes nid from outputs of all kernels
fn store(
    kernels: &mut Slab<Kernel>,
    kid: KernelId,
    nid: TensorId,
    nid_shape: &[usize],
    nid_dtype: DType,
) {
    if kernels[kid].tensors.values().any(|id| *id == nid) {
        return;
    }

    //let _timer = Timer::new("scheduler store");
    // Add store op to kernel
    let x = kernels[kid].outputs[&nid];
    let zview = View::contiguous(nid_shape);

    #[cfg(debug_assertions)]
    if let Some(&Op::Store { z: nz, zview: ref nzview, .. }) = kernels[kid].ops.last() {
        if x == nz && &zview == nzview {
            unreachable!();
        }
    }
    //debug_assert!(zview.numel() < 1024 * 1024 * 1024 * 1024, "Too big store.");
    kernels[kid].max_id += 1;
    let z = kernels[kid].max_id;
    let store_op = Op::Store {
        z,
        zview,
        zscope: Scope::Global,
        zdtype: nid_dtype,
        x,
        xscope: Scope::Register,
        xview: View::none(),
    };
    kernels[kid].ops.push(store_op);
    kernels[kid].tensors.insert(z, nid);
}

// recursive should be faster, since it does not allocate, but in fact the dynamic programming
// version is much faster, apparently cpus really hate recursion
// Check if kidx depends on kidy
fn depends_on(kernels: &Slab<Kernel>, kidx: KernelId, kidy: KernelId) -> bool {
    let mut depends_on = kernels[kidx].depends_on.clone();
    while let Some(kid) = depends_on.pop_last() {
        if kid == kidy {
            return true;
        }
        depends_on.extend(kernels[kid].depends_on.iter().copied());
    }
    false
}

// Check if kidx depends on kidy
/*fn depends_on(kernels: &Slab<Kernel>, kidx: KernelId, kidy: KernelId) -> bool {
    //println!("Checking if {kidx} depends on {kidy}, kidx dependencies: {:?}", kernels[kidx].depends_on);
    if kernels[kidx].depends_on.contains(&kidy) {
        return true;
    }
    kernels[kidx].depends_on.iter().any(|kid| depends_on(kernels, *kid, kidy))
}*/

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
