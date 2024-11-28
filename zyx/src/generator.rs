//! This file contains kernel generator. It converts graph into a sequence of kernels
//! that will be executed by devices.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    graph::Graph,
    ir::Scope,
    kernel::{Kernel, MOp, VOp},
    node::{BOp, Node, ROp},
    tensor::TensorId,
    view::View,
};

/// Generate kernels from graph. This function determines which ops will get fused together
/// and how many kernels will be created.
#[allow(clippy::similar_names)]
#[allow(clippy::cognitive_complexity)]
pub fn generate_kernels(
    graph: &Graph,
    order: &[TensorId],
    debug_sched: bool,
) -> Vec<Kernel> {
    let _t = crate::Timer::new("generate_kernels");
    //let _t = crate::Timer::new("generate_kernels");
    // This function sorts nodes into smallest number of kernels that can be compiled on the device
    // This function defines loops, loads, stores and elementwise ops.
    // The aim is to sort nodes in such a way, that maximum performance is attained.
    // These kernels mostly keep shapes of original nodes.
    // Further optimization is done in optimize kernels function.
    //println!("Eval: {to_eval:?}");
    let mut kernels: Vec<Kernel> = Vec::new();
    for nid in order.iter().copied() {
        let node = &graph[nid];
        if debug_sched { println!("ID({nid})x{}: {node:?}, sh: {:?}", graph.rc(nid), graph.shape(nid)); }
        match node {
            &Node::Const { value } => {
                let _t = crate::Timer::new("Const");
                let mut kernel = Kernel::with_shape(&[1]);
                kernel.ops.push(VOp::Const {
                    z: nid,
                    value,
                    view: View::contiguous(&[1]),
                });
                kernels.push(kernel);
            }
            Node::Leaf => {
                let _t = crate::Timer::new("Leaf");
                kernels.push(Kernel::load(nid, graph));
            }
            &Node::Expand { x } => {
                let _t = crate::Timer::new("Expand");
                let shape = graph.shape(nid);
                let xshape = graph.shape(x);
                assert_eq!(shape.len(), xshape.len());
                let mut kernel = get_kernel(x, &mut kernels, graph);
                // For now no expand on reduce kernels or kernels that store something.
                // Later this can be done if the store or reduce is in different loop,
                // that is if we are expanding loop after reduce and if store is before
                // that expanded loop.
                if kernel.ops.iter().any(|op| matches!(op, VOp::Store { .. })) || kernel.is_reduce()
                {
                    // TODO not sure if this is perfectly correct. Can it contain x in outputs,
                    // but can it be x evaluated to different values, i.e. some intermediate?
                    if !kernel.outputs().contains(&x) {
                        kernel.store(x, View::contiguous(xshape), graph.dtype(x));
                    }
                    kernels.push(Kernel::load(x, graph));
                    kernel = kernels.last_mut().unwrap();
                }
                //println!("Expanding");
                //kernel.debug();
                assert_eq!(shape.len(), kernel.shape().len());
                let mut expand_axes = BTreeSet::new();
                for (a, d) in kernel.shape().into_iter().enumerate() {
                    if d != shape[a] {
                        assert_eq!(d, 1);
                        expand_axes.insert(a);
                    }
                }
                // We go over ops in reverse, increasing last loops dimension
                //println!("expand_axes = {expand_axes:?}");
                let mut done_expanding = BTreeSet::new();
                for op in kernel.ops.iter_mut().rev() {
                    match op {
                        VOp::Loop {
                            axis,
                            len: dimension,
                        } => {
                            if expand_axes.contains(axis) && done_expanding.insert(*axis) {
                                assert_eq!(*dimension, 1);
                                *dimension = shape[*axis];
                            }
                        }
                        VOp::Load { xview: view, .. } | VOp::Const { view, .. } => {
                            // Done expanding marks which loops are behind us,
                            // so we need to only adjust strides to 0 in axes for those axes that are not behind us yet.
                            for a in expand_axes.difference(&done_expanding) {
                                view.expand(*a, shape[*a]);
                            }
                        }
                        VOp::Store { .. } => {
                            unreachable!();
                            // TODO This will do multiple writes to the same index, so this would probably be better solved in different way,
                            // perhaps doing only single write during the whole loop using if condition, but that could also be added
                            // to View in VOp::Store as optimization when converting to IROps
                            //for a in expand_axes.difference(&done_expanding) { zview.expand(*a, shape[*a]); }
                        }
                        _ => {}
                    }
                }
                kernel.ops.push(VOp::Move {
                    z: nid,
                    x,
                    mop: MOp::Expa,
                });
                //kernel.debug();
                assert_eq!(kernel.shape(), shape);
                //println!("Into");
            }
            &Node::Permute { x } => {
                let _t = crate::Timer::new("Permute");
                let axes = graph.axes(nid);
                // Permute shuffles load and store strides
                // It also changes the dimension of loops
                // and shape of kernel
                let kernel = get_kernel(x, &mut kernels, graph);
                kernel.permute(axes);
                kernel.ops.push(VOp::Move {
                    z: nid,
                    x,
                    mop: MOp::Perm,
                });
                assert_eq!(kernel.shape(), graph.shape(nid));
            }
            &Node::Reshape { x } => {
                let _t = crate::Timer::new("Reshape");
                // Reshape needs to add new loops to the end of the kernel if it is unsqueeze

                // If reshape comes after reduce, then if it just aplits axes, it can be merged,
                // otherwise we have to create new kernel.
                //for kernel in &kernels { kernel.debug(); }

                let shape = graph.shape(nid);
                //println!("Reshape node from {:?} to {:?}", graph.shape(x), shape);
                let mut kernel = get_kernel(x, &mut kernels, graph);
                if !kernel.reshape(shape) {
                    // else create new kernel after storing results of previous kernel
                    let xdtype = graph.dtype(x);
                    kernel.store(x, View::contiguous(graph.shape(x)), xdtype);
                    let mut new_kernel = Kernel::with_shape(shape);
                    new_kernel.ops.push(VOp::Load {
                        z: nid,
                        zscope: Scope::Register,
                        zview: View::none(),
                        x,
                        xscope: Scope::Global,
                        xview: View::contiguous(shape),
                        xdtype,
                    });
                    kernels.push(new_kernel);
                    kernel = kernels.last_mut().unwrap();
                }
                kernel.ops.push(VOp::Move {
                    z: nid,
                    x,
                    mop: MOp::Resh,
                });
                //println!("\nKernels {kernels:?}\n");
            }
            &Node::Pad { x } => {
                let _t = crate::Timer::new("Pad");
                let padding = graph.padding(nid);
                //println!("Padding: {padding:?}");
                // Pad shrinks or expands dimension of axes, this is ZERO padding
                let mut kernel = get_kernel(x, &mut kernels, graph);
                // Kernel cannot be padded if it containe max reduce.
                // For now kernel also won't be padded if it contains store,
                // but that can be changed.
                if !kernel.can_be_zero_padded() {
                    kernel.store(x, View::contiguous(graph.shape(x)), graph.dtype(x));
                    kernels.push(Kernel::load(x, graph));
                    kernel = kernels.last_mut().unwrap();
                }
                //kernel.debug();
                let rank = kernel.shape().len();
                // Get which axes are padded
                let mut padded_axes = BTreeMap::new();
                for (op, &p) in kernel.ops[..rank].iter().rev().zip(padding) {
                    let &VOp::Loop { axis, .. } = op else {
                        unreachable!()
                    };
                    padded_axes.insert(axis, p);
                }
                // Apply padding
                let mut num_paddings = padding.len();
                //println!("Padded axes: {padded_axes:?}");
                for op in &mut kernel.ops {
                    match op {
                        VOp::Loop { axis, len } => {
                            if let Some((lp, rp)) = padded_axes.get(axis) {
                                *len = usize::try_from(isize::try_from(*len).unwrap() + lp + rp)
                                    .unwrap();
                            }
                        }
                        VOp::EndLoop => {
                            num_paddings -= 1;
                            if num_paddings == 0 {
                                break;
                            }
                        }
                        VOp::Const { view, .. }
                        | VOp::Load { xview: view, .. }
                        | VOp::Store { zview: view, .. }
                        | VOp::Accumulator { view, .. } => {
                            for (&axis, &(lp, rp)) in &padded_axes {
                                view.pad(axis, lp, rp);
                            }
                        }
                        _ => {}
                    }
                }
                kernel.ops.push(VOp::Move {
                    z: nid,
                    x,
                    mop: MOp::Padd,
                });
                //kernel.debug();
                assert_eq!(kernel.shape(), graph.shape(nid));
            }
            &Node::Reduce { x, rop } => {
                let _t = crate::Timer::new("Reduce");
                let axes = graph.axes(nid);
                // TODO do not apply reduce on a previously fully reduced and expanded kernel, this
                // happens in softmax
                let kernel = get_kernel(x, &mut kernels, graph);
                // Reduce removes loops and adds accumulator before those loops that it removes
                //println!("Axes {axes:?}");
                // Permute the axes such that reduce loops are last
                // and keep the order of axes that are not reduced.
                let permute_axes: Vec<usize> = (0..graph.shape(x).len())
                    .filter(|a| !axes.contains(a))
                    .chain(axes.iter().copied())
                    .collect();
                //println!("Permute axes in reduce: {permute_axes:?}");
                kernel.permute(&permute_axes);

                // We can also just merge these reduce loops into single loop, since it gets removed
                // from the resulting shape either way, but only if there are no ops between those loops.

                // Add accumulator
                let num_axes = graph.shape(x).len();
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
                        rop,
                        view: View::none(),
                        dtype: graph.dtype(nid),
                    },
                );
                kernel.ops.push(VOp::Binary {
                    z: nid,
                    x,
                    y: nid,
                    bop: match rop {
                        ROp::Sum => BOp::Add,
                        ROp::Max => BOp::Max,
                    },
                });
                for _ in 0..axes.len() {
                    kernel.ops.push(VOp::EndLoop);
                }
                if kernel.shape().is_empty() {
                    kernel.insert_loop(0, 0);
                }
                // Optionally merge axes (if possible) for potentially better performance
                //kernel.merge_axes(acc_id + 1, axes.len());
            }
            Node::Unary { x, uop } => {
                let _t = crate::Timer::new("Unary");
                let kernel = get_kernel(*x, &mut kernels, graph);
                kernel.ops.push(VOp::Unary {
                    z: nid,
                    x: *x,
                    uop: *uop,
                });
            }
            &Node::Binary { x, y, bop } => {
                let _t = crate::Timer::new("Binary");
                // Binary ops may allow us to join two kernels together
                if let Some(id) = kernels
                    .iter_mut()
                    .position(|kernel| kernel.vars().is_superset(&[x, y].into()))
                {
                    // If both inputs are in the same kernel
                    /*let kernel = if kernels[id].shape() == graph.shape(x) {
                        &mut kernels[id]
                    } else {
                        // create new kernel using already predefined stores of both x and y
                        println!("Binary op");
                        for kernel in &kernels {
                            kernel.debug();
                        }
                        let mut kernel = Kernel::load(x, graph);
                        kernel.ops.push(VOp::Load {
                            z: y,
                            zscope: Scope::Register,
                            zview: View::none(),
                            x: y,
                            xscope: Scope::Global,
                            xview: View::contiguous(graph.shape(y)),
                            xdtype: graph.dtype(x),
                        });
                        kernels.push(kernel);
                        kernels.last_mut().unwrap()
                    };
                    kernel.ops.push(VOp::Binary { z: nid, x, y, bop });*/
                    kernels[id].ops.push(VOp::Binary { z: nid, x, y, bop });
                } else {
                    // If inputs are in different kernels
                    let _t = crate::Timer::new("Binary first part");
                    let kernel_x_id = kernels
                        .iter()
                        .enumerate()
                        .rev()
                        .filter(|(_, kernel)| kernel.vars().contains(&x))
                        .min_by_key(|(_, kernel)| kernel.ops.len())
                        .unwrap()
                        .0;
                    let kernel_y_id = kernels
                        .iter()
                        .enumerate()
                        .rev()
                        .filter(|(_, kernel)| kernel.vars().contains(&y))
                        .min_by_key(|(_, kernel)| kernel.ops.len())
                        .unwrap()
                        .0;
                    drop(_t);
                    if kernels[kernel_x_id].shape() != kernels[kernel_y_id].shape() {
                        kernels[kernel_x_id].debug();
                        kernels[kernel_y_id].debug();
                        panic!();
                    }

                    // Check which kernel needs to be evaluated first
                    // TODO make sure that depends on is never needed
                    /*match (
                        depends_on(kernel_x_id, kernel_y_id, &kernels),
                        depends_on(kernel_y_id, kernel_x_id, &kernels),
                    ) {
                        (true, true) => {
                            // This should not be possible
                            panic!()
                        }
                        (true, false) => {
                            // kernel x depends on kernel y
                            // This is ok, nothing needs to be done
                        }
                        (false, true) => {
                            println!("Depends on caused changes");
                            // Here we need to do some reordering,
                            // or just swap ids.
                            (kernel_x_id, kernel_y_id) = (kernel_y_id, kernel_x_id);
                        }
                        (false, false) => {
                            // Nothing needs to be done
                        }
                    }*/

                    // Now we know that kernel x depends on kernel y or there is no dependence at all
                    // So kernel y must go first
                    let _t = crate::Timer::new("Binary second part");
                    let (kernel_y, kernel_x) = if kernel_x_id > kernel_y_id {
                        (kernels.remove(kernel_y_id), &mut kernels[kernel_x_id - 1])
                    } else {
                        (kernels.remove(kernel_y_id), &mut kernels[kernel_x_id])
                    };
                    let kernel_y_ops: Vec<VOp> = kernel_y
                        .ops
                        .into_iter()
                        .enumerate()
                        .skip_while(|(i, op)| op == &kernel_x.ops[*i])
                        .map(|(_, op)| op)
                        .collect();
                    kernel_x.ops.extend(kernel_y_ops);
                    kernel_x.ops.push(VOp::Binary { z: nid, x, y, bop });
                    // if kernel is not last, then make it last
                    if kernel_y_id > kernel_x_id {
                        let kernel = kernels.remove(kernel_x_id);
                        kernels.push(kernel);
                    }
                }
            }
        }
        //println!("nid: {nid} to_eval {:?}", graph.to_eval);
        if graph.to_eval.contains(&nid) {
            if let Some(kernel) = kernels
                .iter_mut()
                .find(|kernel| kernel.vars().contains(&nid))
            {
                kernel.store(nid, View::contiguous(graph.shape(nid)), graph.dtype(nid));
            } else {
                unreachable!()
            }
        }
        // TODO only if this is more than user rcs
        if graph.rc(nid) > 1 {
            if let Some(kernel) = kernels
                .iter_mut()
                .find(|kernel| kernel.vars().contains(&nid))
            {
                // if graph.rc(nid) > 1 then just copy that graph if it is not too big graph
                // and if the kernel does not have too big shape.
                // TODO beware of too many copies. We need to make sure that we are not doing
                // the same work twice.
                //if user_leafs.contains(&nid) {
                //kernel.store(nid, View::new(graph.shape(nid)));
                if ((kernel.ops.len() > 100 || graph.rc(nid) > 2)
                    && kernel.shape().into_iter().product::<usize>() < 1024 * 1024 * 1024)
                    || kernel.is_reduce()
                    || !kernel.outputs().is_empty()
                {
                    //println!("Storing {nid}");
                    kernel.store(nid, View::contiguous(graph.shape(nid)), graph.dtype(nid));
                    kernels.push(Kernel::load(nid, graph));
                } else {
                    let kernel2 = kernel.clone();
                    kernels.push(kernel2);
                }
            } else {
                unreachable!()
            }
        }
    }
    // Remove unnecessary kernels
    // TODO these should be only loads for user_rc > 1 kernels, remove this
    kernels.retain(|kernel| !kernel.outputs().is_empty());
    // Remove unnecessary stores not for tensors moved across kernels
    // and not in to_eval that were inserted for rc > 1, but ops got merged,
    // and these stores were not used.
    let mut necessary_stores = graph.to_eval.clone();
    for kernel in &kernels {
        necessary_stores.extend(kernel.inputs().iter());
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

// Checks if kernel_y needs to be evaluated before kernel_x
// This takes 96% of the execution time of the whole generate_kernels function.
// We need to either improve the speed of this function or do it in some other way.
/*fn depends_on(kernel_x_id: usize, kernel_y_id: usize, kernels: &[Kernel]) -> bool {
    let _t = crate::Timer::new("depends_on");
    let mut kernel_x_inputs = kernels[kernel_x_id].inputs();
    let kernel_y_outputs = &kernels[kernel_y_id].outputs();
    //println!("y outputs: {kernel_y_outputs:?}");
    //for kernel in kernels { kernel.debug(); }
    let mut visited = BTreeSet::new();
    while let Some(x) = kernel_x_inputs.pop_last() {
        if visited.insert(x) {
            if kernel_y_outputs.contains(&x) {
                return true;
            } else {
                'a: for kernel in kernels.iter().rev() {
                    if kernel.outputs().contains(&x) {
                        kernel_x_inputs.extend(kernel.inputs());
                        //println!("x inputs: {kernel_x_inputs:?}");
                        break 'a;
                    }
                }
            }
        }
    }
    false
}*/

fn get_kernel<'a>(x: TensorId, kernels: &'a mut Vec<Kernel>, graph: &Graph) -> &'a mut Kernel {
    let _t = crate::Timer::new("get_kernel");
    // First if there is kernel which stores x, then just return new load kernel
    if kernels
        .iter()
        .rev()
        .any(|kernel| kernel.outputs().contains(&x))
    {
        kernels.push(Kernel::load(x, graph));
        return kernels.last_mut().unwrap();
    }
    kernels
        .iter_mut()
        .rev()
        // we need to memorize this so that it is much faster,
        // then we can drop the whole compiled_graph_cache
        .filter(|kernel| kernel.vars().contains(&x)) // todo filter is more ideal, but it is slow...
        .min_by_key(|kernel| kernel.ops.len())
        .unwrap()
}
