use super::{BOp, ROp, UOp};
use crate::runtime::{view::View, Node, Subgraph, TensorId};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "debug1")]
use libc_print::std_name::println;

pub(super) fn create_kernels(
    buffers: &BTreeSet<TensorId>,
    graph: &Subgraph,
    to_eval: BTreeSet<TensorId>,
    order: &[TensorId],
) -> Vec<Kernel> {
    let mut reduce_loop = false;

    // Kernel is a mapping from work size to ops.
    // Kernels must be launched in this order.
    let mut kernels: Vec<Kernel> = Vec::new();

    // TODO change it such, that kernels can not be merged in such a way
    // that there would be deadlock (kernel requires evaluation of args
    // from second kernel which requires evaluation of first kernel)

    for nid in order.iter().copied() {
        if buffers.contains(&nid) {
            let shape: Vec<usize> = graph.shape(nid).into();
            let op = Op::Load {
                z: nid,
                x: nid,
                view: View::from(&shape),
            };
            if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.shape == shape) {
                // kernel with requested work size exists, so we can push this load in there kernel.ops.push(Op::Load { nid, view });
                kernel.ops.push(op);
                kernel.vars.insert(nid);
            } else {
                // Otherwise create kernel with new work size
                kernels.push(Kernel {
                    shape,
                    ops: vec![op],
                    args: BTreeSet::new(),
                    vars: BTreeSet::from([nid]),
                    reduce_vars: BTreeSet::new(),
                });
            }
        }
        match graph[nid] {
            Node::Leaf { .. } => {}
            Node::Expand { x, shape_id } => {
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(&x)) {
                    // Store it, so that it can be accessed by different kernel
                    kernel.ops.push(Op::Store { z: x });
                } else {
                    panic!("Expand on nonexistent tensor.")
                }
                let shape = graph._shape(shape_id);
                let mut view = View::from(graph.shape(x));
                view.expand(shape);
                // Expand always creates new kernel, since it changes the shape,
                // reduce can later allow merging of this kernel with other kernels.
                kernels.push(Kernel {
                    shape: shape.into(),
                    ops: vec![Op::Load { z: nid, x, view }],
                    args: BTreeSet::from([x]),
                    vars: BTreeSet::from([nid]),
                    reduce_vars: BTreeSet::new(),
                });
                // Perhaps we can just merge expand too.
                /*if let Some(kernel) = kernels.values_mut().find(|kernel| kernel.vars.contains(&x)) {
                    let shape = graph._shape(shape_id);
                    kernel.movement_op(graph, x, |view| view.expand(shape), false);
                    kernel.vars.insert(nid);
                } else {
                    panic!("Expand on nonexistent tensor.")
                }*/
            }
            Node::Permute { x, axes_id, .. } => {
                // Permute gets applied to view of all loads that ended in this node
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(&x)) {
                    let axes = graph._axes(axes_id);
                    kernel.movement_op(graph, x, |view| view.permute(axes), true);
                    kernel.vars.insert(nid);
                } else {
                    panic!("Permute on nonexistent tensor.")
                }
            }
            Node::Exp { x } => {
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(&x)) {
                    kernel.ops.push(Op::Unary {
                        z: nid,
                        x,
                        op: UOp::Exp,
                    });
                    kernel.vars.insert(nid);
                    if kernel.reduce_vars.contains(&x) {
                        kernel.reduce_vars.insert(nid);
                    }
                } else {
                    panic!("Exp on nonexistent tensor.")
                }
            }
            Node::Add { x, y } => {
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(&x)) {
                    kernel.ops.push(Op::Binary {
                        z: nid,
                        x,
                        y,
                        op: BOp::Add,
                    });
                    kernel.vars.insert(nid);
                    if kernel.reduce_vars.contains(&x) {
                        kernel.reduce_vars.insert(nid);
                    }
                } else {
                    panic!("Add on nonexistent tensor.")
                }
            }
            Node::Sum {
                x,
                axes_id,
                shape_id,
            } => {
                // Reduce basically means that in current kernel (with x in vars) we make one of
                // the dimensions register

                // TODO If this is a reduce on already reduced values (in reduce_vars), then we can
                // merge two reduces of the same kind together, if these are different ROp, then we
                // must store results from the first reduce and create new reduce.
                // And that should be all that needs to be dealt with in this function :)

                if let Some((i, kernel)) = kernels
                    .iter_mut()
                    .enumerate()
                    .find(|(_, kernel)| kernel.vars.contains(&x))
                {
                    let shape_before_reduce: Vec<usize> = graph.shape(x).into();
                    //println!("{shape_before_reduce:?}, {shape:?}");
                    if shape_before_reduce == kernel.shape {
                        // If kernel has the same shape as non reduced self, we can apply reduce on it
                        kernel.ops.insert(
                            0,
                            Op::ReduceLoop {
                                axes: graph._axes(axes_id).into(),
                                shape_before_reduce,
                                op: ROp::Sum,
                            },
                        );
                        kernel.ops.push(Op::Reduce {
                            z: nid,
                            x,
                            op: ROp::Sum,
                        });
                        let shape: Vec<usize> = graph._shape(shape_id).into();
                        kernel.shape = shape.clone();
                        if let Some(k) = kernels.iter().position(|kernel| kernel.shape == shape) {
                            if !depends_on(k, i, &kernels) {
                                let kernel2 = kernels.remove(i);
                                // If we can merge with some other existing kernel
                                kernels[k].ops.extend(kernel2.ops);
                            }
                        }
                    } else {
                        // Try to find different kernel with appropriate shape that does not have
                        // dependencies preventing merge.
                        let new_shape = graph._shape(shape_id);
                        if let Some(kernel) =
                            kernels.iter_mut().find(|kernel| kernel.shape == new_shape)
                        {
                            // Add this reduce to that other kernel
                            kernel.reduce_vars.insert(nid);
                            kernel.vars.insert(nid);
                            kernel.ops.push(Op::ReduceLoop {
                                axes: graph._axes(axes_id).into(),
                                shape_before_reduce,
                                op: ROp::Sum,
                            });
                            kernel.ops.push(Op::Reduce {
                                z: nid,
                                x,
                                op: ROp::Sum,
                            });
                        } else {
                            // If we can not find any other kernel with correct shape that is dependency free,
                            // then this kernel can not be merged and we have to create new kernel.
                            let view = View::from(new_shape);
                            kernels.push(Kernel {
                                shape: new_shape.into(),
                                args: BTreeSet::from([x]),
                                reduce_vars: BTreeSet::from([nid]),
                                vars: BTreeSet::from([nid]),
                                ops: vec![Op::Load { z: nid, x, view }],
                            });
                            todo!("Add reduce loops.");
                        }
                    }
                } else {
                    panic!("Sum on nonexistent tensor.")
                }
            }
            _ => {
                todo!("{:?}", graph[nid])
            }
        }
        if to_eval.contains(&nid) {
            for kernel in kernels.iter_mut() {
                if kernel.vars.contains(&nid) {
                    kernel.ops.push(Op::Store {
                        z: nid,
                        //dtype: graph.dtype(nid),
                    });
                }
            }
        }
    }

    return kernels;
}

#[derive(Debug)]
pub(super) struct Kernel {
    pub(super) shape: Vec<usize>,
    // Operations in this kernel
    pub(super) ops: Vec<Op>,
    // Variables that must be evaluated before this kernel can run
    args: BTreeSet<TensorId>,
    // Variables evaluated in this kernel (both register and global)
    vars: BTreeSet<TensorId>,
    // These variables come from reduce op. Some ops can be applied on them, while others not.
    reduce_vars: BTreeSet<TensorId>,
}

impl Kernel {
    fn movement_op(
        &mut self,
        graph: &Subgraph,
        x: TensorId,
        mop: impl Fn(&mut View),
        is_permute: bool,
    ) {
        let _ = is_permute;
        let mut params = vec![x];
        while let Some(param) = params.pop() {
            let mut add_load = false;
            if self.reduce_vars.contains(&param) {
                // If this is a result of reduce kernel, then we have to store and load it into
                // global. TODO we may be able to merge permutes, but no other movement ops.
                add_load = true;
            } else {
                for op in &mut self.ops {
                    match op {
                        Op::Unary { z, x, .. } => {
                            if param == *z {
                                params.push(*x);
                                break;
                            }
                        }
                        Op::Binary { z, x, y, .. } => {
                            if param == *z {
                                params.push(*x);
                                params.push(*y);
                                break;
                            }
                        }
                        Op::Load { z, view, .. } => {
                            if param == *z {
                                mop(view);
                                break;
                            }
                        }
                        Op::Store { z, .. } => {
                            if param == *z {
                                // If there is a store, this means global variable z needs not to be
                                // moved and every load before this store may not be moved.
                                // Thus we need to add new load to which we apply the movement op.
                                add_load = true;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
            }
            if add_load {
                let mut add_store = false;
                let mut i = 0;
                while i < self.ops.len() {
                    match self.ops[i] {
                        Op::Unary { z, .. } | Op::Binary { z, .. } => {
                            if param == z {
                                add_store = true;
                                break;
                            }
                        }
                        Op::Store { z, .. } => {
                            if param == z {
                                break;
                            }
                        }
                        Op::Load { .. } => {
                            panic!("Should not be possible.")
                        }
                        _ => {}
                    }
                    i += 1;
                }
                i += 1;
                if add_store {
                    self.ops.insert(i, Op::Store { z: param });
                    i += 1;
                }
                let mut view = View::from(graph.shape(param));
                mop(&mut view);
                self.ops.insert(
                    i,
                    Op::Load {
                        z: param,
                        x: param,
                        view,
                    },
                );
            }
        }
    }
}

#[derive(Debug)]
pub(super) enum Op {
    Store {
        z: TensorId,
    },
    Load {
        z: TensorId,
        x: TensorId,
        view: View,
    },
    Unary {
        z: TensorId,
        x: TensorId,
        op: UOp,
    },
    Binary {
        z: TensorId,
        x: TensorId,
        y: TensorId,
        op: BOp,
    },
    ReduceLoop {
        axes: Vec<usize>,
        shape_before_reduce: Vec<usize>,
        op: ROp,
    },
    Reduce {
        z: TensorId,
        x: TensorId,
        op: ROp,
    },
}

fn depends_on(z: usize, x: usize, kernels: &[Kernel]) -> bool {
    let mut params = kernels[z].args.clone();
    let x_vars = &kernels[x].vars;
    if !params.is_disjoint(x_vars) {
        return true;
    }
    while let Some(param) = params.pop_last() {
        let args = kernels
            .iter()
            .find(|kernel| kernel.vars.contains(&param))
            .unwrap()
            .args
            .clone();
        if !args.is_disjoint(x_vars) {
            return true;
        }
        params.extend(args);
    }
    return false;
}
