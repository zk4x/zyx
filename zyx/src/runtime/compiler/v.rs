use crate::runtime::node::{BOp, Node, ROp, UOp};
use crate::{runtime::graph::Graph, tensor::TensorId};
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::ir::Index;

use std::println;

type Axis = usize;
type Dimension = usize;
type Stride = usize;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum VOp {
    Load {
        z: TensorId,
        x: TensorId,
        view: View,
    },
    Store {
        z: TensorId,
        strides: Vec<Stride>,
    },
    Loop {
        axis: Axis,
        dimension: Dimension,
    },
    Accumulator {
        z: TensorId,
        rop: ROp,
    },
    Reduce {
        num_axes: usize,
        rop: ROp,
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
                VOp::Load { view, .. } => {
                    let n = view.rank();
                    if axes.len() < n {
                        let all_axes: Vec<usize> =
                            axes.iter().copied().chain(axes.len()..n).collect();
                        view.permute(&all_axes);
                    } else {
                        let axes: Vec<usize> = axes.iter().copied().filter(|a| *a < n).collect();
                        view.permute(&axes);
                    }
                }
                VOp::Store { strides, .. } => {
                    let n = strides.len();
                    if axes.len() < n {
                        let all_axes: Vec<usize> =
                            axes.iter().copied().chain(axes.len()..n).collect();
                        *strides = all_axes.iter().map(|axis| strides[*axis]).collect();
                    } else {
                        let axes: Vec<usize> = axes.iter().copied().filter(|a| *a < n).collect();
                        *strides = axes.iter().map(|axis| strides[*axis]).collect();
                    }
                }
                _ => {}
            }
        }
        self.shape = shape.clone();
    }

    pub(super) fn split_axis(&mut self, op_id: usize, dimensions: &[usize]) {
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
        //println!("Splitting {op_id} into {dimensions:?}");
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
                    let mut stride = view.0[axis].stride;
                    view.0.remove(axis);
                    let mut temp_axis = axis + dimensions.len();
                    for dim in dimensions.iter().copied().rev() {
                        temp_axis -= 1;
                        view.0.insert(
                            axis,
                            ViewDim {
                                axis: temp_axis,
                                dim,
                                stride,
                                len: dim,
                                shift: dim,
                            },
                        );
                        stride *= dim;
                    }
                    // Rename all following axes
                    for a in axis + dimensions.len()..view.0.len() {
                        view.0[a].axis += dimensions.len() - 1;
                    }
                }
                VOp::Store { strides, .. } => {
                    // Example of axis split
                    // shape
                    //  2, 6,    2
                    //  2, 3, 2, 2
                    // strides
                    // 12, 2,    1
                    // 12, 4, 2, 1
                    let mut stride = strides[axis];
                    for dim in dimensions[1..].iter().rev() {
                        strides.insert(axis, stride);
                        stride *= dim;
                    }
                }
                _ => {}
            }
        }
    }

    fn merge_axes(&mut self, op_id: usize, num_loops: usize) {
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
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct View(Vec<ViewDim>);

impl View {
    fn new(shape: &[usize]) -> Self {
        let mut stride = 1;
        let mut view: Vec<ViewDim> = shape
            .iter()
            .enumerate()
            .rev()
            .map(|(axis, dim)| {
                let temp = stride;
                stride *= dim;
                ViewDim {
                    axis,
                    stride: temp,
                    dim: *dim,
                    len: *dim,
                    shift: *dim,
                }
            })
            .collect();
        view.reverse();
        return Self(view);
    }

    fn shape(&self) -> Vec<usize> {
        self.0.iter().map(|dim| dim.dim).collect()
    }

    fn rank(&self) -> usize {
        self.0.len()
    }

    /*fn numel(&self) -> usize {
        self.0.iter().map(|dim| dim.dim).product()
    }*/

    fn permute(&mut self, axes: &[usize]) {
        assert_eq!(self.0.len(), axes.len());
        self.0 = axes.iter().map(|axis| self.0[*axis]).collect();
        for (a, dim) in self.0.iter_mut().enumerate() {
            dim.axis = a;
        }
    }

    pub(super) fn index(&self) -> Index {
        // TODO add index for padded views
        if self.is_contiguous() {
            Index::Contiguous {
                dims: self.0.iter().map(|dim| (dim.axis, dim.stride)).collect(),
            }
        } else {
            Index::Strided {
                dims: self.0.iter().map(|dim| (dim.axis, dim.stride)).collect(),
            }
        }
    }

    fn is_contiguous(&self) -> bool {
        &View::new(&self.shape()) == self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ViewDim {
    axis: Axis,
    dim: Dimension,
    stride: Stride,
    len: usize,
    shift: usize,
}

pub(super) fn generate_kernels(
    graph: &Graph,
    order: &[TensorId],
    to_eval: &BTreeSet<TensorId>,
) -> Vec<Kernel> {
    //println!("Graph: {graph:?}");
    let mut kernels: Vec<Kernel> = Vec::new();
    for nid in order.iter().copied() {
        let node = &graph[nid];
        match node {
            Node::Leaf { shape, .. } => {
                let view = View::new(shape);
                let load_op = VOp::Load {
                    z: nid,
                    x: nid,
                    view,
                };
                if let Some(kernel) = kernels.iter_mut().find(|kernel| &kernel.shape == shape) {
                    kernel.ops.push(load_op);
                    kernel.inputs.insert(nid);
                    kernel.vars.insert(nid);
                } else {
                    let mut ops: Vec<VOp> = shape_to_loops(shape);
                    ops.push(load_op);
                    kernels.push(Kernel {
                        shape: shape.clone(),
                        inputs: BTreeSet::from([nid]),
                        outputs: BTreeSet::new(),
                        vars: BTreeSet::from([nid]),
                        ops,
                    });
                }
            }
            Node::Expand { x, shape } => {
                // Expand can just add loops
                // Expand means that global buffer is accessed multiple times. Thus we need to add caching (local, register) here.
                // Expand increases axes with dimension of 1 to bigger dimension
                // and sets strides in those axes to 0 for both loads and stores
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(x)) {
                    //println!("Expanding {kernel:?}");
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
                                    view.0[*a].dim = shape[*a];
                                    view.0[*a].stride = 0;
                                }
                            }
                            VOp::Store { strides, .. } => {
                                for a in expand_axes.difference(&done_expanding) {
                                    // TODO This will do multiple writes to the same index, so this would probably be better solved in different way,
                                    // perhaps doing only single write during the whole loop
                                    strides[*a] = 0;
                                }
                            }
                            _ => {}
                        }
                    }
                    kernel.ops.push(VOp::Unary {
                        z: nid,
                        x: *x,
                        uop: UOp::Noop,
                    });
                    kernel.vars.insert(nid);
                } else {
                    panic!()
                }
            }
            Node::Permute { x, axes, .. } => {
                // Permute shuffles load and store strides
                // It also changes the dimension of loops
                // and shape of kernel
                // TODO but what if it is permute after reduce?
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(x)) {
                    kernel.permute(&axes);
                    kernel.ops.push(VOp::Unary {
                        z: nid,
                        x: *x,
                        uop: UOp::Noop,
                    });
                    kernel.vars.insert(nid);
                } else {
                    panic!()
                }
            }
            Node::Reshape { x, shape } => {
                // If we really want, we can get reshape working with loads and stores
                // simply by using view for loads to have multiple reshapes in single view.
                // But for now it is much simpler to just add new kernel.

                // If reshape comes after reduce, then if it just aplits axes, it can be merged,
                // otherwise we have to create new kernel.

                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(x)) {
                    // If this is just a reshape of kernel with only unary ops and contiguous loads
                    // and stores, we can remove old loops and replace them with new loops.
                    if kernel.ops.iter().all(|op| match op {
                        VOp::Loop { .. } | VOp::Unary { .. } | VOp::Binary { .. } => true,
                        VOp::Load { view, .. } => view.is_contiguous(),
                        VOp::Store { strides, .. } => strides == &shape_to_strides(&kernel.shape),
                        _ => false,
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
                                VOp::Load { view, .. } => {
                                    *view = View::new(shape);
                                }
                                VOp::Store { strides, .. } => {
                                    *strides = shape_to_strides(shape);
                                }
                                _ => {}
                            }
                        }
                        kernel.ops.push(VOp::Unary {
                            z: nid,
                            x: *x,
                            uop: UOp::Noop,
                        });
                        kernel.shape = shape.clone();
                        kernel.vars.insert(nid);
                    } else {
                        // TODO
                        // If we can split axes, split axes by replacing one loop with two loops.
                        // If last axes are unsqueezes with ones, add new loops to the end of the kernel.

                        // else create new kernel after storing results of previous kernel
                        let strides = shape_to_strides(graph.shape(*x));
                        kernel.ops.push(VOp::Store { z: *x, strides });
                        kernel.outputs.insert(*x);
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
                } else {
                    panic!()
                }
            }
            Node::Pad { x, pad, .. } => {
                // Pad shrinks or expands dimension of axes, but if there is store,
                // then it creates new kernel
                todo!()
            }
            Node::Reduce {
                x,
                axes,
                rop,
                shape,
            } => {
                // Reduce removes loops and adds accumulator before those loops that it removes
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(x)) {
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
                    let mut looped_axes: BTreeSet<usize> =
                        (num_axes - axes.len()..num_axes).collect();
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
                    kernel
                        .ops
                        .insert(acc_id, VOp::Accumulator { z: nid, rop: *rop });
                    // End loops
                    kernel.ops.push(VOp::Reduce {
                        num_axes: axes.len(), // Now we are merging them, without merging its axes.len(),
                        rop: *rop,
                        z: nid,
                        x: *x,
                    });
                    kernel.vars.insert(nid);
                    kernel.shape = shape.clone();

                    //kernel.merge_axes(acc_id + 1, axes.len());
                } else {
                    panic!()
                }
            }
            Node::Unary { x, uop } => {
                if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(x)) {
                    kernel.ops.push(VOp::Unary {
                        z: nid,
                        x: *x,
                        uop: *uop,
                    });
                    kernel.vars.insert(nid);
                } else {
                    panic!()
                }
            }
            Node::Binary { x, y, bop } => {
                // Binary ops may allow us to join two kernels together
                if let Some(kernel) = kernels
                    .iter_mut()
                    .find(|kernel| kernel.vars.contains(x) && kernel.vars.contains(y))
                {
                    // If both inputs are in the same kernel
                    //println!("Both inputs are in the same kernel.");
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
                        // TODO do some checks that this join is always valid

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
        if to_eval.contains(&nid) {
            if let Some(kernel) = kernels.iter_mut().find(|kernel| kernel.vars.contains(&nid)) {
                kernel.ops.push(VOp::Store {
                    z: nid,
                    strides: shape_to_strides(graph.shape(nid)),
                });
                kernel.outputs.insert(nid);
            } else {
                panic!()
            }
        }
    }
    println!("Printing kernels");
    for kernel in &kernels {
        println!();
        for op in &kernel.ops {
            println!("{op:?}");
        }
        println!();
    }
    return kernels;
}

fn shape_to_loops(shape: &[usize]) -> Vec<VOp> {
    shape
        .iter()
        .copied()
        .enumerate()
        .map(|(axis, dimension)| VOp::Loop { axis, dimension })
        .collect()
}

fn shape_to_strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1;
    let mut strides: Vec<usize> = shape
        .iter()
        .rev()
        .map(|d| {
            let temp = stride;
            stride *= d;
            temp
        })
        .collect();
    strides.reverse();
    return strides;
}

// Checks if kernel_x depends on kernel_y
fn depends_on(kernel_x_id: usize, kernel_y_id: usize, kernels: &[Kernel]) -> bool {
    let mut kernel_x_inputs = kernels[kernel_x_id].inputs.clone();
    let kernel_y_outputs = &kernels[kernel_y_id].outputs;
    while let Some(x) = kernel_x_inputs.pop_last() {
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
    false
}
