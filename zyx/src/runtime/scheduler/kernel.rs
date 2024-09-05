use std::collections::BTreeSet;

use crate::{runtime::{graph::Graph, ir::Scope, view::{StridedDim, View}}, shape::{Axis, Dimension}, tensor::TensorId};

use super::{shape_to_loops, vop::VOp};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Kernel {
    // Current shape of the kernel after all current ops
    pub(crate) shape: Vec<Dimension>,
    // Global loads
    pub(crate) inputs: BTreeSet<TensorId>,
    // Global stores
    pub(crate) outputs: BTreeSet<TensorId>,
    // Register variables
    pub(super) vars: BTreeSet<TensorId>,
    pub(crate) ops: Vec<VOp>,
}

impl Kernel {
    pub(super) fn debug(&self) {
        println!("Kernel shape: {:?}", self.shape);
        for vop in &self.ops {
            println!("{vop}");
        }
        println!();
    }

    pub(super) fn load(graph: &Graph, x: TensorId) -> Kernel {
        let shape: Vec<usize> = graph.shape(x).into();
        let mut ops: Vec<VOp> = shape_to_loops(&shape);
        ops.push(VOp::Load {
            z: x,
            zscope: Scope::Register,
            x,
            xscope: Scope::Global,
            view: View::new(&shape),
        });
        Kernel {
            shape,
            inputs: BTreeSet::from([x]),
            outputs: BTreeSet::new(),
            vars: BTreeSet::from([x]),
            ops,
        }
    }

    pub(super) fn store(&mut self, z: TensorId, graph: &Graph) {
        let store_op = VOp::Store {
            z,
            zscope: Scope::Global,
            xscope: Scope::Register,
            view: View::new(graph.shape(z)),
        };
        if self.ops.last().unwrap() != &store_op {
            self.ops.push(store_op);
            self.outputs.insert(z);
        }
    }

    pub(super) fn permute(&mut self, axes: &[usize]) {
        //self.debug();
        if (0..axes.len()).zip(axes).all(|(a, ca)| a == *ca) {
            // no permute
            return;
        }
        let shape: Vec<usize> = axes.iter().map(|a| self.shape[*a]).collect();
        //let mut permuted_loops: BTreeSet<usize> = axes.iter().copied().collect();
        let mut skip_loops = 0;
        let mut last_axis = axes.len() - 1;
        for op in self.ops.iter_mut().rev() {
            match op {
                VOp::Loop { dimension, .. } => {
                    if skip_loops > 0 {
                        skip_loops -= 1;
                    } else {
                        *dimension = shape[last_axis];
                        if last_axis > 0 {
                            last_axis -= 1;
                        }
                    }
                }
                VOp::Load { view, .. } | VOp::Store { view, .. } | VOp::Const { view, .. } => {
                    let n = view.rank();
                    let permute_axes: Vec<usize> = if last_axis > n {
                        // We actually need to check which axis view refers to, then check which loops those were
                        // and if and how those loops are permuted
                        todo!()
                    } else {
                        axes[..=last_axis]
                            .iter()
                            .copied()
                            .chain(last_axis + 1..n)
                            .collect()
                    };
                    view.permute(&permute_axes);
                }
                VOp::Reduce { num_axes, .. } => {
                    skip_loops += *num_axes;
                }
                _ => {}
            }
        }
        self.shape = shape.clone();
    }

    // Permutes first found loops, not the kernel as a whole
    pub(super) fn permute_loops(&mut self, op_id: usize, naxes: &[usize]) {
        if naxes.is_empty() { return }
        let mut axes = Vec::new();
        let mut dims = Vec::new();
        // Find which loops will be permuted
        for op in self.ops[op_id..].iter() {
            if let VOp::Loop { axis, dimension } = op {
                axes.push(*axis);
                dims.push(*dimension);
            }
        }
        assert_eq!(dims.len(), axes.len());
        let paxes: Vec<usize> = naxes.iter().map(|a| axes[*a]).collect();
        let pdims: Vec<usize> = naxes.iter().map(|a| dims[*a]).collect();
        // permute them
        let mut id = 0;
        // apply permute to ops
        for op in self.ops[op_id..].iter_mut() {
            match op {
                VOp::Loop { axis, dimension } => {
                    assert_eq!(axes[id], *axis);
                    *dimension = pdims[id];
                    id += 1;
                }
                VOp::Const { view, .. } | VOp::Load { view, .. } | VOp::Store { view, .. } | VOp::Accumulator { view, .. } => {
                    view.arbitrary_permute(&paxes);
                }
                _ => {}
            }
        }
    }

    pub(super) fn split_axis(&mut self, op_id: usize, dimensions: &[usize]) {
        //println!("Splitting {op_id} into {dimensions:?}");
        // First split loop at op_id
        let VOp::Loop { axis, dimension } = &mut self.ops[op_id] else {
            panic!()
        };
        *dimension = dimensions[0];
        let axis = *axis;
        let mut temp_axis = axis;
        let mut id = op_id;
        for dim in &dimensions[1..] {
            id += 1;
            temp_axis += 1;
            self.ops.insert(
                id,
                VOp::Loop {
                    axis: temp_axis,
                    dimension: *dim,
                },
            );
        }
        let mut num_loops = 0;
        // Update loops, loads and stores
        let mut reduce_end = false;
        for i in id + 1..self.ops.len() {
            match &mut self.ops[i] {
                // Then change axis ids for all following loops
                VOp::Loop { axis, .. } => {
                    if reduce_end {
                        break;
                    }
                    *axis += dimensions.len() - 1;
                    num_loops += 1;
                }
                VOp::Reduce { num_axes, .. } => {
                    if num_loops == 0 {
                        *num_axes += dimensions.len() - 1;
                        reduce_end = true;
                    }
                    if *num_axes > num_loops {
                        return
                    }
                    num_loops -= *num_axes;
                }
                // Then change all load and store operations in this loop in the same way.
                VOp::Load { view, .. } | VOp::Const { view, .. } | VOp::Store { view, .. } => {
                    view.split_axis(axis, dimensions);
                }
                _ => {}
            }
        }
        self.shape.remove(axis);
        for dim in dimensions.iter().rev() {
            self.shape.insert(axis, *dim);
        }
    }

    /*fn merge_axes(&mut self, op_id: usize, num_loops: usize) {
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
                VOp::Load { view, .. } | VOp::Const { view, .. } => {
                    let stride = view.0[axis_id + num_loops - 1].stride;
                    view.0[axis_id].dim = dim_size;
                    view.0[axis_id].stride = stride;
                    for _ in 0..num_loops - 1 {
                        view.0.remove(axis_id + 1);
                    }
                }
                _ => {}
            }
        }
    }*/

    /// Inserts loop at op_id, giving it axis id and dimension 1.
    /// All loops and views axis equal or greater then axis are increased by 1
    /// Does not change reduce op's num_axes
    /// This function also does not change kernel's shape!
    pub(super) fn insert_loop(&mut self, op_id: usize, axis: Axis) {
        let naxis = axis;
        for op in &mut self.ops {
            match op {
                VOp::Const { view, .. } | VOp::Load { view, .. } | VOp::Store { view, .. } => match view {
                    View::None => {}
                    View::Strided(dims) => {
                        dims.iter_mut().for_each(|StridedDim { axis, .. }| if *axis >= naxis { *axis += 1 });
                    }
                    View::Padded(dims, axes) => {
                        dims.iter_mut().for_each(|StridedDim { axis, .. }| if *axis >= naxis { *axis += 1 });
                        axes.axes.iter_mut().for_each(|(axes, _)| axes.iter_mut().for_each(|a| if *a >= naxis { *a += 1 }));
                    }
                }
                VOp::Loop { axis, .. } => if *axis >= naxis { *axis += 1 }
                _ => {}
            }
        }
        self.ops.insert(op_id, VOp::Loop { axis, dimension: 1 })
    }

    pub(super) fn shard_axis(&self) -> Option<(Axis, Dimension)> {
        // Shard axis is axis that is not gonna be locally cached,
        // which is usually the batch axis, but it can also be other axes.
        // Since we do not locally cache axis 0, we can for now always just return that
        //Some((0, self.shape[0]))
        None
    }
}