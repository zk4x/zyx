use std::collections::BTreeSet;

use crate::{
    runtime::{
        graph::Graph,
        ir::Scope,
        view::{StridedDim, View},
    },
    shape::{Axis, Dimension},
    tensor::TensorId,
};

use super::{shape_to_loops, vop::VOp};

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Kernel {
    pub(crate) ops: Vec<VOp>,
}

impl Kernel {
    pub(super) fn debug(&self) {
        println!(
            "Kernel shape: {:?}, inputs: {:?}, outputs: {:?}",
            self.shape(),
            self.inputs(),
            self.outputs()
        );
        let mut first_loops = true;
        let mut indent = String::new();
        for vop in &self.ops {
            match vop {
                VOp::Loop { .. } => {
                    println!("{indent}{vop}");
                    if !first_loops {
                        indent += "  ";
                    }
                }
                VOp::EndLoop => {
                    indent.pop();
                    indent.pop();
                    println!("{indent}{vop}");
                }
                _ => {
                    println!("{indent}{vop}");
                    first_loops = false;
                }
            }
        }
        println!();
    }

    pub(super) fn shape(&self) -> Vec<usize> {
        self.ops
            .iter()
            .map_while(|op| {
                if let VOp::Loop { len, .. } = op {
                    Some(*len)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Tensor ids that are read (maybe also written)
    pub(super) fn inputs(&self) -> BTreeSet<TensorId> {
        self.ops
            .iter()
            .flat_map(|op| {
                if let VOp::Load { x, xscope, .. } = op {
                    if *xscope == Scope::Global {
                        Some(*x)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    /// Tensor ids that are written (maybe also read)
    pub(super) fn outputs(&self) -> BTreeSet<TensorId> {
        self.ops
            .iter()
            .flat_map(|op| {
                if let VOp::Store { z, zscope, .. } = op {
                    if *zscope == Scope::Global {
                        Some(*z)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }

    pub(super) fn vars(&self) -> BTreeSet<TensorId> {
        let mut res = BTreeSet::new();
        let mut end_loops = 0;
        for op in self.ops.iter().rev() {
            match op {
                VOp::Const { z, .. }
                | VOp::Accumulator { z, .. }
                | VOp::Move { z, .. }
                | VOp::Unary { z, .. }
                | VOp::Binary { z, .. } => {
                    if end_loops == 0 {
                        res.insert(*z);
                    }
                }
                VOp::Load { z, zscope, .. } => {
                    if end_loops == 0 {
                        if *zscope == Scope::Register {
                            res.insert(*z);
                        }
                    }
                }
                VOp::Store { .. } => {}
                VOp::Loop { .. } => {
                    if end_loops > 0 {
                        end_loops -= 1;
                    }
                }
                VOp::Barrier { .. } => {}
                VOp::EndLoop { .. } => {
                    // Only variables defined after end of loops can be used
                    end_loops += 1;
                }
            }
        }
        res
    }

    pub(super) fn load(x: TensorId, graph: &Graph) -> Kernel {
        let shape: Vec<usize> = graph.shape(x).into();
        let mut ops: Vec<VOp> = shape_to_loops(&shape);
        ops.push(VOp::Load {
            z: x,
            zscope: Scope::Register,
            zview: View::None,
            x,
            xscope: Scope::Global,
            xview: View::new(&shape),
        });
        Kernel { ops }
    }

    /// Store z just after the last operation was executed with it
    pub(super) fn store(&mut self, z: TensorId, zview: View) {
        if let Some(&VOp::Store { z: nz, zview: ref nzview, .. }) = self.ops.last() {
            if z == nz && &zview == nzview {
                return
            }
        }
        assert!(zview.numel() < 1024 * 1024 * 1024,  "Too big store.");
        let store_op = VOp::Store {
            z,
            zview,
            zscope: Scope::Global,
            xscope: Scope::Register,
            xview: View::None,
        };
        self.ops.push(store_op);
        /*for (id, op) in self.ops.iter().enumerate().rev() {
            match op {
                VOp::Load { x, xview, .. } => {
                    if *x == z && xview == &zview {
                        return
                    }
                }
                VOp::Store { z: x, zview: xview, .. } => {
                    if *x == z && xview == &zview {
                        return
                    }
                }
                VOp::Move { z: x, .. } => {
                    if z == *x {
                        self.ops.insert(id+1, store_op);
                        return
                    }
                }
                VOp::Unary { z: x, .. } => {
                    if z == *x {
                        self.ops.insert(id+1, store_op);
                        return
                    }
                }
                VOp::Binary { z: x, .. } => {
                    if z == *x {
                        self.ops.insert(id+1, store_op);
                        return
                    }
                }
                _ => {}
            }
        }*/
    }

    pub(super) fn is_reduce(&self) -> bool {
        self.ops.iter().any(|op| matches!(op, VOp::Accumulator { .. }))
    }

    pub(super) fn permute(&mut self, axes: &[usize]) {
        //self.debug();
        if (0..axes.len()).zip(axes).all(|(a, ca)| a == *ca) {
            // no permute
            return;
        }
        let shape: Vec<usize> = axes.iter().map(|a| self.shape()[*a]).collect();
        //let mut permuted_loops: BTreeSet<usize> = axes.iter().copied().collect();
        let mut skip_loops = 0;
        let mut last_axis = axes.len() - 1;
        for op in self.ops.iter_mut().rev() {
            match op {
                VOp::Loop { len: dimension, .. } => {
                    if skip_loops > 0 {
                        skip_loops -= 1;
                    } else {
                        *dimension = shape[last_axis];
                        if last_axis > 0 {
                            last_axis -= 1;
                        }
                    }
                }
                VOp::Load { xview: view, .. }
                | VOp::Store { zview: view, .. }
                | VOp::Const { view, .. } => {
                    //| VOp::Accumulator { view, .. } => {
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
                VOp::EndLoop => {
                    skip_loops += 1;
                }
                _ => {}
            }
        }
    }

    // Permutes first found loops, not the kernel as a whole
    /*pub(super) fn permute_loops(&mut self, op_id: usize, naxes: &[usize]) {
        if naxes.is_empty() { return }
        let mut axes = Vec::new();
        let mut dims = Vec::new();
        // Find which loops will be permuted
        for op in self.ops[op_id..].iter() {
            if let VOp::Loop { axis, len: dimension } = op {
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
                VOp::Loop { axis, len: dimension } => {
                    assert_eq!(axes[id], *axis);
                    *dimension = pdims[id];
                    id += 1;
                }
                VOp::Const { view, .. } | VOp::Load { xview: view, .. } | VOp::Store { zview: view, .. } | VOp::Accumulator { view, .. } => {
                    view.arbitrary_permute(&paxes);
                }
                _ => {}
            }
        }
    }*/

    pub(super) fn split_axis(&mut self, op_id: usize, dimensions: &[usize]) {
        //println!("Splitting {op_id} into {dimensions:?}");
        // First split loop at op_id
        let VOp::Loop {
            axis,
            len: dimension,
        } = &mut self.ops[op_id]
        else {
            panic!()
        };
        *dimension = dimensions[0];
        let new_dim_count = dimensions.len() - 1;
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
                    len: *dim,
                },
            );
        }
        let mut num_loops = 0;
        // Update loops, loads and stores
        for i in id + 1..self.ops.len() {
            if self.ops[i] == VOp::EndLoop {
                if num_loops == 0 {
                    for _ in 0..new_dim_count {
                        self.ops.insert(i, VOp::EndLoop);
                    }
                    break;
                }
                num_loops -= 1;
            }
            match &mut self.ops[i] {
                // Then change axis ids for all following loops
                VOp::Loop { axis, .. } => {
                    *axis += new_dim_count;
                    num_loops += 1;
                }
                // Then change all load and store operations in this loop in the same way.
                VOp::Load { xview: view, .. }
                | VOp::Store { zview: view, .. }
                | VOp::Const { view, .. }
                | VOp::Accumulator { view, .. } => {
                    view.split_axis(axis, dimensions);
                }
                _ => {}
            }
        }
        //self.debug();
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
                VOp::Const { view, .. }
                | VOp::Store { zview: view, .. }
                | VOp::Load { xview: view, .. }
                | VOp::Accumulator { view, .. } => match view {
                    View::None => {}
                    View::Strided(dims) => {
                        dims.iter_mut().for_each(|StridedDim { axis, .. }| {
                            if *axis >= naxis {
                                *axis += 1
                            }
                        });
                    }
                    View::Padded(dims, axes) => {
                        dims.iter_mut().for_each(|StridedDim { axis, .. }| {
                            if *axis >= naxis {
                                *axis += 1
                            }
                        });
                        axes.iter_mut().for_each(|(axes, _)| {
                            axes.iter_mut().for_each(|a| {
                                if *a >= naxis {
                                    *a += 1
                                }
                            })
                        });
                    }
                },
                VOp::Loop { axis, .. } => {
                    if *axis >= naxis {
                        *axis += 1
                    }
                }
                _ => {}
            }
        }
        self.ops.insert(op_id, VOp::Loop { axis, len: 1 })
    }

    pub(super) fn shard_axis(&self) -> Option<(Axis, Dimension)> {
        // Shard axis is axis that is not gonna be locally cached,
        // which is usually the batch axis, but it can also be other axes.
        // Since we do not locally cache axis 0, we can for now always just return that
        //Some((0, self.shape[0]))
        None
    }

    pub(super) fn flop_mem_rw(&self) -> (u128, u128, u128) {
        let mut shape = Vec::new();
        let mut flop = 0;
        let mut mem_read = 0;
        let mut mem_write = 0;
        for op in &self.ops {
            match op {
                &VOp::Loop { len, .. } => {
                    shape.push(len);
                }
                VOp::Const { .. } => {}
                &VOp::Load { xscope, .. } => {
                    // Note that this calculates actual read speed, even if the load accesses the same
                    // value multiple times. This is usefull so that we can see whether the kernel
                    // is compute bound or memory bound.
                    if xscope == Scope::Global {
                        mem_read += shape.iter().product::<usize>() as u128;
                    }
                }
                &VOp::Store { zscope, .. } => {
                    if zscope == Scope::Global {
                        mem_write += shape.iter().product::<usize>() as u128;
                    }
                }
                VOp::Accumulator { .. } => {}
                VOp::EndLoop => {
                    shape.pop();
                }
                VOp::Move { .. } => {}
                VOp::Unary { .. } => {
                    flop += shape.iter().product::<usize>() as u128;
                }
                VOp::Binary { .. } => {
                    flop += shape.iter().product::<usize>() as u128;
                }
                VOp::Barrier { .. } => {}
            }
        }
        (flop, mem_read, mem_write)
    }

    pub(super) fn can_be_zero_padded(&self) -> bool {
        self.ops.iter().all(|op| match op {
            // For now just do not pad reduce kernels
            VOp::Accumulator { .. } => false, //matches!(rop, ROp::Sum),
            // TODO this can be later removed, but it's a trade-off,
            // it makes kernels bigger,  but they will have to contain branches.
            VOp::Store { .. } => false,
            _ => true,
        })
    }
}
