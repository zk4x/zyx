use std::{collections::BTreeSet, ops::Range};

use crate::{
    dtype::Constant, graph::Graph, ir::Scope, node::{BOp, ROp, UOp}, shape::{Axis, Dimension}, tensor::TensorId, view::View, DType
};

// Should be just Unary, Binary, Const, Copy, Loop, Reduce
#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum VOp {
    Loop {
        axis: Axis,
        len: Dimension,
    },
    // End the latest loop
    EndLoop,
    Const {
        z: TensorId,
        value: Constant,
        view: View,
    },
    Load {
        z: TensorId,
        zscope: Scope,
        zview: View,
        x: TensorId,
        xscope: Scope,
        xview: View,
        xdtype: DType,
    },
    Store {
        z: TensorId,
        zscope: Scope,
        zview: View,
        zdtype: DType,
        xscope: Scope,
        xview: View,
    },
    Accumulator {
        z: TensorId,
        rop: ROp,
        view: View,
        dtype: DType,
    },
    // Move is noop, just a marker for easy debugging
    // and to keep track of tensor ids
    Move {
        z: TensorId,
        x: TensorId,
        mop: MOp,
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
    // Synchronization for local and global memory
    Barrier {
        scope: Scope,
    },
}

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MOp {
    Expa,
    Perm,
    Resh,
    Padd,
}

impl std::fmt::Display for VOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const C_BLUE: &str = "\x1B[34m";
        const C_GREEN: &str = "\x1B[32m";
        const C_MAGENTA: &str = "\x1B[35m";
        const C_RED: &str = "\x1B[31m";
        const C_WHITE: &str = "\x1B[37m";
        const C_YELLOW: &str = "\x1B[33m";
        const C_RESET: &str = "\x1B[39m";
        match self {
            VOp::Const { z, value, view } => f.write_fmt(format_args!(
                "{C_WHITE}Const{C_RESET}       {z} <- value: {value}, {view}"
            )),
            VOp::Load {
                z,
                zscope,
                zview: _,
                x,
                xscope,
                xview,
                xdtype,
            } => f.write_fmt(format_args!(
                "{C_YELLOW}Load{C_RESET}        {z}[{zscope:?}] <- {x}[{xscope:?}, {xdtype}], {xview}"
            )),
            VOp::Store {
                z,
                zview,
                zscope,
                zdtype,
                xscope,
                xview: _,
            } => f.write_fmt(format_args!(
                "{C_RED}Store{C_RESET}        {z}[{zscope:?}] <- {xscope:?}, {zview}, {zdtype}"
            )),
            VOp::Loop {
                axis,
                len: dimension,
            } => f.write_fmt(format_args!(
                "{C_GREEN}Loop{C_RESET}        axis: {axis}, dimension: {dimension}"
            )),
            VOp::Accumulator { z, rop, view, dtype } => f.write_fmt(format_args!(
                "{C_BLUE}Accum{C_RESET}.{rop:?}   {z}, shape: {:?}, {dtype}",
                view.shape()
            )),
            VOp::EndLoop => f.write_fmt(format_args!("{C_BLUE}EndLoop{C_RESET} ")),
            VOp::Move { z, x, mop } => f.write_fmt(format_args!(
                "{C_WHITE}Move{C_RESET}.{mop:?}   {z} <- {x}"
            )),
            VOp::Unary { z, x, uop } => {
                let mut len = format!("{uop:?}").len();
                if len > 5 {
                    len = 5;
                }
                f.write_fmt(format_args!(
                    "{C_WHITE}Unary{C_RESET}.{uop:?}{} {z} <- {x}",
                    " ".repeat(5 - len)
                ))
            }
            VOp::Binary { z, x, y, bop } => f.write_fmt(format_args!(
                "{C_WHITE}Binary{C_RESET}.{bop:?}  {z} <- {x}, {y}"
            )),
            VOp::Barrier { scope } => f.write_fmt(format_args!("{C_MAGENTA}Barrier{C_RESET}({scope})")),
        }
    }
}

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Kernel {
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

    pub(super) fn with_shape(shape: &[usize]) -> Kernel {
        Kernel {
            ops: shape_to_loops(shape),
        }
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
            .filter_map(|op| {
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
            .filter_map(|op| {
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
                    if end_loops == 0 && *zscope == Scope::Register {
                        res.insert(*z);
                    }
                }
                VOp::Loop { .. } => {
                    if end_loops > 0 {
                        end_loops -= 1;
                    }
                }
                VOp::Store { .. } | VOp::Barrier { .. } => {}
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
            zview: View::none(),
            x,
            xscope: Scope::Global,
            xview: View::contiguous(&shape),
            xdtype: graph.dtype(x),
        });
        Kernel { ops }
    }

    /// Store z just after the last operation was executed with it
    pub(super) fn store(&mut self, z: TensorId, zview: View, zdtype: DType) {
        if let Some(&VOp::Store {
            z: nz,
            zview: ref nzview,
            ..
        }) = self.ops.last()
        {
            if z == nz && &zview == nzview {
                return;
            }
        }
        assert!(zview.numel() < 1024 * 1024 * 1024, "Too big store.");
        let store_op = VOp::Store {
            z,
            zview,
            zscope: Scope::Global,
            zdtype,
            xscope: Scope::Register,
            xview: View::none(),
        };
        self.ops.push(store_op);
        //panic!("Storing");
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
        self.ops
            .iter()
            .any(|op| matches!(op, VOp::Accumulator { .. }))
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
                        last_axis = last_axis.saturating_sub(1);
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

    // TODO remove this in favor of reshape
    pub(super) fn split_axis(&mut self, op_id: usize, dimensions: &[usize]) {
        //self.debug();
        //println!("Splitting {op_id} into {dimensions:?}");
        // First split loop at op_id
        let VOp::Loop {
            axis,
            len: dimension,
        } = &mut self.ops[op_id]
        else {
            unreachable!()
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
                    #[allow(clippy::range_plus_one)]
                    {
                        view.reshape(axis..axis + 1, dimensions);
                    }
                }
                _ => {}
            }
        }
        //self.debug();
    }

    /// Inserts loop at `op_id`, giving it axis id and dimension 1.
    /// All loops and views axis equal or greater then axis are increased by 1
    /// Does not change reduce op's `num_axes`
    /// This function also does not change kernel's shape!
    pub(super) fn insert_loop(&mut self, op_id: usize, axis: Axis) {
        let naxis = axis;
        for op in &mut self.ops {
            match op {
                VOp::Const { view, .. }
                | VOp::Store { zview: view, .. }
                | VOp::Load { xview: view, .. }
                | VOp::Accumulator { view, .. } => view.insert_loop(naxis),
                VOp::Loop { axis, .. } => {
                    if *axis >= naxis {
                        *axis += 1;
                    }
                }
                _ => {}
            }
        }
        self.ops.insert(op_id, VOp::Loop { axis, len: 1 });
    }

    #[allow(clippy::unused_self)]
    pub(super) const fn shard_axis(&self) -> Option<(Axis, Dimension)> {
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
                VOp::EndLoop => {
                    shape.pop();
                }
                VOp::Unary { .. } | VOp::Binary { .. } => {
                    flop += shape.iter().product::<usize>() as u128;
                }
                VOp::Accumulator { .. }
                | VOp::Const { .. }
                | VOp::Move { .. }
                | VOp::Barrier { .. } => {}
            }
        }
        (flop, mem_read, mem_write)
    }

    pub(super) fn can_be_zero_padded(&self) -> bool {
        self.ops.iter().all(|op| match op {
            // For now just do not pad reduce kernels
            //matches!(rop, ROp::Sum),
            // TODO this can be later removed, but it's a trade-off,
            // it makes kernels bigger, but harder to reason about
            VOp::Accumulator { .. } | VOp::Store { .. } => false,
            _ => true,
        })
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn get_reshape_pattern(
        &self,
        nshape: &[usize],
    ) -> Option<(
        usize,                             // number of new loops to be inserted
        Vec<(Range<usize>, Range<usize>)>, // range and new shape for reshapes
    )> {
        let mut unmergeable_axes = Vec::new();
        let mut last_op_i = 0;
        for (i, op) in self.ops.iter().enumerate() {
            if let &VOp::Loop { axis, .. } = op {
                if last_op_i != 0 && i != last_op_i + 1 {
                    unmergeable_axes.push(axis);
                }
                last_op_i = i;
            }
        }
        get_reshape_pattern(&self.shape(), nshape, &unmergeable_axes)
    }

    pub(super) fn reshape(&mut self, shape: &[usize]) -> bool {
        // If this is just a reshape of kernel with only unary ops and contiguous loads
        // and stores, we can remove old loops and replace them with new loops.
        //println!("Reshape");
        // TODO this first case can be removed
        if self.ops.iter().all(|op| match op {
            VOp::Loop { .. }
            | VOp::Unary { .. }
            | VOp::Binary { .. }
            | VOp::Barrier { .. }
            | VOp::Move { .. } => true,
            VOp::Load { xview: view, .. }
            | VOp::Store { zview: view, .. }
            | VOp::Const { view, .. } => view.is_contiguous(),
            VOp::Accumulator { .. } | VOp::EndLoop => false, // | VOp::Reduce { .. }
        }) {
            //println!("Before reshape continuous.");
            //kernel.debug();
            // Remove old loops
            for _ in 0..self.shape().len() {
                self.ops.remove(0);
            }
            // Put in new loops
            for op in shape_to_loops(shape).into_iter().rev() {
                self.ops.insert(0, op);
            }
            // Change Reshape loads and stores
            for op in &mut self.ops {
                match op {
                    VOp::Load { xview: view, .. }
                    | VOp::Const { view, .. }
                    | VOp::Store { zview: view, .. } => {
                        *view = View::contiguous(shape);
                    }
                    _ => {}
                }
            }
            //println!("Reshaping continuous.");
            //kernel.debug();
            true
        } else if let Some((new_loops, reshapes)) = self.get_reshape_pattern(shape) {
            let _ = new_loops; // TODO get new_loops working
                               //println!("Reshapes: {reshapes:?}");
            for (org_sh, sh) in reshapes.iter().rev() {
                let mut op_i = self.ops.len();
                'a: loop {
                    op_i -= 1;
                    if let VOp::Loop { axis, .. } = &mut self.ops[op_i] {
                        //println!("{org_sh:?} -> {sh:?}");
                        match (*axis).cmp(&(org_sh.end - 1)) {
                            std::cmp::Ordering::Less => {}
                            std::cmp::Ordering::Equal => {
                                // remove org_sh.end - org_sh.start ops from kernel. They should all be loops.
                                // insert respective loops from new shape
                                let n = org_sh.end - org_sh.start;
                                let i = (op_i + 1) - n;
                                //println!("Removing {i}");
                                for _ in 0..n {
                                    self.ops.remove(i);
                                }
                                //self.debug();
                                for a in sh.clone().rev() {
                                    //println!("Axis {a}, shape {shape:?}");
                                    self.ops.insert(
                                        i,
                                        VOp::Loop {
                                            axis: a + org_sh.start - sh.start,
                                            len: shape[a],
                                        },
                                    );
                                }
                                //self.debug();
                                break 'a;
                            }
                            std::cmp::Ordering::Greater => {
                                *axis += sh.end - sh.start;
                                *axis -= org_sh.end - org_sh.start;
                            }
                        }
                    }
                }
                //self.debug();
            }
            for op in &mut self.ops {
                match op {
                    VOp::Const { view, .. }
                    | VOp::Load { xview: view, .. }
                    | VOp::Store { zview: view, .. }
                    | VOp::Accumulator { view, .. } => {
                        for (org_sh, sh) in reshapes.iter().rev() {
                            view.reshape(org_sh.clone(), &shape[sh.clone()]);
                        }
                    }
                    VOp::Loop { .. }
                    | VOp::EndLoop
                    | VOp::Move { .. }
                    | VOp::Unary { .. }
                    | VOp::Binary { .. }
                    | VOp::Barrier { .. } => {}
                }
            }
            //self.debug();
            // TODO deal with loop inserts
            assert_eq!(
                self.shape(),
                shape,
                "Shape after reshape split is incorrect."
            );
            true
        } else {
            false
        }
    }
}

/// Searches which dimensions can be:
/// 1. insert new loops to the end of the kernel
/// 2. merged
/// 3. split
/// 4. reshaped without affecting reduce dims
///
/// If neither of those are possible, None is returned. last tensor must be stored and new kernel must be created.
#[allow(clippy::type_complexity)]
fn get_reshape_pattern(
    // original shape
    shape: &[usize],
    // new shape
    nshape: &[usize],
    // 6 means there are ops between axes 5 and 6, thus 5 and 6 cannot be merged
    // 3 means there are ops between axes 2 and 3, thus 2 and 3 cannot be merged
    unmergeable_axes: &[usize],
) -> Option<(
    // number of new loops to be inserted
    usize,
    // range and new shape for reshapes
    Vec<(Range<usize>, Range<usize>)>,
)> {
    // reshape
    // 2, 4, 1, 3, 1,    4, 5, 2
    //       8, 3, 1, 2, 2, 2, 5
    let mut reshapes = Vec::new();

    let mut split_axes = 0..1;
    let mut merge_axes = 0..1;
    'a: while merge_axes.end <= shape.len() && split_axes.end <= nshape.len() {
        match shape[merge_axes.clone()]
            .iter()
            .product::<usize>()
            .cmp(&nshape[split_axes.clone()].iter().product())
        {
            std::cmp::Ordering::Less => {
                merge_axes.end += 1;
            }
            std::cmp::Ordering::Greater => {
                split_axes.end += 1;
            }
            std::cmp::Ordering::Equal => {
                if let Some(d) = shape.get(merge_axes.end) {
                    if *d == 1 && split_axes.end == nshape.len() {
                        merge_axes.end += 1;
                        continue 'a;
                    }
                }
                if let Some(d) = nshape.get(split_axes.end) {
                    if *d == 1 && merge_axes.end == shape.len() {
                        split_axes.end += 1;
                        continue 'a;
                    }
                }
                if (merge_axes.len(), split_axes.len()) != (1, 1) {
                    // reshape
                    // If merge range contains unmergeable axes, return None
                    // Axes are not mergeable if there is some ops between those axes
                    if unmergeable_axes
                        .iter()
                        .any(|a| merge_axes.contains(a) && merge_axes.contains(&(a - 1)))
                    {
                        return None;
                    }
                    reshapes.push((merge_axes.clone(), split_axes.clone()));
                }
                #[allow(clippy::range_plus_one)]
                {
                    merge_axes = merge_axes.end..merge_axes.end + 1;
                    split_axes = split_axes.end..split_axes.end + 1;
                }
            }
        }
    }
    Some((0, reshapes))
}

fn shape_to_loops(shape: &[usize]) -> Vec<VOp> {
    let mut res = Vec::with_capacity(20);
    for (axis, dimension) in shape.iter().copied().enumerate() {
        res.push(VOp::Loop {
            axis,
            len: dimension,
        });
    }
    res
}

#[test]
fn reshape_pattern() {
    let shape = [2, 4, 1, 3, 1, 4, 5, 2];
    let nshape = [8, 3, 1, 2, 2, 2, 5];
    let r = get_reshape_pattern(&shape, &nshape, &[]);
    assert_eq!(
        r,
        Some((
            0,
            vec![(0..2, 0..1), (2..4, 1..2), (5..6, 3..5), (6..8, 5..7)]
        ))
    );
    let shape = [2, 2, 1, 2, 2];
    let nshape = [2, 2, 1, 2, 2, 1];
    let r = get_reshape_pattern(&shape, &nshape, &[]);
    assert_eq!(r, Some((0, vec![(4..5, 4..6)])));
    let shape = [1, 3, 4, 5];
    let nshape = [3, 20];
    let r = get_reshape_pattern(&shape, &nshape, &[]);
    assert_eq!(r, Some((0, vec![(0..2, 0..1), (2..4, 1..2)])));
}
