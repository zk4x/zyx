//! Kernel represents hardware programs.

// TODO loosen requirements for is_expandable
// and posdsibly remove requirements for is_paddable
// is_reshapable more or less cannot be loosened.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use crate::{
    backend::{BufferId, Device, MemoryPool},
    dtype::Constant,
    graph::Graph,
    ir::Scope,
    node::{BOp, ROp, UOp},
    optimizer::Optimizer,
    shape::{Axis, Dimension},
    slab::Id,
    tensor::TensorId,
    view::View,
    DType, DebugMask, ZyxError,
};

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Kernel {
    pub(super) ops: Vec<Op>,
    // Mapind from tensors ids to load and store ids
    pub(super) tensors: BTreeMap<TId, TensorId>,
    // Outputs of the kernel that are unused (not stored yet)
    pub(super) outputs: BTreeMap<TensorId, TId>,
    pub(super) max_id: TId,
}

// Tensor id in a kernel
pub(super) type TId = u16;

// TODO this needs to be smaller, since it's stored on the disk
#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Op {
    Loop { axis: Axis, len: Dimension },
    // End the latest loop
    EndLoop,
    Const { z: TId, value: Constant, view: View },
    Load { z: TId, zscope: Scope, zview: View, xscope: Scope, xview: View, xdtype: DType },
    Store { z: TId, zscope: Scope, zview: View, zdtype: DType, xscope: Scope, xview: View },
    Accumulator { z: TId, rop: ROp, view: View, dtype: DType },
    // Move is noop, just a marker for easy debugging
    // and to keep track of tensor ids
    Move { z: TId, x: TId, mop: MOp },
    Unary { z: TId, x: TId, uop: UOp },
    Binary { z: TId, x: TId, y: TId, bop: BOp },
    // Synchronization for local and global memory
    Barrier { scope: Scope },
}

#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MOp {
    Expa,
    Perm,
    Resh,
    Padd,
}

impl Kernel {
    pub(super) fn constant(nid: TensorId, value: Constant) -> Kernel {
        let mut ops = Vec::with_capacity(50);
        ops.push(Op::Loop { axis: 0, len: 1 });
        ops.push(Op::Const { z: 0, value, view: View::contiguous(&[1]) });
        Kernel { max_id: 0, ops, tensors: BTreeMap::new(), outputs: BTreeMap::from([(nid, 0)]) }
    }

    pub(super) fn leaf(nid: TensorId, shape: &[usize], dtype: DType) -> Kernel {
        let mut ops = Vec::with_capacity(50);
        for (axis, dimension) in shape.iter().copied().enumerate() {
            ops.push(Op::Loop { axis, len: dimension });
        }
        ops.push(Op::Load {
            z: 0,
            zscope: Scope::Register,
            zview: View::none(),
            xscope: Scope::Global,
            xview: View::contiguous(&shape),
            xdtype: dtype,
        });
        Kernel {
            max_id: 0,
            ops,
            outputs: BTreeMap::from([(nid, 0)]),
            tensors: BTreeMap::from([(0, nid)]),
        }
    }

    /*pub(super) fn get_tensor_id(&self, tid: TId) -> TensorId {
        *self.tensors.iter().find(|(tidx, _)| **tidx == tid).unwrap().0
    }*/

    pub(super) fn shape(&self) -> Vec<usize> {
        self.ops
            .iter()
            .map_while(|op| {
                if let Op::Loop { len, .. } = op {
                    Some(*len)
                } else {
                    None
                }
            })
            .collect()
    }

    pub(super) fn is_reshapable(&self, shape: &[usize]) -> bool {
        // TODO remove the first case
        self.ops.iter().all(|op| match op {
            Op::Loop { .. }
            | Op::Unary { .. }
            | Op::Binary { .. }
            | Op::Barrier { .. }
            | Op::Move { .. } => true,
            Op::Load { xview: view, .. }
            | Op::Store { zview: view, .. }
            | Op::Const { view, .. } => view.is_contiguous(),
            Op::Accumulator { .. } | Op::EndLoop => false,
        }) | self.get_reshape_pattern(shape).is_some()
    }

    pub(super) fn reshape(&mut self, shape: &[usize]) {
        // If this is just a reshape of kernel with only unary ops and contiguous loads
        // and stores, we can remove old loops and replace them with new loops.
        //println!("Reshape");
        // TODO this first case can be removed
        if self.ops.iter().all(|op| match op {
            Op::Loop { .. }
            | Op::Unary { .. }
            | Op::Binary { .. }
            | Op::Barrier { .. }
            | Op::Move { .. } => true,
            Op::Load { xview: view, .. }
            | Op::Store { zview: view, .. }
            | Op::Const { view, .. } => view.is_contiguous(),
            Op::Accumulator { .. } | Op::EndLoop => false, // | Op::Reduce { .. }
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
                    Op::Load { xview: view, .. }
                    | Op::Const { view, .. }
                    | Op::Store { zview: view, .. } => {
                        *view = View::contiguous(shape);
                    }
                    _ => {}
                }
            }
            //println!("Reshaping continuous.");
            //kernel.debug();
        } else if let Some((new_loops, reshapes)) = self.get_reshape_pattern(shape) {
            let _ = new_loops; // TODO get new_loops working
                               //println!("Reshapes: {reshapes:?}");
            for (org_sh, sh) in reshapes.iter().rev() {
                let mut op_i = self.ops.len();
                'a: loop {
                    op_i -= 1;
                    if let Op::Loop { axis, .. } = &mut self.ops[op_i] {
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
                                        Op::Loop {
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
                    Op::Const { view, .. }
                    | Op::Load { xview: view, .. }
                    | Op::Store { zview: view, .. }
                    | Op::Accumulator { view, .. } => {
                        for (org_sh, sh) in reshapes.iter().rev() {
                            view.reshape(org_sh.clone(), &shape[sh.clone()]);
                        }
                    }
                    Op::Loop { .. }
                    | Op::EndLoop
                    | Op::Move { .. }
                    | Op::Unary { .. }
                    | Op::Binary { .. }
                    | Op::Barrier { .. } => {}
                }
            }
            //self.debug();
            // TODO deal with loop inserts
            assert_eq!(
                self.shape(),
                shape,
                "Shape after reshape split is incorrect."
            );
        }
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
            if let &Op::Loop { axis, .. } = op {
                if last_op_i != 0 && i != last_op_i + 1 {
                    unmergeable_axes.push(axis);
                }
                last_op_i = i;
            }
        }
        get_reshape_pattern(&self.shape(), nshape, &unmergeable_axes)
    }

    // TODO remove this in favor of reshape
    pub(super) fn split_axis(&mut self, op_id: usize, dimensions: &[usize]) {
        //self.debug();
        //println!("Splitting {op_id} into {dimensions:?}");
        // First split loop at op_id
        let Op::Loop { axis, len: dimension } = &mut self.ops[op_id] else { unreachable!() };
        *dimension = dimensions[0];
        let new_dim_count = dimensions.len() - 1;
        let axis = *axis;
        let mut temp_axis = axis;
        let mut id = op_id;
        for dim in &dimensions[1..] {
            id += 1;
            temp_axis += 1;
            self.ops.insert(id, Op::Loop { axis: temp_axis, len: *dim });
        }
        let mut num_loops = 0;
        // Update loops, loads and stores
        for i in id + 1..self.ops.len() {
            if self.ops[i] == Op::EndLoop {
                if num_loops == 0 {
                    for _ in 0..new_dim_count {
                        self.ops.insert(i, Op::EndLoop);
                    }
                    break;
                }
                num_loops -= 1;
            }
            match &mut self.ops[i] {
                // Then change axis ids for all following loops
                Op::Loop { axis, .. } => {
                    *axis += new_dim_count;
                    num_loops += 1;
                }
                // Then change all load and store operations in this loop in the same way.
                Op::Load { xview: view, .. }
                | Op::Store { zview: view, .. }
                | Op::Const { view, .. }
                | Op::Accumulator { view, .. } => {
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

    pub(super) fn debug(&self) {
        println!(
            "Kernel shape: {:?}, outputs: {:?}, tensors: {:?}",
            self.shape(),
            self.outputs,
            self.tensors
        );
        let mut first_loops = true;
        let mut indent = String::new();
        for vop in &self.ops {
            match vop {
                Op::Loop { .. } => {
                    println!("{indent}{vop}");
                    if !first_loops {
                        indent += "  ";
                    }
                }
                Op::EndLoop => {
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

    pub(super) fn is_expandable(&self) -> bool {
        self.ops.iter().all(|op| !matches!(op, Op::Store { .. } | Op::Accumulator { .. }))
    }

    pub(super) fn expand(&mut self, shape: &[usize]) {
        //println!("Expanding");
        //kernel.debug();
        assert_eq!(shape.len(), self.shape().len());
        let mut expand_axes = BTreeSet::new();
        for (a, d) in self.shape().into_iter().enumerate() {
            if d != shape[a] {
                assert_eq!(d, 1);
                expand_axes.insert(a);
            }
        }
        // We go over ops in reverse, increasing last loops dimension
        //println!("expand_axes = {expand_axes:?}");
        let mut done_expanding = BTreeSet::new();
        for op in self.ops.iter_mut().rev() {
            match op {
                Op::Loop { axis, len: dimension } => {
                    if expand_axes.contains(axis) && done_expanding.insert(*axis) {
                        assert_eq!(*dimension, 1);
                        *dimension = shape[*axis];
                    }
                }
                Op::Load { xview: view, .. } | Op::Const { view, .. } => {
                    // Done expanding marks which loops are behind us,
                    // so we need to only adjust strides to 0 in axes for those axes that are not behind us yet.
                    for a in expand_axes.difference(&done_expanding) {
                        view.expand(*a, shape[*a]);
                    }
                }
                Op::Store { .. } => unreachable!(),
                _ => {}
            }
        }
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
                Op::Loop { len: dimension, .. } => {
                    if skip_loops > 0 {
                        skip_loops -= 1;
                    } else {
                        *dimension = shape[last_axis];
                        last_axis = last_axis.saturating_sub(1);
                    }
                }
                Op::Load { xview: view, .. }
                | Op::Store { zview: view, .. }
                | Op::Const { view, .. } => {
                    //| VOp::Accumulator { view, .. } => {
                    let n = view.rank();
                    let permute_axes: Vec<usize> = if last_axis > n {
                        // We actually need to check which axis view refers to, then check which loops those were
                        // and if and how those loops are permuted
                        todo!()
                    } else {
                        axes[..=last_axis].iter().copied().chain(last_axis + 1..n).collect()
                    };
                    view.permute(&permute_axes);
                }
                Op::EndLoop => {
                    skip_loops += 1;
                }
                _ => {}
            }
        }
    }

    pub(super) fn is_paddable(&self) -> bool {
        self.ops.iter().all(|op| match op {
            // For now just do not pad reduce kernels
            //matches!(rop, ROp::Sum),
            // TODO this can be later removed, but it's a trade-off,
            // it makes kernels bigger, but harder to reason about
            Op::Accumulator { .. } | Op::Store { .. } => false,
            _ => true,
        })
    }

    pub(super) fn pad(&mut self, padding: &[(isize, isize)]) {
        //kernel.debug();
        let rank = self.shape().len();
        // Get which axes are padded
        let mut padded_axes = BTreeMap::new();
        for (op, &p) in self.ops[..rank].iter().rev().zip(padding) {
            let &Op::Loop { axis, .. } = op else { unreachable!() };
            padded_axes.insert(axis, p);
        }
        // Apply padding
        let mut num_paddings = padding.len();
        //println!("Padded axes: {padded_axes:?}");
        for op in &mut self.ops {
            match op {
                Op::Loop { axis, len } => {
                    if let Some((lp, rp)) = padded_axes.get(axis) {
                        *len = usize::try_from(isize::try_from(*len).unwrap() + lp + rp).unwrap();
                    }
                }
                Op::EndLoop => {
                    num_paddings -= 1;
                    if num_paddings == 0 {
                        break;
                    }
                }
                Op::Const { view, .. }
                | Op::Load { xview: view, .. }
                | Op::Store { zview: view, .. }
                | Op::Accumulator { view, .. } => {
                    for (&axis, &(lp, rp)) in &padded_axes {
                        view.pad(axis, lp, rp);
                    }
                }
                _ => {}
            }
        }
    }

    pub(super) fn reduce(
        &mut self,
        xt: TId,
        shape: &[usize],
        axes: &[usize],
        dtype: DType,
        rop: ROp,
    ) {
        let permute_axes: Vec<usize> =
            (0..shape.len()).filter(|a| !axes.contains(a)).chain(axes.iter().copied()).collect();
        //println!("Permute axes in reduce: {permute_axes:?}");
        self.permute(&permute_axes);

        // We can also just merge these reduce loops into single loop, since it gets removed
        // from the resulting shape either way, but only if there are no ops between those loops.

        // Add accumulator
        let num_axes = shape.len();
        let mut looped_axes: BTreeSet<usize> = (num_axes - axes.len()..num_axes).collect();
        //println!("Looped axes: {looped_axes:?}");
        let acc_id = self.ops.len()
            - self
                .ops
                .iter()
                .rev()
                .position(|op| {
                    if let Op::Loop { axis, .. } = op {
                        looped_axes.remove(axis);
                    }
                    looped_axes.is_empty()
                })
                .unwrap()
            - 1;
        //println!("Acc id: {acc_id}");
        self.max_id += 1;
        self.ops.insert(
            acc_id,
            Op::Accumulator { z: self.max_id, rop, view: View::none(), dtype },
        );
        self.ops.push(Op::Binary {
            z: self.max_id,
            x: xt,
            y: self.max_id,
            bop: match rop {
                ROp::Sum => BOp::Add,
                ROp::Max => BOp::Max,
            },
        });
        for _ in 0..axes.len() {
            self.ops.push(Op::EndLoop);
        }
        if !matches!(self.ops[0], Op::Loop { .. }) {
            self.insert_loop(0, 0);
        }
    }

    /// Inserts loop at `op_id`, giving it axis id and dimension 1.
    /// All loops and views axis equal or greater then axis are increased by 1
    /// Does not change reduce op's `num_axes`
    /// This function also does not change kernel's shape!
    pub(super) fn insert_loop(&mut self, op_id: usize, axis: Axis) {
        let naxis = axis;
        for op in &mut self.ops {
            match op {
                Op::Const { view, .. }
                | Op::Store { zview: view, .. }
                | Op::Load { xview: view, .. }
                | Op::Accumulator { view, .. } => view.insert_loop(naxis),
                Op::Loop { axis, .. } => {
                    if *axis >= naxis {
                        *axis += 1;
                    }
                }
                _ => {}
            }
        }
        self.ops.insert(op_id, Op::Loop { axis, len: 1 });
    }

    /// Store is the only function that evaluates kernels, it just checks if outputs
    /// are empty after store. Returns ids of evaluated tensors.
    pub(super) fn store(
        &mut self,
        nid: TensorId,
        graph: &Graph,
        devices: &mut [Box<dyn Device>],
        memory_pools: &mut [Box<dyn MemoryPool>],
        tensor_buffer_map: &mut BTreeMap<TensorId, BufferId>,
        optimizer: &mut Optimizer,
        search_iters: usize,
        debug: DebugMask,
    ) -> Result<Option<Vec<TensorId>>, ZyxError> {
        let zview = View::contiguous(graph.shape(nid));
        let zdtype = graph.dtype(nid);
        let z = self.outputs[&nid];
        if let Some(&Op::Store { z: nz, zview: ref nzview, .. }) = self.ops.last() {
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
        self.ops.push(store_op);
        self.tensors.insert(self.outputs.remove(&nid).unwrap(), nid);

        // Delete kernel and dispatch it to device
        if self.outputs.is_empty() {
            if debug.sched() {
                self.debug();
            }

            // Pick a device to run program
            // Find in which memory pool are most of input tensors stored
            let memory_pool_id = 0;
            let memory_pool = memory_pools[memory_pool_id as usize].as_mut();

            // Move all other tensors to that memory pool
            // and finish queues with this kernel's inputs

            // Get device which is associated with that memory pool
            let device = devices[0].as_mut();
            let mut sync = BTreeSet::new();

            // TODO deduplicate buffer ids, so that single tensor is not passed as multiple pointers
            let args: Vec<Id> = self
                .tensors
                .values()
                .map(|&tensor_id| {
                    if let Some(BufferId { buffer_id, .. }) = tensor_buffer_map.get(&tensor_id) {
                        *buffer_id
                    } else {
                        // Allocate bytes for outputs
                        let buffer_id = memory_pool
                            .allocate(
                                graph.shape(tensor_id).iter().product::<usize>()
                                    * graph.dtype(tensor_id).byte_size(),
                            )
                            .unwrap();
                        tensor_buffer_map.insert(tensor_id, BufferId { memory_pool_id, buffer_id });
                        sync.insert(buffer_id);
                        buffer_id
                    }
                })
                .collect();

            optimizer.launch(self, device, memory_pool, &args, sync, search_iters, debug)?;

            // add load kernels for all outputs of this kernel
            return Ok(Some(
                self.ops
                    .iter()
                    .filter_map(|op| {
                        if let Op::Store { z, .. } = op {
                            //tensor_buffer_map
                            //memory_pool.pool_to_host(buffer_id, data);
                            //self.tensors.get(z)
                            Some(*self.tensors.get(z).unwrap())
                        } else {
                            None
                        }
                    })
                    .collect(),
            ));
        } else {
            return Ok(None);
        }
    }
}

impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const C_BLUE: &str = "\x1B[34m";
        const C_GREEN: &str = "\x1B[32m";
        const C_MAGENTA: &str = "\x1B[35m";
        const C_RED: &str = "\x1B[31m";
        const C_WHITE: &str = "\x1B[37m";
        const C_YELLOW: &str = "\x1B[33m";
        const C_RESET: &str = "\x1B[39m";
        match self {
            Op::Const { z, value, view } => f.write_fmt(format_args!(
                "{C_WHITE}Const{C_RESET}       {z} <- value: {value}, {view}"
            )),
            Op::Load { z, zscope, zview: _, xscope, xview, xdtype } => f.write_fmt(format_args!(
                "{C_YELLOW}Load{C_RESET}        {z}[{zscope:?}] <- [{xscope:?}, {xdtype}], {xview}"
            )),
            Op::Store { z, zview, zscope, zdtype, xscope, xview: _ } => f.write_fmt(format_args!(
                "{C_RED}Store{C_RESET}        {z}[{zscope:?}] <- {xscope:?}, {zview}, {zdtype}"
            )),
            Op::Loop { axis, len: dimension } => f.write_fmt(format_args!(
                "{C_GREEN}Loop{C_RESET}        axis: {axis}, dimension: {dimension}"
            )),
            Op::Accumulator { z, rop, view, dtype } => f.write_fmt(format_args!(
                "{C_BLUE}Accum{C_RESET}.{rop:?}   {z}, shape: {:?}, {dtype}",
                view.shape()
            )),
            Op::EndLoop => f.write_fmt(format_args!("{C_BLUE}EndLoop{C_RESET} ")),
            Op::Move { z, x, mop } => {
                f.write_fmt(format_args!("{C_WHITE}Move{C_RESET}.{mop:?}   {z} <- {x}"))
            }
            Op::Unary { z, x, uop } => {
                let mut len = format!("{uop:?}").len();
                if len > 5 {
                    len = 5;
                }
                f.write_fmt(format_args!(
                    "{C_WHITE}Unary{C_RESET}.{uop:?}{} {z} <- {x}",
                    " ".repeat(5 - len)
                ))
            }
            Op::Binary { z, x, y, bop } => f.write_fmt(format_args!(
                "{C_WHITE}Binary{C_RESET}.{bop:?}  {z} <- {x}, {y}"
            )),
            Op::Barrier { scope } => {
                f.write_fmt(format_args!("{C_MAGENTA}Barrier{C_RESET}({scope})"))
            }
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

fn shape_to_loops(shape: &[usize]) -> Vec<Op> {
    let mut res = Vec::with_capacity(20);
    for (axis, dimension) in shape.iter().copied().enumerate() {
        res.push(Op::Loop { axis, len: dimension });
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
