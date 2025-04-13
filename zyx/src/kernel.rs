//! Kernel represents hardware programs.

// Posdsibly remove requirements for is_paddable
// is_reshapable more or less cannot be loosened.

use std::{
    cmp::Ordering, collections::{BTreeMap, BTreeSet}, ops::Range
};

use crate::{
    backend::{Device, Event},
    dtype::Constant,
    graph::Graph,
    ir::Scope,
    kernel_cache::KernelCache,
    node::{BOp, ROp, UOp},
    runtime::Pool,
    shape::{Axis, Dimension},
    slab::Id,
    tensor::TensorId,
    view::View,
    DType, DebugMask, Map, ZyxError,
};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Kernel {
    pub(super) ops: Vec<Op>,
    // Mapind from tensors ids to load and store ids
    pub(super) tensors: BTreeMap<TId, TensorId>,
    // Outputs of the kernel that are unused (not stored yet)
    pub(super) outputs: BTreeMap<TensorId, TId>,
    pub(super) max_id: TId,
    // Which kernels must be evaluated before this kernel (only direct predecessors)
    pub(super) depends_on: BTreeSet<u32>,
}

// Tensor id in a kernel
pub type TId = u16;

// TODO this needs to be smaller, since it's stored on the disk
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Op {
    Loop {
        axis: Axis,
        len: Dimension,
    },
    // End the latest loop
    EndLoop,
    Const {
        z: TId,
        value: Constant,
        view: View,
    },
    Load {
        z: TId,
        zscope: Scope,
        zview: View,
        x: TId,
        xscope: Scope,
        xview: View,
        xdtype: DType,
    },
    Store {
        z: TId,
        zscope: Scope,
        zview: View,
        zdtype: DType,
        x: TId,
        xscope: Scope,
        xview: View,
    },
    Accumulator {
        z: TId,
        rop: ROp,
        dtype: DType,
    },
    // Move is noop, just a marker for easy debugging
    /*Move {
        z: TId,
        x: TId,
        mop: MOp,
    },*/
    Cast {
        z: TId,
        x: TId,
        dtype: DType,
    },
    Unary {
        z: TId,
        x: TId,
        uop: UOp,
    },
    Binary {
        z: TId,
        x: TId,
        y: TId,
        bop: BOp,
    },
    // Synchronization for local and global memory
    #[allow(unused)]
    Barrier {
        scope: Scope,
    },
}

/*#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MOp {
    Expa,
    Perm,
    Resh,
    Padd,
}*/

impl Kernel {
    pub(super) fn constant(nid: TensorId, value: Constant) -> Kernel {
        let mut ops = Vec::with_capacity(50);
        ops.push(Op::Loop { axis: 0, len: 1 });
        ops.push(Op::Const { z: 0, value, view: View::contiguous(&[1]) });
        Kernel {
            max_id: 0,
            ops,
            tensors: BTreeMap::new(),
            outputs: BTreeMap::from([(nid, 0)]),
            depends_on: BTreeSet::new(),
        }
    }

    pub(super) fn leaf(
        nid: TensorId,
        shape: &[Dimension],
        dtype: DType,
        depends_on: BTreeSet<Id>,
    ) -> Kernel {
        let mut ops = Vec::with_capacity(50);
        for (axis, dimension) in shape.iter().copied().enumerate() {
            ops.push(Op::Loop { axis, len: dimension });
        }
        ops.push(Op::Load {
            z: 0,
            zscope: Scope::Register,
            zview: View::none(),
            x: 0,
            xscope: Scope::Global,
            xview: View::contiguous(shape),
            xdtype: dtype,
        });
        Kernel {
            max_id: 0,
            ops,
            outputs: BTreeMap::from([(nid, 0)]),
            tensors: BTreeMap::from([(0, nid)]),
            depends_on,
        }
    }

    /*pub(super) fn get_tensor_id(&self, tid: TId) -> TensorId {
        *self.tensors.iter().find(|(tidx, _)| **tidx == tid).unwrap().0
    }*/

    pub(super) fn shape(&self) -> Vec<Dimension> {
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

    // Returns true if it is cheaper to evaluate this kernel twice as inlined into bigger kernel
    // instead of launching this kernel. separately
    pub(super) fn is_inlinable(&self) -> bool {
        if self.ops.len() > 20 {
            return false;
        }
        for i in 0..self.ops.len() {
            if matches!(self.ops[i], Op::Accumulator { .. }) {
                if let Op::Loop { len, .. } = self.ops[i + 1] {
                    if len > 32 {
                        return false;
                    }
                }
            }
        }
        true
    }

    pub(super) fn has_stores(&self) -> bool {
        self.ops.iter().any(|op| matches!(op, Op::Store { .. }))
    }

    /*#[cfg(debug_assertions)]
    pub(super) fn is_reshapable(&self, shape: &[usize]) -> bool {
        // TODO remove the first case
        self.ops.iter().all(|op| match op {
            Op::Loop { .. } | Op::Unary { .. } | Op::Binary { .. } | Op::Barrier { .. } => true,
            //| Op::Move { .. } => true,
            Op::Load { xview: view, .. }
            | Op::Store { zview: view, .. }
            | Op::Const { view, .. } => view.is_contiguous(),
            Op::Accumulator { .. } | Op::EndLoop => false,
        }) | self.get_reshape_pattern(shape).is_some()
    }*/

    pub(super) fn reshape(&mut self, shape: &[usize]) {
        debug_assert_eq!(
            self.shape().iter().product::<usize>(),
            shape.iter().product(),
            "Cannot reshape kernel from {:?} to {:?}",
            self.shape(),
            shape,
        );
        // If this is just a reshape of kernel with only unary ops and contiguous loads
        // and stores, we can remove old loops and replace them with new loops.
        //println!("Reshape");
        if let Some((new_loops, reshapes)) = self.get_reshape_pattern(shape) {
            let _ = new_loops; // TODO get new_loops working
                               //println!("Reshapes: {reshapes:?}");
            for (org_sh, sh) in reshapes.iter().rev() {
                let mut op_i = self.ops.len();
                'a: loop {
                    op_i -= 1;
                    if let Op::Loop { axis, .. } = &mut self.ops[op_i] {
                        //println!("{org_sh:?} -> {sh:?}");
                        match (*axis).cmp(&(org_sh.end - 1)) {
                            Ordering::Less => {}
                            Ordering::Equal => {
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
                            Ordering::Greater => {
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
                    | Op::Store { zview: view, .. } => {
                        for (org_sh, sh) in reshapes.iter().rev() {
                            view.reshape(org_sh.clone(), &shape[sh.clone()]);
                        }
                    }
                    Op::Accumulator { .. }
                    | Op::Loop { .. }
                    | Op::EndLoop
                    | Op::Cast { .. }
                    | Op::Unary { .. }
                    | Op::Binary { .. }
                    | Op::Barrier { .. } => {}
                }
            }
            //self.debug();
            // TODO deal with loop inserts
            debug_assert_eq!(self.shape(), shape, "Shape after reshape is incorrect.");
        } else {
            unreachable!()
        }
    }

    // TODO we can speed this up in scheduler, since it is always reshapable without unmergeable axes.
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

    pub(super) fn reshape_unchecked(&mut self, nshape: &[usize]) {
        let shape: &[usize] = &self.shape();
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
                                //println!("Axis {a}, nshape {nshape:?}");
                                self.ops.insert(
                                    i,
                                    Op::Loop { axis: a + org_sh.start - sh.start, len: nshape[a] },
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
                | Op::Store { zview: view, .. } => {
                    for (org_sh, sh) in reshapes.iter().rev() {
                        view.reshape(org_sh.clone(), &nshape[sh.clone()]);
                    }
                }
                Op::Accumulator { .. }
                | Op::Loop { .. }
                | Op::EndLoop
                | Op::Cast { .. }
                | Op::Unary { .. }
                | Op::Binary { .. }
                | Op::Barrier { .. } => {}
            }
        }
        //self.debug();
        // TODO deal with loop inserts
        debug_assert_eq!(
            self.shape(),
            nshape,
            "Shape after reshape split is incorrect."
        );
    }

    #[allow(unused)]
    pub(super) fn split_loop(&mut self, op_id: usize, dimensions: &[usize]) {
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
                | Op::Const { view, .. } => {
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
            "Kernel shape: {:?}, outputs: {:?}, tensors: {:?}, depends on: {:?}",
            self.shape(),
            self.outputs,
            self.tensors,
            self.depends_on,
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

    /// Kernel is expandable if it does not contain store and if the result is not too
    /// large when applied on reduce kernel.
    #[allow(clippy::doc_markdown)]
    pub(super) fn is_expandable(&self, shape: &[usize]) -> bool {
        /*!self.ops.iter().any(|op| {
            matches!(op, Op::Store { .. })
                || (matches!(op, Op::Accumulator { .. })
                    && shape.iter().product::<usize>() > 1024 * 1024 * 1024)
        })*/

        //!self.ops.iter().any(|op| matches!(op, Op::Store { .. } | Op::Accumulator { .. }))
        // Small loops with like 32 iterations can be run in big reduce loop and they can also be unrolled.
        !self.ops.iter().any(|op| {
            let is_store = matches!(op, Op::Store { .. });
            let is_reduce = matches!(op, Op::Accumulator { .. });
            let is_large_ws = shape.iter().product::<usize>() > 1024 * 1024 * 1024;
            let is_large_inner_loop = self
                .ops
                .iter()
                .skip_while(|op| matches!(op, Op::Loop { .. }))
                .filter_map(|op| {
                    if let Op::Loop { len, .. } = op {
                        Some(len)
                    } else {
                        None
                    }
                })
                .product::<usize>()
                > 32;
            is_store || is_large_ws || (is_reduce && is_large_inner_loop)
        })
    }

    pub(super) fn expand(&mut self, shape: &[usize]) {
        //println!("Expanding");
        //kernel.debug();
        debug_assert_eq!(shape.len(), self.shape().len());
        let mut expand_axes = BTreeSet::new();
        for (a, d) in self.shape().into_iter().enumerate() {
            if d != shape[a] {
                debug_assert_eq!(d, 1);
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
                        debug_assert_eq!(*dimension, 1);
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
            // Padding stores can be later removed, but it's a trade-off,
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
                | Op::Store { zview: view, .. } => {
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
        nid: TensorId,
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
        self.ops.insert(acc_id, Op::Accumulator { z: self.max_id, rop, dtype });
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
        self.outputs.clear();
        self.outputs.insert(nid, self.max_id);
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
                | Op::Load { xview: view, .. } => view.insert_loop(naxis),
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

    pub(super) fn launch(
        &mut self,
        graph: &Graph,
        devices: &mut [Device],
        memory_pools: &mut [Pool],
        kernel_cache: &mut KernelCache,
        search_iters: usize,
        debug: DebugMask,
    ) -> Result<Option<Event>, ZyxError> {
        // First make a list of all memory pools which can hold all tensors, including outputs
        let total_bytes: Dimension = self
            .tensors
            .values()
            .map(|&tid| {
                graph.shape(tid).iter().product::<Dimension>()
                    * graph.dtype(tid).byte_size() as Dimension
            })
            .sum();
        let mut free_memory_pools = Map::default();
        for (mpid, pool) in memory_pools.iter().enumerate() {
            // TODO subtract tensors stored in that pool from total_bytes
            if pool.pool.free_bytes() > total_bytes {
                free_memory_pools.insert(mpid, (0, 0));
            }
        }

        // Pick memory pool associated with fastest/least occupied device.
        for (device_id, device) in devices.iter().enumerate() {
            let mpid = device.memory_pool_id();
            if let Some((compute, old_device_id)) = free_memory_pools.get_mut(&(mpid as usize)) {
                let free_compute = device.free_compute();
                if free_compute > *compute {
                    *compute = free_compute;
                    *old_device_id = device_id;
                }
            }
        }
        let mpid = free_memory_pools.iter().max_by_key(|x| x.1.0);
        //println!("{free_memory_pools:?} mpid {mpid:?}");
        let (mpid, dev_id) = if let Some(mpid) = mpid {
            (*mpid.0, mpid.1.1)
        } else {
            return Err(ZyxError::AllocationError);
        };

        // Move all tensors to that pool if they are not there already.
        // Allocate space for all outputs.
        let mut args = Vec::new();
        let mut outputs = BTreeSet::new();
        let mut event_wait_list = Vec::new();

        for op in &self.ops {
            match op {
                Op::Load { x, .. } => {
                    let tid = self.tensors[x];
                    if !memory_pools[mpid].buffer_map.contains_key(&tid) {
                        // Check where the tensor is
                        let mut old_mpid = usize::MAX;
                        for (i, pool) in memory_pools.iter().enumerate() {
                            if pool.buffer_map.contains_key(&tid) {
                                old_mpid = i;
                                break;
                            }
                        }
                        debug_assert_ne!(old_mpid, usize::MAX);

                        let bytes = graph.shape(tid).iter().product::<Dimension>() * graph.dtype(tid).byte_size() as Dimension;
                        // No need to initialize here, other than rust is bad.
                        let mut byte_slice = vec![0u8; bytes];
                        let src = memory_pools[old_mpid].buffer_map[&tid];

                        // Move the tensor into mpid pool
                        // Pool to host blocks on event, so we can remove that event.
                        let mut event_wait_list = Vec::new();
                        for buffers in memory_pools[old_mpid].events.keys() {
                            if buffers.contains(&src) {
                                // Pool to host blocks on event, so we can remove that event.
                                let event =
                                    memory_pools[old_mpid].events.remove(&buffers.clone()).unwrap();
                                event_wait_list.push(event);
                                break;
                            }
                        }
                        memory_pools[old_mpid].pool.pool_to_host(
                            src,
                            &mut byte_slice,
                            event_wait_list,
                        )?;
                        memory_pools[old_mpid].pool.deallocate(src, vec![])?;
                        memory_pools[old_mpid].buffer_map.remove(&tid);
                        //println!("{byte_slice:?}");

                        let (dst, event) = memory_pools[mpid].pool.allocate(bytes)?;
                        let event = memory_pools[mpid].pool.host_to_pool(
                            &byte_slice,
                            dst,
                            vec![event],
                        )?;
                        // We have to sync here, because byte_slice does not exist any long.
                        // The other solution would be to put this into temp_data.
                        // But perhaps we should figure some better async.
                        memory_pools[mpid].pool.sync_events(vec![event])?;
                        memory_pools[mpid].buffer_map.insert(tid, dst);
                        //memory_pools[mpid].events.insert(BTreeSet::from([dst]), event);
                    }
                    args.push(memory_pools[mpid].buffer_map[&tid]);
                }
                Op::Store { z, zview, zdtype, .. } => {
                    // Allocate space for output
                    let tensor_id = self.tensors[z];
                    let (buffer_id, event) = memory_pools[mpid]
                        .pool
                        .allocate(zview.original_numel() * (zdtype.byte_size() as Dimension))?;
                    memory_pools[mpid].buffer_map.insert(tensor_id, buffer_id);
                    event_wait_list.push(event);
                    outputs.insert(tensor_id);
                    args.push(buffer_id);
                }
                _ => {}
            }
        }

        /*#[cfg(debug_assertions)]
        {
            let mut visited = BTreeSet::new();
            for &arg in &args {
                debug_assert!(visited.insert(arg));
            }
        }*/
        //println!("args = {args:?}");

        // Send the kernel to kernel cache.
        if let Some(event) = kernel_cache.launch(
            self,
            dev_id as u32,
            &mut devices[dev_id],
            &mut memory_pools[mpid],
            &args,
            event_wait_list,
            search_iters,
            debug,
        )? {
            memory_pools[mpid].events.insert(outputs, event.clone());
            return Ok(Some(event));
        }
        Ok(None)
    }

    /// Store is the only function that evaluates kernels, it just checks if outputs
    /// are empty after store. Returns ids of evaluated tensors.
    /*pub(super) fn launch(
        &mut self,
        graph: &Graph,
        devices: &mut [Device],
        memory_pools: &mut [Pool],
        optimizer: &mut KernelCache,
        search_iters: usize,
        debug: DebugMask,
    ) -> Result<Option<Event>, ZyxError> {
        // Pick a device to run program
        // Find in which memory pool are most of input tensors stored
        let tensors: BTreeSet<TensorId> = self.tensors.values().copied().collect();
        // memory pool id => set(buffer_id), total_memory_used
        let mut used_pools: BTreeMap<usize, (BTreeSet<u32>, usize)> = BTreeMap::new();
        for (memory_pool_id, pool) in memory_pools.iter().enumerate() {
            for &tensor_id in &tensors {
                if pool.buffer_map.contains_key(&tensor_id) {
                    used_pools
                        .entry(memory_pool_id)
                        .and_modify(|(buffers, memory_size)| {
                            buffers.insert(tensor_id);
                            *memory_size += graph.shape(tensor_id).iter().product::<usize>();
                        })
                        .or_insert_with(|| {
                            (
                                BTreeSet::from([tensor_id]),
                                graph.shape(tensor_id).iter().product::<usize>(),
                            )
                        });
                }
            }
        }
        let mut memory_pool_id = used_pools.iter().max_by_key(|x| x.1 .1).map_or(1, |x| *x.0);

        // If this is disk pool, then we need to load it
        if memory_pools[memory_pool_id].pool.disk_pool().is_some() {
            // TODO select fastest available device
            memory_pool_id = 1;
        }

        // Move all other tensors to this memory pool
        // and finish events with this kernel's inputs
        for (pool_id, (tensors, _)) in used_pools {
            if pool_id != memory_pool_id {
                println!("pool id != memory pool id, tensors: {tensors:?}");
                for tensor_id in tensors {
                    let bytes = graph.shape(tensor_id).iter().product::<Dimension>()
                        * graph.dtype(tensor_id).byte_size() as Dimension;

                    // No need to initialize here, other than rust is bad.
                    let mut byte_slice = vec![0; bytes];

                    let src = memory_pools[pool_id].buffer_map[&tensor_id];
                    for buffers in memory_pools[pool_id].events.keys() {
                        if buffers.contains(&memory_pools[pool_id].buffer_map[&tensor_id]) {
                            // Pool to host blocks on event, so we can remove that event.
                            let event =
                                memory_pools[pool_id].events.remove(&buffers.clone()).unwrap();
                            memory_pools[pool_id].pool.pool_to_host(
                                src,
                                &mut byte_slice,
                                vec![event],
                            )?;
                            memory_pools[pool_id].pool.deallocate(src, vec![])?;
                            memory_pools[memory_pool_id].buffer_map.remove(&tensor_id);
                            break;
                        }
                    }

                    let (dst, event) = memory_pools[memory_pool_id].pool.allocate(bytes)?;

                    let event = memory_pools[memory_pool_id].pool.host_to_pool(
                        &byte_slice,
                        dst,
                        vec![event],
                    )?;
                    memory_pools[memory_pool_id].buffer_map.insert(tensor_id, dst);
                    memory_pools[memory_pool_id].events.insert(BTreeSet::from([dst]), event);
                }
            }
        }

        let pool = &mut memory_pools[memory_pool_id];

        // Get fastest device which is associated with that memory pool
        let device = devices
            .iter_mut()
            .filter(|device| device.memory_pool_id() == u32::try_from(memory_pool_id).unwrap())
            .max_by_key(|device| device.free_compute())
            .unwrap();

        let mut outputs = BTreeSet::new();
        let mut event_wait_list: Vec<Event> = Vec::new();
        let mut visited_tensors = BTreeMap::new();

        //println!("Pool contains {:?}", pool.buffer_map.keys());
        //self.debug();

        let mut args: Vec<Id> = Vec::new();
        for op in &mut self.ops {
            match op {
                Op::Load { x, .. } => {
                    let tensor_id = self.tensors[x];
                    let buffer_id = pool.buffer_map[&tensor_id];
                    if let Some(&tx) = visited_tensors.get(&buffer_id) {
                        *x = tx;
                    } else {
                        visited_tensors.insert(buffer_id, *x);
                        if let Some((_, event)) =
                            pool.events.iter().find(|(key, _)| key.contains(&buffer_id))
                        {
                            event_wait_list.push(event.clone());
                        }
                        args.push(buffer_id);
                    }
                }
                Op::Store { z, zview, zdtype, .. } => {
                    let tensor_id = self.tensors[z];
                    //println!("Allocating {zview} {}", zview.original_numel());
                    let (buffer_id, event) = pool
                        .pool
                        .allocate(zview.original_numel() * (zdtype.byte_size() as Dimension))?;
                    debug_assert!(visited_tensors.insert(buffer_id, *z).is_none());
                    pool.buffer_map.insert(tensor_id, buffer_id);
                    event_wait_list.push(event);
                    outputs.insert(tensor_id);
                    args.push(buffer_id);
                }
                _ => (),
            }
        }

        //println!("\nArgs: {args:?}");
        //self.debug();
        #[cfg(debug_assertions)]
        {
            let mut visited = BTreeSet::new();
            for &arg in &args {
                debug_assert!(visited.insert(arg));
            }
        }

        optimizer.launch(
            self,
            device,
            pool,
            &args,
            outputs.clone(),
            event_wait_list,
            search_iters,
            debug,
        )?;
        if let Some(event) = pool.events.get(&outputs) {
            Ok(Some(event.clone()))
        } else {
            Ok(None)
        }
    }*/

    pub(super) fn flop_mem_rw(&self) -> (u128, u128, u128) {
        // TODO This does not yet account for multiple loads from the same buffer.
        let mut shape = Vec::new();
        let mut flop = 0;
        let mut mem_read = 0;
        let mut mem_write = 0;
        for op in &self.ops {
            match op {
                &Op::Loop { len, .. } => {
                    shape.push(len);
                }
                &Op::Load { xscope, ref xview, xdtype, .. } => {
                    // Note that this calculates actual read speed, even if the load accesses the same
                    // value multiple times. This is usefull so that we can see whether the kernel
                    // is compute bound or memory bound.
                    if xscope == Scope::Global {
                        //mem_read += shape.iter().product::<usize>() as u128;
                        mem_read += xview.original_numel() as u128 * xdtype.byte_size() as u128;
                    }
                }
                &Op::Store { zscope, ref zview, zdtype, .. } => {
                    if zscope == Scope::Global {
                        //mem_write += shape.iter().product::<usize>() as u128 * zdtype.byte_size();
                        mem_write += zview.original_numel() as u128 * zdtype.byte_size() as u128;
                    }
                }
                Op::EndLoop => {
                    shape.pop();
                }
                Op::Cast { .. } | Op::Unary { .. } | Op::Binary { .. } => {
                    flop += shape.iter().product::<usize>() as u128;
                }
                Op::Accumulator { .. } | Op::Const { .. } | Op::Barrier { .. } => {}
            }
        }
        (flop, mem_read, mem_write)
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
            Op::Load { z, zscope, zview: _, x, xscope, xview, xdtype } => f.write_fmt(format_args!(
                "{C_MAGENTA}Load{C_RESET}        {z}[{zscope:?}] <- {x}[{xscope:?}, {xdtype}], {xview}"
            )),
            Op::Store { z, zview, zscope, zdtype, x, xscope, xview: _ } => {
                f.write_fmt(format_args!(
                "{C_RED}Store{C_RESET}        {z}[{zscope:?}] <- {x}[{xscope:?}], {zview}, {zdtype}"
            ))
            }
            Op::Loop { axis, len } => f.write_fmt(format_args!(
                "{C_GREEN}Loop{C_RESET}        axis: {axis}, len: {len}"
            )),
            Op::Accumulator { z, rop, dtype } => f.write_fmt(format_args!(
                "{C_BLUE}Accum{C_RESET}.{rop:?}   {z}, {dtype}",
            )),
            Op::EndLoop => f.write_fmt(format_args!("{C_BLUE}EndLoop{C_RESET} ")),
            Op::Cast { z, x, dtype } => {
                let mut len = format!("C-{dtype}").len();
                if len > 5 {
                    len = 5;
                }
                f.write_fmt(format_args!(
                    "{C_WHITE}Unary{C_RESET}.C-{dtype}{} {z} <- {x}",
                    " ".repeat(5 - len)
                ))
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
                f.write_fmt(format_args!("{C_YELLOW}Barrier{C_RESET}({scope})"))
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

/*fn shape_to_loops(shape: &[usize]) -> Vec<Op> {
    let mut res = Vec::with_capacity(20);
    for (axis, dimension) in shape.iter().copied().enumerate() {
        res.push(Op::Loop { axis, len: dimension });
    }
    res
}*/

#[test]
fn reshape_pattern() {
    let shape = [2, 4, 1, 3, 1, 4, 5, 2];
    let nshape = [8, 3, 1, 2, 2, 2, 5];
    let r = get_reshape_pattern(&shape, &nshape, &[]);
    debug_assert_eq!(
        r,
        Some((
            0,
            vec![(0..2, 0..1), (2..4, 1..2), (5..6, 3..5), (6..8, 5..7)]
        ))
    );
    let shape = [2, 2, 1, 2, 2];
    let nshape = [2, 2, 1, 2, 2, 1];
    let r = get_reshape_pattern(&shape, &nshape, &[]);
    debug_assert_eq!(r, Some((0, vec![(4..5, 4..6)])));
    let shape = [1, 3, 4, 5];
    let nshape = [3, 20];
    let r = get_reshape_pattern(&shape, &nshape, &[]);
    debug_assert_eq!(r, Some((0, vec![(0..2, 0..1), (2..4, 1..2)])));
}
