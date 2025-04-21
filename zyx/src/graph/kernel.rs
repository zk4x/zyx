//! Kernel represents hardware programs.

// Posdsibly remove requirements for is_paddable
// is_reshapable more or less cannot be loosened.

use std::{
    cmp::Ordering, collections::{BTreeMap, BTreeSet}, hash::BuildHasherDefault, ops::Range
};
use crate::{
    dtype::Constant, graph::{Graph, Node}, runtime::Pool, shape::{Axis, Dim}, slab::{Slab, SlabId}, tensor::TensorId, DType, DebugMask, Map, Set
};

use super::{view::View, BOp, ROp, UOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId(u32);

impl SlabId for KernelId {
    const ZERO: Self = Self(0);

    fn index(self) -> usize {
        self.0 as usize
    }

    fn from_usize(id: usize) -> Self {
        Self(id as u32)
    }

    fn inc(&mut self) {
        self.0 += 1;
    }
}

// Tensor id in a kernel
pub type TId = u16;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Kernel {
    pub ops: Vec<Op>,
    // Mapind from tensors ids to load and store ids
    pub tensors: BTreeMap<TId, TensorId>,
    // Outputs of the kernel that are unused (not stored yet)
    pub outputs: BTreeMap<TensorId, TId>,
    pub max_id: TId,
    // Which kernels must be evaluated before this kernel (only direct predecessors)
    pub depends_on: BTreeSet<KernelId>,
}

// TODO this needs to be smaller, since it's stored on the disk
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Op {
    Loop {
        axis: Axis,
        len: Dim,
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
        x: TId,
        xview: View,
        xdtype: DType,
    },
    Store {
        z: TId,
        zview: View,
        zdtype: DType,
        x: TId,
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
    }
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
        shape: &[Dim],
        dtype: DType,
        depends_on: BTreeSet<KernelId>,
    ) -> Kernel {
        let mut ops = Vec::with_capacity(50);
        for (axis, dimension) in shape.iter().copied().enumerate() {
            ops.push(Op::Loop { axis, len: dimension });
        }
        ops.push(Op::Load {
            z: 0,
            x: 0,
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

    pub fn shape(&self) -> Vec<Dim> {
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
                    | Op::Binary { .. } => {}
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
                | Op::Binary { .. } => {}
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

    pub fn debug(&self) {
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

    pub fn flop_mem_rw(&self) -> (u128, u128, u128) {
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
                &Op::Load { ref xview, xdtype, .. } => {
                    // Note that this calculates actual read speed, even if the load accesses the same
                    // value multiple times. This is usefull so that we can see whether the kernel
                    // is compute bound or memory bound.
                    //mem_read += shape.iter().product::<usize>() as u128;
                    mem_read += xview.original_numel() as u128 * u128::from(xdtype.byte_size());
                }
                &Op::Store { ref zview, zdtype, .. } => {
                    //mem_write += shape.iter().product::<usize>() as u128 * zdtype.byte_size();
                    mem_write += zview.original_numel() as u128 * u128::from(zdtype.byte_size());
                }
                Op::EndLoop => {
                    shape.pop();
                }
                Op::Cast { .. } | Op::Unary { .. } | Op::Binary { .. } => {
                    flop += shape.iter().product::<Dim>() as u128;
                }
                Op::Accumulator { .. } | Op::Const { .. } => {}
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
            Op::Load { z, x, xview, xdtype: _ } => f.write_fmt(format_args!(
                "{C_MAGENTA}Load{C_RESET}        {z} <- {x}, {xview}"
            )),
            Op::Store { z, zview, zdtype: _, x } => {
                f.write_fmt(format_args!(
                "{C_RED}Store{C_RESET}        {z} <- {x}, {zview}"
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
pub fn kernelize(
    graph: &Graph,
    order: &[TensorId],
    // RCS are only ref counts from parameters, excluding ref counts from being in to_eval
    rcs: Map<TensorId, u32>,
    to_eval: &Set<TensorId>,
    memory_pools: &[Pool],
    realized_nodes: &Set<TensorId>,
    #[allow(unused)]
    debug: DebugMask,
) -> Slab<KernelId, Kernel> {
    // Unary and binary ops do not require duplication of kernels
    // Kernels represented by ops
    let mut kernels: Slab<KernelId, Kernel> = Slab::with_capacity(10);

    //if debug.sched() { println!("To schedule: {} tensors, to eval: {to_eval:?}", order.len()); }

    let mut rcs = if rcs.is_empty() {
        let mut rcs = Map::with_capacity_and_hasher(100, BuildHasherDefault::default());
        // to_eval are not in rcs
        for &nid in order {
            if !realized_nodes.contains(&nid) {
                for nid in graph[nid].parameters() {
                    rcs.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                }
            }
        }
        rcs
    } else {
        rcs
    };

    #[cfg(debug_assertions)]
    {
        let mut rcs2 = Map::with_hasher(BuildHasherDefault::default());
        // to_eval are not in rcs
        for &nid in order {
            if !realized_nodes.contains(&nid) {
                for nid in graph[nid].parameters() {
                    rcs2.entry(nid).and_modify(|rc| *rc += 1).or_insert(1);
                }
            }
        }
        if rcs2 != rcs {
            println!("Realized nodes: {realized_nodes:?}");
            for &nid in order {
                println!(
                    "ID({nid:?}): {:?}, sh: {:?}, rcs: {}, rcs actual: {}, num kernels: {}",
                    graph[nid],
                    graph.shape(nid),
                    rcs.get(&nid).copied().unwrap_or(0),
                    rcs2.get(&nid).copied().unwrap_or(0),
                    kernels.len(),
                );
            }
            panic!("rcs are incorrect, rcs: {rcs:?}\nrcs2: {rcs2:?}");
        }
    }

    for &nid in order {
        /*if debug.sched() {
            println!(
                "ID({nid}): {:?}, sh: {:?}, rcs: {}, num kernels: {}",
                graph[nid],
                graph.shape(nid),
                rcs.get(&nid).copied().unwrap_or(0),
                kernels.len(),
            );
        }*/

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
                        .flat_map(|pool| pool.buffer_map.keys())
                        .copied()
                        .collect();
                    if !realized_nodes.contains(&nid) {
                        println!("tensor {nid:?} not in realized nodes and not in buffers");
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
                            if kernels[kid].is_inlinable() {
                                let mut new_kernel = kernels[kid].clone();
                                if rcs[&x] < 2 {
                                    new_kernel.outputs.remove(&x);
                                }
                                kernels.push(new_kernel);
                            } else {
                                let x_dtype = graph.dtype(x);
                                store(&mut kernels, kid, x, x_shape, x_dtype);
                                if rcs[&x] < 2 {
                                    kernels[kid].outputs.remove(&x).unwrap();
                                }
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

                    if kernels[kid].is_paddable() {
                        if kernels[kid].outputs.len() > 1 || rcs[&x] > 1 {
                            if kernels[kid].is_inlinable() {
                                let mut new_kernel = kernels[kid].clone();
                                if rcs[&x] < 2 {
                                    new_kernel.outputs.remove(&x);
                                }
                                kernels.push(new_kernel);
                            } else {
                                let x_dtype = graph.dtype(x);
                                let x_shape = graph.shape(x);
                                store(&mut kernels, kid, x, x_shape, x_dtype);
                                if rcs[&x] < 2 {
                                    kernels[kid].outputs.remove(&x).unwrap();
                                }
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
                        if kernels[kid].is_inlinable() {
                            let mut new_kernel = kernels[kid].clone();
                            if rcs[&x] < 2 {
                                new_kernel.outputs.remove(&x);
                            }
                            kernels.push(new_kernel);
                        } else {
                            let x_shape = graph.shape(x);
                            let x_dtype = graph.dtype(x);
                            store(&mut kernels, kid, x, x_shape, x_dtype);
                            if rcs[&x] < 2 {
                                kernels[kid].outputs.remove(&x).unwrap();
                            }
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
                        if kernels[kid].is_inlinable() {
                            let mut new_kernel = kernels[kid].clone();
                            if rcs[&x] < 2 {
                                new_kernel.outputs.remove(&x);
                            }
                            kernels.push(new_kernel);
                        } else {
                            //println!("Kernel too big, creating depends on");
                            let x_shape = graph.shape(x);
                            let x_dtype = graph.dtype(x);
                            store(&mut kernels, kid, x, x_shape, x_dtype);
                            if rcs[&x] < 2 {
                                kernels[kid].outputs.remove(&x).unwrap();
                            }
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
                        //if kernels[kid].is_small() {
                        let mut new_kernel = kernels[kid].clone();
                        if rcs[&x] < 2 {
                            new_kernel.outputs.remove(&x);
                        }
                        kernels.push(new_kernel);
                        /*} else {
                            let x_shape = graph.shape(x);
                            let x_dtype = graph.dtype(x);
                            store(&mut kernels, kid, x, x_shape, x_dtype);
                            kernels[kid].outputs.remove(&x).unwrap();
                            let nkid = kernels.push(Kernel::leaf(
                                x,
                                x_shape,
                                x_dtype,
                                BTreeSet::from([kid]),
                            ));
                            xt = 0;
                            kid = nkid;
                        }*/
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
                Node::Cast { x, dtype } => {
                    //let _timer = Timer::new("unary");
                    let (xt, kid) = get_kernel_max(x, &kernels);
                    debug_assert_eq!(kernels[kid].shape(), graph.shape(x));
                    kernels[kid].max_id += 1;
                    let z = kernels[kid].max_id;
                    kernels[kid].ops.push(Op::Cast { z, x: xt, dtype });
                    if rcs[&x] < 2 {
                        kernels[kid].outputs.remove(&x);
                    }
                    kernels[kid].outputs.insert(nid, z);
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
                    // kidx goes first, remove kidy and add it to kidx
                    let (mut xt, mut kidx) = get_kernel_max(x, &kernels);
                    let (mut yt, mut kidy) = get_kernel_max(y, &kernels);

                    //kernels[kidx].debug();
                    //kernels[kidy].debug();

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
                        // can't remove kidy if any kernel depends on kidy
                        // store y and all outputs that are not stored yet in kidy
                        // clear kidy's outputs
                        // write a load kernel for y
                        // mark kidy as new kernel
                        if kernels[kidy].has_stores() && depends_on(&kernels, kidx, kidy) {
                            let old_kidy = kidy;
                            let outputs: Vec<TensorId> =
                                kernels[kidy].outputs.keys().copied().collect();
                            // Stores all outputs that are not stored yet
                            for nid in outputs {
                                debug_assert!(rcs[&nid] > 0);
                                let nid_shape = graph.shape(nid);
                                let nid_dtype = graph.dtype(nid);
                                store(&mut kernels, old_kidy, nid, nid_shape, nid_dtype);
                                let nk = Kernel::leaf(nid, nid_shape, nid_dtype, [old_kidy].into());
                                let nkid = kernels.push(nk);
                                if nid == y {
                                    kidy = nkid;
                                }
                            }
                            yt = 0;
                            // old_kidy is done, can be immediatelly realized
                            kernels[old_kidy].outputs.clear();
                            debug_assert_ne!(kidy, old_kidy);
                        }

                        if kernels[kidx].has_stores() && depends_on(&kernels, kidy, kidx) {
                            let old_kidx = kidx;
                            let outputs: Vec<TensorId> =
                                kernels[kidx].outputs.keys().copied().collect();
                            // Stores all outputs that are not stored yet
                            for nid in outputs {
                                debug_assert!(rcs[&nid] > 0);
                                let nid_shape = graph.shape(nid);
                                let nid_dtype = graph.dtype(nid);
                                store(&mut kernels, old_kidx, nid, nid_shape, nid_dtype);
                                let nk = Kernel::leaf(nid, nid_shape, nid_dtype, [old_kidx].into());
                                let nkid = kernels.push(nk);
                                if nid == x {
                                    kidx = nkid;
                                }
                            }
                            xt = 0;
                            // old_kidx is done, can be immediatelly realized
                            kernels[old_kidx].outputs.clear();
                            debug_assert_ne!(kidx, old_kidx);
                        }

                        let Kernel { ops, tensors, outputs, max_id, depends_on } =
                            unsafe { kernels.remove_and_return(kidy) };

                        // we delete kidy (could by also kidx) and put everything in kidx
                        let n = kernels[kidx].max_id + 1;

                        let mut i = 0;
                        while matches!(ops[i], Op::Loop { .. })
                            && ops[i] == kernels[kidx].ops[i]
                        {
                            i += 1;
                        }
                        for op in ops.into_iter().skip(i) {
                            let new_op = match op {
                                Op::Loop { axis, len } => Op::Loop { axis, len },
                                Op::EndLoop => Op::EndLoop,
                                Op::Const { z, value, ref view } => {
                                    Op::Const { z: z + n, value, view: view.clone() }
                                }
                                Op::Load { z, x, ref xview, xdtype } => {
                                    Op::Load {
                                        z: z + n,
                                        x: x + n,
                                        xview: xview.clone(),
                                        xdtype,
                                    }
                                }
                                Op::Store {
                                    z,
                                    ref zview,
                                    zdtype,
                                    x,
                                } => Op::Store {
                                    z: z + n,
                                    zview: zview.clone(),
                                    zdtype,
                                    x: x + n,
                                },
                                Op::Accumulator { z, rop, dtype } => {
                                    Op::Accumulator { z: z + n, rop, dtype }
                                }
                                Op::Cast { z, x, dtype } => Op::Cast { z: z + n, x: x + n, dtype },
                                Op::Unary { z, x, uop } => Op::Unary { z: z + n, x: x + n, uop },
                                Op::Binary { z, x, y, bop } => {
                                    Op::Binary { z: z + n, x: x + n, y: y + n, bop }
                                }
                            };
                            kernels[kidx].ops.push(new_op);
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

                    if x != y {
                        if rcs[&x] < 2 {
                            kernels[kidx].outputs.remove(&x);
                        }
                        if rcs[&y] < 2 {
                            kernels[kidx].outputs.remove(&y);
                        }
                    } else if rcs[&x] < 3 {
                        kernels[kidx].outputs.remove(&x);
                    }

                    kid
                }
            }
        };

        //for kernel in kernels.values() { kernel.debug(); println!(); }

        for param in graph[nid].parameters() {
            if let Some(rc) = rcs.get_mut(&param) {
                //println!("Decrementing rc of {param} from {rc}");
                *rc -= 1;
                if *rc == 0 {
                    //println!("Removing rc of {param}");
                    rcs.remove(&param);
                }
            }
        }

        debug_assert_eq!(kernels[kid].shape(), graph.shape(nid));

        #[cfg(debug_assertions)]
        {
            if kernels[kid].ops.iter().filter(|op| matches!(op, Op::Loop { .. })).count()
                <= kernels[kid].ops.iter().filter(|op| matches!(op, Op::EndLoop)).count()
            {
                kernels[kid].debug();
                panic!();
            }

            for kernel in kernels.values() {
                if kernel.ops.iter().filter(|op| matches!(op, Op::Loop { .. })).count()
                    <= kernel.ops.iter().filter(|op| matches!(op, Op::EndLoop)).count()
                {
                    kernel.debug();
                    panic!();
                }
            }
        }

        if to_eval.contains(&nid) {
            let nid_shape = graph.shape(nid);
            let nid_dtype = graph.dtype(nid);
            if let Some(&x) = kernels[kid].tensors.get(&0) {
                if x != nid {
                    store(&mut kernels, kid, nid, nid_shape, nid_dtype);
                }
            } else {
                store(&mut kernels, kid, nid, nid_shape, nid_dtype);
            }
        }
    }
    kernels
}

// Adds store to kernel and removes nid from outputs of all kernels
fn store(
    kernels: &mut Slab<KernelId, Kernel>,
    kid: KernelId,
    nid: TensorId,
    nid_shape: &[usize],
    nid_dtype: DType,
) {
    //let _timer = Timer::new("scheduler store");
    if kernels[kid].tensors.values().any(|id| *id == nid) {
        // This check can fail in binary op depends_on stores
        /*#[cfg(debug_assertions)]
        {
            for op in kernels[kid].ops.iter() {
                if let Op::Load { x, .. } = op {
                    if kernels[kid].tensors[x] == nid {
                        kernels[kid].debug();
                        panic!("Trying to store tensor that is used as a load in a kernel.");
                    }
                }
            }
        }*/
        return;
    }

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
        zdtype: nid_dtype,
        x,
    };
    kernels[kid].ops.push(store_op);
    kernels[kid].tensors.insert(z, nid);
}

// recursive should be faster, since it does not allocate, but in fact the dynamic programming
// version is much faster, apparently cpus really hate recursion
/// Check if kidx depends on kidy
fn depends_on(kernels: &Slab<KernelId, Kernel>, kidx: KernelId, kidy: KernelId) -> bool {
    let mut depends_on = kernels[kidx].depends_on.clone();
    while let Some(kid) = depends_on.pop_last() {
        if kid == kidy {
            return true;
        }
        depends_on.extend(kernels[kid].depends_on.iter().copied());
    }
    false
}

// Choose kernel with most outputs (binary, unary)
fn get_kernel_max(x: TensorId, kernels: &Slab<KernelId, Kernel>) -> (TId, KernelId) {
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
fn get_kernel_min(x: TensorId, kernels: &Slab<KernelId, Kernel>) -> (TId, KernelId) {
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
