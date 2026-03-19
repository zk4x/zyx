use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernelize::KMKernelId,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
    view::View,
};
use nanoserde::{DeBin, SerBin};
use std::{fmt::Display, hash::Hash};

mod const_folding;
mod debug;
mod jam_loops;
mod licm;
mod mma;
mod split_loops;
mod unfold;
mod unroll_loops;
mod vectorize;

// TODO later make this dynamic u32 or u64 depending on max range
pub const IDX_T: DType = DType::U32;

#[derive(Debug, Clone)]
pub struct Kernel {
    pub outputs: Vec<TensorId>,
    pub loads: Vec<TensorId>,
    pub stores: Vec<TensorId>,
    pub ops: Slab<OpId, OpNode>,
    pub head: OpId,
    pub tail: OpId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum Scope {
    Global,
    Local,
    Register,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, SerBin, DeBin)]
pub enum UOp {
    Neg,
    BitNot,
    Exp2,
    Log2,
    Reciprocal,
    Sqrt,
    Sin,
    Cos,
    Floor,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash, SerBin, DeBin)]
pub enum BOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    Cmplt,
    Cmpgt,
    Max,
    Or,
    And,
    BitXor,
    BitOr,
    BitAnd,
    BitShiftLeft,
    BitShiftRight,
    NotEq,
    Eq,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MoveOp {
    Reshape { shape: Vec<Dim> },
    Expand { shape: Vec<Dim> },
    Permute { axes: Vec<UAxis>, shape: Vec<Dim> },
    Pad { padding: Vec<(i32, i32)>, shape: Vec<Dim> },
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMADims {
    m8n8k16,
    m16n8k8,
    m16n8k16,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMALayout {
    row_row,
    row_col,
    col_row,
    col_col,
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMADType {
    f16_f16_f16_f32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct OpNode {
    pub prev: OpId,
    pub next: OpId, // Use Vec<OpId> instead for egraph
    pub op: Op,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct OpId(pub u32);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum Op {
    // ops that exist in both
    Cast { x: OpId, dtype: DType },
    Unary { x: OpId, uop: UOp },
    // For binary ops, next of x is y, then next of y is the binary op
    Binary { x: OpId, y: OpId, bop: BOp },

    // ops that only exist after unfolding views and reduces
    Const(Constant),
    Define { dtype: DType, scope: Scope, ro: bool, len: Dim }, // len is 0 for global stores
    Store { dst: OpId, x: OpId, index: OpId, vlen: u8 },
    Load { src: OpId, index: OpId, vlen: u8 },
    Index { len: Dim, scope: Scope, axis: u32 },
    Loop { len: Dim, axis: u32 },
    EndLoop,
    // fused multiply add
    Mad { x: OpId, y: OpId, z: OpId },
    // fused matmul, a, b, c are fragments, each is a vector, c is accumulator, returns new accumulated vector d
    WMMA { dims: MMADims, layout: MMALayout, dtype: MMADType, a: OpId, b: OpId, c: OpId },
    // Vectorization, YAY!
    Vectorize { ops: Vec<OpId> },
    Devectorize { vec: OpId, idx: usize }, // select a single value from a vector

    // ops that exist only in kernelizer, basically they can be eventually removed.
    // TODO Get rid of the view, use whatever ops that are needed directly
    // and then use unfold movement ops function to convert it all into indices.
    // This will make Op smaller and Copy.
    // TODO Use MovementOp instead for all the movement.
    ConstView(Box<(Constant, View)>),
    LoadView(Box<(DType, View)>),
    StoreView { src: OpId, dtype: DType },
    Move { x: OpId, mop: Box<MoveOp> },
    Reduce { x: OpId, rop: BOp, n_axes: UAxis },
}

impl Op {
    // TODO use custom non allocating iterator instead of allocating a vec
    pub fn parameters(&self) -> impl Iterator<Item = OpId> + DoubleEndedIterator {
        match self {
            Op::ConstView { .. } => vec![],
            Op::LoadView { .. } => vec![],
            &Op::Move { x, .. } => vec![x],
            &Op::StoreView { src, .. } => vec![src],
            Op::Reduce { x, .. } => vec![*x],
            &Op::Store { dst, x, index, .. } => vec![dst, x, index],
            Op::Cast { x, .. } => vec![*x],
            Op::Unary { x, .. } => vec![*x],
            &Op::Binary { x, y, .. } => vec![x, y],
            Op::Const { .. } => vec![],
            Op::Define { .. } => vec![],
            &Op::Load { src, index, .. } => vec![src, index],
            Op::Index { .. } => vec![],
            Op::Loop { .. } => vec![],
            Op::EndLoop { .. } => vec![],
            &Op::Mad { x, y, z } => vec![x, y, z],
            Op::Vectorize { ops } => ops.clone(),
            &Op::Devectorize { vec, .. } => vec![vec],
            &Op::WMMA { a, b, c, .. } => vec![a, b, c],
        }
        .into_iter()
    }

    pub fn parameters_mut(&mut self) -> impl Iterator<Item = &mut OpId> + DoubleEndedIterator {
        match self {
            Op::ConstView { .. } => vec![],
            Op::LoadView { .. } => vec![],
            Op::StoreView { src, .. } => vec![src],
            Op::Move { x, .. } => vec![x],
            Op::Reduce { x, .. } => vec![x],
            Op::Store { dst, x, index, .. } => vec![dst, x, index],
            Op::Cast { x, .. } => vec![x],
            Op::Unary { x, .. } => vec![x],
            Op::Binary { x, y, .. } => vec![x, y],
            Op::Const { .. } => vec![],
            Op::Define { .. } => vec![],
            Op::Load { src, index, .. } => vec![src, index],
            Op::Index { .. } => vec![],
            Op::Loop { .. } => vec![],
            Op::EndLoop { .. } => vec![],
            Op::Mad { x, y, z } => vec![x, y, z],
            Op::Vectorize { ops } => ops.iter_mut().collect(),
            Op::Devectorize { vec, .. } => vec![vec],
            Op::WMMA { a, b, c, .. } => vec![a, b, c],
        }
        .into_iter()
    }

    pub fn is_const(&self) -> bool {
        matches!(self, Op::Cast { .. })
    }

    pub fn is_load(&self) -> bool {
        matches!(self, Op::Load { .. })
    }

    pub fn remap_params(&mut self, remapping: &Map<OpId, OpId>) {
        for param in self.parameters_mut() {
            if let Some(remapped_id) = remapping.get(param) {
                *param = *remapped_id;
            }
        }
    }
}

impl OpId {
    pub const NULL: Self = Self(u32::MAX);

    pub fn is_null(&self) -> bool {
        self.0 == u32::MAX
    }
}

impl std::fmt::Display for OpId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl From<usize> for OpId {
    fn from(value: usize) -> Self {
        OpId(value as u32)
    }
}

impl From<OpId> for usize {
    fn from(value: OpId) -> usize {
        value.0 as usize
    }
}

impl SlabId for OpId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Scope::Global => "global",
            Scope::Local => "local",
            Scope::Register => "reg",
        })
    }
}

impl PartialEq for Kernel {
    fn eq(&self, other: &Self) -> bool {
        self.ops == other.ops && self.head == other.head
    }
}

impl Eq for Kernel {}

impl SerBin for Kernel {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        self.ops.ser_bin(output);
        self.head.ser_bin(output);
        self.tail.ser_bin(output);
    }
}

impl DeBin for Kernel {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        let ops = Slab::<OpId, OpNode>::de_bin(offset, bytes)?;
        let start = OpId::de_bin(offset, bytes)?;
        let end = OpId::de_bin(offset, bytes)?;
        Ok(Self { head: start, tail: end, ops, outputs: Vec::new(), loads: Vec::new(), stores: Vec::new() })
    }
}

impl Hash for Kernel {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.head.hash(state);
        self.ops.hash(state);
    }
}

impl Kernel {
    #[track_caller]
    pub fn at(&self, op_id: OpId) -> &Op {
        &self.ops[op_id].op
    }

    pub fn prev_op(&self, op_id: OpId) -> OpId {
        self.ops[op_id].prev
    }

    pub fn next_op(&self, op_id: OpId) -> OpId {
        self.ops[op_id].next
    }

    /*pub fn ops_mut(&mut self) -> impl Iterator<Item = &mut Op> {
        self.ops.values_mut().map(|op_node| &mut op_node.op)
    }*/

    pub fn insert_before(&mut self, before_id: OpId, op: Op) -> OpId {
        debug_assert!(!before_id.is_null());
        debug_assert!(!self.ops.is_empty());

        let prev = self.ops[before_id].prev;
        let op_node = OpNode { prev, next: before_id, op: op };
        let op_id = self.ops.push(op_node);
        self.ops[before_id].prev = op_id;
        if prev.is_null() {
            self.head = op_id;
        } else {
            self.ops[prev].next = op_id;
        }
        op_id
    }

    pub fn insert_after(&mut self, after_id: OpId, op: Op) -> OpId {
        debug_assert!(!after_id.is_null());
        debug_assert!(!self.ops.is_empty());

        let next = self.ops[after_id].next;
        let op_node = OpNode { prev: after_id, next, op: op };
        let op_id = self.ops.push(op_node);
        self.ops[after_id].next = op_id;
        if next.is_null() {
            self.tail = op_id;
        } else {
            self.ops[next].prev = op_id;
        }
        op_id
    }

    pub fn move_op_after(&mut self, op_id: OpId, after_id: OpId) {
        debug_assert!(!op_id.is_null());
        debug_assert!(!after_id.is_null());
        debug_assert!(!self.ops.is_empty());

        //println!("moving op={op_id}, after={after_id}");

        // Remove
        let OpNode { prev, next, .. } = self.ops[op_id];
        if prev.is_null() {
            self.head = next;
        } else {
            self.ops[prev].next = next;
        }
        if next.is_null() {
            self.tail = prev;
        } else {
            self.ops[next].prev = prev;
        }

        // Insert
        self.ops[op_id].prev = after_id;
        let next = self.ops[after_id].next;
        self.ops[op_id].next = next;
        self.ops[after_id].next = op_id;
        if next.is_null() {
            self.tail = op_id;
        } else {
            self.ops[next].prev = op_id;
        }
    }

    pub fn move_op_before(&mut self, op_id: OpId, before_id: OpId) {
        debug_assert!(!op_id.is_null());
        debug_assert!(!before_id.is_null());
        debug_assert!(!self.ops.is_empty());

        //println!("moving op={op_id}, before={before_id}");

        // Remove
        let OpNode { prev, next, .. } = self.ops[op_id];
        if prev.is_null() {
            self.head = next;
        } else {
            self.ops[prev].next = next;
        }
        if next.is_null() {
            self.tail = prev;
        } else {
            self.ops[next].prev = prev;
        }

        // Insert
        self.ops[op_id].next = before_id;
        let prev = self.ops[before_id].prev;
        self.ops[op_id].prev = prev;
        self.ops[before_id].prev = op_id;
        if prev.is_null() {
            self.head = op_id;
        } else {
            self.ops[prev].next = op_id;
        }
    }

    pub fn remove_op(&mut self, op_id: OpId) {
        debug_assert!(!op_id.is_null());
        debug_assert!(!self.ops.is_empty());

        let OpNode { prev, next, .. } = self.ops[op_id];
        if prev.is_null() {
            self.head = next;
        } else {
            self.ops[prev].next = next;
        }
        if next.is_null() {
            self.tail = prev;
        } else {
            self.ops[next].prev = prev;
        }

        self.ops.remove(op_id);
    }

    pub fn iter_unordered(&self) -> impl Iterator<Item = (OpId, &Op)> {
        self.ops.iter().map(|(id, node)| (id, &node.op))
    }

    pub fn flop_mem_rw(&self) -> (u64, u64, u64) {
        #[derive(Clone)]
        struct Info {
            shape: Vec<Dim>,
            flops: u64,
            mem_read: u64,
            mem_write: u64,
        }

        let mut stack: Map<OpId, Info> = Map::default();

        let mut op_id = self.head;
        while !op_id.is_null() {
            let info = match self.at(op_id) {
                Op::ConstView(x) => {
                    let shape = x.1.shape();
                    Info { shape, flops: 0, mem_read: 0, mem_write: 0 }
                }
                Op::LoadView(x) => {
                    let (dtype, view) = x.as_ref();
                    let shape = view.shape();
                    let mem_read = view.original_numel() as u64 * dtype.byte_size() as u64;
                    Info { shape, flops: 0, mem_read, mem_write: 0 }
                }
                Op::StoreView { src, dtype } => {
                    let Info { shape, .. } = stack[src].clone();
                    let mem_write = shape.iter().product::<Dim>() as u64 * dtype.byte_size() as u64;
                    Info { shape, flops: 0, mem_read: 0, mem_write }
                }
                Op::Move { mop, .. } => match mop.as_ref() {
                    MoveOp::Reshape { shape, .. } => Info { shape: shape.clone(), flops: 0, mem_read: 0, mem_write: 0 },
                    MoveOp::Expand { shape } => Info { shape: shape.clone(), flops: 0, mem_read: 0, mem_write: 0 },
                    MoveOp::Permute { shape, .. } => Info { shape: shape.clone(), flops: 0, mem_read: 0, mem_write: 0 },
                    MoveOp::Pad { shape, .. } => Info { shape: shape.clone(), flops: 0, mem_read: 0, mem_write: 0 },
                },
                Op::Reduce { x, n_axes, .. } => {
                    let Info { mut shape, .. } = stack[x].clone();
                    let rd: Dim = shape[shape.len() - n_axes..].iter().product();
                    shape.truncate(shape.len() - n_axes);
                    let n: Dim = shape.iter().product();
                    let flops = n * (rd - 1);
                    let flops = flops as u64;
                    Info { shape, flops, mem_read: 0, mem_write: 0 }
                }
                Op::Cast { x, .. } | Op::Unary { x, .. } => {
                    let Info { shape, .. } = stack[x].clone();
                    let flops = shape.iter().product::<Dim>() as u64;
                    Info { shape, flops, mem_read: 0, mem_write: 0 }
                }
                Op::Binary { x, .. } => {
                    let Info { shape, .. } = stack[x].clone();
                    let flops = shape.iter().product::<Dim>() as u64;
                    Info { shape, flops, mem_read: 0, mem_write: 0 }
                }
                Op::WMMA { .. }
                | Op::Vectorize { .. }
                | Op::Devectorize { .. }
                | Op::Store { .. }
                | Op::Mad { .. }
                | Op::Const(_)
                | Op::Define { .. }
                | Op::Load { .. }
                | Op::Index { .. }
                | Op::Loop { .. }
                | Op::EndLoop => todo!(),
            };
            stack.insert(op_id, info);
            op_id = self.next_op(op_id);
        }

        stack.into_values().fold((0, 0, 0), |acc, info| {
            (acc.0 + info.flops, acc.1 + info.mem_read, acc.2 + info.mem_write)
        })
    }

    pub fn contains_stores(&self) -> bool {
        self.ops.values().any(|x| matches!(x.op, Op::StoreView { .. }))
    }

    pub fn is_reduce(&self) -> bool {
        self.ops.values().any(|x| matches!(x.op, Op::Reduce { .. }))
    }

    pub fn reduce_dims(&self, op_id: OpId) -> Vec<Dim> {
        let mut params = vec![op_id];
        let mut n_reduce_axes = 0;
        let mut visited = Set::default();
        while let Some(param) = params.pop() {
            if visited.insert(param) {
                match self.at(param) {
                    Op::ConstView(x) => {
                        let view = &x.1;
                        let n = view.rank();
                        return view.shape()[n - n_reduce_axes..].into();
                    }
                    Op::LoadView(x) => {
                        let view = &x.1;
                        let n = view.rank();
                        return view.shape()[n - n_reduce_axes..].into();
                    }
                    Op::Reduce { n_axes, .. } => n_reduce_axes += n_axes,
                    Op::Move { mop, .. } => match mop.as_ref() {
                        MoveOp::Reshape { shape, .. } => {
                            return shape[shape.len() - n_reduce_axes..].into();
                        }
                        MoveOp::Expand { shape } => {
                            return shape[shape.len() - n_reduce_axes..].into();
                        }
                        MoveOp::Permute { shape, .. } => {
                            return shape[shape.len() - n_reduce_axes..].into();
                        }
                        MoveOp::Pad { shape, .. } => {
                            return shape[shape.len() - n_reduce_axes..].into();
                        }
                    },
                    _ => {}
                }
                params.extend(self.at(param).parameters());
            }
        }
        unreachable!();
    }

    pub fn shape(&self) -> Vec<Dim> {
        if self.ops.values().any(|x| matches!(x.op, Op::Index { .. })) {
            let mut indices: Vec<(Dim, u32)> = self
                .ops
                .values()
                .filter_map(|x| {
                    // TODO include both global and local, order by axis
                    if let Op::Index { len: dim, axis, .. } = x.op {
                        Some((dim, axis))
                    } else {
                        None
                    }
                })
                .collect();
            indices.sort_by_key(|x| x.1);
            return indices.into_iter().map(|x| x.0).collect();
        }
        let mut reduce_dims = 0;
        let mut op_id = self.tail;
        while !op_id.is_null() {
            match self.at(op_id) {
                Op::ConstView(x) => {
                    let shape = x.1.shape();
                    let shape: Vec<Dim> = shape[..shape.len() - reduce_dims].into();
                    if shape.is_empty() {
                        return vec![1];
                    }
                    return shape;
                }
                Op::LoadView(x) => {
                    let shape = x.1.shape();
                    let shape: Vec<Dim> = shape[..shape.len() - reduce_dims].into();
                    if shape.is_empty() {
                        return vec![1];
                    }
                    return shape;
                }
                Op::Reduce { n_axes, .. } => {
                    reduce_dims += n_axes;
                }
                Op::Move { mop, .. } => match mop.as_ref() {
                    MoveOp::Reshape { shape, .. } => {
                        let shape: Vec<Dim> = shape[..shape.len() - reduce_dims].into();
                        if shape.is_empty() {
                            return vec![1];
                        }
                        return shape;
                    }
                    MoveOp::Expand { shape } => {
                        let shape: Vec<Dim> = shape[..shape.len() - reduce_dims].into();
                        if shape.is_empty() {
                            return vec![1];
                        }
                        return shape;
                    }
                    MoveOp::Permute { shape, .. } => {
                        let shape: Vec<Dim> = shape[..shape.len() - reduce_dims].into();
                        if shape.is_empty() {
                            return vec![1];
                        }
                        return shape;
                    }
                    MoveOp::Pad { shape, .. } => {
                        let shape: Vec<Dim> = shape[..shape.len() - reduce_dims].into();
                        if shape.is_empty() {
                            return vec![1];
                        }
                        return shape;
                    }
                },
                _ => {}
            }
            op_id = self.prev_op(op_id);
        }
        unreachable!()
    }

    #[allow(unused)]
    pub fn is_reshape_contiguous(&self, range: std::ops::Range<UAxis>, shape: &[Dim]) -> bool {
        self.ops.values().all(|node| match &node.op {
            Op::ConstView(x) => x.1.is_reshape_contiguous(range.clone(), shape),
            Op::LoadView(x) => x.1.is_reshape_contiguous(range.clone(), shape),
            _ => true,
        })
    }

    /// Fuses multiple reduce ops together if possible
    pub fn fuse_reduces(&mut self) {
        // TODO
    }

    /// Get index loop ids, dimensions and strides
    /// returns loop_id -> (dimension, stride)
    /// if returned loop_id is OpId::NULL, the stride is constant and dimension is 0 (unknown)
    pub fn get_strides(&self, index: OpId) -> Map<OpId, (Dim, Dim)> {
        use Op::*;
        println!("Get index {index}");

        let mut params = vec![index];
        let mut indices = Map::default();

        while let Some(param) = params.pop() {
            match self.ops[param].op {
                Binary { x, y, bop } => {
                    if bop == BOp::Add {
                        if let Loop { len, .. } = self.ops[x].op {
                            indices.insert(x, (len, 1));
                            params.push(y);
                        } else if let Index { len, .. } = self.ops[x].op {
                            indices.insert(x, (len, 1));
                            params.push(y);
                        } else if let Loop { len, .. } = self.ops[y].op {
                            indices.insert(y, (len, 1));
                            params.push(x);
                        } else if let Index { len, .. } = self.ops[y].op {
                            indices.insert(y, (len, 1));
                            params.push(x);
                        } else {
                            params.push(x);
                            params.push(y);
                        }
                    }
                    if bop == BOp::Mul {
                        match (&self.ops[x].op, &self.ops[y].op) {
                            (Loop { len, .. }, Const(c)) => {
                                indices.insert(x, (*len, c.as_dim()));
                            }
                            (Index { len, .. }, Const(c)) => {
                                indices.insert(x, (*len, c.as_dim()));
                            }
                            (Const(c), Loop { len, .. }) => {
                                indices.insert(y, (*len, c.as_dim()));
                            }
                            (Const(c), Index { len, .. }) => {
                                indices.insert(y, (*len, c.as_dim()));
                            }
                            _ => {} //op => println!("op={op:?}"),
                        }
                    }
                }
                Mad { x, y, z } => {
                    if let Loop { len, .. } = self.ops[z].op {
                        indices.insert(z, (len, 1));
                    } else if let Index { len, .. } = self.ops[z].op {
                        indices.insert(z, (len, 1));
                    } else {
                        params.push(z);
                    }
                    match (&self.ops[x].op, &self.ops[y].op) {
                        (Loop { len: dim, .. }, Const(c)) => {
                            indices.insert(x, (*dim, c.as_dim()));
                        }
                        (Const(c), Loop { len: dim, .. }) => {
                            indices.insert(y, (*dim, c.as_dim()));
                        }
                        _ => {}
                    }
                }
                Const(c) => {
                    indices.insert(OpId::NULL, (0, c.as_dim()));
                }
                ref op => {
                    println!("op={op:?}");
                }
            }
        }

        indices
    }

    fn remap(&mut self, x: OpId, y: OpId) {
        for op_node in self.ops.values_mut() {
            for param in op_node.op.parameters_mut() {
                if *param == x {
                    *param = y;
                }
            }
        }
    }

    pub fn verify(&self) {
        let mut stack = Vec::new();
        stack.push(Set::default());
        let check = |op_id, x: OpId, stack: &[Set<OpId>]| {
            if !stack.iter().any(|set| set.contains(&x)) {
                println!(
                    "{op_id} {:?} uses {x} -> {:?} before declaration.",
                    self.ops[op_id].op, self.ops[x].op
                );
                self.debug();
                panic!();
            }
        };

        let mut op_id = self.head;
        let mut prev: OpId;
        let mut dtypes: Map<OpId, DType> = Map::default();
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::ConstView(ref x) => {
                    dtypes.insert(op_id, x.0.dtype());
                }
                Op::LoadView(ref x) => {
                    dtypes.insert(op_id, x.0);
                }
                Op::StoreView { src, .. } => {
                    check(op_id, src, &stack);
                    dtypes.insert(op_id, dtypes[&src]);
                }
                Op::Store { dst, x, index, vlen: _ } => {
                    check(op_id, dst, &stack);
                    check(op_id, x, &stack);
                    check(op_id, index, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Cast { x, dtype } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtype);
                }
                Op::Reduce { x, .. } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                    if stack.len() > 1 {
                        stack.pop();
                    }
                }
                Op::Unary { x, .. } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Binary { x, y, bop } => {
                    check(op_id, x, &stack);
                    check(op_id, y, &stack);
                    if dtypes[&x] != dtypes[&y] {
                        println!("Binary dtype mismatch on op={op_id}.");
                        self.debug();
                        panic!();
                    }
                    if bop.returns_bool() {
                        dtypes.insert(op_id, DType::Bool);
                    } else {
                        dtypes.insert(op_id, dtypes[&x]);
                    }
                }
                Op::Vectorize { ref ops } => {
                    let dtype = dtypes[&ops[0]];
                    for &x in ops {
                        check(op_id, x, &stack);
                        if dtypes[&x] != dtype {
                            println!("Vectorize dtype mismatch on op={op_id}.");
                            self.debug();
                            panic!();
                        }
                    }
                    dtypes.insert(op_id, dtype);
                }
                Op::Devectorize { .. } => todo!(),
                Op::WMMA { c, a, b, .. } => {
                    let dtype = dtypes[&c];
                    check(op_id, c, &stack);
                    check(op_id, a, &stack);
                    check(op_id, b, &stack);
                    if dtypes[&a] != dtypes[&b] {
                        println!("MMA dtype mismatch on op={op_id}.");
                        self.debug();
                        panic!();
                    }
                    dtypes.insert(op_id, dtype);
                }
                Op::Mad { x, y, z } => {
                    check(op_id, x, &stack);
                    check(op_id, y, &stack);
                    check(op_id, z, &stack);
                    if dtypes[&x] != dtypes[&y] || dtypes[&x] != dtypes[&z] {
                        println!("Mad dtype mismatch on op={op_id}.");
                        self.debug();
                        panic!();
                    }
                    dtypes.insert(op_id, dtypes[&x]);
                }
                Op::Const(v) => {
                    dtypes.insert(op_id, v.dtype());
                }
                Op::Define { dtype, .. } => {
                    dtypes.insert(op_id, dtype);
                }
                Op::Load { src, index, .. } => {
                    check(op_id, src, &stack);
                    check(op_id, index, &stack);
                    dtypes.insert(op_id, dtypes[&src]);
                }
                Op::Index { .. } => {
                    dtypes.insert(op_id, IDX_T);
                }
                Op::Loop { .. } => {
                    stack.push(Set::default());
                    dtypes.insert(op_id, IDX_T);
                }
                Op::EndLoop => {
                    if stack.is_empty() {
                        println!("Endloop without matching loop.");
                        self.debug();
                        panic!();
                    }
                    stack.pop();
                }
                Op::Move { x, .. } => {
                    check(op_id, x, &stack);
                    dtypes.insert(op_id, dtypes[&x]);
                }
            }
            stack.last_mut().unwrap().insert(op_id);
            prev = op_id;
            op_id = self.ops[op_id].next;
            if !op_id.is_null() && self.ops[op_id].prev != prev {
                println!("Inconsistency in prev.");
                self.debug();
                panic!()
            }
        }
        if stack.len() != 1 {
            println!("Wrong {} closing endloops.", stack.len());
            self.debug();
            panic!();
        }
        self.check_oob();
    }

    pub fn check_oob(&self) {
        use std::collections::HashMap;
        let mut ids: Map<OpId, (usize, usize)> = HashMap::default();
        let mut defines = Map::default();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match *self.at(op_id) {
                Op::Const(x) => {
                    if x.is_positive() {
                        let Constant::U64(x) = x.cast(DType::U64) else { unreachable!() };
                        let v = usize::from_le_bytes(x);
                        ids.insert(op_id, (v, v));
                    }
                }
                Op::Define { len, .. } => {
                    defines.insert(op_id, len);
                }
                Op::Cast { x, .. } => {
                    if let Some((l, u)) = ids.get(&x) {
                        ids.insert(op_id, (*l, *u));
                    }
                }
                Op::Binary { x, y, bop } => {
                    if let Some(&(xl, xu)) = ids.get(&x)
                        && let Some(&(yl, yu)) = ids.get(&y)
                    {
                        ids.insert(
                            op_id,
                            match bop {
                                BOp::Add => (xl.wrapping_add(yl), xu.wrapping_add(yu)),
                                BOp::Sub => (xl.wrapping_sub(yl), xu.wrapping_sub(yu)),
                                BOp::Mul => (xl.wrapping_mul(yl), xu.wrapping_mul(yu)),
                                BOp::Div => (xl / yl, xu / yu),
                                BOp::Mod => (xl % yl, xu % yu),
                                BOp::Eq => ((xl == yl) as usize, (xu == yu) as usize),
                                BOp::NotEq => ((xl != yl) as usize, (xu != yu) as usize),
                                BOp::Cmpgt => ((xl > yl) as usize, (xu > yu) as usize),
                                BOp::Cmplt => ((xl < yl) as usize, (xu < yu) as usize),
                                BOp::And => ((xl == 1 && yl == 1) as usize, (xu == 1 && yu == 1) as usize),
                                BOp::BitShiftLeft => (xl << yl, xu << yu),
                                BOp::BitShiftRight => (xl >> yl, xu >> yu),
                                BOp::Pow => (xl.pow(yl as u32), xu.pow(yu as u32)),
                                op => todo!("{:?}", op),
                            },
                        );
                    }
                }
                Op::Mad { x, y, z } => {
                    if let Some(&(xl, xu)) = ids.get(&x)
                        && let Some(&(yl, yu)) = ids.get(&y)
                        && let Some(&(zl, zu)) = ids.get(&z)
                    {
                        ids.insert(
                            op_id,
                            (
                                xl.wrapping_mul(yl).wrapping_add(zl),
                                xu.wrapping_mul(yu).wrapping_add(zu),
                            ),
                        );
                    }
                }
                Op::Index { len: dim, .. } => {
                    ids.insert(op_id, (0, dim - 1));
                }
                Op::Loop { len: dim, .. } => {
                    ids.insert(op_id, (0, dim - 1));
                }
                Op::Load { src, index, .. } => {
                    if !ids.contains_key(&index) {
                        self.debug();
                        panic!("Missing index={index} for op_id={op_id} -> {:?}", self.ops[op_id]);
                    }
                    let idx_range = ids[&index];
                    //println!("Max idx range: {}, define {}", idx_range.1, defines[src]);
                    if idx_range.1 > defines[&src] - 1 {
                        self.debug();
                        panic!(
                            "OOB detected in op {}: index {:?} exceeds buffer length {:?}",
                            op_id, idx_range, defines[&src]
                        );
                    }
                }
                Op::Store { dst, index, .. } => {
                    if !ids.contains_key(&index) {
                        panic!("Missing index={index} for op_id={op_id} -> {:?}", self.ops[op_id]);
                    }
                    let idx_range = ids[&index];
                    //println!("Max idx range: {}, define {}", idx_range.1, defines[src]);
                    if idx_range.1 > defines[&dst] - 1 {
                        self.debug();
                        panic!(
                            "OOB detected in op {}: index {:?} exceeds buffer length {:?}",
                            op_id, idx_range, defines[&dst]
                        );
                    }
                }
                Op::Vectorize { ref ops } => {
                    let mut r = None;
                    for x in ops {
                        if let Some(&(xl, xu)) = ids.get(x) {
                            if let Some((l, u)) = r {
                                r = Some((xl.min(l), xu.max(u)));
                            } else {
                                r = Some((xl, xu));
                            }
                        }
                    }
                    if let Some((xl, xu)) = r {
                        ids.insert(op_id, (xl, xu));
                    }
                }
                _ => {}
            }
            op_id = self.ops[op_id].next;
        }
    }

    pub fn push_back(&mut self, op: Op) -> OpId {
        debug_assert!(!self.ops.is_empty());
        let op_node = OpNode { prev: self.tail, next: OpId::NULL, op };
        let op_id = self.ops.push(op_node);
        self.ops[self.tail].next = op_id;
        self.tail = op_id;
        op_id
    }

    pub fn remove_first_output(&mut self, x: TensorId) {
        //println!("removing tensor {x} from kernel {kid:?}");
        let outputs = &mut self.outputs;
        outputs.iter().position(|elem| *elem == x).map(|i| outputs.remove(i));
    }

    pub fn drop_unused_ops(&mut self, visited: &Map<TensorId, (KMKernelId, OpId)>) {
        let params = self.outputs.iter().map(|tid| visited[tid].1).collect();
        let required = self.get_required_ops(params);
        let mut loaded_tensors = Vec::new();
        let mut load_index = 0;
        let loads = self.loads.clone(); // TODO remove the clone once partial borrows are working in rust
        let mut op_id = self.head;
        while !op_id.is_null() {
            let is_required = required.contains(&op_id);
            if let Op::LoadView { .. } = self.at(op_id) {
                if is_required {
                    loaded_tensors.push(loads[load_index]);
                }
                load_index += 1;
            }
            let temp = op_id;
            op_id = self.next_op(op_id);
            if !is_required {
                self.remove_op(temp);
            }
        }
        self.loads = loaded_tensors;
        #[cfg(debug_assertions)]
        if self.loads.len() != self.ops.values().filter(|op| matches!(op.op, Op::LoadView { .. })).count() {
            self.debug();
            panic!();
        }
    }

    pub fn get_required_ops(&self, mut params: Vec<OpId>) -> Set<OpId> {
        let mut required = Set::default();
        while let Some(param) = params.pop() {
            if required.insert(param) {
                //println!("param={param}");
                match self.at(param) {
                    Op::Reduce { x, .. } | Op::Cast { x, .. } | Op::Unary { x, .. } | Op::Move { x, .. } => {
                        params.push(*x);
                    }
                    Op::Binary { x, y, .. } => {
                        params.push(*x);
                        params.push(*y);
                    }
                    Op::Const { .. } | Op::ConstView { .. } | Op::LoadView { .. } => {}
                    Op::Vectorize { .. }
                    | Op::Devectorize { .. }
                    | Op::WMMA { .. }
                    | Op::Define { .. }
                    | Op::Mad { .. }
                    | Op::StoreView { .. }
                    | Op::Load { .. }
                    | Op::Store { .. }
                    | Op::Index { .. }
                    | Op::Loop { .. }
                    | Op::EndLoop { .. } => unreachable!(),
                }
            }
        }
        required
    }

    /// Find all multiply add operations and fuse them
    pub fn fuse_mad(&mut self) {
        let mut op_id = self.head;
        let mut rcs = Map::default();
        while !op_id.is_null() {
            for param in self.ops[op_id].op.parameters() {
                rcs.entry(param).and_modify(|rc| *rc += 1).or_insert(1);
            }
            if let Op::Binary { x: xo, y: yo, bop } = self.ops[op_id].op {
                if bop == BOp::Add {
                    if let Op::Binary { x, y, bop } = self.ops[xo].op {
                        if bop == BOp::Mul && rcs[&xo] == 1 {
                            self.ops[op_id].op = Op::Mad { x, y, z: yo };
                        }
                    } else if let Op::Binary { x, y, bop } = self.ops[yo].op {
                        if bop == BOp::Mul && rcs[&yo] == 1 {
                            self.ops[op_id].op = Op::Mad { x, y, z: xo };
                        }
                    }
                }
            }
            op_id = self.next_op(op_id);
        }
    }

    pub fn unfold_pows(&mut self) {
        let mut op_id = self.ops.first_id();
        while !op_id.is_null() {
            if let &Op::Binary { x, y, bop } = self.at(op_id) {
                if bop == BOp::Pow {
                    let x = self.insert_before(op_id, Op::Unary { x, uop: UOp::Log2 });
                    let x = self.insert_before(op_id, Op::Binary { x, y, bop: BOp::Mul });
                    self.ops[op_id].op = Op::Unary { x, uop: UOp::Exp2 };
                }
            }
            op_id = self.ops.next_id(op_id);
        }

        #[cfg(debug_assertions)]
        self.verify();
    }
}
