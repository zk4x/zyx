// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

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

pub mod autotune;
pub mod custom;
mod debug;
mod div_mod;
mod fold_constants;
mod fold_loops;
mod fuse;
mod licm;
mod merge_loops;
mod mma;
mod split_loops;
mod tile_registers;
mod tiled_reduce;
mod unfold;
mod unroll_loops;
mod upcast;
mod vectorize;
mod verify;

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
    Trunc,
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
    Pad { padding: Vec<(i64, i64)>, shape: Vec<Dim> },
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
    Cast {
        x: OpId,
        dtype: DType,
    },
    Unary {
        x: OpId,
        uop: UOp,
    },
    // For binary ops, next of x is y, then next of y is the binary op
    Binary {
        x: OpId,
        y: OpId,
        bop: BOp,
    },

    // ops that only exist after unfolding views and reduces
    Const(Constant),
    Define {
        dtype: DType,
        scope: Scope,
        ro: bool,
        len: Dim,
    }, // len is 0 for global stores
    Store {
        dst: OpId,
        x: OpId,
        index: OpId,
        vlen: u8,
    },
    Load {
        src: OpId,
        index: OpId,
        vlen: u8,
    },
    Index {
        len: Dim,
        scope: Scope,
        axis: u32,
    },
    Loop {
        len: Dim,
    },
    EndLoop,
    // fused multiply add
    Mad {
        x: OpId,
        y: OpId,
        z: OpId,
    },
    // fused matmul, a, b, c are fragments, each is a vector, c is accumulator, returns new accumulated vector d
    WMMA {
        dims: MMADims,
        layout: MMALayout,
        dtype: MMADType,
        a: OpId,
        b: OpId,
        c: OpId,
    },
    // Vectorization, YAY!
    Vectorize {
        ops: Vec<OpId>,
    },
    Devectorize {
        vec: OpId,
        idx: usize,
    }, // select a single value from a vector
    Barrier {
        scope: Scope,
    },
    If {
        condition: OpId, // must be boolean variable
    },
    EndIf,

    // ops that exist only in kernelizer, basically they can be eventually removed.
    // TODO Get rid of the view, use whatever ops that are needed directly
    // and then use unfold movement ops function to convert it all into indices.
    // This will make Op smaller and Copy.
    // TODO Use MovementOp instead for all the movement.
    ConstView(Box<(Constant, View)>),
    LoadView(Box<(DType, View)>),
    StoreView {
        src: OpId,
        dtype: DType,
    },
    Move {
        x: OpId,
        mop: Box<MoveOp>,
    },
    Reduce {
        x: OpId,
        rop: BOp,
        n_axes: UAxis,
    },
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
            Op::Index { .. } | Op::Loop { .. } | Op::EndLoop { .. } => vec![],
            &Op::Mad { x, y, z } => vec![x, y, z],
            Op::Vectorize { ops } => ops.clone(),
            &Op::Devectorize { vec, .. } => vec![vec],
            &Op::WMMA { a, b, c, .. } => vec![a, b, c],
            Op::Barrier { .. } => vec![],
            Op::If { condition } => vec![*condition],
            Op::EndIf => vec![],
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
            Op::Index { .. } | Op::Loop { .. } | Op::EndLoop { .. } => vec![],
            Op::Mad { x, y, z } => vec![x, y, z],
            Op::Vectorize { ops } => ops.iter_mut().collect(),
            Op::Devectorize { vec, .. } => vec![vec],
            Op::WMMA { a, b, c, .. } => vec![a, b, c],
            Op::If { condition } => vec![condition],
            Op::EndIf | Op::Barrier { .. } => vec![],
        }
        .into_iter()
    }

    pub const fn is_const(&self) -> bool {
        matches!(self, Op::Cast { .. })
    }

    pub const fn is_load(&self) -> bool {
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

    pub const fn is_null(&self) -> bool {
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
        let op_node = OpNode { prev, next: before_id, op };
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
        let op_node = OpNode { prev: after_id, next, op };
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
                    let mem_read = view.original_numel() as u64 * u64::from(dtype.bit_size() / 8);
                    Info { shape, flops: 0, mem_read, mem_write: 0 }
                }
                Op::StoreView { src, dtype } => {
                    let Info { shape, .. } = stack[src].clone();
                    let mem_write = u64::from(shape.iter().product::<Dim>()) * u64::from(dtype.bit_size() / 8);
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
                Op::Cast { x, .. } => {
                    let Info { shape, .. } = stack[x].clone();
                    let flops = 0; // Cast is not computation
                    Info { shape, flops, mem_read: 0, mem_write: 0 }
                }
                Op::Unary { x, .. } => {
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
                | Op::If { .. }
                | Op::EndIf
                | Op::Barrier { .. }
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

    /*pub fn flop_mem_rw1(&self) -> (u64, u64, u64) {
        self.debug_colorless();
        let mut n_instructions = 0u64;
        let mut bytes_read = 0u64;
        let mut bytes_written = 0u64;

        let mut gws = [1u64, 1, 1];
        let mut lws = [1u64, 1, 1];
        let mut loop_mult = 1u64;
        let mut latest_loop_lengths = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.at(op_id) {
                Op::Cast { .. } => n_instructions += loop_mult,
                Op::Unary { .. } => n_instructions += loop_mult,
                Op::Binary { .. } => n_instructions += loop_mult,
                Op::Const(_) | Op::Define { .. } => {}
                Op::Load { src, vlen, .. } => {
                    n_instructions += loop_mult * 3;
                    let Op::Define { scope, dtype, len, .. } = self.at(*src) else {
                        unreachable!()
                    };
                    let bytes = *len as u64 * (dtype.bit_size() / 8) as u64 * *vlen as u64;
                    match scope {
                        Scope::Global => bytes_read += bytes,
                        Scope::Local => bytes_read += bytes,
                        Scope::Register => bytes_read += bytes,
                    }
                }
                Op::Store { dst, vlen, .. } => {
                    n_instructions += loop_mult * 3;
                    let Op::Define { scope, dtype, len, .. } = self.at(*dst) else {
                        unreachable!()
                    };
                    let bytes = *len as u64 * (dtype.bit_size() / 8) as u64 * *vlen as u64;
                    match scope {
                        Scope::Global => bytes_written += bytes,
                        Scope::Local => bytes_written += bytes,
                        Scope::Register => bytes_written += bytes,
                    }
                }
                Op::Index { len, scope, axis } => match scope {
                    Scope::Global => gws[*axis as usize] = *len as u64,
                    Scope::Local => lws[*axis as usize] = *len as u64,
                    Scope::Register => {}
                },
                Op::Loop { len } => {
                    n_instructions += loop_mult * 3;
                    loop_mult *= *len as u64;
                    latest_loop_lengths.push(*len as u64);
                }
                Op::EndLoop => {
                    loop_mult /= latest_loop_lengths.pop().unwrap();
                }
                Op::Mad { .. } => n_instructions += loop_mult,
                Op::WMMA { dims, .. } => {
                    let (m, n, k) = dims.decompose_mnk();
                    n_instructions += loop_mult * m * n * k;
                }
                Op::Vectorize { .. } | Op::Devectorize { .. } => {}
                _ => {}
            }
            op_id = self.next_op(op_id);
        }

        let total_threads = gws[0] * gws[1] * gws[2] * lws[0] * lws[1] * lws[2];
        (n_instructions * total_threads, bytes_read, bytes_written)
    }*/

    pub fn contains_stores(&self) -> bool {
        self.ops.values().any(|x| matches!(x.op, Op::StoreView { .. }))
    }

    pub fn is_reduce(&self) -> bool {
        self.ops.values().any(|x| matches!(x.op, Op::Reduce { .. }))
    }

    // TODO as of now this is just imprecise estimate, make it better,
    // it does not correctly handle reshapes.
    /*pub fn cumulative_reduce_dim(&self, op_id: OpId, acc: Dim) -> Dim {
        match self.ops[op_id].op {
            Op::Cast { x, .. } | Op::Unary { x, .. } => self.cumulative_reduce_dim(x, acc),
            Op::Binary { x, y, .. } => self.cumulative_reduce_dim(x, acc) * self.cumulative_reduce_dim(y, acc),
            Op::ConstView(_) | Op::LoadView(_) => acc,
            Op::Move { x, .. } => self.cumulative_reduce_dim(op_id, acc),
            Op::Reduce { x, .. } => self.cumulative_reduce_dim(op_id, acc) * shape[n_axes].product9),
            ref op => todo!("{op:?}"),
        }
    }*/

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
        Vec::new()
    }

    #[allow(unused)]
    pub fn is_reshape_contiguous(&self, range: std::ops::Range<UAxis>, shape: &[Dim]) -> bool {
        self.ops.values().all(|node| match &node.op {
            Op::ConstView(x) => x.1.is_reshape_contiguous(range.clone(), shape),
            Op::LoadView(x) => x.1.is_reshape_contiguous(range.clone(), shape),
            _ => true,
        })
    }

    /// Get index loop ids, dimensions and strides
    /// returns loop_id -> (dimension, stride)
    /// if returned loop_id is OpId::NULL, the stride is constant and dimension is 0 (unknown)
    pub fn get_strides(&self, index: OpId) -> Map<OpId, (Dim, Dim)> {
        use Op::*;
        //println!("Get index {index}");

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
                                indices.insert(x, (*len, c.as_dim().unwrap()));
                            }
                            (Index { len, .. }, Const(c)) => {
                                indices.insert(x, (*len, c.as_dim().unwrap()));
                            }
                            (Const(c), Loop { len, .. }) => {
                                indices.insert(y, (*len, c.as_dim().unwrap()));
                            }
                            (Const(c), Index { len, .. }) => {
                                indices.insert(y, (*len, c.as_dim().unwrap()));
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
                            indices.insert(x, (*dim, c.as_dim().unwrap()));
                        }
                        (Const(c), Loop { len: dim, .. }) => {
                            indices.insert(y, (*dim, c.as_dim().unwrap()));
                        }
                        _ => {}
                    }
                }
                Const(c) => {
                    indices.insert(OpId::NULL, (0, c.as_dim().unwrap()));
                }
                _ => {}
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
                    | Op::Barrier { .. }
                    | Op::Define { .. }
                    | Op::Mad { .. }
                    | Op::StoreView { .. }
                    | Op::Load { .. }
                    | Op::Store { .. }
                    | Op::Index { .. }
                    | Op::If { .. }
                    | Op::EndIf
                    | Op::Loop { .. }
                    | Op::EndLoop { .. } => unreachable!(),
                }
            }
        }
        required
    }

    pub fn unfold_pows(&mut self) {
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let &Op::Binary { x, y, bop } = self.at(op_id) {
                if bop == BOp::Pow {
                    let x = self.insert_before(op_id, Op::Unary { x, uop: UOp::Log2 });
                    let x = self.insert_before(op_id, Op::Binary { x, y, bop: BOp::Mul });
                    self.ops[op_id].op = Op::Unary { x, uop: UOp::Exp2 };
                }
            }
            op_id = self.next_op(op_id);
        }

        self.verify();
    }

    pub fn get_global_indices(&self) -> std::collections::BTreeMap<u32, OpId> {
        let mut indices = std::collections::BTreeMap::new();
        for (op_id, op_node) in self.ops.iter() {
            if let Op::Index { scope, axis, .. } = op_node.op {
                if scope == Scope::Global {
                    indices.insert(axis, op_id);
                }
            }
        }
        indices
    }
}

impl MMADims {
    pub const fn decompose_mnk(&self) -> (u64, u64, u64) {
        match self {
            MMADims::m8n8k16 => (8, 8, 16),
            MMADims::m16n8k8 => (16, 8, 8),
            MMADims::m16n8k16 => (16, 8, 16),
        }
    }
}
