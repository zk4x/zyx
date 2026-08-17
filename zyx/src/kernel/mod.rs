// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Kernel Intermediate Representation for building custom compute kernels.
//!
//! This module provides the IR builder API for constructing custom kernels
//! that can be compiled and executed on any backend (CPU, CUDA, Vulkan, etc.).
//!
//! # Quick start — WMMA matrix multiply
//!
//! Tensor-core matmul using the m16n8k8 WMMA instruction with one warp per 16×8 tile.
//! Requires CUDA with tensor cores (compute capability ≥ 7.0).
//!
//! ```rust
//! use zyx::kernel::{DeviceId, Kernel, MMADType, MMADims, MMALayout, MemLayout, MemScope};
//! use zyx::DType;
//!
//! let (m, n, k) = (1024, 1024, 1024);
//! let mut kernel = Kernel::new(DeviceId::AUTO);
//!
//! let a_buf = kernel.define(DType::F16, MemScope::Global, true, &[m, k]);
//! let b_buf = kernel.define(DType::F16, MemScope::Global, true, &[k, n]);
//! let c_buf = kernel.define(DType::F32, MemScope::Global, false, &[m, n]);
//!
//! let gidx = kernel.group_index(0, m / 16);
//! let gidy = kernel.group_index(1, n / 8);
//! let wid = kernel.local_index(0, 32);
//!
//! let [c0, c1, c2, c4, c8, c16] = kernel.const_idxs([0u32, 1, 2, 4, 8, 16]);
//! let n_const = kernel.const_idx(n);
//! let k_const = kernel.const_idx(k);
//!
//! let row_in_tile = kernel.div(wid, c4);
//! let sub_col = kernel.mod_(wid, c4);
//! let col_in_tile = kernel.mul(sub_col, c2);
//!
//! let a_row = kernel.mad(gidx, c16, row_in_tile);
//! let b_col = kernel.mad(gidy, c8, row_in_tile);
//! let tile_base_col = kernel.mul(gidy, c8);
//!
//! let acc = kernel.define(DType::F32, MemScope::Register, false, &[4]);
//! let zf = kernel.const_val(0.0f32);
//! let zero_acc = kernel.vectorize(&[zf, zf, zf, zf]);
//! kernel.store(acc, zero_acc, c0, MemLayout::Vector(4));
//!
//! let k_div_8 = kernel.const_idx(k / 8);
//! let k_loop = kernel.loop_(k_div_8);
//! let k_off = kernel.mul(k_loop, c8);
//!
//! let a_base = kernel.mad(a_row, k_const, k_off);
//! let a_base = kernel.add(a_base, col_in_tile);
//! let a_load_0 = kernel.load(a_buf, a_base, MemLayout::Scalar);
//! let a_base_p1 = kernel.add(a_base, c1);
//! let a_load_1 = kernel.load(a_buf, a_base_p1, MemLayout::Scalar);
//! let a_base2 = kernel.mad(c8, k_const, a_base);
//! let a_load_2 = kernel.load(a_buf, a_base2, MemLayout::Scalar);
//! let a_base2_p1 = kernel.add(a_base2, c1);
//! let a_load_3 = kernel.load(a_buf, a_base2_p1, MemLayout::Scalar);
//! let a_frag = kernel.vectorize(&[a_load_0, a_load_1, a_load_2, a_load_3]);
//!
//! let b_row = kernel.add(k_off, col_in_tile);
//! let b_base = kernel.mad(b_row, n_const, b_col);
//! let b_load_0 = kernel.load(b_buf, b_base, MemLayout::Scalar);
//! let b_base_n = kernel.add(b_base, n_const);
//! let b_load_1 = kernel.load(b_buf, b_base_n, MemLayout::Scalar);
//! let b_frag = kernel.vectorize(&[b_load_0, b_load_1]);
//!
//! let acc_old = kernel.load(acc, c0, MemLayout::Vector(4));
//! let acc_new = kernel.wmma(
//!     MMADims::m16n8k8, MMALayout::row_col, MMADType::f16_f16_f16_f32,
//!     a_frag, b_frag, acc_old,
//! );
//! kernel.store(acc, acc_new, c0, MemLayout::Vector(4));
//! kernel.end_loop();
//!
//! let acc_final = kernel.load(acc, c0, MemLayout::Vector(4));
//! let [co, c1v, c2v, c3v] = kernel.devectorize(acc_final);
//!
//! let c_col = kernel.add(tile_base_col, col_in_tile);
//! let c_base = kernel.mad(a_row, n_const, c_col);
//! kernel.store(c_buf, co, c_base, MemLayout::Scalar);
//! let c_base_p1 = kernel.add(c_base, c1);
//! kernel.store(c_buf, c1v, c_base_p1, MemLayout::Scalar);
//! let c_base2 = kernel.mad(c8, n_const, c_base);
//! kernel.store(c_buf, c2v, c_base2, MemLayout::Scalar);
//! let c_base2_p1 = kernel.add(c_base2, c1);
//! kernel.store(c_buf, c3v, c_base2_p1, MemLayout::Scalar);
//!
//! // kernel.compile()?;  // requires CUDA with tensor cores
//! ```

pub use crate::backend::DeviceId;

use crate::{
    DType, Map, Set,
    dtype::Constant,
    shape::Dim,
    slab::{Slab, SlabId},
};
use nanoserde::{DeBin, SerBin};
use std::collections::BTreeMap;
use std::{hash::BuildHasherDefault, hash::Hash};

pub use custom::CompiledKernel;

mod algebraic;
pub(crate) mod autotune;
mod coarsen;
mod cost;
mod custom;
mod debug;
mod fold_constants;
mod fold_loops;
mod fuse;
mod instr_sched;
mod licm;
mod linearize;
mod local_reduce;
mod merge_loops;
mod mma;
mod ops;
mod pad_index;
mod predict_cost;
mod split_loops;
mod tenstorrent;
mod transforms;
mod unroll_loops;
mod vectorize;
mod verify;

pub(crate) use ops::{BOp, IdxKind, MoveOp, Op, OpId, OpNode, UOp};
pub use ops::{MMADType, MMADims, MMALayout, ParamKind};

// TODO later make this dynamic u32 or u64 depending on max range
/// Type used for indexing into arrays within kernels.
pub(crate) const IDX_T: DType = DType::U32;

/// Kernel builder for constructing custom compute kernels.
///
/// This struct represents a kernel in the intermediate representation (IR)
/// that can be compiled and executed on any backend (CPU, CUDA, Vulkan, etc.).
///
/// The kernel IR supports:
/// - Element-wise operations (add, mul, sin, exp, etc.)
/// - Reductions (sum, max, etc.)
/// - Memory operations (load, store)
/// - Control flow (loops, conditionals)
/// - Tensor transformations (reshape, permute, expand, pad)
///
/// # Example
///
/// Build a kernel that computes `sin(x) + cos(x)` element-wise:
///
/// ```
/// use zyx::kernel::{Kernel, MemScope, MemLayout, DeviceId};
/// use zyx::DType;
///
/// let mut kernel = Kernel::new(DeviceId::AUTO);
/// let n = 256;
/// let inp = kernel.define(DType::F32, MemScope::Global, true, &[n]);
/// let gidx = kernel.group_index(0, n);
/// let loaded = kernel.load(inp, gidx, MemLayout::Scalar);
/// let s = kernel.sin(loaded);
/// let c = kernel.cos(loaded);
/// let result = kernel.add(s, c);
/// let out = kernel.define(DType::F32, MemScope::Global, false, &[n]);
/// kernel.store(out, result, gidx, MemLayout::Scalar);
/// ```
///
/// # Compile
///
/// Build a kernel using fused multiply-add and compile it:
///
/// ```
/// use zyx::kernel::{Kernel, MemScope, MemLayout, DeviceId};
/// use zyx::{DType, Tensor, ZyxError};
///
/// let mut kernel = Kernel::new(DeviceId::AUTO);
/// let n = 4;
/// let inp = kernel.define(DType::F32, MemScope::Global, true, &[n]);
/// let gidx = kernel.group_index(0, n);
/// let loaded = kernel.load(inp, gidx, MemLayout::Scalar);
/// let result = kernel.mad(loaded, loaded, loaded); // x*x + x
/// let out = kernel.define(DType::F32, MemScope::Global, false, &[n]);
/// kernel.store(out, result, gidx, MemLayout::Scalar);
///
/// let compiled = kernel.compile()?;
/// let x = Tensor::from([1.0f32, 2.0, 3.0, 4.0]);
/// let result = compiled.forward(&[&x], vec![n])?;
/// let data: Vec<f32> = result.into_iter().next().unwrap().try_into()?;
/// assert_eq!(data, vec![2.0, 6.0, 12.0, 20.0]);
/// # Ok::<_, ZyxError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Kernel {
    /// Operation slab containing the kernel IR.
    pub(crate) ops: Slab<OpId, OpNode>,
    /// Head of the operation linked list.
    pub(crate) head: OpId,
    /// Tail of the operation linked list.
    pub(crate) tail: OpId,
    /// Target device for compilation.
    pub(crate) device_id: DeviceId,
}

/// Scope for memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MemScope {
    /// Global memory scope (DRAM).
    Global,
    /// Local memory scope (SRAM).
    Local,
    /// Register scope (registers).
    Register,
    /// Circular buffer, SRAM (tenstorrent)
    Circular,
    /// Single scalar variable
    Variable,
}

/// Memory layout for kernel operations.
///
/// Specifies how data is laid out in memory for efficient access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MemLayout {
    /// Scalar layout: one element per memory location
    Scalar,
    /// Vector layout: vector of size `x`
    Vector(u16),
    /// Tile layout: tile of `x` × `y` elements with stride
    Tile {
        /// Width of the tile
        x: u16,
        /// Height of the tile
        y: u16,
        /// Stride between tiles
        stride: u32,
    },
}

impl PartialEq for Kernel {
    fn eq(&self, other: &Self) -> bool {
        self.ops == other.ops && self.head == other.head && self.device_id == other.device_id
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

/*impl DeBin for Kernel {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        let ops = Slab::<OpId, OpNode>::de_bin(offset, bytes)?;
        let start = OpId::de_bin(offset, bytes)?;
        let end = OpId::de_bin(offset, bytes)?;
        Ok(Self { head: start, tail: end, ops, device_id: DeviceId::AUTO })
    }
}*/

impl Hash for Kernel {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.head.hash(state);
        self.ops.hash(state);
        self.device_id.hash(state);
    }
}

// Custom kernel machinery
impl Kernel {
    /// Compute dtypes and reference counts for all operations.
    pub(crate) fn compute_dtypes_and_rcs(&self) -> (Map<OpId, (DType, MemLayout)>, Map<OpId, u32>) {
        let mut rcs: Map<OpId, u32> = Map::with_capacity_and_hasher(self.ops.len().into(), BuildHasherDefault::new());
        let mut dtypes: Map<OpId, (DType, MemLayout)> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());

        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::Move { .. } | Op::Reduce { .. } | Op::ReduceTile { .. } => {
                    unreachable!()
                }
                Op::Const(x) => {
                    dtypes.insert(op_id, (x.dtype(), MemLayout::Scalar));
                }
                Op::Param { dtype, .. } => {
                    dtypes.insert(op_id, (dtype, MemLayout::Scalar));
                }
                Op::Storage { dtype, .. } => {
                    dtypes.insert(op_id, (dtype, MemLayout::Scalar));
                }
                Op::Load { src, index, layout } => {
                    dtypes.insert(op_id, (dtypes[&src].0, layout));
                    *rcs.entry(index).or_insert(0) += 1;
                }
                Op::Store { dst, src: x, index, layout } => {
                    debug_assert_eq!(dtypes[&x].1, layout);
                    dtypes.insert(op_id, dtypes[&x]);
                    *rcs.entry(dst).or_insert(0) += 1;
                    *rcs.entry(x).or_insert(0) += 1;
                    *rcs.entry(index).or_insert(0) += 1;
                }
                Op::Cast { x, dtype } => {
                    dtypes.insert(op_id, (dtype, dtypes[&x].1));
                    *rcs.entry(x).or_insert(0) += 1;
                }
                Op::Unary { x, .. } => {
                    dtypes.insert(op_id, dtypes[&x]);
                    *rcs.entry(x).or_insert(0) += 1;
                }
                Op::Binary { x, y, bop } => {
                    let dtype = if bop.returns_bool() {
                        (DType::Bool, dtypes[&x].1)
                    } else {
                        dtypes[&x]
                    };
                    dtypes.insert(op_id, dtype);
                    *rcs.entry(x).or_insert(0) += 1;
                    *rcs.entry(y).or_insert(0) += 1;
                }
                Op::Asm { ref ops, .. } => {
                    let dtype = dtypes[&ops[0]];
                    dtypes.insert(op_id, (dtype.0, MemLayout::Vector(ops.len().try_into().unwrap())));
                    for &x in ops.iter() {
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                }
                Op::Stack { ref ops } => {
                    let dtype = dtypes[&ops[0]];
                    dtypes.insert(op_id, (dtype.0, MemLayout::Vector(ops.len().try_into().unwrap())));
                    for &x in ops.iter() {
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                }
                Op::Devectorize { vec, idx: _ } => {
                    let dtype = dtypes[&vec];
                    dtypes.insert(op_id, (dtype.0, MemLayout::Scalar));
                    *rcs.entry(vec).or_insert(0) += 1;
                }
                Op::Wmma { dims: _, layout: _, dtype, a, b, c } => {
                    let out_dtype = match dtype {
                        MMADType::f16_f16_f16_f32 => DType::F32,
                    };
                    dtypes.insert(op_id, (out_dtype, MemLayout::Vector(4)));
                    *rcs.entry(a).or_insert(0) += 1;
                    *rcs.entry(b).or_insert(0) += 1;
                    *rcs.entry(c).or_insert(0) += 1;
                }
                Op::MatmulTile { x, y } => {
                    dtypes.insert(op_id, dtypes[&x]);
                    *rcs.entry(x).or_insert(0) += 1;
                    *rcs.entry(y).or_insert(0) += 1;
                }
                Op::TransposeTile { x } => {
                    dtypes.insert(op_id, dtypes[&x]);
                    *rcs.entry(x).or_insert(0) += 1;
                }
                Op::Mad { x, y, z } => {
                    dtypes.insert(op_id, dtypes[&x]);
                    *rcs.entry(x).or_insert(0) += 1;
                    *rcs.entry(y).or_insert(0) += 1;
                    *rcs.entry(z).or_insert(0) += 1;
                }
                Op::Index { .. } | Op::Loop { .. } => {
                    dtypes.insert(op_id, (IDX_T, MemLayout::Scalar));
                }
                Op::If { condition } => {
                    *rcs.entry(condition).or_insert(0) += 1;
                }
                Op::Barrier | Op::EndIf | Op::EndLoop => {}
            }
            op_id = self.next_op(op_id);
        }
        (dtypes, rcs)
    }

    /// Resolve the dtype of an operation's result by walking the IR.
    pub(crate) fn dtype(&self, mut op_id: OpId) -> DType {
        //println!("getting dtype of id: {op_id:?}'");
        for _ in 0..10000 {
            match self.ops[op_id].op {
                Op::Const(c) => return c.dtype(),
                Op::Param { dtype, .. } => return dtype,
                Op::Storage { dtype, .. } => return dtype,
                Op::Cast { dtype, .. } => return dtype,
                Op::Index { .. } => return IDX_T,
                Op::Load { src, .. } => op_id = src,
                Op::Unary { x, .. } => op_id = x,
                Op::Binary { x, .. } => op_id = x,
                Op::Mad { x, .. } => op_id = x,
                Op::Wmma { dtype, .. } => match dtype {
                    MMADType::f16_f16_f16_f32 => return DType::F32,
                },
                Op::MatmulTile { x, .. } => op_id = x,
                Op::TransposeTile { x } => op_id = x,
                Op::Stack { ref ops } => op_id = ops[0],
                Op::Asm { ref ops, .. } => op_id = ops[0],
                Op::Devectorize { vec, .. } => op_id = vec,
                Op::Store { src: x, .. } => op_id = x,
                Op::Move { x, .. } => op_id = x,
                Op::Reduce { x, .. } => op_id = x,
                Op::ReduceTile { x, .. } => op_id = x,
                Op::EndLoop | Op::Loop { .. } => return IDX_T,
                Op::Barrier | Op::If { .. } | Op::EndIf => todo!(),
            }
        }
        panic!("dtype not found for too long time");
    }

    #[track_caller]
    pub(crate) fn at(&self, op_id: OpId) -> &Op {
        &self.ops[op_id].op
    }

    pub(crate) fn prev_op(&self, op_id: OpId) -> OpId {
        self.ops[op_id].prev
    }

    pub(crate) fn next_op(&self, op_id: OpId) -> OpId {
        self.ops[op_id].next
    }

    /*pub fn ops_mut(&mut self) -> impl Iterator<Item = &mut Op> {
        self.ops.values_mut().map(|op_node| &mut op_node.op)
    }*/

    pub(crate) fn insert_before(&mut self, before_id: OpId, op: Op) -> OpId {
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

    pub(crate) fn insert_const_idx_before(&mut self, before_id: OpId, val: impl crate::scalar::Scalar) -> OpId {
        self.insert_before(before_id, Op::Const(Constant::idx(val)))
    }

    pub(crate) fn insert_after(&mut self, after_id: OpId, op: Op) -> OpId {
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

    pub(crate) fn move_op_after(&mut self, op_id: OpId, after_id: OpId) {
        debug_assert!(!op_id.is_null());
        debug_assert!(!after_id.is_null());
        debug_assert!(!self.ops.is_empty());

        if op_id == after_id {
            return;
        }

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

    /// Move an operation before another operation.
    ///
    /// Moves `op_id` to appear immediately before `before_id` in the operation chain.
    pub(crate) fn move_op_before(&mut self, op_id: OpId, before_id: OpId) {
        debug_assert!(!op_id.is_null());
        debug_assert!(!before_id.is_null());
        debug_assert!(!self.ops.is_empty());

        if op_id == before_id {
            return;
        }

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

    /// Remove an operation from the kernel.
    ///
    /// Removes the operation with `op_id` from the kernel IR.
    pub(crate) fn remove_op(&mut self, op_id: OpId) {
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

    /// Remove the transitive dependency chain of `x` that is not needed
    /// by any store or any op in `keep_alive`.
    /// Removes ops from `x` backwards that are no longer needed.
    /// Returns a filtered version of `loads` with entries for removed LoadViews removed.
    /// The i-th entry in `loads` corresponds to the i-th LoadView in op order.
    pub(crate) fn remove_unused_chain(
        &mut self,
        x: OpId,
        keep_alive: &[OpId],
        loads: &[crate::tensor::TensorId],
    ) -> Vec<crate::tensor::TensorId> {
        let mut chain: Set<OpId> = Set::default();
        let mut stack = vec![x];
        while let Some(op) = stack.pop() {
            if chain.insert(op) {
                stack.extend(self.ops[op].op.parameters().filter(|&p| !p.is_null()));
            }
        }

        let mut live: Set<OpId> = Set::default();
        stack.extend_from_slice(keep_alive);
        let mut op_id = self.head;
        while !op_id.is_null() {
            if matches!(self.ops[op_id].op, Op::Store { .. }) {
                stack.push(op_id);
            }
            op_id = self.next_op(op_id);
        }
        while let Some(op) = stack.pop() {
            if live.insert(op) {
                stack.extend(self.ops[op].op.parameters().filter(|&p| !p.is_null()));
            }
        }

        // Collect input (ro) Param OpIds in op order (before removal). These are
        // the Params that correspond 1:1 with `loads`: Global and Variable Params
        // are loads; GlobalMut Params are stores (mapped to `stores`, not `loads`).
        let param_ops: Vec<OpId> = {
            let mut ops = Vec::new();
            let mut id = self.head;
            while !id.is_null() {
                if let Op::Param { kind, .. } = &self.ops[id].op {
                    if *kind != ParamKind::GlobalMut {
                        ops.push(id);
                    }
                }
                id = self.next_op(id);
            }
            ops
        };

        let to_remove: Set<OpId> = chain.difference(&live).copied().collect();
        let mut op_id = self.head;
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if to_remove.contains(&op_id) {
                self.remove_op(op_id);
            }
            op_id = next;
        }

        // Keep only loads whose corresponding input Param was not removed.
        // `loads[i]` maps to the i-th Global/Variable Param in op order.
        param_ops.iter().enumerate().filter(|&(_, &lv_id)| !to_remove.contains(&lv_id)).map(|(i, _)| loads[i]).collect()
    }

    /// Iterate over all operations in the kernel.
    ///
    /// Returns an iterator over all operations without any ordering guarantees.
    pub(crate) fn iter_unordered(&self) -> impl Iterator<Item = (OpId, &Op)> {
        self.ops.iter().map(|(id, node)| (id, &node.op))
    }

    pub(crate) fn name(&self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            match self.at(op_id) {
                Op::Unary { uop, .. } => parts.push(match uop {
                    UOp::Neg => "neg",
                    UOp::BitNot => "bitnot",
                    UOp::Exp => "exp",
                    UOp::Exp2 => "exp2",
                    UOp::Ln => "ln",
                    UOp::Log2 => "log2",
                    UOp::Reciprocal => "reciprocal",
                    UOp::Sqrt => "sqrt",
                    UOp::Sin => "sin",
                    UOp::Cos => "cos",
                    UOp::Floor => "floor",
                    UOp::Trunc => "trunc",
                    UOp::Abs => "abs",
                }),
                Op::Binary { bop, .. } => parts.push(match bop {
                    BOp::Add => "add",
                    BOp::Sub => "sub",
                    BOp::Mul => "mul",
                    BOp::Div => "div",
                    BOp::Pow => "pow",
                    BOp::Mod => "mod",
                    BOp::Cmplt => "cmplt",
                    BOp::Cmpgt => "cmpgt",
                    BOp::Cmpge => "cmpge",
                    BOp::Max => "max",
                    BOp::Or => "or",
                    BOp::And => "and",
                    BOp::BitXor => "bitxor",
                    BOp::BitOr => "bitor",
                    BOp::BitAnd => "bitand",
                    BOp::BitShiftLeft => "shl",
                    BOp::BitShiftRight => "shr",
                    BOp::NotEq => "neq",
                    BOp::Eq => "eq",
                }),
                Op::Reduce { rop, .. } => parts.push(match rop {
                    BOp::Add => "sum",
                    BOp::Max => "max",
                    BOp::Mul => "prod",
                    _ => "reduce",
                }),
                Op::ReduceTile { rop, .. } => parts.push(match rop {
                    BOp::Add => "reduce_tile_sum",
                    BOp::Max => "reduce_tile_max",
                    BOp::Mul => "reduce_tile_prod",
                    _ => "reduce_tile",
                }),
                Op::Mad { .. } => parts.push("mad"),
                Op::Wmma { .. } => parts.push("wmma"),
                Op::Cast { .. } => parts.push("cast"),
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        parts.dedup();
        if parts.is_empty() {
            return "copy".into();
        }
        parts.join("_")
    }

    /// Compute flop and memory statistics for the kernel.
    ///
    /// Returns estimated flops, memory reads, and memory writes.
    pub(crate) fn flop_mem_rw(&self) -> (u64, u64, u64) {
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
                // TODO implement
                Op::Param { .. } => Info { shape: vec![1], flops: 0, mem_read: 0, mem_write: 0 },
                // TODO implement
                &Op::Load { .. } => Info { shape: vec![1], flops: 0, mem_read: 1, mem_write: 0 },
                // TODO implement
                &Op::Store { .. } => Info { shape: vec![1], flops: 0, mem_read: 0, mem_write: 1 },
                Op::Const(_) => Info { shape: vec![1], flops: 0, mem_read: 0, mem_write: 0 },
                Op::Move { x, mop } => match mop.as_ref() {
                    MoveOp::Reshape { shape, .. } => Info { shape: vec![1], flops: 0, mem_read: 0, mem_write: 0 },
                    MoveOp::Permute { .. } | MoveOp::Pad { .. } => todo!(),
                    MoveOp::Expand { .. } => Info { shape: todo!(), flops: 0, mem_read: 0, mem_write: 0 },
                    MoveOp::Flip { .. } => Info { shape: stack[x].shape.clone(), flops: 0, mem_read: 0, mem_write: 0 },
                    MoveOp::Narrow { axis, len, .. } => {
                        todo!()
                    }
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
                Op::ReduceTile { x, .. } => {
                    let Info { shape, .. } = stack[x].clone();
                    let numel: Dim = shape.iter().product();
                    Info { shape: vec![1], flops: numel - 1, mem_read: 0, mem_write: 0 }
                }
                Op::MatmulTile { x, .. } => {
                    let Info { shape, .. } = stack[x].clone();
                    let flops = shape.iter().product::<Dim>();
                    Info { shape, flops, mem_read: 0, mem_write: 0 }
                }
                Op::TransposeTile { x } => {
                    let Info { shape, .. } = stack[x].clone();
                    Info { shape, flops: 0, mem_read: 0, mem_write: 0 }
                }
                Op::Cast { x, .. } => {
                    let Info { shape, .. } = stack[x].clone();
                    let flops = 0; // Cast is not computation
                    Info { shape, flops, mem_read: 0, mem_write: 0 }
                }
                Op::Unary { x, .. } => {
                    let Info { shape, .. } = stack[x].clone();
                    let flops = shape.iter().product::<Dim>();
                    Info { shape, flops, mem_read: 0, mem_write: 0 }
                }
                Op::Binary { x, .. } => {
                    let Info { shape, .. } = stack[x].clone();
                    let flops = shape.iter().product::<Dim>();
                    Info { shape, flops, mem_read: 0, mem_write: 0 }
                }
                &Op::Storage { len, .. } => {
                    let shape: Vec<Dim> = vec![len];
                    Info { shape, flops: 0, mem_read: 0, mem_write: 0 }
                }
                Op::Wmma { .. }
                | Op::Asm { .. }
                | Op::Stack { .. }
                | Op::Devectorize { .. }
                | Op::If { .. }
                | Op::EndIf
                | Op::Barrier
                | Op::Mad { .. }
                | Op::Index { .. }
                | Op::Loop { .. }
                | Op::EndLoop => todo!(),
            };
            stack.insert(op_id, info);
            op_id = self.next_op(op_id);
        }

        stack.into_values().fold((0, 0, 0), |acc, info| (acc.0 + info.flops, acc.1 + info.mem_read, acc.2 + info.mem_write))
    }

    /// Check if the kernel contains any store operations.
    pub(crate) fn contains_stores(&self) -> bool {
        self.ops.values().any(|x| matches!(x.op, Op::Store { .. }))
    }

    /// Check if the kernel is a reduction kernel.
    pub(crate) fn is_reduce(&self) -> bool {
        self.ops.values().any(|x| matches!(x.op, Op::Reduce { .. } | Op::ReduceTile { .. }))
    }

    /// Shape of the value produced by `op_id`. Mirrors [`Self::dtype`]: value
    /// ops resolve through their input, while a `Const` yields a single
    /// dimension, a `Stack` yields one dimension per element (dynamic dims are
    /// `0`), and movement ops apply their transformation to the input shape.
    pub(crate) fn shape(&self, mut op_id: OpId) -> Vec<Dim> {
        let const_dim = |id: OpId| -> Option<Dim> {
            match self.ops[id].op {
                Op::Const(c) => c.as_dim(),
                _ => None,
            }
        };
        for _ in 0..10000 {
            match self.ops[op_id].op {
                Op::Const(c) => return c.as_dim().map(|d| vec![d]).unwrap_or_else(|| vec![0]),
                Op::Param { shape, .. } => return self.shape(shape),
                Op::Stack { ref ops } => {
                    return ops
                        .iter()
                        .map(|&o| match self.ops[o].op {
                            Op::Const(c) => c.as_dim().unwrap_or(0),
                            _ => 0,
                        })
                        .collect();
                }
                Op::Move { x, ref mop } => match mop.as_ref() {
                    MoveOp::Reshape { shape, .. } | MoveOp::Expand { shape } => return self.shape(*shape),
                    MoveOp::Permute { axes } => return crate::shape::permute(&self.shape(x), axes),
                    MoveOp::Flip { .. } => return self.shape(x),
                    MoveOp::Pad { axis, lp, rp } => {
                        let mut s = self.shape(x);
                        match (const_dim(*lp), const_dim(*rp)) {
                            (Some(l), Some(r)) => s[*axis] += l + r,
                            _ => s[*axis] = 0,
                        }
                        return s;
                    }
                    MoveOp::Narrow { axis, len, .. } => {
                        let mut s = self.shape(x);
                        s[*axis] = const_dim(*len).unwrap_or(0);
                        return s;
                    }
                },
                Op::Load { src: x, .. } | Op::Store { src: x, .. } => op_id = x,
                Op::Cast { x, .. }
                | Op::Unary { x, .. }
                | Op::Binary { x, .. }
                | Op::Mad { x, .. }
                | Op::MatmulTile { x, .. }
                | Op::ReduceTile { x, .. }
                | Op::Devectorize { vec: x, .. } => op_id = x,
                Op::Asm { ref ops, .. } => op_id = ops[0],
                Op::Reduce { x, .. } => op_id = x,
                Op::TransposeTile { x } => op_id = x,
                ref op => todo!("shape of {op:?}"),
            }
        }
        panic!("shape not found for too long time");
    }

    /// Builds a single shape op for the value produced by `op_id`, used to size
    /// a store/param buffer. Returns a `Stack` over per-dimension ops, or a bare
    /// const for a rank-1 shape. Padded/narrowed dimensions synthesize new
    /// arithmetic ops; scalars resolve to rank-1 `[1]`.
    pub(crate) fn generate_store_shape(&mut self, op_id: OpId) -> OpId {
        /// Resolves a *shape* op (a `Stack`, a bare `Const` dim, or a `Param`'s
        /// shape) into its per-dimension op ids.
        fn shape_ids(kernel: &Kernel, mut op_id: OpId) -> Vec<OpId> {
            for _ in 0..10000 {
                match kernel.ops[op_id].op {
                    Op::Const(_) => return vec![op_id],
                    Op::Stack { ref ops } => return ops.to_vec(),
                    Op::Param { shape, .. } => op_id = shape,
                    ref op => todo!("shape_ids of {op:?}"),
                }
            }
            panic!("shape_ids not found for too long time");
        }

        /// Resolves the shape of a *value* op into per-dimension ops, following
        /// compute and movement ops and emitting arithmetic for padded dims.
        fn store_shape_ids(kernel: &mut Kernel, mut op_id: OpId) -> Vec<OpId> {
            for _ in 0..10000 {
                match kernel.ops[op_id].op.clone() {
                    Op::Const(_) => return vec![],
                    Op::Param { shape, .. } => return shape_ids(kernel, shape),
                    Op::Stack { ref ops } => return vec![kernel.const_idx(ops.len() as u32)],
                    Op::Move { x, ref mop } => match mop.as_ref() {
                        MoveOp::Reshape { shape, .. } | MoveOp::Expand { shape } => return shape_ids(kernel, *shape),
                        MoveOp::Permute { axes } => return crate::shape::permute(&store_shape_ids(kernel, x), axes),
                        MoveOp::Flip { .. } => op_id = x,
                        MoveOp::Narrow { axis, len, .. } => {
                            let mut dims = store_shape_ids(kernel, x);
                            if dims.is_empty() {
                                dims.push(kernel.const_idx(1));
                            }
                            dims[*axis] = *len;
                            return dims;
                        }
                        MoveOp::Pad { axis, lp, rp } => {
                            let mut dims = store_shape_ids(kernel, x);
                            if dims.is_empty() {
                                dims.push(kernel.const_idx(1));
                            }
                            let orig = dims[*axis];
                            let sum = kernel.push_back(Op::Binary { x: orig, y: *lp, bop: BOp::Add });
                            dims[*axis] = kernel.push_back(Op::Binary { x: sum, y: *rp, bop: BOp::Add });
                            return dims;
                        }
                    },
                    Op::Load { src: x, .. }
                    | Op::Cast { x, .. }
                    | Op::Unary { x, .. }
                    | Op::Binary { x, .. }
                    | Op::Mad { x, .. }
                    | Op::MatmulTile { x, .. }
                    | Op::ReduceTile { x, .. }
                    | Op::Devectorize { vec: x, .. }
                    | Op::TransposeTile { x }
                    | Op::Store { src: x, .. } => op_id = x,
                    Op::Reduce { x, n_axes, .. } => {
                        let mut dims = store_shape_ids(kernel, x);
                        dims.truncate(dims.len().saturating_sub(n_axes));
                        return dims;
                    }
                    Op::Asm { ref ops, .. } => op_id = ops[0],
                    ref op => todo!("store shape of {op:?}"),
                }
            }
            panic!("store shape not found for too long time");
        }

        let dims = store_shape_ids(self, op_id);
        match dims.len() {
            0 => self.const_idx(1),
            1 => dims[0],
            _ => self.stack(&dims),
        }
    }

    /// Get index loop ids, dimensions and strides.
    ///
    /// Returns `loop_id` -> (dimension, stride) where NULL means unknown stride.
    pub(crate) fn get_strides(&self, index: OpId) -> Map<OpId, (Dim, Dim)> {
        //println!("Get index {index}");

        let mut params = vec![(index, 1u64)];
        let mut indices = Map::default();

        while let Some((param, scale)) = params.pop() {
            match self.ops[param].op {
                Op::Binary { x, y, bop } => {
                    if bop == BOp::Add {
                        if let Op::Loop { len, .. } = self.ops[x].op {
                            let d = self.loop_len_dim(len);
                            indices.insert(x, (d, 1));
                            params.push((y, scale));
                        } else if let Op::Index { len, .. } = self.ops[x].op {
                            indices.insert(x, (self.index_len(len), 1));
                            params.push((y, scale));
                        } else if let Op::Loop { len, .. } = self.ops[y].op {
                            let d = self.loop_len_dim(len);
                            indices.insert(y, (d, 1));
                            params.push((x, scale));
                        } else if let Op::Index { len, .. } = self.ops[y].op {
                            indices.insert(y, (self.index_len(len), 1));
                            params.push((x, scale));
                        } else {
                            params.push((x, scale));
                            params.push((y, scale));
                        }
                    }
                    if bop == BOp::Mul {
                        match (&self.ops[x].op, &self.ops[y].op) {
                            (Op::Loop { len, .. }, Op::Const(c)) => {
                                let d = self.loop_len_dim(*len);
                                indices.insert(x, (d, c.as_dim().unwrap() * scale));
                            }
                            (Op::Const(c), Op::Loop { len, .. }) => {
                                let d = self.loop_len_dim(*len);
                                indices.insert(y, (d, c.as_dim().unwrap() * scale));
                            }
                            (Op::Index { len, .. }, Op::Const(c)) => {
                                indices.insert(x, (self.index_len(*len), c.as_dim().unwrap() * scale));
                            }
                            (Op::Const(c), Op::Index { len, .. }) => {
                                indices.insert(y, (self.index_len(*len), c.as_dim().unwrap() * scale));
                            }
                            _ => {}
                        }
                    }
                    if bop == BOp::BitShiftLeft {
                        match (&self.ops[x].op, &self.ops[y].op) {
                            (Op::Loop { len, .. }, Op::Const(c)) => {
                                let d = self.loop_len_dim(*len);
                                indices.insert(x, (d, (1u64 << c.as_dim().unwrap()) * scale));
                            }
                            (Op::Index { len, .. }, Op::Const(c)) => {
                                indices.insert(x, (self.index_len(*len), (1u64 << c.as_dim().unwrap()) * scale));
                            }
                            (Op::Const(c), Op::Index { len, .. }) => {
                                indices.insert(y, (self.index_len(*len), (1u64 << c.as_dim().unwrap()) * scale));
                            }
                            (Op::Const(c), Op::Loop { len, .. }) => {
                                let d = self.loop_len_dim(*len);
                                indices.insert(y, (d, (1u64 << c.as_dim().unwrap()) * scale));
                            }
                            _ => {
                                if let Op::Const(c) = self.ops[y].op {
                                    params.push((x, scale * (1u64 << c.as_dim().unwrap())));
                                }
                            }
                        }
                    }
                }
                Op::Mad { x, y, z } => {
                    match &self.ops[z].op {
                        Op::Loop { len, .. } => {
                            indices.insert(z, (self.loop_len_dim(*len), 1));
                        }
                        Op::Index { len, .. } => {
                            indices.insert(z, (self.index_len(*len), 1));
                        }
                        _ => {
                            params.push((z, scale));
                        }
                    }
                    match (&self.ops[x].op, &self.ops[y].op) {
                        (Op::Loop { len, .. }, Op::Const(c)) => {
                            indices.insert(x, (self.loop_len_dim(*len), c.as_dim().unwrap() * scale));
                        }
                        (Op::Index { len, .. }, Op::Const(c)) => {
                            indices.insert(x, (self.index_len(*len), c.as_dim().unwrap() * scale));
                        }
                        (Op::Const(c), Op::Loop { len, .. }) => {
                            indices.insert(y, (self.loop_len_dim(*len), c.as_dim().unwrap() * scale));
                        }
                        (Op::Const(c), Op::Index { len, .. }) => {
                            indices.insert(y, (self.index_len(*len), c.as_dim().unwrap() * scale));
                        }
                        _ => {}
                    }
                }
                Op::Const(c) => {
                    indices
                        .entry(OpId::NULL)
                        .and_modify(|(_, v)| *v += c.as_dim().unwrap() * scale)
                        .or_insert((0, c.as_dim().unwrap() * scale));
                }
                _ => {}
            }
        }

        indices
    }

    /// Get the dimension value from a loop's length OpId.
    /// Returns 0 if the OpId doesn't point to a Const with a valid dimension.
    pub(crate) fn loop_len_dim(&self, loop_id: OpId) -> Dim {
        if let Op::Const(c) = &self.ops[loop_id].op {
            c.as_dim().unwrap_or(0)
        } else {
            0
        }
    }

    /// Resolve an `Op::Index` length operand (an `OpId`) to a concrete dim.
    ///
    /// The length is a kernel-shape value; for statically-shaped kernels it is a
    /// constant. Returns `0` when the length is not a resolvable constant, so
    /// dynamic workloads are handled by callers that track the op itself.
    pub(crate) fn index_len(&self, id: OpId) -> Dim {
        match &self.ops[id].op {
            Op::Const(c) => c.as_dim().unwrap_or(0),
            Op::Loop { len } => self.loop_len_dim(*len),
            Op::Index { len, .. } => self.index_len(*len),
            _ => 0,
        }
    }

    /// Remap slab indices from x to y
    fn remap(&mut self, x: OpId, y: OpId) {
        for op_node in self.ops.values_mut() {
            for param in op_node.op.parameters_mut() {
                if *param == x {
                    *param = y;
                }
            }
        }
    }

    /// Add an operation to the kernel.
    pub(crate) fn push_back(&mut self, op: Op) -> OpId {
        let op_node = OpNode { prev: self.tail, next: OpId::NULL, op };
        let op_id = self.ops.push(op_node);
        if self.head.is_null() {
            self.head = op_id;
        } else {
            self.ops[self.tail].next = op_id;
        }
        self.tail = op_id;
        op_id
    }

    /// Extract ops reachable from `root_op` into a new kernel.
    ///
    /// `all_outputs` contains all output OpIds in this kernel (including `root_op`).
    /// `loads` — parallel to LoadView ops in linked-list order — is split into
    /// `self_loads` and `new_loads` based on which kernel retains each LoadView.
    /// The new kernel contains only ops that `root_op` transitively depends on.
    /// Removes from `self` ops that are only needed by `root_op` and no other output.
    pub(crate) fn extract_subkernel<T: Copy>(
        &mut self,
        root_op: OpId,
        all_outputs: &[OpId],
        loads: &[T],
    ) -> (Self, OpId, Vec<T>, Vec<T>) {
        // Walk 1: from root_op
        let mut root_required = Set::default();
        let mut stack = vec![root_op];
        while let Some(op) = stack.pop() {
            if root_required.insert(op) {
                stack.extend(self.at(op).parameters());
            }
        }

        // Walk 2: from other outputs
        let mut other_required = Set::default();
        let mut stack = Vec::new();
        for &out in all_outputs {
            stack.push(out);
        }
        while let Some(op) = stack.pop() {
            if other_required.insert(op) {
                stack.extend(self.at(op).parameters());
            }
        }

        // Partition loads: for each kernel Param (Global/Variable), dispatch to
        // the set(s) that keep it. `loads` is parallel to the Param ops. Kernels
        // before linearize contain only Params — Storage (accumulators etc.) is
        // resolved later, so none may appear here.
        let mut self_loads: Vec<T> = Vec::new();
        let mut new_loads: Vec<T> = Vec::new();
        let mut load_idx = 0;
        let mut oid = self.head;
        while !oid.is_null() {
            match self.at(oid) {
                // Global and Variable Params are loads; GlobalMut is a store and
                // is not part of the loads list.
                Op::Param { kind: ParamKind::Global | ParamKind::Variable, .. } => {
                    if other_required.contains(&oid) {
                        self_loads.push(loads[load_idx]);
                    }
                    if root_required.contains(&oid) {
                        new_loads.push(loads[load_idx]);
                    }
                    load_idx += 1;
                }
                Op::Param { kind: ParamKind::GlobalMut, .. } => {}
                Op::Storage { .. } => {
                    panic!("extract_subkernel: unexpected Op::Storage (pre-linearize kernels contain only Params)")
                }
                _ => {}
            }
            oid = self.next_op(oid);
        }

        // Build new kernel by cloning root's ops (in topo order) with remapped OpIds
        let mut new_kernel = Kernel::new(self.device_id);
        let mut remap: Map<OpId, OpId> =
            Map::with_capacity_and_hasher(root_required.len(), core::hash::BuildHasherDefault::default());
        let mut new_root_op = OpId::NULL;
        let mut old_id = self.head;
        while !old_id.is_null() {
            if root_required.contains(&old_id) {
                let mut op = self.at(old_id).clone();
                op.remap_params(&remap);
                let new_id = new_kernel.push_back(op);
                if old_id == root_op {
                    new_root_op = new_id;
                }
                remap.insert(old_id, new_id);
            }
            old_id = self.next_op(old_id);
        }

        // Remove from self ops not needed by other outputs
        let mut old_id = self.head;
        while !old_id.is_null() {
            let next = self.next_op(old_id);
            if !other_required.contains(&old_id) {
                self.remove_op(old_id);
            }
            old_id = next;
        }

        (new_kernel, new_root_op, self_loads, new_loads)
    }

    /// Get all group indices used in the kernel.
    pub(crate) fn get_group_indices(&self) -> std::collections::BTreeMap<u32, OpId> {
        let mut indices = std::collections::BTreeMap::new();
        for (op_id, op_node) in self.ops.iter() {
            if let Op::Index { axis, kind: IdxKind::Group, .. } = op_node.op {
                indices.insert(axis, op_id);
            }
        }
        indices
    }

    /// Renumber indices to be in order.
    pub(crate) fn renumber_indices(&mut self) {
        let mut group_indices = BTreeMap::default();
        let mut local_indices = BTreeMap::default();
        for (op_id, op_node) in self.ops.iter() {
            match op_node.op {
                Op::Index { axis, kind: IdxKind::Group, .. } => group_indices.insert(axis, op_id),
                Op::Index { axis, kind: IdxKind::Local, .. } => local_indices.insert(axis, op_id),
                _ => None,
            };
        }
        let mut ax = 0;
        for &idx_id in group_indices.values() {
            let Op::Index { axis, kind: IdxKind::Group, .. } = &mut self.ops[idx_id].op else {
                unreachable!()
            };
            *axis = ax;
            ax += 1;
        }
        for &idx_id in local_indices.values() {
            let Op::Index { axis, kind: IdxKind::Local, .. } = &mut self.ops[idx_id].op else {
                unreachable!()
            };
            *axis = ax;
            ax += 1;
        }
    }

    pub(crate) fn is_preceded_by_reduce(&self, x: OpId) -> bool {
        //if self.ops.values().filter(|node| matches!(node.op, Op::Reduce { .. })).count() > 1 { return true; }
        let mut params = vec![x];
        while let Some(param) = params.pop() {
            if let &Op::Reduce { x, .. } = self.at(param) {
                params = vec![x];
                break;
            }
            params.extend(self.ops[param].op.parameters());
        }
        //if params.is_empty() { return false; }
        //println!("Found reduce at {params:?}");
        // If there is a load (non constant reduce) or multiple reduces, return true
        let mut seen: Set<OpId> = Set::default();
        while let Some(param) = params.pop() {
            if !seen.insert(param) {
                continue;
            }
            if matches!(self.ops[param].op, Op::Storage { .. } | Op::Reduce { .. }) {
                return true;
            }
            params.extend(self.ops[param].op.parameters());
        }
        false
    }

    pub(crate) fn is_preceded_by_compute(&self, x: OpId) -> bool {
        let mut params = vec![x];
        let mut seen: Set<OpId> = Set::default();
        let (mut has_compute, mut has_storage) = (false, false);
        while let Some(param) = params.pop() {
            if !seen.insert(param) {
                continue;
            }
            match &self.ops[param].op {
                Op::Binary { .. } | Op::Unary { .. } | Op::Reduce { .. } => {
                    has_compute = true;
                    params.extend(self.ops[param].op.parameters());
                }
                Op::Storage { .. } => has_storage = true,
                Op::Const(_) => {}
                _ => params.extend(self.ops[param].op.parameters()),
            }
        }
        has_compute && has_storage
    }
}
