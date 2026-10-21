// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

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
//! use zyx::kernel::{Dev, Kernel, MMADType, MMADims, MMALayout, MemScope, ParamKind};
//! use zyx::DType;
//!
//! let (m, n, k) = (1024, 1024, 1024);
//! let mut kernel = Kernel::new(Dev::Auto);
//!
//! let a_shape = kernel.add_shape(&[m, k]);
//! let b_shape = kernel.add_shape(&[k, n]);
//! let c_shape = kernel.add_shape(&[m, n]);
//! let a_buf = kernel.param(DType::F16);
//! let b_buf = kernel.param(DType::F16);
//! let c_buf = kernel.param_mut(DType::F32);
//!
//! let glen_x = kernel.const_idx(m / 16);
//! let glen_y = kernel.const_idx(n / 8);
//! let gidx = kernel.group_range(0, glen_x);
//! let gidy = kernel.group_range(1, glen_y);
//! let wid = kernel.local_range(0, 32);
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
//! let acc = kernel.storage(DType::F32, MemScope::Register, 4);
//! let zf = kernel.const_val(0.0f32);
//! let zero_acc = kernel.stack(&[zf, zf, zf, zf]);
//! kernel.store_vector(acc, zero_acc, c0, 4);
//!
//! let k_div_8 = kernel.const_idx(k / 8);
//! kernel.loop_over(k_div_8, |kernel, k_loop| {
//! let k_off = kernel.mul(k_loop, c8);
//!
//! let a_base = kernel.mad(a_row, k_const, k_off);
//! let a_base = kernel.add(a_base, col_in_tile);
//! let a_load_0 = kernel.load(a_buf, a_base);
//! let a_base_p1 = kernel.add(a_base, c1);
//! let a_load_1 = kernel.load(a_buf, a_base_p1);
//! let a_base2 = kernel.mad(c8, k_const, a_base);
//! let a_load_2 = kernel.load(a_buf, a_base2);
//! let a_base2_p1 = kernel.add(a_base2, c1);
//! let a_load_3 = kernel.load(a_buf, a_base2_p1);
//! let a_frag = kernel.stack(&[a_load_0, a_load_1, a_load_2, a_load_3]);
//!
//! let b_row = kernel.add(k_off, col_in_tile);
//! let b_base = kernel.mad(b_row, n_const, b_col);
//! let b_load_0 = kernel.load(b_buf, b_base);
//! let b_base_n = kernel.add(b_base, n_const);
//! let b_load_1 = kernel.load(b_buf, b_base_n);
//! let b_frag = kernel.stack(&[b_load_0, b_load_1]);
//!
//! let acc_old = kernel.load_vector(acc, c0, 4);
//! let acc_new = kernel.wmma(
//!     MMADims::m16n8k8, MMALayout::row_col, MMADType::f16_f16_f16_f32,
//!     a_frag, b_frag, acc_old,
//! );
//! kernel.store_vector(acc, acc_new, c0, 4);
//! });
//!
//! let acc_final = kernel.load_vector(acc, c0, 4);
//! let [co, c1v, c2v, c3v] = kernel.unstack(acc_final);
//!
//! let c_col = kernel.add(tile_base_col, col_in_tile);
//! let c_base = kernel.mad(a_row, n_const, c_col);
//! kernel.store(c_buf, co, c_base);
//! let c_base_p1 = kernel.add(c_base, c1);
//! kernel.store(c_buf, c1v, c_base_p1);
//! let c_base2 = kernel.mad(c8, n_const, c_base);
//! kernel.store(c_buf, c2v, c_base2);
//! let c_base2_p1 = kernel.add(c_base2, c1);
//! kernel.store(c_buf, c3v, c_base2_p1);
//!
//! // kernel.compile()?;  // requires CUDA with tensor cores
//! ```

pub use crate::backend::{Dev, DeviceInfo};
pub use crate::error::BackendError;
pub use autotune::BeamSearch;
pub use custom::{Acc, CompiledKernel, LocalPartition, Partition};
pub use ops::{
    BOp, GPUOp, MMADType, MMADims, MMALayout, Op, OpId, PTXOp, ParamKind, SourceBlock, SpirvOp, TTCbConfig, TTOp, TTProgramDesc,
    TileDim,
};
pub(crate) use ops::{OpLinked, RangeKind, UOp};
pub use pat::{Bindings, CoverGroup, DtypeClass, PMatch, Pat, VExpr};

use crate::{
    DType, Map, Set,
    dtype::Constant,
    shape::{Dim, UAxis},
    slab::Slab,
};
use nanoserde::{DeBin, SerBin};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::{hash::BuildHasherDefault, hash::Hash};

mod algebraic;
pub mod autotune;
mod cost;
mod custom;
pub(crate) mod debug;
mod devectorize;
mod fold_constants;
mod fold_loops;
mod fuse;
mod instr_sched;
mod licm;
mod linearize;
mod local_reduce;
mod merge;
mod ops;
mod pad;
mod pat;
mod predict_cost;
mod split;
mod tenstorrent;
mod verify;

// TODO later make this dynamic u32 or u64 depending on max range
/// Type used for indexing into arrays within kernels.
pub(crate) const IDX_T: DType = DType::I64;

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
/// use zyx::kernel::{Kernel, Dev, ParamKind};
/// use zyx::DType;
///
/// let mut kernel = Kernel::new(Dev::Auto);
/// let n = 256;
/// let shape = kernel.add_shape(&[n]);
/// let len = kernel.const_idx(n);
/// let inp = kernel.param(DType::F32);
/// let gidx = kernel.group_range(0, len);
/// let loaded = kernel.load(inp, gidx);
/// let s = kernel.sin(loaded);
/// let c = kernel.cos(loaded);
/// let result = kernel.add(s, c);
/// let out = kernel.param_mut(DType::F32);
/// kernel.store(out, result, gidx);
/// ```
///
/// # Compile
///
/// Build a kernel using fused multiply-add and compile it:
///
/// ```
/// use zyx::kernel::{Kernel, Dev, ParamKind};
/// use zyx::{DType, Tensor, ZyxError};
///
/// let mut kernel = Kernel::new(Dev::Auto);
/// let n = 4;
/// let shape = kernel.add_shape(&[n]);
/// let inp = kernel.param(DType::F32);
/// let len = kernel.const_idx(n);
/// let gidx = kernel.group_range(0, len);
/// let loaded = kernel.load(inp, gidx);
/// let result = kernel.mad(loaded, loaded, loaded); // x*x + x
/// let out = kernel.param_mut(DType::F32);
/// kernel.store(out, result, gidx);
///
/// let compiled = kernel.compile()?;
/// let x = Tensor::from([1.0f32, 2.0, 3.0, 4.0]).to(compiled.device())?;
/// let result = compiled.forward(&[&x], vec![[n]])?;
/// let data: Vec<f32> = result.into_iter().next().unwrap().try_into()?;
/// assert_eq!(data, vec![2.0, 6.0, 12.0, 20.0]);
/// # Ok::<_, ZyxError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Kernel {
    /// Operation slab containing the kernel IR.
    pub(crate) ops: Slab<OpId, OpLinked>,
    /// Head of the operation linked list.
    pub head: OpId,
    /// Tail of the operation linked list.
    pub(crate) tail: OpId,
    /// Target device for compilation.
    pub(crate) dev: Dev,
    /// Snapshot of the device's [`DeviceInfo`] taken when a device is bound —
    /// the warp size and limits live here so kernel passes never need to lock
    /// the RT (it is not reentrant). Shared via `Arc` so kernel clones
    /// (autotune creates thousands) don't re-allocate. `None` only for
    /// placeholder kernels (`Dev::Auto`) before their device is bound.
    /// Hardware properties are fixed, so the snapshot cannot go stale.
    pub(crate) dev_info: Option<Arc<DeviceInfo>>,
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
}

impl PartialEq for Kernel {
    fn eq(&self, other: &Self) -> bool {
        self.ops == other.ops && self.head == other.head && self.dev == other.dev
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
        Ok(Self { head: start, tail: end, ops, device_id: Dev::Auto })
    }
}*/

impl Hash for Kernel {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.head.hash(state);
        self.ops.hash(state);
        self.dev.hash(state);
    }
}

// Custom kernel machinery
impl Kernel {
    /// The device info snapshot bound to this kernel.
    ///
    /// # Panics
    ///
    /// If the kernel has no device bound (placeholder kernels with
    /// `Dev::Auto` that were never compiled).
    pub(crate) fn dev_info(&self) -> &DeviceInfo {
        self.dev_info.as_ref().expect("kernel has no device bound (Dev::Auto placeholder)")
    }

    /// Compute dtypes and reference counts for all operations. The shape
    /// half of each entry is the value's [`Self::shape`]: `[]` for
    /// scalars, `[n]` for vectors, `[x, y]` for tiles.
    pub(crate) fn compute_dtypes_and_rcs(&self) -> (Map<OpId, (DType, Vec<Dim>)>, Map<OpId, u32>) {
        let mut rcs: Map<OpId, u32> = Map::with_capacity_and_hasher(self.ops.len().into(), BuildHasherDefault::new());
        let mut dtypes: Map<OpId, (DType, Vec<Dim>)> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());

        let mut op_id = self.head;
        for _ in 0..250_000 {
            if op_id.is_null() {
                break;
            }
            match self.ops[op_id].op {
                // Embedded source blobs and effect-only ops produce no
                // value: no dtype/shape entry. (Debug printing handles
                // these arms without shapes.)
                Op::Source(_)
                | Op::GPU(_)
                | Op::Spirv(_)
                | Op::PTX(_)
                | Op::After { .. }
                | Op::ToDevice { .. }
                | Op::Contiguous { .. }
                | Op::Program { .. }
                | Op::Kernel(_)
                | Op::TT(TTOp::Copy { .. }) => {}
                // Movement ops carry their output shape in their
                // descriptor operands: resolve each dim, `-1` where
                // symbolic (same convention as `Kernel::shape`).
                Op::Reshape { x, shape } | Op::Expand { x, shape } => {
                    let dims = match self.ops[shape].op {
                        Op::Stack { ref ops } => ops
                            .iter()
                            .map(|&d| self.resolve_const(d).and_then(crate::dtype::Constant::as_dim).unwrap_or(-1))
                            .collect(),
                        _ => {
                            vec![self.resolve_const(shape).and_then(crate::dtype::Constant::as_dim).unwrap_or(-1)]
                        }
                    };
                    let dtype = dtypes[&x].0;
                    dtypes.insert(op_id, (dtype, dims));
                    *rcs.entry(x).or_insert(0) += 1;
                }
                Op::Permute { x, ref axes } => {
                    let dtype = dtypes[&x].0;
                    let dims = crate::shape::permute(&dtypes[&x].1, axes);
                    dtypes.insert(op_id, (dtype, dims));
                    *rcs.entry(x).or_insert(0) += 1;
                }
                Op::Flip { x, .. } => {
                    dtypes.insert(op_id, dtypes[&x].clone());
                    *rcs.entry(x).or_insert(0) += 1;
                }
                Op::Narrow { x, axis, len, .. } | Op::Pad { x, axis, len, .. } => {
                    let mut dims = dtypes[&x].1.clone();
                    let resolved = self.resolve_const(len).and_then(crate::dtype::Constant::as_dim).unwrap_or(-1);
                    if dims.is_empty() {
                        dims = vec![resolved];
                    } else {
                        dims[axis as usize] = resolved;
                    }
                    let dtype = dtypes[&x].0;
                    dtypes.insert(op_id, (dtype, dims));
                    *rcs.entry(x).or_insert(0) += 1;
                }
                Op::Const(x) => {
                    dtypes.insert(op_id, (x.dtype(), Vec::new()));
                }
                Op::Param { dtype, .. } => {
                    dtypes.insert(op_id, (dtype, Vec::new()));
                }
                Op::Storage { dtype, .. } => {
                    dtypes.insert(op_id, (dtype, Vec::new()));
                }
                Op::Load { src, src_idx, .. } => {
                    // Dtype rides on the buffer; the value shape rides on
                    // the access itself (scalar on a null index, else the
                    // index expression's dims).
                    let dims = if src_idx.is_null() {
                        Vec::new()
                    } else {
                        dtypes[&src_idx].1.clone()
                    };
                    dtypes.insert(op_id, (dtypes[&src].0, dims));
                    // Load consumes both the buffer read and the position
                    // it reads at (index-and-load both counted before).
                    *rcs.entry(src).or_insert(0) += 1;
                    *rcs.entry(src_idx).or_insert(0) += 1;
                }
                Op::Store { dst, dst_idx, src: x } => {
                    // The access shape rides on the index arithmetic:
                    // whole-view store (null dst_idx) is scalar by
                    // form, an element store's shape is its index's
                    // lane shape. Implicit broadcast: the value may be
                    // smaller or same rank, never bigger rank into a
                    // smaller-rank access.
                    if !dst_idx.is_null() {
                        debug_assert!(dtypes[&x].1.len() <= dtypes[&dst_idx].1.len(), "store value rank exceeds access rank");
                    }
                    dtypes.insert(op_id, dtypes[&x].clone());
                    *rcs.entry(dst).or_insert(0) += 1;
                    *rcs.entry(dst_idx).or_insert(0) += 1;
                    *rcs.entry(x).or_insert(0) += 1;
                }
                Op::Copy { src, src_idx, dst, dst_idx } => {
                    // Dtype from the buffer, access shape from the
                    // source index expression.
                    dtypes.insert(op_id, (dtypes[&src].0, dtypes[&src_idx].1.clone()));
                    *rcs.entry(src).or_insert(0) += 1;
                    for &o in &[src_idx, dst, dst_idx] {
                        if !o.is_null() {
                            *rcs.entry(o).or_insert(0) += 1;
                        }
                    }
                }
                Op::Cast { x, dtype } => {
                    dtypes.insert(op_id, (dtype, dtypes[&x].1.clone()));
                    *rcs.entry(x).or_insert(0) += 1;
                }
                Op::Bitcast { x, dtype } => {
                    dtypes.insert(op_id, (dtype, dtypes[&x].1.clone()));
                    *rcs.entry(x).or_insert(0) += 1;
                }
                Op::Unary { x, .. } => {
                    dtypes.insert(op_id, dtypes[&x].clone());
                    *rcs.entry(x).or_insert(0) += 1;
                }
                Op::Reduce { x, reduce_axis, .. } => {
                    // Zero the reduced lane's slot when the axis is an
                    // `Op::Unroll`, otherwise drop the last dim.
                    let (dtype, mut dims) = dtypes[&x].clone();
                    match (!reduce_axis.is_null()).then(|| &self.ops[reduce_axis].op) {
                        Some(Op::Unroll { axis, .. }) => {
                            dims[*axis as usize] = 0;
                        }
                        _ => {
                            dims.pop();
                        }
                    }
                    while dims.last() == Some(&0) {
                        dims.pop();
                    }
                    dtypes.insert(op_id, (dtype, dims));
                    *rcs.entry(x).or_insert(0) += 1;
                    *rcs.entry(reduce_axis).or_insert(0) += 1;
                }
                Op::TT(TTOp::ReserveBack { cb, .. })
                | Op::TT(TTOp::PushBack { cb, .. })
                | Op::TT(TTOp::WaitFront { cb, .. })
                | Op::TT(TTOp::PopFront { cb, .. }) => {
                    // Effect-only CB sync: associated with the buffer's
                    // dtype/layout, keeps the CB's refcount (mirrors Copy
                    // minus the dst side — one buffer, not two).
                    dtypes.insert(op_id, dtypes[&cb].clone());
                    *rcs.entry(cb).or_insert(0) += 1;
                }
                Op::Binary { x, y, bop } => {
                    // Implicit broadcast, per slot: equal slots agree; a
                    // -1 (symbolic), 0 (unset) or 1 (scalar) side takes
                    // the max; two differing concrete dims panic. Same
                    // rule as `Kernel::shape`, trailing unsets trimmed
                    // so ranks stay exact.
                    let dtype = if bop.returns_bool() { DType::Bool } else { dtypes[&x].0 };
                    let (xd, yd) = (dtypes[&x].1.clone(), dtypes[&y].1.clone());
                    let rank = xd.len().max(yd.len());
                    let mut dims = Vec::with_capacity(rank);
                    for i in 0..rank {
                        let (a, b) = (xd.get(i).copied().unwrap_or(0), yd.get(i).copied().unwrap_or(0));
                        dims.push(if a == b {
                            a
                        } else if a == -1 || a == 0 || a == 1 || b == -1 || b == 0 || b == 1 {
                            a.max(b)
                        } else {
                            panic!("binary with unequal dims {a} vs {b}");
                        });
                    }
                    while dims.last() == Some(&0) {
                        dims.pop();
                    }
                    dtypes.insert(op_id, (dtype, dims));
                    *rcs.entry(x).or_insert(0) += 1;
                    *rcs.entry(y).or_insert(0) += 1;
                }
                Op::Asm { ref ops, .. } | Op::TT(TTOp::LLK { ref ops, .. }) => {
                    // Result takes ops[0]'s dtype/layout; the template's
                    // `{i}` placeholders substitute the operand expressions.
                    // Operand-free Asm/LLK is an effect-only call (e.g. a
                    // TT LLK init): produces no value, takes no entry.
                    if !ops.is_empty() {
                        let dtype = dtypes[&ops[0]].clone();
                        dtypes.insert(op_id, dtype);
                        for &x in ops.iter() {
                            *rcs.entry(x).or_insert(0) += 1;
                        }
                    }
                }
                Op::TT(TTOp::ReduceTile { cb_in, cb_sc, slot, kind, .. }) => {
                    // Lowered reduce: result takes the input CB's dtype; the
                    // shape drops the trailing dim (Row/Col) or everything
                    // (Scalar), like `Kernel::shape`.
                    let (dtype, _) = dtypes[&cb_in].clone();
                    let shape = if matches!(kind, TileDim::Scalar) {
                        Vec::new()
                    } else {
                        vec![32]
                    };
                    dtypes.insert(op_id, (dtype, shape));
                    for &f in &[cb_in, cb_sc, slot] {
                        *rcs.entry(f).or_insert(0) += 1;
                    }
                }
                Op::TT(TTOp::FillTile { slot, val }) => {
                    // Lowered fill: scalar value broadcast to one tile.
                    dtypes.insert(op_id, (dtypes[&val].0, vec![32, 32]));
                    for &f in &[slot, val] {
                        *rcs.entry(f).or_insert(0) += 1;
                    }
                }
                Op::TT(TTOp::LLKBcast { cb_a, cb_b, mx, plain, .. }) => {
                    // Lowered fused broadcast: same rule over cb_a.
                    dtypes.insert(op_id, dtypes[&cb_a].clone());
                    for &f in &[cb_a, cb_b, mx, plain] {
                        *rcs.entry(f).or_insert(0) += 1;
                    }
                }
                Op::TT(TTOp::MatmulTiles { cb_a, cb_b, slot }) => {
                    // Lowered matmul: one tile out; result takes the
                    // left CB's dtype.
                    let (dtype, _) = dtypes[&cb_a].clone();
                    dtypes.insert(op_id, (dtype, vec![32, 32]));
                    for &f in &[cb_a, cb_b, slot] {
                        *rcs.entry(f).or_insert(0) += 1;
                    }
                }
                Op::Stack { ref ops } => {
                    let dtype = dtypes[&ops[0]].0;
                    dtypes.insert(op_id, (dtype, vec![ops.len() as Dim]));
                    for &x in ops.iter() {
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                }
                Op::Unroll { axis, len } => {
                    // Lane dim sits in its axis slot, the rest unset.
                    let mut dims = vec![0; axis as usize + 1];
                    dims[axis as usize] = len as Dim;
                    dtypes.insert(op_id, (IDX_T, dims));
                }
                Op::Index { vec, idx: _ } => {
                    let dtype = dtypes[&vec].0;
                    let mut shape = dtypes[&vec].1.clone();
                    debug_assert!(!shape.is_empty(), "compute_dtypes: Index of scalar");
                    if !shape.is_empty() {
                        shape.remove(0);
                    }
                    dtypes.insert(op_id, (dtype, shape));
                    *rcs.entry(vec).or_insert(0) += 1;
                }
                Op::Wmma { dims: _, layout: _, dtype, a, b, c } => {
                    let out_dtype = match dtype {
                        MMADType::f16_f16_f16_f32 => DType::F32,
                        MMADType::f16_f16_f16_f16 => DType::F16,
                        MMADType::s8_s8_s32_s32
                        | MMADType::s4_s4_s32_s32
                        | MMADType::b1_b1_s32_xor_popc
                        | MMADType::b1_b1_s32_and_popc => DType::I32,
                    };
                    dtypes.insert(op_id, (out_dtype, vec![4]));
                    *rcs.entry(a).or_insert(0) += 1;
                    *rcs.entry(b).or_insert(0) += 1;
                    *rcs.entry(c).or_insert(0) += 1;
                }
                Op::Range { kind, .. } => {
                    // Group length is a consumed operand (e.g. a Binary
                    // expression) and needs a refcount for codegen register
                    // allocation. Same for the local id a Warp view reads.
                    if let RangeKind::Group(len) = kind {
                        *rcs.entry(len).or_insert(0) += 1;
                    }
                    if let RangeKind::Warp(local_id) = kind {
                        *rcs.entry(local_id).or_insert(0) += 1;
                    }
                    dtypes.insert(op_id, (IDX_T, Vec::new()));
                }
                Op::Loop { len, .. } => {
                    // Loop length is a consumed operand (e.g. a Binary
                    // expression) and needs a refcount for codegen register
                    // allocation.
                    *rcs.entry(len).or_insert(0) += 1;
                    dtypes.insert(op_id, (IDX_T, Vec::new()));
                }
                Op::Barrier
                | Op::EndLoop
                | Op::TT(TTOp::MathLock)
                | Op::TT(TTOp::MathUnlock)
                | Op::TT(TTOp::PackLock)
                | Op::TT(TTOp::PackUnlock)
                | Op::TT(TTOp::NocReadBarrier)
                | Op::TT(TTOp::NocWriteBarrier)
                | Op::TT(TTOp::ReduceUninit)
                | Op::TT(TTOp::EndReader)
                | Op::TT(TTOp::EndCompute)
                | Op::TT(TTOp::ProgramDesc(_))
                | Op::TT(TTOp::TensixGrid(_)) => {}
            }
            op_id = self.next_op(op_id);
        }
        if !op_id.is_null() {
            panic!("compute_dtypes_and_rcs did not finish in 250000 steps");
        }
        for (_, v) in dtypes.iter_mut() {
            v.1.retain(|&d| d != 0);
        }
        (dtypes, rcs)
    }

    /// Resolve the dtype of an operation's result by walking the IR.
    pub(crate) fn dtype(&self, mut op_id: OpId) -> DType {
        //println!("getting dtype of id: {op_id:?}'");
        for _ in 0..250000 {
            match self.ops[op_id].op {
                Op::Source(_) => todo!(),
                Op::GPU(_) => todo!(),
                Op::Spirv(_) => todo!(),
                Op::PTX(_) => todo!(),
                Op::Const(c) => return c.dtype(),
                Op::Param { dtype, .. } => return dtype,
                Op::Storage { dtype, .. } => return dtype,
                Op::Cast { dtype, .. } => return dtype,
                Op::Bitcast { dtype, .. } => return dtype,
                Op::Range { .. } => return IDX_T,
                Op::Load { src, .. } => op_id = src,
                Op::Copy { src, .. } => op_id = src,
                Op::Unary { x, .. } => op_id = x,
                Op::Unroll { .. } => return IDX_T,
                Op::Binary { x, bop, .. } => {
                    if bop.returns_bool() {
                        return DType::Bool;
                    }
                    op_id = x;
                }
                Op::Wmma { dtype, .. } => match dtype {
                    MMADType::f16_f16_f16_f32 => return DType::F32,
                    MMADType::f16_f16_f16_f16 => return DType::F16,
                    MMADType::s8_s8_s32_s32
                    | MMADType::s4_s4_s32_s32
                    | MMADType::b1_b1_s32_xor_popc
                    | MMADType::b1_b1_s32_and_popc => return DType::I32,
                },
                Op::Stack { ref ops } => op_id = ops[0],
                Op::Asm { ref ops, .. } | Op::TT(TTOp::LLK { ref ops, .. }) => {
                    if ops.is_empty() {
                        todo!("dtype: operand-free Asm/LLK produces no value")
                    }
                    op_id = ops[0]
                }
                Op::TT(TTOp::ReduceTile { cb_in, .. }) => op_id = cb_in,
                Op::TT(TTOp::FillTile { val, .. }) => op_id = val,
                Op::TT(TTOp::LLKBcast { cb_a, .. }) => op_id = cb_a,
                Op::TT(TTOp::MatmulTiles { cb_a, .. }) => op_id = cb_a,
                Op::Index { vec, .. } => op_id = vec,
                Op::Store { src: x, .. } => op_id = x,
                Op::Reduce { x, .. } => op_id = x,
                Op::EndLoop | Op::Loop { .. } => return IDX_T,
                Op::Barrier
                | Op::TT(TTOp::MathLock)
                | Op::TT(TTOp::MathUnlock)
                | Op::TT(TTOp::PackLock)
                | Op::TT(TTOp::PackUnlock)
                | Op::TT(TTOp::NocReadBarrier)
                | Op::TT(TTOp::NocWriteBarrier)
                | Op::TT(TTOp::ReduceUninit)
                | Op::TT(TTOp::EndReader)
                | Op::TT(TTOp::EndCompute)
                | Op::TT(TTOp::ProgramDesc(_))
                | Op::TT(TTOp::TensixGrid(_)) => todo!(),
                Op::TT(TTOp::ReserveBack { .. })
                | Op::TT(TTOp::PushBack { .. })
                | Op::TT(TTOp::WaitFront { .. })
                | Op::TT(TTOp::PopFront { .. })
                | Op::TT(TTOp::Copy { .. }) => todo!("dtype: CB sync/traffic op produces no value"),
                Op::After { .. } | Op::ToDevice { .. } | Op::Contiguous { .. } | Op::Program { .. } | Op::Kernel(_) => {
                    todo!()
                }
                Op::Reshape { x, .. }
                | Op::Expand { x, .. }
                | Op::Permute { x, .. }
                | Op::Flip { x, .. }
                | Op::Pad { x, .. }
                | Op::Narrow { x, .. } => op_id = x,
            }
        }
        panic!("dtype not found for too long time");
    }

    /// The op behind an id. The core read end of the op linked list:
    /// external passes walk `head` → `at` → `next_op`.
    ///
    /// # Panics
    ///
    /// Panics if the id is null or out of bounds.
    #[track_caller]
    pub fn at(&self, op_id: OpId) -> &Op {
        &self.ops[op_id].op
    }

    pub(crate) fn prev_op(&self, op_id: OpId) -> OpId {
        self.ops[op_id].prev
    }

    /// The op after `op_id` in head order (`OpId::NULL` at the tail).
    /// External passes walk `head` → `at` → `next_op`.
    pub fn next_op(&self, op_id: OpId) -> OpId {
        self.ops[op_id].next
    }

    /// The ops in head-to-tail emission order (the order `push_back` emitted them).
    /// Backends decode rendered kernels positionally with this instead of
    /// chaining `head` → `next_op` by hand.
    pub fn ops_in_order(&self) -> impl Iterator<Item = &Op> {
        let mut next = self.head;
        std::iter::from_fn(move || {
            if next.is_null() {
                return None;
            }
            let node = &self.ops[next];
            next = node.next;
            Some(&node.op)
        })
    }

    /*pub fn ops_mut(&mut self) -> impl Iterator<Item = &mut Op> {
        self.ops.values_mut().map(|op_node| &mut op_node.op)
    }*/

    pub(crate) fn insert_before(&mut self, before_id: OpId, op: Op) -> OpId {
        debug_assert!(!before_id.is_null());
        debug_assert!(!self.ops.is_empty());

        let prev = self.ops[before_id].prev;
        let op_node = OpLinked { prev, next: before_id, op };
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
        let op_node = OpLinked { prev: after_id, next, op };
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
        let OpLinked { prev, next, .. } = self.ops[op_id];
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
        let OpLinked { prev, next, .. } = self.ops[op_id];
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

        let OpLinked { prev, next, .. } = self.ops[op_id];
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
    /// Generic over the load payload (`TensorId` in runtime kernels,
    /// `ClassId` in kernelizer jit kernels) — it is never inspected.
    pub(crate) fn remove_unused_chain<T: Copy>(&mut self, x: OpId, keep_alive: &[OpId], loads: &[T]) -> Vec<T> {
        let mut chain: Set<OpId> = Set::default();
        let mut stack = vec![x];
        for _ in 0..30_000 {
            let Some(op) = stack.pop() else { break };
            if chain.insert(op) {
                stack.extend(self.ops[op].op.parameters().filter(|&p| !p.is_null()));
            }
        }
        if !stack.is_empty() {
            panic!("remove_unused_chain did not finish in 250000 steps");
        }

        let mut live: Set<OpId> = Set::default();
        stack.extend_from_slice(keep_alive);
        let mut op_id = self.head;
        for _ in 0..30_000 {
            if op_id.is_null() {
                break;
            }
            if matches!(self.ops[op_id].op, Op::Store { .. }) {
                stack.push(op_id);
            }
            op_id = self.next_op(op_id);
        }
        if !op_id.is_null() {
            panic!("remove_unused_chain did not finish in 250000 steps");
        }
        for _ in 0..30_000 {
            let Some(op) = stack.pop() else { break };
            if live.insert(op) {
                stack.extend(self.ops[op].op.parameters().filter(|&p| !p.is_null()));
            }
        }
        if !stack.is_empty() {
            panic!("remove_unused_chain did not finish in 250000 steps");
        }

        // Collect input (ro) Param OpIds in op order (before removal). These are
        // the Params that correspond 1:1 with `loads`: Global and Variable Params
        // are loads; GlobalMut Params are stores (mapped to `stores`, not `loads`).
        let param_ops: Vec<OpId> = {
            let mut ops = Vec::new();
            let mut id = self.head;
            for _ in 0..30_000 {
                if id.is_null() {
                    break;
                }
                if let Op::Param { kind, .. } = &self.ops[id].op {
                    if *kind != ParamKind::GlobalMut {
                        ops.push(id);
                    }
                }
                id = self.next_op(id);
            }
            if !id.is_null() {
                panic!("remove_unused_chain did not finish in 250000 steps");
            }
            ops
        };

        let to_remove: Set<OpId> = chain.difference(&live).copied().collect();
        let mut op_id = self.head;
        for _ in 0..30_000 {
            if op_id.is_null() {
                break;
            }
            let next = self.next_op(op_id);
            if to_remove.contains(&op_id) {
                self.remove_op(op_id);
            }
            op_id = next;
        }
        if !op_id.is_null() {
            panic!("remove_unused_chain did not finish in 250000 steps");
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
        for _ in 0..250_000 {
            if op_id.is_null() {
                break;
            }
            match self.at(op_id) {
                Op::Unary { uop, .. } => parts.push(match uop {
                    UOp::Neg => "neg",
                    UOp::Not => "not",
                    UOp::BitNot => "bitnot",
                    UOp::Exp2 => "exp2",
                    UOp::Log2 => "log2",
                    UOp::Reciprocal => "reciprocal",
                    UOp::Sqrt => "sqrt",
                    UOp::Rsqrt => "rsqrt",
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
                Op::Wmma { .. } => parts.push("wmma"),
                Op::Cast { .. } => parts.push("cast"),
                Op::Bitcast { .. } => parts.push("bitcast"),
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        if !op_id.is_null() {
            panic!("name did not finish in 250000 steps");
        }
        parts.dedup();
        if parts.is_empty() {
            return "copy".into();
        }
        parts.join("_")
    }

    /// Check if the kernel contains any store operations.
    pub(crate) fn contains_stores(&self) -> bool {
        self.ops.values().any(|x| matches!(x.op, Op::Store { .. }))
    }

    /// Estimated flops, global bytes read, global bytes written for a launch
    /// of this kernel. Iteration counts from enclosing `Loop`/`Range`
    /// (Groups/Locals) multiply per-op costs; variable trip counts that
    /// cannot be resolved at compile time are treated as 1.
    pub fn flop_mem_rw(&self) -> (u64, u64, u64) {
        let mut flops: u64 = 0;
        let mut read: u64 = 0;
        let mut write: u64 = 0;
        let mut loop_stack: Vec<u64> = Vec::new();
        let mut group_stack: Vec<u64> = Vec::new();
        let mut local_prod: u64 = 1;

        let mut op_id = self.head;
        for _ in 0..20_000 {
            if op_id.is_null() {
                break;
            }
            // Multiplier for this op: all enclosing loops * all parallel threads.
            let loop_prod: u64 = loop_stack.iter().product::<u64>().max(1);
            let group_prod: u64 = group_stack.iter().product::<u64>().max(1);
            let mult: u64 = loop_prod.saturating_mul(group_prod).saturating_mul(local_prod);

            match self.at(op_id) {
                Op::Loop { len } => {
                    let v = self.resolve_const(*len).and_then(|c| c.as_dim()).unwrap_or(1).max(1) as u64;
                    // The loop body executes v times; push for following ops.
                    // Count is applied via mult on body ops, so push now.
                    loop_stack.push(v);
                }
                Op::EndLoop => {
                    loop_stack.pop();
                }
                Op::Range { kind, .. } => match kind {
                    RangeKind::Group(l) => {
                        let v = self.resolve_const(*l).and_then(|c| c.as_dim()).unwrap_or(1).max(1) as u64;
                        group_stack.push(v);
                    }
                    RangeKind::Local(n) => {
                        local_prod = local_prod.saturating_mul(*n as u64);
                    }
                    RangeKind::Warp(_) => {}
                },
                Op::Unary { .. } => {
                    if self.dtype(op_id).is_float() {
                        flops = flops.saturating_add(mult)
                    }
                }
                Op::Binary { .. } => {
                    if self.dtype(op_id).is_float() {
                        flops = flops.saturating_add(mult)
                    }
                }
                Op::Wmma { dims, .. } => {
                    let (m, n, k) = dims.decompose_mnk();
                    // One wmma per warp, not per thread.
                    let ws = u64::from(self.dev_info().warp_size);
                    let warps = (group_prod * local_prod / ws.max(1)).max(1);
                    let loop_prod: u64 = loop_stack.iter().product::<u64>().max(1);
                    flops = flops.saturating_add(2 * m * n * k * warps * loop_prod);
                }
                Op::TT(TTOp::MatmulTiles { .. }) => {
                    // Lowered matmul: single 32x32x32 tile MAC.
                    flops = flops.saturating_add(2 * 32 * 32 * 32 * mult);
                }
                Op::Load { src, .. } => {
                    if let Op::Param { kind: ParamKind::Global, dtype, .. } = &self.ops[*src].op {
                        let n_elems: Dim = self
                            .shape(op_id)
                            .iter()
                            .map(|&d| {
                                assert!(d >= 0, "dram bytes: symbolic access width");
                                d
                            })
                            .product();
                        let n_bytes = (dtype.bit_size() as u64 / 8) * n_elems as u64;
                        read = read.saturating_add(n_bytes.saturating_mul(mult));
                    }
                }
                Op::Store { dst, .. } => {
                    if let Op::Param { kind: ParamKind::GlobalMut, dtype, .. } = &self.ops[*dst].op {
                        let n_elems: Dim = self
                            .shape(op_id)
                            .iter()
                            .map(|&d| {
                                assert!(d >= 0, "dram bytes: symbolic access width");
                                d
                            })
                            .product();
                        let n_bytes = (dtype.bit_size() as u64 / 8) * n_elems as u64;
                        write = write.saturating_add(n_bytes.saturating_mul(mult));
                    }
                }
                _ => {}
            }
            op_id = self.next_op(op_id);
        }
        (flops, read, write)
    }

    /// Check if the kernel is a reduction kernel.
    pub(crate) fn is_reduce(&self) -> bool {
        self.ops.values().any(|x| matches!(x.op, Op::Reduce { .. }))
    }

    /// Unpacks a shape descriptor (a `Stack` of dims or a bare dim op,
    /// as stored in `Param { shape }`, `Reshape { shape }`,
    /// `Expand { shape }`) into its dimension op ids.
    ///
    /// Each returned entry is the op computing that dim's length. A
    /// dynamic (runtime-known only) dim length is [`OpId::NULL`]: rank is
    /// always statically known, individual lengths may not be. An empty
    /// vec means a scalar (rank 0).
    ///
    /// A Param as a shape descriptor is a shape *tensor*: scalar
    /// (`shape == NULL`) → one dynamic dim; 1d vector → one dynamic dim
    /// per element, loaded element-wise from the param. Higher rank is
    /// invalid. Shape tensors must be [`IDX_T`].
    ///
    /// Cast wrappers around dim expressions are looked through
    /// (recurse — nested casts and any other descriptor ops inside are
    /// handled by their own arms). Never fold: lengths may be dynamic.
    pub(crate) fn shape_descriptor(&mut self, id: OpId) -> Vec<OpId> {
        if id.is_null() {
            return Vec::new();
        }
        let (shape, dtype) = match self.ops[id].op.clone() {
            Op::Stack { ops } => return ops.into_vec(),
            // A bare Const is a single concrete dim length.
            Op::Const(_) => return vec![id],
            // A scalar dim *expression* (over consts, params or loads)
            // computes its own single dim length.
            Op::Unary { .. } | Op::Binary { .. } | Op::Load { .. } => return vec![id],
            Op::Cast { x, .. } => return self.shape_descriptor(x),
            Op::Param { shape, dtype, .. } => (shape, dtype),
            ref op => todo!("shape_ids: invalid shape descriptor {op:?}"),
        };
        debug_assert!(dtype == IDX_T, "shape tensor must be {IDX_T:?}, got {dtype:?}");
        // A scalar shape tensor (shape == NULL) IS its single dim: the
        // param op itself is the length (a variable is its value — no
        // load). Only a 1-d shape tensor needs element-wise loads.
        if shape.is_null() {
            return vec![id];
        }
        let rank = self.shape(shape).len();
        debug_assert!(rank <= 1, "shape_ids: param shape descriptor must be 0d or 1d, got rank {rank}");
        let mut dims = Vec::with_capacity(rank);
        for i in 0..rank {
            dims.push(self.push_back(Op::Index { vec: id, idx: i }));
        }
        dims
    }

    /// Tensor shape of the value produced by `op_id`, as per-dimension op
    /// ids. Symbolic walk over existing ops with a local per-call memo
    /// (shared sub-DAGs are re-entered by many parents; re-walking them is
    /// exponential without the memo). A `Stack` accessed *directly* (as a
    /// value) produces `[len] + first_element_shape`: its length is emitted
    /// as a const dim. A `Stack` referenced as a *shape descriptor* (through
    /// `Param { shape }`, `Reshape { shape }`, `Expand { shape }`) yields
    /// its element ops directly. Movement ops apply their transformation
    /// (`Permute` reorders, `Pad`/`Narrow` substitute their stored `len`);
    /// a scalar value (`Const`, scalar `Param`) has an empty shape.
    #[must_use]
    pub(crate) fn shape_ids(&mut self, op_id: OpId) -> Vec<OpId> {
        if op_id.is_null() {
            return Vec::new();
        }
        let mut stack = vec![op_id];
        let mut visited: Map<OpId, Vec<OpId>> = Map::default();
        // Lane-axis attribution for binary lane-merge semantics (see
        // the `binary` shape rules): which axis each lane dim came from.
        let mut axes: Map<OpId, Vec<u32>> = Map::default();
        for _ in 0..250_000 {
            let Some(op_id) = stack.pop() else {
                return visited.remove(&op_id).unwrap();
            };
            if visited.contains_key(&op_id) {
                continue;
            }
            let node_op = self.ops[op_id].op.clone();
            match node_op {
                Op::Const(_) => {
                    visited.insert(op_id, vec![]);
                }
                // Unroll lane values: one dim of `len` lanes on `axis`.
                Op::Unroll { axis, len } => {
                    visited.insert(op_id, vec![self.const_idx(u32::from(len))]);
                    axes.insert(op_id, vec![axis]);
                }
                Op::Param { shape, .. } => {
                    visited.insert(op_id, self.shape_descriptor(shape));
                }
                // Direct access of a stack op: it is a rank-1+ tensor whose
                // leading dim is the element count (emitted as a const).
                Op::Stack { ref ops } => {
                    if ops.is_empty() {
                        visited.insert(op_id, vec![]);
                    } else if let Some(mut dims) = visited.get(&ops[0]).cloned() {
                        dims.insert(0, self.const_idx(ops.len() as u32));
                        visited.insert(op_id, dims);
                    } else {
                        stack.push(op_id);
                        stack.push(ops[0]);
                    }
                }
                Op::Reshape { shape, .. } | Op::Expand { shape, .. } => {
                    visited.insert(op_id, self.shape_descriptor(shape));
                }
                Op::Permute { x, ref axes } => match visited.get(&x) {
                    Some(dims) => {
                        visited.insert(op_id, crate::shape::permute(dims, axes));
                    }
                    None => {
                        stack.push(op_id);
                        stack.push(x);
                    }
                },
                Op::Flip { x, .. } => match visited.get(&x) {
                    Some(dims) => {
                        visited.insert(op_id, dims.clone());
                    }
                    None => {
                        stack.push(op_id);
                        stack.push(x);
                    }
                },
                Op::Narrow { x, axis, len, .. } | Op::Pad { x, axis, len, .. } => match visited.get(&x).cloned() {
                    Some(mut dims) => {
                        if dims.is_empty() {
                            dims = vec![len];
                        } else {
                            dims[axis as usize] = len;
                        }
                        visited.insert(op_id, dims);
                    }
                    None => {
                        stack.push(op_id);
                        stack.push(x);
                    }
                },
                // Value shape of a binary: scalars broadcast (the other
                // operand's shape); lane operands merge axis-aware (see
                // the `binary` shape rules).
                Op::Binary { x, y, .. } => {
                    let x_done = visited.contains_key(&x);
                    let y_done = visited.contains_key(&y);
                    if !x_done || !y_done {
                        // Not both operands resolved yet: revisit after.
                        stack.push(op_id);
                        if !x_done {
                            stack.push(x);
                        }
                        if !y_done {
                            stack.push(y);
                        }
                        continue;
                    }
                    let x_dims = visited[&x].clone();
                    let y_dims = visited[&y].clone();
                    // Lane algebra: scalar broadcast, union by lane axis.
                    // A binary is elementwise: scalars broadcast, shared
                    // axes keep x's dim, disjoint axes outer (sorted
                    // ascending). No compatibility proof demanded.
                    let (dims, merged_axes) = match (x_dims.as_slice(), y_dims.as_slice()) {
                        (&[], &[]) => (Vec::new(), Vec::new()),
                        (&[], _) => (visited[&y].clone(), axes.get(&y).cloned().unwrap_or_default()),
                        (_, &[]) => (visited[&x].clone(), axes.get(&x).cloned().unwrap_or_default()),
                        _ => {
                            let x_axes = axes.get(&x).cloned().unwrap_or_default();
                            let y_axes = axes.get(&y).cloned().unwrap_or_default();
                            if x_axes.is_empty() || y_axes.is_empty() {
                                // At least one side unattributed: take the
                                // attributed side's dims.
                                if x_axes.is_empty() {
                                    (y_dims, y_axes)
                                } else {
                                    (x_dims, x_axes)
                                }
                            } else {
                                // Union by axis, sorted ascending. Shared
                                // axes keep x's dim.
                                let mut merged: Vec<(UAxis, OpId)> = x_axes.iter().copied().zip(x_dims.iter().copied()).collect();
                                for (a, d) in y_axes.iter().copied().zip(y_dims.iter().copied()) {
                                    match merged.iter().find(|(b, _)| *b == a) {
                                        Some(_) => {}
                                        None => merged.push((a, d)),
                                    }
                                }
                                merged.sort_by_key(|&(a, _)| a);
                                (merged.iter().map(|&(_, d)| d).collect(), merged.iter().map(|&(a, _)| a).collect())
                            }
                        }
                    };
                    visited.insert(op_id, dims);
                    if !merged_axes.is_empty() {
                        axes.insert(op_id, merged_axes);
                    }
                }
                Op::Reduce { x, reduce_axis, .. } => match visited.get(&x).cloned() {
                    Some(dims) => {
                        // Lane-space reduce: dims of x minus the dims of the
                        // axes the reduce iterates. The reduce axes are the
                        // unroll nodes the axis expression names; dim/axis
                        // attribution rides the axes map. A legacy
                        // pre-linearize reduce (its axis is the loop being
                        // collapsed, no unroll nodes in the cone) keeps the
                        // old trailing-dim semantics: it drops the last
                        // value dim.
                        let mut reduce_axes: Vec<u32> = Vec::new();
                        let mut axis_stack = vec![reduce_axis];
                        let mut axis_seen: Set<OpId> = Set::default();
                        while let Some(id) = axis_stack.pop() {
                            if !axis_seen.insert(id) {
                                continue;
                            }
                            let node_op = self.ops[id].op.clone();
                            match node_op {
                                Op::Unroll { axis, .. } => reduce_axes.push(axis),
                                Op::Binary { x: a, y: b, .. } => {
                                    axis_stack.push(a);
                                    axis_stack.push(b);
                                }
                                Op::Const(_) => {}
                                _ => {}
                            }
                        }
                        if reduce_axes.is_empty() {
                            let mut legacy = dims;
                            legacy.truncate(legacy.len().saturating_sub(1));
                            visited.insert(op_id, legacy);
                        } else {
                            match axes.get(&x).cloned() {
                                Some(xs) => {
                                    let xs: Vec<u32> = xs;
                                    let surviving: Vec<u32> = xs.iter().copied().filter(|a| !reduce_axes.contains(a)).collect();
                                    let surviving_dims: Vec<_> = dims
                                        .iter()
                                        .copied()
                                        .zip(xs)
                                        .filter(|&(_, a)| !reduce_axes.contains(&a))
                                        .map(|(d, _)| d)
                                        .collect();
                                    visited.insert(op_id, surviving_dims);
                                    if !surviving.is_empty() {
                                        axes.insert(op_id, surviving);
                                    }
                                }
                                None => {
                                    visited.insert(op_id, vec![]);
                                }
                            }
                        }
                    }
                    None => {
                        stack.push(op_id);
                        stack.push(x);
                    }
                },
                // A scalar access (null index) is scalar by form; a
                // vector/tile load has the shape of its index expression.
                Op::Load { src_idx, .. } => {
                    if src_idx.is_null() {
                        visited.insert(op_id, vec![]);
                    } else {
                        match visited.get(&src_idx).cloned() {
                            Some(dims) => {
                                visited.insert(op_id, dims);
                                if let Some(a) = axes.get(&src_idx).cloned() {
                                    axes.insert(op_id, a);
                                }
                            }
                            None => {
                                stack.push(op_id);
                                stack.push(src_idx);
                                continue;
                            }
                        }
                    }
                }
                // `Copy` moves elements between buffers: its shape is the
                // access shape, not either buffer's.
                Op::Copy { src_idx, .. } => {
                    if src_idx.is_null() {
                        visited.insert(op_id, vec![]);
                    } else {
                        match visited.get(&src_idx).cloned() {
                            Some(dims) => {
                                visited.insert(op_id, dims);
                                if let Some(a) = axes.get(&src_idx).cloned() {
                                    axes.insert(op_id, a);
                                }
                            }
                            None => {
                                stack.push(op_id);
                                stack.push(src_idx);
                                continue;
                            }
                        }
                    }
                }
                Op::Store { src: x, .. }
                | Op::Cast { x, .. }
                | Op::Bitcast { x, .. }
                | Op::Unary { x, .. }
                | Op::TT(TTOp::LLKBcast { cb_a: x, .. }) => match visited.get(&x) {
                    Some(dims) => {
                        visited.insert(op_id, dims.clone());
                        if let Some(a) = axes.get(&x).cloned() {
                            axes.insert(op_id, a);
                        }
                    }
                    None => {
                        stack.push(op_id);
                        stack.push(x);
                    }
                },
                // Lowered calls report no shape ids: no buffer is ever
                // sized from them (writer CBs are pre-sized). `shape`
                // and the dtype map carry their tile shapes.
                Op::TT(TTOp::MatmulTiles { .. }) | Op::TT(TTOp::ReduceTile { .. }) => {
                    visited.insert(op_id, Vec::new());
                }
                Op::Asm { ref ops, .. } | Op::TT(TTOp::LLK { ref ops, .. }) => {
                    if ops.is_empty() {
                        todo!("shape: operand-free Asm/LLK produces no value")
                    }
                    match visited.get(&ops[0]) {
                        Some(dims) => {
                            visited.insert(op_id, dims.clone());
                            if let Some(a) = axes.get(&ops[0]).cloned() {
                                axes.insert(op_id, a);
                            }
                        }
                        None => {
                            stack.push(op_id);
                            stack.push(ops[0]);
                        }
                    }
                }
                // Lane extraction drops the leading dim: scalar from a
                // vector, row vector from a tile. Lane reduction yields
                // a scalar. The leading axis goes with the leading dim.
                Op::Index { vec: x, .. } => match visited.get(&x).cloned() {
                    Some(mut dims) => {
                        debug_assert!(!dims.is_empty(), "shape_ids: Index of scalar");
                        if !dims.is_empty() {
                            dims.remove(0);
                            let mut axs = axes.get(&x).cloned().unwrap_or_default();
                            if !axs.is_empty() {
                                axs.remove(0);
                            }
                            if !axs.is_empty() {
                                axes.insert(op_id, axs);
                            }
                        }
                        visited.insert(op_id, dims);
                    }
                    None => {
                        stack.push(op_id);
                        stack.push(x);
                    }
                },
                // A fragment value has its native width as a single dim
                // (widths match the hardware's native fragment sizes).
                Op::Wmma { dims, .. } => {
                    let n: u32 = match dims {
                        MMADims::m8n8k16 => 2,
                        MMADims::m16n8k8 => 4,
                        MMADims::m16n8k16 => 4,
                        MMADims::m32n8k16 => 8,
                        MMADims::m8n32k16 => 8,
                        MMADims::m8n8k32 => 2,
                        MMADims::m8n8k128 => 2,
                    };
                    let dim = self.const_idx(n);
                    visited.insert(op_id, vec![dim]);
                }
                // Group/loop indices and scalar storages are scalar values.
                Op::Range { .. } | Op::Loop { .. } => {
                    visited.insert(op_id, vec![]);
                }
                // A buffer's shape is a 1-D vector of its element count.
                Op::Storage { len, .. } => {
                    let dim = self.const_idx(len);
                    visited.insert(op_id, vec![dim]);
                }
                ref op => todo!("shape_ids of {op:?}"),
            }
        }
        panic!("shape_ids did not resolve in 250000 steps");
    }

    /// Numeric tensor shape of the value produced by `op_id`: each dim
    /// resolved to its const value, `-1` where resolution fails (symbolic).
    /// Fast accessor, no validity checks: binary merges take x's side on
    /// any mismatch, symbolic dims are assumed fine. Immutable — unlike
    /// [`Self::shape_ids`] this never emits ops; a directly accessed
    /// `Stack` contributes its element count as a plain number instead of
    /// a const dim op.
    pub(crate) fn shape(&self, op_id: OpId) -> Vec<Dim> {
        // Slot value: resolved dim length, 0 = unset, 1 = scalar, -1 =
        // symbolic (unresolvable). Only `Unroll` sets slots; loops and
        // ranges are scalar iterators and set nothing.
        //
        // Allocation budget: one stack `Vec`, one visited `Map`, one
        // result `Vec`. Everything else borrows (`Stack`/descriptor
        // dims iterate by reference, never `to_vec`).
        fn resolve_dim(k: &Kernel, id: OpId) -> i32 {
            k.resolve_const(id).and_then(crate::dtype::Constant::as_dim).map(|d| d as i32).unwrap_or(-1)
        }
        if op_id.is_null() {
            return Vec::new();
        }
        let mut stack = vec![op_id];
        let mut visited: Map<OpId, [i32; 8]> = Map::default();
        for _ in 0..250_000 {
            let Some(id) = stack.pop() else {
                let slots = visited[&op_id];
                return slots.iter().filter(|&&d| d != 0).map(|&d| d as Dim).collect();
            };
            if visited.contains_key(&id) {
                continue;
            }
            match self.ops[id].op {
                Op::Const(_) => {
                    visited.insert(id, [0; 8]);
                }
                // Only unroll sets slots: one dim of `len` lanes on `axis`.
                Op::Unroll { axis, len } => {
                    let mut dims = [0; 8];
                    dims[axis as usize] = i32::from(len);
                    visited.insert(id, dims);
                }
                // Buffer shapes are positional: slot[i] = dim[i], -1 where
                // symbolic. The descriptor borrows, never allocates.
                Op::Param { shape, .. } => {
                    let mut dims = [0; 8];
                    if !shape.is_null() {
                        match self.ops[shape].op {
                            Op::Stack { ref ops } => {
                                for (i, &d) in ops.iter().enumerate() {
                                    if i >= dims.len() {
                                        break;
                                    }
                                    dims[i] = resolve_dim(self, d);
                                }
                            }
                            _ => {
                                dims[0] = resolve_dim(self, shape);
                            }
                        }
                    }
                    visited.insert(id, dims);
                }
                // Stacks of IDX_T dims are shape descriptors (positional,
                // like params); value stacks carry their length in slot 0.
                Op::Stack { ref ops } => {
                    if ops.iter().all(|&o| self.dtype(o) == IDX_T) {
                        let mut dims = [0; 8];
                        for (i, &d) in ops.iter().enumerate() {
                            if i >= dims.len() {
                                break;
                            }
                            dims[i] = resolve_dim(self, d);
                        }
                        visited.insert(id, dims);
                    } else {
                        let mut dims = [0; 8];
                        dims[0] = ops.len() as i32;
                        visited.insert(id, dims);
                    }
                }
                Op::Reshape { shape, .. } | Op::Expand { shape, .. } => {
                    let mut dims = [0; 8];
                    if !shape.is_null() {
                        match self.ops[shape].op {
                            Op::Stack { ref ops } => {
                                for (i, &d) in ops.iter().enumerate() {
                                    if i >= dims.len() {
                                        break;
                                    }
                                    dims[i] = resolve_dim(self, d);
                                }
                            }
                            _ => {
                                dims[0] = resolve_dim(self, shape);
                            }
                        }
                    }
                    visited.insert(id, dims);
                }
                Op::Permute { x, ref axes } => match visited.get(&x) {
                    Some(dims) => {
                        // In-place permute into a fresh array: no Vec.
                        let dims = *dims;
                        let mut out = [0; 8];
                        for (i, &a) in axes.iter().enumerate() {
                            if i >= out.len() {
                                break;
                            }
                            out[i] = dims[a as usize];
                        }
                        visited.insert(id, out);
                    }
                    None => {
                        stack.push(id);
                        stack.push(x);
                    }
                },
                Op::Flip { x, .. } => match visited.get(&x) {
                    Some(dims) => {
                        visited.insert(id, dims.clone());
                    }
                    None => {
                        stack.push(id);
                        stack.push(x);
                    }
                },
                Op::Narrow { x, axis, len, .. } | Op::Pad { x, axis, len, .. } => match visited.get(&x).cloned() {
                    Some(mut dims) => {
                        dims[axis as usize] = resolve_dim(self, len);
                        visited.insert(id, dims);
                    }
                    None => {
                        stack.push(id);
                        stack.push(x);
                    }
                },
                // Binary is implicit broadcast, per slot: equal slots
                // agree; a -1 (symbolic), 0 (unset) or 1 (scalar) side
                // takes the max; two differing concrete dims panic.
                Op::Binary { x, y, .. } => {
                    let x_done = visited.contains_key(&x);
                    let y_done = visited.contains_key(&y);
                    if !x_done || !y_done {
                        // Not both operands resolved yet: revisit after.
                        stack.push(id);
                        if !x_done {
                            stack.push(x);
                        }
                        if !y_done {
                            stack.push(y);
                        }
                        continue;
                    }
                    let xd = visited[&x];
                    let yd = visited[&y];
                    let mut dims = [0; 8];
                    for i in 0..dims.len() {
                        let (a, b) = (xd[i], yd[i]);
                        dims[i] = if a == b {
                            a
                        } else if a == -1 || a == 0 || a == 1 || b == -1 || b == 0 || b == 1 {
                            a.max(b)
                        } else {
                            panic!("binary with unequal dims {a} vs {b}");
                        };
                    }
                    visited.insert(id, dims);
                }
                // Reduce zeroes the reduced lane's slot when the axis is
                // an `Op::Unroll`, otherwise it drops the last dim.
                Op::Reduce { x, reduce_axis, .. } => match visited.get(&x).cloned() {
                    Some(mut dims) => {
                        match (!reduce_axis.is_null()).then(|| &self.ops[reduce_axis].op) {
                            Some(Op::Unroll { axis, .. }) => {
                                dims[*axis as usize] = 0;
                            }
                            _ => {
                                for i in (0..dims.len()).rev() {
                                    if dims[i] != 0 {
                                        dims[i] = 0;
                                        break;
                                    }
                                }
                            }
                        }
                        visited.insert(id, dims);
                    }
                    None => {
                        stack.push(id);
                        stack.push(x);
                    }
                },
                Op::Store { src: x, .. }
                | Op::Cast { x, .. }
                | Op::Bitcast { x, .. }
                | Op::Unary { x, .. }
                | Op::TT(TTOp::LLKBcast { cb_a: x, .. }) => match visited.get(&x) {
                    Some(dims) => {
                        visited.insert(id, *dims);
                    }
                    None => {
                        stack.push(id);
                        stack.push(x);
                    }
                },
                // A reduce over a 32-tile yields one [32] row (Row/Col)
                // or a scalar (Scalar): kind-constant, since the
                // slot-indexed feeder load itself shapes scalar.
                // A matmul yields one [32, 32] tile: kind-constant.
                Op::TT(TTOp::ReduceTile { kind, .. }) => {
                    let mut out = [0; 8];
                    if !matches!(kind, TileDim::Scalar) {
                        out[0] = 32;
                    }
                    visited.insert(id, out);
                }
                Op::TT(TTOp::MatmulTiles { .. }) => {
                    visited.insert(id, [32, 32, 0, 0, 0, 0, 0, 0]);
                }
                // A fill produces one full tile: fixed [32, 32], no
                // unroll dims anywhere.
                Op::TT(TTOp::FillTile { .. }) => {
                    visited.insert(id, [32, 32, 0, 0, 0, 0, 0, 0]);
                }
                Op::Asm { ref ops, .. } | Op::TT(TTOp::LLK { ref ops, .. }) => {
                    if ops.is_empty() {
                        // Operand-free Asm/LLK is an effect-only call:
                        // no value, scalar shape.
                        visited.insert(id, [0; 8]);
                    } else {
                        match visited.get(&ops[0]) {
                            Some(dims) => {
                                visited.insert(id, *dims);
                            }
                            None => {
                                stack.push(id);
                                stack.push(ops[0]);
                            }
                        }
                    }
                }
                // Lane extraction drops slot 0, shifting the rest down.
                Op::Index { vec: x, .. } => match visited.get(&x).cloned() {
                    Some(dims) => {
                        let mut out = [0; 8];
                        for i in 1..dims.len() {
                            out[i - 1] = dims[i];
                        }
                        visited.insert(id, out);
                    }
                    None => {
                        stack.push(id);
                        stack.push(x);
                    }
                },
                // A fragment value has its native width in slot 0
                // (widths match the hardware's native fragment sizes).
                Op::Wmma { dims, .. } => {
                    let mut out = [0; 8];
                    out[0] = match dims {
                        MMADims::m8n8k16 => 2,
                        MMADims::m16n8k8 => 4,
                        MMADims::m16n8k16 => 4,
                        MMADims::m32n8k16 => 8,
                        MMADims::m8n32k16 => 8,
                        MMADims::m8n8k32 => 2,
                        MMADims::m8n8k128 => 2,
                    };
                    visited.insert(id, out);
                }
                // Loops and ranges are scalar iterators: no slots.
                Op::Range { .. } | Op::Loop { .. } => {
                    visited.insert(id, [0; 8]);
                }
                // Loads and copies ride their index expression; a null
                // index is scalar by form.
                Op::Load { src_idx, .. } => {
                    if src_idx.is_null() {
                        visited.insert(id, [0; 8]);
                    } else {
                        match visited.get(&src_idx).cloned() {
                            Some(dims) => {
                                visited.insert(id, dims);
                            }
                            None => {
                                stack.push(id);
                                stack.push(src_idx);
                                continue;
                            }
                        };
                    }
                }
                Op::Copy { src_idx, .. } => {
                    if src_idx.is_null() {
                        visited.insert(id, [0; 8]);
                    } else {
                        match visited.get(&src_idx).cloned() {
                            Some(dims) => {
                                visited.insert(id, dims);
                            }
                            None => {
                                stack.push(id);
                                stack.push(src_idx);
                                continue;
                            }
                        };
                    }
                }
                // A buffer's element count sits in slot 0.
                Op::Storage { len, .. } => {
                    let mut dims = [0; 8];
                    dims[0] = len as i32;
                    visited.insert(id, dims);
                }
                // Effect-only and embedded ops carry no value shape.
                _ => {
                    visited.insert(id, [0; 8]);
                }
            }
        }
        panic!("shape did not resolve in 250000 steps");
    }

    /// Builds a single shape-descriptor op for the value produced by
    /// `op_id`, used to size a store/param buffer: `NULL` for scalars, the
    /// bare dim for rank-1, a `Stack` otherwise.
    #[must_use]
    pub(crate) fn stack_shape_dims(&mut self, op_id: OpId) -> OpId {
        match self.shape_ids(op_id).as_slice() {
            [] => OpId::NULL,
            [dim] => *dim,
            dims => self.stack(dims),
        }
    }

    /// Get index loop ids, dimensions and strides.
    ///
    /// Returns `loop_id` -> (dimension, stride) where NULL means unknown stride.
    /// Returns `None` when a group length or loop bound is dynamic (a `Param`
    /// variable with no constant value): there is no concrete dim to report.
    /// Callers must bail (not optimize) on `None`, mirroring `resolve_const`.
    pub(crate) fn get_strides(&self, index: OpId) -> Option<Map<OpId, (Dim, Dim)>> {
        //println!("Get index {index}");

        let index_len_of = |op: &Op| -> Option<Dim> {
            match op {
                Op::Range { kind, .. } => match kind {
                    RangeKind::Group(len) => self.resolve_const(*len).and_then(crate::dtype::Constant::as_dim),
                    RangeKind::Local(len) => Some(i64::from(*len)),
                    RangeKind::Warp(_) => Some(i64::from(self.dev_info().warp_size)),
                },
                Op::Loop { len, .. } => self.resolve_const(*len).and_then(crate::dtype::Constant::as_dim),
                _ => unreachable!(),
            }
        };
        // Patterns built once per call, matched by reference: no
        // allocation on the per-node path below.
        let range_or_loop = Pat::any([Pat::Range, Pat::Loop]);
        let bare_idx = Pat::all([Pat::bind('r'), range_or_loop.clone()]);
        let add_idx = Pat::binary(BOp::Add, Pat::all([Pat::bind('r'), range_or_loop.clone()]), Pat::bind('s'));
        let add = Pat::binary(BOp::Add, Pat::bind('a'), Pat::bind('b'));
        let sub = Pat::binary(BOp::Sub, Pat::bind('a'), Pat::bind('b'));
        let neg = Pat::unary(Pat::bind('a'), UOp::Neg);
        let mul_idx = Pat::binary(BOp::Mul, Pat::all([Pat::bind('r'), range_or_loop.clone()]), Pat::bind_const('k'));
        let shl_idx = Pat::binary(BOp::BitShiftLeft, Pat::all([Pat::bind('r'), range_or_loop]), Pat::bind_const('k'));
        let shl = Pat::binary(BOp::BitShiftLeft, Pat::bind('e'), Pat::bind_const('k'));

        // Stride accumulation: one var may appear several times
        // (i + i*4 has stride 5, not 4); overwriting drops terms.
        let add_stride = |indices: &mut Map<OpId, (Dim, Dim)>, var: OpId, len: Dim, stride: Dim| {
            indices.entry(var).and_modify(|(_, s)| *s += stride).or_insert((len, stride));
        };

        let mut params = vec![(index, 1i64)];
        let mut indices = Map::default();

        for _ in 0..250_000 {
            let Some((param, scale)) = params.pop() else { break };
            // Const leaf folds into the NULL offset.
            if self.match_pat(param, Pat::Const).is_some() {
                let Op::Const(c) = self.at(param) else { unreachable!() };
                let Some(k) = c.as_integer() else { return None };
                indices.entry(OpId::NULL).and_modify(|(_, v)| *v += k * scale).or_insert((0, k * scale));
                continue;
            }
            // Bare index var: stride 1 (times the inherited scale).
            if let Some(m) = self.match_pat(param, &bare_idx) {
                let Some(d) = index_len_of(self.at(m.op('r'))) else {
                    return None;
                };
                add_stride(&mut indices, m.op('r'), d, scale);
                continue;
            }
            // Add with an index operand: stride 1 (times the inherited
            // scale), descend the rest.
            if let Some(m) = self.match_pat(param, &add_idx) {
                let Some(d) = index_len_of(self.at(m.op('r'))) else {
                    return None;
                };
                add_stride(&mut indices, m.op('r'), d, scale);
                params.push((m.op('s'), scale));
                continue;
            }
            // Plain add descends both sides.
            if let Some(m) = self.match_pat(param, &add) {
                params.push((m.op('a'), scale));
                params.push((m.op('b'), scale));
                continue;
            }
            // Sub descends with the subtrahend negated.
            if let Some(m) = self.match_pat(param, &sub) {
                params.push((m.op('a'), scale));
                params.push((m.op('b'), -scale));
                continue;
            }
            // Negation flips the scale.
            if let Some(m) = self.match_pat(param, &neg) {
                params.push((m.op('a'), -scale));
                continue;
            }
            // Index times const: stride folds the factor in.
            if let Some(m) = self.match_pat(param, &mul_idx) {
                let Op::Const(c) = self.at(m.op('k')) else { unreachable!() };
                let Some(d) = index_len_of(self.at(m.op('r'))) else {
                    return None;
                };
                let Some(k) = c.as_integer() else { return None };
                add_stride(&mut indices, m.op('r'), d, k * scale);
                continue;
            }
            // Shifted index records 2^k; any other shifted expr descends.
            if let Some(m) = self.match_pat(param, &shl_idx) {
                let Op::Const(c) = self.at(m.op('k')) else { unreachable!() };
                let Some(d) = index_len_of(self.at(m.op('r'))) else {
                    return None;
                };
                let Some(k) = c.as_dim() else { return None };
                add_stride(&mut indices, m.op('r'), d, (1i64 << k) * scale);
                continue;
            }
            if let Some(m) = self.match_pat(param, &shl) {
                if let Op::Const(c) = self.at(m.op('k')) {
                    let Some(k) = c.as_dim() else { return None };
                    params.push((m.op('e'), scale * (1i64 << k)));
                }
                continue;
            }
            // Unroll lane var: post-tile lane trees index over
            // `Op::Unroll` nodes, not Range/Loop. Bare lane var takes
            // stride 1 (times the inherited scale), length is the lane
            // count; lane var scaled by a const (lane-tree terms) folds
            // the factor into the stride.
            if let Op::Unroll { len, .. } = self.at(param) {
                add_stride(&mut indices, param, i64::from(*len), scale);
                continue;
            }
            if let Op::Binary { x, y, bop: BOp::Mul } = self.at(param) {
                match (self.at(*x), self.at(*y)) {
                    (Op::Unroll { len, .. }, Op::Const(c)) | (Op::Const(c), Op::Unroll { len, .. }) => {
                        let Some(k) = c.as_integer() else { return None };
                        let lane = if matches!(self.at(*x), Op::Unroll { .. }) { *x } else { *y };
                        add_stride(&mut indices, lane, i64::from(*len), k * scale);
                        continue;
                    }
                    _ => {}
                }
            }
            // Anything else (Div/Mod/Min/Max, dynamic factors,
            // data-dependent leaves) is not an affine index expression:
            // report unanalyzable explicitly instead of returning a
            // partial map that silently drops terms.
            return None;
        }
        if !params.is_empty() {
            panic!("get_strides did not finish in 250000 steps");
        }

        Some(indices)
    }

    /// True if the value's expression references an `Op::Unroll` node.
    /// Traffic base indices must be unroll-free (lanes stripped at
    /// lowering); anything else renders per-lane addresses the hardware
    /// cannot execute.
    pub(crate) fn contains_unroll(&self, id: OpId) -> bool {
        let mut stack = vec![id];
        for _ in 0..250_000 {
            let Some(n) = stack.pop() else { return false };
            if n.is_null() {
                continue;
            }
            match self.ops[n].op {
                Op::Unroll { .. } => return true,
                Op::Const(_) | Op::Param { .. } | Op::Range { .. } | Op::Storage { .. } => {}
                Op::Binary { x, y, .. } => {
                    stack.push(x);
                    stack.push(y);
                }
                Op::Unary { x, .. } | Op::Cast { x, .. } | Op::Index { vec: x, .. } => stack.push(x),
                _ => return true,
            }
        }
        panic!("contains_unroll did not finish in 250000 steps");
    }

    /// Fully resolve a kernel-shape value (an `OpId`) down to a constant dim.
    ///
    /// Follows `Cast`, `Unary`, `Binary`, `Loop`, and `Index` operands,
    /// evaluating constant expressions with `Constant::unary`/`Constant::binary`.
    /// Returns `Some(dim)` when the value depends only on constants, and `None`
    /// when it is genuinely dynamic (e.g. involves a `Param` variable). Callers
    /// must bail (not optimize) on `None`.
    pub(crate) fn resolve_const(&self, op_id: OpId) -> Option<Constant> {
        // Collect the constant-expression subgraph reachable through `Cast`,
        // `Unary`, `Binary`, `Loop`, and `Index` operands in postorder
        // (children before parents), then evaluate it bottom-up
        // (iteratively, no recursion). Postorder matters: reversed preorder
        // evaluates a parent before a shared child first reached through a
        // later branch, poisoning the parent to None (e.g. an Add whose
        // addend is also used deeper in the other addend's subtree).
        let mut seen: Set<OpId> = Set::default();
        let mut order: Vec<OpId> = Vec::new();
        let mut stack = vec![(op_id, false)];
        for _ in 0..250_000 {
            let Some((id, emit)) = stack.pop() else { break };
            if id.is_null() {
                continue;
            }
            if emit {
                order.push(id);
                continue;
            }
            if !seen.insert(id) {
                continue;
            }
            stack.push((id, true));
            match &self.ops[id].op {
                Op::Cast { x, .. } => stack.push((*x, false)),
                Op::Bitcast { x, .. } => stack.push((*x, false)),
                Op::Unary { x, .. } => stack.push((*x, false)),
                Op::Binary { x, y, .. } => {
                    stack.push((*x, false));
                    stack.push((*y, false));
                }
                Op::Stack { ops } => stack.extend(ops.iter().copied().map(|o| (o, false))),
                Op::Loop { len } => stack.push((*len, false)),
                &Op::Range { kind, .. } => match kind {
                    RangeKind::Group(len) => stack.push((len, false)),
                    RangeKind::Local(_) | RangeKind::Warp(_) => {}
                },
                _ => {}
            }
        }
        if !stack.is_empty() {
            panic!("resolve_const did not finish in 250000 steps");
        }

        let mut values: Map<OpId, Option<Constant>> = Map::default();
        for &id in &order {
            let v = match &self.ops[id].op {
                Op::Const(c) => Some(*c),
                // A cast preserves the integer value of a length.
                Op::Cast { x, dtype } => values.get(x).copied().flatten().map(|v| v.cast(*dtype)),
                // A bitcast reinterprets the bits (equal widths asserted upstream).
                Op::Bitcast { x, dtype } => values.get(x).copied().flatten().map(|v| v.bitcast(*dtype)),
                Op::Unary { x, uop } => values.get(x).copied().flatten().map(|v| v.unary(*uop)),
                Op::Binary { x, y, bop } => {
                    values.get(x).copied().flatten().zip(values.get(y).copied().flatten()).map(|(a, b)| {
                        // Shape expressions may mix dtypes (e.g. a U32 group
                        // index with an I64 pad constant); evaluate in their
                        // least upper dtype.
                        let dt = a.dtype().least_upper_dtype(b.dtype());
                        Constant::binary(a.cast(dt), b.cast(dt), *bop)
                    })
                }
                Op::Loop { len } => values.get(len).copied().flatten(),
                &Op::Range { kind, .. } => match kind {
                    RangeKind::Group(len) => values.get(&len).copied().flatten(),
                    RangeKind::Local(len) => Some(Constant::idx(len)),
                    // The warp op's value is the lane id — runtime data.
                    RangeKind::Warp(_) => None,
                },
                // Param is dynamic
                Op::Param { .. } => None,
                // A loaded value (e.g. a lowered variable: Binary{mask, Load})
                // is runtime data — never compile-time constant, but a
                // legitimate dynamic length (same as Param).
                Op::Load { .. } => None,
                // The unroll value op is the lane id vector — runtime data.
                Op::Unroll { .. } => None,
                ref op => todo!("{op:?}"),
            };
            values.insert(id, v);
        }
        values[&op_id]
    }

    /// True iff the dim expression at `id` is provably a multiple of
    /// `divisor`, without resolving it to a constant.
    ///
    /// Divisibility over the const-expr ops: `Const(c)` checks the value;
    /// `Cast` preserves integer values; `Mul` holds when either side is a
    /// multiple; `Add`/`Sub` hold when both sides are. Everything else
    /// (params, loads, divisions, bitcasts, unknown ops) is unprovable
    /// and returns false — the caller pads as usual. Iterative, never
    /// panics.
    pub(crate) fn is_sym_multiple(&self, id: OpId, divisor: Dim) -> bool {
        debug_assert!(divisor > 0, "is_sym_multiple: divisor {divisor} <= 0");
        if divisor <= 0 {
            return false;
        }
        // Postorder collection (same walk as resolve_const), then
        // bottom-up divisibility.
        let mut seen: Set<OpId> = Set::default();
        let mut order: Vec<OpId> = Vec::new();
        let mut stack = vec![(id, false)];
        for _ in 0..250_000 {
            let Some((nid, emit)) = stack.pop() else { break };
            if nid.is_null() {
                continue;
            }
            if emit {
                order.push(nid);
                continue;
            }
            if !seen.insert(nid) {
                continue;
            }
            stack.push((nid, true));
            match self.at(nid) {
                Op::Cast { x, .. } => stack.push((*x, false)),
                Op::Bitcast { x, .. } => stack.push((*x, false)),
                Op::Unary { x, .. } => stack.push((*x, false)),
                Op::Binary { x, y, .. } => {
                    stack.push((*x, false));
                    stack.push((*y, false));
                }
                _ => {}
            }
        }
        if !stack.is_empty() {
            panic!("is_sym_multiple did not finish in 250000 steps");
        }
        let mut multiple: Map<OpId, bool> = Map::default();
        for &nid in &order {
            let m = match self.at(nid) {
                Op::Const(c) => c.as_dim().map(|v| v % divisor == 0).unwrap_or(false),
                Op::Cast { x, .. } => multiple.get(x).copied().unwrap_or(false),
                Op::Binary { x, y, bop } => match bop {
                    BOp::Mul => multiple.get(x).copied().unwrap_or(false) || multiple.get(y).copied().unwrap_or(false),
                    BOp::Add | BOp::Sub => multiple.get(x).copied().unwrap_or(false) && multiple.get(y).copied().unwrap_or(false),
                    _ => false,
                },
                _ => false,
            };
            multiple.insert(nid, m);
        }
        multiple.get(&id).copied().unwrap_or(false)
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
        let op_node = OpLinked { prev: self.tail, next: OpId::NULL, op };
        let op_id = self.ops.push(op_node);
        if self.head.is_null() {
            self.head = op_id;
        } else {
            self.ops[self.tail].next = op_id;
        }
        self.tail = op_id;
        op_id
    }

    /// Duplicates the dependency chain of `root_op` into a fresh kernel.
    ///
    /// # The duplication design (runtime vs kernelizer)
    ///
    /// Kernels can share their ops' results in two ways, and the two callers
    /// of this method use two different ownership models:
    ///
    /// - **Runtime** (eager tensors): ops never *consume* — they only create
    ///   new tensors. When a kernel must be split, the original kernel keeps
    ///   the output tensor in its `outputs` and keeps ALL of its ops; the
    ///   fresh kernel receives a **clone** of the root chain and *recomputes*
    ///   the same values from the shared input loads. Nothing is removed,
    ///   no refcount changes hands — the only bookkeeping is that every load
    ///   the fresh kernel reads gains one reference (retain per `new_loads`
    ///   occurrence). The fresh kernel's `outputs` stays empty: it produces
    ///   values only for the ops built on it afterwards.
    /// - **Kernelizer** (graph classes): ops DO consume edges (`rcs`,
    ///   `visited`, `outputs`). The split keeps the same observable
    ///   accounting as a move: one `outputs` occurrence of the child class
    ///   leaves the original kernel and is pushed onto the fresh kernel,
    ///   where the caller's `consume` immediately removes it. The original
    ///   kernel keeps its output registration for its remaining consumers;
    ///   if the split consumer was the last one, the original's orphaned
    ///   chain is cleaned up separately via [`Kernel::remove_unused_chain`].
    ///
    /// In both models this method is a **pure duplication**: `self` is left
    /// completely untouched (ops AND loads), and the fresh kernel contains
    /// exactly the ops `root_op` transitively depends on, cloned in topo
    /// order with remapped OpIds.
    ///
    /// Returns the fresh kernel, the remapped `root_op` OpId inside it, and
    /// `new_loads` — the caller's `loads` entries (parallel to the input
    /// `Param` ops, linked-list order) for the Params the fresh kernel
    /// cloned, in the fresh kernel's Param order. Every `new_loads` entry is
    /// an additional reader of the underlying tensor: the runtime caller
    /// retains one reference per occurrence.
    pub(crate) fn duplicate_subkernel<T: Copy>(&mut self, root_op: OpId, loads: &[T]) -> (Self, OpId, Vec<T>) {
        // Walk 1: from root_op
        let mut root_required = Set::default();
        let mut stack = vec![root_op];
        for _ in 0..250_000 {
            let Some(op) = stack.pop() else { break };
            if root_required.insert(op) {
                stack.extend(self.at(op).parameters());
            }
        }
        if !stack.is_empty() {
            panic!("duplicate_subkernel did not finish in 250000 steps");
        }

        // Partition loads: the fresh kernel receives the caller's `loads`
        // entries for every input Param it cloned. `self` keeps ALL of its
        // loads — it is untouched. `loads` is parallel to the kernel Param
        // ops. Kernels before linearize contain only Params — Storage
        // (accumulators etc.) is resolved later, so none may appear here.
        let mut new_loads: Vec<T> = Vec::new();
        let mut load_idx = 0;
        let mut oid = self.head;
        for _ in 0..250_000 {
            if oid.is_null() {
                break;
            }
            match self.at(oid) {
                // Global and Variable Params are loads; GlobalMut is a store and
                // is not part of the loads list.
                Op::Param { kind: ParamKind::Global | ParamKind::Variable, .. } => {
                    if root_required.contains(&oid) {
                        new_loads.push(loads[load_idx]);
                    }
                    load_idx += 1;
                }
                Op::Param { kind: ParamKind::GlobalMut, .. } => {}
                Op::Storage { .. } => {
                    panic!("duplicate_subkernel: unexpected Op::Storage (pre-linearize kernels contain only Params)")
                }
                _ => {}
            }
            oid = self.next_op(oid);
        }
        if !oid.is_null() {
            panic!("duplicate_subkernel did not finish in 250000 steps");
        }

        let mut new_kernel =
            Self { ops: Slab::new(), head: OpId::NULL, tail: OpId::NULL, dev: self.dev, dev_info: self.dev_info.clone() };
        let mut remap: Map<OpId, OpId> =
            Map::with_capacity_and_hasher(root_required.len(), core::hash::BuildHasherDefault::default());
        let mut new_root_op = OpId::NULL;
        let mut old_id = self.head;
        for _ in 0..250_000 {
            if old_id.is_null() {
                break;
            }
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
        if !old_id.is_null() {
            panic!("duplicate_subkernel did not finish in 250000 steps");
        }

        (new_kernel, new_root_op, new_loads)
    }

    /// Get all group indices used in the kernel.
    pub(crate) fn get_group_indices(&self) -> std::collections::BTreeMap<u32, OpId> {
        let mut indices = std::collections::BTreeMap::new();
        for (op_id, op_node) in self.ops.iter() {
            if let Op::Range { axis, kind: RangeKind::Group(_), .. } = op_node.op {
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
                Op::Range { axis, kind: RangeKind::Group(_), .. } => group_indices.insert(axis, op_id),
                Op::Range { axis, kind: RangeKind::Local(_), .. } => local_indices.insert(axis, op_id),
                _ => None,
            };
        }
        let mut ax = 0;
        for &idx_id in group_indices.values() {
            let Op::Range { axis, kind: RangeKind::Group(_), .. } = &mut self.ops[idx_id].op else {
                unreachable!()
            };
            *axis = ax;
            ax += 1;
        }
        for &idx_id in local_indices.values() {
            let Op::Range { axis, kind: RangeKind::Local(_), .. } = &mut self.ops[idx_id].op else {
                unreachable!()
            };
            *axis = ax;
            ax += 1;
        }
    }

    pub(crate) fn is_preceded_by_reduce(&self, x: OpId) -> bool {
        //if self.ops.values().filter(|node| matches!(node.op, Op::Reduce { .. })).count() > 1 { return true; }
        let mut params = vec![x];
        let mut found = false;
        let mut seen: Set<OpId> = Set::default();
        for _ in 0..250_000 {
            let Some(param) = params.pop() else { break };
            if !seen.insert(param) {
                continue;
            }
            if let &Op::Reduce { x, .. } = self.at(param) {
                params = vec![x];
                found = true;
                break;
            }
            params.extend(self.ops[param].op.parameters());
        }
        if !found && !params.is_empty() {
            panic!("is_preceded_by_reduce did not finish in 250000 steps");
        }
        //if params.is_empty() { return false; }
        //println!("Found reduce at {params:?}");
        // If there is a load (non constant reduce) or multiple reduces, return true
        let mut seen: Set<OpId> = Set::default();
        for _ in 0..250_000 {
            let Some(param) = params.pop() else { break };
            if !seen.insert(param) {
                continue;
            }
            if matches!(self.ops[param].op, Op::Param { .. } | Op::Reduce { .. }) {
                return true;
            }
            params.extend(self.ops[param].op.parameters());
        }
        if !params.is_empty() {
            panic!("is_preceded_by_reduce did not finish in 250000 steps");
        }
        false
    }

    pub(crate) fn is_preceded_by_compute(&self, x: OpId) -> bool {
        let mut params = vec![x];
        let mut seen: Set<OpId> = Set::default();
        let (mut has_compute, mut has_param) = (false, false);
        for _ in 0..250_000 {
            let Some(param) = params.pop() else { break };
            if !seen.insert(param) {
                continue;
            }
            match &self.ops[param].op {
                Op::Binary { .. } | Op::Unary { .. } => {
                    has_compute = true;
                    params.extend(self.ops[param].op.parameters());
                }
                Op::Param { .. } => has_param = true,
                Op::Const(_) => {}
                // Shape/bound edges carry dim math, not data: walking
                // them mistakes symbolic shape expressions (Binary/Cast
                // over dims) for compute and force-stores pure movement
                // chains at every broadcast.
                Op::Reshape { x, .. } | Op::Expand { x, .. } | Op::Narrow { x, .. } => params.push(*x),
                Op::Reduce { x, .. } => {
                    has_compute = true;
                    params.push(*x);
                }
                _ => params.extend(self.ops[param].op.parameters()),
            }
        }
        if !params.is_empty() {
            panic!("is_preceded_by_compute did not finish in 250000 steps");
        }
        has_compute && has_param
    }
}
