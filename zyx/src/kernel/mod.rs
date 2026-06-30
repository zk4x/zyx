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
//! use zyx::kernel::{DeviceId, Kernel, MMADType, MMADims, MMALayout, MemLayout, Scope};
//! use zyx::DType;
//!
//! let (m, n, k) = (1024, 1024, 1024);
//! let mut kernel = Kernel::new(DeviceId::AUTO);
//!
//! let a_buf = kernel.define(DType::F16, Scope::Global, true, m * k);
//! let b_buf = kernel.define(DType::F16, Scope::Global, true, k * n);
//! let c_buf = kernel.define(DType::F32, Scope::Global, false, m * n);
//!
//! let gidx = kernel.gidx(0, m / 16);
//! let gidy = kernel.gidx(1, n / 8);
//! let wid = kernel.lidx(0, 32);
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
//! let acc = kernel.define(DType::F32, Scope::Register, false, 4);
//! let zf = kernel.const_val(0.0f32);
//! let zero_acc = kernel.vectorize(vec![zf, zf, zf, zf]);
//! kernel.store(acc, zero_acc, c0, MemLayout::Vector(4));
//!
//! let k_loop = kernel.loop_(k / 8);
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
//! let a_frag = kernel.vectorize(vec![a_load_0, a_load_1, a_load_2, a_load_3]);
//!
//! let b_row = kernel.add(k_off, col_in_tile);
//! let b_base = kernel.mad(b_row, n_const, b_col);
//! let b_load_0 = kernel.load(b_buf, b_base, MemLayout::Scalar);
//! let b_base_n = kernel.add(b_base, n_const);
//! let b_load_1 = kernel.load(b_buf, b_base_n, MemLayout::Scalar);
//! let b_frag = kernel.vectorize(vec![b_load_0, b_load_1]);
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
use crate::view::View;

use crate::{
    DType, Map, Set,
    dtype::Constant,
    kernel_cache::KernelId,
    kernelize::KMKernelId,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
};
use nanoserde::{DeBin, SerBin};
use std::{fmt::Display, hash::BuildHasherDefault, hash::Hash};

pub use custom::CompiledKernel;
pub(crate) use custom::CustomKernel;

mod algebraic;
/// Autotuning optimizations for kernel compilation.
pub(crate) mod autotune;
/// Cost estimation for kernel selection.
mod cost;
/// Custom kernel compilation for GPU-specific operations.
mod custom;
mod debug;
mod exp2_to_exp;
mod fold_constants;
mod fold_loops;
mod fuse;
mod instr_sched;
mod licm;
mod local_reduce;
mod local_tile;
mod log2_to_ln;
mod merge_loops;
mod mma;
mod pad_index;
/// Cost prediction for kernel selection.
mod predict_cost;
mod split_loops;
mod thread_coarse;
mod tile;
mod unfold;
mod unroll_loops;
mod vectorize;
mod verify;

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
/// use zyx::kernel::{Kernel, Scope, MemLayout, DeviceId};
/// use zyx::DType;
///
/// let mut kernel = Kernel::new(DeviceId::AUTO);
/// let n = 256;
/// let inp = kernel.define(DType::F32, Scope::Global, true, n);
/// let gidx = kernel.gidx(0, n);
/// let loaded = kernel.load(inp, gidx, MemLayout::Scalar);
/// let s = kernel.sin(loaded);
/// let c = kernel.cos(loaded);
/// let result = kernel.add(s, c);
/// let out = kernel.define(DType::F32, Scope::Global, false, n);
/// kernel.store(out, result, gidx, MemLayout::Scalar);
/// ```
///
/// # Compile
///
/// Build a kernel using fused multiply-add and compile it:
///
/// ```
/// use zyx::kernel::{Kernel, Scope, MemLayout, DeviceId};
/// use zyx::{DType, Tensor, ZyxError};
///
/// let mut kernel = Kernel::new(DeviceId::AUTO);
/// let n = 4;
/// let inp = kernel.define(DType::F32, Scope::Global, true, n);
/// let gidx = kernel.gidx(0, n);
/// let loaded = kernel.load(inp, gidx, MemLayout::Scalar);
/// let result = kernel.mad(loaded, loaded, loaded); // x*x + x
/// let out = kernel.define(DType::F32, Scope::Global, false, n);
/// kernel.store(out, result, gidx, MemLayout::Scalar);
///
/// let compiled = kernel.compile()?;
/// let x = Tensor::from([1.0f32, 2.0, 3.0, 4.0]);
/// let result = compiled.forward(&[&x], [n]);
/// let data: Vec<f32> = result.try_into().unwrap();
/// assert_eq!(data, vec![2.0, 6.0, 12.0, 20.0]);
/// # Ok::<_, ZyxError>(())
/// ```
#[derive(Debug, Clone)]
pub struct Kernel {
    /// Tensor IDs that this kernel produces.
    pub(crate) outputs: Vec<TensorId>,
    /// Tensor IDs loaded from memory.
    pub(crate) loads: Vec<TensorId>,
    /// Tensor IDs stored to memory.
    pub(crate) stores: Vec<TensorId>,
    /// Operation slab containing the kernel IR.
    pub(crate) ops: Slab<OpId, OpNode>,
    /// Head of the operation linked list.
    pub(crate) head: OpId,
    /// Tail of the operation linked list.
    pub(crate) tail: OpId,
    /// Target device for compilation.
    pub(crate) device_id: DeviceId,
    /// ID of custom kernel if applicable.
    pub(crate) custom_kernel_id: Option<KernelId>,
}

/// Execution scope for kernel indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum Scope {
    /// Global memory scope (shared across all threads).
    Global,
    /// Local memory scope (per-thread or per-block).
    Local,
    /// Register scope (per-thread fast storage).
    Register,
}

/// Unary operations for element-wise kernel transformations.
///
/// These operations are applied to a single input tensor.
///
/// # Variants
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub(crate) enum UOp {
    /// Negation: -x
    Neg,
    /// Bitwise NOT: ~x
    BitNot,
    /// Exponential: e^x
    Exp,
    /// Exponential with base 2: 2^x
    Exp2,
    /// Natural logarithm: ln(x)
    Ln,
    /// Logarithm with base 2: log2(x)
    Log2,
    /// Reciprocal: 1/x
    Reciprocal,
    /// Square root: sqrt(x)
    Sqrt,
    /// Sine: sin(x)
    Sin,
    /// Cosine: cos(x)
    Cos,
    /// Floor: floor(x)
    Floor,
    /// Truncate toward zero: trunc(x)
    Trunc,
    /// Absolute value: |x|
    Abs,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
/// Binary operations for element-wise or reduction kernel operations.
///
/// These operations take two input tensors and produce an output.
///
/// # Variants
pub(crate) enum BOp {
    /// Addition: x + y
    Add,
    /// Subtraction: x - y
    Sub,
    /// Multiplication: x * y
    Mul,
    /// Division: x / y
    Div,
    /// Power: x^y
    Pow,
    /// Modulo: x % y
    Mod,
    /// Compare less than: x < y
    Cmplt,
    /// Compare greater than: x > y
    Cmpgt,
    /// Maximum: max(x, y)
    Max,
    /// Bitwise OR: x | y
    Or,
    /// Bitwise AND: x & y
    And,
    /// Bitwise XOR: x ^ y
    BitXor,
    /// Bitwise OR: x | y
    BitOr,
    /// Bitwise AND: x & y
    BitAnd,
    /// Left shift: x << y
    BitShiftLeft,
    /// Right shift: x >> y
    BitShiftRight,
    /// Not equal: x != y
    NotEq,
    /// Equal: x == y
    Eq,
}

/// Movement operations for tensor shape transformations.
///
/// These operations change the shape of tensors without changing their data.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum MoveOp {
    /// Reshape to a new shape.
    Reshape { shape: Vec<Dim> },
    /// Expand dimensions.
    Expand { shape: Vec<Dim> },
    /// Permute axes.
    Permute { axes: Vec<UAxis>, shape: Vec<Dim> },
    /// Pad dimensions.
    Pad { padding: Vec<(i64, i64)>, shape: Vec<Dim> },
}

/// Matrix multiply dimensions for tensor core operations.
///
/// Represents the shape (m, n, k) for matrix multiplication.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMADims {
    /// 8x8 with k=16
    m8n8k16,
    /// 16x8 with k=8
    m16n8k8,
    /// 16x8 with k=16
    m16n8k16,
}

/// Memory layout for tensor core matrix operands.
///
/// Describes how matrix data is stored in memory.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMALayout {
    /// Row-major for both matrices
    row_row,
    /// Row-major for A, column-major for B
    row_col,
    /// Column-major for A, row-major for B
    col_row,
    /// Column-major for both matrices
    col_col,
}

/// Data type for matrix multiply operations.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub enum MMADType {
    /// FP16 input with FP32 accumulator
    f16_f16_f16_f32,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct OpNode {
    pub(crate) prev: OpId,
    pub(crate) next: OpId, // Use Vec<OpId> instead for egraph
    pub(crate) op: Op,
}

impl SerBin for OpNode {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        self.prev.ser_bin(output);
        self.next.ser_bin(output);
        self.op.ser_bin(output);
    }
}

impl DeBin for OpNode {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        let prev = OpId::de_bin(offset, bytes)?;
        let next = OpId::de_bin(offset, bytes)?;
        let op = Op::de_bin(offset, bytes)?;
        Ok(OpNode { prev, next, op })
    }
}

/// Operation ID for kernel operations.
///
/// This is a unique identifier for each operation in the kernel IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct OpId(pub(crate) u32);

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

impl MemLayout {
    /// Get the number of elements in the memory layout.
    pub(crate) fn n_elements(self) -> Dim {
        match self {
            MemLayout::Scalar => 1,
            MemLayout::Vector(x) => x.into(),
            MemLayout::Tile { x, y, .. } => x as Dim * y as Dim,
        }
    }
}

impl std::fmt::Display for MemLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemLayout::Scalar => f.write_fmt(format_args!("Scalar")),
            MemLayout::Vector(x) => f.write_fmt(format_args!("Vec({x})")),
            MemLayout::Tile { x, y, stride } => f.write_fmt(format_args!("Tile({x}x{y} st={stride})")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Op {
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
        layout: MemLayout,
    },
    Load {
        src: OpId,
        index: OpId,
        layout: MemLayout,
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
    Wmma {
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

impl SerBin for Op {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        match self {
            Op::Cast { x, dtype } => {
                output.push(0);
                x.ser_bin(output);
                dtype.ser_bin(output);
            }
            Op::Unary { x, uop } => {
                output.push(1);
                x.ser_bin(output);
                uop.ser_bin(output);
            }
            Op::Binary { x, y, bop } => {
                output.push(2);
                x.ser_bin(output);
                y.ser_bin(output);
                bop.ser_bin(output);
            }
            Op::Const(c) => {
                output.push(3);
                c.ser_bin(output);
            }
            Op::Define { dtype, scope, ro, len } => {
                output.push(4);
                dtype.ser_bin(output);
                scope.ser_bin(output);
                output.push(u8::from(*ro));
                len.ser_bin(output);
            }
            Op::Store { dst, x, index, layout } => {
                output.push(5);
                dst.ser_bin(output);
                x.ser_bin(output);
                index.ser_bin(output);
                layout.ser_bin(output);
            }
            Op::Load { src, index, layout } => {
                output.push(6);
                src.ser_bin(output);
                index.ser_bin(output);
                layout.ser_bin(output);
            }
            Op::Index { len, scope, axis } => {
                output.push(7);
                len.ser_bin(output);
                scope.ser_bin(output);
                axis.ser_bin(output);
            }
            Op::Loop { len } => {
                output.push(8);
                len.ser_bin(output);
            }
            Op::EndLoop => output.push(9),
            Op::Mad { x, y, z } => {
                output.push(10);
                x.ser_bin(output);
                y.ser_bin(output);
                z.ser_bin(output);
            }
            Op::Wmma {
                dims,
                layout,
                dtype,
                a,
                b,
                c,
            } => {
                output.push(11);
                dims.ser_bin(output);
                layout.ser_bin(output);
                dtype.ser_bin(output);
                a.ser_bin(output);
                b.ser_bin(output);
                c.ser_bin(output);
            }
            Op::Vectorize { ops } => {
                output.push(12);
                ops.ser_bin(output);
            }
            Op::Devectorize { vec, idx } => {
                output.push(13);
                vec.ser_bin(output);
                idx.ser_bin(output);
            }
            Op::Barrier { scope } => {
                output.push(14);
                scope.ser_bin(output);
            }
            Op::If { condition } => {
                output.push(15);
                condition.ser_bin(output);
            }
            Op::EndIf => output.push(16),
            Op::ConstView(t) => {
                output.push(17);
                t.ser_bin(output);
            }
            Op::LoadView(t) => {
                output.push(18);
                t.ser_bin(output);
            }
            Op::StoreView { src, dtype } => {
                output.push(19);
                src.ser_bin(output);
                dtype.ser_bin(output);
            }
            Op::Move { x, mop } => {
                output.push(20);
                x.ser_bin(output);
                mop.ser_bin(output);
            }
            Op::Reduce { x, rop, n_axes } => {
                output.push(21);
                x.ser_bin(output);
                rop.ser_bin(output);
                n_axes.ser_bin(output);
            }
        }
    }
}

impl DeBin for Op {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        let tag = bytes[*offset];
        *offset += 1;
        match tag {
            0 => {
                let x = OpId::de_bin(offset, bytes)?;
                let dtype = DType::de_bin(offset, bytes)?;
                Ok(Op::Cast { x, dtype })
            }
            1 => {
                let x = OpId::de_bin(offset, bytes)?;
                let uop = UOp::de_bin(offset, bytes)?;
                Ok(Op::Unary { x, uop })
            }
            2 => {
                let x = OpId::de_bin(offset, bytes)?;
                let y = OpId::de_bin(offset, bytes)?;
                let bop = BOp::de_bin(offset, bytes)?;
                Ok(Op::Binary { x, y, bop })
            }
            3 => {
                let c = Constant::de_bin(offset, bytes)?;
                Ok(Op::Const(c))
            }
            4 => {
                let dtype = DType::de_bin(offset, bytes)?;
                let scope = Scope::de_bin(offset, bytes)?;
                let ro = bytes[*offset] != 0;
                *offset += 1;
                let len = Dim::de_bin(offset, bytes)?;
                Ok(Op::Define { dtype, scope, ro, len })
            }
            5 => {
                let dst = OpId::de_bin(offset, bytes)?;
                let x = OpId::de_bin(offset, bytes)?;
                let index = OpId::de_bin(offset, bytes)?;
                let layout = MemLayout::de_bin(offset, bytes)?;
                Ok(Op::Store { dst, x, index, layout })
            }
            6 => {
                let src = OpId::de_bin(offset, bytes)?;
                let index = OpId::de_bin(offset, bytes)?;
                let layout = MemLayout::de_bin(offset, bytes)?;
                Ok(Op::Load { src, index, layout })
            }
            7 => {
                let len = Dim::de_bin(offset, bytes)?;
                let scope = Scope::de_bin(offset, bytes)?;
                let axis = u32::de_bin(offset, bytes)?;
                Ok(Op::Index { len, scope, axis })
            }
            8 => {
                let len = Dim::de_bin(offset, bytes)?;
                Ok(Op::Loop { len })
            }
            9 => Ok(Op::EndLoop),
            10 => {
                let x = OpId::de_bin(offset, bytes)?;
                let y = OpId::de_bin(offset, bytes)?;
                let z = OpId::de_bin(offset, bytes)?;
                Ok(Op::Mad { x, y, z })
            }
            11 => {
                let dims = MMADims::de_bin(offset, bytes)?;
                let layout = MMALayout::de_bin(offset, bytes)?;
                let dtype = MMADType::de_bin(offset, bytes)?;
                let a = OpId::de_bin(offset, bytes)?;
                let b = OpId::de_bin(offset, bytes)?;
                let c = OpId::de_bin(offset, bytes)?;
                Ok(Op::Wmma {
                    dims,
                    layout,
                    dtype,
                    a,
                    b,
                    c,
                })
            }
            12 => {
                let ops = Vec::<OpId>::de_bin(offset, bytes)?;
                Ok(Op::Vectorize { ops })
            }
            13 => {
                let vec = OpId::de_bin(offset, bytes)?;
                let idx = usize::de_bin(offset, bytes)?;
                Ok(Op::Devectorize { vec, idx })
            }
            14 => {
                let scope = Scope::de_bin(offset, bytes)?;
                Ok(Op::Barrier { scope })
            }
            15 => {
                let condition = OpId::de_bin(offset, bytes)?;
                Ok(Op::If { condition })
            }
            16 => Ok(Op::EndIf),
            17 => {
                let t = Box::<(Constant, View)>::de_bin(offset, bytes)?;
                Ok(Op::ConstView(t))
            }
            18 => {
                let t = Box::<(DType, View)>::de_bin(offset, bytes)?;
                Ok(Op::LoadView(t))
            }
            19 => {
                let src = OpId::de_bin(offset, bytes)?;
                let dtype = DType::de_bin(offset, bytes)?;
                Ok(Op::StoreView { src, dtype })
            }
            20 => {
                let x = OpId::de_bin(offset, bytes)?;
                let mop = Box::<MoveOp>::de_bin(offset, bytes)?;
                Ok(Op::Move { x, mop })
            }
            21 => {
                let x = OpId::de_bin(offset, bytes)?;
                let rop = BOp::de_bin(offset, bytes)?;
                let n_axes = UAxis::de_bin(offset, bytes)?;
                Ok(Op::Reduce { x, rop, n_axes })
            }
            _ => Err(nanoserde::DeBinErr::new(*offset - 1, 1, bytes.len())),
        }
    }
}

impl Op {
    // TODO use custom non allocating iterator instead of allocating a vec
    #[allow(clippy::match_same_arms)]
    pub(crate) fn parameters(&self) -> impl DoubleEndedIterator<Item = OpId> {
        match self {
            Op::ConstView { .. }
            | Op::LoadView { .. }
            | Op::Const { .. }
            | Op::Define { .. }
            | Op::Index { .. }
            | Op::Loop { .. }
            | Op::EndLoop
            | Op::Barrier { .. }
            | Op::EndIf => {
                vec![]
            }
            &Op::Move { x, .. } => vec![x],
            &Op::StoreView { src, .. } => vec![src],
            Op::Reduce { x, .. } => vec![*x],
            &Op::Store { dst, x, index, .. } => vec![dst, x, index],
            Op::Cast { x, .. } => vec![*x],
            Op::Unary { x, .. } => vec![*x],
            &Op::Binary { x, y, .. } => vec![x, y],
            &Op::Load { src, index, .. } => vec![src, index],
            &Op::Mad { x, y, z } => vec![x, y, z],
            Op::Vectorize { ops } => ops.clone(),
            &Op::Devectorize { vec, .. } => vec![vec],
            &Op::Wmma { a, b, c, .. } => vec![a, b, c],
            Op::If { condition } => vec![*condition],
        }
        .into_iter()
    }

    #[allow(clippy::match_same_arms)]
    pub(crate) fn parameters_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut OpId> {
        match self {
            Op::ConstView { .. }
            | Op::LoadView { .. }
            | Op::Const { .. }
            | Op::Define { .. }
            | Op::Index { .. }
            | Op::Loop { .. }
            | Op::EndLoop
            | Op::EndIf
            | Op::Barrier { .. } => vec![],
            Op::StoreView { src, .. } => vec![src],
            Op::Move { x, .. } => vec![x],
            Op::Reduce { x, .. } => vec![x],
            Op::Store { dst, x, index, .. } => vec![dst, x, index],
            Op::Cast { x, .. } => vec![x],
            Op::Unary { x, .. } => vec![x],
            Op::Binary { x, y, .. } => vec![x, y],
            Op::Load { src, index, .. } => vec![src, index],
            Op::Mad { x, y, z } => vec![x, y, z],
            Op::Vectorize { ops } => ops.iter_mut().collect(),
            Op::Devectorize { vec, .. } => vec![vec],
            Op::Wmma { a, b, c, .. } => vec![a, b, c],
            Op::If { condition } => vec![condition],
        }
        .into_iter()
    }

    /// Check if this operation is a constant.
    pub(crate) const fn is_const(&self) -> bool {
        matches!(self, Op::Cast { .. })
    }

    /// Check if this operation is a load.
    pub(crate) const fn is_load(&self) -> bool {
        matches!(self, Op::Load { .. })
    }

    /// Remap parameter IDs according to a mapping.
    pub(crate) fn remap_params(&mut self, remapping: &Map<OpId, OpId>) {
        for param in self.parameters_mut() {
            if let Some(remapped_id) = remapping.get(param) {
                *param = *remapped_id;
            }
        }
    }
}

impl OpId {
    pub(crate) const NULL: Self = Self(u32::MAX);

    /// Check if this OpId is null.
    pub const fn is_null(self) -> bool {
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
        Ok(Self {
            head: start,
            tail: end,
            ops,
            outputs: Vec::new(),
            loads: Vec::new(),
            stores: Vec::new(),
            device_id: DeviceId::AUTO,
            custom_kernel_id: None,
        })
    }
}

impl Hash for Kernel {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.head.hash(state);
        self.ops.hash(state);
    }
}

// Custom kernel machinery
impl Kernel {
    /// Create a new custom kernel targeting a specific device.
    ///
    /// Two approaches for inputs:
    /// - **Manual gidx**: `define(dtype, Scope::Global, true, len)` + [`Kernel::gidx`]
    /// - **LoadView**: `push_back(Op::LoadView(...))` — `compile()` adds thread indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zyx::kernel::{Kernel, Scope, MemLayout, DeviceId};
    /// use zyx::DType;
    ///
    /// let mut kernel = Kernel::new(DeviceId::AUTO);
    /// let n = 4;
    /// let inp = kernel.define(DType::F32, Scope::Global, true, n);
    /// let gidx = kernel.gidx(0, n);
    /// let loaded = kernel.load(inp, gidx, MemLayout::Scalar);
    /// let doubled = kernel.add(loaded, loaded);
    /// let out = kernel.define(DType::F32, Scope::Global, false, n);
    /// kernel.store(out, doubled, gidx, MemLayout::Scalar);
    /// ```
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            outputs: Vec::new(),
            loads: Vec::new(),
            stores: Vec::new(),
            ops: Slab::new(),
            head: OpId::NULL,
            tail: OpId::NULL,
            device_id,
            custom_kernel_id: None,
        }
    }

    /// Compute dtypes and reference counts for all operations.
    pub(crate) fn compute_dtypes_and_rcs(&self) -> (Map<OpId, (DType, MemLayout)>, Map<OpId, u32>) {
        let mut rcs: Map<OpId, u32> = Map::with_capacity_and_hasher(self.ops.len().into(), BuildHasherDefault::new());
        let mut dtypes: Map<OpId, (DType, MemLayout)> = Map::with_capacity_and_hasher(100, BuildHasherDefault::new());

        let mut op_id = self.head;
        while !op_id.is_null() {
            match &self.ops[op_id].op {
                Op::ConstView { .. } | Op::StoreView { .. } | Op::LoadView { .. } | Op::Move { .. } | Op::Reduce { .. } => {
                    unreachable!()
                }
                Op::Const(x) => {
                    dtypes.insert(op_id, (x.dtype(), MemLayout::Scalar));
                }
                &Op::Define { dtype, .. } => {
                    dtypes.insert(op_id, (dtype, MemLayout::Scalar));
                }
                &Op::Load { src, index, layout } => {
                    dtypes.insert(op_id, (dtypes[&src].0, layout));
                    *rcs.entry(index).or_insert(0) += 1;
                }
                &Op::Store { dst, x, index, layout } => {
                    debug_assert_eq!(dtypes[&x].1, layout);
                    dtypes.insert(op_id, dtypes[&x]);
                    *rcs.entry(dst).or_insert(0) += 1;
                    *rcs.entry(x).or_insert(0) += 1;
                    *rcs.entry(index).or_insert(0) += 1;
                }
                &Op::Cast { x, dtype } => {
                    dtypes.insert(op_id, (dtype, dtypes[&x].1));
                    *rcs.entry(x).or_insert(0) += 1;
                }
                &Op::Unary { x, .. } => {
                    dtypes.insert(op_id, dtypes[&x]);
                    *rcs.entry(x).or_insert(0) += 1;
                }
                &Op::Binary { x, y, bop } => {
                    let dtype = if bop.returns_bool() {
                        (DType::Bool, dtypes[&x].1)
                    } else {
                        dtypes[&x]
                    };
                    dtypes.insert(op_id, dtype);
                    *rcs.entry(x).or_insert(0) += 1;
                    *rcs.entry(y).or_insert(0) += 1;
                }
                Op::Vectorize { ops } => {
                    let dtype = dtypes[&ops[0]];
                    dtypes.insert(op_id, (dtype.0, MemLayout::Vector(ops.len().try_into().unwrap())));
                    for &x in ops {
                        *rcs.entry(x).or_insert(0) += 1;
                    }
                }
                Op::Devectorize { vec, idx: _ } => {
                    let dtype = dtypes[vec];
                    dtypes.insert(op_id, (dtype.0, MemLayout::Scalar));
                    *rcs.entry(*vec).or_insert(0) += 1;
                }
                Op::Wmma {
                    dims: _,
                    layout: _,
                    dtype,
                    a,
                    b,
                    c,
                } => {
                    let out_dtype = match dtype {
                        MMADType::f16_f16_f16_f32 => DType::F32,
                    };
                    dtypes.insert(op_id, (out_dtype, MemLayout::Vector(4)));
                    *rcs.entry(*a).or_insert(0) += 1;
                    *rcs.entry(*b).or_insert(0) += 1;
                    *rcs.entry(*c).or_insert(0) += 1;
                }
                &Op::Mad { x, y, z } => {
                    dtypes.insert(op_id, dtypes[&x]);
                    *rcs.entry(x).or_insert(0) += 1;
                    *rcs.entry(y).or_insert(0) += 1;
                    *rcs.entry(z).or_insert(0) += 1;
                }
                Op::Index { .. } | Op::Loop { .. } => {
                    dtypes.insert(op_id, (IDX_T, MemLayout::Scalar));
                }
                &Op::If { condition } => {
                    *rcs.entry(condition).or_insert(0) += 1;
                }
                Op::Barrier { .. } | Op::EndIf | Op::EndLoop => {}
            }
            op_id = self.next_op(op_id);
        }
        (dtypes, rcs)
    }

    /// Resolve the dtype of an operation's result by walking the IR.
    pub(crate) fn dtype(&self, op_id: OpId) -> DType {
        match &self.ops[op_id].op {
            Op::Const(c) => c.dtype(),
            Op::Define { dtype, .. } => *dtype,
            Op::Cast { dtype, .. } => *dtype,
            Op::Index { .. } => IDX_T,
            Op::Load { src, .. } => self.dtype(*src),
            Op::Unary { x, .. } => self.dtype(*x),
            Op::Binary { x, .. } => self.dtype(*x),
            Op::Mad { x, .. } => self.dtype(*x),
            Op::Wmma { dtype, .. } => match dtype {
                MMADType::f16_f16_f16_f32 => DType::F32,
            },
            Op::Vectorize { ops } => self.dtype(ops[0]),
            Op::Devectorize { vec, .. } => self.dtype(*vec),
            Op::Store { x, .. } => self.dtype(*x),
            Op::StoreView { src, .. } => self.dtype(*src),
            Op::ConstView(b) => b.0.dtype(),
            Op::LoadView(b) => b.0,
            Op::Move { x, .. } => self.dtype(*x),
            Op::Reduce { x, .. } => self.dtype(*x),
            Op::Barrier { .. } | Op::If { .. } | Op::EndIf | Op::EndLoop | Op::Loop { .. } => {
                panic!("operation has no dtype")
            }
        }
    }

    /// Compile the kernel. Consumes `self`.
    ///
    /// Runs [`Kernel::unfold_movement_ops`] and [`Kernel::verify`] before compilation.
    ///
    /// # Panics
    ///
    /// If the kernel IR is invalid (see [`Kernel::verify`]).
    ///
    /// # Errors
    ///
    /// If device initialization or compilation fails.
    ///
    /// # Example
    ///
    /// Build a simple element-wise doubling kernel using [`DeviceId::AUTO`] to
    /// let the runtime pick the first available device:
    ///
    /// ```rust
    /// use zyx::kernel::{Kernel, Scope, MemLayout, DeviceId};
    /// use zyx::{DType, Tensor, ZyxError};
    ///
    /// let mut kernel = Kernel::new(DeviceId::AUTO);
    /// let n = 4;
    /// let inp = kernel.define(DType::F32, Scope::Global, true, n);
    /// let gidx = kernel.gidx(0, n);
    /// let loaded = kernel.load(inp, gidx, MemLayout::Scalar);
    /// let doubled = kernel.add(loaded, loaded);
    /// let out = kernel.define(DType::F32, Scope::Global, false, n);
    /// kernel.store(out, doubled, gidx, MemLayout::Scalar);
    ///
    /// let compiled = kernel.compile()?;
    /// let x = Tensor::from([1.0f32, 2.0, 3.0, 4.0]);
    /// let result = compiled.forward(&[&x], [n]);
    /// let data: Vec<f32> = result.try_into().unwrap();
    /// assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    /// # Ok::<_, ZyxError>(())
    /// ```
    pub fn compile(mut self) -> Result<CompiledKernel, crate::ZyxError> {
        self.unfold_movement_ops();
        self.sort_global_defines();
        self.dead_code_elimination();
        self.verify();

        let device_id = self.device_id;
        let dtype = self
            .ops
            .values()
            .find_map(|n| {
                if let Op::Define {
                    dtype,
                    scope: Scope::Global,
                    ro: false,
                    ..
                } = n.op
                {
                    Some(dtype)
                } else {
                    None
                }
            })
            .expect("custom kernel must have exactly one mutable global define");
        let mut rt = crate::RT.lock();
        rt.initialize_devices()?;
        let device_id = if device_id == DeviceId::AUTO {
            rt.devices.ids().next().expect("no devices available")
        } else {
            device_id
        };
        if rt.debug.ir() {
            self.debug();
        }
        let debug_asm = rt.debug.asm();
        let program_id = rt.devices[device_id].compile(&self, debug_asm)?;
        let prog = crate::backend::ProgramId {
            device: device_id,
            program: program_id,
        };
        let kid = rt.kernel_cache.insert_kernel(self);
        rt.kernel_cache.programs.insert((kid, device_id), program_id);
        Ok(crate::kernel::custom::CompiledKernel {
            program: prog,
            dtype,
            kernel_id: kid,
        })
    }

    /// Run autotuning then compile the kernel.
    /// Consumes the kernel.
    ///
    /// TODO: real autotune — must allocate temp buffers and call [`Kernel::autotune_`].
    /// For now this is identical to [`Kernel::compile`].
    #[allow(unused)]
    fn autotune(self) -> Result<CompiledKernel, crate::ZyxError> {
        self.compile()
    }

    /// Load a contiguous tensor from device memory.
    pub fn load_contiguous(&mut self, dtype: DType, shape: &[Dim]) -> OpId {
        self.push_back(Op::LoadView(Box::new((dtype, View::contiguous(shape)))))
    }

    /// Permute tensor axes.
    pub fn permute(&mut self, x: OpId, axes: &[UAxis]) -> OpId {
        let axes = axes.to_vec();
        let shape = self.shape();
        let shape = crate::shape::permute(&shape, &axes);
        self.push_back(Op::Move {
            x,
            mop: Box::new(MoveOp::Permute { axes, shape }),
        })
    }

    /// Reshape tensor.
    pub fn reshape(&mut self, x: OpId, shape: &[Dim]) -> OpId {
        let shape = shape.to_vec();
        self.push_back(Op::Move {
            x,
            mop: Box::new(MoveOp::Reshape { shape }),
        })
    }

    /// Expand tensor (adds singleton dims).
    pub fn expand(&mut self, x: OpId, shape: &[Dim]) -> OpId {
        let shape = shape.to_vec();
        self.push_back(Op::Move {
            x,
            mop: Box::new(MoveOp::Expand { shape }),
        })
    }

    /// Pad tensor with zeros.
    pub fn pad(&mut self, x: OpId, padding: &[(i64, i64)]) -> OpId {
        let padding = padding.to_vec();
        let mut shape = self.shape();
        crate::shape::pad(&mut shape, &padding);
        self.push_back(Op::Move {
            x,
            mop: Box::new(MoveOp::Pad { padding, shape }),
        })
    }

    /// Sum over the last `n_axes` dimensions.
    pub fn reduce_sum(&mut self, x: OpId, n_axes: usize) -> OpId {
        self.push_back(Op::Reduce {
            x,
            rop: BOp::Add,
            n_axes,
        })
    }

    /// Max over the last `n_axes` dimensions.
    pub fn reduce_max(&mut self, x: OpId, n_axes: usize) -> OpId {
        self.push_back(Op::Reduce {
            x,
            rop: BOp::Max,
            n_axes,
        })
    }

    /// Product over the last `n_axes` dimensions.
    pub fn reduce_prod(&mut self, x: OpId, n_axes: usize) -> OpId {
        self.push_back(Op::Reduce {
            x,
            rop: BOp::Mul,
            n_axes,
        })
    }

    /// Store tensor to contiguous device memory.
    pub fn store_contiguous(&mut self, src: OpId, dtype: DType) {
        self.push_back(Op::StoreView { src, dtype });
    }

    /// Constant data value (uses natural dtype).
    /// For index constants, use [`Kernel::const_idx`].
    pub fn const_val<T: crate::scalar::Scalar>(&mut self, val: T) -> OpId {
        self.push_back(Op::Const(Constant::new(val)))
    }

    /// Constant index value (normalized to index type).
    /// For data constants, use [`Kernel::const_val`].
    pub fn const_idx<T: crate::scalar::Scalar>(&mut self, val: T) -> OpId {
        self.push_back(Op::Const(Constant::idx(val)))
    }

    /// Create multiple constant indices.
    pub fn const_idxs<const N: usize>(&mut self, vals: [u32; N]) -> [OpId; N] {
        core::array::from_fn(|i| self.const_idx(vals[i]))
    }

    /// Define a tensor buffer.
    pub fn define(&mut self, dtype: DType, scope: Scope, ro: bool, len: Dim) -> OpId {
        self.push_back(Op::Define { dtype, scope, ro, len })
    }

    /// Global thread index.
    pub fn gidx(&mut self, axis: u32, len: Dim) -> OpId {
        self.push_back(Op::Index {
            len,
            scope: Scope::Global,
            axis,
        })
    }

    /// Local thread index.
    pub fn lidx(&mut self, axis: u32, len: Dim) -> OpId {
        self.push_back(Op::Index {
            len,
            scope: Scope::Local,
            axis,
        })
    }

    /// Store `x` to `dst` at `index`.
    pub fn store(&mut self, dst: OpId, x: OpId, index: OpId, layout: MemLayout) {
        self.push_back(Op::Store { dst, x, index, layout });
    }

    /// Load from `src` at `index`.
    pub fn load(&mut self, src: OpId, index: OpId, layout: MemLayout) -> OpId {
        self.push_back(Op::Load { src, index, layout })
    }

    /// Begin a loop.
    pub fn loop_(&mut self, len: Dim) -> OpId {
        self.push_back(Op::Loop { len })
    }

    /// End the current loop.
    pub fn end_loop(&mut self) {
        self.push_back(Op::EndLoop);
    }

    pub(crate) fn unary(&mut self, x: OpId, uop: UOp) -> OpId {
        self.push_back(Op::Unary { x, uop })
    }

    /// `-x`
    pub fn neg(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Neg)
    }

    /// `~x`
    pub fn bit_not(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::BitNot)
    }

    /// `e^x`
    pub fn exp(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Exp)
    }

    /// `2^x`
    pub fn exp2(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Exp2)
    }

    /// `ln(x)`
    pub fn ln(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Ln)
    }

    /// `log2(x)`
    pub fn log2(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Log2)
    }

    /// `1/x`
    pub fn reciprocal(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Reciprocal)
    }

    /// `sqrt(x)`
    pub fn sqrt(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Sqrt)
    }

    /// `sin(x)`
    pub fn sin(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Sin)
    }

    /// `cos(x)`
    pub fn cos(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Cos)
    }

    /// `floor(x)`
    pub fn floor(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Floor)
    }

    /// `trunc(x)`
    pub fn trunc(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Trunc)
    }

    /// `|x|`
    pub fn abs(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Abs)
    }

    pub(crate) fn binary(&mut self, x: OpId, y: OpId, bop: BOp) -> OpId {
        self.push_back(Op::Binary { x, y, bop })
    }

    /// `x + y`
    pub fn add(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Add)
    }

    /// `x - y`
    pub fn sub(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Sub)
    }

    /// `x * y`
    pub fn mul(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Mul)
    }

    /// `x / y`
    pub fn div(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Div)
    }

    /// `x^y`
    pub fn pow(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Pow)
    }

    /// `x % y`
    pub fn mod_(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Mod)
    }

    /// `x < y`
    pub fn cmplt(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Cmplt)
    }

    /// `x > y`
    pub fn cmpgt(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Cmpgt)
    }

    /// `max(x, y)`
    pub fn max(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Max)
    }

    /// `x | y`
    pub fn or_(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Or)
    }

    /// `x & y`
    pub fn and_(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::And)
    }

    /// `x ^ y`
    pub fn bit_xor(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::BitXor)
    }

    /// `x | y`
    pub fn bit_or(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::BitOr)
    }

    /// `x & y`
    pub fn bit_and(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::BitAnd)
    }

    /// `x << y`
    pub fn bit_shift_left(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::BitShiftLeft)
    }

    /// `x >> y`
    pub fn bit_shift_right(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::BitShiftRight)
    }

    /// `x != y`
    pub fn not_eq(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::NotEq)
    }

    /// `x == y`
    pub fn eq(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Eq)
    }

    /// Warp matrix multiply-accumulate.
    pub fn wmma(&mut self, dims: MMADims, layout: MMALayout, dtype: MMADType, a: OpId, b: OpId, c: OpId) -> OpId {
        self.push_back(Op::Wmma {
            dims,
            layout,
            dtype,
            a,
            b,
            c,
        })
    }

    /// Vectorize ops into a single value.
    pub fn vectorize(&mut self, ops: Vec<OpId>) -> OpId {
        self.push_back(Op::Vectorize { ops })
    }

    /// Extract one element from a vectorized value.
    pub fn devectorize_one(&mut self, vec: OpId, idx: usize) -> OpId {
        self.push_back(Op::Devectorize { vec, idx })
    }

    /// Extract all elements from a vectorized value.
    pub fn devectorize<const N: usize>(&mut self, vec: OpId) -> [OpId; N] {
        core::array::from_fn(|i| self.devectorize_one(vec, i))
    }

    /// Local thread barrier.
    pub fn local_barrier(&mut self) {
        self.push_back(Op::Barrier { scope: Scope::Local });
    }

    /// Global thread barrier.
    pub fn global_barrier(&mut self) {
        self.push_back(Op::Barrier { scope: Scope::Global });
    }

    /// Begin conditional block.
    pub fn if_(&mut self, condition: OpId) {
        self.push_back(Op::If { condition });
    }

    /// End conditional block.
    pub fn end_if(&mut self) {
        self.push_back(Op::EndIf);
    }

    /// Cast to a different dtype.
    pub fn cast(&mut self, x: OpId, dtype: DType) -> OpId {
        self.push_back(Op::Cast { x, dtype })
    }

    /// `x * y + z`
    pub fn mad(&mut self, x: OpId, y: OpId, z: OpId) -> OpId {
        self.push_back(Op::Mad { x, y, z })
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
        let op_node = OpNode {
            prev,
            next: before_id,
            op,
        };
        let op_id = self.ops.push(op_node);
        self.ops[before_id].prev = op_id;
        if prev.is_null() {
            self.head = op_id;
        } else {
            self.ops[prev].next = op_id;
        }
        op_id
    }

    pub(crate) fn insert_after(&mut self, after_id: OpId, op: Op) -> OpId {
        debug_assert!(!after_id.is_null());
        debug_assert!(!self.ops.is_empty());

        let next = self.ops[after_id].next;
        let op_node = OpNode {
            prev: after_id,
            next,
            op,
        };
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

    /// Iterate over all operations in the kernel.
    ///
    /// Returns an iterator over all operations without any ordering guarantees.
    pub(crate) fn iter_unordered(&self) -> impl Iterator<Item = (OpId, &Op)> {
        self.ops.iter().map(|(id, node)| (id, &node.op))
    }

    /// Sort global defines to the beginning of the operation chain.
    ///
    /// Moves all `Define` operations with `Scope::Global` to appear at the beginning.
    pub(crate) fn sort_global_defines(&mut self) {
        let mut insert_after = OpId::NULL;
        let mut op_id = self.head;
        while !op_id.is_null() {
            if matches!(
                self.ops[op_id].op,
                Op::Define {
                    scope: Scope::Global,
                    ..
                }
            ) {
                insert_after = op_id;
            } else {
                break;
            }
            op_id = self.next_op(op_id);
        }
        if insert_after.is_null() || op_id.is_null() {
            return;
        }
        while !op_id.is_null() {
            let next = self.next_op(op_id);
            if matches!(
                self.ops[op_id].op,
                Op::Define {
                    scope: Scope::Global,
                    ..
                }
            ) {
                self.move_op_after(op_id, insert_after);
                insert_after = op_id;
            }
            op_id = next;
        }
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
                Op::ConstView(x) => {
                    let shape = x.1.shape();
                    Info {
                        shape,
                        flops: 0,
                        mem_read: 0,
                        mem_write: 0,
                    }
                }
                Op::LoadView(x) => {
                    let (dtype, view) = x.as_ref();
                    let shape = view.shape();
                    let mem_read = view.original_numel() * u64::from(dtype.bit_size()) / 8;
                    Info {
                        shape,
                        flops: 0,
                        mem_read,
                        mem_write: 0,
                    }
                }
                Op::StoreView { src, dtype } => {
                    let Info { shape, .. } = stack[src].clone();
                    let mem_write = shape.iter().product::<Dim>() * u64::from(dtype.bit_size()) / 8;
                    Info {
                        shape,
                        flops: 0,
                        mem_read: 0,
                        mem_write,
                    }
                }
                Op::Move { mop, .. } => match mop.as_ref() {
                    MoveOp::Reshape { shape, .. }
                    | MoveOp::Expand { shape }
                    | MoveOp::Permute { shape, .. }
                    | MoveOp::Pad { shape, .. } => Info {
                        shape: shape.clone(),
                        flops: 0,
                        mem_read: 0,
                        mem_write: 0,
                    },
                },
                Op::Reduce { x, n_axes, .. } => {
                    let Info { mut shape, .. } = stack[x].clone();
                    let rd: Dim = shape[shape.len() - n_axes..].iter().product();
                    shape.truncate(shape.len() - n_axes);
                    let n: Dim = shape.iter().product();
                    let flops = n * (rd - 1);
                    let flops = flops as u64;
                    Info {
                        shape,
                        flops,
                        mem_read: 0,
                        mem_write: 0,
                    }
                }
                Op::Cast { x, .. } => {
                    let Info { shape, .. } = stack[x].clone();
                    let flops = 0; // Cast is not computation
                    Info {
                        shape,
                        flops,
                        mem_read: 0,
                        mem_write: 0,
                    }
                }
                Op::Unary { x, .. } => {
                    let Info { shape, .. } = stack[x].clone();
                    let flops = shape.iter().product::<Dim>() as u64;
                    Info {
                        shape,
                        flops,
                        mem_read: 0,
                        mem_write: 0,
                    }
                }
                Op::Binary { x, .. } => {
                    let Info { shape, .. } = stack[x].clone();
                    let flops = shape.iter().product::<Dim>() as u64;
                    Info {
                        shape,
                        flops,
                        mem_read: 0,
                        mem_write: 0,
                    }
                }
                Op::Wmma { .. }
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

    /// Check if the kernel contains any store operations.
    pub(crate) fn contains_stores(&self) -> bool {
        self.ops.values().any(|x| matches!(x.op, Op::StoreView { .. }))
    }

    /// Check if the kernel is a reduction kernel.
    pub(crate) fn is_reduce(&self) -> bool {
        self.ops.values().any(|x| matches!(x.op, Op::Reduce { .. }))
    }

    /// Shape of the kernel output.
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
                Op::Move { mop, .. } => {
                    let shape = match mop.as_ref() {
                        MoveOp::Reshape { shape, .. }
                        | MoveOp::Expand { shape }
                        | MoveOp::Permute { shape, .. }
                        | MoveOp::Pad { shape, .. } => shape,
                    };
                    let shape: Vec<Dim> = shape[..shape.len() - reduce_dims].into();
                    if shape.is_empty() {
                        return vec![1];
                    }
                    return shape;
                }
                _ => {}
            }
            op_id = self.prev_op(op_id);
        }
        Vec::new()
    }

    #[allow(unused)]
    /// Check if a reshape is contiguous.
    pub(crate) fn is_reshape_contiguous(&self, range: std::ops::Range<UAxis>, shape: &[Dim]) -> bool {
        self.ops.values().all(|node| match &node.op {
            Op::ConstView(x) => x.1.is_reshape_contiguous(range.clone(), shape),
            Op::LoadView(x) => x.1.is_reshape_contiguous(range.clone(), shape),
            _ => true,
        })
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
                            indices.insert(x, (len, 1));
                            params.push((y, scale));
                        } else if let Op::Index { len, .. } = self.ops[x].op {
                            indices.insert(x, (len, 1));
                            params.push((y, scale));
                        } else if let Op::Loop { len, .. } = self.ops[y].op {
                            indices.insert(y, (len, 1));
                            params.push((x, scale));
                        } else if let Op::Index { len, .. } = self.ops[y].op {
                            indices.insert(y, (len, 1));
                            params.push((x, scale));
                        } else {
                            params.push((x, scale));
                            params.push((y, scale));
                        }
                    }
                    if bop == BOp::Mul {
                        match (&self.ops[x].op, &self.ops[y].op) {
                            (Op::Loop { len, .. }, Op::Const(c)) | (Op::Index { len, .. }, Op::Const(c)) => {
                                indices.insert(x, (*len, c.as_dim().unwrap() * scale));
                            }
                            (Op::Const(c), Op::Loop { len, .. }) | (Op::Const(c), Op::Index { len, .. }) => {
                                indices.insert(y, (*len, c.as_dim().unwrap() * scale));
                            }
                            _ => {} //op => println!("op={op:?}"),
                        }
                    }
                    if bop == BOp::BitShiftLeft {
                        match (&self.ops[x].op, &self.ops[y].op) {
                            (Op::Loop { len, .. }, Op::Const(c)) | (Op::Index { len, .. }, Op::Const(c)) => {
                                indices.insert(x, (*len, (1u64 << c.as_dim().unwrap()) * scale));
                            }
                            (Op::Const(c), Op::Loop { len, .. }) | (Op::Const(c), Op::Index { len, .. }) => {
                                indices.insert(y, (*len, (1u64 << c.as_dim().unwrap()) * scale));
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
                    if let Some(len) = match &self.ops[z].op {
                        Op::Loop { len, .. } | Op::Index { len, .. } => Some(*len),
                        _ => None,
                    } {
                        indices.insert(z, (len, 1));
                    } else {
                        params.push((z, scale));
                    }
                    match (&self.ops[x].op, &self.ops[y].op) {
                        (Op::Loop { len: dim, .. }, Op::Const(c))
                        | (Op::Index { len: dim, .. }, Op::Const(c))
                        | (Op::Const(c), Op::Loop { len: dim, .. })
                        | (Op::Const(c), Op::Index { len: dim, .. }) => {
                            let target = if matches!(self.ops[x].op, Op::Loop { .. } | Op::Index { .. }) {
                                x
                            } else {
                                y
                            };
                            indices.insert(target, (*dim, c.as_dim().unwrap() * scale));
                        }
                        _ => {}
                    }
                }
                Op::Const(c) => {
                    indices.insert(OpId::NULL, (0, c.as_dim().unwrap() * scale));
                }
                _ => {}
            }
        }

        indices
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
        let op_node = OpNode {
            prev: self.tail,
            next: OpId::NULL,
            op,
        };
        let op_id = self.ops.push(op_node);
        if self.head.is_null() {
            self.head = op_id;
        } else {
            self.ops[self.tail].next = op_id;
        }
        self.tail = op_id;
        op_id
    }

    /// Remove the first output tensor.
    pub(crate) fn remove_first_output(&mut self, x: TensorId) {
        //println!("removing tensor {x} from kernel {kid:?}");
        let outputs = &mut self.outputs;
        outputs.iter().position(|elem| *elem == x).map(|i| outputs.remove(i));
    }

    /// Drop unused operations from the kernel.
    ///
    /// Removes operations that are not required by the outputs.
    pub(crate) fn drop_unused_ops(&mut self, visited: &Map<TensorId, (KMKernelId, OpId)>) {
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

    /// Get all required operations for a set of parameters.
    pub(crate) fn get_required_ops(&self, mut params: Vec<OpId>) -> Set<OpId> {
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
                    | Op::Wmma { .. }
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
                    | Op::EndLoop => unreachable!(),
                }
            }
        }
        required
    }

    /// Get all global indices used in the kernel.
    pub(crate) fn get_global_indices(&self) -> std::collections::BTreeMap<u32, OpId> {
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

    /// Renumber indices to be in order.
    pub(crate) fn renumber_indices(&mut self) {
        let mut indices = std::collections::BTreeMap::new();
        indices.insert(Scope::Global, std::collections::BTreeMap::new());
        indices.insert(Scope::Local, std::collections::BTreeMap::new());
        for (op_id, op_node) in self.ops.iter() {
            if let Op::Index { scope, axis, .. } = op_node.op {
                indices.get_mut(&scope).unwrap().insert(axis, op_id);
            }
        }
        for (_, scoped_indices) in indices {
            let mut ax = 0;
            for &idx_id in scoped_indices.values() {
                let Op::Index { axis, .. } = &mut self.ops[idx_id].op else {
                    unreachable!()
                };
                *axis = ax;
                ax += 1;
            }
        }
    }
}

impl MMADims {
    /// Decompose MMAD dimensions into m, n, k components.
    pub const fn decompose_mnk(self) -> (u64, u64, u64) {
        match self {
            MMADims::m8n8k16 => (8, 8, 16),
            MMADims::m16n8k8 => (16, 8, 8),
            MMADims::m16n8k16 => (16, 8, 16),
        }
    }
}

// Manual SerBin/DeBin implementations for private enums

impl SerBin for UOp {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        match self {
            UOp::Neg => output.push(0),
            UOp::BitNot => output.push(1),
            UOp::Exp => output.push(2),
            UOp::Exp2 => output.push(3),
            UOp::Ln => output.push(4),
            UOp::Log2 => output.push(5),
            UOp::Reciprocal => output.push(6),
            UOp::Sqrt => output.push(7),
            UOp::Sin => output.push(8),
            UOp::Cos => output.push(9),
            UOp::Floor => output.push(10),
            UOp::Trunc => output.push(11),
            UOp::Abs => output.push(12),
        }
    }
}

impl DeBin for UOp {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        let tag = bytes[*offset];
        *offset += 1;
        match tag {
            0 => Ok(UOp::Neg),
            1 => Ok(UOp::BitNot),
            2 => Ok(UOp::Exp),
            3 => Ok(UOp::Exp2),
            4 => Ok(UOp::Ln),
            5 => Ok(UOp::Log2),
            6 => Ok(UOp::Reciprocal),
            7 => Ok(UOp::Sqrt),
            8 => Ok(UOp::Sin),
            9 => Ok(UOp::Cos),
            10 => Ok(UOp::Floor),
            11 => Ok(UOp::Trunc),
            12 => Ok(UOp::Abs),
            _ => Err(nanoserde::DeBinErr::new(*offset - 1, 1, bytes.len())),
        }
    }
}

impl SerBin for BOp {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        match self {
            BOp::Add => output.push(0),
            BOp::Sub => output.push(1),
            BOp::Mul => output.push(2),
            BOp::Div => output.push(3),
            BOp::Pow => output.push(4),
            BOp::Mod => output.push(5),
            BOp::Cmplt => output.push(6),
            BOp::Cmpgt => output.push(7),
            BOp::Max => output.push(8),
            BOp::Or => output.push(9),
            BOp::And => output.push(10),
            BOp::BitXor => output.push(11),
            BOp::BitOr => output.push(12),
            BOp::BitAnd => output.push(13),
            BOp::BitShiftLeft => output.push(14),
            BOp::BitShiftRight => output.push(15),
            BOp::NotEq => output.push(16),
            BOp::Eq => output.push(17),
        }
    }
}

impl DeBin for BOp {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        let tag = bytes[*offset];
        *offset += 1;
        match tag {
            0 => Ok(BOp::Add),
            1 => Ok(BOp::Sub),
            2 => Ok(BOp::Mul),
            3 => Ok(BOp::Div),
            4 => Ok(BOp::Pow),
            5 => Ok(BOp::Mod),
            6 => Ok(BOp::Cmplt),
            7 => Ok(BOp::Cmpgt),
            8 => Ok(BOp::Max),
            9 => Ok(BOp::Or),
            10 => Ok(BOp::And),
            11 => Ok(BOp::BitXor),
            12 => Ok(BOp::BitOr),
            13 => Ok(BOp::BitAnd),
            14 => Ok(BOp::BitShiftLeft),
            15 => Ok(BOp::BitShiftRight),
            16 => Ok(BOp::NotEq),
            17 => Ok(BOp::Eq),
            _ => Err(nanoserde::DeBinErr::new(*offset - 1, 1, bytes.len())),
        }
    }
}

impl SerBin for MoveOp {
    fn ser_bin(&self, output: &mut Vec<u8>) {
        match self {
            MoveOp::Reshape { shape } => {
                output.push(0);
                shape.ser_bin(output);
            }
            MoveOp::Expand { shape } => {
                output.push(1);
                shape.ser_bin(output);
            }
            MoveOp::Permute { axes, shape } => {
                output.push(2);
                axes.ser_bin(output);
                shape.ser_bin(output);
            }
            MoveOp::Pad { padding, shape } => {
                output.push(3);
                padding.ser_bin(output);
                shape.ser_bin(output);
            }
        }
    }
}

impl DeBin for MoveOp {
    fn de_bin(offset: &mut usize, bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        let tag = bytes[*offset];
        *offset += 1;
        match tag {
            0 => {
                let shape = Vec::<Dim>::de_bin(offset, bytes)?;
                Ok(MoveOp::Reshape { shape })
            }
            1 => {
                let shape = Vec::<Dim>::de_bin(offset, bytes)?;
                Ok(MoveOp::Expand { shape })
            }
            2 => {
                let axes = Vec::<UAxis>::de_bin(offset, bytes)?;
                let shape = Vec::<Dim>::de_bin(offset, bytes)?;
                Ok(MoveOp::Permute { axes, shape })
            }
            3 => {
                let padding = Vec::<(i64, i64)>::de_bin(offset, bytes)?;
                let shape = Vec::<Dim>::de_bin(offset, bytes)?;
                Ok(MoveOp::Pad { padding, shape })
            }
            _ => Err(nanoserde::DeBinErr::new(*offset - 1, 1, bytes.len())),
        }
    }
}
