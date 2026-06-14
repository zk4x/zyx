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
    kernel::custom::CompiledKernel,
    kernel_cache::KernelId,
    kernelize::KMKernelId,
    shape::{Dim, UAxis},
    slab::{Slab, SlabId},
    tensor::TensorId,
};
use nanoserde::{DeBin, SerBin};
use std::{fmt::Display, hash::Hash};

mod algebraic;
/// Autotuning optimizations for kernel compilation.
pub(crate) mod autotune;
/// Cost estimation for kernel selection.
mod cost;
/// Custom kernel compilation for GPU-specific operations.
pub mod custom;
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
pub const IDX_T: DType = DType::U32;

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
    fn ser_bin(&self, _output: &mut Vec<u8>) {
        todo!()
    }
}

impl DeBin for OpNode {
    fn de_bin(_offset: &mut usize, _bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        todo!()
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
    Vector(u8),
    /// Tile layout: tile of `x` × `y` elements with stride
    Tile {
        /// Width of the tile
        x: u8,
        /// Height of the tile
        y: u8,
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
    fn ser_bin(&self, _output: &mut Vec<u8>) {
        todo!()
    }
}

impl DeBin for Op {
    fn de_bin(_offset: &mut usize, _bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        todo!()
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
    /// Use the builder methods (`push_back`, `binary`, `store`, `load`, etc.) to
    /// construct the kernel IR, then finalize with [`Kernel::compile`].
    ///
    /// [`Kernel::compile`] automatically unfolds movement ops (LoadView, StoreView,
    /// Load, Store, Move) via [`Kernel::unfold_movement_ops`] and validates the IR
    /// via [`Kernel::verify`].
    ///
    /// There are two approaches for specifying kernel inputs:
    ///
    /// **A — Explicit global input with manual gidx** (shown below): use
    /// `define(dtype, Scope::Global, true, len)` for each input, then create
    /// [`Kernel::gidx`] / [`Kernel::lidx`] ops for thread indexing.
    ///
    /// **B — LoadView**: use `push_back(Op::LoadView(...))` for inputs and let
    /// `compile()` compute thread indices automatically from the view shape.
    /// Do NOT manually add `gidx` ops — `unfold_movement_ops` adds them.
    ///
    /// Pass [`DeviceId::AUTO`] to let the runtime pick the first available device.
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

    /// Compile the kernel and produce a compiled kernel for repeated execution.
    /// Consumes the kernel.
    ///
    /// Automatically unfolds movement ops ([`Kernel::unfold_movement_ops`])
    /// and validates the IR ([`Kernel::verify`]) before compilation.
    ///
    /// # Panics
    ///
    /// Panics if the kernel IR is invalid (see [`Kernel::verify`]).
    ///
    /// # Errors
    ///
    /// Returns [`crate::ZyxError`] if device initialization or compilation fails.
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
        self.verify();

        let device_id = self.device_id;
        let dtype = self
            .ops
            .values()
            .find_map(|n| {
                if let Op::Define { dtype, scope: Scope::Global, ro: false, .. } = n.op {
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
        let prog = crate::backend::ProgramId { device: device_id, program: program_id };
        let kid = rt.kernel_cache.insert_kernel(self);
        rt.kernel_cache.programs.insert((kid, device_id), program_id);
        Ok(crate::kernel::custom::CompiledKernel { program: prog, dtype, kernel_id: kid })
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
    ///
    /// Creates a `LoadView` operation that loads all elements of a tensor
    /// with the given shape and dtype from device memory.
    pub fn load_contiguous(&mut self, dtype: DType, shape: &[Dim]) -> OpId {
        self.push_back(Op::LoadView(Box::new((dtype, View::contiguous(shape)))))
    }

    /// Permute tensor axes.
    ///
    /// Reorders the dimensions of a tensor by applying the given axis permutation.
    pub fn permute(&mut self, x: OpId, axes: &[UAxis]) -> OpId {
        let axes = axes.to_vec();
        let shape = self.shape();
        let shape = crate::shape::permute(&shape, &axes);
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Permute { axes, shape }) })
    }

    /// Reshape tensor to a new shape.
    ///
    /// Changes the shape of a tensor without changing its underlying data.
    pub fn reshape(&mut self, x: OpId, shape: &[Dim]) -> OpId {
        let shape = shape.to_vec();
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Reshape { shape }) })
    }

    /// Expand tensor to a larger shape.
    ///
    /// Adds singleton dimensions (size 1) to the tensor shape without duplicating any data.
    pub fn expand(&mut self, x: OpId, shape: &[Dim]) -> OpId {
        let shape = shape.to_vec();
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Expand { shape }) })
    }

    /// Pad tensor with zero padding.
    ///
    /// Adds padding to the tensor along specified axes.
    pub fn pad(&mut self, x: OpId, padding: &[(i64, i64)]) -> OpId {
        let padding = padding.to_vec();
        let mut shape = self.shape();
        crate::shape::pad(&mut shape, &padding);
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Pad { padding, shape }) })
    }

    /// Reduce tensor with sum operation.
    ///
    /// Sums over the last `n_axes` dimensions of the tensor.
    pub fn reduce_sum(&mut self, x: OpId, n_axes: usize) -> OpId {
        self.push_back(Op::Reduce { x, rop: BOp::Add, n_axes })
    }

    /// Reduce tensor with max operation.
    ///
    /// Finds the maximum value over the last `n_axes` dimensions of the tensor.
    pub fn reduce_max(&mut self, x: OpId, n_axes: usize) -> OpId {
        self.push_back(Op::Reduce { x, rop: BOp::Max, n_axes })
    }

    /// Reduce tensor with product operation.
    ///
    /// Computes the product of elements over the last `n_axes` dimensions of the tensor.
    pub fn reduce_prod(&mut self, x: OpId, n_axes: usize) -> OpId {
        self.push_back(Op::Reduce { x, rop: BOp::Mul, n_axes })
    }

    /// Store tensor to contiguous device memory.
    ///
    /// Creates a `StoreView` operation that stores a tensor to device memory in row-major order.
    pub fn store_contiguous(&mut self, src: OpId, dtype: DType) {
        self.push_back(Op::StoreView { src, dtype });
    }

    /// Create a constant data value.
    ///
    /// Creates a constant from a scalar using its natural dtype.
    /// Use for computation operands (e.g., `const_val(1.0f32)` in arithmetic).
    /// For index/address constants (strides, offsets), use [`Kernel::const_idx`].
    pub fn const_val<T: crate::scalar::Scalar>(&mut self, val: T) -> OpId {
        self.push_back(Op::Const(Constant::new(val)))
    }

    /// Create a constant index value.
    ///
    /// Creates a constant normalized to the kernel's index type ([`IDX_T`]).
    /// Use for strides, offsets, sizes, and any value fed into index arithmetic.
    /// For general data constants, use [`Kernel::const_val`].
    pub fn const_idx<T: crate::scalar::Scalar>(&mut self, val: T) -> OpId {
        self.push_back(Op::Const(Constant::idx(val)))
    }

    /// Create multiple constant indices in one call.
    pub fn const_idxs<const N: usize>(&mut self, vals: [u32; N]) -> [OpId; N] {
        core::array::from_fn(|i| self.const_idx(vals[i]))
    }

    /// Define a tensor in the kernel.
    ///
    /// Creates a new tensor with the given dtype, scope, and length.
    pub fn define(&mut self, dtype: DType, scope: Scope, ro: bool, len: Dim) -> OpId {
        self.push_back(Op::Define { dtype, scope, ro, len })
    }

    /// Get global thread index.
    ///
    /// Creates an operation that returns the global thread index for the given axis.
    pub fn gidx(&mut self, axis: u32, len: Dim) -> OpId {
        self.push_back(Op::Index { len, scope: Scope::Global, axis })
    }

    /// Get local thread index.
    ///
    /// Creates an operation that returns the local thread index for the given axis.
    pub fn lidx(&mut self, axis: u32, len: Dim) -> OpId {
        self.push_back(Op::Index { len, scope: Scope::Local, axis })
    }

    /// Store a value to device memory.
    ///
    /// Stores the value `x` to the destination tensor `dst` at the position specified by `index`.
    pub fn store(&mut self, dst: OpId, x: OpId, index: OpId, layout: MemLayout) {
        self.push_back(Op::Store { dst, x, index, layout });
    }

    /// Load a value from device memory.
    ///
    /// Loads a value from the source tensor `src` at the position specified by `index`.
    pub fn load(&mut self, src: OpId, index: OpId, layout: MemLayout) -> OpId {
        self.push_back(Op::Load { src, index, layout })
    }

    /// Begin a loop.
    ///
    /// Starts a loop that iterates over the given length.
    pub fn loop_(&mut self, len: Dim) -> OpId {
        self.push_back(Op::Loop { len })
    }

    /// End a loop.
    ///
    /// Ends the most recently started loop.
    pub fn end_loop(&mut self) {
        self.push_back(Op::EndLoop);
    }

    pub(crate) fn unary(&mut self, x: OpId, uop: UOp) -> OpId {
        self.push_back(Op::Unary { x, uop })
    }

    /// Negate a tensor.
    ///
    /// Computes `-x` element-wise.
    pub fn neg(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Neg)
    }

    /// Compute bitwise NOT.
    ///
    /// Computes `~x` element-wise.
    pub fn bit_not(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::BitNot)
    }

    /// Compute exponential function.
    ///
    /// Computes `e^x` element-wise.
    pub fn exp(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Exp)
    }

    /// Compute base-2 exponential.
    ///
    /// Computes `2^x` element-wise.
    pub fn exp2(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Exp2)
    }

    /// Compute natural logarithm.
    ///
    /// Computes `ln(x)` element-wise.
    pub fn ln(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Ln)
    }

    /// Compute base-2 logarithm.
    ///
    /// Computes `log2(x)` element-wise.
    pub fn log2(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Log2)
    }

    /// Compute reciprocal.
    ///
    /// Computes `1/x` element-wise.
    pub fn reciprocal(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Reciprocal)
    }

    /// Compute square root.
    ///
    /// Computes `sqrt(x)` element-wise.
    pub fn sqrt(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Sqrt)
    }

    /// Compute sine function.
    ///
    /// Computes `sin(x)` element-wise.
    pub fn sin(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Sin)
    }

    /// Compute cosine function.
    ///
    /// Computes `cos(x)` element-wise.
    pub fn cos(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Cos)
    }

    /// Compute floor function.
    ///
    /// Computes `floor(x)` element-wise, rounding down to the nearest integer.
    pub fn floor(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Floor)
    }

    /// Compute truncation.
    ///
    /// Computes `trunc(x)` element-wise, removing the fractional part.
    pub fn trunc(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Trunc)
    }

    /// Compute absolute value.
    ///
    /// Computes `|x|` element-wise.
    pub fn abs(&mut self, x: OpId) -> OpId {
        self.unary(x, UOp::Abs)
    }

    pub(crate) fn binary(&mut self, x: OpId, y: OpId, bop: BOp) -> OpId {
        self.push_back(Op::Binary { x, y, bop })
    }

    /// Add two tensors.
    ///
    /// Performs element-wise addition.
    pub fn add(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Add)
    }

    /// Subtract two tensors.
    ///
    /// Performs element-wise subtraction.
    pub fn sub(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Sub)
    }

    /// Multiply two tensors.
    ///
    /// Performs element-wise multiplication.
    pub fn mul(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Mul)
    }

    /// Divide two tensors.
    ///
    /// Performs element-wise division.
    pub fn div(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Div)
    }

    /// Compute power of two tensors.
    ///
    /// Computes `x^y` element-wise.
    pub fn pow(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Pow)
    }

    /// Compute modulo of two tensors.
    ///
    /// Computes `x % y` element-wise.
    pub fn mod_(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Mod)
    }

    /// Compare less than of two tensors.
    ///
    /// Computes `x < y` element-wise, returning a boolean tensor.
    pub fn cmplt(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Cmplt)
    }

    /// Compare greater than of two tensors.
    ///
    /// Computes `x > y` element-wise, returning a boolean tensor.
    pub fn cmpgt(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Cmpgt)
    }

    /// Compute element-wise maximum.
    ///
    /// Computes `max(x, y)` element-wise.
    pub fn max(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Max)
    }

    /// Compute element-wise bitwise OR.
    ///
    /// Computes `x | y` element-wise.
    pub fn or_(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Or)
    }

    /// Compute element-wise bitwise AND.
    ///
    /// Computes `x & y` element-wise.
    pub fn and_(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::And)
    }

    /// Compute element-wise bitwise XOR.
    ///
    /// Computes `x ^ y` element-wise.
    pub fn bit_xor(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::BitXor)
    }

    /// Compute element-wise bitwise OR.
    ///
    /// Computes `x | y` element-wise.
    pub fn bit_or(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::BitOr)
    }

    /// Compute element-wise bitwise AND.
    ///
    /// Computes `x & y` element-wise.
    pub fn bit_and(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::BitAnd)
    }

    /// Compute element-wise left bit shift.
    ///
    /// Computes `x << y` element-wise.
    pub fn bit_shift_left(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::BitShiftLeft)
    }

    /// Compute element-wise right bit shift.
    ///
    /// Computes `x >> y` element-wise.
    pub fn bit_shift_right(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::BitShiftRight)
    }

    /// Compute element-wise not equal comparison.
    ///
    /// Computes `x != y` element-wise, returning a boolean tensor.
    pub fn not_eq(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::NotEq)
    }

    /// Compute element-wise equality comparison.
    ///
    /// Computes `x == y` element-wise, returning a boolean tensor.
    pub fn eq(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Eq)
    }

    /// Execute fused multiply-add (wmma) operation.
    ///
    /// Performs a fused multiply-add operation for matrix multiplication.
    pub fn wmma(&mut self, dims: MMADims, layout: MMALayout, dtype: MMADType, a: OpId, b: OpId, c: OpId) -> OpId {
        self.push_back(Op::Wmma { dims, layout, dtype, a, b, c })
    }

    /// Vectorize multiple operations.
    ///
    /// Combines multiple operations into a single vectorized operation.
    pub fn vectorize(&mut self, ops: Vec<OpId>) -> OpId {
        self.push_back(Op::Vectorize { ops })
    }

    /// Devectorize a vector operation.
    ///
    /// Extracts a single element from a vectorized operation.
    pub fn devectorize_one(&mut self, vec: OpId, idx: usize) -> OpId {
        self.push_back(Op::Devectorize { vec, idx })
    }

    /// Extract all elements from a vectorized operation.
    pub fn devectorize<const N: usize>(&mut self, vec: OpId) -> [OpId; N] {
        core::array::from_fn(|i| self.devectorize_one(vec, i))
    }

    /// Insert a local barrier.
    ///
    /// Synchronizes threads within a local scope.
    pub fn local_barrier(&mut self) {
        self.push_back(Op::Barrier { scope: Scope::Local });
    }

    /// Insert a global barrier.
    ///
    /// Synchronizes all threads globally.
    pub fn global_barrier(&mut self) {
        self.push_back(Op::Barrier { scope: Scope::Global });
    }

    /// Begin a conditional block.
    ///
    /// Starts a conditional block that executes based on the given condition.
    pub fn if_(&mut self, condition: OpId) {
        self.push_back(Op::If { condition });
    }

    /// End a conditional block.
    ///
    /// Ends the most recently started conditional block.
    pub fn end_if(&mut self) {
        self.push_back(Op::EndIf);
    }

    /// Cast a tensor to a different dtype.
    ///
    /// Converts the input tensor to the specified data type.
    pub fn cast(&mut self, x: OpId, dtype: DType) -> OpId {
        self.push_back(Op::Cast { x, dtype })
    }

    /// Compute fused multiply-add (MAD).
    ///
    /// Computes `x * y + z` element-wise in a single fused operation.
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
            if matches!(self.ops[op_id].op, Op::Define { scope: Scope::Global, .. }) {
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
            if matches!(self.ops[op_id].op, Op::Define { scope: Scope::Global, .. }) {
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
                    Info { shape, flops: 0, mem_read: 0, mem_write: 0 }
                }
                Op::LoadView(x) => {
                    let (dtype, view) = x.as_ref();
                    let shape = view.shape();
                    let mem_read = view.original_numel() * u64::from(dtype.bit_size()) / 8;
                    Info { shape, flops: 0, mem_read, mem_write: 0 }
                }
                Op::StoreView { src, dtype } => {
                    let Info { shape, .. } = stack[src].clone();
                    let mem_write = shape.iter().product::<Dim>() * u64::from(dtype.bit_size()) / 8;
                    Info { shape, flops: 0, mem_read: 0, mem_write }
                }
                Op::Move { mop, .. } => match mop.as_ref() {
                    MoveOp::Reshape { shape, .. }
                    | MoveOp::Expand { shape }
                    | MoveOp::Permute { shape, .. }
                    | MoveOp::Pad { shape, .. } => Info { shape: shape.clone(), flops: 0, mem_read: 0, mem_write: 0 },
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

    /// Get the shape of the kernel output.
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

    /// Compute work sizes for different scopes.
    pub(crate) fn work_sizes(&self) -> (Vec<Dim>, Vec<Dim>) {
        let mut gws = Vec::new();
        let mut lws = Vec::new();
        for node in self.ops.values() {
            if let Op::Index { len, scope, axis } = node.op {
                let a = axis as usize;
                match scope {
                    Scope::Global => {
                        while gws.len() <= a {
                            gws.push(1);
                        }
                        gws[a] = len;
                    }
                    Scope::Local => {
                        while lws.len() <= a {
                            lws.push(1);
                        }
                        lws[a] = len;
                    }
                    Scope::Register => {}
                }
            }
        }
        (gws, lws)
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

        let mut params = vec![index];
        let mut indices = Map::default();

        while let Some(param) = params.pop() {
            match self.ops[param].op {
                Op::Binary { x, y, bop } => {
                    if bop == BOp::Add {
                        if let Op::Loop { len, .. } = self.ops[x].op {
                            indices.insert(x, (len, 1));
                            params.push(y);
                        } else if let Op::Index { len, .. } = self.ops[x].op {
                            indices.insert(x, (len, 1));
                            params.push(y);
                        } else if let Op::Loop { len, .. } = self.ops[y].op {
                            indices.insert(y, (len, 1));
                            params.push(x);
                        } else if let Op::Index { len, .. } = self.ops[y].op {
                            indices.insert(y, (len, 1));
                            params.push(x);
                        } else {
                            params.push(x);
                            params.push(y);
                        }
                    }
                    if bop == BOp::Mul {
                        match (&self.ops[x].op, &self.ops[y].op) {
                            (Op::Loop { len, .. }, Op::Const(c)) | (Op::Index { len, .. }, Op::Const(c)) => {
                                indices.insert(x, (*len, c.as_dim().unwrap()));
                            }
                            (Op::Const(c), Op::Loop { len, .. }) | (Op::Const(c), Op::Index { len, .. }) => {
                                indices.insert(y, (*len, c.as_dim().unwrap()));
                            }
                            _ => {} //op => println!("op={op:?}"),
                        }
                    }
                    if bop == BOp::BitShiftLeft {
                        match (&self.ops[x].op, &self.ops[y].op) {
                            (Op::Loop { len, .. }, Op::Const(c)) | (Op::Index { len, .. }, Op::Const(c)) => {
                                indices.insert(x, (*len, 1u64 << c.as_dim().unwrap()));
                            }
                            (Op::Const(c), Op::Loop { len, .. }) | (Op::Const(c), Op::Index { len, .. }) => {
                                indices.insert(y, (*len, 1u64 << c.as_dim().unwrap()));
                            }
                            _ => {
                                params.push(x);
                                params.push(y);
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
                        params.push(z);
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
                            indices.insert(target, (*dim, c.as_dim().unwrap()));
                        }
                        _ => {}
                    }
                }
                Op::Const(c) => {
                    indices.insert(OpId::NULL, (0, c.as_dim().unwrap()));
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
                let Op::Index { axis, .. } = &mut self.ops[idx_id].op else { unreachable!() };
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
        match bytes[*offset] {
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
            _ => Err(nanoserde::DeBinErr::new(*offset, 1, bytes.len())),
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
        match bytes[*offset] {
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
            _ => Err(nanoserde::DeBinErr::new(*offset, 1, bytes.len())),
        }
    }
}

impl SerBin for MoveOp {
    fn ser_bin(&self, _output: &mut Vec<u8>) {
        todo!()
    }
}

impl DeBin for MoveOp {
    fn de_bin(_offset: &mut usize, _bytes: &[u8]) -> Result<Self, nanoserde::DeBinErr> {
        todo!()
    }
}
