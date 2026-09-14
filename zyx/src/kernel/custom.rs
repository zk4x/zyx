// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Custom kernel compilation for GPU-specific operations.
//!
//! This module provides support for custom kernel compilation,
//! allowing backends to generate and execute custom kernels
//! for operations not covered by the standard kernel IR.
//!
//! Custom kernels are typically used for:
//!
//! - GPU-specific operations (e.g., WMMA, tensor cores)
//! - Specialized kernels with unique memory access patterns
//! - Backend-specific optimizations
//!
//! The custom kernel system allows backends to compile kernels
//! to their native instruction set and cache them for repeated use.

use std::collections::BTreeSet;
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use std::sync::Arc;

use crate::backend::{BufferId, DeviceInfo, LaunchArg, MemoryPool, ProgramId};
use crate::dtype::Constant;
use crate::error::BackendError;
use crate::graph::{ClassId, EClass, Node, NodeData};
use crate::kernel::{
    BOp, DeviceId, IDX_T, Kernel, MMADType, MMADims, MMALayout, MemLayout, MemScope, MoveOp, Op, OpId, ParamKind, RangeKind, UOp,
    ops::TileReduceKind,
};
use crate::runtime::{KernelId, Runtime, TensorData};
use crate::shape::UAxis;
use crate::slab::{Slab, SlabId};
use crate::tensor::TensorId;
use crate::types::{TinyString, TinyVec};
use crate::{DType, Tensor, ZyxError, bf16, f16, shape::Dim};
use crate::{Dev, Map, Scalar};

/// A compiled kernel ready for repeated execution.
///
/// Dropping a `CompiledKernel` does **not** release the device program:
/// the program registry is intentionally append-only (programs leak, bounded
/// by kernel-hash deduplication), so `ProgramId`s held by egraph nodes and
/// execution plans stay valid for the process lifetime. Releasing a program
/// whose id is still referenced elsewhere could silently launch a *different*
/// program after slab id reuse.
#[derive(Debug)]
pub struct CompiledKernel {
    program: ProgramId,
    inputs: Vec<DType>,
    outputs: Vec<DType>,
}

impl Kernel {
    /// Create a new custom kernel targeting a specific device.
    ///
    /// Two approaches for inputs:
    /// - **Manual gidx**: `param(dtype, shape)` + manual global index computation
    /// - **LoadView**: `push_back(Op::LoadView(...))` — `compile()` adds thread indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zyx::kernel::{Kernel, MemLayout, DeviceId, ParamKind};
    /// use zyx::DType;
    ///
    /// let mut kernel = Kernel::new(DeviceId::AUTO);
    /// let n = 4;
    /// let inp = kernel.param(DType::F32);
    /// let len = kernel.const_idx(n);
    /// let gidx = kernel.group_range(0, len);
    /// let loaded = kernel.load(inp, gidx);
    /// let doubled = kernel.add(loaded, loaded);
    /// let out = kernel.param_mut(DType::F32);
    /// kernel.store(out, doubled, gidx);
    /// ```
    pub fn new(dev: Dev) -> Self {
        let mut rt = crate::RT.lock();
        rt.initialize_backends();
        let device_id = rt.resolve_dev(dev);
        let dev_info = Some(rt.devices[device_id].info());
        Self { ops: Slab::new(), head: OpId::NULL, tail: OpId::NULL, device_id, dev_info, shape_cache: Map::default() }
    }

    /// Compile the kernel. Consumes `self`.
    ///
    /// Runs movement-op unfolding and [`Kernel::verify`] before compilation.
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
    /// use zyx::kernel::{Kernel, MemLayout, DeviceId, ParamKind};
    /// use zyx::{DType, Tensor, ZyxError};
    ///
    /// let mut kernel = Kernel::new(DeviceId::AUTO);
    /// let n = 4;
    /// let inp = kernel.param(DType::F32);
    /// let len = kernel.const_idx(n);
    /// let gidx = kernel.group_range(0, len);
    /// let loaded = kernel.load(inp, gidx);
    /// let doubled = kernel.add(loaded, loaded);
    /// let out = kernel.param_mut(DType::F32);
    /// kernel.store(out, doubled, gidx);
    ///
    /// let compiled = kernel.compile()?;
    /// let x = Tensor::from([1.0f32, 2.0, 3.0, 4.0]);
    /// let result = compiled.forward(&[&x], vec![[n]])?;
    /// let data: Vec<f32> = result.into_iter().next().unwrap().try_into()?;
    /// assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    /// # Ok::<_, ZyxError>(())
    /// ```
    pub fn compile(mut self) -> Result<CompiledKernel, ZyxError> {
        let _compile_start = std::time::Instant::now();
        let mut _t = std::time::Instant::now();
        _t = std::time::Instant::now();
        self.linearize();
        eprintln!("[compile] linearize {}us", _t.elapsed().as_micros());
        // After linearization the parameter shapes are no longer meaningful
        // (the same clear happens inside `linearize` for kernels it processes);
        // clear them here too so kernels that skip linearization (already
        // lowered by hand) don't require shape consts to be ordered before the
        // params that reference them.
        for node in self.ops.values_mut() {
            if let Op::Param { shape, .. } = &mut node.op {
                *shape = OpId::NULL;
            }
        }
        _t = std::time::Instant::now();
        self.instruction_schedule();
        eprintln!("[compile] instruction_schedule {}us", _t.elapsed().as_micros());
        _t = std::time::Instant::now();
        self.constant_folding();
        eprintln!("[compile] constant_folding {}us", _t.elapsed().as_micros());
        _t = std::time::Instant::now();
        self.dead_code_elimination();
        eprintln!("[compile] dead_code_elimination {}us", _t.elapsed().as_micros());
        _t = std::time::Instant::now();
        self.verify();
        eprintln!("[compile] verify {}us", _t.elapsed().as_micros());

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Param { dtype, kind, .. } = self.ops[op_id].op {
                match kind {
                    ParamKind::Variable | ParamKind::Global => inputs.push(dtype),
                    ParamKind::GlobalMut => outputs.push(dtype),
                }
            }
            op_id = self.next_op(op_id);
        }

        if outputs.is_empty() {
            return Err(ZyxError::BackendError(BackendError {
                status: crate::error::ErrorStatus::KernelCompilation,
                context: "Kernel must have at least one output.".to_string().into(),
            }));
        }

        // Get shapes and dtypes for inputs and outputs

        let mut rt = crate::RT.lock();
        rt.initialize_backends();
        let device_id = if self.device_id == DeviceId::AUTO {
            rt.devices.ids().next().expect("no devices available")
        } else {
            self.device_id
        };
        // Bind the resolved device so codegen can read dev_info
        // (mirrors Kernel::new; from_device_id placeholders carry None).
        self.device_id = device_id;
        self.dev_info = Some(rt.devices[device_id].info());
        if rt.debug.ir() {
            self.debug();
        }
        let debug_asm = rt.debug.asm();
        _t = std::time::Instant::now();
        let program_id = rt.devices[device_id].compile(&self, debug_asm)?;
        eprintln!("[compile] device.compile {}us", _t.elapsed().as_micros());
        eprintln!("[compile] total {}us", _compile_start.elapsed().as_micros());
        let program = crate::backend::ProgramId { device_id, program_id };
        Ok(CompiledKernel { program, inputs, outputs })
    }

    /// Permute tensor axes.
    pub fn permute(&mut self, x: OpId, axes: &[UAxis]) -> OpId {
        let axes = axes.into();
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Permute { axes }) })
    }

    /// Reshape tensor. `shape` is the (pre-built) output shape op: a single
    /// const for rank-1, or a `stack` of per-dimension ops otherwise.
    pub fn reshape(&mut self, x: OpId, shape: OpId) -> OpId {
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Reshape { shape }) })
    }

    /// Expand tensor (adds singleton dims). `shape` is the pre-built output
    /// shape op (const for rank-1, or a `stack` of per-dimension ops).
    pub fn expand(&mut self, x: OpId, shape: OpId) -> OpId {
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Expand { shape }) })
    }

    /// Pad axis `axis` with `lp` zeros on the left, to total length `len`
    /// (tinygrad convention; right padding is `len - lp - orig_len`).
    pub fn pad(&mut self, x: OpId, axis: UAxis, lp: impl IntoOp, len: impl IntoOp) -> OpId {
        let lp = lp.into_op(self);
        let len = len.into_op(self);
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Pad { axis, lp, len }) })
    }

    /// Flip tensor axes.
    pub fn flip(&mut self, x: OpId, axes: &[UAxis]) -> OpId {
        let axes: Box<[UAxis]> = axes.into();
        debug_assert!(!axes.is_empty(), "flip: axes must not be empty");
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Flip { axes }) })
    }

    /// Sum over the last dimension (given by `reduce_axis`).
    pub fn reduce_sum(&mut self, x: OpId, reduce_axis: impl IntoOp) -> OpId {
        let reduce_axis = reduce_axis.into_op(self);
        self.push_back(Op::Reduce { x, rop: BOp::Add, reduce_axis })
    }

    /// Max over the last dimension (given by `reduce_axis`).
    pub fn reduce_max(&mut self, x: OpId, reduce_axis: impl IntoOp) -> OpId {
        let reduce_axis = reduce_axis.into_op(self);
        self.push_back(Op::Reduce { x, rop: BOp::Max, reduce_axis })
    }

    /// Product over the last dimension (given by `reduce_axis`).
    pub fn reduce_prod(&mut self, x: OpId, reduce_axis: impl IntoOp) -> OpId {
        let reduce_axis = reduce_axis.into_op(self);
        self.push_back(Op::Reduce { x, rop: BOp::Mul, reduce_axis })
    }

    /// Constant data value (uses natural dtype).
    /// For index constants, use [`Kernel::const_idx`].
    pub fn const_val<T: Scalar>(&mut self, val: T) -> OpId {
        self.push_back(Op::Const(Constant::new(val)))
    }

    /// Constant index value (normalized to index type).
    /// For data constants, use [`Kernel::const_val`].
    pub fn const_idx<T: Scalar>(&mut self, val: T) -> OpId {
        self.push_back(Op::Const(Constant::idx(val)))
    }

    /// Create multiple constant indices.
    pub fn const_idxs<const N: usize, T: Scalar>(&mut self, vals: [T; N]) -> [OpId; N] {
        core::array::from_fn(|i| self.const_idx(vals[i]))
    }

    /// Define a kernel input param (global memory read-only argument).
    pub fn param(&mut self, dtype: DType) -> OpId {
        self.push_back(Op::Param { dtype, kind: ParamKind::Global, shape: OpId::NULL })
    }

    /// Define a kernel output param (global memory mutable argument).
    pub fn param_mut(&mut self, dtype: DType) -> OpId {
        self.push_back(Op::Param { dtype, kind: ParamKind::GlobalMut, shape: OpId::NULL })
    }

    /// Define a scalar variable param (its value lives in the backend pools' variable slots).
    pub fn variable(&mut self, dtype: DType) -> OpId {
        self.push_back(Op::Param { dtype, kind: ParamKind::Variable, shape: OpId::NULL })
    }

    /// Define multiple scalar variable params (see [`Kernel::variable`]).
    pub fn variables<const N: usize>(&mut self, dtypes: [DType; N]) -> [OpId; N] {
        dtypes.map(|dtype| self.variable(dtype))
    }

    /// Define multiple kernel input params (see [`Kernel::param`]).
    pub fn params<const N: usize>(&mut self, dtypes: [DType; N]) -> [OpId; N] {
        dtypes.map(|dtype| self.param(dtype))
    }

    /// Build a shape op from dimension values.
    ///
    /// A negative dim (`-1`) marks a dynamic/symbolic dimension and becomes a
    /// scalar `Param { kind: Variable }` of `IDX_T`; any nonnegative dim
    /// becomes a const index. Returns `OpId::NULL` for rank-0, the single dim op for rank-1,
    /// or a `Stack` for higher ranks.
    pub fn add_shape(&mut self, shape: &[Dim]) -> OpId {
        let dim_ops: Vec<OpId> = shape.iter().map(|&d| if d < 0 { self.variable(IDX_T) } else { self.const_idx(d) }).collect();
        match dim_ops.len() {
            0 => OpId::NULL,
            1 => dim_ops[0],
            _ => self.stack(&dim_ops),
        }
    }

    /// Define a storage (kernel-internal memory). The `len` must resolve to
    /// a constant (the buffer size is baked into the IR).
    pub fn storage(&mut self, dtype: DType, scope: MemScope, len: impl IntoOp) -> OpId {
        let len_op = len.into_op(self);
        let len =
            self.resolve_const(len_op).and_then(crate::dtype::Constant::as_dim).expect("storage: len must resolve to a constant");
        self.push_back(Op::Storage { dtype, scope, len })
    }

    /// Create a zero-initialized `MemScope::Register` storage of `len` elements.
    ///
    /// Returns the storage id; the zero-init is emitted as a loop storing the
    /// dtype's zero value over the whole storage.
    pub fn zeros(&mut self, dtype: DType, len: impl IntoOp) -> OpId {
        let acc = self.storage(dtype, MemScope::Register, len);
        let len_c = self.const_idx(match self.ops[acc].op {
            Op::Storage { len, .. } => len,
            _ => unreachable!("zeros: storage op expected"),
        });
        let zero = self.push_back(Op::Const(dtype.zero_constant()));
        let l = self.push_back(Op::Loop { len: len_c });
        self.store(acc, zero, l);
        self.push_back(Op::EndLoop);
        acc
    }

    /// Define a `MemScope::Circular` storage of `ntiles` standard 32 x 32
    /// tiles (`ntiles * 1024` elements).
    pub fn circular_storage(&mut self, dtype: DType, ntiles: i64) -> OpId {
        self.storage(dtype, MemScope::Circular, ntiles * 1024)
    }

    /// Group (block) index.
    pub fn group_range(&mut self, axis: u32, len: impl IntoOp) -> OpId {
        let len = len.into_op(self);
        self.push_back(Op::Range { axis, kind: RangeKind::Group(len) })
    }

    /// Group (block) indices, one per axis in order (see [`Kernel::group_range`]).
    pub fn group_ranges<const N: usize>(&mut self, lens: [impl IntoOp; N]) -> [OpId; N] {
        core::array::from_fn(|axis| self.group_range(axis as u32, lens[axis]))
    }

    /// Local thread indices, one per axis in order (see [`Kernel::local_range`]).
    pub fn local_ranges<const N: usize>(&mut self, lens: [u32; N]) -> [OpId; N] {
        core::array::from_fn(|axis| self.local_range(axis as u32, lens[axis]))
    }

    /// Local thread index.
    pub fn local_range(&mut self, axis: u32, len: u32) -> OpId {
        self.push_back(Op::Range { axis, kind: RangeKind::Local(len) })
    }

    /// Warp lane index derived from a local thread range: the threads of
    /// `local_id` form hardware warps and the op's value is the lane id
    /// within the warp (`0..warp_size`, warp size from the device info).
    pub fn warp(&mut self, local_id: OpId) -> OpId {
        let axis = match &self.ops[local_id].op {
            Op::Range { axis, kind: RangeKind::Local(_), .. } => *axis,
            _ => panic!("warp: local_id must reference a local range op"),
        };
        self.push_back(Op::Range { axis, kind: RangeKind::Warp(local_id) })
    }

    /// Load from `src` at `index` (scalar layout: one element).
    pub fn load(&mut self, src: OpId, index: impl IntoOp) -> OpId {
        let index = index.into_op(self);
        self.load_op(src, index, MemLayout::Scalar)
    }

    /// Load a vector of `size` elements from `src` at `index`.
    pub fn load_vector(&mut self, src: OpId, index: impl IntoOp, size: u16) -> OpId {
        let index = index.into_op(self);
        self.load_op(src, index, MemLayout::Vector(size))
    }

    /// Load an `x` × `y` tile with `stride` from `src` at `index`.
    pub fn load_tile(&mut self, src: OpId, index: impl IntoOp, x: u16, y: u16, stride: u32) -> OpId {
        let index = index.into_op(self);
        self.load_op(src, index, MemLayout::Tile { x, y, stride })
    }

    /// Load a standard 32 x 32 circular tile from `src` at `index`.
    pub fn load_circular(&mut self, src: OpId, index: impl IntoOp) -> OpId {
        debug_assert!(
            matches!(self.ops[src].op, Op::Storage { scope: MemScope::Circular, .. }),
            "load_circular: src {src} is not a Circular storage"
        );
        self.load_tile(src, index, 32, 32, 32)
    }

    /// Load a standard 32 x 32 tile from a `MemScope::Register` acc
    /// storage at `index` (SSA threading, no traffic).
    pub fn load_register_tile(&mut self, src: OpId, index: impl IntoOp) -> OpId {
        debug_assert!(
            matches!(self.ops[src].op, Op::Storage { scope: MemScope::Register, .. }),
            "load_register_tile: src {src} is not a Register storage"
        );
        self.load_tile(src, index, 32, 32, 32)
    }

    /// Load a standard 32 x 32 tile from a DRAM (`Global`/`GlobalMut`)
    /// param at `index`.
    pub fn load_global_tile(&mut self, src: OpId, index: impl IntoOp) -> OpId {
        debug_assert!(
            matches!(self.ops[src].op, Op::Param { kind: ParamKind::Global, .. } | Op::Param { kind: ParamKind::GlobalMut, .. }),
            "load_global_tile: src {src} is not a DRAM param"
        );
        self.load_tile(src, index, 32, 32, 32)
    }

    fn load_op(&mut self, src: OpId, index: OpId, layout: MemLayout) -> OpId {
        self.push_back(Op::Load { src, index, layout })
    }

    /// Store `x` to `dst` at `index` (scalar layout: one element).
    pub fn store(&mut self, dst: OpId, x: OpId, index: impl IntoOp) {
        let index = index.into_op(self);
        self.store_op(dst, x, index, MemLayout::Scalar)
    }

    /// Store a vector of `size` elements to `dst` at `index`.
    pub fn store_vector(&mut self, dst: OpId, x: OpId, index: impl IntoOp, size: u16) {
        let index = index.into_op(self);
        self.store_op(dst, x, index, MemLayout::Vector(size))
    }

    /// Store an `x` × `y` tile with `stride` to `dst` at `index`.
    pub fn store_tile(&mut self, dst: OpId, x: OpId, index: impl IntoOp, x_size: u16, y_size: u16, stride: u32) {
        let index = index.into_op(self);
        self.store_op(dst, x, index, MemLayout::Tile { x: x_size, y: y_size, stride })
    }

    /// Store a standard 32 x 32 circular tile `x` to `dst` at `index`.
    pub fn store_circular(&mut self, dst: OpId, x: OpId, index: impl IntoOp) {
        debug_assert!(
            matches!(self.ops[dst].op, Op::Storage { scope: MemScope::Circular, .. }),
            "store_circular: dst {dst} is not a Circular storage"
        );
        self.store_tile(dst, x, index, 32, 32, 32)
    }

    /// Store a standard 32 x 32 tile `x` to a `MemScope::Register` acc
    /// storage at `index` (SSA threading, no traffic).
    pub fn store_register_tile(&mut self, dst: OpId, x: OpId, index: impl IntoOp) {
        debug_assert!(
            matches!(self.ops[dst].op, Op::Storage { scope: MemScope::Register, .. }),
            "store_register_tile: dst {dst} is not a Register storage"
        );
        self.store_tile(dst, x, index, 32, 32, 32)
    }

    /// Store a standard 32 x 32 tile `x` to a DRAM (`GlobalMut`) param
    /// at `index`.
    pub fn store_global_tile(&mut self, dst: OpId, x: OpId, index: impl IntoOp) {
        debug_assert!(
            matches!(self.ops[dst].op, Op::Param { kind: ParamKind::GlobalMut, .. }),
            "store_global_tile: dst {dst} is not a GlobalMut param"
        );
        self.store_tile(dst, x, index, 32, 32, 32)
    }

    fn store_op(&mut self, dst: OpId, x: OpId, index: OpId, layout: MemLayout) {
        self.push_back(Op::Store { dst, src: x, index, layout });
    }

    /// Emit a loop over `len`, call `f` to build the body (the closure
    /// receives the kernel and the loop variable), then close the loop.
    pub fn loop_over(&mut self, len: impl IntoOp, f: impl FnOnce(&mut Kernel, OpId)) {
        let len = len.into_op(self);
        let lv = self.push_back(Op::Loop { len });
        f(self, lv);
        self.push_back(Op::EndLoop);
    }

    /// Emit a partition loop whose length is bound later, by the body: the
    /// first [`Kernel::mma`] inside derives `len = shape[K] / chunk` from
    /// its bind and patches the `Op::Loop` in place (the IR is the state).
    /// Panics after the body if no bind ever touched the loop.
    pub fn loop_partition(&mut self, f: impl FnOnce(&mut Kernel, OpId)) {
        let lv = self.push_back(Op::Loop { len: OpId::NULL });
        f(self, lv);
        if matches!(self.ops[lv].op, Op::Loop { len } if len.is_null()) {
            panic!("loop_partition: loop length never bound (no mma used this loop's variable)");
        }
        self.push_back(Op::EndLoop);
    }

    pub(crate) fn unary(&mut self, x: OpId, uop: UOp) -> OpId {
        self.push_back(Op::Unary { x, uop })
    }

    /// `-x`
    pub fn neg(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        self.unary(x, UOp::Neg)
    }

    /// `~x`
    pub fn bit_not(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        self.unary(x, UOp::BitNot)
    }

    /// `e^x`. Emits the raw `Exp` op; `default_epilogue` converts it to
    /// `exp2` on devices that prefer it, and the CUDA codegen emits `exp`
    /// directly.
    pub fn exp(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        self.unary(x, UOp::Exp)
    }

    /// `2^x`
    pub fn exp2(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        self.unary(x, UOp::Exp2)
    }

    /// Dequant Q4_K nibble: `((input >> (intra*4)) & 15) * scale - offset` as F16.
    /// Returns Vector(8) F16 — one dequantized value per nibble in the u32.
    pub fn dequant_q4_k(&mut self, input: OpId, scale: OpId, offset: OpId) -> OpId {
        let fifteen = self.const_val(15u32);
        let mut outs = Vec::with_capacity(8);
        for intra in 0..8u32 {
            let shift = self.const_idx(intra * 4);
            let shift_u32 = self.cast(shift, DType::U32);
            let shifted = self.binary(input, shift_u32, BOp::BitShiftRight);
            let nibble = self.binary(shifted, fifteen, BOp::BitAnd);
            let nibble_f16 = self.cast(nibble, DType::F16);
            let scaled = self.mul(nibble_f16, scale);
            let v = self.sub(scaled, offset);
            outs.push(v);
        }
        self.stack(&outs)
    }

    /// Dequant Q8_1 int8: cast each of 8 bytes to F16, `* scale + offset` as F16.
    /// Returns Vector(8) F16 — one dequantized value per byte in the u32.
    pub fn dequant_q8_1(&mut self, input: OpId, scale: OpId, offset: OpId) -> OpId {
        let eight = self.const_idx(8u32);
        let ff = self.const_val(0xFFu32);
        let mut outs = Vec::with_capacity(8);
        for byte in 0..8u32 {
            let shift = self.const_idx(byte * 8);
            let shift_u32 = self.cast(shift, DType::U32);
            let shifted = self.binary(input, shift_u32, BOp::BitShiftRight);
            let byte_u32 = self.binary(shifted, ff, BOp::BitAnd);
            let byte_i64 = self.cast(byte_u32, DType::I64);
            let signed = self.sub(byte_i64, eight);
            let signed_f16 = self.cast(signed, DType::F16);
            let scaled = self.mad(signed_f16, scale, offset);
            outs.push(scaled);
        }
        self.stack(&outs)
    }

    /// `ln(x) = log2(x) * ln(2)`
    pub fn ln(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let l = self.log2(x);
        let dtype = self.dtype(x);
        let ln2 = self.const_val(core::f32::consts::LN_2);
        let ln2 = self.cast(ln2, dtype);
        self.mul(l, ln2)
    }

    /// `log2(x)`
    pub fn log2(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        self.unary(x, UOp::Log2)
    }

    /// `1/x`
    pub fn reciprocal(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        self.unary(x, UOp::Reciprocal)
    }

    /// `sqrt(x)`
    pub fn sqrt(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        self.unary(x, UOp::Sqrt)
    }

    /// `sin(x)`
    pub fn sin(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        self.unary(x, UOp::Sin)
    }

    /// `cos(x)`
    pub fn cos(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        self.unary(x, UOp::Cos)
    }

    /// `floor(x)`
    pub fn floor(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        self.unary(x, UOp::Floor)
    }

    /// `trunc(x)`
    pub fn trunc(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        self.unary(x, UOp::Trunc)
    }

    /// `|x|`
    pub fn abs(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        self.unary(x, UOp::Abs)
    }

    /// `1 / (1 + exp(-x))`.
    pub fn sigmoid(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let nx = self.neg(x);
        let e = self.exp(nx);
        let dtype = self.dtype(x);
        let one = self.const_val(1.0f32);
        let one = self.cast(one, dtype);
        let den = self.add(one, e);
        self.reciprocal(den)
    }

    /// `x * sigmoid(x)`.
    pub fn silu(&mut self, x: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let s = self.sigmoid(x);
        self.mul(x, s)
    }

    /// `softplus(x) = log(1 + exp(x))` with the overflow guard: returns `x`
    /// directly above `threshold` (HF default 20.0).
    pub fn softplus(&mut self, x: impl IntoOp, threshold: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let threshold = threshold.into_op(self);
        let big = self.cmpge(x, threshold);
        let e = self.exp(x);
        let dtype = self.dtype(x);
        let one = self.const_val(1.0f32);
        let one = self.cast(one, dtype);
        let sp_in = self.add(one, e);
        let small = self.ln(sp_in);
        self.branchless_where(big, x, small)
    }

    pub(crate) fn binary(&mut self, x: OpId, y: OpId, bop: BOp) -> OpId {
        self.push_back(Op::Binary { x, y, bop })
    }

    /// `x + y`
    pub fn add(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::Add)
    }

    /// `x - y`
    pub fn sub(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::Sub)
    }

    /// `x * y`
    pub fn mul(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::Mul)
    }

    /// `x / y`
    pub fn div(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::Div)
    }

    /// `x^y`
    pub fn pow(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::Pow)
    }

    /// `x % y`
    pub fn mod_(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::Mod)
    }

    /// `x and y`
    pub fn and(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::And)
    }

    /// `x < y`
    pub fn cmplt(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::Cmplt)
    }

    /// `x > y`
    pub fn cmpgt(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::Cmpgt)
    }

    /// `x >= y`
    pub fn cmpge(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::Cmpge)
    }

    /// `max(x, y)`
    pub fn max(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::Max)
    }

    /// `x | y`
    pub fn or_(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::Or)
    }

    /// `x & y`
    pub fn and_(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::And)
    }

    /// `x ^ y`
    pub fn bit_xor(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::BitXor)
    }

    /// `x | y`
    pub fn bit_or(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::BitOr)
    }

    /// `x & y`
    pub fn bit_and(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::BitAnd)
    }

    /// `x << y`
    pub fn bit_shift_left(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::BitShiftLeft)
    }

    /// `x >> y`
    pub fn bit_shift_right(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::BitShiftRight)
    }

    /// `x != y`
    pub fn not_eq(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::NotEq)
    }

    /// `x == y`
    pub fn eq(&mut self, x: impl IntoOp, y: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        self.binary(x, y, BOp::Eq)
    }

    /// Warp matrix multiply-accumulate.
    pub fn wmma(&mut self, dims: MMADims, layout: MMALayout, dtype: MMADType, a: OpId, b: OpId, c: OpId) -> OpId {
        self.push_back(Op::Wmma { dims, layout, dtype, a, b, c })
    }

    /// Hardware tile matmul: folds `x @ y` into accumulator tile `acc`,
    /// returning the new accumulator value (explicit SSA threading).
    pub fn matmul_tile(&mut self, x: OpId, y: OpId, acc: OpId) -> OpId {
        self.push_back(Op::MatmulTile { x, y, acc })
    }

    /// Hardware tile transpose.
    pub fn transpose_tile(&mut self, x: OpId) -> OpId {
        self.push_back(Op::TransposeTile { x })
    }

    /// Hardware tile reduce: folds tile `x` into accumulator tile `acc`
    /// with `rop`, returning the new accumulator value (explicit SSA
    /// threading). `scaler` is the LLK-mandated scale tile (ones when
    /// unused). On TT (`reduce_tile`) the result tile carries the
    /// values in its first row, rest zeros/unchanged.
    pub fn reduce_tile(&mut self, x: OpId, scaler: OpId, acc: OpId, rop: BOp, kind: TileReduceKind) -> OpId {
        self.push_back(Op::ReduceTile { x, scaler, acc, rop, kind })
    }

    /// Backend-specific assembly instruction applied to `ops`.
    pub fn asm(&mut self, asm: &str, ops: &[OpId]) -> OpId {
        let asm = TinyString::new(asm);
        let ops = TinyVec::new(ops);
        self.push_back(Op::Asm { asm, ops })
    }

    /// Warp-wide xor-butterfly sum reduction over `x` (all 32 lanes of the
    /// warp must participate; every lane ends with the sum). CUDA C syntax:
    /// `__shfl_xor_sync` — the caller targets the CUDA backend.
    pub fn warp_reduce(&mut self, mut x: OpId) -> OpId {
        for mask in [16u32, 8, 4, 2, 1] {
            let sh = self.asm(&format!("__shfl_xor_sync(0xffffffff, {{0}}, {mask})"), &[x]);
            x = self.add(x, sh);
        }
        x
    }

    /// Vectorize ops into a single value.
    pub fn stack(&mut self, ops: &[OpId]) -> OpId {
        self.push_back(Op::Stack { ops: ops.into() })
    }

    /// Extract one element from a vectorized value.
    pub fn devectorize_one(&mut self, vec: OpId, idx: usize) -> OpId {
        self.push_back(Op::Index { vec, idx })
    }

    /// Extract all elements from a vectorized value.
    pub fn devectorize<const N: usize>(&mut self, vec: OpId) -> [OpId; N] {
        core::array::from_fn(|i| self.devectorize_one(vec, i))
    }

    /// Local thread barrier.
    /// Thread barrier (synchronization point).
    pub fn barrier(&mut self) {
        self.push_back(Op::Barrier);
    }

    /// Begin conditional block.
    pub fn if_(&mut self, condition: impl IntoOp) {
        let condition = condition.into_op(self);
        self.push_back(Op::If { condition });
    }

    /// End conditional block.
    pub fn end_if(&mut self) {
        self.push_back(Op::EndIf);
    }

    /// Cast to a different dtype.
    pub fn cast(&mut self, x: impl IntoOp, dtype: DType) -> OpId {
        let x = x.into_op(self);
        self.push_back(Op::Cast { x, dtype })
    }

    /// Branchless select: `cond ? a : b` as `a*sel + b*(1-sel)` where `sel` is
    /// `cond` cast to `a`'s dtype. `cond` must be bool; `a` and `b` must share a
    /// dtype (taken from `a`).
    pub fn branchless_where(&mut self, cond: impl IntoOp, a: impl IntoOp, b: impl IntoOp) -> OpId {
        let cond = cond.into_op(self);
        let a = a.into_op(self);
        let b = b.into_op(self);
        let dtype = self.dtype(a);
        //debug_assert_eq!(self.dtype(cond), DType::Bool, "branchless_where: cond must be bool");
        //debug_assert_eq!(self.dtype(b), dtype, "branchless_where: a and b must share a dtype");
        let sel = self.cast(cond, dtype);
        let one = self.push_back(Op::Const(dtype.one_constant()));
        let term_a = self.push_back(Op::Binary { x: a, y: sel, bop: BOp::Mul });
        let not_sel = self.push_back(Op::Binary { x: one, y: sel, bop: BOp::Sub });
        let term_b = self.push_back(Op::Binary { x: b, y: not_sel, bop: BOp::Mul });
        self.push_back(Op::Binary { x: term_a, y: term_b, bop: BOp::Add })
    }

    /// `cond ? a : b` as real control flow (`Op::If` / `Op::EndIf`).
    ///
    /// Unlike [`Kernel::branchless_where`] this works with any operand values,
    /// including `±inf` (the arithmetic version computes `a*sel + b*(1-sel)`,
    /// which turns `±inf * 0` into `NaN`). Returns the selected value (loaded
    /// from a temporary register).
    pub fn ternary_where(&mut self, cond: impl IntoOp, a: impl IntoOp, b: impl IntoOp) -> OpId {
        let cond = cond.into_op(self);
        let a = a.into_op(self);
        let b = b.into_op(self);
        let dtype = self.dtype(a);
        let out = self.storage(dtype, MemScope::Register, 1);
        let idx0 = self.const_idx(0);
        let false_c = self.push_back(Op::Const(DType::Bool.zero_constant()));
        let not_cond = self.eq(cond, false_c);

        self.if_(cond);
        self.store(out, a, idx0);
        self.end_if();
        self.if_(not_cond);
        self.store(out, b, idx0);
        self.end_if();
        self.load(out, idx0)
    }

    /// Bitcast to a different dtype: reinterprets the raw bits of `x` without
    /// a value conversion. Requires equal bit widths of `x`'s dtype and
    /// `dtype` (`debug_assert`s it; the user-facing check lives in
    /// `Tensor::bitcast`).
    pub fn bitcast(&mut self, x: impl IntoOp, dtype: DType) -> OpId {
        let x = x.into_op(self);
        debug_assert_eq!(self.dtype(x).bit_size(), dtype.bit_size(), "bitcast requires equal bit widths");
        self.push_back(Op::Bitcast { x, dtype })
    }

    /// `x * y + z`
    pub fn mad(&mut self, x: impl IntoOp, y: impl IntoOp, z: impl IntoOp) -> OpId {
        let x = x.into_op(self);
        let y = y.into_op(self);
        let z = z.into_op(self);
        self.push_back(Op::Mad { x, y, z })
    }
}

impl Kernel {
    /// Returns the DeviceInfo of the device this kernel is bound to.
    pub fn device_info(&self) -> Arc<DeviceInfo> {
        self.dev_info.clone().expect("kernel has no bound device")
    }
}

impl CompiledKernel {
    /// Returns the DeviceInfo for the device this kernel was compiled on.
    pub fn device_info(&self) -> Arc<DeviceInfo> {
        crate::RT.lock().devices[self.program.device_id].info()
    }

    /// Execute the compiled kernel with new input tensors.
    ///
    /// Routing mirrors every other op: if **any** input is a graph tensor of
    /// the current tape, the kernel is appended to the egraph as a
    /// `Node::Custom` and the outputs are returned lazily (mixed graph/eager
    /// inputs promote the eager ones); with all-eager inputs the kernel
    /// launches directly. The node references the device program, which is
    /// never released (append-only registry) — a dropped `CompiledKernel`
    /// leaves it valid.
    pub fn forward(
        &self,
        inputs: &[&Tensor],
        shapes: Vec<impl IntoIterator<Item = impl Into<Tensor>>>,
    ) -> Result<Vec<Tensor>, ZyxError> {
        debug_assert_eq!(inputs.len(), self.inputs.len());
        debug_assert_eq!(shapes.len(), self.outputs.len());
        let shape_tensors: Vec<Vec<Tensor>> = shapes.into_iter().map(|s| s.into_iter().map(|t| t.into()).collect()).collect();
        // Handles must stay alive until the locked body is done: an inline
        // temporary (e.g. `Tensor::from(1024i64)` as a shape dim) drops at the
        // end of the expression and its slab entry is freed with it.
        let shape_tids: Vec<Vec<TensorId>> = shape_tensors.iter().map(|s| s.iter().map(|t| t.id).collect()).collect();
        let shape_tids: Vec<&[TensorId]> = shape_tids.iter().map(|s| s.as_slice()).collect();
        let ids = crate::RT.lock().forward(
            self.program,
            &inputs.iter().map(|t| t.id).collect::<Vec<_>>(),
            &self.outputs,
            &shape_tids,
        )?;
        Ok(ids.into_iter().map(Tensor::from_id).collect())
    }
}

impl Runtime {
    /// Launches a compiled custom kernel: routes by operands (graph path
    /// promotes inputs into the egraph and emits a `Node::Custom`; eager path
    /// materializes inputs, binds launch args, and launches), returning one
    /// tensor per kernel output.
    ///
    /// `inputs[i]` binds to kernel param `i`; `shapes[i]` is output i's
    /// shape expression (dim tensor ids, possibly empty for a scalar output).
    /// `program` is the compiled kernel's program id; `output_dtypes` its
    /// per-output dtypes.
    pub(crate) fn forward(
        &mut self,
        program: ProgramId,
        inputs: &[TensorId],
        output_dtypes: &[DType],
        shapes: &[&[TensorId]],
    ) -> Result<Vec<TensorId>, ZyxError> {
        // Routing mirrors `Runtime::stack`: the graph path runs iff any
        // operand is a graph tensor of the current tape; otherwise the
        // kernel launches eagerly.
        let any_graph = inputs.iter().any(|&input| self.is_graph(input));
        if any_graph {
            let graph_id = inputs
                .iter()
                .find(|&&input| self.is_graph(input))
                .map(|&input| match self.tensors[input] {
                    TensorData::Graph { graph_id, .. } | TensorData::Promoted { graph_id, .. } => graph_id,
                    ref t => unreachable!("{t:?}"),
                })
                .unwrap();
            self.assert_graph_alive(graph_id);

            // Materialized eager inputs must live in the program's memory
            // pool (they are read as buffers at launch). Unmaterialized eager
            // kernels have no device yet; cross-device placement for them and
            // for graph inputs is resolved at compile time by
            // `Graph::add_memory_ops`.
            let prog_pool = self.devices[program.device_id].memory_pool_id();
            for &input in inputs {
                if !self.is_graph(input) && self.buffer_map.contains_key(&input) && self.buffer_map[&input].pool != prog_pool {
                    return Err(ZyxError::BackendError(BackendError {
                        status: crate::error::ErrorStatus::IncorrectKernelArg,
                        context: format!("custom kernel input tensor {input} is on a different device than the compiled kernel")
                            .into(),
                    }));
                }
            }

            // Promote eager inputs into the graph and resolve every input to
            // a class (mirrors `Runtime::binary` / `stack`'s graph arm).
            let mut input_classes = Vec::with_capacity(inputs.len());
            for &input in inputs {
                if !self.is_graph(input) && !matches!(self.tensors[input], TensorData::Constant { .. }) {
                    self.promote_to_graph(input, graph_id)?;
                }
                input_classes.push(match self.tensors[input] {
                    TensorData::Graph { class_id, .. } | TensorData::Promoted { class_id, .. } => class_id,
                    TensorData::Constant { value, .. } => self.push_const(graph_id, value),
                    ref t => todo!("forward: promote symbolic scalar tid {input} ({t:?}) into a graph"),
                });
            }

            // Per output: the shape expression. `Runtime::stack` routes by
            // the dims themselves — all-slab dims (constants, variables, dim
            // arithmetic) build a slab shape expression; graph dims promote
            // and stack in the egraph. The kernel-node shape CLASS comes from
            // the same call: a graph stack tensor already IS classes; a slab
            // expression replays into the graph.
            let mut shape_classes = Vec::with_capacity(shapes.len());
            let mut shape_ids = Vec::with_capacity(shapes.len());
            for shape in shapes.iter() {
                if shape.is_empty() {
                    shape_classes.push(ClassId::NULL);
                    shape_ids.push(TensorId::NULL);
                    continue;
                }
                let sid = self.stack(shape)?;
                let shape_class = match self.tensors[sid] {
                    TensorData::Graph { class_id, .. } => class_id,
                    TensorData::Constant { .. }
                    | TensorData::Variable { .. }
                    | TensorData::Cast { .. }
                    | TensorData::Unary { .. }
                    | TensorData::Binary { .. }
                    | TensorData::Stack { .. }
                    | TensorData::Stack2 { .. }
                    | TensorData::Stack3 { .. }
                    | TensorData::Stack4 { .. }
                    | TensorData::Stack5 { .. } => self.replay_symbolic_into_graph(graph_id, sid),
                    ref t => todo!("forward: output shape dim tid {sid} is neither slab nor graph ({t:?})"),
                };
                shape_classes.push(shape_class);
                shape_ids.push(sid);
            }

            // Fresh output classes (empty until the Custom node joins them),
            // then the Custom node itself: member of every output class,
            // `class_of` the first. Hashcons is bypassed — the node references
            // output classes that must exist before it, mirroring how
            // `Node::Kernel`s are minted in `autotune_jit_kernels`.
            let mut out_cids = Vec::with_capacity(shapes.len());
            for _ in 0..shapes.len() {
                out_cids.push(self.graphs[graph_id].classes.push(EClass { nodes: vec![] }));
            }
            let outputs: Vec<(ClassId, ClassId, DType)> = out_cids
                .iter()
                .copied()
                .zip(shape_classes)
                .zip(output_dtypes.iter().copied())
                .map(|((cid, shape), dtype)| (cid, shape, dtype))
                .collect();
            let node = Node::Custom { inputs: input_classes.into(), outputs: outputs.into(), program_id: program, time: 10 };
            let nid = self.graphs[graph_id].nodes.push(NodeData { node: node.clone(), class_of: out_cids[0] });
            self.graphs[graph_id].hashcons.insert(node, nid);
            for &ocid in &out_cids {
                self.graphs[graph_id].classes[ocid].nodes.push(nid);
            }

            // Output tensors: lazy graph tensors backed by the output classes.
            let mut tensors = Vec::with_capacity(out_cids.len());
            for ((cid, shape_id), dtype) in out_cids.into_iter().zip(shape_ids).zip(output_dtypes.iter().copied()) {
                self.graphs[graph_id].ref_count += 1;
                let id = self.tensors.push(TensorData::Graph { class_id: cid, graph_id, shape_id, dtype, rc: 1 });
                tensors.push(id);
            }
            return Ok(tensors);
        }

        // Eager launch: materialize inputs, then resolve the shape args.
        // `Runtime::stack` builds each shape expression (routing by the dims
        // themselves) and `resolve_symbolic_dims` evaluates it — variables
        // read their slots, exprs fold; every variable is resolvable.
        // Scalar inputs (Constant/Variable tensors and const expressions)
        // resolve to a `Constant` here and are passed as `LaunchArg::Variable`
        // (mirroring the graph path's `class_vars` binding in plan.rs) — they
        // are kernel params, never buffers.
        // NOTE: all async — allocate is pool bump, launch is stream enqueue, sync is deferred to to_vec/item.
        let _fwd_start = std::time::Instant::now();
        let device_id = program.device_id;
        let pool_id = self.devices[device_id].memory_pool_id();
        let mut input_args: Vec<LaunchArg> = Vec::with_capacity(inputs.len());
        let mut all_bufs = BTreeSet::new();
        let mut event_wait_list = Vec::new();
        for &input in inputs {
            if let Some(value) = self.resolve_symbolic(input) {
                input_args.push(LaunchArg::Variable(value));
                continue;
            }
            if !self.buffer_map.contains_key(&input) {
                self.add_store(input)?;
            }
            let buf_id = self.buffer_map[&input];
            if buf_id.pool != pool_id {
                return Err(ZyxError::BackendError(BackendError {
                    status: crate::error::ErrorStatus::IncorrectKernelArg,
                    context: format!("custom kernel input tensor {input} is on a different device than the compiled kernel")
                        .into(),
                }));
            }
            let keys: Vec<BTreeSet<BufferId>> = self.events.keys().filter(|k| k.contains(&buf_id)).cloned().collect();
            for key in keys {
                event_wait_list.push(self.events.remove(&key).unwrap());
            }
            input_args.push(LaunchArg::Buffer(buf_id.buffer));
            all_bufs.insert(buf_id);
        }
        debug_assert!(inputs.iter().all(|&input| self.buffer_map.contains_key(&input) || self.resolve_symbolic(input).is_some()));

        let mut dims: Vec<Vec<Dim>> = Vec::with_capacity(shapes.len());
        for shape in shapes.iter() {
            if shape.is_empty() {
                dims.push(Vec::new());
                continue;
            }
            let sid = self.stack(shape)?;
            dims.push(self.resolve_symbolic_dims(sid));
            self.release(sid);
        }
        let shapes = dims;

        let mut output_bufs = Vec::new();
        for (i, dtype) in output_dtypes.iter().enumerate() {
            let shape = &shapes[i];
            let bytes = ((shape.iter().product::<Dim>() * dtype.bit_size() as Dim) + 7) / 8;
            let (buf, ev) = self.pools[pool_id].allocate(bytes)?;
            event_wait_list.push(ev);
            let buf_id = BufferId { pool: pool_id, buffer: buf };
            output_bufs.push(buf_id);
            all_bufs.insert(buf_id);
        }

        let mut args = input_args;
        for buf in &output_bufs {
            args.push(LaunchArg::Buffer(buf.buffer));
        }
        let pool_ptr = &mut self.pools[pool_id] as *mut MemoryPool;
        let device = &mut self.devices[device_id];
        let _launch_t = std::time::Instant::now();
        let event = unsafe { device.launch(program.program_id, &mut *pool_ptr, &args, event_wait_list)? };
        /*eprintln!(
            "[forward async] launch enqueue {}us total {}us (async, no sync)",
            _launch_t.elapsed().as_micros(),
            _fwd_start.elapsed().as_micros()
        );*/
        self.events.insert(all_bufs, event);

        // Put to tensors. Each output becomes a **Leaf**: the launched buffer
        // is its backing store (set in buffer_map), no kernel is created.
        // Consumers mint their own load kernels (Runtime::leaf_load), so no
        // NULL op ids ever leak into eager ops built on the result.
        let mut tensors = Vec::new();
        for ((dtype, buf_id), shape) in output_dtypes.iter().copied().zip(output_bufs).zip(shapes) {
            // Build the slab-side shape expression (constant dims) for the
            // new tensor before pushing it.
            let dim_tids: Vec<TensorId> =
                shape.iter().map(|&d| self.new_constant_tensor(crate::dtype::Constant::idx(d))).collect();
            let shape_id = if dim_tids.is_empty() {
                TensorId::NULL
            } else {
                self.stack(&dim_tids).expect("custom kernel output: failed to build shape stack")
            };
            let id = self.tensors.push(TensorData::Leaf {
                depends_on: KernelId::NULL,
                shape_id,
                dtype,
                device_id: program.device_id,
                rc: 1,
            });
            self.buffer_map.insert(id, buf_id);
            tensors.push(id);
        }

        Ok(tensors)
    }
}

/// Partition view: a pure view over a global tensor — source, iteration
/// shape and strides (row-major unless given explicitly). A builder-side
/// handle that emits no IR; tile geometry is bound at use sites ([`Kernel::mma`]),
/// never here.
pub struct Partition {
    /// Global tensor (param) the view reads from / writes to.
    src: OpId,
    /// Iteration shape, row-major. Fully symbolic: every dim is an op
    /// (const or `Param { Variable }`), so shapes are runtime values.
    shape: Vec<OpId>,
    /// Per-axis strides in elements. Derived row-major from `shape`
    /// unless given explicitly ([`Kernel::partition_strided`]).
    strides: Vec<OpId>,
    /// Element dtype.
    dtype: DType,
}

/// Accumulator handle: a per-warp register tile born at [`Kernel::acc`]
/// (its creation is a bind site — an accumulator has no source tensor to
/// defer against, so the tile size lives here; it is also what sizes
/// shared-memory tiles once those exist), filled by [`Kernel::mma`] and
/// written out by [`Kernel::store_partition`].
///
/// The tile is LOGICAL: the kernel author states the output tile they want
/// (e.g. `[128, 8]`); [`Kernel::acc`] splits it into as many hardware
/// fragment accumulators as the device's mma shape requires (CUDA SM 7.5:
/// m16n8k8 → one 16x8 fragment per 16 rows). [`Kernel::mma`] and
/// [`Kernel::store_partition`] iterate the fragments internally — the
/// fragment geometry never reaches the call site.
pub struct Acc {
    /// Accumulator dtype.
    dtype: DType,
    /// Fragment grid width (logical cols / fragment cols); fragments are
    /// stored row-major, so the grid height is `frags.len() / frag_cols`.
    frag_cols: usize,
    /// Per-fragment per-lane register storages (each holds
    /// `frag_m * frag_n / warp_size` elements).
    frags: Vec<OpId>,
}

impl Kernel {
    /// View a global tensor in registers: `src` plus a fully symbolic
    /// iteration `shape`. Strides are derived row-major. Emits no IR.
    pub fn partition<const N: usize>(&mut self, src: OpId, shape: [impl IntoOp; N]) -> Partition {
        let mut shape_ops = Vec::with_capacity(N);
        for d in shape {
            shape_ops.push(d.into_op(self));
        }
        let mut strides = Vec::with_capacity(N);
        for axis in 0..N {
            strides.push(self.row_major_stride(&shape_ops, axis));
        }
        Partition { src, shape: shape_ops, strides, dtype: self.dtype(src) }
    }

    /// Partition with explicit strides (e.g. transposed or offset views).
    pub fn partition_strided<const N: usize>(
        &mut self,
        src: OpId,
        shape: [impl IntoOp; N],
        strides: [impl IntoOp; N],
    ) -> Partition {
        let mut shape_ops = Vec::with_capacity(N);
        for d in shape {
            shape_ops.push(d.into_op(self));
        }
        let mut stride_ops = Vec::with_capacity(N);
        for s in strides {
            stride_ops.push(s.into_op(self));
        }
        debug_assert!(shape_ops.iter().all(|&d| !d.is_null()), "partition: shape dims must be bound ops");
        debug_assert!(stride_ops.iter().all(|&s| !s.is_null()), "partition: strides must be bound ops");
        Partition { src, shape: shape_ops, strides: stride_ops, dtype: self.dtype(src) }
    }

    /// Create a logical accumulator: a zero-initialized per-lane register
    /// tile over `tile`, split into device-native mma fragments (see
    /// [`Kernel::mma_frag_dims`]). The tile must be a whole multiple of the
    /// fragment shape.
    pub fn acc<const N: usize>(&mut self, tile: [impl IntoOp; N], dtype: DType) -> Acc {
        let mut tile_ops = Vec::with_capacity(N);
        for d in tile {
            tile_ops.push(d.into_op(self));
        }
        let dims: Vec<Dim> = tile_ops
            .iter()
            .map(|&d| {
                self.resolve_const(d)
                    .and_then(crate::dtype::Constant::as_dim)
                    .expect("acc: tile dim must resolve to a constant (const or variable op)")
            })
            .collect();
        debug_assert_eq!(dims.len(), 2, "acc: tile must be rank 2 [m, n]");
        debug_assert_eq!(dtype, DType::F32, "acc: f32 accumulator");
        let info = self.device_info();
        let (fm, fn_) = Self::mma_frag_dims(&info, dtype);
        debug_assert!(
            dims[0] > 0 && dims[0] % fm == 0 && dims[1] > 0 && dims[1] % fn_ == 0,
            "acc: logical tile {:?} must be a whole multiple of the fragment shape {:?} (masking is a \
             global-params concern, not an acc one)",
            dims,
            [fm, fn_]
        );
        let frag_rows = (dims[0] / fm) as usize;
        let frag_cols = (dims[1] / fn_) as usize;
        let mut frags = Vec::with_capacity(frag_rows * frag_cols);
        for _ in 0..frag_rows * frag_cols {
            frags.push(self.zeros(dtype, fm * fn_ / 32));
        }
        Acc { dtype, frag_cols, frags }
    }

    /// The device-native mma fragment shape `(m, n)` for an f32 accumulator
    /// with f16 inputs. Fixed per device for now: CUDA SM >= 7.0 uses the
    /// common m16n8k8 shape; everything else is `todo!()` until per-device
    /// branches land (e.g. Tenstorrent's 32x32).
    fn mma_frag_dims(info: &DeviceInfo, _dtype: DType) -> (Dim, Dim) {
        if info.cc[0] >= 7 {
            (16, 8)
        } else {
            todo!("mma: no fragment shape for device cc {:?} (f32 acc, f16 inputs)", info.cc)
        }
    }

    /// Find the innermost open warp op by walking back from the tail.
    ///
    /// The IR is the state: `k.warp(k.local_range(..))` is emitted once, up
    /// front; every partition method finds it here. No plumbing.
    fn open_warp(&self) -> OpId {
        let mut op_id = self.tail;
        while !op_id.is_null() {
            if matches!(self.ops[op_id].op, Op::Range { kind: RangeKind::Warp(_), .. }) {
                return op_id;
            }
            op_id = self.prev_op(op_id);
        }
        panic!("partition: no warp op behind this point — call k.warp(k.local_range(..)) before using partition methods");
    }

    /// `idx * stride`, collapsing to `idx` when the stride resolves to 1
    /// (row-major innermost axis), so fragment addressing stays tight even
    /// though the compile pipeline does not run algebraic simplification.
    fn stride_mul(&mut self, idx: OpId, stride: OpId) -> OpId {
        match self.resolve_const(stride).and_then(crate::dtype::Constant::as_dim) {
            Some(1) => idx,
            _ => self.mul(idx, stride),
        }
    }

    /// Find the innermost OPEN loop by walking back from the tail.
    ///
    /// The IR is the state: ops inside a loop body always sit after their
    /// `Op::Loop`, so the nearest `Op::Loop` behind the tail *is* the open
    /// loop's variable — closed loops (their `Op::EndLoop` already emitted)
    /// are skipped via nesting depth. No builder-side mutable state needed.
    fn open_loop_var(&self) -> OpId {
        let mut depth = 0usize;
        let mut op_id = self.tail;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::EndLoop => depth += 1,
                Op::Loop { .. } => {
                    if depth == 0 {
                        return op_id;
                    }
                    depth -= 1;
                }
                _ => {}
            }
            op_id = self.prev_op(op_id);
        }
        panic!("partition auto indexing: no open loop (call inside loop_partition)");
    }

    /// Row-major stride op for `axis` of a tensor with dims `shape`.
    fn row_major_stride(&mut self, shape: &[OpId], axis: usize) -> OpId {
        let mut stride = self.const_idx(1u32);
        for &dim in &shape[axis + 1..] {
            stride = self.mul(stride, dim);
        }
        stride
    }

    /// Warp matrix multiply-accumulate with fully automatic indexing AND
    /// fully automatic fragment scheduling.
    ///
    /// The TILE form: `a` and `b` are exactly the tiles this call consumes
    /// (e.g. shared-memory register views), so all origins are zero and the
    /// K base is 0 — no coords. The A/B views' K extent is processed in
    /// `frag_k` (8)-wide rounds. See [`Kernel::mma_at`] for the general
    /// form and [`Kernel::mma_frag_dims`] for the fragment shape.
    pub fn mma(&mut self, acc: &Acc, a: &Partition, b: &Partition) {
        let c0 = self.const_idx(0u32);
        let coords = vec![c0; a.shape.len() + b.shape.len() - 1];
        self.mma_coords(acc, a, b, &coords, false);
    }

    /// Warp matrix multiply-accumulate over a chunk of views that are
    /// LARGER than one call's tile (e.g. global-memory views): `coords`
    /// select the chunk, assigned positionally: `rank(a) - 1` coords for
    /// `a`'s non-K axes (in axis order), then `rank(b) - 1` for `b`'s, and
    /// the LAST coord is the K base of both views. Two modes: if the last
    /// coord is the open loop variable, the call derives and patches the
    /// open loop's length (`shape[K] / frag_k`) and emits one fragment per
    /// iteration (K base = `loop_var * frag_k`) — the view's K extent is
    /// then the loop chunk. Otherwise the last coord is a fixed K base op
    /// and the view's whole K extent is processed per call (it must resolve
    /// to a multiple of `frag_k`) — the k-splitting is internal. The coords
    /// live in the A/B views' own coordinate spaces; output coords for
    /// [`Kernel::store_partition`] are passed there independently.
    pub fn mma_at<const N: usize>(&mut self, acc: &Acc, a: &Partition, b: &Partition, coords: [impl IntoOp; N]) {
        let mut coord_ops: Vec<OpId> = Vec::with_capacity(N);
        for c in coords {
            coord_ops.push(c.into_op(self));
        }
        self.mma_coords(acc, a, b, &coord_ops, true);
    }

    /// Shared mma body: `allow_loop_mode` gates the loop-patched K mode
    /// (only meaningful when coords address into a bigger view).
    fn mma_coords(&mut self, acc: &Acc, a: &Partition, b: &Partition, coords: &[OpId], allow_loop_mode: bool) {
        debug_assert_eq!(acc.dtype, DType::F32, "mma: f32 accumulator");
        debug_assert_eq!(a.dtype, b.dtype, "mma: a/b dtype mismatch");
        debug_assert_eq!(a.dtype, DType::F16, "mma: f16 inputs");
        debug_assert_eq!(a.shape.len(), 2, "mma: a must be rank 2");
        debug_assert_eq!(b.shape.len(), 2, "mma: b must be rank 2");
        debug_assert_eq!(
            coords.len(),
            a.shape.len() + b.shape.len() - 1,
            "mma: coords must be [a non-K coords..., b non-K coords..., K base]"
        );

        let frag_k: Dim = 8; // m16n8k8
        let dims = MMADims::m16n8k8;

        // Two modes: loop-patched (last coord is the open loop variable,
        // K base = loop_var * frag_k, one fragment per iteration) or fixed
        // K base (last coord is any op; the view's whole K extent is
        // processed in frag_k-wide rounds).
        let mut lv = OpId::NULL;
        let loop_mode = allow_loop_mode && {
            lv = self.open_loop_var();
            coords[coords.len() - 1] == lv
        };
        let rounds: Dim = if loop_mode {
            // Bind the loop length on first use: K chunk is `frag_k`. The
            // const and the div are inserted BEFORE the loop so the bound is
            // loop-invariant in the linear order.
            if matches!(self.ops[lv].op, Op::Loop { len } if len.is_null()) {
                let frag_k_op = self.insert_before(lv, Op::Const(crate::dtype::Constant::idx(frag_k as u32)));
                let a_k = a.shape[1];
                let len = self.insert_before(lv, Op::Binary { x: a_k, y: frag_k_op, bop: BOp::Div });
                self.ops[lv].op = Op::Loop { len };
            }
            1
        } else {
            let k_extent = self
                .resolve_const(a.shape[1])
                .and_then(crate::dtype::Constant::as_dim)
                .expect("mma: fixed mode requires the view's K extent to resolve (it is the whole per-call chunk)");
            debug_assert!(
                k_extent > 0 && k_extent % frag_k == 0,
                "mma: K extent {k_extent} must be a multiple of frag_k {frag_k}"
            );
            k_extent / frag_k
        };

        let lane = self.open_warp();
        let [c1, c2, c4, c8] = self.const_idxs([1u32, 2, 4, 8]);
        // Fixed m16n8 lane geometry: gid = lane/4, tig = lane%4.
        let gid = self.div(lane, c4);
        let tig = self.mod_(lane, c4);
        let tig2 = self.mul(tig, c2);
        let k_base = if loop_mode {
            self.mul(lv, c8)
        } else {
            coords[coords.len() - 1]
        };

        for fi in 0..acc.frags.len() {
            let frag_row = (fi / acc.frag_cols) as Dim;
            let frag_col = (fi % acc.frag_cols) as Dim;
            // Fragment position within the logical tile.
            let frag_row_op = if frag_row == 0 {
                coords[0]
            } else {
                let off = self.const_idx((frag_row * 16) as u32);
                self.add(coords[0], off)
            };
            let frag_col_op = if frag_col == 0 {
                coords[1]
            } else {
                let off = self.const_idx((frag_col * 8) as u32);
                self.add(coords[1], off)
            };

            // A fragment: rows {r + gid, r + 8 + gid}, k cols {k0 + 2*tig, +1}.
            // Register order: each 2-col pair holds (row, col0), (row, col1),
            // (row_hi, col0), (row_hi, col1).
            let a_row = self.add(frag_row_op, gid);
            let a_row_hi = self.add(a_row, c8);
            let b_row = self.add(frag_col_op, gid);

            for r in 0..rounds {
                let k0 = if r == 0 {
                    k_base
                } else {
                    let off = self.const_idx((8 * r) as u32);
                    self.add(k_base, off)
                };
                let a_col = self.add(k0, tig2);
                let a_col_p1 = self.add(a_col, c1);
                let a_cols = [a_col, a_col_p1];

                let mut a_elems = Vec::new();
                for row in [a_row, a_row_hi] {
                    for col in a_cols {
                        let c = self.stride_mul(col, a.strides[1]);
                        let idx = self.mad(row, a.strides[0], c);
                        a_elems.push(self.load(a.src, idx));
                    }
                }
                let a_frag = self.stack(&a_elems);

                // B fragment: rows {n + gid}, the same shared k cols (consecutive K).
                let mut b_elems = Vec::new();
                for col in a_cols {
                    let bc = self.stride_mul(col, b.strides[1]);
                    let idx = self.mad(b_row, b.strides[0], bc);
                    b_elems.push(self.load(b.src, idx));
                }
                let b_frag = self.stack(&b_elems);

                let idx0 = self.const_idx(0u32);
                let acc_old = self.load_vector(acc.frags[fi], idx0, 4);
                let acc_new = self.wmma(dims, MMALayout::row_col, MMADType::f16_f16_f16_f32, a_frag, b_frag, acc_old);
                self.store_vector(acc.frags[fi], acc_new, idx0, 4);
            }
        }
    }

    /// Store a logical accumulator to a global output view with fully
    /// automatic indexing: the fragments are iterated internally (row-major,
    /// matching [`Kernel::acc`]), each lane scatters its C-fragment values
    /// to their global positions (mma.sync C mapping: rows {r + gid + 8*b},
    /// cols {c + 2*tig, + 1}, one 2-col pair per 8-row block `b`, offset by
    /// the fragment's position in the logical tile). `coords` are the
    /// logical output tile's position in `c`'s own coordinate space (global
    /// for a gmem view) — mma's fragment coords live in the A/B views'
    /// spaces and are independent of these. The lane id is found via
    /// `open_warp`.
    pub fn store_partition<const N: usize>(&mut self, c: &Partition, acc: &Acc, coords: [impl IntoOp; N]) {
        let mut coord_ops: Vec<OpId> = Vec::with_capacity(N);
        for coord in coords {
            coord_ops.push(coord.into_op(self));
        }
        let coords: &[OpId] = &coord_ops;
        debug_assert_eq!(acc.dtype, DType::F32, "store_partition: f32 accumulator");
        debug_assert_eq!(c.shape.len(), 2, "store_partition: output must be rank 2");
        debug_assert_eq!(coords.len(), 2, "store_partition: one coord per output axis");

        let lane = self.open_warp();
        let [c2, c4] = self.const_idxs([2u32, 4]);
        let gid = self.div(lane, c4);
        let tig = self.mod_(lane, c4);

        let row = self.add(coords[0], gid);
        let col = self.mad(tig, c2, coords[1]);
        let idx0 = self.const_idx(0u32);
        let col0 = self.stride_mul(col, c.strides[1]);
        for fi in 0..acc.frags.len() {
            let frag_row = (fi / acc.frag_cols) as Dim;
            let frag_col = (fi % acc.frag_cols) as Dim;
            let acc_final = self.load_vector(acc.frags[fi], idx0, 4);
            let mut elems = Vec::new();
            for i in 0..4usize {
                elems.push(self.devectorize_one(acc_final, i));
            }
            // Fragment offset within the logical tile.
            let row_f = if frag_row == 0 {
                row
            } else {
                let off = self.const_idx((16 * frag_row) as u32);
                self.add(row, off)
            };
            let col_f0 = if frag_col == 0 {
                col0
            } else {
                let off = self.const_idx((8 * frag_col) as u32);
                let off_strided = self.mul(off, c.strides[1]);
                let col_f = self.add(col, off_strided);
                self.stride_mul(col_f, c.strides[1])
            };
            // Each 16x8 fragment contributes two 8-row blocks of one 2-element
            // pair per lane.
            for b in 0..2 {
                let o = if b == 0 {
                    self.mad(row_f, c.strides[0], col_f0)
                } else {
                    let rb = self.const_idx((8 * b) as u32);
                    let row_b = self.add(row_f, rb);
                    self.mad(row_b, c.strides[0], col_f0)
                };
                let o_p1 = self.add(o, c.strides[1]);
                self.store(c.src, elems[2 * b], o);
                self.store(c.src, elems[2 * b + 1], o_p1);
            }
        }
    }
}

/// Scalar/op argument for builder calls: an already-bound op, or ANY scalar
/// constant written inline (`[128, 16]` vs `[c128, c16]`, `kernel.add(x, 1)`).
/// Constants become `Op::Const` index ops (normalized to `IDX_T`); floats are
/// accepted but every consumer of index math asserts its dims/tiles resolve to
/// integer `IDX_T` constants.
pub trait IntoOp: Copy {
    /// Convert to a bound op (constants become index `Op::Const` ops).
    fn into_op(self, kernel: &mut Kernel) -> OpId;
}

impl IntoOp for OpId {
    fn into_op(self, _kernel: &mut Kernel) -> OpId {
        self
    }
}

/// Integer scalars (Dim = i64).
impl IntoOp for Dim {
    fn into_op(self, kernel: &mut Kernel) -> OpId {
        kernel.const_idx(self)
    }
}

impl IntoOp for bool {
    fn into_op(self, kernel: &mut Kernel) -> OpId {
        kernel.const_idx(self as u32)
    }
}

/// 32-bit integer scalars (normalized to index type, like `Dim`).
impl IntoOp for i32 {
    fn into_op(self, kernel: &mut Kernel) -> OpId {
        kernel.const_idx(self)
    }
}

macro_rules! impl_into_op_float {
    ($($t:ty),*) => {$(
        impl IntoOp for $t {
            fn into_op(self, kernel: &mut Kernel) -> OpId {
                kernel.const_val(self)
            }
        }
    )*}
}
impl_into_op_float!(f16, bf16, f32, f64);

/// Shared-memory tile: a `MemScope::Local` buffer of `depth` generations of
/// `total(tile)` elements, cut from the global view `view_shape`. Created by
/// [`Kernel::view_global_local`]; staged by [`Kernel::load_global_local`], one
/// element per call. The tile is viewed in registers via
/// [`Kernel::view_local_register`].
/// `load_global_local` writes generation `k_coord % depth`, so `depth = 1` is
/// classic single buffering and `depth = 2` is double buffering — the
/// surrounding barrier structure (explicit, user-written) is identical.
/// Generic over the view/tile rank `N`.
pub struct LocalPartition<const N: usize> {
    /// Global source the tiles are cut from.
    src: OpId,
    /// Element dtype.
    dtype: DType,
    /// Global view iteration shape — the view of the global tensor the
    /// tiles are cut from; staging strides derive row-major from it.
    view_shape: [OpId; N],
    /// Tile geometry as bound ops (one generation).
    tile: [OpId; N],
    /// Bank-conflict padding: elements added to the LAST axis's smem row
    /// stride (0 = packed).
    pad: Dim,
    /// Elements per generation, padded: `total(tile) + pad * rows(tile)`.
    gen_total: Dim,
    /// `MemScope::Local` storage of `depth * gen_total` elements.
    storage: OpId,
    /// Buffering depth (generations).
    depth: u32,
}

impl Kernel {
    /// View a global tensor as tiles in shared memory: raw global source +
    /// the view shape of the global tensor + the tile geometry + a
    /// bank-conflict padding + buffering depth. `pad` elements are added to
    /// the LAST axis's smem row stride (0 = packed row-major), so rows of a
    /// 16-element f16 tile stagger their bank phase; it must resolve to a
    /// constant (the smem buffer needs a compile-time size). Emits the
    /// `MemScope::Local` storage of `depth * gen_total(tile, pad)` elements;
    /// staging is performed by [`Kernel::load_global_local`], one element
    /// per call. The rank `N` is carried by the returned
    /// [`LocalPartition<N>`].
    pub fn partition_local<const N: usize>(
        &mut self,
        src: OpId,
        view_shape: [impl IntoOp; N],
        tile: [impl IntoOp; N],
        pad: impl IntoOp,
        depth: u32,
    ) -> LocalPartition<N> {
        debug_assert!(N > 0, "view_global_local: rank must be non-zero");
        debug_assert!(depth >= 1, "view_global_local: depth must be >= 1");
        let view_shape: [OpId; N] = view_shape.map(|s| s.into_op(self));
        let tile: [OpId; N] = tile.map(|s| s.into_op(self));
        let pad_op = pad.into_op(self);
        let tile_dims: Vec<Dim> = tile
            .iter()
            .map(|&d| {
                self.resolve_const(d)
                    .and_then(crate::dtype::Constant::as_dim)
                    .expect("view_global_local: tile dim must resolve to a constant (the smem buffer needs a compile-time size)")
            })
            .collect();
        let pad: Dim = self
            .resolve_const(pad_op)
            .and_then(crate::dtype::Constant::as_dim)
            .expect("view_global_local: pad must resolve to a constant (the smem buffer needs a compile-time size)");
        let tile_total: Dim = tile_dims.iter().product();
        debug_assert!(tile_total > 0, "view_global_local: tile must be non-empty");
        // Padded generation size: every row block of `tile[last]` elements
        // grows by `pad`; rank-1 tiles have no row structure to pad.
        let rows: Dim = if N > 1 { tile_dims[..N - 1].iter().product() } else { 0 };
        let gen_total: Dim = tile_total + pad * rows;
        let storage = self.storage(self.dtype(src), MemScope::Local, gen_total * depth as Dim);
        LocalPartition { src, dtype: self.dtype(src), view_shape, tile, pad, gen_total, storage, depth }
    }

    /// Load ONE element global → local of a [`LocalPartition`] tile — the
    /// fully generic primitive. Emits a single gmem load + smem store:
    ///
    /// - gmem address = `Σ origins[a] * (tile[a] * view_stride[a]) + Σ coord[a] * view_stride[a]`
    /// - smem address = `Σ coord[a] * padded_stride[a] + (origins[last] % depth) * gen_total`
    ///
    /// where `coord[a] = (id / tile_stride[a]) % tile[a]` is derived
    /// INTERNALLY from the flat row-major element index `id` — the caller
    /// never writes index arithmetic. `origins` are per-axis tile origins in
    /// TILE units (raw index ops — group ids, loop variables, anything).
    /// Coverage of the tile is user responsibility (thread-cooperative
    /// mappings like `id = la * threads + tid` are exactly how papers write
    /// it). Returns `()` — view the staged tile with
    /// [`Kernel::view_local_register`].
    pub fn load_global_local<const N: usize>(&mut self, shared: &LocalPartition<N>, origins: [impl IntoOp; N], id: impl IntoOp) {
        debug_assert!(N > 0, "load_global_local: rank must be non-zero");
        let origins: [OpId; N] = origins.map(|o| o.into_op(self));
        let id = id.into_op(self);

        // Generation offset: tile lands in buffer `origins[last] % depth`
        // (single buffering: offset 0).
        let smem_base = if shared.depth > 1 {
            let cdepth = self.const_idx(shared.depth);
            let generation = self.mod_(origins[N - 1], cdepth);
            let cgen = self.const_idx(shared.gen_total);
            self.mul(generation, cgen)
        } else {
            self.const_idx(0u32)
        };

        // Decompose the flat element index through the row-major tile
        // strides: coord_a = (id / tile_stride[a]) % tile[a]. Axis 0 needs
        // no modulo (id < tile_total by the coverage contract).
        let mut coords: Vec<OpId> = Vec::with_capacity(N);
        for a in 0..N {
            let t_stride = self.row_major_stride(&shared.tile, a);
            let coord = if a == 0 {
                self.div(id, t_stride)
            } else {
                let q = self.div(id, t_stride);
                self.mod_(q, shared.tile[a])
            };
            coords.push(coord);
        }

        // Per-axis stride arithmetic for gmem, accumulated row-major. The
        // smem side needs no recomposition: the coords are `id`'s row-major
        // decomposition, so the flat `id` IS the smem address within the
        // generation.
        let mut g_idx: Option<OpId> = None;
        for a in 0..N {
            let v_stride = self.row_major_stride(&shared.view_shape, a);
            // gmem: (origins[a] * tile[a] + coord[a]) * view_stride[a]
            let o = self.mul(origins[a], shared.tile[a]);
            let o = self.add(o, coords[a]);
            g_idx = Some(match g_idx {
                Some(g) => self.mad(o, v_stride, g),
                None => self.mul(o, v_stride),
            });
        }
        let v = self.load(shared.src, g_idx.unwrap());
        // smem address: Σ coord[a] * padded_stride[a]. Last axis stride 1,
        // axis N-2 stride = tile[N-1] + pad, outer axes multiply by tile[a+1].
        let mut s_idx = self.add(smem_base, coords[N - 1]);
        if N > 1 {
            let cpad = self.const_idx(shared.pad);
            let mut stride = self.add(shared.tile[N - 1], cpad);
            s_idx = self.mad(coords[N - 2], stride, s_idx);
            for a in (0..N - 2).rev() {
                stride = self.mul(stride, shared.tile[a + 1]);
                s_idx = self.mad(coords[a], stride, s_idx);
            }
        }
        self.store(shared.storage, v, s_idx);
    }

    /// Stage the whole tile of a [`LocalPartition`] cooperatively across all
    /// threads of the block: creates the flat staging loop (`tile_total /
    /// threads` iterations), derives the thread id from the open local
    /// ranges (`tid = Σ lid[a] · stride[a]`, CUDA's own linearization, both
    /// axes participating) and calls [`Kernel::load_global_local`] for each
    /// element. The tile size must be a multiple of the thread count.
    /// Barriers are user-written: stage, then `barrier()`, consume, then
    /// `barrier()` before the next stage may overwrite.
    pub fn stage_global_local<const N: usize>(&mut self, shared: &LocalPartition<N>, origins: [impl IntoOp; N]) {
        debug_assert!(N > 0, "stage_global_local: rank must be non-zero");
        let origins: [OpId; N] = origins.map(|o| o.into_op(self));
        let lids = self.open_local_ids();
        debug_assert!(!lids.is_empty(), "stage_global_local: no open local ranges — call k.local_range(..) before staging");

        // Tile size in elements (packed decomposition; smem padding is
        // handled inside load_global_local).
        let tile_total: Dim = shared
            .tile
            .iter()
            .map(|&d| {
                self.resolve_const(d)
                    .and_then(crate::dtype::Constant::as_dim)
                    .expect("stage_global_local: tile dim must resolve to a constant")
            })
            .product();
        // Threads = product of the open local range lengths (axis 0 fastest).
        let lens: Vec<Dim> = lids
            .iter()
            .map(|&id| match self.ops[id].op {
                Op::Range { kind: RangeKind::Local(len), .. } => len as Dim,
                _ => unreachable!("open_local_ids returned a non-local-range op"),
            })
            .collect();
        let threads: Dim = lens.iter().product();
        debug_assert!(
            tile_total % threads == 0,
            "stage_global_local: tile size {tile_total} must be a multiple of the thread count {threads}"
        );

        // tid = Σ lid[a] * stride[a] — emitted once, before the loop.
        let mut tid = lids[0];
        let mut stride: Dim = lens[0];
        for (a, lid) in lids.iter().enumerate().skip(1) {
            let cstride = self.const_idx(stride as u32);
            tid = self.mad(*lid, cstride, tid);
            stride *= lens[a];
        }

        let cthreads = self.const_idx(threads as u32);
        let trip = tile_total / threads;
        let ctrip = self.const_idx(trip as u32);
        self.loop_over(ctrip, |kernel, la| {
            let id = kernel.mad(la, cthreads, tid);
            kernel.load_global_local(shared, origins, id);
        });
    }

    /// Stage a tile with a fused transformation: load from global, apply `f` to
    /// each element, store to shared memory. Same cooperative staging as
    /// [`Kernel::stage_global_local`] but the transformation runs on the raw
    /// loaded value before storing. Works for any quantization scheme — pass
    /// the appropriate dequant closure.
    ///
    /// `f` receives the raw loaded value and returns the transformed value to
    /// store to smem.
    pub fn stage_global_local_fused<const N: usize, F>(&mut self, shared: &LocalPartition<N>, origins: [impl IntoOp; N], f: F)
    where
        F: Fn(&mut Kernel, OpId) -> OpId,
    {
        debug_assert!(N > 0, "stage_global_local_fused: rank must be non-zero");
        let origins: [OpId; N] = origins.map(|o| o.into_op(self));
        let lids = self.open_local_ids();
        debug_assert!(!lids.is_empty(), "stage_global_local_fused: no open local ranges");

        let tile_total: Dim = shared
            .tile
            .iter()
            .map(|&d| {
                self.resolve_const(d)
                    .and_then(crate::dtype::Constant::as_dim)
                    .expect("stage_global_local_fused: tile dim must resolve to a constant")
            })
            .product();
        let lens: Vec<Dim> = lids
            .iter()
            .map(|&id| match self.ops[id].op {
                Op::Range { kind: RangeKind::Local(len), .. } => len as Dim,
                _ => unreachable!(),
            })
            .collect();
        let threads: Dim = lens.iter().product();
        debug_assert!(tile_total % threads == 0, "tile size must be multiple of thread count");

        let mut tid = lids[0];
        let mut stride: Dim = lens[0];
        for (a, lid) in lids.iter().enumerate().skip(1) {
            let cstride = self.const_idx(stride as u32);
            tid = self.mad(*lid, cstride, tid);
            stride *= lens[a];
        }

        let cthreads = self.const_idx(threads as u32);
        let trip = tile_total / threads;
        let ctrip = self.const_idx(trip as u32);

        self.loop_over(ctrip, |kernel, la| {
            let id = kernel.mad(la, cthreads, tid);

            // Inlined load_global_local with transformation fused in
            let mut coords: Vec<OpId> = Vec::with_capacity(N);
            for a in 0..N {
                let t_stride = kernel.row_major_stride(&shared.tile, a);
                let coord = if a == 0 {
                    kernel.div(id, t_stride)
                } else {
                    let q = kernel.div(id, t_stride);
                    kernel.mod_(q, shared.tile[a])
                };
                coords.push(coord);
            }

            let mut g_idx: Option<OpId> = None;
            for a in 0..N {
                let v_stride = kernel.row_major_stride(&shared.view_shape, a);
                let o = kernel.mul(origins[a], shared.tile[a]);
                let o = kernel.add(o, coords[a]);
                g_idx = Some(match g_idx {
                    Some(g) => kernel.mad(o, v_stride, g),
                    None => kernel.mul(o, v_stride),
                });
            }
            let v = kernel.load(shared.src, g_idx.unwrap());

            // Apply transformation
            let v_transformed = f(kernel, v);

            // Compute smem address and store
            let smem_base = if shared.depth > 1 {
                let cdepth = kernel.const_idx(shared.depth);
                let generation = kernel.mod_(origins[N - 1], cdepth);
                let cgen = kernel.const_idx(shared.gen_total);
                kernel.mul(generation, cgen)
            } else {
                kernel.const_idx(0u32)
            };

            let mut s_idx = kernel.add(smem_base, coords[N - 1]);
            if N > 1 {
                let cpad = kernel.const_idx(shared.pad);
                let mut stride = kernel.add(shared.tile[N - 1], cpad);
                s_idx = kernel.mad(coords[N - 2], stride, s_idx);
                for a in (0..N - 2).rev() {
                    stride = kernel.mul(stride, shared.tile[a + 1]);
                    s_idx = kernel.mad(coords[a], stride, s_idx);
                }
            }
            let layout = kernel.layout(v_transformed);
            match layout {
                MemLayout::Vector(n) => {
                    kernel.store_op(shared.storage, v_transformed, s_idx, MemLayout::Vector(n));
                }
                _ => {
                    kernel.store(shared.storage, v_transformed, s_idx);
                }
            }
        });
    }

    /// The open local ranges (thread axes), axis 0 first, walking back from
    /// the tail. Closed loops are skipped via nesting depth, like
    /// `open_loop_var`; a local range declared inside a closed loop is
    /// ignored.
    fn open_local_ids(&self) -> Vec<OpId> {
        let mut depth = 0usize;
        let mut lids = Vec::new();
        let mut op_id = self.tail;
        while !op_id.is_null() {
            match self.ops[op_id].op {
                Op::EndLoop => depth += 1,
                Op::Loop { .. } => {
                    if depth > 0 {
                        depth -= 1;
                    }
                }
                Op::Range { kind: RangeKind::Local(_), .. } if depth == 0 => lids.push(op_id),
                _ => {}
            }
            op_id = self.prev_op(op_id);
        }
        lids.reverse();
        lids
    }

    /// View the smem tile of a [`LocalPartition`] in registers: row-major
    /// strides over the tile (with the `pad`-padded last-axis row stride),
    /// src = the smem storage. Emits no IR beyond the stride ops. Call it
    /// AFTER the staging loop + barrier, so the stride ops live in the same
    /// scope as their consumers (e.g. `mma`).
    pub fn partition_register<const N: usize>(&mut self, shared: &LocalPartition<N>) -> Partition {
        debug_assert!(N > 0, "view_local_register: rank must be non-zero");
        let mut strides = vec![self.const_idx(1u32); N];
        if N > 1 {
            let cpad = self.const_idx(shared.pad);
            strides[N - 2] = self.add(shared.tile[N - 1], cpad);
            for a in (0..N - 2).rev() {
                strides[a] = self.mul(strides[a + 1], shared.tile[a + 1]);
            }
        }
        Partition { src: shared.storage, shape: shared.tile.to_vec(), strides, dtype: shared.dtype }
    }

    /// RoPE rotate on 32×32 tiles: `y = x*cos + (x @ trans)*sin`.
    ///
    /// `x`, `cos`, `sin`, `trans` are 2-D `Partition`s (row-major). `x`
    /// is `[M,D]`, `cos`/`sin` `[S,rot_dim]` with `H` broadcast `row%S`,
    /// `trans` `[D,D]` (when `rot_dim==D`) else `[rot_dim,rot_dim]` unused in
    /// partial case. `rot_dim` dims rotate (`half=rot_dim/2`), rest pass through.
    pub fn rope_rotate_tile(
        &mut self,
        x: &Partition,
        cos: &Partition,
        sin: &Partition,
        trans: &Partition,
        out: &Partition,
        rot_dim: i64,
    ) {
        debug_assert_eq!(x.shape.len(), 2, "rope_rotate_tile: x must be rank 2");
        debug_assert_eq!(cos.shape.len(), 2, "rope_rotate_tile: cos must be rank 2");
        debug_assert_eq!(sin.shape.len(), 2, "rope_rotate_tile: sin must be rank 2");
        debug_assert_eq!(trans.shape.len(), 2, "rope_rotate_tile: trans must be rank 2");
        debug_assert_eq!(out.shape.len(), 2, "rope_rotate_tile: out must be rank 2");
        debug_assert!(rot_dim > 0 && rot_dim % 2 == 0, "rope_rotate_tile: rot_dim must be positive even");
        let tile_m: i64 = self
            .resolve_const(x.shape[0])
            .and_then(crate::dtype::Constant::as_dim)
            .expect("rope_rotate_tile: tile dim must resolve");
        let tile_n: i64 = self
            .resolve_const(x.shape[1])
            .and_then(crate::dtype::Constant::as_dim)
            .expect("rope_rotate_tile: tile dim must resolve");
        debug_assert!(rot_dim <= tile_n, "rope_rotate_tile: rot_dim must be <= D");
        let info = self.device_info();
        // Reusable: tile -> wmma -> vector -> scalar, based on dims and device caps.
        // Tile/wmma only handle full D rotation (rot_dim==D) with 32×32 / 16×8 and f16 inputs (mma).
        let tile_ok = rot_dim == tile_n
            && !info.tile_sizes.is_empty()
            && tile_m % 32 == 0
            && tile_n % 32 == 0
            && info.tile_sizes.iter().any(|[tx, ty]| *tx == 32 && *ty == 32);
        let wmma_ok =
            rot_dim == tile_n && x.dtype == DType::F16 && !info.wmma_layouts.is_empty() && tile_m % 16 == 0 && tile_n % 8 == 0;
        let vec_ok = !info.supported_vec_lens.is_empty();
        if tile_ok {
            // Tile path (TT 32×32): one CB tile, matmul_tile for rotate_half
            let idx0 = self.const_idx(0);
            let x_tile = self.load_tile(x.src, idx0, 32, 32, 32);
            let cos_tile = self.load_tile(cos.src, idx0, 32, 32, 32);
            let sin_tile = self.load_tile(sin.src, idx0, 32, 32, 32);
            let trans_tile = self.load_tile(trans.src, idx0, 32, 32, 32);
            // Pure (non-accumulating) matmul: seed the explicit acc with zeros.
            let zero_acc_storage = self.zeros(x.dtype, 1024);
            let zero_acc = self.load_tile(zero_acc_storage, idx0, 32, 32, 32);
            let rot = self.matmul_tile(x_tile, trans_tile, zero_acc);
            let y1 = self.mul(x_tile, cos_tile);
            let y2 = self.mul(rot, sin_tile);
            let y = self.add(y1, y2);
            self.store_tile(out.src, y, idx0, 32, 32, 32);
        } else if wmma_ok {
            // Tensor-core path (CUDA 16×8 fragments): e.g. 32×32 -> 2×4, 16×16 -> 1×2, etc.
            let acc_rot = self.acc([tile_m, tile_n], x.dtype);
            self.mma(&acc_rot, x, trans);
            let tmp = self.storage(x.dtype, MemScope::Register, tile_m * tile_n);
            let tmp_part = Partition {
                src: tmp,
                shape: vec![self.const_idx(tile_m), self.const_idx(tile_n)],
                strides: vec![self.const_idx(tile_n), self.const_idx(1)],
                dtype: x.dtype,
            };
            let zero = self.const_idx(0);
            self.store_partition(&tmp_part, &acc_rot, [zero, zero]);
            let cos_m: i64 = self.resolve_const(cos.shape[0]).and_then(crate::dtype::Constant::as_dim).expect("rope cos dim");
            let c_m = self.const_idx(tile_m);
            let c_n = self.const_idx(tile_n);
            let c_rot = self.const_idx(rot_dim);
            let c_s = self.const_idx(cos_m);
            let c_zero = self.const_idx(0);
            let lane = self.open_warp();
            self.loop_over(c_m, |kernel, row| {
                let col = lane;
                let valid = kernel.cmplt(col, c_n);
                let is_rot = kernel.cmplt(col, c_rot);
                let idx = kernel.mad(row, c_n, col);
                let x_val = kernel.load(x.src, idx);
                let s = kernel.mod_(row, c_s);
                let safe_col = kernel.branchless_where(is_rot, col, c_zero);
                let cos_idx = kernel.mad(s, c_rot, safe_col);
                let cos_val = kernel.load(cos.src, cos_idx);
                let sin_val = kernel.load(sin.src, cos_idx);
                let rot_val = kernel.load(tmp, idx);
                let y1 = kernel.mul(x_val, cos_val);
                let y2 = kernel.mul(rot_val, sin_val);
                let rot_y = kernel.add(y1, y2);
                let y = kernel.branchless_where(is_rot, rot_y, x_val);
                let zero = kernel.push_back(crate::kernel::Op::Const(x.dtype.zero_constant()));
                let y = kernel.branchless_where(valid, y, zero);
                kernel.store(out.src, y, idx);
            });
        } else if vec_ok {
            // Vector path: use supported_vec_lens[0] as width, any M×D, first rot_dim rotate.
            let vec_len = info.supported_vec_lens[0] as i64;
            let cos_m: i64 = self.resolve_const(cos.shape[0]).and_then(crate::dtype::Constant::as_dim).expect("rope cos dim");
            let c_m = self.const_idx(tile_m);
            let c_n = self.const_idx(tile_n);
            let c_rot = self.const_idx(rot_dim);
            let c_half = self.const_idx(rot_dim / 2);
            let c_s = self.const_idx(cos_m);
            let c_zero = self.const_idx(0);
            let lane = self.open_warp();
            self.loop_over(c_m, |kernel, row| {
                let col = lane;
                let valid = kernel.cmplt(col, c_n);
                let is_rot = kernel.cmplt(col, c_rot);
                let idx = kernel.mad(row, c_n, col);
                let x_val = kernel.load(x.src, idx);
                let s = kernel.mod_(row, c_s);
                let safe_col = kernel.branchless_where(is_rot, col, c_zero);
                let cos_idx = kernel.mad(s, c_rot, safe_col);
                let cos_val = kernel.load(cos.src, cos_idx);
                let sin_val = kernel.load(sin.src, cos_idx);
                let is_first = kernel.cmplt(col, c_half);
                let col_plus = kernel.add(col, c_half);
                let col_minus = kernel.sub(col, c_half);
                let rot_col = kernel.branchless_where(is_first, col_plus, col_minus);
                let safe_rot = kernel.branchless_where(is_rot, rot_col, c_zero);
                let rot_idx = kernel.mad(row, c_n, safe_rot);
                let x_rot_raw = kernel.load(x.src, rot_idx);
                let neg = kernel.neg(x_rot_raw);
                let x_rot = kernel.branchless_where(is_first, neg, x_rot_raw);
                let y1 = kernel.mul(x_val, cos_val);
                let y2 = kernel.mul(x_rot, sin_val);
                let rot_y = kernel.add(y1, y2);
                let y = kernel.branchless_where(is_rot, rot_y, x_val);
                let zero = kernel.push_back(crate::kernel::Op::Const(x.dtype.zero_constant()));
                let y = kernel.branchless_where(valid, y, zero);
                let _ = vec_len;
                kernel.store(out.src, y, idx);
            });
        } else {
            // Scalar fallback: any M×D with H broadcast, first rot_dim rotate.
            let cos_m: i64 = self.resolve_const(cos.shape[0]).and_then(crate::dtype::Constant::as_dim).expect("rope cos dim");
            let c_m = self.const_idx(tile_m);
            let c_n = self.const_idx(tile_n);
            let c_rot = self.const_idx(rot_dim);
            let c_half = self.const_idx(rot_dim / 2);
            let c_s = self.const_idx(cos_m);
            let c_zero = self.const_idx(0);
            let lane = self.open_warp();
            self.loop_over(c_m, |kernel, row| {
                let col = lane;
                let valid = kernel.cmplt(col, c_n);
                let is_rot = kernel.cmplt(col, c_rot);
                let idx = kernel.mad(row, c_n, col);
                let x_val = kernel.load(x.src, idx);
                let s = kernel.mod_(row, c_s);
                let safe_col = kernel.branchless_where(is_rot, col, c_zero);
                let cos_idx = kernel.mad(s, c_rot, safe_col);
                let cos_val = kernel.load(cos.src, cos_idx);
                let sin_val = kernel.load(sin.src, cos_idx);
                let is_first = kernel.cmplt(col, c_half);
                let col_plus = kernel.add(col, c_half);
                let col_minus = kernel.sub(col, c_half);
                let rot_col = kernel.branchless_where(is_first, col_plus, col_minus);
                let safe_rot = kernel.branchless_where(is_rot, rot_col, c_zero);
                let rot_idx = kernel.mad(row, c_n, safe_rot);
                let x_rot_raw = kernel.load(x.src, rot_idx);
                let neg = kernel.neg(x_rot_raw);
                let x_rot = kernel.branchless_where(is_first, neg, x_rot_raw);
                let y1 = kernel.mul(x_val, cos_val);
                let y2 = kernel.mul(x_rot, sin_val);
                let rot_y = kernel.add(y1, y2);
                let y = kernel.branchless_where(is_rot, rot_y, x_val);
                let zero = kernel.push_back(crate::kernel::Op::Const(x.dtype.zero_constant()));
                let y = kernel.branchless_where(valid, y, zero);
                kernel.store(out.src, y, idx);
            });
        }
    }
}

/// View over a tensor: source, iteration shape, strides, offset, and an
/// optional validity mask.
///
/// The mask holds one `[lo, hi)` valid interval per axis in view coordinates;
/// `None` means fully valid. Only [`Kernel::pad_view`] introduces invalid
/// coords (padded regions); the other view ops preserve or transform the
/// intervals. Intervals are authoritative only for coords within `[0, dim)` —
/// consumers iterate within the shape. [`Kernel::load_view`] and
/// [`Kernel::store_view`] build their predicates from these intervals; plain
/// [`Kernel::index`] ignores them.
pub struct View {
    x: OpId,
    shape: Vec<OpId>,
    strides: Vec<OpId>,
    offset: OpId,
    mask: Option<Vec<(OpId, OpId)>>,
}

impl Kernel {
    /// View `x` with the given iteration `shape`.
    ///
    /// Strides derive row-major from `shape`, offset is const 0. Emits the
    /// stride/index const ops only; the view itself is a builder-side handle.
    pub fn view(&mut self, x: OpId, shape: &[impl IntoOp]) -> View {
        let shape_ops: Vec<OpId> = shape.iter().copied().map(|d| d.into_op(self)).collect();
        debug_assert!(shape_ops.iter().all(|&d| !d.is_null()), "view: shape dims must be bound ops");
        let mut strides = Vec::with_capacity(shape_ops.len());
        for axis in 0..shape_ops.len() {
            strides.push(self.row_major_stride(&shape_ops, axis));
        }
        let offset = self.const_idx(0u32);
        View { x, shape: shape_ops, strides, offset, mask: None }
    }

    /// Add offset
    pub fn offset_view(&mut self, view: &View, offset: impl IntoOp) -> View {
        let offset = offset.into_op(self);
        let offset = self.add(view.offset, offset);
        View { x: view.x, shape: view.shape.clone(), strides: view.strides.clone(), offset, mask: view.mask.clone() }
    }

    /// Reshape a view: new row-major strides over `shape`, offset unchanged.
    ///
    /// The input must be contiguous with the same element count. Both are
    /// enforced when they resolve to constants (fully concrete shapes); with
    /// symbolic dims the caller guarantees them.
    pub fn reshape_view(&mut self, view: &View, shape: &[impl IntoOp]) -> View {
        assert!(view.mask.is_none(), "reshape_view: cannot reshape a masked (padded) view");
        let new_shape: Vec<OpId> = shape.iter().copied().map(|d| d.into_op(self)).collect();
        debug_assert!(new_shape.iter().all(|&d| !d.is_null()), "reshape_view: shape dims must be bound ops");
        let old_dims: Option<Vec<Dim>> = view.shape.iter().map(|&d| self.resolve_const(d).and_then(Constant::as_dim)).collect();
        let new_dims: Option<Vec<Dim>> = new_shape.iter().map(|&d| self.resolve_const(d).and_then(Constant::as_dim)).collect();
        if let (Some(old), Some(new)) = (old_dims, new_dims) {
            let old_total: Dim = old.iter().product();
            let new_total: Dim = new.iter().product();
            assert!(old_total == new_total, "reshape_view: element count mismatch {old_total} != {new_total}");
            let strides: Option<Vec<Dim>> =
                view.strides.iter().map(|&s| self.resolve_const(s).and_then(Constant::as_dim)).collect();
            if let Some(strides) = strides {
                for axis in 0..old.len() {
                    let expected: Dim = old[axis + 1..].iter().product();
                    assert!(
                        strides[axis] == expected,
                        "reshape_view: input is not contiguous (axis {axis} stride {} != row-major {expected})",
                        strides[axis]
                    );
                }
            }
        }
        let mut strides = Vec::with_capacity(new_shape.len());
        for axis in 0..new_shape.len() {
            strides.push(self.row_major_stride(&new_shape, axis));
        }
        View { x: view.x, shape: new_shape, strides, offset: view.offset, mask: view.mask.clone() }
    }

    /// Expand a view (broadcast): same rank; each axis either keeps its dim
    /// (the same op, or a provably equal value) or grows a dim resolving to 1.
    /// Grown axes get stride 0. Offset unchanged.
    pub fn expand_view(&mut self, view: &View, shape: &[impl IntoOp]) -> View {
        let new_shape: Vec<OpId> = shape.iter().copied().map(|d| d.into_op(self)).collect();
        assert!(new_shape.len() == view.shape.len(), "expand_view: rank mismatch {} != {}", new_shape.len(), view.shape.len());
        debug_assert!(new_shape.iter().all(|&d| !d.is_null()), "expand_view: shape dims must be bound ops");
        let zero = self.const_idx(0u32);
        let mut new_strides = Vec::with_capacity(new_shape.len());
        let mut new_mask: Option<Vec<(OpId, OpId)>> = view.mask.clone();
        for (axis, ((&old_d, &old_s), &new_d)) in view.shape.iter().zip(view.strides.iter()).zip(new_shape.iter()).enumerate() {
            if new_d == old_d {
                new_strides.push(old_s);
                continue;
            }
            let old_dim = self.resolve_const(old_d).and_then(Constant::as_dim);
            let new_dim = self.resolve_const(new_d).and_then(Constant::as_dim);
            match (old_dim, new_dim) {
                (Some(o), Some(n)) if o == n => new_strides.push(old_s),
                (Some(1), _) => {
                    new_strides.push(zero);
                    // Every new coord reads the single base element (stride
                    // 0), so the grown axis is fully valid iff coord 0 is.
                    if let Some(mask) = new_mask.as_mut() {
                        let (lo, hi) = mask[axis];
                        let lo_ok = self.resolve_const(lo).and_then(Constant::as_dim).is_some_and(|l| l <= 0);
                        let hi_ok = self.resolve_const(hi).and_then(Constant::as_dim).is_some_and(|h| h >= 1);
                        assert!(
                            lo_ok && hi_ok,
                            "expand_view: cannot broadcast a masked axis whose base element is not provably valid"
                        );
                        mask[axis] = (zero, new_d);
                    }
                }
                (Some(o), _) => panic!("expand_view: axis dim {o} is neither kept nor 1 (cannot broadcast)"),
                (None, _) => panic!("expand_view: grown axis dim is symbolic and differs from the source op"),
            }
        }
        View { x: view.x, shape: new_shape, strides: new_strides, offset: view.offset, mask: new_mask }
    }

    /// Permute view axes: reorders shape/strides by `axes`, offset unchanged.
    pub fn permute_view(&mut self, view: &View, axes: &[UAxis]) -> View {
        let rank = view.shape.len();
        assert!(axes.len() == rank, "permute_view: axes len {} != rank {rank}", axes.len());
        let mut seen = vec![false; rank];
        for &a in axes {
            assert!(a < rank, "permute_view: axis {a} out of range for rank {rank}");
            assert!(!seen[a], "permute_view: duplicate axis {a}");
            seen[a] = true;
        }
        let shape = axes.iter().map(|&a| view.shape[a]).collect();
        let strides = axes.iter().map(|&a| view.strides[a]).collect();
        let mask = view.mask.as_ref().map(|m| axes.iter().map(|&a| m[a]).collect());
        View { x: view.x, shape, strides, offset: view.offset, mask }
    }

    /// Pad a view (torch `functional.pad` convention): `padding` holds
    /// `(left, right)` pairs starting from the LAST axis backwards —
    /// `[left_last, right_last, left_next, right_next, ...]`. Length must be
    /// even with at most one pair per axis; leading axes without a pair stay
    /// unchanged. Padded axes grow (`dim + left + right`, strides unchanged)
    /// and the offset shifts back by `left * stride` per padded axis. Padded
    /// regions lie outside the source; consumers mask them.
    pub fn pad_view(&mut self, view: &View, padding: &[impl IntoOp]) -> View {
        let rank = view.shape.len();
        assert!(padding.len() % 2 == 0, "pad_view: padding len {} must be even (left/right pairs)", padding.len());
        let pairs = padding.len() / 2;
        assert!(pairs <= rank, "pad_view: {} pairs for rank {rank}", pairs);
        let pads: Vec<OpId> = padding.iter().copied().map(|p| p.into_op(self)).collect();
        let mut new_shape = view.shape.clone();
        let mut offset = view.offset;
        let mut mask = view.mask.clone();
        for k in 0..pairs {
            let axis = rank - 1 - k;
            let left = pads[2 * k];
            let right = pads[2 * k + 1];
            let l = self.resolve_const(left).and_then(Constant::as_dim);
            let r = self.resolve_const(right).and_then(Constant::as_dim);
            if let (Some(l), Some(r)) = (l, r) {
                assert!(l >= 0 && r >= 0, "pad_view: negative pad ({l}, {r}) on axis {axis}");
            }
            // All-zero pads change nothing: shape, offset, and mask stay as-is.
            if l == Some(0) && r == Some(0) {
                continue;
            }
            let grown = self.add(new_shape[axis], left);
            new_shape[axis] = self.add(grown, right);
            // Zero left pads leave the offset alone, keeping the IR tight.
            if l != Some(0) {
                let shift = self.mul(left, view.strides[axis]);
                offset = self.sub(offset, shift);
            }
            // The padded axis is valid over the old data window shifted into
            // the new coordinates; untouched axes keep full `[0, dim)` windows.
            let m = mask.get_or_insert_with(|| {
                let c0 = self.const_idx(0u32);
                view.shape.iter().map(|&d| (c0, d)).collect()
            });
            if l != Some(0) {
                let (lo, hi) = m[axis];
                m[axis] = (self.add(lo, left), self.add(hi, left));
            }
        }
        View { x: view.x, shape: new_shape, strides: view.strides.clone(), offset, mask }
    }

    /// Flat index into a view: `offset + Σ coord[i] * stride[i]`.
    /// `coords` covers every axis exactly once, in axis order.
    /// Private addressing primitive behind [`Kernel::load_view`] and
    /// [`Kernel::store_view`]; ignores the view's mask.
    fn index<const N: usize>(&mut self, view: &View, coords: [impl IntoOp; N]) -> OpId {
        assert!(N == view.shape.len(), "index: {} coords for rank {}", N, view.shape.len());
        let coords: [OpId; N] = coords.map(|c| c.into_op(self));
        let mut idx = view.offset;
        for (c, &s) in coords.into_iter().zip(view.strides.iter()) {
            // Zero coords contribute nothing; skipping keeps the IR tight.
            if self.resolve_const(c).and_then(Constant::as_dim) == Some(0) {
                continue;
            }
            idx = self.mad(c, s, idx);
        }
        idx
    }

    /// Load one element through a view: clamp coords into their valid
    /// windows, `index`, load, then predicate with the mask — masked-off
    /// coords read the source dtype's zero. The clamp keeps the issued load
    /// in-bounds (a `branchless_where` alone would not: the OOB load would
    /// still execute). Unmasked views emit a plain load.
    pub fn load_view<const N: usize>(&mut self, view: &View, coords: [impl IntoOp; N]) -> OpId {
        let coords: [OpId; N] = coords.map(|c| c.into_op(self));
        let Some(mask) = &view.mask else {
            let idx = self.index(view, coords);
            return self.load(view.x, idx);
        };
        // Clamp into `[lo, hi - 1]` per axis (`min` via negated `max`);
        // the predicate below runs on the ORIGINAL coords.
        let one = self.const_idx(1u32);
        let mut clamped = coords;
        for (cc, (lo, hi)) in clamped.iter_mut().zip(mask.iter().copied()) {
            let a = self.max(*cc, lo);
            let hi_m1 = self.sub(hi, one);
            let na = self.neg(a);
            let nhi = self.neg(hi_m1);
            let m = self.max(na, nhi);
            *cc = self.neg(m);
        }
        let idx = self.index(view, clamped);
        let v = self.load(view.x, idx);
        let mut pred: Option<OpId> = None;
        for (c, (lo, hi)) in coords.into_iter().zip(mask.iter().copied()) {
            let lo_t = self.cmpge(c, lo);
            let hi_t = self.cmplt(c, hi);
            let term = self.and(lo_t, hi_t);
            pred = Some(match pred {
                Some(p) => self.and(p, term),
                None => term,
            });
        }
        let pred = pred.expect("load_view: masked view with no axes");
        let dtype = self.dtype(view.x);
        let zero = self.push_back(Op::Const(dtype.zero_constant()));
        self.branchless_where(pred, v, zero)
    }

    /// Store one element through a view: `index` the coords, then guard with
    /// the mask — masked-off coords skip the store. Unmasked views emit a
    /// plain store.
    pub fn store_view<const N: usize>(&mut self, view: &View, x: impl IntoOp, coords: [impl IntoOp; N]) {
        let x = x.into_op(self);
        let coords: [OpId; N] = coords.map(|c| c.into_op(self));
        let idx = self.index(view, coords);
        let Some(mask) = &view.mask else {
            self.store(view.x, x, idx);
            return;
        };
        let mut pred: Option<OpId> = None;
        for (c, (lo, hi)) in coords.into_iter().zip(mask.iter().copied()) {
            let lo_t = self.cmpge(c, lo);
            let hi_t = self.cmplt(c, hi);
            let term = self.and(lo_t, hi_t);
            pred = Some(match pred {
                Some(p) => self.and(p, term),
                None => term,
            });
        }
        let pred = pred.expect("store_view: masked view with no axes");
        self.if_(pred);
        self.store(view.x, x, idx);
        self.end_if();
    }

    /// Slice a view with index specs (`..`, `a..b`, `a..`, `..b`, `a..=b`,
    /// `..=b` per axis): a single spec covers axis 0, a tuple covers the
    /// leading axes (up to 4), the rest stay untouched. Each covered axis
    /// becomes `len = end - start` with `offset += start * stride`, strides
    /// unchanged. Bounds are checked when they resolve to constants (empty
    /// slices rejected); symbolic bounds skip the checks.
    pub fn slice_view(&mut self, view: &View, indices: impl IntoIndex) -> View {
        let rank = view.shape.len();
        let axes = indices.into_index(self, &view.shape);
        assert!(!axes.is_empty(), "slice_view: no indices given");
        assert!(axes.len() <= rank, "slice_view: {} indices for rank {rank}", axes.len());
        let mut new_shape = view.shape.clone();
        let mut offset = view.offset;
        let mut mask = view.mask.clone();
        for (axis, (start, end)) in axes.into_iter().enumerate() {
            let s = self.resolve_const(start).and_then(Constant::as_dim);
            let e = self.resolve_const(end).and_then(Constant::as_dim);
            let d = self.resolve_const(view.shape[axis]).and_then(Constant::as_dim);
            if let Some(s) = s {
                assert!(s >= 0, "slice_view: negative start {s} on axis {axis}");
            }
            if let (Some(s), Some(e)) = (s, e) {
                assert!(e > s, "slice_view: empty slice [{s}, {e}) on axis {axis}");
            }
            if let (Some(e), Some(d)) = (e, d) {
                assert!(e <= d, "slice_view: end {e} past dim {d} on axis {axis}");
            }
            new_shape[axis] = self.sub(end, start);
            // Zero starts leave offset and mask alone, keeping the IR tight.
            if s != Some(0) {
                let shift = self.mul(start, view.strides[axis]);
                offset = self.add(offset, shift);
                if let Some(m) = mask.as_mut() {
                    let (lo, hi) = m[axis];
                    m[axis] = (self.sub(lo, start), self.sub(hi, start));
                }
            }
        }
        View { x: view.x, shape: new_shape, strides: view.strides.clone(), offset, mask }
    }
}

/// Per-axis slice spec for [`Kernel::slice_view`]: `..`, `a..b`, `a..`,
/// `..b`, `a..=b`, `..=b`. Bounds accept anything [`IntoOp`] (bound ops,
/// `i64`/`i32` consts); negative const bounds count from the end of the axis.
pub trait IntoSliceAxis {
    /// Bind to `(start, end-exclusive)` ops normalized against `dim`.
    fn into_axis(self, kernel: &mut Kernel, dim: OpId) -> (OpId, OpId);

    /// Normalize one bound: `None` is the axis edge (0 for start, `dim` for
    /// end, plus 1 for inclusive ends); negative const bounds fold against
    /// `dim`; everything else passes through untouched.
    fn norm_bound(kernel: &mut Kernel, dim: OpId, bound: Option<OpId>, is_start: bool, inclusive: bool) -> OpId {
        let Some(b) = bound else {
            if is_start {
                return kernel.const_idx(0u32);
            }
            if inclusive {
                let one = kernel.const_idx(1u32);
                return kernel.add(dim, one);
            }
            return dim;
        };
        let b = match kernel.resolve_const(b).and_then(Constant::as_dim) {
            Some(v) if v < 0 => kernel.add(dim, b),
            _ => b,
        };
        if !is_start && inclusive {
            let one = kernel.const_idx(1u32);
            kernel.add(b, one)
        } else {
            b
        }
    }
}

impl IntoSliceAxis for RangeFull {
    fn into_axis(self, kernel: &mut Kernel, dim: OpId) -> (OpId, OpId) {
        (Self::norm_bound(kernel, dim, None, true, false), Self::norm_bound(kernel, dim, None, false, false))
    }
}

impl<T: IntoOp> IntoSliceAxis for Range<T> {
    fn into_axis(self, kernel: &mut Kernel, dim: OpId) -> (OpId, OpId) {
        let s = self.start.into_op(kernel);
        let e = self.end.into_op(kernel);
        (Self::norm_bound(kernel, dim, Some(s), true, false), Self::norm_bound(kernel, dim, Some(e), false, false))
    }
}

impl<T: IntoOp> IntoSliceAxis for RangeFrom<T> {
    fn into_axis(self, kernel: &mut Kernel, dim: OpId) -> (OpId, OpId) {
        let s = self.start.into_op(kernel);
        (Self::norm_bound(kernel, dim, Some(s), true, false), Self::norm_bound(kernel, dim, None, false, false))
    }
}

impl<T: IntoOp> IntoSliceAxis for RangeTo<T> {
    fn into_axis(self, kernel: &mut Kernel, dim: OpId) -> (OpId, OpId) {
        let e = self.end.into_op(kernel);
        (Self::norm_bound(kernel, dim, None, true, false), Self::norm_bound(kernel, dim, Some(e), false, false))
    }
}

impl<T: IntoOp> IntoSliceAxis for RangeInclusive<T> {
    fn into_axis(self, kernel: &mut Kernel, dim: OpId) -> (OpId, OpId) {
        let (s, e) = self.into_inner();
        let s = s.into_op(kernel);
        let e = e.into_op(kernel);
        (Self::norm_bound(kernel, dim, Some(s), true, false), Self::norm_bound(kernel, dim, Some(e), false, true))
    }
}

impl<T: IntoOp> IntoSliceAxis for RangeToInclusive<T> {
    fn into_axis(self, kernel: &mut Kernel, dim: OpId) -> (OpId, OpId) {
        let e = self.end.into_op(kernel);
        (Self::norm_bound(kernel, dim, None, true, false), Self::norm_bound(kernel, dim, Some(e), false, true))
    }
}

/// Slice indices for [`Kernel::slice_view`]: a single axis spec (applies to
/// axis 0) or a tuple of up to 4 specs (applies to the leading axes, the rest
/// untouched).
pub trait IntoIndex {
    /// Normalize to one `(start, end-exclusive)` pair per covered axis.
    fn into_index(self, kernel: &mut Kernel, dims: &[OpId]) -> Vec<(OpId, OpId)>;
}

impl<S: IntoSliceAxis> IntoIndex for S {
    fn into_index(self, kernel: &mut Kernel, dims: &[OpId]) -> Vec<(OpId, OpId)> {
        let dim = dims.first().copied().expect("slice_view: no axes to slice");
        vec![self.into_axis(kernel, dim)]
    }
}

macro_rules! impl_into_index_tuple {
    ($($($t:ident),+);+) => {$(
        impl<$($t: IntoSliceAxis),+> IntoIndex for ($($t,)+) {
            fn into_index(self, kernel: &mut Kernel, dims: &[OpId]) -> Vec<(OpId, OpId)> {
                #[allow(non_snake_case)]
                let ($($t,)+) = self;
                let mut i = 0;
                let mut out = Vec::new();
                $(
                    let dim = dims.get(i).copied().expect("slice_view: more indices than axes");
                    out.push($t.into_axis(kernel, dim));
                    i += 1;
                )+
                let _ = i;
                out
            }
        }
    )+};
}
impl_into_index_tuple!(A; A, B; A, B, C; A, B, C, D);
