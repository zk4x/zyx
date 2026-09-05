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
use std::sync::Arc;

use crate::backend::{BufferId, DeviceInfo, LaunchArg, MemoryPool, ProgramId};
use crate::dtype::Constant;
use crate::error::BackendError;
use crate::graph::{ClassId, EClass, Node, NodeData};
use crate::kernel::{
    BOp, DeviceId, IDX_T, Kernel, MMADType, MMADims, MMALayout, MemLayout, MemScope, MoveOp, Op, OpId, ParamKind, RangeKind, UOp,
};
use crate::runtime::{KernelId, Runtime, TensorData};
use crate::shape::UAxis;
use crate::slab::{Slab, SlabId};
use crate::tensor::TensorId;
use crate::types::{TinyString, TinyVec};
use crate::{DType, Tensor, ZyxError, shape::Dim};
use crate::{Dev, Map};

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
        self.linearize();
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
        self.instruction_schedule();
        self.constant_folding();
        self.dead_code_elimination();
        self.verify();

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
        if rt.debug.ir() {
            self.debug();
        }
        let debug_asm = rt.debug.asm();
        let program_id = rt.devices[device_id].compile(&self, debug_asm)?;
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
    pub fn pad(&mut self, x: OpId, axis: UAxis, lp: OpId, len: OpId) -> OpId {
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Pad { axis, lp, len }) })
    }

    /// Flip tensor axes.
    pub fn flip(&mut self, x: OpId, axes: &[UAxis]) -> OpId {
        let axes: Box<[UAxis]> = axes.into();
        debug_assert!(!axes.is_empty(), "flip: axes must not be empty");
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Flip { axes }) })
    }

    /// Sum over the last dimension (given by `reduce_axis`).
    pub fn reduce_sum(&mut self, x: OpId, reduce_axis: OpId) -> OpId {
        self.push_back(Op::Reduce { x, rop: BOp::Add, reduce_axis })
    }

    /// Max over the last dimension (given by `reduce_axis`).
    pub fn reduce_max(&mut self, x: OpId, reduce_axis: OpId) -> OpId {
        self.push_back(Op::Reduce { x, rop: BOp::Max, reduce_axis })
    }

    /// Product over the last dimension (given by `reduce_axis`).
    pub fn reduce_prod(&mut self, x: OpId, reduce_axis: OpId) -> OpId {
        self.push_back(Op::Reduce { x, rop: BOp::Mul, reduce_axis })
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

    /// Define a storage (kernel-internal memory).
    pub fn storage(&mut self, dtype: DType, scope: MemScope, len: Dim) -> OpId {
        self.push_back(Op::Storage { dtype, scope, len })
    }

    /// Create a zero-initialized `MemScope::Register` storage of `len` elements.
    ///
    /// Returns the storage id; the zero-init is emitted as a loop storing the
    /// dtype's zero value over the whole storage.
    pub fn zeros(&mut self, dtype: DType, len: Dim) -> OpId {
        let acc = self.storage(dtype, MemScope::Register, len);
        let len_c = self.const_idx(len);
        let zero = self.push_back(Op::Const(dtype.zero_constant()));
        let l = self.push_back(Op::Loop { len: len_c });
        self.store(acc, zero, l);
        self.push_back(Op::EndLoop);
        acc
    }

    /// Group (block) index.
    pub fn group_range(&mut self, axis: u32, len: OpId) -> OpId {
        self.push_back(Op::Range { axis, kind: RangeKind::Group(len) })
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
    pub fn load(&mut self, src: OpId, index: OpId) -> OpId {
        self.load_op(src, index, MemLayout::Scalar)
    }

    /// Load a vector of `size` elements from `src` at `index`.
    pub fn load_vector(&mut self, src: OpId, index: OpId, size: u16) -> OpId {
        self.load_op(src, index, MemLayout::Vector(size))
    }

    /// Load an `x` × `y` tile with `stride` from `src` at `index`.
    pub fn load_tile(&mut self, src: OpId, index: OpId, x: u16, y: u16, stride: u32) -> OpId {
        self.load_op(src, index, MemLayout::Tile { x, y, stride })
    }

    fn load_op(&mut self, src: OpId, index: OpId, layout: MemLayout) -> OpId {
        self.push_back(Op::Load { src, index, layout })
    }

    /// Store `x` to `dst` at `index` (scalar layout: one element).
    pub fn store(&mut self, dst: OpId, x: OpId, index: OpId) {
        self.store_op(dst, x, index, MemLayout::Scalar)
    }

    /// Store a vector of `size` elements to `dst` at `index`.
    pub fn store_vector(&mut self, dst: OpId, x: OpId, index: OpId, size: u16) {
        self.store_op(dst, x, index, MemLayout::Vector(size))
    }

    /// Store an `x` × `y` tile with `stride` to `dst` at `index`.
    pub fn store_tile(&mut self, dst: OpId, x: OpId, index: OpId, x_size: u16, y_size: u16, stride: u32) {
        self.store_op(dst, x, index, MemLayout::Tile { x: x_size, y: y_size, stride })
    }

    fn store_op(&mut self, dst: OpId, x: OpId, index: OpId, layout: MemLayout) {
        self.push_back(Op::Store { dst, src: x, index, layout });
    }

    /// Emit a loop over `len`, call `f` to build the body (the closure
    /// receives the kernel and the loop variable), then close the loop.
    pub fn loop_over(&mut self, len: OpId, f: impl FnOnce(&mut Kernel, OpId)) {
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

    /// Accumulate `a * b` over the currently open loop into a fresh scalar
    /// accumulator (a `MemScope::Register` storage of `dtype`), returning the
    /// accumulator's storage id.
    ///
    /// The loop must be open: an `Op::Loop` emitted, its `Op::EndLoop` not yet. The
    /// helper hoists the accumulator and its zero-init before the loop
    /// (via `move_op_before`) and appends the per-iteration
    /// `acc = a[i] * b[i] + acc` block inside the loop body. The caller closes
    /// the loop and loads the result from the returned storage afterwards.
    ///
    /// # Arguments
    ///
    /// - `loop_op`: the open loop (`loop_()` return value)
    /// - `a`, `b`: sources to multiply
    /// - `index_a`, `index_b`: index ops for `a` and `b`, typically built from `loop_op`
    pub fn dot(&mut self, dtype: DType, loop_op: OpId, a: OpId, index_a: OpId, b: OpId, index_b: OpId) -> OpId {
        let acc = self.storage(dtype, MemScope::Register, 1);
        let idx0 = self.const_idx(0);
        let zero = self.push_back(Op::Const(dtype.zero_constant()));
        self.move_op_before(acc, loop_op);
        self.move_op_before(idx0, loop_op);
        self.move_op_before(zero, loop_op);
        let init = self.push_back(Op::Store { dst: acc, src: zero, index: idx0, layout: MemLayout::Scalar });
        self.move_op_before(init, loop_op);

        let av = self.load(a, index_a);
        let bv = self.load(b, index_b);
        let old = self.load(acc, idx0);
        let sum = self.mad(av, bv, old);
        self.store(acc, sum, idx0);
        acc
    }

    /// Cooperatively copy a tile into a `MemScope::Local` storage (shared memory).
    ///
    /// Emits a loop over `cols`; iteration `c` copies
    /// `src[(row_base + thread_row) * cols + c]` to `dst[thread_row * cols + c]`
    /// — i.e. each of `rows` workgroup threads fetches the `thread_row`-th row of
    /// the tile. The caller is responsible for the surrounding `barrier()`s.
    ///
    /// Returns nothing; `dst` must be a `MemScope::Local` storage with room for
    /// `rows * cols` elements.
    ///
    /// # Arguments
    ///
    /// - `src`, `dst`: global source and local destination storages
    /// - `rows`, `cols`: tile dimensions (`rows` = number of threads participating)
    /// - `row_base`: op giving the first global row to copy (tile origin)
    /// - `thread_row`: the calling thread's row within the tile
    pub fn copy_tile_local(&mut self, src: OpId, dst: OpId, rows: OpId, row_base: OpId, cols: OpId, thread_row: OpId) {
        let c_loop = self.push_back(Op::Loop { len: cols });
        let dst_idx = self.mad(thread_row, cols, c_loop);
        let src_row = self.mad(row_base, rows, thread_row);
        let src_idx = self.mad(src_row, cols, c_loop);
        let v = self.load(src, src_idx);
        self.store(dst, v, dst_idx);
        self.push_back(Op::EndLoop);
    }

    /// For each column `c` of the `row`-th row of a `MemScope::Local` tile,
    /// update a `MemScope::Register` accumulator vector:
    /// `acc[c] = sa * acc[c] + sb * src[row * cols + c]`.
    ///
    /// Emits the loop over `cols` itself. `acc` must be a Register storage of
    /// length `cols`; `sa` and `sb` are scalar ops broadcast across the row.
    pub fn mad_tile_local(&mut self, acc: OpId, sa: OpId, sb: OpId, src: OpId, row: OpId, cols: OpId) {
        let c_loop = self.push_back(Op::Loop { len: cols });
        let tile_idx = self.mad(row, cols, c_loop);
        let v = self.load(src, tile_idx);
        let old = self.load(acc, c_loop);
        let scaled = self.mul(old, sa);
        let new = self.mad(sb, v, scaled);
        self.store(acc, new, c_loop);
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
    ///
    /// Decomposed as `exp2(x * log2(e))` — backends only implement `Exp2`
    /// natively (e.g. CUDA has no `expf` in its supported instruction set).
    pub fn exp(&mut self, x: OpId) -> OpId {
        let log2e = self.const_val(std::f32::consts::LOG2_E);
        let scaled = self.mul(x, log2e);
        self.exp2(scaled)
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

    /// `x and y`
    pub fn and(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::And)
    }

    /// `x < y`
    pub fn cmplt(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Cmplt)
    }

    /// `x > y`
    pub fn cmpgt(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Cmpgt)
    }

    /// `x >= y`
    pub fn cmpge(&mut self, x: OpId, y: OpId) -> OpId {
        self.binary(x, y, BOp::Cmpge)
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
        self.push_back(Op::Wmma { dims, layout, dtype, a, b, c })
    }

    /// Hardware tile matmul: multiplies two tiles into an accumulator tile.
    pub fn matmul_tile(&mut self, x: OpId, y: OpId) -> OpId {
        self.push_back(Op::MatmulTile { x, y })
    }

    /// Hardware tile transpose.
    pub fn transpose_tile(&mut self, x: OpId) -> OpId {
        self.push_back(Op::TransposeTile { x })
    }

    /// Backend-specific assembly instruction applied to `ops`.
    pub fn asm(&mut self, asm: &str, ops: &[OpId]) -> OpId {
        let asm = TinyString::new(asm);
        let ops = TinyVec::new(ops);
        self.push_back(Op::Asm { asm, ops })
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

    /// Branchless select: `cond ? a : b` as `a*sel + b*(1-sel)` where `sel` is
    /// `cond` cast to `a`'s dtype. `cond` must be bool; `a` and `b` must share a
    /// dtype (taken from `a`).
    pub fn branchless_where(&mut self, cond: OpId, a: OpId, b: OpId) -> OpId {
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
    pub fn ternary_where(&mut self, cond: OpId, a: OpId, b: OpId) -> OpId {
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
    pub fn bitcast(&mut self, x: OpId, dtype: DType) -> OpId {
        debug_assert_eq!(self.dtype(x).bit_size(), dtype.bit_size(), "bitcast requires equal bit widths");
        self.push_back(Op::Bitcast { x, dtype })
    }

    /// `x * y + z`
    pub fn mad(&mut self, x: OpId, y: OpId, z: OpId) -> OpId {
        self.push_back(Op::Mad { x, y, z })
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
        let event = unsafe { device.launch(program.program_id, &mut *pool_ptr, &args, event_wait_list)? };
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
/// shared-memory tiles once those exist), filled by [`Kernel::mma`] (which
/// also captures the output coords for the store) and written out by
/// [`Kernel::store_partition`].
pub struct Acc {
    /// Per-warp tile shape as bound ops (e.g. m16n8k8: two consts 16, 8).
    tile: Vec<OpId>,
    /// Accumulator dtype.
    dtype: DType,
    /// Per-lane register storage (`total / 32` elements).
    storage: OpId,
    /// Output coords captured at the [`Kernel::mma`] bind.
    c_coords: Option<Vec<OpId>>,
}

impl Kernel {
    /// Partition a global tensor into a view: `src` plus a fully symbolic
    /// iteration `shape`. Strides are derived row-major. Emits no IR.
    pub fn partition<const N: usize>(&mut self, src: OpId, shape: [OpId; N]) -> Partition {
        let mut strides = Vec::with_capacity(N);
        for axis in 0..N {
            strides.push(self.row_major_stride(&shape, axis));
        }
        Partition { src, shape: shape.to_vec(), strides, dtype: self.dtype(src) }
    }

    /// Partition with explicit strides (e.g. transposed or offset views).
    pub fn partition_strided<const N: usize>(&mut self, src: OpId, shape: [OpId; N], strides: [OpId; N]) -> Partition {
        debug_assert!(shape.iter().all(|&d| !d.is_null()), "partition: shape dims must be bound ops");
        debug_assert!(strides.iter().all(|&s| !s.is_null()), "partition: strides must be bound ops");
        Partition { src, shape: shape.to_vec(), strides: strides.to_vec(), dtype: self.dtype(src) }
    }

    /// Create an accumulator: a zero-initialized per-lane register tile of
    /// `total(tile) / 32` elements.
    pub fn acc<const N: usize>(&mut self, tile: [OpId; N], dtype: DType) -> Acc {
        let mut total: Dim = 1;
        for &d in &tile {
            let dim = self
                .resolve_const(d)
                .and_then(crate::dtype::Constant::as_dim)
                .expect("acc: tile dim must resolve to a constant (const or variable op)");
            total *= dim;
        }
        debug_assert!(total > 0 && total % 32 == 0, "acc: tile must cover whole 32-lane warps");
        let storage = self.zeros(dtype, total / 32);
        Acc { tile: tile.to_vec(), dtype, storage, c_coords: None }
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

    /// Find the innermost open loop's variable by walking back from the tail.
    ///
    /// The IR is the state: ops inside a loop body always sit after their
    /// `Op::Loop`, so the nearest `Op::Loop` behind the tail *is* the open
    /// loop's variable. No builder-side mutable state needed.
    fn open_loop_var(&self) -> OpId {
        let mut op_id = self.tail;
        while !op_id.is_null() {
            if matches!(self.ops[op_id].op, Op::Loop { .. }) {
                return op_id;
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

    /// Warp matrix multiply-accumulate with fully automatic indexing.
    ///
    /// `acc`'s tile defines the mma output geometry (v1: `[16, 8]` = m16n8,
    /// f16 inputs, f32 accumulator). `k` selects the instruction variant and
    /// must resolve to a constant: 8 → `m16n8k8`, 16 → `m16n8k16`. `a` and
    /// `b` are partition views; `coords` are the chunk coordinates, assigned
    /// positionally: `rank(a) - 1` coords for `a`'s non-loop axes (in axis
    /// order), then `rank(b) - 1` for `b`'s, and the LAST coord must be the
    /// open loop variable — the K axis of both views, with a chunk of `k`.
    /// The call derives and patches the open loop's length (`shape[K] / k`),
    /// emits the lane-mapped A/B fragment loads (lane id found via
    /// `open_warp`), the wmma and the accumulator update, and captures the
    /// output coords on `acc` for [`Kernel::store_partition`].
    pub fn mma(&mut self, acc: &mut Acc, k: OpId, a: &Partition, b: &Partition, coords: &[OpId]) {
        let tile_dims: Vec<Dim> = acc
            .tile
            .iter()
            .map(|&d| self.resolve_const(d).and_then(crate::dtype::Constant::as_dim).expect("mma: acc tile dim must resolve"))
            .collect();
        debug_assert_eq!(tile_dims.len(), 2, "mma: acc tile must be rank 2 [m, n]");
        debug_assert_eq!(acc.dtype, DType::F32, "mma v1: f32 accumulator");
        debug_assert_eq!(a.dtype, b.dtype, "mma: a/b dtype mismatch");
        debug_assert_eq!(a.dtype, DType::F16, "mma v1: f16 inputs");
        debug_assert_eq!(a.shape.len(), 2, "mma v1: a must be rank 2");
        debug_assert_eq!(b.shape.len(), 2, "mma v1: b must be rank 2");
        debug_assert_eq!(
            coords.len(),
            a.shape.len() + b.shape.len() - 1,
            "mma: coords must be [a non-loop coords..., b non-loop coords..., loop coord]"
        );
        let lv = self.open_loop_var();
        debug_assert_eq!(coords[coords.len() - 1], lv, "mma: the LAST coord must be the open loop variable");

        // k must resolve to a constant; (m, n, k) must be a real mma.sync
        // shape. mma v1 emits f16 inputs with f32 accumulator, so only the
        // f16 shapes are reachable.
        let k_dim = self
            .resolve_const(k)
            .and_then(crate::dtype::Constant::as_dim)
            .expect("mma: k must resolve to a constant to select the wmma variant");
        let dims = match (tile_dims[0], tile_dims[1], k_dim) {
            (16, 8, 8) => MMADims::m16n8k8,
            (16, 8, 16) => MMADims::m16n8k16,
            other => panic!(
                "mma: unsupported (m, n, k) = {:?} for f16 inputs with f32 accumulator (supported: \
                 (16, 8, 8) -> m16n8k8, (16, 8, 16) -> m16n8k16; the s8/s4/b1 shapes need dtype \
                 parameterization in mma)",
                other
            ),
        };
        // m16n8k16 A/B fragments additionally hold the k+8..+16 half.
        let k_half = if k_dim == 8 { 0 } else { 8 };

        // Bind the loop length on first use: K chunk is `k_dim`. The div is
        // inserted BEFORE the loop so the bound is loop-invariant in the
        // linear order.
        if matches!(self.ops[lv].op, Op::Loop { len } if len.is_null()) {
            let a_k = a.shape[1];
            let len = self.insert_before(lv, Op::Binary { x: a_k, y: k, bop: BOp::Div });
            self.ops[lv].op = Op::Loop { len };
        }

        let lane = self.open_warp();
        let [c1, c2, c4, c8] = self.const_idxs([1u32, 2, 4, 8]);
        // Fixed m16n8 lane geometry: gid = lane/4, tig = lane%4.
        let gid = self.div(lane, c4);
        let tig = self.mod_(lane, c4);
        let tig2 = self.mul(tig, c2);
        let k0 = self.mul(lv, k);

        // A fragment: rows {r + gid, r + 8 + gid}, k cols {k0 + 2*tig, +1}
        // and (k=16) {k0 + 8 + 2*tig, +1}. Register order: each 2-col pair
        // holds (row, col0), (row, col1), (row_hi, col0), (row_hi, col1).
        let a_row = self.add(coords[0], gid);
        let a_row_hi = self.add(a_row, c8);
        let a_col = self.add(k0, tig2);
        let a_col_p1 = self.add(a_col, c1);
        let a_cols: Vec<OpId> = if k_half == 0 {
            vec![a_col, a_col_p1]
        } else {
            let k0_hi = self.add(k0, c8);
            let a_col_hi = self.add(k0_hi, tig2);
            let a_col_hi_p1 = self.add(a_col_hi, c1);
            vec![a_col, a_col_p1, a_col_hi, a_col_hi_p1]
        };
        let mut a_elems = Vec::new();
        for col_pair in a_cols.chunks(2) {
            for row in [a_row, a_row_hi] {
                for col in col_pair {
                    let c = self.stride_mul(*col, a.strides[1]);
                    let idx = self.mad(row, a.strides[0], c);
                    a_elems.push(self.load(a.src, idx));
                }
            }
        }
        let a_frag = self.stack(&a_elems);

        // B fragment: rows {n + gid}, the same shared k cols (consecutive K).
        let b_row = self.add(coords[1], gid);
        let mut b_elems = Vec::new();
        for col in &a_cols {
            let bc = self.stride_mul(*col, b.strides[1]);
            let idx = self.mad(b_row, b.strides[0], bc);
            b_elems.push(self.load(b.src, idx));
        }
        let b_frag = self.stack(&b_elems);

        let idx0 = self.const_idx(0u32);
        let acc_old = self.load_vector(acc.storage, idx0, 4);
        let acc_new = self.wmma(dims, MMALayout::row_col, MMADType::f16_f16_f16_f32, a_frag, b_frag, acc_old);
        self.store_vector(acc.storage, acc_new, idx0, 4);

        // Capture the output coords for the store: a's non-loop coords then
        // b's (v1 rank-2: one each).
        acc.c_coords = Some(vec![coords[0], coords[1]]);
    }

    /// Store an accumulator to a global output view with fully automatic
    /// indexing: each lane scatters its C-fragment values to their global
    /// positions (mma.sync C mapping: rows {r + gid + 8*b}, cols
    /// {c + 2*tig, + 1}, one 2-col pair per 8-row block `b`). The output
    /// coords were captured at the [`Kernel::mma`] bind; the lane id is
    /// found via `open_warp`.
    pub fn store_partition(&mut self, c: &Partition, acc: &Acc) {
        let tile_dims: Vec<Dim> = acc
            .tile
            .iter()
            .map(|&d| {
                self.resolve_const(d)
                    .and_then(crate::dtype::Constant::as_dim)
                    .expect("store_partition: acc tile dim must resolve")
            })
            .collect();
        debug_assert_eq!(tile_dims.len(), 2, "store_partition: acc tile must be rank 2 [m, n]");
        debug_assert_eq!(acc.dtype, DType::F32, "store_partition: f32 accumulator");
        debug_assert_eq!(c.shape.len(), 2, "store_partition: output must be rank 2");
        let c_coords = acc.c_coords.as_ref().expect("store_partition: acc was never bound by mma");
        debug_assert_eq!(c_coords.len(), 2, "store_partition: captured coords/output rank mismatch");

        // Row-block count per tile: each 8-row block contributes one 2-element
        // C-fragment pair per lane (m8n8: 1, m16n8: 2, m32n8: 4).
        let (m, n) = (tile_dims[0], tile_dims[1]);
        let row_blocks = match (m, n) {
            (8, 8) => 1,
            (16, 8) => 2,
            (32, 8) => 4,
            (8, 32) => todo!("store_partition: m8n32 C-fragment column layout not implemented"),
            other => panic!(
                "store_partition: unsupported acc tile (m, n) = {other:?} (valid mma.sync output tiles: \
                 8x8, 16x8, 32x8, 8x32)"
            ),
        };

        let lane = self.open_warp();
        let [c2, c4] = self.const_idxs([2u32, 4]);
        let gid = self.div(lane, c4);
        let tig = self.mod_(lane, c4);

        let row = self.add(c_coords[0], gid);
        let col = self.mad(tig, c2, c_coords[1]);
        let idx0 = self.const_idx(0u32);
        let acc_final = self.load_vector(acc.storage, idx0, (m * n / 32) as u16);
        let col0 = self.stride_mul(col, c.strides[1]);
        let mut elems = Vec::new();
        for i in 0..(m * n / 32) as usize {
            elems.push(self.devectorize_one(acc_final, i));
        }
        for b in 0..row_blocks {
            let o = if b == 0 {
                self.mad(row, c.strides[0], col0)
            } else {
                let rb = self.const_idx((8 * b) as u32);
                let row_b = self.add(row, rb);
                self.mad(row_b, c.strides[0], col0)
            };
            let o_p1 = self.add(o, c.strides[1]);
            self.store(c.src, elems[2 * b], o);
            self.store(c.src, elems[2 * b + 1], o_p1);
        }
    }
}
