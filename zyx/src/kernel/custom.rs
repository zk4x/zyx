// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

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

use crate::Map;
use crate::Set;
use crate::backend::{BufferId, DeviceInfo, MemoryPool, ProgramId};
use crate::dtype::Constant;
use crate::error::BackendError;
use crate::graph::{ClassId, GraphId};
use crate::kernel::{
    BOp, DeviceId, IDX_T, IdxKind, Kernel, MMADType, MMADims, MMALayout, MemLayout, MemScope, MoveOp, Op, OpId, ParamKind, UOp,
};
use crate::runtime::{KernelData, KernelId, TensorData};
use crate::shape::UAxis;
use crate::slab::{Slab, SlabId};
use crate::types::{TinyString, TinyVec};
use crate::{DType, IntoShape, Tensor, ZyxError, shape::Dim};

/// A compiled kernel ready for repeated execution.
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
    /// - **Manual gidx**: `param(dtype, ParamKind::Global, shape)` + [`Kernel::gidx`]
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
    /// let shape = kernel.add_shape(&[n]);
    /// let len = kernel.const_idx(n);
    /// let inp = kernel.param(DType::F32, ParamKind::Global, shape);
    /// let gidx = kernel.group_index(0, len);
    /// let loaded = kernel.load(inp, gidx, MemLayout::Scalar);
    /// let doubled = kernel.add(loaded, loaded);
    /// let out = kernel.param(DType::F32, ParamKind::GlobalMut, shape);
    /// kernel.store(out, doubled, gidx, MemLayout::Scalar);
    /// ```
    pub fn new(device_id: DeviceId) -> Self {
        Self { ops: Slab::new(), head: OpId::NULL, tail: OpId::NULL, device_id, shape_cache: Map::default() }
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
    /// use zyx::kernel::{Kernel, MemLayout, DeviceId, ParamKind};
    /// use zyx::{DType, Tensor, ZyxError};
    ///
    /// let mut kernel = Kernel::new(DeviceId::AUTO);
    /// let n = 4;
    /// let shape = kernel.add_shape(&[n]);
    /// let len = kernel.const_idx(n);
    /// let inp = kernel.param(DType::F32, ParamKind::Global, shape);
    /// let gidx = kernel.group_index(0, len);
    /// let loaded = kernel.load(inp, gidx, MemLayout::Scalar);
    /// let doubled = kernel.add(loaded, loaded);
    /// let out = kernel.param(DType::F32, ParamKind::GlobalMut, shape);
    /// kernel.store(out, doubled, gidx, MemLayout::Scalar);
    ///
    /// let compiled = kernel.compile()?;
    /// let x = Tensor::from([1.0f32, 2.0, 3.0, 4.0]);
    /// let result = compiled.forward(&[&x], vec![n])?;
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
        let program = crate::backend::ProgramId { device: device_id, program: program_id };
        Ok(CompiledKernel { program, inputs, outputs })
    }

    // Run autotuning then compile the kernel.
    // Consumes the kernel.
    /*#[allow(unused)]
    fn autotune(self) -> Result<CompiledKernel, crate::ZyxError> {
        self.compile()
    }*/

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

    /// Define a kernel param (a launch argument).
    pub fn param(&mut self, dtype: DType, kind: ParamKind, shape: OpId) -> OpId {
        self.push_back(Op::Param { dtype, kind, shape })
    }

    /// Build a shape op from dimension values.
    ///
    /// A negative dim (`-1`) marks a dynamic/symbolic dimension and becomes a
    /// scalar `Param { kind: Variable }` of `IDX_T`; any nonnegative dim
    /// becomes a const index. Returns `OpId::NULL` for rank-0, the single dim op for rank-1,
    /// or a `Stack` for higher ranks.
    pub fn add_shape(&mut self, shape: &[Dim]) -> OpId {
        let dim_ops: Vec<OpId> = shape
            .iter()
            .map(|&d| {
                if d < 0 {
                    self.param(IDX_T, ParamKind::Variable, OpId::NULL)
                } else {
                    self.const_idx(d)
                }
            })
            .collect();
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

    /// Group (block) index.
    pub fn group_index(&mut self, axis: u32, len: OpId) -> OpId {
        self.push_back(Op::Index { axis, kind: IdxKind::Group(len) })
    }

    /// Local thread index.
    pub fn local_index(&mut self, axis: u32, len: u32) -> OpId {
        self.push_back(Op::Index { axis, kind: IdxKind::Local(len) })
    }

    /// Store `x` to `dst` at `index`.
    pub fn store(&mut self, dst: OpId, x: OpId, index: OpId, layout: MemLayout) {
        self.push_back(Op::Store { dst, src: x, index, layout });
    }

    /// Load from `src` at `index`.
    pub fn load(&mut self, src: OpId, index: OpId, layout: MemLayout) -> OpId {
        self.push_back(Op::Load { src, index, layout })
    }

    /// Begin a loop.
    pub fn loop_(&mut self, len: OpId) -> OpId {
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
        self.push_back(Op::Devectorize { vec, idx })
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

    /// Bitcast to a different dtype.
    pub fn bitcast(&mut self, _x: OpId, _dtype: DType) -> OpId {
        todo!()
    }

    /// `x * y + z`
    pub fn mad(&mut self, x: OpId, y: OpId, z: OpId) -> OpId {
        self.push_back(Op::Mad { x, y, z })
    }
}

impl CompiledKernel {
    /// Returns the DeviceInfo for the device this kernel was compiled on.
    pub fn device_info(&self) -> DeviceInfo {
        crate::RT.lock().devices[self.program.device].info().clone()
    }

    /// Execute the compiled kernel with new input tensors.
    pub fn forward(&self, inputs: &[&Tensor], shapes: Vec<impl IntoShape>) -> Result<Vec<Tensor>, ZyxError> {
        debug_assert_eq!(inputs.len(), self.inputs.len());
        let shapes: Vec<Vec<Dim>> = shapes.into_iter().map(|s| s.into_shape().collect()).collect();
        debug_assert_eq!(shapes.len(), self.outputs.len());

        let mut rt = crate::RT.lock();
        for input in inputs {
            if !rt.buffer_map.contains_key(&input.id) {
                rt.add_store(input.id)?;
            }
        }
        debug_assert!(inputs.iter().all(|input| rt.buffer_map.contains_key(&input.id)));

        // Launch kernel
        let device_id = self.program.device;
        let pool_id = rt.devices[device_id].memory_pool_id();

        let mut input_bufs = Vec::new();
        let mut all_bufs = BTreeSet::new();
        let mut event_wait_list = Vec::new();
        for input in inputs {
            let buf_id = rt.buffer_map[&input.id];
            let dev_buf_id = if buf_id.pool != pool_id {
                let mut pool_events = Vec::new();
                let keys: Vec<BTreeSet<BufferId>> = rt.events.keys().filter(|k| k.contains(&buf_id)).cloned().collect();
                for key in keys {
                    pool_events.push(rt.events.remove(&key).unwrap());
                }
                let dtype = rt.dtype(input.id);
                let bytes = ((rt.shape(input.id).iter().product::<Dim>() * dtype.bit_size() as Dim) + 7) / 8;
                let alloc_bytes = bytes + dtype.bit_size() as Dim / 8;
                let (dev_buf, alloc_ev) = rt.pools[pool_id].allocate(alloc_bytes)?;
                pool_events.push(alloc_ev);
                let dev_buf_id = BufferId { pool: pool_id, buffer: dev_buf };
                let src_pool_ptr: *mut MemoryPool = &mut rt.pools[buf_id.pool];
                let copy_ev = rt.pools[pool_id].pool_to_pool(
                    unsafe { &mut *src_pool_ptr },
                    buf_id.buffer,
                    dev_buf_id.buffer,
                    pool_events,
                )?;
                event_wait_list.push(copy_ev);
                dev_buf_id
            } else {
                let keys: Vec<BTreeSet<BufferId>> = rt.events.keys().filter(|k| k.contains(&buf_id)).cloned().collect();
                for key in keys {
                    event_wait_list.push(rt.events.remove(&key).unwrap());
                }
                buf_id
            };
            input_bufs.push(dev_buf_id.buffer);
            all_bufs.insert(dev_buf_id);
        }

        let mut output_bufs = Vec::new();
        for (i, shape) in shapes.iter().enumerate() {
            let dtype = self.outputs[i];
            let bytes = ((shape.iter().product::<Dim>() * dtype.bit_size() as Dim) + 7) / 8;
            let (buf, ev) = rt.pools[pool_id].allocate(bytes)?;
            event_wait_list.push(ev);
            let buf_id = BufferId { pool: pool_id, buffer: buf };
            output_bufs.push(buf_id);
            all_bufs.insert(buf_id);
        }

        let mut args = input_bufs;
        for buf in &output_bufs {
            args.push(buf.buffer);
        }
        let pool_ptr = &mut rt.pools[pool_id] as *mut MemoryPool;
        let device = &mut rt.devices[device_id];
        let event = unsafe { device.launch(self.program.program, &mut *pool_ptr, &args, event_wait_list)? };
        rt.events.insert(all_bufs, event);

        // Put to tensors. Each output gets its own load kernel: a realized
        // tensor is referenced through a read-only global Param (like
        // Runtime::eagerify/add_store does), never a NULL op id. NULL op ids
        // break any eager op built on the forward result (e.g. a .cast()).
        let mut tensors = Vec::new();
        for ((dtype, buf_id), shape) in self.outputs.iter().copied().zip(output_bufs).zip(shapes) {
            let id = rt.tensors.push(TensorData {
                kernel_id: KernelId::NULL,
                op_id: OpId::NULL,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });
            let mut kernel = Kernel::new(DeviceId::AUTO);
            let shape_op = kernel.add_shape(&shape);
            let op_id = kernel.push_back(Op::Param { dtype, kind: ParamKind::Global, shape: shape_op });
            let load_kid =
                rt.kernels.push(KernelData { outputs: Set::from_iter([id]), loads: vec![id], stores: Vec::new(), kernel });
            rt.tensors[id].kernel_id = load_kid;
            rt.tensors[id].op_id = op_id;
            rt.retain(id);
            rt.buffer_map.insert(id, buf_id);
            tensors.push(Tensor { id })
        }

        Ok(tensors)
    }
}

impl Drop for CompiledKernel {
    fn drop(&mut self) {
        crate::RT.lock().devices[self.program.device].release(self.program.program);
    }
}
