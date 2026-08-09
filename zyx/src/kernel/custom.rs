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

use crate::backend::{BufferId, DeviceInfo, MemoryPool, ProgramId};
use crate::dtype::Constant;
use crate::error::BackendError;
use crate::graph::{ClassId, GraphId};
use crate::kernel::{BOp, DeviceId, IdxScope, Kernel, MMADType, MMADims, MMALayout, MemLayout, MemScope, MoveOp, Op, OpId, UOp};
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
    /// - **Manual gidx**: `define(dtype, MemScope::Global, true, len)` + [`Kernel::gidx`]
    /// - **LoadView**: `push_back(Op::LoadView(...))` — `compile()` adds thread indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use zyx::kernel::{Kernel, MemScope, MemLayout, DeviceId};
    /// use zyx::DType;
    ///
    /// let mut kernel = Kernel::new(DeviceId::AUTO);
    /// let n = 4;
    /// let inp = kernel.define(DType::F32, MemScope::Global, true, n);
    /// let gidx = kernel.group_index(0, n);
    /// let loaded = kernel.load(inp, gidx, MemLayout::Scalar);
    /// let doubled = kernel.add(loaded, loaded);
    /// let out = kernel.define(DType::F32, MemScope::Global, false, n);
    /// kernel.store(out, doubled, gidx, MemLayout::Scalar);
    /// ```
    pub fn new(device_id: DeviceId) -> Self {
        Self { ops: Slab::new(), head: OpId::NULL, tail: OpId::NULL, device_id }
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
    /// use zyx::kernel::{Kernel, MemScope, MemLayout, DeviceId};
    /// use zyx::{DType, Tensor, ZyxError};
    ///
    /// let mut kernel = Kernel::new(DeviceId::AUTO);
    /// let n = 4;
    /// let inp = kernel.define(DType::F32, MemScope::Global, true, n);
    /// let gidx = kernel.group_index(0, n);
    /// let loaded = kernel.load(inp, gidx, MemLayout::Scalar);
    /// let doubled = kernel.add(loaded, loaded);
    /// let out = kernel.define(DType::F32, MemScope::Global, false, n);
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
        self.instruction_schedule();
        self.dead_code_elimination();
        self.verify();

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Define { dtype, scope: MemScope::Global, ro, .. } = self.ops[op_id].op {
                if ro {
                    inputs.push(dtype);
                } else {
                    outputs.push(dtype);
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
        rt.initialize_devices()?;
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
        let axes = axes.to_vec();
        let in_shape = self.shape_of(x);
        debug_assert_eq!(axes.len(), in_shape.len(), "permute: axes length {} != rank {}", axes.len(), in_shape.len());
        {
            let mut sorted = axes.clone();
            sorted.sort();
            debug_assert!(
                sorted.iter().copied().eq(0..in_shape.len() as UAxis),
                "permute: axes not a valid permutation: {axes:?} for rank {}",
                in_shape.len()
            );
        }
        let shape = crate::shape::permute(&in_shape, &axes);
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Permute { axes, shape }) })
    }

    /// Reshape tensor.
    pub fn reshape(&mut self, x: OpId, shape: &[Dim]) -> OpId {
        let shape = shape.to_vec();
        let in_shape = self.shape_of(x);
        debug_assert_eq!(
            shape.iter().product::<Dim>(),
            in_shape.iter().product::<Dim>(),
            "reshape: element count mismatch: {:?} -> {:?}",
            in_shape,
            shape
        );
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Reshape { shape }) })
    }

    /// Expand tensor (adds singleton dims).
    pub fn expand(&mut self, x: OpId, shape: &[Dim]) -> OpId {
        let shape = shape.to_vec();
        let in_shape = self.shape_of(x);
        debug_assert!(
            in_shape.len() <= shape.len(),
            "expand: input rank {} > target rank {}: {:?} -> {:?}",
            in_shape.len(),
            shape.len(),
            in_shape,
            shape
        );
        for (old, new) in in_shape.iter().copied().rev().zip(shape.iter().copied().rev()) {
            debug_assert!(old == new || old == 1, "expand: incompatible dims: {old} vs {new} in {:?} -> {:?}", in_shape, shape);
        }
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Expand { shape }) })
    }

    /// Pad tensor with zeros.
    pub fn pad(&mut self, x: OpId, padding: &[(i64, i64)]) -> OpId {
        let padding = padding.to_vec();
        let in_shape = self.shape_of(x);
        debug_assert_eq!(padding.len(), in_shape.len(), "pad: padding length {} != rank {}", padding.len(), in_shape.len());
        let mut shape = in_shape.clone();
        crate::shape::pad(&mut shape, &padding);
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Pad { padding, shape }) })
    }

    /// Flip tensor axes.
    pub fn flip(&mut self, x: OpId, axes: &[UAxis]) -> OpId {
        let axes = axes.to_vec();
        let in_shape = self.shape_of(x);
        debug_assert!(!axes.is_empty(), "flip: axes must not be empty");
        for &axis in &axes {
            debug_assert!((axis as usize) < in_shape.len(), "flip: axis {axis} out of range for rank {}", in_shape.len());
        }
        self.push_back(Op::Move { x, mop: Box::new(MoveOp::Flip { axes }) })
    }

    /// Sum over the last `n_axes` dimensions.
    pub fn reduce_sum(&mut self, x: OpId, n_axes: usize) -> OpId {
        let in_shape = self.shape_of(x);
        debug_assert!(n_axes <= in_shape.len(), "reduce_sum: n_axes {} > rank {}", n_axes, in_shape.len());
        debug_assert!(n_axes > 0, "reduce_sum: n_axes == 0");
        self.push_back(Op::Reduce { x, rop: BOp::Add, n_axes })
    }

    /// Max over the last `n_axes` dimensions.
    pub fn reduce_max(&mut self, x: OpId, n_axes: usize) -> OpId {
        let in_shape = self.shape_of(x);
        debug_assert!(n_axes <= in_shape.len(), "reduce_max: n_axes {} > rank {}", n_axes, in_shape.len());
        debug_assert!(n_axes > 0, "reduce_max: n_axes == 0");
        self.push_back(Op::Reduce { x, rop: BOp::Max, n_axes })
    }

    /// Product over the last `n_axes` dimensions.
    pub fn reduce_prod(&mut self, x: OpId, n_axes: usize) -> OpId {
        self.push_back(Op::Reduce { x, rop: BOp::Mul, n_axes })
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
    pub fn define(&mut self, dtype: DType, scope: MemScope, ro: bool, shape: &[Dim]) -> OpId {
        self.push_back(Op::Define { dtype, scope, ro, shape: shape.into() })
    }

    /// Group (block) index.
    pub fn group_index(&mut self, axis: u32, len: Dim) -> OpId {
        let len = self.const_idx(len);
        self.push_back(Op::Index { len, axis, scope: IdxScope::Group })
    }

    /// Local thread index.
    pub fn local_index(&mut self, axis: u32, len: Dim) -> OpId {
        let len = self.const_idx(len);
        self.push_back(Op::Index { len, axis, scope: IdxScope::Local })
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
    pub fn vectorize(&mut self, ops: &[OpId]) -> OpId {
        let ops = TinyVec::new(ops);
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
                let dtype = rt.tensors[input.id].dtype;
                let bytes = (rt.shape(input.id).iter().product::<Dim>() * dtype.bit_size() as Dim).div_ceil(8);
                let (dev_buf, alloc_ev) = rt.pools[pool_id].allocate(bytes)?;
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
            let bytes = (shape.iter().product::<Dim>() * dtype.bit_size() as Dim).div_ceil(8);
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

        let kernel_id = rt.kernels.push(KernelData {
            outputs: Vec::new(),
            loads: Vec::new(),
            stores: Vec::new(),
            kernel: Kernel::new(device_id),
        });

        // Put to tensors
        let mut tensors = Vec::new();
        for ((dtype, shape), buf_id) in self.outputs.iter().copied().zip(shapes).zip(output_bufs) {
            let shape_id = rt.push_shape(shape);
            let id = rt.tensors.push(TensorData {
                shape_id,
                dtype,
                kernel_id,
                op_id: OpId::NULL,
                depends_on: KernelId::NULL,
                class_id: ClassId::NULL,
                graph_id: GraphId::NULL,
                rc: 1,
            });
            rt.kernels[kernel_id].outputs.push(id);
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
