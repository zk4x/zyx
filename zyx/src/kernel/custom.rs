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
use crate::error::BackendError;
use crate::kernel::{DeviceId, Kernel, Op, OpId, Scope};
use crate::runtime::{KernelData, TensorData, TensorState};
use crate::{DType, IntoShape, Tensor, ZyxError, shape::Dim};

/// A compiled kernel ready for repeated execution.
#[derive(Debug)]
pub struct CompiledKernel {
    program: ProgramId,
    inputs: Vec<DType>,
    outputs: Vec<DType>,
}

impl Kernel {
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
    /// let gidx = kernel.global_index(0, n);
    /// let loaded = kernel.load(inp, gidx, MemLayout::Scalar);
    /// let doubled = kernel.add(loaded, loaded);
    /// let out = kernel.define(DType::F32, Scope::Global, false, n);
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
        self.unfold_movement_ops();
        self.sort_global_defines();
        self.dead_code_elimination();
        self.verify();

        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let mut op_id = self.head;
        while !op_id.is_null() {
            if let Op::Define { dtype, scope: Scope::Global, ro, .. } = self.ops[op_id].op {
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
                context: format!("Kernel must have at least one output.").into(),
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
                let bytes = (rt.shape(input.id).iter().product::<Dim>() * dtype.bit_size() as Dim + 7) / 8;
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
            let bytes = (shape.iter().product::<Dim>() * dtype.bit_size() as Dim + 7) / 8;
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
                state: TensorState::Eager { kernel_id, op_id: OpId::NULL, pending_store: false },
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
