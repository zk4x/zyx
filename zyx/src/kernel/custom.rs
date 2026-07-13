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

use crate::backend::{DeviceInfo, ProgramId};
use crate::error::BackendError;
use crate::kernel::{DeviceId, Kernel, Op, Scope};
use crate::tensor::TensorId;
use crate::{DType, IntoShape, ZyxError};

/// A compiled kernel ready for repeated execution.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// Compiled program handle (includes device).
    pub program: ProgramId,
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
    pub fn compile(mut self) -> Result<CompiledKernel, ZyxError> {
        self.unfold_movement_ops();
        self.sort_global_defines();
        self.dead_code_elimination();
        self.verify();

        let device_id = self.device_id;
        let output_dtypes: Vec<DType> = self
            .ops
            .values()
            .filter_map(|n| {
                if let Op::Define { dtype, scope: Scope::Global, ro: false, .. } = n.op {
                    Some(dtype)
                } else {
                    None
                }
            })
            .collect();
        if output_dtypes.is_empty() {
            return Err(ZyxError::BackendError(BackendError {
                status: crate::error::ErrorStatus::KernelCompilation,
                context: format!("Kernel must have at least one output.").into(),
            }));
        }
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
        let program = crate::backend::ProgramId { device: device_id, program: program_id };
        Ok(CompiledKernel { program })
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
    pub fn forward(&self, inputs: &[&crate::tensor::Tensor], shape: impl IntoShape) -> crate::tensor::Tensor {
        let ids: Vec<TensorId> = inputs.iter().map(|t| t.id).collect();
        todo!()
    }
}

impl Drop for CompiledKernel {
    fn drop(&mut self) {
        crate::RT.lock().devices[self.program.device].release(self.program.program);
    }
}
