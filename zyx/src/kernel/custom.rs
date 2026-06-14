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

use crate::IntoShape;
use crate::backend::ProgramId;
use crate::kernel_cache::KernelId;
use crate::tensor::TensorId;

/// Custom kernel referencing a pre-compiled program.
///
/// This struct represents a custom kernel that has been compiled
/// to a backend-specific program and cached for repeated execution.
///
/// Custom kernels are used for operations that require backend-specific
/// compilation, such as GPU tensor core operations or specialized kernels.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub(crate) struct CustomKernel {
    /// Compiled program handle.
    pub program: ProgramId,
    /// Input tensors that this kernel reads from.
    pub inputs: Vec<TensorId>,
    /// Output dtype.
    pub dtype: crate::DType,
    /// Kernel cache id for the compiled kernel IR.
    pub kernel_id: KernelId,
}

/// A compiled kernel ready for repeated execution.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// Compiled program handle (includes device).
    pub program: ProgramId,
    /// Output dtype.
    pub dtype: crate::DType,
    /// Kernel cache id for the compiled kernel IR.
    pub kernel_id: KernelId,
}

impl CompiledKernel {
    /// Execute the compiled kernel with new input tensors.
    pub fn forward(&self, inputs: &[&crate::tensor::Tensor], shape: impl IntoShape) -> crate::tensor::Tensor {
        let ids: Vec<_> = inputs.iter().map(|t| t.id).collect();
        let ck = CustomKernel { program: self.program, inputs: ids, dtype: self.dtype, kernel_id: self.kernel_id };
        let tensor_id = crate::RT
            .lock()
            .graph
            .push_wshape(crate::graph::Node::Custom(Box::new(ck)), shape.into_shape().collect());
        crate::tensor::Tensor { id: tensor_id }
    }
}
