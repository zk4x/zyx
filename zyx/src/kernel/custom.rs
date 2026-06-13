// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::backend::ProgramId;
use crate::tensor::TensorId;

/// Custom kernel referencing a pre-compiled program.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub(crate) struct CustomKernel {
    /// Compiled program handle.
    pub program: ProgramId,
    /// Input tensors that this kernel reads from.
    pub inputs: Vec<TensorId>,
    /// Output dtype.
    pub dtype: crate::DType,
}

/// A compiled kernel ready for repeated execution.
#[derive(Debug, Clone)]
pub struct CompiledKernel {
    /// Compiled program handle (includes device).
    pub program: ProgramId,
    /// Output shape.
    pub shape: Vec<crate::shape::Dim>,
    /// Output dtype.
    pub dtype: crate::DType,
}

impl CompiledKernel {
    /// Execute the compiled kernel with new input tensors.
    pub fn forward(&self, inputs: &[&crate::tensor::Tensor]) -> crate::tensor::Tensor {
        let ids: Vec<_> = inputs.iter().map(|t| t.id).collect();
        let ck = CustomKernel { program: self.program, inputs: ids, dtype: self.dtype };
        let tensor_id = crate::RT
            .lock()
            .graph
            .push_wshape(crate::graph::Node::Custom(Box::new(ck)), self.shape.clone());
        crate::tensor::Tensor { id: tensor_id }
    }
}
