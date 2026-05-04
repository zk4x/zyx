// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::{Tensor, kernel::Kernel, kernel_cache::KernelId, tensor::TensorId};

/// Custom kernel, for now custom kernels can have only 1 output
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub struct CustomKernel {
    params: Vec<TensorId>,
    kernel: KernelId,
}

impl Kernel {
    #[allow(unused)]
    pub fn into_tensor_op(self) -> fn(&Tensor) -> Tensor {
        todo!()
    }

    #[allow(unused)]
    pub fn into_backward_op(self) -> fn(&Tensor) -> Tensor {
        todo!()
    }
}
