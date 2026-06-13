// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use crate::backend::DeviceId;
use crate::{kernel_cache::KernelId, tensor::TensorId};

/// Custom kernel identified by a cached `KernelId` with input tensor mappings.
///
/// The actual `Kernel` IR lives in the kernel cache, avoiding duplication
/// in graph nodes and enabling optimization reuse across identical kernels.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub struct CustomKernel {
    /// Handle into the kernel cache for the pre-built Kernel IR.
    pub kernel: KernelId,
    /// Input tensors that this kernel reads from.
    pub inputs: Vec<TensorId>,
    /// Device the kernel should run on.
    pub device: DeviceId,
}
