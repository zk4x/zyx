//! Converts graph to kernels and schedules them to devices

use crate::{runtime::Runtime, slab::SlabId, tensor::TensorId, Set, ZyxError};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct KernelId(u32);

impl SlabId for KernelId {
    const ZERO: Self = Self(0);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl From<usize> for KernelId {
    fn from(value: usize) -> Self {
        KernelId(value as u32)
    }
}

impl From<KernelId> for usize {
    fn from(value: KernelId) -> Self {
        value.0 as usize
    }
}

impl Runtime {
    // 1. gets a set of tensors which need to be processed and in which order
    // 2. generates kernels from them
    // 3. assigns those kernels to devices, compiles and launches them
    #[allow(clippy::cognitive_complexity)]
    pub fn realize2(&mut self, to_eval: &Set<TensorId>) -> Result<(), ZyxError> {
        Ok(())
    }
}
