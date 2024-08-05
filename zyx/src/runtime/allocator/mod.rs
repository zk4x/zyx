use std::{collections::BTreeMap, ffi::c_void};
use core::mem::transmute;
use crate::{tensor::TensorId, Scalar};

#[derive(Debug)]
pub(crate) enum MemoryError {}

// Untyped unsized (no length) memory pointer
#[derive(Debug, Clone, Copy)]
pub(crate) struct MemoryHandle {
    ptr: *mut c_void,
}

unsafe impl Send for MemoryHandle {}

#[derive(Debug)]
pub(crate) struct Allocator {
    // Single host memory pool
    host_memory: MemoryPool,
    // Memory pools for all devices
    device_memory: Vec<MemoryPool>,
}

// Id 0 is host memory, id 1 is first device, id 2 is second device, etc.
type MemoryPoolId = usize;
const HOST_MEMORY: MemoryPoolId = 0;

impl Allocator {
    pub(super) const fn new() -> Allocator {
        Allocator {
            host_memory: MemoryPool::Host {
                blocks: Vec::new(),
                buffers: BTreeMap::new(),
            },
            device_memory: Vec::new(),
        }
    }

    pub(super) fn store_host<T: Scalar>(&mut self, data: &[T], tensor_id: TensorId) -> Result<(), MemoryError> {
        self.host_memory.store(data, tensor_id)
    }

    // TODO functions for storing on devices by copying from host or from disk
}

#[derive(Debug, Clone, Copy)]
struct Buffer {
    handle: MemoryHandle,
    offset: usize,
}

// Each memory pool represents one physical device
// That is RAM is one pool (even when accessable by multiple executors),
// VRAM on multiple GPUs is represented by multiple pools
// Memory pool allocates lazily, reuses memory and never deallocates.
// Only way to deallocate memory pool is to drop it.
#[derive(Debug)]
enum MemoryPool {
    Host {
        // Allocated memory blocks
        blocks: Vec<MemoryBlock>,
        // Just mapping from tensors to memory blocks
        buffers: BTreeMap<TensorId, Buffer>,
    }
    // CUDA
    // HSA
    // OpenCL
}

#[derive(Debug)]
struct MemoryBlock {
    handle: MemoryHandle,
    byte_size: usize,
    alignment: usize,
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        match self {
            Self::Host { blocks, .. } => {
                for block in blocks {
                    todo!()
                }
            }
        }
    }
}

impl MemoryPool {
    fn store<T: Scalar>(&mut self, data: &[T], tensor_id: TensorId) -> Result<(), MemoryError> {
        let byte_size = data.len() * T::dtype().byte_size();
        let buffer = self.alloc(byte_size)?;
        self.copy(buffer, unsafe { transmute(data) })?;
        self.map(buffer, tensor_id);
        Ok(())
    }

    fn alloc(&mut self, byte_size: usize) -> Result<Buffer, MemoryError> {
        todo!()
    }

    fn copy(&mut self, buffer: Buffer, data: &[u8]) -> Result<(), MemoryError> {
        todo!()
    }

    fn map(&mut self, buffer: Buffer, tensor_id: TensorId) {
        match self {
            Self::Host { buffers, .. } => {
                buffers.insert(tensor_id, buffer);
            }
        }
    }
}
