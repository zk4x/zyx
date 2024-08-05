use std::collections::BTreeMap;
use crate::{tensor::TensorId, Scalar};

type MemoryPoolId = u32;
const HOST_MEMORY: u32 = 0;

#[derive(Debug)]
pub(crate) enum MemoryError {
    OutOfMemory,
}

// Untyped unsized (no length) memory pointer
#[derive(Debug, Clone)]
pub(crate) struct Buffer {
    ptr: *mut u8,
    byte_size: usize,
    offset: usize,
    memory_pool_id: MemoryPoolId,
}

unsafe impl Send for Buffer {}

#[derive(Debug)]
pub(crate) struct Memory {
    // Memory pools for all devices
    // Id 0 is host memory, id 1 is first device, id 2 is second device, etc.
    memory_pools: Vec<MemoryPool>,
    // Just mapping from tensors to buffers
    // Each buffer represents memory stored in single physical memory
    buffers: BTreeMap<TensorId, Vec<Buffer>>,
}

impl Memory {
    pub(super) const fn new() -> Memory {
        Memory {
            memory_pools: Vec::new(),
            buffers: BTreeMap::new(),
        }
    }

    pub(super) fn store_host<T: Scalar>(&mut self, data: &[T], tensor_id: TensorId) -> Result<(), MemoryError> {
        if self.memory_pools.len() == 0 {
            self.init_host();
        }
        let buffer = self.memory_pools[0].store(data, HOST_MEMORY)?;
        self.buffers.insert(tensor_id, vec![buffer]);
        Ok(())
    }

    pub(super) fn is_stored(&self, x: TensorId) -> bool {
        self.buffers.contains_key(&x)
    }

    pub(super) fn load<T: Scalar>(&mut self, x: TensorId) -> Vec<T> {
        // Load buffers from
        let mut byte_size = 0;
        for buffer in &self.buffers[&x] {
            byte_size += buffer.byte_size;
        }
        let mut res: Vec<T> = Vec::with_capacity(byte_size/T::byte_size());
        unsafe { res.set_len(byte_size/T::byte_size()) };
        let mut offset = 0;
        for buffer in &self.buffers[&x] {
            self.memory_pools[buffer.memory_pool_id as usize].copy_to_vec(res.as_mut_ptr().cast(), buffer, offset);
            offset += buffer.byte_size;
        }
        res
    }

    fn init_host(&mut self) {
        let system = sysinfo::System::new();
        self.memory_pools.push(MemoryPool::Host {
            blocks: Vec::new(),
            // Use up to 80% of total memory
            capacity: system.total_memory() as usize * 4 / 5,
        });
    }

    // TODO functions for storing on devices by copying from host or from disk
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
        // Maximum available memory capacity
        capacity: usize,
    }
    // CUDA
    // HSA
    // OpenCL
}

#[derive(Debug)]
struct MemoryBlock {
    ptr: *mut u8,
    byte_size: usize,
    alignment: usize,
}

// ptr is in RAM, so this should be sendable
unsafe impl Send for MemoryBlock {}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        match self {
            Self::Host { blocks, .. } => {
                for block in blocks {
                    let _ = unsafe { Vec::from_raw_parts(block.ptr, block.byte_size, block.byte_size) };
                }
            }
        }
    }
}

impl MemoryPool {
    fn store<T: Scalar>(&mut self, data: &[T], memory_pool_id: MemoryPoolId) -> Result<Buffer, MemoryError> {
        let byte_size = data.len() * T::dtype().byte_size();
        let mut buffer = self.alloc(byte_size, memory_pool_id)?;
        self.copy(&mut buffer, data.as_ptr().cast(), data.len()*T::byte_size())?;
        Ok(buffer)
    }

    // Always aligned to 16 bytes (that is the biggest dtype)
    // Allocate new memory, or return already allocated memory that can be reused
    fn alloc(&mut self, byte_size: usize, memory_pool_id: MemoryPoolId) -> Result<Buffer, MemoryError> {
        match self {
            Self::Host { blocks, .. } => {
                // For now we will just alloc byte_size rounded to 100 MB of memory per block
                let r = 100*1024*1024;
                let alloc_size = byte_size / r + if byte_size % r == 0 { 0 } else { r };
                let mut block = Vec::<u128>::with_capacity(alloc_size/16);
                unsafe { block.set_len(alloc_size/16) };
                let block = block.into_boxed_slice();
                let block = Box::into_raw(block);
                let ptr = block.cast();
                blocks.push(MemoryBlock { ptr, byte_size: alloc_size, alignment: 16 });
                Ok(Buffer {
                    ptr,
                    offset: 0,
                    byte_size,
                    memory_pool_id,
                })
            }
        }
    }

    fn copy(&mut self, buffer: &mut Buffer, data: *const u8, bytes: usize) -> Result<(), MemoryError> {
        match self {
            MemoryPool::Host { .. } => {
                unsafe { std::ptr::copy_nonoverlapping(data, buffer.ptr.cast(), bytes) };
                Ok(())
            },
        }
    }

    fn copy_to_vec(&mut self, res: *mut u8, buffer: &Buffer, offset: usize) {
        let src = buffer.ptr.wrapping_add(offset);
        unsafe { std::ptr::copy_nonoverlapping(src, res, buffer.byte_size) };
    }
}
