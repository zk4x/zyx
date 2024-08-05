#[derive(Debug)]
pub(crate) struct HostMemoryPool {
    blocks: Vec<MemoryBlock>,
    capacity: usize,
}

#[derive(Debug)]
struct MemoryBlock {
    ptr: *mut u8,
    byte_size: usize,
    alignment: usize,
}

impl HostMemoryPool {
    fn new() -> HostMemoryPool {
        let system = sysinfo::System::new();
        HostMemoryPool {
            blocks: Vec::new(),
            // Use up to 80% of total memory
            capacity: system.total_memory() as usize * 4 / 5,
        }
    }
}

// ptr is in RAM, so this should be sendable
unsafe impl Send for MemoryBlock {}

impl Drop for HostMemoryPool {
    fn drop(&mut self) {
        for block in &mut self.blocks {
            let _ = unsafe { Vec::from_raw_parts(block.ptr, block.byte_size, block.byte_size) };
        }
    }
}
