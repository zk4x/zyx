use crate::{runtime::Pool, shape::Dimension, slab::{Id, Slab}};
use super::{BackendError, Event, MemoryPool};

pub struct DiskMemoryPool {
    free_bytes: Dimension,
    buffers: Slab<DiskBuffer>,
}

struct DiskBuffer {
    bytes: Dimension,
}

#[derive(Debug, Clone)]
pub struct DiskEvent {}

pub(super) fn initialize_pool(
    memory_pools: &mut Vec<Pool>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if debug_dev {
        println!("Using disk backend");
    }
    let pool = MemoryPool::Disk(DiskMemoryPool {
        free_bytes: 1024 * 1024 * 1024 * 1024 * 1024,
        buffers: Slab::new(),
    });
    memory_pools.push(Pool::new(pool));
    Ok(())
}

impl DiskMemoryPool {
    pub fn deinitialize(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    pub fn free_bytes(&self) -> Dimension {
        self.free_bytes
    }

    pub fn from_path(&self, path: impl AsRef<std::path::Path>, offset_bytes: u64) -> Result<(Id, Event), BackendError> {
        todo!()
    }

    pub fn deallocate(
        &mut self,
        buffer_id: crate::slab::Id,
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        let _ = event_wait_list;
        let buffer = unsafe { self.buffers.remove_and_return(buffer_id) };
        self.free_bytes += buffer.bytes;
        Ok(())
    }

    pub fn pool_to_host(
        &mut self,
        src: crate::slab::Id,
        dst: &mut [u8],
        event_wait_list: Vec<super::Event>,
    ) -> Result<(), BackendError> {
        let _ = src;
        let _ = dst;
        let _ = event_wait_list;
        Ok(())
    }

    pub fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        let _ = events;
        Ok(())
    }

    pub fn release_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        let _ = events;
        Ok(())
    }
}