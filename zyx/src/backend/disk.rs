use crate::{runtime::Pool, shape::Dimension, slab::{Id, Slab}};
use super::{BackendError, Device, ErrorStatus, Event, MemoryPool};

pub struct DiskConfig {
    enabled: bool,
}

pub struct DiskMemoryPool {
    free_bytes: Dimension,
    buffers: Slab<DiskBuffer>,
}

struct DiskBuffer {
    bytes: Dimension,
}

#[derive(Debug, Clone)]
pub struct DiskEvent {}

pub(super) fn initialize_device(
    config: &DiskConfig,
    memory_pools: &mut Vec<Pool>,
    _devices: &mut Vec<Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if !config.enabled {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: "Configured out.".into(),
        });
    }
    if debug_dev {
        println!("Using dummy backend");
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

    pub fn allocate(&mut self, bytes: Dimension) -> Result<(Id, Event), BackendError> {
        if self.free_bytes > bytes {
            self.free_bytes -= bytes;
        } else {
            return Err(BackendError {
                status: ErrorStatus::MemoryAllocation,
                context: "OOM".into(),
            });
        }
        todo!();
        //Ok((id, Event::Disk(DiskEvent { })))
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

    pub fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: crate::slab::Id,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let _ = src;
        let _ = dst;
        let _ = event_wait_list;
        Ok(Event::Disk(DiskEvent { }))
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