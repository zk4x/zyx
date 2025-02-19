use std::{fs::File, os::unix::fs::FileExt, path::{Path, PathBuf}};

use crate::{runtime::Pool, shape::Dimension, slab::{Id, Slab}};
use super::{BackendError, Event, MemoryPool};

#[derive(Debug)]
pub struct DiskMemoryPool {
    free_bytes: Dimension,
    buffers: Slab<DiskBuffer>,
}

#[derive(Debug)]
struct DiskBuffer {
    bytes: Dimension,
    path: PathBuf,
    offset_bytes: u64,
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
        free_bytes: 0, // Non allocatable
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

    pub fn from_path(&mut self, bytes: Dimension, path: &Path, offset_bytes: u64) -> Result<Id, BackendError> {
        let id = self.buffers.push(DiskBuffer { bytes, path: path.into(), offset_bytes });
        // TODO perhaps add verification that the file exists and it contains enough bytes at given offset
        Ok(id)
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
        let _ = event_wait_list;
        let buffer = &self.buffers[src];
        let f = File::open(&buffer.path).unwrap();
        f.read_exact_at(dst, buffer.offset_bytes).unwrap();
        //println!("Read from disk: {buffer:?}, data: {:?}", dst[..50]);
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