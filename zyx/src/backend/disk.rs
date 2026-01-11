use std::{
    fs::File,
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
};

use super::{BufferId, Event, MemoryPool};
use crate::{
    error::{BackendError, ErrorStatus},
    runtime::Pool,
    shape::Dim,
    slab::Slab,
};

#[derive(Debug)]
pub struct DiskMemoryPool {
    free_bytes: Dim,
    buffers: Slab<BufferId, DiskBuffer>,
}

#[derive(Debug)]
struct DiskBuffer {
    bytes: Dim,
    path: PathBuf,
    offset_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct DiskEvent {}

#[allow(clippy::unnecessary_wraps)]
pub(super) fn initialize_pool(memory_pools: &mut Vec<Pool>, debug_dev: bool) -> Result<(), BackendError> {
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
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub const fn deinitialize(&mut self) {
        let _ = self;
    }

    pub const fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn buffer_from_path(&mut self, bytes: Dim, path: &Path, offset_bytes: u64) -> BufferId {
        // TODO perhaps add verification that the file exists and it contains enough bytes at given offset
        let id = self.buffers.push(DiskBuffer { bytes, path: path.into(), offset_bytes });
        //println!("Create buffer={id:?} on disk from path={path:?}");
        id
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn deallocate(&mut self, buffer_id: BufferId, event_wait_list: Vec<Event>) {
        //println!("Deallocate buffer={buffer_id:?} from the disk");
        let _ = event_wait_list;
        if self.buffers.contains_key(buffer_id) {
            let buffer = unsafe { self.buffers.remove_and_return(buffer_id) };
            self.free_bytes += buffer.bytes;
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn pool_to_host(
        &mut self,
        src: BufferId,
        dst: &mut [u8],
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        let _ = event_wait_list;
        let buffer = &self.buffers[src];
        let f = File::open(&buffer.path).unwrap();
        f.read_exact_at(dst, buffer.offset_bytes)
            .map_err(|err| BackendError { status: ErrorStatus::MemoryCopyP2H, context: format!("{err}").into() })
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        let _ = self;
        let _ = events;
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn release_events(&mut self, events: Vec<Event>) {
        let _ = self;
        let _ = events;
    }
}
