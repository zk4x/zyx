// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

use super::{Event, MemoryPool, PoolBufferId, PoolId};
use crate::{
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    shape::Dim,
    slab::Slab,
};

#[derive(Debug)]
pub struct HostMemoryPool {
    free_bytes: Dim,
    buffers: Slab<PoolBufferId, HostBuffer>,
}

#[derive(Debug)]
enum HostBuffer {
    Variable(Box<[u8]>),
    Buffer(Box<[u8]>),
}

#[derive(Debug, Clone)]
pub struct HostEvent;

fn detect_host_memory_bytes() -> u64 {
    let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: u64 = rest.split_whitespace().next().and_then(|s| s.parse().ok()).unwrap_or(0);
            if kb > 0 {
                return kb * 1024;
            }
        }
    }
    1024 * 1024 * 1024
}

#[allow(clippy::unnecessary_wraps)]
pub(super) fn initialize_pool(memory_pools: &mut Slab<PoolId, MemoryPool>, debug_dev: bool) -> Result<(), BackendError> {
    if debug_dev {
        println!("[host] initialized");
    }
    let total_bytes = detect_host_memory_bytes();
    let pool = MemoryPool::Host(HostMemoryPool { free_bytes: total_bytes, buffers: Slab::new() });
    if debug_dev {
        println!("[host] device total memory: {} MB", total_bytes / (1024 * 1024));
    }
    memory_pools.push(pool);
    Ok(())
}

impl HostMemoryPool {
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub const fn deinitialize(&mut self) {
        let _ = self;
    }

    pub const fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn store_variable(&mut self, variable: Constant) -> PoolBufferId {
        let buffer = HostBuffer::Variable(variable.to_le_bytes().into_boxed_slice());
        self.buffers.push(buffer)
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
        let bytes: usize = bytes
            .try_into()
            .map_err(|_| BackendError { status: ErrorStatus::MemoryAllocation, context: "allocation size too large".into() })?;
        if self.free_bytes < bytes as Dim {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "OOM".into() });
        }
        self.free_bytes -= bytes as Dim;
        let buffer = HostBuffer::Buffer(vec![0u8; bytes].into_boxed_slice());
        let id = self.buffers.push(buffer);
        Ok((id, Event::Host(HostEvent)))
    }

    pub fn insert(&mut self, buf: Box<[u8]>) -> PoolBufferId {
        self.free_bytes -= buf.len() as Dim;
        self.buffers.push(HostBuffer::Buffer(buf))
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn deallocate(&mut self, buffer_id: PoolBufferId, event_wait_list: Vec<Event>) {
        let _ = event_wait_list;
        if self.buffers.contains_key(buffer_id) {
            match unsafe { self.buffers.remove_and_return(buffer_id) } {
                HostBuffer::Variable(..) => {}
                HostBuffer::Buffer(buffer) => self.free_bytes += buffer.len() as Dim,
            }
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn host_to_pool(&mut self, src: &[u8], dst: PoolBufferId, event_wait_list: Vec<Event>) -> Result<Event, BackendError> {
        let _ = event_wait_list;
        let buffer = self
            .buffers
            .get_mut(dst)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyH2P, context: "invalid buffer id".into() })?;
        let HostBuffer::Buffer(buffer) = buffer else { unreachable!() };
        let len = src.len().min(buffer.len());
        buffer[..len].copy_from_slice(&src[..len]);
        Ok(Event::Host(HostEvent))
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    #[allow(clippy::unnecessary_wraps)]
    pub fn pool_to_host(&mut self, src: PoolBufferId, dst: &mut [u8], event_wait_list: Vec<Event>) -> Result<(), BackendError> {
        let _ = event_wait_list;
        let buffer = &self.buffers[src];
        let HostBuffer::Buffer(buffer) = buffer else { unreachable!() };
        let len = dst.len().min(buffer.len());
        dst[..len].copy_from_slice(&buffer[..len]);
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        let _ = self;
        let _ = events;
        Ok(())
    }

    #[allow(unused)]
    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn release_events(&mut self, events: Vec<Event>) {
        let _ = self;
        let _ = events;
    }

    pub fn get_buffer(&self, id: PoolBufferId) -> &[u8] {
        match &self.buffers[id] {
            HostBuffer::Variable(items) | HostBuffer::Buffer(items) => items,
        }
    }

    /// Get a mutable raw pointer to the buffer's data
    pub fn buffer_ptr_mut(&mut self, id: PoolBufferId) -> *mut u8 {
        match &mut self.buffers[id] {
            HostBuffer::Variable(items) | HostBuffer::Buffer(items) => items.as_mut_ptr(),
        }
    }
}
