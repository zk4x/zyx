// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use super::{Event, Pool, PoolBufferId};
use crate::{
    error::{BackendError, ErrorStatus},
    shape::Dim,
    slab::Slab,
};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug)]
pub struct HostMemoryPool {
    free_bytes: Dim,
    buffers: Slab<PoolBufferId, Box<[u8]>>,
}

#[derive(Debug, Clone)]
pub struct HostEvent;

/// Constructs the global host pool. Infallible — `Host` is always available.
pub(super) fn ensure_pool() -> HostMemoryPool {
    let total_bytes = detect_host_memory_bytes();
    if super::debug_backends() {
        println!("[host] initialized");
        println!("[host] device total memory: {} MB", total_bytes / (1024 * 1024));
    }
    HostMemoryPool { free_bytes: total_bytes as i64, buffers: Slab::new() }
}

/// Process-wide global host pool. Owned here — `mod.rs` only holds the `Pool::Host` handle.
static HOST_POOL: OnceLock<Arc<Mutex<HostMemoryPool>>> = OnceLock::new();

pub(super) fn pool() -> Arc<Mutex<HostMemoryPool>> {
    HOST_POOL.get_or_init(|| Arc::new(Mutex::new(ensure_pool()))).clone()
}

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

impl HostMemoryPool {
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub const fn deinitialize(&mut self) {
        let _ = self;
    }

    pub const fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
        let bytes: usize = bytes
            .try_into()
            .map_err(|_| BackendError { status: ErrorStatus::MemoryAllocation, context: "allocation size too large".into() })?;
        if self.free_bytes < bytes as Dim {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "OOM".into() });
        }
        self.free_bytes -= bytes as Dim;
        let buffer = vec![0u8; bytes].into_boxed_slice();
        let id = self.buffers.push(buffer);
        Ok((id, Event::Host(HostEvent)))
    }

    pub fn insert(&mut self, buf: Box<[u8]>) -> PoolBufferId {
        self.free_bytes -= buf.len() as Dim;
        self.buffers.push(buf)
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn deallocate(&mut self, buffer_id: PoolBufferId, event_wait_list: Vec<Event>) {
        let _ = event_wait_list;
        if self.buffers.contains_id(buffer_id) {
            let buffer = unsafe { self.buffers.remove_and_return(buffer_id) };
            self.free_bytes += buffer.len() as Dim;
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
        let len = dst.len().min(buffer.len());
        dst[..len].copy_from_slice(&buffer[..len]);
        Ok(())
    }

    pub fn pool_to_pool(
        &mut self,
        src: Pool,
        src_buf: PoolBufferId,
        dst_buf: PoolBufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        match src {
            Pool::Host => {
                let len = self.buffers[src_buf].len().min(self.buffers[dst_buf].len());
                let src_bytes = self.buffers[src_buf][..len].to_vec();
                self.buffers[dst_buf][..len].copy_from_slice(&src_bytes);
                let _ = event_wait_list;
                Ok(Event::Host(HostEvent))
            }
            Pool::Disk => {
                let src_pool = super::disk::pool();
                let mut src_pool = super::lock(src, &src_pool);
                let mut byte_slice = vec![0u8; src_pool.buffer_bytes(src_buf) as usize];
                src_pool.pool_to_host(src_buf, &mut byte_slice, Vec::new())?;
                drop(src_pool);
                self.host_to_pool(&byte_slice, dst_buf, event_wait_list)
            }
            // CUDA -> host: download the device buffer into host bytes,
            // then memcpy into the host buffer.
            Pool::Cuda(id) => {
                let src_pool = super::cuda::pool(id)?;
                let mut src_pool = super::lock(src, &src_pool);
                let mut byte_slice = vec![0u8; self.buffers[dst_buf].len()];
                src_pool.pool_to_host(src_buf, &mut byte_slice, Vec::new())?;
                drop(src_pool);
                self.host_to_pool(&byte_slice, dst_buf, event_wait_list)
            }
            // TT -> host: read the device DRAM buffer into host bytes via the
            // runtime shim (read_buf), then memcpy into the host buffer.
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => {
                let src_pool = super::tenstorrent::pool(id)?;
                let mut src_pool = super::lock(src, &src_pool);
                let mut byte_slice = vec![0u8; src_pool.buffers[src_buf].size as usize];
                src_pool.pool_to_host(src_buf, &mut byte_slice, Vec::new())?;
                drop(src_pool);
                self.host_to_pool(&byte_slice, dst_buf, event_wait_list)
            }
            _ => todo!("host pool_to_pool from {src:?}"),
        }
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
        &self.buffers[id]
    }

    /// Get a mutable raw pointer to the buffer's data
    pub fn buffer_ptr_mut(&mut self, id: PoolBufferId) -> *mut u8 {
        self.buffers[id].as_mut_ptr()
    }
}
