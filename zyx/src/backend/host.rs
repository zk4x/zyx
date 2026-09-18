// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use super::{Pool, PoolBufferId};
use crate::{
    error::{BackendError, ErrorStatus},
    shape::Dim,
    slab::Slab,
};
use std::sync::{Arc, Mutex, OnceLock};

#[derive(Debug)]
pub struct HostBuffer {
    data: Box<[u8]>,
    rc: u16,
}

#[derive(Debug)]
pub struct HostMemoryPool {
    free_bytes: Dim,
    buffers: Slab<PoolBufferId, HostBuffer>,
}

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
    pub const fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<PoolBufferId, BackendError> {
        let bytes: usize = bytes
            .try_into()
            .map_err(|_| BackendError { status: ErrorStatus::MemoryAllocation, context: "allocation size too large".into() })?;
        if self.free_bytes < bytes as Dim {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "OOM".into() });
        }
        self.free_bytes -= bytes as Dim;
        let buffer = vec![0u8; bytes].into_boxed_slice();
        Ok(self.buffers.push(HostBuffer { data: buffer, rc: 1 }))
    }

    /// Insert an already-filled buffer, starting its reference count at 1.
    pub fn insert(&mut self, buf: Box<[u8]>) -> PoolBufferId {
        self.free_bytes -= buf.len() as Dim;
        self.buffers.push(HostBuffer { data: buf, rc: 1 })
    }

    /// Increment the reference count. Checked math: overflow panics.
    pub fn retain(&mut self, buffer_id: PoolBufferId) {
        match self.buffers.get_mut(buffer_id) {
            Some(buffer) => buffer.rc = buffer.rc.checked_add(1).expect("HostBuffer rc overflow"),
            None => debug_assert!(false, "retain of unknown host buffer {buffer_id:?}"),
        }
    }

    /// Decrement the reference count. At zero the buffer is freed immediately:
    /// the host pool is synchronous and holds no in-flight work — async
    /// consumers elsewhere retain the buffer while they still need it.
    pub fn release(&mut self, buffer_id: PoolBufferId) {
        let Some(buffer) = self.buffers.get_mut(buffer_id) else {
            debug_assert!(false, "release of unknown host buffer {buffer_id:?}");
            return;
        };
        buffer.rc = buffer.rc.checked_sub(1).expect("HostBuffer rc underflow");
        if buffer.rc == 0 {
            let buffer = unsafe { self.buffers.remove_and_return(buffer_id) };
            self.free_bytes += buffer.data.len() as Dim;
        }
    }

    /// Memcpy into a host-pool buffer. Synchronous — the host pool never
    /// defers work, so a plain borrow is safe.
    pub fn host_to_pool(&mut self, src: &[u8], dst: PoolBufferId) -> Result<(), BackendError> {
        let buffer = self
            .buffers
            .get_mut(dst)
            .ok_or_else(|| BackendError { status: ErrorStatus::MemoryCopyH2P, context: "invalid buffer id".into() })?;
        let len = src.len().min(buffer.data.len());
        buffer.data[..len].copy_from_slice(&src[..len]);
        Ok(())
    }

    pub fn pool_to_host(&mut self, src: PoolBufferId, dst: &mut [u8]) -> Result<(), BackendError> {
        let buffer = &self.buffers[src];
        let len = dst.len().min(buffer.data.len());
        dst[..len].copy_from_slice(&buffer.data[..len]);
        Ok(())
    }

    pub fn pool_to_pool(&mut self, src: Pool, src_buf: PoolBufferId, dst_buf: PoolBufferId) -> Result<(), BackendError> {
        match src {
            Pool::Host => {
                let len = self.buffers[src_buf].data.len().min(self.buffers[dst_buf].data.len());
                let src_bytes = self.buffers[src_buf].data[..len].to_vec();
                self.buffers[dst_buf].data[..len].copy_from_slice(&src_bytes);
                Ok(())
            }
            Pool::Disk => {
                let src_pool = super::disk::pool();
                let mut src_pool = super::lock(src, &src_pool);
                let mut byte_slice = vec![0u8; src_pool.buffer_bytes(src_buf) as usize];
                src_pool.pool_to_host(src_buf, &mut byte_slice)?;
                drop(src_pool);
                self.host_to_pool(&byte_slice, dst_buf)
            }
            // CUDA -> host: download the device buffer into host bytes,
            // then memcpy into the host buffer.
            Pool::Cuda(id) => {
                let src_pool = super::cuda::pool(id)?;
                let mut src_pool = super::lock(src, &src_pool);
                let mut byte_slice = vec![0u8; self.buffers[dst_buf].data.len()];
                src_pool.pool_to_host(src_buf, &mut byte_slice)?;
                drop(src_pool);
                self.host_to_pool(&byte_slice, dst_buf)
            }
            // TT -> host: read the device DRAM buffer into host bytes via the
            // runtime shim (read_buf), then memcpy into the host buffer.
            #[cfg(feature = "tenstorrent")]
            Pool::TT(id) => {
                let src_pool = super::tenstorrent::pool(id)?;
                let mut src_pool = super::lock(src, &src_pool);
                let mut byte_slice = vec![0u8; src_pool.buffers[src_buf].size as usize];
                src_pool.pool_to_host(src_buf, &mut byte_slice)?;
                drop(src_pool);
                self.host_to_pool(&byte_slice, dst_buf)
            }
            Pool::OpenCL(_) | Pool::Vulkan(_) | Pool::Dummy => todo!("host pool_to_pool from {src:?}"),
            #[cfg(feature = "wgpu")]
            Pool::WGPU(_) => todo!("host pool_to_pool from WGPU"),
        }
    }

    pub fn get_buffer(&self, id: PoolBufferId) -> &[u8] {
        &self.buffers[id].data
    }

    /// Get a mutable raw pointer to the buffer's data
    pub fn buffer_ptr_mut(&mut self, id: PoolBufferId) -> *mut u8 {
        self.buffers[id].data.as_mut_ptr()
    }
}
