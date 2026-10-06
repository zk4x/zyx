// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use super::{ChunkId, Placement, Pool};
use crate::{
    Set,
    error::{BackendError, ErrorStatus},
    shape::Dim,
    slab::Slab,
};
use std::sync::{Mutex, OnceLock};

// ── Global state ──────────────────────────────────────────────────────────────

/// Process-wide global host pool. Owned here — `mod.rs` only holds the `Pool::Host` handle.
static HOST_POOL: OnceLock<Mutex<HostMemoryPool>> = OnceLock::new();

#[derive(Debug)]
pub struct HostBuffer {
    data: Box<[u8]>,
}

#[derive(Debug)]
pub struct HostMemoryPool {
    free_bytes: Dim,
    buffers: Slab<ChunkId, HostBuffer>,
    free_set: Set<ChunkId>,
}

/// Constructs the global host pool. Infallible — `Host` is always available.
pub(super) fn ensure_pool() -> HostMemoryPool {
    let total_bytes = detect_host_memory_bytes();
    if super::debug_backends() {
        println!("[host] initialized");
        println!("[host] device total memory: {} MB", total_bytes / (1024 * 1024));
    }
    HostMemoryPool { free_bytes: total_bytes as i64, buffers: Slab::new(), free_set: Set::default() }
}

pub(super) fn pool() -> &'static Mutex<HostMemoryPool> {
    HOST_POOL.get_or_init(|| Mutex::new(ensure_pool()))
}

/// Copy executing a transfer into the host pool, matching on the source
/// pool: host memcpys directly, disk reads straight from the file
/// mapping, CUDA DMAs straight into the destination, every other
/// source is later work. The destination is always host-resident.
/// Single-shard placements only. Needs no device init — the host pool
/// is always available, which is what makes [`super::Dev::Host`] work
/// with every compute backend configured out.
pub(super) fn copy(src: &Placement, dst: &Placement, bytes: Dim) -> Result<(), BackendError> {
    debug_assert!(bytes >= 0, "host copy of negative bytes");
    let host = pool();
    let mut pool = super::lock(Pool::Host, host);
    let [src_shard] = &src.shards[..] else {
        todo!("host copy of multi-shard source placement")
    };
    let [dst_shard] = &dst.shards[..] else {
        todo!("host copy of multi-shard destination placement")
    };
    debug_assert_eq!(dst_shard.pool, Pool::Host, "host copy destination is not host-resident");
    match src_shard.pool {
        Pool::Host => {
            let src_ptr = pool.get_buffer(src_shard.chunk).as_ptr();
            let dst_ptr = pool.buffer_ptr_mut(dst_shard.chunk);
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, bytes as usize);
            }
        }
        Pool::Disk => {
            // Straight into the destination: the mapping is the source,
            // no staging buffer. The destination is over-allocated (one
            // extra element); the extent holds exact tensor bytes only.
            let disk = super::disk::pool();
            let mut dpool = super::lock(Pool::Disk, disk);
            let n = bytes.min(dpool.buffer_bytes(src_shard.chunk));
            let dst_ptr = pool.buffer_ptr_mut(dst_shard.chunk);
            dpool.pool_to_host(src_shard.chunk, unsafe { std::slice::from_raw_parts_mut(dst_ptr, n as usize) })?;
        }
        Pool::Cuda(id) => {
            // Device-to-host DMA straight into the destination, no
            // staging buffer: the worker drains the stream first, the
            // reply arrives after the data did.
            let cuda = super::cuda::pool(id)?;
            let mut cpool = super::lock(Pool::Cuda(id), cuda);
            let dst_ptr = pool.buffer_ptr_mut(dst_shard.chunk);
            cpool.pool_to_host(src_shard.chunk, unsafe { std::slice::from_raw_parts_mut(dst_ptr, bytes as usize) })?;
        }
        Pool::OpenCL(id) => {
            // Device-to-host read straight into the destination, no
            // staging buffer: the worker drains every queue first, the
            // reply arrives after the data did.
            let ocl = super::opencl::pool(id)?;
            let mut opool = super::lock(Pool::OpenCL(id), ocl);
            let dst_ptr = pool.buffer_ptr_mut(dst_shard.chunk);
            opool.pool_to_host(src_shard.chunk, unsafe { std::slice::from_raw_parts_mut(dst_ptr, bytes as usize) })?;
        }
        Pool::Vulkan(_) => {
            todo!("host copy from {:?}", src_shard.pool)
        }
        #[cfg(feature = "tenstorrent")]
        Pool::TT(id) => {
            // Device-to-host download straight into the destination, no
            // staging buffer: the shim IPC is synchronous, so the reply
            // arrives after the data did.
            let tt = super::tenstorrent::pool(id)?;
            let mut tpool = super::lock(Pool::TT(id), tt);
            let dst_ptr = pool.buffer_ptr_mut(dst_shard.chunk);
            tpool.pool_to_host(src_shard.chunk, unsafe { std::slice::from_raw_parts_mut(dst_ptr, bytes as usize) })?;
        }
        #[cfg(feature = "wgpu")]
        Pool::WGPU(_) => todo!("host copy from {:?}", src_shard.pool),
    }
    Ok(())
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

    pub fn allocate(&mut self, bytes: Dim) -> Result<ChunkId, BackendError> {
        // Best-fit from the free list first (stable addresses across
        // repeats — those bytes were already paid for); else fresh.
        let best = self
            .free_set
            .iter()
            .filter_map(|id| {
                let len = self.buffers[*id].data.len() as Dim;
                (len >= bytes).then_some((len, *id))
            })
            .min();
        if let Some((_, id)) = best {
            self.free_set.remove(&id);
            return Ok(id);
        }
        let bytes: usize = bytes
            .try_into()
            .map_err(|_| BackendError { status: ErrorStatus::MemoryAllocation, context: "allocation size too large".into() })?;
        if self.free_bytes < bytes as Dim {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "OOM".into() });
        }
        self.free_bytes -= bytes as Dim;
        let buffer = vec![0u8; bytes].into_boxed_slice();
        Ok(self.buffers.push(HostBuffer { data: buffer }))
    }

    /// Insert an already-filled buffer.
    pub fn insert(&mut self, buf: Box<[u8]>) -> ChunkId {
        self.free_bytes -= buf.len() as Dim;
        self.buffers.push(HostBuffer { data: buf })
    }

    /// Tries to reuse existing allocations, all or nothing, returns true if possible, otherwise returns false
    pub fn try_reuse_allocations(&mut self, buffer_ids: &Set<ChunkId>) -> bool {
        let ok = buffer_ids.iter().all(|id| self.free_set.contains(id));
        if ok {
            for id in buffer_ids {
                self.free_set.remove(id);
            }
        }
        ok
    }

    /// Put single buffer into the free list
    pub fn release(&mut self, buffer_id: ChunkId) {
        // Frees nothing: memory is reclaimed only by Dispose.
        if !self.buffers.contains_id(buffer_id) {
            debug_assert!(false, "release of unknown host buffer {buffer_id:?}");
            return;
        }
        debug_assert!(!self.free_set.contains(&buffer_id), "double release of host buffer {buffer_id:?}");
        self.free_set.insert(buffer_id);
    }

    /// Free all buffers in the free list at once
    pub fn dispose(&mut self) {
        // Synchronous pool, no in-flight work: drop every free buffer and
        // give the bytes back.
        for id in core::mem::take(&mut self.free_set) {
            if let Some(buffer) = self.buffers.get(id) {
                self.free_bytes += buffer.data.len() as Dim;
            }
            self.buffers.remove(id);
        }
    }

    pub fn pool_to_host(&mut self, src: ChunkId, dst: &mut [u8]) -> Result<(), BackendError> {
        let buffer = &self.buffers[src];
        let len = dst.len().min(buffer.data.len());
        dst[..len].copy_from_slice(&buffer.data[..len]);
        Ok(())
    }

    pub fn get_buffer(&self, id: ChunkId) -> &[u8] {
        &self.buffers[id].data
    }

    /// Get a mutable raw pointer to the buffer's data
    pub fn buffer_ptr_mut(&mut self, id: ChunkId) -> *mut u8 {
        self.buffers[id].data.as_mut_ptr()
    }
}
