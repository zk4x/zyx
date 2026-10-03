// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use super::{ChunkId, DTypeCapability, DeviceInfo, DeviceProgramId, LaunchArg, Pool};
use crate::{
    DType, Set,
    error::{BackendError, ErrorStatus},
    kernel::Kernel,
    shape::Dim,
    slab::{Slab, SlabId},
};
use nanoserde::DeJson;
use std::sync::{Arc, Mutex, OnceLock};

// ── Global state ──────────────────────────────────────────────────────────────

static DUMMY_POOL: OnceLock<Mutex<DummyMemoryPool>> = OnceLock::new();
static DUMMY_DEVICE: OnceLock<Mutex<DummyDevice>> = OnceLock::new();

#[derive(Default, Debug, DeJson)]
#[nserde(default)]
pub struct DummyConfig {
    enabled: bool,
}

#[derive(Debug)]
pub struct DummyBuffer {
    bytes: Dim,
}

#[derive(Debug)]
pub struct DummyMemoryPool {
    free_bytes: Dim,
    buffers: Slab<ChunkId, DummyBuffer>,
    free_set: Set<ChunkId>,
}

#[derive(Debug)]
pub struct DummyDevice {
    device_info: Arc<DeviceInfo>,
    memory_pool: Pool,
}

pub(super) fn pool() -> Result<&'static Mutex<DummyMemoryPool>, BackendError> {
    if let Some(pool) = DUMMY_POOL.get() {
        return Ok(pool);
    }
    let pool = Mutex::new(ensure_pool()?);
    let _ = DUMMY_POOL.set(pool);
    DUMMY_POOL
        .get()
        .ok_or_else(|| BackendError { status: ErrorStatus::Initialization, context: "dummy pool init failed".into() })
}

/// Constructs the global dummy pool. Fails when dummy is configured out.
fn ensure_pool() -> Result<DummyMemoryPool, BackendError> {
    let config = super::config();
    if !config.dummy.enabled {
        if super::debug_backends() {
            println!("[dummy] configured out");
        }
        return Err(BackendError { status: ErrorStatus::Initialization, context: "[dummy] configured out".into() });
    }
    if super::debug_backends() {
        println!("[dummy] initialized");
        println!("[dummy] device total memory: {} MB", 1024 * 1024);
    }
    Ok(DummyMemoryPool { free_bytes: 1024 * 1024 * 1024 * 1024, buffers: Slab::new(), free_set: Set::default() })
}

pub(super) fn device() -> Result<&'static Mutex<DummyDevice>, BackendError> {
    if let Some(dev) = DUMMY_DEVICE.get() {
        return Ok(dev);
    }
    let config = super::config();
    if !config.dummy.enabled {
        if super::debug_backends() {
            println!("[dummy] configured out");
        }
        return Err(BackendError { status: ErrorStatus::Initialization, context: "[dummy] configured out".into() });
    }
    if super::debug_backends() {
        println!("[dummy] initialized");
    }
    let dev = Mutex::new(DummyDevice {
        device_info: Arc::new(DeviceInfo {
            compute: 20 * 1024 * 1024 * 1024 * 1024 * 1024,
            max_global_work_dims: vec![Dim::from(u32::MAX); 3],
            max_local_threads: 256 * 256,
            max_local_work_dims: vec![1, 256, 256],
            preferred_vector_size: 8,
            local_mem_size: 1024 * 1024 * 1024,
            max_register_bytes: 128,
            tensor_cores: true,
            warp_size: 32,
            cc: [0, 0],
            dtype_capability: [DTypeCapability::all(); DType::N_DTYPES],
            has_native_exp2: true,
            supported_vec_lens: vec![2, 4, 8, 16],
            tenstorrent: false,
            tile: [1, 1],
            tile_sizes: vec![],
            wmma_layouts: vec![],
            num_circular_buffers: 0,
            has_openmp: false,
        }),
        memory_pool: Pool::Dummy,
    });
    let _ = DUMMY_DEVICE.set(dev);
    Ok(DUMMY_DEVICE.get().unwrap())
}

impl DummyMemoryPool {
    pub const fn free_bytes(&self) -> Dim {
        //println!("Free bytes {} B", self.free_bytes);
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<ChunkId, BackendError> {
        if self.free_bytes > bytes {
            self.free_bytes -= bytes;
        } else {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "OOM".into() });
        }
        Ok(self.buffers.push(DummyBuffer { bytes }))
    }

    /// Put a buffer into the free list for stable-address reuse. Frees
    /// nothing; memory is reclaimed only by [`DummyMemoryPool::dispose`].
    pub fn release(&mut self, buffer_id: ChunkId) {
        if !self.buffers.contains_id(buffer_id) {
            debug_assert!(false, "release of unknown dummy buffer {buffer_id:?}");
            return;
        }
        debug_assert!(!self.free_set.contains(&buffer_id), "double release of dummy buffer {buffer_id:?}");
        self.free_set.insert(buffer_id);
    }

    /// Tries to reuse existing allocations, all or nothing: every id must
    /// be on the free list (stable addresses) or nothing is claimed.
    pub fn try_reuse_allocations(&mut self, buffer_ids: &Set<ChunkId>) -> bool {
        let ok = buffer_ids.iter().all(|id| self.free_set.contains(id));
        if ok {
            for id in buffer_ids {
                self.free_set.remove(id);
            }
        }
        ok
    }

    /// Scratch for queue intermediaries: reuses the smallest fitting free
    /// buffer, else allocates fresh. Never fails on fragmentation.
    pub fn allocate_scratch(&mut self, bytes: Dim) -> Result<ChunkId, BackendError> {
        let best = self
            .free_set
            .iter()
            .filter_map(|id| {
                let len = self.buffers[*id].bytes;
                (len >= bytes).then_some((len, *id))
            })
            .min();
        if let Some((_, id)) = best {
            self.free_set.remove(&id);
            return Ok(id);
        }
        self.allocate(bytes)
    }

    /// Free all buffers in the free list at once.
    pub fn dispose(&mut self) {
        // Synchronous pool, no in-flight work: drop every free buffer and
        // give the bytes back.
        for id in core::mem::take(&mut self.free_set) {
            if let Some(buffer) = self.buffers.get(id) {
                self.free_bytes += buffer.bytes;
            }
            self.buffers.remove(id);
        }
    }

    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn pool_to_host(&mut self, src: ChunkId, dst: &mut [u8]) -> Result<(), BackendError> {
        // The dummy pool holds no data — nothing to read back.
        let _ = (self, src, dst);
        Ok(())
    }
}

impl DummyDevice {
    pub fn info(&self) -> Arc<DeviceInfo> {
        self.device_info.clone()
    }

    pub fn free_compute(&self) -> u128 {
        self.device_info.compute
    }

    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub const fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        let _ = self;
        let _ = kernel;
        let _ = debug_asm;
        Ok(DeviceProgramId::ZERO)
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub const fn release(&mut self, program_id: DeviceProgramId) {
        let _ = self;
        let _ = program_id;
    }

    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn launch(&mut self, program_id: DeviceProgramId, pool_handle: Pool, args: &[LaunchArg]) -> Result<(), BackendError> {
        debug_assert_eq!(pool_handle, self.memory_pool);
        let _ = program_id;
        let memory_pool = pool()?;
        let memory_pool = super::lock(pool_handle, memory_pool);
        for arg in args {
            match arg {
                LaunchArg::Buffer(_) => {
                    todo!("placement check in dummy launch")
                }
                LaunchArg::Variable(_) => {}
            }
        }
        Ok(())
    }
}
