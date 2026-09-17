// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

use super::{
    DTypeCapability, DeviceInfo, DeviceProgramId, Event, LaunchArg, Pool, PoolBufferId, opencl::OpenCLEvent,
};
use crate::{
    DType,
    error::{BackendError, ErrorStatus},
    kernel::Kernel,
    shape::Dim,
    slab::{Slab, SlabId},
};
use nanoserde::DeJson;
use std::sync::{Mutex, OnceLock};
use std::{ptr, sync::Arc};

#[derive(Default, Debug, DeJson)]
#[nserde(default)]
pub struct DummyConfig {
    enabled: bool,
}

#[derive(Debug)]
pub struct DummyMemoryPool {
    free_bytes: Dim,
    buffers: Slab<PoolBufferId, Dim>,
}

#[derive(Debug)]
pub struct DummyDevice {
    device_info: Arc<DeviceInfo>,
    memory_pool: Pool,
}

/// Process-wide global dummy pool. Owned here — `mod.rs` only holds the
/// `Pool::Dummy` handle. `INIT` serializes first construction only; the
/// alloc/free path never takes it.
static DUMMY_POOL: OnceLock<Arc<Mutex<DummyMemoryPool>>> = OnceLock::new();
static DUMMY_INIT: Mutex<()> = Mutex::new(());

pub(super) fn pool() -> Result<Arc<Mutex<DummyMemoryPool>>, BackendError> {
    if let Some(pool) = DUMMY_POOL.get() {
        return Ok(pool.clone());
    }
    let _init = DUMMY_INIT.lock().unwrap_or_else(|_| panic!("dummy pool init lock poisoned"));
    if let Some(pool) = DUMMY_POOL.get() {
        return Ok(pool.clone());
    }
    let pool = Arc::new(Mutex::new(ensure_pool()?));
    DUMMY_POOL.set(pool.clone()).expect("dummy pool set twice under init lock");
    Ok(pool)
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
    Ok(DummyMemoryPool { free_bytes: 1024 * 1024 * 1024 * 1024, buffers: Slab::new() })
}

/// Process-wide global dummy device. Owned here — `mod.rs` only holds the
/// `Dev::Dummy` handle. Lazy like the dummy pool; fails when configured out.
static DUMMY_DEVICE: OnceLock<Arc<Mutex<DummyDevice>>> = OnceLock::new();
static DUMMY_DEV_INIT: Mutex<()> = Mutex::new(());

pub(super) fn device() -> Result<Arc<Mutex<DummyDevice>>, BackendError> {
    if let Some(dev) = DUMMY_DEVICE.get() {
        return Ok(dev.clone());
    }
    let _init = DUMMY_DEV_INIT.lock().unwrap_or_else(|_| panic!("dummy device init lock poisoned"));
    if let Some(dev) = DUMMY_DEVICE.get() {
        return Ok(dev.clone());
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
    let dev = Arc::new(Mutex::new(DummyDevice {
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
    }));
    DUMMY_DEVICE.set(dev.clone()).expect("dummy device set twice under init lock");
    Ok(dev)
}

impl DummyMemoryPool {
    pub const fn free_bytes(&self) -> Dim {
        //println!("Free bytes {} B", self.free_bytes);
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
        if self.free_bytes > bytes {
            self.free_bytes -= bytes;
        } else {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "OOM".into() });
        }
        let id = self.buffers.push(bytes);
        Ok((id, Event::OpenCL(OpenCLEvent { event: ptr::null_mut() })))
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn deallocate(&mut self, buffer_id: PoolBufferId, event_wait_list: Vec<Event>) {
        let _ = event_wait_list;
        let bytes = self.buffers[buffer_id];
        self.buffers.remove(buffer_id);
        self.free_bytes += bytes;
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn host_to_pool(&mut self, src: &[u8], dst: PoolBufferId, event_wait_list: Vec<Event>) -> Result<Event, BackendError> {
        let _ = self;
        let _ = src;
        let _ = dst;
        let _ = event_wait_list;
        Ok(Event::OpenCL(OpenCLEvent { event: ptr::null_mut() }))
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn pool_to_host(
        &mut self,
        src: PoolBufferId,
        dst: &mut [u8],
        event_wait_list: Vec<super::Event>,
    ) -> Result<(), BackendError> {
        let _ = self;
        let _ = src;
        let _ = dst;
        let _ = event_wait_list;
        Ok(())
    }

    pub fn pool_to_pool(
        &mut self,
        src: Pool,
        src_buf: PoolBufferId,
        dst_buf: PoolBufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let _ = (src, src_buf, dst_buf, event_wait_list);
        todo!("copies into dummy pool")
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
    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        pool_handle: Pool,
        args: &[LaunchArg],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        debug_assert_eq!(pool_handle, self.memory_pool);
        let _ = self;
        let _ = program_id;
        let _ = event_wait_list;
        let memory_pool = pool()?;
        let memory_pool = super::lock(pool_handle, &memory_pool);
        for arg in args {
            match arg {
                LaunchArg::Buffer(buffer_id) => {
                    let _ = memory_pool.buffers[*buffer_id];
                }
                LaunchArg::Variable(_) => {}
            }
        }
        Ok(Event::OpenCL(OpenCLEvent { event: ptr::null_mut() }))
    }
}
