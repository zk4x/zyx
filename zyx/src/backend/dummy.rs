use super::{BufferId, Device, DeviceInfo, Event, MemoryPool, ProgramId, opencl::OpenCLEvent};
use crate::{
    error::{BackendError, ErrorStatus},
    kernel::Kernel,
    runtime::Pool,
    shape::Dim,
    slab::{Slab, SlabId},
};
use nanoserde::DeJson;
use std::ptr;

#[derive(Default, Debug, DeJson)]
pub struct DummyConfig {
    enabled: bool,
}

#[derive(Debug)]
pub struct DummyMemoryPool {
    free_bytes: Dim,
    buffers: Slab<BufferId, Dim>,
}

#[derive(Debug)]
pub struct DummyDevice {
    device_info: DeviceInfo,
    memory_pool_id: u32,
}

pub(super) fn initialize_device(
    config: &DummyConfig,
    memory_pools: &mut Vec<Pool>,
    devices: &mut Vec<Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if !config.enabled {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: "Dummy backend configured out.".into(),
        });
    }
    if debug_dev {
        println!("Using dummy backend");
    }
    let pool = MemoryPool::Dummy(DummyMemoryPool { free_bytes: 1024 * 1024 * 1024 * 1024, buffers: Slab::new() });
    memory_pools.push(Pool::new(pool));
    devices.push(Device::Dummy(DummyDevice {
        device_info: DeviceInfo {
            compute: 20 * 1024 * 1024 * 1024 * 1024 * 1024,
            max_global_work_dims: vec![u32::MAX as Dim; 3],
            max_local_threads: 256 * 256,
            max_local_work_dims: vec![1, 256, 256],
            preferred_vector_size: 8,
            local_mem_size: 1024 * 1024 * 1024,
            num_registers: 128,
            tensor_cores: false,
        },
        memory_pool_id: (memory_pools.len() - 1) as u32,
    }));
    Ok(())
}

impl DummyMemoryPool {
    pub const fn deinitialize(&mut self) {
        let _ = self;
    }

    pub fn free_bytes(&self) -> Dim {
        //println!("Free bytes {} B", self.free_bytes);
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(BufferId, Event), BackendError> {
        if self.free_bytes > bytes {
            self.free_bytes -= bytes;
        } else {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "OOM".into() });
        }
        let id = self.buffers.push(bytes);
        Ok((id, Event::OpenCL(OpenCLEvent { event: ptr::null_mut() })))
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn deallocate(&mut self, buffer_id: BufferId, event_wait_list: Vec<Event>) {
        let _ = event_wait_list;
        let bytes = self.buffers[buffer_id];
        self.buffers.remove(buffer_id);
        self.free_bytes += bytes;
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unnecessary_wraps)]
    pub fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: BufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let _ = self;
        let _ = src;
        let _ = dst;
        let _ = event_wait_list;
        Ok(Event::OpenCL(OpenCLEvent { event: ptr::null_mut() }))
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unnecessary_wraps)]
    pub fn pool_to_host(
        &mut self,
        src: BufferId,
        dst: &mut [u8],
        event_wait_list: Vec<super::Event>,
    ) -> Result<(), BackendError> {
        let _ = self;
        let _ = src;
        let _ = dst;
        let _ = event_wait_list;
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unnecessary_wraps)]
    pub fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        let _ = self;
        let _ = events;
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn release_events(&mut self, events: Vec<Event>) {
        let _ = self;
        let _ = events;
    }
}

impl DummyDevice {
    pub const fn deinitialize(&mut self) {
        let _ = self;
    }

    pub const fn info(&self) -> &super::DeviceInfo {
        &self.device_info
    }

    pub const fn memory_pool_id(&self) -> u32 {
        self.memory_pool_id
    }

    pub const fn free_compute(&self) -> u128 {
        self.device_info.compute
    }

    #[allow(clippy::unnecessary_wraps)]
    pub const fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<ProgramId, BackendError> {
        let _ = self;
        let _ = kernel;
        let _ = debug_asm;
        Ok(ProgramId::ZERO)
    }

    pub const fn release(&mut self, program_id: ProgramId) {
        let _ = self;
        let _ = program_id;
    }

    #[allow(clippy::unnecessary_wraps)]
    #[allow(clippy::needless_pass_by_value)]
    pub fn launch(
        &mut self,
        program_id: ProgramId,
        memory_pool: &mut DummyMemoryPool,
        args: &[BufferId],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let _ = self;
        let _ = program_id;
        let _ = event_wait_list;
        for &arg in args {
            let _ = memory_pool.buffers[arg];
        }
        Ok(Event::OpenCL(OpenCLEvent { event: ptr::null_mut() }))
    }
}
