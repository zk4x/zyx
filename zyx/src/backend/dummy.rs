use std::ptr;

use nanoserde::DeJson;

use crate::{
    runtime::Pool, shape::Dimension, slab::{Id, Slab}
};

use super::{
    opencl::OpenCLEvent, BackendError, BufferMut, Device, DeviceInfo, ErrorStatus, Event,
    MemoryPool,
};

#[derive(Default, Debug, DeJson)]
pub struct DummyConfig {
    enabled: bool,
}

struct DummyMemoryPool {
    free_bytes: Dimension,
    buffers: Slab<Dimension>,
}

struct DummyDevice {
    device_info: DeviceInfo,
}

pub(super) fn initialize_device(
    config: &DummyConfig,
    memory_pools: &mut Vec<Pool>,
    devices: &mut Vec<Box<dyn Device>>,
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
    let pool = Box::new(DummyMemoryPool {
        free_bytes: 1024 * 1024 * 1024 * 1024 * 1024,
        buffers: Slab::new(),
    });
    memory_pools.push(Pool::new(pool));
    devices.push(Box::new(DummyDevice {
        device_info: DeviceInfo {
            compute: 20 * 1024 * 1024 * 1024 * 1024 * 1024,
            max_global_work_dims: [u32::MAX as Dimension, u32::MAX as Dimension, u32::MAX as Dimension],
            max_local_threads: 256 * 256,
            max_local_work_dims: [1, 256, 256],
            preferred_vector_size: 8,
            local_mem_size: 1024 * 1024 * 1024,
            num_registers: 128,
            tensor_cores: false,
        },
    }));
    Ok(())
}

impl MemoryPool for DummyMemoryPool {
    fn deinitialize(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    fn free_bytes(&self) -> Dimension {
        self.free_bytes
    }

    fn get_buffer(&self, buffer: crate::slab::Id) -> BufferMut {
        #[cfg(debug_assertions)]
        self.buffers[buffer];
        BufferMut::Dummy(buffer)
    }

    fn allocate(&mut self, bytes: Dimension) -> Result<(Id, Event), BackendError> {
        if self.free_bytes > bytes {
            self.free_bytes -= bytes;
        } else {
            return Err(BackendError {
                status: ErrorStatus::MemoryAllocation,
                context: "OOM".into(),
            });
        }
        let id = self.buffers.push(bytes);
        Ok((id, Event::OpenCL(OpenCLEvent { event: ptr::null_mut() })))
    }

    fn deallocate(
        &mut self,
        buffer_id: crate::slab::Id,
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        let _ = event_wait_list;
        let bytes = self.buffers[buffer_id];
        self.buffers.remove(buffer_id);
        self.free_bytes += bytes;
        Ok(())
    }

    fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: crate::slab::Id,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let _ = src;
        let _ = dst;
        let _ = event_wait_list;
        Ok(Event::OpenCL(OpenCLEvent { event: ptr::null_mut() }))
    }

    fn pool_to_host(
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

    fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        let _ = events;
        Ok(())
    }

    fn release_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        let _ = events;
        Ok(())
    }
}

impl Device for DummyDevice {
    fn deinitialize(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    fn info(&self) -> &super::DeviceInfo {
        &self.device_info
    }

    fn memory_pool_id(&self) -> u32 {
        0
    }

    fn compute(&self) -> u128 {
        self.device_info.compute
    }

    fn compile(
        &mut self,
        kernel: &crate::ir::IRKernel,
        debug_asm: bool,
    ) -> Result<Id, BackendError> {
        let _ = kernel;
        let _ = debug_asm;
        Ok(0)
    }

    fn release(&mut self, program_id: Id) -> Result<(), BackendError> {
        let _ = program_id;
        Ok(())
    }

    fn launch(
        &mut self,
        program_id: Id,
        memory_pool: &mut dyn MemoryPool,
        args: &[Id],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let _ = program_id;
        let _ = event_wait_list;
        for &arg in args {
            let BufferMut::Dummy(_) = memory_pool.get_buffer(arg) else { panic!() };
        }
        Ok(Event::OpenCL(OpenCLEvent { event: ptr::null_mut() }))
    }
}
