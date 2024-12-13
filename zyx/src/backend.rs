//! This file creates backend agnostic API to backends
//! That is it contains enums that dispatch function calls to appropriate backends.
//! Backend automatically keeps track of hardware queues and events
//! and synchronizes events as needed at latest possible moment
//! for maximum concurrency.

// Because I don't want to write struct and inner enum for MemoryPool and Device
#![allow(private_interfaces)]

use std::collections::BTreeMap;

use super::{ir::IRKernel, DeviceConfig, ZyxError};
use crate::{
    kernel::Op,
    slab::{Id, Slab},
    Scalar,
};
use cuda::{CUDABuffer, CUDADevice, CUDAEvent, CUDAMemoryPool, CUDAProgram, CUDAQueue};
use hip::{HIPBuffer, HIPDevice, HIPMemoryPool, HIPProgram, HIPQueue};
use opencl::{
    OpenCLBuffer, OpenCLDevice, OpenCLEvent, OpenCLMemoryPool, OpenCLProgram, OpenCLQueue,
};

#[cfg(feature = "wgsl")]
use wgsl::{WGSLBuffer, WGSLDevice, WGSLMemoryPool, WGSLProgram, WGSLQueue};

mod cuda;
mod hip;
mod opencl;
#[cfg(feature = "vulkan")]
mod vulkan;
#[cfg(feature = "wgsl")]
mod wgsl;

// Export configs and errors, nothing more
pub use cuda::{CUDAConfig, CUDAError};
pub use hip::{HIPConfig, HIPError};
pub use opencl::{OpenCLConfig, OpenCLError};
#[cfg(feature = "vulkan")]
pub use vulkan::{VulkanConfig, VulkanError};
#[cfg(feature = "wgsl")]
pub use wgsl::{WGSLConfig, WGSLError};

/// Hardware information needed for applying optimizations
#[cfg_attr(feature = "disk_cache", derive(bitcode::Encode, bitcode::Decode))]
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeviceInfo {
    /// Device compute in flops
    pub compute: u128,
    /// Biggest kernel dimensions
    pub max_global_work_dims: [usize; 3],
    /// Maximum local work size threads
    pub max_local_threads: usize,
    pub max_local_work_dims: [usize; 3],
    /// Preferred vector size in bytes
    pub preferred_vector_size: usize,
    /// Local memory size in bytes
    pub local_mem_size: usize,
    /// Number of registers per thread
    pub num_registers: usize,
    /// Does this hardware have tensor cores?
    pub tensor_cores: bool,
}

pub type MemoryPoolId = u32;

/*trait HMemoryPool {
    type Error;
    type Buffer;
    fn deinitialize(self) -> Result<(), Self::Error>;
    fn free_bytes(&self) -> usize;
    fn allocate(&mut self, bytes: usize) -> Result<Self::Buffer, Self::Error>;
    fn host_to_pool(&mut self, src: &[u8], dst: &Self::Buffer) -> Result<(), Self::Error>;
    fn pool_to_host(&mut self, src: &Self::Buffer, dst: &mut [u8]) -> Result<(), Self::Error>;
}

trait HDevice {
    type Error;
    type Program;
    type Queue: HQueue<Program = Self::Program>;
    fn deinitialize(self) -> Result<(), Self::Error>;
    fn info(&self) -> &DeviceInfo;
    fn memory_pool_id(&self) -> usize;
    fn release_program(&self, program: Self::Program) -> Result<(), Self::Error>;
    fn release_queue(&self, queue: Self::Queue) -> Result<(), Self::Error>;
    fn compile(&mut self, kernel: &IRKernel, debug_asm: bool)
        -> Result<Self::Program, Self::Error>;
}

trait HQueue {
    type Buffer;
    type Program;
    type Error;
    fn launch(
        &mut self,
        program: &mut Self::Program,
        buffers: &mut Slab<Self::Buffer>,
        args: &[Id],
    ) -> Result<(), Self::Error>;
    fn sync(&mut self) -> Result<(), Self::Error>;
    fn load(&self) -> usize;
}*/

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub enum MemoryPool {
    CUDA {
        memory_pool: CUDAMemoryPool,
        buffers: Slab<CUDABuffer>,
        events: BTreeMap<Id, CUDAEvent>,
    },
    HIP {
        memory_pool: HIPMemoryPool,
        buffers: Slab<HIPBuffer>,
    },
    OpenCL {
        memory_pool: OpenCLMemoryPool,
        buffers: Slab<OpenCLBuffer>,
        // Buffers can have associated events that must be finished
        // before accessing given buffer
        events: BTreeMap<Id, OpenCLEvent>,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        memory_pool: VulkanMemoryPool,
        buffers: Slab<VulkanBuffer>,
    },
    #[cfg(feature = "wgsl")]
    WGSL {
        memory_pool: WGSLMemoryPool,
        buffers: Slab<WGSLBuffer>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId {
    pub(super) memory_pool_id: u32,
    pub(super) buffer_id: Id,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub enum Device {
    CUDA {
        memory_pool_id: MemoryPoolId,
        device: CUDADevice,
        queues: Vec<CUDAQueue>,
        kernels: BTreeMap<Vec<Op>, CUDAProgram>,
    },
    HIP {
        memory_pool_id: MemoryPoolId,
        device: HIPDevice,
        queues: Vec<HIPQueue>,
        kernels: BTreeMap<Vec<Op>, HIPProgram>,
    },
    OpenCL {
        memory_pool_id: MemoryPoolId,
        device: OpenCLDevice,
        queues: Vec<OpenCLQueue>,
        kernels: BTreeMap<Vec<Op>, OpenCLProgram>,
    },
    #[cfg(feature = "vulkan")]
    Vulkan {
        memory_pool_id: MemoryPoolId,
        device: VulkanDevice,
        queues: Vec<VulkanQueue>,
        kernels: BTreeMap<Vec<Op>, VulkanProgram>,
    },
    #[cfg(feature = "wgsl")]
    WGSL {
        memory_pool_id: MemoryPoolId,
        device: WGSLDevice,
        queues: Vec<WGSLQueue>,
        kernels: BTreeMap<Vec<Op>, WGSLProgram>,
    },
}

pub enum Event {
    CUDA(CUDAEvent),
    HIP,
    OpenCL(OpenCLEvent),
}

impl Event {
    pub(super) fn finish(self) -> Result<(), ZyxError> {
        match self {
            Event::CUDA(event) => event.finish()?,
            Event::HIP => todo!(),
            Event::OpenCL(event) => event.finish()?,
        }
        Ok(())
    }
}

pub fn initialize_backends(
    device_config: &DeviceConfig,
    memory_pools: &mut Vec<MemoryPool>,
    devices: &mut Vec<Device>,
    debug_dev: bool,
) -> Result<(), ZyxError> {
    if let Ok((mem_pools, devs)) = cuda::initialize_devices(&device_config.cuda, debug_dev) {
        let n = u32::try_from(memory_pools.len()).unwrap();
        memory_pools.extend(mem_pools.into_iter().map(|m| MemoryPool::CUDA {
            memory_pool: m,
            buffers: Slab::new(),
            events: BTreeMap::new(),
        }));
        devices.extend(devs.into_iter().map(|(device, queues)| Device::CUDA {
            memory_pool_id: device.memory_pool_id() + n,
            device,
            queues,
            kernels: BTreeMap::new(),
        }));
    }
    if let Ok((mem_pools, devs)) = hip::initialize_device(&device_config.hip, debug_dev) {
        let n = u32::try_from(memory_pools.len()).unwrap();
        memory_pools.extend(
            mem_pools.into_iter().map(|m| MemoryPool::HIP { memory_pool: m, buffers: Slab::new() }),
        );
        devices.extend(devs.into_iter().map(|(device, queues)| Device::HIP {
            memory_pool_id: device.memory_pool_id() + n,
            device,
            queues,
            kernels: BTreeMap::new(),
        }));
    }
    if let Ok((mem_pools, devs)) = opencl::initialize_devices(&device_config.opencl, debug_dev) {
        let n = u32::try_from(memory_pools.len()).unwrap();
        memory_pools.extend(mem_pools.into_iter().map(|m| MemoryPool::OpenCL {
            memory_pool: m,
            buffers: Slab::new(),
            events: BTreeMap::new(),
        }));
        devices.extend(devs.into_iter().map(|(device, queues)| Device::OpenCL {
            memory_pool_id: device.memory_pool_id() + n,
            device,
            queues,
            kernels: BTreeMap::new(),
        }));
    }
    #[cfg(feature = "vulkan")]
    if let Ok((mem_pools, devs)) = vulkan::initialize_devices(&device_config.vulkan, debug_dev) {
        let n = u32::try_from(memory_pools.len()).unwrap();
        memory_pools.extend(
            mem_pools
                .into_iter()
                .map(|m| MemoryPool::Vulkan { memory_pool: m, buffers: Slab::new() }),
        );
        devices.extend(devs.into_iter().map(|(device, queues)| Device::Vulkan {
            memory_pool_id: device.memory_pool_id() + n,
            device,
            programs: Slab::new(),
            queues,
        }));
    }
    #[cfg(feature = "wgsl")]
    if let Ok((mem_pools, devs)) = wgsl::initialize_backend(&device_config.wgsl, debug_dev) {
        let n = u32::try_from(memory_pools.len()).unwrap();
        memory_pools.extend(
            mem_pools
                .into_iter()
                .map(|m| MemoryPool::WGSL { memory_pool: m, buffers: Slab::new() }),
        );
        devices.extend(devs.into_iter().map(|(device, queues)| Device::WGSL {
            memory_pool_id: device.memory_pool_id() + n,
            device,
            programs: Slab::new(),
            queues,
        }));
    }
    if devices.is_empty() {
        return Err(ZyxError::NoBackendAvailable);
    }
    Ok(())
}

impl MemoryPool {
    pub(super) fn deinitialize(self) -> Result<(), ZyxError> {
        match self {
            MemoryPool::CUDA { mut memory_pool, mut buffers, .. } => {
                let ids: Vec<Id> = buffers.ids().collect();
                for id in ids {
                    let buffer = buffers.remove(id).unwrap();
                    memory_pool.deallocate(buffer)?;
                }
                memory_pool.deinitialize()?;
            }
            MemoryPool::HIP { mut memory_pool, mut buffers } => {
                let ids: Vec<Id> = buffers.ids().collect();
                for id in ids {
                    let buffer = buffers.remove(id).unwrap();
                    memory_pool.deallocate(buffer)?;
                }
                memory_pool.deinitialize()?;
            }
            MemoryPool::OpenCL { mut memory_pool, mut buffers, .. } => {
                let ids: Vec<Id> = buffers.ids().collect();
                for id in ids {
                    let buffer = buffers.remove(id).unwrap();
                    memory_pool.deallocate(buffer)?;
                }
                memory_pool.deinitialize()?;
            }
            #[cfg(feature = "vulkan")]
            MemoryPool::Vulkan { mut memory_pool, mut buffers } => {
                let ids: Vec<Id> = buffers.ids().collect();
                for id in ids {
                    let buffer = buffers.remove(id).unwrap();
                    memory_pool.deallocate(buffer)?;
                }
                memory_pool.deinitialize()?;
            }
            #[cfg(feature = "wgsl")]
            MemoryPool::WGSL { mut memory_pool, mut buffers } => {
                let ids: Vec<Id> = buffers.ids().collect();
                for id in ids {
                    let buffer = buffers.remove(id).unwrap();
                    memory_pool.deallocate(buffer)?;
                }
                memory_pool.deinitialize()?;
            }
        }
        Ok(())
    }

    pub(super) const fn free_bytes(&self) -> usize {
        match self {
            MemoryPool::CUDA { memory_pool, .. } => memory_pool.free_bytes(),
            MemoryPool::HIP { memory_pool, .. } => memory_pool.free_bytes(),
            MemoryPool::OpenCL { memory_pool, .. } => memory_pool.free_bytes(),
            #[cfg(feature = "vulkan")]
            MemoryPool::Vulkan { memory_pool, .. } => memory_pool.free_bytes(),
            #[cfg(feature = "wgsl")]
            MemoryPool::WGSL { memory_pool, .. } => memory_pool.free_bytes(),
        }
    }

    // Allocates bytes on memory pool and returns buffer id
    pub(super) fn allocate(&mut self, bytes: usize) -> Result<Id, ZyxError> {
        let id = match self {
            MemoryPool::CUDA { memory_pool, buffers, .. } => {
                buffers.push(memory_pool.allocate(bytes)?)
            }
            MemoryPool::HIP { memory_pool, buffers } => buffers.push(memory_pool.allocate(bytes)?),
            MemoryPool::OpenCL { memory_pool, buffers, .. } => {
                buffers.push(memory_pool.allocate(bytes)?)
            }
            #[cfg(feature = "vulkan")]
            MemoryPool::Vulkan { memory_pool, buffers } => {
                buffers.push(memory_pool.allocate(bytes)?)
            }
            #[cfg(feature = "wgsl")]
            MemoryPool::WGSL { memory_pool, buffers } => buffers.push(memory_pool.allocate(bytes)?),
        };
        //println!("Allocate {bytes} bytes into buffer id {id}");
        Ok(id)
    }

    pub(super) fn deallocate(&mut self, buffer_id: Id) -> Result<(), ZyxError> {
        //println!("Deallocate buffer id {buffer_id}");
        match self {
            MemoryPool::CUDA { memory_pool, buffers, .. } => {
                let buffer = buffers.remove(buffer_id).unwrap();
                memory_pool.deallocate(buffer)?;
            }
            MemoryPool::HIP { memory_pool, buffers } => {
                let buffer = buffers.remove(buffer_id).unwrap();
                memory_pool.deallocate(buffer)?;
            }
            MemoryPool::OpenCL { memory_pool, buffers, .. } => {
                let buffer = buffers.remove(buffer_id).unwrap();
                memory_pool.deallocate(buffer)?;
            }
            #[cfg(feature = "vulkan")]
            MemoryPool::Vulkan { memory_pool, buffers } => {
                let buffer = buffers.remove(buffer_id).unwrap();
                memory_pool.deallocate(buffer)?;
            }
            #[cfg(feature = "wgsl")]
            MemoryPool::WGSL { memory_pool, buffers } => {
                let buffer = buffers.remove(buffer_id).unwrap();
                memory_pool.deallocate(buffer)?;
            }
        }
        Ok(())
    }

    pub(super) fn host_to_pool<T: Scalar>(
        &mut self,
        data: &[T],
        buffer_id: Id,
    ) -> Result<(), ZyxError> {
        let bytes = data.len() * T::byte_size();
        match self {
            MemoryPool::CUDA { memory_pool, buffers, .. } => {
                let ptr: *const u8 = data.as_ptr().cast();
                memory_pool.host_to_pool(
                    unsafe { std::slice::from_raw_parts(ptr, bytes) },
                    &buffers[buffer_id],
                )?;
            }
            MemoryPool::HIP { memory_pool, buffers } => {
                let ptr: *const u8 = data.as_ptr().cast();
                memory_pool.host_to_pool(
                    unsafe { std::slice::from_raw_parts(ptr, bytes) },
                    &buffers[buffer_id],
                )?;
            }
            MemoryPool::OpenCL { memory_pool, buffers, .. } => {
                let ptr: *const u8 = data.as_ptr().cast();
                memory_pool.host_to_pool(
                    unsafe { std::slice::from_raw_parts(ptr, bytes) },
                    &buffers[buffer_id],
                )?;
            }
            #[cfg(feature = "vulkan")]
            MemoryPool::Vulkan { memory_pool, buffers } => {
                let ptr: *const u8 = data.as_ptr().cast();
                memory_pool.host_to_pool(
                    unsafe { std::slice::from_raw_parts(ptr, bytes) },
                    &buffers[buffer_id],
                )?;
            }
            #[cfg(feature = "wgsl")]
            MemoryPool::WGSL { memory_pool, buffers } => {
                let ptr: *const u8 = data.as_ptr().cast();
                memory_pool.host_to_pool(
                    unsafe { std::slice::from_raw_parts(ptr, bytes) },
                    &buffers[buffer_id],
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn pool_to_host<T: Scalar>(
        &mut self,
        buffer_id: Id,
        data: &mut [T],
    ) -> Result<(), ZyxError> {
        let slice = unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), data.len() * T::byte_size())
        };
        match self {
            MemoryPool::CUDA { memory_pool, buffers, events } => {
                if let Some(event) = events.remove(&buffer_id) {
                    event.finish()?;
                }
                memory_pool.pool_to_host(&buffers[buffer_id], slice)?;
            }
            MemoryPool::HIP { memory_pool, buffers } => {
                memory_pool.pool_to_host(&buffers[buffer_id], slice)?;
            }
            MemoryPool::OpenCL { memory_pool, buffers, events } => {
                if let Some(event) = events.remove(&buffer_id) {
                    event.finish()?;
                }
                memory_pool.pool_to_host(&buffers[buffer_id], slice)?;
            }
            #[cfg(feature = "vulkan")]
            MemoryPool::Vulkan { memory_pool, buffers } => {
                memory_pool.pool_to_host(&buffers[buffer_id], slice)?;
            }
            #[cfg(feature = "wgsl")]
            MemoryPool::WGSL { memory_pool, buffers } => {
                memory_pool.pool_to_host(&buffers[buffer_id], slice)?;
            }
        }
        Ok(())
    }

    #[rustfmt::skip]
    pub(super) fn pool_to_pool(&mut self, sbid: Id, dst_mp: &mut MemoryPool, dbid: Id, bytes: usize) -> Result<(), ZyxError> {
        macro_rules! cross_backend {
            ($sm: expr, $sb: expr, $dm: expr, $db: expr) => {{
                let mut data = vec![0; bytes];
                $sm.pool_to_host(&$sb[sbid], &mut data)?;
                $dm.host_to_pool(&data, &$db[dbid])?;
            }};
        }
        // Finish necessary events
        match self {
            MemoryPool::CUDA { .. } => todo!(),
            MemoryPool::HIP { .. } => todo!(),
            MemoryPool::OpenCL { events, .. } => {
                if let Some(event) = events.remove(&sbid) {
                    event.finish()?;
                }
            }
        }
        match (self, dst_mp) {
            #[rustfmt::skip]
            (MemoryPool::CUDA { buffers: sb, .. }, MemoryPool::CUDA { memory_pool: dm, buffers: db, .. }) => { dm.pool_to_pool(&sb[sbid], &db[dbid])?; }
            #[rustfmt::skip]
            (MemoryPool::CUDA { memory_pool: sm, buffers: sb, .. }, MemoryPool::HIP { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[rustfmt::skip]
            (MemoryPool::CUDA { memory_pool: sm, buffers: sb, .. }, MemoryPool::OpenCL { memory_pool: dm, buffers: db, .. }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "vulkan")]
            #[rustfmt::skip]
            (MemoryPool::CUDA { memory_pool: sm, buffers: sb }, MemoryPool::Vulkan { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::CUDA { memory_pool: sm, buffers: sb }, MemoryPool::WGSL { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[rustfmt::skip]
            (MemoryPool::HIP { memory_pool: sm, buffers: sb }, MemoryPool::CUDA { memory_pool: dm, buffers: db, .. }) => { cross_backend!(sm, sb, dm, db) }
            #[rustfmt::skip]
            (MemoryPool::HIP { buffers: sb, .. }, MemoryPool::HIP { memory_pool: dm, buffers: db }) => { dm.pool_to_pool(&sb[sbid], &db[dbid])?; }
            #[rustfmt::skip]
            (MemoryPool::HIP { memory_pool: sm, buffers: sb }, MemoryPool::OpenCL { memory_pool: dm, buffers: db, .. }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "vulkan")]
            #[rustfmt::skip]
            (MemoryPool::HIP { memory_pool: sm, buffers: sb }, MemoryPool::Vulkan { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::HIP { memory_pool: sm, buffers: sb }, MemoryPool::WGSL { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[rustfmt::skip]
            (MemoryPool::OpenCL { memory_pool: sm, buffers: sb, .. }, MemoryPool::CUDA { memory_pool: dm, buffers: db, .. }) => { cross_backend!(sm, sb, dm, db) }
            #[rustfmt::skip]
            (MemoryPool::OpenCL { memory_pool: sm, buffers: sb, .. }, MemoryPool::HIP { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[rustfmt::skip]
            (MemoryPool::OpenCL { buffers: sb, .. }, MemoryPool::OpenCL { memory_pool: dm, buffers: db, .. }) => { dm.pool_to_pool(&sb[sbid], &db[dbid])?; }
            #[cfg(feature = "vulkan")]
            #[rustfmt::skip]
            (MemoryPool::OpenCL { memory_pool: sm, buffers: sb }, MemoryPool::Vulkan { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::OpenCL { memory_pool: sm, buffers: sb }, MemoryPool::WGSL { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "vulkan")]
            #[rustfmt::skip]
            (MemoryPool::Vulkan { memory_pool: sm, buffers: sb }, MemoryPool::CUDA { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "vulkan")]
            #[rustfmt::skip]
            (MemoryPool::Vulkan { memory_pool: sm, buffers: sb }, MemoryPool::HIP { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "vulkan")]
            #[rustfmt::skip]
            (MemoryPool::Vulkan { memory_pool: sm, buffers: sb }, MemoryPool::OpenCL { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "vulkan")]
            #[rustfmt::skip]
            (MemoryPool::Vulkan { buffers: sb, .. }, MemoryPool::Vulkan { memory_pool: dm, buffers: db }) => { dm.pool_to_pool(&sb[sbid], &db[dbid])?; }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::Vulkan { memory_pool: sm, buffers: sb }, MemoryPool::WGSL { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::WGSL { memory_pool: sm, buffers: sb }, MemoryPool::CUDA { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::WGSL { memory_pool: sm, buffers: sb }, MemoryPool::HIP { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::WGSL { memory_pool: sm, buffers: sb }, MemoryPool::OpenCL { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::WGSL { buffers: sb, .. }, MemoryPool::WGSL { memory_pool: dm, buffers: db }) => { dm.pool_to_pool(&sb[sbid], &db[dbid])?; }
            #[cfg(feature = "wgsl")]
            #[rustfmt::skip]
            (MemoryPool::WGSL { memory_pool: sm, buffers: sb }, MemoryPool::Vulkan { memory_pool: dm, buffers: db }) => { cross_backend!(sm, sb, dm, db) }
        }
        Ok(())
    }
}

impl Device {
    pub(super) fn deinitialize(self) -> Result<(), ZyxError> {
        match self {
            Device::CUDA { device, mut queues, mut kernels, .. } => {
                while let Some((_, program)) = kernels.pop_last() {
                    device.release_program(program)?;
                }
                while let Some(queue) = queues.pop() {
                    device.release_queue(queue)?;
                }
                device.deinitialize()?;
            }
            Device::HIP { device, mut queues, mut kernels, .. } => {
                while let Some((_, program)) = kernels.pop_last() {
                    device.release_program(program)?;
                }
                while let Some(queue) = queues.pop() {
                    device.release_queue(queue)?;
                }
                device.deinitialize()?;
            }
            Device::OpenCL { device, mut queues, mut kernels, .. } => {
                while let Some((_, program)) = kernels.pop_last() {
                    device.release_program(program)?;
                }
                while let Some(queue) = queues.pop() {
                    device.release_queue(queue)?;
                }
                device.deinitialize()?;
            }
            #[cfg(feature = "vulkan")]
            Device::Vulkan { device, mut programs, mut queues, .. } => {
                let ids: Vec<Id> = programs.ids().collect();
                for id in ids {
                    let program = programs.remove(id).unwrap();
                    device.release_program(program)?;
                }
                while let Some(queue) = queues.pop() {
                    device.release_queue(queue)?;
                }
                device.deinitialize()?;
            }
            #[cfg(feature = "wgsl")]
            Device::WGSL { device, mut programs, mut queues, .. } => {
                let ids: Vec<Id> = programs.ids().collect();
                for id in ids {
                    let program = programs.remove(id).unwrap();
                    device.release_program(program)?;
                }
                while let Some(queue) = queues.pop() {
                    device.release_queue(queue)?;
                }
                device.deinitialize()?;
            }
        }
        Ok(())
    }

    // NOTE returns memory pool id out of runtime memory pools
    pub(super) const fn memory_pool_id(&self) -> MemoryPoolId {
        match self {
            Device::CUDA { memory_pool_id, .. } => *memory_pool_id,
            Device::HIP { memory_pool_id, .. } => *memory_pool_id,
            Device::OpenCL { memory_pool_id, .. } => *memory_pool_id,
            #[cfg(feature = "vulkan")]
            Device::Vulkan { memory_pool_id, .. } => *memory_pool_id,
            #[cfg(feature = "wgsl")]
            Device::WGSL { memory_pool_id, .. } => *memory_pool_id,
        }
    }

    pub(super) const fn info(&self) -> &DeviceInfo {
        match self {
            Device::CUDA { device, .. } => device.info(),
            Device::HIP { device, .. } => device.info(),
            Device::OpenCL { device, .. } => device.info(),
            #[cfg(feature = "vulkan")]
            Device::Vulkan { device, .. } => device.info(),
            #[cfg(feature = "wgsl")]
            Device::WGSL { device, .. } => device.info(),
        }
    }

    pub(super) const fn compute(&self) -> u128 {
        self.info().compute
    }

    /*pub(super) fn sync(&mut self, event: Event) -> Result<(), ZyxError> {
        let id = event.id as usize;
        match self {
            Device::CUDA { queues, .. } => queues[id].sync()?,
            Device::HIP { queues, .. } => queues[id].sync()?,
            Device::OpenCL { queues, .. } => queues[id].sync()?,
            #[cfg(feature = "vulkan")]
            Device::Vulkan { queues, .. } => queues[id].sync()?,
            #[cfg(feature = "wgsl")]
            Device::WGSL { queues, .. } => queues[id].sync()?,
        }
        Ok(())
    }*/

    pub(super) fn release_program(&mut self, kernel: &[Op]) -> Result<(), ZyxError> {
        //println!("Release program {program_id}");
        match self {
            Device::CUDA { device, kernels, .. } => {
                device.release_program(kernels.remove(kernel).unwrap())?
            }
            Device::HIP { device, kernels, .. } => {
                device.release_program(kernels.remove(kernel).unwrap())?
            }
            Device::OpenCL { device, kernels, .. } => {
                device.release_program(kernels.remove(kernel).unwrap())?
            }
            #[cfg(feature = "vulkan")]
            Device::Vulkan { device, programs, .. } => {
                device.release_program(programs.remove(program_id).unwrap())?
            }
            #[cfg(feature = "wgsl")]
            Device::WGSL { device, programs, .. } => {
                device.release_program(programs.remove(program_id).unwrap())?
            }
        }
        Ok(())
    }

    pub(super) fn compile(
        &mut self,
        kernel: Vec<Op>,
        ir_kernel: &IRKernel,
        debug_asm: bool,
    ) -> Result<(), ZyxError> {
        match self {
            Device::CUDA { device, kernels, .. } => {
                kernels.insert(kernel, device.compile(ir_kernel, debug_asm)?);
            }
            Device::HIP { device, kernels, .. } => {
                kernels.insert(kernel, device.compile(ir_kernel, debug_asm)?);
            }
            Device::OpenCL { device, kernels, .. } => {
                kernels.insert(kernel, device.compile(ir_kernel, debug_asm)?);
            }
            #[cfg(feature = "vulkan")]
            Device::Vulkan { device, programs, .. } => {
                programs.push(device.compile(ir_kernel, debug_asm)?)
            }
            #[cfg(feature = "wgsl")]
            Device::WGSL { device, programs, .. } => {
                programs.push(device.compile(ir_kernel, debug_asm)?)
            }
        };
        Ok(())
    }

    pub(super) fn is_cached(&self, kernel: &[Op]) -> bool {
        match self {
            Device::CUDA { kernels, .. } => kernels.contains_key(kernel),
            Device::HIP { kernels, .. } => kernels.contains_key(kernel),
            Device::OpenCL { kernels, .. } => kernels.contains_key(kernel),
            #[cfg(feature = "vulkan")]
            Device::Vulkan { kernels, .. } => kernels.contains_key(kernel),
            #[cfg(feature = "wgsl")]
            Device::WGSL { kernels, .. } => kernels.contains_key(kernel),
        }
    }

    pub(super) fn launch(
        &mut self,
        kernel: &[Op],
        memory_pool: &mut MemoryPool,
        buffer_ids: &[Id],
    ) -> Result<Event, ZyxError> {
        Ok(match self {
            Device::CUDA { kernels, queues, .. } => {
                let mut queue = queues.iter_mut().min_by_key(|queue| queue.load()).unwrap();
                if queue.load() > 20 {
                    queue = queues.iter_mut().max_by_key(|queue| queue.load()).unwrap();
                    queue.sync()?;
                }
                let MemoryPool::CUDA { buffers, .. } = memory_pool else { unreachable!() };
                let event = queue.launch(kernels.get_mut(kernel).unwrap(), buffers, buffer_ids)?;
                Event::CUDA(event)
            }
            Device::HIP { kernels, queues, .. } => {
                let mut queue = queues.iter_mut().min_by_key(|queue| queue.load()).unwrap();
                if queue.load() > 20 {
                    queue = queues.iter_mut().max_by_key(|queue| queue.load()).unwrap();
                    queue.sync()?;
                }
                let MemoryPool::HIP { buffers, .. } = memory_pool else { unreachable!() };
                queue.launch(kernels.get_mut(kernel).unwrap(), buffers, buffer_ids)?;
                Event::HIP
            }
            Device::OpenCL { kernels, queues, .. } => {
                let mut queue = queues.iter_mut().min_by_key(|queue| queue.load()).unwrap();
                if queue.load() > 20 {
                    queue.sync()?;
                    queue = queues.iter_mut().max_by_key(|queue| queue.load()).unwrap();
                }
                let MemoryPool::OpenCL { buffers, .. } = memory_pool else { unreachable!() };
                let event = queue.launch(kernels.get_mut(kernel).unwrap(), buffers, buffer_ids)?;
                Event::OpenCL(event)
            }
            #[cfg(feature = "vulkan")]
            Device::Vulkan { programs, queues, .. } => {
                let (mut id, mut queue) =
                    queues.iter_mut().enumerate().min_by_key(|(_, queue)| queue.load()).unwrap();
                if queue.load() > 10 {
                    (id, queue) = queues
                        .iter_mut()
                        .enumerate()
                        .max_by_key(|(_, queue)| queue.load())
                        .unwrap();
                    queue.sync()?;
                }
                let MemoryPool::Vulkan { buffers, .. } = memory_pool else { unreachable!() };
                queue.launch(&mut programs[program_id], buffers, buffer_ids)?;
            }
            #[cfg(feature = "wgsl")]
            Device::WGSL { programs, queues, .. } => {
                let (mut id, mut queue) =
                    queues.iter_mut().enumerate().min_by_key(|(_, queue)| queue.load()).unwrap();
                if queue.load() > 10 {
                    (id, queue) = queues
                        .iter_mut()
                        .enumerate()
                        .max_by_key(|(_, queue)| queue.load())
                        .unwrap();
                    queue.sync()?;
                }
                let MemoryPool::WGSL { buffers, .. } = memory_pool else { unreachable!() };
                queue.launch(&mut programs[program_id], buffers, buffer_ids)?;
            }
        })
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CUDA { memory_pool_id, .. } => f.write_fmt(format_args!(
                "Device {{ memory_pool_id: {memory_pool_id} }})"
            )),
            Device::HIP { memory_pool_id, .. } => f.write_fmt(format_args!(
                "Device {{ memory_pool_id: {memory_pool_id} }})"
            )),
            Device::OpenCL { memory_pool_id, .. } => f.write_fmt(format_args!(
                "Device {{ memory_pool_id: {memory_pool_id} }})"
            )),
            #[cfg(feature = "vulkan")]
            Device::Vulkan { memory_pool_id, .. } => f.write_fmt(format_args!(
                "Device {{ memory_pool_id: {memory_pool_id} }})"
            )),
            #[cfg(feature = "wgsl")]
            Device::WGSL { memory_pool_id, .. } => f.write_fmt(format_args!(
                "Device {{ memory_pool_id: {memory_pool_id} }})"
            )),
        }
    }
}
