//! This file creates backend agnostic API to backends
//! That is it contains enums that dispatch function calls to appropriate backends.
//! Backend automatically keeps track of hardware queues.
//! Interfaces use events independent from underlying implementation.
//! Events are used to achieve maximum asynchronous execution.

// Because I don't want to write struct and inner enum for MemoryPool and Device

use crate::{ir::IRKernel, runtime::Pool, shape::Dimension, slab::Id, ZyxError};
use cuda::{CUDADevice, CUDAMemoryPool};
use disk::DiskMemoryPool;
use dummy::{DummyDevice, DummyMemoryPool};
use nanoserde::DeJson;
use opencl::{OpenCLDevice, OpenCLMemoryPool};
#[cfg(feature = "wgpu")]
use wgpu::{WGPUDevice, WGPUMemoryPool};
use std::fmt::Display;

mod disk;
mod cuda;
mod dummy;
mod opencl;
/*mod hip;
#[cfg(feature = "vulkan")]
mod vulkan;*/
#[cfg(feature = "wgpu")]
mod wgpu;

impl From<BackendError> for ZyxError {
    fn from(value: BackendError) -> Self {
        ZyxError::BackendError(value)
    }
}

impl Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}: {}", self.status, self.context))
    }
}

pub fn initialize_backends(
    device_config: &DeviceConfig,
    memory_pools: &mut Vec<Pool>,
    devices: &mut Vec<Device>,
    debug_backends: bool,
) -> Result<(), BackendError> {
    if let Err(err) = disk::initialize_pool(memory_pools, debug_backends) {
        if debug_backends {
            println!("Disk init error: {err}");
        }
    }
    if let Err(err) = dummy::initialize_device(&device_config.dummy, memory_pools, devices, debug_backends) {
        if debug_backends {
            println!("dummy init error: {err}");
        }
    }
    if let Err(err) = cuda::initialize_device(&device_config.cuda, memory_pools, devices, debug_backends) {
        if debug_backends {
            println!("cuda init error: {err}");
        }
    }
    //let _ = hip::initialize_device(&device_config.hip, memory_pools, devices, debug_dev);
    if let Err(err) = opencl::initialize_device(&device_config.opencl, memory_pools, devices, debug_backends) {
        if debug_backends {
            println!("opencl init error: {err}");
        }
    }
    //#[cfg(feature = "vulkan")]
    //let _ = vulkan::initialize_device(&device_config.vulkan, memory_pools, devices, debug_dev);
    #[cfg(feature = "wgpu")]
    if let Err(err) = wgpu::initialize_device(&device_config.wgpu, memory_pools, devices, debug_backends) {
        if debug_backends {
            println!("wgpu init error: {err}");
        }
    }
    //println!("Memory pools: {memory_pools:?}");

    if devices.is_empty() || memory_pools.is_empty() {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: "All backends failed to initialize or were configured out.".into(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub enum Event {
    #[allow(unused)]
    Disk(disk::DiskEvent),
    OpenCL(opencl::OpenCLEvent),
    CUDA(cuda::CUDAEvent),
    #[cfg(feature = "wgpu")]
    WGPU(wgpu::WGPUEvent),
}

/// Device configuration
#[cfg_attr(feature = "py", pyo3::pyclass)]
#[derive(DeJson, Debug, Default)]
pub struct DeviceConfig {
    /// Configuration of dummy device for testing
    pub dummy: dummy::DummyConfig,
    /// CUDA configuration
    pub cuda: cuda::CUDAConfig,
    /// HIP configuration
    //pub hip: hip::HIPConfig,
    /// `OpenCL` configuration
    pub opencl: opencl::OpenCLConfig,
    // Vulkan configuration
    //#[cfg(feature = "vulkan")]
    //pub vulkan: vulkan::VulkanConfig,
    /// WGSL configuration
    #[cfg(feature = "wgpu")]
    pub wgpu: wgpu::WGPUConfig,
}

#[derive(Debug)]
pub enum ErrorStatus {
    /// Dynamic library was not found on the disk
    DyLibNotFound,
    /// Backend initialization failure
    Initialization,
    /// Backend deinitialization failure
    Deinitialization,
    /// Failed to enumerate devices
    DeviceEnumeration,
    /// Failed to query device for information
    DeviceQuery,
    /// Failed to allocate memory
    MemoryAllocation,
    /// Failed to deallocate memory
    MemoryDeallocation,
    /// Failed to copy memory to pool
    MemoryCopyH2P,
    /// Failed to copy memory to host
    MemoryCopyP2H,
    /// Kernel argument was not correct
    IncorrectKernelArg,
    /// Failed to compile kernel
    KernelCompilation,
    /// Failed to launch kernel
    KernelLaunch,
    /// Failed to synchronize kernel
    KernelSync,
}

#[derive(Debug)]
pub struct BackendError {
    status: ErrorStatus,
    context: String,
}

/// Hardware information needed for applying optimizations
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeviceInfo {
    /// Device compute in flops
    pub compute: u128,
    /// Biggest kernel dimensions
    pub max_global_work_dims: [Dimension; 3],
    /// Maximum local work size threads
    pub max_local_threads: Dimension,
    /// Maximum local work size dimensions
    pub max_local_work_dims: [Dimension; 3],
    /// Preferred vector size in bytes
    pub preferred_vector_size: u8,
    /// Local memory size in bytes
    pub local_mem_size: Dimension,
    /// Number of registers per thread
    pub num_registers: u16,
    /// Does this hardware have tensor cores?
    pub tensor_cores: bool,
}

#[derive(Debug)]
pub(super) enum MemoryPool {
    Dummy(DummyMemoryPool),
    Disk(DiskMemoryPool),
    CUDA(CUDAMemoryPool),
    OpenCL(OpenCLMemoryPool),
    #[cfg(feature = "wgpu")]
    WGPU(WGPUMemoryPool),
}

impl MemoryPool {
    pub fn deinitialize(&mut self) -> Result<(), BackendError> {
        match self {
            MemoryPool::Disk(pool) => pool.deinitialize(),
            MemoryPool::CUDA(pool) => pool.deinitialize(),
            MemoryPool::OpenCL(pool) => pool.deinitialize(),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.deinitialize(),
            MemoryPool::Dummy(pool) => pool.deinitialize(),
        }
    }

    pub fn disk_pool(&mut self) -> Option<&mut DiskMemoryPool> {
        match self {
            Self::Disk(disk) => Some(disk),
            _ => None,
        }
    }

    pub fn free_bytes(&self) -> Dimension {
        match self {
            MemoryPool::Disk(pool) => pool.free_bytes(),
            MemoryPool::CUDA(pool) => pool.free_bytes(),
            MemoryPool::OpenCL(pool) => pool.free_bytes(),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.free_bytes(),
            MemoryPool::Dummy(pool) => pool.free_bytes(),
        }
    }

    pub fn allocate(&mut self, bytes: Dimension) -> Result<(Id, Event), BackendError> {
        match self {
            MemoryPool::Disk(_) => todo!(),
            MemoryPool::CUDA(pool) => pool.allocate(bytes),
            MemoryPool::OpenCL(pool) => pool.allocate(bytes),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.allocate(bytes),
            MemoryPool::Dummy(pool) => pool.allocate(bytes),
        }
    }

    // Deallocate drops events without synchronization
    pub fn deallocate(
        &mut self,
        buffer_id: Id,
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        match self {
            MemoryPool::Disk(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::CUDA(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::OpenCL(pool) => pool.deallocate(buffer_id, event_wait_list),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::Dummy(pool) => pool.deallocate(buffer_id, event_wait_list),
        }
    }

    // Host to pool does not synchronize events, it keeps them alive
    // src must be alive as long as Event is not synchronized
    pub fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: Id,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        match self {
            MemoryPool::Disk(_) => todo!(),
            MemoryPool::CUDA(pool) => pool.host_to_pool(src, dst, event_wait_list),
            MemoryPool::OpenCL(pool) => pool.host_to_pool(src, dst, event_wait_list),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.host_to_pool(src, dst, event_wait_list),
            MemoryPool::Dummy(pool) => pool.host_to_pool(src, dst, event_wait_list),
        }
    }

    /// Pool to host is blocking operation, synchronizes events and drops them
    pub fn pool_to_host(
        &mut self,
        src: Id,
        dst: &mut [u8],
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        match self {
            MemoryPool::Disk(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::CUDA(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::OpenCL(pool) => pool.pool_to_host(src, dst, event_wait_list),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::Dummy(pool) => pool.pool_to_host(src, dst, event_wait_list),
        }
    }

    // Synchronize events, blocking, drops those events
    pub fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        match self {
            MemoryPool::Disk(pool) => pool.sync_events(events),
            MemoryPool::CUDA(pool) => pool.sync_events(events),
            MemoryPool::OpenCL(pool) => pool.sync_events(events),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.sync_events(events),
            MemoryPool::Dummy(pool) => pool.sync_events(events),
        }
    }

    // Drop events without synchronization, non-blocking
    pub fn release_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        match self {
            MemoryPool::Disk(pool) => pool.release_events(events),
            MemoryPool::CUDA(pool) => pool.release_events(events),
            MemoryPool::OpenCL(pool) => pool.release_events(events),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.release_events(events),
            MemoryPool::Dummy(pool) => pool.release_events(events),
        }
    }
}

pub(super) enum Device {
    CUDA(CUDADevice),
    OpenCL(OpenCLDevice),
    #[cfg(feature = "wgpu")]
    WGPU(WGPUDevice),
    Dummy(DummyDevice),
}

impl Device {
    pub fn deinitialize(&mut self) -> Result<(), BackendError> {
        match self {
            Device::CUDA(dev) => dev.deinitialize(),
            Device::OpenCL(dev) => dev.deinitialize(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.deinitialize(),
            Device::Dummy(dev) => dev.deinitialize(),
        }
    }

    pub fn info(&self) -> &DeviceInfo {
        match self {
            Device::CUDA(dev) => dev.info(),
            Device::OpenCL(dev) => dev.info(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.info(),
            Device::Dummy(dev) => dev.info(),
        }
    }

    pub fn memory_pool_id(&self) -> u32 {
        match self {
            Device::CUDA(dev) => dev.memory_pool_id(),
            Device::OpenCL(dev) => dev.memory_pool_id(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.memory_pool_id(),
            Device::Dummy(dev) => dev.memory_pool_id(),
        }
    }

    /// How much compute is available on the device, this should be multiplied by (1 - device_usage),
    /// so that we spread the laod across all available devices appropriatelly.
    pub fn free_compute(&self) -> u128 {
        match self {
            Device::CUDA(dev) => dev.free_compute(),
            Device::OpenCL(dev) => dev.free_compute(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.compute(),
            Device::Dummy(dev) => dev.free_compute(),
        }
    }

    pub fn compile(&mut self, kernel: &IRKernel, debug_asm: bool) -> Result<Id, BackendError> {
        match self {
            Device::CUDA(dev) => dev.compile(kernel, debug_asm),
            Device::OpenCL(dev) => dev.compile(kernel, debug_asm),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.compile(kernel, debug_asm),
            Device::Dummy(dev) => dev.compile(kernel, debug_asm),
        }
    }

    pub fn release(&mut self, program_id: Id) -> Result<(), BackendError> {
        match self {
            Device::CUDA(dev) => dev.release(program_id),
            Device::OpenCL(dev) => dev.release(program_id),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.release(program_id),
            Device::Dummy(dev) => dev.release(program_id),
        }
    }

    pub fn launch(
        &mut self,
        program_id: Id,
        memory_pool: &mut MemoryPool,
        args: &[Id],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        match self {
            Device::CUDA(dev) => {
                let MemoryPool::CUDA(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            Device::OpenCL(dev) => {
                let MemoryPool::OpenCL(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => {
                let MemoryPool::WGPU(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            Device::Dummy(dev) => {
                let MemoryPool::Dummy(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
        }
    }
}
