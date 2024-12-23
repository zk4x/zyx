//! This file creates backend agnostic API to backends
//! That is it contains enums that dispatch function calls to appropriate backends.
//! Backend automatically keeps track of hardware queues.
//! Interfaces use events independent from underlying implementation.
//! Events are used to achieve maximum asynchronous execution.

// Because I don't want to write struct and inner enum for MemoryPool and Device
//#![allow(private_interfaces)]

use std::fmt::Display;

use nanoserde::DeJson;

mod cuda;
mod dummy;
mod opencl;
/*mod hip;
#[cfg(feature = "vulkan")]
mod vulkan;*/
#[cfg(feature = "wgpu")]
mod wgpu;

pub fn initialize_backends(
    device_config: &DeviceConfig,
    memory_pools: &mut Vec<Pool>,
    devices: &mut Vec<Box<dyn Device>>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    let _ = dummy::initialize_device(&device_config.dummy, memory_pools, devices, debug_dev);
    let _ = cuda::initialize_device(&device_config.cuda, memory_pools, devices, debug_dev);
    //let _ = hip::initialize_device(&device_config.hip, memory_pools, devices, debug_dev);
    let _ = opencl::initialize_device(&device_config.opencl, memory_pools, devices, debug_dev);
    #[cfg(feature = "vulkan")]
    let _ = vulkan::initialize_device(&device_config.vulkan, memory_pools, devices, debug_dev);
    #[cfg(feature = "wgpu")]
    let _ = wgpu::initialize_device(&device_config.wgpu, memory_pools, devices, debug_dev);

    if devices.is_empty() || memory_pools.is_empty() {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: "All backends failed to initialize or were configured out.".into(),
        });
    }
    Ok(())
}

use crate::{ir::IRKernel, runtime::Pool, slab::Id, ZyxError};

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
    /// Vulkan configuration
    #[cfg(feature = "vulkan")]
    pub vulkan: vulkan::VulkanConfig,
    /// WGSL configuration
    #[cfg(feature = "wgpu")]
    pub wgpu: wgpu::WGPUConfig,
}

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
    /// Maximum local work size dimensions
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

pub trait MemoryPool: Send {
    fn deinitialize(&mut self) -> Result<(), BackendError>;
    fn free_bytes(&self) -> usize;
    fn get_buffer(&mut self, buffer: Id) -> BufferMut;
    fn allocate(&mut self, bytes: usize) -> Result<(Id, Event), BackendError>;
    fn deallocate(
        &mut self,
        buffer_id: Id,
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError>;
    fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: Id,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError>;
    /// Pool to host is blocking operation
    fn pool_to_host(
        &mut self,
        src: Id,
        dst: &mut [u8],
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError>;
}

pub trait Device: Send {
    fn deinitialize(&mut self) -> Result<(), BackendError>;
    fn info(&self) -> &DeviceInfo;
    fn memory_pool_id(&self) -> u32;
    fn compute(&self) -> u128;
    fn compile(&mut self, kernel: &IRKernel, debug_asm: bool) -> Result<Id, BackendError>;
    fn release(&mut self, program_id: Id) -> Result<(), BackendError>;
    fn launch(
        &mut self,
        program_id: Id,
        memory_pool: &mut dyn MemoryPool,
        args: &[Id],
        event_wait_list: Vec<Event>,
        // Immediatelly synchronize?
        sync: bool,
    ) -> Result<Event, BackendError>;
    fn sync(&mut self, event_wait_list: Vec<Event>) -> Result<(), BackendError>;
}

#[allow(private_interfaces)]
pub enum BufferMut<'a> {
    Dummy,
    OpenCL(&'a mut opencl::OpenCLBuffer),
    CUDA(&'a mut cuda::CUDABuffer),
}

#[derive(Debug)]
pub enum Event {
    OpenCL(opencl::OpenCLEvent),
    CUDA(cuda::CUDAEvent),
}

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
