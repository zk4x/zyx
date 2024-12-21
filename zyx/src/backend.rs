//! This file creates backend agnostic API to backends
//! That is it contains enums that dispatch function calls to appropriate backends.
//! Backend automatically keeps track of hardware queues and events
//! and synchronizes events as needed at latest possible moment
//! for maximum concurrency.

// Because I don't want to write struct and inner enum for MemoryPool and Device
#![allow(private_interfaces)]

use std::{collections::BTreeSet, fmt::Display};

use nanoserde::DeJson;
#[cfg(feature = "wgsl")]
use wgsl::{WGSLBuffer, WGSLDevice, WGSLMemoryPool, WGSLProgram, WGSLQueue};

mod opencl;
mod cuda;
/*mod hip;
#[cfg(feature = "vulkan")]
mod vulkan;
#[cfg(feature = "wgsl")]
mod wgsl;*/

use crate::{ir::IRKernel, slab::Id, ZyxError};

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
    /// Failed to copy memory
    MemoryCopy,
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
    #[cfg(feature = "wgsl")]
    pub wgsl: wgsl::WGSLConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId {
    pub memory_pool_id: u32,
    pub buffer_id: Id,
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
    fn allocate(&mut self, bytes: usize) -> Result<Id, BackendError>;
    fn deallocate(&mut self, buffer_id: Id) -> Result<(), BackendError>;
    fn host_to_pool(&mut self, src: &[u8], dst: Id) -> Result<(), BackendError>;
    fn pool_to_host(&mut self, src: Id, dst: &mut [u8]) -> Result<(), BackendError>;
    fn get_buffer(&mut self, buffer: Id) -> BufferMut;
    fn event_wait_list(&mut self, buffers: &BTreeSet<Id>) -> Vec<Event>;
    fn bind_event(&mut self, event: Event, buffers: BTreeSet<Id>);
}

pub trait Device: Send {
    fn deinitialize(&mut self) -> Result<(), BackendError>;
    fn info(&self) -> &DeviceInfo;
    fn memory_pool_id(&self) -> u32;
    fn compute(&self) -> u128;
    fn compile(&mut self, kernel: &IRKernel, debug_asm: bool) -> Result<Id, BackendError>;
    // Returns if this is the first time running the kernel
    fn launch(
        &mut self,
        program_id: Id,
        memory_pool: &mut dyn MemoryPool,
        args: &[Id],
        // If sync is empty, kernel will be immediatelly synchronized
        sync: BTreeSet<Id>,
    ) -> Result<(), BackendError>;
    fn release(&mut self, program_id: Id) -> Result<(), BackendError>;
}

enum BufferMut<'a> {
    OpenCL(&'a mut opencl::OpenCLBuffer),
    CUDA(&'a mut cuda::CUDABuffer),
}

enum Event {
    OpenCL(opencl::OpenCLEvent),
    CUDA(cuda::CUDAEvent),
}

pub fn initialize_backends(
    device_config: &DeviceConfig,
    memory_pools: &mut Vec<Box<dyn MemoryPool>>,
    devices: &mut Vec<Box<dyn Device>>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    let _ = cuda::initialize_device(&device_config.cuda, memory_pools, devices, debug_dev);

    //let _ = hip::initialize_device(&device_config.hip, memory_pools, devices, debug_dev);

    let _ = opencl::initialize_device(&device_config.opencl, memory_pools, devices, debug_dev);

    #[cfg(feature = "vulkan")]
    let _ = vulkan::initialize_devices(&device_config.opencl, memory_pools, devices, debug_dev);

    #[cfg(feature = "wgsl")]
    let _ = wgsl::initialize_devices(&device_config.opencl, memory_pools, devices, debug_dev);

    if devices.is_empty() || memory_pools.is_empty() {
        return Err(BackendError {
            status: ErrorStatus::Initialization,
            context: "All backends failed to initialize or were configured out.".into(),
        });
    }
    Ok(())
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
