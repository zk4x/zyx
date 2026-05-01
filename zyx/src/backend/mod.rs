// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! This file creates backend agnostic API to backends
//! That is it contains enums that dispatch function calls to appropriate backends.
//! Backend automatically keeps track of hardware queues.
//! Interfaces use events independent from underlying implementation.
//! Events are used to achieve maximum asynchronous execution.

#![allow(clippy::needless_pass_by_ref_mut)]
#![allow(clippy::upper_case_acronyms)]

// Because I don't want to write struct and inner enum for MemoryPool and Device

use crate::{
    backend::hip::{HIPDevice, HIPMemoryPool},
    dtype::DType,
    error::{BackendError, ErrorStatus},
    kernel::Kernel,
    shape::Dim,
    slab::{Slab, SlabId},
};
use cuda::{CUDADevice, CUDAMemoryPool};
use disk::DiskMemoryPool;
use dummy::{DummyDevice, DummyMemoryPool};
use host::HostMemoryPool;
use nanoserde::{DeBin, DeJson, SerBin};
use opencl::{OpenCLDevice, OpenCLMemoryPool};
#[cfg(feature = "wgpu")]
use wgpu::{WGPUDevice, WGPUMemoryPool};

mod cuda;
mod disk;
mod dummy;
mod hip;
mod host;
mod opencl;
/*#[cfg(feature = "vulkan")]
mod vulkan;*/
#[cfg(feature = "wgpu")]
mod wgpu;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PoolBufferId(u32);

impl From<usize> for PoolBufferId {
    fn from(value: usize) -> Self {
        PoolBufferId(u32::try_from(value).unwrap())
    }
}

impl From<PoolBufferId> for usize {
    fn from(value: PoolBufferId) -> Self {
        value.0 as usize
    }
}

impl SlabId for PoolBufferId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DeBin, SerBin)]
pub struct DeviceProgramId(u32);

impl From<usize> for DeviceProgramId {
    fn from(value: usize) -> Self {
        DeviceProgramId(u32::try_from(value).unwrap())
    }
}

impl From<DeviceProgramId> for usize {
    fn from(value: DeviceProgramId) -> Self {
        value.0 as usize
    }
}

impl SlabId for DeviceProgramId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

/// Pool identifier for use with `Slab<PoolId, MemoryPool>`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PoolId(u32);

impl From<usize> for PoolId {
    fn from(value: usize) -> Self {
        PoolId(u32::try_from(value).unwrap())
    }
}

impl From<PoolId> for usize {
    fn from(value: PoolId) -> Self {
        value.0 as usize
    }
}

impl SlabId for PoolId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

impl std::ops::AddAssign<u32> for PoolId {
    fn add_assign(&mut self, rhs: u32) {
        self.0 += rhs;
    }
}

/// Device identifier for use with `Slab<DeviceId, Device>`
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, DeBin, SerBin)]
pub struct DeviceId(pub u32);

impl From<usize> for DeviceId {
    fn from(value: usize) -> Self {
        DeviceId(u32::try_from(value).unwrap())
    }
}

impl From<DeviceId> for usize {
    fn from(value: DeviceId) -> Self {
        value.0 as usize
    }
}

impl SlabId for DeviceId {
    const ZERO: Self = Self(0);
    const NULL: Self = Self(u32::MAX);

    fn inc(&mut self) {
        self.0 += 1;
    }
}

/// Globally unique buffer identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferId {
    pub pool: PoolId,
    pub buffer: PoolBufferId,
}

impl BufferId {
    pub const NULL: Self = Self { pool: PoolId::NULL, buffer: PoolBufferId(u32::MAX) };
}

impl From<usize> for BufferId {
    fn from(value: usize) -> Self {
        BufferId { pool: PoolId::ZERO, buffer: PoolBufferId(u32::try_from(value).unwrap()) }
    }
}

impl From<BufferId> for usize {
    fn from(value: BufferId) -> Self {
        value.buffer.0 as usize
    }
}

/// Globally unique program identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramId {
    pub device: DeviceId,
    pub program: DeviceProgramId,
}

impl ProgramId {
    pub const NULL: Self = Self { device: DeviceId::NULL, program: DeviceProgramId(u32::MAX) };
}

impl From<usize> for ProgramId {
    fn from(value: usize) -> Self {
        ProgramId { device: DeviceId::ZERO, program: DeviceProgramId(u32::try_from(value).unwrap()) }
    }
}

impl From<ProgramId> for usize {
    fn from(value: ProgramId) -> Self {
        value.program.0 as usize
    }
}

impl From<libloading::Error> for BackendError {
    fn from(value: libloading::Error) -> Self {
        BackendError { status: ErrorStatus::Initialization, context: value.to_string().into() }
    }
}

pub fn initialize_backends(
    device_config: &Config,
    memory_pools: &mut Slab<PoolId, MemoryPool>,
    devices: &mut Slab<DeviceId, Device>,
    debug_backends: bool,
) -> Result<(), BackendError> {
    if let Err(err) = host::initialize_pool(memory_pools, debug_backends) {
        if debug_backends {
            println!("{err}");
        }
    }
    if let Err(err) = disk::initialize_pool(memory_pools, debug_backends) {
        if debug_backends {
            println!("{err}");
        }
    }
    if let Err(err) = dummy::initialize_device(&device_config.dummy, memory_pools, devices, debug_backends) {
        if debug_backends {
            println!("{err}");
        }
    }
    if let Err(err) = cuda::initialize_device(&device_config.cuda, memory_pools, devices, debug_backends) {
        if debug_backends {
            println!("{err}");
        }
    }
    if let Err(err) = hip::initialize_device(&device_config.hip, memory_pools, devices, debug_backends) {
        if debug_backends {
            println!("{err}");
        }
    }
    if let Err(err) = opencl::initialize_device(&device_config.opencl, memory_pools, devices, debug_backends) {
        if debug_backends {
            println!("{err}");
        }
    }
    #[cfg(feature = "wgpu")]
    if let Err(err) = wgpu::initialize_device(&device_config.wgpu, memory_pools, devices, debug_backends) {
        if debug_backends {
            println!("{err}");
        }
    }

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
    Host(host::HostEvent),
    CUDA(cuda::CUDAEvent),
    OpenCL(opencl::OpenCLEvent),
    HIP(hip::HIPEvent),
    #[cfg(feature = "wgpu")]
    WGPU(wgpu::WGPUEvent),
}

#[cfg_attr(feature = "py", pyo3::pyclass)]
#[derive(DeJson, Debug)]
pub struct AutotuneConfig {
    #[allow(unused)]
    /// Should the autotuned kernel be stored to disk?
    pub save_to_disk: bool,
    /// Max number of kernel launches
    pub n_launches: usize, // = 10;
    /// Number of initial optimization seeds
    pub n_seeds: usize, // = 100;
    /// How many optimizations to try each iteration
    pub n_added_per_step: usize, //: usize = 10;
    /// How many iterations to remove each iteration
    pub n_removed_per_step: usize, // = 5;
    /// Max number of optimizations that can be tried
    pub n_total_opts: usize, // = 1000;
}

impl Default for AutotuneConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl AutotuneConfig {
    pub const fn new() -> AutotuneConfig {
        AutotuneConfig {
            save_to_disk: true,
            n_added_per_step: 10,
            n_launches: 20,
            n_removed_per_step: 5,
            n_seeds: 100,
            n_total_opts: 1000,
        }
    }
}

/// Device configuration
#[cfg_attr(feature = "py", pyo3::pyclass)]
#[derive(DeJson, Debug, Default)]
pub struct Config {
    /// Kernel autotune configuration
    pub autotune: AutotuneConfig,
    /// Configuration of dummy device for testing
    pub dummy: dummy::DummyConfig,
    /// CUDA configuration
    pub cuda: cuda::CUDAConfig,
    /// HIP configuration
    pub hip: hip::HIPConfig,
    /// `OpenCL` configuration
    pub opencl: opencl::OpenCLConfig,
    // Vulkan configuration
    //#[cfg(feature = "vulkan")]
    //pub vulkan: vulkan::VulkanConfig,
    /// WGSL configuration
    #[cfg(feature = "wgpu")]
    pub wgpu: wgpu::WGPUConfig,
}

/// Hardware information needed for applying optimizations
#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, SerBin, DeBin)]
pub struct DeviceInfo {
    /// Device compute in flops
    pub compute: u128,
    /// Biggest kernel dimensions
    pub max_global_work_dims: Vec<Dim>,
    /// Maximum local work size threads
    pub max_local_threads: Dim,
    /// Maximum local work size dimensions
    pub max_local_work_dims: Vec<Dim>,
    /// Preferred vector size in bytes
    pub preferred_vector_size: u8,
    /// Local memory size in bytes
    pub local_mem_size: Dim,
    /// private memory size in bytes
    pub max_register_bytes: Dim,
    /// Does this hardware have tensor cores?
    pub tensor_cores: bool,
    /// Warp size
    pub warp_size: u16,
    /// Bitmask of supported DTypes (bit index = DType as u16)
    pub supported_dtypes: u32,
}

impl DeviceInfo {
    /// Check if a dtype is supported by this device
    pub fn supports_dtype(&self, dtype: DType) -> bool {
        self.supported_dtypes & (1 << dtype as u32) != 0
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub enum MemoryPool {
    Dummy(DummyMemoryPool),
    Disk(DiskMemoryPool),
    Host(HostMemoryPool),
    CUDA(CUDAMemoryPool),
    OpenCL(OpenCLMemoryPool),
    HIP(HIPMemoryPool),
    #[cfg(feature = "wgpu")]
    WGPU(WGPUMemoryPool),
}

impl MemoryPool {
    #[allow(unused)]
    pub fn deinitialize(&mut self) {
        match self {
            MemoryPool::Dummy(pool) => pool.deinitialize(),
            MemoryPool::Disk(pool) => pool.deinitialize(),
            MemoryPool::Host(pool) => pool.deinitialize(),
            MemoryPool::CUDA(pool) => pool.deinitialize(),
            MemoryPool::OpenCL(pool) => pool.deinitialize(),
            MemoryPool::HIP(pool) => pool.deinitialize(),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.deinitialize(),
        }
    }

    pub const fn disk_pool(&mut self) -> Option<&mut DiskMemoryPool> {
        match self {
            Self::Disk(disk) => Some(disk),
            _ => None,
        }
    }

    pub const fn free_bytes(&self) -> Dim {
        match self {
            MemoryPool::Dummy(pool) => pool.free_bytes(),
            MemoryPool::Disk(pool) => pool.free_bytes(),
            MemoryPool::Host(pool) => pool.free_bytes(),
            MemoryPool::CUDA(pool) => pool.free_bytes(),
            MemoryPool::OpenCL(pool) => pool.free_bytes(),
            MemoryPool::HIP(pool) => pool.free_bytes(),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.free_bytes(),
        }
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
        match self {
            MemoryPool::Dummy(pool) => pool.allocate(bytes),
            MemoryPool::Disk(_) => todo!(),
            MemoryPool::Host(pool) => pool.allocate(bytes),
            MemoryPool::CUDA(pool) => pool.allocate(bytes),
            MemoryPool::OpenCL(pool) => pool.allocate(bytes),
            MemoryPool::HIP(pool) => pool.allocate(bytes),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.allocate(bytes),
        }
    }

    // Deallocate drops events without synchronization
    pub fn deallocate(&mut self, buffer_id: PoolBufferId, event_wait_list: Vec<Event>) {
        match self {
            MemoryPool::Dummy(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::Disk(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::Host(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::CUDA(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::OpenCL(pool) => pool.deallocate(buffer_id, event_wait_list),
            MemoryPool::HIP(pool) => pool.deallocate(buffer_id, event_wait_list),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.deallocate(buffer_id, event_wait_list),
        }
    }

    // Host to pool does not synchronize events, it keeps them alive
    // src must be alive as long as Event is not synchronized
    pub fn host_to_pool(
        &mut self,
        src: &[u8], // TODO this will likely have to be Vec<u8> for better lifetimes handling and less synchronization
        dst: PoolBufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        match self {
            MemoryPool::Dummy(pool) => pool.host_to_pool(src, dst, event_wait_list),
            MemoryPool::Disk(_) => todo!(),
            MemoryPool::Host(pool) => pool.host_to_pool(src, dst, event_wait_list),
            MemoryPool::CUDA(pool) => pool.host_to_pool(src, dst, event_wait_list),
            MemoryPool::OpenCL(pool) => pool.host_to_pool(src, dst, event_wait_list),
            MemoryPool::HIP(pool) => pool.host_to_pool(src, dst, event_wait_list),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.host_to_pool(src, dst, event_wait_list),
        }
    }

    /// Pool to host is blocking operation, synchronizes events and drops them
    pub fn pool_to_host(&mut self, src: PoolBufferId, dst: &mut [u8], event_wait_list: Vec<Event>) -> Result<(), BackendError> {
        match self {
            MemoryPool::Dummy(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::Disk(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::Host(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::CUDA(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::OpenCL(pool) => pool.pool_to_host(src, dst, event_wait_list),
            MemoryPool::HIP(pool) => pool.pool_to_host(src, dst, event_wait_list),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.pool_to_host(src, dst, event_wait_list),
        }
    }

    // Synchronize events, blocking, drops those events
    pub fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        match self {
            MemoryPool::Dummy(pool) => pool.sync_events(events),
            MemoryPool::Disk(pool) => pool.sync_events(events),
            MemoryPool::Host(pool) => pool.sync_events(events),
            MemoryPool::CUDA(pool) => pool.sync_events(events),
            MemoryPool::OpenCL(pool) => pool.sync_events(events),
            MemoryPool::HIP(pool) => pool.sync_events(events),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.sync_events(events),
        }
    }

    // Drop events without synchronization, non-blocking
    #[allow(unused)]
    pub fn release_events(&mut self, events: Vec<Event>) {
        match self {
            MemoryPool::Dummy(pool) => pool.release_events(events),
            MemoryPool::Disk(pool) => pool.release_events(events),
            MemoryPool::Host(pool) => pool.release_events(events),
            MemoryPool::CUDA(pool) => pool.release_events(events),
            MemoryPool::OpenCL(pool) => pool.release_events(events),
            MemoryPool::HIP(pool) => pool.release_events(events),
            #[cfg(feature = "wgpu")]
            MemoryPool::WGPU(pool) => pool.release_events(events),
        }
    }
}

#[derive(Debug)]
pub enum Device {
    Dummy(DummyDevice),
    CUDA(CUDADevice),
    OpenCL(OpenCLDevice),
    HIP(HIPDevice),
    #[cfg(feature = "wgpu")]
    WGPU(WGPUDevice),
}

impl Device {
    #[allow(unused)]
    pub const fn deinitialize(&mut self) {
        match self {
            Device::Dummy(dev) => dev.deinitialize(),
            Device::CUDA(dev) => dev.deinitialize(),
            Device::OpenCL(dev) => dev.deinitialize(),
            Device::HIP(dev) => dev.deinitialize(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.deinitialize(),
        }
    }

    pub const fn info(&self) -> &DeviceInfo {
        match self {
            Device::Dummy(dev) => dev.info(),
            Device::CUDA(dev) => dev.info(),
            Device::OpenCL(dev) => dev.info(),
            Device::HIP(dev) => dev.info(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.info(),
        }
    }

    pub const fn memory_pool_id(&self) -> PoolId {
        match self {
            Device::Dummy(dev) => dev.memory_pool_id(),
            Device::CUDA(dev) => dev.memory_pool_id(),
            Device::OpenCL(dev) => dev.memory_pool_id(),
            Device::HIP(dev) => dev.memory_pool_id(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.memory_pool_id(),
        }
    }

    /// How much compute is available on the device,
    /// Internally this should be adjusted for current `device_usage`,
    /// so that we spread the laod across all available devices appropriatelly.
    pub const fn free_compute(&self) -> u128 {
        match self {
            Device::Dummy(dev) => dev.free_compute(),
            Device::CUDA(dev) => dev.free_compute(),
            Device::OpenCL(dev) => dev.free_compute(),
            Device::HIP(dev) => dev.free_compute(),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.free_compute(),
        }
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        match self {
            Device::Dummy(dev) => dev.compile(kernel, debug_asm),
            Device::CUDA(dev) => dev.compile(kernel, debug_asm),
            Device::OpenCL(dev) => dev.compile(kernel, debug_asm),
            Device::HIP(dev) => dev.compile(kernel, debug_asm),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.compile(kernel, debug_asm),
        }
    }

    pub fn release(&mut self, program_id: DeviceProgramId) {
        match self {
            Device::Dummy(dev) => dev.release(program_id),
            Device::CUDA(dev) => dev.release(program_id),
            Device::OpenCL(dev) => dev.release(program_id),
            Device::HIP(dev) => dev.release(program_id),
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => dev.release(program_id),
        }
    }

    pub fn launch(
        &mut self,
        program_id: DeviceProgramId,
        memory_pool: &mut MemoryPool,
        args: &[PoolBufferId],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        match self {
            Device::Dummy(dev) => {
                let MemoryPool::Dummy(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            Device::CUDA(dev) => {
                let MemoryPool::CUDA(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            Device::OpenCL(dev) => {
                let MemoryPool::OpenCL(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            Device::HIP(dev) => {
                let MemoryPool::HIP(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
            #[cfg(feature = "wgpu")]
            Device::WGPU(dev) => {
                let MemoryPool::WGPU(pool) = memory_pool else { unreachable!() };
                dev.launch(program_id, pool, args, event_wait_list)
            }
        }
    }
}
