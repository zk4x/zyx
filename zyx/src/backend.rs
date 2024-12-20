//! This file creates backend agnostic API to backends
//! That is it contains enums that dispatch function calls to appropriate backends.
//! Backend automatically keeps track of hardware queues and events
//! and synchronizes events as needed at latest possible moment
//! for maximum concurrency.

// Because I don't want to write struct and inner enum for MemoryPool and Device
#![allow(private_interfaces)]

use std::collections::BTreeSet;

use nanoserde::DeJson;
#[cfg(feature = "wgsl")]
use wgsl::{WGSLBuffer, WGSLDevice, WGSLMemoryPool, WGSLProgram, WGSLQueue};

mod opencl;
/*mod cuda;
mod hip;
#[cfg(feature = "vulkan")]
mod vulkan;
#[cfg(feature = "wgsl")]
mod wgsl;*/

use crate::{
    ir::IRKernel,
    kernel::{Kernel, Op},
    optimizer::Optimization,
    slab::Id,
    ZyxError,
};

#[derive(Debug)]
pub enum ErrorStatus {
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
    //pub cuda: cuda::CUDAConfig,
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

pub struct BufferId {
    memory_pool_id: u32,
    buffer_id: Id,
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
    fn deinitialize(self) -> Result<(), BackendError>;
    fn free_bytes(&self) -> usize;
    fn allocate(&mut self, bytes: usize) -> Result<Id, BackendError>;
    fn deallocate(&mut self, buffer: Id) -> Result<(), BackendError>;
    fn host_to_pool(&mut self, src: &[u8], dst: Id) -> Result<(), BackendError>;
    fn pool_to_host(&mut self, src: Id, dst: &mut [u8]) -> Result<(), BackendError>;
    fn pool_to_pool(
        &mut self,
        src: Id,
        dst_pool: &mut dyn MemoryPool,
        dst: Id,
    ) -> Result<(), BackendError>;
}

pub trait Device: Send {
    fn deinitialize(self) -> Result<(), BackendError>;
    fn info(&self) -> &DeviceInfo;
    fn memory_pool_id(&self) -> u32;
    fn compile(&self, kernel: &IRKernel, debug_asm: bool) -> Result<Id, BackendError>;
    // Returns if this is the first time running the kernel
    fn launch(
        &mut self,
        program_id: Id,
        memory_pool: &mut dyn MemoryPool,
        args: &[Id],
        // If sync is empty, kernel will be immediatelly synchronized
        sync: &BTreeSet<Id>,
    ) -> Result<(), BackendError>;
    fn release(&self, program_id: Id) -> Result<(), BackendError>;
}

pub fn initialize_backends(
    device_config: &DeviceConfig,
    memory_pools: &mut Vec<Box<dyn MemoryPool>>,
    devices: &mut Vec<Box<dyn Device>>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    //let _ = cuda::initialize_device(&device_config.cuda, memory_pools, devices, debug_dev);

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

/*impl MemoryPool {
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
                for key in events.keys() {
                    if key.contains(&buffer_id) {
                        // Useless clone because of borrowck ...
                        let event = events.remove(&key.clone()).unwrap();
                        event.finish()?;
                        break;
                    }
                }
                memory_pool.pool_to_host(&buffers[buffer_id], slice)?;
            }
            MemoryPool::HIP { memory_pool, buffers } => {
                memory_pool.pool_to_host(&buffers[buffer_id], slice)?;
            }
            MemoryPool::OpenCL { memory_pool, buffers, events } => {
                for key in events.keys() {
                    if key.contains(&buffer_id) {
                        // Useless clone because of borrowck ...
                        let event = events.remove(&key.clone()).unwrap();
                        event.finish()?;
                        break;
                    }
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
            MemoryPool::CUDA { events, .. } => {
                for key in events.keys() {
                    if key.contains(&sbid) {
                        // Useless clone because of borrowck ...
                        let event = events.remove(&key.clone()).unwrap();
                        event.finish()?;
                        break;
                    }
                }
            },
            MemoryPool::HIP { .. } => todo!(),
            MemoryPool::OpenCL { events, .. } => {
                for key in events.keys() {
                    if key.contains(&sbid) {
                        // Useless clone because of borrowck ...
                        let event = events.remove(&key.clone()).unwrap();
                        event.finish()?;
                        break;
                    }
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
        output_buffers: BTreeSet<Id>,
    ) -> Result<(), ZyxError> {
        match self {
            Device::CUDA { kernels, queues, .. } => {
                let mut queue = queues.iter_mut().min_by_key(|queue| queue.load()).unwrap();
                if queue.load() > 20 {
                    queue = queues.iter_mut().max_by_key(|queue| queue.load()).unwrap();
                    queue.sync()?;
                }
                let MemoryPool::CUDA { buffers, events, .. } = memory_pool else { unreachable!() };
                let event = queue.launch(
                    kernels.get_mut(kernel).unwrap(),
                    buffers,
                    buffer_ids,
                    output_buffers.is_empty(),
                )?;
                events.insert(output_buffers, event);
            }
            Device::HIP { kernels, queues, .. } => {
                let mut queue = queues.iter_mut().min_by_key(|queue| queue.load()).unwrap();
                if queue.load() > 20 {
                    queue = queues.iter_mut().max_by_key(|queue| queue.load()).unwrap();
                    queue.sync()?;
                }
                let MemoryPool::HIP { buffers, .. } = memory_pool else { unreachable!() };
                queue.launch(kernels.get_mut(kernel).unwrap(), buffers, buffer_ids)?;
            }
            Device::OpenCL { kernels, queues, .. } => {
                let mut queue = queues.iter_mut().min_by_key(|queue| queue.load()).unwrap();
                if queue.load() > 20 {
                    queue.sync()?;
                    queue = queues.iter_mut().max_by_key(|queue| queue.load()).unwrap();
                }
                let MemoryPool::OpenCL { buffers, events, .. } = memory_pool else {
                    unreachable!()
                };
                let event = queue.launch(
                    kernels.get_mut(kernel).unwrap(),
                    buffers,
                    buffer_ids,
                    output_buffers.is_empty(),
                )?;
                events.insert(output_buffers, event);
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
        }
        Ok(())
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

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub enum MemoryPool {
    CUDA {
        memory_pool: CUDAMemoryPool,
        buffers: Slab<CUDABuffer>,
        // Buffers that depend on events
        events: BTreeMap<BTreeSet<Id>, CUDAEvent>,
    },
    HIP {
        memory_pool: HIPMemoryPool,
        buffers: Slab<HIPBuffer>,
    },
    OpenCL {
        memory_pool: OpenCLMemoryPool,
        buffers: Slab<OpenCLBuffer>,
        events: BTreeMap<BTreeSet<Id>, OpenCLEvent>,
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
}*/
