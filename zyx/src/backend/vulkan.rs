//! Vulkan backend

#![allow(unused)]

use std::num::NonZero;
use std::ptr;
use std::{ffi::CString, sync::Arc};

use vulkano::buffer::{AllocateBufferError, Buffer, BufferCreateInfo, BufferUsage};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::device::{
    DeviceCreateInfo, DeviceExtensions, QueueCreateFlags, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::{
    AllocationCreateInfo, DeviceLayout, MemoryTypeFilter, StandardMemoryAllocator,
};
use vulkano::memory::DeviceAlignment;
use vulkano::{LoadingError, NonExhaustive, Validated, VulkanLibrary};

use crate::{
    index_map::{Id, IndexMap},
    ir::IRKernel,
};

use super::DeviceInfo;

#[derive(serde::Deserialize, Debug, Default)]
pub struct VulkanConfig {
    device_ids: Option<Vec<i32>>,
}

#[derive(Debug)]
pub struct VulkanError {
    info: String,
    status: vulkano::VulkanError
}

#[derive(Debug)]
pub(super) struct VulkanMemoryPool {
    free_bytes: usize,
    memory_allocator: Arc<StandardMemoryAllocator>,
    //device: Arc<vulkano::device::Device>,
    //queue: Arc<vulkano::device::Queue>,
}

#[derive(Debug)]
pub(super) struct VulkanBuffer {
    buffer: Arc<Buffer>,
}

#[derive(Debug)]
pub(super) struct VulkanDevice {
    dev_info: DeviceInfo,
    memory_pool_id: u32,
    device: Arc<vulkano::device::Device>,
}

#[derive(Debug)]
pub(super) struct VulkanProgram {
    name: String,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
    read_only_args: Vec<bool>,
    shader: (),
}

#[derive(Debug)]
pub(super) struct VulkanQueue {
    load: usize,
    queue: Arc<vulkano::device::Queue>,
}

type VulkanQueuePool = Vec<(VulkanDevice, Vec<VulkanQueue>)>;

#[allow(clippy::unnecessary_wraps)]
pub(super) fn initialize_devices(
    config: &VulkanConfig,
    debug_dev: bool,
) -> Result<(Vec<VulkanMemoryPool>, VulkanQueuePool), VulkanError> {
    let lib = VulkanLibrary::new()?;
    let api_version = lib.api_version();
    let instance = Instance::new(lib, InstanceCreateInfo::default())?;

    let mut memory_pools = Vec::new();
    let mut devices = Vec::new();

    let physical_devices = instance.enumerate_physical_devices()?;

    let device_ids: Vec<usize> = (0..physical_devices.len())
        .filter(|id| {
            config
                .device_ids
                .as_ref()
                .map_or(true, |ids| ids.contains(&(*id as i32)))
        })
        .collect();
    if debug_dev && !device_ids.is_empty() {
        println!(
            "Using Vulkan backend API version {} on devices:",
            api_version
        );
    }

    // TODO apply this as filter on physical_devices
    if device_ids.is_empty() {
        return Err(VulkanError { info: "".into(), status: vulkano::VulkanError::Unknown })
    }

    for (phys_device, queue_family_index) in
        physical_devices.filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| q.queue_flags.intersects(QueueFlags::COMPUTE))
                .map(|i| (p, i as u32))
        })
    {
        if debug_dev {
            println!("{}", phys_device.properties().device_name);
        }
        let (device, queues) = vulkano::device::Device::new(
            phys_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: DeviceExtensions::empty(),
                enabled_features: Default::default(),
                physical_devices: Default::default(),
                private_data_slot_request_count: Default::default(),
                ..Default::default()
            },
        )
        .unwrap();
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let memory_pool = VulkanMemoryPool {
            free_bytes: 1024 * 1024 * 1024,
            memory_allocator,
        };
        let device = VulkanDevice {
            dev_info: DeviceInfo::default(),
            memory_pool_id: 0,
            device,
        };
        let queues = queues.map(|queue| VulkanQueue { load: 0, queue }).collect();
        devices.push((device, queues));
        memory_pools.push(memory_pool);
    }

    Ok((memory_pools, devices))
}

impl VulkanMemoryPool {
    pub(super) const fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn deinitialize(self) -> Result<(), VulkanError> {
        Ok(())
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn allocate(&mut self, bytes: usize) -> Result<VulkanBuffer, VulkanError> {
        let buffer = vulkano::buffer::Buffer::new(
            self.memory_allocator.clone(),
            BufferCreateInfo::default(),
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            DeviceLayout::new(NonZero::new(bytes as u64).unwrap(), DeviceAlignment::default()).unwrap(),
        )?;
        /*let data_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                ..Default::default()
            },
            // Iterator that produces the data.
            0..65536u32,
        )
        .unwrap();*/
        Ok(VulkanBuffer {
            buffer,
        })
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn deallocate(&mut self, buffer: VulkanBuffer) -> Result<(), VulkanError> {
        todo!()
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: &VulkanBuffer,
    ) -> Result<(), VulkanError> {
        todo!()
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn pool_to_host(
        &mut self,
        src: &VulkanBuffer,
        dst: &mut [u8],
    ) -> Result<(), VulkanError> {
        todo!()
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn pool_to_pool(
        &mut self,
        src: &VulkanBuffer,
        dst: &VulkanBuffer,
    ) -> Result<(), VulkanError> {
        todo!()
    }
}

impl VulkanDevice {
    pub(super) const fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    // Memory pool id out of VulkanMemoryPools
    pub(super) const fn memory_pool_id(&self) -> u32 {
        self.memory_pool_id
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn release_program(&self, program: VulkanProgram) -> Result<(), VulkanError> {
        todo!()
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn release_queue(&self, queue: VulkanQueue) -> Result<(), VulkanError> {
        todo!()
    }

    pub(super) fn deinitialize(self) -> Result<(), VulkanError> {
        todo!()
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn compile(
        &mut self,
        kernel: &IRKernel,
        debug_asm: bool,
    ) -> Result<VulkanProgram, VulkanError> {
        todo!()
    }
}

impl VulkanQueue {
    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn launch(
        &mut self,
        program: &mut VulkanProgram,
        buffers: &mut IndexMap<VulkanBuffer>,
        args: &[Id],
    ) -> Result<(), VulkanError> {
        todo!()
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn sync(&mut self) -> Result<(), VulkanError> {
        todo!()
    }

    pub(super) const fn load(&self) -> usize {
        self.load
    }
}

impl From<LoadingError> for VulkanError {
    fn from(value: LoadingError) -> Self {
        match value {
            LoadingError::VulkanError(status) => Self {
                info: "LoadingError".into(),
                status
            },
            LoadingError::LibraryLoadFailure(error) => Self {
                info: format!("LoadingError {error}"),
                status: vulkano::VulkanError::Unknown,
            },
        }
    }
}

impl From<Validated<vulkano::VulkanError>> for VulkanError {
    fn from(value: Validated<vulkano::VulkanError>) -> Self {
        match value {
            Validated::Error(status) => Self {
                info: "Vulkan error".into(),
                status
            },
            Validated::ValidationError(err) => Self {
                info: format!("Validation error {err}"),
                status: vulkano::VulkanError::Unknown
            },
        }
    }
}

impl From<vulkano::VulkanError> for VulkanError {
    fn from(status: vulkano::VulkanError) -> Self {
        Self {
            info: "".into(),
            status,
        }
    }
}

impl From<vulkano::Validated<AllocateBufferError>> for VulkanError {
    fn from(value: vulkano::Validated<AllocateBufferError>) -> Self {
        match value {
            Validated::Error(value) => Self {
                info: "Buffer allocation failure".into(),
                status: vulkano::VulkanError::OutOfDeviceMemory
            },
            Validated::ValidationError(err) => Self {
                info: format!("Buffer allocation failure {err}"),
                status: vulkano::VulkanError::Unknown,
            },
        }
    }
}
