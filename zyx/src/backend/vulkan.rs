// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Vulkan backend using vulkano.

use std::sync::Arc;

use super::vulkan_vulkano::VulkanLibrary;
use super::vulkan_vulkano::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use super::vulkan_vulkano::{BufferUsage, Subbuffer};
use super::vulkan_vulkano::StandardCommandBufferAllocator;
use super::vulkan_vulkano::{AutoCommandBufferBuilder, CommandBufferUsage};
use super::vulkan_vulkano::StandardDescriptorSetAllocator;
use super::vulkan_vulkano::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
use super::vulkan_vulkano::{CopyDescriptorSet, DescriptorSet, WriteDescriptorSet};
use super::vulkan_vulkano::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
use super::vulkan_vulkano::{Instance, InstanceCreateInfo};
use super::vulkan_vulkano::{MemoryTypeFilter, StandardMemoryAllocator};
use super::vulkan_vulkano::{ComputePipeline, ComputePipelineCreateInfo};
use super::vulkan_vulkano::{PipelineLayout, PipelineLayoutCreateInfo};
use super::vulkan_vulkano::{PipelineBindPoint, PipelineShaderStageCreateInfo};
use super::vulkan_vulkano::ShaderStages;
use super::vulkan_vulkano::{ShaderModule, ShaderModuleCreateInfo};
use super::vulkan_vulkano::sync;
use super::vulkan_vulkano::GpuFuture;

use nanoserde::DeJson;

use crate::{
    error::{BackendError, ErrorStatus},
    kernel::Kernel,
    shape::Dim,
    slab::Slab,
};

use super::{DeviceInfo, DeviceProgramId, Event, MemoryPool, PoolBufferId, PoolId};

#[derive(DeJson, Debug, Default)]
#[nserde(default)]
pub struct VulkanConfig {
    device_ids: Option<Vec<i32>>,
}

#[derive(Debug)]
pub struct VulkanMemoryPool {
    free_bytes: usize,
    memory_allocator: Arc<StandardMemoryAllocator>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    buffers: Slab<PoolBufferId, (Subbuffer<[u8]>, usize)>,
}

pub(crate) struct VulkanEvent {
    fence: Option<Arc<std::sync::Mutex<Option<Box<dyn FnOnce() + Send + 'static>>>>>,
}

impl std::fmt::Debug for VulkanEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanEvent").finish_non_exhaustive()
    }
}

impl Clone for VulkanEvent {
    fn clone(&self) -> Self {
        Self { fence: self.fence.clone() }
    }
}

impl VulkanEvent {
    fn wait(&self) {
        if let Some(fence) = &self.fence {
            let mut guard = fence.lock().unwrap();
            if let Some(f) = guard.take() {
                f();
            }
        }
    }
}

#[derive(Debug)]
pub struct VulkanDevice {
    dev_info: DeviceInfo,
    memory_pool_id: PoolId,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    programs: Slab<DeviceProgramId, VulkanProgram>,
}

#[derive(Debug)]
pub(super) struct VulkanProgram {
    //name: String,
    gws: Vec<Dim>,
    lws: Vec<Dim>,
    pipeline: Arc<ComputePipeline>,
    pipeline_layout: Arc<PipelineLayout>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
}

#[allow(clippy::unnecessary_wraps)]
pub(super) fn initialize_device(
    config: &VulkanConfig,
    memory_pools: &mut Slab<super::PoolId, MemoryPool>,
    devices: &mut Slab<super::DeviceId, super::Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if let Some(device_ids) = &config.device_ids
        && device_ids.is_empty()
    {
        if debug_dev {
            println!("[vulkan] configured out");
        }
        return Ok(());
    }

    let lib = match VulkanLibrary::new() {
        Ok(lib) => lib,
        Err(e) => {
            if debug_dev {
                println!("[vulkan] {e}");
            }
            return Err(BackendError { status: ErrorStatus::Initialization, context: format!("[vulkan] {e}").into() });
        }
    };
    let instance = match Instance::new(lib, InstanceCreateInfo::default()) {
        Ok(inst) => inst,
        Err(e) => {
            if debug_dev {
                println!("[vulkan] instance: {e}");
            }
            return Err(BackendError { status: ErrorStatus::Initialization, context: format!("[vulkan] instance: {e}").into() });
        }
    };

    let physical_devices = match instance.enumerate_physical_devices() {
        Ok(devs) => devs,
        Err(e) => {
            return Err(BackendError { status: ErrorStatus::Initialization, context: format!("[vulkan] enumerate: {e}").into() });
        }
    };

    for phys_device in physical_devices {
        let name = phys_device.properties().device_name.to_string();
        let qfps = phys_device.queue_family_properties();
        // Use the first queue family that has compute support
        let qfi = qfps
            .iter()
            .enumerate()
            .position(|(_, q)| q.queue_flags.intersects(QueueFlags::COMPUTE));

        let Some(qfi) = qfi else {
            eprintln!("[vulkan] device {name}: no compute queue family found");
            continue;
        };

        let max_wg_count = phys_device.properties().max_compute_work_group_count;
        let max_wg_invocations = phys_device.properties().max_compute_work_group_invocations;
        let max_wg_size = phys_device.properties().max_compute_work_group_size;

        if debug_dev {
            println!("[vulkan] {name}");
        }

        let (device, mut queues) = Device::new(
            phys_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo { queue_family_index: qfi as u32, ..Default::default() }],
                ..Default::default()
            },
        )
        .map_err(|e| {
            eprintln!("[vulkan] device creation failed: {e}");
            BackendError { status: ErrorStatus::Initialization, context: format!("[vulkan] device: {e}").into() }
        })?;

        let queues: Vec<_> = queues.collect();

        let queue = queues
            .first()
            .cloned()
            .ok_or_else(|| {
                eprintln!("[vulkan] no queues for device {name} queue family {qfi}");
                BackendError { status: ErrorStatus::Initialization, context: "[vulkan] no queue".into() }
            })?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(device.clone(), Default::default()));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(device.clone(), Default::default()));

        let memory_pool = VulkanMemoryPool {
            free_bytes: 1024 * 1024 * 1024,
            memory_allocator: memory_allocator.clone(),
            device: device.clone(),
            queue: queue,
            command_buffer_allocator: command_buffer_allocator.clone(),
            buffers: Slab::new(),
        };

        let mem_pool_id = memory_pools.push(MemoryPool::Vulkan(memory_pool));

        let dev_info = DeviceInfo {
            compute: 1_000_000_000_000,
            max_global_work_dims: vec![Dim::from(max_wg_count[0]); max_wg_count.len()],
            max_local_threads: Dim::from(max_wg_invocations),
            max_local_work_dims: vec![Dim::from(max_wg_size[0]); max_wg_size.len()],
            ..Default::default()
        };

        let queue = queues
            .into_iter()
            .next()
            .ok_or_else(|| BackendError { status: ErrorStatus::Initialization, context: "[vulkan] no queues".into() })?;

        let vk_device = VulkanDevice {
            dev_info,
            memory_pool_id: mem_pool_id,
            device: device.clone(),
            queue: queue.clone(),
            command_buffer_allocator: command_buffer_allocator.clone(),
            descriptor_set_allocator,
            programs: Slab::new(),
        };

        devices.push(super::Device::Vulkan(vk_device));
        break;
    }

    Ok(())
}

impl VulkanMemoryPool {
    pub(super) fn free_bytes(&self) -> Dim {
        self.free_bytes as Dim
    }

    #[allow(clippy::unused_self)]
    pub const fn deinitialize(&mut self) {}

    pub(super) fn allocate(&mut self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
        // Round up to at least 4 bytes — bool elements are stored as u32 (4 bytes) in Vulkan
        let aligned = bytes.next_multiple_of(4);
        // Use host-visible memory so we can directly read/write from the host
        use super::vulkan_vulkano::MemoryPropertyFlags;
        let allocator = SubbufferAllocator::new(
            self.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
                memory_type_filter: MemoryTypeFilter {
                    required_flags: MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
                    preferred_flags: MemoryPropertyFlags::empty(),
                    not_preferred_flags: MemoryPropertyFlags::empty(),
                },
                ..Default::default()
            },
        );
        let buffer = allocator
            .allocate_slice::<u8>(aligned)
            .map_err(|e| BackendError { status: ErrorStatus::MemoryAllocation, context: format!("buffer alloc: {e}").into() })?;

        let id = self.buffers.push((buffer, bytes as usize));
        Ok((PoolBufferId::from(id), Event::Vulkan(VulkanEvent { fence: None })))
    }

    pub(super) fn deallocate(&mut self, buffer_id: PoolBufferId, _event_wait_list: Vec<Event>) {
        self.buffers.remove(buffer_id);
    }

    pub(super) fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: PoolBufferId,
        _event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let (buffer, _) = &self.buffers[dst];
        let mut content = buffer
            .write()
            .map_err(|e| BackendError { status: ErrorStatus::MemoryCopyH2P, context: format!("write: {e}").into() })?;
        let copy_len = content.len().min(src.len());
        content[..copy_len].copy_from_slice(&src[..copy_len]);
        Ok(Event::Vulkan(VulkanEvent { fence: None }))
    }

    pub(super) fn pool_to_host(
        &mut self,
        src: PoolBufferId,
        dst: &mut [u8],
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        for event in &event_wait_list {
            if let Event::Vulkan(event) = event {
                event.wait();
            }
        }
        let (buffer, _) = &self.buffers[src];
        let content = buffer
            .read()
            .map_err(|e| BackendError { status: ErrorStatus::MemoryCopyP2H, context: format!("read: {e}").into() })?;
        let copy_len = dst.len().min(content.len());
        dst[..copy_len].copy_from_slice(&content[..copy_len]);
        Ok(())
    }

    pub(super) fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        for event in &events {
            if let Event::Vulkan(event) = event {
                event.wait();
            }
        }
        Ok(())
    }

    pub(super) fn release_events(&mut self, _events: Vec<Event>) {}
}

impl VulkanDevice {
    pub const fn deinitialize(&mut self) {}

    pub(super) const fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    pub(super) const fn memory_pool_id(&self) -> PoolId {
        self.memory_pool_id
    }

    pub(super) const fn free_compute(&self) -> u128 {
        1_000_000_000_000
    }

    pub(super) fn release(&mut self, program_id: DeviceProgramId) {
        self.programs.remove(program_id);
    }

    pub(super) fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        let (spirv_words, gws, lws) = crate::backend::spirv::compile(kernel, debug_asm)
            .map_err(|e| BackendError { status: ErrorStatus::KernelCompilation, context: format!("SPIR-V: {e}").into() })?;

        let shader_module = unsafe {
            ShaderModule::new(self.device.clone(), ShaderModuleCreateInfo::new(&spirv_words)).map_err(|e| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("shader module: {e}").into(),
            })?
        };

        let entry_point = shader_module
            .single_entry_point()
            .ok_or_else(|| BackendError { status: ErrorStatus::KernelCompilation, context: "no entry point".into() })?;

        // Scan kernel for read-only flags
        let mut arg_ro_flags = Vec::new();
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            if let crate::kernel::Op::Define { ro, scope, .. } = kernel.at(op_id) {
                if *scope == crate::kernel::Scope::Global {
                    arg_ro_flags.push(*ro);
                }
            }
            op_id = kernel.next_op(op_id);
        }

        // Create descriptor set layout matching SPIR-V bindings
        let ds_bindings: std::collections::BTreeMap<u32, DescriptorSetLayoutBinding> = arg_ro_flags
            .iter()
            .enumerate()
            .map(|(i, _ro)| {
                let mut binding = DescriptorSetLayoutBinding::descriptor_type(DescriptorType::StorageBuffer);
                binding.stages = ShaderStages::COMPUTE;
                (i as u32, binding)
            })
            .collect();

        let ds_layout = DescriptorSetLayout::new(
            self.device.clone(),
            DescriptorSetLayoutCreateInfo { bindings: ds_bindings, ..Default::default() },
        )
        .map_err(|e| BackendError { status: ErrorStatus::KernelCompilation, context: format!("ds layout: {e}").into() })?;

        let pipeline_layout = PipelineLayout::new(
            self.device.clone(),
            PipelineLayoutCreateInfo { set_layouts: vec![ds_layout.clone()], ..Default::default() },
        )
        .map_err(|e| BackendError { status: ErrorStatus::KernelCompilation, context: format!("pipeline layout: {e}").into() })?;

        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout.clone()),
        )
        .map_err(|e| BackendError { status: ErrorStatus::KernelCompilation, context: format!("compute pipeline: {e}").into() })?;

        let program = VulkanProgram { gws, lws, pipeline, pipeline_layout, descriptor_set_layout: ds_layout };

        let id = self.programs.push(program);
        Ok(id)
    }

    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn launch(
        &mut self,
        program_id: DeviceProgramId,
        memory_pool: &mut VulkanMemoryPool,
        args: &[PoolBufferId],
        _event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let program = &self.programs[program_id];

        let write_descriptors: Vec<WriteDescriptorSet> = args
            .iter()
            .enumerate()
            .map(|(i, arg_id)| {
                let (buffer, _) = &memory_pool.buffers[*arg_id];
                WriteDescriptorSet::buffer(i as u32, buffer.clone())
            })
            .collect();

        let descriptor_set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            program.descriptor_set_layout.clone(),
            write_descriptors,
            std::iter::empty::<CopyDescriptorSet>(),
        )
        .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("descriptor set: {e}").into() })?;

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("cmd buffer: {e}").into() })?;

        builder
            .bind_pipeline_compute(program.pipeline.clone())
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("bind pipeline: {e}").into() })?;

        builder
            .bind_descriptor_sets(PipelineBindPoint::Compute, program.pipeline_layout.clone(), 0, descriptor_set)
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("bind ds: {e}").into() })?;

        let group_count_x = program.gws[0];
        let group_count_y = program.gws[1];
        let group_count_z = program.gws[2];

        unsafe {
            builder
                .dispatch([group_count_x as u32, group_count_y as u32, group_count_z as u32])
                .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("dispatch: {e}").into() })?;
        }

        let command_buffer = builder
            .build()
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("build: {e}").into() })?;

        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .map_err(|e| BackendError { status: ErrorStatus::KernelLaunch, context: format!("execute: {e}").into() })?
            .then_signal_fence();

        let wait = Box::new(move || {
            future.wait(None).unwrap();
        });

        Ok(Event::Vulkan(VulkanEvent {
            fence: Some(Arc::new(std::sync::Mutex::new(Some(wait)))),
        }))
    }
}
