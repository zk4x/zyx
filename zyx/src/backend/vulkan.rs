// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Vulkan backend using vulkano.

use std::sync::Arc;

use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
use vulkano::buffer::{BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::{DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType};
use vulkano::shader::ShaderStages;
use vulkano::descriptor_set::{CopyDescriptorSet, DescriptorSet, WriteDescriptorSet};
use vulkano::device::{Device, DeviceCreateInfo, QueueCreateInfo, QueueFlags};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::compute::{ComputePipeline, ComputePipelineCreateInfo};
use vulkano::pipeline::layout::{PipelineLayout, PipelineLayoutCreateInfo};
use vulkano::pipeline::{PipelineBindPoint, PipelineShaderStageCreateInfo};
use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
use vulkano::sync::{self, GpuFuture};
use vulkano::VulkanLibrary;

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
#[allow(dead_code)]
pub struct VulkanConfig {
    device_ids: Option<Vec<i32>>,
}

#[derive(Debug)]
pub struct VulkanMemoryPool {
    free_bytes: usize,
    memory_allocator: Arc<StandardMemoryAllocator>,
    buffers: Slab<PoolBufferId, (Subbuffer<[u8]>, usize)>,
}

#[derive(Clone, Debug)]
pub(crate) struct VulkanEvent;

#[derive(Debug)]
#[allow(dead_code)]
pub struct VulkanDevice {
    dev_info: DeviceInfo,
    memory_pool_id: PoolId,
    device: Arc<Device>,
    queue: Arc<vulkano::device::Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    programs: Slab<DeviceProgramId, VulkanProgram>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct VulkanProgram {
    name: String,
    gws: [u64; 3],
    lws: [u64; 3],
    pipeline: Arc<ComputePipeline>,
    pipeline_layout: Arc<PipelineLayout>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
}

#[allow(clippy::unnecessary_wraps)]
#[allow(unused_variables)]
pub(super) fn initialize_device(
    config: &VulkanConfig,
    memory_pools: &mut Slab<super::PoolId, MemoryPool>,
    devices: &mut Slab<super::DeviceId, super::Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    let lib = match VulkanLibrary::new() {
        Ok(lib) => lib,
        Err(e) => {
            if debug_dev {
                println!("[vulkan] {e}");
            }
            return Err(BackendError {
                status: ErrorStatus::Initialization,
                context: format!("[vulkan] {e}").into(),
            });
        }
    };
    let instance = match Instance::new(lib, InstanceCreateInfo::default()) {
        Ok(inst) => inst,
        Err(e) => {
            if debug_dev {
                println!("[vulkan] instance: {e}");
            }
            return Err(BackendError {
                status: ErrorStatus::Initialization,
                context: format!("[vulkan] instance: {e}").into(),
            });
        }
    };

    let physical_devices = match instance.enumerate_physical_devices() {
        Ok(devs) => devs,
        Err(e) => {
            return Err(BackendError {
                status: ErrorStatus::Initialization,
                context: format!("[vulkan] enumerate: {e}").into(),
            });
        }
    };

    for phys_device in physical_devices {
        let qfps = phys_device.queue_family_properties();
        let queue_family_index = qfps
            .iter()
            .enumerate()
            .position(|(_, q)| q.queue_flags.intersects(QueueFlags::COMPUTE));

        let Some(qfi) = queue_family_index else {
            continue;
        };

        let name = phys_device.properties().device_name.to_string();
        let max_wg_count = phys_device.properties().max_compute_work_group_count;
        let max_wg_invocations = phys_device.properties().max_compute_work_group_invocations;
        let max_wg_size = phys_device.properties().max_compute_work_group_size;

        if debug_dev {
            println!("[vulkan] {name}");
        }

        let (device, queues) = Device::new(
            phys_device,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index: qfi as u32,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )
        .map_err(|e| BackendError {
            status: ErrorStatus::Initialization,
            context: format!("[vulkan] device: {e}").into(),
        })?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let memory_pool = VulkanMemoryPool {
            free_bytes: 1024 * 1024 * 1024,
            memory_allocator: memory_allocator.clone(),
            buffers: Slab::new(),
        };

        let mem_pool_id = memory_pools.push(MemoryPool::Vulkan(memory_pool));

        let dev_info = DeviceInfo {
            compute: 1_000_000_000_000,
            max_global_work_dims: vec![
                Dim::from(max_wg_count[0]);
                max_wg_count.len()
            ],
            max_local_threads: Dim::from(max_wg_invocations),
            max_local_work_dims: vec![
                Dim::from(max_wg_size[0]);
                max_wg_size.len()
            ],
            ..Default::default()
        };

        let queue = queues.into_iter().next().ok_or_else(|| BackendError {
            status: ErrorStatus::Initialization,
            context: "[vulkan] no queues".into(),
        })?;

        let vk_device = VulkanDevice {
            dev_info,
            memory_pool_id: mem_pool_id,
            device: device.clone(),
            queue: queue.clone(),
            command_buffer_allocator: command_buffer_allocator.clone(),
            descriptor_set_allocator,
            memory_allocator,
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
        let allocator = SubbufferAllocator::new(
            self.memory_allocator.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::STORAGE_BUFFER | BufferUsage::TRANSFER_DST | BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
        );
        let buffer = allocator
            .allocate_slice::<u8>(aligned)
            .map_err(|e| BackendError {
                status: ErrorStatus::MemoryAllocation,
                context: format!("buffer alloc: {e}").into(),
            })?;

        let id = self.buffers.push((buffer, bytes as usize));
        Ok((PoolBufferId::from(id), Event::Vulkan(VulkanEvent)))
    }

    pub(super) fn deallocate(
        &mut self,
        buffer_id: PoolBufferId,
        _event_wait_list: Vec<Event>,
    ) {
        self.buffers.remove(buffer_id);
    }

    pub(super) fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: PoolBufferId,
        _event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let (buffer, _) = &self.buffers[dst];
        let mut content = buffer.write().map_err(|e| BackendError {
            status: ErrorStatus::MemoryCopyH2P,
            context: format!("write: {e}").into(),
        })?;
        let copy_len = content.len().min(src.len());
        content[..copy_len].copy_from_slice(&src[..copy_len]);
        Ok(Event::Vulkan(VulkanEvent))
    }

    pub(super) fn pool_to_host(
        &mut self,
        src: PoolBufferId,
        dst: &mut [u8],
        _event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        let (buffer, _) = &self.buffers[src];
        let content = buffer.read().map_err(|e| BackendError {
            status: ErrorStatus::MemoryCopyP2H,
            context: format!("read: {e}").into(),
        })?;
        let copy_len = dst.len().min(content.len());
        dst[..copy_len].copy_from_slice(&content[..copy_len]);
        Ok(())
    }

    #[allow(unused)]
    pub(super) fn pool_to_pool(
        &mut self,
        _src: PoolBufferId,
        _dst: PoolBufferId,
    ) -> Result<(), BackendError> {
        todo!()
    }

    pub(super) fn sync_events(&mut self, _events: Vec<Event>) -> Result<(), BackendError> {
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

    pub(super) fn compile(
        &mut self,
        kernel: &Kernel,
        debug_asm: bool,
    ) -> Result<DeviceProgramId, BackendError> {
        let spirv_words =
            crate::backend::spirv::compile(kernel, debug_asm).map_err(|e| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("SPIR-V: {e}").into(),
            })?;

        let shader_module = unsafe {
            ShaderModule::new(
                self.device.clone(),
                ShaderModuleCreateInfo::new(&spirv_words),
            )
            .map_err(|e| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("shader module: {e}").into(),
            })?
        };

        let mut gws: [u64; 3] = [1, 1, 1];
        let mut lws: [u64; 3] = [1, 1, 1];
        let mut op_id = kernel.head;
        while !op_id.is_null() {
            if let crate::kernel::Op::Index { len, scope, axis } = kernel.at(op_id) {
                match scope {
                    crate::kernel::Scope::Global if *axis < 3 => {
                        gws[*axis as usize] = gws[*axis as usize].max(*len)
                    }
                    crate::kernel::Scope::Local if *axis < 3 => {
                        lws[*axis as usize] = lws[*axis as usize].max(*len)
                    }
                    _ => {}
                }
            }
            op_id = kernel.next_op(op_id);
        }

        let name = format!(
            "k_{}__{}",
            gws.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("_"),
            lws.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("_"),
        );

        let entry_point = shader_module.single_entry_point().ok_or_else(|| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: "no entry point".into(),
        })?;

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
            DescriptorSetLayoutCreateInfo {
                bindings: ds_bindings,
                ..Default::default()
            },
        )
        .map_err(|e| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("ds layout: {e}").into(),
        })?;

        let pipeline_layout = PipelineLayout::new(
            self.device.clone(),
            PipelineLayoutCreateInfo {
                set_layouts: vec![ds_layout.clone()],
                ..Default::default()
            },
        )
        .map_err(|e| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("pipeline layout: {e}").into(),
        })?;

        let stage = PipelineShaderStageCreateInfo::new(entry_point);
        let pipeline = ComputePipeline::new(
            self.device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, pipeline_layout.clone()),
        )
        .map_err(|e| BackendError {
            status: ErrorStatus::KernelCompilation,
            context: format!("compute pipeline: {e}").into(),
        })?;

        let program = VulkanProgram {
            name,
            gws,
            lws,
            pipeline,
            pipeline_layout,
            descriptor_set_layout: ds_layout,
        };

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
        .map_err(|e| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: format!("descriptor set: {e}").into(),
        })?;

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .map_err(|e| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: format!("cmd buffer: {e}").into(),
        })?;

        builder
            .bind_pipeline_compute(program.pipeline.clone())
            .map_err(|e| BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("bind pipeline: {e}").into(),
            })?;

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                program.pipeline_layout.clone(),
                0,
                descriptor_set,
            )
            .map_err(|e| BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("bind ds: {e}").into(),
            })?;

        let group_count_x = (program.gws[0] + program.lws[0] - 1) / program.lws[0];
        let group_count_y = (program.gws[1] + program.lws[1] - 1) / program.lws[1];
        let group_count_z = (program.gws[2] + program.lws[2] - 1) / program.lws[2];

        unsafe {
            builder
                .dispatch([group_count_x as u32, group_count_y as u32, group_count_z as u32])
                .map_err(|e| BackendError {
            status: ErrorStatus::KernelLaunch,
                    context: format!("dispatch: {e}").into(),
                })?;
        }

        let command_buffer = builder.build().map_err(|e| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: format!("build: {e}").into(),
        })?;

        let future = sync::now(self.device.clone())
            .then_execute(self.queue.clone(), command_buffer)
            .map_err(|e| BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("execute: {e}").into(),
            })?
            .then_signal_fence();

        future.wait(None).map_err(|e| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: format!("fence: {e}").into(),
        })?;

        Ok(Event::Vulkan(VulkanEvent))
    }
}
