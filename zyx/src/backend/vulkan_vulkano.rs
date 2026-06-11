// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

// Shim module: re-exports from vulkano crate.
// Each function/type will be replaced with ash-based implementation one by one.

pub use vulkano::VulkanLibrary;
pub use vulkano::buffer::allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo};
pub use vulkano::buffer::{BufferUsage, Subbuffer};
pub use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
pub use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage};
pub use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
pub use vulkano::descriptor_set::layout::{
    DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo, DescriptorType,
};
pub use vulkano::descriptor_set::{CopyDescriptorSet, DescriptorSet, WriteDescriptorSet};
pub use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags};
pub use vulkano::memory::MemoryPropertyFlags;
pub use vulkano::instance::{Instance, InstanceCreateInfo};
pub use vulkano::memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator};
pub use vulkano::pipeline::compute::{ComputePipeline, ComputePipelineCreateInfo};
pub use vulkano::pipeline::layout::{PipelineLayout, PipelineLayoutCreateInfo};
pub use vulkano::pipeline::{PipelineBindPoint, PipelineShaderStageCreateInfo};
pub use vulkano::shader::ShaderStages;
pub use vulkano::shader::{ShaderModule, ShaderModuleCreateInfo};
pub use vulkano::sync::{self, GpuFuture};
