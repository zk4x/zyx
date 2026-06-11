// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Vulkan backend using ash.

use std::cell::Cell;
use std::ffi::CStr;
use std::ffi::CString;
use std::sync::Arc;

use ash::vk;
use nanoserde::DeJson;

use crate::{
    error::{BackendError, ErrorStatus},
    kernel::Kernel,
    shape::Dim,
    slab::Slab,
};

use super::{DeviceInfo, DeviceProgramId, Event, MemoryPool, PoolBufferId, PoolId};

// Mapped pointer wrapper — Send+Sync so the slab compiles inside Runtime.
#[derive(Clone, Copy)]
struct Mapped(*mut u8);
unsafe impl Send for Mapped {}
unsafe impl Sync for Mapped {}

// ── Config ─────────────────────────────────────────────────────────────────

#[derive(DeJson, Debug, Default)]
#[nserde(default)]
pub struct VulkanConfig {
    device_ids: Option<Vec<i32>>,
}

// ── Core: Arc'd Vulkan state ───────────────────────────────────────────────

struct Core {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    gpu: vk::PhysicalDevice,
    queue: vk::Queue,
    qfi: u32,
    cmd_pool: vk::CommandPool,
    desc_pool: vk::DescriptorPool,
}

impl Drop for Core {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_descriptor_pool(self.desc_pool, None);
            self.device.destroy_command_pool(self.cmd_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn find_mem_type(
    inst: &ash::Instance,
    gpu: vk::PhysicalDevice,
    type_filter: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let mem = unsafe { inst.get_physical_device_memory_properties(gpu) };
    (0..mem.memory_type_count).find(|&i| {
        (type_filter & (1 << i)) != 0
            && mem.memory_types[i as usize].property_flags & required == required
    })
}

// ── Memory Pool ─────────────────────────────────────────────────────────────

pub struct VulkanMemoryPool {
    free_bytes: usize,
    core: Arc<Core>,
    // (buffer, memory, mapped_ptr, requested_bytes)
    buffers: Slab<PoolBufferId, (vk::Buffer, vk::DeviceMemory, Mapped, usize)>,
}

impl std::fmt::Debug for VulkanMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanMemoryPool")
            .field("free_bytes", &self.free_bytes)
            .field("n_buffers", &self.buffers.len())
            .finish()
    }
}

impl Drop for VulkanMemoryPool {
    fn drop(&mut self) {
        let dev = &self.core.device;
        for &(buf, mem, Mapped(ptr), _) in self.buffers.values() {
            if !ptr.is_null() {
                unsafe { dev.unmap_memory(mem) };
            }
            unsafe {
                dev.destroy_buffer(buf, None);
                dev.free_memory(mem, None);
            }
        }
    }
}

impl VulkanMemoryPool {
    pub(super) fn free_bytes(&self) -> Dim {
        self.free_bytes as Dim
    }

    pub(super) const fn deinitialize(&mut self) {}

    pub(super) fn allocate(
        &mut self,
        bytes: Dim,
    ) -> Result<(PoolBufferId, Event), BackendError> {
        let size = bytes.next_multiple_of(4) as u64;
        let (buf, mem, ptr) = self.create_buffer(size)?;
        let id = self.buffers.push((buf, mem, Mapped(ptr), bytes as usize));
        Ok((PoolBufferId::from(id), Event::Vulkan(VulkanEvent::none())))
    }

    fn create_buffer(
        &self,
        size: u64,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), BackendError> {
        let dev = &self.core.device;
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC;
        let ci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buf = unsafe { dev.create_buffer(&ci, None) }.map_err(|e| BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: format!("create buffer: {e}").into(),
        })?;
        let req = unsafe { dev.get_buffer_memory_requirements(buf) };
        let mem_type = find_mem_type(
            &self.core.instance,
            self.core.gpu,
            req.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or_else(|| BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: "no suitable memory type".into(),
        })?;
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let mem = unsafe { dev.allocate_memory(&alloc, None) }.map_err(|e| BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: format!("alloc memory: {e}").into(),
        })?;
        unsafe { dev.bind_buffer_memory(buf, mem, 0) }.map_err(|e| BackendError {
            status: ErrorStatus::MemoryAllocation,
            context: format!("bind memory: {e}").into(),
        })?;
        let ptr = unsafe { dev.map_memory(mem, 0, size, vk::MemoryMapFlags::empty()) }
            .map_err(|e| BackendError {
                status: ErrorStatus::MemoryAllocation,
                context: format!("map memory: {e}").into(),
            })?;
        Ok((buf, mem, ptr.cast::<u8>()))
    }

    pub(super) fn deallocate(
        &mut self,
        buffer_id: PoolBufferId,
        event_wait_list: Vec<Event>,
    ) {
        for event in &event_wait_list {
            if let Event::Vulkan(ev) = event {
                ev.wait();
            }
        }
        let (buf, mem, Mapped(ptr), _) =
            unsafe { self.buffers.remove_and_return(buffer_id) };
        let dev = &self.core.device;
        if !ptr.is_null() {
            unsafe { dev.unmap_memory(mem) };
        }
        unsafe {
            dev.destroy_buffer(buf, None);
            dev.free_memory(mem, None);
        }
    }

    pub(super) fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: PoolBufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        for event in &event_wait_list {
            if let Event::Vulkan(ev) = event {
                ev.wait();
            }
        }
        let &(_, _, Mapped(ptr), _) = &self.buffers[dst];
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, src.len());
        }
        Ok(Event::Vulkan(VulkanEvent::none()))
    }

    pub(super) fn pool_to_host(
        &mut self,
        src: PoolBufferId,
        dst: &mut [u8],
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        for event in &event_wait_list {
            if let Event::Vulkan(ev) = event {
                ev.wait();
            }
        }
        let &(_, _, Mapped(ptr), _) = &self.buffers[src];
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, dst.as_mut_ptr(), dst.len());
        }
        Ok(())
    }

    pub(super) fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        for event in &events {
            if let Event::Vulkan(ev) = event {
                ev.wait();
            }
        }
        Ok(())
    }

    pub(super) fn release_events(&mut self, _events: Vec<Event>) {}
}

// ── Event ───────────────────────────────────────────────────────────────────

/// Wraps raw handles with take-once semantics.
/// `wait()` waits on the fence and frees everything. Subsequent calls are no-ops.
/// `Drop` also calls `wait()` to prevent resource leaks / GPU memory corruption.
pub(crate) struct VulkanEvent {
    core: Option<Arc<Core>>,
    fence: Cell<Option<vk::Fence>>,
    cmd: Cell<Option<vk::CommandBuffer>>,
    desc_set: Cell<Option<vk::DescriptorSet>>,
}

impl std::fmt::Debug for VulkanEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanEvent").finish_non_exhaustive()
    }
}

impl Clone for VulkanEvent {
    fn clone(&self) -> Self {
        Self { core: None, fence: Cell::new(None), cmd: Cell::new(None), desc_set: Cell::new(None) }
    }
}

impl Drop for VulkanEvent {
    fn drop(&mut self) {
        self.wait();
    }
}

impl VulkanEvent {
    const fn none() -> Self {
        Self { core: None, fence: Cell::new(None), cmd: Cell::new(None), desc_set: Cell::new(None) }
    }

    fn create(
        core: Arc<Core>,
        fence: vk::Fence,
        cmd: vk::CommandBuffer,
        desc_set: vk::DescriptorSet,
    ) -> Self {
        Self {
            core: Some(core),
            fence: Cell::new(Some(fence)),
            cmd: Cell::new(Some(cmd)),
            desc_set: Cell::new(Some(desc_set)),
        }
    }

    fn wait(&self) {
        let Some(ref core) = self.core else { return };
        if let Some(fence) = self.fence.take() {
            unsafe {
                let _ = core.device.wait_for_fences(&[fence], true, std::u64::MAX);
                core.device.destroy_fence(fence, None);
            }
        }
        if let (Some(cmd), Some(desc_set)) = (self.cmd.take(), self.desc_set.take()) {
            unsafe {
                let _ = core.device.free_command_buffers(core.cmd_pool, &[cmd]);
                let _ = core.device.free_descriptor_sets(core.desc_pool, &[desc_set]);
            }
        }
    }
}

// ── Program ─────────────────────────────────────────────────────────────────

pub(super) struct VulkanProgram {
    gws: Vec<Dim>,
    lws: Vec<Dim>,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    desc_layout: vk::DescriptorSetLayout,
}

// ── Device ──────────────────────────────────────────────────────────────────

pub struct VulkanDevice {
    dev_info: DeviceInfo,
    memory_pool_id: PoolId,
    core: Arc<Core>,
    programs: Slab<DeviceProgramId, VulkanProgram>,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("dev_info", &self.dev_info)
            .field("memory_pool_id", &self.memory_pool_id)
            .field("n_programs", &self.programs.len())
            .finish()
    }
}

impl VulkanDevice {
    pub(super) const fn deinitialize(&mut self) {}

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
        let prog = unsafe { self.programs.remove_and_return(program_id) };
        let dev = &self.core.device;
        unsafe {
            dev.destroy_pipeline(prog.pipeline, None);
            dev.destroy_pipeline_layout(prog.pipeline_layout, None);
            dev.destroy_descriptor_set_layout(prog.desc_layout, None);
        }
    }

    pub(super) fn compile(
        &mut self,
        kernel: &Kernel,
        debug_asm: bool,
    ) -> Result<DeviceProgramId, BackendError> {
        let (spirv, gws, lws) = crate::backend::spirv::compile(kernel, debug_asm)
            .map_err(|e| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("SPIR-V: {e}").into(),
            })?;
        let dev = &self.core.device;

        let shader_ci = vk::ShaderModuleCreateInfo::default().code(&spirv);
        let shader = unsafe { dev.create_shader_module(&shader_ci, None) }.map_err(|e| {
            BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("shader: {e}").into(),
            }
        })?;

        // Count kernel args for descriptor bindings
        let n_args = {
            let mut n = 0usize;
            let mut op = kernel.head;
            while !op.is_null() {
                if let crate::kernel::Op::Define { ro: _, scope, .. } = kernel.at(op) {
                    if *scope == crate::kernel::Scope::Global {
                        n += 1;
                    }
                }
                op = kernel.next_op(op);
            }
            n
        };

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_args as u32)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let desc_layout =
            unsafe { dev.create_descriptor_set_layout(&layout_ci, None) }.map_err(|e| {
                BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("ds layout: {e}").into(),
                }
            })?;

        let desc_layouts = [desc_layout];
        let pl_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&desc_layouts);
        let pipeline_layout =
            unsafe { dev.create_pipeline_layout(&pl_ci, None) }.map_err(|e| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("pipeline layout: {e}").into(),
            })?;

        // Entry point name must match spirv.rs format: "k_{gws}__{lws}"
        let ep_name = format!(
            "k_{}__{}",
            gws.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("_"),
            lws.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("_"),
        );
        let entry_name = CString::new(ep_name).unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&entry_name);
        let cp_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        let pipeline = unsafe {
            dev.create_compute_pipelines(vk::PipelineCache::null(), &[cp_ci], None)
                .map_err(|(_, e)| BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("pipeline: {e}").into(),
                })?
                .remove(0)
        };

        unsafe { dev.destroy_shader_module(shader, None) };

        let id = self.programs.push(VulkanProgram {
            gws,
            lws,
            pipeline,
            pipeline_layout,
            desc_layout,
        });
        Ok(id)
    }

    pub(super) fn launch(
        &mut self,
        program_id: DeviceProgramId,
        memory_pool: &mut VulkanMemoryPool,
        args: &[PoolBufferId],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        // Wait for all input dependencies before submitting this kernel
        for event in &event_wait_list {
            if let Event::Vulkan(ev) = event {
                ev.wait();
            }
        }
        let prog = &self.programs[program_id];
        let dev = &self.core.device;

        // Allocate descriptor set
        let ds_layouts = [prog.desc_layout];
        let ds_alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.core.desc_pool)
            .set_layouts(&ds_layouts);
        let desc_sets = unsafe { dev.allocate_descriptor_sets(&ds_alloc) }.map_err(|e| {
            BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("alloc ds: {e}").into(),
            }
        })?;
        let desc_set = desc_sets[0];

        // Write descriptor set — one SSBO per argument
        let mut buf_infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(args.len());
        for &arg_id in args {
            let &(buf, _, _, _) = &memory_pool.buffers[arg_id];
            buf_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(buf)
                    .offset(0)
                    .range(vk::WHOLE_SIZE),
            );
        }
        let writes: Vec<vk::WriteDescriptorSet> = args
            .iter()
            .enumerate()
            .map(|(i, _)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(desc_set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&buf_infos[i]))
            })
            .collect();
        unsafe { dev.update_descriptor_sets(&writes, &[]) };

        // Allocate + begin command buffer
        let cmd_alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.core.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd_bufs = unsafe { dev.allocate_command_buffers(&cmd_alloc) }.map_err(|e| {
            BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("alloc cmd: {e}").into(),
            }
        })?;
        let cmd = cmd_bufs[0];

        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { dev.begin_command_buffer(cmd, &begin) }.map_err(|e| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: format!("begin cmd: {e}").into(),
        })?;

        let gx = prog.gws.first().copied().unwrap_or(1) as u32;
        let gy = prog.gws.get(1).copied().unwrap_or(1) as u32;
        let gz = prog.gws.get(2).copied().unwrap_or(1) as u32;

        if gx == 0 || gy == 0 || gz == 0 {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("dispatch dims zero: ({gx},{gy},{gz})").into(),
            });
        }
        let dyn_offsets: [u32; 0] = [];
        unsafe {
            dev.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, prog.pipeline);
            dev.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                prog.pipeline_layout,
                0,
                &[desc_set],
                &dyn_offsets,
            );
            dev.cmd_dispatch(cmd, gx, gy, gz);
        }

        unsafe { dev.end_command_buffer(cmd) }.map_err(|e| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: format!("end cmd: {e}").into(),
        })?;

        // Fence for GPU sync
        let fence = unsafe { dev.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("fence: {e}").into(),
            })?;

        // Submit
        let cmd_bufs = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmd_bufs);
        unsafe { dev.queue_submit(self.core.queue, &[submit], fence) }.map_err(|e| {
            BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("submit: {e}").into(),
            }
        })?;

        Ok(Event::Vulkan(VulkanEvent::create(
            self.core.clone(),
            fence,
            cmd,
            desc_set,
        )))
    }
}

// ── Initialization ──────────────────────────────────────────────────────────

#[allow(clippy::unnecessary_wraps)]
pub(super) fn initialize_device(
    config: &VulkanConfig,
    memory_pools: &mut Slab<super::PoolId, MemoryPool>,
    devices: &mut Slab<super::DeviceId, super::Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if let Some(ids) = &config.device_ids
        && ids.is_empty()
    {
        if debug_dev {
            println!("[vulkan] configured out");
        }
        return Ok(());
    }

    // ── Load Vulkan ──
    let entry = unsafe { ash::Entry::load() }.map_err(|e| {
        eprintln!("[vulkan] load error: {e}");
        BackendError {
            status: ErrorStatus::Initialization,
            context: format!("[vulkan] load: {e}").into(),
        }
    })?;

    let app_name = CString::new("zyx").unwrap();
    let engine_name = CString::new("zyx").unwrap();
    let app = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(0)
        .engine_name(&engine_name)
        .engine_version(0)
        .api_version(vk::API_VERSION_1_2);

    let ext_ptrs: Vec<*const i8> = vec![];
    let layer_ptrs: Vec<*const i8> = vec![];

    let ici = vk::InstanceCreateInfo::default()
        .application_info(&app)
        .enabled_layer_names(&layer_ptrs)
        .enabled_extension_names(&ext_ptrs);

    let instance = unsafe { entry.create_instance(&ici, None) }.map_err(|e| {
        eprintln!("[vulkan] instance error: {e}");
        BackendError {
            status: ErrorStatus::Initialization,
            context: format!("[vulkan] instance: {e}").into(),
        }
    })?;

    let gpus =
        unsafe { instance.enumerate_physical_devices() }.map_err(|e| BackendError {
            status: ErrorStatus::Initialization,
            context: format!("[vulkan] enumerate: {e}").into(),
        })?;

    // ── Find first suitable GPU, build Core ──
    for gpu in gpus {
        let props = unsafe { instance.get_physical_device_properties(gpu) };
        let name = {
            let cstr = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
            cstr.to_string_lossy().into_owned()
        };

        let qfps =
            unsafe { instance.get_physical_device_queue_family_properties(gpu) };
        let qfi = match qfps
            .iter()
            .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
        {
            Some(i) => i,
            None => {
                if debug_dev {
                    println!("[vulkan] {name}: no compute queue family");
                }
                continue;
            }
        };

        if debug_dev {
            println!("[vulkan] {name}");
        }

        let max_wg_count = props.limits.max_compute_work_group_count;
        let max_wg_invocations = props.limits.max_compute_work_group_invocations;
        let max_wg_size = props.limits.max_compute_work_group_size;

        // Device + queue
        let priority = [1.0f32];
        let qci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfi as u32)
            .queue_priorities(&priority);

        let dev_features = vk::PhysicalDeviceFeatures::default();
        let dev_ext_ptrs: Vec<*const i8> = vec![];

        let qcis = [qci];
        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qcis)
            .enabled_features(&dev_features)
            .enabled_extension_names(&dev_ext_ptrs);

        let device =
            unsafe { instance.create_device(gpu, &dci, None) }.map_err(|e| {
                BackendError {
                    status: ErrorStatus::Initialization,
                    context: format!("[vulkan] device: {e}").into(),
                }
            })?;

        let queue = unsafe { device.get_device_queue(qfi as u32, 0) };

        // Command pool
        let cp_ci = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(qfi as u32);
        let cmd_pool =
            unsafe { device.create_command_pool(&cp_ci, None) }.map_err(|e| {
                BackendError {
                    status: ErrorStatus::Initialization,
                    context: format!("[vulkan] cmd pool: {e}").into(),
                }
            })?;

        // Descriptor pool
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 1024,
        }];
        let dp_ci = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(1024)
            .pool_sizes(&pool_sizes);
        let desc_pool =
            unsafe { device.create_descriptor_pool(&dp_ci, None) }.map_err(|e| {
                BackendError {
                    status: ErrorStatus::Initialization,
                    context: format!("[vulkan] desc pool: {e}").into(),
                }
            })?;

        let core = Arc::new(Core {
            _entry: entry,
            instance,
            device,
            gpu,
            queue,
            qfi: qfi as u32,
            cmd_pool,
            desc_pool,
        });

        let mem_pool = VulkanMemoryPool {
            free_bytes: 1024 * 1024 * 1024,
            core: core.clone(),
            buffers: Slab::new(),
        };

        let dev_info = DeviceInfo {
            compute: 1_000_000_000_000,
            max_global_work_dims: vec![
                Dim::from(max_wg_count[0]);
                max_wg_count.len()
            ],
            max_local_threads: Dim::from(max_wg_invocations),
            max_local_work_dims: vec![Dim::from(max_wg_size[0]); max_wg_size.len()],
            ..Default::default()
        };

        let vk_dev = VulkanDevice {
            dev_info,
            memory_pool_id: memory_pools.push(MemoryPool::Vulkan(mem_pool)),
            core,
            programs: Slab::new(),
        };

        devices.push(super::Device::Vulkan(vk_dev));
        return Ok(());
    }

    Ok(())
}
