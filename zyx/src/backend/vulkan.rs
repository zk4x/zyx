// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Vulkan backend using ash.

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
        _event_wait_list: Vec<Event>,
    ) {
        // Safety: we own the buffer and it's not in use (event_wait_list enforces sync).
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
        _event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
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

pub(crate) struct VulkanEvent {
    fence: Option<(vk::Fence, Arc<Core>)>,
}

impl std::fmt::Debug for VulkanEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanEvent").finish_non_exhaustive()
    }
}

impl Clone for VulkanEvent {
    fn clone(&self) -> Self {
        Self {
            fence: self.fence.clone(),
        }
    }
}

impl VulkanEvent {
    const fn none() -> Self {
        Self { fence: None }
    }

    fn wait(&self) {
        if let Some((fence, ref core)) = self.fence {
            unsafe {
                core.device
                    .wait_for_fences(&[fence], true, std::u64::MAX)
                    .unwrap();
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
        eprintln!("[vulkan] compile step 1: SPIR-V");
        let (spirv, gws, lws) = crate::backend::spirv::compile(kernel, debug_asm)
            .map_err(|e| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("SPIR-V: {e}").into(),
            })?;
        eprintln!("[vulkan] compile step 2: SPIR-V done");
        let dev = &self.core.device;

        eprintln!("[vulkan] compile step 3: shader module");
        let shader_ci = vk::ShaderModuleCreateInfo::default().code(&spirv);
        let shader = unsafe { dev.create_shader_module(&shader_ci, None) }.map_err(|e| {
            BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("shader: {e}").into(),
            }
        })?;
        eprintln!("[vulkan] compile step 4: shader module done");

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
        eprintln!("[vulkan] compile step 5: n_args={n_args}");

        // Descriptor set layout (one SSBO per arg)
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
        eprintln!("[vulkan] compile step 6: ds layout");
        let desc_layout =
            unsafe { dev.create_descriptor_set_layout(&layout_ci, None) }.map_err(|e| {
                BackendError {
                    status: ErrorStatus::KernelCompilation,
                    context: format!("ds layout: {e}").into(),
                }
            })?;
        eprintln!("[vulkan] compile step 7: ds layout done");

        // Pipeline layout
        let desc_layouts = [desc_layout];
        let pl_ci = vk::PipelineLayoutCreateInfo::default().set_layouts(&desc_layouts);
        eprintln!("[vulkan] compile step 8: pipeline layout");
        let pipeline_layout =
            unsafe { dev.create_pipeline_layout(&pl_ci, None) }.map_err(|e| BackendError {
                status: ErrorStatus::KernelCompilation,
                context: format!("pipeline layout: {e}").into(),
            })?;
        eprintln!("[vulkan] compile step 9: pipeline layout done");

        // Compute pipeline
        let entry_name = CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader)
            .name(&entry_name);
        let cp_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(pipeline_layout);
        eprintln!("[vulkan] compile step 10: compute pipeline");
        let pipeline = unsafe {
            let r = dev.create_compute_pipelines(vk::PipelineCache::null(), &[cp_ci], None);
            match r {
                Ok(mut pipelines) => {
                    eprintln!("[vulkan] compile step 10a: pipeline creation OK");
                    pipelines.remove(0)
                }
                Err((pipelines, errors)) => {
                    eprintln!("[vulkan] compile step 10a: pipeline creation FAILED, errors={:?}", errors);
                    return Err(BackendError {
                        status: ErrorStatus::KernelCompilation,
                        context: format!("pipeline: {errors:?}").into(),
                    });
                }
            }
        };
        eprintln!("[vulkan] compile step 11: pipeline created, destroying shader");

        unsafe { dev.destroy_shader_module(shader, None) };

        let id = self.programs.push(VulkanProgram {
            gws,
            lws,
            pipeline,
            pipeline_layout,
            desc_layout,
        });
        eprintln!("[vulkan] compile step 12: done, id={id:?}");
        Ok(id)
    }

    pub(super) fn launch(
        &mut self,
        program_id: DeviceProgramId,
        memory_pool: &mut VulkanMemoryPool,
        args: &[PoolBufferId],
        _event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        eprintln!("[vulkan] LAUNCH ENTERED");
        panic!("LAUNCH CALLED");
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
    eprintln!("[vulkan] initialize_device called, device_ids={:?}", config.device_ids);
    if let Some(ids) = &config.device_ids
        && ids.is_empty()
    {
        eprintln!("[vulkan] configured out");
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

    let layer_names = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
    let layer_ptrs: Vec<*const i8> = layer_names.iter().map(|c| c.as_ptr()).collect();

    let ext_names = [CString::new(VK_EXT_DEBUG_UTILS_EXTENSION_NAME).unwrap()];
    let ext_ptrs: Vec<*const i8> = ext_names.iter().map(|c| c.as_ptr()).collect();

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
        eprintln!("[vulkan] initialized device {name}");
        return Ok(());
    }

    eprintln!("[vulkan] no suitable device found or init failed");
    Ok(())
}
