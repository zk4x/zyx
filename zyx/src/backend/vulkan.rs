// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only

//! Vulkan backend using raw libloading FFI (no ash) with worker-thread dispatch.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::{CStr, CString};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
    mpsc::{Receiver, Sender, channel},
};

use libloading::Library;
use nanoserde::DeJson;

use crate::{
    DType,
    error::{BackendError, ErrorStatus},
    kernel::Kernel,
    shape::Dim,
    slab::Slab,
};

use super::{DeviceInfo, DeviceProgramId, Event, MemoryPool, OpCapability, PoolBufferId, PoolId};

// ── Vulkan FFI types ─────────────────────────────────────────────────────────

type VkInstance = *mut std::ffi::c_void;
type VkPhysicalDevice = *mut std::ffi::c_void;
type VkDevice = *mut std::ffi::c_void;
type VkQueue = *mut std::ffi::c_void;
type VkCommandPool = *mut std::ffi::c_void;
type VkDescriptorPool = *mut std::ffi::c_void;
type VkBuffer = *mut std::ffi::c_void;
type VkDeviceMemory = *mut std::ffi::c_void;
type VkFence = *mut std::ffi::c_void;
type VkCommandBuffer = *mut std::ffi::c_void;
type VkDescriptorSet = *mut std::ffi::c_void;
type VkPipeline = *mut std::ffi::c_void;
type VkPipelineLayout = *mut std::ffi::c_void;
type VkDescriptorSetLayout = *mut std::ffi::c_void;
type VkShaderModule = *mut std::ffi::c_void;
type VkPipelineCache = *mut std::ffi::c_void;
type VkSampler = *mut std::ffi::c_void;
type VkResult = i32;

const VK_SUCCESS: VkResult = 0;
const VK_WHOLE_SIZE: u64 = !0;
const VK_API_VERSION_1_2: u32 = (1 << 22) | (2 << 12) | 0;
const VK_NULL_HANDLE: VkPipelineCache = std::ptr::null_mut();

const VK_STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO: u32 = 2;
const VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO: u32 = 3;
const VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO: u32 = 16;
const VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO: u32 = 12;
const VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO: u32 = 5;
const VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO: u32 = 18;
const VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO: u32 = 30;
const VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO: u32 = 29;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO: u32 = 32;
const VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO: u32 = 33;
const VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO: u32 = 34;
const VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET: u32 = 35;
const VK_STRUCTURE_TYPE_SUBMIT_INFO: u32 = 4;
const VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO: u32 = 39;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO: u32 = 40;
const VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO: u32 = 42;
const VK_STRUCTURE_TYPE_FENCE_CREATE_INFO: u32 = 8;

const VK_BUFFER_USAGE_STORAGE_BUFFER_BIT: u32 = 0x0080;
const VK_BUFFER_USAGE_TRANSFER_DST_BIT: u32 = 0x0002;
const VK_BUFFER_USAGE_TRANSFER_SRC_BIT: u32 = 0x0001;
const VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT: u32 = 0x0001;
const VK_MEMORY_PROPERTY_HOST_COHERENT_BIT: u32 = 0x0004;
const VK_SHARING_MODE_EXCLUSIVE: u32 = 0;
const VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT: u32 = 0x0001;
const VK_COMMAND_BUFFER_LEVEL_PRIMARY: u32 = 0;
const VK_PIPELINE_BIND_POINT_COMPUTE: u32 = 1;
const VK_SHADER_STAGE_COMPUTE_BIT: u32 = 0x0020;
const VK_QUEUE_COMPUTE_BIT: u32 = 0x0004;
const VK_DESCRIPTOR_TYPE_STORAGE_BUFFER: u32 = 7;
const VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT: u32 = 1;
const VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT: u32 = 0x0004;

#[repr(C)]
struct VkApplicationInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    pApplicationName: *const i8,
    applicationVersion: u32,
    pEngineName: *const i8,
    engineVersion: u32,
    apiVersion: u32,
}
#[repr(C)]
struct VkInstanceCreateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    flags: u32,
    pApplicationInfo: *const VkApplicationInfo,
    enabledLayerCount: u32,
    ppEnabledLayerNames: *const *const i8,
    enabledExtensionCount: u32,
    ppEnabledExtensionNames: *const *const i8,
}
#[repr(C)]
struct VkDeviceQueueCreateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    flags: u32,
    queueFamilyIndex: u32,
    queueCount: u32,
    pQueuePriorities: *const f32,
}
#[repr(C)]
struct VkDeviceCreateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    flags: u32,
    queueCreateInfoCount: u32,
    pQueueCreateInfos: *const VkDeviceQueueCreateInfo,
    enabledLayerCount: u32,
    ppEnabledLayerNames: *const *const i8,
    enabledExtensionCount: u32,
    ppEnabledExtensionNames: *const *const i8,
    pEnabledFeatures: *const std::ffi::c_void,
}
#[repr(C)]
struct VkShaderModuleCreateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    flags: u32,
    codeSize: usize,
    pCode: *const u32,
}
#[repr(C)]
struct VkBufferCreateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    flags: u32,
    size: u64,
    usage: u32,
    sharingMode: u32,
    queueFamilyIndexCount: u32,
    pQueueFamilyIndices: *const u32,
}
#[repr(C)]
struct VkMemoryAllocateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    allocationSize: u64,
    memoryTypeIndex: u32,
}
#[repr(C)]
struct VkMemoryRequirements {
    size: u64,
    alignment: u64,
    memoryTypeBits: u32,
}
#[repr(C)]
struct VkDescriptorSetLayoutBinding {
    binding: u32,
    descriptorType: u32,
    descriptorCount: u32,
    stageFlags: u32,
    pImmutableSamplers: *const VkSampler,
}
#[repr(C)]
struct VkDescriptorSetLayoutCreateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    flags: u32,
    bindingCount: u32,
    pBindings: *const VkDescriptorSetLayoutBinding,
}
#[repr(C)]
struct VkPipelineLayoutCreateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    flags: u32,
    setLayoutCount: u32,
    pSetLayouts: *const VkDescriptorSetLayout,
    pushConstantRangeCount: u32,
    pPushConstantRanges: *const std::ffi::c_void,
}
#[repr(C)]
struct VkPipelineShaderStageCreateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    flags: u32,
    stage: u32,
    module: VkShaderModule,
    pName: *const i8,
    pSpecializationInfo: *const std::ffi::c_void,
}
#[repr(C)]
struct VkComputePipelineCreateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    flags: u32,
    stage: VkPipelineShaderStageCreateInfo,
    layout: VkPipelineLayout,
    basePipelineHandle: VkPipeline,
    basePipelineIndex: i32,
}
#[repr(C)]
struct VkDescriptorSetAllocateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    descriptorPool: VkDescriptorPool,
    descriptorSetCount: u32,
    pSetLayouts: *const VkDescriptorSetLayout,
}
#[repr(C)]
struct VkDescriptorBufferInfo {
    buffer: VkBuffer,
    offset: u64,
    range: u64,
}
#[repr(C)]
struct VkWriteDescriptorSet {
    sType: u32,
    pNext: *const std::ffi::c_void,
    dstSet: VkDescriptorSet,
    dstBinding: u32,
    dstArrayElement: u32,
    descriptorCount: u32,
    descriptorType: u32,
    pImageInfo: *const std::ffi::c_void,
    pBufferInfo: *const VkDescriptorBufferInfo,
    pTexelBufferView: *const std::ffi::c_void,
}
#[repr(C)]
struct VkCommandBufferAllocateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    commandPool: VkCommandPool,
    level: u32,
    commandBufferCount: u32,
}
#[repr(C)]
struct VkCommandBufferBeginInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    flags: u32,
    pInheritanceInfo: *const std::ffi::c_void,
}
#[repr(C)]
struct VkSubmitInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    waitSemaphoreCount: u32,
    pWaitSemaphores: *const std::ffi::c_void,
    pWaitDstStageMask: *const u32,
    commandBufferCount: u32,
    pCommandBuffers: *const VkCommandBuffer,
    signalSemaphoreCount: u32,
    pSignalSemaphores: *const std::ffi::c_void,
}
#[repr(C)]
struct VkFenceCreateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    flags: u32,
}
#[repr(C)]
struct VkCommandPoolCreateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    flags: u32,
    queueFamilyIndex: u32,
}
#[repr(C)]
struct VkDescriptorPoolSize {
    ty: u32,
    descriptorCount: u32,
}
#[repr(C)]
struct VkDescriptorPoolCreateInfo {
    sType: u32,
    pNext: *const std::ffi::c_void,
    flags: u32,
    maxSets: u32,
    poolSizeCount: u32,
    pPoolSizes: *const VkDescriptorPoolSize,
}
#[repr(C)]
struct VkPhysicalDeviceProperties {
    api_version: u32,
    driver_version: u32,
    vendor_id: u32,
    device_id: u32,
    device_type: u32,
    device_name: [u8; 256],
    pipeline_cache_uuid: [u8; 16],
    _pad0: [u8; 216],
    max_compute_shared_memory_size: u32,
    max_compute_work_group_count: [u32; 3],
    max_compute_work_group_invocations: u32,
    max_compute_work_group_size: [u32; 3],
}
#[repr(C)]
#[derive(Clone)]
struct VkQueueFamilyProperties {
    queueFlags: u32,
    queueCount: u32,
    timestampValidBits: u32,
    minImageTransferGranularity: [u32; 3],
}
#[repr(C)]
struct VkPhysicalDeviceMemoryProperties {
    memoryTypeCount: u32,
    memoryTypes: [VkMemoryType; 32],
    memoryHeapCount: u32,
    memoryHeaps: [VkMemoryHeap; 16],
}
#[repr(C)]
struct VkMemoryHeap {
    size: u64,
    flags: u32,
}
#[repr(C)]
struct VkMemoryType {
    propertyFlags: u32,
    heapIndex: u32,
}

// ── Config ───────────────────────────────────────────────────────────────────

#[derive(DeJson, Debug, Default)]
#[nserde(default)]
pub struct VulkanConfig {
    device_ids: Option<Vec<i32>>,
}

// ── Worker-thread command enum ───────────────────────────────────────────────

enum VulkanCommand {
    Allocate {
        bytes: Dim,
        reply: Sender<Result<(PoolBufferId, Event), BackendError>>,
    },
    Deallocate {
        buffer_id: PoolBufferId,
        event_wait_list: Vec<Event>,
    },
    HostToPool {
        src: *const u8,
        bytes: usize,
        dst: PoolBufferId,
        event_wait_list: Vec<Event>,
        reply: Sender<Result<Event, BackendError>>,
    },
    PoolToHost {
        src: PoolBufferId,
        dst: *mut u8,
        bytes: usize,
        event_wait_list: Vec<Event>,
        reply: Sender<Result<(), BackendError>>,
    },
    Compile {
        kernel: Box<Kernel>,
        debug_asm: bool,
        reply: Sender<Result<DeviceProgramId, BackendError>>,
    },
    Launch {
        program_id: DeviceProgramId,
        args: Vec<PoolBufferId>,
        event_wait_list: Vec<Event>,
        reply: Sender<Result<Event, BackendError>>,
    },
    SyncEvents {
        events: Vec<Event>,
        reply: Sender<Result<(), BackendError>>,
    },
    ReleaseProgram(DeviceProgramId),
    ReleaseEvents(Vec<Event>),
}

unsafe impl Send for VulkanCommand {}

// ── Memory Pool ──────────────────────────────────────────────────────────────

pub struct VulkanMemoryPool {
    tx: Sender<VulkanCommand>,
    free_bytes: Arc<AtomicU64>,
}

unsafe impl Send for VulkanMemoryPool {}

impl std::fmt::Debug for VulkanMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanMemoryPool")
            .field("free_bytes", &self.free_bytes)
            .finish()
    }
}

impl VulkanMemoryPool {
    pub(super) fn free_bytes(&self) -> Dim {
        self.free_bytes.load(Ordering::SeqCst)
    }
    pub(super) const fn deinitialize(&mut self) {}
    pub(super) fn allocate(&mut self, bytes: Dim) -> Result<(PoolBufferId, Event), BackendError> {
        let (reply, rx) = channel();
        self.tx.send(VulkanCommand::Allocate { bytes, reply }).unwrap();
        rx.recv().unwrap()
    }
    pub(super) fn deallocate(&mut self, buffer_id: PoolBufferId, event_wait_list: Vec<Event>) {
        self.tx
            .send(VulkanCommand::Deallocate { buffer_id, event_wait_list })
            .unwrap();
    }
    pub(super) fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: PoolBufferId,
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let (reply, rx) = channel();
        self.tx
            .send(VulkanCommand::HostToPool { src: src.as_ptr(), bytes: src.len(), dst, event_wait_list, reply })
            .unwrap();
        rx.recv().unwrap()
    }
    pub(super) fn pool_to_host(
        &mut self,
        src: PoolBufferId,
        dst: &mut [u8],
        event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        let (reply, rx) = channel();
        self.tx
            .send(VulkanCommand::PoolToHost { src, dst: dst.as_mut_ptr(), bytes: dst.len(), event_wait_list, reply })
            .unwrap();
        rx.recv().unwrap()
    }
    pub(super) fn sync_events(&mut self, events: Vec<Event>) -> Result<(), BackendError> {
        let (reply, rx) = channel();
        self.tx.send(VulkanCommand::SyncEvents { events, reply }).unwrap();
        rx.recv().unwrap()
    }
    pub(super) fn release_events(&mut self, events: Vec<Event>) {
        self.tx.send(VulkanCommand::ReleaseEvents(events)).unwrap();
    }
}

// ── Event ────────────────────────────────────────────────────────────────────

pub struct VulkanEvent {
    fence: VkFence,
    cmd: VkCommandBuffer,
    desc_set: VkDescriptorSet,
}

unsafe impl Send for VulkanEvent {}

impl std::fmt::Debug for VulkanEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanEvent").finish_non_exhaustive()
    }
}

// ── Program ──────────────────────────────────────────────────────────────────

struct VulkanProgram {
    gws: Vec<Dim>,
    pipeline: VkPipeline,
    pipeline_layout: VkPipelineLayout,
    desc_layout: VkDescriptorSetLayout,
}

// ── Device ───────────────────────────────────────────────────────────────────

pub struct VulkanDevice {
    tx: Sender<VulkanCommand>,
    dev_info: DeviceInfo,
    memory_pool_id: PoolId,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice")
            .field("dev_info", &self.dev_info)
            .field("memory_pool_id", &self.memory_pool_id)
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
        self.tx.send(VulkanCommand::ReleaseProgram(program_id)).unwrap();
    }
    pub(super) fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        let (reply, rx) = channel();
        self.tx
            .send(VulkanCommand::Compile { kernel: Box::new(kernel.clone()), debug_asm, reply })
            .unwrap();
        rx.recv().unwrap()
    }
    pub(super) fn launch(
        &mut self,
        program_id: DeviceProgramId,
        _memory_pool: &mut VulkanMemoryPool,
        args: &[PoolBufferId],
        event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let (reply, rx) = channel();
        self.tx
            .send(VulkanCommand::Launch { program_id, args: args.to_vec(), event_wait_list, reply })
            .unwrap();
        rx.recv().unwrap()
    }
}

// ── Helper ───────────────────────────────────────────────────────────────────

fn find_mem_type(
    gpu: VkPhysicalDevice,
    type_filter: u32,
    required: u32,
    vkGetPhysicalDeviceMemoryProperties: unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties),
) -> Option<u32> {
    let mut mem: VkPhysicalDeviceMemoryProperties = unsafe { std::mem::zeroed() };
    unsafe { vkGetPhysicalDeviceMemoryProperties(gpu, &mut mem) };
    (0..mem.memoryTypeCount)
        .find(|&i| (type_filter & (1 << i)) != 0 && mem.memoryTypes[i as usize].propertyFlags & required == required)
}

// ── Initialization ───────────────────────────────────────────────────────────

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

    let vulkan_paths = [
        "/lib64/libvulkan.so",
        "/lib64/libvulkan.so.1",
        "/lib/libvulkan.so",
        "/lib/libvulkan.so.1",
        "/usr/lib64/libvulkan.so",
        "/usr/lib64/libvulkan.so.1",
        "/usr/lib/libvulkan.so",
        "/usr/lib/libvulkan.so.1",
        "/lib/x86_64-linux-gnu/libvulkan.so",
        "/lib/x86_64-linux-gnu/libvulkan.so.1",
        "/lib64/x86_64-linux-gnu/libvulkan.so",
        "/lib64/x86_64-linux-gnu/libvulkan.so.1",
    ];
    let lib = vulkan_paths
        .into_iter()
        .find_map(|path| unsafe { Library::new(path) }.ok())
        .ok_or_else(|| BackendError { status: ErrorStatus::DyLibNotFound, context: "[vulkan] libvulkan.so not found.".into() })?;
    let vkGetInstanceProcAddr: unsafe extern "system" fn(VkInstance, *const i8) -> *mut std::ffi::c_void =
        *unsafe { lib.get(b"vkGetInstanceProcAddr\0") }?;
    let vkCreateInstance: unsafe extern "system" fn(
        *const VkInstanceCreateInfo,
        *const std::ffi::c_void,
        *mut VkInstance,
    ) -> VkResult = *unsafe { lib.get(b"vkCreateInstance\0") }?;

    let app_name = CString::new("zyx").unwrap();
    let engine_name = CString::new("zyx").unwrap();
    let app = VkApplicationInfo {
        sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pNext: std::ptr::null(),
        pApplicationName: app_name.as_ptr(),
        applicationVersion: 0,
        pEngineName: engine_name.as_ptr(),
        engineVersion: 0,
        apiVersion: VK_API_VERSION_1_2,
    };
    let ici = VkInstanceCreateInfo {
        sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext: std::ptr::null(),
        flags: 0,
        pApplicationInfo: &app,
        enabledLayerCount: 0,
        ppEnabledLayerNames: std::ptr::null(),
        enabledExtensionCount: 0,
        ppEnabledExtensionNames: std::ptr::null(),
    };
    let mut instance = std::ptr::null_mut();
    let res = unsafe { vkCreateInstance(&ici, std::ptr::null(), &mut instance) };
    if res != VK_SUCCESS {
        return Err(BackendError { status: ErrorStatus::Initialization, context: format!("[vulkan] instance: {res}").into() });
    }

    // Instance-level function pointers (loaded once)
    macro_rules! get_inst_proc {
        ($name:literal) => {
            unsafe {
                std::mem::transmute::<*mut std::ffi::c_void, _>(vkGetInstanceProcAddr(
                    instance,
                    concat!($name, "\0").as_ptr() as *const i8,
                ))
            }
        };
    }
    let vkDestroyInstance: unsafe extern "system" fn(VkInstance, *const std::ffi::c_void) = get_inst_proc!("vkDestroyInstance");
    let vkEnumeratePhysicalDevices: unsafe extern "system" fn(VkInstance, *mut u32, *mut VkPhysicalDevice) -> VkResult =
        get_inst_proc!("vkEnumeratePhysicalDevices");
    let vkGetPhysicalDeviceProperties: unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceProperties) =
        get_inst_proc!("vkGetPhysicalDeviceProperties");
    let vkGetPhysicalDeviceQueueFamilyProperties: unsafe extern "system" fn(
        VkPhysicalDevice,
        *mut u32,
        *mut VkQueueFamilyProperties,
    ) = get_inst_proc!("vkGetPhysicalDeviceQueueFamilyProperties");
    let vkGetPhysicalDeviceMemoryProperties: unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceMemoryProperties) =
        get_inst_proc!("vkGetPhysicalDeviceMemoryProperties");
    let vkCreateDevice: unsafe extern "system" fn(
        VkPhysicalDevice,
        *const VkDeviceCreateInfo,
        *const std::ffi::c_void,
        *mut VkDevice,
    ) -> VkResult = get_inst_proc!("vkCreateDevice");
    let vkGetDeviceProcAddr: unsafe extern "system" fn(VkDevice, *const i8) -> *mut std::ffi::c_void =
        get_inst_proc!("vkGetDeviceProcAddr");

    // Wrap library in Arc so the worker thread can hold a reference (OpenCL pattern)
    let library = Arc::new(lib);

    let mut gpu_count: u32 = 0;
    let _ = unsafe { vkEnumeratePhysicalDevices(instance, &mut gpu_count, std::ptr::null_mut()) };
    let mut gpus: Vec<VkPhysicalDevice> = vec![std::ptr::null_mut(); gpu_count as usize];
    let _ = unsafe { vkEnumeratePhysicalDevices(instance, &mut gpu_count, gpus.as_mut_ptr()) };

    for gpu in gpus {
        let mut props: VkPhysicalDeviceProperties = unsafe { std::mem::zeroed() };
        unsafe { vkGetPhysicalDeviceProperties(gpu, &mut props) };
        let name = {
            let cstr = unsafe { CStr::from_ptr(props.device_name.as_ptr() as *const i8) };
            cstr.to_string_lossy().into_owned()
        };

        if name.contains("llvmpipe") {
            if debug_dev {
                println!("[vulkan] skipping sw rasterizer: {name}");
            }
            continue;
        }

        let mut qfp_count: u32 = 0;
        unsafe { vkGetPhysicalDeviceQueueFamilyProperties(gpu, &mut qfp_count, std::ptr::null_mut()) };
        let mut qfps: Vec<VkQueueFamilyProperties> = vec![unsafe { std::mem::zeroed() }; qfp_count as usize];
        unsafe { vkGetPhysicalDeviceQueueFamilyProperties(gpu, &mut qfp_count, qfps.as_mut_ptr()) };
        let qfi = match qfps.iter().position(|q| q.queueFlags & VK_QUEUE_COMPUTE_BIT != 0) {
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

        let max_wg_count = props.max_compute_work_group_count;
        let max_wg_invocations = props.max_compute_work_group_invocations;
        let max_wg_size = props.max_compute_work_group_size;

        let priority = [1.0f32];
        let qci = VkDeviceQueueCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext: std::ptr::null(),
            flags: 0,
            queueFamilyIndex: qfi as u32,
            queueCount: 1,
            pQueuePriorities: priority.as_ptr(),
        };
        let dci = VkDeviceCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext: std::ptr::null(),
            flags: 0,
            queueCreateInfoCount: 1,
            pQueueCreateInfos: &qci,
            enabledLayerCount: 0,
            ppEnabledLayerNames: std::ptr::null(),
            enabledExtensionCount: 0,
            ppEnabledExtensionNames: std::ptr::null(),
            pEnabledFeatures: std::ptr::null(),
        };
        let mut device = std::ptr::null_mut();
        let res = unsafe { vkCreateDevice(gpu, &dci, std::ptr::null(), &mut device) };
        if res != VK_SUCCESS {
            return Err(BackendError { status: ErrorStatus::Initialization, context: format!("[vulkan] device: {res}").into() });
        }

        let vkGetDeviceQueue: unsafe extern "system" fn(VkDevice, u32, u32, *mut VkQueue) = unsafe {
            std::mem::transmute::<*mut std::ffi::c_void, _>(vkGetDeviceProcAddr(
                device,
                concat!("vkGetDeviceQueue", "\0").as_ptr() as *const i8,
            ))
        };
        let mut queue = std::ptr::null_mut();
        unsafe { vkGetDeviceQueue(device, qfi as u32, 0, &mut queue) };

        // Device-level function pointers (loaded per-device)
        macro_rules! ld {
            ($name:literal) => {
                unsafe {
                    std::mem::transmute::<*mut std::ffi::c_void, _>(vkGetDeviceProcAddr(
                        device,
                        concat!($name, "\0").as_ptr() as *const i8,
                    ))
                }
            };
        }
        let vkDestroyDevice: unsafe extern "system" fn(VkDevice, *const std::ffi::c_void) = ld!("vkDestroyDevice");
        let vkDestroyBuffer: unsafe extern "system" fn(VkDevice, VkBuffer, *const std::ffi::c_void) = ld!("vkDestroyBuffer");
        let vkDestroyCommandPool: unsafe extern "system" fn(VkDevice, VkCommandPool, *const std::ffi::c_void) =
            ld!("vkDestroyCommandPool");
        let vkDestroyDescriptorPool: unsafe extern "system" fn(VkDevice, VkDescriptorPool, *const std::ffi::c_void) =
            ld!("vkDestroyDescriptorPool");
        let vkDestroyShaderModule: unsafe extern "system" fn(VkDevice, VkShaderModule, *const std::ffi::c_void) =
            ld!("vkDestroyShaderModule");
        let vkDestroyPipeline: unsafe extern "system" fn(VkDevice, VkPipeline, *const std::ffi::c_void) =
            ld!("vkDestroyPipeline");
        let vkDestroyPipelineLayout: unsafe extern "system" fn(VkDevice, VkPipelineLayout, *const std::ffi::c_void) =
            ld!("vkDestroyPipelineLayout");
        let vkDestroyDescriptorSetLayout: unsafe extern "system" fn(VkDevice, VkDescriptorSetLayout, *const std::ffi::c_void) =
            ld!("vkDestroyDescriptorSetLayout");
        let vkDestroyFence: unsafe extern "system" fn(VkDevice, VkFence, *const std::ffi::c_void) = ld!("vkDestroyFence");
        let vkCreateBuffer: unsafe extern "system" fn(
            VkDevice,
            *const VkBufferCreateInfo,
            *const std::ffi::c_void,
            *mut VkBuffer,
        ) -> VkResult = ld!("vkCreateBuffer");
        let vkCreateCommandPoolFn: unsafe extern "system" fn(
            VkDevice,
            *const VkCommandPoolCreateInfo,
            *const std::ffi::c_void,
            *mut VkCommandPool,
        ) -> VkResult = ld!("vkCreateCommandPool");
        let vkCreateDescriptorPoolFn: unsafe extern "system" fn(
            VkDevice,
            *const VkDescriptorPoolCreateInfo,
            *const std::ffi::c_void,
            *mut VkDescriptorPool,
        ) -> VkResult = ld!("vkCreateDescriptorPool");
        let vkCreateFence: unsafe extern "system" fn(
            VkDevice,
            *const VkFenceCreateInfo,
            *const std::ffi::c_void,
            *mut VkFence,
        ) -> VkResult = ld!("vkCreateFence");
        let vkCreateShaderModule: unsafe extern "system" fn(
            VkDevice,
            *const VkShaderModuleCreateInfo,
            *const std::ffi::c_void,
            *mut VkShaderModule,
        ) -> VkResult = ld!("vkCreateShaderModule");
        let vkCreateDescriptorSetLayout: unsafe extern "system" fn(
            VkDevice,
            *const VkDescriptorSetLayoutCreateInfo,
            *const std::ffi::c_void,
            *mut VkDescriptorSetLayout,
        ) -> VkResult = ld!("vkCreateDescriptorSetLayout");
        let vkCreatePipelineLayout: unsafe extern "system" fn(
            VkDevice,
            *const VkPipelineLayoutCreateInfo,
            *const std::ffi::c_void,
            *mut VkPipelineLayout,
        ) -> VkResult = ld!("vkCreatePipelineLayout");
        let vkCreateComputePipelines: unsafe extern "system" fn(
            VkDevice,
            VkPipelineCache,
            u32,
            *const VkComputePipelineCreateInfo,
            *const std::ffi::c_void,
            *mut VkPipeline,
        ) -> VkResult = ld!("vkCreateComputePipelines");
        let vkAllocateMemory: unsafe extern "system" fn(
            VkDevice,
            *const VkMemoryAllocateInfo,
            *const std::ffi::c_void,
            *mut VkDeviceMemory,
        ) -> VkResult = ld!("vkAllocateMemory");
        let vkFreeMemory: unsafe extern "system" fn(VkDevice, VkDeviceMemory, *const std::ffi::c_void) = ld!("vkFreeMemory");
        let vkBindBufferMemory: unsafe extern "system" fn(VkDevice, VkBuffer, VkDeviceMemory, u64) -> VkResult =
            ld!("vkBindBufferMemory");
        let vkMapMemory: unsafe extern "system" fn(
            VkDevice,
            VkDeviceMemory,
            u64,
            u64,
            u32,
            *mut *mut std::ffi::c_void,
        ) -> VkResult = ld!("vkMapMemory");
        let vkUnmapMemory: unsafe extern "system" fn(VkDevice, VkDeviceMemory) = ld!("vkUnmapMemory");
        let vkGetBufferMemoryRequirements: unsafe extern "system" fn(VkDevice, VkBuffer, *mut VkMemoryRequirements) =
            ld!("vkGetBufferMemoryRequirements");
        let vkWaitForFences: unsafe extern "system" fn(VkDevice, u32, *const VkFence, u32, u64) -> VkResult =
            ld!("vkWaitForFences");
        let vkAllocateDescriptorSets: unsafe extern "system" fn(
            VkDevice,
            *const VkDescriptorSetAllocateInfo,
            *mut VkDescriptorSet,
        ) -> VkResult = ld!("vkAllocateDescriptorSets");
        let vkFreeDescriptorSets: unsafe extern "system" fn(VkDevice, VkDescriptorPool, u32, *const VkDescriptorSet) -> VkResult =
            ld!("vkFreeDescriptorSets");
        let vkUpdateDescriptorSets: unsafe extern "system" fn(
            VkDevice,
            u32,
            *const VkWriteDescriptorSet,
            u32,
            *const std::ffi::c_void,
        ) = ld!("vkUpdateDescriptorSets");
        let vkAllocateCommandBuffers: unsafe extern "system" fn(
            VkDevice,
            *const VkCommandBufferAllocateInfo,
            *mut VkCommandBuffer,
        ) -> VkResult = ld!("vkAllocateCommandBuffers");
        let vkFreeCommandBuffers: unsafe extern "system" fn(VkDevice, VkCommandPool, u32, *const VkCommandBuffer) =
            ld!("vkFreeCommandBuffers");
        let vkBeginCommandBuffer: unsafe extern "system" fn(VkCommandBuffer, *const VkCommandBufferBeginInfo) -> VkResult =
            ld!("vkBeginCommandBuffer");
        let vkEndCommandBuffer: unsafe extern "system" fn(VkCommandBuffer) -> VkResult = ld!("vkEndCommandBuffer");
        let vkCmdBindPipeline: unsafe extern "system" fn(VkCommandBuffer, u32, VkPipeline) = ld!("vkCmdBindPipeline");
        let vkCmdBindDescriptorSets: unsafe extern "system" fn(
            VkCommandBuffer,
            u32,
            VkPipelineLayout,
            u32,
            u32,
            *const VkDescriptorSet,
            u32,
            *const u32,
        ) = ld!("vkCmdBindDescriptorSets");
        let vkCmdDispatch: unsafe extern "system" fn(VkCommandBuffer, u32, u32, u32) = ld!("vkCmdDispatch");
        let vkQueueSubmit: unsafe extern "system" fn(VkQueue, u32, *const VkSubmitInfo, VkFence) -> VkResult =
            ld!("vkQueueSubmit");

        // Cast raw handles through usize for Send capture
        let instance_raw = instance as usize;
        let device_raw = device as usize;
        let gpu_raw = gpu as usize;
        let queue_raw = queue as usize;

        let total_bytes = 1024 * 1024 * 1024; // 1 GB
        let free_bytes_atomic = Arc::new(AtomicU64::new(total_bytes as u64));
        let (tx, rx): (Sender<VulkanCommand>, Receiver<VulkanCommand>) = channel();

        // Clone library Arc for worker thread (OpenCL pattern)
        let worker_library = Arc::clone(&library);

        std::thread::spawn({
            let free_bytes_atomic = Arc::clone(&free_bytes_atomic);
            move || {
                let _worker_library = worker_library; // keep libvulkan.so alive
                let instance = instance_raw as VkInstance;
                let device = device_raw as VkDevice;
                let gpu = gpu_raw as VkPhysicalDevice;
                let queue = queue_raw as VkQueue;

                let cp_ci = VkCommandPoolCreateInfo {
                    sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    pNext: std::ptr::null(),
                    flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                    queueFamilyIndex: qfi as u32,
                };
                let mut cmd_pool = std::ptr::null_mut();
                let res = unsafe { vkCreateCommandPoolFn(device, &cp_ci, std::ptr::null(), &mut cmd_pool) };
                if res != VK_SUCCESS {
                    if debug_dev {
                        println!("[vulkan] cmd pool: {res}");
                    }
                    return;
                }

                let pool_sizes = [VkDescriptorPoolSize { ty: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount: 1024 }];
                let dp_ci = VkDescriptorPoolCreateInfo {
                    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    pNext: std::ptr::null(),
                    flags: VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                    maxSets: 1024,
                    poolSizeCount: 1,
                    pPoolSizes: pool_sizes.as_ptr(),
                };
                let mut desc_pool = std::ptr::null_mut();
                let res = unsafe { vkCreateDescriptorPoolFn(device, &dp_ci, std::ptr::null(), &mut desc_pool) };
                if res != VK_SUCCESS {
                    if debug_dev {
                        println!("[vulkan] desc pool: {res}");
                    }
                    return;
                }

                let mut buffers: Slab<PoolBufferId, (VkBuffer, VkDeviceMemory, *mut u8, usize)> = Slab::new();
                let mut programs: Slab<DeviceProgramId, VulkanProgram> = Slab::new();

                macro_rules! send_or_continue {
                    ($expr:expr, $tx:expr) => {
                        match $expr {
                            Ok(v) => v,
                            Err(e) => {
                                let _ = $tx.send(Err(e));
                                continue;
                            }
                        }
                    };
                }

                let create_buffer = |size: u64| -> Result<(VkBuffer, VkDeviceMemory, *mut u8), BackendError> {
                    let usage =
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
                    let ci = VkBufferCreateInfo {
                        sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                        pNext: std::ptr::null(),
                        flags: 0,
                        size,
                        usage,
                        sharingMode: VK_SHARING_MODE_EXCLUSIVE,
                        queueFamilyIndexCount: 0,
                        pQueueFamilyIndices: std::ptr::null(),
                    };
                    let mut buf = std::ptr::null_mut();
                    let res = unsafe { vkCreateBuffer(device, &ci, std::ptr::null(), &mut buf) };
                    if res != VK_SUCCESS {
                        return Err(BackendError {
                            status: ErrorStatus::MemoryAllocation,
                            context: format!("vkCreateBuffer: {res}").into(),
                        });
                    }
                    let mut req: VkMemoryRequirements = unsafe { std::mem::zeroed() };
                    unsafe { vkGetBufferMemoryRequirements(device, buf, &mut req) };
                    let mem_type = find_mem_type(
                        gpu,
                        req.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        vkGetPhysicalDeviceMemoryProperties,
                    )
                    .ok_or_else(|| BackendError {
                        status: ErrorStatus::MemoryAllocation,
                        context: "no suitable memory type".into(),
                    })?;
                    let alloc = VkMemoryAllocateInfo {
                        sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                        pNext: std::ptr::null(),
                        allocationSize: req.size,
                        memoryTypeIndex: mem_type,
                    };
                    let mut mem = std::ptr::null_mut();
                    let res = unsafe { vkAllocateMemory(device, &alloc, std::ptr::null(), &mut mem) };
                    if res != VK_SUCCESS {
                        return Err(BackendError {
                            status: ErrorStatus::MemoryAllocation,
                            context: format!("vkAllocateMemory: {res}").into(),
                        });
                    }
                    let res = unsafe { vkBindBufferMemory(device, buf, mem, 0) };
                    if res != VK_SUCCESS {
                        return Err(BackendError {
                            status: ErrorStatus::MemoryAllocation,
                            context: format!("vkBindBufferMemory: {res}").into(),
                        });
                    }
                    let mut ptr = std::ptr::null_mut();
                    let res = unsafe { vkMapMemory(device, mem, 0, size, 0, &mut ptr) };
                    if res != VK_SUCCESS {
                        return Err(BackendError {
                            status: ErrorStatus::MemoryAllocation,
                            context: format!("vkMapMemory: {res}").into(),
                        });
                    }
                    Ok((buf, mem, ptr.cast::<u8>()))
                };

                while let Ok(cmd) = rx.recv() {
                    match cmd {
                        VulkanCommand::Allocate { bytes, reply } => {
                            let size = bytes.next_multiple_of(4) as u64;
                            let (buf, mem, ptr) = send_or_continue!(create_buffer(size), reply);
                            let id = buffers.push((buf, mem, ptr, bytes as usize));
                            free_bytes_atomic.fetch_sub(size, Ordering::SeqCst);
                            let _ = reply.send(Ok((
                                PoolBufferId::from(id),
                                Event::Vulkan(VulkanEvent {
                                    fence: std::ptr::null_mut(),
                                    cmd: std::ptr::null_mut(),
                                    desc_set: std::ptr::null_mut(),
                                }),
                            )));
                        }
                        VulkanCommand::Deallocate { buffer_id, mut event_wait_list } => {
                            while let Some(Event::Vulkan(ev)) = event_wait_list.pop() {
                                if !ev.fence.is_null() {
                                    unsafe {
                                        let _ = vkWaitForFences(device, 1, &ev.fence, 1, u64::MAX);
                                        vkDestroyFence(device, ev.fence, std::ptr::null());
                                    }
                                }
                                if !ev.cmd.is_null() {
                                    unsafe {
                                        vkFreeCommandBuffers(device, cmd_pool, 1, &ev.cmd);
                                    }
                                }
                                if !ev.desc_set.is_null() {
                                    unsafe {
                                        vkFreeDescriptorSets(device, desc_pool, 1, &ev.desc_set);
                                    }
                                }
                            }
                            let (buf, mem, ptr, size) = unsafe { buffers.remove_and_return(buffer_id) };
                            if !ptr.is_null() {
                                unsafe { vkUnmapMemory(device, mem) };
                            }
                            unsafe {
                                vkDestroyBuffer(device, buf, std::ptr::null());
                                vkFreeMemory(device, mem, std::ptr::null());
                            }
                            free_bytes_atomic.fetch_add(size as u64, Ordering::SeqCst);
                        }
                        VulkanCommand::HostToPool { src, bytes, dst, mut event_wait_list, reply } => {
                            while let Some(Event::Vulkan(ev)) = event_wait_list.pop() {
                                if !ev.fence.is_null() {
                                    unsafe {
                                        let _ = vkWaitForFences(device, 1, &ev.fence, 1, u64::MAX);
                                        vkDestroyFence(device, ev.fence, std::ptr::null());
                                    }
                                }
                            }
                            let &(_, _, ptr, _) = &buffers[dst];
                            unsafe { std::ptr::copy_nonoverlapping(src, ptr, bytes) };
                            let _ = reply.send(Ok(Event::Vulkan(VulkanEvent {
                                fence: std::ptr::null_mut(),
                                cmd: std::ptr::null_mut(),
                                desc_set: std::ptr::null_mut(),
                            })));
                        }
                        VulkanCommand::PoolToHost { src, dst, bytes, mut event_wait_list, reply } => {
                            while let Some(Event::Vulkan(ev)) = event_wait_list.pop() {
                                if !ev.fence.is_null() {
                                    unsafe {
                                        let _ = vkWaitForFences(device, 1, &ev.fence, 1, u64::MAX);
                                        vkDestroyFence(device, ev.fence, std::ptr::null());
                                    }
                                }
                            }
                            let &(_, _, ptr, _) = &buffers[src];
                            unsafe { std::ptr::copy_nonoverlapping(ptr, dst, bytes) };
                            let _ = reply.send(Ok(()));
                        }
                        VulkanCommand::Compile { kernel, debug_asm, reply } => {
                            let (spirv, gws, lws) = send_or_continue!(
                                crate::backend::spirv::compile(&kernel, debug_asm).map_err(|e| BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: format!("SPIR-V: {e}").into()
                                }),
                                reply
                            );

                            let shader_ci = VkShaderModuleCreateInfo {
                                sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                pNext: std::ptr::null(),
                                flags: 0,
                                codeSize: spirv.len() * 4,
                                pCode: spirv.as_ptr(),
                            };
                            let mut shader = std::ptr::null_mut();
                            {
                                let res = unsafe { vkCreateShaderModule(device, &shader_ci, std::ptr::null(), &mut shader) };
                                if res != VK_SUCCESS {
                                    let _ = reply.send(Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("vkCreateShaderModule: {res}").into(),
                                    }));
                                    continue;
                                }
                            }

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

                            let bindings: Vec<VkDescriptorSetLayoutBinding> = (0..n_args as u32)
                                .map(|i| VkDescriptorSetLayoutBinding {
                                    binding: i,
                                    descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    descriptorCount: 1,
                                    stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
                                    pImmutableSamplers: std::ptr::null(),
                                })
                                .collect();
                            let layout_ci = VkDescriptorSetLayoutCreateInfo {
                                sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                pNext: std::ptr::null(),
                                flags: 0,
                                bindingCount: bindings.len() as u32,
                                pBindings: bindings.as_ptr(),
                            };
                            let mut desc_layout = std::ptr::null_mut();
                            {
                                let res = unsafe {
                                    vkCreateDescriptorSetLayout(device, &layout_ci, std::ptr::null(), &mut desc_layout)
                                };
                                if res != VK_SUCCESS {
                                    let _ = reply.send(Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("vkCreateDescriptorSetLayout: {res}").into(),
                                    }));
                                    continue;
                                }
                            }

                            let pl_ci = VkPipelineLayoutCreateInfo {
                                sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                pNext: std::ptr::null(),
                                flags: 0,
                                setLayoutCount: 1,
                                pSetLayouts: &desc_layout,
                                pushConstantRangeCount: 0,
                                pPushConstantRanges: std::ptr::null(),
                            };
                            let mut pipeline_layout = std::ptr::null_mut();
                            {
                                let res =
                                    unsafe { vkCreatePipelineLayout(device, &pl_ci, std::ptr::null(), &mut pipeline_layout) };
                                if res != VK_SUCCESS {
                                    let _ = reply.send(Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("vkCreatePipelineLayout: {res}").into(),
                                    }));
                                    continue;
                                }
                            }

                            let ep_name = format!(
                                "k_{}__{}",
                                gws.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("_"),
                                lws.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("_"),
                            );
                            let entry_name = CString::new(ep_name).unwrap();
                            let stage = VkPipelineShaderStageCreateInfo {
                                sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                pNext: std::ptr::null(),
                                flags: 0,
                                stage: VK_SHADER_STAGE_COMPUTE_BIT,
                                module: shader,
                                pName: entry_name.as_ptr(),
                                pSpecializationInfo: std::ptr::null(),
                            };
                            let cp_ci = VkComputePipelineCreateInfo {
                                sType: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                pNext: std::ptr::null(),
                                flags: 0,
                                stage,
                                layout: pipeline_layout,
                                basePipelineHandle: std::ptr::null_mut(),
                                basePipelineIndex: -1,
                            };
                            let mut pipeline = std::ptr::null_mut();
                            {
                                let res = unsafe {
                                    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cp_ci, std::ptr::null(), &mut pipeline)
                                };
                                if res != VK_SUCCESS {
                                    let _ = reply.send(Err(BackendError {
                                        status: ErrorStatus::KernelCompilation,
                                        context: format!("vkCreateComputePipelines: {res}").into(),
                                    }));
                                    continue;
                                }
                            }

                            unsafe { vkDestroyShaderModule(device, shader, std::ptr::null()) };

                            let id = programs.push(VulkanProgram { gws, pipeline, pipeline_layout, desc_layout });
                            let _ = reply.send(Ok(id));
                        }
                        VulkanCommand::Launch { program_id, args, mut event_wait_list, reply } => {
                            while let Some(Event::Vulkan(ev)) = event_wait_list.pop() {
                                if !ev.fence.is_null() {
                                    unsafe {
                                        let _ = vkWaitForFences(device, 1, &ev.fence, 1, u64::MAX);
                                        vkDestroyFence(device, ev.fence, std::ptr::null());
                                    }
                                }
                            }

                            let prog = &programs[program_id];

                            let ds_layouts = [prog.desc_layout];
                            let ds_alloc = VkDescriptorSetAllocateInfo {
                                sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                pNext: std::ptr::null(),
                                descriptorPool: desc_pool,
                                descriptorSetCount: 1,
                                pSetLayouts: ds_layouts.as_ptr(),
                            };
                            let mut desc_set = std::ptr::null_mut();
                            let res = unsafe { vkAllocateDescriptorSets(device, &ds_alloc, &mut desc_set) };
                            if res != VK_SUCCESS {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelLaunch,
                                    context: format!("vkAllocateDescriptorSets: {res}").into(),
                                }));
                                continue;
                            }
                            let n = args.len();
                            let mut buf_infos: Vec<VkDescriptorBufferInfo> = Vec::with_capacity(n);
                            for &arg_id in &args {
                                let &(buf, _, _, _) = &buffers[arg_id];
                                buf_infos.push(VkDescriptorBufferInfo { buffer: buf, offset: 0, range: VK_WHOLE_SIZE });
                            }
                            let mut writes: Vec<VkWriteDescriptorSet> = Vec::with_capacity(n);
                            for i in 0..n {
                                writes.push(VkWriteDescriptorSet {
                                    sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                    pNext: std::ptr::null(),
                                    dstSet: desc_set,
                                    dstBinding: i as u32,
                                    dstArrayElement: 0,
                                    descriptorCount: 1,
                                    descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    pImageInfo: std::ptr::null(),
                                    pBufferInfo: &buf_infos[i],
                                    pTexelBufferView: std::ptr::null(),
                                });
                            }
                            unsafe { vkUpdateDescriptorSets(device, writes.len() as u32, writes.as_ptr(), 0, std::ptr::null()) };

                            let cmd_alloc = VkCommandBufferAllocateInfo {
                                sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                pNext: std::ptr::null(),
                                commandPool: cmd_pool,
                                level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                commandBufferCount: 1,
                            };
                            let mut cmd = std::ptr::null_mut();
                            let res = unsafe { vkAllocateCommandBuffers(device, &cmd_alloc, &mut cmd) };
                            if res != VK_SUCCESS {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelLaunch,
                                    context: format!("vkAllocateCommandBuffers: {res}").into(),
                                }));
                                continue;
                            }

                            let begin = VkCommandBufferBeginInfo {
                                sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                pNext: std::ptr::null(),
                                flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                                pInheritanceInfo: std::ptr::null(),
                            };
                            let res = unsafe { vkBeginCommandBuffer(cmd, &begin) };
                            if res != VK_SUCCESS {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelLaunch,
                                    context: format!("vkBeginCommandBuffer: {res}").into(),
                                }));
                                continue;
                            }

                            let gx = prog.gws.first().copied().unwrap_or(1) as u32;
                            let gy = prog.gws.get(1).copied().unwrap_or(1) as u32;
                            let gz = prog.gws.get(2).copied().unwrap_or(1) as u32;

                            if gx == 0 || gy == 0 || gz == 0 {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelLaunch,
                                    context: format!("dispatch dims zero: ({gx},{gy},{gz})").into(),
                                }));
                                continue;
                            }

                            unsafe {
                                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prog.pipeline);
                                vkCmdBindDescriptorSets(
                                    cmd,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    prog.pipeline_layout,
                                    0,
                                    1,
                                    &desc_set,
                                    0,
                                    std::ptr::null(),
                                );
                                vkCmdDispatch(cmd, gx, gy, gz);
                            }

                            let res = unsafe { vkEndCommandBuffer(cmd) };
                            if res != VK_SUCCESS {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelLaunch,
                                    context: format!("vkEndCommandBuffer: {res}").into(),
                                }));
                                continue;
                            }

                            let fence_ci = VkFenceCreateInfo {
                                sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                pNext: std::ptr::null(),
                                flags: 0,
                            };
                            let mut fence = std::ptr::null_mut();
                            let res = unsafe { vkCreateFence(device, &fence_ci, std::ptr::null(), &mut fence) };
                            if res != VK_SUCCESS {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelLaunch,
                                    context: format!("vkCreateFence: {res}").into(),
                                }));
                                continue;
                            }

                            let submit = VkSubmitInfo {
                                sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                pNext: std::ptr::null(),
                                waitSemaphoreCount: 0,
                                pWaitSemaphores: std::ptr::null(),
                                pWaitDstStageMask: std::ptr::null(),
                                commandBufferCount: 1,
                                pCommandBuffers: &cmd,
                                signalSemaphoreCount: 0,
                                pSignalSemaphores: std::ptr::null(),
                            };
                            let res = unsafe { vkQueueSubmit(queue, 1, &submit, fence) };
                            if res != VK_SUCCESS {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelLaunch,
                                    context: format!("vkQueueSubmit: {res}").into(),
                                }));
                                continue;
                            }

                            let _ = reply.send(Ok(Event::Vulkan(VulkanEvent { fence, cmd, desc_set })));
                        }
                        VulkanCommand::SyncEvents { mut events, reply } => {
                            for event in &mut events {
                                if let Event::Vulkan(ev) = event {
                                    if !ev.fence.is_null() {
                                        unsafe {
                                            let _ = vkWaitForFences(device, 1, &ev.fence, 1, u64::MAX);
                                            vkDestroyFence(device, ev.fence, std::ptr::null());
                                        }
                                        ev.fence = std::ptr::null_mut();
                                    }
                                    if !ev.cmd.is_null() {
                                        unsafe { vkFreeCommandBuffers(device, cmd_pool, 1, &ev.cmd) };
                                        ev.cmd = std::ptr::null_mut();
                                    }
                                    if !ev.desc_set.is_null() {
                                        unsafe { vkFreeDescriptorSets(device, desc_pool, 1, &ev.desc_set) };
                                        ev.desc_set = std::ptr::null_mut();
                                    }
                                }
                            }
                            let _ = reply.send(Ok(()));
                        }
                        VulkanCommand::ReleaseProgram(program_id) => {
                            if programs.contains_key(program_id) {
                                let prog = unsafe { programs.remove_and_return(program_id) };
                                unsafe {
                                    vkDestroyPipeline(device, prog.pipeline, std::ptr::null());
                                    vkDestroyPipelineLayout(device, prog.pipeline_layout, std::ptr::null());
                                    vkDestroyDescriptorSetLayout(device, prog.desc_layout, std::ptr::null());
                                }
                            }
                        }
                        VulkanCommand::ReleaseEvents(events) => {
                            for event in events {
                                if let Event::Vulkan(ev) = event {
                                    if !ev.fence.is_null() {
                                        unsafe { vkDestroyFence(device, ev.fence, std::ptr::null()) };
                                    }
                                    if !ev.cmd.is_null() {
                                        unsafe { vkFreeCommandBuffers(device, cmd_pool, 1, &ev.cmd) };
                                    }
                                    if !ev.desc_set.is_null() {
                                        unsafe { vkFreeDescriptorSets(device, desc_pool, 1, &ev.desc_set) };
                                    }
                                }
                            }
                        }
                    }
                }

                // Cleanup all resources
                for id in buffers.ids().collect::<Vec<_>>() {
                    let (buf, mem, ptr, _) = unsafe { buffers.remove_and_return(id) };
                    if !ptr.is_null() {
                        unsafe { vkUnmapMemory(device, mem) };
                    }
                    unsafe {
                        vkDestroyBuffer(device, buf, std::ptr::null());
                        vkFreeMemory(device, mem, std::ptr::null());
                    }
                }
                for id in programs.ids().collect::<Vec<_>>() {
                    let prog = unsafe { programs.remove_and_return(id) };
                    unsafe {
                        vkDestroyPipeline(device, prog.pipeline, std::ptr::null());
                        vkDestroyPipelineLayout(device, prog.pipeline_layout, std::ptr::null());
                        vkDestroyDescriptorSetLayout(device, prog.desc_layout, std::ptr::null());
                    }
                }

                unsafe {
                    vkDestroyDescriptorPool(device, desc_pool, std::ptr::null());
                    vkDestroyCommandPool(device, cmd_pool, std::ptr::null());
                    vkDestroyDevice(device, std::ptr::null());
                    vkDestroyInstance(instance, std::ptr::null());
                }
            }
        });

        let mem_pool = VulkanMemoryPool { tx: tx.clone(), free_bytes: Arc::clone(&free_bytes_atomic) };
        memory_pools.push(MemoryPool::Vulkan(mem_pool));
        let dev = VulkanDevice {
            tx,
            dev_info: DeviceInfo {
                compute: 1_000_000_000_000,
                max_global_work_dims: vec![Dim::from(max_wg_count[0]); max_wg_count.len()],
                max_local_threads: Dim::from(max_wg_invocations),
                max_local_work_dims: vec![Dim::from(max_wg_size[0]); max_wg_size.len()],
                preferred_vector_size: 4,
                local_mem_size: Dim::from(props.max_compute_shared_memory_size),
                max_register_bytes: 1024,
                tensor_cores: false,
                warp_size: 32,
                supported_dtype_ops: {
                    let mut all = [OpCapability::all(); DType::N_DTYPES];
                    // Vulkan/SPIR-V f64 transcendentals crash or produce garbage
                    all[DType::F64 as usize].0 &=
                        !(OpCapability::EXP | OpCapability::EXP2 | OpCapability::LN
                        | OpCapability::LOG2 | OpCapability::SIN | OpCapability::COS
                        | OpCapability::POW);
                    all
                },
                has_native_exp2: false,
            },
            memory_pool_id: PoolId::from(usize::from(memory_pools.len()) - 1),
        };

        devices.push(super::Device::Vulkan(dev));
    }

    Ok(())
}
