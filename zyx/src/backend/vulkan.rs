// Copyright (C) 2025 zk4x
// SPDX-License-Identifier: LGPL-3.0-only WITH Classpath-exception-2.0

//! Vulkan backend using raw libloading FFI (no ash) with worker-thread dispatch.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use crate::Map;
use crate::Set;
use std::ffi::{CStr, CString};
use std::sync::{
    Arc, Mutex, OnceLock,
    atomic::{AtomicU64, Ordering},
    mpsc::{Receiver, Sender, channel},
};

use libloading::Library;
use nanoserde::DeJson;
use std::time::Instant;

use crate::kernel::{GPUOp, Op, OpId, SpirvOp};
use crate::{
    DType,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::Kernel,
    shape::Dim,
    slab::Slab,
};

use super::{
    AllocPlan, ChunkId, Cmd, DTypeCapability, DeviceInfo, DeviceProgramId, GwsDim, LaunchArg, Placement, PlanDim, Pool, Shard,
};

// ── Global state ──────────────────────────────────────────────────────────────

static VULKAN_POOLS: OnceLock<Vec<Mutex<VulkanMemoryPool>>> = OnceLock::new();
static VULKAN_DEVICES: OnceLock<Vec<Mutex<VulkanDevice>>> = OnceLock::new();

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
const VK_API_VERSION_1_2: u32 = (1 << 22) | (2 << 12);
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
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2: u32 = 1000059000;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES: u32 = 1000082000;
const VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR: u32 = 1000141000;

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
struct VkPushConstantRange {
    stageFlags: u32,
    offset: u32,
    size: u32,
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
    pPushConstantRanges: *const VkPushConstantRange,
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
    _pad_to_limits: [u8; 8],
    _limits_prefix: [u8; 216],
    max_compute_shared_memory_size: u32,
    max_compute_work_group_count: [u32; 3],
    max_compute_work_group_invocations: u32,
    max_compute_work_group_size: [u32; 3],
    _limits_suffix: [u8; 256],
    _sparse: [u8; 20],
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
#[repr(C)]
struct VkPhysicalDeviceFeatures2 {
    sType: u32,
    pNext: *mut std::ffi::c_void,
    features: [u32; 55],
}
#[repr(C)]
struct VkPhysicalDeviceShaderFloat16Int8Features {
    sType: u32,
    pNext: *mut std::ffi::c_void,
    shaderFloat16: u32,
    shaderInt8: u32,
}
#[repr(C)]
struct VkPhysicalDeviceShaderBfloat16FeaturesKHR {
    sType: u32,
    pNext: *mut std::ffi::c_void,
    shaderBFloat16Type: u32,
    shaderBFloat16DotProduct: u32,
    shaderBFloat16CooperativeMatrix: u32,
}
#[repr(C)]
#[derive(Clone)]
struct VkExtensionProperties {
    extensionName: [i8; 256],
    specVersion: u32,
}

// ── Config ───────────────────────────────────────────────────────────────────

#[derive(DeJson, Debug, Default)]
#[nserde(default)]
pub struct VulkanConfig {
    device_ids: Option<Vec<i32>>,
}

// ── Worker-thread command enum ───────────────────────────────────────────────

/// Pending commands accumulate until the micro-batch window is flushed.
const MICRO_BATCH_WINDOW: usize = 100;

enum VulkanCommand {
    Allocate {
        bytes: Dim,
        reply: Sender<Result<ChunkId, BackendError>>,
    },
    /// Put a buffer on the free list for stable-address reuse. Frees
    /// nothing; VRAM is reclaimed only by `Dispose`.
    Release {
        buffer_id: ChunkId,
    },
    /// Free every free-list buffer at once (submit + full drain first).
    /// The only VRAM reclamation; every released id becomes invalid.
    Dispose,
    /// All-or-nothing claim of free-list ids for stable addresses.
    TryReuse {
        buffer_ids: Set<ChunkId>,
        reply: Sender<Result<bool, BackendError>>,
    },
    /// Async copy into this pool's buffer (host-mapped memcpy on the worker,
    /// ordered against pending/in-flight GPU work). The staging chunk is
    /// caller-owned: the caller releases it after the reply, holding the
    /// host lock throughout. The reply fires after the memcpy.
    Copy {
        src_ptr: *const u8,
        bytes: usize,
        dst_buf: ChunkId,
        reply: Sender<Result<(), BackendError>>,
    },
    /// Blocking read-back: the reply is sent after the data arrived in host
    /// memory. This is a sync point — all pending work is submitted and
    /// every in-flight batch is drained first.
    PoolToHost {
        src: ChunkId,
        dst: *mut u8,
        bytes: usize,
        reply: Sender<Result<(), BackendError>>,
    },
    Compile {
        kernel: Box<Kernel>,
        debug_asm: bool,
        reply: Sender<Result<DeviceProgramId, BackendError>>,
    },
    /// Timed launch for autotune: flush + drain first (uncontended timing),
    /// then launch solo and measure enqueue-to-fence.
    LaunchTimed {
        program_id: DeviceProgramId,
        args: Vec<LaunchArg>,
        reply: Sender<Result<u64, BackendError>>,
    },
    /// Partition replay: one roundtrip per replay. Carries schedule's
    /// decisions (commands, per-def alloc plans, deaths) plus the
    /// boundary table and symbolic values. The worker flushes prior
    /// work, drains (proving bound slots and free-list chunks complete),
    /// allocates defs, records launches into the micro-batch window, and
    /// replies after submit — kernels still running (async). The next
    /// queue-touching command fences first.
    Replay {
        cmds: Vec<Cmd>,
        allocs: Vec<Vec<AllocPlan>>,
        bound: Vec<(OpId, ChunkId, usize, usize)>,
        vars: Vec<(OpId, Constant)>,
        deaths: Vec<Vec<OpId>>,
        reply: Sender<Result<Vec<(OpId, ChunkId, usize)>, BackendError>>,
    },
    ReleaseProgram(DeviceProgramId),
}

unsafe impl Send for VulkanCommand {}

enum Pending {
    Launch {
        program_id: DeviceProgramId,
        args: Vec<ResolvedArg>,
    },
}

/// A launch arg resolved to worker-local handles: buffer args carry
/// their chunk plus the live region (offset, length) on it — descriptors
/// index the buffer table at record time. Deliberately NOT a Placement:
/// an `Arc<Placement>` dropped here would run `Placement::drop` →
/// `pool.release` → send `Release` to this same worker and double-free
/// the buffer. Plain ids; ownership stays caller-side.
enum ResolvedArg {
    Buffer { chunk: ChunkId, offset: usize, len: usize },
    Variable(Constant),
}
/// One batched submission: the window's command buffers were recorded and
/// submitted with a single fence. Resources are freed once the fence signals
/// (sweep/drain) — never behind in-flight GPU work.
struct InFlight {
    fence: VkFence,
    cmds: Vec<VkCommandBuffer>,
    desc_sets: Vec<VkDescriptorSet>,
}

/// Single backend initializer: builds pools + devices together in one pass,
/// publishes both tables. Reads config directly; no init locks.
fn backend() -> Result<(&'static Vec<Mutex<VulkanMemoryPool>>, &'static Vec<Mutex<VulkanDevice>>), BackendError> {
    if let Some(pools) = VULKAN_POOLS.get()
        && let Some(devs) = VULKAN_DEVICES.get()
    {
        return Ok((pools, devs));
    }
    let config = super::config();
    let debug_dev = super::debug_backends();
    let pools = ensure_pool_table(&config.vulkan, debug_dev)?;
    let mut devs = Vec::with_capacity(pools.len());
    for (idx, pool) in pools.iter().enumerate() {
        let pool_id = Pool::Vulkan(u16::try_from(idx).expect("So many Vulkan devices..."));
        let guard = super::lock(pool_id, pool);
        let tx = guard.tx.clone();
        let dev_info = Arc::new(guard.dev_info.clone());
        drop(guard);
        devs.push(Mutex::new(VulkanDevice { tx, dev_info, memory_pool: pool_id }));
    }
    let _ = VULKAN_POOLS.set(pools);
    let _ = VULKAN_DEVICES.set(devs);
    match (VULKAN_POOLS.get(), VULKAN_DEVICES.get()) {
        (Some(pools), Some(devs)) => Ok((pools, devs)),
        _ => Err(BackendError { status: ErrorStatus::Initialization, context: "Vulkan init failed".into() }),
    }
}

pub(super) fn pool(id: u16) -> Result<&'static Mutex<VulkanMemoryPool>, BackendError> {
    backend()?.0.get(id as usize).ok_or_else(|| no_pool(id))
}

pub(super) fn pool_count() -> u16 {
    backend().map(|(pools, _)| pools.len() as u16).unwrap_or(0)
}

fn no_pool(id: u16) -> BackendError {
    BackendError { status: ErrorStatus::Initialization, context: format!("Pool::Vulkan({id}) is not available").into() }
}

pub struct VulkanMemoryPool {
    tx: Sender<VulkanCommand>,
    free_bytes: Arc<AtomicU64>,
    dev_info: DeviceInfo,
}

impl std::fmt::Debug for VulkanMemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanMemoryPool").field("free_bytes", &self.free_bytes).finish()
    }
}

impl VulkanMemoryPool {
    pub(super) fn free_bytes(&self) -> Dim {
        self.free_bytes.load(Ordering::SeqCst) as i64
    }
    pub(super) fn allocate(&mut self, bytes: Dim) -> Result<ChunkId, BackendError> {
        let (reply, rx) = channel();
        self.tx.send(VulkanCommand::Allocate { bytes, reply }).unwrap();
        rx.recv().unwrap()
    }
    /// Put a buffer on the free list for stable-address reuse. Frees
    /// nothing; VRAM is reclaimed only by [`VulkanMemoryPool::dispose`].
    pub(super) fn release(&mut self, buffer_id: ChunkId) {
        self.tx.send(VulkanCommand::Release { buffer_id }).unwrap();
    }
    /// Free every free-list buffer at once (submit + full drain first).
    /// The only VRAM reclamation; every released id becomes invalid.
    pub(super) fn dispose(&mut self) {
        self.tx.send(VulkanCommand::Dispose).unwrap();
    }
    /// Tries to reuse existing allocations, all or nothing: every id must
    /// be on the free list (stable addresses) or nothing is claimed.
    pub(super) fn try_reuse_allocations(&mut self, buffer_ids: &Set<ChunkId>) -> bool {
        let (reply, reply_rx) = channel();
        self.tx.send(VulkanCommand::TryReuse { buffer_ids: buffer_ids.clone(), reply }).unwrap();
        reply_rx.recv().unwrap().unwrap_or(false)
    }
    /// Blocking read-back (sync point).
    pub(super) fn pool_to_host(&mut self, src: ChunkId, dst: &mut [u8]) -> Result<(), BackendError> {
        let (reply, rx) = channel();
        self.tx.send(VulkanCommand::PoolToHost { src, dst: dst.as_mut_ptr(), bytes: dst.len(), reply }).unwrap();
        rx.recv().unwrap()
    }
}

// ── Program ──────────────────────────────────────────────────────────────────

struct VulkanProgram {
    pipeline: VkPipeline,
    pipeline_layout: VkPipelineLayout,
    desc_layout: VkDescriptorSetLayout,
    push_constants_size: u32,
    gws: Vec<GwsDim>,
}

// ── Buffer ───────────────────────────────────────────────────────────────────

/// A buffer in the Vulkan memory pool: a device buffer with mapped host pointer.
#[derive(Debug)]
pub(super) struct VulkanBuffer {
    buf: VkBuffer,
    mem: VkDeviceMemory,
    ptr: *mut u8,
    bytes: usize,
}

// ── Device ───────────────────────────────────────────────────────────────────

pub struct VulkanDevice {
    tx: Sender<VulkanCommand>,
    dev_info: Arc<DeviceInfo>,
    memory_pool: Pool,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanDevice").field("dev_info", &self.dev_info).field("memory_pool", &self.memory_pool).finish()
    }
}

impl VulkanDevice {
    pub(super) fn info(&self) -> Arc<DeviceInfo> {
        self.dev_info.clone()
    }
    pub(super) const fn free_compute(&self) -> u128 {
        1_000_000_000_000
    }
    pub(super) fn release(&mut self, program_id: DeviceProgramId) {
        self.tx.send(VulkanCommand::ReleaseProgram(program_id)).unwrap();
    }
    pub(super) fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<DeviceProgramId, BackendError> {
        let rendered = kernel.render()?;
        let (reply, rx) = channel();
        self.tx.send(VulkanCommand::Compile { kernel: Box::new(rendered), debug_asm, reply }).unwrap();
        rx.recv().unwrap()
    }
    /// Timed launch for autotune: the worker's pending window is submitted
    /// and every in-flight batch drained first, then the kernel runs solo and
    /// the wall-clock nanos are measured around submit-to-fence.
    pub(super) fn launch_timed(&mut self, program_id: DeviceProgramId, args: &[LaunchArg]) -> Result<u64, BackendError> {
        let (reply, rx) = channel();
        self.tx.send(VulkanCommand::LaunchTimed { program_id, args: args.to_vec(), reply }).unwrap();
        rx.recv().unwrap()
    }

    /// Copy executing a transfer into this device's pool. Every source
    /// stages through a proper [`Pool::Host`] chunk — never a temp `Vec`:
    /// the source lands in the staging chunk first (device sources via
    /// their own ordered read-back, host/disk via memcpy), then the worker
    /// memcpys chunk → device buffer in channel order and releases the
    /// staging chunk itself. The host lock is held across both blocking
    /// roundtrips: the staging pointer stays valid. The worker never
    /// takes it. Single-shard placements only.
    pub fn copy(&self, src: &Placement, dst: &Placement, bytes: Dim) -> Result<(), BackendError> {
        debug_assert!(bytes >= 0, "Vulkan copy of negative bytes");
        let [src_shard] = &src.shards[..] else {
            todo!("Vulkan copy of multi-shard source placement")
        };
        let [dst_shard] = &dst.shards[..] else {
            todo!("Vulkan copy of multi-shard destination placement")
        };
        debug_assert_eq!(dst_shard.pool, self.memory_pool, "Vulkan copy destination is not on this device");
        let dead = |_| BackendError { status: ErrorStatus::MemoryCopyP2P, context: "vulkan worker thread died".into() };
        let dead_rx = |_: std::sync::mpsc::RecvError| BackendError {
            status: ErrorStatus::MemoryCopyP2P,
            context: "vulkan worker hung up".into(),
        };
        let stage = Pool::Host.allocate(bytes)?;
        let host = super::host::pool();
        let mut hpool = super::lock(Pool::Host, host);
        let stage_ptr = hpool.buffer_ptr_mut(stage);
        let n = bytes as usize;
        let stage_slice = unsafe { std::slice::from_raw_parts_mut(stage_ptr, n) };
        match src_shard.pool {
            p if p == self.memory_pool => {
                let Pool::Vulkan(id) = p else {
                    unreachable!("Vulkan copy source pool is not Vulkan")
                };
                super::lock(self.memory_pool, pool(id)?).pool_to_host(src_shard.chunk, stage_slice)?;
            }
            Pool::Host => {
                let src_bytes = hpool.get_buffer(src_shard.chunk);
                debug_assert!(src_bytes.len() >= n, "Vulkan copy source host buffer is short");
                unsafe { std::ptr::copy_nonoverlapping(src_bytes.as_ptr(), stage_ptr, n) };
            }
            #[cfg(unix)]
            Pool::Disk => {
                // Straight from the file mapping into the staging chunk.
                // The chunk stays mapped across the memcpy (released
                // caller-side with the plan's deaths).
                let disk = super::disk::pool();
                let dpool = super::lock(Pool::Disk, disk);
                let (ptr, extent) = dpool.mapped_ptr(src_shard.chunk);
                let m = bytes.min(extent);
                debug_assert!(m >= 0, "Vulkan copy mapped extent is negative");
                unsafe { std::ptr::copy_nonoverlapping(ptr, stage_ptr, m as usize) };
            }
            #[cfg(windows)]
            Pool::Disk => todo!("Vulkan copy from disk on windows"),
            p => {
                // Other device pools read back through their own ordered path.
                p.pool_to_host(src_shard.chunk, stage_slice)?;
            }
        }
        let (reply, reply_rx) = channel();
        self.tx.send(VulkanCommand::Copy { src_ptr: stage_ptr, bytes: n, dst_buf: dst_shard.chunk, reply }).map_err(dead)?;
        let out = reply_rx.recv().map_err(dead_rx)?;
        // Staging is caller-owned: release it through the held guard
        // (Pool::release would re-lock the held host pool).
        hpool.release(stage);
        out
    }
}

/// A scheduled Vulkan partition: commands plus per-def allocation plans
/// and deaths. No queues, no wait edges: the worker records every launch
/// into one in-order micro-batch window on its single queue, so program
/// order is the ordering guarantee. Replay resolves plans against runtime
/// state (chunks) and submits; it computes nothing.
#[derive(Debug)]
pub(crate) struct VulkanPartition {
    pub(crate) cmds: Vec<Cmd>,
    deaths: Vec<Vec<OpId>>,
    allocs: Vec<Vec<AllocPlan>>,
    pub(crate) dev: u16,
}

impl VulkanDevice {
    pub(crate) fn schedule(cmds: Vec<Cmd>, outputs: &Set<OpId>, live_out: Set<OpId>, dev: u16) -> VulkanPartition {
        // Deaths: a slot dies at its last read unless pinned (a plan
        // output or read after this partition).
        // output or read after this partition).
        let mut last_use: Map<OpId, usize> = Map::default();
        for (idx, cmd) in cmds.iter().enumerate() {
            for r in cmd.reads() {
                last_use.insert(r, idx);
            }
        }
        let mut pinned = outputs.clone();
        pinned.extend(live_out);
        let mut deaths: Vec<Vec<OpId>> = vec![Vec::new(); cmds.len()];
        for (slot, idx) in last_use {
            if !pinned.contains(&slot) {
                deaths[idx].push(slot);
            }
        }
        // Allocation pairing is structural: a def reuses a dead slot's
        // chunk only for an identical (dtype, dims) spec — equal bytes
        // by construction, no evaluation. Each dead chunk is paired at
        // most once. Reuse scan covers deaths strictly before this
        // command (a slot dying here may be read by this very launch).
        // Death order is static; the first structural match wins.
        // Aliases are transparent via union-find over rebinds.
        let mut parent: Map<OpId, OpId> = Map::default();
        let canon = |mut slot: OpId, parent: &Map<OpId, OpId>| -> OpId {
            while let Some(&p) = parent.get(&slot) {
                slot = p;
            }
            slot
        };
        let mut spec: Map<OpId, (DType, Vec<PlanDim>)> = Map::default();
        let mut paired: Set<OpId> = Set::default();
        let mut allocs: Vec<Vec<AllocPlan>> = Vec::with_capacity(cmds.len());
        for (idx, cmd) in cmds.iter().enumerate() {
            match cmd {
                Cmd::Launch { outputs: defs, .. } => {
                    let mut plans: Vec<AllocPlan> = Vec::with_capacity(defs.len());
                    for (slot, dtype, dims) in defs {
                        let mut reuse = None;
                        for dead_list in deaths.iter().take(idx) {
                            for dead in dead_list {
                                if paired.contains(dead) {
                                    continue;
                                }
                                if spec.get(dead).is_some_and(|s| s.0 == *dtype && s.1 == *dims) {
                                    reuse = Some(*dead);
                                    break;
                                }
                            }
                            if reuse.is_some() {
                                break;
                            }
                        }
                        if let Some(dead) = reuse {
                            paired.insert(dead);
                            plans.push(AllocPlan::Reuse(dead));
                        } else {
                            plans.push(AllocPlan::Fresh);
                        }
                        spec.insert(*slot, (*dtype, dims.clone()));
                    }
                    allocs.push(plans);
                }
                Cmd::Alias { class, to } => {
                    // Zero-cost rebind: no executable, no queue traffic.
                    // Union the slots so later tracking sees one value.
                    let root = canon(*to, &parent);
                    parent.insert(*class, root);
                    allocs.push(Vec::new());
                }
                Cmd::Copy { .. } => unreachable!("copies are Copy partitions, never device runs"),
            }
        }
        VulkanPartition { cmds, deaths, allocs, dev }
    }
}

impl VulkanPartition {
    pub(crate) fn replay(
        &self,
        dev: &mut VulkanDevice,
        resolved: &mut Map<OpId, Arc<Placement>>,
        vars: &Map<OpId, Constant>,
    ) -> Result<(), BackendError> {
        // Bound slots addressed to this pool; caller-side defs stay
        // unbound for the worker's per-command allocator.
        let my_pool = dev.memory_pool;
        let mut bound = Vec::with_capacity(resolved.len());
        for (slot, placement) in resolved.iter() {
            if let Some(shard) = placement.shards.iter().find(|s| s.pool == my_pool) {
                bound.push((*slot, shard.chunk, shard.offset, shard.len));
            }
        }
        let vars_vec: Vec<(OpId, Constant)> = vars.iter().map(|(s, c)| (*s, *c)).collect();
        let dead = |_| BackendError { status: ErrorStatus::KernelLaunch, context: "vulkan worker thread died".into() };
        let dead_rx = |_: std::sync::mpsc::RecvError| BackendError {
            status: ErrorStatus::KernelLaunch,
            context: "vulkan worker hung up".into(),
        };
        let (reply, reply_rx) = channel();
        dev.tx
            .send(VulkanCommand::Replay {
                cmds: self.cmds.clone(),
                allocs: self.allocs.clone(),
                bound,
                vars: vars_vec,
                deaths: self.deaths.clone(),
                reply,
            })
            .map_err(dead)?;
        for (slot, chunk, len) in reply_rx.recv().map_err(dead_rx)?? {
            resolved.insert(slot, Arc::new(Placement { shards: vec![Shard { pool: my_pool, chunk, offset: 0, len }] }));
        }
        for dead in self.deaths.iter().flatten() {
            resolved.remove(dead);
        }
        Ok(())
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

// ── Batched submission ───────────────────────────────────────────────────────

/// Records the pending micro-batch window and submits it as ONE batched
/// `vkQueueSubmit` with a single fence. The batch is pushed to `inflight`;
/// its resources are freed once the fence signals (`sweep_inflight` /
/// `drain_*`) — never behind in-flight GPU work. Commands are recorded in
/// program order on the single in-order queue, so no cross-command waits are
/// needed.
#[allow(clippy::too_many_arguments)]
/// Resolve one eager launch arg to a worker-local handle: buffer args
/// carry their chunk on this pool, variables carry their constant.
/// Single-pool launches only — a buffer arg with no shard resident here
/// is a missing Copy, never a default.
/// Resolve one eager launch arg to a worker-local handle: copy the
/// region fields, variables carry their constant. No pool lookup — the
/// region is fully described, single-pool by construction.
fn resolve_arg(arg: &LaunchArg) -> ResolvedArg {
    match arg {
        LaunchArg::Buffer { chunk, offset, len } => ResolvedArg::Buffer { chunk: *chunk, offset: *offset, len: *len },
        LaunchArg::Variable(c) => ResolvedArg::Variable(*c),
    }
}

fn submit_window(
    pending: &mut Vec<Pending>,
    queue: VkQueue,
    device: VkDevice,
    cmd_pool: VkCommandPool,
    desc_pool: VkDescriptorPool,
    buffers: &Slab<ChunkId, VulkanBuffer>,
    programs: &Slab<DeviceProgramId, VulkanProgram>,
    max_grid: &[Dim],
    inflight: &mut Vec<InFlight>,
    debug_dev: bool,
    vkAllocateDescriptorSets: unsafe extern "system" fn(
        VkDevice,
        *const VkDescriptorSetAllocateInfo,
        *mut VkDescriptorSet,
    ) -> VkResult,
    vkUpdateDescriptorSets: unsafe extern "system" fn(VkDevice, u32, *const VkWriteDescriptorSet, u32, *const std::ffi::c_void),
    vkAllocateCommandBuffers: unsafe extern "system" fn(
        VkDevice,
        *const VkCommandBufferAllocateInfo,
        *mut VkCommandBuffer,
    ) -> VkResult,
    vkBeginCommandBuffer: unsafe extern "system" fn(VkCommandBuffer, *const VkCommandBufferBeginInfo) -> VkResult,
    vkEndCommandBuffer: unsafe extern "system" fn(VkCommandBuffer) -> VkResult,
    vkCmdBindPipeline: unsafe extern "system" fn(VkCommandBuffer, u32, VkPipeline),
    vkCmdPushConstants: unsafe extern "system" fn(VkCommandBuffer, VkPipelineLayout, u32, u32, u32, *const std::ffi::c_void),
    vkCmdBindDescriptorSets: unsafe extern "system" fn(
        VkCommandBuffer,
        u32,
        VkPipelineLayout,
        u32,
        u32,
        *const VkDescriptorSet,
        u32,
        *const u32,
    ),
    vkCmdDispatch: unsafe extern "system" fn(VkCommandBuffer, u32, u32, u32),
    vkCreateFence: unsafe extern "system" fn(
        VkDevice,
        *const VkFenceCreateInfo,
        *const std::ffi::c_void,
        *mut VkFence,
    ) -> VkResult,
    vkQueueSubmit: unsafe extern "system" fn(VkQueue, u32, *const VkSubmitInfo, VkFence) -> VkResult,
) -> Result<(), BackendError> {
    if pending.is_empty() {
        return Ok(());
    }
    let mut cmds: Vec<VkCommandBuffer> = Vec::with_capacity(pending.len());
    let mut desc_sets: Vec<VkDescriptorSet> = Vec::with_capacity(pending.len());
    let mut submit_infos: Vec<VkSubmitInfo> = Vec::with_capacity(pending.len());
    for Pending::Launch { program_id, args } in pending.drain(..) {
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
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("vkAllocateDescriptorSets: {res}").into(),
            });
        }
        // Separate buffer args (descriptors) from variable args (push constants).
        // Buffer bindings are 0..nbuffers in param order; push-constant members
        // use the same std140 layout as the SPIR-V block, in param order.
        let mut buf_infos: Vec<VkDescriptorBufferInfo> = Vec::with_capacity(args.len());
        let mut push_constants: Vec<u8> = vec![0u8; prog.push_constants_size as usize];
        let mut push_off: u32 = 0;
        for arg in &args {
            match arg {
                ResolvedArg::Variable(constant) => {
                    let storage_bits = if constant.dtype() == crate::DType::Bool {
                        32
                    } else {
                        constant.dtype().bit_size()
                    };
                    let size = storage_bits as u32 / 8;
                    let align = if size >= 8 { 8 } else { 4 };
                    push_off = push_off.next_multiple_of(align);
                    let bytes = constant.to_le_bytes();
                    push_constants[push_off as usize..push_off as usize + bytes.len()].copy_from_slice(&bytes);
                    push_off += size;
                }
                ResolvedArg::Buffer { chunk, offset, len } => {
                    debug_assert!(buffers.contains_id(*chunk), "vulkan launch arg addresses an unknown buffer");
                    let resolved = &buffers[*chunk];
                    debug_assert!(!resolved.buf.is_null(), "vulkan launch arg addresses a null buffer");
                    debug_assert!(offset.saturating_add(*len) <= resolved.bytes, "vulkan launch arg region overruns its chunk");
                    buf_infos.push(VkDescriptorBufferInfo { buffer: resolved.buf, offset: *offset as u64, range: *len as u64 });
                }
            }
        }
        let n_buffers = buf_infos.len();
        let mut writes: Vec<VkWriteDescriptorSet> = Vec::with_capacity(n_buffers);
        for (i, buf_info) in buf_infos.iter().enumerate() {
            writes.push(VkWriteDescriptorSet {
                sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext: std::ptr::null(),
                dstSet: desc_set,
                dstBinding: i as u32,
                dstArrayElement: 0,
                descriptorCount: 1,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pImageInfo: std::ptr::null(),
                pBufferInfo: buf_info,
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
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("vkAllocateCommandBuffers: {res}").into(),
            });
        }

        let begin = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: std::ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo: std::ptr::null(),
        };
        let res = unsafe { vkBeginCommandBuffer(cmd, &begin) };
        if res != VK_SUCCESS {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("vkBeginCommandBuffer: {res}").into(),
            });
        }

        let default_gws = GwsDim::Const(1);
        let grid = |gdim: &GwsDim| -> Dim {
            gdim.eval(&mut |ordinal| match &args[ordinal] {
                ResolvedArg::Variable(c) => c.as_dim().unwrap(),
                ResolvedArg::Buffer { .. } => unreachable!("gws param must be a Variable launch arg"),
            })
        };
        let gx = grid(prog.gws.first().unwrap_or(&default_gws));
        let gy = grid(prog.gws.get(1).unwrap_or(&default_gws));
        let gz = grid(prog.gws.get(2).unwrap_or(&default_gws));

        if gx <= 0 || gy <= 0 || gz <= 0 {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("dispatch dims non-positive: ({gx},{gy},{gz})").into(),
            });
        }
        if gx > max_grid[0] || gy > max_grid[1] || gz > max_grid[2] {
            return Err(BackendError {
                status: ErrorStatus::KernelLaunch,
                context: format!("grid dims ({gx},{gy},{gz}) exceed device max {max_grid:?}").into(),
            });
        }
        let (gx, gy, gz) = (u32::try_from(gx).unwrap(), u32::try_from(gy).unwrap(), u32::try_from(gz).unwrap());

        unsafe {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, prog.pipeline);
            if prog.push_constants_size > 0 {
                vkCmdPushConstants(
                    cmd,
                    prog.pipeline_layout,
                    VK_SHADER_STAGE_COMPUTE_BIT,
                    0,
                    push_constants.len() as u32,
                    push_constants.as_ptr().cast(),
                );
            }
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
            return Err(BackendError { status: ErrorStatus::KernelLaunch, context: format!("vkEndCommandBuffer: {res}").into() });
        }

        submit_infos.push(VkSubmitInfo {
            sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: std::ptr::null(),
            waitSemaphoreCount: 0,
            pWaitSemaphores: std::ptr::null(),
            pWaitDstStageMask: std::ptr::null(),
            commandBufferCount: 1,
            pCommandBuffers: &cmd,
            signalSemaphoreCount: 0,
            pSignalSemaphores: std::ptr::null(),
        });
        cmds.push(cmd);
        desc_sets.push(desc_set);
    }

    let fence_ci = VkFenceCreateInfo { sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: std::ptr::null(), flags: 0 };
    let mut fence = std::ptr::null_mut();
    let res = unsafe { vkCreateFence(device, &fence_ci, std::ptr::null(), &mut fence) };
    if res != VK_SUCCESS {
        return Err(BackendError { status: ErrorStatus::KernelLaunch, context: format!("vkCreateFence: {res}").into() });
    }

    // pCommandBuffers must point at stable storage — cmds outlives the submit.
    // NOTE: each entry must address its OWN command buffer (cmds[i]), not
    // the loop variable: every submit_info created in the loop above holds
    // the address of the same stack slot, which ends up holding only the
    // last recorded buffer (earlier kernels would silently never run).
    for (info, cmd) in submit_infos.iter_mut().zip(cmds.iter()) {
        info.pCommandBuffers = cmd as *const VkCommandBuffer;
    }
    let res = unsafe { vkQueueSubmit(queue, submit_infos.len() as u32, submit_infos.as_ptr(), fence) };
    if res != VK_SUCCESS {
        if debug_dev {
            println!("[vulkan] batched submission error: vkQueueSubmit: {res}");
        }
        return Err(BackendError { status: ErrorStatus::KernelLaunch, context: format!("vkQueueSubmit: {res}").into() });
    }

    inflight.push(InFlight { fence, cmds, desc_sets });
    Ok(())
}

/// Frees a completed batch's fence, command buffers and descriptor sets.
fn destroy_batch(
    device: VkDevice,
    cmd_pool: VkCommandPool,
    desc_pool: VkDescriptorPool,
    batch: InFlight,
    vkFreeCommandBuffers: unsafe extern "system" fn(VkDevice, VkCommandPool, u32, *const VkCommandBuffer),
    vkFreeDescriptorSets: unsafe extern "system" fn(VkDevice, VkDescriptorPool, u32, *const VkDescriptorSet) -> VkResult,
    vkDestroyFence: unsafe extern "system" fn(VkDevice, VkFence, *const std::ffi::c_void),
) {
    if !batch.fence.is_null() {
        unsafe { vkDestroyFence(device, batch.fence, std::ptr::null()) };
    }
    if !batch.cmds.is_empty() {
        unsafe { vkFreeCommandBuffers(device, cmd_pool, batch.cmds.len() as u32, batch.cmds.as_ptr()) };
    }
    if !batch.desc_sets.is_empty() {
        unsafe { vkFreeDescriptorSets(device, desc_pool, batch.desc_sets.len() as u32, batch.desc_sets.as_ptr()) };
    }
}

/// Waits for and destroys every in-flight batch (a full sync point).
#[allow(clippy::too_many_arguments)]
fn drain_all(
    inflight: &mut Vec<InFlight>,
    device: VkDevice,
    cmd_pool: VkCommandPool,
    desc_pool: VkDescriptorPool,
    vkWaitForFences: unsafe extern "system" fn(VkDevice, u32, *const VkFence, u32, u64) -> VkResult,
    vkFreeCommandBuffers: unsafe extern "system" fn(VkDevice, VkCommandPool, u32, *const VkCommandBuffer),
    vkFreeDescriptorSets: unsafe extern "system" fn(VkDevice, VkDescriptorPool, u32, *const VkDescriptorSet) -> VkResult,
    vkDestroyFence: unsafe extern "system" fn(VkDevice, VkFence, *const std::ffi::c_void),
) {
    for batch in inflight.drain(..) {
        let res = unsafe { vkWaitForFences(device, 1, &batch.fence, 1, u64::MAX) };
        if res != VK_SUCCESS {
            // The device is in a broken state; still reap the host-side
            // resources and let the error surface at the sync point.
        }
        destroy_batch(device, cmd_pool, desc_pool, batch, vkFreeCommandBuffers, vkFreeDescriptorSets, vkDestroyFence);
    }
}

/// Waits for and destroys every in-flight batch that touched `buffer_id`.
/// Single in-order queue: waiting the batch fence completes all earlier
/// batches too, so every use of the buffer up to the release is covered.
#[allow(clippy::too_many_arguments)]
/// Reaps batches whose fence has signaled (a cheap `vkGetFenceStatus` poll).
#[allow(clippy::too_many_arguments)]
fn sweep_inflight(
    inflight: &mut Vec<InFlight>,
    device: VkDevice,
    vkGetFenceStatus: unsafe extern "system" fn(VkDevice, VkFence) -> VkResult,
    cmd_pool: VkCommandPool,
    desc_pool: VkDescriptorPool,
    vkFreeCommandBuffers: unsafe extern "system" fn(VkDevice, VkCommandPool, u32, *const VkCommandBuffer),
    vkFreeDescriptorSets: unsafe extern "system" fn(VkDevice, VkDescriptorPool, u32, *const VkDescriptorSet) -> VkResult,
    vkDestroyFence: unsafe extern "system" fn(VkDevice, VkFence, *const std::ffi::c_void),
) {
    let mut remaining = Vec::new();
    for batch in inflight.drain(..) {
        if unsafe { vkGetFenceStatus(device, batch.fence) } == VK_SUCCESS {
            destroy_batch(device, cmd_pool, desc_pool, batch, vkFreeCommandBuffers, vkFreeDescriptorSets, vkDestroyFence);
        } else {
            remaining.push(batch);
        }
    }
    *inflight = remaining;
}

// ── Initialization ───────────────────────────────────────────────────────────

#[allow(clippy::unnecessary_wraps)]
pub(super) fn ensure_pool_table(config: &VulkanConfig, debug_dev: bool) -> Result<Vec<Mutex<VulkanMemoryPool>>, BackendError> {
    let mut pools: Vec<Mutex<VulkanMemoryPool>> = Vec::new();
    if let Some(ids) = &config.device_ids
        && ids.is_empty()
    {
        if debug_dev {
            println!("[vulkan] configured out");
        }
        return Ok(pools);
    }

    let vulkan_paths = [
        "libvulkan.so.1",
        "libvulkan.so",
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
    let lib = vulkan_paths.into_iter().find_map(|path| unsafe { Library::new(path) }.ok()).ok_or_else(|| {
        if debug_dev {
            println!("[vulkan] libvulkan.so not found");
        }
        BackendError { status: ErrorStatus::DyLibNotFound, context: "[vulkan] libvulkan.so not found.".into() }
    })?;
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
        if debug_dev {
            println!("[vulkan] instance: {res}");
        }
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
    let vkGetPhysicalDeviceFeatures2: unsafe extern "system" fn(VkPhysicalDevice, *mut VkPhysicalDeviceFeatures2) =
        get_inst_proc!("vkGetPhysicalDeviceFeatures2");
    let vkEnumerateDeviceExtensionProperties: unsafe extern "system" fn(
        VkPhysicalDevice,
        *const i8,
        *mut u32,
        *mut VkExtensionProperties,
    ) -> VkResult = get_inst_proc!("vkEnumerateDeviceExtensionProperties");
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
    if debug_dev {
        println!("[vulkan] physical devices: {gpu_count}");
    }
    let mut gpus: Vec<VkPhysicalDevice> = vec![std::ptr::null_mut(); gpu_count as usize];
    let _ = unsafe { vkEnumeratePhysicalDevices(instance, &mut gpu_count, gpus.as_mut_ptr()) };

    // device_ids filter: Some([0,1]) means use those indices, None means all
    let indices: Vec<usize> = match &config.device_ids {
        Some(ids) => {
            let v: Vec<usize> = ids.iter().map(|&i| i as usize).filter(|&i| i < gpus.len()).collect();
            if debug_dev {
                println!("[vulkan] device_ids {:?}, indices {:?}, gpus {}", ids, v, gpus.len());
            }
            v
        }
        None => {
            let mut v = Vec::new();
            for (i, gpu) in gpus.iter().enumerate() {
                let mut props: VkPhysicalDeviceProperties = unsafe { std::mem::zeroed() };
                unsafe { vkGetPhysicalDeviceProperties(*gpu, &mut props) };
                v.push(i);
            }
            v
        }
    };

    for &gpu_i in &indices {
        let gpu = gpus[gpu_i];
        let mut props: VkPhysicalDeviceProperties = unsafe { std::mem::zeroed() };
        unsafe { vkGetPhysicalDeviceProperties(gpu, &mut props) };
        let name = {
            let cstr = unsafe { CStr::from_ptr(props.device_name.as_ptr() as *const i8) };
            cstr.to_string_lossy().into_owned()
        };

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

        // shaderBFloat16Type can be reported as supported even when the
        // VK_KHR_shader_bfloat16 extension isn't actually available, so gate on
        // the extension being enumerated explicitly.
        let ext_supports_bf16 = {
            let mut count: u32 = 0;
            unsafe { vkEnumerateDeviceExtensionProperties(gpu, std::ptr::null(), &mut count, std::ptr::null_mut()) };
            let mut props = vec![unsafe { std::mem::zeroed::<VkExtensionProperties>() }; count as usize];
            unsafe { vkEnumerateDeviceExtensionProperties(gpu, std::ptr::null(), &mut count, props.as_mut_ptr()) };
            props.iter().any(|p| {
                let name = unsafe { CStr::from_ptr(p.extensionName.as_ptr()) };
                name.to_bytes() == b"VK_KHR_shader_bfloat16"
            })
        };
        if debug_dev {
            println!("[vulkan] {name}: VK_KHR_shader_bfloat16 extension: {ext_supports_bf16}");
        }

        let bf16_device_features;
        let has_shader_bf16;
        // VkPhysicalDeviceFeatures index of shaderInt64 (index math is i64).
        const SHADER_INT64_IDX: usize = 38;
        let mut has_shader_int64 = false;
        let has_shader_float16 = if vkGetPhysicalDeviceFeatures2 as usize != 0 {
            let mut float16_features = VkPhysicalDeviceShaderFloat16Int8Features {
                sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
                pNext: std::ptr::null_mut(),
                shaderFloat16: 0,
                shaderInt8: 0,
            };
            let mut bf16_features = VkPhysicalDeviceShaderBfloat16FeaturesKHR {
                sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
                pNext: &mut float16_features as *mut VkPhysicalDeviceShaderFloat16Int8Features as *mut std::ffi::c_void,
                shaderBFloat16Type: 0,
                shaderBFloat16DotProduct: 0,
                shaderBFloat16CooperativeMatrix: 0,
            };
            let mut features2 = VkPhysicalDeviceFeatures2 {
                sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                pNext: &mut bf16_features as *mut VkPhysicalDeviceShaderBfloat16FeaturesKHR as *mut std::ffi::c_void,
                features: [0u32; 55],
            };
            unsafe { vkGetPhysicalDeviceFeatures2(gpu, &mut features2) };
            has_shader_int64 = features2.features[SHADER_INT64_IDX] != 0;
            has_shader_bf16 = ext_supports_bf16 && bf16_features.shaderBFloat16Type != 0;
            bf16_device_features = if has_shader_bf16 {
                VkPhysicalDeviceShaderBfloat16FeaturesKHR {
                    sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
                    pNext: std::ptr::null_mut(),
                    shaderBFloat16Type: 1,
                    shaderBFloat16DotProduct: 0,
                    shaderBFloat16CooperativeMatrix: 0,
                }
            } else {
                VkPhysicalDeviceShaderBfloat16FeaturesKHR {
                    sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
                    pNext: std::ptr::null_mut(),
                    shaderBFloat16Type: 0,
                    shaderBFloat16DotProduct: 0,
                    shaderBFloat16CooperativeMatrix: 0,
                }
            };
            float16_features.shaderFloat16 != 0
        } else {
            has_shader_bf16 = false;
            bf16_device_features = VkPhysicalDeviceShaderBfloat16FeaturesKHR {
                sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR,
                pNext: std::ptr::null_mut(),
                shaderBFloat16Type: 0,
                shaderBFloat16DotProduct: 0,
                shaderBFloat16CooperativeMatrix: 0,
            };
            false
        };

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

        let ext_names: Vec<Vec<u8>> = {
            let mut exts = Vec::new();
            if has_shader_bf16 {
                exts.push(b"VK_KHR_shader_bfloat16\0".to_vec());
            }
            exts
        };
        let ext_ptrs: Vec<*const i8> = ext_names.iter().map(|s| s.as_ptr() as *const i8).collect();

        let dci_pnext: *mut std::ffi::c_void = if has_shader_bf16 {
            &bf16_device_features as *const VkPhysicalDeviceShaderBfloat16FeaturesKHR as *mut std::ffi::c_void
        } else {
            std::ptr::null_mut()
        };
        if debug_dev {
            println!("[vulkan] {name}: shaderInt64: {has_shader_int64}");
        }

        let mut enabled_features = [0u32; 55];
        if has_shader_int64 {
            enabled_features[SHADER_INT64_IDX] = 1;
        }
        let dci = VkDeviceCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext: dci_pnext,
            flags: 0,
            queueCreateInfoCount: 1,
            pQueueCreateInfos: &qci,
            enabledLayerCount: 0,
            ppEnabledLayerNames: std::ptr::null(),
            enabledExtensionCount: ext_ptrs.len() as u32,
            ppEnabledExtensionNames: ext_ptrs.as_ptr(),
            pEnabledFeatures: enabled_features.as_ptr() as *const std::ffi::c_void,
        };
        let mut device = std::ptr::null_mut();
        let res = unsafe { vkCreateDevice(gpu, &dci, std::ptr::null(), &mut device) };
        if res != VK_SUCCESS {
            if debug_dev {
                println!("[vulkan] {name}: device: {res}");
            }
            return Err(BackendError {
                status: ErrorStatus::Initialization,
                context: format!("[vulkan] {name}: device: {res}").into(),
            });
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
        let vkGetFenceStatus: unsafe extern "system" fn(VkDevice, VkFence) -> VkResult = ld!("vkGetFenceStatus");
        let vkDeviceWaitIdle: unsafe extern "system" fn(VkDevice) -> VkResult = ld!("vkDeviceWaitIdle");
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
        let vkCmdPushConstants: unsafe extern "system" fn(
            VkCommandBuffer,
            VkPipelineLayout,
            u32,
            u32,
            u32,
            *const std::ffi::c_void,
        ) = ld!("vkCmdPushConstants");
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
        // Built up-front so the worker thread can validate group lengths at
        // compile and launch time against the device grid limits.
        let dev_info = DeviceInfo {
            compute: 1_000_000_000_000,
            max_global_work_dims: max_wg_count.iter().map(|&c| Dim::from(c)).collect(),
            max_local_threads: max_wg_invocations,
            max_local_work_dims: vec![u32::from(max_wg_size[0]); max_wg_size.len()],
            preferred_vector_size: 4,
            local_mem_size: Dim::from(props.max_compute_shared_memory_size),
            max_register_bytes: 1024,
            tensor_cores: false,
            warp_size: 32,
            cc: [0, 0],
            dtype_capability: {
                let mut all = [DTypeCapability::all(); DType::N_DTYPES];
                // Vulkan/SPIR-V f64 transcendentals crash or produce garbage
                all[DType::F64 as usize] = all[DType::F64 as usize].exclude(
                    DTypeCapability::EXP
                        | DTypeCapability::EXP2
                        | DTypeCapability::LN
                        | DTypeCapability::LOG2
                        | DTypeCapability::SIN
                        | DTypeCapability::COS
                        | DTypeCapability::POW,
                );
                // Turing/NVIDIA driver crashes on BF16 compute even when
                // VK_KHR_shader_bfloat16 is enabled. SPIR-V codegen would
                // need explicit OpFConvert around GLSL.std.450 intrinsics.
                // Disable until that's implemented.
                all[DType::BF16 as usize] = DTypeCapability::none();
                if !has_shader_float16 {
                    all[DType::F16 as usize] = DTypeCapability::none();
                }
                all
            },
            has_native_exp2: false,
            supported_vec_lens: vec![2, 3, 4],
            tenstorrent: false,
            tile: [1, 1],
            tile_sizes: vec![],
            wmma_layouts: vec![],
            num_circular_buffers: 0,
            has_openmp: false,
        };

        std::thread::spawn({
            let free_bytes_atomic = Arc::clone(&free_bytes_atomic);
            let dev_info = Arc::new(dev_info.clone());
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

                let mut buffers: Slab<ChunkId, VulkanBuffer> = Slab::new();
                // Free-list ids for stable-address reuse (`Release`
                // inserts, `TryReuse`/`Allocate` claim, `Dispose`
                // frees). Slab entries stay so ChunkIds are monotonic and
                // never alias a different buffer.
                let mut free_set: Set<ChunkId> = Set::default();
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

                // Pending micro-batch window: launches accumulate here in
                // program order until MICRO_BATCH_WINDOW is reached (or a
                // sync point arrives), then the whole window is recorded and
                // submitted as ONE batched vkQueueSubmit.
                let mut pending: Vec<Pending> = Vec::new();
                // Submitted-but-maybe-in-flight batches, freed once their
                // fence signals (swept each loop iteration, drained at sync
                // points and buffer releases).
                let mut inflight: Vec<InFlight> = Vec::new();
                // Same-replay deaths: their last readers may still be in
                // flight, so they join the free list only at the next
                // full drain (Replay opening, PoolToHost, LaunchTimed,
                // Dispose), when completion is proven. No polling, ever.
                let mut pending_free: Vec<ChunkId> = Vec::new();
                // First async submission error since the last sync point.
                let mut last_error: Option<BackendError> = None;

                while let Ok(cmd) = rx.recv() {
                    // Poll completed batches: reap resources whose fence
                    // signaled. Cheap (vkGetFenceStatus) and usually a no-op.
                    sweep_inflight(
                        &mut inflight,
                        device,
                        vkGetFenceStatus,
                        cmd_pool,
                        desc_pool,
                        vkFreeCommandBuffers,
                        vkFreeDescriptorSets,
                        vkDestroyFence,
                    );
                    match cmd {
                        VulkanCommand::Allocate { bytes, reply } => {
                            // Best-fit from the free list first (stable
                            // addresses across repeats); else fresh.
                            let best = free_set
                                .iter()
                                .filter_map(|id| {
                                    let len = buffers[*id].bytes;
                                    (len >= bytes as usize).then_some((len, *id))
                                })
                                .min();
                            if let Some((_, id)) = best {
                                free_set.remove(&id);
                                let _ = reply.send(Ok(id));
                                continue;
                            }
                            let size = (bytes + 3) & !3;
                            let (buf, mem, ptr) = send_or_continue!(create_buffer(size as u64), reply);
                            let id = buffers.push(VulkanBuffer { buf, mem, ptr, bytes: bytes as usize });
                            free_bytes_atomic.fetch_sub(size as u64, Ordering::SeqCst);
                            let _ = reply.send(Ok(id));
                        }
                        VulkanCommand::Release { buffer_id } => {
                            // Put the id on the free list for
                            // stable-address reuse. Frees nothing: VRAM is
                            // reclaimed only by Dispose. A released id may
                            // still be in flight (async replays release
                            // early) — claims stay safe because every
                            // CPU-side use (Copy, PoolToHost) and every
                            // replay opening drains first, and GPU-side
                            // order comes from the single in-order queue.
                            if !buffers.contains_id(buffer_id) {
                                debug_assert!(false, "release of unknown Vulkan buffer {buffer_id:?}");
                                continue;
                            }
                            debug_assert!(!free_set.contains(&buffer_id), "double release of Vulkan buffer {buffer_id:?}");
                            free_set.insert(buffer_id);
                        }
                        VulkanCommand::Dispose => {
                            // The only reclamation: submit the pending
                            // window, then drain every in-flight batch —
                            // completion implies completion of every queued
                            // use of every free buffer. Then destroy the
                            // whole free list at once; every released id
                            // becomes invalid.
                            if let Err(err) = submit_window(
                                &mut pending,
                                queue,
                                device,
                                cmd_pool,
                                desc_pool,
                                &buffers,
                                &programs,
                                &dev_info.max_global_work_dims,
                                &mut inflight,
                                debug_dev,
                                vkAllocateDescriptorSets,
                                vkUpdateDescriptorSets,
                                vkAllocateCommandBuffers,
                                vkBeginCommandBuffer,
                                vkEndCommandBuffer,
                                vkCmdBindPipeline,
                                vkCmdPushConstants,
                                vkCmdBindDescriptorSets,
                                vkCmdDispatch,
                                vkCreateFence,
                                vkQueueSubmit,
                            ) && last_error.is_none()
                            {
                                last_error = Some(err);
                            }
                            drain_all(
                                &mut inflight,
                                device,
                                cmd_pool,
                                desc_pool,
                                vkWaitForFences,
                                vkFreeCommandBuffers,
                                vkFreeDescriptorSets,
                                vkDestroyFence,
                            );
                            free_set.extend(pending_free.drain(..));
                            for buffer_id in core::mem::take(&mut free_set) {
                                let VulkanBuffer { buf, mem, ptr, bytes } = unsafe { buffers.remove_and_return(buffer_id) };
                                if !ptr.is_null() {
                                    unsafe { vkUnmapMemory(device, mem) };
                                }
                                unsafe {
                                    vkDestroyBuffer(device, buf, std::ptr::null());
                                    vkFreeMemory(device, mem, std::ptr::null());
                                }
                                free_bytes_atomic.fetch_add(((bytes + 3) & !3) as u64, Ordering::SeqCst);
                            }
                        }
                        VulkanCommand::TryReuse { buffer_ids, reply } => {
                            // All-or-nothing: every id must be on the free
                            // list, else some address moved and the graph
                            // must be retraced.
                            let ok = buffer_ids.iter().all(|id| free_set.contains(id));
                            if ok {
                                for id in &buffer_ids {
                                    free_set.remove(id);
                                }
                            }
                            let _ = reply.send(Ok(ok));
                        }
                        VulkanCommand::Copy { src_ptr, bytes, dst_buf, reply } => {
                            // Host-mapped memcpy done on the worker. Full
                            // sync point (like PoolToHost): the destination
                            // chunk may have been claimed from the free
                            // list while its last reader was still in
                            // flight (async replays release early), so only
                            // a fence proves it safe to overwrite. Copies
                            // are off the hot path — correctness over
                            // pipelining here.
                            if let Err(err) = submit_window(
                                &mut pending,
                                queue,
                                device,
                                cmd_pool,
                                desc_pool,
                                &buffers,
                                &programs,
                                &dev_info.max_global_work_dims,
                                &mut inflight,
                                debug_dev,
                                vkAllocateDescriptorSets,
                                vkUpdateDescriptorSets,
                                vkAllocateCommandBuffers,
                                vkBeginCommandBuffer,
                                vkEndCommandBuffer,
                                vkCmdBindPipeline,
                                vkCmdPushConstants,
                                vkCmdBindDescriptorSets,
                                vkCmdDispatch,
                                vkCreateFence,
                                vkQueueSubmit,
                            ) && last_error.is_none()
                            {
                                last_error = Some(err);
                            }
                            drain_all(
                                &mut inflight,
                                device,
                                cmd_pool,
                                desc_pool,
                                vkWaitForFences,
                                vkFreeCommandBuffers,
                                vkFreeDescriptorSets,
                                vkDestroyFence,
                            );
                            if let Some(err) = last_error.take() {
                                let _ = reply.send(Err(err));
                                continue;
                            }
                            free_set.extend(pending_free.drain(..));
                            let VulkanBuffer { ptr, .. } = buffers[dst_buf];
                            unsafe { std::ptr::copy_nonoverlapping(src_ptr, ptr, bytes) };
                            // No release here: the caller releases the
                            // staging chunk after the reply, holding the
                            // host lock throughout (releasing here would
                            // deadlock on the caller's held lock).
                            let _ = reply.send(Ok(()));
                        }
                        VulkanCommand::PoolToHost { src, dst, bytes, reply } => {
                            // Sync point: submit everything pending, drain all
                            // in-flight batches, surface async submission
                            // errors, then read back.
                            if let Err(err) = submit_window(
                                &mut pending,
                                queue,
                                device,
                                cmd_pool,
                                desc_pool,
                                &buffers,
                                &programs,
                                &dev_info.max_global_work_dims,
                                &mut inflight,
                                debug_dev,
                                vkAllocateDescriptorSets,
                                vkUpdateDescriptorSets,
                                vkAllocateCommandBuffers,
                                vkBeginCommandBuffer,
                                vkEndCommandBuffer,
                                vkCmdBindPipeline,
                                vkCmdPushConstants,
                                vkCmdBindDescriptorSets,
                                vkCmdDispatch,
                                vkCreateFence,
                                vkQueueSubmit,
                            ) && last_error.is_none()
                            {
                                last_error = Some(err);
                            }
                            drain_all(
                                &mut inflight,
                                device,
                                cmd_pool,
                                desc_pool,
                                vkWaitForFences,
                                vkFreeCommandBuffers,
                                vkFreeDescriptorSets,
                                vkDestroyFence,
                            );
                            free_set.extend(pending_free.drain(..));
                            if let Some(err) = last_error.take() {
                                let _ = reply.send(Err(err));
                                continue;
                            }
                            let VulkanBuffer { ptr, .. } = buffers[src];
                            unsafe { std::ptr::copy_nonoverlapping(ptr, dst, bytes) };
                            let _ = reply.send(Ok(()));
                        }
                        VulkanCommand::Compile { kernel, debug_asm: _debug_asm, reply } => {
                            // Debugging TBD: render takes no debug flag, so
                            // SPIR-V disassembly is off here.
                            let mut order = kernel.ops_in_order();
                            let Some(GPUOp::Params(params)) = order.next().and_then(Op::as_gpu) else {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "head op is not Params".into(),
                                }));
                                continue;
                            };
                            let Some(GPUOp::Grid(gws)) = order.next().and_then(Op::as_gpu) else {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "second op is not Grid".into(),
                                }));
                                continue;
                            };
                            let Some(GPUOp::LocalWorkSize(lws)) = order.next().and_then(Op::as_gpu) else {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "third op is not LocalWorkSize".into(),
                                }));
                                continue;
                            };
                            let Some(Op::Spirv(words)) = order.next() else {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "fourth op is not Spirv".into(),
                                }));
                                continue;
                            };
                            let SpirvOp::WordBytes(words) = words.as_ref() else {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "fourth op is not WordBytes".into(),
                                }));
                                continue;
                            };
                            let Some(Op::Spirv(push)) = order.next() else {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "fifth op is not Spirv".into(),
                                }));
                                continue;
                            };
                            let SpirvOp::PushConstants(push_constants_size) = push.as_ref() else {
                                let _ = reply.send(Err(BackendError {
                                    status: ErrorStatus::KernelCompilation,
                                    context: "fifth op is not PushConstants".into(),
                                }));
                                continue;
                            };
                            let lws: [u32; 3] = *lws;
                            let spirv: &[u32] = words;

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

                            let n_args = params
                                .iter()
                                .filter(|&kind| {
                                    matches!(kind, crate::kernel::ParamKind::Global | crate::kernel::ParamKind::GlobalMut)
                                })
                                .count();

                            // Same layout as the SPIR-V push-constant block (std140 scalars; bool stored as u32)
                            let push_constants_size = *push_constants_size;

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

                            let push_constant_range = VkPushConstantRange {
                                stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
                                offset: 0,
                                size: push_constants_size,
                            };
                            let pl_ci = VkPipelineLayoutCreateInfo {
                                sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                pNext: std::ptr::null(),
                                flags: 0,
                                setLayoutCount: 1,
                                pSetLayouts: &desc_layout,
                                pushConstantRangeCount: u32::from(push_constants_size > 4),
                                pPushConstantRanges: &push_constant_range,
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

                            let ep_name = format!("k_lws_{}", lws.iter().map(|v| v.to_string()).collect::<Vec<_>>().join("_"),);
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

                            let gws = gws.to_vec();
                            let id =
                                programs.push(VulkanProgram { pipeline, pipeline_layout, desc_layout, push_constants_size, gws });
                            let _ = reply.send(Ok(id));
                        }
                        VulkanCommand::LaunchTimed { program_id, args, reply } => {
                            // Uncontended timing for autotune: submit the
                            // pending window, drain every in-flight batch,
                            // surface async errors, then run the kernel solo
                            // and measure submit-to-fence.
                            if let Err(err) = submit_window(
                                &mut pending,
                                queue,
                                device,
                                cmd_pool,
                                desc_pool,
                                &buffers,
                                &programs,
                                &dev_info.max_global_work_dims,
                                &mut inflight,
                                debug_dev,
                                vkAllocateDescriptorSets,
                                vkUpdateDescriptorSets,
                                vkAllocateCommandBuffers,
                                vkBeginCommandBuffer,
                                vkEndCommandBuffer,
                                vkCmdBindPipeline,
                                vkCmdPushConstants,
                                vkCmdBindDescriptorSets,
                                vkCmdDispatch,
                                vkCreateFence,
                                vkQueueSubmit,
                            ) && last_error.is_none()
                            {
                                last_error = Some(err);
                            }
                            drain_all(
                                &mut inflight,
                                device,
                                cmd_pool,
                                desc_pool,
                                vkWaitForFences,
                                vkFreeCommandBuffers,
                                vkFreeDescriptorSets,
                                vkDestroyFence,
                            );
                            free_set.extend(pending_free.drain(..));
                            if let Some(err) = last_error.take() {
                                let _ = reply.send(Err(err));
                                continue;
                            }
                            let resolved: Vec<ResolvedArg> = args.iter().map(resolve_arg).collect();
                            let start = Instant::now();
                            let result = submit_window(
                                &mut vec![Pending::Launch { program_id, args: resolved }],
                                queue,
                                device,
                                cmd_pool,
                                desc_pool,
                                &buffers,
                                &programs,
                                &dev_info.max_global_work_dims,
                                &mut inflight,
                                debug_dev,
                                vkAllocateDescriptorSets,
                                vkUpdateDescriptorSets,
                                vkAllocateCommandBuffers,
                                vkBeginCommandBuffer,
                                vkEndCommandBuffer,
                                vkCmdBindPipeline,
                                vkCmdPushConstants,
                                vkCmdBindDescriptorSets,
                                vkCmdDispatch,
                                vkCreateFence,
                                vkQueueSubmit,
                            )
                            .and_then(|()| {
                                // The solo batch is the most recent one.
                                match inflight.pop() {
                                    Some(batch) => {
                                        let r = unsafe { vkWaitForFences(device, 1, &batch.fence, 1, u64::MAX) };
                                        destroy_batch(
                                            device,
                                            cmd_pool,
                                            desc_pool,
                                            batch,
                                            vkFreeCommandBuffers,
                                            vkFreeDescriptorSets,
                                            vkDestroyFence,
                                        );
                                        if r == VK_SUCCESS {
                                            Ok(())
                                        } else {
                                            Err(BackendError {
                                                status: ErrorStatus::KernelSync,
                                                context: format!("vkWaitForFences: {r}").into(),
                                            })
                                        }
                                    }
                                    None => Err(BackendError {
                                        status: ErrorStatus::KernelSync,
                                        context: "timed launch produced no batch".into(),
                                    }),
                                }
                            });
                            let nanos = start.elapsed().as_nanos() as u64;
                            let _ = reply.send(result.map(|()| nanos));
                        }
                        VulkanCommand::Replay { cmds, allocs, bound, vars, deaths, reply } => {
                            // One roundtrip per partition replay. Opening:
                            // flush the pending window, then drain every
                            // in-flight batch — the previous replay's
                            // launches are async, and only a fence proves
                            // bound slots and free-list chunks complete.
                            // Proven deaths join the free list. NO end
                            // drain: this replay's launches overlap
                            // caller-side CPU work; the next queue-touching
                            // command fences first.
                            if let Err(err) = submit_window(
                                &mut pending,
                                queue,
                                device,
                                cmd_pool,
                                desc_pool,
                                &buffers,
                                &programs,
                                &dev_info.max_global_work_dims,
                                &mut inflight,
                                debug_dev,
                                vkAllocateDescriptorSets,
                                vkUpdateDescriptorSets,
                                vkAllocateCommandBuffers,
                                vkBeginCommandBuffer,
                                vkEndCommandBuffer,
                                vkCmdBindPipeline,
                                vkCmdPushConstants,
                                vkCmdBindDescriptorSets,
                                vkCmdDispatch,
                                vkCreateFence,
                                vkQueueSubmit,
                            ) && last_error.is_none()
                            {
                                last_error = Some(err);
                            }
                            drain_all(
                                &mut inflight,
                                device,
                                cmd_pool,
                                desc_pool,
                                vkWaitForFences,
                                vkFreeCommandBuffers,
                                vkFreeDescriptorSets,
                                vkDestroyFence,
                            );
                            if let Some(err) = last_error.take() {
                                let _ = reply.send(Err(err));
                                continue;
                            }
                            free_set.extend(pending_free.drain(..));
                            // Ownership split (OpenCL pattern): bound inputs
                            // and escaping defs are caller-side Placements
                            // (Drop releases them). Intermediaries never
                            // become Placements: the worker allocates them
                            // here and parks them aside at death.
                            let result = (|| -> Result<Vec<(OpId, ChunkId, usize)>, BackendError> {
                                let vars_map: Map<OpId, Constant> = vars.into_iter().collect();
                                let mut slot_chunk: Map<OpId, (ChunkId, usize, usize)> =
                                    bound.into_iter().map(|(s, c, o, l)| (s, (c, o, l))).collect();
                                // Fresh defs' region lengths (offset is 0
                                // for worker-allocated defs; the length
                                // travels back so the caller names the
                                // live region on the chunk).
                                let mut fresh: Map<OpId, usize> = Map::default();
                                let mut stashed: Map<OpId, (ChunkId, usize, usize)> = Map::default();
                                // Proven-complete by the opening drain;
                                // same-replay deaths park in the stash, then
                                // pending_free — invisible to the scan by
                                // construction. Cloned for iteration safety;
                                // claims remove from both.
                                let mut snapshot: Set<ChunkId> = free_set.clone();
                                debug_assert_eq!(cmds.len(), allocs.len(), "replay alloc plan length mismatch");
                                debug_assert_eq!(cmds.len(), deaths.len(), "replay death length mismatch");
                                for (idx, cmd) in cmds.iter().enumerate() {
                                    match cmd {
                                        Cmd::Launch { program, args, outputs } => {
                                            for ((slot, dtype, dims), plan) in outputs.iter().zip(allocs[idx].iter()) {
                                                if slot_chunk.contains_key(slot) {
                                                    continue;
                                                }
                                                let chunk = match plan {
                                                    AllocPlan::Reuse(dead) => stashed.remove(dead),
                                                    AllocPlan::Fresh => None,
                                                };
                                                // Fresh defs name their region length
                                                // (evaluated bytes, offset 0); reused
                                                // defs inherit the dead slot's region.
                                                let (chunk, offset, len) = match chunk {
                                                    Some(region) => region,
                                                    None => {
                                                        let el = Dim::from(dtype.bit_size() / 8);
                                                        let bytes = dims.iter().map(|d| d.eval(&vars_map)).fold(el, |a, b| a * b);
                                                        if bytes < 0 {
                                                            return Err(BackendError {
                                                                status: ErrorStatus::MemoryAllocation,
                                                                context: format!("replay allocated negative bytes for {slot:?}")
                                                                    .into(),
                                                            });
                                                        }
                                                        // Best-fit over the
                                                        // snapshot only; fresh
                                                        // VRAM last.
                                                        let best = snapshot
                                                            .iter()
                                                            .filter_map(|id| {
                                                                let b = &buffers[*id];
                                                                (b.bytes >= bytes as usize).then_some((b.bytes, *id))
                                                            })
                                                            .min();
                                                        if let Some((_, id)) = best {
                                                            snapshot.remove(&id);
                                                            free_set.remove(&id);
                                                            (id, 0, bytes as usize)
                                                        } else {
                                                            let size = (bytes + 3) & !3;
                                                            let (buf, mem, ptr) = create_buffer(size as u64)?;
                                                            free_bytes_atomic.fetch_sub(size as u64, Ordering::SeqCst);
                                                            let id = buffers.push(VulkanBuffer {
                                                                buf,
                                                                mem,
                                                                ptr,
                                                                bytes: bytes as usize,
                                                            });
                                                            (id, 0, bytes as usize)
                                                        }
                                                    }
                                                };
                                                slot_chunk.insert(*slot, (chunk, offset, len));
                                                fresh.insert(*slot, len);
                                            }
                                            // Args in param order (GWS
                                            // ordinals index this vec):
                                            // placed slots become buffer
                                            // regions, variable slots
                                            // become push-constant args.
                                            // Bare ids, never
                                            // Placements (see ResolvedArg).
                                            let mut launch_args: Vec<ResolvedArg> = Vec::with_capacity(args.len());
                                            for slot in args {
                                                if let Some((chunk, offset, len)) = slot_chunk.get(slot) {
                                                    launch_args.push(ResolvedArg::Buffer {
                                                        chunk: *chunk,
                                                        offset: *offset,
                                                        len: *len,
                                                    });
                                                } else if let Some(c) = vars_map.get(slot) {
                                                    launch_args.push(ResolvedArg::Variable(*c));
                                                } else {
                                                    return Err(BackendError {
                                                        status: ErrorStatus::KernelLaunch,
                                                        context: format!(
                                                            "replay: launch slot {slot:?} is neither placed nor bound"
                                                        )
                                                        .into(),
                                                    });
                                                }
                                            }
                                            debug_assert!(
                                                programs.contains_id(program.program_id),
                                                "replay: launch of unknown program"
                                            );
                                            pending.push(Pending::Launch { program_id: program.program_id, args: launch_args });
                                            if pending.len() >= MICRO_BATCH_WINDOW
                                                && let Err(err) = submit_window(
                                                    &mut pending,
                                                    queue,
                                                    device,
                                                    cmd_pool,
                                                    desc_pool,
                                                    &buffers,
                                                    &programs,
                                                    &dev_info.max_global_work_dims,
                                                    &mut inflight,
                                                    debug_dev,
                                                    vkAllocateDescriptorSets,
                                                    vkUpdateDescriptorSets,
                                                    vkAllocateCommandBuffers,
                                                    vkBeginCommandBuffer,
                                                    vkEndCommandBuffer,
                                                    vkCmdBindPipeline,
                                                    vkCmdPushConstants,
                                                    vkCmdBindDescriptorSets,
                                                    vkCmdDispatch,
                                                    vkCreateFence,
                                                    vkQueueSubmit,
                                                )
                                                && last_error.is_none()
                                            {
                                                last_error = Some(err);
                                            }
                                        }
                                        Cmd::Alias { class, to } => {
                                            let region = *slot_chunk.get(to).ok_or_else(|| BackendError {
                                                status: ErrorStatus::KernelLaunch,
                                                context: format!("replay: alias target {to:?} is unplaced").into(),
                                            })?;
                                            slot_chunk.insert(*class, region);
                                        }
                                        Cmd::Copy { .. } => {
                                            unreachable!("copies are Copy partitions, never device runs")
                                        }
                                    }
                                    for dead in &deaths[idx] {
                                        if let Some(region) = slot_chunk.remove(dead) {
                                            // Worker-owned intermediaries
                                            // park in the stash: a baked
                                            // reuse takes them by slot,
                                            // leftovers join pending at
                                            // the end. Bound inputs stay
                                            // caller-owned — the caller's
                                            // Drop releases those.
                                            // Alias-shared chunks die
                                            // twice; the chunk guard keeps
                                            // one entry.
                                            if fresh.contains_key(dead)
                                                && !stashed.values().any(|&c| c == region)
                                                && !pending_free.contains(&region.0)
                                            {
                                                stashed.insert(*dead, region);
                                            }
                                        }
                                    }
                                }
                                // Unclaimed stashes never had a taker:
                                // park them for the next fence.
                                pending_free.extend(stashed.drain().map(|(_, region)| region.0));
                                // Submit the remainder. Errors abort the
                                // replay; submitted launches keep running
                                // (async) and the caller propagates.
                                submit_window(
                                    &mut pending,
                                    queue,
                                    device,
                                    cmd_pool,
                                    desc_pool,
                                    &buffers,
                                    &programs,
                                    &dev_info.max_global_work_dims,
                                    &mut inflight,
                                    debug_dev,
                                    vkAllocateDescriptorSets,
                                    vkUpdateDescriptorSets,
                                    vkAllocateCommandBuffers,
                                    vkBeginCommandBuffer,
                                    vkEndCommandBuffer,
                                    vkCmdBindPipeline,
                                    vkCmdPushConstants,
                                    vkCmdBindDescriptorSets,
                                    vkCmdDispatch,
                                    vkCreateFence,
                                    vkQueueSubmit,
                                )?;
                                // Survivors escape the partition: caller
                                // turns them into Placements, naming the
                                // live region length on each chunk.
                                Ok(fresh
                                    .iter()
                                    .filter(|(s, _)| slot_chunk.contains_key(s))
                                    .map(|(s, len)| (*s, slot_chunk[s].0, *len))
                                    .collect())
                            })();
                            let _ = reply.send(result);
                        }
                        VulkanCommand::ReleaseProgram(program_id) => {
                            if programs.contains_id(program_id) {
                                let prog = unsafe { programs.remove_and_return(program_id) };
                                unsafe {
                                    vkDestroyPipeline(device, prog.pipeline, std::ptr::null());
                                    vkDestroyPipelineLayout(device, prog.pipeline_layout, std::ptr::null());
                                    vkDestroyDescriptorSetLayout(device, prog.desc_layout, std::ptr::null());
                                }
                            }
                        }
                    }
                }

                // Idle device before destroying any resources
                unsafe {
                    let _ = vkDeviceWaitIdle(device);
                };

                // Cleanup all resources
                drain_all(
                    &mut inflight,
                    device,
                    cmd_pool,
                    desc_pool,
                    vkWaitForFences,
                    vkFreeCommandBuffers,
                    vkFreeDescriptorSets,
                    vkDestroyFence,
                );
                for id in buffers.ids().collect::<Vec<_>>() {
                    let VulkanBuffer { buf, mem, ptr, bytes: _, .. } = unsafe { buffers.remove_and_return(id) };
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

        pools.push(Mutex::new(VulkanMemoryPool { tx, free_bytes: Arc::clone(&free_bytes_atomic), dev_info }));
    }

    Ok(pools)
}

pub(super) fn device(id: u16) -> Result<&'static Mutex<VulkanDevice>, BackendError> {
    backend()?.1.get(id as usize).ok_or_else(|| BackendError {
        status: ErrorStatus::Initialization,
        context: format!("Dev::Vulkan({id}) is not available").into(),
    })
}

pub(super) fn device_count() -> u16 {
    backend().map(|(_, devs)| devs.len() as u16).unwrap_or(0)
}
