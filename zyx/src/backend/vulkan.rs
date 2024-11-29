//! Vulkan backend

#![allow(unused)]

use std::ptr;
use std::{ffi::CString, sync::Arc};

//use vulkano::instance::{Instance, InstanceCreateInfo};
//use vulkano::{LoadingError, Validated, VulkanLibrary};

use crate::{
    index_map::{Id, IndexMap},
    ir::IRKernel,
};

use super::DeviceInfo;

#[derive(serde::Deserialize, Debug, Default)]
pub struct VulkanConfig {}

#[derive(Debug)]
pub struct VulkanError; //(vulkano::VulkanError);

#[derive(Debug)]
pub(super) struct VulkanMemoryPool {
    free_bytes: usize,
}

#[derive(Debug)]
pub(super) struct VulkanBuffer {}

#[derive(Debug)]
pub(super) struct VulkanDevice {
    dev_info: DeviceInfo,
    memory_pool_id: u32,
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
}

type VulkanQueuePool = Vec<(VulkanDevice, Vec<VulkanQueue>)>;

#[allow(clippy::unnecessary_wraps)]
pub(super) fn initialize_devices(
    config: &VulkanConfig,
    debug_dev: bool,
) -> Result<(Vec<VulkanMemoryPool>, VulkanQueuePool), VulkanError> {
    // Load libvulkan.so dynamically
    /*let lib =
        unsafe { libloading::Library::new("libvulkan.so") }.expect("Could not load libvulkan.so");

    // Load Vulkan functions
    let vk_create_instance: PFN_vkCreateInstance =
        *unsafe { lib.get(b"vkCreateInstance") }.unwrap();
    let vk_enumerate_physical_devices: PFN_vkEnumeratePhysicalDevices =
        *unsafe { lib.get(b"vkEnumeratePhysicalDevices") }.unwrap();
    let vk_create_device: PFN_vkCreateDevice = *unsafe { lib.get(b"vkCreateDevice") }.unwrap();
    let vk_allocate_memory: PFN_vkAllocateMemory =
        *unsafe { lib.get(b"vkAllocateMemory") }.unwrap();
    // Create Vulkan instance
    let app_name = CString::new("Vulkan Example").unwrap();
    let engine_name = CString::new("No Engine").unwrap();
    let app_info = VkApplicationInfo {
        p_application_name: app_name.as_ptr(),
        application_version: 0x00010000, // Version 1.0.0
        p_engine_name: engine_name.as_ptr(),
        engine_version: 0x00010000, // Version 1.0.0
        api_version: 0x01000000,    // Vulkan 1.0
    };

    let instance_create_info = VkInstanceCreateInfo {
        s_type: 0, // VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO
        p_next: ptr::null(),
        flags: 0,
        p_application_info: &app_info,
        enabled_layer_count: 0,
        pp_enabled_layer_names: ptr::null(),
        enabled_extension_count: 0,
        pp_enabled_extension_names: ptr::null(),
    };
    let lib = Arc::new(lib);

    let mut instance: VkInstance = ptr::null_mut();
    unsafe {
        let result = vk_create_instance(&instance_create_info, ptr::null(), &mut instance);
        if result != 0 {
            panic!("Failed to create Vulkan instance");
        }
    }

    // Enumerate physical devices
    let mut device_count = 0;
    unsafe {
        vk_enumerate_physical_devices(instance, &mut device_count, ptr::null_mut());
    }
    let mut devices: Vec<VkPhysicalDevice> = Vec::with_capacity(device_count as usize);
    unsafe {
        vk_enumerate_physical_devices(instance, &mut device_count, devices.as_mut_ptr());
    }

    // Create logical device (just a simple example, no extensions or queues)
    let mut device: VkDevice = ptr::null_mut();
    unsafe {
        vk_create_device(devices[0], ptr::null(), ptr::null(), &mut device);
    }

    // Allocate memory on the device
    let memory_info = VkMemoryAllocateInfo {
        s_type: 0, // VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO
        p_next: ptr::null(),
        allocation_size: 1024 * 1024, // 1 MB
        memory_type_index: 0,         // You need to select a proper memory type
    };

    let mut memory: VkDeviceMemory = ptr::null_mut();
    unsafe {
        let result = vk_allocate_memory(device, &memory_info, ptr::null(), &mut memory);
        if result != 0 {
            panic!("Memory allocation failed");
        }
    }

    println!("Memory allocated successfully!");

    // Cleanup resources
    unsafe {
        // Free memory (in reality, you should call vkFreeMemory and destroy instances/devices)
    }*/

    /*let lib = VulkanLibrary::new()?;
    if debug_dev {
        println!(
            "Using Vulkan backend API version {} on devices:",
            lib.api_version()
        );
    }

    let instance = Instance::new(lib, InstanceCreateInfo::default())?;

    for device in instance.enumerate_physical_devices()? {
        println!("{}", device.properties().device_name);
    }*/

    let memory_pools = Vec::new();
    let devices = Vec::new();

    Ok((memory_pools, devices))
}

impl VulkanMemoryPool {
    pub(super) const fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    #[allow(clippy::unused_self)]
    #[allow(clippy::unnecessary_wraps)]
    pub(super) const fn deinitialize(self) -> Result<(), VulkanError> {
        Ok(())
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub(super) fn allocate(&mut self, bytes: usize) -> Result<VulkanBuffer, VulkanError> {
        todo!()
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

/*impl From<LoadingError> for VulkanError {
    fn from(value: LoadingError) -> Self {
        match value {
            LoadingError::LibraryLoadFailure(error) => Self(vulkano::VulkanError::Unknown),
            LoadingError::VulkanError(vulkan_error) => Self(vulkan_error),
        }
    }
}

impl From<Validated<vulkano::VulkanError>> for VulkanError {
    fn from(value: Validated<vulkano::VulkanError>) -> Self {
        match value {
            Validated::Error(value) => Self(value),
            Validated::ValidationError(_) => Self(vulkano::VulkanError::Unknown),
        }
    }
}

impl From<vulkano::VulkanError> for VulkanError {
    fn from(value: vulkano::VulkanError) -> Self {
        Self(value)
    }
}*/

/*#[repr(C)]
#[derive(Debug)]
struct VkApplicationInfo {
    p_application_name: *const i8,
    application_version: u32,
    p_engine_name: *const i8,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
#[derive(Debug)]
struct VkInstanceCreateInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    flags: u32,
    p_application_info: *const VkApplicationInfo,
    enabled_layer_count: u32,
    pp_enabled_layer_names: *const *const i8,
    enabled_extension_count: u32,
    pp_enabled_extension_names: *const *const i8,
}

#[repr(C)]
#[derive(Debug)]
struct VkMemoryAllocateInfo {
    s_type: u32,
    p_next: *const std::ffi::c_void,
    allocation_size: u64,
    memory_type_index: u32,
}

type VkDevice = *mut VkDeviceData;
#[repr(C)]
#[derive(Debug)]
struct VkDeviceData {
    _unused: [u8; 0],
}

type VkInstance = *mut VkInstanceData;
#[repr(C)]
#[derive(Debug)]
struct VkInstanceData {
    _unused: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct VkPhysicalDevice {
    _unused: [u8; 0],
}

type VkDeviceMemory = *mut VkDeviceMemoryData;
#[repr(C)]
#[derive(Debug)]
struct VkDeviceMemoryData {
    _unused: [u8; 0],
}

// Function types for Vulkan API functions
type PFN_vkCreateInstance = unsafe extern "system" fn(
    create_info: *const VkInstanceCreateInfo,
    p_allocator: *const std::ffi::c_void,
    p_instance: *mut VkInstance,
) -> u32;

type PFN_vkEnumeratePhysicalDevices = unsafe extern "system" fn(
    instance: VkInstance,
    p_device_count: *mut u32,
    p_devices: *mut VkPhysicalDevice,
) -> u32;

type PFN_vkCreateDevice = unsafe extern "system" fn(
    physical_device: VkPhysicalDevice,
    p_create_info: *const std::ffi::c_void,
    p_allocator: *const std::ffi::c_void,
    p_device: *mut VkDevice,
) -> u32;

type PFN_vkAllocateMemory = unsafe extern "system" fn(
    device: VkDevice,
    p_allocate_info: *const VkMemoryAllocateInfo,
    p_allocator: *const std::ffi::c_void,
    p_memory: *mut VkDeviceMemory,
) -> u32;*/
