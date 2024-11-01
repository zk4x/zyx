#![allow(unused)]

use crate::{
    index_map::{Id, IndexMap},
    runtime::ir::IRKernel,
};

use super::DeviceInfo;

#[derive(serde::Deserialize, Debug, Default)]
pub struct VulkanConfig {}

#[derive(Debug)]
pub struct VulkanError {}

#[derive(Debug)]
pub(super) struct VulkanMemoryPool {
    free_bytes: usize,
}

#[derive(Debug)]
pub(super) struct VulkanBuffer {}

#[derive(Debug)]
pub(super) struct VulkanDevice {
    dev_info: DeviceInfo,
    memory_pool_id: usize,
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

pub(super) fn initialize_devices(
    config: &VulkanConfig,
    debug_dev: bool,
) -> Result<(Vec<VulkanMemoryPool>, VulkanQueuePool), VulkanError> {
    let memory_pools = Vec::new();
    let devices = Vec::new();

    Ok((memory_pools, devices))
}

impl VulkanMemoryPool {
    pub(super) fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    pub(super) fn deinitialize(self) -> Result<(), VulkanError> {
        Ok(())
    }

    pub(super) fn allocate(&mut self, bytes: usize) -> Result<VulkanBuffer, VulkanError> {
        todo!()
    }

    pub(super) fn deallocate(&mut self, buffer: VulkanBuffer) -> Result<(), VulkanError> {
        todo!()
    }

    pub(super) fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: &VulkanBuffer,
    ) -> Result<(), VulkanError> {
        todo!()
    }

    pub(super) fn pool_to_host(
        &mut self,
        src: &VulkanBuffer,
        dst: &mut [u8],
    ) -> Result<(), VulkanError> {
        todo!()
    }

    pub(super) fn pool_to_pool(
        &mut self,
        src: &VulkanBuffer,
        dst: &VulkanBuffer,
    ) -> Result<(), VulkanError> {
        todo!()
    }
}

impl VulkanDevice {
    pub(super) fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    // Memory pool id out of VulkanMemoryPools
    pub(super) fn memory_pool_id(&self) -> usize {
        self.memory_pool_id
    }

    pub(super) fn release_program(&self, program: VulkanProgram) -> Result<(), VulkanError> {
        todo!()
    }

    pub(super) fn release_queue(&self, queue: VulkanQueue) -> Result<(), VulkanError> {
        todo!()
    }

    pub(super) fn deinitialize(self) -> Result<(), VulkanError> {
        todo!()
    }

    pub(super) fn compile(
        &mut self,
        kernel: &IRKernel,
        debug_asm: bool,
    ) -> Result<VulkanProgram, VulkanError> {
        todo!()
    }
}

impl VulkanQueue {
    pub(super) fn launch(
        &mut self,
        program: &mut VulkanProgram,
        buffers: &mut IndexMap<VulkanBuffer>,
        args: &[Id],
    ) -> Result<(), VulkanError> {
        todo!()
    }

    pub(super) fn sync(&mut self) -> Result<(), VulkanError> {
        todo!()
    }

    pub(super) fn load(&self) -> usize {
        self.load
    }
}
