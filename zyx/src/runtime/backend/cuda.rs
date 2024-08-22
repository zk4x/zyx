use crate::{index_map::IndexMap, runtime::ir::IRKernel};

use super::{opencl::OpenCLBuffer, DeviceInfo};


#[derive(Debug)]
pub(crate) struct CUDAConfig {}

#[derive(Debug)]
pub(crate) struct CUDAError {}

#[derive(Debug)]
pub(crate) struct CUDAStatus {}

#[derive(Debug)]
pub(crate) struct CUDAMemoryPool {
    free_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct CUDABuffer {
    bytes: usize,
}

#[derive(Debug)]
pub(crate) struct CUDADevice {
    dev_info: DeviceInfo,
    memory_pool_id: usize,
}

#[derive(Debug)]
pub(crate) struct CUDAProgram {}

#[derive(Debug)]
pub(crate) struct CUDAEvent {}

pub(crate) fn initialize_cuda_backend(config: &CUDAConfig) -> Result<(Vec<CUDAMemoryPool>, Vec<CUDADevice>), CUDAError> {
    Err(CUDAError {})
}

impl CUDAMemoryPool {
    pub(crate) fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    pub(crate) fn allocate(&mut self, bytes: usize) -> Result<CUDABuffer, CUDAError> {
        //println!("Allocated buffer {ptr:?}");
        self.free_bytes -= bytes;
        /*let mut dptr = 0;
        check(
            unsafe { cuMemAlloc_v2(&mut dptr, bytes) },
            "Failed to allocate memory",
        )?;
        return Ok(CUDABuffer { mem: dptr });*/
        todo!()
    }

    pub(crate) fn deallocate(&mut self, buffer: CUDABuffer) -> Result<(), CUDAError> {
        //let status = unsafe { (self.clReleaseMemObject)(buffer.ptr) };
        //check(status, "Unable to free allocated memory")?;
        self.free_bytes += buffer.bytes;
        //Ok(())
        todo!()
    }

    pub(crate) fn host_to_cuda(
        &mut self,
        src: &[u8],
        dst: &CUDABuffer,
    ) -> Result<(), CUDAError> {
        todo!()
    }

    pub(crate) fn cuda_to_host(
        &mut self,
        src: &CUDABuffer,
        dst: &mut [u8],
    ) -> Result<(), CUDAError> {
        todo!()
    }

    pub(crate) fn cuda_to_cuda(
        &mut self,
        src: &CUDABuffer,
        dst: &CUDABuffer,
    ) -> Result<(), CUDAError> {
        todo!()
    }
}

impl CUDADevice {
    pub(crate) fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    // Memory pool id out of OpenCLMemoryPools
    pub(crate) fn memory_pool_id(&self) -> usize {
        self.memory_pool_id
    }

    pub(crate) fn compile(&mut self, kernel: &IRKernel) -> Result<CUDAProgram, CUDAError> {
        todo!()
    }
}

impl CUDAProgram {
    pub(crate) fn launch(
        &mut self,
        buffers: &mut IndexMap<CUDABuffer>,
        args: &[usize],
    ) -> Result<CUDAEvent, CUDAError> {
        todo!()
    }
}
