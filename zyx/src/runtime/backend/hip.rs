#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use super::DeviceInfo;
use crate::{index_map::IndexMap, runtime::ir::IRKernel};
use libloading::Library;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::rc::Rc;

#[derive(Debug, serde::Deserialize)]
pub struct HIPConfig {}

#[derive(Debug)]
pub struct HIPError {
    info: String,
    status: HIPStatus,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum HIPStatus {
    HIP_SUCCESS,
    HIP_ERROR_UNKNOWN,
    HIP_ERROR_OUT_OF_MEMORY,
}

#[derive(Debug)]
pub(crate) struct HIPMemoryPool {
    #[allow(unused)]
    cuda: Rc<Library>,
    context: HIPcontext,
    device: HIPdevice,
    free_bytes: usize,
    hipMemAlloc: unsafe extern "C" fn(*mut HIPdeviceptr, usize) -> HIPStatus,
    hipMemcpyHtoD: unsafe extern "C" fn(HIPdeviceptr, *const c_void, usize) -> HIPStatus,
    hipMemcpyDtoH: unsafe extern "C" fn(*mut c_void, HIPdeviceptr, usize) -> HIPStatus,
    hipMemFree: unsafe extern "C" fn(HIPdeviceptr) -> HIPStatus,
    hipMemcpyPeer:
      unsafe extern "C" fn(HIPdeviceptr, HIPcontext, HIPdeviceptr, HIPcontext, usize) -> HIPStatus,
    hipCtxDestroy: unsafe extern "C" fn(HIPcontext) -> HIPStatus,
}

#[derive(Debug)]
pub(crate) struct HIPBuffer {
    ptr: u64,
    context: HIPcontext,
    bytes: usize,
}

#[derive(Debug)]
pub(crate) struct HIPDevice {
    device: HIPdevice,
    memory_pool_id: usize,
    dev_info: DeviceInfo,
    compute_capability: [c_int; 2],
}

#[derive(Debug)]
pub(crate) struct HIPProgram {}

#[derive(Debug)]
pub(crate) struct HIPEvent {}

unsafe impl Send for HIPMemoryPool {}
unsafe impl Send for HIPBuffer {}
unsafe impl Send for HIPProgram {}

pub(crate) fn initialize_hip_backend(
    config: &HIPConfig,
) -> Result<(Vec<HIPMemoryPool>, Vec<HIPDevice>), HIPError> {
    let _ = config;

    let hip_paths = ["/lib64/libamdhip64.so"];
    let hip = hip_paths.iter().find_map(|path| {
        if let Ok(lib) = unsafe { Library::new(path) } {
            Some(lib)
        } else {
            None
        }
    });
    let Some(hip) = hip else {
        return Err(HIPError {
            info: "HIP runtime not found.".into(),
            status: HIPStatus::HIP_ERROR_UNKNOWN,
        });
    };

    let hipInit: unsafe extern "C" fn(c_uint) -> HIPStatus =
        *unsafe { hip.get(b"hipInit\0") }.unwrap();
    let hipDriverGetVersion: unsafe extern "C" fn(*mut c_int) -> HIPStatus =
        *unsafe { hip.get(b"hipDriverGetVersion\0") }.unwrap();
    let hipDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> HIPStatus =
        *unsafe { hip.get(b"hipGetDeviceCount\0") }.unwrap();
    let hipDeviceGet: unsafe extern "C" fn(*mut HIPdevice, c_int) -> HIPStatus =
        *unsafe { hip.get(b"hipDeviceGet\0") }.unwrap();
    let hipDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, HIPdevice) -> HIPStatus =
        *unsafe { hip.get(b"hipDeviceGetName\0") }.unwrap();
    let hipDeviceComputeCapability: unsafe extern "C" fn(
        *mut c_int,
        *mut c_int,
        HIPdevice,
    ) -> HIPStatus = *unsafe { hip.get(b"hipDeviceComputeCapability\0") }.unwrap();
    let hipDeviceTotalMem: unsafe extern "C" fn(*mut usize, HIPdevice) -> HIPStatus =
        *unsafe { hip.get(b"hipDeviceTotalMem\0") }.unwrap();
    let hipDeviceGetAttribute: unsafe extern "C" fn(
        *mut c_int,
        HIPdevice_attribute,
        HIPdevice,
    ) -> HIPStatus = *unsafe { hip.get(b"hipDeviceGetAttribute\0") }.unwrap();
    let hipCtxCreate: unsafe extern "C" fn(*mut HIPcontext, c_uint, HIPdevice) -> HIPStatus =
        *unsafe { hip.get(b"hipCtxCreate\0") }.unwrap();
    let hipMemAlloc = *unsafe { hip.get(b"hipMalloc\0") }.unwrap();
    let hipMemcpyHtoD = *unsafe { hip.get(b"hipMemcpyHtoD\0") }.unwrap();
    let hipMemFree = *unsafe { hip.get(b"hipFree\0") }.unwrap();
    let hipMemcpyDtoH = *unsafe { hip.get(b"hipMemcpyDtoH\0") }.unwrap();
    let hipMemcpyPeer = *unsafe { hip.get(b"hipMemcpyPeer\0") }.unwrap();
    let hipCtxDestroy = *unsafe { hip.get(b"hipCtxDestroy\0") }.unwrap();
    //let hipModuleLoadDataEx = *unsafe { hip.get(b"hipModuleLoadDataEx\0") }.unwrap();
    //let hipModuleGetFunction = *unsafe { hip.get(b"hipModuleGetFunction\0") }.unwrap();
    //let hipLaunchKernel = *unsafe { hip.get(b"hipLaunchKernel\0") }.unwrap();

    unsafe { hipInit(0) }.check("Failed to init HIP")?;
    let mut driver_version = 0;
    unsafe { hipDriverGetVersion(&mut driver_version) }
        .check("Failed to get HIP driver version")?;
    #[cfg(feature = "debug_dev")]
    println!(
        "Using HIP backend, driver version: {}.{} on devices:",
        driver_version / 1000,
        (driver_version - (driver_version / 1000 * 1000)) / 10
    );
    let mut num_devices = 0;
    unsafe { hipDeviceGetCount(&mut num_devices) }.check("Failed to get HIP device count")?;
    if num_devices == 0 {
        return Err(HIPError {
            info: "No available hip device.".into(),
            status: HIPStatus::HIP_ERROR_UNKNOWN,
        });
    }

    let hip = Rc::new(hip);
    let mut memory_pools = Vec::new();
    let mut devices = Vec::new();
    for dev_id in 0..num_devices {
        let mut device = 0;
        unsafe { hipDeviceGet(&mut device, dev_id) }.check("Failed to access HIP device")?;
        let mut device_name = [0; 100];
        let Ok(_) = unsafe { hipDeviceGetName(device_name.as_mut_ptr(), 100, device) }
            .check("Failed to get HIP device name") else { continue; };
        let mut major = 0;
        let mut minor = 0;
        let Ok(_) = unsafe { hipDeviceComputeCapability(&mut major, &mut minor, device) }
            .check("Failed to get HIP device compute capability.") else { continue; };
        #[cfg(feature = "debug_dev")]
        println!("{:?}, compute capability: {major}.{minor}", unsafe {
            std::ffi::CStr::from_ptr(device_name.as_ptr())
        });
        let mut free_bytes = 0;
        let Ok(_) = unsafe { hipDeviceTotalMem(&mut free_bytes, device) }.check("Failed to get dev mem.") else { continue; };
        let mut context: HIPcontext = ptr::null_mut();
        unsafe { hipCtxCreate(&mut context, 0, device) }.check("Unable to create HIP context.")?;
        memory_pools.push(HIPMemoryPool {
            cuda: hip.clone(),
            context,
            device,
            free_bytes,
            hipMemAlloc,
            hipMemcpyHtoD,
            hipMemFree,
            hipMemcpyDtoH,
            hipMemcpyPeer,
            hipCtxDestroy,
        });
        devices.push(HIPDevice {
            device,
            dev_info: DeviceInfo::default(),
            memory_pool_id: 0,
            //hipModuleLoadDataEx,
            //hipModuleGetFunction,
            //hipModuleEnumerateFunctions,
            //hipLaunchKernel,
            compute_capability: [major, minor],
        })
    }

    Ok((memory_pools, devices))
}

impl HIPMemoryPool {
    pub(crate) fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    pub(crate) fn allocate(&mut self, bytes: usize) -> Result<HIPBuffer, HIPError> {
        if bytes > self.free_bytes {
            return Err(HIPError {
                info: "Insufficient free memory.".into(),
                status: HIPStatus::HIP_ERROR_OUT_OF_MEMORY,
            });
        }
        self.free_bytes -= bytes;
        let mut ptr = self.device as u64;
        unsafe { (self.hipMemAlloc)(&mut ptr, bytes) }.check("Failed to allocate memory.")?;
        return Ok(HIPBuffer {
            ptr,
            bytes,
            context: self.context,
        });
    }

    pub(crate) fn deallocate(&mut self, buffer: HIPBuffer) -> Result<(), HIPError> {
        unsafe { (self.hipMemFree)(buffer.ptr) }.check("Failed to free memory.")?;
        self.free_bytes += buffer.bytes;
        Ok(())
    }

    pub(crate) fn host_to_pool(&mut self, src: &[u8], dst: &HIPBuffer) -> Result<(), HIPError> {
        unsafe { (self.hipMemcpyHtoD)(dst.ptr, src.as_ptr().cast(), src.len()) }
            .check("Failed to copy memory from host to pool.")
    }

    pub(crate) fn pool_to_host(&mut self, src: &HIPBuffer, dst: &mut [u8]) -> Result<(), HIPError> {
        unsafe { (self.hipMemcpyDtoH)(dst.as_mut_ptr().cast(), src.ptr, dst.len()) }
            .check("Failed to copy memory from pool to host.")
    }

    pub(crate) fn pool_to_pool(
        &mut self,
        src: &HIPBuffer,
        dst: &HIPBuffer,
    ) -> Result<(), HIPError> {
        unsafe { (self.hipMemcpyPeer)(dst.ptr, dst.context, src.ptr, src.context, dst.bytes) }
            .check("Failed copy memory from pool to pool.")
    }
}

impl Drop for HIPMemoryPool {
    fn drop(&mut self) {
        unsafe { (self.hipCtxDestroy)(self.context) };
    }
}

impl HIPDevice {
    pub(crate) fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    // Memory pool id out of OpenCLMemoryPools
    pub(crate) fn memory_pool_id(&self) -> usize {
        self.memory_pool_id
    }

    pub(crate) fn compile(&mut self, kernel: &IRKernel) -> Result<HIPProgram, HIPError> {
        todo!()
    }
}

impl HIPProgram {
    pub(crate) fn launch(
        &mut self,
        buffers: &mut IndexMap<HIPBuffer>,
        args: &[usize],
    ) -> Result<HIPEvent, HIPError> {
        let mut kernel_params: Vec<*mut core::ffi::c_void> = Vec::new();
        for arg in args {
            let arg = &mut buffers[*arg];
            //let ptr = &mut arg.mem;
            let ptr: *mut _ = &mut arg.ptr;
            kernel_params.push(ptr.cast());
        }
        /*unsafe {
            (self.hipLaunchKernel)(
                self.function,
                self.global_work_size[0] as u32,
                self.global_work_size[1] as u32,
                self.global_work_size[2] as u32,
                self.local_work_size[0] as u32,
                self.local_work_size[1] as u32,
                self.local_work_size[2] as u32,
                0,
                ptr::null_mut(),
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            )
        }
        .check("Failed to launch kernel.")?;*/
        // For now just empty event, later we can deal with streams to make it async
        Ok(HIPEvent {})
    }
}

impl HIPStatus {
    fn check(self, info: &str) -> Result<(), HIPError> {
        if self != HIPStatus::HIP_SUCCESS {
            return Err(HIPError {
                info: info.into(),
                status: self,
            });
        } else {
            return Ok(());
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct HIPctx_st {
    _unused: [u8; 0],
}
type HIPcontext = *mut HIPctx_st;
type HIPdevice = c_int;
type HIPdeviceptr = u64;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct HIPmod_st {
    _unused: [u8; 0],
}
type HIPmodule = *mut HIPmod_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct HIPfunc_st {
    _unused: [u8; 0],
}
type HIPfunction = *mut HIPfunc_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum HIPdevice_attribute {
    HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
}
