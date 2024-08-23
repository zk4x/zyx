#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_uint};
use std::ptr;
use libloading::Library;
use crate::{index_map::IndexMap, runtime::ir::IRKernel};
use super::DeviceInfo;

#[derive(Debug)]
pub(crate) struct HIPError {
    info: String,
    status: HIPStatus,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum HIPStatus {
    HIP_SUCCESS,
    HIP_ERROR_UNKNOWN,
}

#[derive(Debug)]
pub(crate) struct HIPMemoryPool {
    free_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct HIPBuffer {
    bytes: usize,
}

#[derive(Debug)]
pub(crate) struct HIPDevice {
    dev_info: DeviceInfo,
    memory_pool_id: usize,
}

#[derive(Debug)]
pub(crate) struct HIPProgram {}

#[derive(Debug)]
pub(crate) struct HIPEvent {}

#[derive(Debug)]
pub struct HIPConfig {}

pub(crate) fn initialize_hip_backend(config: &HIPConfig) -> Result<(Vec<HIPMemoryPool>, Vec<HIPDevice>), HIPError> {
    let _ = config;

    let hip_paths = ["/lib64/libamdhip64.so"];
    let hip = hip_paths.iter().find_map(|path| if let Ok(lib) = unsafe { Library::new(path) } { Some(lib) } else { None } );
    let Some(hip) = hip else { return Err(HIPError { info: "HIP runtime not found.".into(), status: HIPStatus::HIP_ERROR_UNKNOWN }) };

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
    let hipDeviceComputeCapability: unsafe extern "C" fn(*mut c_int, *mut c_int, HIPdevice) -> HIPStatus =
        *unsafe { hip.get(b"hipDeviceComputeCapability\0") }.unwrap();
    let hipCtxCreate: unsafe extern "C" fn(*mut HIPcontext, c_uint, HIPdevice) -> HIPStatus =
        *unsafe { hip.get(b"hipCtxCreate\0") }.unwrap();

    unsafe { hipInit(0) }.check("Failed to init HIP")?;

    let mut driver_version = 0;
    unsafe { hipDriverGetVersion(&mut driver_version) }.check("Failed to get HIP driver version")?;
    #[cfg(feature = "debug_dev")]
    println!(
        "Using HIP backend, driver version: {}.{} on devices:",
        driver_version / 1000,
        (driver_version - (driver_version / 1000 * 1000)) / 10
    );
    let mut num_devices = 0;
    unsafe { hipDeviceGetCount(&mut num_devices) }.check("Failed to get HIP device count")?;
    if num_devices == 0 {
        return Err(HIPError { info: "No available hip device.".into(), status: HIPStatus::HIP_ERROR_UNKNOWN });
    }

    let mut memory_pools = Vec::new();
    let mut devices = Vec::new();
    for dev_id in 0..num_devices {
        let mut device = 0;
        unsafe { hipDeviceGet(&mut device, dev_id) }.check("Failed to access HIP device")?;
        let mut device_name = [0; 100];
        unsafe { hipDeviceGetName(device_name.as_mut_ptr(), 100, device) }.check("Failed to get HIP device name")?;
        let mut major = 0;
        let mut minor = 0;
        unsafe { hipDeviceComputeCapability(&mut major, &mut minor, device) }.check("Failed to get HIP device compute capability.")?;
        #[cfg(feature = "debug_dev")]
        println!("{:?}, compute capability: {major}.{minor}", unsafe {
            std::ffi::CStr::from_ptr(device_name.as_ptr())
        });
        let mut context: HIPcontext = ptr::null_mut();
        unsafe { hipCtxCreate(&mut context, 0, device) }.check("Unable to create HIP context.")?;

        //memory_pools.push(HIPMemoryPool { free_bytes: () });
        devices.push(HIPDevice { dev_info: DeviceInfo::default(), memory_pool_id: 0 })
    }

    Ok((memory_pools, devices))
}

impl HIPMemoryPool {
    pub(crate) fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    pub(crate) fn allocate(&mut self, bytes: usize) -> Result<HIPBuffer, HIPError> {
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

    pub(crate) fn deallocate(&mut self, buffer: HIPBuffer) -> Result<(), HIPError> {
        //let status = unsafe { (self.clReleaseMemObject)(buffer.ptr) };
        //check(status, "Unable to free allocated memory")?;
        self.free_bytes += buffer.bytes;
        //Ok(())
        todo!()
    }

    pub(crate) fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: &HIPBuffer,
    ) -> Result<(), HIPError> {
        todo!()
    }

    pub(crate) fn pool_to_host(
        &mut self,
        src: &HIPBuffer,
        dst: &mut [u8],
    ) -> Result<(), HIPError> {
        todo!()
    }

    pub(crate) fn pool_to_pool(
        &mut self,
        src: &HIPBuffer,
        dst: &HIPBuffer,
    ) -> Result<(), HIPError> {
        todo!()
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
        todo!()
    }
}

impl HIPStatus {
    fn check(self, info: &str) -> Result<(), HIPError> {
        if self != HIPStatus::HIP_SUCCESS {
            return Err(HIPError { info: info.into(), status: self });
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
