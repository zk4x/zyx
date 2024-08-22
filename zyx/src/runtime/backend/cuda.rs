#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::{c_char, c_int, c_uint};
use std::ptr;

use libloading::Library;

use crate::{index_map::IndexMap, runtime::ir::IRKernel};
use super::DeviceInfo;


#[derive(Debug)]
pub(crate) struct CUDAConfig {}

#[derive(Debug)]
pub struct CUDAError {
    info: String,
    status: CUDAStatus,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum CUDAStatus {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_UNKNOWN = 999,
}

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
    let _ = config;

    let cuda_paths = ["/lib/x86_64-linux-gnu/libcuda.so"];
    let cuda = cuda_paths.iter().find_map(|path| if let Ok(lib) = unsafe { Library::new(path) } { Some(lib) } else { None } );
    let Some(cuda) = cuda else { return Err(CUDAError { info: "CUDA runtime not found.".into(), status: CUDAStatus::CUDA_ERROR_UNKNOWN }) };

    let cuInit: unsafe extern "C" fn(c_uint) -> CUDAStatus =
        *unsafe { cuda.get(b"cuInit\0") }.unwrap();
    let cuDriverGetVersion: unsafe extern "C" fn(*mut c_int) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDriverGetVersion\0") }.unwrap();
    let cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDeviceGetCount\0") }.unwrap();
    let cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDeviceGet\0") }.unwrap();
    let cuDeviceGetName: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDeviceGetName\0") }.unwrap();
    let cuDeviceComputeCapability: unsafe extern "C" fn(*mut c_int, *mut c_int, CUdevice) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDeviceComputeCapability\0") }.unwrap();
    let cuCtxCreate_v2: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUDAStatus =
        *unsafe { cuda.get(b"cuCtxCreate_v2\0") }.unwrap();

    unsafe { cuInit(0) }.check("Failed to init CUDA")?;

    let mut driver_version = 0;
    unsafe { cuDriverGetVersion(&mut driver_version) }.check("Failed to get CUDA driver version")?;
    #[cfg(feature = "debug_dev")]
    println!(
        "Using CUDA backend, driver version: {}.{} on devices:",
        driver_version / 1000,
        (driver_version - (driver_version / 1000 * 1000)) / 10
    );
    let mut num_devices = 0;
    unsafe { cuDeviceGetCount(&mut num_devices) }.check("Failed to get CUDA device count")?;
    if num_devices == 0 {
        return Err(CUDAError { info: "No available cuda device.".into(), status: CUDAStatus::CUDA_ERROR_UNKNOWN });
    }

    let mut memory_pools = Vec::new();
    let mut devices = Vec::new();
    for dev_id in 0..num_devices {
        let mut device = 0;
        unsafe { cuDeviceGet(&mut device, dev_id) }.check("Failed to access CUDA device")?;
        let mut device_name = [0; 100];
        unsafe { cuDeviceGetName(device_name.as_mut_ptr(), 100, device) }.check("Failed to get CUDA device name")?;
        let mut major = 0;
        let mut minor = 0;
        unsafe { cuDeviceComputeCapability(&mut major, &mut minor, device) }.check("Failed to get CUDA device compute capability.")?;
        #[cfg(feature = "debug_dev")]
        println!("{:?}, compute capability: {major}.{minor}", unsafe {
            std::ffi::CStr::from_ptr(device_name.as_ptr())
        });
        let mut context: CUcontext = ptr::null_mut();
        unsafe { cuCtxCreate_v2(&mut context, 0, device) }.check("Unable to create CUDA context.")?;

        devices.push(CUDADevice { dev_info: DeviceInfo::default(), memory_pool_id: 0 })
    }

    Ok((memory_pools, devices))
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

impl CUDAStatus {
    fn check(self, info: &str) -> Result<(), CUDAError> {
        if self != CUDAStatus::CUDA_SUCCESS {
            return Err(CUDAError { info: info.into(), status: self });
        } else {
            return Ok(());
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUctx_st {
    _unused: [u8; 0],
}
type CUcontext = *mut CUctx_st;
type CUdevice = c_int;
