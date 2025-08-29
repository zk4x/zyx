//! HIP backend

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(unused)]

use super::{Device, DeviceInfo, MemoryPool};
use crate::DType;
use crate::backend::{BufferId, Event, ProgramId};
use crate::dtype::Constant;
use crate::error::{BackendError, ErrorStatus};
use crate::kernel::Kernel;
use crate::runtime::Pool;
use crate::shape::Dim;
use crate::slab::Slab;
use libloading::Library;
use nanoserde::DeJson;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::ptr;
use std::sync::Arc;

#[derive(Debug, Default, DeJson)]
pub struct HIPConfig {
    device_ids: Option<Vec<i32>>,
}

#[derive(Debug)]
pub struct HIPMemoryPool {
    #[allow(unused)]
    cuda: Arc<Library>,
    context: HIPcontext,
    device: HIPdevice,
    free_bytes: usize,
    buffers: Slab<BufferId, HIPBuffer>,
    stream: HIPstream,
    hipMemAlloc: unsafe extern "C" fn(*mut HIPdeviceptr, usize) -> HIPStatus,
    hipMemcpyHtoDAsync: unsafe extern "C" fn(HIPdeviceptr, *const c_void, usize, HIPstream) -> HIPStatus,
    hipMemcpyDtoHAsync: unsafe extern "C" fn(*mut c_void, HIPdeviceptr, usize, HIPstream) -> HIPStatus,
    hipMemFree: unsafe extern "C" fn(HIPdeviceptr) -> HIPStatus,
    //hipMemcpyPeer: unsafe extern "C" fn(HIPdeviceptr, HIPcontext, HIPdeviceptr, HIPcontext, usize) -> HIPStatus,
    hipEventCreate: unsafe extern "C" fn(*mut HIPevent, c_uint) -> HIPStatus,
    hipEventRecord: unsafe extern "C" fn(HIPevent, HIPstream) -> HIPStatus,
    hipStreamWaitEvent: unsafe extern "C" fn(HIPstream, HIPevent, c_uint) -> HIPStatus,
    hipEventSynchronize: unsafe extern "C" fn(HIPevent) -> HIPStatus,
    hipEventDestroy: unsafe extern "C" fn(HIPevent) -> HIPStatus,
    hipCtxDestroy: unsafe extern "C" fn(HIPcontext) -> HIPStatus,
}

#[derive(Debug)]
pub(super) struct HIPBuffer {
    ptr: u64,
    bytes: usize,
}

#[derive(Debug)]
pub struct HIPDevice {
    device: HIPdevice,
    memory_pool_id: u32,
    dev_info: DeviceInfo,
    compute_capability: [c_int; 2],
    streams: Vec<HIPStream>,
    programs: Slab<ProgramId, HIPProgram>,
    hipModuleLoadData: unsafe extern "C" fn(*mut HIPmodule, *const u8) -> HIPStatus,
    hipModuleGetFunction: unsafe extern "C" fn(*mut HIPfunction, HIPmodule, *const c_char) -> HIPStatus,
    hipModuleUnload: unsafe extern "C" fn(HIPmodule) -> HIPStatus,
    hipStreamSynchronize: unsafe extern "C" fn(HIPstream) -> HIPStatus,
    hipEventCreate: unsafe extern "C" fn(*mut HIPevent, c_uint) -> HIPStatus,
    hipLaunchKernel: unsafe extern "C" fn(
        HIPfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        HIPstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> HIPStatus,
    hipEventRecord: unsafe extern "C" fn(HIPevent, HIPstream) -> HIPStatus,
    hipStreamWaitEvent: unsafe extern "C" fn(HIPstream, HIPevent, c_uint) -> HIPStatus,
    //hipStreamDestroy: unsafe extern "C" fn(HIPstream) -> HIPStatus,
}

#[derive(Debug)]
pub(super) struct HIPProgram {
    name: String,
    module: HIPmodule,
    function: HIPfunction,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
}

#[derive(Debug)]
pub(super) struct HIPStream {
    stream: HIPstream,
    load: usize,
}

#[derive(Debug, Clone)]
pub struct HIPEvent {
    event: HIPevent,
}

// TODO remove this using channels
unsafe impl Send for HIPMemoryPool {}
unsafe impl Send for HIPDevice {}
unsafe impl Send for HIPBuffer {}
unsafe impl Send for HIPProgram {}
unsafe impl Send for HIPStream {}
unsafe impl Send for HIPEvent {}

pub(super) fn initialize_device(
    config: &HIPConfig,
    memory_pools: &mut Vec<Pool>,
    devices: &mut Vec<Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    let _ = config;

    let hip_paths = ["/lib64/libamdhip64.so", "/lib/x86_64-linux-gnu/libamdhip64.so"];
    let hip = hip_paths.iter().find_map(|path| unsafe { Library::new(path) }.ok());
    let Some(hip) = hip else {
        return Err(BackendError { status: ErrorStatus::DyLibNotFound, context: "HIP runtime not found.".into() });
    };

    let hipInit: unsafe extern "C" fn(c_uint) -> HIPStatus = *unsafe { hip.get(b"hipInit\0") }.unwrap();
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
    let hipDeviceTotalMem: unsafe extern "C" fn(*mut usize, HIPdevice) -> HIPStatus =
        *unsafe { hip.get(b"hipDeviceTotalMem\0") }.unwrap();
    //let hipDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, HIPdevice_attribute, HIPdevice) -> HIPStatus =
    //*unsafe { hip.get(b"hipDeviceGetAttribute\0") }.unwrap();
    let hipCtxCreate: unsafe extern "C" fn(*mut HIPcontext, c_uint, HIPdevice) -> HIPStatus =
        *unsafe { hip.get(b"hipCtxCreate\0") }.unwrap();
    let hipMemAlloc = *unsafe { hip.get(b"hipMalloc\0") }.unwrap();
    let hipMemFree = *unsafe { hip.get(b"hipFree\0") }.unwrap();
    let hipMemcpyHtoDAsync = *unsafe { hip.get(b"hipMemcpyHtoDAsync\0") }.unwrap();
    //let hipMemcpyHtoD = *unsafe { hip.get(b"hipMemcpyHtoD\0") }.unwrap();
    let hipMemcpyDtoHAsync = *unsafe { hip.get(b"hipMemcpyDtoHAsync\0") }.unwrap();
    //let hipMemcpyDtoH = *unsafe { hip.get(b"hipMemcpyDtoH\0") }.unwrap();
    //let hipMemcpyPeer = *unsafe { hip.get(b"hipMemcpyPeer\0") }.unwrap();
    let hipModuleLoadData = *unsafe { hip.get(b"hipModuleLoadData\0") }.unwrap();
    let hipModuleGetFunction = *unsafe { hip.get(b"hipModuleGetFunction\0") }.unwrap();
    let hipLaunchKernel = *unsafe { hip.get(b"hipLaunchKernel\0") }.unwrap();
    let hipStreamCreate: unsafe extern "C" fn(*mut HIPstream, c_uint) -> HIPStatus =
        *unsafe { hip.get(b"hipStreamCreate\0") }.unwrap();
    let hipStreamSynchronize = *unsafe { hip.get(b"hipStreamSynchronize\0") }.unwrap();
    //let hipStreamDestroy = *unsafe { hip.get(b"hipStreamDestroy\0") }.unwrap();
    //let hipModuleUnload = *unsafe { hip.get(b"hipModuleUnload\0") }.unwrap();

    let hipStreamWaitEvent = *unsafe { hip.get(b"hipStreamWaitEvent\0") }.unwrap();
    //let cuStreamDestroy = *unsafe { cuda.get(b"cuStreamDestroy\0") }.unwrap();
    let hipModuleUnload = *unsafe { hip.get(b"hipModuleUnload\0") }.unwrap();
    let hipEventCreate = *unsafe { hip.get(b"hipEventCreate\0") }.unwrap();
    let hipEventRecord = *unsafe { hip.get(b"hipEventRecord\0") }.unwrap();
    let hipEventSynchronize = *unsafe { hip.get(b"hipEventSynchronize\0") }.unwrap();
    let hipEventDestroy = *unsafe { hip.get(b"hipEventDestroy\0") }.unwrap();
    let hipCtxDestroy = *unsafe { hip.get(b"hipCtxDestroy\0") }.unwrap();
    //let cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUDAStatus = *unsafe { cuda.get(b"cuDevicePrimaryCtxRetain\0") }.unwrap();

    unsafe { hipInit(0) }.check(ErrorStatus::Initialization)?;
    let mut driver_version = 0;
    unsafe { hipDriverGetVersion(&mut driver_version) }.check(ErrorStatus::Initialization)?;
    let mut num_devices = 0;
    unsafe { hipDeviceGetCount(&mut num_devices) }.check(ErrorStatus::DeviceEnumeration)?;
    if num_devices == 0 {
        return Err(BackendError { status: ErrorStatus::DeviceEnumeration, context: "HIP no devices found.".into() });
    }
    let device_ids: Vec<_> =
        (0..num_devices).filter(|id| config.device_ids.as_ref().map_or(true, |ids| ids.contains(id))).collect();
    if device_ids.is_empty() {
        return Err(BackendError {
            status: ErrorStatus::DeviceEnumeration,
            context: "HIP all available devices configured out.".into(),
        });
    }
    if debug_dev {
        println!(
            "Using HIP runtime, driver version: {}.{} on devices:",
            driver_version / 1000,
            (driver_version - (driver_version / 1000 * 1000)) / 10
        );
    }

    let hip = Arc::new(hip);
    for dev_id in device_ids {
        let mut device = 0;
        unsafe { hipDeviceGet(&mut device, dev_id) }.check(ErrorStatus::DeviceEnumeration)?;
        let mut device_name = [0; 100];
        let Ok(()) = unsafe { hipDeviceGetName(device_name.as_mut_ptr(), 100, device) }.check(ErrorStatus::DeviceQuery)
        else {
            continue;
        };
        let mut major = 0;
        let mut minor = 0;
        let Ok(()) =
            unsafe { hipDeviceComputeCapability(&mut major, &mut minor, device) }.check(ErrorStatus::DeviceQuery)
        else {
            continue;
        };
        if debug_dev {
            println!("{:?}, compute capability: {major}.{minor}", unsafe {
                std::ffi::CStr::from_ptr(device_name.as_ptr())
            });
        }
        let mut free_bytes = 0;
        let Ok(()) = unsafe { hipDeviceTotalMem(&mut free_bytes, device) }.check(ErrorStatus::DeviceQuery) else {
            continue;
        };
        let mut context: HIPcontext = ptr::null_mut();
        unsafe { hipCtxCreate(&mut context, 0, device) }.check(ErrorStatus::Initialization)?;
        let mut stream = ptr::null_mut();
        unsafe { hipStreamCreate(&raw mut stream, 0) }.check(ErrorStatus::Initialization)?;
        let pool = HIPMemoryPool {
            cuda: hip.clone(),
            context,
            device,
            free_bytes,
            buffers: Slab::new(),
            stream,
            hipEventCreate,
            hipMemAlloc,
            hipMemcpyHtoDAsync,
            hipMemFree,
            //hipMemcpyPeer,
            hipMemcpyDtoHAsync,
            hipEventRecord,
            hipStreamWaitEvent,
            hipEventSynchronize,
            hipEventDestroy,
            hipCtxDestroy,
        };
        memory_pools.push(Pool::new(MemoryPool::HIP(pool)));
        let mut streams = Vec::new();
        for _ in 0..8 {
            let mut stream = ptr::null_mut();
            if let Err(err) = unsafe { hipStreamCreate(&raw mut stream, 0) }.check(ErrorStatus::Initialization) {
                if debug_dev {
                    println!("Device with id {dev_id} requested, but cuda stream initialization failed. {err:?}");
                }
                continue;
            }
            streams.push(HIPStream { stream, load: 0 });
        }
        let dev = HIPDevice {
            device,
            dev_info: DeviceInfo {
                compute: 1024 * 1024 * 1024 * 1024,
                max_global_work_dims: vec![64, 64, 64],
                max_local_threads: 1,
                max_local_work_dims: vec![1, 1, 1],
                local_mem_size: 0,
                num_registers: 96,
                preferred_vector_size: 16,
                tensor_cores: major > 7,
            },
            streams,
            programs: Slab::new(),
            memory_pool_id: u32::try_from(memory_pools.len()).unwrap() - 1,
            hipModuleLoadData,
            hipModuleGetFunction,
            hipModuleUnload,
            compute_capability: [major, minor],
            hipLaunchKernel,
            hipStreamSynchronize,
            hipEventCreate,
            hipEventRecord,
            hipStreamWaitEvent,
        };
        devices.push(Device::HIP(dev));
        //queues,
    }
    Ok(())
}

impl HIPMemoryPool {
    #[allow(clippy::unused_self)]
    #[allow(clippy::unnecessary_wraps)]
    pub(super) fn deinitialize(&mut self) {
        // TODO
    }

    pub(super) const fn free_bytes(&self) -> usize { self.free_bytes }

    pub(super) fn allocate(&mut self, bytes: usize) -> Result<(BufferId, Event), BackendError> {
        if bytes > self.free_bytes {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "Allocation failure".into() });
        }
        //println!("Allocating to context {:?}, device {:?}", self.context, self.device);
        let mut ptr = u64::try_from(self.device).unwrap();
        //unsafe { (self.cuCtxSetCurrent)(self.context) }.check("Failed to set current CUDA context.")?;
        let mut event = ptr::null_mut();
        unsafe { (self.hipEventCreate)(&raw mut event, 0x2) }.check(ErrorStatus::MemoryAllocation)?;
        debug_assert!(!self.stream.is_null());
        //unsafe { (self.cuMemAllocAsync)(&mut ptr, bytes, self.stream) }.check(ErrorStatus::MemoryAllocation)?;
        unsafe { (self.hipMemAlloc)(&raw mut ptr, bytes as usize) }.check(ErrorStatus::MemoryAllocation)?;
        unsafe { (self.hipEventRecord)(event, self.stream) }.check(ErrorStatus::MemoryAllocation)?;
        self.free_bytes = self.free_bytes.checked_sub(bytes).unwrap();
        Ok((
            self.buffers.push(HIPBuffer { ptr, bytes }),
            Event::HIP(HIPEvent { event }),
        ))
    }

    #[allow(clippy::needless_pass_by_value)]
    pub(super) fn deallocate(&mut self, buffer_id: BufferId, mut event_wait_list: Vec<Event>) {
        while let Some(Event::HIP(HIPEvent { event })) = event_wait_list.pop() {
            if !event.is_null() {
                unsafe { (self.hipStreamWaitEvent)(self.stream, event, 0) }
                    .check(ErrorStatus::MemoryDeallocation)
                    .unwrap();
                unsafe { (self.hipEventDestroy)(event) }.check(ErrorStatus::MemoryCopyP2H).unwrap();
            }
        }
        let buffer = &mut self.buffers[buffer_id];
        //unsafe { (self.hipMemFreeAsync)(buffer.ptr, self.stream) }.check(ErrorStatus::MemoryDeallocation).unwrap();
        unsafe { (self.hipMemFree)(buffer.ptr) }.check(ErrorStatus::MemoryDeallocation).unwrap();
        self.free_bytes += buffer.bytes;
        self.buffers.remove(buffer_id);
    }

    pub(super) fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: BufferId,
        mut event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        //unsafe { (self.hipMemcpyHtoD)(dst.ptr, src.as_ptr().cast(), src.len()) }.check("Failed to copy memory from host to pool.")
        let dst = &self.buffers[dst];
        while let Some(Event::HIP(HIPEvent { event })) = event_wait_list.pop() {
            if !event.is_null() {
                unsafe { (self.hipStreamWaitEvent)(self.stream, event, 0) }.check(ErrorStatus::MemoryCopyH2P)?;
            }
        }
        let mut event = ptr::null_mut();
        unsafe { (self.hipEventCreate)(&raw mut event, 0x2) }.check(ErrorStatus::MemoryCopyH2P)?;
        debug_assert!(!self.stream.is_null());
        //unsafe { (self.cuStreamSynchronize)(self.stream) }.check(ErrorStatus::MemoryCopyH2P).unwrap();
        unsafe { (self.hipMemcpyHtoDAsync)(dst.ptr, src.as_ptr().cast(), src.len(), self.stream) }
            .check(ErrorStatus::MemoryCopyH2P)?;
        //unsafe { (self.cuMemcpyHtoD)(dst.ptr, src.as_ptr().cast(), src.len()) }.check(ErrorStatus::MemoryCopyH2P)?;
        unsafe { (self.hipEventRecord)(event, self.stream) }.check(ErrorStatus::MemoryCopyH2P)?;
        Ok(Event::HIP(HIPEvent { event }))
    }

    pub(super) fn pool_to_host(
        &mut self,
        src: BufferId,
        dst: &mut [u8],
        mut event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        //unsafe { (self.hipMemcpyDtoH)(dst.as_mut_ptr().cast(), src.ptr, dst.len()) }.check("Failed to copy memory from pool to host.")
        while let Some(Event::HIP(HIPEvent { event })) = event_wait_list.pop() {
            if !event.is_null() {
                unsafe { (self.hipStreamWaitEvent)(self.stream, event, 0) }.check(ErrorStatus::MemoryCopyP2H)?;
                // Should we destroy the event here?
            }
        }
        let src = &self.buffers[src];
        let mut event = ptr::null_mut();
        unsafe { (self.hipEventCreate)(&raw mut event, 0x2) }.check(ErrorStatus::MemoryCopyP2H)?;
        unsafe { (self.hipMemcpyDtoHAsync)(dst.as_mut_ptr().cast(), src.ptr, dst.len(), self.stream) }
            .check(ErrorStatus::MemoryCopyP2H)?;
        unsafe { (self.hipEventRecord)(event, self.stream) }.check(ErrorStatus::MemoryCopyP2H)?;
        //unsafe { (self.cuStreamSynchronize)(self.stream) }.check(ErrorStatus::MemoryCopyP2H)?;
        unsafe { (self.hipEventSynchronize)(event) }.check(ErrorStatus::MemoryCopyP2H)?;
        unsafe { (self.hipEventDestroy)(event) }.check(ErrorStatus::MemoryCopyP2H)?;
        Ok(())
    }

    /*pub(super) fn pool_to_pool(&mut self, src: &HIPBuffer, dst: &HIPBuffer) -> Result<(), BackendError> {
        unsafe { (self.hipMemcpyPeer)(dst.ptr, dst.context, src.ptr, src.context, dst.bytes) }.check("Failed copy memory from pool to pool.")
    }*/

    pub fn sync_events(&mut self, mut events: Vec<Event>) -> Result<(), BackendError> {
        while let Some(Event::HIP(HIPEvent { event })) = events.pop() {
            if !event.is_null() {
                unsafe { (self.hipEventSynchronize)(event) }.check(ErrorStatus::KernelSync)?;
                unsafe { (self.hipEventDestroy)(event) }.check(ErrorStatus::KernelSync)?;
            }
        }
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn release_events(&mut self, events: Vec<Event>) {
        for event in events {
            let Event::HIP(HIPEvent { event }) = event else { unreachable!() };
            unsafe { (self.hipEventDestroy)(event) }.check(ErrorStatus::Deinitialization).unwrap();
        }
    }
}

impl Drop for HIPMemoryPool {
    fn drop(&mut self) { unsafe { (self.hipCtxDestroy)(self.context) }; }
}

impl HIPDevice {
    #[allow(clippy::unused_self)]
    #[allow(clippy::unnecessary_wraps)]
    pub(super) const fn deinitialize(&mut self) {
        // TODO
    }

    fn next_stream(&mut self) -> Result<usize, BackendError> {
        let mut id = self.streams.iter().enumerate().min_by_key(|(_, s)| s.load).unwrap().0;
        if self.streams[id].load > 20 {
            unsafe { (self.hipStreamSynchronize)(self.streams[id].stream) }.check(ErrorStatus::KernelSync)?;
            self.streams[id].load = 0;
            id = self.streams.iter().enumerate().min_by_key(|(_, q)| q.load).unwrap().0;
        }
        Ok(id)
    }

    pub(super) const fn info(&self) -> &DeviceInfo { &self.dev_info }

    // Memory pool id out of OpenCLMemoryPools
    pub(super) const fn memory_pool_id(&self) -> u32 { self.memory_pool_id }

    /*#[allow(clippy::needless_pass_by_value)]
    pub(super) fn release_program(&self, program: HIPProgram) -> Result<(), BackendError> {
        unsafe { (self.hipModuleUnload)(program.module) }.check(ErrorStatus::Deinitialization)
    }*/

    /*#[allow(clippy::needless_pass_by_value)]
    pub(super) fn release_queue(&self, queue: HIPQueue) -> Result<(), HIPError> {
        unsafe { (self.hipStreamDestroy)(queue.stream) }.check("Failed to release HIP stream.")
    }*/

    /*pub(super) fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<ProgramId, BackendError> {
        #[repr(C)]
        #[derive(Debug)]
        struct _hiprtcProgram {
            _unused: [u8; 0],
        }
        type hiprtcProgram = *mut _hiprtcProgram;

        // INFO: MUST BE NULL TERMINATED!
        let source = format!("{pragma}extern \"C\" __global__ void {name}{source}\0");
        name += "\0";
        if debug_asm {
            println!("{source}");
        }
        let hiprtc_paths = ["/lib64/libhiprtc.so"];
        let hiprtc = hiprtc_paths.iter().find_map(|path| unsafe { Library::new(path) }.ok());
        let Some(hiprtc) = hiprtc else {
            return Err(HIPError {
                info: "HIP runtime compiler (HIPRTC) not found.".into(),
                status: HIPStatus::hipErrorTbd,
                hiprtc: hiprtcResult::HIPRTC_SUCCESS,
            });
        };
        let hiprtcCreateProgram: unsafe extern "C" fn(
            *mut hiprtcProgram,
            *const c_char,
            *const c_char,
            c_int,
            *const *const c_char,
            *const *const c_char,
        ) -> hiprtcResult = *unsafe { hiprtc.get(b"hiprtcCreateProgram\0") }.unwrap();
        let hiprtcCompileProgram: unsafe extern "C" fn(hiprtcProgram, c_int, *const *const c_char) -> hiprtcResult =
            *unsafe { hiprtc.get(b"hiprtcCompileProgram\0") }.unwrap();
        let hiprtcGetCodeSize: unsafe extern "C" fn(hiprtcProgram, *mut usize) -> hiprtcResult =
            *unsafe { hiprtc.get(b"hiprtcGetCodeSize\0") }.unwrap();
        let hiprtcGetCode: unsafe extern "C" fn(hiprtcProgram, *mut c_char) -> hiprtcResult =
            *unsafe { hiprtc.get(b"hiprtcGetCode\0") }.unwrap();
        let hiprtcGetProgramLogSize: unsafe extern "C" fn(hiprtcProgram, *mut usize) -> hiprtcResult =
            *unsafe { hiprtc.get(b"hiprtcGetProgramLogSize\0") }.unwrap();
        let hiprtcGetProgramLog: unsafe extern "C" fn(hiprtcProgram, *mut c_char) -> hiprtcResult =
            *unsafe { hiprtc.get(b"hiprtcGetProgramLog\0") }.unwrap();
        let hiprtcDestroyProgram: unsafe extern "C" fn(*mut hiprtcProgram) -> hiprtcResult =
            *unsafe { hiprtc.get(b"hiprtcDestroyProgram\0") }.unwrap();

        let mut program = ptr::null_mut();
        unsafe {
            hiprtcCreateProgram(
                &mut program,
                source.as_ptr().cast(),
                name.as_ptr().cast(),
                0,
                ptr::null(),
                ptr::null(),
            )
        }
        .check("hiprtcCreateProgram")?;

        let df = format!(
            "--gpu-architecture=compute_{}{}\0",
            self.compute_capability[0], self.compute_capability[1]
        );
        //let df = format!("");
        let opts = [df.as_str()];
        if let Err(e) = unsafe { hiprtcCompileProgram(program, 0, opts.as_ptr().cast()) }.check("hiprtcCompileProgram")
        {
            //println!("Error during compilation {e:?}");
            let mut program_log_size: usize = 0;
            unsafe { hiprtcGetProgramLogSize(program, &mut program_log_size) }.check("hiprtcGetProgramLogSize")?;
            //program_log_size = 1000;
            println!("Program log size: {program_log_size}");
            let mut program_log: Vec<u8> = vec![0; program_log_size];
            unsafe { hiprtcGetProgramLog(program, program_log.as_mut_ptr().cast()) }.check("hiprtcGetProgramLog")?;
            if let Ok(log) = String::from_utf8(program_log) {
                println!("HIPRTC program log:\n{log}",);
            } else {
                println!("HIPRTC program log is not valid utf8");
            }
            return Err(e);
        }

        let mut code_size: usize = 0;
        unsafe { hiprtcGetCodeSize(program, &mut code_size) }.check("hiprtcGetCodeSize")?;

        let mut code_vec: Vec<u8> = vec![0; code_size];
        unsafe { hiprtcGetCode(program, code_vec.as_mut_ptr().cast()) }.check("hiprtcGetCode")?;
        unsafe { hiprtcDestroyProgram(&mut program) }.check("hiprtcDestroyProgram")?;

        let mut module = ptr::null_mut();
        unsafe { (self.hipModuleLoadData)(&mut module, code_vec.as_ptr()) }.check("Module load failed.")?;
        let mut function: HIPfunction = ptr::null_mut();
        unsafe { (self.hipModuleGetFunction)(&mut function, module, name.as_ptr().cast()) }
            .check("Failed to load function.")?;

        Ok(HIPProgram { name, module, function, global_work_size, local_work_size })
    }*/

    #[allow(unused)]
    #[allow(clippy::type_complexity)]
    fn compile_hip(
        &self,
        kernel: &Kernel,
        debug_asm: bool,
    ) -> Result<([Dim; 3], [Dim; 3], String, Vec<u8>), BackendError> {
        todo!()
    }

    #[allow(unused)]
    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<ProgramId, BackendError> {
        /*let (gws, lws, name, ptx) = self.compile_hip(kernel, debug_asm)?;
        //let (gws, lws, name, ptx) = self.compile_ptx(kernel, debug_asm)?;

        let mut module = ptr::null_mut();
        if let Err(err) = unsafe {
            (self.hipModuleLoadData)(
                &raw mut module,
                ptx.as_ptr().cast(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        }
        .check(ErrorStatus::KernelCompilation)
        {
            if debug_asm {
                println!("Failed to compile kernel with err: {err:?}");
            }
            return Err(err);
        }
        let mut function: CUfunction = ptr::null_mut();
        // Don't forget that the name is null terminated string
        if let Err(err) = unsafe { (self.cuModuleGetFunction)(&raw mut function, module, name.as_ptr().cast()) }
            .check(ErrorStatus::KernelLaunch)
        {
            if debug_asm {
                println!("Failed to launch kernel with err: {err:?}\n");
            }
            return Err(err);
        }

        let program_id = self.programs.push(CUDAProgram {
            //name,
            module,
            function,
            global_work_size: gws,
            local_work_size: lws,
        });
        Ok(program_id)*/
        todo!()
    }

    pub fn launch(
        &mut self,
        program_id: ProgramId,
        memory_pool: &mut HIPMemoryPool,
        args: &[BufferId],
        mut event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let stream_id = self.next_stream()?;
        let program = &self.programs[program_id];

        //println!("CUDA launch program id: {program_id}, gws: {:?}, lws: {:?}", program.global_work_size, program.local_work_size);

        let mut kernel_params: Vec<*mut core::ffi::c_void> = Vec::new();
        for &arg in args {
            let arg = &memory_pool.buffers[arg];
            //let ptr = &mut arg.mem;
            let ptr: *const u64 = &raw const arg.ptr;
            let ptr: *mut u64 = ptr.cast_mut();
            kernel_params.push(ptr.cast());
        }

        while let Some(Event::HIP(HIPEvent { event })) = event_wait_list.pop() {
            if !event.is_null() {
                unsafe { (self.hipStreamWaitEvent)(self.streams[stream_id].stream, event, 0) }
                    .check(ErrorStatus::KernelLaunch)?;
            }
        }

        //unsafe { (self.cuStreamSynchronize)(self.streams[stream_id].stream) }.check(ErrorStatus::KernelLaunch).unwrap();

        let mut event = ptr::null_mut();
        unsafe { (self.hipEventCreate)(&raw mut event, 0) }.check(ErrorStatus::KernelLaunch)?;
        unsafe {
            (self.hipLaunchKernel)(
                program.function,
                u32::try_from(program.global_work_size[0]).unwrap(),
                u32::try_from(program.global_work_size[1]).unwrap(),
                u32::try_from(program.global_work_size[2]).unwrap(),
                u32::try_from(program.local_work_size[0]).unwrap(),
                u32::try_from(program.local_work_size[1]).unwrap(),
                u32::try_from(program.local_work_size[2]).unwrap(),
                0,
                self.streams[stream_id].stream,
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            )
        }
        .check(ErrorStatus::KernelLaunch)?;
        unsafe { (self.hipEventRecord)(event, self.streams[stream_id].stream) }.check(ErrorStatus::KernelLaunch)?;

        //unsafe { (self.hipStreamSynchronize)(self.streams[stream_id].stream) }.check(ErrorStatus::KernelLaunch).unwrap();

        self.streams[stream_id].load += 1;
        Ok(Event::HIP(HIPEvent { event }))
    }

    pub fn release(&mut self, program_id: ProgramId) {
        let _ =
            unsafe { (self.hipModuleUnload)(self.programs[program_id].module) }.check(ErrorStatus::Deinitialization);
        self.programs.remove(program_id);
    }

    pub const fn free_compute(&self) -> u128 { self.dev_info.compute }
}

impl HIPStatus {
    fn check(self, status: ErrorStatus) -> Result<(), BackendError> {
        if self == Self::hipSuccess {
            Ok(())
        } else {
            Err(BackendError { status, context: format!("Try rerunning with env var AMD_LOG_LEVEL=2 {self:?}").into() })
        }
    }
}

impl DType {
    pub(super) fn hip(&self) -> &str {
        match self {
            Self::BF16 => "hip_bfloat16",
            Self::F16 => "half",
            Self::F32 => "float",
            Self::F64 => "double",
            Self::U8 => "unsigned char",
            Self::I8 => "char",
            Self::I16 => "short",
            Self::I32 => "int",
            Self::I64 => "long",
            Self::Bool => "bool",
            Self::U16 => "unsigned short",
            Self::U32 => "unsigned int",
            Self::U64 => "unsigned long",
        }
    }
}

impl Constant {
    fn hip(self) -> String {
        match self {
            Self::BF16(x) => format!("{}f", half::bf16::from_le_bytes(x)),
            Self::F16(x) => format!("{}f", half::f16::from_le_bytes(x)),
            Self::F32(x) => format!("{}f", f32::from_le_bytes(x)),
            Self::F64(x) => format!("{}f", f64::from_le_bytes(x)),
            Self::U8(x) => format!("{x}"),
            Self::I8(x) => format!("{x}"),
            Self::I16(x) => format!("{x}"),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}"),
            Self::U64(x) => format!("{}", u64::from_le_bytes(x)),
            Self::I32(x) => format!("{x}"),
            Self::I64(x) => format!("{}", i64::from_le_bytes(x)),
            Self::Bool(x) => format!("{x}"),
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
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct HIPstream_st {
    _unused: [u8; 0],
}
type HIPstream = *mut HIPstream_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct HIPevent_st {
    _unused: [u8; 0],
}
type HIPevent = *mut HIPevent_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum hiprtcResult {
    HIPRTC_SUCCESS = 0,                                     // Success
    HIPRTC_ERROR_OUT_OF_MEMORY = 1,                         // Out of memory
    HIPRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,              // Failed to create program
    HIPRTC_ERROR_INVALID_INPUT = 3,                         // Invalid input
    HIPRTC_ERROR_INVALID_PROGRAM = 4,                       // Invalid program
    HIPRTC_ERROR_INVALID_OPTION = 5,                        // Invalid option
    HIPRTC_ERROR_COMPILATION = 6,                           // Compilation error
    HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,             // Failed in builtin operation
    HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8, // No name expression after compilation
    HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,   // No lowered names before compilation
    HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,            // Invalid name expression
    HIPRTC_ERROR_INTERNAL_ERROR = 11,                       // Internal error
    HIPRTC_ERROR_LINKING = 100,                             // Error in linking
}

impl hiprtcResult {
    fn check(self, status: ErrorStatus) -> Result<(), BackendError> {
        if self == Self::HIPRTC_SUCCESS {
            Ok(())
        } else {
            Err(BackendError { status, context: format!("Try rerunning with env var AMD_LOG_LEVEL=2 {self:?}").into() })
        }
    }
}

#[allow(clippy::enum_variant_names)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum HIPStatus {
    hipSuccess = 0,
    /// Successful completion.
    hipErrorInvalidValue = 1,
    /// One or more of the parameters passed to the API call is NULL
    /// or not in an acceptable range.
    hipErrorOutOfMemory = 2,
    /// out of memory range.
    hipErrorNotInitialized = 3,
    /// Invalid not initialized
    hipErrorDeinitialized = 4,
    /// Deinitialized
    hipErrorProfilerDisabled = 5,
    hipErrorProfilerNotInitialized = 6,
    hipErrorProfilerAlreadyStarted = 7,
    hipErrorProfilerAlreadyStopped = 8,
    hipErrorInvalidConfiguration = 9,
    /// Invalide configuration
    hipErrorInvalidPitchValue = 12,
    /// Invalid pitch value
    hipErrorInvalidSymbol = 13,
    /// Invalid symbol
    hipErrorInvalidDevicePointer = 17,
    /// Invalid Device Pointer
    hipErrorInvalidMemcpyDirection = 21,
    /// Invalid memory copy direction
    hipErrorInsufficientDriver = 35,
    hipErrorMissingConfiguration = 52,
    hipErrorPriorLaunchFailure = 53,
    hipErrorInvalidDeviceFunction = 98,
    /// Invalid device function
    hipErrorNoDevice = 100,
    /// Call to hipGetDeviceCount returned 0 devices
    hipErrorInvalidDevice = 101,
    /// `DeviceID` must be in range from 0 to compute-devices.
    hipErrorInvalidImage = 200,
    /// Invalid image
    hipErrorInvalidContext = 201,
    /// Produced when input context is invalid.
    hipErrorContextAlreadyCurrent = 202,
    hipErrorMapFailed = 205,
    hipErrorUnmapFailed = 206,
    hipErrorArrayIsMapped = 207,
    hipErrorAlreadyMapped = 208,
    hipErrorNoBinaryForGpu = 209,
    hipErrorAlreadyAcquired = 210,
    hipErrorNotMapped = 211,
    hipErrorNotMappedAsArray = 212,
    hipErrorNotMappedAsPointer = 213,
    hipErrorECCNotCorrectable = 214,
    hipErrorUnsupportedLimit = 215,
    /// Unsupported limit
    hipErrorContextAlreadyInUse = 216,
    /// The context is already in use
    hipErrorPeerAccessUnsupported = 217,
    hipErrorInvalidKernelFile = 218,
    /// In CUDA DRV, it is `CUDA_ERROR_INVALID_PTX`
    hipErrorInvalidGraphicsContext = 219,
    hipErrorInvalidSource = 300,
    /// Invalid source.
    hipErrorFileNotFound = 301,
    /// the file is not found.
    hipErrorSharedObjectSymbolNotFound = 302,
    hipErrorSharedObjectInitFailed = 303,
    /// Failed to initialize shared object.
    hipErrorOperatingSystem = 304,
    /// Not the correct operating system
    hipErrorInvalidHandle = 400,
    /// Invalide handle
    hipErrorIllegalState = 401,
    /// Resource required is not in a valid state to perform operation.
    hipErrorNotFound = 500,
    /// Not found
    hipErrorNotReady = 600,
    /// Indicates that asynchronous operations enqueued earlier are not
    /// ready.  This is not actually an error, but is used to distinguish
    /// from hipSuccess (which indicates completion).  APIs that return
    /// this error include hipEventQuery and hipStreamQuery.
    hipErrorIllegalAddress = 700,
    hipErrorLaunchOutOfResources = 701,
    /// Out of resources error.
    hipErrorLaunchTimeOut = 702,
    /// Timeout for the launch.
    hipErrorPeerAccessAlreadyEnabled = 704,
    /// Peer access was already enabled from the current
    /// device.
    hipErrorPeerAccessNotEnabled = 705,
    /// Peer access was never enabled from the current device.
    hipErrorSetOnActiveProcess = 708,
    /// The process is active.
    hipErrorContextIsDestroyed = 709,
    /// The context is already destroyed
    hipErrorAssert = 710,
    /// Produced when the kernel calls assert.
    hipErrorHostMemoryAlreadyRegistered = 712,
    /// Produced when trying to lock a page-locked
    /// memory.
    hipErrorHostMemoryNotRegistered = 713,
    /// Produced when trying to unlock a non-page-locked
    /// memory.
    hipErrorLaunchFailure = 719,
    /// An exception occurred on the device while executing a kernel.
    hipErrorCooperativeLaunchTooLarge = 720,
    /// This error indicates that the number of blocks
    /// launched per grid for a kernel that was launched
    /// via cooperative launch APIs exceeds the maximum
    /// number of allowed blocks for the current device.
    hipErrorNotSupported = 801,
    /// Produced when the hip API is not supported/implemented
    hipErrorStreamCaptureUnsupported = 900,
    /// The operation is not permitted when the stream
    /// is capturing.
    hipErrorStreamCaptureInvalidated = 901,
    /// The current capture sequence on the stream
    /// has been invalidated due to a previous error.
    hipErrorStreamCaptureMerge = 902,
    /// The operation would have resulted in a merge of
    /// two independent capture sequences.
    hipErrorStreamCaptureUnmatched = 903,
    /// The capture was not initiated in this stream.
    hipErrorStreamCaptureUnjoined = 904,
    /// The capture sequence contains a fork that was not
    /// joined to the primary stream.
    hipErrorStreamCaptureIsolation = 905,
    /// A dependency would have been created which crosses
    /// the capture sequence boundary. Only implicit
    /// in-stream ordering dependencies  are allowed
    /// to cross the boundary
    hipErrorStreamCaptureImplicit = 906,
    /// The operation would have resulted in a disallowed
    /// implicit dependency on a current capture sequence
    /// from hipStreamLegacy.
    hipErrorCapturedEvent = 907,
    /// The operation is not permitted on an event which was last
    /// recorded in a capturing stream.
    hipErrorStreamCaptureWrongThread = 908,
    /// A stream capture sequence not initiated with
    /// the hipStreamCaptureModeRelaxed argument to
    /// hipStreamBeginCapture was passed to
    /// hipStreamEndCapture in a different thread.
    hipErrorGraphExecUpdateFailure = 910,
    /// This error indicates that the graph update
    /// not performed because it included changes which
    /// violated constraintsspecific to instantiated graph
    /// update.
    hipErrorInvalidChannelDescriptor = 911,
    /// Invalid channel descriptor.
    hipErrorInvalidTexture = 912,
    /// Invalid texture.
    hipErrorUnknown = 999,
    /// Unknown error.
    // HSA Runtime Error Codes start here.
    hipErrorRuntimeMemory = 1052,
    /// HSA runtime memory call returned error.  Typically not seen
    /// in production systems.
    hipErrorRuntimeOther = 1053,
    /// HSA runtime call other than memory returned error.  Typically
    /// not seen in production systems.
    hipErrorTbd, // Marker that more error codes are needed.
}
