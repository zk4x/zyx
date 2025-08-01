#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(clippy::question_mark)]

// TODO properly deallocate events

use std::{
    ffi::{c_char, c_int, c_uint, c_void},
    fmt::Write,
    ptr,
    sync::Arc,
};

use libloading::Library;
use nanoserde::DeJson;

use crate::{
    DType,
    dtype::Constant,
    error::{BackendError, ErrorStatus},
    kernel::Kernel,
    shape::Dim,
    slab::Slab,
};

use super::{BufferId, Device, DeviceInfo, Event, MemoryPool, Pool, ProgramId};

/// CUDA configuration
#[derive(Debug, Default, DeJson)]
pub struct CUDAConfig {
    /// If set to None, then it will automatically use all CUDA devices,
    /// otherwise it uses only selected devices
    device_ids: Option<Vec<i32>>,
}

#[derive(Debug)]
pub struct CUDAMemoryPool {
    // Just to close the connection
    #[allow(unused)]
    lib: Arc<Library>,
    #[allow(unused)]
    context: CUcontext,
    device: CUdevice,
    free_bytes: Dim,
    buffers: Slab<BufferId, CUDABuffer>,
    stream: CUstream,
    //cuMemAllocAsync: unsafe extern "C" fn(*mut CUdeviceptr, usize, CUstream) -> CUDAStatus,
    cuMemAlloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUDAStatus,
    cuMemcpyHtoDAsync:
        unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUDAStatus,
    cuMemcpyDtoHAsync:
        unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUDAStatus,
    //cuMemFreeAsync: unsafe extern "C" fn(CUdeviceptr, CUstream) -> CUDAStatus,
    cuMemFree: unsafe extern "C" fn(CUdeviceptr) -> CUDAStatus,
    cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUDAStatus,
    cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUDAStatus,
    cuStreamWaitEvent: unsafe extern "C" fn(CUstream, CUevent, c_uint) -> CUDAStatus,
    cuEventSynchronize: unsafe extern "C" fn(CUevent) -> CUDAStatus,
    cuEventDestroy: unsafe extern "C" fn(CUevent) -> CUDAStatus,
    //cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUDAStatus,
    //cuMemcpyHtoD: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUDAStatus,
    //cuMemcpyPeer: unsafe extern "C" fn(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, usize) -> CUDAStatus,
    //cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUDAStatus,
    //cuCtxDestroy: unsafe extern "C" fn(CUcontext) -> CUDAStatus,
}

#[derive(Debug)]
pub(super) struct CUDABuffer {
    ptr: u64,
    bytes: Dim,
}

#[derive(Debug)]
pub struct CUDADevice {
    device: CUdevice,
    memory_pool_id: u32,
    dev_info: DeviceInfo,
    compute_capability: [c_int; 2],
    streams: Vec<CUDAStream>,
    programs: Slab<ProgramId, CUDAProgram>,
    cuModuleLoadDataEx: unsafe extern "C" fn(
        *mut CUmodule,
        *const c_void,
        c_uint,
        *mut CUjit_option,
        *mut *mut c_void,
    ) -> CUDAStatus,
    cuModuleGetFunction:
        unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUDAStatus,
    cuModuleUnload: unsafe extern "C" fn(CUmodule) -> CUDAStatus,
    cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUDAStatus,
    //cuStreamDestroy: unsafe extern "C" fn(CUstream) -> CUDAStatus,
    cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUDAStatus,
    cuLaunchKernel: unsafe extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUDAStatus,
    cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUDAStatus,
    cuStreamWaitEvent: unsafe extern "C" fn(CUstream, CUevent, c_uint) -> CUDAStatus,
}

#[derive(Debug)]
pub(super) struct CUDAProgram {
    //name: String,
    module: CUmodule,
    function: CUfunction,
    global_work_size: [Dim; 3],
    local_work_size: [Dim; 3],
}

#[derive(Debug)]
pub(super) struct CUDAStream {
    stream: CUstream,
    load: usize,
}

#[derive(Debug, Clone)]
pub struct CUDAEvent {
    event: CUevent,
}

unsafe impl Send for CUDAMemoryPool {}
unsafe impl Send for CUDADevice {}
unsafe impl Send for CUDABuffer {}
unsafe impl Send for CUDAProgram {}
unsafe impl Send for CUDAStream {}
unsafe impl Send for CUDAEvent {}

pub(super) fn initialize_device(
    config: &CUDAConfig,
    memory_pools: &mut Vec<Pool>,
    devices: &mut Vec<Device>,
    debug_dev: bool,
) -> Result<(), BackendError> {
    if let Some(device_ids) = &config.device_ids {
        if device_ids.is_empty() {
            if debug_dev {
                println!("CUDA won't be used, as it was configured out");
            }
            return Ok(());
        }
    }
    let cuda_paths = [
        "/lib/x86_64-linux-gnu/libcuda.so",
        "/lib64/x86_64-linux-gnu/libcuda.so",
        "/lib/libcuda.so",
        "/lib64/libcuda.so",
        "/usr/lib/libcuda.so",
        "/usr/lib64/libcuda.so",
    ];
    let cuda = cuda_paths.iter().find_map(|path| unsafe { Library::new(path) }.ok());
    let Some(cuda) = cuda else {
        if debug_dev {
            println!("libcuda.so not found");
        }
        return Err(BackendError {
            status: ErrorStatus::DyLibNotFound,
            context: "CUDA libcuda.so not found.".into(),
        });
    };

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
    let cuDeviceComputeCapability: unsafe extern "C" fn(
        *mut c_int,
        *mut c_int,
        CUdevice,
    ) -> CUDAStatus = *unsafe { cuda.get(b"cuDeviceComputeCapability\0") }.unwrap();
    let cuDeviceTotalMem: unsafe extern "C" fn(*mut usize, CUdevice) -> CUDAStatus =
        *unsafe { cuda.get(b"cuDeviceTotalMem\0") }.unwrap();
    let cuDeviceGetAttribute: unsafe extern "C" fn(
        *mut c_int,
        CUdevice_attribute,
        CUdevice,
    ) -> CUDAStatus = *unsafe { cuda.get(b"cuDeviceGetAttribute\0") }.unwrap();
    let cuCtxCreate: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUDAStatus =
        *unsafe { cuda.get(b"cuCtxCreate\0") }.unwrap();
    //let cuMemAllocAsync = *unsafe { cuda.get(b"cuMemAllocAsync\0") }.unwrap();
    let cuMemAlloc = *unsafe { cuda.get(b"cuMemAlloc\0") }.unwrap();
    //let cuMemFreeAsync = *unsafe { cuda.get(b"cuMemFreeAsync\0") }.unwrap();
    let cuMemFree = *unsafe { cuda.get(b"cuMemFree\0") }.unwrap();
    let cuMemcpyHtoDAsync = *unsafe { cuda.get(b"cuMemcpyHtoDAsync\0") }.unwrap();
    //let cuMemcpyHtoD = *unsafe { cuda.get(b"cuMemcpyHtoD\0") }.unwrap();
    let cuMemcpyDtoHAsync = *unsafe { cuda.get(b"cuMemcpyDtoHAsync\0") }.unwrap();
    //let cuMemcpyPeer = *unsafe { cuda.get(b"cuMemcpyPeer\0") }.unwrap();
    //let cuCtxSetCurrent = *unsafe { cuda.get(b"cuCtxGetCurrent\0") }.unwrap();
    //let cuCtxDestroy = *unsafe { cuda.get(b"cuCtxDestroy\0") }.unwrap();
    let cuModuleLoadDataEx = *unsafe { cuda.get(b"cuModuleLoadDataEx\0") }.unwrap();
    let cuModuleGetFunction = *unsafe { cuda.get(b"cuModuleGetFunction\0") }.unwrap();
    let cuLaunchKernel = *unsafe { cuda.get(b"cuLaunchKernel\0") }.unwrap();
    let cuStreamCreate: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUDAStatus =
        *unsafe { cuda.get(b"cuStreamCreate\0") }.unwrap();
    let cuStreamSynchronize = *unsafe { cuda.get(b"cuStreamSynchronize\0") }.unwrap();
    let cuStreamWaitEvent = *unsafe { cuda.get(b"cuStreamWaitEvent\0") }.unwrap();
    //let cuStreamDestroy = *unsafe { cuda.get(b"cuStreamDestroy\0") }.unwrap();
    let cuModuleUnload = *unsafe { cuda.get(b"cuModuleUnload\0") }.unwrap();
    let cuEventCreate = *unsafe { cuda.get(b"cuEventCreate\0") }.unwrap();
    let cuEventRecord = *unsafe { cuda.get(b"cuEventRecord\0") }.unwrap();
    let cuEventSynchronize = *unsafe { cuda.get(b"cuEventSynchronize\0") }.unwrap();
    let cuEventDestroy = *unsafe { cuda.get(b"cuEventDestroy\0") }.unwrap();
    //let cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUDAStatus = *unsafe { cuda.get(b"cuDevicePrimaryCtxRetain\0") }.unwrap();

    if let Err(err) = unsafe { cuInit(0) }.check(ErrorStatus::Initialization) {
        if debug_dev {
            println!("CUDA requested, but cuInit failed. {err:?}");
        }
        return Err(err);
    }
    let mut driver_version = 0;
    unsafe { cuDriverGetVersion(&mut driver_version) }.check(ErrorStatus::DeviceQuery)?;
    let mut num_devices = 0;
    unsafe { cuDeviceGetCount(&mut num_devices) }.check(ErrorStatus::DeviceQuery)?;
    if num_devices == 0 {
        return Err(BackendError {
            status: ErrorStatus::DeviceEnumeration,
            context: "No available cuda device.".into(),
        });
    }
    let device_ids: Vec<i32> = (0..num_devices)
        .filter(|id| config.device_ids.as_ref().is_none_or(|ids| ids.contains(id)))
        .collect();
    if debug_dev && !device_ids.is_empty() {
        println!(
            "Using CUDA driver, driver version: {}.{} on devices:",
            driver_version / 1000,
            (driver_version - (driver_version / 1000 * 1000)) / 10
        );
    }

    let cuda = Arc::new(cuda);
    for dev_id in device_ids {
        let mut device = 0;
        if let Err(err) =
            unsafe { cuDeviceGet(&mut device, dev_id) }.check(ErrorStatus::DeviceEnumeration)
        {
            if debug_dev {
                println!("Device with id {dev_id} requested, but could not be enumerated.");
            }
            return Err(err);
        }

        let mut device_name = [0; 100];
        let Ok(()) = unsafe { cuDeviceGetName(device_name.as_mut_ptr(), 100, device) }
            .check(ErrorStatus::DeviceQuery)
        else {
            continue;
        };
        let mut major = 0;
        let mut minor = 0;
        let Ok(()) = unsafe { cuDeviceComputeCapability(&mut major, &mut minor, device) }
            .check(ErrorStatus::DeviceQuery)
        else {
            continue;
        };
        if debug_dev {
            println!("{:?}, compute capability: {major}.{minor}", unsafe {
                std::ffi::CStr::from_ptr(device_name.as_ptr())
            });
        }
        let mut free_bytes = 0;
        let Ok(()) =
            unsafe { cuDeviceTotalMem(&mut free_bytes, device) }.check(ErrorStatus::DeviceQuery)
        else {
            continue;
        };
        let free_bytes = free_bytes as Dim;
        let mut context: CUcontext = ptr::null_mut();
        if let Err(e) =
            unsafe { cuCtxCreate(&mut context, 0, device) }.check(ErrorStatus::Initialization)
        {
            if debug_dev {
                println!(
                    "Device with id {dev_id} requested, but cuda context initialization failed. {e:?}"
                );
            }
            continue;
        }
        /*if let Err(e) = unsafe { cuDevicePrimaryCtxRetain(&mut context, device) }.check("Failed to create CUDA context.") {
            println!("{e:?}");
            continue;
        }*/
        //println!("Using context {context:?} and device {device:?}");
        let mut stream = ptr::null_mut();
        if let Err(err) =
            unsafe { cuStreamCreate(&mut stream, 0) }.check(ErrorStatus::Initialization)
        {
            if debug_dev {
                println!(
                    "Device with id {dev_id} requested, but cuda stream initialization failed. {err:?}"
                );
            }
            continue;
        }
        let pool = MemoryPool::CUDA(CUDAMemoryPool {
            lib: cuda.clone(),
            context,
            device,
            free_bytes,
            buffers: Slab::new(),
            stream,
            cuEventCreate,
            //cuMemAllocAsync,
            cuMemAlloc,
            cuMemcpyHtoDAsync,
            //cuMemFreeAsync,
            cuMemFree,
            cuMemcpyDtoHAsync,
            cuEventRecord,
            cuStreamWaitEvent,
            cuEventSynchronize,
            cuEventDestroy,
            //cuStreamSynchronize,
            //cuMemcpyHtoD,
            //cuMemcpyPeer,
            //cuCtxSetCurrent,
            //cuCtxDestroy,
        });
        memory_pools.push(Pool::new(pool));
        let mut streams = Vec::new();
        for _ in 0..8 {
            let mut stream = ptr::null_mut();
            if let Err(err) =
                unsafe { cuStreamCreate(&mut stream, 0) }.check(ErrorStatus::Initialization)
            {
                if debug_dev {
                    println!(
                        "Device with id {dev_id} requested, but cuda stream initialization failed. {err:?}"
                    );
                }
                continue;
            }
            streams.push(CUDAStream { stream, load: 0 });
        }
        let mut dev = CUDADevice {
            device,
            dev_info: DeviceInfo {
                compute: 1024 * 1024 * 1024 * 1024,
                max_global_work_dims: [64, 64, 64],
                max_local_threads: 1,
                max_local_work_dims: [1, 1, 1],
                local_mem_size: 0,
                num_registers: 96,
                preferred_vector_size: 16,
                tensor_cores: major > 7,
            },
            streams,
            programs: Slab::new(),
            memory_pool_id: u32::try_from(memory_pools.len()).unwrap() - 1,
            cuModuleLoadDataEx,
            cuModuleGetFunction,
            cuModuleUnload,
            compute_capability: [major, minor],
            cuLaunchKernel,
            cuStreamSynchronize,
            cuEventCreate,
            cuEventRecord,
            cuStreamWaitEvent,
            //cuStreamDestroy,
        };
        dev.dev_info = DeviceInfo {
            compute: 1024 * 1024 * 1024 * 1024,
            max_global_work_dims: [
                Dim::try_from(dev.get(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                    cuDeviceGetAttribute,
                )?)
                .unwrap(),
                Dim::try_from(dev.get(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                    cuDeviceGetAttribute,
                )?)
                .unwrap(),
                Dim::try_from(dev.get(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
                    cuDeviceGetAttribute,
                )?)
                .unwrap(),
            ],
            max_local_threads: Dim::try_from(dev.get(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                cuDeviceGetAttribute,
            )?)
            .unwrap(),
            max_local_work_dims: [
                Dim::try_from(dev.get(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                    cuDeviceGetAttribute,
                )?)
                .unwrap(),
                Dim::try_from(dev.get(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                    cuDeviceGetAttribute,
                )?)
                .unwrap(),
                Dim::try_from(dev.get(
                    CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                    cuDeviceGetAttribute,
                )?)
                .unwrap(),
            ],
            local_mem_size: Dim::try_from(dev.get(
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                cuDeviceGetAttribute,
            )?)
            .unwrap(),
            num_registers: 96,
            preferred_vector_size: 16,
            tensor_cores: major > 7,
        };
        devices.push(Device::CUDA(dev));
    }
    Ok(())
}

impl CUDAMemoryPool {
    pub const fn deinitialize(&mut self) {
        let _ = self;
    }

    pub const fn free_bytes(&self) -> Dim {
        self.free_bytes
    }

    pub fn allocate(&mut self, bytes: Dim) -> Result<(BufferId, Event), BackendError> {
        if bytes > self.free_bytes {
            return Err(BackendError {
                status: ErrorStatus::MemoryAllocation,
                context: "Allocation failure.".into(),
            });
        }
        //println!("Allocating to context {:?}, device {:?}", self.context, self.device);
        let mut ptr = u64::try_from(self.device).unwrap();
        //unsafe { (self.cuCtxSetCurrent)(self.context) }.check("Failed to set current CUDA context.")?;
        let mut event = ptr::null_mut();
        unsafe { (self.cuEventCreate)(&mut event, 0x2) }.check(ErrorStatus::MemoryAllocation)?;
        debug_assert!(!self.stream.is_null());
        //unsafe { (self.cuMemAllocAsync)(&mut ptr, bytes, self.stream) }.check(ErrorStatus::MemoryAllocation)?;
        unsafe { (self.cuMemAlloc)(&mut ptr, bytes as usize) }
            .check(ErrorStatus::MemoryAllocation)?;
        unsafe { (self.cuEventRecord)(event, self.stream) }.check(ErrorStatus::MemoryAllocation)?;
        self.free_bytes = self.free_bytes.checked_sub(bytes).unwrap();
        Ok((
            self.buffers.push(CUDABuffer { ptr, bytes }),
            Event::CUDA(CUDAEvent { event }),
        ))
    }

    pub fn deallocate(&mut self, buffer_id: BufferId, mut event_wait_list: Vec<Event>) {
        while let Some(Event::CUDA(CUDAEvent { event })) = event_wait_list.pop() {
            if !event.is_null() {
                unsafe { (self.cuStreamWaitEvent)(self.stream, event, 0) }
                    .check(ErrorStatus::MemoryDeallocation)
                    .unwrap();
                unsafe { (self.cuEventDestroy)(event) }.check(ErrorStatus::MemoryCopyP2H).unwrap();
            }
        }
        let buffer = &mut self.buffers[buffer_id];
        //unsafe { (self.cuMemFreeAsync)(buffer.ptr, self.stream) }.check(ErrorStatus::MemoryDeallocation).unwrap();
        unsafe { (self.cuMemFree)(buffer.ptr) }.check(ErrorStatus::MemoryDeallocation).unwrap();
        self.free_bytes += buffer.bytes;
        self.buffers.remove(buffer_id);
    }

    pub fn host_to_pool(
        &mut self,
        src: &[u8],
        dst: BufferId,
        mut event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let dst = &self.buffers[dst];
        while let Some(Event::CUDA(CUDAEvent { event })) = event_wait_list.pop() {
            if !event.is_null() {
                unsafe { (self.cuStreamWaitEvent)(self.stream, event, 0) }
                    .check(ErrorStatus::MemoryCopyH2P)?;
            }
        }
        let mut event = ptr::null_mut();
        unsafe { (self.cuEventCreate)(&mut event, 0x2) }.check(ErrorStatus::MemoryCopyH2P)?;
        debug_assert!(!self.stream.is_null());
        //unsafe { (self.cuStreamSynchronize)(self.stream) }.check(ErrorStatus::MemoryCopyH2P).unwrap();
        unsafe { (self.cuMemcpyHtoDAsync)(dst.ptr, src.as_ptr().cast(), src.len(), self.stream) }
            .check(ErrorStatus::MemoryCopyH2P)?;
        //unsafe { (self.cuMemcpyHtoD)(dst.ptr, src.as_ptr().cast(), src.len()) }.check(ErrorStatus::MemoryCopyH2P)?;
        unsafe { (self.cuEventRecord)(event, self.stream) }.check(ErrorStatus::MemoryCopyH2P)?;
        Ok(Event::CUDA(CUDAEvent { event }))
    }

    pub fn pool_to_host(
        &mut self,
        src: BufferId,
        dst: &mut [u8],
        mut event_wait_list: Vec<Event>,
    ) -> Result<(), BackendError> {
        while let Some(Event::CUDA(CUDAEvent { event })) = event_wait_list.pop() {
            if !event.is_null() {
                unsafe { (self.cuStreamWaitEvent)(self.stream, event, 0) }
                    .check(ErrorStatus::MemoryCopyP2H)?;
                // Should we destroy the event here?
            }
        }
        let src = &self.buffers[src];
        let mut event = ptr::null_mut();
        unsafe { (self.cuEventCreate)(&mut event, 0x2) }.check(ErrorStatus::MemoryCopyP2H)?;
        unsafe {
            (self.cuMemcpyDtoHAsync)(dst.as_mut_ptr().cast(), src.ptr, dst.len(), self.stream)
        }
        .check(ErrorStatus::MemoryCopyP2H)?;
        unsafe { (self.cuEventRecord)(event, self.stream) }.check(ErrorStatus::MemoryCopyP2H)?;
        //unsafe { (self.cuStreamSynchronize)(self.stream) }.check(ErrorStatus::MemoryCopyP2H)?;
        unsafe { (self.cuEventSynchronize)(event) }.check(ErrorStatus::MemoryCopyP2H)?;
        unsafe { (self.cuEventDestroy)(event) }.check(ErrorStatus::MemoryCopyP2H)?;
        Ok(())
    }

    pub fn sync_events(&mut self, mut events: Vec<Event>) -> Result<(), BackendError> {
        while let Some(Event::CUDA(CUDAEvent { event })) = events.pop() {
            if !event.is_null() {
                unsafe { (self.cuEventSynchronize)(event) }.check(ErrorStatus::KernelSync)?;
                unsafe { (self.cuEventDestroy)(event) }.check(ErrorStatus::KernelSync)?;
            }
        }
        Ok(())
    }

    pub fn release_events(&mut self, events: Vec<Event>) {
        for event in events {
            let Event::CUDA(CUDAEvent { event }) = event else { unreachable!() };
            unsafe { (self.cuEventDestroy)(event) }.check(ErrorStatus::Deinitialization).unwrap();
        }
    }
}

impl CUDADevice {
    pub const fn deinitialize(&mut self) {
        let _ = self;
    }

    pub const fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    pub const fn memory_pool_id(&self) -> u32 {
        self.memory_pool_id
    }

    pub const fn free_compute(&self) -> u128 {
        self.dev_info.compute
    }

    pub fn compile(&mut self, kernel: &Kernel, debug_asm: bool) -> Result<ProgramId, BackendError> {
        let (gws, lws, name, ptx) = self.compile_cuda(kernel, debug_asm)?;
        //let (gws, lws, name, ptx) = self.compile_ptx(kernel, debug_asm)?;

        let mut module = ptr::null_mut();
        if let Err(err) = unsafe {
            (self.cuModuleLoadDataEx)(
                &mut module,
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
        if let Err(err) =
            unsafe { (self.cuModuleGetFunction)(&mut function, module, name.as_ptr().cast()) }
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
        Ok(program_id)
    }

    pub fn launch(
        &mut self,
        program_id: ProgramId,
        memory_pool: &mut CUDAMemoryPool,
        args: &[BufferId],
        // If sync is empty, kernel will be immediatelly synchronized
        mut event_wait_list: Vec<Event>,
    ) -> Result<Event, BackendError> {
        let stream_id = self.next_stream()?;
        let program = &self.programs[program_id];

        //println!("CUDA launch program id: {program_id}, gws: {:?}, lws: {:?}", program.global_work_size, program.local_work_size);

        let mut kernel_params: Vec<*mut core::ffi::c_void> = Vec::new();
        for &arg in args {
            let arg = &memory_pool.buffers[arg];
            //let ptr = &mut arg.mem;
            let ptr: *const u64 = &arg.ptr;
            let ptr: *mut u64 = ptr.cast_mut();
            kernel_params.push(ptr.cast());
        }

        while let Some(Event::CUDA(CUDAEvent { event })) = event_wait_list.pop() {
            if !event.is_null() {
                unsafe { (self.cuStreamWaitEvent)(self.streams[stream_id].stream, event, 0) }
                    .check(ErrorStatus::KernelLaunch)?;
            }
        }

        //unsafe { (self.cuStreamSynchronize)(self.streams[stream_id].stream) }.check(ErrorStatus::KernelLaunch).unwrap();

        let mut event = ptr::null_mut();
        unsafe { (self.cuEventCreate)(&mut event, 0) }.check(ErrorStatus::KernelLaunch)?;
        unsafe {
            (self.cuLaunchKernel)(
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
        unsafe { (self.cuEventRecord)(event, self.streams[stream_id].stream) }
            .check(ErrorStatus::KernelLaunch)?;

        //unsafe { (self.cuStreamSynchronize)(self.streams[stream_id].stream) }.check(ErrorStatus::KernelLaunch).unwrap();

        self.streams[stream_id].load += 1;
        Ok(Event::CUDA(CUDAEvent { event }))
    }

    pub fn release(&mut self, program_id: ProgramId) {
        let _ = unsafe { (self.cuModuleUnload)(self.programs[program_id].module) }
            .check(ErrorStatus::Deinitialization);
        self.programs.remove(program_id);
    }
}

impl CUDADevice {
    fn next_stream(&mut self) -> Result<usize, BackendError> {
        let mut id = self.streams.iter().enumerate().min_by_key(|(_, s)| s.load).unwrap().0;
        if self.streams[id].load > 20 {
            unsafe { (self.cuStreamSynchronize)(self.streams[id].stream) }
                .check(ErrorStatus::KernelSync)?;
            self.streams[id].load = 0;
            id = self.streams.iter().enumerate().min_by_key(|(_, q)| q.load).unwrap().0;
        }
        Ok(id)
    }

    fn get(
        &mut self,
        attr: CUdevice_attribute,
        cuDeviceGetAttribute: unsafe extern "C" fn(
            *mut c_int,
            CUdevice_attribute,
            CUdevice,
        ) -> CUDAStatus,
    ) -> Result<c_int, BackendError> {
        let mut v = 0;
        unsafe { cuDeviceGetAttribute(&mut v, attr, self.device) }
            .check(ErrorStatus::DeviceQuery)?;
        Ok(v)
    }

    #[allow(unused)]
    #[allow(clippy::type_complexity)]
    fn compile_cuda(
        &self,
        kernel: &Kernel,
        debug_asm: bool,
    ) -> Result<([Dim; 3], [Dim; 3], String, Vec<u8>), BackendError> {
        let mut source = String::from("(\n");
        let mut indent = String::from("  ");

        let mut global_work_size = [0; 3];
        let mut local_work_size = [0; 3];

        /*for (i, op) in kernel.ops[..6].iter().enumerate() {
            if let &IROp::Loop { len } = op {
                if i < 3 {
                    global_work_size[i] = len;
                } else {
                    local_work_size[i - 3] = len;
                }
            } else {
                unreachable!()
            }
        }*/

        // Declare global variables
        /*for (id, (read_only, dtype)) in kernel.global_variables.iter().enumerate() {
            writeln!(
                source,
                "{indent}{}{}* p{id},",
                if *read_only { "const " } else { "" },
                dtype.cu(),
            );
        }*/

        source.pop();
        source.pop();
        source.push_str("\n) {\n");

        // Declare local variables
        /*for (id, (scope, dtype, len, _)) in kernel.local_variables.iter().enumerate() {
            writeln!(
                source,
                "{indent}__shared__ {} p{id}[{len}];",
                //if *read_only { "const " } else { "" },
                dtype.cu(),
            );
        }*/

        // Declare register variables
        /*for (id, dtype) in kernel.registers.iter().enumerate() {
            writeln!(source, "{indent}{} r{id};", dtype.cu());
        }*/

        // Add indices for global and local loops
        writeln!(
            source,
            "{indent}r0 = blockIdx.x;  /* 0..{} */\n{indent}r1 = blockIdx.y;  /* 0..{} */\n{indent}r2 = blockIdx.z;  /* 0..{} */\n{indent}r3 = threadIdx.x;  /* 0..{} */\n{indent}r4 = threadIdx.y;  /* 0..{} */\n{indent}r5 = threadIdx.z;  /* 0..{} */",
            global_work_size[0],
            global_work_size[1],
            global_work_size[2],
            local_work_size[0],
            local_work_size[1],
            local_work_size[2]
        );

        /*for op in kernel.ops[6..kernel.ops.len() - 6].iter().copied() {
            match op {
                IROp::Load { z, address, offset } => {
                    writeln!(source, "{indent}r{z} = p{address}[{}];", offset.cu());
                }
                IROp::Store { address, offset, x } => {
                    writeln!(source, "{indent}p{address}[{}] = {};", offset.cu(), x.cu());
                }
                IROp::Set { z, value } => {
                    writeln!(source, "{indent}r{z} = {};", value.cu());
                }
                IROp::Cast { z, x, dtype } => {
                    writeln!(source, "{indent}r{} = ({})r{};", z, dtype.cu(), x);
                }
                IROp::Unary { z, x, uop } => {
                    let dtype = kernel.registers[z as usize];
                    let zero = Constant::new(0).cast(dtype).cu();
                    match uop {
                        UOp::ReLU => {
                            if dtype == DType::F16 {
                                writeln!(
                                    source,
                                    "{indent}r{z} = r{x} * __float2half(r{x} > {zero});"
                                )
                            } else {
                                writeln!(source, "{indent}r{z} = r{x} * (r{x} > {zero});")
                            }
                        }
                        UOp::Neg => writeln!(source, "{indent}r{z} = -r{x};"),
                        UOp::Exp2 => writeln!(source, "{indent}r{z} = exp2(r{x});"),
                        UOp::Log2 => writeln!(source, "{indent}r{z} = log2(r{x});"),
                        UOp::Reciprocal => writeln!(source, "{indent}r{z} = 1/r{x};"),
                        UOp::Sqrt => writeln!(source, "{indent}r{z} = sqrt(r{x});"),
                        UOp::Sin => writeln!(source, "{indent}r{z} = sin(r{x});"),
                        UOp::Cos => writeln!(source, "{indent}r{z} = cos(r{x});"),
                        UOp::Not => writeln!(source, "{indent}r{z} = !r{x};"),
                    };
                }
                IROp::Binary { z, x, y, bop } => {
                    writeln!(
                        source,
                        "{indent}r{z} = {};",
                        match bop {
                            BOp::Add => format!("{} + {}", x.cu(), y.cu()),
                            BOp::Sub => format!("{} - {}", x.cu(), y.cu()),
                            BOp::Mul => format!("{} * {}", x.cu(), y.cu()),
                            BOp::Div => format!("{} / {}", x.cu(), y.cu()),
                            BOp::Mod => format!("{} % {}", x.cu(), y.cu()),
                            BOp::Pow => format!("pow({}, {})", x.cu(), y.cu()),
                            BOp::Cmplt => format!("{} < {}", x.cu(), y.cu()),
                            BOp::Cmpgt => format!("{} > {}", x.cu(), y.cu()),
                            BOp::Max => format!("max({}, {})", x.cu(), y.cu()),
                            BOp::Or => format!("{} || {}", x.cu(), y.cu()),
                            BOp::And => format!("{} && {}", x.cu(), y.cu()),
                            BOp::BitOr => format!("{} | {}", x.cu(), y.cu()),
                            BOp::BitAnd => format!("{} & {}", x.cu(), y.cu()),
                            BOp::BitXor => format!("{} ^ {}", x.cu(), y.cu()),
                            BOp::NotEq => format!("{} != {}", x.cu(), y.cu()),
                            BOp::BitShiftLeft => format!("{} << {}", x.cu(), y.cu()),
                            BOp::BitShiftRight => format!("{} >> {}", x.cu(), y.cu()),
                        }
                    );
                }
                IROp::MAdd { z, a, b, c } => {
                    writeln!(
                        source,
                        "{indent}r{z} = {} * {} + {};",
                        a.cu(),
                        b.cu(),
                        c.cu()
                    );
                }
                IROp::Loop { id, len } => {
                    writeln!(
                        source,
                        "{indent}for (unsigned int r{id} = 0; r{id} < {len}; r{id} += 1) {{"
                    );
                    indent.push_str("  ");
                }
                IROp::EndLoop { .. } => {
                    indent.pop();
                    indent.pop();
                    writeln!(source, "{indent}}}");
                }
                IROp::LocalBarrier => {
                    writeln!(source, "__syncthreads()");
                }
                _ => {
                    todo!()
                }
            }
        }*/
        source += "}\n";

        let mut name = format!(
            "k_{}_{}_{}__{}_{}_{}",
            global_work_size[0],
            global_work_size[1],
            global_work_size[2],
            local_work_size[0],
            local_work_size[1],
            local_work_size[2],
        );
        let mut pragma = String::new();
        if source.contains("__half") {
            pragma += "#include <cuda_fp16.h>\n";
        }
        let source = format!("{pragma}extern \"C\" __global__ void {name}{source}\0");
        name += "\0";
        if debug_asm {
            println!("{source}");
        }

        let cudartc_paths = [
            "/lib/x86_64-linux-gnu/libnvrtc.so",
            "/usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so",
            "/usr/lib64/x86_64-linux/lib/libnvrtc.so",
            "/usr/lib/libnvrtc.so",
            "/usr/lib64/libnvrtc.so",
        ];
        let cudartc = cudartc_paths.iter().find_map(|path| unsafe { Library::new(path) }.ok());
        let Some(cudartc) = cudartc else {
            return Err(BackendError {
                status: ErrorStatus::KernelCompilation,
                context: "CUDA libnvrtc.so not found.".into(),
            });
        };
        let nvrtcCreateProgram: unsafe extern "C" fn(
            *mut nvrtcProgram,
            *const c_char,
            *const c_char,
            c_int,
            *const *const c_char,
            *const *const c_char,
        ) -> nvrtcResult = *unsafe { cudartc.get(b"nvrtcCreateProgram\0") }.unwrap();
        let nvrtcCompileProgram: unsafe extern "C" fn(
            nvrtcProgram,
            c_int,
            *const *const c_char,
        ) -> nvrtcResult = *unsafe { cudartc.get(b"nvrtcCompileProgram\0") }.unwrap();
        let nvrtcGetPTXSize: unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult =
            *unsafe { cudartc.get(b"nvrtcGetPTXSize\0") }.unwrap();
        let nvrtcGetPTX: unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> nvrtcResult =
            *unsafe { cudartc.get(b"nvrtcGetPTX\0") }.unwrap();
        let nvrtcGetProgramLogSize: unsafe extern "C" fn(nvrtcProgram, *mut usize) -> nvrtcResult =
            *unsafe { cudartc.get(b"nvrtcGetProgramLogSize\0") }.unwrap();
        let nvrtcGetProgramLog: unsafe extern "C" fn(nvrtcProgram, *mut c_char) -> nvrtcResult =
            *unsafe { cudartc.get(b"nvrtcGetProgramLog\0") }.unwrap();
        let nvrtcDestroyProgram: unsafe extern "C" fn(*mut nvrtcProgram) -> nvrtcResult =
            *unsafe { cudartc.get(b"nvrtcDestroyProgram\0") }.unwrap();

        let mut program = ptr::null_mut();
        unsafe {
            nvrtcCreateProgram(
                &mut program,
                source.as_ptr().cast(),
                name.as_ptr().cast(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        }
        .check(ErrorStatus::KernelCompilation)?;
        let df = format!(
            "--gpu-architecture=compute_{}{}\0",
            self.compute_capability[0], self.compute_capability[1]
        );
        let opts = [
            df.as_ptr().cast(),
            c"-I/usr/local/cuda-12.8/include".as_ptr().cast(),
        ];
        if let Err(e) = unsafe { nvrtcCompileProgram(program, 2, opts.as_ptr()) }
            .check(ErrorStatus::KernelCompilation)
        {
            println!("CUDA compilation error {e:?}");
            let mut program_log_size: usize = 0;
            unsafe { nvrtcGetProgramLogSize(program, &mut program_log_size) }
                .check(ErrorStatus::KernelCompilation)?;
            let mut program_log_vec: Vec<u8> = vec![0; program_log_size + 1];
            unsafe { nvrtcGetProgramLog(program, program_log_vec.as_mut_ptr().cast()) }
                .check(ErrorStatus::KernelCompilation)?;
            if let Ok(log) = String::from_utf8(program_log_vec) {
                println!("NVRTC program log:\n{log}",);
            } else {
                println!("NVRTC program log is not valid utf8");
            }
        }
        let mut ptx_size: usize = 0;
        unsafe { nvrtcGetPTXSize(program, &mut ptx_size) }.check(ErrorStatus::KernelCompilation)?;
        let mut ptx_vec: Vec<u8> = vec![0; ptx_size];
        unsafe { nvrtcGetPTX(program, ptx_vec.as_mut_ptr().cast()) }
            .check(ErrorStatus::KernelCompilation)?;
        unsafe { nvrtcDestroyProgram(&mut program) }.check(ErrorStatus::KernelCompilation)?;
        Ok((global_work_size, local_work_size, name, ptx_vec))
    }

    #[allow(unused)]
    fn compile_ptx(
        &mut self,
        kernel: Kernel,
        debug_asm: bool,
    ) -> ([Dim; 3], [Dim; 3], Box<str>, Vec<u8>) {
        let mut global_work_size = [0; 3];
        let mut local_work_size = [0; 3];
        /*for (i, op) in kernel.ops[..3].iter().enumerate() {
            if let IROp::Loop { len } = op {
                global_work_size[i] = *len;
            }
        }
        for (i, op) in kernel.ops[3..6].iter().enumerate() {
            if let IROp::Loop { len } = op {
                local_work_size[i] = *len;
            }
        }*/
        let name = format!(
            "k__{}_{}__{}_{}__{}_{}",
            global_work_size[0],
            local_work_size[0],
            global_work_size[1],
            local_work_size[1],
            global_work_size[2],
            local_work_size[2],
        );

        let indent = "    ";
        let mut source = format!(
            ".version {0}.{1}
.target sm_{0}{1}
.address_size 64
.visible .entry {name}(\n",
            self.compute_capability[0], self.compute_capability[1]
        );
        // Declare global variables
        /*for (id, (_, _)) in kernel.global_variables.iter().enumerate() {
            writeln!(source, "{indent}.param    .u64 g{id},");
        }*/
        source.pop();
        source.pop();
        source += "\n) {\n";

        // TOOD declare local variables

        // Temporaries
        writeln!(source, "{indent}.reg  .pred    p;");
        writeln!(source, "{indent}.reg  .s64    a0;");
        writeln!(source, "{indent}.reg  .s64    a1;");
        // Declare register variables
        /*for (id, dtype) in kernel.registers.iter().enumerate() {
            writeln!(source, "{indent}.reg  .{}    r{id};", dtype.ptx());
        }*/
        // Add indices for global and local loops
        writeln!(source, "{indent}mov.u32    r0, %ctaid.x;");
        writeln!(source, "{indent}mov.u32    r1, %tid.x;");
        writeln!(source, "{indent}mov.u32    r2, %ctaid.y;");
        writeln!(source, "{indent}mov.u32    r3, %tid.y;");
        writeln!(source, "{indent}mov.u32    r4, %ctaid.z;");
        writeln!(source, "{indent}mov.u32    r5, %tid.z;");

        let mut loop_id = 6;
        /*for op in kernel.ops[6..kernel.ops.len() - 6].iter().copied() {
            match op {
                IROp::Set { z, value } => {
                    writeln!(source, "{indent}mov.{}  r{z}, {};", value.dtype().ptx(), value.cu());
                }
                IROp::Load { z, address, offset } => {
                    let dtype = kernel.registers[z as usize];
                    // Get address
                    writeln!(
                        source,
                        "{indent}ld.param.u64    a0, [a{address}+{}];",
                        offset.cu()
                    );
                    // Convert address to global
                    writeln!(source, "{indent}cvta.to.global.u64    a1, a0;");
                    // Load from global to register
                    writeln!(source, "{indent}ld.global.{}    r{}, [a1];", dtype.ptx(), z);
                }
                IROp::Store { address, offset: _, x } => {
                    let Reg::Var(id) = x else { panic!() };
                    let dtype = kernel.registers[id as usize];
                    // Get address
                    writeln!(source, "{indent}ld.param.u64    a0, [a{address}];");
                    // Convert address to global
                    writeln!(source, "{indent}cvta.to.global.u64    a1, a0;");
                    // Load from global to register
                    writeln!(
                        source,
                        "{indent}st.global.{}    [a1], {};",
                        dtype.ptx(),
                        x.cu()
                    );
                }
                IROp::Unary { z, x, uop } => {
                    let dtype = kernel.registers[z as usize];
                    match uop {
                        UOp::ReLU => todo!(),
                        UOp::Neg => _ = writeln!(source, "{indent}neg.{}   r{z}, r{x};", dtype.ptx()),
                        UOp::Exp2 => {
                            _ = writeln!(source, "{indent}ex2.approx.{}   r{z}, r{x};", dtype.ptx());
                        }
                        UOp::Log2 => {
                            _ = writeln!(source, "{indent}lg2.approx.{}   r{z}, r{x};", dtype.ptx());
                        }
                        UOp::Sqrt => {
                            _ = writeln!(source, "{indent}sqrt.approx.{}   r{z}, r{x};", dtype.ptx());
                        }
                        UOp::Sin => {
                            _ = writeln!(source, "{indent}sin.approx.{}   r{z}, r{x};", dtype.ptx());
                        }
                        UOp::Cos => {
                            _ = writeln!(source, "{indent}cos.approx.{}   r{z}, r{x};", dtype.ptx());
                        }
                        UOp::Not => {
                            _ = writeln!(source, "{indent}not.{}   r{z}, r{x};", dtype.ptx());
                        }
                        UOp::Reciprocal => todo!(),
                    }
                }
                IROp::Binary { z, x, y, bop } => {
                    let dtype = kernel.registers[z as usize];
                    //println!("Adding binary {bop:?}");
                    writeln!(
                        source,
                        "{indent}{}.{}   r{z}, r{x}, r{y};",
                        match bop {
                            BOp::Add => "add",
                            BOp::Sub => "sub",
                            BOp::Mul => "mul",
                            BOp::Div => "div",
                            BOp::Pow => todo!(),
                            BOp::Cmplt => "set.lt",
                            BOp::Cmpgt => "set.gt",
                            BOp::NotEq => "set.ne",
                            BOp::Max => todo!(),
                            BOp::Or => todo!(),
                            BOp::And => todo!(),
                            BOp::BitOr => todo!(),
                            BOp::BitAnd => todo!(),
                            BOp::BitXor => todo!(),
                            BOp::Mod => todo!(),
                            BOp::BitShiftLeft => todo!(),
                            BOp::BitShiftRight => todo!(),
                        },
                        dtype.ptx(),
                    );
                }
                /*IROp::MAdd { z, a, b, c } => {
                    let dtype = kernel.registers[z as usize];
                    writeln!(
                        source,
                        "{indent}mad.lo.{}    r{z}, {}, {}, {};",
                        dtype.ptx(),
                        a.cu(),
                        b.cu(),
                        c.cu()
                    );
                }*/
                IROp::Loop { .. } => {
                    writeln!(source, "LOOP_{id}:");
                }
                IROp::EndLoop { len } => {
                    // Increment counter
                    writeln!(source, "{indent}add.u32    r{id}, r{id}, 1;");
                    // Set condition
                    writeln!(source, "{indent}setp.lt.u32    p, r{id}, {len};");
                    // Branch
                    writeln!(source, "@p  bra    LOOP_{id};");
                }
                IROp::LocalBarrier => {
                    writeln!(source, "__syncthreas()");
                }
                IROp::Cast { z, x, dtype } => {
                    _ = writeln!(
                        source,
                        "{indent}cvt.{}.{}    r{z}, r{x};",
                        dtype.ptx(),
                        dtype.ptx(),
                    );
                }
            }
        }*/
        // End kernel
        writeln!(source, "{indent}ret;\n}}\0");
        if debug_asm {
            println!("Compiling kernel {name}, PTX source:\n{source}");
        }
        (
            global_work_size,
            local_work_size,
            name.into(),
            source.into(),
        )
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUctx_st {
    _unused: [u8; 0],
}
type CUcontext = *mut CUctx_st;
type CUdevice = c_int;
type CUdeviceptr = u64;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUmod_st {
    _unused: [u8; 0],
}
type CUmodule = *mut CUmod_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUfunc_st {
    _unused: [u8; 0],
}
type CUfunction = *mut CUfunc_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUstream_st {
    _unused: [u8; 0],
}
type CUstream = *mut CUstream_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct CUevent_st {
    _unused: [u8; 0],
}
type CUevent = *mut CUevent_st;
#[allow(unused)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum CUjit_option {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_NUM_OPTIONS = 20,
}
#[allow(unused)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED = 135,
    CU_DEVICE_ATTRIBUTE_MAX,
}

impl DType {
    pub(super) fn cu(&self) -> &str {
        match self {
            Self::BF16 => todo!("BF16 is not native to OpenCL, workaround is WIP."),
            Self::F16 => "__half",
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

    pub(super) fn ptx(&self) -> &str {
        match self {
            Self::BF16 => todo!("BF16 is not native to OpenCL, workaround is WIP."),
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I8 => "s8",
            Self::I16 => "s16",
            Self::I32 => "s32",
            Self::I64 => "s64",
            Self::Bool => "bool",
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
        }
    }
}

impl Constant {
    fn cu(&self) -> String {
        match self {
            &Self::BF16(x) => format!("{}f", half::bf16::from_le_bytes(x)),
            &Self::F16(x) => format!("__float2half({:.6}f)", half::f16::from_le_bytes(x)),
            &Self::F32(x) => format!("{:.16}f", f32::from_le_bytes(x)),
            &Self::F64(x) => format!("{:.16}", f64::from_le_bytes(x)),
            Self::U8(x) => format!("{x}"),
            Self::I8(x) => format!("{x}"),
            Self::I16(x) => format!("{x}"),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}"),
            &Self::U64(x) => format!("{}", u64::from_le_bytes(x)),
            Self::I32(x) => format!("{x}"),
            &Self::I64(x) => format!("{}", i64::from_le_bytes(x)),
            Self::Bool(x) => format!("{x}"),
        }
    }
}

#[repr(C)]
#[derive(Debug)]
struct _nvrtcProgram {
    _unused: [u8; 0],
}
type nvrtcProgram = *mut _nvrtcProgram;

#[allow(unused)]
#[derive(Debug, PartialEq, Eq)]
#[repr(C)]
enum nvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
}

impl nvrtcResult {
    fn check(self, status: ErrorStatus) -> Result<(), BackendError> {
        if self == Self::NVRTC_SUCCESS {
            Ok(())
        } else {
            Err(BackendError { status, context: format!("{self:?}").into() })
        }
    }
}

#[allow(unused)]
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

impl CUDAStatus {
    fn check(self, status: ErrorStatus) -> Result<(), BackendError> {
        if self == Self::CUDA_SUCCESS {
            Ok(())
        } else {
            /*let cuda_paths = ["/lib/x86_64-linux-gnu/libcuda.so", "/lib64/libcuda.so"];
            let cuda = cuda_paths.iter().find_map(|path| unsafe { Library::new(path) }.ok());
            let Some(cuda) = cuda else {
                return Err(BackendError {
                    status: ErrorStatus::DyLibNotFound,
                    context: "CUDA runtime not found.".into(),
                }
                .into());
            };

            let cudaPeek: unsafe extern "C" fn(c_uint) -> CUDAStatus =
            *unsafe { cuda.get(b"cudaPeekAtLastError\0") }.unwrap();*/

            Err(BackendError { status, context: format!("{self:?}").into() })
        }
    }
}
