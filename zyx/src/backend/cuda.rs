use std::{collections::{BTreeMap, BTreeSet}, ffi::{c_char, c_int, c_uint, c_void}, ptr, sync::Arc};

use float8::F8E4M3;
use libloading::Library;
use nanoserde::DeJson;

use crate::{dtype::Constant, ir::Reg, slab::{Id, Slab}, DType};

use super::{BackendError, Buffer, Device, DeviceInfo, ErrorStatus, Event, MemoryPool};


/// CUDA configuration
#[derive(Debug, Default, DeJson)]
pub struct CUDAConfig {
    device_ids: Option<Vec<i32>>,
}

#[derive(Debug)]
pub(super) struct CUDAMemoryPool {
    // Just to close the connection
    #[allow(unused)]
    cuda: Arc<Library>,
    context: CUcontext,
    device: CUdevice,
    free_bytes: usize,
    buffers: Slab<CUDABuffer>,
    events: BTreeMap<BTreeSet<Id>, *mut c_void>,
    cuMemAlloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUDAStatus,
    cuMemcpyHtoD: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUDAStatus,
    cuMemcpyDtoH: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUDAStatus,
    cuMemFree: unsafe extern "C" fn(CUdeviceptr) -> CUDAStatus,
    cuMemcpyPeer:
        unsafe extern "C" fn(CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, usize) -> CUDAStatus,
    //cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUDAStatus,
    //cuCtxDestroy: unsafe extern "C" fn(CUcontext) -> CUDAStatus,
}

#[derive(Debug)]
pub(super) struct CUDABuffer {
    ptr: u64,
    bytes: usize,
}

#[derive(Debug)]
pub(super) struct CUDADevice {
    device: CUdevice,
    memory_pool_id: u32,
    dev_info: DeviceInfo,
    compute_capability: [c_int; 2],
    queues: Vec<CUDAQueue>,
    programs: Slab<CUDAProgram>,
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
    cuStreamDestroy: unsafe extern "C" fn(CUstream) -> CUDAStatus,
    /*cuLaunchKernel: unsafe extern "C" fn(
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
    cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUDAStatus,*/
}

#[derive(Debug)]
pub(super) struct CUDAProgram {
    //name: String,
    module: CUmodule,
    function: CUfunction,
    global_work_size: [usize; 3],
    local_work_size: [usize; 3],
}

#[derive(Debug)]
pub(super) struct CUDAQueue {
    stream: CUstream,
    load: usize,
}

#[derive(Debug)]
pub struct CUDAEvent {}

unsafe impl Send for CUDAMemoryPool {}
unsafe impl Send for CUDADevice {}
unsafe impl Send for CUDABuffer {}
unsafe impl Send for CUDAProgram {}
unsafe impl Send for CUDAQueue {}
unsafe impl Send for CUDAEvent {}

impl MemoryPool for CUDAMemoryPool {
    fn deinitialize(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    fn free_bytes(&self) -> usize {
        self.free_bytes
    }

    fn allocate(&mut self, bytes: usize) -> Result<crate::slab::Id, BackendError> {
        if bytes > self.free_bytes {
            return Err(BackendError { status: ErrorStatus::MemoryAllocation, context: "".into() });
        }
        //println!("Allocating to context {:?}, device {:?}", self.context, self.device);
        let mut ptr = u64::try_from(self.device).unwrap();
        //unsafe { (self.cuCtxSetCurrent)(self.context) }.check("Failed to set current CUDA context.")?;
        unsafe { (self.cuMemAlloc)(&mut ptr, bytes) }.check(ErrorStatus::MemoryAllocation)?;
        self.free_bytes = self.free_bytes.checked_sub(bytes).unwrap();
        Ok(self.buffers.push(CUDABuffer { ptr, bytes }))
    }

    fn deallocate(&mut self, buffer_id: crate::slab::Id) -> Result<(), BackendError> {
        if let Some(buffer) = self.buffers.remove(buffer_id) {
            unsafe { (self.cuMemFree)(buffer.ptr) }.check(ErrorStatus::Deinitialization)?;
            self.free_bytes += buffer.bytes;
        }
        Ok(())
    }

    fn host_to_pool(&mut self, src: &[u8], dst: crate::slab::Id) -> Result<(), BackendError> {
        let dst = &self.buffers[dst];
        unsafe { (self.cuMemcpyHtoD)(dst.ptr, src.as_ptr().cast(), src.len()) }
            .check(ErrorStatus::MemoryCopy)
    }

    fn pool_to_host(&mut self, src: crate::slab::Id, dst: &mut [u8]) -> Result<(), BackendError> {
        if let Some((_, event)) = self.events.iter().find(|(key, _)| key.contains(&src)) {
            unsafe { (self.clWaitForEvents)(1, std::slice::from_ref(&event).as_ptr().cast()) }
                .check(ErrorStatus::MemoryCopy)?;
        }
        let src = &self.buffers[src];
        unsafe { (self.cuMemcpyDtoH)(dst.as_mut_ptr().cast(), src.ptr, dst.len()) }
            .check(ErrorStatus::MemoryCopy)
    }

    fn get_buffer(&self, buffer: crate::slab::Id) -> super::Buffer {
        Buffer::CUDA(&self.buffers[buffer])
    }

    fn synchronize(&self, buffers: &std::collections::BTreeSet<crate::slab::Id>) -> Result<(), BackendError> {
        if let Some((_, event)) = self.events.iter().find(|(key, _)| key.is_disjoint(buffers)) {
            unsafe { (self.clWaitForEvents)(1, std::slice::from_ref(&event).as_ptr().cast()) }
                .check(ErrorStatus::MemoryCopy)?;
        }
        Ok(())
    }

    fn bind_event(&mut self, event: super::Event, buffers: std::collections::BTreeSet<crate::slab::Id>) {
        let Event::CUDA(CUDAEvent { ptr }) = event else { unreachable!() };
        self.events.insert(buffers, ptr);
    }
}

impl Device for CUDADevice {
    fn deinitialize(&mut self) -> Result<(), BackendError> {
        Ok(())
    }

    fn info(&self) -> &DeviceInfo {
        &self.dev_info
    }

    fn memory_pool_id(&self) -> u32 {
        self.memory_pool_id
    }

    fn compute(&self) -> u128 {
        self.dev_info.compute
    }

    fn compile(&mut self, kernel: &crate::ir::IRKernel, debug_asm: bool) -> Result<crate::slab::Id, BackendError> {
        let (global_work_size, local_work_size, name, ptx_vec) =
            self.compile_cuda(kernel, debug_asm)?;
        //self.compile_ptx(kernel, debug_asm)?;

        let mut module = ptr::null_mut();
        unsafe {
            (self.cuModuleLoadDataEx)(
                &mut module,
                ptx_vec.as_ptr().cast(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        }
        .check(ErrorStatus::KernelCompilation)?;
        let mut function: CUfunction = ptr::null_mut();
        // Don't forget that the name is null terminated string
        unsafe { (self.cuModuleGetFunction)(&mut function, module, name.as_ptr().cast()) }
            .check(ErrorStatus::KernelLaunch)?;

        let program_id = self.programs.insert(CUDAProgram {
            //name,
            module,
            function,
            global_work_size,
            local_work_size,
        });
        Ok(program_id)
    }

    fn launch(
        &mut self,
        program_id: crate::slab::Id,
        memory_pool: &mut dyn MemoryPool,
        args: &[crate::slab::Id],
        // If sync is empty, kernel will be immediatelly synchronized
        sync: std::collections::BTreeSet<crate::slab::Id>,
    ) -> Result<(), BackendError> {
        memory_pool.synchronize(&sync)?;
        let queue_id = self.next_queue()?;
        let program = &self.programs[program_id];
        let mut kernel_params: Vec<*mut core::ffi::c_void> = Vec::new();
        for &arg in args {
            let arg = &mut buffers[arg];
            //let ptr = &mut arg.mem;
            let ptr: *mut _ = &mut arg.ptr;
            kernel_params.push(ptr.cast());
        }
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
                self.stream,
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            )
        }
        .check("Failed to launch kernel.")?;
        if sync {
            todo!()
        }
        Ok(CUDAEvent {})
    }

    fn release(&mut self, program_id: crate::slab::Id) -> Result<(), BackendError> {
        if let Some(program) = self.programs.get(program_id) {
            unsafe { (self.cuModuleUnload)(program.module) }.check(ErrorStatus::Deinitialization)?;
        }
        Ok(())
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
    pub(super) fn ptx(&self) -> &str {
        match self {
            Self::BF16 => panic!("BF16 is not native to OpenCL, workaround is WIP."),
            Self::F8 => "f8",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::U8 => "u8",
            Self::I8 => "s8",
            Self::I16 => "s16",
            Self::I32 => "s32",
            Self::I64 => "s64",
            Self::Bool => "b8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
        }
    }
}

impl Constant {
    fn ptx(&self) -> String {
        match self {
            &Self::F16(x) => format!("{:.12}", half::f16::from_bits(x)),
            &Self::BF16(x) => format!("{:.12}", half::bf16::from_bits(x)),
            Self::F8(x) => {
                /*let bytes = unsafe { t::<_, f32>(*x).to_ne_bytes() };
                let hex = format!("{:02X}{:02X}{:02X}{:02X}", bytes[0], bytes[1], bytes[2], bytes[3]);
                format!("0f{}", hex)*/
                use float8::F8E4M3 as f8;
                format!("{:.12}", f8::from_bits(*x))
            }
            &Self::F32(x) => {
                /*let bytes = unsafe { t::<_, f32>(*x).to_ne_bytes() };
                let hex = format!("{:02X}{:02X}{:02X}{:02X}", bytes[0], bytes[1], bytes[2], bytes[3]);
                format!("0f{}", hex)*/
                format!("{:.12}", f32::from_bits(x))
            }
            &Self::F64(x) => {
                /*let bytes = unsafe { t::<_, f64>(*x).to_ne_bytes() };
                let hex = format!("{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}", bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7]);
                format!("0d{}", hex)*/
                format!("{:.12}", f64::from_bits(x))
            }
            Self::U8(_) => todo!(),
            Self::I8(_) => todo!(),
            Self::I16(_) => todo!(),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}"),
            Self::U64(x) => format!("{x}"),
            Self::I32(x) => format!("{x}"),
            Self::I64(x) => format!("{x}"),
            Self::Bool(_) => todo!(),
        }
    }
}

impl Reg {
    fn ptx(&self) -> String {
        match self {
            Self::Var(id) => format!("r{id}"),
            Self::Const(value) => value.ptx(),
        }
    }
}

impl DType {
    pub(super) fn cu(&self) -> &str {
        match self {
            Self::BF16 => todo!("BF16 is not native to OpenCL, workaround is WIP."),
            Self::F8 => todo!("F8 is not native to OpenCL, workaround is WIP."),
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
}

impl Reg {
    fn cu(&self) -> String {
        match self {
            Self::Var(id) => format!("r{id}"),
            Self::Const(value) => value.cu(),
        }
    }
}

impl Constant {
    fn cu(&self) -> String {
        match self {
            &Self::BF16(x) => format!("{}f", half::bf16::from_bits(x)),
            &Self::F8(x) => format!("{:.16}f", F8E4M3::from_bits(x)),
            &Self::F16(x) => format!("__float2half({:.6}f)", half::f16::from_bits(x)),
            &Self::F32(x) => format!("{:.16}f", f32::from_bits(x)),
            &Self::F64(x) => format!("{:.16}", f64::from_bits(x)),
            Self::U8(x) => format!("{x}"),
            Self::I8(x) => format!("{x}"),
            Self::I16(x) => format!("{x}"),
            Self::U16(x) => format!("{x}"),
            Self::U32(x) => format!("{x}"),
            Self::U64(x) => format!("{x}"),
            Self::I32(x) => format!("{x}"),
            Self::I64(x) => format!("{x}"),
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
            Err(BackendError { status, context: format!("{self:?}") })
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
            Err(BackendError { status, context: format!("{self:?}") })
        }
    }
}